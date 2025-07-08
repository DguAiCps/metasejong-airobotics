# === task_implementation.py ===

import time
print("i")
import numpy as np
print("i")
import os
print("i")
import yaml
print("i")
from . import astar
print("i")
from . import order_decision
print("i")
from pathlib import Path
print("i")
from typing import Dict, List
print("i")
from ultralytics import YOLO
print("i")
from .competition_task_base import CompetitionTask
print("i")
from .robot_node import RobotNode, MobileBaseCommander
print("i")
from .vision_manager import VisionManager
print("i")
from .manipulation_manager import ManipulationManager
print("i")
from .mpc_controller import MPCController
print("i")
from .path_generator import generate_final_path_with_frontal_pickup
print("i")
from .order_decision import genetic_algorithm
print("i")
from .util import quaternion_to_yaw, normalize_angle
from .obj_detect_mlp import *
from scipy.spatial.transform import Rotation as R

class TaskImplementation(CompetitionTask):
    def __init__(self, team_name: str, team_token: str, target_stage: str, robot_node: RobotNode):
        super().__init__(team_name, team_token, target_stage)
        self.logger.info('[TaskImplementation] __init__ 진입')

        self.robot_node = robot_node
        model_root = Path(__file__).resolve().parent

        
        self.yolo_model = YOLO(model_root / ".." / "resource" / "final.pt")
        self.logger.info("YOLOv8 모델 로딩 완료 (final.pt)")

        scenario_id = os.getenv("ENV_METASEJONG_SCENARIO", "demo")
        answer_sheet_file_path = f"/data/{scenario_id}_answer_sheet.yaml"
        self.answer_sheet = self.load_yaml_to_dict(answer_sheet_file_path)
        self.robot_node.start_position = self.answer_sheet['start_position']

        self.vision_manager = VisionManager(self.robot_node, self.yolo_model, self.logger)
        self.manip_manager = ManipulationManager(self.robot_node, self.logger)
        self.mpc = MPCController()

        self.object_detection_result = []
        self.logger.info("[Competitor] Answer sheet 로딩 완료")

    def load_yaml_to_dict(self, yaml_path: str) -> Dict:
        self.logger.info(f'YAML 로딩: {yaml_path}')
        with open(yaml_path, 'r') as yaml_file:
            return yaml.safe_load(yaml_file)

    def start_stage_1_task(self) -> List[Dict[str, List[int]]]:
        """
        Start stage 1 task.

        Returns:
            List[Dict[str, List[int]]]: List of detected objects with their classes and positions.
                Each object is a dictionary with:
                - 'class_name': str - Object class name
                - 'position': List[int] - [x, y, z] coordinates
        """
        stage1_answer = []

        #   Delay for 10 seconds for demo. emulate the time for object detection and pose estimation
        time.sleep(5)
        centers = detect_objects(use_mlp=True)

        self.object_detection_result = self.answer_sheet['mission_objects']
        for object_detection in centers:
            if object_detection['recyclable'] == True:
                stage1_answer.append({
                    "class_name": object_detection['class_name'],
                    "position": object_detection['position']
                })

        return stage1_answer


    def start_stage_2_task(self) -> bool:
        self.logger.info('[Stage2] 시작')
        order_decision._map_msg, order_decision._tf = order_decision._get_map_and_tf()

        if not self.object_detection_result:
            self.logger.warn("[Stage2] Stage1 결과 없음")
            return False

        recyclable_objects = [o for o in self.object_detection_result if o['recyclable']]
        if not recyclable_objects:
            self.logger.info("[Stage2] 재활용 쓰레기 없음")
            return False

        trash_positions = np.array([obj['position'][:2] for obj in recyclable_objects])
        obstacle_list = [{'center': np.array(obj['position'][:2]), 'radius': 0.2} for obj in self.object_detection_result]

        # visit_order 함수 호출, 밑에 함수도 수정해야함
        astar.build_soft_cost_grid(
            [astar.world_to_grid(obj['position'][:2], order_decision._map_msg, order_decision._tf) for obj in recyclable_objects],
            (order_decision._map_msg.info.width, order_decision._map_msg.info.height)
        )
        visited_order = order_decision.visit_order(
            recyclable_objects,
            entry_pos=self.robot_node.get_robot_position(),
            exit_pos=None
        )
        self.logger.info(f"[Stage2] 방문 순서 결정: {visited_order}")

        # 지우기
        #order, _ = genetic_algorithm(trash_positions)
        ordered_positions = trash_positions[visited_order]
        start_pos = self.robot_node.get_robot_position()[:2]
        _, pickup_infos = generate_final_path_with_frontal_pickup(ordered_positions, obstacle_list, start_pos)

        self._follow_path_and_pick_objects(pickup_infos, visited_order, recyclable_objects)
        return True

    def _follow_path_and_pick_objects(self, pickup_infos, object_order, recyclable_objects):
        self.logger.info("[Stage2] 경로 추종 시작")
        collected_positions = []

        for i, ((tx, ty), ttheta) in enumerate(pickup_infos):
            if self.force_stop:
                break

            while True:
                pos = self.robot_node.get_robot_position()
                ori = self.robot_node.get_robot_orientation()
                yaw = quaternion_to_yaw(ori)

                dx, dy = tx - pos[0], ty - pos[1]
                dist = np.hypot(dx, dy)
                target_heading = np.arctan2(dy, dx)
                if dist < 0.3:
                    break

                x0 = [pos[0], pos[1], yaw]
                goal = [tx, ty, target_heading]
                obstacles = []

                for j, obj in enumerate(recyclable_objects):
                    if j != object_order[i]:
                        if obj['position'][:2] not in collected_positions:
                            obstacles.append({'center': np.array(obj['position'][:2]), 'radius': 0.2})
                for obj in self.object_detection_result:
                    if not obj['recyclable']:
                        obstacles.append({'center': np.array(obj['position'][:2]), 'radius': 0.2})

                v, w = self.mpc.solve(x0, goal, obstacles=[o['center'] for o in obstacles])
                self.robot_node.move_robot(MobileBaseCommander(linear_x=v, angular_z=w))
                time.sleep(0.1)

            while True:
                yaw = quaternion_to_yaw(self.robot_node.get_robot_orientation())
                angle_diff = normalize_angle(ttheta - yaw)
                if abs(angle_diff) < np.radians(10):
                    break
                w = float(np.clip(angle_diff, -0.6, 0.6))
                self.robot_node.move_robot(MobileBaseCommander(0.0, w))
                time.sleep(0.05)

            self.robot_node.move_robot(MobileBaseCommander(0.0, 0.0))
            self.logger.info(f"[Stage2] #{i+1} 수거 위치 도달 완료")
            

            obj = recyclable_objects[object_order[i]]
            obj_dict = {
                'position': obj['position'],
                'class_name': obj['class_name'],
                'rotation': obj.get('rotation', [0, 0, 0]),
                'recyclable': True
            }
            self.vision_manager.center_align(obj_dict)
            vision_result = self.vision_manager.get_object_pose(obj_dict)
            gripper_quat = self.vision_manager.compute_grasp_quaternion(vision_result['quaternion'])
            self.manip_manager.pick_object(obj_dict, np.array(vision_result['position']), gripper_quat, vision_result.get('closest_box'))

            collected_positions.append(obj['position'][:2])
            time.sleep(2)

    def stop_task_and_quit(self):
        super().stop_task_and_quit()
        self.robot_node.stop_and_destroy_robot()
