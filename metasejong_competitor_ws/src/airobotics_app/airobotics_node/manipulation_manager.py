# === manipulation_manager.py ===

import time
import numpy as np
from scipy.spatial.transform import Rotation as R
from .robot_node import MobileBaseCommander, ManipulatorCommander

class ManipulationManager:
    def __init__(self, robot_node, logger):
        self.robot_node = robot_node
        self.logger = logger

    def pick_object(self, object_detection: dict, object_pos_world: np.ndarray, object_quat_world: list, closest_box: list):
        """
        주어진 객체 정보 및 YOLO + Depth 기반 위치/회전 결과를 이용해 집기 동작을 수행
        """
        robot_pos_world = np.array(self.robot_node.get_robot_position())
        robot_quat_world = np.array(self.robot_node.get_robot_orientation())
        rot_robot = R.from_quat(robot_quat_world)

        delta_pos = object_pos_world - robot_pos_world
        delta_pos[2] = 0
        if np.linalg.norm(delta_pos) < 1e-3:
            delta_pos = np.array([1.0, 0.0, 0.0])
        approach_dir = delta_pos / np.linalg.norm(delta_pos)

        x_axis = approach_dir
        z_axis = np.array([0, 0, -1])
        y_axis = np.cross(z_axis, x_axis)
        y_axis /= np.linalg.norm(y_axis)
        x_axis = np.cross(y_axis, z_axis)

        grasp_rot_matrix = np.column_stack((x_axis, y_axis, z_axis))
        grasp_quat_world = R.from_matrix(grasp_rot_matrix).as_quat()

        relative_pos = object_pos_world - robot_pos_world
        relative_pos[1]=0##############반드시 수정
        relative_pos[2] = -0.24

        #relative_quat = (R.from_quat(grasp_quat_world) * rot_robot.inv()).as_quat()
        relative_quat = object_quat_world
        class_name = object_detection['class_name']
        drop_position = 0.0
        if class_name in ['master_chef_can', 'cola_can']:
            drop_position = -0.15
        elif class_name in ['juice', 'disposable_cup']:
            drop_position = 0.0
        elif class_name in ['tissue', 'cracker_box']:
            drop_position = 0.15

        pick_and_place_command = ManipulatorCommander(
            start_target_orientation_quat=relative_quat,
            start_target_position=relative_pos.tolist(),
            end_target_orientation_quat=[1, 0, 0, 0],
            end_target_position=[0.5, drop_position, -0.1]
        )

        self.robot_node.pick_up_object(pick_and_place_command)

        # 검증 및 로그
        self.logger.info(f"[검증] 실제 쓰레기 위치 (정답지): {object_detection['position']}")
        self.logger.info(f"[검증] 실제 쓰레기 회전 (정답지): {object_detection['rotation']}")

        quat = object_quat_world
        euler_deg = R.from_quat(quat).as_euler('zyx', degrees=True)
        self.logger.info(f"[검증] 추정된 쓰레기 위치 (YOLO+Depth): {object_pos_world.tolist()}")
        self.logger.info(f"[검증] 추정된 회전 (Quaternion): {quat}")
        self.logger.info(f"[검증] 추정된 회전 (Euler XYZ degrees): {euler_deg.tolist()}")
        self.logger.info(f"[검증] 로봇 위치: {robot_pos_world.tolist()}")
        self.logger.info(f"[검증] 로봇-쓰레기 상대 위치: {relative_pos.tolist()}")

        if closest_box is not None:
            x1, y1, x2, y2 = map(int, closest_box)
            corners = [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]
            for i, (u, v) in enumerate(corners):
                if 0 <= v < self.robot_node.depth_image.shape[0] and 0 <= u < self.robot_node.depth_image.shape[1]:
                    d = float(self.robot_node.depth_image[v, u])
                    self.logger.info(f'[검증] bbox corner {i+1}: (u={u}, v={v}) → depth = {d:.3f}m')
                else:
                    self.logger.info(f"[검증] bbox corner {i+1}: (u={u}, v={v}) → 범위 초과")

        time.sleep(20)
