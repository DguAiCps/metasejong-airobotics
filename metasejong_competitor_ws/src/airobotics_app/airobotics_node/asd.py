# Copyright (c) 2025, IoT Convergence & Open Sharing System (IoTCOSS)
#
# All rights reserved. This software and its documentation are proprietary and confidential.
# The IoT Convergence & Open Sharing System (IoTCOSS) retains all intellectual property rights,
# including but not limited to copyrights, patents, and trade secrets, in and to this software
# and related documentation. Any use, reproduction, disclosure, or distribution of this software
# and related documentation without explicit written permission from IoTCOSS is strictly prohibited.

from tkinter import Image

import math
import time
import os
import yaml
import numpy as np
from typing import Dict, List, Union
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import Image
import scipy.optimize
from pathlib import Path
from .competition_task_base import CompetitionTask
from .robot_node import RobotNode, MobileBaseCommander, ManipulatorCommander
from .robot_util import RobotUtil

from ultralytics import YOLO

CONST_GLOBAL_NAVIGATION_DISTANCE = 1.0
CONST_GLOBAL_PICK_DISTANCE = 0.9
import numpy as np
import time
import random
import math

# === 경로 생성기 ===
def get_heading(from_point, to_point):
    vec = to_point - from_point
    return np.arctan2(vec[1], vec[0])

def generate_arc_path(start, goal, steps=20, force_final_heading=None):
    path = []
    for i in range(1, steps + 1):
        t = i / steps
        intermediate = (1 - t) * start + t * goal
        heading = get_heading(start, goal) if force_final_heading is None or i < steps else force_final_heading
        path.append((intermediate[0], intermediate[1], heading))
    return path

def is_path_colliding(p1, p2, obstacle, buffer=0.6):
    p1, p2 = np.array(p1), np.array(p2)
    center = obstacle['center']
    radius = obstacle['radius'] + buffer
    d = p2 - p1
    f = p1 - center
    a = np.dot(d, d)
    b = 2 * np.dot(f, d)
    c = np.dot(f, f) - radius**2
    discriminant = b**2 - 4*a*c
    if discriminant < 0:
        return False
    sqrt_disc = np.sqrt(discriminant)
    t1 = (-b - sqrt_disc) / (2*a)
    t2 = (-b + sqrt_disc) / (2*a)
    return (0 <= t1 <= 1) or (0 <= t2 <= 1)

def create_waypoint(p1, p2, obstacle, buffer=0.6):
    center = obstacle['center']
    radius = obstacle['radius'] + buffer
    path_vec = p2 - p1
    path_dir = path_vec / (np.linalg.norm(path_vec) + 1e-6)
    perp_dir = np.array([-path_dir[1], path_dir[0]])
    wp1 = center + perp_dir * radius
    wp2 = center - perp_dir * radius
    dist1 = np.linalg.norm(p1 - wp1) + np.linalg.norm(wp1 - p2)
    dist2 = np.linalg.norm(p1 - wp2) + np.linalg.norm(wp2 - p2)
    return wp1 if dist1 <= dist2 else wp2

def generate_final_path_with_frontal_pickup(trash_positions, obstacles, start_position):
    final_path = []
    pickup_infos = []
    cur_pos = np.array(start_position)
    used_obstacles = obstacles.copy()

    for t_i in trash_positions:
        # 쓰레기와의 상대 위치(-> 상대 방향 단위벡터)
        direction = t_i - cur_pos
        direction /= (np.linalg.norm(direction) + 1e-6)

        # 쓰레기 위치에서 상대 방향의 반대로 0.9만큼 -> 픽업 위치
        pickup_point = t_i - direction * 0.9
        # 픽업 위치와 쓰레기 위치의 radian 각도
        goal_heading = get_heading(pickup_point, t_i)
        # 현재 위치에서 픽업 위치까지의 호 경로 생성
        arc = generate_arc_path(cur_pos, pickup_point, steps=20, force_final_heading=goal_heading)
        # 호 경로가 장애물과 충돌하는지 확인
        collision = False
        for j in range(len(arc) - 1):
            for obs in used_obstacles:
                if is_path_colliding(arc[j][:2], arc[j+1][:2], obs):
                    collision = True
                    collision_obs = obs
                    break
            if collision:
                break
        
        # 충돌이 없으면 호 경로를 최종 경로에 추가
        if not collision:
            final_path.extend(arc)
        # 충돌이 있으면 장애물 회피를 위해 웨이포인트 생성하고 최종 경로에 추가
        else:
            wp = create_waypoint(cur_pos, pickup_point, collision_obs)
            arc1 = generate_arc_path(cur_pos, wp, steps=15)
            arc2 = generate_arc_path(wp, pickup_point, steps=15, force_final_heading=goal_heading)
            final_path.extend(arc1)
            final_path.extend(arc2)
        # 픽업 위치와 목표 각도를 저장
        pickup_infos.append((pickup_point, goal_heading))
        # 현재 위치를 픽업 위치로 업데이트
        cur_pos = pickup_point
        # 장애물 리스트에서 현재 쓰레기 위치와 동일한(가장 가까운) 장애물 제거
        used_obstacles = [obs for obs in used_obstacles if not np.allclose(obs['center'], t_i)]

    return final_path, pickup_infos

def fitness(order, positions):
    path_len = 0
    cur = positions[0]
    for idx in order:
        next_pos = positions[idx]
        path_len += np.linalg.norm(next_pos - cur)
        cur = next_pos
    return -path_len

def mutate(order):
    a, b = random.sample(range(len(order)), 2)
    order[a], order[b] = order[b], order[a]

def crossover(p1, p2):
    size = len(p1)
    a, b = sorted(random.sample(range(size), 2))
    child = [-1] * size
    child[a:b+1] = p1[a:b+1]
    fill = [x for x in p2 if x not in child]
    idx = 0
    for i in range(size):
        if child[i] == -1:
            child[i] = fill[idx]
            idx += 1
    return child

def genetic_algorithm(trash_positions, pop_size=30, generations=100, mutation_rate=0.2):
    num = len(trash_positions)
    population = [random.sample(range(num), num) for _ in range(pop_size)]
    best_chr = None
    best_fit = float('-inf')

    for gen in range(generations):
        scored = [(indiv, fitness(indiv, trash_positions)) for indiv in population]
        scored.sort(key=lambda x: x[1], reverse=True)
        population = [ind for ind, _ in scored[:pop_size//2]]
        if scored[0][1] > best_fit:
            best_chr = scored[0][0][:]
            best_fit = scored[0][1]
        while len(population) < pop_size:
            p1, p2 = random.sample(population[:10], 2)
            child = crossover(p1, p2)
            if random.random() < mutation_rate:
                mutate(child)
            population.append(child)
    return best_chr, best_fit

# 원통의 긴 축 = local Z → world 기준으로 변환
def _compute_grasp_quaternion(o_euler_deg, r_quat_xyzw):
    # 1. 원통의 긴 축 = local Z → world 기준으로 변환
    rot_o = R.from_euler('xyz', o_euler_deg, degrees=True)
    long_axis = rot_o.apply([0, 0, 1])  # 원통 긴 축 (world)

    # 2. 긴 축에 수직한 접근 방향 (XY 평면)
    approach_vec_2d = np.array([-long_axis[1], long_axis[0]])
    approach_vec_2d /= np.linalg.norm(approach_vec_2d)

    # 3. 전방 방향 (X축): 접근 방향
    x_axis = np.array([approach_vec_2d[0], approach_vec_2d[1], 0])

    # 4. 아래 방향 (Z축): 항상 월드 기준 -Z
    z_axis = np.array([0, 0, -1])

    # 5. Y축 = Z x X (오른손 좌표계)
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)

    # 6. 정규 직교 좌표계 구성
    rot_matrix = np.column_stack((x_axis, y_axis, z_axis))

    # 7. 회전 행렬 → 쿼터니언
    grasp_rot = R.from_matrix(rot_matrix)
    return grasp_rot.as_quat()  # [x, y, z, w]


class TaskImplementation(CompetitionTask):
    """
    Competition task implementation class.
    Implements actual tasks by inheriting from the CompetitionTask class.
    """
    
    def __init__(self, team_name: str, team_token: str, target_stage: str, robot_node: RobotNode):
        """
        Initialize task implementation
        
        Args:
            node (Node): ROS2 node instance
            team_name (str): Team name
            team_token (str): Team token
            target_stage (str): Target stage
        """
        super().__init__(team_name, team_token, target_stage)
        self.logger.info('[TaskImplementation] __init__ 진입')

        self.current_position = [0, 0, 0]
        self.current_orientation = [0, 0, 0, 1]

        self.object_detection_result = []

        self.robot_node = robot_node
        model_root = Path(__file__).resolve().parent
        self.yolo_model = YOLO(model_root / ".." / "resource" / "final.pt")
        
        self.logger.info("YOLOv8 모델 로딩 완료 (final.pt)")
        try: 
            scenario_id = os.getenv("ENV_METASEJONG_SCENARIO", "demo")
            answer_sheet_file_path = f"/data/{scenario_id}_answer_sheet.yaml"

            self.answer_sheet = self.load_yaml_to_dict(answer_sheet_file_path)
            self.robot_node.start_position = self.answer_sheet['start_position']

            self.logger.info(f"[Competitor] Successfully loaded answer sheet")
            self.logger.info(f"  - Path for answer sheet: {answer_sheet_file_path}")

        except Exception as e:
            self.logger.error(f"Failed to load answer sheet: {str(e)}")
            self.logger.error(f"  - Path for answer sheet: {answer_sheet_file_path}")
            raise

        self.robot_util = RobotUtil()

        # 맵 관련 변수 및 구독 제거 (RobotNode에서 관리)
        pass

    def load_yaml_to_dict(self, yaml_path: str) -> Dict:
        self.logger.info(f'[TaskImplementation] load_yaml_to_dict: {yaml_path}')
        try:
            with open(yaml_path, 'r') as yaml_file:
                yaml_dict = yaml.safe_load(yaml_file)

                self.logger.info('[TaskImplementation] YAML 파일 로드 성공')
                return yaml_dict
        except FileNotFoundError:
            self.logger.error(f'[TaskImplementation] 파일 없음: {yaml_path}')
            raise
        except yaml.YAMLError as e:
            self.logger.error(f'[TaskImplementation] YAML 파싱 에러: {str(e)}')
            raise

    # overriding abstract method. 
    #   Stage 1 task implementation: Objecect detection and pose estimation from Fixed camera image
    def start_stage_1_task(self) -> List[Dict[str, List[int]]]:
        self.logger.info('[TaskImplementation] start_stage_1_task 진입')
        stage1_answer = []

        #   Delay for 10 seconds for demo. emulate the time for object detection and pose estimation
        time.sleep(5)

        self.object_detection_result = self.answer_sheet['mission_objects']
        for object_detection in self.object_detection_result:
            self.logger.debug(f'[TaskImplementation] Detected object: {object_detection}')
            if object_detection['recyclable'] == True:
                stage1_answer.append({
                    "class_name": object_detection['class_name'],
                    "position": object_detection['position']
                })
        self.logger.info(f'[TaskImplementation] stage1_answer: {stage1_answer}')
        return stage1_answer  
    
    #   Demo implementation for navigate to the target position
    #   TODO: Implement actual navigation logic
    #   README: This function is only for demo implementation.
    #           Competitor must implement actual navigation logic.  

    def mpc_control_with_scipy_full(self, x0, goal, obstacles=None, N=7, dt=0.2, v_min=-2.0, v_max=2.0, w_min=-3.14, w_max=3.14, min_obs_dist=0.7):
        """
        scipy.optimize 기반 풀 MPC: 목표점, 제약조건, 장애물 회피, 입력 변화 최소화, 마지막 θ 정렬
        x0: [x, y, theta]
        goal: [x, y, theta]
        obstacles: [[x1, y1], ...]
        N: 예측지평선
        dt: 샘플링 타임
        """
        def cost(u_flat):
            u = u_flat.reshape(N, 2)
            x = np.array(x0)
            total = 0
            for k in range(N):
                v, w = u[k]
                x = x + dt * np.array([v*np.cos(x[2]), v*np.sin(x[2]), w])
                dist_to_goal = np.linalg.norm(x[:2] - goal[:2])
                #total += 10 * dist_to_goal**2                          # 기본 위치 비용
                #total += 0.1 * (v**2 + w**2)                           # 속도 크기 비용
                #total += 5.0 * dist_to_goal * v**2                     # 거리 가까울수록 속도 줄이는 항
                total += 10*np.sum((x[:2] - goal[:2])**2) + 0.1*(v**2 + w**2)
                if k > 0:
                    total += 0.5*((u[k][0]-u[k-1][0])**2 + (u[k][1]-u[k-1][1])**2)  # 입력 변화 최소화
            # 마지막 상태에서 θ 정렬 비용 강화
            total += 10*((x[2]-goal[2])**2)
            return total
        def constraint_factory(obs, k):
            def constr(u_flat):
                u = u_flat.reshape(N, 2)
                x = np.array(x0)
                for i in range(k+1):
                    v, w = u[i]
                    x = x + dt * np.array([v*np.cos(x[2]), v*np.sin(x[2]), w])
                return np.sqrt((x[0]-obs[0])**2 + (x[1]-obs[1])**2) - min_obs_dist
            return constr
        u0 = np.zeros((N,2))
        bounds = [(v_min, v_max), (w_min, w_max)]*N
        cons = []
        if obstacles:
            # 가까운 장애물 10개만 사용
            obstacles = sorted(obstacles, key=lambda obs: (x0[0]-obs[0])**2 + (x0[1]-obs[1])**2)[:10]
            for k in range(N):
                for obs in obstacles:
                    cons.append({'type': 'ineq', 'fun': constraint_factory(obs, k)})
        res = scipy.optimize.minimize(cost, u0.flatten(), bounds=bounds, constraints=cons, method='SLSQP', options={'maxiter': 100, 'ftol': 1e-2})
        u_opt = res.x[:2]
        return u_opt  # [v, w]

    def _demo_implementation___locate_robot_to_target_position(
            self, 
            target_position: List[float], 
            distance_threshold: float = CONST_GLOBAL_NAVIGATION_DISTANCE,
            angle_threshold: float = math.pi/72,    # 360도 중 5도
            sleep_time: float = 0.1
        ) -> None:
        self.logger.info(f'[TaskImplementation] 이동 시작: target_position={target_position}')
        try:
            # 1. 목표점 0.9m 앞에서 정지, 정면 정렬
            while True:
                robot_pos = self.robot_node.get_robot_position()
                robot_ori = self.robot_node.get_robot_orientation()
                # 쿼터니언 -> yaw
                qx, qy, qz, qw = robot_ori
                siny_cosp = 2 * (qw * qz + qx * qy)
                cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
                robot_yaw = np.arctan2(siny_cosp, cosy_cosp)
                dx = target_position[0] - robot_pos[0]
                dy = target_position[1] - robot_pos[1]
                dist = np.hypot(dx, dy)
                # 목표점 0.9m 앞에서 멈춤
                if dist < 0.9 + 0.05:
                    self.logger.info('[TaskImplementation] 목표점 0.9m 앞 도달')
                    break
                # 장애물 리스트(맵에서 100 이상 셀 중심)
                obstacles = []
                if self.robot_node.map_array is not None:
                    arr = self.robot_node.map_array
                    for y in range(arr.shape[0]):
                        for x in range(arr.shape[1]):
                            if arr[y, x] >= 100:
                                wx = self.robot_node.map_origin[0] + x * self.robot_node.map_resolution
                                wy = self.robot_node.map_origin[1] + y * self.robot_node.map_resolution
                                # 수거한 쓰레기 위치 근처는 장애물에서 제외
                                if not is_near_collected(wx, wy, collected_positions):
                                    obstacles.append([wx, wy])
                # MPC 호출
                x0 = [robot_pos[0], robot_pos[1], robot_yaw]
                goal = [target_position[0], target_position[1], np.arctan2(dy, dx)]
                v, w = self.mpc_control_with_scipy_full(x0, goal, obstacles=obstacles, N=7, dt=0.2)
                # 제약조건 적용
                v = float(np.clip(v, -2.0, 2.0))
                w = float(np.clip(w, -3.14, 3.14))
                self.robot_node.move_robot(MobileBaseCommander(linear_x=v, angular_z=w))
                time.sleep(sleep_time)
            # 목표점에서 정렬
            robot_pos = self.robot_node.get_robot_position()
            dx = target_position[0] - robot_pos[0]
            dy = target_position[1] - robot_pos[1]
            angle_to_obj = np.arctan2(dy, dx)
            robot_ori = self.robot_node.get_robot_orientation()
            qx, qy, qz, qw = robot_ori
            siny_cosp = 2 * (qw * qz + qx * qy)
            cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
            robot_yaw = np.arctan2(siny_cosp, cosy_cosp)
            while abs(angle_to_obj - robot_yaw) > 0.05:
                angle_diff = angle_to_obj - robot_yaw
                while angle_diff > np.pi:
                    angle_diff -= 2 * np.pi
                while angle_diff < -np.pi:
                    angle_diff += 2 * np.pi
                angular_z = float(np.clip(angle_diff, -0.6, 0.6))
                self.robot_node.move_robot(MobileBaseCommander(linear_x=0.0, angular_z=angular_z))

                time.sleep(0.05)
                robot_ori = self.robot_node.get_robot_orientation()
                qx, qy, qz, qw = robot_ori
                siny_cosp = 2 * (qw * qz + qx * qy)
                cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
                robot_yaw = np.arctan2(siny_cosp, cosy_cosp)
            self.robot_node.move_robot(MobileBaseCommander(linear_x=0.0, angular_z=0.0))
            self.logger.info('[TaskImplementation] 이동 및 정렬 완료')
        except Exception as e:
            self.logger.error(f'[TaskImplementation] 이동 중 예외: {e}')
            import traceback
            self.logger.error(traceback.format_exc())
            print(f'[EXCEPTION] {e}')
            print(traceback.format_exc())

    #   Demo implementation for obstacle detection
    #   TODO: Implement actual obstacle detection logic
    #   README: This function is only for demo implementation.
    #           Competitor must implement actual obstacle detection logic.  
    def _demo_implementation___is_movement_blocked(self, robot_position: List[float], linear_x: float, angular_z: float) -> bool:
        
        # TODO: Implement actual obstacle detection logic
        # This is a placeholder that should be replaced with actual sensor data
        # THis demo implementation assumes that there is no obstacle in front of the robot

        return False

    #   Demo implementation for get object center position using robot vision
    #   TODO: Implement actual object center position estimation logic
    #   README: This function is only for demo implementation.
    #           Competitor must implement actual object center position estimation logic.     
    """
    def _demo_implementation___get_object_center_position_using_robot_vision(self, object_detection: Dict, robot_capture_image: Image):
        
        Get the center position of an object using robot vision
        
        Returns:
            List[float]: [x, y, z] coordinates of the object center in the robot's camera coordinate system 
        

        return object_detection['position']
    """
    
    def _demo_implementation___get_object_center_position_using_robot_vision(
        self,
        object_detection: Dict,
    ) -> Dict[str, List[float]]:
        """
        YOLO + Depth + PCA 기반으로 지정된 객체의 월드 좌표 위치와 회전(quaternion) 추정

        Returns:
            Dict {
                "position": [x, y, z],          # 월드 좌표
                "quaternion": [qx, qy, qz, qw]  # 월드 회전
            }
        """
        import time
        import numpy as np
        from sklearn.decomposition import PCA
        from scipy.spatial.transform import Rotation as R

        # 카메라 오프셋 (로봇 기준 좌표계에서 카메라 위치)
        camera_offset_robot_frame = np.array([-0.047, 0.0, -0.617])

        # 1. 센서 데이터 수신 대기
        timeout = 5
        t_start = time.time()
        while (
            self.robot_node.rgb_image is None or
            self.robot_node.depth_image is None or
            self.robot_node.camera_info is None
        ):
            if time.time() - t_start > timeout:
                self.logger.error("[VISION] 센서 데이터 수신 실패 (5초 초과)")
                return {"position": [0.0, 0.0, 0.0], "quaternion": [0.0, 0.0, 0.0, 1.0]}
            time.sleep(0.1)

        rgb = self.robot_node.rgb_image.copy()
        depth = self.robot_node.depth_image.copy()
        cam_info = self.robot_node.camera_info
        fx, fy = cam_info.k[0], cam_info.k[4]
        cx, cy = cam_info.k[2], cam_info.k[5]

        # 2. YOLO 추론
        results = self.yolo_model(rgb)
        detections = results[0]

        target_class = object_detection['class_name']
        target_world_pos = np.array(object_detection['position'][:2])

        # 3. 로봇 pose 추출
        robot_pos = np.array(self.robot_node.get_robot_position())
        robot_ori = np.array(self.robot_node.get_robot_orientation())
        rot_robot = R.from_quat(robot_ori)

        # 4. 가장 가까운 bbox 선택
        closest_box = None
        min_dist = float('inf')
        for box, cls_id in zip(detections.boxes.xyxy, detections.boxes.cls):
            cls_name = self.yolo_model.names[int(cls_id)]
            if cls_name != target_class:
                continue

            x1, y1, x2, y2 = box.int().tolist()
            u_center = int((x1 + x2) / 2)
            v_center = int((y1 + y2) / 2)
            z = float(depth[v_center, u_center]) / 1000.0
            if z == 0.0 or np.isnan(z):
                continue

            x = (u_center - cx) * z / fx
            y = (v_center - cy) * z / fy

            # 카메라 기준 쓰레기 상대 좌표(옆, 위, 깊이)
            pos_cam = np.array([x, y, z])
            # 로봇 기준 쓰레기 상대 좌표
            pos_robot = pos_cam + camera_offset_robot_frame
            # 월드 기준 쓰레기 절대 좌표
            pos_world = rot_robot.apply(pos_robot) + robot_pos

            # 정답지의 쓰레기 위치와 로봇에서 추정한 쓰레기 위치의 거리
            dist = np.linalg.norm(pos_world[:2] - target_world_pos)

            # 충분히 가까우면 가장 가까운 bbox로 선택
            if dist < min_dist:
                min_dist = dist
                closest_box = [x1, y1, x2, y2]

        if closest_box is None:
            self.logger.warning(f"[YOLO] '{target_class}' 객체를 주변에서 찾을 수 없음")
            return {"position": [0.0, 0.0, 0.0], "quaternion": [0.0, 0.0, 0.0, 1.0]}

        # 5. bbox 내부 포인트 3D 복원
        x1, y1, x2, y2 = closest_box
        points_3d = []
        for v in range(y1, y2):
            for u in range(x1, x2):
                z = float(depth[v, u]) / 1000.0
                if z == 0.0 or np.isnan(z):
                    continue
                x = (u - cx) * z / fx
                y = (v - cy) * z / fy
                points_3d.append([x, y, z])

        if len(points_3d) < 10:
            self.logger.warning("[PCA] 유효한 포인트 부족 → 회전 추정 실패")
            return {"position": [0.0, 0.0, 0.0], "quaternion": [0.0, 0.0, 0.0, 1.0]}

        points_3d = np.array(points_3d)
        points_3d_robot = points_3d + camera_offset_robot_frame

        # 6. PCA로 회전 추정
        pca = PCA(n_components=3)
        pca.fit(points_3d_robot)
        principal_axis = pca.components_[0]  # 가장 긴 축

        x_axis = principal_axis
        z_axis = np.array([0, 0, 1])
        y_axis = np.cross(z_axis, x_axis)
        y_axis /= np.linalg.norm(y_axis)
        x_axis = np.cross(y_axis, z_axis)
        x_axis /= np.linalg.norm(x_axis)
        rot_mat = np.column_stack((x_axis, y_axis, z_axis))
        rot_obj = R.from_matrix(rot_mat)

        # 7. 위치 및 자세 (월드 좌표계 기준)
        object_pos_robot = np.mean(points_3d_robot, axis=0)
        object_pos_world = rot_robot.apply(object_pos_robot) + robot_pos
        object_quat_world = (rot_robot * rot_obj).as_quat()

        self.logger.info(f"[VISION] '{target_class}' 위치: {object_pos_world}, 회전(quat): {object_quat_world}")

        return {
            "position": object_pos_world.tolist(),
            "quaternion": object_quat_world.tolist(),
            "closest_box": closest_box
        }

    #   Demo implementation for pick object
    #   TODO: Implement actual pick object logic
    #   README: This function is only for demo implementation.
    #           Competitor must implement actual pick object logic.   
    
    def _demo_implementation___pick_object(self, object_detection: Dict) -> None:
        """
        로봇 비전 기반 객체 위치 및 회전 추정 후 물체를 집는 함수.
        입력으로 받은 object_detection 객체를 기준으로 해당 물체만 인식하여 픽업 수행.
        """
        time.sleep(2)  # 픽업 준비 시간

        # 1. 비전 기반 객체 중심 위치 및 회전 추정 (월드 좌표 기준)
        center_result = self._demo_implementation___get_object_center_position_using_robot_vision(object_detection)

        object_pos_world = np.array(center_result["position"])      # 월드 좌표 [x, y, z]
        object_quat_world = center_result["quaternion"]             # 월드 회전 쿼터니언 [x, y, z, w]
        closest_box = center_result.get("closest_box", None)
        # 2. 로봇 현재 위치/자세 가져오기
        robot_pos_world = np.array(self.robot_node.get_robot_position())
        robot_quat_world = np.array(self.robot_node.get_robot_orientation())
        rot_robot = R.from_quat(robot_quat_world)

        # 3. grasp 방향 계산 (접근 벡터)
        delta_pos = object_pos_world - robot_pos_world
        delta_pos[2] = 0  # Z축 무시 (XY 평면 기준)
        if np.linalg.norm(delta_pos) < 1e-3:
            delta_pos = np.array([1.0, 0.0, 0.0])  # fallback

        approach_dir = delta_pos / np.linalg.norm(delta_pos)

        # 접근 방향을 기준으로 회전 행렬 생성
        x_axis = approach_dir
        z_axis = np.array([0, 0, -1])
        y_axis = np.cross(z_axis, x_axis)
        y_axis /= np.linalg.norm(y_axis)
        x_axis = np.cross(y_axis, z_axis)

        grasp_rot_matrix = np.column_stack((x_axis, y_axis, z_axis))
        grasp_quat_world = R.from_matrix(grasp_rot_matrix).as_quat()

        # 4. 로봇 기준 상대 좌표 및 회전 변환
        relative_pos = rot_robot.inv().apply(object_pos_world - robot_pos_world)
        relative_pos[2] = -0.24  # 픽업 높이 보정

        relative_quat = (R.from_quat(grasp_quat_world) * rot_robot.inv()).as_quat()

        # 5. 너무 멀면 이동 (0.75m 이상)
        dist = np.linalg.norm(relative_pos[:2])
        if dist > CONST_GLOBAL_PICK_DISTANCE:
            self.logger.info(f"[로봇 이동] 쓰레기까지의 거리 {dist:.2f} > 0.75 → 이동 시작")
            self._demo_implementation___locate_robot_to_target_position(
                object_pos_world.tolist(),
                CONST_GLOBAL_PICK_DISTANCE,
                math.pi / 720,
                0.1
            )
            time.sleep(2)  # 이동 안정화 대기

            # 이동 후 다시 계산
            robot_pos_world = np.array(self.robot_node.get_robot_position())
            robot_quat_world = np.array(self.robot_node.get_robot_orientation())
            rot_robot = R.from_quat(robot_quat_world)

            relative_pos = rot_robot.inv().apply(object_pos_world - robot_pos_world)
            relative_pos[0] = abs(relative_pos[0])
            relative_pos[2] = -0.24

            relative_quat = (R.from_quat(grasp_quat_world) * rot_robot.inv()).as_quat()

        # 6. 재질에 따라 드롭 위치 분류
        class_name = object_detection['class_name']
        drop_position = 0.0
        if class_name in ['master_chef_can', 'cola_can']:
            drop_position = -0.15
        elif class_name in ['juice', 'disposable_cup']:
            drop_position = 0.0
        elif class_name in ['tissue', 'cracker_box']:
            drop_position = 0.15

        # 7. 픽업 및 드롭 명령 생성
        pick_and_place_command = ManipulatorCommander(
            start_target_orientation_quat=relative_quat.tolist(),
            start_target_position=relative_pos.tolist(),
            end_target_orientation_quat=[1, 0, 0, 0],
            end_target_position=[0.5, drop_position, -0.1]
        )

        # 8. 픽업 수행
        self.robot_node.pick_up_object(pick_and_place_command)
        self.logger.info(f"[검증] 실제 쓰레기 위치 (정답지): {object_detection['position']}")
        self.logger.info(f"[검증] 실제 쓰레기 회전 (정답지): {object_detection['rotation']}")
        # 2. 추정 위치 및 회전 정보
        from scipy.spatial.transform import Rotation as R
        quat = object_quat_world
        euler_deg = R.from_quat(quat).as_euler('xyz', degrees=True)
        self.logger.info(f"[검증] 추정된 쓰레기 위치 (YOLO+Depth): {object_pos_world.tolist()}")
        self.logger.info(f"[검증] 추정된 회전 (Quaternion): {quat}")
        self.logger.info(f"[검증] 추정된 회전 (Euler XYZ degrees): {euler_deg.tolist()}")

        # 3. 네 모서리 depth 출력
        if closest_box is not None:
            x1, y1, x2, y2 = map(int, closest_box)
            corners = [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]
            for i, (u, v) in enumerate(corners):
                if 0 <= v < self.robot_node.depth_image.shape[0] and 0 <= u < self.robot_node.depth_image.shape[1]:
                    d = float(self.robot_node.depth_image[v, u]) / 1000.0
                    self.logger.info(f'[검증] bbox corner {i+1}: (u={u}, v={v}) → depth = {d:.3f}m')
                else:
                    self.logger.info(f"[검증] bbox corner {i+1}: (u={u}, v={v}) → 범위 초과")

        # 4. 로봇 현재 위치 출력
        self.logger.info(f"[검증] 로봇 위치: {robot_pos_world.tolist()}")
        # 9. 동작 간 대기 시간
        time.sleep(20)

    # === 경로 추종 제어 (MPC) ===
    def follow_path_and_pick_objects(self, pickup_infos, object_order, recyclable_objects, angle_threshold=math.radians(10)):
        collected_positions = []  # 이미 수거한 쓰레기 위치들

        # 픽업 위치와 헤딩 정보들에 대해
        for i, ((tx, ty), ttheta) in enumerate(pickup_infos):
            if self.force_stop:
                break
            
            # 최종 거리 정렬
            while True:
                pos = self.robot_node.get_robot_position()
                ori = self.robot_node.get_robot_orientation()
                # 쿼터니언 → yaw 변환
                qx, qy, qz, qw = ori
                siny_cosp = 2 * (qw * qz + qx * qy)
                cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
                yaw = math.atan2(siny_cosp, cosy_cosp)

                # 픽업 위치와 현재 위치 간 거리 및 각도 계산
                dx, dy = tx - pos[0], ty - pos[1]
                dist = math.hypot(dx, dy)
                target_heading = math.atan2(dy, dx)

                # 픽업 위치와 거리가 충분히 가까워질 때까지 반복
                if dist < 0.3:
                    break
                
                # 현재 로봇의 pose와 픽업 pose
                x0 = [pos[0], pos[1], yaw]
                goal = [tx, ty, target_heading]
                obstacles = []

                # 각 재활용 쓰레기의 인덱스와 정보에 대해
                for j, obj in enumerate(recyclable_objects):
                    # 현재 수거해야할 쓰레기가 아니면 장애물 리스트에 추가
                    if j != object_order[i]:
                        pos_j = obj['position'][:2]
                        if pos_j not in collected_positions:
                            obstacles.append({'center': np.array(pos_j), 'radius': 0.2})
                # 재활용 불가 쓰레기도 장애물 리스트에 추가
                for obj in self.object_detection_result:
                    if not obj['recyclable']:
                        obstacles.append({'center': np.array(obj['position'][:2]), 'radius': 0.2})

                # MPC 제어 호출
                # 장애물 회피, 입력 변화 최소화, 마지막 θ 정렬 포함
                # v, w는 각각 선형 속도와 각속도
                v, w = self.mpc_control_with_scipy_full(x0, goal, obstacles=[o['center'] for o in obstacles])
                v = float(np.clip(v, -2.0, 2.0))
                w = float(np.clip(w, -3.14, 3.14))

                # MPC 제어 결과를 이용해 로봇 이동
                self.robot_node.move_robot(MobileBaseCommander(linear_x=v, angular_z=w))
                time.sleep(0.1)

            # 최종 헤딩 정렬
            while True:

                # 쿼터니언 → yaw 변환
                ori = self.robot_node.get_robot_orientation()
                qx, qy, qz, qw = ori
                siny_cosp = 2 * (qw * qz + qx * qy)
                cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
                yaw = math.atan2(siny_cosp, cosy_cosp)

                # 픽업 헤딩과 현재 헤딩 간의 각도 차이 계산
                angle_diff = ttheta - yaw
                while angle_diff > math.pi:
                    angle_diff -= 2 * math.pi
                while angle_diff < -math.pi:
                    angle_diff += 2 * math.pi

                # 각도 차이가 충분히 작으면 정렬 완료
                if abs(angle_diff) < angle_threshold:
                    break
                
                # 각도 차이에 따라 회전 속도 조정
                w = float(np.clip(angle_diff, -0.6, 0.6))
                # 로봇 회전
                self.robot_node.move_robot(MobileBaseCommander(linear_x=0.0, angular_z=w))
                time.sleep(0.05)

            # 목표 거리와 헤딩에 도달하면 정지
            self.robot_node.move_robot(MobileBaseCommander(linear_x=0.0, angular_z=0.0))
            self.logger.info(f"[MPC] 쓰레기 {i+1} 위치 도달 및 정렬 완료")

            # 수거할 쓰레기 객체 정보
            obj = recyclable_objects[object_order[i]]
            object_detection = {
                'position': obj['position'],
                'class_name': obj['class_name'],
                'rotation': obj.get('rotation', [0, 0, 0]),
                'recyclable': True
            }
            self._demo_implementation___pick_object(object_detection)
            collected_positions.append(obj['position'][:2])  # 수거한 쓰레기 위치 추가
            self.logger.info(f"[Stage2] 쓰레기 #{i+1} 수거 완료, 대기 중...")
            time.sleep(120)

    #   Demo implementation for select next object to pick
    #   TODO: Implement actual select next object to pick logic
    #   README: This function is only for demo implementation.
    #           Competitor must implement actual select next object to pick logic.     
    def _demo_implementation___select_next_object(self):

        #   initialize recyclable_object_list
        if not hasattr(self, 'recyclable_object_list'):
            self.recyclable_object_list = [obj for obj in self.object_detection_result if obj['recyclable']]
            
        #   if there is no recyclable object, return None
        if not self.recyclable_object_list:
            return None
            
        #   get current robot position
        robot_position = self.robot_node.get_robot_position()
        
        #   find the closest object
        min_distance = float('inf')
        closest_object = None
        closest_idx = -1
        
        for idx, obj in enumerate(self.recyclable_object_list):
            obj_position = obj['position']
            distance = math.sqrt(
                (obj_position[0] - robot_position[0])**2 + 
                (obj_position[1] - robot_position[1])**2
            )
            
            if distance < min_distance:
                min_distance = distance
                closest_object = obj
                closest_idx = idx
                
        #   select the closest object and remove it from the list
        if closest_idx >= 0:
            self.recyclable_object_list.pop(closest_idx)
            
        return closest_object

    def stop_task_and_quit(self) :
        super().stop_task_and_quit()
        self.robot_node.stop_and_destroy_robot()


    #   Demo implementation for stage 2 task
    #   TODO: Implement actual stage 2 task logic
    #   README: This function is only for demo implementation.
    #           Competitor must implement actual stage 2 task logic.     
    def start_stage_2_task(self) -> bool:
        self.logger.info('[TaskImplementation] Stage 2 시작')

        if not self.object_detection_result:
            self.logger.warn("  - Stage 1 결과가 존재하지 않습니다.")
            return False

        recyclable_objects = [obj for obj in self.object_detection_result if obj['recyclable']]
        non_recyclable_objects = [obj for obj in self.object_detection_result if not obj['recyclable']]

        if not recyclable_objects:
            self.logger.info("  - 수거할 재활용 쓰레기가 없습니다.")
            return False

        trash_positions = np.array([obj['position'][:2] for obj in recyclable_objects])
        obstacle_list = [{'center': np.array(obj['position'][:2]), 'radius': 0.2} for obj in self.object_detection_result]

        order, _ = genetic_algorithm(trash_positions)
        ordered_trash_positions = trash_positions[order]
        start_position = self.robot_node.get_robot_position()[:2]
        # final path는 안 쓰임?
        final_path, pickup_infos = generate_final_path_with_frontal_pickup(ordered_trash_positions, obstacle_list, start_position)

        self.follow_path_and_pick_objects(pickup_infos, order, recyclable_objects)
        self.logger.info("[Stage2] 모든 쓰레기 수거 완료.")
        return True
