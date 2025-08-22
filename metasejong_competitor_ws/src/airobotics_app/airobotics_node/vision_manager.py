# === vision_manager.py ===

import time
import numpy as np
from scipy.spatial.transform import Rotation as R
from sklearn.decomposition import PCA
from .robot_node import MobileBaseCommander
from pathlib import Path
import cv2
import torch
from .mlp_model import ResidualMLP

class VisionManager:
    def __init__(self, robot_node, yolo_model, logger, *, collect_mode: bool = False, save_dir: str = "./data"):
        self.robot_node = robot_node
        self.yolo_model = yolo_model
        self.logger = logger
        self.camera_offset_robot_frame = np.array([0.41, -0.01067, 0.09926])
        self.collect_mode = collect_mode
        self.save_dir = Path(save_dir)
        if self.collect_mode:
            self.save_dir.mkdir(parents=True, exist_ok=True)
        # === MLP 로드 (일단 비활성화) ===
        self.mlp_model = None
        self.logger.info("[MLP] 비활성화 - 기하학적 추정만 사용")

    def _save_sample(self, bbox, depth_arr, estimated_pos, gt_pos):
        import json, numpy as np, datetime as dt

        idx = len(list(self.save_dir.glob("metadata_*.json")))
        depth_path = self.save_dir / f"depth_image_{idx:05d}.npy"
        meta_path  = self.save_dir / f"metadata_{idx:05d}.json"
        rgb_path = self.save_dir / f"rgb_image_{idx:05d}.png"

        np.save(depth_path, depth_arr)
        bgr_image = cv2.cvtColor(self.robot_node.rgb_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(rgb_path), bgr_image)
        metadata = {
            "bounding_box": list(map(int, bbox)),
            "pixel_center": [ (bbox[0]+bbox[2])//2, (bbox[1]+bbox[3])//2 ],
            "camera_intrinsics": { "K": self.robot_node.camera_info.k.reshape(3,3).tolist() },
            "robot_pose": {
                "position": self.robot_node.get_robot_position(),
                "orientation_quat": self.robot_node.get_robot_orientation(),
            },
            "estimated_position": estimated_pos,
            "ground_truth_position": gt_pos,
            "timestamp": dt.datetime.now().isoformat(),
        }
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

        self.logger.info(f"✅ 샘플 저장 → {meta_path.name}, {depth_path.name}")

    def get_object_pose(self, object_detection: dict) -> dict:
        """
        YOLO + Depth + PCA 기반으로 지정된 객체의 월드 좌표 위치와 회전(quaternion) 추정
        """
        timeout = 5
        t_start = time.time()
        while (
            self.robot_node.rgb_image is None or
            self.robot_node.depth_image is None or
            self.robot_node.camera_info is None
        ):
            if time.time() - t_start > timeout:
                self.logger.error("[VISION] 센서 데이터 수신 실패 (5초 초과)")
                return {"position": [0,0,0], "quaternion": [0,0,0,1]}
            time.sleep(0.1)

        rgb = self.robot_node.rgb_image.copy()
        depth = self.robot_node.depth_image
        cam_info = self.robot_node.camera_info
        fx, fy = cam_info.k[0], cam_info.k[4]
        cx, cy = cam_info.k[2], cam_info.k[5]

        results = self.yolo_model(rgb)
        detections = results[0]
        target_class = object_detection['class_name']
        target_world_xy = np.array(object_detection['position'][:2])

        robot_pos = np.array(self.robot_node.get_robot_position())
        robot_ori = np.array(self.robot_node.get_robot_orientation())
        rot_robot = R.from_quat(robot_ori)

        closest_box = None
        min_dist = float('inf')
        best_pos_world = None

        for box, cls_id in zip(detections.boxes.xyxy, detections.boxes.cls):
            if self.yolo_model.names[int(cls_id)] != target_class:
                continue

            x1, y1, x2, y2 = map(int, box.int().tolist())
            roi = depth[y1:y2, x1:x2].flatten()
            valid = roi[np.isfinite(roi) & (roi > 0.1) & (roi < 5.0)]
            if valid.size == 0:
                continue
            z_med = float(np.median(valid))

            u_center = int((x1 + x2) / 2)
            v_center = int((y1 + y2) / 2)
            
            # 중심점의 실제 depth 사용 (핵심 수정)
            z_center = depth[v_center, u_center] if (0 <= v_center < depth.shape[0] and 
                                                    0 <= u_center < depth.shape[1] and 
                                                    np.isfinite(depth[v_center, u_center]) and 
                                                    depth[v_center, u_center] > 0.1) else z_med
            
            self.logger.info(f"[DEPTH] 픽셀({u_center},{v_center}): 중심점={depth[v_center, u_center]:.3f}m, 박스median={z_med:.3f}m, 사용={z_center:.3f}m")
            
            # 1. 픽셀 → 카메라 3D 좌표
            Xo = (u_center - cx) * z_center / fx
            Yo = (v_center - cy) * z_center / fy
            Zo = z_center
            self.logger.info(f"[좌표변환1] 카메라 3D: X={Xo:.3f}, Y={Yo:.3f}, Z={Zo:.3f}")
            
            # 2. 카메라 → 로봇 기준 좌표
            pos_cam_robot = np.array([Zo, -Xo, -Yo]) + self.camera_offset_robot_frame
            self.logger.info(f"[좌표변환2] 로봇 기준: X={pos_cam_robot[0]:.3f}, Y={pos_cam_robot[1]:.3f}, Z={pos_cam_robot[2]:.3f}")
            self.logger.info(f"[좌표변환2] 로봇 기준 거리: {np.linalg.norm(pos_cam_robot):.3f}m")
            
            # 3. 로봇 기준 → 월드 절대좌표
            pos_world = rot_robot.apply(pos_cam_robot) + robot_pos
            self.logger.info(f"[좌표변환3] 월드 절대: X={pos_world[0]:.3f}, Y={pos_world[1]:.3f}, Z={pos_world[2]:.3f}")
            
            # 4. 검증: 월드 → 다시 상대좌표로 계산한 거리
            relative_from_world = pos_world - robot_pos
            distance_from_world = np.linalg.norm(relative_from_world)
            self.logger.info(f"[좌표변환4] 월드→상대 재계산 거리: {distance_from_world:.3f}m")
            self.logger.info(f"[오차분석] 직접계산 vs 재계산: {np.linalg.norm(pos_cam_robot):.3f}m vs {distance_from_world:.3f}m (차이: {abs(np.linalg.norm(pos_cam_robot) - distance_from_world):.6f}m)")

            dist = np.linalg.norm(pos_world[:2] - target_world_xy)
            if dist < min_dist:
                min_dist = dist
                closest_box = (x1, y1, x2, y2)
                best_pos_world = pos_world

        if closest_box is None:
            self.logger.warning(f"[YOLO] '{target_class}' 객체를 찾을 수 없음")
            return {"position": [0,0,0], "quaternion": [0,0,0,1]}

        x1, y1, x2, y2 = closest_box
        points_3d = []
        for v in range(y1, y2):
            for u in range(x1, x2):
                z = float(depth[v, u])
                if z <= 0 or np.isnan(z): continue
                Xo = (u - cx) * z / fx
                Yo = (v - cy) * z / fy
                Zo = z
                pr = np.array([Zo, -Xo, -Yo]) + self.camera_offset_robot_frame
                points_3d.append(pr)

        if len(points_3d) < 10:
            self.logger.warning("[PCA] 유효 포인트 부족 → 회전 추정 실패")
            return {"position": best_pos_world.tolist(), "quaternion": [0,0,0,1]}

        pts = np.vstack(points_3d)
        pca = PCA(n_components=3)
        pca.fit(pts)
        principal_axis = pca.components_[0]
        x_axis = principal_axis / np.linalg.norm(principal_axis)
        z_axis = np.array([0, 0, 1])
        y_axis = np.cross(z_axis, x_axis)
        y_axis /= np.linalg.norm(y_axis)
        x_axis = np.cross(y_axis, z_axis)
        x_axis /= np.linalg.norm(x_axis)

        rot_obj = R.from_matrix(np.column_stack((x_axis, y_axis, z_axis)))
        object_quat_world = (rot_robot * rot_obj).as_quat()
        self.logger.info(f"[이게 로봇기준 쿼터니언 월드 아님 이름만 월드]: {object_quat_world.tolist()}")
        result = {
            "position": best_pos_world.tolist(),
            "quaternion": object_quat_world.tolist(),
            "closest_box": list(closest_box)
        }

        # [추가] 수집 모드일 경우 Ground Truth 입력 받고 저장
        if self.collect_mode:
            self.logger.info("📝 G.T. 월드 좌표(x y z)를 입력하세요 (예: -64.2 132.8 0.05):")
            try:
                #t = list(map(float, input("GT > ").strip().split()))
                gt =[0.0, 0.0, 0.0]
                self._save_sample(closest_box, depth,best_pos_world.tolist(), gt)
            except Exception as e:
                self.logger.error(f"GT 입력 실패: {e}")
        # MLP 보정 비활성화 - 기하학적 추정만 사용
        # if self.mlp_model:
        #     mlp_input = torch.tensor([[u_center, v_center, best_pos_world[0], best_pos_world[1]]], dtype=torch.float32)
        #     correction = self.mlp_model(mlp_input).detach().numpy().flatten()
        #     result["position"] = (best_pos_world + correction).tolist()
        #     self.logger.info(f"[MLP 보정 결과] x={correction[0]:.3f}, y={correction[1]:.3f}")
        
        # 기하학적 추정값만 사용
        result["position"] = best_pos_world.tolist()

        return result
    def compute_grasp_quaternion(self, object_quat_world: list, angle_offset_deg: float = 90.0) -> list:
        """
        물체의 회전 쿼터니언(object_quat_world)을 기반으로
        로봇 집게가 수직 방향으로 접근할 수 있는 회전 쿼터니언 반환
        → Isaac Sim 포맷: [w, x, y, z]
        """
        rot_obj = R.from_quat(object_quat_world)
        x_dir = rot_obj.apply([1, 0, 0])
        angle_rad = np.arctan2(x_dir[1], x_dir[0])
        angle_deg = np.degrees(angle_rad)
        grasp_angle = angle_deg + angle_offset_deg

        rot_gripper = R.from_euler('x', grasp_angle, degrees=True)
        q = rot_gripper.as_quat()  # [x, y, z, w]

        return [q[3], q[0], q[1], q[2]]  # Isaac Sim 포맷 [w, x, y, z]
    def log_vector(logger, label: str, vec, precision: int = 4):
        """배열이나 벡터를 보기 좋게 로깅하는 유틸 함수"""
        if isinstance(vec, np.ndarray):
            vec = vec.tolist()
        formatted = [round(float(v), precision) for v in vec]
        logger.info(f"[{label}]: {formatted}")

    def center_align(self, object_detection: dict):
        rgb = self.robot_node.rgb_image.copy()
        depth = self.robot_node.depth_image
        cam_info = self.robot_node.camera_info
        fx, cx = cam_info.k[0], cam_info.k[2]

        results = self.yolo_model(rgb)
        detections = results[0]
        target_class = object_detection['class_name']
        target_world_xy = np.array(object_detection['position'][:2])

        best_box = None
        min_dist = float('inf')
        for box, cls_id in zip(detections.boxes.xyxy, detections.boxes.cls):
            if self.yolo_model.names[int(cls_id)] != target_class:
                continue

            x1, y1, x2, y2 = map(int, box.int().tolist())
            cx_box = int((x1 + x2) / 2)
            cy_box = int((y1 + y2) / 2)
            d = depth[cy_box, cx_box]
            if not np.isfinite(d) or not (0.1 < d < 5.0):
                continue

            # 카메라 기준 3D 좌표 추정
            x = (cx_box - cx) * d / fx
            y = (cy_box - cam_info.k[5]) * d / cam_info.k[4]
            est_xy = self.robot_node.get_robot_position()[:2] + np.array([x, y])
            dist = np.linalg.norm(est_xy - target_world_xy)
            if dist < min_dist:
                min_dist = dist
                best_box = box
                best_u_center = cx_box

        if best_box is None:
            self.logger.warn("[center_align] 대상 클래스 탐지 실패 또는 유효한 Depth 없음")
            return

        # === PID 기반 회전 정렬 ===
        Kp = 0.002
        max_speed = 0.25
        tol_pixel = 5

        while True:
            rgb = self.robot_node.rgb_image.copy()
            results = self.yolo_model(rgb)
            detections = results[0]

            u_center = None
            best_dist = float('inf')
            for box, cls_id in zip(detections.boxes.xyxy, detections.boxes.cls):
                if self.yolo_model.names[int(cls_id)] != target_class:
                    continue
                x1, y1, x2, y2 = map(int, box.int().tolist())
                cx_box = int((x1 + x2) / 2)
                cy_box = int((y1 + y2) / 2)
                
                # depth 확인 및 거리 계산
                d = depth[cy_box, cx_box]
                if not np.isfinite(d) or not (0.1 < d < 5.0):
                    continue
                    
                # 목표 위치와의 거리 비교
                x_cam = (cx_box - cx) * d / fx
                y_cam = (cy_box - cam_info.k[5]) * d / cam_info.k[4]
                est_xy = self.robot_node.get_robot_position()[:2] + np.array([x_cam, y_cam])
                dist = np.linalg.norm(est_xy - target_world_xy)
                
                if dist < best_dist:
                    best_dist = dist
                    u_center = cx_box

            if u_center is None:
                self.logger.warn("[center_align] 회전 중 객체 사라짐")
                break

            error = u_center - cx
            if abs(error) <= tol_pixel:
                break  # 중심 정렬 완료

            angular_z = np.clip(Kp * error, -max_speed, max_speed)
            self.robot_node.move_robot(MobileBaseCommander(linear_x=0.0, angular_z=angular_z*(-1)))
            time.sleep(0.05)

        self.robot_node.move_robot(MobileBaseCommander(0.0, 0.0))  # 정지
        
        # === 거리 조정 단계 추가 ===
        self.logger.info("[center_align] 회전 완료, 거리 조정 시작")
        target_distance = 0.8  # 목표 거리 (0.7-0.9m 중간값)
        distance_tolerance = 0.08  # 거리 허용 오차 (완화)
        
        while True:
            # 현재 거리 측정 (3D 유클리드 거리)
            rgb = self.robot_node.rgb_image.copy()
            depth = self.robot_node.depth_image
            cam_info = self.robot_node.camera_info
            fx, fy = cam_info.k[0], cam_info.k[4]
            cx, cy = cam_info.k[2], cam_info.k[5]
            
            results = self.yolo_model(rgb)
            detections = results[0]
            
            current_distance = None
            for box, cls_id in zip(detections.boxes.xyxy, detections.boxes.cls):
                if self.yolo_model.names[int(cls_id)] != target_class:
                    continue
                x1, y1, x2, y2 = map(int, box.int().tolist())
                cx_box = int((x1 + x2) / 2)
                cy_box = int((y1 + y2) / 2)
                
                # depth 값 추출
                z = depth[cy_box, cx_box]
                if not (np.isfinite(z) and z > 0.1):
                    continue
                
                # 3D 카메라 좌표 계산
                x_cam = (cx_box - cx) * z / fx
                y_cam = (cy_box - cy) * z / fy
                z_cam = z
                
                # 카메라 → 로봇 좌표계 변환 (카메라 오프셋 고려)
                pos_cam_robot = np.array([z_cam, -x_cam, -y_cam]) + self.camera_offset_robot_frame
                
                # 로봇 중심에서 객체까지의 거리 계산
                current_distance = np.linalg.norm(pos_cam_robot)
                break
            
            if current_distance is None:
                self.logger.warn("[center_align] 거리 측정 실패")
                break
                
            distance_error = current_distance - target_distance
            self.logger.info(f"[center_align] 현재거리: {current_distance:.3f}m, 목표: {target_distance:.3f}m, 오차: {distance_error:.3f}m")
            
            if abs(distance_error) <= distance_tolerance:
                self.logger.info(f"[center_align] 거리 조정 완료: {current_distance:.3f}m")
                break
            
            # 거리 조정 (전진/후진) - 속도 중간값
            linear_speed = np.clip(distance_error * 0.4, -0.3, 0.3)  # 적당한 속도
            self.robot_node.move_robot(MobileBaseCommander(linear_x=linear_speed, angular_z=0.0))
            time.sleep(0.05)
        
        self.robot_node.move_robot(MobileBaseCommander(0.0, 0.0))  # 최종 정지
        
        # === 재정렬 단계 (거리 조정 후 각도가 틀어졌을 수 있음) ===
        self.logger.info("[center_align] 거리 조정 완료, 재정렬 시작")
        
        while True:
            rgb = self.robot_node.rgb_image.copy()
            results = self.yolo_model(rgb)
            detections = results[0]

            u_center = None
            best_dist = float('inf')
            for box, cls_id in zip(detections.boxes.xyxy, detections.boxes.cls):
                if self.yolo_model.names[int(cls_id)] != target_class:
                    continue
                x1, y1, x2, y2 = map(int, box.int().tolist())
                cx_box = int((x1 + x2) / 2)
                cy_box = int((y1 + y2) / 2)
                
                # depth 확인 및 거리 계산
                d = depth[cy_box, cx_box]
                if not np.isfinite(d) or not (0.1 < d < 5.0):
                    continue
                    
                # 목표 위치와의 거리 비교
                x_cam = (cx_box - cx) * d / fx
                y_cam = (cy_box - cam_info.k[5]) * d / cam_info.k[4]
                est_xy = self.robot_node.get_robot_position()[:2] + np.array([x_cam, y_cam])
                dist = np.linalg.norm(est_xy - target_world_xy)
                
                if dist < best_dist:
                    best_dist = dist
                    u_center = cx_box

            if u_center is None:
                self.logger.warn("[center_align] 재정렬 중 객체 사라짐")
                break

            error = u_center - cx
            if abs(error) <= tol_pixel:
                self.logger.info("[center_align] 재정렬 완료")
                break  # 중심 정렬 완료

            angular_z = np.clip(Kp * error, -max_speed, max_speed)
            self.robot_node.move_robot(MobileBaseCommander(linear_x=0.0, angular_z=angular_z*(-1)))
            time.sleep(0.05)

        self.robot_node.move_robot(MobileBaseCommander(0.0, 0.0))  # 최종 정지
        self.logger.info("[center_align] 회전 및 거리 조정 완료")


