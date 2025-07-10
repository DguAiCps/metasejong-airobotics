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
        self.camera_offset_robot_frame = np.array([-0.047, 0.0, -0.617])
        self.collect_mode = collect_mode
        self.save_dir = Path(save_dir)
        if self.collect_mode:
            self.save_dir.mkdir(parents=True, exist_ok=True)
        # === MLP 로드 ===
        model_path = Path(__file__).resolve().parent / ".." / "resource" / "mlp_model.pth"
        self.mlp_model = ResidualMLP()
        self.mlp_model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        self.mlp_model.eval()
        self.logger.info(f"[MLP 보정 모델 로딩 성공]: {model_path}")

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
            Xo = (u_center - cx) * z_med / fx
            Yo = (v_center - cy) * z_med / fy
            Zo = z_med
            pos_cam_robot = np.array([Zo, -Xo, -Yo]) + self.camera_offset_robot_frame
            pos_world = rot_robot.apply(pos_cam_robot) + robot_pos

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
        if self.mlp_model:
                mlp_input = torch.tensor([[u_center, v_center, best_pos_world[0], best_pos_world[1]]], dtype=torch.float32)
                correction = self.mlp_model(mlp_input).detach().numpy().flatten()
                result["position"] = (best_pos_world + correction).tolist()
                self.logger.info(f"[MLP 보정 결과] x={correction[0]:.3f}, y={correction[1]:.3f}")

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
        Kp = 0.003
        max_speed = 0.4
        tol_pixel = 5

        while True:
            rgb = self.robot_node.rgb_image.copy()
            results = self.yolo_model(rgb)
            detections = results[0]

            u_center = None
            for box, cls_id in zip(detections.boxes.xyxy, detections.boxes.cls):
                if self.yolo_model.names[int(cls_id)] != target_class:
                    continue
                x1, y1, x2, y2 = map(int, box.int().tolist())
                u_center = int((x1 + x2) / 2)
                break  # 가장 처음 찾은 것 사용

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


