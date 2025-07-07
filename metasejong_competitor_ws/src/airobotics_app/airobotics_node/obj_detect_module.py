#!/usr/bin/env python3
from pixel_to_cor import *
from single_capture import capture_single_image
from pathlib import Path
from topic_collect import collect_demo_data
import math
import subprocess
import time
import os
from collections import defaultdict
from ultralytics import YOLO
import shutil
import signal

def detect_objects() -> dict[str, list[dict[str, list[float]]]]:
    """
    Capture images from cameras, detect objects using YOLO, and return object centers.

    Returns:
        defaultdict: Dictionary with object class names as keys and list of center coordinates as values
                    Example: {'juice': [(100.5, 200.3), (150.2, 180.7)], 'apple': [(300.1, 400.2)]}
    """
    print("오브젝트 감지 시작")

    # Capture images from cameras
    capture_single_image('/metasejong2025/cameras/demo_1/image_raw', '1.jpg')
    capture_single_image('/metasejong2025/cameras/demo_2/image_raw', '2.jpg')
    capture_single_image('/metasejong2025/cameras/demo_3/image_raw', '3.jpg')
    q2star = np.array([ 0.5, -0.5,  -0.5,  0.5])
    k, demo1_pos, demo1_rot, demo2_pos, demo2_rot = collect_demo_data()
    # Initialize object centers storage
    object_centers = defaultdict(list)
    real_rot_1 = qmul([demo1_rot['x'], demo1_rot['y'], demo1_rot['z'], demo1_rot['w']],q2star)
    real_rot_2 = qmul([demo2_rot['x'], demo2_rot['y'], demo2_rot['z'], demo2_rot['w']],q2star)
    rpy_1 = quat_to_rpy(real_rot_1)
    rpy_2 = quat_to_rpy(real_rot_2)
    rpy_1_list = [rpy_1[2], rpy_1[1], rpy_1[0]]
    rpy_2_list = [rpy_2[2], rpy_2[1], rpy_2[0]]
    C_1 = [demo1_pos['x'], demo1_pos['y'], demo1_pos['z']]
    C_2 = [demo2_pos['x'], demo2_pos['y'], demo2_pos['z']]
    
    # Load YOLO model
    model_root = Path(__file__).resolve().parent
    model = YOLO(model_root / ".." / "resource" / "final.pt")
    image_files = ['1.jpg', '2.jpg', '3.jpg']

    object_centers: dict[str, list[dict[str, list[float]]]] = {
        img: [] for img in image_files
    }

    for image_file in image_files:
        if not os.path.exists(image_file):
            print(f"Warning: {image_file} not found, skipping...")
            continue

        print(f"\nProcessing {image_file}...")
        results = model.predict(image_file, save=False)
        detections = results[0]

        if detections.boxes is not None and len(detections.boxes) > 0:
            for box in detections.boxes:
                class_id   = int(box.cls[0])
                class_name = model.names[class_id]
                bbox       = box.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]

                # 픽셀 중심 계산
                cx = float((bbox[0] + bbox[2]) / 2)
                cy = float((bbox[1] + bbox[3]) / 2)

                # 월드 좌표로 변환 (Z 값도 주의)
                if image_file == '1.jpg':
                    world_cor = pixel_to_world_Z(cx, cy, k, C_1, rpy_1_list, 17.0)
                elif image_file == '2.jpg':
                    world_cor = pixel_to_world_Z(cx, cy, k, C_2, rpy_2_list, 17.3)
                else:
                    # demo_3 카메라가 없다면 넘어가거나, 동일 처리
                    continue

                # 최종 결과 리스트에 추가
                object_centers[image_file].append({
                    class_name: [float(world_cor[0]),
                                 float(world_cor[1]),
                                 float(world_cor[2])]
                })
        else:
            print(f"No objects detected in {image_file}")

    # 4) 임시 이미지 파일 삭제
    for image_file in image_files:
        if os.path.exists(image_file):
            os.remove(image_file)
            print(f"Removed {image_file}")

    return object_centers


# Optional: If you want to run this file directly for testing
if __name__ == "__main__":
    centers = detect_objects()
    print("Detected object centers:")
    print(centers)

    # Example: Access specific object
    #if "juice" in centers:
    #    print(f"\nJuice centers: {centers['juice']}")

