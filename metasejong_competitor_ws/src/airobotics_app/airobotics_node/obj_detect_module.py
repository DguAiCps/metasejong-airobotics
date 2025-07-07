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

def detect_objects() -> list[dict[str, list[float]]]:
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

    # Process each image
    for image_file in image_files:
        if not os.path.exists(image_file):
            print(f"Warning: {image_file} not found, skipping...")
            continue

        print(f"\nProcessing {image_file}...")

        # Run inference
        results = model.predict(image_file, save=False)
        result = results[0]

        if result.boxes is not None and len(result.boxes) > 0:
            print(f"Found {len(result.boxes)} objects in {image_file}:")
            for i, box in enumerate(result.boxes):
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                confidence = float(box.conf[0])
                bbox = box.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]

                # Calculate center coordinates
                center_x = (bbox[0] + bbox[2]) / 2
                center_y = (bbox[1] + bbox[3]) / 2
                # z cor for demo_1 : 16.5, demo_2 : 16.8
                # Store center coordinates by object class name
                if image_file == '1.jpg':
                    world_cor = pixel_to_world_Z(center_x, center_y, k, C_1, rpy_1_list, 17.0)
                    object_centers[class_name].append(world_cor.tolist()) 
                elif image_file == '2.jpg':
                    world_cor = pixel_to_world_Z(center_x, center_y, k, C_2, rpy_2_list, 17.3)
                    object_centers[class_name].append(world_cor.tolist())

                print(f"  {i+1}. {class_name}: confidence={confidence:.2f}")
                print(f"     Center: ({center_x:.1f}, {center_y:.1f})")
        else:
            print(f"No objects detected in {image_file}")

    for image_file in image_files:
        if os.path.exists(image_file):
            os.remove(image_file)
            print(f"Removed {image_file}")
    
    result = [
    {key: sub_list}           # 새 딕셔너리
    for key, list_of_lists in object_centers.items()
    for sub_list in list_of_lists
    ]
    #print(result)
    #TODO: list[dict[str, list[dict[str, list]]]]로 만들기
    return result


# Optional: If you want to run this file directly for testing
if __name__ == "__main__":
    centers = detect_objects()
    print("Detected object centers:")
    for item in centers:
        obj_class = item.keys()
        coords = item.values()
        print(f"{obj_class}: {coords}")

    # Example: Access specific object
    #if "juice" in centers:
    #    print(f"\nJuice centers: {centers['juice']}")

