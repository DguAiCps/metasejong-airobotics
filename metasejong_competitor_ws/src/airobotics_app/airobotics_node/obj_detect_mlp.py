#!/usr/bin/env python3
from .pixel_to_cor import *
from .single_capture import capture_single_image
from pathlib import Path
from .topic_collect import collect_demo_data
import math
import subprocess
import time
import os
from collections import defaultdict
from ultralytics import YOLO
import shutil
import signal
import numpy as np
from scipy.spatial.transform import Rotation
import torch
import torch.nn as nn
import sys


class MyStandardScaler:
    """numpy를 사용한 StandardScaler 구현"""
    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def fit_transform(self, data):
        self.mean_ = np.mean(data, axis=0)
        self.scale_ = np.std(data, axis=0)
        # 0으로 나누는 것을 방지
        self.scale_[self.scale_ == 0] = 1
        return (data - self.mean_) / self.scale_

    def transform(self, data):
        return (data - self.mean_) / self.scale_

    def inverse_transform(self, data):
        return (data * self.scale_) + self.mean_

sys.modules['__main__'].MyStandardScaler = MyStandardScaler

class MLPNet(nn.Module):
    def __init__(self, input_dim=2, output_dim=2, hidden_dims=[512, 512, 512, 512, 256, 128]):
        super(MLPNet, self).__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class PixelToWorldConverter:
    """픽셀-좌표 변환기 클래스"""
    def __init__(self, camera1_model_path, camera2_model_path):
        try:
            torch.serialization.add_safe_globals([MyStandardScaler])

            self.cam1_checkpoint = torch.load(camera1_model_path, map_location='cpu', weights_only=False)
            
            self.cam1_model = MLPNet()
            self.cam1_model.load_state_dict(self.cam1_checkpoint['model_state_dict'])
            self.cam1_model.eval()
            self.cam1_pixel_scaler = self.cam1_checkpoint['pixel_scaler']
            self.cam1_world_scaler = self.cam1_checkpoint['world_scaler']
            print(f"Successfully loaded Camera 1 model from {camera1_model_path}")

            self.cam2_checkpoint = torch.load(camera2_model_path, map_location='cpu', weights_only=False)
            self.cam2_model = MLPNet()
            self.cam2_model.load_state_dict(self.cam2_checkpoint['model_state_dict'])
            self.cam2_model.eval()
            self.cam2_pixel_scaler = self.cam2_checkpoint['pixel_scaler']
            self.cam2_world_scaler = self.cam2_checkpoint['world_scaler']
            print(f"Successfully loaded Camera 2 model from {camera2_model_path}")

            self.models_loaded = True

        except Exception as e:
            print(f"Error loading models: {e}")
            self.models_loaded = False

    def convert(self, pixel_x, pixel_y, camera_id):
        """픽셀 좌표를 실제 좌표로 변환"""
        if not self.models_loaded:
            return None

        if camera_id == 1:
            model = self.cam1_model
            pixel_scaler = self.cam1_pixel_scaler
            world_scaler = self.cam1_world_scaler
        elif camera_id == 2:
            model = self.cam2_model
            pixel_scaler = self.cam2_pixel_scaler
            world_scaler = self.cam2_world_scaler
        else:
            print(f"Invalid camera_id: {camera_id}. Must be 1 or 2.")
            return None

        # 입력 정규화
        pixel_input = np.array([[pixel_x, pixel_y]])
        pixel_normalized = pixel_scaler.transform(pixel_input)

        # 예측
        with torch.no_grad():
            prediction_norm = model(torch.FloatTensor(pixel_normalized))
            prediction = world_scaler.inverse_transform(prediction_norm.numpy())

        return prediction[0][0], prediction[0][1]  # x, y 좌표 반환

def interpole(tR, tL, bR, bL, u, v):
    """Keep original interpole function as fallback"""
    s2 =  u / 1280.0
    t2 = v / 720.0
    tx2 = (1-s2) * tR[0] + s2 * tL[0]
    ty2 = (1-s2) * tR[1] + s2 * tL[1]
    bx2 = (1-s2) * bR[0] + s2 * bL[0]
    by2 = (1-s2) * bR[1] + s2 * bL[1]
    fx2 = (1-t2) * tx2 + t2 * bx2
    fy2 = (1-t2) * ty2 + t2 * by2
    return [fx2, fy2]

def detect_objects(use_mlp=True) -> list[dict[str, list[float]]]:
    """
    Capture images from cameras, detect objects using YOLO, and return object centers.

    Args:
        use_mlp: If True, use MLP models for correction. If False, use original interpole method.

    Returns:
        list: List of dictionaries containing class_name, position, and recyclable status
    """
    # Capture images from cameras (only cam1 and cam2)
    scenario_id = os.getenv("ENV_METASEJONG_SCENARIO", "demo")
    capture_single_image(f'/metasejong2025/cameras/{scenario_id}_1/image_raw', '1.jpg')
    capture_single_image(f'/metasejong2025/cameras/{scenario_id}_2/image_raw', '2.jpg')

    q2star = np.array([0.5, -0.5, -0.5, 0.5])
    k, demo1_pos, demo1_rot, demo2_pos, demo2_rot = collect_demo_data()

    # Initialize object centers storage
    object_centers = defaultdict(list)

    # Setup for MLP method
    converter = None

    if use_mlp:
        try:
            # Use same pattern as the reference code
            model_root = Path(__file__).resolve().parent
            camera1_model_path = model_root / ".." / "resource" / "cam1.pth"
            camera2_model_path = model_root / ".." / "resource" / "cam2.pth"
            print(f"Loading models from: {model_root / '..' / 'resource'}")

            if not camera1_model_path.exists() or not camera2_model_path.exists():
                print("Warning: MLP model files not found. Falling back to interpole method.")
                use_mlp = False
            else:
                converter = PixelToWorldConverter(camera1_model_path, camera2_model_path)
                if not converter.models_loaded:
                    print("Warning: Failed to load MLP models. Falling back to interpole method.")
                    use_mlp = False

        except Exception as e:
            print(f"Error initializing converter: {e}. Falling back to interpole method.")
            use_mlp = False

    # Setup for interpole method (fallback or if not using MLP)
    if not use_mlp:
        real_rot_1 = qmul([demo1_rot['x'], demo1_rot['y'], demo1_rot['z'], demo1_rot['w']], q2star)
        real_rot_2 = qmul([demo2_rot['x'], demo2_rot['y'], demo2_rot['z'], demo2_rot['w']], q2star)
        rpy_1 = quat_to_rpy(real_rot_1)
        rpy_2 = quat_to_rpy(real_rot_2)
        rpy_1_list = [rpy_1[2], rpy_1[1], rpy_1[0]]
        rpy_2_list = [rpy_2[2], rpy_2[1], rpy_2[0]]
        C_1 = [demo1_pos['x'], demo1_pos['y'], demo1_pos['z']]
        C_2 = [demo2_pos['x'], demo2_pos['y'], demo2_pos['z']]

        topRight_1 = pixel_to_world_Z(0, 0, k, C_1, rpy_1_list, 16.5)
        topRight_2 = pixel_to_world_Z(0, 0, k, C_2, rpy_2_list, 16.8)
        topLeft_1 = pixel_to_world_Z(1280, 0, k, C_1, rpy_1_list, 16.5)
        topLeft_2 = pixel_to_world_Z(1280, 0, k, C_2, rpy_2_list, 16.8)
        bottomRight_1 = pixel_to_world_Z(0, 720, k, C_1, rpy_1_list, 16.5)
        bottomRight_2 = pixel_to_world_Z(0, 720, k, C_2, rpy_2_list, 16.8)
        bottomLeft_1 = pixel_to_world_Z(1280, 720, k, C_1, rpy_1_list, 16.5)
        bottomLeft_2 = pixel_to_world_Z(1280, 720, k, C_2, rpy_2_list, 16.8)

    # Load YOLO model
    model_root = Path(__file__).resolve().parent
    model = YOLO(model_root / ".." / "resource" / "final.pt")
    image_files = ['1.jpg', '2.jpg']  # Only process camera 1 and 2

    # Process each image
    for image_file in image_files:
        if not os.path.exists(image_file):
            print(f"Warning: {image_file} not found, skipping...")
            continue

        print(f"\nProcessing {image_file}...")
        camera_id = int(image_file.split('.')[0])  # Get camera number from filename

        # Run inference
        results = model.predict(image_file, save=False)
        result = results[0]

        if result.boxes is not None and len(result.boxes) > 0:
            for i, box in enumerate(result.boxes):
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                confidence = float(box.conf[0])
                bbox = box.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]

                # Calculate center coordinates
                center_x = (bbox[0] + bbox[2]) / 2
                center_y = (bbox[1] + bbox[3]) / 2

                # Calculate world coordinates
                world_cor = None

                if use_mlp and converter:
                    # Use MLP model for correction
                    world_xy = converter.convert(center_x, center_y, camera_id)
                    if world_xy is not None:
                        if camera_id == 1:
                            world_cor = [world_xy[0], world_xy[1], 17.0]
                        elif camera_id == 2:
                            world_cor = [world_xy[0], world_xy[1], 17.3]
                    else:
                        print(f"  Warning: MLP conversion failed for {class_name}, skipping...")
                        continue
                else:
                    # Use original interpole method
                    if camera_id == 1:
                        world_cor_xy = interpole(topRight_1, topLeft_1, bottomRight_1, bottomLeft_1, center_x, center_y)
                        world_cor = [world_cor_xy[0], world_cor_xy[1], 17.0]
                    elif camera_id == 2:
                        world_cor_xy = interpole(topRight_2, topLeft_2, bottomRight_2, bottomLeft_2, center_x, center_y)
                        world_cor = [world_cor_xy[0], world_cor_xy[1], 17.3]

                if world_cor is not None:
                    object_centers[class_name].append(world_cor)

                    print(f"  {i+1}. {class_name}: confidence={confidence:.2f}")
                    print(f"      Pixel Center: ({center_x:.1f}, {center_y:.1f})")
                    print(f"      World Coords: ({world_cor[0]:.3f}, {world_cor[1]:.3f}, {world_cor[2]:.3f})")
        else:
            print(f"No objects detected in {image_file}")

    # Clean up images
    for image_file in image_files:
        if os.path.exists(image_file):
            os.remove(image_file)
            print(f"Removed {image_file}")

    # Format results
    result = []
    for key, list_of_lists in object_centers.items():
        for sub_list in list_of_lists:
            recyclable = not (key == 'mug' or key == 'wood_block')
            result.append({'class_name': key, 'position': sub_list, 'recyclable': recyclable})

    method_used = "MLP models" if use_mlp else "Interpole method"
    print(f"\nDetection completed using {method_used}")
    print(f"Total objects detected: {len(result)}")

    return result


# Optional: If you want to run this file directly for testing
if __name__ == "__main__":
    # Test with MLP models
    print("=== Testing with MLP models ===")
    centers_mlp = detect_objects(use_mlp=True)
    print("\nDetected objects with MLP correction:")
    for item in centers_mlp:
        print(f"  {item['class_name']}: {item['position']} (recyclable: {item['recyclable']})")

    # Optionally test with original interpole method
    print("\n=== Testing with interpole method ===")
    centers_interpole = detect_objects(use_mlp=False)
    print("\nDetected objects with interpole method:")
    for item in centers_interpole:
        print(f"  {item['class_name']}: {item['position']} (recyclable: {item['recyclable']})")
