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

# Define the MLP model structure (same as training)
class ResidualMLP(nn.Module):
    def __init__(self, input_size=4, output_size=3):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 64), 
            nn.ReLU(), 
            nn.Linear(64, 32), 
            nn.ReLU(), 
            nn.Linear(32, output_size)
        )
    def forward(self, x): 
        return self.model(x)

def setup_camera_params():
    """Setup camera parameters for both cameras"""
    cam_params = {}
    
    # Camera 1 parameters
    rpy1_deg = [-74.492, -23.79, -168.446]
    K1 = np.array([[10666.666, 0, 640], 
                   [0, 10666.666, 360], 
                   [0, 0, 1]])
    cam_params[1] = {
        "cam_pos": np.array([-60.864, 152.947, 21.475]), 
        "R": Rotation.from_euler('xyz', rpy1_deg, degrees=True).as_matrix(), 
        "K": K1
    }
    
    # Camera 2 parameters
    rpy2_deg = [-72.435, 30.821, 167.397]
    K2 = np.array([[10666.666, 0, 640], 
                   [0, 10666.666, 360], 
                   [0, 0, 1]])
    cam_params[2] = {
        "cam_pos": np.array([-31.510, 136.458, 21.271]), 
        "R": Rotation.from_euler('xyz', rpy2_deg, degrees=True).as_matrix(), 
        "K": K2
    }
    
    return cam_params

def estimate_world_coords(pixel_coord, cam_params, ground_z):
    """Convert pixel coordinates to world coordinates using camera geometry"""
    K = cam_params['K']
    R_mat = cam_params['R']
    cam_pos = cam_params['cam_pos']
    
    u, v = pixel_coord
    
    # Normalize pixel coordinates
    cam_dir = np.linalg.inv(K) @ np.array([u, v, 1])
    
    # Transform to world direction
    world_dir = R_mat.T @ cam_dir
    
    # Find intersection with ground plane
    if abs(world_dir[2]) < 1e-6:
        return None
    
    lambda_val = (ground_z - cam_pos[2]) / world_dir[2]
    world_coords = cam_pos + lambda_val * world_dir
    
    return world_coords

def predict_with_mlp(u, v, camera_id, ground_z, cam_params, mlp_model):
    """
    Use MLP model to predict corrected world coordinates from pixel coordinates
    
    Args:
        u, v: Pixel coordinates
        camera_id: Camera number (1 or 2)
        ground_z: Ground plane height
        cam_params: Camera parameters dictionary
        mlp_model: Loaded MLP model
    
    Returns:
        Corrected world coordinates [x, y, z] or None if failed
    """
    # Get geometric estimation
    geometric_coords = estimate_world_coords([u, v], cam_params[camera_id], ground_z)
    
    if geometric_coords is None:
        return None
    
    x_geo, y_geo, z_geo = geometric_coords
    
    # Prepare input for MLP
    input_features = torch.tensor([[u, v, x_geo, y_geo]], dtype=torch.float32)
    
    # Predict residual
    with torch.no_grad():
        residual = mlp_model(input_features).numpy()[0]
    
    # Calculate corrected coordinates
    x_corrected = x_geo + residual[0]
    y_corrected = y_geo + residual[1]
    z_corrected = z_geo + residual[2]
    
    return [x_corrected, y_corrected, z_corrected]

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

def detect_objects(use_mlp=True, mlp_model_path='mlp_model_multi.pth') -> list[dict[str, list[float]]]:
    """
    Capture images from cameras, detect objects using YOLO, and return object centers.
    
    Args:
        use_mlp: If True, use MLP model for correction. If False, use original interpole method.
        mlp_model_path: Path to the trained MLP model file
    
    Returns:
        list: List of dictionaries containing class_name, position, and recyclable status
    """
    # Capture images from cameras
    capture_single_image('/metasejong2025/cameras/demo_1/image_raw', '1.jpg')
    capture_single_image('/metasejong2025/cameras/demo_2/image_raw', '2.jpg')
    capture_single_image('/metasejong2025/cameras/demo_3/image_raw', '3.jpg')
    
    q2star = np.array([ 0.5, -0.5,  -0.5,  0.5])
    k, demo1_pos, demo1_rot, demo2_pos, demo2_rot = collect_demo_data()
    
    # Initialize object centers storage
    object_centers = defaultdict(list)
    
    # Setup for MLP method
    mlp_model = None
    cam_params = None
    
    if use_mlp:
        # Load MLP model
        try:
            mlp_model_path = Path(__file__).resolve().parent / ".." / "resource" / "mlp_model_multi.pth"
            mlp_model = ResidualMLP(input_size=4, output_size=3)
            mlp_model.load_state_dict(torch.load(mlp_model_path))
            mlp_model.eval()
            print(f"Successfully loaded MLP model from {mlp_model_path}")
            
            # Setup camera parameters for MLP method
            cam_params = setup_camera_params()
        except FileNotFoundError:
            print(f"Warning: MLP model file '{mlp_model_path}' not found. Falling back to interpole method.")
            use_mlp = False
        except Exception as e:
            print(f"Error loading MLP model: {e}. Falling back to interpole method.")
            use_mlp = False
    
    # Setup for interpole method (fallback or if not using MLP)
    if not use_mlp:
        real_rot_1 = qmul([demo1_rot['x'], demo1_rot['y'], demo1_rot['z'], demo1_rot['w']],q2star)
        real_rot_2 = qmul([demo2_rot['x'], demo2_rot['y'], demo2_rot['z'], demo2_rot['w']],q2star)
        rpy_1 = quat_to_rpy(real_rot_1)
        rpy_2 = quat_to_rpy(real_rot_2)
        rpy_1_list = [rpy_1[2], rpy_1[1], rpy_1[0]]
        rpy_2_list = [rpy_2[2], rpy_2[1], rpy_2[0]]
        C_1 = [demo1_pos['x'], demo1_pos['y'], demo1_pos['z']]
        C_2 = [demo2_pos['x'], demo2_pos['y'], demo2_pos['z']]
        
        topRight_1 = pixel_to_world_Z(0, 0, k, C_1, rpy_1_list, 16.5)
        topRight_2 = pixel_to_world_Z(0, 0, k, C_2, rpy_2_list, 16.8)
        topLeft_1 =  pixel_to_world_Z(1280, 0, k, C_1, rpy_1_list, 16.5)
        topLeft_2 =  pixel_to_world_Z(1280, 0, k, C_2, rpy_2_list, 16.8)
        bottomRight_1 = pixel_to_world_Z(0, 720, k, C_1, rpy_1_list, 16.5)
        bottomRight_2 = pixel_to_world_Z(0, 720, k, C_2, rpy_2_list, 16.8)
        bottomLeft_1 =  pixel_to_world_Z(1280, 720, k, C_1, rpy_1_list, 16.5)
        bottomLeft_2 = pixel_to_world_Z(1280, 720, k, C_2, rpy_2_list, 16.8)
    
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
            for i, box in enumerate(result.boxes):
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                confidence = float(box.conf[0])
                bbox = box.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]

                # Calculate center coordinates
                center_x = (bbox[0] + bbox[2]) / 2
                center_y = (bbox[1] + bbox[3]) / 2
                
                # Calculate world coordinates
                if image_file == '1.jpg':
                    if use_mlp:
                        # Use MLP model for correction
                        world_cor = predict_with_mlp(center_x, center_y, 1, 17.0, cam_params, mlp_model)
                        if world_cor is None:
                            print(f"  Warning: Failed to predict coordinates for {class_name}, skipping...")
                            continue
                    else:
                        # Use original interpole method
                        world_cor_xy = interpole(topRight_1, topLeft_1, bottomRight_1, bottomLeft_1, center_x, center_y)
                        world_cor = [world_cor_xy[0], world_cor_xy[1], 17.0]
                    
                    object_centers[class_name].append(world_cor)
                    
                elif image_file == '2.jpg':
                    if use_mlp:
                        # Use MLP model for correction
                        world_cor = predict_with_mlp(center_x, center_y, 2, 17.3, cam_params, mlp_model)
                        if world_cor is None:
                            print(f"  Warning: Failed to predict coordinates for {class_name}, skipping...")
                            continue
                    else:
                        # Use original interpole method
                        world_cor_xy = interpole(topRight_2, topLeft_2, bottomRight_2, bottomLeft_2, center_x, center_y)
                        world_cor = [world_cor_xy[0], world_cor_xy[1], 17.3]
                    
                    object_centers[class_name].append(world_cor)
                
                print(f"  {i+1}. {class_name}: confidence={confidence:.2f}")
                print(f"     Pixel Center: ({center_x:.1f}, {center_y:.1f})")
                if image_file in ['1.jpg', '2.jpg']:
                    print(f"     World Coords: ({world_cor[0]:.3f}, {world_cor[1]:.3f}, {world_cor[2]:.3f})")
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
    
    method_used = "MLP correction" if use_mlp else "Interpole method"
    print(f"\nDetection completed using {method_used}")
    
    return result


# Optional: If you want to run this file directly for testing
if __name__ == "__main__":
    # Test with MLP model
    print("=== Testing with MLP model ===")
    centers_mlp = detect_objects(use_mlp=True)
    print("\nDetected objects with MLP correction:")
    for item in centers_mlp:
        print(f"  {item['class_name']}: {item['position']} (recyclable: {item['recyclable']})")
    
    # Optionally test with original interpole method
    # print("\n=== Testing with interpole method ===")
    # centers_interpole = detect_objects(use_mlp=False)
    # print("\nDetected objects with interpole method:")
    # for item in centers_interpole:
    #     print(f"  {item['class_name']}: {item['position']} (recyclable: {item['recyclable']})")
