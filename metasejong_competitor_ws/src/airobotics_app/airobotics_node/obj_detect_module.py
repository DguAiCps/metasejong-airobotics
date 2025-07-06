#!/usr/bin/env python3
from single_capture import capture_single_image
from pathlib import Path
import subprocess
import time
import os
from collections import defaultdict
from ultralytics import YOLO
import shutil
import signal

def detect_objects():
    """
    Capture images from cameras, detect objects using YOLO, and return object centers.

    Returns:
        defaultdict: Dictionary with object class names as keys and list of center coordinates as values
                    Example: {'juice': [(100.5, 200.3), (150.2, 180.7)], 'apple': [(300.1, 400.2)]}
    """
    # Capture images from cameras
    capture_single_image('/metasejong2025/cameras/demo_1/image_raw', '1.jpg')
    capture_single_image('/metasejong2025/cameras/demo_2/image_raw', '2.jpg')
    capture_single_image('/metasejong2025/cameras/demo_3/image_raw', '3.jpg')

    # Initialize object centers storage
    object_centers = defaultdict(list)

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

                # Store center coordinates by object class name
                object_centers[class_name].append((center_x, center_y))

                print(f"  {i+1}. {class_name}: confidence={confidence:.2f}")
                print(f"     Center: ({center_x:.1f}, {center_y:.1f})")
        else:
            print(f"No objects detected in {image_file}")

    # Clean up image files
    for image_file in image_files:
        if os.path.exists(image_file):
            os.remove(image_file)
            print(f"Removed {image_file}")

    return object_centers

# Optional: If you want to run this file directly for testing
if __name__ == "__main__":
    centers = detect_objects()
    print("Detected object centers:")
    for obj_class, coords in centers.items():
        print(f"{obj_class}: {coords}")

    # Example: Access specific object
    if "juice" in centers:
        print(f"\nJuice centers: {centers['juice']}")

