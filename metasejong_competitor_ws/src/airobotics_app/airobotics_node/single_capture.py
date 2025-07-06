#!/usr/bin/env python3
import time
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge

class SingleImageSaver(Node):
    def __init__(self, topic_name, output_filename='captured.jpg', timeout_sec = 1.0):
        super().__init__('single_image_saver')
        self.bridge = CvBridge()
        self.output_filename = output_filename
        self.saved = False
        self.subscription = self.create_subscription(
            Image,
            topic_name,
            self.save_and_exit,
            10)
        self.start_time = time.time()
        self.timeout_sec = timeout_sec

    def save_and_exit(self, msg):
        if not self.saved:
            # Convert and save image
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            cv2.imwrite(self.output_filename, cv_image)
            print(f'Image saved as {self.output_filename}')
            self.saved = True
    def is_timeout(self):
        return time.time() - self.start_time > self.timeout_sec

def capture_single_image(topic_name, output_filename='captured.jpg'):
    """Capture a single image from ROS2 topic and save it"""
    rclpy.init()
    node = SingleImageSaver(topic_name, output_filename)

    # Spin until image is saved
    while not node.saved and rclpy.ok() and not node.is_timeout():
        rclpy.spin_once(node, timeout_sec=0.1)

    rclpy.shutdown()

# Usage example
if __name__ == '__main__':
    capture_single_image('/metasejong2025/cameras/demo_1/image_raw')
