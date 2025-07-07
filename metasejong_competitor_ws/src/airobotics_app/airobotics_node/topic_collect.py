#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from tf2_msgs.msg import TFMessage

class DemoTFExtractor(Node):
    def __init__(self, target_frames):
        super().__init__('demo_tf_extractor')
        self.target_frames = target_frames  # ['demo_1', 'demo_2']
        self.collected_transforms = {}
        self.camera_info = None
        
    def tf_callback(self, msg):
        # Check each transform in the message
        for transform in msg.transforms:
            child_frame_id = transform.child_frame_id
            
            # If this is one of our target frames, save it
            if child_frame_id in self.target_frames:
                self.get_logger().info(f"Found transform for {child_frame_id}")
                
                # Extract translation and rotation
                translation = {
                    'x': transform.transform.translation.x,
                    'y': transform.transform.translation.y,
                    'z': transform.transform.translation.z
                }
                
                rotation = {
                    'x': transform.transform.rotation.x,
                    'y': transform.transform.rotation.y,
                    'z': transform.transform.rotation.z,
                    'w': transform.transform.rotation.w
                }
                
                self.collected_transforms[child_frame_id] = {
                    'frame_id': transform.header.frame_id,
                    'child_frame_id': child_frame_id,
                    'translation': translation,
                    'rotation': rotation,
                    'timestamp': {
                        'sec': transform.header.stamp.sec,
                        'nanosec': transform.header.stamp.nanosec
                    }
                }
                
                # Print the data
                print(f"\n=== {child_frame_id} Transform ===")
                print(f"Frame: {transform.header.frame_id} -> {child_frame_id}")
                print(f"Translation: x={translation['x']:.6f}, y={translation['y']:.6f}, z={translation['z']:.6f}")
                print(f"Rotation: x={rotation['x']:.6f}, y={rotation['y']:.6f}, z={rotation['z']:.6f}, w={rotation['w']:.6f}")
        
        # Check if we have collected all target frames
        if len(self.collected_transforms) == len(self.target_frames):
            self.get_logger().info("All target frames collected. Stopping subscription.")
            return True  # Signal that we're done
        
        return False
    
    def camera_info_callback(self, msg):
        if self.camera_info is None:
            self.get_logger().info("Camera info received")
            
            # Extract camera intrinsic matrix K
            k_matrix = msg.k  # This is a list of 9 elements (3x3 matrix in row-major order)
            
            self.camera_info = {
                'frame_id': msg.header.frame_id,
                'width': msg.width,
                'height': msg.height,
                'k_matrix': {
                    'fx': k_matrix[0],  # focal length x
                    'fy': k_matrix[4],  # focal length y
                    'cx': k_matrix[2],  # principal point x
                    'cy': k_matrix[5],  # principal point y
                    'full_matrix': k_matrix
                },
                'distortion_model': msg.distortion_model,
                'distortion_coeffs': msg.d,
                'timestamp': {
                    'sec': msg.header.stamp.sec,
                    'nanosec': msg.header.stamp.nanosec
                }
            }
            
            print(f"\n=== Camera Info ===")
            print(f"Frame ID: {msg.header.frame_id}")
            print(f"Resolution: {msg.width}x{msg.height}")
            print(f"Focal Length: fx={k_matrix[0]:.2f}, fy={k_matrix[4]:.2f}")
            print(f"Principal Point: cx={k_matrix[2]:.2f}, cy={k_matrix[5]:.2f}")
    
    def collect_data(self):
        # Subscribe to /tf topic
        self.tf_subscription = self.create_subscription(
            TFMessage,
            '/tf',
            self.tf_callback,
            10
        )
        
        # Subscribe to camera_info topic
        from sensor_msgs.msg import CameraInfo
        self.camera_subscription = self.create_subscription(
            CameraInfo,
            '/metasejong2025/cameras/demo_1/camera_info',
            self.camera_info_callback,
            10
        )
        
        self.get_logger().info(f"Waiting for transforms: {self.target_frames}")
        self.get_logger().info("Also collecting camera info...")
        
        # Spin until we have all the data we need
        while (len(self.collected_transforms) < len(self.target_frames) or 
               self.camera_info is None) and rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.1)
        
        return self.collected_transforms, self.camera_info

def collect_demo_data():
    """
    Collect camera K matrix and demo_1, demo_2 transforms from ROS2 topics
    
    Returns:
        tuple: (k_matrix, demo1_pos, demo1_rot, demo2_pos, demo2_rot)
        - k_matrix: dict with 'fx', 'fy', 'cx', 'cy' keys
        - demo1_pos, demo2_pos: dict with 'x', 'y', 'z' keys  
        - demo1_rot, demo2_rot: dict with 'x', 'y', 'z', 'w' keys
        Returns (None, None, None, None, None) if collection fails
    """
    if not rclpy.ok():
        rclpy.init()
    
    # Specify which demo frames to collect
    target_frames = ['demo_1', 'demo_2']
    
    extractor = DemoTFExtractor(target_frames)
    
    try:
        tf_data, camera_data = extractor.collect_data()
        
        # Extract data for return
        k_matrix = None
        demo1_pos = None
        demo1_rot = None
        demo2_pos = None
        demo2_rot = None
        
        # Extract camera K matrix
        if camera_data:
            k_matrix = camera_data['k_matrix']
        
        # Extract demo_1 data
        if 'demo_1' in tf_data:
            demo1_pos = tf_data['demo_1']['translation']
            demo1_rot = tf_data['demo_1']['rotation']
        
        # Extract demo_2 data  
        if 'demo_2' in tf_data:
            demo2_pos = tf_data['demo_2']['translation']
            demo2_rot = tf_data['demo_2']['rotation']
        
        # Print summary
        print("\n=== DATA COLLECTION COMPLETE ===")
        if k_matrix:
            print(f"Camera K: fx={k_matrix['fx']:.2f}, fy={k_matrix['fy']:.2f}, cx={k_matrix['cx']:.2f}, cy={k_matrix['cy']:.2f}")
        if demo1_pos:
            print(f"Demo1 - Pos: ({demo1_pos['x']:.3f}, {demo1_pos['y']:.3f}, {demo1_pos['z']:.3f}), "
                  f"Rot: ({demo1_rot['x']:.3f}, {demo1_rot['y']:.3f}, {demo1_rot['z']:.3f}, {demo1_rot['w']:.3f})")
        if demo2_pos:
            print(f"Demo2 - Pos: ({demo2_pos['x']:.3f}, {demo2_pos['y']:.3f}, {demo2_pos['z']:.3f}), "
                  f"Rot: ({demo2_rot['x']:.3f}, {demo2_rot['y']:.3f}, {demo2_rot['z']:.3f}, {demo2_rot['w']:.3f})")
        
        extractor.destroy_node()
        return k_matrix, demo1_pos, demo1_rot, demo2_pos, demo2_rot
        
    except KeyboardInterrupt:
        print("\nProgram interrupted")
        extractor.destroy_node()
        return None, None, None, None, None
    except Exception as e:
        print(f"Error: {e}")
        extractor.destroy_node()
        return None, None, None, None, None

def main():
    """Example usage of collect_demo_data function"""
    k, demo1_pos, demo1_rot, demo2_pos, demo2_rot = collect_demo_data()
    
    if k is not None:
        print(f"\nReturned data:")
        print(f"K matrix: {k}")
        print(f"Demo1 position: {demo1_pos}")
        print(f"Demo1 rotation: {demo1_rot}")
        print(f"Demo2 position: {demo2_pos}")
        print(f"Demo2 rotation: {demo2_rot}")
    else:
        print("Failed to collect data")
    
    # Shutdown ROS2 if needed
    if rclpy.ok():
        rclpy.shutdown()

if __name__ == "__main__":
    main()
