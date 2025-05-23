# Copyright (c) 2025, IoT Convergence & Open Sharing System (IoTCOSS)
#
# All rights reserved. This software and its documentation are proprietary and confidential.
# The IoT Convergence & Open Sharing System (IoTCOSS) retains all intellectual property rights,
# including but not limited to copyrights, patents, and trade secrets, in and to this software
# and related documentation. Any use, reproduction, disclosure, or distribution of this software
# and related documentation without explicit written permission from IoTCOSS is strictly prohibited.

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from typing import Optional, Dict, Any, List
from geometry_msgs.msg import Twist, Pose, Point, Quaternion
from std_msgs.msg import String, Float32, Bool
from sensor_msgs.msg import JointState
from nav_msgs.msg import Odometry
from dataclasses import dataclass
import math
import time

@dataclass
class ManipulatorCommander:
    """
    Data class for storing initial and final target positions and orientations of the robot arm
    
    Attributes:
        start_target_orientation_quat (List[float]): Initial target orientation (quaternion format [x, y, z, w])
        start_target_position (List[float]): Initial target position (coordinates [x, y, z])
        end_target_orientation_quat (List[float]): Final target orientation (quaternion format [x, y, z, w])
        end_target_position (List[float]): Final target position (coordinates [x, y, z])
    """
    start_target_orientation_quat: List[float]  # [x, y, z, w]
    start_target_position: List[float]  # [x, y, z]
    end_target_orientation_quat: List[float]  # [x, y, z, w]
    end_target_position: List[float]  # [x, y, z]

    def __init__(self, start_target_orientation_quat: List[float], start_target_position: List[float], end_target_orientation_quat: List[float], end_target_position: List[float]):
        self._start_target_orientation_quat = [0.0, 0.0, 0.0, 1.0]
        self._start_target_position = [0.0, 0.0, 0.0]
        self._end_target_orientation_quat = [0.0, 0.0, 0.0, 1.0]
        self._end_target_position = [0.0, 0.0, 0.0]

        # Set initial values with validation by calling setters
        self.start_target_orientation_quat = start_target_orientation_quat
        self.start_target_position = start_target_position
        self.end_target_orientation_quat = end_target_orientation_quat
        self.end_target_position = end_target_position


    @property
    def start_target_orientation_quat(self) -> List[float]:
        return self._start_target_orientation_quat

    @start_target_orientation_quat.setter
    def start_target_orientation_quat(self, value: List[float]) -> None:
        self._start_target_orientation_quat = value

    @property
    def start_target_position(self) -> List[float]:
        return self._start_target_position

    @start_target_position.setter
    def start_target_position(self, value: List[float]) -> None:
        self._start_target_position = value

    @property
    def end_target_orientation_quat(self) -> List[float]:
        return self._end_target_orientation_quat

    @end_target_orientation_quat.setter
    def end_target_orientation_quat(self, value: List[float]) -> None:
        self._end_target_orientation_quat = value

    @property
    def end_target_position(self) -> List[float]:
        return self._end_target_position

    @end_target_position.setter
    def end_target_position(self, value: List[float]) -> None:
        self._end_target_position = value


    def to_string(self) -> str:
        # Combine all lists into one list
        all_values = (
            list(self.start_target_orientation_quat) +
            list(self.start_target_position) +
            list(self.end_target_orientation_quat) +
            list(self.end_target_position)
        )
        # Convert each value to string and join with spaces
        return ' '.join(map(str, all_values))

class MobileBaseCommander:
    """
    Data class for robot velocity control.
    Validates linear and angular velocity values.

    Attributes:
        linear_x (float): Forward/backward velocity (m/s)
        angular_z (float): Rotational velocity (rad/s)
    """
    def __init__(self, linear_x: float = 0.0, angular_z: float = 0.0):
        self._linear_x = 0.0
        self._angular_z = 0.0
        
        # Set velocity with validation (setter가 호출됨)
        self.linear_x = linear_x
        self.angular_z = angular_z
    
    @property
    def linear_x(self) -> float:
        return self._linear_x
    
    @linear_x.setter
    def linear_x(self, value: float) -> None:
        if not -2.0 <= value <= 2.0:  # Example: max velocity 2.0 m/s
            raise ValueError(f"Linear velocity must be between -2.0 and 2.0 m/s. Input: {value}")
        self._linear_x = value
    
    @property
    def angular_z(self) -> float:
        return self._angular_z
    
    @angular_z.setter
    def angular_z(self, value: float) -> None:
        if not -3.14 <= value <= 3.14:  # Example: max angular velocity π rad/s
            raise ValueError(f"Angular velocity must be between -3.14 and 3.14 rad/s. Input: {value}")
        self._angular_z = value
    
    def to_twist(self) -> Twist:
        twist = Twist()
        twist.linear.x = self._linear_x
        twist.angular.z = self._angular_z
        return twist


METASEJONG_DEFAULT_NAMESPACE = ""

ROBOT_TOPIC_CMD_VEL = "/metasejong2025/robot/cmd_vel"
ROBOT_TOPIC_ARM_MANIPULATOR = "/metasejong2025/robot/ppcmd"
ROBOT_TOPIC_ODOM = "/metasejong2025/robot/odom"


#   RobotNode is the main node for the robot status .
#   It controls robot movement and robotic arm through ROS2 topics and actions.
class RobotNode(Node):
    """
    Utility class for robot control.
    Controls robot movement and robotic arm through ROS2 topics and actions.
    """
    
    def __init__(self, namespace: str = METASEJONG_DEFAULT_NAMESPACE):
        super().__init__('airobotics_robot_node')

        self.logger = self.get_logger()
        self.namespace = namespace

        self.stop_flag = False

        # QoS settings
        self.odom_qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            durability=DurabilityPolicy.VOLATILE
        )

        self.pub_qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.robot_position = [0.0, 0.0, 0.0]
        self.robot_orientation = [0.0, 0.0, 0.0, 0.0]
        
        # Node가 완전히 초기화될 때까지 대기
        time.sleep(1)

        # Initialize subscribers
        self._init_subscribers()
        
        # Initialize publishers
        self._init_publishers()


    #   Stop robot movement and clean up resources
    def stop_and_destroy_robot(self) -> None:
        """Stop robot movement and clean up resources"""
        try:
            self.stop_flag = True
            
            # Stop all robot movements
            stop_cmd = MobileBaseCommander(0.0, 0.0)
            self.move_robot(stop_cmd)
            
        except Exception as e:
            self.logger.error(f"Error while cleanup: {str(e)}")
        finally:
            # Ensure node is destroyed
            try:
                self.destroy_node()
            except:
                pass


    #   Initialize publishers for robot control
    def _init_publishers(self) -> None:
        """Initialize publishers for robot control"""
        # Movement control publisher
        self.cmd_vel_pub = self.create_publisher(
            Twist,
            f"{self.namespace}{ROBOT_TOPIC_CMD_VEL}",
            self.pub_qos_profile    
        )
        
        self.cmd_manipulator_pub = self.create_publisher(
            String,
            f"{self.namespace}{ROBOT_TOPIC_ARM_MANIPULATOR}",
            self.pub_qos_profile
        )

    
    #   Initialize subscribers for robot state monitoring   
    def _init_subscribers(self) -> None:
        """Initialize subscribers for robot state monitoring"""
        # Robot position information

        self.odom_sub = self.create_subscription(
            Odometry,
            f"{self.namespace}{ROBOT_TOPIC_ODOM}",
            self._odom_callback,
            self.odom_qos_profile
        )
        
    #   Callback function for processing robot position information
    def _odom_callback(self, msg: Odometry) -> None:
        """Callback function for processing robot position information"""

        try: 
            self.robot_position = [msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z]
            self.robot_orientation = [msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w]
        except Exception as e:
            self.logger.error(f"Error while processing /odom message: {str(e)}")
            raise  
     

    #   Publish robot movement command
    def move_robot(self, velocity: MobileBaseCommander) -> None:
        """
        Move the robot
        
        Args:
            velocity (MobileBaseCommander): Robot velocity object
            
        Raises:
            ValueError: When velocity is out of allowed range
        """
        try:
            twist = velocity.to_twist()
            self.cmd_vel_pub.publish(twist)
        except ValueError as e:
            self.logger.error(f"Error while publish robot movement command: {str(e)}")
            raise
    
    #   Publish robot arm manipulator command
    def pick_up_object(self, manipulator_commander: ManipulatorCommander) -> None:
        """
        Pick up an object
        """
        try:
            if self.stop_flag:
                return

            manipulator = String()

            manipulator.data = manipulator_commander.to_string()
            self.cmd_manipulator_pub.publish(manipulator)
        except ValueError as e:
            self.logger.error(f"Error while publlish robot arm manipulator command: {str(e)}")
            raise

    def get_robot_position(self) -> List[float]:
        """
        Get the robot position
        """
        return [self.start_position[0]+ self.robot_position[0], self.start_position[1]+ self.robot_position[1], self.start_position[2]+ self.robot_position[2]]
    
    def get_robot_orientation(self) -> List[float]:
        """
        Get the robot orientation
        """
        return self.robot_orientation
    
    