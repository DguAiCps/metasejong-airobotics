# Copyright (c) 2025, IoT Convergence & Open Sharing System (IoTCOSS)
#
# All rights reserved. This software and its documentation are proprietary and confidential.
# The IoT Convergence & Open Sharing System (IoTCOSS) retains all intellectual property rights,
# including but not limited to copyrights, patents, and trade secrets, in and to this software
# and related documentation. Any use, reproduction, disclosure, or distribution of this software
# and related documentation without explicit written permission from IoTCOSS is strictly prohibited.

from tkinter import Image

import math
import time
from typing import Dict, List, Union
from sensor_msgs.msg import Image

from .competition_task_base import CompetitionTask
from .robot_node import RobotNode, MobileBaseCommander, ManipulatorCommander
from .robot_util import RobotUtil

import yaml
import os

import numpy as np
from scipy.spatial.transform import Rotation as R


CONST_GLOBAL_NAVIGATION_DISTANCE = 1.5
CONST_GLOBAL_PICK_DISTANCE = 0.78

# 원통의 긴 축 = local Z → world 기준으로 변환
def _compute_grasp_quaternion(o_euler_deg, r_quat_xyzw):
    # 1. 원통의 긴 축 = local Z → world 기준으로 변환
    rot_o = R.from_euler('xyz', o_euler_deg, degrees=True)
    long_axis = rot_o.apply([0, 0, 1])  # 원통 긴 축 (world)

    # 2. 긴 축에 수직한 접근 방향 (XY 평면)
    approach_vec_2d = np.array([-long_axis[1], long_axis[0]])
    approach_vec_2d /= np.linalg.norm(approach_vec_2d)

    # 3. 전방 방향 (X축): 접근 방향
    x_axis = np.array([approach_vec_2d[0], approach_vec_2d[1], 0])

    # 4. 아래 방향 (Z축): 항상 월드 기준 -Z
    z_axis = np.array([0, 0, -1])

    # 5. Y축 = Z x X (오른손 좌표계)
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)

    # 6. 정규 직교 좌표계 구성
    rot_matrix = np.column_stack((x_axis, y_axis, z_axis))

    # 7. 회전 행렬 → 쿼터니언
    grasp_rot = R.from_matrix(rot_matrix)
    return grasp_rot.as_quat()  # [x, y, z, w]


class TaskImplementation(CompetitionTask):
    """
    Competition task implementation class.
    Implements actual tasks by inheriting from the CompetitionTask class.
    """
    
    def __init__(self, team_name: str, team_token: str, target_stage: str, robot_node: RobotNode):
        """
        Initialize task implementation
        
        Args:
            node (Node): ROS2 node instance
            team_name (str): Team name
            team_token (str): Team token
            target_stage (str): Target stage
        """
        super().__init__(team_name, team_token, target_stage)

        self.current_position = [0, 0, 0]
        self.current_orientation = [0, 0, 0, 1]

        self.object_detection_result = []

        self.robot_node = robot_node

        try: 
            scenario_id = os.getenv("ENV_METASEJONG_SCENARIO", "demo")
            answer_sheet_file_path = f"/data/{scenario_id}_answer_sheet.yaml"

            self.answer_sheet = self.load_yaml_to_dict(answer_sheet_file_path)
            self.robot_node.start_position = self.answer_sheet['start_position']

            self.logger.info(f"[Competitor] Successfully loaded answer sheet")
            self.logger.info(f"  - Path for answer sheet: {answer_sheet_file_path}")

        except Exception as e:
            self.logger.error(f"Failed to load answer sheet: {str(e)}")
            self.logger.error(f"  - Path for answer sheet: {answer_sheet_file_path}")
            raise

        self.robot_util = RobotUtil()

    def load_yaml_to_dict(self, yaml_path: str) -> Dict:
        """
        YAML 파일을 읽어서 딕셔너리로 변환합니다.
        
        Args:
            yaml_path (str): YAML 파일 경로
            
        Returns:
            Dict: YAML 파일의 내용이 담긴 딕셔너리
            
        Raises:
            FileNotFoundError: YAML 파일을 찾을 수 없는 경우
            yaml.YAMLError: YAML 파싱 중 에러가 발생한 경우
        """
        try:
            with open(yaml_path, 'r') as yaml_file:
                yaml_dict = yaml.safe_load(yaml_file)

                return yaml_dict
        except FileNotFoundError:
            self.logger.error(f"  - File not found: {yaml_path}")
            raise
        except yaml.YAMLError as e:
            self.logger.error(f"  - Fail to parse yaml file: {str(e)}")
            raise

    # overriding abstract method. 
    #   Stage 1 task implementation: Objecect detection and pose estimation from Fixed camera image
    def start_stage_1_task(self) -> List[Dict[str, List[int]]]:
        """
        Start stage 1 task.
        
        Returns:
            List[Dict[str, List[int]]]: List of detected objects with their classes and positions.
                Each object is a dictionary with:
                - 'class_name': str - Object class name
                - 'position': List[int] - [x, y, z] coordinates
        """
        stage1_answer = []

        #   Delay for 10 seconds for demo. emulate the time for object detection and pose estimation
        time.sleep(10)

        self.object_detection_result = self.answer_sheet['mission_objects']
        for object_detection in self.object_detection_result:
            if object_detection['recyclable'] == True:
                stage1_answer.append({
                    "class_name": object_detection['class_name'],
                    "position": object_detection['position']
                })
        
        return stage1_answer  
    
    #   Demo implementation for navigate to the target position
    #   TODO: Implement actual navigation logic
    #   README: This function is only for demo implementation.
    #           Competitor must implement actual navigation logic.  
    def _demo_implementation___locate_robot_to_target_position(
            self, 
            target_position: List[float], 
            distance_threshold: float = CONST_GLOBAL_NAVIGATION_DISTANCE,
            angle_threshold: float = math.pi/72,    # 360도 중 5도
            sleep_time: float = 0.5
        ) -> None:
        """
        Move the robot to the target position
        
        Args:
            target_position (List[float]): Target position [x, y, z]
            distance (float): Target distance threshold
        """
        # Speed control parameters definition
        max_linear_speed = 1.0  # Linear speed increased (0.5 -> 2.0)
        max_angular_speed = 0.1  # Angular speed maintained
        min_speed = 0.5  # Minimum speed increased (0.1 -> 0.5)
        while self.force_stop == False:
            try:
                # Get current position and orientation
                robot_position = self.robot_node.get_robot_position()
                robot_orientation = self.robot_node.get_robot_orientation()
                
                # Calculate distance to target
                distance = math.sqrt(
                    (target_position[0] - robot_position[0]) ** 2 +
                    (target_position[1] - robot_position[1]) ** 2
                )
                
                # Current robot direction (yaw angle)
                # Calculate yaw angle from quaternion
                x, y, z, w = robot_orientation
                yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
                
                # Calculate angle to target
                angle_to_target = math.atan2(
                    target_position[1] - robot_position[1],
                    target_position[0] - robot_position[0]
                )
                
                # Calculate angle difference and normalize (-pi ~ pi range)
                angle_diff = angle_to_target - yaw
                if angle_diff > math.pi:
                    angle_diff -= 2 * math.pi
                elif angle_diff < -math.pi:
                    angle_diff += 2 * math.pi
                    
                
                # Check target reach condition
                if distance <= distance_threshold and abs(angle_diff) <= angle_threshold:

                    velocity = MobileBaseCommander(linear_x=0.0, angular_z=0.0)
                    self.robot_node.move_robot(velocity)
                    break
                    
                # Calculate speed
                if abs(angle_diff) > angle_threshold:
                    # Prioritize rotation with slight forward movement when angle difference is large
                    linear_x = 0.0 # min_speed * (1 - abs(angle_diff)/math.pi)
                    angular_z = max_angular_speed if angle_diff > 0 else -max_angular_speed
                else:
                    # Move forward with fine adjustment when facing the target
                    linear_x = min_speed + (max_linear_speed - min_speed) * (1 - distance_threshold/distance)
                    linear_x = max(min_speed, min(linear_x, max_linear_speed))
                    angular_z = 0.0 #  0.5 * angle_diff  # Fine adjustment coefficient increased (0.2 -> 0.5)
                    
                # Obstacle check
                if self._demo_implementation___is_movement_blocked(robot_position, linear_x, angular_z):
                    linear_x = 0.0
                    angular_z = 0.5  # Clockwise rotation
                    
                # Create and send velocity command
                velocity = MobileBaseCommander(linear_x=linear_x, angular_z=angular_z)
                self.robot_node.move_robot(velocity)
                
                time.sleep(sleep_time)  # Control cycle
                
            except Exception as e:
                self.logger.error(f"Error while publishing robot navigation: {str(e)}")
                import traceback
                self.logger.error(f"  Stack trace:\n{traceback.format_exc()}")
                break

    #   Demo implementation for obstacle detection
    #   TODO: Implement actual obstacle detection logic
    #   README: This function is only for demo implementation.
    #           Competitor must implement actual obstacle detection logic.  
    def _demo_implementation___is_movement_blocked(self, robot_position: List[float], linear_x: float, angular_z: float) -> bool:
        
        # TODO: Implement actual obstacle detection logic
        # This is a placeholder that should be replaced with actual sensor data
        # THis demo implementation assumes that there is no obstacle in front of the robot

        return False

    #   Demo implementation for get object center position using robot vision
    #   TODO: Implement actual object center position estimation logic
    #   README: This function is only for demo implementation.
    #           Competitor must implement actual object center position estimation logic.     
    def _demo_implementation___get_object_center_position_using_robot_vision(self, object_detection: Dict, robot_capture_image: Image):
        """
        Get the center position of an object using robot vision
        
        Returns:
            List[float]: [x, y, z] coordinates of the object center in the robot's camera coordinate system 
        """

        return object_detection['position']
        

    #   Demo implementation for pick object
    #   TODO: Implement actual pick object logic
    #   README: This function is only for demo implementation.
    #           Competitor must implement actual pick object logic.   
    def _demo_implementation___pick_object(self, object_detection: Dict) -> None:

        #   로봇 카메라로 촬영한 이미지를 얻어서 
        robot_image = None # self._robot_image

        #   촬영 이미지에 대한 vision 분석을 통해 로봇이 이동해야 할지('mobilebase', pick&place를 수행하면 될지 판단 
        center_position = self._demo_implementation___get_object_center_position_using_robot_vision(object_detection, robot_image)
        robot_position = self.robot_node.get_robot_position()

        distance = math.sqrt(
            (center_position[0] - robot_position[0]) ** 2 +
            (center_position[1] - robot_position[1]) ** 2
        )

        if distance > CONST_GLOBAL_PICK_DISTANCE:
            self._demo_implementation___locate_robot_to_target_position(
                center_position, 
                CONST_GLOBAL_PICK_DISTANCE, 
                math.pi/720,      # 360도 중 1도
                0.1             # 0.1초 마다 움직임
            )
            
        robot_position = self.robot_node.get_robot_position()
        distance = math.sqrt(
            (center_position[0] - robot_position[0]) ** 2 +
            (center_position[1] - robot_position[1]) ** 2
        )            

        time.sleep(2)

        class_name = object_detection['class_name']

        target_rotation = object_detection['rotation']
        robot_rotation_quaternion = self.robot_node.get_robot_orientation()

        drop_position = 0.0

        
        #   Please use below code to determine the drop position depending on the object class name


        # aluminum type
        if class_name == 'master_chef_can' or class_name == 'cola_can':
            drop_position = -0.15
        # plastic type
        elif class_name == 'juice' or class_name == 'disposable_cup':
            drop_position = 0.0
        # paper type
        elif class_name == 'tissue' or class_name == 'cracker_box':
            drop_position = 0.15


        grasp_approach_quaternion = _compute_grasp_quaternion(target_rotation, robot_rotation_quaternion)
        pick_and_place_command = ManipulatorCommander(
            start_target_orientation_quat=grasp_approach_quaternion, # [0.7091, 0.7091, 0, 0],
            start_target_position=[distance, 0, -0.24],
            end_target_orientation_quat=[1, 0, 0, 0],
            end_target_position=[0.5, drop_position, -0.1]
        )

        self.robot_node.pick_up_object(pick_and_place_command)

        # loop interval time 
        time.sleep(15)


    #   Demo implementation for select next object to pick
    #   TODO: Implement actual select next object to pick logic
    #   README: This function is only for demo implementation.
    #           Competitor must implement actual select next object to pick logic.     
    def _demo_implementation___select_next_object(self):

        #   initialize recyclable_object_list
        if not hasattr(self, 'recyclable_object_list'):
            self.recyclable_object_list = [obj for obj in self.object_detection_result if obj['recyclable']]
            
        #   if there is no recyclable object, return None
        if not self.recyclable_object_list:
            return None
            
        #   get current robot position
        robot_position = self.robot_node.get_robot_position()
        
        #   find the closest object
        min_distance = float('inf')
        closest_object = None
        closest_idx = -1
        
        for idx, obj in enumerate(self.recyclable_object_list):
            obj_position = obj['position']
            distance = math.sqrt(
                (obj_position[0] - robot_position[0])**2 + 
                (obj_position[1] - robot_position[1])**2
            )
            
            if distance < min_distance:
                min_distance = distance
                closest_object = obj
                closest_idx = idx
                
        #   select the closest object and remove it from the list
        if closest_idx >= 0:
            self.recyclable_object_list.pop(closest_idx)
            
        return closest_object

    def stop_task_and_quit(self) :
        super().stop_task_and_quit()
        self.robot_node.stop_and_destroy_robot()


    #   Demo implementation for stage 2 task
    #   TODO: Implement actual stage 2 task logic
    #   README: This function is only for demo implementation.
    #           Competitor must implement actual stage 2 task logic.     
    def start_stage_2_task(self) -> bool:

        ## for all object_detection in object_detection_result
        while not self.force_stop:
            object_detection = self._demo_implementation___select_next_object()
            if object_detection is None:
                self.logger.info("  - No more objects to pick")
                break
            
            if not object_detection['recyclable'] == True:
                continue

            target_position = object_detection['position']
            
            # control robot to move to the object_detection position
            self._demo_implementation___locate_robot_to_target_position(target_position)

            time.sleep(3)

            # pick object
            self._demo_implementation___pick_object(object_detection)

        return True
