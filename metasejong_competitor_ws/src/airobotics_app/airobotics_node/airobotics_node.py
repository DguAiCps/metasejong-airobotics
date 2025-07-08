# Copyright (c) 2025, IoT Convergence & Open Sharing System (IoTCOSS)
#
# All rights reserved. This software and its documentation are proprietary and confidential.
# The IoT Convergence & Open Sharing System (IoTCOSS) retains all intellectual property rights,
# including but not limited to copyrights, patents, and trade secrets, in and to this software
# and related documentation. Any use, reproduction, disclosure, or distribution of this software
# and related documentation without explicit written permission from IoTCOSS is strictly prohibited.

#!/usr/bin/env python3

import os
import threading
from .robot_node import RobotNode
import rclpy
from rclpy.node import Node
from .competition_task_implementation import TaskImplementation
import traceback
from rclpy.executors import MultiThreadedExecutor

###############################################################################
# AiroboticsNode is the main node for the Airobotics application.
###############################################################################
class AiroboticsNode(Node):
    def __init__(self):
        super().__init__('airobotics_node')
        
        # Check required environment variables
        self.team_name = os.getenv('ENV_METASEJONG_TEAM_NAME', '')
        self.team_token = os.getenv('ENV_METASEJONG_TEAM_TOKEN', '')
        self.target_stage = int(os.getenv('ENV_METASEJONG_TEAM_TARGET_STAGE', 1))
        self.scenario_field = os.getenv('ENV_METASEJONG_SCENARIO', 'demo')

        if self.scenario_field == 'demo':
            if self.team_name == "REPLACE_YOUR_TEAM_NAME_HEAR" or self.team_name == "" or self.team_name == None:
                self.team_name = os.getenv('ENV_DEMO_TEAM_NAME')
            if self.team_token == "REPLACE_YOUR_TEAM_TOKEN_HEAR" or self.team_token == "" or self.team_token == None:
                self.team_token = os.getenv('ENV_DEMO_TEAM_TOKEN')

        self.logger = self.get_logger()

        # Validate environment variables
        if not all([self.team_name, self.team_token, self.target_stage]):
            self.logger.error('--[ERROR]---------------------------------------------------')
            self.logger.error('Required environment variables are not set.')
            self.logger.error(' # Please set the following environment variables:')
            self.logger.error('   - ENV_METASEJONG_TEAM_NAME')
            self.logger.error('   - ENV_METASEJONG_TEAM_TOKEN')
            self.logger.error('   - ENV_METASEJONG_TEAM_TARGET_STAGE')
            self.logger.error('----------------------------------------------------------')
            raise EnvironmentError('Required environment variables are not set.')

        # Print competitor information
        self.logger.info('=======================================================')
        self.logger.info('         MetaCom 2025 Student Compatition')
        self.logger.info('-------------------------------------------------------')
        self.logger.info(' Welcome to the MetaSejong AI Robotics Competition')
        self.logger.info(' # Competitor Information ')
        self.logger.info(f'   - Team name: {self.team_name}')
        self.logger.info(f'   - Target stage: {self.target_stage}')
        self.logger.info(' # Applying Scenario')
        self.logger.info(f'   - Scenario Field: {self.scenario_field}')
        self.logger.info('=======================================================')

###############################################################################
# Main function
###############################################################################
def main(args=None):
    print("m")
    rclpy.init(args=args)
    print("m")
    # Create AiroboticsNode instance
    airobotics_node = AiroboticsNode()
    print("m")
    # Create RobotNode instance
    # RobotNode is the main node for the robot status .
    robot_node = RobotNode()
    print("m")
    # Create TaskImplementation instance
    # TaskImplementation is the main node for the competition protocol implementation.
    # This node shows an example of how to implement the competition protocol.
    task_impl = TaskImplementation(airobotics_node.team_name, airobotics_node.team_token, airobotics_node.target_stage, robot_node)
    print("m")
    try:
        # Create MultiThreadedExecutor
        # MultiThreadedExecutor is the main node for the competition protocol implementation.
        executor = MultiThreadedExecutor(num_threads=4)
        executor.add_node(airobotics_node)
        executor.add_node(robot_node)
        executor.add_node(task_impl)
    
        # Start ROS thread
        ros_thread = threading.Thread(
            target=executor.spin,
            daemon=True
        )
        ros_thread.start()

        task_impl.report_competitor_app_started()

        # Wait until thread terminates
        ros_thread.join()
                
    except Exception as e:
        logger = airobotics_node.get_logger()
        logger.error(f"Error while task execution: {str(e)}")
        logger.error(traceback.format_exc())
    finally:
        # Proper shutdown sequence
        try:
            logger = airobotics_node.get_logger()
            logger.info('=======================================================')
            logger.info(' Completed mission of MetaSejong AI Robotics Challenge')
            logger.info(' Thanks for your participation!')
            logger.info('=======================================================')
   
            # Destroy nodes
            airobotics_node.destroy_node()
            robot_node.destroy_node()
            task_impl.destroy_node()
            
            # Shutdown ROS
            rclpy.shutdown()

            # Force exit if needed
            os._exit(0)
        except Exception as e:
            logger.error(f"Error while shutting down: {str(e)}")
            os._exit(1)

if __name__ == '__main__':
    main()
