# Copyright (c) 2025, IoT Convergence & Open Sharing System (IoTCOSS)
#
# All rights reserved. This software and its documentation are proprietary and confidential.
# The IoT Convergence & Open Sharing System (IoTCOSS) retains all intellectual property rights,
# including but not limited to copyrights, patents, and trade secrets, in and to this software
# and related documentation. Any use, reproduction, disclosure, or distribution of this software
# and related documentation without explicit written permission from IoTCOSS is strictly prohibited.

from multiprocessing import dummy
import time
import traceback
from .competitor_request_message import (
    CompetitorAppStartedPayload,
    CompetitorNotificationMessage,
    CompetitorRequestMessage,
    CompetitorResponseMessage,
    MessageStatus,
    MessageType,
    ReportStage1CompletedPayload,
    ReportStage2CompletedPayload,
)
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import String
from typing import Optional, Dict, Any, List, Union
import queue
import sys

CONST_COMPETITOR_INTERFACE_NAMESPACE = 'metasejong2025'
CONST_TOPIC_COMPETITOR_REQUEST = f"/{CONST_COMPETITOR_INTERFACE_NAMESPACE}/competitor_request"
CONST_TOPIC_COMPETITOR_RESPONSE = f"/{CONST_COMPETITOR_INTERFACE_NAMESPACE}/competitor_response"
CONST_TOPIC_COMPETITOR_NOTIFICATION = f"/{CONST_COMPETITOR_INTERFACE_NAMESPACE}/competitor_notification"

class CompetitionTask(Node):
    """
    Base class for competition tasks.
    All competition tasks must inherit from this class.
    """
    
    def __init__(self, team_name: str, team_token: str, target_stage: int):
        """
        Initialize competition task
        
        Args:
            team_name (str): Team name
            team_token (str): Team token
            target_stage (int): Target stage
        """
        super().__init__('airobotics_node')

        self.team_name = team_name
        self.team_token = team_token
        self.session = None
        self.target_stage = target_stage
        self.logger = self.get_logger()
        
        # Task state
        self.force_stop = False

        # QoS settings
        self.qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        # Initialize competitor request publisher
        self.competitor_request_pub = self.create_publisher(
            String,
            CONST_TOPIC_COMPETITOR_REQUEST,
            self.qos_profile
        )

        self.message_queue = queue.Queue()

        # Initialize competitor response subscriber
        def competitor_response_callback(msg: String):
            """
            Callback for competitor response message
            
            Args:
                msg (String): Response message
            """
            self.logger.info(f"[Platform->Competitor] RESPONSE message received")
            
            # Parse the response message
            response_message = CompetitorResponseMessage.from_json(msg.data)
            self.logger.info(f"  - message type: {response_message.msg.name}({response_message.msg.value})")

            # Process the response message
            if response_message.msg.value == MessageType.COMPETITOR_APP_STARTED_RESPONSE.value:
                self.message_queue.put(response_message)
                  
            elif response_message.msg.value == MessageType.REPORT_STAGE1_COMPLETED_RESPONSE.value:
                self.message_queue.put(response_message)

            elif response_message.msg.value == MessageType.REPORT_STAGE2_COMPLETED_RESPONSE.value:
                self.message_queue.put(response_message)

        # Initialize competitor notification subscriber
        def competitor_notification_callback(msg: String):
            """
            Callback for competitor notification message
            
            Args:
                msg (String): Notification message
            """
            self.logger.info(f"[Platform->Competitor] NOTIFICATION message received")
            
            # Parse the notification message
            notification_message = CompetitorNotificationMessage.from_json(msg.data)
            self.logger.info(f"  - message type: {notification_message.msg.name}({notification_message.msg.value})")

            if notification_message.msg.value == MessageType.TIME_CONSTRAINT_EXPIRED.value:
                self.stop_task_and_quit()
                self.message_queue.put(notification_message)

        # Initialize ROS topic subscribers
        self.competitor_response_sub = self.create_subscription(
            String,
            CONST_TOPIC_COMPETITOR_RESPONSE,
            competitor_response_callback,
            self.qos_profile
        )

        self.competitor_notification_sub = self.create_subscription(
            String,
            CONST_TOPIC_COMPETITOR_NOTIFICATION,
            competitor_notification_callback,
            self.qos_profile
        )

    # Fetch one message from the MetaSejong Platform     
    def update_one(self):
        try : 
            mesage = self.message_queue.get_nowait()
            return mesage
        except queue.Empty: 
            pass
        except Exception as e:
            self.logger.error(f"Error while fetch message from MetaSejong Prlatform: {e}")
            traceback.print_exc()
            raise e


    def start_competition_tasks(self) -> None:
        """
        Start the competition tasks and set up necessary subscriptions.
        This function should be called after creating the task implementation.
        """

        # Start stage 1 task
        self.logger.info(f"[Competitor] Start Competition")
        stage1_task_result = self.start_stage_1_task()

        self.report_stage1_completed(stage1_task_result)
        self.logger.info("  - Completed stage 1 task and report the result to MetaSejong Platform")



    def report_competitor_app_started(self) -> None:
        """
        Report that the competitor application has started.
        """
        self.logger.info("[Competitor->Platform] Reporting competitor application started")

        competitor_request_message = CompetitorRequestMessage(
            msg=MessageType.COMPETITOR_APP_STARTED,
            session="dummy_session",
            payload=CompetitorAppStartedPayload(
                team=self.team_name,
                token=self.team_token,
                stage=self.target_stage,
            )
        )

        self._send_competitor_request_messsage(competitor_request_message)

        # Wait for the response or notification from the MetaSejong Platform
        while not self.force_stop:

            # Fetch one message from the MetaSejong Platform
            message = self.update_one()
            if message is not None:

                # Start the competition tasks
                if message.msg.value == MessageType.COMPETITOR_APP_STARTED_RESPONSE.value:
                    if message.status == MessageStatus.FAILED:
                        self.logger.error("  - Error whild verify competitor authentication token: ", message.status_message)
                        self.cleanup()
                        self.stop_task_and_quit()
                        break
                    else:
                        self.logger.info("  - Start the competition")
                        self.session = message.result.session
                        self.start_competition_tasks()

                elif message.msg.value == MessageType.REPORT_STAGE1_COMPLETED_RESPONSE.value:
                    self.logger.info("  - Stage 1 completed")

                    if self.target_stage == 1:
                        self.logger.info("  - Completed stage 1 and finish competition")
                        self.cleanup()
                        self.stop_task_and_quit()
                        break

                    elif self.target_stage == 2:
                        self.logger.info("  - Completed stage 1 and start stage 2 task")
                        self.start_stage_2_task()

                elif message.msg.value == MessageType.REPORT_STAGE2_COMPLETED_RESPONSE.value:
                    self.logger.info("  - Completed stage 1 and finish competition")
                    self.cleanup()
                    self.stop_task_and_quit()
                    break

                elif message.msg.value == MessageType.TIME_CONSTRAINT_EXPIRED.value:
                    self.logger.info("  - Time constraint expired. Finish competition")
                    self.cleanup()
                    self.stop_task_and_quit()
                    break

            time.sleep(0.5)


    def _send_competitor_request_messsage(self, competitor_request_message: CompetitorRequestMessage) -> None:
        """
        Send the competitor request message.
        
        Args:
            competitor_request_message (CompetitorRequestMessage): Message to send
        """
        self.logger.info(f"[Competitor->Platform]   - Send competitor request message: {competitor_request_message.msg.name}({competitor_request_message.msg.value})")

        ros_msg = String()
        ros_msg.data = competitor_request_message.to_json()
        self.competitor_request_pub.publish(ros_msg)
      
    
    def report_stage1_completed(self, object_detection_result: List[Dict[str, List[int]]]) -> None:
        """
        Report stage 1 competition results.
        
        Args:
            object_detection_result (List[Dict[str, List[int]]]): List of detected objects with their classes and positions.
                Each object is a dictionary with:
                - 'class_name': str - Object class name
                - 'position': List[int] - [x, y, z] coordinates  
        Returns:
            None
        """

        # save the result for stage 2 task  
        self.stage1_task_result = object_detection_result

        self.logger.info("[Competitor->Platform] Reporting stage 1 completed")

        # report the result
        stage1_completed_message = CompetitorRequestMessage(
            msg=MessageType.REPORT_STAGE1_COMPLETED,
            session=self.session,
            payload=ReportStage1CompletedPayload(
                object_detections=object_detection_result
            )
        )   
        self._send_competitor_request_messsage(stage1_completed_message)

    def report_stage2_completed(self) -> None:
        """
        Report stage 2 competition results.
        
        Returns:
            None
        """

        self.logger.info("[Competitor->Platform] Reporting stage 2 completed")

        stage2_completed_message = CompetitorRequestMessage(
            msg=MessageType.REPORT_STAGE2_COMPLETED,
            session=self.session,
            payload=ReportStage2CompletedPayload(
                dummy = "dummy"
            )
        )
        self._send_competitor_request_messsage(stage2_completed_message)

    def stop_task_and_quit(self) :
        self.force_stop = True

        self.cleanup()
        self.destroy_node()

    # Abstract method for stage 1 task
    #   Competitor must override this method
    def start_stage_1_task(self) -> List[Dict[str, List[int]]]:
        """
        Start stage 1 task.
        
        Returns:
            bool: Task start success status
        """
        raise NotImplementedError("start_stage_1_task method must be implemented in the subclass.")
    
    # Abstract method for stage 2 task
    #   Competitor must override this method
    def start_stage_2_task(self) -> bool:
        """
        Start stage 2 task.
        
        Returns:
            bool: Task start success status
        """
        raise NotImplementedError("start_stage_2_task method must be implemented in the subclass.")
    
    
    # Abstract method for cleanup resources
    #   Competitor must override this method
    def cleanup(self) -> None:
        """
        Clean up resources after task completion or interruption
        """
