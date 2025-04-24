# Copyright (c) 2025, IoT Convergence & Open Sharing System (IoTCOSS)
#
# All rights reserved. This software and its documentation are proprietary and confidential.
# The IoT Convergence & Open Sharing System (IoTCOSS) retains all intellectual property rights,
# including but not limited to copyrights, patents, and trade secrets, in and to this software
# and related documentation. Any use, reproduction, disclosure, or distribution of this software
# and related documentation without explicit written permission from IoTCOSS is strictly prohibited.

import json
from typing import Dict, Union, List
from dataclasses import dataclass, asdict
from enum import Enum

class MessageStatus(Enum):
    """
    Enumeration of possible message statuses.
    """
    SUCCESS = 1
    FAILED = 0

class MessageType(Enum):
    """
    Enumeration of possible message types for competitor messages.
        10x: Request from competitor to server
        20x: Response from server to competitor drived from 10x competitor request message
        30x: Notification from server to competitor

    message format: 
        10x: From competitor to server
        {
            "msg": {message_type_number}
            "session": "session_id",
            "payload": { # depends on the message type
                "stage": 1,
            }
        }
        20x: Response from server to competitor drived from 10x competitor request message
        {
            "msg": {message_type_number}
            "status": MessageStatus.SUCCESS(1) or MessageStatus.FAILED(0),
            "status_message": "message",   # "OK" if success, error message if failed
            "result": { # depends on the message type
            }
        } 
        30x: Notification from server to competitor
        {
            "msg": {message_type_number}
            "session": "session_id",
            "payload": { # depends on the message type
            }
        }
    """
    # 10x: Request from competitor to server
    COMPETITOR_APP_STARTED = 101
    REPORT_STAGE1_COMPLETED = 102
    REPORT_STAGE2_COMPLETED = 103

    # 20x: Response from server to competitor drived from 10x competitor request message
    COMPETITOR_APP_STARTED_RESPONSE = 201
    REPORT_STAGE1_COMPLETED_RESPONSE = 202
    REPORT_STAGE2_COMPLETED_RESPONSE = 203

    # 30x: Notification from server to competitor
    TIME_CONSTRAINT_EXPIRED = 301
    COMPETITOR_REQUEST_ERROR = 302


@dataclass
class CompetitorAppStartedPayload:
    """
    Payload for COMPETITOR_APP_STARTED message type.
    """
    team: str
    token: str
    stage: int

    def __post_init__(self):
        """
        Validates the payload data.
        """
        if not isinstance(self.team, str):
            raise ValueError("team must be a string")       
        if not isinstance(self.token, str):
            raise ValueError("token must be a string")
        if not isinstance(self.stage, int):
            raise ValueError("stage must be an integer")
        if self.stage < 1 or self.stage > 2:
            raise ValueError("stage must be 1 or 2")    
        

@dataclass
class ReportStage1CompletedPayload:
    """
    Payload for REPORT_STAGE1_COMPLETED message type.
    Contains a list of object_detections with their classes and positions.
    """
    object_detections: List[Dict[str, List[int]]]

    def __post_init__(self):
        """
        Validates the payload data.
        """
        if not isinstance(self.object_detections, list):
            raise ValueError("object_detections must be a list")
        
        for obj in self.object_detections:
            if not isinstance(obj, dict):
                raise ValueError("Each object must be an instance of Dict")
            if not isinstance(obj['class_name'], str):
                raise ValueError("'class_name' must be a string value")
            if not isinstance(obj['position'], list) or len(obj['position']) != 3:
                raise ValueError("'position' must be a list of 3 integers [x, y, z]")

@dataclass
class ReportStage2CompletedPayload:
    """
    Payload for REPORT_STAGE2_COMPLETED message type.
    """
    dummy: str

@dataclass
class FinishCompetitionPayload:
    """
    Payload for FINISH_COMPETITION message type.
    """
    dummy: str


@dataclass
class CompetitorRequestMessage:
    """
    Class for handling competitor request messages.
    Provides conversion between JSON string and dict format, and performs required attribute validation.
    """
    msg: MessageType
    session: str
    payload: Union[CompetitorAppStartedPayload, ReportStage1CompletedPayload, ReportStage2CompletedPayload]

    def __post_init__(self):
        """
        Performs validation of required attributes after object initialization.
        """
        if not isinstance(self.msg, MessageType):
            raise ValueError("msg must be a valid MessageType enum value")
        if not self.session:
            raise ValueError("session is a required attribute")
        if not self.payload:
            raise ValueError("payload is a required attribute")
        
        # Validate payload based on message type
        if self.msg.value == MessageType.COMPETITOR_APP_STARTED.value:
            if not isinstance(self.payload, CompetitorAppStartedPayload):
                raise ValueError("payload must be CompetitorAppStartedPayload for COMPETITOR_APP_STARTED message type")
        elif self.msg.value == MessageType.REPORT_STAGE1_COMPLETED.value:
            if not isinstance(self.payload, ReportStage1CompletedPayload):
                raise ValueError("payload must be ReportStage1CompletedPayload for REPORT_STAGE1_COMPLETED message type")
        elif self.msg.value == MessageType.REPORT_STAGE2_COMPLETED.value:
            if not isinstance(self.payload, ReportStage2CompletedPayload):
                raise ValueError("payload must be ReportStage2CompletedPayload for REPORT_STAGE2_COMPLETED message type")

    @classmethod
    def from_json(cls, json_str: str) -> 'CompetitorRequestMessage':
        """
        Creates a CompetitorRequestMessage object from a JSON string.
        
        Args:
            json_str (str): JSON formatted string
            
        Returns:
            CompetitorRequestMessage: Created object
        """
        try:
            data = json.loads(json_str)
            # data['msg'] is int as MessageType.value. So, it is not needed to be converted.
            msg_int = data['msg']
            data['msg'] = MessageType(msg_int)
            
            # 'session' is required too. But the type of session string type. So, it is not needed to be converted.

            # Convert payload to appropriate type based on message type
            if 'payload' in data and 'msg' in data:
                msg_type = data['msg']
                payload_data = data['payload']
                
                if msg_type == MessageType.COMPETITOR_APP_STARTED:
                    data['payload'] = CompetitorAppStartedPayload(**payload_data)
                elif msg_type == MessageType.REPORT_STAGE1_COMPLETED:
                    data['payload'] = ReportStage1CompletedPayload(**payload_data)
                elif msg_type == MessageType.REPORT_STAGE2_COMPLETED:
                    data['payload'] = ReportStage2CompletedPayload(**payload_data)
            
            return cls(**data)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format: {str(e)}")
        except TypeError as e:
            raise ValueError(f"Required attributes are missing: {str(e)}")
        except ValueError as e:
            raise ValueError(f"Invalid message type or payload: {str(e)}")

    def to_json(self) -> str:
        """
        Converts the object to a JSON string.
        """

        data = asdict(self)
        # Convert enum to string for JSON serialization
        data['msg'] = data['msg'].value
        return json.dumps(data, ensure_ascii=False)


@dataclass
class CompetitorAppStartedResponsePayload:
    """
    Payload for COMPETITOR_APP_STARTED message type.
    """
    session: str

@dataclass
class ReportStage1CompletedResponsePayload:
    """
    Payload for REPORT_STAGE1_COMPLETED message type.
    """
    dummy: str

@dataclass
class ReportStage2CompletedResponsePayload:
    """
    Payload for REPORT_STAGE2_COMPLETED message type.
    """
    dummy: str

@dataclass
class CompetitorResponseMessage:
    """
    Class for handling competitor notification messages.

        "msg": "message_type",
        "status": MessageStatus
        "status_message": "message",   # "OK" if success, error message if failed
        "result": { # depends on the message type
        }
    """
    msg: MessageType
    status: MessageStatus
    status_message: str
    result: Union[CompetitorAppStartedResponsePayload, ReportStage1CompletedResponsePayload, ReportStage2CompletedResponsePayload]

    def __post_init__(self):
        """
        Validates the payload data.
        """
        if not isinstance(self.msg, MessageType):
            raise ValueError("msg must be a MessageType enum")
        if not isinstance(self.status, MessageStatus):
            raise ValueError("status must be a MessageStatus enum")
        if not isinstance(self.status_message, str):
            raise ValueError("status_message must be a string")
        if not isinstance(self.result, (CompetitorAppStartedResponsePayload, ReportStage1CompletedResponsePayload, ReportStage2CompletedResponsePayload)):
            raise ValueError("result must be a valid response payload type")

    def to_json(self) -> str:
        """
        Converts the object to a JSON string.
        """
        data = asdict(self)
        # Convert enums to their values for JSON serialization
        data['msg'] = data['msg'].value
        data['status'] = data['status'].value
        return json.dumps(data, ensure_ascii=False)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'CompetitorResponseMessage':
        """
        Creates a CompetitorResponseMessage object from a JSON string.
        """
        try:
            data = json.loads(json_str)
            # Convert int values to enums
            data['msg'] = MessageType(data['msg'])
            data['status'] = MessageStatus(data['status'])

            if 'result' in data and 'msg' in data:
                msg_type = data['msg']
                result_data = data['result']
                
                if msg_type == MessageType.COMPETITOR_APP_STARTED_RESPONSE:
                    data['result'] = CompetitorAppStartedResponsePayload(**result_data)
                elif msg_type == MessageType.REPORT_STAGE1_COMPLETED_RESPONSE:
                    data['result'] = ReportStage1CompletedResponsePayload(**result_data)
                elif msg_type == MessageType.REPORT_STAGE2_COMPLETED_RESPONSE:
                    data['result'] = ReportStage2CompletedResponsePayload(**result_data)
            
            return cls(**data)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format: {str(e)}")
        except (TypeError, ValueError) as e:
            raise ValueError(f"Invalid message data: {str(e)}")


@dataclass
class TimeConstraintExpiredPayload:
    """
    Payload for TIME_CONSTRAINT_EXPIRED message type.
    """
    dummy: str

@dataclass
class CompetitorRequestErrorPayload:
    """
    Payload for COMPETITOR_REQUEST_ERROR message type.
    """
    error_message: str


@dataclass
class CompetitorNotificationMessage:
    """
    Class for handling competitor notification messages.
    """
    msg: MessageType
    session: str
    payload: Union[TimeConstraintExpiredPayload]

    def __post_init__(self):
        """
        Validates the payload data.
        """
        if not isinstance(self.msg, MessageType):
            raise ValueError("msg must be a MessageType enum")
        if not isinstance(self.session, str):
            raise ValueError("session must be a string")
        if not isinstance(self.payload, TimeConstraintExpiredPayload):
            raise ValueError("payload must be a TimeConstraintExpiredPayload")

    def to_json(self) -> str:
        """
        Converts the object to a JSON string.
        """
        data = asdict(self)
        # Convert enum to its value for JSON serialization
        data['msg'] = data['msg'].value
        return json.dumps(data, ensure_ascii=False)

    @classmethod
    def from_json(cls, json_str: str) -> 'CompetitorNotificationMessage':
        """
        Creates a CompetitorNotificationMessage object from a JSON string.
        """
        try:
            data = json.loads(json_str)
            # Convert int value to enum
            data['msg'] = MessageType(data['msg'])

            if 'payload' in data and 'msg' in data:
                msg_type = data['msg']
                payload_data = data['payload']
                
                if msg_type == MessageType.TIME_CONSTRAINT_EXPIRED:
                    data['payload'] = TimeConstraintExpiredPayload(**payload_data)

            return cls(**data)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format: {str(e)}")
        except (TypeError, ValueError) as e:
            raise ValueError(f"Invalid message data: {str(e)}")
