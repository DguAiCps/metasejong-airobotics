#!/usr/bin/env python3

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from obj_detect_module import detect_objects
#import order_decision
import numpy as np
import time

from rclpy.node import Node
from sensor_msgs.msg import Image as ROSImage, CameraInfo
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import time
import rclpy

class DepthImageSubscriber:
    """
    소형 클래스: ROS2로부터 depth image, rgb image, camera info를
    구독하고 최신 프레임을 반환하는 유틸리티입니다.
    """
    def __init__(self,
                 node: Node,
                 rgb_topic: str = '/metasejong2025/robot/center_camera_image',
                 depth_topic: str = '/metasejong2025/robot/center_camera_depth',
                 info_topic: str = '/metasejong2025/robot/center_camera_info',
                 qos_profile=None):
        self.node = node
        self.bridge = CvBridge()
        self.rgb_image: np.ndarray | None = None
        self.depth_image: np.ndarray | None = None
        self.camera_info: CameraInfo | None = None
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # 구독자 설정
        self.rgb_sub = node.create_subscription(
            ROSImage, rgb_topic, self._rgb_callback, qos_profile)
        self.depth_sub = node.create_subscription(
            ROSImage, depth_topic, self._depth_callback, qos_profile)
        self.info_sub = node.create_subscription(
            CameraInfo, info_topic, self._info_callback, qos_profile)

    def _rgb_callback(self, msg: ROSImage) -> None:
        try:
            self.rgb_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except CvBridgeError as e:
            self.node.get_logger().error(f"[DepthImageSubscriber] RGB 변환 실패: {e}")

    def _depth_callback(self, msg: ROSImage) -> None:
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except CvBridgeError as e:
            self.node.get_logger().error(f"[DepthImageSubscriber] Depth 변환 실패: {e}")

    def _info_callback(self, msg: CameraInfo) -> None:
        self.camera_info = msg

    def get_latest_depth(self, timeout: float = 5.0) -> np.ndarray:
        """
        depth image (float32, 미터 단위)를 반환합니다.
        timeout 초 내에 수신되지 않으면 TimeoutError 발생.
        """
        start = time.time()
        while self.depth_image is None:
            rclpy.spin_once(self.node, timeout_sec=0.1)
            if time.time() - start > timeout:
                raise TimeoutError(f"Depth frame not received within {timeout}s")
        return self.depth_image

    def get_latest_rgb(self, timeout: float = 5.0) -> np.ndarray:
        start = time.time()
        while self.rgb_image is None:
            rclpy.spin_once(self.node, timeout_sec=0.1)
            if time.time() - start > timeout:
                raise TimeoutError(f"RGB frame not received within {timeout}s")
        return self.rgb_image

    def get_camera_info(self, timeout: float = 5.0) -> CameraInfo:
        start = time.time()
        while self.camera_info is None:
            rclpy.spin_once(self.node, timeout_sec=0.1)
            if time.time() - start > timeout:
                raise TimeoutError(f"CameraInfo not received within {timeout}s")
        return self.camera_info


"""start_time = time.time()
#foo = detect_objects()
foo = dict({'1.jpg': [{'wood_block': [-53.06285117133758, 137.29700161160693, 17.0]}, {'juice': [-54.425261844349265, 137.4853240945785, 17.0]}, {'juice': [-53.217919537093145, 137.45358902554864, 17.0]}, {'master_chef_can': [-53.75209348188695, 135.9827471889623, 17.0]}, {'cola_can': [-53.45874406505757, 137.17811334207306, 17.0]}, {'tissue': [-52.880689039948564, 137.3048138727833, 17.0]}, {'wood_block': [-53.37086052590742, 136.25426619110493, 17.0]}, {'cola_can': [-52.23711788638051, 136.22418806580478, 17.0]}, {'tissue': [-52.60468976656616, 135.16959894410434, 17.0]}, {'master_chef_can': [-53.62042180274355, 137.20263783949557, 17.0]}, {'tissue': [-53.14706170229538, 136.95583662803915, 17.0]}, {'juice': [-53.03760945435819, 135.70967856671524, 17.0]}, {'cola_can': [-54.141706322153475, 137.18633225622375, 17.0]}, {'wood_block': [-53.35097877215893, 137.56706016941033, 17.0]}, {'master_chef_can': [-53.99972249045059, 136.2296448148203, 17.0]}], '2.jpg': [{'cola_can': [-38.65427058876027, 124.8921493705295, 17.3]}, {'wood_block': [-39.28358271058694, 124.89326656024217, 17.3]}, {'wood_block': [-38.905923945637106, 124.49820869357002, 17.3]}, {'master_chef_can': [-39.428501646110384, 123.61864899644864, 17.3]}, {'juice': [-39.08102100741113, 124.72412736831672, 17.3]}, {'juice': [-39.0239218415492, 123.7969688435851, 17.3]}, {'wood_block': [-39.71766381881872, 124.39187122147163, 17.3]}, {'juice': [-40.311500337033316, 123.70699076612304, 17.3]}, {'cola_can': [-39.714695375395024, 122.86572024007988, 17.3]}, {'master_chef_can': [-39.68431479192893, 124.01238802212632, 17.3]}, {'master_chef_can': [-38.833574595329054, 124.72315524025252, 17.3]}, {'cola_can': [-39.08248666497089, 124.19503918863296, 17.3]}], '3.jpg': []})
print(foo)

# TODO: 현재 로봇 위치(혹은 시작위치) 받아오도록 변경
robot_position = [-65.0, 130.0, 17.0]

clusters = list(foo.keys())
center_points = dict()
print(clusters)
final_order = []

# 각 클러스터의 center point 구하기
for cluster in clusters:
    print(cluster)
    items = foo[cluster]
    if np.size(items) == 0: continue
    coords = [vals for entry in items for vals in entry.values()]
    xyz = [list(axis_vals) for axis_vals in zip(*coords)]
    print(xyz)
    #plt.scatter(xyz[0], xyz[1])
    avg = [sum(axis_vals) / len(axis_vals) for axis_vals in zip(*coords)]
    center_points[cluster] = avg

center_points = [{k: v} for k, v in center_points.items()]

print(center_points)
#plt.scatter([center_points[:, 0]], [center_points[:, 1]], c='orange')

# 로봇 + 각 클러스터 center point의 방문 순서 정하기
cluster_order = order_decision.visit_order(center_points, robot_position)
print(cluster_order)

# 각 클러스터 내에서 방문 순서 정하기
# TODO: 단계별로, 수거가 완료되면 다음 클러스터 계산하도록 변경
print(f"cluster num: {len(cluster_order)}")
for i in range(len(cluster_order)):
    
    entry_pos = None
    exit_pos = None
    if i == 0:
        entry_pos = robot_position
        exit_pos = list(center_points[i+1].values())
    elif i == len(cluster_order)-1:
        entry_pos = list(center_points[i-1].values())
        exit_pos = None
    else:
        entry_pos = list(center_points[i-1].values())
        exit_pos = list(center_points[i+1].values())
    
    cluster = clusters[i]
    items = foo[cluster]
    
    intracluster_order = order_decision.visit_order(items, entry_pos, exit_pos)
    final_order += intracluster_order

print(final_order)

print(f"총 실행 시간: {time.time() - start_time:.5f} s")


object_coords = np.array([value for d in final_order for value in d.values()])
#print(object_coords)

plt.scatter(object_coords[:,0], object_coords[:,1])
plt.scatter(robot_position[0], robot_position[1], c='green')
plt.plot([robot_position[0], object_coords[0,0]], [robot_position[1], object_coords[0,1]], 'r')
for i in range(np.size(object_coords, axis=0)-1):
    p, q = object_coords[i], object_coords[i+1]
    plt.plot([p[0], q[0]], [p[1], q[1]], 'r')
plt.show()"""
rclpy.init()
node = DepthImageSubscriber(Node('test_node'))
depth = node.get_latest_depth().copy()
plt.matshow(depth, cmap='gray')
plt.colorbar()
plt.show()
print(depth[500, 500])