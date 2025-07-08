
import numpy as np
from scipy.spatial.transform import Rotation as R
import math

def _compute_grasp_quaternion(o_euler_deg, r_quat_xyzw):
    """
    물체의 Euler 각도(o_euler_deg)와 로봇의 쿼터니언(r_quat_xyzw)을 이용해
    grasp 방향 쿼터니언을 계산합니다.
    """
    rot_o = R.from_euler('xyz', o_euler_deg, degrees=True)
    long_axis = rot_o.apply([0, 0, 1])  # 물체 Z축 (world)

    approach_vec_2d = np.array([-long_axis[1], long_axis[0]])
    approach_vec_2d /= np.linalg.norm(approach_vec_2d)

    x_axis = np.array([approach_vec_2d[0], approach_vec_2d[1], 0])
    z_axis = np.array([0, 0, -1])
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)

    rot_matrix = np.column_stack((x_axis, y_axis, z_axis))
    grasp_rot = R.from_matrix(rot_matrix)
    return grasp_rot.as_quat()  # [x, y, z, w]

def quaternion_to_yaw(q):
    """
    쿼터니언 [x, y, z, w] -> yaw (heading angle)
    """
    qx, qy, qz, qw = q
    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)

def normalize_angle(theta):
    """
    radian 각도를 -pi ~ pi 범위로 정규화
    """
    while theta > math.pi:
        theta -= 2 * math.pi
    while theta < -math.pi:
        theta += 2 * math.pi
    return theta