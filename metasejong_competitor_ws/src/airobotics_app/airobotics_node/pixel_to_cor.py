import numpy as np
from  scipy.spatial.transform import Rotation as R

def qmul(a,b):
    x1,y1,z1,w1 = a;  x2,y2,z2,w2 = b
    return np.array([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2])

def quat_to_rpy(q):
    roll, pitch, yaw = R.from_quat(q).as_euler("zyx", degrees=True)
    if yaw > 180:              # 191.554° → -168.446°
        yaw -= 360
    return roll, pitch, yaw



def rpy_to_rot(rpy_deg):
    """Rotate X→Y→Z, 입력: [roll, pitch, yaw] (deg)"""
    roll, pitch, yaw = np.deg2rad(rpy_deg)
    c, s = np.cos, np.sin
    cx, cy, cz = c(roll), c(pitch), c(yaw)
    sx, sy, sz = s(roll), s(pitch), s(yaw)

    return np.array([
        [ cy*cz,               -cy*sz,              sy          ],
        [ sx*sy*cz + cx*sz,   -sx*sy*sz + cx*cz,  -sx*cy       ],
        [-cx*sy*cz + sx*sz,    cx*sy*sz + sx*cz,   cx*cy       ],
    ])                 # R_wc  (카메라 → 월드 회전 행렬)

def pixel_to_world_Z(u, v, K, C, rpy_deg, z_plane=1.0):
    """
    u,v      : 픽셀 좌표 (origin at image left-top)
    K        : 3×3 내부행렬 (fx, fy, cx, cy)
    C        : 카메라 위치 (world) 3-vector
    rpy_deg  : [roll, pitch, yaw] in degrees
    z_plane  : 투영할 월드 z 높이 (default = 1 m)
    """
    fx, fy, cx, cy = K['fx'], K['fy'], K['cx'], K['cy']

    # 픽셀 -> 카메라 좌표 단위 방향벡터
    d_c = np.array([(u - cx)/fx, (v - cy)/fy, 1.0])
    d_c /= np.linalg.norm(d_c)

    # 카메라 -> 월드 회전
    R_wc = rpy_to_rot(rpy_deg)
    d_w  = R_wc @ d_c

    # 직선 C + s*d_w 와 z = z_plane 평면의 교점
    s = (z_plane - C[2]) / d_w[2]
    X_w = C + s * d_w
    return X_w


