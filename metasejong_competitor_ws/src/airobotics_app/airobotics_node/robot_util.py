# Copyright (c) 2025, IoT Convergence & Open Sharing System (IoTCOSS)
#
# All rights reserved. This software and its documentation are proprietary and confidential.
# The IoT Convergence & Open Sharing System (IoTCOSS) retains all intellectual property rights,
# including but not limited to copyrights, patents, and trade secrets, in and to this software
# and related documentation. Any use, reproduction, disclosure, or distribution of this software
# and related documentation without explicit written permission from IoTCOSS is strictly prohibited.

import math

class RobotUtil:
    def euler_to_quaternion(self, euler):
        roll, pitch, yaw = euler
        
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)   
    
        quaternion = [
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy
        ]   

        return quaternion
    
    def quaternion_to_euler(self, quaternion):
        x, y, z, w = quaternion
        
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = math.atan2(sinr_cosp, cosr_cosp)
        
        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = math.copysign(math.pi / 2, sinp)  # Use 90 degrees if out of range
        else:
            pitch = math.asin(sinp)
        
        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        
        # Convert to degrees
        roll = math.degrees(roll)
        pitch = math.degrees(pitch)
        yaw = math.degrees(yaw)
        
        return [roll, pitch, yaw]

    def quaternion_difference(self, q1, q2):
        # Get components
        x1, y1, z1, w1 = q1
        x2, y2, z2, w2 = q2
        
        # Calculate conjugate of q1
        q1_conj = [-x1, -y1, -z1, w1]
        
        # Multiply q2 by conjugate of q1 to get the difference
        # Quaternion multiplication: q2 * q1_conj
        x = w2 * q1_conj[0] + x2 * q1_conj[3] + y2 * q1_conj[2] - z2 * q1_conj[1]
        y = w2 * q1_conj[1] - x2 * q1_conj[2] + y2 * q1_conj[3] + z2 * q1_conj[0]
        z = w2 * q1_conj[2] + x2 * q1_conj[1] - y2 * q1_conj[0] + z2 * q1_conj[3]
        w = w2 * q1_conj[3] - x2 * q1_conj[0] - y2 * q1_conj[1] - z2 * q1_conj[2]
        
        # Normalize the result
        norm = math.sqrt(x*x + y*y + z*z + w*w)
        if norm > 0:
            x /= norm
            y /= norm
            z /= norm
            w /= norm
        
        return [x, y, z, w] 