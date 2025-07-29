import numpy as np


class Kinematics:
    def __init__(self):
        self.path = []
        self.joints_stack = []

    def get_current_joints(self):
        """Returns the last solved joint configuration, or home if none."""
        return self.joints_stack[-1] if self.joints_stack else [0, 0, 0, 0, 0, 0]

    def DH_transform_std(self, theta, d, a, alpha):
        return np.array([
            [np.cos(theta), -np.sin(theta) * np.cos(alpha), np.sin(theta) * np.sin(alpha), a * np.cos(theta)],
            [np.sin(theta), np.cos(theta) * np.cos(alpha), -np.cos(theta) * np.sin(alpha), a * np.sin(theta)],
            [0, np.sin(alpha), np.cos(alpha), d],
            [0, 0, 0, 1]
        ])

    def FK(self, joints):
        """
        Forward kinematics using Standard DH parameters for Meca500:
        Based on standard Meca500 DH table
        """
        dh = [
            [joints[0], 135, 0, -np.pi / 2],  # Joint 1: base rotation
            [joints[1], 0, 0, np.pi / 2],  # Joint 2: shoulder pitch
            [joints[2], 0, 120, 0],  # Joint 3: upper arm (120mm)
            [joints[3], 0, 70, -np.pi / 2],  # Joint 4: forearm (70mm)
            [joints[4], 0, 0, np.pi / 2],  # Joint 5: wrist pitch
            [joints[5], 0, 0, 0]  # Joint 6: wrist roll
        ]
        T = np.eye(4)
        for params in dh:
            T = T @ self.DH_transform_std(*params)
        return T

    def FK_position_only(self, joints):
        """Return only the flange XYZ position (mm)"""
        return self.FK(joints)[0:3, 3]

    def _euler_to_rotation_matrix(self, alpha, beta, gamma):
        """Convert Euler XYZ angles to rotation matrix"""
        ca, sa = np.cos(alpha), np.sin(alpha)
        cb, sb = np.cos(beta), np.sin(beta)
        cc, sc = np.cos(gamma), np.sin(gamma)

        # XYZ Euler convention (intrinsic rotations)
        return np.array([
            [cb * cc, -cb * sc, sb],
            [sa * sb * cc + ca * sc, -sa * sb * sc + ca * cc, -sa * cb],
            [-ca * sb * cc + sa * sc, ca * sb * sc + sa * cc, ca * cb]
        ])

    def IK_smart(self, x, y, z, alpha, beta, gamma):
        """Tries all config seeds and picks the solution closest to previous joints."""
        prev = self.get_current_joints()
        configs = [{'cs': cs, 'ce': ce, 'cw': cw, 'ct': 0}
                   for cs in (1, -1)
                   for ce in (1, -1)
                   for cw in (1, -1)]
        best_sol, best_cfg, best_dist = None, None, float('inf')

        for cfg in configs:
            sol, ok = self.IK(x, y, z, alpha, beta, gamma, cfg)
            if ok:
                dist = sum((s - p) ** 2 for s, p in zip(sol, prev))
                if dist < best_dist:
                    best_dist, best_sol, best_cfg = dist, sol, cfg

        if best_sol is None:
            return None, False, None

        # Add the best solution to joints_stack
        self.joints_stack.append(best_sol)
        return best_sol, True, best_cfg

    def IK(self, x, y, z, alpha, beta, gamma, config_params=None):
        """
        Inverse Kinematics for Meca500 using correct DH parameters
        """
        if config_params is None:
            config_params = {'cs': 1, 'ce': 1, 'cw': 1, 'ct': 0}

        try:
            # Convert Euler angles to rotation matrix
            R_desired = self._euler_to_rotation_matrix(np.radians(alpha),
                                                       np.radians(beta),
                                                       np.radians(gamma))

            # Robot parameters (matching FK DH table)
            d1 = 135  # Base height
            a3 = 120  # Upper arm length (link 3)
            a4 = 70  # Forearm length (link 4)

            # Wrist center calculation (target position is at flange)
            wrist_center = np.array([x, y, z])

            # Joint 1: Base rotation
            theta1 = np.arctan2(wrist_center[1], wrist_center[0])

            # Apply shoulder configuration
            if config_params['cs'] == -1:
                theta1 += np.pi
                if theta1 > np.pi:
                    theta1 -= 2 * np.pi

            # 2D planar problem for joints 2 and 3
            r = np.sqrt(wrist_center[0] ** 2 + wrist_center[1] ** 2)
            s = wrist_center[2] - d1

            # Distance to wrist center from joint 2
            D = np.sqrt(r ** 2 + s ** 2)

            # Check if target is reachable
            if D > (a3 + a4) or D < abs(a3 - a4):
                return None, False

            # Joint 3: Law of cosines
            cos_theta3 = (a3 ** 2 + a4 ** 2 - D ** 2) / (2 * a3 * a4)
            cos_theta3 = np.clip(cos_theta3, -1, 1)  # Numerical safety

            theta3 = np.arccos(cos_theta3)

            # Apply elbow configuration
            if config_params['ce'] == -1:  # Elbow down
                theta3 = -theta3

            # Joint 2: Calculate shoulder angle
            alpha_angle = np.arctan2(s, r)
            beta_angle = np.arccos((a3 ** 2 + D ** 2 - a4 ** 2) / (2 * a3 * D))

            if config_params['ce'] == 1:  # Elbow up
                theta2 = alpha_angle - beta_angle
            else:  # Elbow down
                theta2 = alpha_angle + beta_angle

            # Calculate transformation matrix for first 3 joints
            T03 = self._forward_kinematics_partial([theta1, theta2, theta3], 3)

            # Calculate required rotation for last 3 joints
            R03 = T03[0:3, 0:3]
            R36 = R03.T @ R_desired

            # Extract Euler angles for joints 4, 5, 6 from R36
            theta4, theta5, theta6 = self._rotation_matrix_to_euler_zyz(R36)

            # Apply wrist configuration
            if config_params['cw'] == -1 and abs(theta5) > 1e-6:
                theta4 += np.pi
                theta5 = -theta5
                theta6 += np.pi

            # Apply turn configuration
            theta6 += config_params['ct'] * 2 * np.pi

            # Normalize angles to proper ranges
            joint_angles = [theta1, theta2, theta3, theta4, theta5, theta6]
            joint_angles = self._normalize_joint_angles(joint_angles)

            # Check joint limits
            if not self._check_joint_limits(joint_angles):
                return None, False

            return joint_angles, True

        except Exception as e:
            return None, False

    def _forward_kinematics_partial(self, joints, num_joints):
        """Calculate forward kinematics for first num_joints"""
        dh_params = [
            [joints[0], 135, 0, -np.pi / 2],  # Joint 1
            [joints[1] if len(joints) > 1 else 0, 0, 0, np.pi / 2],  # Joint 2
            [joints[2] if len(joints) > 2 else 0, 0, 120, 0],  # Joint 3
            [joints[3] if len(joints) > 3 else 0, 0, 70, -np.pi / 2],  # Joint 4
            [joints[4] if len(joints) > 4 else 0, 0, 0, np.pi / 2],  # Joint 5
            [joints[5] if len(joints) > 5 else 0, 0, 0, 0]  # Joint 6
        ]

        T = np.eye(4)
        for i in range(num_joints):
            T_i = self.DH_transform_std(*dh_params[i])
            T = T @ T_i
        return T

    def _rotation_matrix_to_euler_zyz(self, R):
        """Convert rotation matrix to ZYZ Euler angles for spherical wrist"""
        theta5 = np.arccos(np.clip(R[2, 2], -1, 1))

        if abs(np.sin(theta5)) > 1e-6:  # Not at singularity
            theta4 = np.arctan2(R[1, 2], R[0, 2])
            theta6 = np.arctan2(R[2, 1], -R[2, 0])
        else:  # At singularity
            theta4 = 0
            theta6 = np.arctan2(R[1, 0], R[0, 0])

        return theta4, theta5, theta6

    def _normalize_joint_angles(self, angles):
        """Normalize joint angles to [-pi, pi] range"""
        out = []
        for ang in angles:
            while ang > np.pi:
                ang -= 2 * np.pi
            while ang <= -np.pi:
                ang += 2 * np.pi
            out.append(ang)
        return out

    def _check_joint_limits(self, angles):
        """Check if joint angles are within mechanical limits"""
        limits = [
            (-175, 175),  # Joint 1
            (-70, 90),  # Joint 2
            (-135, 70),  # Joint 3
            (-170, 170),  # Joint 4
            (-115, 115),  # Joint 5
            (-np.inf, np.inf)  # Joint 6 (unlimited)
        ]

        for ang, (min_deg, max_deg) in zip(angles, limits):
            angle_deg = np.degrees(ang)
            if not (min_deg <= angle_deg <= max_deg):
                return False
        return True

    def test_ik_fk_consistency(self, x, y, z, alpha, beta, gamma):
        """Test IK/FK consistency for debugging"""
        print(f"\n=== Testing pose [{x}, {y}, {z}] [{alpha}, {beta}, {gamma}] ===")

        # Try IK
        joints, success, config = self.IK_smart(x, y, z, alpha, beta, gamma)

        if not success:
            print("❌ IK failed")
            return False

        print(f"✅ IK success with config: {config}")
        joints_deg = [np.degrees(j) for j in joints]
        print(f"Joint angles (deg): {[f'{j:.2f}' for j in joints_deg]}")

        # Verify with FK
        fk_result = self.FK_position_only(joints)
        print(f"FK result: [{fk_result[0]:.2f}, {fk_result[1]:.2f}, {fk_result[2]:.2f}]")

        # Calculate error
        target = np.array([x, y, z])
        error = np.linalg.norm(fk_result - target)
        print(f"Position error: {error:.3f} mm")

        if error < 0.1:  # Less than 0.1mm error
            print("✅ IK/FK consistency verified")
            return True
        else:
            print("❌ IK/FK mismatch!")
            return False




