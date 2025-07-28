import numpy as np


class Kinematics:
    def __init__(self):  # Fixed: was **init**
        self.path = []
        self.joints_stack = []

    def get_current_joints(self):
        """Get the current joint configuration from the stack"""
        if self.joints_stack:
            return self.joints_stack[-1]
        else:
            return [0, 0, 0, 0, 0, 0]

    def DH_transform_std(self, theta, d, a, alpha):
        return np.array([
            [np.cos(theta), -np.sin(theta) * np.cos(alpha), np.sin(theta) * np.sin(alpha), a * np.cos(theta)],
            [np.sin(theta), np.cos(theta) * np.cos(alpha), -np.cos(theta) * np.sin(alpha), a * np.sin(theta)],
            [0, np.sin(alpha), np.cos(alpha), d],
            [0, 0, 0, 1]
        ])

    def DH_transform_mod(self, theta, d, a, alpha):
        return np.array([
            [np.cos(theta), -np.sin(theta), 0, a],
            [np.sin(theta) * np.cos(alpha), np.cos(theta) * np.cos(alpha), -np.sin(alpha), -np.sin(alpha) * d],
            [np.sin(theta) * np.sin(alpha), np.cos(theta) * np.sin(alpha), np.cos(alpha), np.cos(alpha) * d],
            [0, 0, 0, 1]
        ])

    def FK(self, joints):
        # Standard DH method
        dh_params_std = [
            [joints[0], 135, 0, -np.pi / 2],  # Joint 1: base rotation
            [joints[1], 0, 0, np.pi / 2],  # Joint 2: shoulder pitch
            [joints[2], 0, 120, 0],  # Joint 3: upper arm
            [joints[3], 0, 70, -np.pi / 2],  # Joint 4: forearm
            [joints[4], 0, 0, np.pi / 2],  # Joint 5: wrist pitch
            [joints[5], 0, 0, 0]  # Joint 6: wrist roll
        ]

        T = np.eye(4)

        for i, params in enumerate(dh_params_std):
            T_i = self.DH_transform_std(*params)
            T = T @ T_i
            # Uncomment below for debugging
            # print(f"Joint {i+1}: theta={np.degrees(params[0]):.1f}°, d={params[1]}, a={params[2]}, alpha={np.degrees(params[3]):.1f}°")
            # print(f"T_{i+1}:")
            # print(T_i)
            # print(f"Cumulative T_0^{i+1}:")
            # print(T)
            # print("-" * 50)

        return T

    def FK_position_only(self, joints):
        """Return only the position vector [x, y, z]"""
        T = self.FK(joints)
        return T[0:3, 3]

    def FK_full(self, joints):
        """Return full transformation matrix"""
        return self.FK(joints)

    def FK_with_base_offset(self, joints, base_z_offset=0):
        """
        Return transformation matrix with base offset
        base_z_offset: distance from base bottom to joint 1 axis (estimated ~50-70mm)
        """
        T = self.FK(joints)
        # Add base offset to Z position
        T[2, 3] += base_z_offset
        return T

    def IK_smart(self, x, y, z, alpha, beta, gamma):
        """
        Smart IK that considers previous joint configuration from joints_stack
        Tries all configurations and picks the one closest to previous position
        """
        # Get previous joints from stack
        previous_joints = self.get_current_joints()

        # All possible configurations
        all_configs = [
            {'cs': 1, 'ce': 1, 'cw': 1, 'ct': 0},  # Default
            {'cs': -1, 'ce': 1, 'cw': 1, 'ct': 0},  # Back shoulder
            {'cs': 1, 'ce': -1, 'cw': 1, 'ct': 0},  # Elbow down
            {'cs': -1, 'ce': -1, 'cw': 1, 'ct': 0},  # Back shoulder + elbow down
            {'cs': 1, 'ce': 1, 'cw': -1, 'ct': 0},  # Wrist flip
            {'cs': -1, 'ce': 1, 'cw': -1, 'ct': 0},  # Back shoulder + wrist flip
            {'cs': 1, 'ce': -1, 'cw': -1, 'ct': 0},  # Elbow down + wrist flip
            {'cs': -1, 'ce': -1, 'cw': -1, 'ct': 0}  # All flipped
        ]

        valid_solutions = []

        # Try all configurations
        for config in all_configs:
            joints, success = self.IK(x, y, z, alpha, beta, gamma, config)
            if success:
                # Calculate distance from previous configuration
                distance = sum((j_new - j_prev) ** 2 for j_new, j_prev in zip(joints, previous_joints))
                valid_solutions.append((joints, distance, config))

        if not valid_solutions:
            return None, False, None

        # Sort by distance and return closest solution
        valid_solutions.sort(key=lambda x: x[1])
        best_joints, _, best_config = valid_solutions[0]

        return best_joints, True, best_config

    def IK(self, x, y, z, alpha, beta, gamma, config_params=None):
        """
        Original IK method (kept for compatibility)
        """
        if config_params is None:
            config_params = {'cs': 1, 'ce': 1, 'cw': 1, 'ct': 0}

        # Convert Euler angles to rotation matrix
        R_desired = self._euler_to_rotation_matrix(np.radians(alpha),
                                                   np.radians(beta),
                                                   np.radians(gamma))

        # Robot parameters (from DH table)
        d1 = 135  # Base height
        a3 = 120  # Upper arm length
        a4 = 70  # Forearm length

        try:
            # Calculate wrist center position
            wrist_center = np.array([x, y, z])

            # Joint 1: Simple atan2 from wrist center position
            theta1 = np.arctan2(wrist_center[1], wrist_center[0])

            # Apply shoulder configuration
            if config_params['cs'] == -1:
                theta1 += np.pi
                if theta1 > np.pi:
                    theta1 -= 2 * np.pi

            # Distance calculations for joints 2 and 3
            r = np.sqrt(wrist_center[0] ** 2 + wrist_center[1] ** 2)
            s = wrist_center[2] - d1

            # Distance to wrist center from joint 2
            D = np.sqrt(r ** 2 + s ** 2)

            # Check if target is reachable
            if D > (a3 + a4) or D < abs(a3 - a4):
                return None, False

            # Joint 3: Law of cosines
            cos_theta3 = (a3 ** 2 + a4 ** 2 - D ** 2) / (2 * a3 * a4)
            cos_theta3 = np.clip(cos_theta3, -1, 1)

            theta3 = np.arccos(cos_theta3)

            # Apply elbow configuration
            if config_params['ce'] == -1:
                theta3 = -theta3

            # Joint 2: More complex calculation
            alpha_angle = np.arctan2(s, r)
            beta_angle = np.arccos((a3 ** 2 + D ** 2 - a4 ** 2) / (2 * a3 * D))

            if config_params['ce'] == 1:
                theta2 = alpha_angle - beta_angle
            else:
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
            print(f"IK calculation failed: {e}")
            return None, False

    def FK_tcp_position(self, joints, base_z_offset=50, tool_z_offset=123):
        """
        Return TCP position matching online mode
        base_z_offset: base bottom to joint 1 (~50mm estimate)
        tool_z_offset: flange to TCP (173-50=123mm estimate)
        """
        T = self.FK_with_base_offset(joints, base_z_offset)
        # Add tool offset along flange Z-axis (joint 6 axis)
        tool_offset = T[0:3, 2] * tool_z_offset  # Z-axis of flange frame
        T[0:3, 3] += tool_offset
        return T[0:3, 3]

    def IK(self, x, y, z, alpha, beta, gamma, config_params=None):
        """
        Inverse Kinematics for Meca500

        Args:
            x, y, z: desired TCP position (mm)
            alpha, beta, gamma: desired orientation (degrees) - Euler XYZ convention
            config_params: dict with 'cs', 'ce', 'cw', 'ct' (optional)

        Returns:
            joint_angles: list of 6 joint angles in radians
            success: boolean indicating if solution was found
        """
        if config_params is None:
            config_params = {'cs': 1, 'ce': 1, 'cw': 1, 'ct': 0}  # Default config

        # Convert Euler angles to rotation matrix
        R_desired = self._euler_to_rotation_matrix(np.radians(alpha),
                                                   np.radians(beta),
                                                   np.radians(gamma))

        # Robot parameters (from DH table)
        d1 = 135  # Base height
        a3 = 120  # Upper arm length
        a4 = 70  # Forearm length

        try:
            # Calculate wrist center position (subtract tool offset if needed)
            wrist_center = np.array([x, y, z]) - R_desired @ np.array([0, 0, 0])  # No tool offset for now

            # Joint 1: Simple atan2 from wrist center position
            theta1 = np.arctan2(wrist_center[1], wrist_center[0])

            # Apply shoulder configuration
            if config_params['cs'] == -1:
                theta1 += np.pi
                if theta1 > np.pi:
                    theta1 -= 2 * np.pi

            # Distance calculations for joints 2 and 3
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
            elbow_threshold = -np.arctan(60 / 19)  # -72.43 degrees from manual
            if config_params['ce'] == -1:  # Elbow down
                theta3 = -theta3

            # Joint 2: More complex calculation
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
            if config_params['cw'] == -1 and theta5 > 0:
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
            print(f"IK calculation failed: {e}")
            return None, False

    def _euler_to_rotation_matrix(self, alpha, beta, gamma):
        """Convert Euler XYZ angles to rotation matrix"""
        ca, sa = np.cos(alpha), np.sin(alpha)
        cb, sb = np.cos(beta), np.sin(beta)
        cc, sc = np.cos(gamma), np.sin(gamma)

        # Mobile XYZ convention (intrinsic rotations)
        R = np.array([
            [cb * cc, -cb * sc, sb],
            [sa * sb * cc + ca * sc, -sa * sb * sc + ca * cc, -sa * cb],
            [-ca * sb * cc + sa * sc, ca * sb * sc + sa * cc, ca * cb]
        ])
        return R

    def _forward_kinematics_partial(self, joints, num_joints):
        """Calculate forward kinematics for first num_joints"""
        dh_params_std = [
            [joints[0], 135, 0, -np.pi / 2],  # Joint 1
            [joints[1], 0, 0, np.pi / 2],  # Joint 2
            [joints[2], 0, 120, 0],  # Joint 3
            [joints[3] if len(joints) > 3 else 0, 0, 70, -np.pi / 2],  # Joint 4
            [joints[4] if len(joints) > 4 else 0, 0, 0, np.pi / 2],  # Joint 5
            [joints[5] if len(joints) > 5 else 0, 0, 0, 0]  # Joint 6
        ]

        T = np.eye(4)
        for i in range(num_joints):
            T_i = self.DH_transform_std(*dh_params_std[i])
            T = T @ T_i
        return T

    def _rotation_matrix_to_euler_zyz(self, R):
        """Convert rotation matrix to ZYZ Euler angles for spherical wrist"""
        # This is a simplified version - you may need more robust implementation
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
        normalized = []
        for angle in angles:
            while angle > np.pi:
                angle -= 2 * np.pi
            while angle <= -np.pi:
                angle += 2 * np.pi
            normalized.append(angle)
        return normalized

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

        for i, (angle, (min_deg, max_deg)) in enumerate(zip(angles, limits)):
            angle_deg = np.degrees(angle)
            if angle_deg < min_deg or angle_deg > max_deg:
                return False
        return True

    def IK_debug(self, x, y, z, alpha, beta, gamma, config_params=None):
        """
        Debug version of IK to see where it's failing
        """
        if config_params is None:
            config_params = {'cs': 1, 'ce': 1, 'cw': 1, 'ct': 0}

        print(f"\n=== IK Debug for pose [{x}, {y}, {z}] [{alpha}, {beta}, {gamma}] ===")
        print(f"Config: {config_params}")

        try:
            # Convert Euler angles to rotation matrix
            R_desired = self._euler_to_rotation_matrix(np.radians(alpha),
                                                       np.radians(beta),
                                                       np.radians(gamma))
            print(f"✓ Rotation matrix calculated")

            # Robot parameters (from DH table)
            d1 = 135  # Base height
            a3 = 120  # Upper arm length
            a4 = 70  # Forearm length

            # Calculate wrist center position
            wrist_center = np.array([x, y, z])
            print(f"Wrist center: {wrist_center}")

            # Joint 1: Simple atan2 from wrist center position
            theta1 = np.arctan2(wrist_center[1], wrist_center[0])
            print(f"Theta1 (before config): {np.degrees(theta1):.2f}°")

            # Apply shoulder configuration
            if config_params['cs'] == -1:
                theta1 += np.pi
                if theta1 > np.pi:
                    theta1 -= 2 * np.pi
            print(f"Theta1 (after config): {np.degrees(theta1):.2f}°")

            # Distance calculations for joints 2 and 3
            r = np.sqrt(wrist_center[0] ** 2 + wrist_center[1] ** 2)
            s = wrist_center[2] - d1
            print(f"r = {r:.2f}, s = {s:.2f}")

            # Distance to wrist center from joint 2
            D = np.sqrt(r ** 2 + s ** 2)
            print(f"D = {D:.2f}")

            # Check if target is reachable
            max_reach = a3 + a4
            min_reach = abs(a3 - a4)
            print(f"Reachability check: {min_reach:.2f} <= {D:.2f} <= {max_reach:.2f}")

            if D > max_reach:
                print(f"✗ Target too far! D={D:.2f} > max_reach={max_reach:.2f}")
                return None, False
            if D < min_reach:
                print(f"✗ Target too close! D={D:.2f} < min_reach={min_reach:.2f}")
                return None, False
            print("✓ Target is reachable")

            # Joint 3: Law of cosines
            cos_theta3 = (a3 ** 2 + a4 ** 2 - D ** 2) / (2 * a3 * a4)
            print(f"cos_theta3 = {cos_theta3:.4f}")

            if cos_theta3 > 1 or cos_theta3 < -1:
                print(f"✗ cos_theta3 out of range: {cos_theta3}")
                return None, False

            cos_theta3 = np.clip(cos_theta3, -1, 1)
            theta3 = np.arccos(cos_theta3)
            print(f"Theta3 (before config): {np.degrees(theta3):.2f}°")

            # Apply elbow configuration
            if config_params['ce'] == -1:
                theta3 = -theta3
            print(f"Theta3 (after config): {np.degrees(theta3):.2f}°")

            # Joint 2: More complex calculation
            alpha_angle = np.arctan2(s, r)
            beta_angle = np.arccos((a3 ** 2 + D ** 2 - a4 ** 2) / (2 * a3 * D))
            print(f"alpha_angle = {np.degrees(alpha_angle):.2f}°")
            print(f"beta_angle = {np.degrees(beta_angle):.2f}°")

            if config_params['ce'] == 1:
                theta2 = alpha_angle - beta_angle
            else:
                theta2 = alpha_angle + beta_angle
            print(f"Theta2: {np.degrees(theta2):.2f}°")

            # Calculate transformation matrix for first 3 joints
            T03 = self._forward_kinematics_partial([theta1, theta2, theta3], 3)
            print("✓ T03 calculated")

            # Calculate required rotation for last 3 joints
            R03 = T03[0:3, 0:3]
            R36 = R03.T @ R_desired
            print("✓ R36 calculated")

            # Extract Euler angles for joints 4, 5, 6 from R36
            theta4, theta5, theta6 = self._rotation_matrix_to_euler_zyz(R36)
            print(
                f"Wrist angles (before config): θ4={np.degrees(theta4):.2f}°, θ5={np.degrees(theta5):.2f}°, θ6={np.degrees(theta6):.2f}°")

            # Apply wrist configuration
            if config_params['cw'] == -1 and abs(theta5) > 1e-6:
                theta4 += np.pi
                theta5 = -theta5
                theta6 += np.pi

            # Apply turn configuration
            theta6 += config_params['ct'] * 2 * np.pi

            print(
                f"Wrist angles (after config): θ4={np.degrees(theta4):.2f}°, θ5={np.degrees(theta5):.2f}°, θ6={np.degrees(theta6):.2f}°")

            # Normalize angles to proper ranges
            joint_angles = [theta1, theta2, theta3, theta4, theta5, theta6]
            joint_angles = self._normalize_joint_angles(joint_angles)

            joint_angles_deg = [np.degrees(j) for j in joint_angles]
            print(f"Final joint angles (deg): {[f'{j:.2f}' for j in joint_angles_deg]}")

            # Check joint limits
            limits_passed = self._check_joint_limits(joint_angles)
            print(f"Joint limits check: {'✓ PASSED' if limits_passed else '✗ FAILED'}")

            if not limits_passed:
                # Show which joints failed
                limits = [(-175, 175), (-70, 90), (-135, 70), (-170, 170), (-115, 115), (-np.inf, np.inf)]
                for i, (angle, (min_deg, max_deg)) in enumerate(zip(joint_angles, limits)):
                    angle_deg = np.degrees(angle)
                    if angle_deg < min_deg or angle_deg > max_deg:
                        print(f"  Joint {i + 1}: {angle_deg:.2f}° violates limits [{min_deg}, {max_deg}]")
                return None, False

            print("✓ IK SUCCESS!")
            return joint_angles, True

        except Exception as e:
            print(f"✗ Exception in IK: {e}")
            import traceback
            traceback.print_exc()
            return None, False

    # Test function to debug all configurations
    def debug_all_configs(self, x, y, z, alpha, beta, gamma):
        """Test all configurations and see which ones work"""
        all_configs = [
            {'cs': 1, 'ce': 1, 'cw': 1, 'ct': 0},  # Default
            {'cs': -1, 'ce': 1, 'cw': 1, 'ct': 0},  # Back shoulder
            {'cs': 1, 'ce': -1, 'cw': 1, 'ct': 0},  # Elbow down
            {'cs': -1, 'ce': -1, 'cw': 1, 'ct': 0},  # Back shoulder + elbow down
            {'cs': 1, 'ce': 1, 'cw': -1, 'ct': 0},  # Wrist flip
            {'cs': -1, 'ce': 1, 'cw': -1, 'ct': 0},  # Back shoulder + wrist flip
            {'cs': 1, 'ce': -1, 'cw': -1, 'ct': 0},  # Elbow down + wrist flip
            {'cs': -1, 'ce': -1, 'cw': -1, 'ct': 0}  # All flipped
        ]

        print(f"\n=== Testing all configurations for pose [{x}, {y}, {z}] [{alpha}, {beta}, {gamma}] ===")

        working_configs = []
        for i, config in enumerate(all_configs):
            print(f"\n--- Configuration {i + 1}: {config} ---")
            joints, success = self.IK_debug(x, y, z, alpha, beta, gamma, config)
            if success:
                working_configs.append((i + 1, config, joints))
                print(f"✓ Config {i + 1} WORKS!")
            else:
                print(f"✗ Config {i + 1} FAILED")

        print(f"\n=== SUMMARY ===")
        print(f"Working configurations: {len(working_configs)}")
        for config_num, config, joints in working_configs:
            joints_deg = [np.degrees(j) for j in joints]
            print(f"Config {config_num}: {[f'{j:.1f}' for j in joints_deg]}")

        return working_configs


# Test the kinematics
if __name__ == "__main__":
    kin = Kinematics()

    print("=== Forward Kinematics Test ===")
    # Test home position
    joints_home = [0, 0, 0, 0, 0, 0]  # radians
    T_home = kin.FK_full(joints_home)
    pos_home = kin.FK_position_only(joints_home)

    print("Home position test:")
    print(f"Position: [{pos_home[0]:.2f}, {pos_home[1]:.2f}, {pos_home[2]:.2f}] mm")
    print()

    print("=== Inverse Kinematics Test ===")
    # Test IK with home position
    x, y, z = pos_home[0], pos_home[1], pos_home[2]
    alpha, beta, gamma = 0, 0, 0  # Home orientation

    print(f"Target pose: [{x:.2f}, {y:.2f}, {z:.2f}] mm, [{alpha}, {beta}, {gamma}]°")

    # Try IK
    joint_solution, success = kin.IK(x, y, z, alpha, beta, gamma)

    if success:
        print("IK Success!")
        print(f"Joint solution (deg): {[np.degrees(j) for j in joint_solution]}")

        # Verify by running FK on the solution
        verification_pos = kin.FK_position_only(joint_solution)
        print(f"FK verification: [{verification_pos[0]:.2f}, {verification_pos[1]:.2f}, {verification_pos[2]:.2f}] mm")

        # Check error
        error = np.linalg.norm(np.array([x, y, z]) - verification_pos)
        print(f"Position error: {error:.3f} mm")
    else:
        print("IK Failed!")

    print("\n=== Testing Different Configurations ===")
    # Test different configurations
    configs = [
        {'cs': 1, 'ce': 1, 'cw': 1, 'ct': 0},  # Default
        {'cs': -1, 'ce': 1, 'cw': 1, 'ct': 0},  # Back shoulder
        {'cs': 1, 'ce': -1, 'cw': 1, 'ct': 0},  # Elbow down
        {'cs': 1, 'ce': 1, 'cw': -1, 'ct': 0},  # Wrist flip
    ]

    test_pose = [150, 100, 200, 0, 0, 0]  # Reachable pose

    for i, config in enumerate(configs):
        print(f"\nConfig {i + 1}: cs={config['cs']}, ce={config['ce']}, cw={config['cw']}")
        joints, success = kin.IK_debug(*test_pose, config_params=config)
        if success:
            joints_deg = [np.degrees(j) for j in joints]
            print(f"  Joints (deg): {[f'{j:.1f}' for j in joints_deg]}")

            # Verify
            verify_pos = kin.FK_position_only(joints)
            error = np.linalg.norm(np.array(test_pose[:3]) - verify_pos)
            print(f"  Verification error: {error:.3f} mm")
        else:
            print("  Failed to find solution")
























