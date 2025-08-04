import numpy as np
import math


class Kinematics:
    """
    Kinematics for the Mecademic Meca500 6-axis robot using Standard Denavit–Hartenberg parameters.
    Tracks the end-effector path for visualization or logging.
    """

    def __init__(self, tol=1e-3, max_iters=50, lambda2=0.01):
        # DH parameters per joint: [theta_offset_deg, d_mm, a_mm, alpha_deg]
        self.dh_params = [
            [0, 135.0, 0.0, -90],  # Joint 1: d1=135mm, a1=0
            [-90, 0.0, 135.0, 0],  # Joint 2: a2=135mm
            [0, 0.0, 38.0, -90],  # Joint 3: a3=38mm
            [0, 120.0, 0.0, 90],  # Joint 4: d4=120mm
            [0, 0.0, 0.0, -90],  # Joint 5
            [180, 70.0, 0.0, 0],  # Joint 6: d6=70mm
        ]

        self.joint_limits = [
            (-175, 175),  # θ1
            (-70, 90),  # θ2
            (-135, 70),  # θ3
            (-170, 170),  # θ4
            (-115, 115),  # θ5
            (None, None)  # θ6 no mechanical limits
        ]



        # Initialize storage for sampled joint sets and TCP positions
        self.joints_stack = []
        self.path = []
        self.tol = tol
        self.max_iters = max_iters
        self.lambda2 = lambda2



    def DH_transform(self, theta_rad, d, a, alpha_rad):
        """
        Build the individual joint transform using Standard DH convention.
        """
        cth = np.cos(theta_rad)
        sth = np.sin(theta_rad)
        cal = np.cos(alpha_rad)
        sal = np.sin(alpha_rad)
        return np.array([
            [cth, -sth * cal, sth * sal, a * cth],
            [sth, cth * cal, -cth * sal, a * sth],
            [0, sal, cal, d],
            [0, 0, 0, 1],
        ])

    def _within_limits(self, angles):
        """Return True if each angle is within its joint limit."""
        for i, a in enumerate(angles):
            mn, mx = self.joint_limits[i]
            if mn is not None and a < mn: return False
            if mx is not None and a > mx: return False
        return True
    def FK(self, joint_angles_deg):
        """
        Forward kinematics: compute the TCP position and orientation.
        Appends each computed TCP position to self.path.

        Args:
            joint_angles_deg (list of 6 floats): joint positions in degrees [θ1…θ6]
        Returns:
            position (3-array): [x, y, z] in mm
            rotation (3x3 array): rotation matrix
        """
        if len(joint_angles_deg) != 6:
            raise ValueError(f"Expected 6 joint angles, got {len(joint_angles_deg)}")

        T = np.eye(4)
        # Chain transforms
        for i, angle_deg in enumerate(joint_angles_deg):
            theta_off, d, a, alpha = self.dh_params[i]
            theta = np.deg2rad(angle_deg + theta_off)
            alpha_rad = np.deg2rad(alpha)
            T_i = self.DH_transform(theta, d, a, alpha_rad)
            T = T @ T_i

        # Extract TCP
        position = T[0:3, 3]
        position_mm = np.round(position, decimals=6)
        rotation = T[0:3, 0:3]

        # Store for path logging
        self.joints_stack.append(joint_angles_deg.copy())
        self.path.append(position_mm.tolist())

        return position, rotation




