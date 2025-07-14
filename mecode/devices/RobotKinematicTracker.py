import numpy as np  # NumPy is used for math functions and matrix operations
class RobotKinematicTracker:
    def __init__(self):
        self.joints_stack=[[0,0,0,0,0,0]] # home pos
        self.path=[] # store path


    def matrix(self,theta, d, a, alpha):
        """
        This function constructs the Denavit-Hartenberg (DH) transformation matrix
        for a single joint of a robot arm using four DH parameters:

        Parameters:
        - theta (θ): Joint angle (rotation about z-axis), in radians
        - d     : Link offset (distance along z-axis)
        - a     : Link length (distance along x-axis)
        - alpha (α): Link twist (rotation about x-axis), in radians

        The DH matrix represents the transformation from one joint's coordinate
        frame to the next in the robotic kinematic chain.

        It performs 4 steps in order:
        1. Rotate by θ about the z-axis
        2. Translate by d along the z-axis
        3. Translate by a along the x-axis
        4. Rotate by α about the x-axis

        The resulting 4x4 matrix includes both rotation and translation.
        """
        # Precompute trigonometric values for efficiency and readability
        ct, st = np.cos(theta), np.sin(theta)  # cos(θ), sin(θ)
        ca, sa = np.cos(alpha), np.sin(alpha)  # cos(α), sin(α)

        # Construct the DH transformation matrix using the standard formula
        return np.array([
            [ct, -st * ca, st * sa, a * ct],
            [st, ct * ca, -ct * sa, a * st],
            [0, sa, ca, d],
            [0, 0, 0, 1]
        ])


    def forward_kin(self,joints_deg):
        """
        Computes the forward kinematics for the Meca500 robot arm using DH parameters.

        Parameters:
        - joints_deg: A list of 6 joint angles in degrees from a command like MoveJoints()

        Steps:
        1. Convert joint angles to radians (since numpy trig functions use radians)
        2. Define the DH parameter table for the meca500
        3. For each joint, compute the DH transformation matrix using the matrix() function
        4. Multiply the matrices in order to get the full transformation from base to end-effector
        5. Extract the (x, y, z) position of the end-effector from the final transformation matrix

        Returns:
        - A 3-element NumPy array containing the (x, y, z) position of the end-effector in meters
        """
        # Step 1: Convert input joint angles from degrees to radians
        joints = np.radians(joints_deg)

        # Step 2: Define the DH parameters table
        # Format per row: [thetaᵢ, dᵢ, aᵢ, alphaᵢ]
        # All units in meters and radians
        dh_params = [
            [joints[0], 0.0, 0.0, np.pi / 2],  # Joint 1: rotates, no offset, 90° twist
            [joints[1], 0.0, 0.135, 0.0],  # Joint 2: 135mm link, same axis
            [joints[2], 0.0, 0.135, 0.0],  # Joint 3: 135mm link, same axis
            [joints[3], 0.038, 0.0, np.pi / 2],  # Joint 4: 38mm z offset, 90° twist
            [joints[4], 0.0, 0.0, -np.pi / 2],  # Joint 5: twist back -90°
            [joints[5], 0.07, 0.0, 0.0],  # Joint 6: tool offset 70mm along z
        ]

        # Step 3: Initialize the final transformation matrix as the identity matrix
        # This represents the base frame before applying any transformations
        T = np.eye(4)

        # Step 4: Multiply transformation matrices for each joint
        # This chains the transformations from base to end-effector
        for theta, d, a, alpha in dh_params:
            T = np.dot(T, self.matrix(theta, d, a, alpha))  # Matrix multiplication: T = T × Tᵢ

        # Step 5: Extract the position (x, y, z) from the final transformation matrix
        # The position is stored in the last column of the 4x4 matrix
        position = T[:3, 3]  # Extract top 3 values of the last column
        rotation_matrix = T[:3, :3] # Orientation

        return position*1000,rotation_matrix  # Return end-effector position in mm

    def track_abs(self,joints_angles): # track movejoint abs commands
        self.joints_stack.append(joints_angles.copy())
    def track_rel(self,joints_angles):
        last=self.joints_stack[-1]
        new_angle= [a+b for a,b in zip(last,joints_angles.copy())]
        self.joints_stack.append(new_angle)

    def euler_xyz_to_rotMatrix(self, alpha_deg, beta_deg, gamma_deg):
        """
        Converts Euler angles (α, β, γ) using XYZ intrinsic convention to a 3x3 rotation matrix.
        Angles should be provided in degrees.
        """
        α = np.radians(alpha_deg)
        β = np.radians(beta_deg)
        γ = np.radians(gamma_deg)

        # Rotation about X (α)
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(α), -np.sin(α)],
            [0, np.sin(α), np.cos(α)]
        ])
        # Rotation about Y (β)
        Ry = np.array([
            [np.cos(β), 0, np.sin(β)],
            [0, 1, 0],
            [-np.sin(β), 0, np.cos(β)]
        ])

        # Rotation about Z (γ)
        Rz = np.array([
            [np.cos(γ), -np.sin(γ), 0],
            [np.sin(γ), np.cos(γ), 0],
            [0, 0, 1]
        ])

        R=Rz@Ry@Rx
        return R

    def compute_wrist_center(self, x, y, z, alpha, beta, gamma):
        """
        Computes the wrist center (origin of joint 4) given end-effector pose.

        All units are in millimeters and degrees.

        Returns:
        - P_wc: Wrist center position [x, y, z] in mm
        - R: Rotation matrix (3x3)
        """
        R = self.euler_xyz_to_rotMatrix(alpha, beta, gamma)
        P_tcp = np.array([x, y, z])  # already in mm
        d6 = 70  # Distance from wrist center to TCP in mm
        z_axis = R[:, 2]  # ẑ of tool frame
        P_wc = P_tcp - d6 * z_axis  # Wrist center = TCP - d6 * ẑ
        return P_wc, R

    def solve_position_ik(self, P_wc):
        """
        Solves for joints 1–3 given the wrist center position.
        Returns joint angles [j1, j2, j3] in degrees.
        """
        x, y, z = P_wc
        a2 = 135
        a3 = 135
        d4 = 38  # z-offset between joint 3 and 4

        # 1. Joint 1
        j1 = np.arctan2(y, x)

        # 2. Planar projection for joints 2 and 3
        r = np.hypot(x, y)  # distance from base to wrist in XY
        z_prime = z - 0  # if d1 = 0

        # 3. Cosine Law to solve j3
        D = (r ** 2 + z_prime ** 2 - a2 ** 2 - a3 ** 2) / (2 * a2 * a3)
        if abs(D) > 1:
            raise ValueError("Unreachable position for IK.")

        j3 = np.arccos(D)

        # 4. Solve for j2 using triangle geometry
        phi1 = np.arctan2(z_prime, r)
        phi2 = np.arctan2(a3 * np.sin(j3), a2 + a3 * np.cos(j3))
        j2 = phi1 - phi2

        # Convert to degrees
        j1_deg = np.degrees(j1)
        j2_deg = np.degrees(j2)
        j3_deg = np.degrees(j3)

        return [j1_deg, j2_deg, j3_deg]

    def compute_T0_3(self, joints_deg):
        """
        Computes transformation matrix from base to joint 3 (T0_3) for given joint angles.
        """
        joints = np.radians(joints_deg[:3])  # Only first 3 joints

        dh_params = [
            [joints[0], 0.0, 0.0, np.pi / 2],
            [joints[1], 0.0, 0.135, 0.0],
            [joints[2], 0.0, 0.135, 0.0],
        ]

        T = np.eye(4)
        for theta, d, a, alpha in dh_params:
            T = np.dot(T, self.matrix(theta, d, a, alpha))

        return T

    def solve_orientation_ik(self, R_desired, j1_to_j3):
        """
        Given the rotation matrix of the desired pose (R_desired) and the first 3 joint angles,
        compute joint angles j4, j5, j6 based on wrist orientation (rotation matrix).

        Returns [j4, j5, j6] in degrees.
        """
        # Step 1: Compute R0_3
        T0_3 = self.compute_T0_3(j1_to_j3)
        R0_3 = T0_3[:3, :3]

        # Step 2: Compute R3_6
        R3_6 = R0_3.T @ R_desired

        # Step 3: Extract Euler angles using ZYZ (common for wrist rotation)
        # Note: This assumes standard configuration and avoids gimbal lock
        if abs(R3_6[2, 2]) < 1.0:
            j5 = np.arccos(R3_6[2, 2])
            j4 = np.arctan2(R3_6[1, 2], R3_6[0, 2])
            j6 = np.arctan2(R3_6[2, 1], -R3_6[2, 0])
        else:
            # Gimbal lock: j5 = 0 or pi
            j5 = 0.0 if R3_6[2, 2] > 0 else np.pi
            j4 = 0.0
            j6 = np.arctan2(R3_6[1, 0], R3_6[0, 0])

        # Convert to degrees
        j4_deg = np.degrees(j4)
        j5_deg = np.degrees(j5)
        j6_deg = np.degrees(j6)

        return [j4_deg, j5_deg, j6_deg]

    def solve_full_ik(self, x, y, z, alpha, beta, gamma):
        P_wc, R = self.compute_wrist_center(x, y, z, alpha, beta, gamma)
        j1_to_j3 = self.solve_position_ik(P_wc)
        j4_to_j6 = self.solve_orientation_ik(R, j1_to_j3)
        return j1_to_j3 + j4_to_j6






