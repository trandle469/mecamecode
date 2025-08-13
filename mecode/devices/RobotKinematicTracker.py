# RobotKinematicTracker.py
# Meca500 URDF-based FK (TRF==FRF). Degrees in, mm/deg out.

import numpy as np
from pathlib import Path
from typing import List, Optional
from urchin import URDF  # maintained fork of urdfpy




def _wrap_deg(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0

def _clip_deg(v: float, lo_hi):
    lo, hi = lo_hi
    if lo is not None: v = max(v, lo)
    if hi is not None: v = min(v, hi)
    return v

def _safe_axis(j) -> np.ndarray:
    """Return a 3-vector axis for the joint, defaulting to [0,0,1]."""
    ax = getattr(j, "axis", None)
    if ax is None:
        return np.array([0.0, 0.0, 1.0], float)
    ax = np.asarray(ax, dtype=float).reshape(-1)
    if ax.size != 3 or not np.isfinite(ax).all():
        return np.array([0.0, 0.0, 1.0], float)
    return ax

def Rx(a):
    ca, sa = np.cos(a), np.sin(a)
    return np.array([[1,0,0,0],[0,ca,-sa,0],[0,sa,ca,0],[0,0,0,1]], float)

def Ry(b):
    cb, sb = np.cos(b), np.sin(b)
    return np.array([[cb,0,sb,0],[0,1,0,0],[-sb,0,cb,0],[0,0,0,1]], float)

def Rz(c):
    cc, sc = np.cos(c), np.sin(c)
    return np.array([[cc,-sc,0,0],[sc,cc,0,0],[0,0,1,0],[0,0,0,1]], float)

def hat(v):
    x,y,z = v
    return np.array([[0,-z,y],[z,0,-x],[-y,x,0]], float)

def rodrigues(axis, theta):
    axis = np.asarray(axis, float)
    n = np.linalg.norm(axis)
    if n == 0:
        return np.eye(3)
    a = axis / n
    K = hat(a)
    return np.eye(3) + np.sin(theta)*K + (1 - np.cos(theta))*(K@K)

def xyzrpy_to_T(xyz, rpy):
    # URDF joint origin uses fixed-axis RPY: Rz(yaw)*Ry(pitch)*Rx(roll)
    x,y,z = xyz
    rr,pp,yy = rpy
    T = Rz(yy) @ Ry(pp) @ Rx(rr)
    T[0:3, 3] = [x, y, z]
    return T

def T_from_R_t(R, t):
    T = np.eye(4)
    T[:3,:3] = R
    T[:3, 3] = t
    return T

def rot_to_euler_xyz(R):
    """ Returns (α,β,γ) in radians."""
    sy = float(-R[2,0])
    sy = np.clip(sy, -1.0, 1.0)              # guard for round-off
    cy = np.sqrt(max(0.0, 1 - sy*sy))
    if cy > 1e-9:
        alpha = np.arctan2(R[2,1], R[2,2])   # about X
        beta  = np.arcsin(sy)                # about Y
        gamma = np.arctan2(R[1,0], R[0,0])   # about Z
    else:
        # gimbal lock fallback
        alpha = np.arctan2(-R[1,2], R[1,1])
        beta  = np.arcsin(sy)
        gamma = 0.0
    return alpha, beta, gamma

def euler_xyz_to_R(alpha, beta, gamma):
    """ Angles in radians."""
    return (Rx(alpha) @ Ry(beta) @ Rz(gamma))[:3, :3]

def _skew(a):
    x, y, z = a
    return np.array([[0, -z,  y],
                     [z,  0, -x],
                     [-y, x,  0]], float)

def _se3_log(T):
    """
    Body log map
    """
    R = T[:3, :3]
    p = T[:3, 3]
    tr = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    theta = np.arccos(tr)

    if theta < 1e-7:
        # small-angle approximation
        w = 0.5 * np.array([R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1]])
        V_inv = np.eye(3) - 0.5 * _skew(w)
        v = V_inv @ p
    else:
        w_hat = (R - R.T) * (0.5/np.sin(theta))
        w = np.array([w_hat[2,1], w_hat[0,2], w_hat[1,0]]) * theta
        wx = _skew(w/theta)
        A = (1 - np.cos(theta)) / (theta**2)
        B = (theta - np.sin(theta)) / (theta**3)
        V = np.eye(3) + A*wx + B*(wx@wx)
        v = np.linalg.solve(V, p)

    return np.r_[v, w]

def _body_error(T_curr, T_target, w_pos=1.0, w_ori=1.0):
    """
    6x1 body twist error e
    """
    R = T_curr[:3,:3].T
    p = -R @ T_curr[:3,3]
    Tinv = np.eye(4); Tinv[:3,:3] = R; Tinv[:3,3] = p
    Delta = Tinv @ T_target
    xi = _se3_log(Delta)
    xi[:3] *= w_pos
    xi[3:] *= w_ori
    return xi

def _unwrap_near(prev, cand):
    """Choose equivalent angle (cand + 2πk) nearest to prev."""
    k = np.round((prev - cand) / (2*np.pi))
    return cand + k*(2*np.pi)

def _wrap_pi(a):
    return (a + np.pi) % (2*np.pi) - np.pi

def normalize_euler_mecademic_xyz(a, b, c):
    """
    Mecademic canonicalization for mobile XYZ:
    """
    a = _wrap_pi(a)
    b = np.clip(b, -np.pi/2, np.pi/2)
    c = _wrap_pi(c)
    if abs(abs(b) - np.pi/2) < 1e-9:
        c = _wrap_pi(c + a)
        a = 0.0
    return a, b, c

rad = np.deg2rad
deg = np.rad2deg

# ---------- Meca500 limits ----------
MECA_LIMITS_DEG = [
    (-175.0,  175.0),  # J1
    ( -70.0,   90.0),  # J2
    (-135.0,   70.0),  # J3
    (-170.0,  170.0),  # J4
    (-115.0,  115.0),  # J5
    (None, None)       # J6 continuous; software ±100 turns
]
J6_SW_LIMIT_RAD = 100 * 2*np.pi  # ±100 turns

def _name_of(x):
    """Return name if object has .name, else assume it's already a string."""
    return x if isinstance(x, str) else getattr(x, "name", x)

def _origin_to_T(origin):
    """
     convert a joint.origin to a 4x4 transform.
    - urchin typically stores origin as a (4,4) ndarray
    - urdfpy sometimes used objects with .xyz/.rpy
    """
    if origin is None:
        return np.eye(4)
    # ndarray case
    if isinstance(origin, np.ndarray) and origin.shape == (4,4):
        return origin
    # object with xyz/rpy
    xyz = getattr(origin, "xyz", None)
    rpy = getattr(origin, "rpy", None)
    if xyz is not None and rpy is not None:
        return xyzrpy_to_T(xyz, rpy)
    # try array-like fallback
    arr = np.asarray(origin)
    if arr.shape == (4,4):
        return arr
    # last resort
    return np.eye(4)


class Kinematics:
    """
    Forward kinematics from URDF (base: meca_base_link -> tip: meca_axis_6_link).
    takes degrees; returns mm & degrees (mobile XYZ).
    - TRF == FRF (no tool).
    """
    def __init__(self, urdf_path: Optional[str] = None,
                 base_link: str = "meca_base_link",
                 tip_link: str  = "meca_axis_6_link",
                 enforce_limits: bool = True):
        if urdf_path is None:
            urdf_path = Path(__file__).resolve().parent / "meca_500_r3.urdf"
        self.urdf_path = Path(urdf_path)

        # urchin: skip mesh I/O entirely
        self.robot = URDF.load(str(self.urdf_path), lazy_load_meshes=True)
        self.base_link = base_link
        self.tip_link  = tip_link
        self.enforce_limits = enforce_limits

        # drawing / logs
        self.path:   List[List[float]] = []   # global pose log (mm,deg)
        self.joints: List[List[float]] = []   # global joint log (deg)
        self.pen_down: bool = False
        self.strokes: list = []               # strokes with points/joints
        self.teleports: list = []             # pen-up moves

        # ---- build joint chain (base->tip) without URDF.get_chain ----
        by_child = { _name_of(j.child): j for j in self.robot.joints }

        seq = []
        cur = self.tip_link
        guard = 0
        while cur != self.base_link:
            guard += 1
            if guard > 1000:
                raise RuntimeError("Chain walk overflow—check base_link/tip_link names.")
            j = by_child.get(cur)
            if j is None:
                raise ValueError(
                    f"No joint found whose child is '{cur}'. "
                    f"Check base_link='{self.base_link}', tip_link='{self.tip_link}', "
                    f"and confirm the URDF contains that chain."
                )
            seq.append(j)
            cur = _name_of(j.parent)

        seq.reverse()  # base->tip order

        self._chain = []
        for j in seq:
            T_pj = _origin_to_T(getattr(j, "origin", None))
            jt = getattr(j, "joint_type", None) or getattr(j, "type", None) or "fixed"
            axis = _safe_axis(j)

            limit = getattr(j, "limit", None)
            lower = getattr(limit, "lower", None) if limit is not None else None
            upper = getattr(limit, "upper", None) if limit is not None else None

            self._chain.append({
                "name":  getattr(j, "name", "<unnamed>"),
                "type":  jt,
                "axis":  axis,
                "T_pj":  T_pj,        #
                "lower": lower,
                "upper": upper,
            })

        self._movables = [i for i,e in enumerate(self._chain)
                          if e["type"] in ("revolute","continuous","prismatic")]
        if len(self._movables) != 6:
            print(f"[WARN] Expected 6 actuated joints; found {len(self._movables)}. Check base/tip names.")

    # ----- limits -----
    def _check_limits(self, q_deg: List[float]):
        if len(q_deg) != 6:
            raise ValueError("Expected 6 joint values (deg) for Meca500.")
        for i, qd in enumerate(q_deg, start=1):
            lo, hi = MECA_LIMITS_DEG[i-1]
            if i == 6:
                if abs(rad(qd)) > J6_SW_LIMIT_RAD:
                    raise ValueError(f"J6 violates ±100 turns (got {qd} deg).")
            else:
                if lo is not None and qd < lo - 1e-9:
                    raise ValueError(f"J{i} below limit: {qd} < {lo} deg.")
                if hi is not None and qd > hi + 1e-9:
                    raise ValueError(f"J{i} above limit: {qd} > {hi} deg.")

    # ----- FK core -----
    def fk_T(self, q_deg: List[float]) -> np.ndarray:
        """Return 4x4 base->FRF transform."""
        if self.enforce_limits:
            self._check_limits(q_deg)

        q_rad = [rad(v) for v in q_deg]
        T = np.eye(4)
        qi = 0
        for e in self._chain:
            T = T @ e["T_pj"]
            t = e["type"]
            if t in ("revolute","continuous"):
                R = rodrigues(e["axis"], q_rad[qi])
                T = T @ T_from_R_t(R, np.zeros(3))
                qi += 1
            elif t == "prismatic":
                # (Meca500 has no prismatic joints, but keep generic)
                T = T @ T_from_R_t(np.eye(3), e["axis"] * q_rad[qi])
                qi += 1
            elif t == "fixed":
                pass
            else:
                raise NotImplementedError(f"Unsupported joint type: {t}")
        return T

    def fk_pose(self, q_deg: List[float]):
        """Return (x,y,z, α,β,γ) with x,y,z in mm and α,β,γ in deg."""
        T = self.fk_T(q_deg)
        R, t = T[:3,:3], T[:3,3]
        a,b,c = rot_to_euler_xyz(R)
        a,b,c = normalize_euler_mecademic_xyz(a,b,c)
        return float(t[0]*1e3), float(t[1]*1e3), float(t[2]*1e3), float(deg(a)), float(deg(b)), float(deg(c))

    # ----- optional: quick introspection -----
    def describe_chain(self) -> str:
        rows = []
        for i,e in enumerate(self._chain):
            rows.append(
                f"{i:02d}  {e['name']:<22} {e['type']:<10} axis={e['axis']}\n"
                f"{np.array_str(e['T_pj'], precision=3)}"
            )
        return "\n".join(rows)

    # ---------- IK (Jacobian DLS) ----------

    def _fk_T_rad(self, q_rad: np.ndarray) -> np.ndarray:
        """FK wrapper that takes radians  converting to degrees"""
        return self.fk_T([float(deg(v)) for v in q_rad])

    def _numeric_body_jacobian(self, q_rad: np.ndarray, h: float = 1e-4) -> np.ndarray:
        """
        Body Jacobian J_b at q_rad (6x6) using limit-aware finite differences.
        Uses central difference when both sides are feasible; otherwise one-sided.
        """
        T0 = self._fk_T_rad(q_rad)
        J = np.zeros((6, 6))

        # mechanical limits (rad). J6 has software limit ±J6_SW_LIMIT_RAD
        lims = []
        for i in range(6):
            if i < 5:
                lo, hi = MECA_LIMITS_DEG[i]
                lo = -np.inf if lo is None else np.deg2rad(lo)
                hi = np.inf if hi is None else np.deg2rad(hi)
            else:
                lo, hi = -J6_SW_LIMIT_RAD, J6_SW_LIMIT_RAD
            lims.append((lo, hi))

        eye = np.eye(6)
        for i in range(6):
            lo, hi = lims[i]
            hp = h if q_rad[i] + h <= hi else 0.0
            hm = h if q_rad[i] - h >= lo else 0.0

            try:
                if hp and hm:  # central
                    Tp = self._fk_T_rad(q_rad + eye[i] * hp)
                    Tm = self._fk_T_rad(q_rad - eye[i] * hm)
                    xp = _se3_log(np.linalg.inv(T0) @ Tp)
                    xm = _se3_log(np.linalg.inv(T0) @ Tm)
                    J[:, i] = (xp - xm) / (hp + hm)
                elif hp:  # forward only
                    Tp = self._fk_T_rad(q_rad + eye[i] * hp)
                    xp = _se3_log(np.linalg.inv(T0) @ Tp)
                    J[:, i] = xp / hp
                elif hm:  # backward only
                    Tm = self._fk_T_rad(q_rad - eye[i] * hm)
                    xm = _se3_log(np.linalg.inv(T0) @ Tm)
                    J[:, i] = -xm / hm
                else:
                    J[:, i] = 0.0
            except Exception:
                # extremely rare; keep column zero and continue
                J[:, i] = 0.0

        return J

    def _build_target_T(self, xyz_mm, euler_deg) -> np.ndarray:
        """Build 4x4 target transform from (mm, deg) using mobile XYZ Euler."""
        x, y, z = [float(v)/1000.0 for v in xyz_mm]  # mm -> m
        a, b, c = [rad(float(v)) for v in euler_deg] # deg -> rad
        R = euler_xyz_to_R(a, b, c)
        T = np.eye(4); T[:3,:3] = R; T[:3,3] = [x, y, z]
        return T

    def ik_solve_dls(self,
                     target_xyz_mm=None,
                     target_euler_deg=None,
                     target_T: np.ndarray = None,
                     q0_deg: List[float] = None,
                     w_pos: float = 1.0,
                     w_ori: float = 1.0,
                     lam0: float = 1e-2,
                     max_iters: int = 120,
                     step_deg_max: float = 5.0,
                     tol_pos_mm: float = 0.2,
                     tol_ori_deg: float = 0.2):
        """
        Damped least-squares IK with simple backtracking line search.

        Provide either:
          - target_xyz_mm=(x,y,z) and target_euler_deg=(α,β,γ)  [mobile XYZ], or
          - target_T (4x4 SE(3) in meters/radians).

        q0_deg: optional seed (deg). If None, uses zeros.

        Returns:
          (q_deg: list[6], success: bool, iters: int)
        """
        if target_T is None:
            if target_xyz_mm is None or target_euler_deg is None:
                raise ValueError("Provide either (target_xyz_mm, target_euler_deg) or target_T.")
            T_star = self._build_target_T(target_xyz_mm, target_euler_deg)
        else:
            T_star = np.array(target_T, float)

        # Seed (radians)
        if q0_deg is None:
            q = np.zeros(6)
        else:
            q = np.array([np.deg2rad(float(v)) for v in q0_deg], float)

        lam = lam0
        step_max = np.deg2rad(step_deg_max)
        pos_tol_m = tol_pos_mm / 1000.0
        ori_tol = np.deg2rad(tol_ori_deg)

        for it in range(max_iters):
            T = self._fk_T_rad(q)
            e = _body_error(T, T_star, w_pos, w_ori)
            pos_err = np.linalg.norm(e[:3]) / max(w_pos, 1e-9)
            ori_err = np.linalg.norm(e[3:]) / max(w_ori, 1e-9)
            if pos_err < pos_tol_m and ori_err < ori_tol:
                return [float(np.rad2deg(v)) for v in q], True, it

            # DLS step
            J = self._numeric_body_jacobian(q)
            JTJ = J.T @ J
            step = np.linalg.solve(JTJ + (lam ** 2) * np.eye(6), J.T @ e)

            # Trust region clamp (max absolute joint motion per iter)
            m = float(np.max(np.abs(step)))
            if m > step_max:
                step *= (step_max / m)

            # Candidate update with limits
            q_new = q + step

            # Enforce J1..J5 mechanical limits (in radians)
            for i in range(5):
                lo, hi = MECA_LIMITS_DEG[i]
                if lo is not None: q_new[i] = max(q_new[i], np.deg2rad(lo))
                if hi is not None: q_new[i] = min(q_new[i], np.deg2rad(hi))

            # J6: unwrap nearest + software ±100 turns
            q_new[5] = _unwrap_near(q[5], q_new[5])
            q_new[5] = np.clip(q_new[5], -J6_SW_LIMIT_RAD, J6_SW_LIMIT_RAD)

            # --- Accept/reject with simple backtracking line search ---
            T_new = self._fk_T_rad(q_new)
            e_new = _body_error(T_new, T_star, w_pos, w_ori)

            if np.linalg.norm(e_new) < np.linalg.norm(e):
                # Good full step: accept and relax damping
                q = q_new
                lam = max(lam / 2.0, 1e-6)
            else:
                # Try scaled steps before giving up (helps near limits/singularities)
                improved = False
                for s in (0.5, 0.25, 0.1):
                    q_try = q + s * step

                    # Enforce limits again on the trial
                    for i in range(5):
                        lo, hi = MECA_LIMITS_DEG[i]
                        if lo is not None: q_try[i] = max(q_try[i], np.deg2rad(lo))
                        if hi is not None: q_try[i] = min(q_try[i], np.deg2rad(hi))
                    q_try[5] = _unwrap_near(q[5], q_try[5])
                    q_try[5] = np.clip(q_try[5], -J6_SW_LIMIT_RAD, J6_SW_LIMIT_RAD)

                    T_try = self._fk_T_rad(q_try)
                    e_try = _body_error(T_try, T_star, w_pos, w_ori)
                    if np.linalg.norm(e_try) < np.linalg.norm(e):
                        # Accept smaller step; modestly increase damping
                        q = q_try
                        lam = min(lam * 2.0, 1e2)
                        improved = True
                        break

                if not improved:
                    # No improvement even with shrunken steps: increase damping a lot
                    lam = min(lam * 10.0, 1e2)

        # Return best effort even if not converged
        return [float(np.rad2deg(v)) for v in q], False, max_iters

    def _auto_seed_candidates(self, target_xyz_mm, q_last_deg: Optional[List[float]] = None):
        """
        Build a small set of reasonable seeds that cover shoulder/front-back,
        elbow up/down, and wrist flip/no-flip. All angles in degrees.
        """
        x_mm, y_mm, z_mm = [float(v) for v in target_xyz_mm]
        # rough base yaw guess from target XY
        theta1_guess = _wrap_deg(np.degrees(np.arctan2(y_mm, x_mm)))

        # elbow-up/down heuristics (stay well inside limits)
        elbow_up = (_clip_deg(+20.0, MECA_LIMITS_DEG[1]),
                    _clip_deg(-40.0, MECA_LIMITS_DEG[2]))
        elbow_down = (_clip_deg(-20.0, MECA_LIMITS_DEG[1]),
                      _clip_deg(-100.0, MECA_LIMITS_DEG[2]))

        # wrist flip / no-flip via theta5 sign
        wrist_pos = _clip_deg(+10.0, MECA_LIMITS_DEG[4])
        wrist_neg = _clip_deg(-10.0, MECA_LIMITS_DEG[4])

        base_front = theta1_guess
        base_back = _wrap_deg(theta1_guess + 180.0)

        # Start list
        seeds = []

        # 0) continuity: use last joints if provided
        if q_last_deg is not None:
            seeds.append(list(q_last_deg))

        # 1) front, elbow-up, no-flip
        seeds.append([base_front, elbow_up[0], elbow_up[1], 0.0, wrist_pos, 0.0])
        # 2) front, elbow-down, no-flip
        seeds.append([base_front, elbow_down[0], elbow_down[1], 0.0, wrist_pos, 0.0])
        # 3) back, elbow-up, no-flip
        seeds.append([base_back, elbow_up[0], elbow_up[1], 0.0, wrist_pos, 0.0])
        # 4) front, elbow-up, flip
        seeds.append([base_front, elbow_up[0], elbow_up[1], 0.0, wrist_neg, 0.0])

        # clip seeds to mechanical limits (J1..J5); J6 left as-is
        fixed = []
        for s in seeds:
            s = list(s)
            for i in range(5):
                s[i] = _clip_deg(s[i], MECA_LIMITS_DEG[i])
            fixed.append(s)
        return fixed

    def _ik_score(self, T, T_star, w_pos=1.0, w_ori=1.0):
        """Scalar error score (lower is better) for selecting best multi-start result."""
        e = _body_error(T, T_star, w_pos, w_ori)
        pos_err = np.linalg.norm(e[:3]) / max(w_pos, 1e-9)
        ori_err = np.linalg.norm(e[3:]) / max(w_ori, 1e-9)
        return pos_err + ori_err

    def ik_solve_dls_auto(self,
                          target_xyz_mm=None,
                          target_euler_deg=None,
                          target_T: np.ndarray = None,
                          q_last_deg: Optional[List[float]] = None,
                          w_pos: float = 1.0,
                          w_ori: float = 1.0,
                          lam0: float = 1e-2,
                          max_iters_each: int = 80,
                          step_deg_max: float = 4.0,
                          tol_pos_mm: float = 0.2,
                          tol_ori_deg: float = 0.2,
                          *,
                          prefer_continuity: bool = True):
        """
        Multi-start Jacobian DLS IK:
          - if prefer_continuity and q_last_deg: try continuity seed first; if it
            converges, return immediately (prevents posture flips on MoveLin legs)
          - otherwise test posture seeds and return the best-scoring solution
        """
        # Build target transform
        if target_T is None:
            if target_xyz_mm is None or target_euler_deg is None:
                raise ValueError("Provide either (target_xyz_mm, target_euler_deg) or target_T.")
            T_star = self._build_target_T(target_xyz_mm, target_euler_deg)
        else:
            T_star = np.array(target_T, float)

        best = None  # (score, q_deg, ok, iters)

        # 1) continuity-first
        if prefer_continuity and q_last_deg is not None:
            q_deg, ok, iters = self.ik_solve_dls(
                target_xyz_mm=target_xyz_mm,
                target_euler_deg=target_euler_deg,
                target_T=T_star,
                q0_deg=q_last_deg,
                w_pos=w_pos, w_ori=w_ori,
                lam0=lam0,
                max_iters=max_iters_each,
                step_deg_max=step_deg_max,
                tol_pos_mm=tol_pos_mm,
                tol_ori_deg=tol_ori_deg
            )
            if ok:
                return q_deg, True, iters
            T_sol = self.fk_T(q_deg)
            score = self._ik_score(T_sol, T_star, w_pos, w_ori)
            best = (score, q_deg, ok, iters)

        # 2) posture seeds
        seeds = self._auto_seed_candidates(target_xyz_mm, q_last_deg=q_last_deg)
        for seed in seeds:
            if prefer_continuity and q_last_deg is not None and np.allclose(seed, q_last_deg, atol=1e-6):
                continue
            q_deg, ok, iters = self.ik_solve_dls(
                target_xyz_mm=target_xyz_mm,
                target_euler_deg=target_euler_deg,
                target_T=T_star,
                q0_deg=seed,
                w_pos=w_pos, w_ori=w_ori,
                lam0=lam0,
                max_iters=max_iters_each,
                step_deg_max=step_deg_max,
                tol_pos_mm=tol_pos_mm,
                tol_ori_deg=tol_ori_deg
            )
            T_sol = self.fk_T(q_deg)
            score = self._ik_score(T_sol, T_star, w_pos, w_ori)
            if best is None or (ok and not best[2]) or (ok == best[2] and score < best[0]):
                best = (score, q_deg, ok, iters)

        # 3) fallback
        if best is None:
            seed = q_last_deg if q_last_deg is not None else [0, 0, 0, 0, 0, 0]
            q_deg, ok, iters = self.ik_solve_dls(
                target_xyz_mm=target_xyz_mm,
                target_euler_deg=target_euler_deg,
                target_T=T_star,
                q0_deg=seed,
                w_pos=w_pos, w_ori=w_ori,
                lam0=lam0,
                max_iters=max_iters_each,
                step_deg_max=step_deg_max,
                tol_pos_mm=tol_pos_mm,
                tol_ori_deg=tol_ori_deg
            )
            return q_deg, ok, iters

        return best[1], best[2], best[3]

    # -------- recorder helpers (pen metaphor) --------
    def _begin_stroke_if_needed(self, source: str):
        """Open a new stroke if pen is up."""
        if not self.pen_down:
            self.strokes.append({
                "points": [],        # list[(x,y,z) in mm]
                "joints": [],        # list[[j1..j6] in deg]
                "source": source,    # "MoveLin" | "MoveJoints" | "MoveJointsRel"
            })
            self.pen_down = True

    def _append_point_to_stroke(self, pose_mm_deg, joints_deg, source: str):
        """Append one point to the current stroke; start a stroke if needed."""
        self._begin_stroke_if_needed(source)
        x, y, z, a, b, c = pose_mm_deg
        self.strokes[-1]["points"].append((float(x), float(y), float(z)))
        self.strokes[-1]["joints"].append([float(v) for v in joints_deg])

    def _record_teleport(self, start_pose_mm_deg, end_pose_mm_deg, from_cmd: str = "MovePose"):
        """Record a pen-up move (not drawn). Sets pen to up."""
        sx, sy, sz, *_ = start_pose_mm_deg
        ex, ey, ez, *_ = end_pose_mm_deg
        self.teleports.append({
            "start": (float(sx), float(sy), float(sz)),
            "end":   (float(ex), float(ey), float(ez)),
            "from_cmd": from_cmd,
        })
        # pen up after a teleport (must start a new stroke before next draw)
        self.pen_down = False

    # -------- plotting & reset --------
    def clear_path(self):
        """Reset all logs: global path/joints and drawing structures."""
        self.path = []
        self.joints = []
        self.strokes = []
        self.teleports = []
        self.pen_down = False

    def plot3d_path(self, ax=None, *, show_teleports: bool = False,
                    color: str = "C0", linewidth: float = 2.0,
                    teleport_style: dict | None = None,
                    equal_axes: bool = True,
                    title: str | None = "TCP Path (mm)",
                    save_path: str | None = None):
        """
        Plot strokes as 3D polylines (single color). Everything in mm.
        Teleports are hidden by default (dashed lines if enabled).
        Forces autoscale using the data and overlays small point markers so
        even degenerate/overlapping segments are visible.
        """
        import numpy as np
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        if ax is None:
            fig = plt.figure(figsize=(7, 6))
            ax = fig.add_subplot(111, projection="3d")
        else:
            fig = ax.figure

        # collect bounds to drive autoscale
        all_xyz = []

        # draw strokes
        for s in self.strokes:
            pts = np.array(s["points"], float)
            if len(pts) == 0:
                continue
            all_xyz.append(pts)
            if len(pts) == 1:
                ax.scatter(pts[0, 0], pts[0, 1], pts[0, 2], c=color, s=20)
            else:
                # line + markers so degenerate segments are visible
                ax.plot(pts[:, 0], pts[:, 1], pts[:, 2],
                        color=color, linewidth=linewidth, marker="o", markersize=3)

        # optional teleports (dashed)
        if show_teleports and self.teleports:
            style = {"linestyle": "--", "linewidth": 1.0, "alpha": 0.6}
            if teleport_style:
                style.update(teleport_style)
            for t in self.teleports:
                xs = [t["start"][0], t["end"][0]]
                ys = [t["start"][1], t["end"][1]]
                zs = [t["start"][2], t["end"][2]]
                seg = np.array([[xs[0], ys[0], zs[0]], [xs[1], ys[1], zs[1]]], float)
                all_xyz.append(seg)
                ax.plot(xs, ys, zs, color=color, **style)

        # fallback: use self.path if no strokes
        if not all_xyz and self.path:
            pts = np.array([[p[0], p[1], p[2]] for p in self.path], float)
            if len(pts) >= 2:
                all_xyz.append(pts)
                ax.plot(pts[:, 0], pts[:, 1], pts[:, 2],
                        color=color, linewidth=linewidth, marker="o", markersize=3)
            elif len(pts) == 1:
                all_xyz.append(pts)
                ax.scatter(pts[0, 0], pts[0, 1], pts[0, 2], c=color, s=20)

        # force autoscale from the actual data
        if all_xyz:
            pts = np.vstack(all_xyz)
            xmin, ymin, zmin = np.nanmin(pts, axis=0)
            xmax, ymax, zmax = np.nanmax(pts, axis=0)
            # add a small pad so points on the boundary are visible
            pad = 0.05 * max(xmax - xmin, ymax - ymin, zmax - zmin, 1.0)
            ax.set_xlim(xmin - pad, xmax + pad)
            ax.set_ylim(ymin - pad, ymax + pad)
            ax.set_zlim(zmin - pad, zmax + pad)

            if equal_axes:
                # equal aspect after setting limits
                def _set_axes_equal(ax3d):
                    limits = np.array([ax3d.get_xlim3d(),
                                       ax3d.get_ylim3d(),
                                       ax3d.get_zlim3d()], float)
                    spans = limits[:, 1] - limits[:, 0]
                    centers = limits.mean(axis=1)
                    radius = 0.5 * spans.max()
                    ax3d.set_xlim3d([centers[0] - radius, centers[0] + radius])
                    ax3d.set_ylim3d([centers[1] - radius, centers[1] + radius])
                    ax3d.set_zlim3d([centers[2] - radius, centers[2] + radius])

                _set_axes_equal(ax)

        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        ax.set_zlabel("Z (mm)")
        if title:
            ax.set_title(title)

        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150)
        return ax

    def plot3d_path_plotly(self, *, show_teleports: bool = False,
                           color: str = "royalblue", title: str = "TCP Path (mm)"):
        import numpy as np
        import plotly.graph_objects as go

        traces = []

        # strokes (drawn paths)
        for s in getattr(self, "strokes", []):
            pts = np.array(s["points"], float)
            if len(pts) == 0:
                continue
            traces.append(go.Scatter3d(
                x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
                mode="lines",
                line=dict(width=5, color=color),
                name=s.get("source", "stroke")
            ))

        # optional teleports (pen-up moves)
        if show_teleports and getattr(self, "teleports", None):
            for t in self.teleports:
                x = [t["start"][0], t["end"][0]]
                y = [t["start"][1], t["end"][1]]
                z = [t["start"][2], t["end"][2]]
                traces.append(go.Scatter3d(
                    x=x, y=y, z=z,
                    mode="lines",
                    line=dict(width=2, color=color, dash="dash"),
                    name="teleport"
                ))

        fig = go.Figure(
            data=traces,
            layout=go.Layout(
                title=title,
                showlegend=False,
                scene=dict(
                    xaxis_title="X (mm)",
                    yaxis_title="Y (mm)",
                    zaxis_title="Z (mm)",
                    aspectmode="data"  # equal scale
                )
            )
        )
        return fig

    # convenience for callers that expect getpath()
    def getpath(self):
        return list(self.path)
