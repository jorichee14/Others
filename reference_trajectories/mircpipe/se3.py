"""SE(3) maths, trajectory interpolation and TUM I/O.

Conventions used everywhere in the pipeline:
  * a pose is a 4x4 numpy matrix T_a_b = pose of frame b in frame a, and it
    maps b-points into a: p_a = T_a_b @ p_b
  * a trajectory is (ts (N,), Ts (N,4,4)) with ts strictly increasing
  * quaternions are [x, y, z, w] (scipy order), as in TUM files
"""
import math
import numpy as np
from scipy.spatial.transform import Rotation as Rot


# --------------------------------------------------------------- primitives
def Rt(R, t):
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = t; return T


def inv(T):
    R = T[:3, :3]; o = np.eye(4); o[:3, :3] = R.T; o[:3, 3] = -R.T @ T[:3, 3]
    return o


def hat(v):
    x, y, z = v
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])


def log_R(R):
    return Rot.from_matrix(R).as_rotvec()


def exp_r(w):
    return Rot.from_rotvec(w).as_matrix()


def jr_inv(w):
    """Exact inverse right Jacobian of SO(3). The small-angle form stalls
    Gauss-Newton when the pose-graph rotation residuals are large at
    iteration 0, so the closed form is used instead."""
    th = float(np.linalg.norm(w)); W = hat(w)
    if th < 1e-6:
        return np.eye(3) + 0.5 * W
    a = 1.0 / th ** 2 - (1.0 + math.cos(th)) / (2.0 * th * math.sin(th))
    return np.eye(3) + 0.5 * W + a * (W @ W)


def q_of_R(R):
    return Rot.from_matrix(R).as_quat()


def make_T_xyzq(v):
    """[x, y, z, qx, qy, qz, qw] -> 4x4."""
    return Rt(Rot.from_quat(v[3:7]).as_matrix(), np.asarray(v[0:3], float))


def apply(T, P):
    """Transform a point set: (N,3) -> (N,3)."""
    return np.asarray(P) @ T[:3, :3].T + T[:3, 3]


def compose_all(Ts, T_right):
    """(N,4,4) @ (4,4) for every pose - e.g. lidar poses -> camera poses."""
    return np.asarray(Ts) @ np.tile(T_right, (len(Ts), 1, 1))


def se3_scale(D, s):
    """Fraction s of the rigid motion D (constant-velocity extrapolation)."""
    return Rt(exp_r(s * log_R(D[:3, :3])), s * D[:3, 3])


def yaw_of(R):
    return math.atan2(R[1, 0], R[0, 0])


def level_parts(T):
    """Split a pose into (x, y, yaw) and the roll/pitch rotation R_rp such
    that R = R_z(yaw) @ R_rp. Used by 2D localisers."""
    yaw = yaw_of(T[:3, :3])
    Rz = exp_r([0, 0, yaw])
    return np.array([T[0, 3], T[1, 3], yaw]), Rz.T @ T[:3, :3]


# -------------------------------------------------------------- trajectories
def interp_traj(ts, Ts, tq):
    """Pose at each query stamp: linear in translation, SLERP in rotation.
    Queries are clamped to the trajectory's own span."""
    from scipy.spatial.transform import Slerp
    ts = np.asarray(ts, float); Ts = np.asarray(Ts, float)
    tq = np.clip(np.asarray(tq, float), ts[0], ts[-1])
    i = np.clip(np.searchsorted(ts, tq) - 1, 0, len(ts) - 2)
    d = ts[i + 1] - ts[i]
    a = np.where(d > 0, (tq - ts[i]) / np.where(d > 0, d, 1), 0.0)
    sl = Slerp(ts, Rot.from_matrix(Ts[:, :3, :3]))
    out = np.tile(np.eye(4), (len(tq), 1, 1))
    out[:, :3, :3] = sl(tq).as_matrix()
    out[:, :3, 3] = Ts[i, :3, 3] * (1 - a)[:, None] + Ts[i + 1, :3, 3] * a[:, None]
    return out


def traj_gap(Ta, Tb):
    """Per-sample translation (m) and rotation (rad) gap of two pose arrays
    given at the SAME stamps and in the SAME body frame."""
    Ta = np.asarray(Ta); Tb = np.asarray(Tb)
    dt = np.linalg.norm(Ta[:, :3, 3] - Tb[:, :3, 3], axis=1)
    dr = np.array([np.linalg.norm(log_R(A[:3, :3].T @ B[:3, :3]))
                   for A, B in zip(Ta, Tb)])
    return dt, dr


def path_length(Ts):
    return float(np.sum(np.linalg.norm(np.diff(np.asarray(Ts)[:, :3, 3], axis=0),
                                       axis=1)))


def decimate_idx(ts, rate_hz):
    """Indices of the first sample at or after every 1/rate mark."""
    ts = np.asarray(ts, float)
    if not rate_hz or rate_hz <= 0:
        return np.arange(len(ts))
    marks = np.arange(ts[0], ts[-1] + 1e-9, 1.0 / rate_hz)
    return np.unique(np.clip(np.searchsorted(ts, marks), 0, len(ts) - 1))


def subsample(P, n):
    """Evenly spaced n points (or all of them if fewer)."""
    P = np.asarray(P)
    if len(P) > n:
        P = P[np.linspace(0, len(P) - 1, n).astype(int)]
    return P


def report_gap(label, ts, dt, dr, path_len=None, printer=print):
    printer("  %s: translation median %.1f cm  p95 %.1f cm  max %.1f cm | "
            "rotation median %.2f deg  max %.2f deg  (%d stamps)"
            % (label, np.median(dt) * 100, np.percentile(dt, 95) * 100,
               dt.max() * 100, math.degrees(np.median(dr)),
               math.degrees(dr.max()), len(dt)))
    qs = np.linspace(0, len(ts) - 1, 6).astype(int)
    printer("     over time: " + "  ".join(
        "t=%.0fs %.0fcm" % (ts[i] - ts[0], dt[i] * 100) for i in qs))
    if path_len is not None and path_len > 1.0:
        printer("     end gap %.1f cm over %.1f m of path = %.2f%% of distance "
                "travelled" % (dt[-1] * 100, path_len, 100 * dt[-1] / path_len))


# ---------------------------------------------------------------- TUM files
def read_tum(path):
    """TUM (t x y z qx qy qz qw) -> (ts, Ts), sorted by time."""
    A = np.loadtxt(path)
    if A.ndim == 1:
        A = A[None]
    A = A[np.argsort(A[:, 0])]
    Ts = np.array([Rt(Rot.from_quat(r[4:8]).as_matrix(), r[1:4]) for r in A])
    return A[:, 0], Ts


def write_tum(path, ts, Ts, printer=print):
    import os
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "w") as f:
        for t, T in zip(ts, Ts):
            q = q_of_R(T[:3, :3])
            f.write("%.9f %.6f %.6f %.6f %.9f %.9f %.9f %.9f\n"
                    % (t, T[0, 3], T[1, 3], T[2, 3], q[0], q[1], q[2], q[3]))
    if printer:
        printer("  wrote %s (%d poses)" % (path, len(ts)))
