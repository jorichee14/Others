"""SE(3) helpers. Rotations are 3x3, poses are 4x4, both float64.

Quaternions are (x, y, z, w) — the ROS/TUM order, not the (w, x, y, z) order
scipy uses internally. Everything that reads a file gets converted here once, so
no other module has to remember which convention it is holding.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "quat_to_rot", "rot_to_quat", "pose_from_quat", "pose_to_quat",
    "invert", "rpy_to_rot", "rot_to_rpy", "pose_from_rpy", "so3_log",
    "rotation_angle",
]


def quat_to_rot(q: np.ndarray) -> np.ndarray:
    """(x, y, z, w) -> 3x3. The quaternion is normalised first."""
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q)
    if n < 1e-12:
        raise ValueError("zero-norm quaternion")
    x, y, z, w = q / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def rot_to_quat(R: np.ndarray) -> np.ndarray:
    """3x3 -> (x, y, z, w), w >= 0 so the sign is deterministic."""
    R = np.asarray(R, dtype=np.float64)
    t = np.trace(R)
    if t > 0.0:
        s = np.sqrt(t + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([x, y, z, w], dtype=np.float64)
    if q[3] < 0:
        q = -q
    return q / np.linalg.norm(q)


def pose_from_quat(t: np.ndarray, q: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = quat_to_rot(q)
    T[:3, 3] = np.asarray(t, dtype=np.float64)
    return T


def pose_to_quat(T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.array(T[:3, 3], dtype=np.float64), rot_to_quat(T[:3, :3])


def invert(T: np.ndarray) -> np.ndarray:
    """Rigid inverse. Not np.linalg.inv: exact, and it cannot introduce shear."""
    R = T[:3, :3]
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ T[:3, 3]
    return out


def rpy_to_rot(roll: float, pitch: float, yaw: float, degrees: bool = True) -> np.ndarray:
    """Z-Y-X intrinsic (yaw, then pitch, then roll) — the convention the
    rosbag_to_opv2v configs use for every `extrinsic` block."""
    if degrees:
        roll, pitch, yaw = np.radians([roll, pitch, yaw])
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
    return Rz @ Ry @ Rx


def rot_to_rpy(R: np.ndarray, degrees: bool = True) -> np.ndarray:
    pitch = np.arcsin(np.clip(-R[2, 0], -1.0, 1.0))
    if abs(R[2, 0]) < 1.0 - 1e-9:
        roll = np.arctan2(R[2, 1], R[2, 2])
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:                                    # gimbal lock: fold roll into yaw
        roll = 0.0
        yaw = np.arctan2(-R[0, 1], R[1, 1])
    out = np.array([roll, pitch, yaw], dtype=np.float64)
    return np.degrees(out) if degrees else out


def pose_from_rpy(x: float, y: float, z: float, roll: float, pitch: float,
                  yaw: float, degrees: bool = True) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = rpy_to_rot(roll, pitch, yaw, degrees)
    T[:3, 3] = [x, y, z]
    return T


def so3_log(R: np.ndarray) -> np.ndarray:
    """Rotation vector, radians."""
    ang = rotation_angle(R)
    if ang < 1e-9:
        return np.zeros(3)
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return w * (ang / (2.0 * np.sin(ang)))


def rotation_angle(R: np.ndarray) -> float:
    """Geodesic angle of a rotation, radians, in [0, pi].

    Via the quaternion rather than arccos((tr - 1) / 2). The trace form loses
    half its digits near identity — arccos is vertical there, so a rotation
    error of 1e-16 reads as 1e-8 rad — and near-identity is exactly where a
    rotation ATE is evaluated.
    """
    q = rot_to_quat(R[:3, :3])
    return float(2.0 * np.arctan2(np.linalg.norm(q[:3]), abs(q[3])))
