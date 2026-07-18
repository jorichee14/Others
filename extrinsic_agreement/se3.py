"""Minimal SE(3) helpers with an explicit, single convention.

Convention (used everywhere in this package):

    A 4x4 matrix ``T_a_b`` maps a point expressed in frame ``b`` into frame ``a``::

        p_a = T_a_b @ p_b          # p_* are homogeneous [x, y, z, 1]

    A "transform  parent -> child  with (translation t, quaternion q)" as printed
    on a TF / calibration sheet expresses the *child frame's pose in the parent*
    (the ROS TF meaning: header.frame_id = parent, child_frame_id = child). That
    is exactly ``T_parent_child`` in the notation above: its columns are the child
    axes written in parent coordinates and ``t`` is the child origin in the parent.

Quaternions are ``[x, y, z, w]`` (scipy order), matching every quat on the
dataset sheet ("quat xyzw").
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def se3(translation, quat_xyzw) -> np.ndarray:
    """Build ``T_parent_child`` from a translation and an xyzw quaternion.

    ``translation`` is the child origin in the parent frame; ``quat_xyzw`` is the
    child orientation in the parent frame. The result maps child points to parent.
    """
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat(np.asarray(quat_xyzw, float)).as_matrix()
    T[:3, 3] = np.asarray(translation, float)
    return T


def inv(T: np.ndarray) -> np.ndarray:
    """Inverse of an SE(3) matrix (``inv(T_a_b) == T_b_a``)."""
    R = T[:3, :3]
    t = T[:3, 3]
    out = np.eye(4)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


def apply(T: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply ``T`` to an (N,3) or (3,) array of points. Returns the same shape."""
    p = np.asarray(points, float)
    single = p.ndim == 1
    p = np.atleast_2d(p)
    ph = np.hstack([p, np.ones((len(p), 1))])
    out = (T @ ph.T).T[:, :3]
    return out[0] if single else out


def chain(*transforms: np.ndarray) -> np.ndarray:
    """Compose left-to-right: ``chain(T_a_b, T_b_c) == T_a_c``."""
    out = np.eye(4)
    for T in transforms:
        out = out @ T
    return out


def rotation_angle_deg(Ra: np.ndarray, Rb: np.ndarray) -> float:
    """Geodesic angle (deg) between two rotation matrices."""
    rel = Ra.T @ Rb
    return float(np.degrees(Rotation.from_matrix(rel).magnitude()))


def slerp_pose(T0: np.ndarray, T1: np.ndarray, t0: float, t1: float, t: float) -> np.ndarray:
    """Interpolate a pose between two stamped poses (linear t, slerp R).

    Used to evaluate a moving platform's pose at an image timestamp that falls
    between two pose samples. ``t`` is clamped to [t0, t1].
    """
    if t1 == t0:
        return T0.copy()
    a = float(np.clip((t - t0) / (t1 - t0), 0.0, 1.0))
    out = np.eye(4)
    R0 = Rotation.from_matrix(T0[:3, :3])
    R1 = Rotation.from_matrix(T1[:3, :3])
    # scipy Slerp for two keyframes
    from scipy.spatial.transform import Slerp

    slerp = Slerp([0.0, 1.0], Rotation.concatenate([R0, R1]))
    out[:3, :3] = slerp([a])[0].as_matrix()
    out[:3, 3] = (1 - a) * T0[:3, 3] + a * T1[:3, 3]
    return out
