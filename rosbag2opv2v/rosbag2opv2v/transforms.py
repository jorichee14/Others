"""Rigid-transform helpers shared by the converter.

The one non-obvious piece here is the Euler convention.  OpenCOOD stores every
pose as ``[x, y, z, roll, yaw, pitch]`` in **degrees** and rebuilds the matrix
with ``opencood.utils.transformation_utils.x_to_world``.  That function builds

    R_carla(roll, yaw, pitch) = Rz(yaw) @ Ry(-pitch) @ Rx(-roll)

i.e. a standard right-handed Z-Y-X rotation with the pitch and roll angles
negated (a leftover of CARLA's left-handed world).  So a ROS rotation matrix is
encoded for OpenCOOD as

    yaw   =  yaw_zyx
    pitch = -pitch_zyx
    roll  = -roll_zyx

With that encoding ``x_to_world`` reproduces the original right-handed matrix
exactly, which is all OpenCOOD needs: it only ever uses these poses relatively
(``x1_to_x2``), so a self-consistent export behaves identically to real OPV2V.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import numpy as np


def quat_to_matrix(x: float, y: float, z: float, w: float) -> np.ndarray:
    """Quaternion (ROS xyzw order) -> 3x3 rotation matrix."""
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if n < 1e-12:
        return np.eye(3)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def matrix_to_quat(rot: np.ndarray) -> Tuple[float, float, float, float]:
    """3x3 rotation matrix -> quaternion in ROS xyzw order."""
    m = np.asarray(rot, dtype=float)
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    return x, y, z, w


def rpy_deg_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Standard ROS/URDF fixed-axis roll-pitch-yaw (degrees) -> 3x3 matrix."""
    r, p, y = math.radians(roll), math.radians(pitch), math.radians(yaw)
    cr, sr, cp, sp, cy, sy = (math.cos(r), math.sin(r), math.cos(p),
                              math.sin(p), math.cos(y), math.sin(y))
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    return rz @ ry @ rx


def make_matrix(xyz: Sequence[float], rot: np.ndarray) -> np.ndarray:
    out = np.eye(4)
    out[:3, :3] = rot
    out[:3, 3] = np.asarray(xyz, dtype=float)
    return out


def invert(matrix: np.ndarray) -> np.ndarray:
    out = np.eye(4)
    rot = matrix[:3, :3]
    out[:3, :3] = rot.T
    out[:3, 3] = -rot.T @ matrix[:3, 3]
    return out


def matrix_to_opv2v_pose(matrix: np.ndarray) -> List[float]:
    """4x4 right-handed transform -> OPV2V ``[x, y, z, roll, yaw, pitch]`` (deg).

    Inverse of :func:`opv2v_pose_to_matrix` / OpenCOOD's ``x_to_world``.
    """
    m = np.asarray(matrix, dtype=float)
    # standard ZYX decomposition of the right-handed matrix
    sy = -m[2, 0]
    sy = max(-1.0, min(1.0, sy))
    pitch_zyx = math.asin(sy)
    if abs(sy) < 1.0 - 1e-9:
        yaw_zyx = math.atan2(m[1, 0], m[0, 0])
        roll_zyx = math.atan2(m[2, 1], m[2, 2])
    else:  # gimbal lock: fold roll into yaw
        yaw_zyx = math.atan2(-m[0, 1], m[1, 1])
        roll_zyx = 0.0
    return [
        float(m[0, 3]), float(m[1, 3]), float(m[2, 3]),
        float(-math.degrees(roll_zyx)),   # OPV2V roll  = -roll_zyx
        float(math.degrees(yaw_zyx)),     # OPV2V yaw   =  yaw_zyx
        float(-math.degrees(pitch_zyx)),  # OPV2V pitch = -pitch_zyx
    ]


def opv2v_pose_to_matrix(pose: Sequence[float]) -> np.ndarray:
    """Byte-for-byte re-implementation of OpenCOOD's ``x_to_world``."""
    x, y, z, roll, yaw, pitch = pose[:6]
    c_y, s_y = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
    c_r, s_r = math.cos(math.radians(roll)), math.sin(math.radians(roll))
    c_p, s_p = math.cos(math.radians(pitch)), math.sin(math.radians(pitch))

    matrix = np.identity(4)
    matrix[0, 3], matrix[1, 3], matrix[2, 3] = x, y, z
    matrix[0, 0] = c_p * c_y
    matrix[0, 1] = c_y * s_p * s_r - s_y * c_r
    matrix[0, 2] = -c_y * s_p * c_r - s_y * s_r
    matrix[1, 0] = s_y * c_p
    matrix[1, 1] = s_y * s_p * s_r + c_y * c_r
    matrix[1, 2] = -s_y * s_p * c_r + c_y * s_r
    matrix[2, 0] = s_p
    matrix[2, 1] = -c_p * s_r
    matrix[2, 2] = c_p * c_r
    return matrix


def slerp(q0: np.ndarray, q1: np.ndarray, alpha: float) -> np.ndarray:
    """Spherical linear interpolation between two xyzw quaternions."""
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        out = q0 + alpha * (q1 - q0)
        return out / np.linalg.norm(out)
    theta_0 = math.acos(max(-1.0, min(1.0, dot)))
    theta = theta_0 * alpha
    q2 = q1 - q0 * dot
    q2 = q2 / np.linalg.norm(q2)
    return q0 * math.cos(theta) + q2 * math.sin(theta)


class PoseTrack:
    """Time-ordered 6-DoF trajectory with gap-aware interpolation."""

    def __init__(self) -> None:
        self._t: List[float] = []
        self._pos: List[np.ndarray] = []
        self._quat: List[np.ndarray] = []
        self._sorted = True

    def add(self, t: float, pos: Sequence[float], quat: Sequence[float]) -> None:
        if self._t and t < self._t[-1]:
            self._sorted = False
        self._t.append(float(t))
        self._pos.append(np.asarray(pos, dtype=float))
        self._quat.append(np.asarray(quat, dtype=float))

    def finalize(self) -> None:
        if not self._sorted:
            order = np.argsort(np.asarray(self._t))
            self._t = [self._t[i] for i in order]
            self._pos = [self._pos[i] for i in order]
            self._quat = [self._quat[i] for i in order]
            self._sorted = True
        self.times = np.asarray(self._t)

    def __len__(self) -> int:
        return len(self._t)

    @property
    def t_start(self) -> float:
        return self._t[0]

    @property
    def t_end(self) -> float:
        return self._t[-1]

    def sample(self, t: float, max_gap: float) -> Optional[np.ndarray]:
        """Interpolated 4x4 pose at ``t``; ``None`` if no data within ``max_gap``."""
        if not self._t:
            return None
        idx = int(np.searchsorted(self.times, t))
        if idx == 0:
            if self._t[0] - t > max_gap:
                return None
            return make_matrix(self._pos[0], quat_to_matrix(*self._quat[0]))
        if idx >= len(self._t):
            if t - self._t[-1] > max_gap:
                return None
            return make_matrix(self._pos[-1], quat_to_matrix(*self._quat[-1]))
        t0, t1 = self._t[idx - 1], self._t[idx]
        if (t1 - t0) > max_gap:
            # straddling a dropout: only accept if one side is close enough
            if min(t - t0, t1 - t) > max_gap:
                return None
        alpha = 0.0 if t1 <= t0 else (t - t0) / (t1 - t0)
        pos = self._pos[idx - 1] * (1.0 - alpha) + self._pos[idx] * alpha
        quat = slerp(self._quat[idx - 1], self._quat[idx], alpha)
        return make_matrix(pos, quat_to_matrix(*quat))

    def velocity(self, t: float, window: float = 0.2) -> float:
        """Finite-difference speed (m/s) around ``t``."""
        a = self.sample(max(self.t_start, t - window * 0.5), window * 2 + 0.05)
        b = self.sample(min(self.t_end, t + window * 0.5), window * 2 + 0.05)
        if a is None or b is None:
            return 0.0
        dt = min(self.t_end, t + window * 0.5) - max(self.t_start, t - window * 0.5)
        if dt <= 1e-6:
            return 0.0
        return float(np.linalg.norm(b[:3, 3] - a[:3, 3]) / dt)


class StaticTFTree:
    """Small static transform graph built from ``/tf_static`` (and optionally ``/tf``)."""

    def __init__(self) -> None:
        self._edges: dict = {}

    def add(self, parent: str, child: str, matrix: np.ndarray) -> None:
        parent = parent.lstrip('/')
        child = child.lstrip('/')
        self._edges.setdefault(parent, {})[child] = matrix
        self._edges.setdefault(child, {})[parent] = invert(matrix)

    def frames(self) -> List[str]:
        return sorted(self._edges)

    def lookup(self, target: str, source: str) -> Optional[np.ndarray]:
        """T_target_source, i.e. the transform that maps a point in ``source``
        coordinates into ``target`` coordinates."""
        target = target.lstrip('/')
        source = source.lstrip('/')
        if target == source:
            return np.eye(4)
        if target not in self._edges or source not in self._edges:
            return None
        # BFS from target to source, composing along the way
        seen = {target}
        queue = [(target, np.eye(4))]
        while queue:
            node, acc = queue.pop(0)
            for nxt, mat in self._edges[node].items():
                if nxt in seen:
                    continue
                composed = acc @ mat
                if nxt == source:
                    return composed
                seen.add(nxt)
                queue.append((nxt, composed))
        return None
