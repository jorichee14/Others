# -*- coding: utf-8 -*-
"""
Pose algebra shared by the ROS 2 -> OPV2V converter.

The only subtle piece in this file is the pose parameterisation.  OpenCOOD stores
a pose as ``[x, y, z, roll, yaw, pitch]`` (degrees) and turns it into a 4x4 matrix
with ``opencood.utils.transformation_utils.x_to_world``, whose rotation block is

    R = Rz(yaw) @ Ry(-pitch) @ Rx(-roll)

i.e. CARLA's left-handed Euler convention.  We do **not** convert ROS poses into
CARLA's handedness.  Instead we keep everything in the right-handed ROS world and
solve for the ``(roll, yaw, pitch)`` triple that makes ``x_to_world`` reproduce our
rotation matrix *exactly*.  That is always possible (the parameterisation covers
all of SO(3)), and it is sufficient because OpenCOOD only ever consumes these
numbers through ``x_to_world`` / ``x1_to_x2`` — every transform it derives is then
correct, and every relative geometry (agent-to-ego, box-to-lidar) is preserved.

Consequence to keep in mind: the angles written to the yaml are *not* CARLA angles
and must not be compared numerically against OPV2V's own yaml values.  Only the
matrices they generate are meaningful.
"""

from __future__ import annotations

import numpy as np

EPS = 1e-9


def x_to_world(pose) -> np.ndarray:
    """Bit-exact replica of ``opencood.utils.transformation_utils.x_to_world``.

    Kept local so the converter and its self-tests never need OpenCOOD installed.
    """
    x, y, z, roll, yaw, pitch = pose[:6]

    c_y, s_y = np.cos(np.radians(yaw)), np.sin(np.radians(yaw))
    c_r, s_r = np.cos(np.radians(roll)), np.sin(np.radians(roll))
    c_p, s_p = np.cos(np.radians(pitch)), np.sin(np.radians(pitch))

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


def matrix_to_opencood_pose(matrix: np.ndarray) -> list:
    """Inverse of :func:`x_to_world`: 4x4 -> ``[x, y, z, roll, yaw, pitch]`` (deg).

    ``x_to_world(matrix_to_opencood_pose(T))`` reproduces ``T`` to float precision
    for any rigid ``T`` (verified over random rotations in the self-tests).
    """
    m = np.asarray(matrix, dtype=np.float64)
    s_p = float(np.clip(m[2, 0], -1.0, 1.0))
    pitch = np.arcsin(s_p)
    c_p = np.cos(pitch)

    if abs(c_p) < 1e-7:
        # Gimbal lock: pitch = +/-90 deg. Fold the residual rotation into yaw.
        roll = 0.0
        yaw = np.arctan2(-m[0, 1], m[1, 1])
    else:
        roll = np.arctan2(-m[2, 1], m[2, 2])
        yaw = np.arctan2(m[1, 0], m[0, 0])

    # ``+ 0.0`` normalises -0.0 away so the yaml never carries a signed zero.
    return [float(m[0, 3]) + 0.0, float(m[1, 3]) + 0.0, float(m[2, 3]) + 0.0,
            float(np.degrees(roll)) + 0.0, float(np.degrees(yaw)) + 0.0,
            float(np.degrees(pitch)) + 0.0]


def rpy_to_matrix(roll: float, pitch: float, yaw: float, degrees: bool = True) -> np.ndarray:
    """Right-handed ROS rotation ``Rz(yaw) @ Ry(pitch) @ Rx(roll)`` as a 4x4.

    This is the convention used for every extrinsic in the converter config, so a
    user typing ``yaw: 90`` gets the usual ROS meaning (counter-clockwise seen
    from +z), not CARLA's.
    """
    if degrees:
        roll, pitch, yaw = np.radians([roll, pitch, yaw])
    c_r, s_r = np.cos(roll), np.sin(roll)
    c_p, s_p = np.cos(pitch), np.sin(pitch)
    c_y, s_y = np.cos(yaw), np.sin(yaw)

    rot = np.array([
        [c_y * c_p, c_y * s_p * s_r - s_y * c_r, c_y * s_p * c_r + s_y * s_r],
        [s_y * c_p, s_y * s_p * s_r + c_y * c_r, s_y * s_p * c_r - c_y * s_r],
        [-s_p, c_p * s_r, c_p * c_r],
    ])
    out = np.identity(4)
    out[:3, :3] = rot
    return out


def make_transform(x=0.0, y=0.0, z=0.0, roll=0.0, pitch=0.0, yaw=0.0,
                   degrees: bool = True) -> np.ndarray:
    """Rigid transform from translation + ROS RPY."""
    out = rpy_to_matrix(roll, pitch, yaw, degrees=degrees)
    out[:3, 3] = [x, y, z]
    return out


def quat_to_matrix(x: float, y: float, z: float, w: float) -> np.ndarray:
    """Quaternion (ROS order x, y, z, w) -> 4x4 rigid transform."""
    n = np.sqrt(x * x + y * y + z * z + w * w)
    if n < EPS:
        return np.identity(4)
    x, y, z, w = x / n, y / n, z / n, w / n
    out = np.identity(4)
    out[:3, :3] = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
    return out


def matrix_to_quat(matrix: np.ndarray):
    """4x4 -> quaternion (x, y, z, w). Shepperd's method, numerically stable."""
    m = np.asarray(matrix, dtype=np.float64)[:3, :3]
    trace = np.trace(m)
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    return np.array([x, y, z, w])


def slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """Spherical linear interpolation between two quaternions."""
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:            # take the short way round
        q1, dot = -q1, -dot
    if dot > 0.9995:         # nearly parallel: linear is numerically safer
        out = q0 + t * (q1 - q0)
        return out / np.linalg.norm(out)
    theta = np.arccos(np.clip(dot, -1.0, 1.0))
    s0 = np.sin((1.0 - t) * theta) / np.sin(theta)
    s1 = np.sin(t * theta) / np.sin(theta)
    return s0 * q0 + s1 * q1


def interpolate_transforms(t_a: np.ndarray, t_b: np.ndarray, frac: float) -> np.ndarray:
    """Interpolate two rigid transforms: linear on translation, slerp on rotation."""
    frac = float(np.clip(frac, 0.0, 1.0))
    out = np.identity(4)
    out[:3, 3] = (1.0 - frac) * t_a[:3, 3] + frac * t_b[:3, 3]
    q = slerp(matrix_to_quat(t_a), matrix_to_quat(t_b), frac)
    out[:3, :3] = quat_to_matrix(*q)[:3, :3]
    return out


def invert(matrix: np.ndarray) -> np.ndarray:
    """Inverse of a rigid transform (transpose + rotated translation)."""
    m = np.asarray(matrix, dtype=np.float64)
    out = np.identity(4)
    out[:3, :3] = m[:3, :3].T
    out[:3, 3] = -m[:3, :3].T @ m[:3, 3]
    return out


def transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Apply a 4x4 rigid transform to an (N, 3) point array."""
    if points.size == 0:
        return points
    return points @ np.asarray(matrix[:3, :3], dtype=points.dtype).T + \
        np.asarray(matrix[:3, 3], dtype=points.dtype)


# Optical frame (x right, y down, z forward) -> body/FLU frame (x fwd, y left, z up).
OPTICAL_TO_FLU = np.array([
    [0.0, 0.0, 1.0, 0.0],
    [-1.0, 0.0, 0.0, 0.0],
    [0.0, -1.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
])
