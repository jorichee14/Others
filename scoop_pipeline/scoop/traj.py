# -*- coding: utf-8 -*-
"""
Trajectories: TUM I/O, pose lookup by time, and the rotation conversions.

One copy for every stage. The stage scripts each carried their own
load_traj / write_traj / nearest_idx / interp_poses; they differed in small
ways (quaternion sign, sorting, clamping) that nobody chose on purpose.

A :class:`Traj` is float64 seconds plus (N,4,4) poses, strictly increasing in
time. Everything is vectorised over query times, because deskew asks for a
hundred poses per scan and the board stage asks for one per image.

The file format matches ``slam_benchmark/slambench/trajectory.py`` exactly
(``t tx ty tz qx qy qz qw``, ``#`` comments, zero quaternion = lost row,
sorted, duplicate stamps dropped), so a trajectory written here is scored
there without conversion.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

__all__ = ["Traj", "load_tum", "save_tum", "rot_to_quat", "quat_to_rot",
           "inv_se3", "slerp"]


# --------------------------------------------------------------------------- #
# rotations
# --------------------------------------------------------------------------- #
def quat_to_rot(q: np.ndarray) -> np.ndarray:
    """(K,4) or (4,) xyzw -> (K,3,3) or (3,3). Normalised first."""
    q = np.asarray(q, np.float64)
    one = q.ndim == 1
    q = q.reshape(-1, 4)
    n = np.linalg.norm(q, axis=1, keepdims=True)
    if np.any(n < 1e-12):
        raise ValueError("zero-norm quaternion")
    x, y, z, w = (q / n).T
    R = np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
                  2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
                  2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
                 1).reshape(-1, 3, 3)
    return R[0] if one else R


def rot_to_quat(R: np.ndarray) -> np.ndarray:
    """(K,3,3) or (3,3) -> (K,4) or (4,) xyzw, unit norm, w >= 0.

    Shepperd's method, branch chosen per rotation by the largest diagonal term
    so no branch divides by a small number. w >= 0 makes the sign deterministic
    (slambench does the same), so two files of one trajectory compare equal."""
    R = np.asarray(R, np.float64)
    one = R.ndim == 2
    R = R.reshape(-1, 3, 3)
    m00, m11, m22 = R[:, 0, 0], R[:, 1, 1], R[:, 2, 2]
    tr = m00 + m11 + m22
    q = np.empty((len(R), 4))
    b0 = tr > 0
    b1 = ~b0 & (m00 >= m11) & (m00 >= m22)
    b2 = ~b0 & ~b1 & (m11 >= m22)
    b3 = ~(b0 | b1 | b2)
    if b0.any():
        s = np.sqrt(tr[b0] + 1.0) * 2
        q[b0] = np.stack([(R[b0, 2, 1] - R[b0, 1, 2]) / s,
                          (R[b0, 0, 2] - R[b0, 2, 0]) / s,
                          (R[b0, 1, 0] - R[b0, 0, 1]) / s, 0.25 * s], 1)
    for b, (i, j, k) in ((b1, (0, 1, 2)), (b2, (1, 2, 0)), (b3, (2, 0, 1))):
        if not b.any():
            continue
        s = np.sqrt(np.maximum(1e-12, 1.0 + R[b, i, i] - R[b, j, j] - R[b, k, k])) * 2
        qq = np.empty((int(b.sum()), 4))
        qq[:, i] = 0.25 * s
        qq[:, j] = (R[b, j, i] + R[b, i, j]) / s
        qq[:, k] = (R[b, k, i] + R[b, i, k]) / s
        qq[:, 3] = (R[b, k, j] - R[b, j, k]) / s
        q[b] = qq
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    q[q[:, 3] < 0] *= -1.0
    return q[0] if one else q


def slerp(q0: np.ndarray, q1: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Row-wise slerp of (K,4) xyzw at fractions (K,). Shortest arc; lerp where
    the two are within ~1.8 deg, where it is exact to float precision."""
    q0 = np.asarray(q0, np.float64)
    q1 = np.array(q1, np.float64)
    u = np.asarray(u, np.float64)[:, None]
    d = np.sum(q0 * q1, 1)
    q1[d < 0] *= -1.0
    d = np.abs(d)
    q = q0 + u * (q1 - q0)
    far = d <= 0.9995
    if far.any():
        th = np.arccos(np.clip(d[far], -1.0, 1.0))[:, None]
        uf = u[far]
        q[far] = (np.sin((1 - uf) * th) * q0[far] + np.sin(uf * th) * q1[far]) / np.sin(th)
    return q / np.linalg.norm(q, axis=1, keepdims=True)


def inv_se3(T: np.ndarray) -> np.ndarray:
    """Analytic inverse of (4,4) or (K,4,4) rigid transforms."""
    T = np.asarray(T, np.float64)
    Ti = np.zeros_like(T)
    Rt = np.swapaxes(T[..., :3, :3], -1, -2)
    Ti[..., :3, :3] = Rt
    Ti[..., :3, 3] = -np.einsum("...ij,...j->...i", Rt, T[..., :3, 3])
    Ti[..., 3, 3] = 1.0
    return Ti


# --------------------------------------------------------------------------- #
# trajectory
# --------------------------------------------------------------------------- #
@dataclass
class Traj:
    t: np.ndarray                    # (N,) float64 s, strictly increasing
    T: np.ndarray                    # (N,4,4) float64
    lost: int = 0                    # zero-quaternion rows skipped at load
    _q: Optional[np.ndarray] = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        self.t = np.asarray(self.t, np.float64).reshape(-1)
        self.T = np.asarray(self.T, np.float64).reshape(-1, 4, 4)
        if len(self.t) != len(self.T):
            raise ValueError(f"{len(self.t)} stamps for {len(self.T)} poses")
        if len(self.t) == 0:
            raise ValueError("empty trajectory")
        if np.any(np.diff(self.t) <= 0):
            raise ValueError("stamps are not strictly increasing")

    def __len__(self):
        return len(self.t)

    @property
    def q(self) -> np.ndarray:
        """(N,4) xyzw, computed once."""
        if self._q is None:
            self._q = rot_to_quat(self.T[:, :3, :3])
        return self._q

    @property
    def span(self) -> Tuple[float, float]:
        return float(self.t[0]), float(self.t[-1])

    def nearest(self, times, tol: float = np.inf) -> np.ndarray:
        """Index of the nearest pose per query time, -1 beyond ``tol``.

        Same rule as the scripts' nearest_idx and ``bag.nearest_pose``: an
        exact tie goes to the earlier pose."""
        s = np.atleast_1d(np.asarray(times, np.float64))
        n = len(self.t)
        i = np.searchsorted(self.t, s)
        lo = np.clip(i - 1, 0, n - 1)
        hi = np.clip(i, 0, n - 1)
        j = np.where((self.t[hi] - s) < (s - self.t[lo]), hi, lo)
        j = np.where(i <= 0, 0, np.where(i >= n, n - 1, j))
        return np.where(np.abs(self.t[j] - s) <= tol, j, -1)

    def interp(self, times) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Poses at arbitrary times: (R (K,3,3), p (K,3), gap (K,)).

        Slerp on rotation, linear on translation, CLAMPED to the end poses
        outside the span (what deskew wants at the first and last sweep).
        ``gap`` is the distance in seconds to the nearest trajectory sample,
        so a caller that must not extrapolate or bridge a dropout tests it."""
        s = np.atleast_1d(np.asarray(times, np.float64))
        if len(self.t) == 1:
            K = len(s)
            return (np.repeat(self.T[:1, :3, :3], K, 0),
                    np.repeat(self.T[:1, :3, 3], K, 0), np.abs(s - self.t[0]))
        j = np.clip(np.searchsorted(self.t, s), 1, len(self.t) - 1)
        t0, t1 = self.t[j - 1], self.t[j]
        u = np.clip((s - t0) / (t1 - t0), 0.0, 1.0)
        R = quat_to_rot(slerp(self.q[j - 1], self.q[j], u))
        p0, p1 = self.T[j - 1, :3, 3], self.T[j, :3, 3]
        p = p0 + u[:, None] * (p1 - p0)
        gap = np.minimum(np.abs(s - t0), np.abs(s - t1))
        return R, p, gap

    def poses_at(self, times) -> Tuple[np.ndarray, np.ndarray]:
        """(T (K,4,4), gap (K,)) -- :meth:`interp` as homogeneous matrices."""
        R, p, gap = self.interp(times)
        T = np.zeros((len(p), 4, 4))
        T[:, :3, :3] = R
        T[:, :3, 3] = p
        T[:, 3, 3] = 1.0
        return T, gap

    def with_poses(self, T: np.ndarray) -> "Traj":
        """Same stamps, new poses (a refinement round's output)."""
        return Traj(self.t.copy(), T)


def load_tum(path) -> Traj:
    """TUM file -> :class:`Traj`. Sorted, duplicate stamps dropped (first
    kept), zero-quaternion rows skipped and counted in ``Traj.lost``."""
    p = Path(path).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"no trajectory at {p}")
    rows = []
    for lineno, raw in enumerate(p.read_text().splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 8:
            raise ValueError(f"{p}:{lineno}: expected 8 fields (TUM), got {len(parts)}")
        rows.append([float(v) for v in parts])
    if not rows:
        raise ValueError(f"{p}: no poses")
    d = np.array(rows)
    lost_rows = np.linalg.norm(d[:, 4:8], axis=1) < 1e-9
    d = d[~lost_rows]
    if len(d) == 0:
        raise ValueError(f"{p}: all {int(lost_rows.sum())} rows are lost")
    d = d[np.argsort(d[:, 0], kind="stable")]
    d = d[np.concatenate([[True], np.diff(d[:, 0]) > 0])]
    T = np.zeros((len(d), 4, 4))
    T[:, :3, :3] = quat_to_rot(d[:, 4:8])
    T[:, :3, 3] = d[:, 1:4]
    T[:, 3, 3] = 1.0
    return Traj(d[:, 0], T, lost=int(lost_rows.sum()))


def save_tum(path, traj: Traj, comment: str = "") -> Path:
    """:class:`Traj` -> TUM, stamps to the nanosecond as the scripts wrote them."""
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    q = rot_to_quat(traj.T[:, :3, :3])
    out = np.column_stack([traj.t, traj.T[:, :3, 3], q])
    header = "timestamp tx ty tz qx qy qz qw" + (f"\n{comment}" if comment else "")
    np.savetxt(p, out, fmt="%.9f %.6f %.6f %.6f %.9f %.9f %.9f %.9f",
               header=header, comments="# ")
    return p
