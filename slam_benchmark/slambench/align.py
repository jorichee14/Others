"""Alignment. Every SLAM system starts at its own identity; the reference does
not. Something has to put them in one frame, and *which* something is a result,
not a detail.

Four modes, in decreasing strength of the claim they let you make:

  none    the estimate is already in the reference's world frame (a method
          initialised from a known pose, or a re-localiser). ATE then means
          absolute error. This is the only mode whose number is comparable
          across methods without caveat.
  se3     rigid alignment. The estimate's origin and heading are free; scale is
          not. Correct for LiDAR and RGB-D, where scale is observed.
  sim3    rigid plus one scale. Necessary for monocular; reporting it for a
          LiDAR method hides a scale error that is a real defect.
  yaw     translation plus yaw only, with roll/pitch and z left alone. For
          comparing against a reference that shares a gravity axis (the coop2
          `map` frame keeps the mapping session's gravity), this is the
          strongest alignment that does not let a tilted estimate buy accuracy
          by rotating out of the horizontal.

`sim3` on a metric sensor is the classic way to make a bad trajectory look
good, so `fit` records the scale it found even in `se3` mode, where it is
applied as 1.0 and reported as evidence.
"""
from __future__ import annotations

import dataclasses

import numpy as np

from . import se3 as _se3
from .trajectory import Trajectory

__all__ = ["Alignment", "fit", "MODES"]

MODES = ("none", "se3", "sim3", "yaw")


@dataclasses.dataclass
class Alignment:
    mode: str
    T: np.ndarray              # 4x4, ref <- est (rotation and translation only)
    scale: float               # applied scale (1.0 unless mode == "sim3")
    # THE SCALE THAT MULTIPLIES THE **ESTIMATE** TO MATCH THE REFERENCE:
    # `ref ~ scale * R @ est + t`. The direction is the opposite of how it
    # reads, and it was reported backwards for a day.
    #   scale_observed > 1  ->  the estimate is SMALLER than the reference
    #   scale_observed < 1  ->  the estimate is LARGER  than the reference
    # Use `size_ratio` / `size_note()` rather than inverting it by hand.
    scale_observed: float      # the scale Umeyama found, whether or not applied

    @property
    def size_ratio(self) -> float:
        """The estimate's size as a fraction of the reference's: 1.05 means the
        estimate is 5% too big."""
        return 1.0 / self.scale_observed if self.scale_observed else float("nan")

    def size_note(self) -> str:
        """One clause a reader cannot invert."""
        r = self.size_ratio
        if abs(r - 1.0) < 0.005:
            return "estimate is within 0.5% of the reference's size"
        return (f"estimate is {abs(r - 1.0) * 100:.1f}% too "
                f"{'large' if r > 1 else 'small'}")

    def apply(self, traj: Trajectory) -> Trajectory:
        poses = traj.poses.copy()
        poses[:, :3, 3] *= self.scale
        return Trajectory(traj.stamps, np.einsum("ij,njk->nik", self.T, poses),
                          traj.name, traj.frame, traj.world)


def fit(est: Trajectory, ref: Trajectory, mode: str = "se3",
        n_first: int | None = None) -> Alignment:
    """Solve for `ref <- est` from associated, equal-length trajectories.

    `n_first` restricts the fit to the first N pose pairs. That turns ATE from
    "how close can these be made to agree" into "how far apart do they drift
    once anchored at the start", which is the question a drift-free claim is
    actually about. Use it, and say which you used.
    """
    if mode not in MODES:
        raise ValueError(f"alignment mode {mode!r} not in {MODES}")
    if len(est) != len(ref):
        raise ValueError(f"align needs paired trajectories: {len(est)} vs {len(ref)}")
    if len(est) < 3:
        raise ValueError("align needs at least 3 pose pairs")

    P = est.positions if n_first is None else est.positions[:n_first]
    Q = ref.positions if n_first is None else ref.positions[:n_first]
    if len(P) < 3:
        raise ValueError(f"n_first={n_first} leaves fewer than 3 pose pairs")

    R_u, t_u, s_u = _umeyama(P, Q)

    if mode == "none":
        return Alignment("none", np.eye(4), 1.0, s_u)
    if mode == "sim3":
        T = np.eye(4); T[:3, :3] = R_u; T[:3, 3] = t_u
        return Alignment("sim3", T, float(s_u), float(s_u))
    if mode == "se3":
        R, t, _ = _umeyama(P, Q, with_scale=False)
        T = np.eye(4); T[:3, :3] = R; T[:3, 3] = t
        return Alignment("se3", T, 1.0, float(s_u))

    # yaw: the horizontal problem is a 2-D Procrustes; z is a shift only.
    Pc, Qc = P - P.mean(0), Q - Q.mean(0)
    H = Pc[:, :2].T @ Qc[:, :2]
    psi = float(np.arctan2(H[0, 1] - H[1, 0], H[0, 0] + H[1, 1]))
    R = _se3.rpy_to_rot(0.0, 0.0, psi, degrees=False)
    t = Q.mean(0) - R @ P.mean(0)
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = t
    return Alignment("yaw", T, 1.0, float(s_u))


def _umeyama(P: np.ndarray, Q: np.ndarray, with_scale: bool = True):
    """Least-squares similarity Q ~ s R P + t (Umeyama 1991).

    The reflection guard is the part that matters: without it a degenerate or
    near-planar trajectory — which an indoor pushcart on a flat floor is — can
    fit a mirrored rotation with lower residual than the true one.
    """
    mp, mq = P.mean(0), Q.mean(0)
    Pc, Qc = P - mp, Q - mq
    C = (Qc.T @ Pc) / len(P)
    U, D, Vt = np.linalg.svd(C)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1.0
    R = U @ S @ Vt
    var_p = float(np.mean(np.sum(Pc ** 2, axis=1)))
    s = float(np.trace(np.diag(D) @ S) / var_p) if (with_scale and var_p > 1e-12) else 1.0
    t = mq - s * (R @ mp)
    return R, t, s
