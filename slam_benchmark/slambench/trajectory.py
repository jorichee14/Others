"""Trajectories: TUM I/O, frame re-expression, and time association.

A `Trajectory` is timestamps (float seconds) plus 4x4 poses, strictly increasing
in time. Everything a SLAM system writes and everything the reference supplies
lands here first, so the comparison code never touches a file format.

The TUM format is `timestamp tx ty tz qx qy qz qw`, one pose per line, seconds
and metres. It is the only format this harness accepts from a method, because
every SLAM system in the shortlist can already emit it and a second format is a
second way to get the quaternion order wrong.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np

from . import se3

__all__ = ["Trajectory", "load_tum", "save_tum", "associate"]


@dataclasses.dataclass
class Trajectory:
    stamps: np.ndarray          # (N,) float64 seconds
    poses: np.ndarray           # (N, 4, 4) float64
    name: str = ""
    frame: str = ""             # what the poses describe, e.g. "os_lidar"
    world: str = ""             # what they are expressed in, e.g. "map"
    lost: int = 0               # rows the source marked as "odometry lost" (a
                                # zeroed pose); skipped at load, quoted beside ATE

    def __post_init__(self):
        self.stamps = np.asarray(self.stamps, dtype=np.float64).reshape(-1)
        self.poses = np.asarray(self.poses, dtype=np.float64).reshape(-1, 4, 4)
        if len(self.stamps) != len(self.poses):
            raise ValueError(f"{len(self.stamps)} stamps for {len(self.poses)} poses")
        if len(self.stamps) and np.any(np.diff(self.stamps) <= 0):
            raise ValueError(f"{self.name or 'trajectory'}: stamps are not strictly increasing")

    def __len__(self) -> int:
        return len(self.stamps)

    @property
    def positions(self) -> np.ndarray:
        return self.poses[:, :3, 3]

    @property
    def duration(self) -> float:
        return float(self.stamps[-1] - self.stamps[0]) if len(self) > 1 else 0.0

    def path_length(self) -> float:
        """Metres travelled. The denominator of every drift-per-metre number."""
        if len(self) < 2:
            return 0.0
        return float(np.sum(np.linalg.norm(np.diff(self.positions, axis=0), axis=1)))

    def arc_length(self) -> np.ndarray:
        """Cumulative path length at each pose, (N,)."""
        if len(self) < 2:
            return np.zeros(len(self))
        step = np.linalg.norm(np.diff(self.positions, axis=0), axis=1)
        return np.concatenate([[0.0], np.cumsum(step)])

    def subset(self, idx: np.ndarray) -> "Trajectory":
        return Trajectory(self.stamps[idx], self.poses[idx], self.name, self.frame, self.world)

    def transform_left(self, T: np.ndarray) -> "Trajectory":
        """world' <- world. Changes what the poses are expressed in."""
        return Trajectory(self.stamps, np.einsum("ij,njk->nik", T, self.poses),
                          self.name, self.frame, self.world)

    def transform_right(self, T: np.ndarray) -> "Trajectory":
        """sensor <- body. Changes which rigid frame the poses describe.

        This is the one that decides whether a comparison is meaningful at all.
        A method that tracks the Ouster reports `map <- os_lidar`; the coop2
        reference reports `map <- zed_left_camera_optical_frame`. They differ by
        a 9.3 cm lever arm and a ~90 deg rotation, which shows up as a rotation-
        coupled position error no alignment can absorb — the trajectories have
        the same shape only where the platform does not rotate.
        """
        return Trajectory(self.stamps, np.einsum("nij,jk->nik", self.poses, T),
                          self.name, self.frame, self.world)

    def interpolate(self, stamps: np.ndarray, max_gap_s: float = 0.2) -> "Trajectory":
        """Resample onto `stamps`. Linear in position, SLERP in rotation.

        Queries outside the trajectory's span, or inside a gap wider than
        `max_gap_s`, are dropped rather than extrapolated — a reference is not
        evidence where it was not recorded.
        """
        stamps = np.asarray(stamps, dtype=np.float64).reshape(-1)
        if len(self) < 2:
            raise ValueError("cannot interpolate a trajectory of fewer than 2 poses")
        keep, out = [], []
        hi = np.searchsorted(self.stamps, stamps, side="left")
        for k, (t, j) in enumerate(zip(stamps, hi)):
            # An exact hit is taken as the pose itself, whatever the neighbouring
            # spacing. Without this, a query landing on the far edge of a dropout
            # is discarded even though the trajectory has a pose at exactly that
            # instant — which silently costs every method a frame per gap.
            if j < len(self) and abs(t - self.stamps[j]) < 1e-9:
                keep.append(k); out.append(self.poses[j]); continue
            if j == 0:
                continue                                   # before the first pose
            if j >= len(self):
                continue                                   # after the last pose
            t0, t1 = self.stamps[j - 1], self.stamps[j]
            if t1 - t0 > max_gap_s:
                continue                                   # a real dropout
            u = (t - t0) / (t1 - t0)
            out.append(_interp_pose(self.poses[j - 1], self.poses[j], u))
            keep.append(k)
        if not out:
            raise ValueError(f"{self.name or 'trajectory'}: no query stamp fell inside it")
        return Trajectory(stamps[np.array(keep)], np.stack(out), self.name, self.frame, self.world)


def _interp_pose(A: np.ndarray, B: np.ndarray, u: float) -> np.ndarray:
    T = np.eye(4)
    T[:3, 3] = (1 - u) * A[:3, 3] + u * B[:3, 3]
    T[:3, :3] = se3.quat_to_rot(_slerp(se3.rot_to_quat(A[:3, :3]),
                                       se3.rot_to_quat(B[:3, :3]), u))
    return T


def _slerp(q0: np.ndarray, q1: np.ndarray, u: float) -> np.ndarray:
    d = float(np.dot(q0, q1))
    if d < 0.0:
        q1, d = -q1, -d
    if d > 0.9995:                                  # near-parallel: lerp is exact enough
        q = q0 + u * (q1 - q0)
        return q / np.linalg.norm(q)
    th0 = np.arccos(np.clip(d, -1.0, 1.0))
    th = th0 * u
    q2 = q1 - q0 * d
    q2 /= np.linalg.norm(q2)
    return q0 * np.cos(th) + q2 * np.sin(th)


def load_tum(path: str | Path, name: str = "", frame: str = "", world: str = "") -> Trajectory:
    # A missing file is a SETUP problem and has to say so. A bare
    # FileNotFoundError traceback out of the middle of a scoring loop is how
    # `runs/reference/mobile_2.tum` went unexported and unnoticed while a whole
    # agent's rows were quietly absent from the tier-2 table.
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"no trajectory at {p}"
            + (f" (wanted as {name})" if name else "")
            + ". A reference is produced by scripts/export_reference.py "
              "--config <dataset> --agent <agent> --out <path>; a run's own "
              "output is written by the container.")
    stamps, poses, lost = [], [], 0
    for lineno, raw in enumerate(p.read_text().splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 8:
            raise ValueError(f"{path}:{lineno}: expected 8 fields (TUM), got {len(parts)}")
        v = [float(p) for p in parts]
        # A zero quaternion is not a corrupt row, it is RTAB-Map's convention
        # for "odometry lost": the message is still published, with the pose
        # zeroed. Ten such rows crashed the evaluator on a run that was 99.4%
        # tracked. They are skipped and COUNTED -- a lost frame is a result
        # (rule 8), and the count travels on the trajectory so it can be quoted
        # beside the ATE rather than vanishing into a skipped line.
        if np.linalg.norm(v[4:8]) < 1e-9:
            lost += 1
            continue
        stamps.append(v[0])
        poses.append(se3.pose_from_quat(v[1:4], v[4:8]))
    if not stamps:
        raise ValueError(f"{path}: no poses" + (f" ({lost} rows, all lost)" if lost else ""))
    stamps = np.array(stamps)
    order = np.argsort(stamps, kind="stable")
    stamps, poses = stamps[order], [poses[i] for i in order]
    # Duplicate stamps are a replay artefact (two messages, one clock tick), not
    # information; keep the first and say how many went.
    uniq = np.concatenate([[True], np.diff(stamps) > 0])
    return Trajectory(stamps[uniq], np.stack(poses)[uniq],
                      name or Path(path).stem, frame, world, lost=lost)


def save_tum(traj: Trajectory, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# timestamp tx ty tz qx qy qz qw",
             f"# frame={traj.frame or '?'} world={traj.world or '?'} name={traj.name or '?'}"]
    for t, T in zip(traj.stamps, traj.poses):
        p, q = se3.pose_to_quat(T)
        lines.append(f"{t:.9f} {p[0]:.6f} {p[1]:.6f} {p[2]:.6f} "
                     f"{q[0]:.9f} {q[1]:.9f} {q[2]:.9f} {q[3]:.9f}")
    path.write_text("\n".join(lines) + "\n")


def associate(est: Trajectory, ref: Trajectory, max_diff_s: float = 0.02,
              ) -> tuple[Trajectory, Trajectory]:
    """Nearest-neighbour association, one reference pose per estimate at most.

    Used when the reference is not to be interpolated — a sparser reference on
    an irregular grid (which the coop2 stage-09 trajectories are) is better
    interpolated, so `interpolate` is the default path and this is here for
    references that are themselves discrete measurements, such as the surveyed
    board dwells.
    """
    if not len(est) or not len(ref):
        raise ValueError("cannot associate an empty trajectory")
    j = np.searchsorted(ref.stamps, est.stamps)
    lo = np.clip(j - 1, 0, len(ref) - 1)
    hi = np.clip(j, 0, len(ref) - 1)
    pick = np.where(np.abs(ref.stamps[lo] - est.stamps) <=
                    np.abs(ref.stamps[hi] - est.stamps), lo, hi)
    ok = np.abs(ref.stamps[pick] - est.stamps) <= max_diff_s
    # One reference pose may be nearest to several estimates; keep the closest.
    seen: dict[int, int] = {}
    for i in np.nonzero(ok)[0]:
        r = int(pick[i])
        if r not in seen or abs(ref.stamps[r] - est.stamps[i]) < abs(ref.stamps[r] - est.stamps[seen[r]]):
            seen[r] = int(i)
    if not seen:
        raise ValueError(f"no pose pair within {max_diff_s * 1e3:.0f} ms "
                         f"(est spans {est.stamps[0]:.3f}..{est.stamps[-1]:.3f}, "
                         f"ref spans {ref.stamps[0]:.3f}..{ref.stamps[-1]:.3f}) — "
                         "a whole-trajectory offset usually means two different clocks")
    ei = np.array(sorted(seen.values()))
    ri = np.array([r for r, _ in sorted(seen.items(), key=lambda kv: kv[1])])
    return est.subset(ei), ref.subset(ri)
