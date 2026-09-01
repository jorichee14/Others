# -*- coding: utf-8 -*-
"""
Turning asynchronous topics into OPV2V frames.

OPV2V's contract is stricter than it looks: OpenCOOD reads the timestamp list
from the *first* agent folder and then indexes every other agent with the same
key, so a frame that exists for one agent and not another is a ``KeyError`` at
training time, not a gracefully skipped sample.  This module therefore builds a
frame table in which every required agent has a message within tolerance, and
drops the rest.
"""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .geometry import interpolate_transforms

NS = 1_000_000_000


class StampIndex:
    """Sorted message stamps for one topic, with nearest-neighbour lookup."""

    def __init__(self, stamps: Sequence[int]):
        self.stamps: List[int] = sorted(int(s) for s in stamps)

    def __len__(self) -> int:
        return len(self.stamps)

    def nearest(self, t_ns: int, tolerance_ns: Optional[int] = None) -> Optional[int]:
        """The stamp closest to ``t_ns``, or ``None`` if it is beyond tolerance."""
        if not self.stamps:
            return None
        pos = bisect_left(self.stamps, t_ns)
        candidates = []
        if pos < len(self.stamps):
            candidates.append(self.stamps[pos])
        if pos > 0:
            candidates.append(self.stamps[pos - 1])
        best = min(candidates, key=lambda s: abs(s - t_ns))
        if tolerance_ns is not None and abs(best - t_ns) > tolerance_ns:
            return None
        return best

    def rate_hz(self) -> float:
        if len(self.stamps) < 2:
            return 0.0
        span = (self.stamps[-1] - self.stamps[0]) / NS
        return (len(self.stamps) - 1) / span if span > 0 else 0.0


class PoseTrack:
    """A time series of rigid transforms with interpolated lookup.

    Holds the raw ``odom -> child`` transforms from an odometry/pose topic; the
    conversion into the shared world frame happens in :mod:`ros2opv2v.convert`,
    so a track can be re-aligned without re-reading the bag.
    """

    def __init__(self):
        self._stamps: List[int] = []
        self._transforms: List[np.ndarray] = []
        self._speeds: List[float] = []
        self._sorted = True

    def add(self, stamp_ns: int, transform: np.ndarray, speed_mps: float = 0.0) -> None:
        if self._stamps and stamp_ns < self._stamps[-1]:
            self._sorted = False
        self._stamps.append(int(stamp_ns))
        self._transforms.append(np.asarray(transform, dtype=np.float64))
        self._speeds.append(float(speed_mps))

    def finish(self) -> "PoseTrack":
        """Sort by stamp and drop duplicates (bags are not always monotonic)."""
        if not self._sorted:
            order = np.argsort(np.asarray(self._stamps, dtype=np.int64), kind="stable")
            self._stamps = [self._stamps[i] for i in order]
            self._transforms = [self._transforms[i] for i in order]
            self._speeds = [self._speeds[i] for i in order]
            self._sorted = True
        keep_s, keep_t, keep_v = [], [], []
        for stamp, transform, speed in zip(self._stamps, self._transforms, self._speeds):
            if keep_s and stamp == keep_s[-1]:
                keep_t[-1], keep_v[-1] = transform, speed     # last write wins
                continue
            keep_s.append(stamp)
            keep_t.append(transform)
            keep_v.append(speed)
        self._stamps, self._transforms, self._speeds = keep_s, keep_t, keep_v
        return self

    def __len__(self) -> int:
        return len(self._stamps)

    @property
    def stamps(self) -> List[int]:
        return self._stamps

    @property
    def transforms(self) -> List[np.ndarray]:
        return self._transforms

    @property
    def speeds(self) -> List[float]:
        return self._speeds

    def set_speeds(self, speeds: Sequence[float]) -> None:
        if len(speeds) != len(self._stamps):
            raise ValueError("speed series must match the number of samples")
        self._speeds = [float(v) for v in speeds]

    def lookup(self, t_ns: int, mode: str = "linear",
               max_gap_ns: Optional[int] = None) -> Optional[Tuple[np.ndarray, float]]:
        """``(transform, speed)`` at ``t_ns``, or ``None`` when out of range.

        ``linear`` interpolates translation linearly and rotation by slerp between
        the bracketing samples; ``nearest`` snaps.  Either way a sample must exist
        within ``max_gap_ns`` or the frame is reported as unavailable — silently
        extrapolating a pose across a SLAM dropout is how a converted dataset ends
        up with plausible-looking but wrong geometry.
        """
        if not self._stamps:
            return None
        pos = bisect_left(self._stamps, t_ns)

        if pos == 0:
            before = after = 0
        elif pos >= len(self._stamps):
            before = after = len(self._stamps) - 1
        else:
            before, after = pos - 1, pos

        gap = min(abs(self._stamps[before] - t_ns), abs(self._stamps[after] - t_ns))
        if max_gap_ns is not None and gap > max_gap_ns:
            return None

        if mode == "nearest" or before == after:
            idx = before if abs(self._stamps[before] - t_ns) <= \
                abs(self._stamps[after] - t_ns) else after
            return self._transforms[idx].copy(), self._speeds[idx]

        span = self._stamps[after] - self._stamps[before]
        frac = 0.0 if span <= 0 else (t_ns - self._stamps[before]) / span
        transform = interpolate_transforms(self._transforms[before],
                                           self._transforms[after], frac)
        speed = (1.0 - frac) * self._speeds[before] + frac * self._speeds[after]
        return transform, speed


@dataclass
class Frame:
    """One OPV2V timestamp: which message each agent contributes."""
    index: int
    t_ns: int
    cloud_stamps: Dict[str, int] = field(default_factory=dict)     # agent name -> stamp
    camera_stamps: Dict[Tuple[str, int], int] = field(default_factory=dict)
    local_key: str = ""            # index within its scenario folder (set at write time)

    @property
    def key(self) -> str:
        return f"{self.index:06d}"


@dataclass
class FrameTable:
    frames: List[Frame]
    dropped: Dict[str, int]        # reason -> count
    master: str
    tolerance_ns: int

    def __len__(self) -> int:
        return len(self.frames)


def frame_times(master_stamps: Sequence[int], rate_hz: float = 0.0,
                start_offset_s: float = 0.0, duration_s: float = 0.0) -> List[int]:
    """The candidate frame times for the dataset.

    With ``rate_hz == 0`` the master agent's own message stamps are used, which
    keeps its cloud unresampled (no nearest-neighbour error on the ego).  With a
    positive rate a uniform grid is generated instead — use that when you need a
    strict 10 Hz cadence, e.g. because a downstream model converts an integer
    frame offset into milliseconds.
    """
    stamps = sorted(int(s) for s in master_stamps)
    if not stamps:
        return []

    t_start = stamps[0] + int(start_offset_s * NS)
    t_end = stamps[-1] if duration_s <= 0 else min(stamps[-1], t_start + int(duration_s * NS))
    if t_end < t_start:
        return []

    if rate_hz <= 0:
        return [s for s in stamps if t_start <= s <= t_end]

    step = int(round(NS / rate_hz))
    return list(range(t_start, t_end + 1, step))


def build_frame_table(times: Sequence[int],
                      cloud_indices: Dict[str, StampIndex],
                      required: Dict[str, bool],
                      tolerance_ns: int,
                      master: str,
                      drop_incomplete: bool = True,
                      camera_indices: Optional[Dict[Tuple[str, int], StampIndex]] = None,
                      stride: int = 1) -> FrameTable:
    """Match every agent's cloud to each candidate frame time.

    A frame survives only if each *required* agent has a message within
    ``tolerance_ns``; optional agents simply contribute nothing when they are
    late.  Reusing the same message for two consecutive frames is allowed (a slow
    agent legitimately repeats), but it is counted and reported.
    """
    camera_indices = camera_indices or {}
    frames: List[Frame] = []
    dropped: Dict[str, int] = {}

    for t_ns in times[::max(1, stride)]:
        cloud_stamps: Dict[str, int] = {}
        missing: List[str] = []
        for name, index in cloud_indices.items():
            stamp = index.nearest(t_ns, tolerance_ns)
            if stamp is None:
                if required.get(name, True):
                    missing.append(name)
                continue
            cloud_stamps[name] = stamp

        if missing:
            if drop_incomplete:
                key = "missing:" + ",".join(sorted(missing))
                dropped[key] = dropped.get(key, 0) + 1
                continue
            # Without drop_incomplete a frame is kept only if *some* agent has data.
            if not cloud_stamps:
                dropped["empty"] = dropped.get("empty", 0) + 1
                continue

        frame = Frame(index=len(frames), t_ns=int(t_ns), cloud_stamps=cloud_stamps)
        for (agent_name, slot), index in camera_indices.items():
            stamp = index.nearest(t_ns, tolerance_ns)
            if stamp is not None:
                frame.camera_stamps[(agent_name, slot)] = stamp
        frames.append(frame)

    return FrameTable(frames=frames, dropped=dropped, master=master,
                      tolerance_ns=tolerance_ns)


def reuse_statistics(table: FrameTable) -> Dict[str, Dict[str, float]]:
    """Per-agent share of frames that reuse the previous frame's message.

    A high reuse rate means the agent is slower than the frame grid and its data
    is effectively stale — worth knowing before reading anything into a
    cooperative-perception result computed on this dataset.
    """
    stats: Dict[str, Dict[str, float]] = {}
    names = set()
    for frame in table.frames:
        names.update(frame.cloud_stamps)

    for name in sorted(names):
        seen, reused, offsets = None, 0, []
        present = 0
        for frame in table.frames:
            stamp = frame.cloud_stamps.get(name)
            if stamp is None:
                continue
            present += 1
            offsets.append(abs(stamp - frame.t_ns) / 1e6)
            if stamp == seen:
                reused += 1
            seen = stamp
        stats[name] = {
            "frames": present,
            "reuse_rate": reused / present if present else 0.0,
            "mean_offset_ms": float(np.mean(offsets)) if offsets else 0.0,
            "max_offset_ms": float(np.max(offsets)) if offsets else 0.0,
        }
    return stats
