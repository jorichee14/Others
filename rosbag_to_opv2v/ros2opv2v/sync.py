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
    """Sorted message stamps for one topic, with nearest-neighbour lookup.

    Two stamps per message, not one, once cross-host clock correction is in play
    (``ros2opv2v/clock.py``). ``stamps`` are on the *reference* clock and are what
    frame matching compares; ``raw`` are the bag's own values and are what the
    write pass must ask the reader for. Conflating them yields either a dataset
    matched on uncorrected times or a write pass that selects nothing at all, and
    both fail quietly, so they are kept side by side rather than derived.
    """

    def __init__(self, stamps: Sequence[int], raw: Optional[Sequence[int]] = None):
        pairs = sorted(zip((int(s) for s in stamps),
                           (int(r) for r in (raw if raw is not None else stamps))))
        self.stamps: List[int] = [c for c, _ in pairs]
        self.raw: List[int] = [r for _, r in pairs]

    def __len__(self) -> int:
        return len(self.stamps)

    def _nearest_index(self, t_ns: int, tolerance_ns: Optional[int]) -> Optional[int]:
        if not self.stamps:
            return None
        pos = bisect_left(self.stamps, t_ns)
        candidates = []
        if pos < len(self.stamps):
            candidates.append(pos)
        if pos > 0:
            candidates.append(pos - 1)
        best = min(candidates, key=lambda i: abs(self.stamps[i] - t_ns))
        if tolerance_ns is not None and abs(self.stamps[best] - t_ns) > tolerance_ns:
            return None
        return best

    def nearest(self, t_ns: int, tolerance_ns: Optional[int] = None) -> Optional[int]:
        """The corrected stamp closest to ``t_ns``, or ``None`` beyond tolerance."""
        index = self._nearest_index(t_ns, tolerance_ns)
        return None if index is None else self.stamps[index]

    def nearest_pair(self, t_ns: int,
                     tolerance_ns: Optional[int] = None) -> Optional[Tuple[int, int]]:
        """``(corrected_stamp, raw_stamp)`` for the nearest message."""
        index = self._nearest_index(t_ns, tolerance_ns)
        return None if index is None else (self.stamps[index], self.raw[index])

    def rate_hz(self) -> float:
        if len(self.stamps) < 2:
            return 0.0
        span = (self.stamps[-1] - self.stamps[0]) / NS
        return (len(self.stamps) - 1) / span if span > 0 else 0.0

    def half_period_ns(self) -> float:
        """The best |skew| nearest-neighbour matching can ever achieve on this
        stream. Half a publication period is a property of the *recording*: no
        matching strategy improves on it, so a tolerance below it rejects frames
        for a reason no amount of processing can fix."""
        rate = self.rate_hz()
        return 0.5 * NS / rate if rate > 0 else float("inf")


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
    """One OPV2V timestamp: which message each agent contributes.

    ``cloud_stamps`` hold the bag's own (raw) stamps, because the write pass
    selects messages by them. ``skew_ns`` holds each contribution's signed
    distance from ``t_ns`` **on the corrected timeline** — the quantity that says
    how synchronous this frame actually is. It is carried into the frame yaml
    rather than aggregated away: a converter that reports only a mean offset lets
    an individual frame be arbitrarily stale without anyone downstream noticing.
    """
    index: int
    t_ns: int
    cloud_stamps: Dict[str, int] = field(default_factory=dict)     # agent name -> raw stamp
    camera_stamps: Dict[Tuple[str, int], int] = field(default_factory=dict)
    skew_ns: Dict[str, int] = field(default_factory=dict)          # agent -> corrected - t_ns
    camera_skew_ns: Dict[Tuple[str, int], int] = field(default_factory=dict)
    local_key: str = ""            # index within its scenario folder (set at write time)

    def worst_skew_ns(self, agents: Optional[Sequence[str]] = None) -> int:
        values = [abs(v) for name, v in self.skew_ns.items()
                  if agents is None or name in agents]
        return max(values) if values else 0

    @property
    def key(self) -> str:
        return f"{self.index:06d}"


@dataclass
class FrameTable:
    frames: List[Frame]
    dropped: Dict[str, int]        # reason -> count
    master: str
    tolerance_ns: int
    dropped_at: List[Tuple[int, str]] = field(default_factory=list)   # (t_ns, reason)

    def dropped_runs(self, max_gap_ns: int) -> List[dict]:
        """Contiguous stretches of dropped candidates, per reason.

        A count of dropped frames does not say whether a sensor blinked a few
        hundred times or went away for fifteen seconds, and those are different
        datasets: scattered drops are a small loss, one long outage is a hole in
        the middle of a trajectory that any temporal model will feel."""
        runs: List[dict] = []
        for t_ns, reason in sorted(self.dropped_at):
            last = runs[-1] if runs else None
            if last and last["reason"] == reason and t_ns - last["end_ns"] <= max_gap_ns:
                last["end_ns"] = t_ns
                last["frames"] += 1
            else:
                runs.append({"reason": reason, "start_ns": t_ns, "end_ns": t_ns, "frames": 1})
        return runs

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
    dropped_at: List[Tuple[int, str]] = []

    for t_ns in times[::max(1, stride)]:
        cloud_stamps: Dict[str, int] = {}
        skews: Dict[str, int] = {}
        missing: List[str] = []
        for name, index in cloud_indices.items():
            found = index.nearest_pair(t_ns, tolerance_ns)
            if found is None:
                if required.get(name, True):
                    missing.append(name)
                continue
            corrected, raw = found
            cloud_stamps[name] = raw
            skews[name] = corrected - t_ns

        if missing:
            if drop_incomplete:
                key = "missing:" + ",".join(sorted(missing))
                dropped[key] = dropped.get(key, 0) + 1
                dropped_at.append((int(t_ns), key))
                continue
            # Without drop_incomplete a frame is kept only if *some* agent has data.
            if not cloud_stamps:
                dropped["empty"] = dropped.get("empty", 0) + 1
                dropped_at.append((int(t_ns), "empty"))
                continue

        frame = Frame(index=len(frames), t_ns=int(t_ns), cloud_stamps=cloud_stamps,
                      skew_ns=skews)
        for (agent_name, slot), index in camera_indices.items():
            found = index.nearest_pair(t_ns, tolerance_ns)
            if found is not None:
                corrected, raw = found
                frame.camera_stamps[(agent_name, slot)] = raw
                frame.camera_skew_ns[(agent_name, slot)] = corrected - t_ns
        frames.append(frame)

    return FrameTable(frames=frames, dropped=dropped, master=master,
                      tolerance_ns=tolerance_ns, dropped_at=dropped_at)


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
            offsets.append(abs(frame.skew_ns.get(name, stamp - frame.t_ns)) / 1e6)
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


def tightness_curve(table: FrameTable, required: Sequence[str],
                    clock_residual_ms: Optional[Dict[str, float]] = None,
                    grid_ms: Sequence[float] = (5, 10, 15, 20, 25, 30, 40, 50, 60, 80, 100),
                    ) -> Dict[str, object]:
    """How many frames survive at each candidate synchronisation budget.

    The right tolerance is a trade — how many frames am I willing to lose to halve
    the residual asynchrony? — and it cannot be made without seeing both columns.
    Reporting a single pass/fail against one hard-coded number hides the choice and
    usually hides the fact that the chosen number was unreachable anyway.

    ``clock_residual_ms`` adds each agent's un-correctable clock uncertainty to its
    selection skew, so the budget is against the total error a frame carries rather
    than the part that happens to be easy to measure.
    """
    residual = clock_residual_ms or {}
    totals = []
    for frame in table.frames:
        worst = 0.0
        for name in required:
            if name not in frame.skew_ns:
                worst = float("inf")
                break
            worst = max(worst, abs(frame.skew_ns[name]) / 1e6 + residual.get(name, 0.0))
        totals.append(worst)
    complete = [v for v in totals if v != float("inf")]
    return {
        "candidate_frames": len(table.frames),
        "complete_frames": len(complete),
        "worst_ms": {
            "p50": round(sorted(complete)[len(complete) // 2], 3) if complete else None,
            "max": round(max(complete), 3) if complete else None,
        },
        "curve": [{"budget_ms": float(g),
                   "frames": sum(1 for v in complete if v <= g),
                   "fraction": round(sum(1 for v in complete if v <= g) / len(table.frames), 4)
                   if table.frames else 0.0}
                  for g in grid_ms],
    }
