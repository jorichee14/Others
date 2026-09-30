# -*- coding: utf-8 -*-
"""
How evenly a topic's messages are spaced -- what a frame count cannot show.

Per topic: the intervals between header stamps (median, percentiles, gaps
longer than 1.5x the median), stamps seen twice, consecutive messages with
identical bytes past the header (a frame repeated under a new stamp), and
log time - stamp.

    from scoop import timing
    for t in timing.topic_timing(bag_dir, ["/mobile_1/zed/left/image_rect_color"]):
        print(t.summary())
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from . import bag
from .retime import header_stamp_ns

__all__ = ["TopicTiming", "topic_timing"]


@dataclass
class TopicTiming:
    topic: str
    stamps: np.ndarray                          # header stamps, ns, in log order
    logs: np.ndarray                            # log times, ns
    repeats: int = 0                            # same content as the message before
    repeat_at: List[float] = field(default_factory=list)   # seconds from the start

    @property
    def n(self) -> int:
        return len(self.stamps)

    def summary(self, gap_factor: float = 1.5) -> str:
        if self.n < 2:
            return f"{self.topic}: {self.n} messages"
        st = np.sort(self.stamps)
        dt = np.diff(st) * 1e-6                               # ms
        med = float(np.median(dt))
        gaps = np.flatnonzero(dt > gap_factor * med)
        span = (st[-1] - st[0]) * 1e-9
        out = [f"{self.topic}: {self.n} msgs over {span:.1f} s, {(self.n - 1) / span:.2f} Hz",
               f"    interval ms: median {med:.1f}  p1 {np.percentile(dt, 1):.1f}  "
               f"p99 {np.percentile(dt, 99):.1f}  min {dt.min():.1f}  max {dt.max():.1f}",
               f"    gaps > {gap_factor}x median: {len(gaps)}"]
        if len(gaps):
            worst = gaps[np.argsort(dt[gaps])[::-1][:5]]
            out[-1] += "  (longest: " + ", ".join(
                f"{dt[i]:.0f} ms at {(st[i] - st[0]) * 1e-9:.1f} s" for i in worst) + ")"
        out.append(f"    same stamp twice: {int(np.sum(dt == 0))}   "
                   f"out of order in the bag: {int(np.sum(np.diff(self.stamps) < 0))}   "
                   f"same content as the previous message: {self.repeats}")
        if self.repeat_at:
            out[-1] += " (at " + ", ".join(f"{s:.1f} s" for s in self.repeat_at[:5]) + \
                (" ...)" if len(self.repeat_at) > 5 else ")")
        lag = (self.logs - self.stamps) * 1e-6
        out.append(f"    log - stamp ms: median {np.median(lag):.1f}  "
                   f"min {lag.min():.1f}  max {lag.max():.1f}")
        return "\n".join(out)


def topic_timing(bag_dir, topics: List[str]) -> List[TopicTiming]:
    """Read ``topics`` of ``bag_dir`` once and time them."""
    reader = bag.open_bag(bag_dir)
    have = reader.topics()
    missing = [t for t in topics if t not in have]
    if missing:
        raise bag.BagError(f"not in {bag_dir}: {missing}")
    stamps: Dict[str, list] = {t: [] for t in topics}
    logs: Dict[str, list] = {t: [] for t in topics}
    last: Dict[str, bytes] = {}
    repeats: Dict[str, list] = {t: [] for t in topics}
    t0 = None
    for _, _, ch, msg in bag.iter_mcap_records(reader, topics):
        data = msg.data
        st = header_stamp_ns(data)
        t0 = st if t0 is None else t0
        stamps[ch.topic].append(st)
        logs[ch.topic].append(msg.log_time)
        h = hashlib.blake2b(memoryview(data)[12:], digest_size=16).digest()   # past the stamp
        if last.get(ch.topic) == h:
            repeats[ch.topic].append((st - t0) * 1e-9)
        last[ch.topic] = h
    return [TopicTiming(t, np.array(stamps[t], np.int64), np.array(logs[t], np.int64),
                        len(repeats[t]), repeats[t]) for t in topics]
