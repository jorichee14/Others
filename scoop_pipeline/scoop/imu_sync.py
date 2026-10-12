# -*- coding: utf-8 -*-
"""
How far one IMU's header stamps run behind another's on the same robot.

Two IMUs on one rigid body turn together, so the size of their angular rate
|w| is the same signal whatever each one's mounting. The shift that
correlates the two best is how far ``topic``'s stamps run behind ``ref``'s
(the reference: on mobile_1 the Ouster's IMU, on the capture clock by the
clock fit). Found on the coop_2 pass of 20260924: the ZED's live IMU
(/zed/zed_node/imu/data, stamped by the wrapper on the robot) 2.12 s behind
its images, its LiDAR and the Ouster's IMU, the same over the whole pass.

    from scoop import imu_sync
    r = imu_sync.measure(reader, "/mobile_1/zed/imu/data", "/mobile_1/ouster/imu",
                         ref_stamp=clock.host_ns)
    r.shift_ns        # take it off the topic's stamps (retime_bag does)
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable, Optional, Tuple

import numpy as np

from . import bag

__all__ = ["ImuSync", "angular_rates", "line_up", "measure", "MIN_CORR"]

MIN_CORR = 0.8             # below this the two do not agree enough to correct anything
MIN_TURN = 0.02            # rad/s rms of the reference: the robot has to turn at all


@dataclass
class ImuSync:
    topic: str
    ref: str
    shift_s: float          # topic's stamps behind ref's (+ = late)
    correlation: float
    scale: float            # |w| of topic over ref's (about 1)
    samples: int
    turn_rms: float         # rad/s, the reference's |w| less its mean
    ok: bool                # correlation >= min_corr and the robot turned

    @property
    def shift_ns(self) -> int:
        return int(round(self.shift_s * 1e9))

    def as_dict(self) -> dict:
        return asdict(self)


def angular_rates(reader, topic: str, stamp: Optional[Callable[[int], int]] = None
                  ) -> Tuple[np.ndarray, np.ndarray]:
    """(header stamps ns, |w| rad/s) of every sensor_msgs/Imu on ``topic``;
    ``stamp`` maps each header stamp (e.g. a clock fit's host_ns)."""
    from . import rosmsg
    from .retime import header_stamp_ns
    ts = rosmsg.typestore()
    t, w = [], []
    for _, _, _, msg in bag.iter_mcap_records(reader, [topic]):
        s = header_stamp_ns(msg.data)
        av = ts.deserialize_cdr(msg.data, "sensor_msgs/msg/Imu").angular_velocity
        t.append(stamp(s) if stamp else s)
        w.append(np.sqrt(av.x * av.x + av.y * av.y + av.z * av.z))
    t, w = np.array(t, np.int64), np.array(w)
    o = np.argsort(t, kind="stable")
    return t[o], w[o]


def _smooth(T, t, x, width):
    y = np.interp(T, t, x)
    n = max(1, int(round(width / (T[1] - T[0]))))
    return np.convolve(y, np.ones(n) / n, "same")


def line_up(t_ref, w_ref, t, w, max_shift=5.0, smooth_s=0.2):
    """(shift s, correlation, scale, samples, turn rms) with w(t + shift) ~ w_ref(t);
    times in seconds. A coarse search (0.05 s) then a fine one (2 ms)."""
    lo = max(t_ref[0], t[0]) + max_shift + 1
    hi = min(t_ref[-1], t[-1]) - max_shift - 1
    if hi - lo < 10:
        return 0.0, float("nan"), float("nan"), 0, 0.0
    dt = 0.005
    G = np.arange(lo - max_shift - 0.5, hi + max_shift + 0.5, dt)
    a_full, b_full = _smooth(G, t_ref, w_ref, smooth_s), _smooth(G, t, w, smooth_s)
    T = np.arange(lo, hi, dt)
    a = np.interp(T, G, a_full)
    a = a - a.mean()

    def corr(s):
        b = np.interp(T + s, G, b_full)
        b = b - b.mean()
        d = np.sqrt((a @ a) * (b @ b))
        return (a @ b) / d if d > 0 else -1.0

    s = max(np.arange(-max_shift, max_shift + 1e-9, 0.05), key=corr)
    s = max(np.arange(s - 0.06, s + 0.06 + 1e-9, 0.002), key=corr)
    b = np.interp(T + s, G, b_full)
    b = b - b.mean()
    scale = float((a @ b) / (a @ a)) if a @ a > 0 else float("nan")
    return float(s), float(corr(s)), scale, len(T), float(np.sqrt(np.mean(a * a)))


def measure(reader, topic: str, ref: str, ref_stamp: Optional[Callable[[int], int]] = None,
            stamp: Optional[Callable[[int], int]] = None, max_shift: float = 5.0,
            min_corr: float = MIN_CORR) -> ImuSync:
    """How far ``topic``'s IMU stamps run behind ``ref``'s in ``reader``."""
    tr, wr = angular_rates(reader, ref, ref_stamp)
    tt, wt = angular_rates(reader, topic, stamp)
    if len(tr) < 100 or len(tt) < 100:
        return ImuSync(topic, ref, 0.0, float("nan"), float("nan"), 0, 0.0, False)
    s, c, sc, n, turn = line_up(tr * 1e-9, wr, tt * 1e-9, wt, max_shift)
    ok = bool(n and c >= min_corr and turn >= MIN_TURN)
    return ImuSync(topic, ref, round(s, 3), c, sc, n, turn, ok)
