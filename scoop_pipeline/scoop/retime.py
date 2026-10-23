# -*- coding: utf-8 -*-
"""
Sensor time -> host capture time for an Ouster points bag. ``retime_bag.py``
without ROS.

``retime_bag.py`` needed ROS only to deserialize each PointCloud2/Imu, change
its header stamp and serialize it again. The stamp sits at a fixed place in
every CDR message (right after the 4-byte encapsulation header: int32 sec,
uint32 nanosec), so here it is rewritten in the bytes directly; everything
else is the same procedure:

1. clock fit on the ORIGINAL packets bag (:func:`scoop.ouster.fit_clock`, the
   port of ``fit_clock_map``);
2. ``<points_ns>/points`` and ``<points_ns>/imu``: header stamp -> host time,
   log and publish time = the new stamp;
3. every other message shifted by the median (mapped stamp - log time) of
   those two topics, so the whole bag lives on the capture timeline;
4. written in non-decreasing log-time order through a bounded reorder buffer;
   metadata.yaml regenerated from the input's, keeping its QoS profiles;
5. ``imu_sync``: another IMU copied through (the ZED's live one), whose
   driver stamps it late, measured against the Ouster's IMU on the capture
   clock (:mod:`scoop.imu_sync`) and its header stamps moved back by that
   (only when the two correlate; its log time is shifted as in 3).

    from scoop import retime
    retime.retime_bag(original, decoded, out_dir, "/ouster", "/mobile_1/ouster")
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np

from . import bag, imu_sync as IS, ouster
from .bag import BagError, TopicSchema
from .bagwrite import BagWriter, write_in_order

__all__ = ["REORDER_S", "header_stamp_ns", "with_header_stamp", "replay_shift_ns",
           "retime_bag", "RetimeResult"]

REORDER_S = 2.0          # retime_bag.REORDER_S: must exceed pipeline-latency variation
MAX_REORDER_S = 60.0     # a wider window (a message logged later than this) is refused


def _endian(payload) -> str:
    if len(payload) < 12 or payload[0] != 0:
        raise BagError("not a CDR message with a header")
    return "<" if payload[1] & 1 else ">"


def header_stamp_ns(payload) -> int:
    """Header stamp (ns) of any stamped ROS 2 message, read from its bytes."""
    sec, nsec = struct.unpack_from(_endian(payload) + "iI", payload, 4)
    return sec * 1_000_000_000 + nsec


def with_header_stamp(payload, stamp_ns: int) -> bytes:
    """The same message bytes with the header stamp replaced."""
    out = bytearray(payload)
    struct.pack_into(_endian(payload) + "iI", out, 4,
                     int(stamp_ns // 1_000_000_000), int(stamp_ns % 1_000_000_000))
    return bytes(out)


def replay_shift_ns(reader, topics, clock: ouster.ClockMap,
                    reorder_s: float = REORDER_S, log=print, stats=None) -> int:
    """Median (mapped capture time - log time) over ``topics``: the constant
    that moves every other message onto the capture timeline. ``stats``
    (a dict, if given) gets "late_s": how far the most out-of-place of them
    was logged from that (a recorder that fell behind and caught up)."""
    deltas = [clock.host_ns(header_stamp_ns(msg.data)) - msg.log_time
              for _, _, _, msg in bag.iter_mcap_records(reader, topics)]
    if not deltas:
        raise BagError(f"no messages on {topics}; check the points namespace")
    delta = int(np.median(deltas))
    if stats is not None:          # either way: early ones hold the rest back as much
        stats["late_s"] = max(0.0, (delta - min(deltas)) * 1e-9, (max(deltas) - delta) * 1e-9)
    spread = (np.percentile(deltas, 99) - np.percentile(deltas, 1)) * 1e-9
    if spread > reorder_s:
        raise BagError(f"replay-latency spread {spread:.2f} s exceeds the "
                       f"{reorder_s} s reorder window")
    log(f"replay -> capture shift: {delta * 1e-9:+.3f} s "
        f"(latency spread {spread * 1e3:.0f} ms over {len(deltas)} msgs)")
    return delta


@dataclass
class RetimeResult:
    bag_dir: Path
    clock: ouster.ClockMap
    shift_ns: int
    retimed: Dict[str, int]                  # messages retimed per topic
    counts: Dict[str, int] = field(default_factory=dict)
    t_min: int = 0
    t_max: int = 0
    imu_sync: List["IS.ImuSync"] = field(default_factory=list)


def retime_bag(packets_bag, sensortime_bag, out_dir, packets_ns: str = "/ouster",
               points_ns: str = "/mobile_1/ouster",
               clock: Optional[ouster.ClockMap] = None,
               reorder_s: float = REORDER_S, metadata=None, imu_sync: Iterable[dict] = (),
               log=print) -> RetimeResult:
    """Retime ``sensortime_bag`` (decoded or replayed, sensor-time stamps)
    into ``out_dir`` using the clock of ``packets_bag`` (the original).
    ``clock`` skips the fit (e.g. a saved :class:`~scoop.ouster.ClockMap`);
    ``metadata`` overrides the packets bag's sensor metadata. ``imu_sync``:
    [{"topic": ..., "ref": (default <points_ns>/imu), "max_shift": s,
    "min_corr": c}], see step 5."""
    if clock is None:
        clock = ouster.fit_clock(bag.open_bag(packets_bag), packets_ns,
                                 metadata=metadata, log=log)
    reader = bag.open_bag(sensortime_bag)
    retime = [f"{points_ns}/points", f"{points_ns}/imu"]
    st = {}
    delta = replay_shift_ns(reader, retime, clock, reorder_s, log, stats=st)
    # a LiDAR/IMU message logged long after its capture lands that far back in
    # the output: the reorder window must cover it (20261015 mobile_2's
    # recorder fell behind in its Ouster gap)
    need = st.get("late_s", 0.0) + 0.5
    if need > reorder_s:
        if need > MAX_REORDER_S:
            raise BagError(f"a {retime[0]} / imu message was logged {need - 0.5:.1f} s after its "
                           f"capture: more than the {MAX_REORDER_S:.0f} s reorder window allowed")
        log(f"reorder window {need:.1f} s: a LiDAR/IMU message was logged {need - 0.5:.1f} s "
            f"after its capture (the recorder fell behind)")
        reorder_s = need
    synced, shift = [], {}
    for c in imu_sync or ():
        r = IS.measure(reader, c["topic"], c.get("ref", f"{points_ns}/imu"), ref_stamp=clock.host_ns,
                       max_shift=float(c.get("max_shift", 5.0)),
                       min_corr=float(c.get("min_corr", IS.MIN_CORR)))
        synced.append(r)
        if r.ok:
            shift[r.topic] = r.shift_ns
            log(f"imu sync: {r.topic} {r.shift_s:+.3f} s behind {r.ref} (correlation "
                f"{r.correlation:.2f}, scale {r.scale:.2f}): stamps moved back by it")
        else:
            log(f"imu sync: WARNING {r.topic} vs {r.ref}: correlation {r.correlation:.2f}, "
                f"turning {r.turn_rms:.3f} rad/s -- not sure enough, stamps left as they are")

    template = Path(sensortime_bag) / "metadata.yaml"
    retimed = {t: 0 for t in retime}

    def records():
        for _, sc, ch, msg in bag.iter_mcap_records(reader):
            data = msg.data
            if ch.topic in retimed:
                new_ns = clock.host_ns(header_stamp_ns(data))
                data = with_header_stamp(data, new_ns)
                log_time = pub_time = new_ns
                retimed[ch.topic] += 1
            elif ch.topic in shift:
                data = with_header_stamp(data, header_stamp_ns(data) - shift[ch.topic])
                log_time = msg.log_time + delta
                pub_time = msg.publish_time + delta
            else:
                log_time = msg.log_time + delta
                pub_time = msg.publish_time + delta
            schema = TopicSchema(sc.name, sc.encoding, bytes(sc.data), ch.message_encoding)
            yield (log_time, ch.topic, schema, pub_time, msg.sequence,
                   dict(ch.metadata), data)

    with BagWriter(out_dir, library="scoop.retime",
                   metadata_template=template if template.exists() else None) as w:
        try:
            write_in_order(w, records(), reorder_s)
        except ValueError as e:
            raise BagError(str(e)) from None

    res = RetimeResult(w.dir, clock, delta, retimed, dict(w.counts),
                       int(w.t_min or 0), int(w.t_max or 0), synced)
    log(f"wrote {w.n} messages -> {w.dir}")
    for t, c in retimed.items():
        log(f"retimed {c} msgs on {t}")
    log(f"output time range : {res.t_min * 1e-9:.3f} .. {res.t_max * 1e-9:.3f} "
        f"({(res.t_max - res.t_min) * 1e-9:.1f} s), non-decreasing")
    return res
