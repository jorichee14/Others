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
   metadata.yaml regenerated from the input's, keeping its QoS profiles.

    from scoop import retime
    retime.retime_bag(original, decoded, out_dir, "/ouster", "/mobile_1/ouster")
"""
from __future__ import annotations

import heapq
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from . import bag, ouster
from .bag import BagError, TopicSchema
from .bagwrite import BagWriter

__all__ = ["REORDER_S", "header_stamp_ns", "with_header_stamp", "replay_shift_ns",
           "retime_bag", "RetimeResult"]

REORDER_S = 2.0          # retime_bag.REORDER_S: must exceed pipeline-latency variation


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
                    reorder_s: float = REORDER_S, log=print) -> int:
    """Median (mapped capture time - log time) over ``topics``: the constant
    that moves every other message onto the capture timeline."""
    deltas = [clock.host_ns(header_stamp_ns(msg.data)) - msg.log_time
              for _, _, _, msg in bag.iter_mcap_records(reader, topics)]
    if not deltas:
        raise BagError(f"no messages on {topics}; check the points namespace")
    delta = int(np.median(deltas))
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


def retime_bag(packets_bag, sensortime_bag, out_dir, packets_ns: str = "/ouster",
               points_ns: str = "/mobile_1/ouster",
               clock: Optional[ouster.ClockMap] = None,
               reorder_s: float = REORDER_S, metadata=None, log=print) -> RetimeResult:
    """Retime ``sensortime_bag`` (decoded or replayed, sensor-time stamps)
    into ``out_dir`` using the clock of ``packets_bag`` (the original).
    ``clock`` skips the fit (e.g. a saved :class:`~scoop.ouster.ClockMap`);
    ``metadata`` overrides the packets bag's sensor metadata."""
    if clock is None:
        clock = ouster.fit_clock(bag.open_bag(packets_bag), packets_ns,
                                 metadata=metadata, log=log)
    reader = bag.open_bag(sensortime_bag)
    retime = [f"{points_ns}/points", f"{points_ns}/imu"]
    delta = replay_shift_ns(reader, retime, clock, reorder_s, log)

    template = Path(sensortime_bag) / "metadata.yaml"
    retimed = {t: 0 for t in retime}
    horizon_ns = int(reorder_s * 1e9)
    heap = []
    tiebreak = 0
    watermark = -1
    last = [-1]

    with BagWriter(out_dir, library="scoop.retime",
                   metadata_template=template if template.exists() else None) as w:

        def emit(item):
            log_time, _, topic, schema, pub, seq, meta, data = item
            if log_time < last[0]:
                raise BagError("reorder window too small: output would go back in time")
            last[0] = log_time
            w.write_raw(topic, schema, data, log_time, publish_ns=pub, sequence=seq,
                        channel_metadata=meta)

        for _, sc, ch, msg in bag.iter_mcap_records(reader):
            data = msg.data
            if ch.topic in retimed:
                new_ns = clock.host_ns(header_stamp_ns(data))
                data = with_header_stamp(data, new_ns)
                log_time = pub_time = new_ns
                retimed[ch.topic] += 1
            else:
                log_time = msg.log_time + delta
                pub_time = msg.publish_time + delta
            schema = TopicSchema(sc.name, sc.encoding, bytes(sc.data), ch.message_encoding)
            heapq.heappush(heap, (log_time, tiebreak, ch.topic, schema, pub_time,
                                  msg.sequence, dict(ch.metadata), data))
            tiebreak += 1
            watermark = max(watermark, log_time)
            while heap and heap[0][0] <= watermark - horizon_ns:
                emit(heapq.heappop(heap))
        while heap:
            emit(heapq.heappop(heap))

    res = RetimeResult(w.dir, clock, delta, retimed, dict(w.counts),
                       int(w.t_min or 0), int(w.t_max or 0))
    log(f"wrote {w.n} messages -> {w.dir}")
    for t, c in retimed.items():
        log(f"retimed {c} msgs on {t}")
    log(f"output time range : {res.t_min * 1e-9:.3f} .. {res.t_max * 1e-9:.3f} "
        f"({(res.t_max - res.t_min) * 1e-9:.1f} s), non-decreasing")
    return res
