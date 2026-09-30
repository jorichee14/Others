# -*- coding: utf-8 -*-
"""
Several bags -> one, in log-time order: what ``ros2 bag convert`` with a
merge.yaml (``all_topics: true``) does, without ROS.

Every input is already in log-time order (the recorder writes it so, and so
does every step here), so the merge is exact: a k-way merge, no reordering
window to fall out of. Messages are copied byte for byte with their schema,
sequence, publish time and channel metadata; each topic keeps its input's QoS.

The output can be zstd-compressed inside the MCAP (every MCAP reader --
``ros2 bag play``, GLIM, scoop.bag -- undoes it), which is what makes a pass
of uncompressed images fit next to its inputs. Writing stops, and the partial
bag is removed, before the disk fills up.

    from scoop import merge
    merge.merge_bags([retimed, zed, zed_right, mobile_2_bag], out_dir)
"""
from __future__ import annotations

import heapq
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

from . import bag
from .bag import BagError, TopicSchema
from .bagwrite import BagWriter, write_in_order

__all__ = ["merge_bags", "MergeResult", "bag_bytes"]

MIN_FREE_GB = 10.0        # stop before the disk is fuller than this


def bag_bytes(bag_dir) -> int:
    return sum(f.stat().st_size for f in Path(bag_dir).rglob("*") if f.is_file())


@dataclass
class MergeResult:
    bag_dir: Path
    counts: Dict[str, int]
    sources: Dict[str, List[str]] = field(default_factory=dict)   # topic -> input bags
    t_min: int = 0
    t_max: int = 0


def _stream(b: Path, reader, total: List[int]):
    for _, sc, ch, msg in bag.iter_mcap_records(reader):
        total[0] += len(msg.data)
        yield (msg.log_time, ch.topic,
               TopicSchema(sc.name, sc.encoding, bytes(sc.data), ch.message_encoding),
               msg.publish_time, msg.sequence, dict(ch.metadata), msg.data)


def merge_bags(inputs, out_dir, compression: str = "zstd",
               min_free_gb: float = MIN_FREE_GB, log=print) -> MergeResult:
    """Merge the rosbag2 folders ``inputs`` into ``out_dir`` (must not exist)."""
    inputs = [Path(b) for b in inputs]
    readers = [bag.open_bag(b) for b in inputs]
    sources: Dict[str, List[str]] = {}
    types: Dict[str, str] = {}
    for b, r in zip(inputs, readers):
        for t, info in r.topics().items():
            if t in types and types[t] != info.msgtype:
                raise BagError(f"{t} is {types[t]} in {sources[t][0]} but "
                               f"{info.msgtype} in {b.name}")
            types[t] = info.msgtype
            sources.setdefault(t, []).append(b.name)
    for t, srcs in sorted(sources.items()):
        if len(srcs) > 1:
            log(f"    note: {t} is in {len(srcs)} inputs ({', '.join(srcs)}); all kept")
    total = sum(bag_bytes(b) for b in inputs) or 1
    log(f"    merging {len(inputs)} bags, {total / 1e9:.1f} GB -> {out_dir} "
        f"({compression})")

    done = [0]
    streams = [_stream(b, r, done) for b, r in zip(inputs, readers)]
    out_dir = Path(out_dir)

    def watched(records):
        shown = None                             # check at once, then every 30 s
        for rec in records:
            if shown is None or time.monotonic() - shown >= 30:
                shown = time.monotonic()
                free = shutil.disk_usage(out_dir).free / 1e9
                log(f"    merged {done[0] / 1e9:.1f} of {total / 1e9:.1f} GB "
                    f"({100 * done[0] / total:.0f}%), output {bag_bytes(out_dir) / 1e9:.1f} GB, "
                    f"{free:.0f} GB free")
                if free < min_free_gb:
                    raise BagError(f"only {free:.0f} GB left on the disk; stopped")
            yield rec

    try:
        with BagWriter(out_dir, library="scoop.merge", compression=compression) as w:
            for b, r in zip(inputs, readers):
                for t, q in bag.topic_qos(r).items():
                    if q and not w.qos.get(t):
                        w.set_qos(t, q)
            merged = heapq.merge(*streams, key=lambda rec: rec[0])
            write_in_order(w, watched(merged), reorder_s=2.0)   # tolerance for raw bags
    except ValueError as e:                  # an input out of log-time order
        shutil.rmtree(out_dir, ignore_errors=True)
        raise BagError(f"{e}: an input bag is not in log-time order") from None
    except BaseException:
        shutil.rmtree(out_dir, ignore_errors=True)
        raise
    res = MergeResult(w.dir, dict(w.counts), sources, int(w.t_min or 0), int(w.t_max or 0))
    log(f"wrote {w.n} messages, {bag_bytes(out_dir) / 1e9:.1f} GB -> {w.dir} "
        f"({(res.t_max - res.t_min) * 1e-9:.1f} s)")
    return res
