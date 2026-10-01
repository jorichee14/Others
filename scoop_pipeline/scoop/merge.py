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

import fnmatch
import heapq
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from . import bag, rosmsg
from .bag import BagError, TopicSchema
from .bagwrite import BagWriter, write_in_order

__all__ = ["merge_bags", "MergeResult", "bag_bytes", "select_topics", "list_topics",
           "resolve_sources", "duplicated"]

MIN_FREE_GB = 10.0        # stop before the disk is fuller than this
LATCHED = ("- history: 1\n  depth: 1\n  reliability: 1\n  durability: 1\n  deadline:\n"
           "    sec: 9223372036\n    nsec: 854775807\n  lifespan:\n    sec: 9223372036\n"
           "    nsec: 854775807\n  liveliness: 1\n  liveliness_lease_duration:\n"
           "    sec: 9223372036\n    nsec: 854775807\n  avoid_ros_namespace_conventions: false")


def bag_bytes(bag_dir) -> int:
    return sum(f.stat().st_size for f in Path(bag_dir).rglob("*") if f.is_file())


@dataclass
class MergeResult:
    bag_dir: Path
    counts: Dict[str, int]
    sources: Dict[str, List[str]] = field(default_factory=dict)   # topic -> input bags
    t_min: int = 0
    t_max: int = 0


def resolve_sources(inputs, topics: Iterable[str], rules, log=print) -> Dict[str, int]:
    """``rules`` = [(pattern, bag index)] -> {topic: bag index} for the
    ``topics`` that match and that bag has. A matched topic the bag does not
    have comes from wherever it is (noted); a rule whose bag has none of its
    topics is an error (wrong N, most likely)."""
    names = [set(bag.open_bag(b).topics()) for b in inputs]
    out: Dict[str, int] = {}
    for pat, i in rules:
        if not 0 <= i < len(inputs):
            raise BagError(f"--source {pat}={i + 1}: there are {len(inputs)} bags")
        hit = [t for t in topics if fnmatch.fnmatchcase(t, pat)]
        here = [t for t in hit if t in names[i]]
        if not here:
            raise BagError(f"--source {pat}={i + 1}: no chosen topic matching it is in "
                           f"{Path(inputs[i]).name}")
        for t in hit:
            if t in names[i]:
                out[t] = i
            else:
                log(f"note: {t} is not in {Path(inputs[i]).name}; taken from where it is")
    return out


def duplicated(inputs, topics: Iterable[str], source: Dict[str, int]) -> Dict[str, List[str]]:
    """Chosen topics that more than one bag would contribute."""
    names = [set(bag.open_bag(b).topics()) for b in inputs]
    out = {}
    for t in topics:
        if t in source:
            continue
        bags = [Path(b).name for b, n in zip(inputs, names) if t in n]
        if len(bags) > 1:
            out[t] = bags
    return out


def list_topics(inputs) -> Dict[str, List]:
    """topic -> [type, messages, [bag names]] over ``inputs``."""
    out: Dict[str, List] = {}
    for b in inputs:
        for t, info in bag.open_bag(b).topics().items():
            e = out.setdefault(t, [info.msgtype, 0, []])
            e[1] += int(info.count)
            e[2].append(Path(b).name)
    return dict(sorted(out.items()))


def select_topics(available: Iterable[str], include=(), exclude=()) -> List[str]:
    """The topics matching any ``include`` pattern (all if none) and no
    ``exclude`` pattern; patterns as in a shell (``/mobile_1/zed/*``).
    An include pattern matching nothing is an error (a typo, most likely)."""
    available = sorted(available)
    include, exclude = list(include or ()), list(exclude or ())
    for p in include:
        if not any(fnmatch.fnmatchcase(t, p) for t in available):
            raise BagError(f"--topics {p}: matches no topic in the bags")
    keep = [t for t in available
            if (not include or any(fnmatch.fnmatchcase(t, p) for p in include))
            and not any(fnmatch.fnmatchcase(t, p) for p in exclude)]
    if not keep:
        raise BagError("no topics selected")
    return keep


def _stream(b: Path, reader, total: List[int], topics, skip=()):
    if not topics:
        return
    for _, sc, ch, msg in bag.iter_mcap_records(reader, list(topics)):
        total[0] += 1
        if ch.topic in skip:
            continue
        yield (msg.log_time, ch.topic,
               TopicSchema(sc.name, sc.encoding, bytes(sc.data), ch.message_encoding),
               msg.publish_time, msg.sequence, dict(ch.metadata), msg.data)


def merge_bags(inputs, out_dir, compression: str = "zstd",
               min_free_gb: float = MIN_FREE_GB, static_tf=(),
               topics: Optional[Iterable[str]] = None,
               source: Optional[Dict[str, int]] = None, log=print) -> MergeResult:
    """Merge the rosbag2 folders ``inputs`` into ``out_dir`` (must not exist);
    ``topics`` limits it to those (default: all, see :func:`select_topics`),
    ``source`` = {topic: index into inputs} takes a topic from that bag only.
    ``static_tf``: ``[(parent, child, t, q)]`` added to /tf_static (see
    :func:`scoop.tftree.attach`).

    /tf_static is written as ONE message at the start holding every static
    transform of the inputs (the latest one per child frame) plus
    ``static_tf``, as tf2's StaticTransformBroadcaster does: a latched topic
    replays only its last message to a late subscriber (rviz, tf2_echo), so
    with several messages all but one set of transforms would go missing."""
    inputs = [Path(b) for b in inputs]
    readers = [bag.open_bag(b) for b in inputs]
    sources: Dict[str, List[str]] = {}
    types: Dict[str, str] = {}
    wanted = None if topics is None else set(topics)
    source = dict(source or {})
    sel = [[t for t in r.topics() if (wanted is None or t in wanted)
            and source.get(t, i) == i] for i, r in enumerate(readers)]
    n_total = sum(int(r.topics()[t].count) for r, ts_ in zip(readers, sel) for t in ts_) or 1
    for b, r, ts_ in zip(inputs, readers, sel):
        for t in ts_:
            info = r.topics()[t]
            if t in types and types[t] != info.msgtype:
                raise BagError(f"{t} is {types[t]} in {sources[t][0]} but "
                               f"{info.msgtype} in {b.name}")
            types[t] = info.msgtype
            sources.setdefault(t, []).append(b.name)
    for t, srcs in sorted(sources.items()):
        if len(srcs) > 1:
            log(f"    note: {t} is in {len(srcs)} inputs ({', '.join(srcs)}); all kept")
    log(f"    merging {len(types)} topics, {n_total} messages from {len(inputs)} bags "
        f"-> {out_dir} ({compression})")

    done = [0]
    name = "tf2_msgs/msg/TFMessage"
    combine = "/tf_static" in types or bool(static_tf)
    if combine and types.get("/tf_static", name) != name:
        raise BagError(f"/tf_static is {types['/tf_static']} in {sources['/tf_static'][0]}, "
                       f"not {name}")
    streams = [_stream(b, r, done, ts_, skip={"/tf_static"} if combine else ())
               for b, r, ts_ in zip(inputs, readers, sel)]
    if combine:
        ts = rosmsg.typestore()
        t0 = min(r.time_range()[0] for r in readers)
        by_child: Dict[str, object] = {}
        n_in = 0
        for r, ts_ in zip(readers, sel):
            if "/tf_static" not in ts_:
                continue
            for _, _, _, msg in bag.iter_mcap_records(r, ["/tf_static"]):
                n_in += 1
                for tr in ts.deserialize_cdr(msg.data, name).transforms:
                    old = by_child.get(tr.child_frame_id)
                    if old is not None and old.header.frame_id != tr.header.frame_id:
                        raise BagError(f"/tf_static gives {tr.child_frame_id} two parents: "
                                       f"{old.header.frame_id} and {tr.header.frame_id}")
                    by_child[tr.child_frame_id] = tr
        for tr in rosmsg.tf_message(static_tf, t0).transforms:
            if tr.child_frame_id in by_child:
                raise BagError(f"{tr.child_frame_id} already has a parent in /tf_static")
            by_child[tr.child_frame_id] = tr
            log(f"    /tf_static += {tr.header.frame_id} -> {tr.child_frame_id}")
        msg = rosmsg.msg_type(name)(transforms=list(by_child.values()))
        schema = TopicSchema(name, "ros2msg", rosmsg.msgdef(name).encode(), "cdr")
        streams.insert(0, iter([(t0, "/tf_static", schema, t0, 0, {}, rosmsg.serialize(msg))]))
        log(f"    /tf_static: {n_in} messages + {len(static_tf)} added -> one message with "
            f"{len(by_child)} transforms")
    out_dir = Path(out_dir)

    def watched(records):
        shown = None                             # check at once, then every 30 s
        for rec in records:
            if shown is None or time.monotonic() - shown >= 30:
                shown = time.monotonic()
                free = shutil.disk_usage(out_dir).free / 1e9
                log(f"    merged {done[0]} of {n_total} messages "
                    f"({100 * done[0] / n_total:.0f}%), output {bag_bytes(out_dir) / 1e9:.1f} GB, "
                    f"{free:.0f} GB free")
                if free < min_free_gb:
                    raise BagError(f"only {free:.0f} GB left on the disk; stopped")
            yield rec

    try:
        with BagWriter(out_dir, library="scoop.merge", compression=compression) as w:
            for b, r, ts_ in zip(inputs, readers, sel):
                for t, q in bag.topic_qos(r).items():
                    if q and t in ts_ and not w.qos.get(t):
                        w.set_qos(t, q)
            if combine:
                w.set_qos("/tf_static", LATCHED)          # keep last 1, transient local: tf2's
            merged = heapq.merge(*streams, key=lambda rec: rec[0])
            write_in_order(w, watched(merged), reorder_s=2.0)   # tolerance for raw bags
    except ValueError as e:                  # an input out of log-time order, a type clash
        shutil.rmtree(out_dir, ignore_errors=True)
        raise BagError(str(e)) from None
    except BaseException:
        shutil.rmtree(out_dir, ignore_errors=True)
        raise
    res = MergeResult(w.dir, dict(w.counts), sources, int(w.t_min or 0), int(w.t_max or 0))
    log(f"wrote {w.n} messages, {bag_bytes(out_dir) / 1e9:.1f} GB -> {w.dir} "
        f"({(res.t_max - res.t_min) * 1e-9:.1f} s)")
    return res
