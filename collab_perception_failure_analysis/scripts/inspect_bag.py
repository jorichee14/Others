#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage A of the ROS 2 -> OPV2V conversion: look at the bag before converting it.

Answers the questions the converter config has to be built from, and that the
recording's own ``metadata.yaml`` cannot answer:

* what each topic actually carries (type, true rate, ``frame_id``)
* how far each topic's header stamp sits from its log time — i.e. whether the
  agents' clocks are disciplined well enough for cross-agent synchronisation
* what the TF tree looks like, and in particular whether any frame is shared
  between agents (if none is, every agent's odometry lives in its own origin and
  the alignment transforms have to be supplied by hand)
* point-cloud field layouts, image encodings, camera intrinsics

``--emit-config`` writes a converter config skeleton with the topics filled in
and every extrinsic left null, so the parts a human must supply fail loudly.

Usage:
    python scripts/inspect_bag.py --bag ~/bags/mirc_dataset_coop2_20260828 \
        [--emit-config configs/my_bag.yaml] [--sample 200]
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ros2opv2v.bagreader import BagReader, stamp_from_cdr          # noqa: E402

NS = 1_000_000_000

CLOUD_TYPES = ("sensor_msgs/msg/PointCloud2",)
IMAGE_TYPES = ("sensor_msgs/msg/Image",)
INFO_TYPES = ("sensor_msgs/msg/CameraInfo",)
POSE_TYPES = ("nav_msgs/msg/Odometry", "geometry_msgs/msg/PoseStamped",
              "geometry_msgs/msg/PoseWithCovarianceStamped")


def namespace_of(topic: str) -> str:
    """Leading namespace of a topic, used to guess which agent it belongs to."""
    parts = [p for p in topic.split("/") if p]
    return parts[0] if parts else ""


def human_hz(count: int, span_ns: int) -> float:
    return (count - 1) / (span_ns / NS) if count > 1 and span_ns > 0 else 0.0


def summarize(reader: BagReader, sample: int) -> dict:
    """Per-topic stamp statistics from the index pass (no message decoding)."""
    topics = reader.topics()
    indexes = reader.index(sorted(topics))
    out = {}
    for name, info in sorted(topics.items()):
        entry = indexes.get(name)
        stamps = entry.header_stamps if entry else []
        logs = entry.log_times if entry else []
        if not stamps:
            out[name] = {"type": info.msgtype, "count": info.count, "rate_hz": 0.0}
            continue
        skew = [(h - l) / 1e6 for h, l in zip(stamps, logs)]
        gaps = [(b - a) / 1e6 for a, b in zip(stamps, stamps[1:])]
        gaps_sorted = sorted(gaps)
        out[name] = {
            "type": info.msgtype,
            "count": len(stamps),
            "rate_hz": round(human_hz(len(stamps), stamps[-1] - stamps[0]), 2),
            "start_ns": stamps[0],
            "end_ns": stamps[-1],
            "headerless": entry.headerless,
            "stamp_minus_log_ms": round(sum(skew) / len(skew), 2),
            "median_gap_ms": round(gaps_sorted[len(gaps_sorted) // 2], 2) if gaps else 0.0,
            "max_gap_ms": round(max(gaps), 2) if gaps else 0.0,
        }
    return out


def sample_details(reader: BagReader, stats: dict, sample: int) -> dict:
    """Decode one message per interesting topic to read frame_ids and layouts."""
    wanted = [name for name, s in stats.items()
              if s["type"] in CLOUD_TYPES + IMAGE_TYPES + INFO_TYPES + POSE_TYPES]
    details = {}
    remaining = set(wanted)
    if not remaining:
        return details

    for topic, stamp, msg in reader.iter_messages(sorted(remaining)):
        if topic not in remaining:
            continue
        remaining.discard(topic)
        entry = {"frame_id": getattr(getattr(msg, "header", None), "frame_id", "")}
        msgtype = stats[topic]["type"]

        if msgtype in CLOUD_TYPES:
            entry["fields"] = [f"{f.name}:{f.datatype}" for f in msg.fields]
            entry["size"] = f"{msg.width}x{msg.height}"
            entry["point_step"] = int(msg.point_step)
            entry["is_dense"] = bool(msg.is_dense)
        elif msgtype in IMAGE_TYPES:
            entry["encoding"] = str(msg.encoding)
            entry["size"] = f"{msg.width}x{msg.height}"
        elif msgtype in INFO_TYPES:
            k = list(getattr(msg, "k", None) or getattr(msg, "K", []) or [])
            entry["size"] = f"{msg.width}x{msg.height}"
            entry["fx_fy_cx_cy"] = [round(float(k[0]), 2), round(float(k[4]), 2),
                                    round(float(k[2]), 2), round(float(k[5]), 2)] \
                if len(k) >= 9 else None
        elif msgtype in POSE_TYPES:
            entry["child_frame_id"] = str(getattr(msg, "child_frame_id", "") or "")
        details[topic] = entry
        if not remaining:
            break
    return details


def tf_tree(reader: BagReader, sample: int) -> dict:
    """Parent -> children edges seen on /tf and /tf_static."""
    topics = [t for t in ("/tf", "/tf_static") if t in reader.topics()]
    edges = defaultdict(Counter)
    if not topics:
        return {}
    seen = 0
    for topic, stamp, msg in reader.iter_messages(topics):
        for transform in getattr(msg, "transforms", []):
            parent = str(transform.header.frame_id)
            child = str(transform.child_frame_id)
            edges[parent][child] += 1
        seen += 1
        if seen >= sample:
            break
    return {parent: dict(children) for parent, children in edges.items()}


def guess_agents(stats: dict, details: dict) -> dict:
    """Group topics by namespace and guess each agent's role topics."""
    agents = defaultdict(lambda: {"clouds": [], "depth": [], "poses": [],
                                  "cameras": [], "infos": []})
    for topic, entry in stats.items():
        namespace = namespace_of(topic)
        if not namespace or namespace in ("tf", "tf_static", "rosout", "parameter_events"):
            continue
        bucket = agents[namespace]
        if entry["type"] in CLOUD_TYPES:
            bucket["clouds"].append(topic)
        elif entry["type"] in POSE_TYPES:
            bucket["poses"].append(topic)
        elif entry["type"] in INFO_TYPES:
            bucket["infos"].append(topic)
        elif entry["type"] in IMAGE_TYPES:
            encoding = (details.get(topic) or {}).get("encoding", "")
            if "16uc1" in encoding.lower() or "32fc1" in encoding.lower() \
                    or "depth" in topic.lower():
                bucket["depth"].append(topic)
            else:
                bucket["cameras"].append(topic)
    return {k: v for k, v in sorted(agents.items())}


def emit_config(path: str, bag: str, stats: dict, details: dict, agents: dict) -> None:
    """Write a converter config skeleton: topics filled in, geometry left null."""
    lines = [
        "# Converter config skeleton generated by scripts/inspect_bag.py.",
        "#",
        "# Every `extrinsic:` and `align:` below is null on purpose. They are the",
        "# facts the bag cannot tell you: where each sensor sits on its robot, and",
        "# what transform puts each robot's odometry into ONE shared world frame.",
        "# The converter refuses to run while any of them is null, because a wrong",
        "# guess produces a dataset that looks fine and is geometrically meaningless.",
        "",
        f"bag: {bag}",
        "",
        "output:",
        "  root: ~/cpfa/data/OPV2V_from_bag",
        "  split: test",
        "  scenario_name: scenario_00",
        "  frames_per_scenario: 0      # 0 = one scenario for the whole bag",
        "  frame_stride: 1",
        "",
        "time:",
        "  stamp_source: header",
        "  master_agent: null          # defaults to the ego (lowest cav id)",
        "  rate_hz: 0                  # 0 = keep the master agent's own cadence",
        "  match_tolerance_ms: 60",
        "  drop_incomplete_frames: true",
        "",
        "world:",
        "  frame_id: world",
        "",
        "agents:",
    ]

    for index, (namespace, bucket) in enumerate(agents.items()):
        clouds = bucket["clouds"] + bucket["depth"]
        if not clouds:
            continue
        cloud_topic = clouds[0]
        is_depth = cloud_topic in bucket["depth"]
        pose_topic = bucket["poses"][0] if bucket["poses"] else None
        info_topic = None
        if is_depth:
            base = cloud_topic.rsplit("/", 1)[0]
            info_topic = next((t for t in bucket["infos"] if t.startswith(base)),
                              bucket["infos"][0] if bucket["infos"] else None)

        lines += [
            f"  - name: {namespace}",
            f"    id: {index + 1}                     # OpenCOOD uses the lowest id as ego",
            "    role: cav                 # 'rsu' for infrastructure (needs a negative id)",
            "    enabled: true",
            "    pose:",
        ]
        if pose_topic:
            lines += [
                "      source: odometry",
                f"      topic: {pose_topic}",
                "      interpolation: linear",
                "      max_gap_ms: 200",
                "      align: null             # REQUIRED: world <- this agent's odom frame",
                "      child_to_base: {x: 0, y: 0, z: 0, roll: 0, pitch: 0, yaw: 0}",
            ]
        else:
            lines += [
                "      source: static",
                "      world_pose: null        # REQUIRED: world <- this agent's base frame",
            ]
        lines += [
            "    cloud:",
            f"      kind: {'depth_image' if is_depth else 'pointcloud2'}",
            f"      topic: {cloud_topic}",
            "      extrinsic: null           # REQUIRED: agent base -> this sensor",
        ]
        if is_depth:
            lines += [
                f"      camera_info_topic: {info_topic}",
                "      depth_scale: 0.001      # 16UC1 counts -> metres",
                "      min_depth: 0.2",
                "      max_depth: 12.0",
                "      pixel_stride: 2",
                "      optical_frame: true",
            ]
        else:
            lines += [
                "      intensity: {field: intensity, scale: 1.0, default: 0.5}",
            ]
        lines += [
            "      ground_lift: 0.0          # see docs/ROS2OPV2V.md",
            "      min_points: 1",
            "    object:",
            "      emit: true                # this agent is a labelled box for the others",
            "      extent: null              # REQUIRED if emit: [dx, dy, dz] HALF-dimensions",
            "      center: [0.0, 0.0, 0.0]   # box centre offset in the base frame",
            "",
        ]

    with open(path, "w") as handle:
        handle.write("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bag", required=True, help="rosbag2 directory or .mcap/.db3 file")
    parser.add_argument("--emit-config", default=None,
                        help="write a converter config skeleton to this path")
    parser.add_argument("--sample", type=int, default=200,
                        help="how many /tf messages to sample for the frame tree")
    parser.add_argument("--stamp-source", default="header", choices=["header", "log"])
    args = parser.parse_args()

    reader = BagReader(args.bag, args.stamp_source)
    print(f"bag      : {args.bag}")
    print(f"storage  : {reader.storage}  ({len(reader.files)} file(s))")
    start, end = reader.time_range()
    print(f"log span : {(end - start) / NS:.2f} s\n")

    stats = summarize(reader, args.sample)
    details = sample_details(reader, stats, args.sample)

    print(f"{'topic':<52} {'type':<38} {'msgs':>7} {'Hz':>7} {'maxgap':>8} {'skew ms':>8}")
    print("-" * 124)
    for name, entry in stats.items():
        print(f"{name:<52} {entry['type'].split('/')[-1]:<38} {entry['count']:>7} "
              f"{entry.get('rate_hz', 0):>7.2f} {entry.get('max_gap_ms', 0):>8.1f} "
              f"{entry.get('stamp_minus_log_ms', 0):>8.1f}")

    print("\nframe ids and payload layout")
    print("-" * 124)
    for name in sorted(details):
        entry = details[name]
        extras = {k: v for k, v in entry.items() if k != "frame_id"}
        print(f"  {name}\n      frame_id={entry['frame_id']!r}  {extras}")

    tree = tf_tree(reader, args.sample)
    print("\nTF tree (parent -> children, from the first "
          f"{args.sample} /tf and /tf_static messages)")
    print("-" * 124)
    if not tree:
        print("  no /tf or /tf_static in this bag")
    else:
        for parent, children in sorted(tree.items()):
            print(f"  {parent} -> {', '.join(sorted(children))}")
        roots = set(tree) - {c for children in tree.values() for c in children}
        print(f"\n  roots: {sorted(roots) if roots else '(cycle or none)'}")
        if len(roots) > 1:
            print("  NOTE: more than one root means the agents are NOT in a shared "
                  "frame.\n        Supply each agent's `align` transform in the "
                  "converter config.")

    agents = guess_agents(stats, details)
    print("\nagent grouping guess (by topic namespace)")
    print("-" * 124)
    for namespace, bucket in agents.items():
        print(f"  {namespace}: clouds={len(bucket['clouds'])} depth={len(bucket['depth'])} "
              f"poses={len(bucket['poses'])} cameras={len(bucket['cameras'])}")
        labels = {"clouds": "cloud", "depth": "depth", "poses": "pose",
                  "cameras": "camera"}
        for key, label in labels.items():
            for topic in bucket[key]:
                print(f"      {label:<7} {topic}")

    if args.emit_config:
        emit_config(args.emit_config, args.bag, stats, details, agents)
        print(f"\nwrote config skeleton: {args.emit_config}")
        print("Fill in every `null` before running scripts/convert_rosbag.py.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
