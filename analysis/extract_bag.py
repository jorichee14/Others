#!/usr/bin/env python3
"""Extract light-weight topics from a rosbag2 MCAP file into per-topic Parquet tables.

No ROS installation is needed: message definitions are read from the schemas
embedded in the MCAP file and decoded with mcap-ros2-support.

Usage
-----
    python extract_bag.py BAG.mcap --out extracts/            # light preset + stamp audit
    python extract_bag.py BAG.mcap --out extracts/ --topics /infra_1/ntp/status /mobile_2/ntp/status
    python extract_bag.py BAG.mcap --out extracts/ --no-audit

Outputs
-------
    <out>/<topic>.parquet      one flat table per extracted topic; every row carries
                               log_time_ns, publish_time_ns and (when the message has a
                               std_msgs/Header) header_stamp_ns and stamp_minus_log_ms
    <out>/stamp_audit.parquet  one row per message of EVERY topic whose type starts with a
                               std_msgs/Header (header parsed from raw CDR, no full decode):
                               topic, node, log_time_ns, publish_time_ns, header_stamp_ns
    <out>/metadata.json        topics, types, counts, extracted files
"""
from __future__ import annotations

import argparse
import json
import math
import re
import struct
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory

# Types that are large and not needed for the NTP / Wi-Fi / CSI / geometry analysis.
HEAVY_TYPES = {
    "sensor_msgs/msg/Image",
    "sensor_msgs/msg/PointCloud2",
    "sensor_msgs/msg/CameraInfo",
    "sensor_msgs/msg/LaserScan",
    "nav_msgs/msg/Path",
    "ouster_sensor_msgs/msg/PacketMsg",
}

NODE_RE = re.compile(r"^/([A-Za-z0-9_]+)/")


def node_of(topic: str) -> str:
    """'/mobile_1/ntp/status' -> 'mobile_1'; '/tf' -> 'tf'."""
    m = NODE_RE.match(topic)
    return m.group(1) if m else topic.strip("/")


def topic_to_filename(topic: str) -> str:
    return topic.strip("/").replace("/", "__")


def flatten(obj, prefix: str = "", out: dict | None = None) -> dict:
    """Flatten a decoded ROS message into {dotted.field: value}.

    Nested messages recurse; arrays of primitives stay lists; arrays of messages
    are stored as JSON strings (they are rare in the topics we care about)."""
    if out is None:
        out = {}
    fields = getattr(obj, "__slots__", None) or [k for k in vars(obj) if not k.startswith("_")]
    for name in fields:
        val = getattr(obj, name)
        key = f"{prefix}{name}"
        if hasattr(val, "__slots__") or (hasattr(val, "__dict__") and not isinstance(val, (str, bytes))):
            flatten(val, key + ".", out)
        elif isinstance(val, (list, tuple)):
            if val and (hasattr(val[0], "__slots__") or hasattr(val[0], "__dict__")):
                out[key] = json.dumps([flatten(v) for v in val], default=str)
            else:
                out[key] = list(val)
        elif isinstance(val, bytes):
            out[key] = val
        else:
            out[key] = val
    return out


def first_field_is_header(schema_text: str) -> bool:
    """True if the top-level definition's first field is a std_msgs/Header."""
    top = schema_text.split("\n===")[0]
    for line in top.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        return s.startswith("std_msgs/Header ") or s.startswith("std_msgs/msg/Header ")
    return False


def cdr_header_stamp_ns(data: bytes) -> int | None:
    """Parse header.stamp from raw CDR bytes when the header is the first field.

    CDR encapsulation is 4 bytes (0x00 0x01 = little-endian CDR), then int32 sec,
    uint32 nanosec."""
    if len(data) < 12:
        return None
    endian = "<" if data[1] == 0x01 else ">"
    sec, nsec = struct.unpack(endian + "iI", data[4:12])
    if sec < 0 or nsec >= 1_000_000_000:
        return None
    return sec * 1_000_000_000 + nsec


def extract(
    bag: Path | str,
    out: Path | str,
    topics: list[str] | None = None,
    include_heavy: bool = False,
    audit: bool = True,
    audit_max_per_topic: int = 0,
) -> dict:
    """Extract `bag` into per-topic Parquet files under `out`. Returns the metadata dict.

    Input:
        bag: path to the rosbag2 MCAP file
        out: output directory (created if missing)
        topics: explicit topic list; None = every topic except the heavy types
        include_heavy: also decode Image / PointCloud2 / LaserScan / CameraInfo / Path / PacketMsg
        audit: also write stamp_audit.parquet (header stamp vs log time for every header-bearing topic)
        audit_max_per_topic: cap audit rows per topic (0 = all)
    Output:
        metadata dict (also written to <out>/metadata.json)
    """
    args = argparse.Namespace(
        bag=Path(bag), out=Path(out), topics=topics, include_heavy=include_heavy,
        no_audit=not audit, audit_max_per_topic=audit_max_per_topic,
    )
    args.out.mkdir(parents=True, exist_ok=True)

    with open(args.bag, "rb") as f:
        reader = make_reader(f, decoder_factories=[DecoderFactory()])
        summary = reader.get_summary()
        if summary is None:
            raise RuntimeError("MCAP has no summary section; cannot list topics")
        channels = summary.channels
        schemas = summary.schemas
        stats = summary.statistics
        counts = dict(stats.channel_message_counts) if stats else {}

        topic_type = {ch.topic: schemas[ch.schema_id].name for ch in channels.values()}
        topic_count = defaultdict(int)
        for ch in channels.values():
            topic_count[ch.topic] += counts.get(ch.id, 0)

        if args.topics is not None:
            wanted = [t for t in args.topics if t in topic_type]
            missing = sorted(set(args.topics) - set(wanted))
            if missing:
                print(f"warning: topics not in bag: {missing}", file=sys.stderr)
        else:
            wanted = [t for t, ty in topic_type.items() if args.include_heavy or ty not in HEAVY_TYPES]
        wanted = sorted(set(wanted))

        print(f"bag: {args.bag}")
        print(f"topics in bag: {len(topic_type)}; extracting {len(wanted)}")

        # ---- full decode of the wanted topics --------------------------------
        rows_by_topic: dict[str, list[dict]] = defaultdict(list)
        for schema, channel, message, ros_msg in reader.iter_decoded_messages(topics=wanted, log_time_order=True):
            row = flatten(ros_msg)
            row["log_time_ns"] = message.log_time
            row["publish_time_ns"] = message.publish_time
            if "header.stamp.sec" in row:
                row["header_stamp_ns"] = int(row["header.stamp.sec"]) * 1_000_000_000 + int(row["header.stamp.nanosec"])
                row["stamp_minus_log_ms"] = (row["header_stamp_ns"] - message.log_time) / 1e6
            rows_by_topic[channel.topic].append(row)

        written = {}
        for topic, rows in rows_by_topic.items():
            df = pd.DataFrame(rows)
            path = args.out / f"{topic_to_filename(topic)}.parquet"
            try:
                df.to_parquet(path, index=False)
            except (pa.ArrowInvalid, pa.ArrowTypeError, ValueError):
                # mixed-type columns (e.g. bytes / objects) -> stringify the offenders
                for col in df.columns:
                    if df[col].dtype == object:
                        try:
                            pa.array(df[col])
                        except (pa.ArrowInvalid, pa.ArrowTypeError):
                            df[col] = df[col].astype(str)
                df.to_parquet(path, index=False)
            written[topic] = {"file": path.name, "rows": len(df), "type": topic_type[topic]}
            print(f"  {topic:55s} {len(df):8d} rows -> {path.name}")

        # ---- cheap header-stamp audit over every header-bearing topic ---------
        audit_file = None
        if not args.no_audit:
            header_topics = {
                ch.topic for ch in channels.values() if first_field_is_header(schemas[ch.schema_id].data.decode("utf-8", "replace"))
            }
            audit_rows = []
            per_topic = defaultdict(int)
            for schema, channel, message in reader.iter_messages(topics=sorted(header_topics), log_time_order=True):
                per_topic[channel.topic] += 1
                if args.audit_max_per_topic and per_topic[channel.topic] > args.audit_max_per_topic:
                    continue
                audit_rows.append(
                    {
                        "topic": channel.topic,
                        "node": node_of(channel.topic),
                        "type": schema.name,
                        "log_time_ns": message.log_time,
                        "publish_time_ns": message.publish_time,
                        "header_stamp_ns": cdr_header_stamp_ns(message.data),
                    }
                )
            audit = pd.DataFrame(audit_rows)
            if len(audit):
                audit["stamp_minus_log_ms"] = (audit["header_stamp_ns"] - audit["log_time_ns"]) / 1e6
                audit["stamp_minus_pub_ms"] = (audit["header_stamp_ns"] - audit["publish_time_ns"]) / 1e6
            audit_file = args.out / "stamp_audit.parquet"
            audit.to_parquet(audit_file, index=False)
            print(f"  stamp audit: {len(audit)} rows over {len(header_topics)} header-bearing topics -> {audit_file.name}")

    meta = {
        "bag": str(args.bag),
        "topics": {t: {"type": ty, "count": topic_count[t]} for t, ty in sorted(topic_type.items())},
        "extracted": written,
        "stamp_audit": audit_file.name if audit_file else None,
    }
    (args.out / "metadata.json").write_text(json.dumps(meta, indent=2))
    return meta


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bag", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--topics", nargs="*", default=None, help="explicit topic list (default: light preset)")
    ap.add_argument("--include-heavy", action="store_true", help="also decode Image/PointCloud2/... topics")
    ap.add_argument("--no-audit", action="store_true", help="skip the header-stamp audit over all topics")
    ap.add_argument("--audit-max-per-topic", type=int, default=0, help="subsample audit rows per topic (0 = all)")
    args = ap.parse_args()
    extract(args.bag, args.out, args.topics, args.include_heavy, not args.no_audit, args.audit_max_per_topic)
    return 0


if __name__ == "__main__":
    sys.exit(main())
