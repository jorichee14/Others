#!/usr/bin/env python3
"""Extract light-weight topics from a rosbag2 into per-topic Parquet tables.

Reads both rosbag2 storage formats:

* **MCAP** (``.mcap``) -- no ROS installation needed: message definitions come
  from the schemas embedded in the file and are decoded with mcap-ros2-support.
* **SQLite3** (``.db3``) -- the rosbag2 default on Humble. The ``.db3`` stores
  only the *name* of each message type, not its definition, so decoding needs
  the message packages importable: source your ROS workspace first (including
  ``comms_msgs``) or the affected topics are skipped with a warning.

Either storage may be given as the bag directory or as a single file, and a
split recording (``*_0.db3``, ``*_1.db3``, ...) is read in name order.

Prefer MCAP when recording: it is self-describing, so an extraction never
depends on having the right packages sourced, and the ``.db3`` schema-free
format is why a bag recorded today can become undecodable later.

Usage
-----
    python extract_bag.py BAG.mcap --out extracts/            # light preset + stamp audit
    python extract_bag.py bagdir/  --out extracts/            # .mcap or .db3, auto-detected
    python extract_bag.py BAG.mcap --out extracts/ --topics /infra_1/ntp/status
    python extract_bag.py BAG.db3  --out extracts/ --no-audit

Outputs
-------
    <out>/<topic>.parquet      one flat table per extracted topic; every row carries
                               log_time_ns, publish_time_ns and (when the message has a
                               std_msgs/Header) header_stamp_ns and stamp_minus_log_ms
    <out>/stamp_audit.parquet  one row per message of EVERY topic whose type starts with a
                               std_msgs/Header (header parsed from raw CDR, no full decode):
                               topic, node, log_time_ns, publish_time_ns, header_stamp_ns
    <out>/metadata.json        topics, types, counts, extracted files, storage format

Note on SQLite3 bags: the ``messages`` table records one timestamp per message,
the receive time, and has no separate publish time. ``publish_time_ns`` is set
equal to ``log_time_ns`` there, and ``metadata.json`` records
``"publish_time_is_log_time": true`` so an analysis can tell the two cases
apart rather than reading a fabricated zero-latency figure as real.
"""
from __future__ import annotations

import argparse
import array
import json
import math
import re
import struct
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
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


def field_names(obj) -> list[str]:
    """Public field names of a decoded message, whichever decoder produced it.

    mcap-ros2's dynamic classes carry plain names in ``__slots__``; rosidl's
    generated classes carry underscore-prefixed ones ('_header') with public
    properties over them. Both are normalised to the bare field name so the
    two storage formats yield identical Parquet columns.
    """
    fields_and_types = getattr(obj, "get_fields_and_field_types", None)
    if callable(fields_and_types):
        return list(fields_and_types().keys())
    slots = getattr(obj, "__slots__", None)
    if slots:
        return [s[1:] if s.startswith("_") else s for s in slots]
    return [k for k in vars(obj) if not k.startswith("_")]


def _is_message(val) -> bool:
    if isinstance(val, (str, bytes, bytearray, array.array, np.ndarray)):
        return False
    return hasattr(val, "__slots__") or hasattr(val, "__dict__")


def flatten(obj, prefix: str = "", out: dict | None = None) -> dict:
    """Flatten a decoded ROS message into {dotted.field: value}.

    Nested messages recurse; arrays of primitives stay lists; arrays of messages
    are stored as JSON strings (they are rare in the topics we care about)."""
    if out is None:
        out = {}
    for name in field_names(obj):
        val = getattr(obj, name)
        key = f"{prefix}{name}"
        if _is_message(val):
            flatten(val, key + ".", out)
        elif isinstance(val, (bytes, bytearray)):
            out[key] = bytes(val)
        elif isinstance(val, (list, tuple, array.array, np.ndarray)):
            seq = list(val)
            if seq and _is_message(seq[0]):
                out[key] = json.dumps([flatten(v) for v in seq], default=str)
            else:
                out[key] = seq
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


def first_field_is_header_class(cls) -> bool:
    """Same test for a rosidl-generated class, which has no definition text."""
    fields_and_types = getattr(cls, "get_fields_and_field_types", None)
    if not callable(fields_and_types):
        return False
    items = list(fields_and_types().items())
    return bool(items) and items[0][1] in ("std_msgs/Header", "std_msgs/msg/Header")


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


# --------------------------------------------------------------------- storage

MCAP_SUFFIXES = (".mcap",)
DB3_SUFFIXES = (".db3",)


def resolve_bag(bag: Path | str) -> tuple[str, list[Path]]:
    """Locate the storage files of `bag` and say which format they are.

    Accepts a rosbag2 directory or a single file. A split recording is returned
    in name order, which for rosbag2 is chronological.
    """
    p = Path(bag)
    if p.is_dir():
        mcaps = sorted(p.glob("*.mcap"))
        db3s = sorted(p.glob("*.db3"))
        if mcaps and db3s:
            # A converted bag can hold both; the self-describing one wins.
            print(f"note: {p} holds both .mcap and .db3; reading the .mcap", file=sys.stderr)
            return "mcap", mcaps
        if mcaps:
            return "mcap", mcaps
        if db3s:
            return "db3", db3s
        raise SystemExit(f"no .mcap or .db3 files in {p}")
    if not p.exists():
        raise SystemExit(f"no such bag: {p}")
    if p.suffix in MCAP_SUFFIXES:
        return "mcap", [p]
    if p.suffix in DB3_SUFFIXES:
        return "db3", [p]
    raise SystemExit(f"unrecognised bag file {p} (expected .mcap or .db3)")


class _McapBag:
    """Read a (possibly split) MCAP recording."""

    publish_time_is_log_time = False

    def __init__(self, files: list[Path]) -> None:
        self.files = files
        self.topic_type: dict[str, str] = {}
        self.topic_count: dict[str, int] = defaultdict(int)
        self.undecodable: set[str] = set()
        self._schema_text: dict[str, str] = {}
        for path in files:
            with open(path, "rb") as fh:
                summary = make_reader(fh, decoder_factories=[DecoderFactory()]).get_summary()
                if summary is None:
                    raise RuntimeError(f"{path}: MCAP has no summary section; cannot list topics")
                counts = dict(summary.statistics.channel_message_counts) if summary.statistics else {}
                for ch in summary.channels.values():
                    schema = summary.schemas[ch.schema_id]
                    self.topic_type[ch.topic] = schema.name
                    self.topic_count[ch.topic] += counts.get(ch.id, 0)
                    # A channel whose schema was recorded empty (the message package
                    # was not sourced when the bag was written) cannot be decoded.
                    if schema.data:
                        self._schema_text[ch.topic] = schema.data.decode("utf-8", "replace")
                    else:
                        self.undecodable.add(ch.topic)

    def header_topics(self) -> set[str]:
        return {t for t, text in self._schema_text.items() if first_field_is_header(text)}

    def iter_decoded(self, topics):
        for path in self.files:
            with open(path, "rb") as fh:
                reader = make_reader(fh, decoder_factories=[DecoderFactory()])
                for _schema, channel, message, ros_msg in reader.iter_decoded_messages(
                    topics=topics, log_time_order=True
                ):
                    yield channel.topic, message.log_time, message.publish_time, ros_msg

    def iter_raw(self, topics):
        for path in self.files:
            with open(path, "rb") as fh:
                reader = make_reader(fh, decoder_factories=[DecoderFactory()])
                for schema, channel, message in reader.iter_messages(
                    topics=topics, log_time_order=True
                ):
                    yield channel.topic, schema.name, message.log_time, message.publish_time, message.data


class _Db3Bag:
    """Read a (possibly split) SQLite3 recording.

    The ``.db3`` format stores the type *name* only, so every type has to be
    importable here. That is the one real cost of the format and the reason a
    missing ``source install/setup.bash`` shows up as silently empty output
    rather than as a decode error.
    """

    publish_time_is_log_time = True

    def __init__(self, files: list[Path]) -> None:
        import sqlite3

        self._sqlite3 = sqlite3
        try:
            from rclpy.serialization import deserialize_message
            from rosidl_runtime_py.utilities import get_message
        except ImportError as exc:  # pragma: no cover - depends on the host
            raise SystemExit(
                "reading a .db3 bag needs the ROS 2 Python packages (rclpy, "
                f"rosidl_runtime_py): {exc}\n"
                "Source your ROS setup first, or re-record with "
                "'ros2 bag record --storage mcap', which needs no ROS to read."
            ) from exc
        self._deserialize = deserialize_message
        self.files = files
        self.topic_type: dict[str, str] = {}
        self.topic_count: dict[str, int] = defaultdict(int)
        self.undecodable: set[str] = set()
        self._cls: dict[str, type] = {}
        self._unimportable: dict[str, str] = {}

        for path in files:
            con = self._connect(path)
            try:
                for _tid, name, type_name in con.execute("SELECT id, name, type FROM topics"):
                    self.topic_type[name] = type_name
                for name, n in con.execute(
                    "SELECT t.name, COUNT(*) FROM messages m JOIN topics t ON t.id = m.topic_id "
                    "GROUP BY t.name"
                ):
                    self.topic_count[name] += n
            finally:
                con.close()

        for topic, type_name in self.topic_type.items():
            if type_name in self._cls:
                continue
            try:
                self._cls[type_name] = get_message(type_name)
            except (ImportError, AttributeError, ValueError, KeyError) as exc:
                self._unimportable[type_name] = str(exc)
        for topic, type_name in self.topic_type.items():
            if type_name not in self._cls:
                self.undecodable.add(topic)
        if self._unimportable:
            names = ", ".join(sorted(self._unimportable))
            print(
                f"warning: message types not importable, their topics are skipped: {names}\n"
                "         source the workspace that defines them and re-run.",
                file=sys.stderr,
            )

    def _connect(self, path: Path):
        # immutable=1: open read-only without needing to create -wal/-shm files,
        # so an extraction never modifies the recording it is reading.
        return self._sqlite3.connect(f"file:{path}?immutable=1", uri=True)

    def _iter_rows(self, topics):
        wanted = sorted(set(topics))
        for path in self.files:
            con = self._connect(path)
            try:
                id_to_topic = {
                    tid: name for tid, name in con.execute("SELECT id, name FROM topics")
                }
                ids = [tid for tid, name in id_to_topic.items() if name in wanted]
                if not ids:
                    continue
                placeholders = ",".join("?" * len(ids))
                rows = con.execute(
                    "SELECT topic_id, timestamp, data FROM messages "
                    f"WHERE topic_id IN ({placeholders}) ORDER BY timestamp",
                    ids,
                )
                for topic_id, timestamp, blob in rows:
                    yield id_to_topic[topic_id], timestamp, bytes(blob)
            finally:
                con.close()

    def header_topics(self) -> set[str]:
        return {
            topic
            for topic, type_name in self.topic_type.items()
            if type_name in self._cls and first_field_is_header_class(self._cls[type_name])
        }

    def iter_decoded(self, topics):
        for topic, timestamp, blob in self._iter_rows(topics):
            cls = self._cls.get(self.topic_type[topic])
            if cls is None:
                continue
            yield topic, timestamp, timestamp, self._deserialize(blob, cls)

    def iter_raw(self, topics):
        for topic, timestamp, blob in self._iter_rows(topics):
            yield topic, self.topic_type[topic], timestamp, timestamp, blob


def open_bag(bag: Path | str):
    """Open `bag` in whichever storage format it uses."""
    kind, files = resolve_bag(bag)
    return (_McapBag if kind == "mcap" else _Db3Bag)(files), kind, files


# --------------------------------------------------------------------- extract


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
        bag: rosbag2 directory, or a single .mcap / .db3 file
        out: output directory (created if missing)
        topics: explicit topic list; None = every topic except the heavy types
        include_heavy: also decode Image / PointCloud2 / LaserScan / CameraInfo / Path / PacketMsg
        audit: also write stamp_audit.parquet (header stamp vs log time for every header-bearing topic)
        audit_max_per_topic: cap audit rows per topic (0 = all)
    Output:
        metadata dict (also written to <out>/metadata.json)
    """
    bag = Path(bag)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)

    reader, kind, files = open_bag(bag)
    topic_type = reader.topic_type
    topic_count = reader.topic_count

    if topics is not None:
        wanted = [t for t in topics if t in topic_type]
        missing = sorted(set(topics) - set(wanted))
        if missing:
            print(f"warning: topics not in bag: {missing}", file=sys.stderr)
    else:
        wanted = [t for t, ty in topic_type.items() if include_heavy or ty not in HEAVY_TYPES]

    skipped = sorted(t for t in wanted if t in reader.undecodable)
    if skipped:
        print(f"warning: not decodable, skipping: {skipped}", file=sys.stderr)
    wanted = sorted(set(wanted) - reader.undecodable)

    n_files = f" ({len(files)} files)" if len(files) > 1 else ""
    print(f"bag: {bag} [{kind}]{n_files}")
    print(f"topics in bag: {len(topic_type)}; extracting {len(wanted)}")

    # ---- full decode of the wanted topics --------------------------------
    rows_by_topic: dict[str, list[dict]] = defaultdict(list)
    if wanted:
        for topic, log_time, publish_time, ros_msg in reader.iter_decoded(wanted):
            row = flatten(ros_msg)
            row["log_time_ns"] = log_time
            row["publish_time_ns"] = publish_time
            if "header.stamp.sec" in row:
                row["header_stamp_ns"] = (
                    int(row["header.stamp.sec"]) * 1_000_000_000 + int(row["header.stamp.nanosec"])
                )
                row["stamp_minus_log_ms"] = (row["header_stamp_ns"] - log_time) / 1e6
            rows_by_topic[topic].append(row)

    written = {}
    for topic, rows in rows_by_topic.items():
        df = pd.DataFrame(rows)
        path = out / f"{topic_to_filename(topic)}.parquet"
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
    if audit:
        header_topics = sorted(reader.header_topics())
        audit_rows = []
        per_topic = defaultdict(int)
        if header_topics:
            for topic, type_name, log_time, publish_time, data in reader.iter_raw(header_topics):
                per_topic[topic] += 1
                if audit_max_per_topic and per_topic[topic] > audit_max_per_topic:
                    continue
                audit_rows.append(
                    {
                        "topic": topic,
                        "node": node_of(topic),
                        "type": type_name,
                        "log_time_ns": log_time,
                        "publish_time_ns": publish_time,
                        "header_stamp_ns": cdr_header_stamp_ns(data),
                    }
                )
        audit_df = pd.DataFrame(audit_rows)
        if len(audit_df):
            audit_df["stamp_minus_log_ms"] = (
                audit_df["header_stamp_ns"] - audit_df["log_time_ns"]
            ) / 1e6
            audit_df["stamp_minus_pub_ms"] = (
                audit_df["header_stamp_ns"] - audit_df["publish_time_ns"]
            ) / 1e6
        audit_file = out / "stamp_audit.parquet"
        audit_df.to_parquet(audit_file, index=False)
        print(
            f"  stamp audit: {len(audit_df)} rows over {len(header_topics)} "
            f"header-bearing topics -> {audit_file.name}"
        )

    meta = {
        "bag": str(bag),
        "storage": kind,
        "files": [str(f) for f in files],
        # SQLite3 keeps one timestamp per message, so the two are the same there;
        # say so rather than let an analysis read a zero difference as measured.
        "publish_time_is_log_time": reader.publish_time_is_log_time,
        "topics": {t: {"type": ty, "count": topic_count[t]} for t, ty in sorted(topic_type.items())},
        "extracted": written,
        "stamp_audit": audit_file.name if audit_file else None,
    }
    (out / "metadata.json").write_text(json.dumps(meta, indent=2))
    return meta


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bag", type=Path, help="rosbag2 directory, or a .mcap / .db3 file")
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
