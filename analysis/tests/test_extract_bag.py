#!/usr/bin/env python3
"""Checks that a .db3 recording extracts to the same tables as a .mcap one.

The robots record SQLite3 by default on Humble, so the analysis has to read it;
the point of these checks is that WHICH storage was used never changes the
Parquet an analysis sees. The two decoders name fields differently
(mcap-ros2 uses plain names, rosidl underscore-prefixed ones with properties
over them), which is exactly the kind of difference that would silently rename
every column. Run: python analysis/tests/test_extract_bag.py
"""
from __future__ import annotations

import sqlite3
import struct
import sys
import tempfile
import types
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

FAILED = []

TYPE = "test_msgs/msg/Sample"
DEF = """std_msgs/Header header
float64 value
int32 seq

================================================================================
MSG: std_msgs/Header
builtin_interfaces/Time stamp
string frame_id
"""

# (sec, nanosec, frame_id, value, seq) -- the rows both bags carry.
SAMPLES = [
    (1790082417, 187269014, "wlan0", -53.0, 11),
    (1790082418, 287269014, "wlan0", -51.5, 12),
    (1790082419, 387269014, "wlan0", -49.25, 13),
]


def check(name, got, want):
    ok = got == want
    print(f"  {'ok  ' if ok else 'FAIL'}  {name:46s} got {got!r}  want {want!r}")
    if not ok:
        FAILED.append(name)


# ----------------------------------------------------------------- CDR codec
# Hand-rolled for this one layout so the .db3 blobs are real CDR: the stamp
# audit parses them without any decoder, so a stub that skipped encoding would
# test nothing.

def _pad(buf: bytearray, align: int) -> None:
    while (len(buf) - 4) % align:      # alignment is measured from the body
        buf.append(0)


def encode(sec, nsec, frame_id, value, seq) -> bytes:
    buf = bytearray(b"\x00\x01\x00\x00")           # little-endian CDR
    buf += struct.pack("<i", sec)
    buf += struct.pack("<I", nsec)
    raw = frame_id.encode() + b"\x00"
    buf += struct.pack("<I", len(raw)) + raw
    _pad(buf, 8)
    buf += struct.pack("<d", value)
    _pad(buf, 4)
    buf += struct.pack("<i", seq)
    return bytes(buf)


def decode(data: bytes):
    sec, nsec, n = struct.unpack_from("<iII", data, 4)
    off = 16 + n
    off += (-(off - 4)) % 8
    (value,) = struct.unpack_from("<d", data, off)
    off += 8
    off += (-(off - 4)) % 4
    (seq,) = struct.unpack_from("<i", data, off)
    return sec, nsec, data[16:16 + n - 1].decode(), value, seq


# ------------------------------------------------------- fake rosidl / rclpy
# extract_bag imports these lazily for .db3 only. Standing them up here lets the
# SQLite path run end to end without a ROS install, and the stand-ins mimic the
# part that matters: rosidl messages expose get_fields_and_field_types() and
# keep their storage in underscore-prefixed slots.

class _Time:
    __slots__ = ["_sec", "_nanosec"]

    @classmethod
    def get_fields_and_field_types(cls):
        return {"sec": "int32", "nanosec": "uint32"}

    def __init__(self, sec, nanosec):
        self._sec, self._nanosec = sec, nanosec

    sec = property(lambda self: self._sec)
    nanosec = property(lambda self: self._nanosec)


class _Header:
    __slots__ = ["_stamp", "_frame_id"]

    @classmethod
    def get_fields_and_field_types(cls):
        return {"stamp": "builtin_interfaces/Time", "frame_id": "string"}

    def __init__(self, stamp, frame_id):
        self._stamp, self._frame_id = stamp, frame_id

    stamp = property(lambda self: self._stamp)
    frame_id = property(lambda self: self._frame_id)


class _Sample:
    __slots__ = ["_header", "_value", "_seq"]

    @classmethod
    def get_fields_and_field_types(cls):
        return {"header": "std_msgs/Header", "value": "float64", "seq": "int32"}

    def __init__(self, header, value, seq):
        self._header, self._value, self._seq = header, value, seq

    header = property(lambda self: self._header)
    value = property(lambda self: self._value)
    seq = property(lambda self: self._seq)


def install_ros_stubs():
    rclpy = types.ModuleType("rclpy")
    serialization = types.ModuleType("rclpy.serialization")

    def deserialize_message(blob, cls):
        sec, nsec, frame_id, value, seq = decode(blob)
        return cls(_Header(_Time(sec, nsec), frame_id), value, seq)

    serialization.deserialize_message = deserialize_message
    rclpy.serialization = serialization

    utilities = types.ModuleType("rosidl_runtime_py.utilities")
    utilities.get_message = lambda name: {TYPE: _Sample}[name]
    runtime = types.ModuleType("rosidl_runtime_py")
    runtime.utilities = utilities

    sys.modules.update({
        "rclpy": rclpy,
        "rclpy.serialization": serialization,
        "rosidl_runtime_py": runtime,
        "rosidl_runtime_py.utilities": utilities,
    })


# ------------------------------------------------------------------- writers

def write_db3(path: Path, topic: str = "/infra_1/csi") -> None:
    con = sqlite3.connect(path)
    con.executescript(
        "CREATE TABLE topics (id INTEGER PRIMARY KEY, name TEXT, type TEXT,"
        " serialization_format TEXT, offered_qos_profiles TEXT);"
        "CREATE TABLE messages (id INTEGER PRIMARY KEY, topic_id INTEGER,"
        " timestamp INTEGER, data BLOB);"
    )
    con.execute("INSERT INTO topics VALUES (1, ?, ?, 'cdr', '')", (topic, TYPE))
    # A second topic whose type is not importable: it must be reported and
    # skipped, not crash the extraction.
    con.execute("INSERT INTO topics VALUES (2, '/ghost/csi', 'nope_msgs/msg/Ghost', 'cdr', '')")
    for i, s in enumerate(SAMPLES):
        # deliberately inserted out of order: the reader must sort by timestamp
        con.execute(
            "INSERT INTO messages (topic_id, timestamp, data) VALUES (1, ?, ?)",
            (s[0] * 1_000_000_000 + s[1] + 1_000_000, encode(*s)),
        )
    con.commit()
    con.close()


def write_mcap(path: Path, topic: str = "/infra_1/csi") -> None:
    from mcap_ros2.writer import Writer

    with open(path, "wb") as f:
        w = Writer(f)
        schema = w.register_msgdef(TYPE.replace("/msg/", "/"), DEF)
        for s in SAMPLES:
            sec, nsec, frame_id, value, seq = s
            w.write_message(
                topic=topic,
                schema=schema,
                message={
                    "header": {"stamp": {"sec": sec, "nanosec": nsec}, "frame_id": frame_id},
                    "value": value,
                    "seq": seq,
                },
                log_time=sec * 1_000_000_000 + nsec + 1_000_000,
                publish_time=sec * 1_000_000_000 + nsec,
            )
        w.finish()


# --------------------------------------------------------------------- checks

install_ros_stubs()
from extract_bag import extract, resolve_bag  # noqa: E402

tmp = Path(tempfile.mkdtemp())
print("resolve_bag")
(tmp / "bagdir").mkdir()
write_db3(tmp / "bagdir" / "bagdir_0.db3")
check("directory of .db3 -> db3", resolve_bag(tmp / "bagdir")[0], "db3")
check("single .db3 file -> db3", resolve_bag(tmp / "bagdir" / "bagdir_0.db3")[0], "db3")
write_mcap(tmp / "bagdir" / "bagdir_0.mcap")
check("mixed directory prefers mcap", resolve_bag(tmp / "bagdir")[0], "mcap")

print("\nextraction")
write_db3(tmp / "only.db3")
write_mcap(tmp / "only.mcap")
meta_db3 = extract(tmp / "only.db3", tmp / "out_db3")
meta_mcap = extract(tmp / "only.mcap", tmp / "out_mcap")

df_db3 = pd.read_parquet(tmp / "out_db3" / "infra_1__csi.parquet")
df_mcap = pd.read_parquet(tmp / "out_mcap" / "infra_1__csi.parquet")

check("db3 row count", len(df_db3), len(SAMPLES))
check("same columns as mcap", sorted(df_db3.columns), sorted(df_mcap.columns))
check("nested field flattened", "header.stamp.sec" in df_db3.columns, True)
check("no underscore-prefixed columns",
      [c for c in df_db3.columns if c.startswith("_") or "._" in c], [])
check("values match mcap", df_db3[["value", "seq"]].values.tolist(),
      df_mcap[["value", "seq"]].values.tolist())
check("sorted by timestamp", df_db3["seq"].tolist(), [s[4] for s in SAMPLES])
check("header_stamp_ns derived", df_db3["header_stamp_ns"].tolist(),
      [s[0] * 1_000_000_000 + s[1] for s in SAMPLES])
check("unimportable type skipped", "/ghost/csi" in meta_db3["extracted"], False)
check("unimportable type still listed", "/ghost/csi" in meta_db3["topics"], True)
check("storage recorded", meta_db3["storage"], "db3")
check("db3 flags publish==log", meta_db3["publish_time_is_log_time"], True)
check("mcap does not", meta_mcap["publish_time_is_log_time"], False)

print("\nstamp audit (raw CDR, no decoder)")
audit = pd.read_parquet(tmp / "out_db3" / "stamp_audit.parquet")
check("audit rows", len(audit), len(SAMPLES))
check("audit stamps parsed from CDR", sorted(audit["header_stamp_ns"].tolist()),
      sorted(s[0] * 1_000_000_000 + s[1] for s in SAMPLES))

print()
if FAILED:
    print(f"{len(FAILED)} failed: {', '.join(FAILED)}")
    sys.exit(1)
print("all checks passed")
