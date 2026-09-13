#!/usr/bin/env python3
"""
ntp_bag_postprocess.py
──────────────────────
Post-processing utility: reads NtpStatus topics from a rosbag2 bag and
produces corrected timestamps for every sensor message on a given topic.

Usage
  python3 ntp_bag_postprocess.py <bag_dir> <sensor_topic> [ntp_topic]

Example
  python3 ntp_bag_postprocess.py ./my_bag /lidar/points /ntp/client/status

Requirements
  pip install rosbags numpy

The script outputs a CSV:
  original_ns, corrected_ns, offset_s, delta_us, stepped
one row per sensor message.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

try:
    from rosbags.rosbag2 import Reader
    from rosbags.typesys import get_typestore, Stores
except ImportError:
    sys.exit("Install rosbags:  pip install rosbags")


# helpers

def stamp_to_ns(stamp) -> int:
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


# NTP sample loading

def load_ntp_samples(
    bag_path: str,
    topic: str,
    typestore,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load NTP samples from bag.

    Returns four parallel numpy arrays:
      timestamps_ns   ROS wall-clock stamp (int64)
      offsets_s       clock offset in seconds (float64)
      monos           monotonic_seconds at publish time (float64)
      stepped         clock_stepped flag (bool)
    """
    timestamps, offsets, monos, stepped_flags = [], [], [], []

    with Reader(bag_path) as reader:
        conns = [c for c in reader.connections if c.topic == topic]
        if not conns:
            sys.exit(f"Topic {topic!r} not found in bag. "
                     f"Available: {[c.topic for c in reader.connections]}")

        for conn, _ts, raw in reader.messages(connections=conns):
            msg = typestore.deserialize_cdr(raw, conn.msgtype)
            timestamps.append(stamp_to_ns(msg.header.stamp))
            offsets.append(float(msg.offset_seconds))
            monos.append(float(msg.monotonic_seconds))
            stepped_flags.append(bool(msg.clock_stepped))

    if not timestamps:
        sys.exit(f"No messages found on {topic!r}")

    return (
        np.array(timestamps,    dtype=np.int64),
        np.array(offsets,       dtype=np.float64),
        np.array(monos,         dtype=np.float64),
        np.array(stepped_flags, dtype=bool),
    )


# segment on step boundaries

def segment_samples(
    times_ns:  np.ndarray,
    offsets_s: np.ndarray,
    stepped:   np.ndarray,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Split NTP sample arrays at every clock-step boundary.

    Returns a list of (times_ns, offsets_s) segment pairs. Interpolation
    must stay within a single segment — crossing a step boundary is invalid.
    """
    segments = []
    start = 0
    for i, flag in enumerate(stepped):
        if flag and i > start:
            segments.append((times_ns[start:i], offsets_s[start:i]))
            start = i
    segments.append((times_ns[start:], offsets_s[start:]))
    return [s for s in segments if len(s[0]) >= 2]


# interpolation

def interpolate_offset(
    sensor_ts_ns: int,
    segments: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[float, bool]:
    """
    Return (offset_seconds, valid) for a given sensor timestamp.

    Searches each segment for one that brackets the timestamp and
    interpolates linearly within it.

    Returns (0.0, False) if the timestamp falls in a step gap or
    outside all segment windows — caller should flag that message.
    """
    for times_ns, offsets_s in segments:
        if sensor_ts_ns < times_ns[0] or sensor_ts_ns > times_ns[-1]:
            continue
        offset = float(np.interp(sensor_ts_ns, times_ns, offsets_s))
        return offset, True

    return 0.0, False


# sensor topic iteration

def iter_sensor_stamps(
    bag_path: str,
    topic: str,
    typestore,
):
    """Yield (original_ns,) for each message on the sensor topic."""
    with Reader(bag_path) as reader:
        conns = [c for c in reader.connections if c.topic == topic]
        if not conns:
            sys.exit(f"Sensor topic {topic!r} not found in bag.")
        for conn, _ts, raw in reader.messages(connections=conns):
            msg = typestore.deserialize_cdr(raw, conn.msgtype)
            yield stamp_to_ns(msg.header.stamp)


# main

def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    bag_path     = sys.argv[1]
    sensor_topic = sys.argv[2]
    ntp_topic    = sys.argv[3] if len(sys.argv) > 3 else "/ntp/client/status"

    typestore = get_typestore(Stores.ROS2_HUMBLE)

    print(f"Loading NTP samples from {ntp_topic!r} ...")
    times_ns, offsets_s, monos, stepped = load_ntp_samples(
        bag_path, ntp_topic, typestore
    )
    print(f"  {len(times_ns)} samples, "
          f"{stepped.sum()} step events, "
          f"offset range [{offsets_s.min()*1e3:.3f}, {offsets_s.max()*1e3:.3f}] ms")

    segments = segment_samples(times_ns, offsets_s, stepped)
    print(f"  {len(segments)} continuous segment(s) after step splitting")

    out_path = Path(bag_path).stem + "_corrected.csv"
    print(f"Processing {sensor_topic!r} -> {out_path} ...")

    total = 0
    invalid = 0

    with open(out_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "original_ns", "corrected_ns",
            "offset_s", "delta_us", "valid",
        ])

        for orig_ns in iter_sensor_stamps(bag_path, sensor_topic, typestore):
            offset_s, valid = interpolate_offset(orig_ns, segments)

            if valid:
                corrected_ns = orig_ns - int(offset_s * 1_000_000_000)
                delta_us     = (corrected_ns - orig_ns) / 1_000.0
            else:
                corrected_ns = orig_ns
                delta_us     = 0.0
                invalid     += 1

            writer.writerow([orig_ns, corrected_ns, f"{offset_s:.9f}",
                             f"{delta_us:.3f}", valid])
            total += 1

    print(f"Done. {total} messages processed, {invalid} flagged invalid ")
    print(f"      (invalid = timestamp in step gap or outside NTP window)")
    print(f"Output: {out_path}")


if __name__ == "__main__":
    main()
