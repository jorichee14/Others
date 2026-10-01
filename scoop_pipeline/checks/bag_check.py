#!/usr/bin/env python3
"""Check scoop.bag against the generic decoder on a REAL bag, and time both.

    python3 checks/bag_check.py <bag> [--points /ouster/points]
                                 [--image /zed/.../image_rect_color]
                                 [--check 50] [--time 300]

1. Lists the PointCloud2 / Image topics (file index only, nothing decoded).
2. For the first --check messages of each chosen topic, decodes the payload
   both ways and asserts identical xyz, per-point time and pixels. Exits
   non-zero on any mismatch -- run this once per new sensor/driver version.
3. Times --time messages through the old per-message path (generic decode,
   bytes() copy, per-field extraction, finite + range gate, as the stage
   scripts did) and through scoop.bag.iter_scans, and prints msg/s for each.
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scoop import bag                                               # noqa: E402
from ros2opv2v.pointclouds import pointcloud2_to_array             # noqa: E402

_PF_DT = {1: np.int8, 2: np.uint8, 3: np.int16, 4: np.uint16,
          5: np.int32, 6: np.uint32, 7: np.float32, 8: np.float64}


def old_xyz(msg):
    """The stage scripts' pc2_xyz, verbatim in behaviour."""
    off = {f.name: (f.offset, f.datatype) for f in msg.fields}
    raw = np.frombuffer(bytes(msg.data), np.uint8).reshape(-1, msg.point_step)

    def col(name):
        o, dt = off[name]
        t = _PF_DT[dt]
        w = np.dtype(t).itemsize
        return raw[:, o:o + w].copy().view(t).ravel()
    return np.stack([col("x"), col("y"), col("z")], 1).astype(np.float64)


def check_points(reader, topic, n):
    bad = 0
    for i, (_, _, payload, decode) in enumerate(reader.iter_raw([topic])):
        if i >= n:
            break
        _, fast = bag.parse_pointcloud2(payload)
        ref = pointcloud2_to_array(decode(payload))
        for f in ref.dtype.names:
            a, b = np.asarray(fast[f]), np.asarray(ref[f])
            if a.shape != b.shape or not np.array_equal(a, b, equal_nan=a.dtype.kind == "f"):
                print(f"  MISMATCH msg {i} field {f!r}")
                bad += 1
    print(f"  {topic}: {min(i + 1, n)} msgs compared field by field, {bad} mismatches")
    return bad


def check_image(reader, topic, n):
    compressed = "Compressed" in reader.topics()[topic].msgtype
    bad = 0
    for i, (_, _, payload, decode) in enumerate(reader.iter_raw([topic])):
        if i >= n:
            break
        _, e1, a = bag.parse_image(payload, compressed)
        _, e2, b = bag._generic_image(decode(payload), compressed)
        if e1 != e2 or not np.array_equal(a, b):
            print(f"  MISMATCH image {i}")
            bad += 1
    print(f"  {topic}: {min(i + 1, n)} images compared, {bad} mismatches")
    return bad


def time_points(reader, topic, n, lo, hi):
    t0 = time.perf_counter()
    k = 0
    for _, _, payload, decode in reader.iter_raw([topic]):
        p = old_xyz(decode(payload))                 # the scripts' iter_scans body
        keep = np.isfinite(p).all(1)
        d = np.linalg.norm(p, axis=1)
        keep &= (d > lo) & (d < hi)
        p[keep]
        k += 1
        if k >= n:
            break
    t_old = time.perf_counter() - t0

    t0 = time.perf_counter()
    st = {}
    for s in bag.iter_scans(reader, topic, stats=st, min_range=lo, max_range=hi):
        if st["seen"] >= n:
            break
    t_new = time.perf_counter() - t0
    print(f"  {topic}: old {k / t_old:7.1f} msg/s | scoop.bag {st['seen'] / t_new:7.1f} "
          f"msg/s  ({t_old / max(t_new, 1e-9):.1f}x; fast {st['fast']}, "
          f"fallback {st['fallback']})")
    print("    (both include reading the file; with a trajectory selector, "
          "scans outside time_tol are not decoded at all)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bag")
    ap.add_argument("--points", default="")
    ap.add_argument("--image", default="")
    ap.add_argument("--check", type=int, default=50)
    ap.add_argument("--time", type=int, default=300)
    ap.add_argument("--min-range", type=float, default=0.5)
    ap.add_argument("--max-range", type=float, default=40.0)
    a = ap.parse_args()

    reader = bag.open_bag(a.bag)
    print(f"{a.bag}: {len(reader.files)} file(s), storage {reader.storage}")
    for t in sorted(reader.topics().values(), key=lambda t: t.name):
        if "PointCloud2" in t.msgtype or "Image" in t.msgtype:
            print(f"  {t.name:55s} {t.msgtype:32s} {t.count:7d}")

    points = a.points or bag.detect_points_topic(reader)
    bad = 0
    print("\n[check] fast path vs generic decoder")
    bad += check_points(reader, points, a.check)
    if a.image:
        bad += check_image(reader, a.image, a.check)
    print("\n[time]")
    time_points(reader, points, a.time, a.min_range, a.max_range)
    if bad:
        sys.exit(f"\n{bad} mismatches -- do not use the fast path on this bag")
    print("\nall identical")


if __name__ == "__main__":
    main()
