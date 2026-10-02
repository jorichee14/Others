#!/usr/bin/env python3
"""Bag -> a folder of PNG frames, their real stamps, and a calibration file.

For the methods that are not ROS nodes. MASt3R-SLAM reads a directory of images
and has no concept of a bag, so the bag has to be unrolled first -- but the
stamps must survive, because a trajectory on the wrong clock associates against
the reference and scores, and that is the worst possible failure mode (container
contract rule 2).

MASt3R-SLAM's own folder loader fabricates timestamps as `index / 30.0`. So the
real stamps are written out here, in order, and `restamp_tum.py` puts them back
afterwards by inverting that map exactly. The alternative -- naming the folder
`tum` so its TUM loader picks it up -- silently applies hard-coded Freiburg
intrinsics, which is why it is not done.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# encoding -> (channels, numpy dtype, the channel order to slice into RGB)
ENCODINGS = {
    "rgb8":   (3, np.uint8, (0, 1, 2)),
    "bgr8":   (3, np.uint8, (2, 1, 0)),
    "rgba8":  (4, np.uint8, (0, 1, 2)),
    "bgra8":  (4, np.uint8, (2, 1, 0)),
    "mono8":  (1, np.uint8, (0, 0, 0)),
}


def to_rgb(data: bytes, height: int, width: int, encoding: str, step: int) -> np.ndarray:
    """Raw ROS image bytes -> (H,W,3) uint8 RGB.

    `step` is honoured rather than assumed: a row-padded image reshaped by
    width alone comes out sheared, and a sheared image still tracks -- badly,
    and with no error anywhere to say why.
    """
    if encoding not in ENCODINGS:
        raise ValueError(f"encoding {encoding!r} is not one this exporter knows "
                         f"({sorted(ENCODINGS)}). Add it deliberately.")
    ch, dtype, order = ENCODINGS[encoding]
    flat = np.frombuffer(data, dtype=dtype)
    row = step // np.dtype(dtype).itemsize
    img = flat[:height * row].reshape(height, row)[:, :width * ch].reshape(height, width, ch)
    return np.ascontiguousarray(img[:, :, order])


def main() -> int:                                           # pragma: no cover
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag", required=True)
    ap.add_argument("--topic", required=True)
    ap.add_argument("--info-topic", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import rosbag2_py
    import yaml
    from PIL import Image as PILImage
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=args.bag, storage_id=""),
                rosbag2_py.ConverterOptions("", ""))
    types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    for t in (args.topic, args.info_topic):
        if t not in types:
            print(f"topic {t!r} is not in the bag", file=sys.stderr)
            return 2
    reader.set_filter(rosbag2_py.StorageFilter(topics=[args.topic, args.info_topic]))
    img_cls, info_cls = get_message(types[args.topic]), get_message(types[args.info_topic])

    stamps, info = [], None
    while reader.has_next():
        topic, data, _ = reader.read_next()
        if topic == args.info_topic:
            if info is None:
                info = deserialize_message(data, info_cls)
            continue
        m = deserialize_message(data, img_cls)
        rgb = to_rgb(bytes(m.data), m.height, m.width, m.encoding, m.step)
        PILImage.fromarray(rgb).save(out / f"{len(stamps):06d}.png")
        stamps.append(m.header.stamp.sec + m.header.stamp.nanosec * 1e-9)

    if not stamps:
        print(f"{args.topic} produced no frames", file=sys.stderr)
        return 3
    if info is None:
        print(f"{args.info_topic} carried no message -- no intrinsics, and running "
              f"uncalibrated is a DIFFERENT experiment from the one configured",
              file=sys.stderr)
        return 4

    (out / "stamps.txt").write_text("".join(f"{t:.9f}\n" for t in stamps))
    (out / "calib.yaml").write_text(yaml.safe_dump({
        "width": int(info.width), "height": int(info.height),
        # fx, fy, cx, cy off the rectified projection matrix. These images are
        # `*_rect_*`, so the distortion is already out and adding the plumb-bob
        # coefficients here would undistort a second time.
        "calibration": [float(info.p[0]), float(info.p[5]),
                        float(info.p[2]), float(info.p[6])],
    }))
    print(f"{len(stamps)} frames, {stamps[-1] - stamps[0]:.1f} s, "
          f"{info.width}x{info.height} -> {out}")
    return 0


if __name__ == "__main__":                                   # pragma: no cover
    sys.exit(main())
