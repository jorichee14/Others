#!/usr/bin/env python3
"""Reproject a depth image into a PointCloud2, so the LiDAR estimator can eat it.

This node exists for exactly one row of the plan: KISS-ICP is the geometry-only
backbone, and on a LiDAR-free platform the only geometry available is the depth
camera. It is the bridge, and it is deliberately the dumbest possible bridge --
deprojection with the stream's own camera_info and nothing else. No filtering,
no normal estimation, no downsampling beyond a declared pixel stride, because
every one of those is a tuning knob that would belong in the method config
rather than in a format adapter.

THE FAILURE THIS GUARDS AGAINST. `configs/methods/kiss_icp.yaml` warns that the
ZED's depth is 32FC1 metres and the RealSense's is 16UC1 millimetres, "a factor
of 1000 apart, and a run with the wrong one completes and produces a map of a
room 1000 m across". So the scale is not inferred from the encoding: it is
passed in from the dataset config, and then CHECKED against the encoding. A
mismatch is a refusal, not a correction -- correcting it silently would hide a
config that is wrong about a stream, which is the more dangerous of the two.

The cloud is published in the depth image's own optical frame, unmodified. The
container contract (docker/README.md rule 1) says poses describe the sensor
frame the method was given; the evaluator applies `reference_frame_from_sensor`
from the dataset config. A helpful transform here would be applied twice.
"""
from __future__ import annotations

import os
import sys

import numpy as np

# encoding -> (numpy dtype, the ONLY depth_scale that can be right for it)
ENCODINGS = {
    "32FC1": (np.float32, 1.0),      # metres, as the ZED publishes
    "16UC1": (np.uint16, 0.001),     # millimetres, as the RealSense publishes
}


def deproject(depth_m: np.ndarray, fx: float, fy: float, cx: float, cy: float,
              stride: int = 1, r_min: float = 0.0, r_max: float = np.inf
              ) -> np.ndarray:
    """(H,W) depth in METRES -> (N,3) float32 points in the optical frame.

    Optical convention, which is the one camera_info is defined in: +z forward
    along the boresight, +x right, +y down. Invalid depth in ROS is 0 or NaN;
    both are dropped here rather than turned into a point at the origin, which
    would put a dense blob at the camera centre in every single frame and give
    the ICP a rigid feature that does not move.

    `r_min`/`r_max` gate on z (the depth channel), not on the euclidean radius:
    a depth camera's stated range is a depth range, and `configs/coop2.yaml`
    carries it per stream as `range_m`.
    """
    if depth_m.ndim != 2:
        raise ValueError(f"depth must be (H,W), got {depth_m.shape}")
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")

    z = depth_m[::stride, ::stride]
    h, w = z.shape
    # Pixel centres of the SUBSAMPLED grid, expressed in FULL-resolution pixel
    # coordinates -- the intrinsics are full-resolution, so striding the image
    # without striding the coordinates is a silent focal-length error.
    us = np.arange(w, dtype=np.float64) * stride
    vs = np.arange(h, dtype=np.float64) * stride

    good = np.isfinite(z) & (z > 0) & (z >= r_min) & (z <= r_max)
    v_idx, u_idx = np.nonzero(good)
    if v_idx.size == 0:
        return np.zeros((0, 3), dtype=np.float32)

    zz = z[v_idx, u_idx].astype(np.float64)
    xx = (us[u_idx] - cx) * zz / fx
    yy = (vs[v_idx] - cy) * zz / fy
    return np.column_stack([xx, yy, zz]).astype(np.float32)


def check_scale(encoding: str, depth_scale: float) -> float:
    """Return the scale to use, or raise. See the module docstring."""
    if encoding not in ENCODINGS:
        raise ValueError(
            f"depth encoding {encoding!r} is not one this adapter knows "
            f"({sorted(ENCODINGS)}). Add it deliberately -- guessing the units "
            f"of a depth image is how a room becomes 1000 m across.")
    _, expected = ENCODINGS[encoding]
    if not np.isclose(depth_scale, expected, rtol=1e-6):
        raise ValueError(
            f"the stream declares depth_scale={depth_scale} but the bag publishes "
            f"{encoding}, whose only correct scale is {expected}. One of the two is "
            f"wrong and this adapter will not pick. Fix configs/coop2.yaml.")
    return expected


def _main() -> int:                                          # pragma: no cover
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import CameraInfo, Image, PointCloud2, PointField

    depth_topic = os.environ["DEPTH_TOPIC"]
    info_topic = os.environ["DEPTH_INFO_TOPIC"]
    cloud_topic = os.environ.get("CLOUD_TOPIC", "/slambench/cloud")
    depth_scale = float(os.environ["DEPTH_SCALE"])
    r_min = float(os.environ.get("RANGE_MIN", "0.0"))
    r_max = float(os.environ.get("RANGE_MAX", "inf"))
    stride = int(os.environ.get("DEPTH_STRIDE", "2"))

    class Bridge(Node):
        def __init__(self):
            super().__init__("depth_to_cloud")
            # NOT declare_parameter("use_sim_time", ...) -- rclpy's Node constructor
        # already declared it (TimeSource.attach_node), so declaring it again
        # raises ParameterAlreadyDeclaredException and kills the node on
        # construction. It also would not have worked: sim time is switched on by
        # a parameter OVERRIDE at launch, which the entrypoint now passes.
            self.K = None
            self.n = 0
            qos = QoSProfile(depth=20, reliability=ReliabilityPolicy.RELIABLE)
            self.pub = self.create_publisher(PointCloud2, cloud_topic, qos)
            self.create_subscription(CameraInfo, info_topic, self.on_info, 5)
            self.create_subscription(Image, depth_topic, self.on_depth, qos)

        def on_info(self, msg):
            if self.K is None:
                self.K = (msg.k[0], msg.k[4], msg.k[2], msg.k[5])
                self.get_logger().info(f"intrinsics fx,fy,cx,cy = {self.K}")

        def on_depth(self, msg):
            if self.K is None:
                return                      # camera_info has not arrived yet
            scale = check_scale(msg.encoding, depth_scale)
            dtype, _ = ENCODINGS[msg.encoding]
            raw = np.frombuffer(msg.data, dtype=dtype).reshape(msg.height, msg.width)
            pts = deproject(raw.astype(np.float32) * scale, *self.K,
                            stride=stride, r_min=r_min, r_max=r_max)
            out = PointCloud2()
            out.header = msg.header         # SAME stamp, SAME frame. Both load-bearing.
            out.height, out.width = 1, len(pts)
            out.fields = [PointField(name=n, offset=4 * i, datatype=7, count=1)
                          for i, n in enumerate("xyz")]
            out.is_bigendian = False
            out.point_step, out.row_step = 12, 12 * len(pts)
            out.is_dense = True
            out.data = np.ascontiguousarray(pts).tobytes()
            self.pub.publish(out)
            self.n += 1
            if self.n % 100 == 0:
                self.get_logger().info(f"{self.n} clouds, last {len(pts)} points")

    rclpy.init()
    node = Bridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
    return 0


if __name__ == "__main__":                                   # pragma: no cover
    sys.exit(_main())
