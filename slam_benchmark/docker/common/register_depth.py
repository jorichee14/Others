#!/usr/bin/env python3
"""Register a depth image into the colour camera, for RGB-D front-ends.

RTAB-Map's rgbd_odometry takes ONE colour image and ONE depth image and requires
the depth to be in the colour camera's frame with the colour image's aspect
ratio (rgbd_odometry.cpp: "imageWidth/depthWidth == imageHeight/depthHeight").
A RealSense driver does that before publishing (`align_depth`); this bag's
RealSense publishes 640x480 depth in `camera_depth_optical_frame` beside
1280x720 colour in `camera_color_optical_frame`, and the first mobile_2 run
died on exactly that check. So the registration is done here.

It is a FORMAT adapter, like depth_to_cloud.py: deproject with the depth
intrinsics, move the points by the declared colour<-depth extrinsic (a number
in configs/coop2.yaml, rule 4), project with the colour intrinsics, keep the
nearest point per pixel. No hole filling, no smoothing -- those would be
knobs. The output is at the colour resolution divided by an integer, with
intrinsics scaled by exactly that ratio, because that is the scaling RTAB-Map
applies internally when the depth is smaller than the colour image.

The consequence for the contract (docker/README.md rule 1): the method's poses
come out in the COLOUR optical frame, and the method config declares that
with an identity `extrinsic_by_stream` for the stream. Registered depth in
the depth frame is not a thing; this is what registration means.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from depth_to_cloud import ENCODINGS, check_scale, deproject  # noqa: E402


def quat_to_rot(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    x, y, z, w = qx, qy, qz, qw
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def register(depth_m: np.ndarray, K_depth: tuple, K_color: tuple, T_color_from_depth: np.ndarray,
             out_hw: tuple[int, int], divisor: int = 1) -> np.ndarray:
    """(H,W) depth in METRES in the depth camera -> (h,w) depth in METRES in the
    colour camera at colour_resolution/divisor. K_* are (fx, fy, cx, cy) at
    FULL resolution; K_color is scaled by 1/divisor here, exactly as RTAB-Map
    scales it. Unfilled pixels are 0 (the ROS "no depth" value). Where several
    depth points land on one output pixel the NEAREST wins: a z-buffer, made
    deterministic with a sort rather than relying on the order numpy applies
    repeated fancy-index writes.
    """
    h, w = out_hw
    fx, fy, cx, cy = (v / divisor for v in K_color)
    pts = deproject(depth_m, *K_depth).astype(np.float64)
    out = np.zeros((h, w), dtype=np.float32)
    if len(pts) == 0:
        return out
    R, t = T_color_from_depth[:3, :3], T_color_from_depth[:3, 3]
    p = pts @ R.T + t
    z = p[:, 2]
    keep = z > 0
    u = np.rint(fx * p[keep, 0] / z[keep] + cx).astype(np.int64)
    v = np.rint(fy * p[keep, 1] / z[keep] + cy).astype(np.int64)
    z = z[keep]
    inside = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    u, v, z = u[inside], v[inside], z[inside]
    if z.size == 0:
        return out
    lin = v * w + u
    order = np.lexsort((z, lin))           # by pixel, then nearest first
    lin_s, z_s = lin[order], z[order]
    first = np.concatenate(([True], lin_s[1:] != lin_s[:-1]))
    out.flat[lin_s[first]] = z_s[first]
    return out


def output_shape(color_hw: tuple[int, int], divisor: int) -> tuple[int, int]:
    H, W = color_hw
    if H % divisor or W % divisor:
        raise ValueError(f"colour {W}x{H} is not divisible by {divisor}; the registered depth "
                         f"must keep the colour aspect ratio exactly")
    return H // divisor, W // divisor


def _main() -> int:                                          # pragma: no cover
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import CameraInfo, Image

    depth_topic = os.environ["DEPTH_TOPIC"]
    depth_info_topic = os.environ["DEPTH_INFO_TOPIC"]
    color_info_topic = os.environ["COLOR_INFO_TOPIC"]
    out_topic = os.environ.get("REGISTERED_TOPIC", "/slambench/depth_registered")
    color_frame = os.environ["COLOR_FRAME"]
    depth_scale = float(os.environ["DEPTH_SCALE"])
    divisor = int(os.environ.get("REGISTER_DIVISOR", "2"))
    x, y, z_, qx, qy, qz, qw = (float(v) for v in os.environ["COLOR_FROM_DEPTH"].split())
    T = np.eye(4)
    T[:3, :3] = quat_to_rot(qx, qy, qz, qw)
    T[:3, 3] = (x, y, z_)

    class Register(Node):
        def __init__(self):
            super().__init__("register_depth")
            self.Kd = self.Kc = self.color_hw = None
            self.n = 0
            qos = QoSProfile(depth=20, reliability=ReliabilityPolicy.RELIABLE)
            self.pub = self.create_publisher(Image, out_topic, qos)
            self.create_subscription(CameraInfo, depth_info_topic, self.on_depth_info, 5)
            self.create_subscription(CameraInfo, color_info_topic, self.on_color_info, 5)
            self.create_subscription(Image, depth_topic, self.on_depth, qos)

        def on_depth_info(self, msg):
            if self.Kd is None:
                self.Kd = (msg.k[0], msg.k[4], msg.k[2], msg.k[5])
                self.get_logger().info(f"depth intrinsics {self.Kd} {msg.width}x{msg.height}")

        def on_color_info(self, msg):
            if self.Kc is None:
                self.Kc = (msg.k[0], msg.k[4], msg.k[2], msg.k[5])
                self.color_hw = (msg.height, msg.width)
                self.get_logger().info(f"colour intrinsics {self.Kc} {msg.width}x{msg.height}; "
                                       f"registering at /{divisor} = "
                                       f"{msg.width // divisor}x{msg.height // divisor} in {color_frame}")

        def on_depth(self, msg):
            if self.Kd is None or self.Kc is None:
                return
            scale = check_scale(msg.encoding, depth_scale)
            dtype, _ = ENCODINGS[msg.encoding]
            raw = np.frombuffer(msg.data, dtype=dtype).reshape(msg.height, msg.width)
            reg = register(raw.astype(np.float32) * scale, self.Kd, self.Kc, T,
                           output_shape(self.color_hw, divisor), divisor)
            out = Image()
            out.header.stamp = msg.header.stamp        # the depth's own stamp
            out.header.frame_id = color_frame          # in the colour camera now
            out.height, out.width = reg.shape
            out.encoding = msg.encoding                # same units as the input
            out.is_bigendian = False
            if dtype == np.uint16:
                data = np.rint(reg / scale).astype(np.uint16)
            else:
                data = reg.astype(np.float32)
            out.step = data.shape[1] * data.dtype.itemsize
            out.data = np.ascontiguousarray(data).tobytes()
            self.pub.publish(out)
            self.n += 1
            if self.n % 100 == 0:
                filled = float((reg > 0).mean())
                self.get_logger().info(f"{self.n} registered, last {filled:.0%} of pixels filled")

    rclpy.init()
    node = Register()
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
