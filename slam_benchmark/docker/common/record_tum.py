#!/usr/bin/env python3
"""Subscribe to the method's pose output and write TUM; snapshot its map to PLY.

Runs inside the method container. It writes the poses EXACTLY as the method
publishes them — same frame, same world, same stamps — because every
re-expression belongs in the evaluator, where it is declared in a config that
travels with the result.
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import time
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy


class Recorder(Node):
    def __init__(self, pose_topic: str, map_topic: str, out: Path):
        super().__init__("slambench_recorder")
        self.declare_parameter("use_sim_time", True)
        self.out = out
        self.n = 0
        self.t0 = time.time()
        self.fh = (out / "trajectory.tum").open("w")
        self.fh.write("# timestamp tx ty tz qx qy qz qw\n")
        self.cloud = None

        from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
        from nav_msgs.msg import Odometry
        qos = QoSProfile(depth=2000, reliability=ReliabilityPolicy.RELIABLE)
        for mt in (Odometry, PoseStamped, PoseWithCovarianceStamped):
            self.create_subscription(mt, pose_topic, self.on_pose, qos)
        if map_topic:
            from sensor_msgs.msg import PointCloud2
            self.create_subscription(PointCloud2, map_topic, self.on_map, 1)

    def on_pose(self, msg):
        h = msg.header
        p = msg.pose
        p = getattr(p, "pose", p)
        t = h.stamp.sec + h.stamp.nanosec * 1e-9
        self.fh.write(f"{t:.9f} {p.position.x:.6f} {p.position.y:.6f} {p.position.z:.6f} "
                      f"{p.orientation.x:.9f} {p.orientation.y:.9f} "
                      f"{p.orientation.z:.9f} {p.orientation.w:.9f}\n")
        self.n += 1
        if self.n % 200 == 0:
            self.fh.flush()

    def on_map(self, msg):
        self.cloud = msg                       # keep only the latest; that is the map

    def finish(self):
        self.fh.close()
        if self.cloud is not None:
            self._write_ply(self.cloud, self.out / "map.ply")
        ru = resource.getrusage(resource.RUSAGE_SELF)
        (self.out / "timing.json").write_text(json.dumps({
            "frames": self.n, "wall_s": time.time() - self.t0,
            "cpu_s": ru.ru_utime + ru.ru_stime,
            "peak_rss_mb": ru.ru_maxrss / 1024.0,
        }, indent=2) + "\n")

    @staticmethod
    def _write_ply(msg, path: Path):
        import numpy as np
        from sensor_msgs_py import point_cloud2
        pts = np.array([[p[0], p[1], p[2]] for p in point_cloud2.read_points(
            msg, field_names=("x", "y", "z"), skip_nans=True)], dtype=np.float32)
        header = ("ply\nformat binary_little_endian 1.0\n"
                  f"element vertex {len(pts)}\n"
                  "property float x\nproperty float y\nproperty float z\n"
                  "end_header\n").encode()
        path.write_bytes(header + np.ascontiguousarray(pts).tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pose-topic", required=True)
    ap.add_argument("--map-topic", default="")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rclpy.init()
    node = Recorder(args.pose_topic, args.map_topic, Path(args.out))
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.finish()
        node.destroy_node()
        rclpy.try_shutdown()
        print(f"recorded {node.n} poses to {args.out}")


if __name__ == "__main__":
    main()
