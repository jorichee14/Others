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


HEADER = "# timestamp tx ty tz qx qy qz qw\n"


def _row(header, pose) -> str:
    p = getattr(pose, "pose", pose)          # Odometry/PoseWithCovariance nest it
    t = header.stamp.sec + header.stamp.nanosec * 1e-9
    return (f"{t:.9f} {p.position.x:.6f} {p.position.y:.6f} {p.position.z:.6f} "
            f"{p.orientation.x:.9f} {p.orientation.y:.9f} "
            f"{p.orientation.z:.9f} {p.orientation.w:.9f}\n")


class Recorder(Node):
    """Records the estimate, and -- where the method has two -- BOTH of them.

    A loop-closing SLAM system publishes two different things and they are not
    interchangeable. The odometry topic carries the incremental estimate, which
    never benefits from a loop closure. The optimised graph carries the answer
    the method actually claims, with every closure applied retroactively to the
    poses behind it.

    Recording only the first and labelling the row `loop_closure: true` would put
    a drift number in a table that says it has none. So the graph, when the image
    declares one, becomes `trajectory.tum` -- and the odometry is kept beside it
    as `odometry.tum`, which is free and is exactly the drift floor the closing
    row has to beat.
    """

    def __init__(self, pose_topic: str, map_topic: str, out: Path,
                 path_topic: str = ""):
        super().__init__("slambench_recorder")
        self.declare_parameter("use_sim_time", True)
        self.out = out
        self.n = 0
        self.t0 = time.time()
        self.fh = (out / "odometry.tum").open("w")
        self.fh.write(HEADER)
        self.cloud = None
        self.path = None

        from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
        from nav_msgs.msg import Odometry
        qos = QoSProfile(depth=2000, reliability=ReliabilityPolicy.RELIABLE)
        for mt in (Odometry, PoseStamped, PoseWithCovarianceStamped):
            self.create_subscription(mt, pose_topic, self.on_pose, qos)
        if path_topic:
            from nav_msgs.msg import Path as PathMsg
            self.create_subscription(PathMsg, path_topic, self.on_path, 5)
        if map_topic:
            from sensor_msgs.msg import PointCloud2
            self.create_subscription(PointCloud2, map_topic, self.on_map, 1)

    def on_pose(self, msg):
        self.fh.write(_row(msg.header, msg.pose))
        self.n += 1
        if self.n % 200 == 0:
            self.fh.flush()

    def on_path(self, msg):
        # Keep only the latest. Every republish is the whole graph re-optimised,
        # so the last one is the method's final answer and the earlier ones are
        # snapshots of it mid-run.
        self.path = msg

    def on_map(self, msg):
        self.cloud = msg                       # keep only the latest; that is the map

    def finish(self):
        self.fh.close()
        odom = self.out / "odometry.tum"
        traj = self.out / "trajectory.tum"
        if self.path is not None and self.path.poses:
            with traj.open("w") as f:
                f.write(HEADER)
                for ps in self.path.poses:
                    f.write(_row(ps.header, ps.pose))
            source, n = "optimised_graph", len(self.path.poses)
        else:
            # No graph: the odometry IS the estimate. Copied rather than moved so
            # the two files mean the same thing in every run directory.
            traj.write_text(odom.read_text())
            source, n = "odometry", self.n
        if self.cloud is not None:
            self._write_ply(self.cloud, self.out / "map.ply")
        ru = resource.getrusage(resource.RUSAGE_SELF)
        (self.out / "timing.json").write_text(json.dumps({
            "frames": n, "odometry_poses": self.n,
            "trajectory_source": source,
            "wall_s": time.time() - self.t0,
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
    ap.add_argument("--path-topic", default="",
                    help="the OPTIMISED graph, if the method publishes one. It "
                         "becomes trajectory.tum and the odometry is kept beside it.")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rclpy.init()
    node = Recorder(args.pose_topic, args.map_topic, Path(args.out), args.path_topic)
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
