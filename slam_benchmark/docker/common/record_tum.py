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
import sys
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
                 path_topic: str = "", pose_type: str = "nav_msgs/msg/Odometry",
                 count_topic: str = ""):
        super().__init__("slambench_recorder")
        # NOT declare_parameter("use_sim_time", ...) -- rclpy's Node constructor
        # already declared it (TimeSource.attach_node), so declaring it again
        # raises ParameterAlreadyDeclaredException and kills the node on
        # construction. It also would not have worked: sim time is switched on by
        # a parameter OVERRIDE at launch, which the entrypoint now passes.
        self.out = out
        self.n = 0
        self.t0 = time.time()
        self.fh = (out / "odometry.tum").open("w")
        self.fh.write(HEADER)
        self.cloud = None
        self.path = None

        # ONE subscription, of the type the image DECLARES. The original version
        # subscribed three times -- Odometry, PoseStamped, PoseWithCovariance --
        # so that whichever the method published would land. ROS 2 forbids two
        # subscriptions on one topic with different types in one node, so the
        # second create_subscription() threw and the recorder died on
        # construction, on every run, from the first day. A method image knows
        # what it publishes; it says so in SLAM_POSE_TYPE rather than having the
        # recorder guess.
        from rosidl_runtime_py.utilities import get_message
        qos = QoSProfile(depth=2000, reliability=ReliabilityPolicy.RELIABLE)
        self.create_subscription(get_message(pose_type), pose_topic, self.on_pose, qos)
        if path_topic:
            from nav_msgs.msg import Path as PathMsg
            self.create_subscription(PathMsg, path_topic, self.on_path, 5)
        # A witness. When the estimator processes a third of the frames and its
        # own log cannot say why, the question is whether the bag DELIVERED them.
        # This subscriber does nothing but count, so if it sees every frame the
        # loss is inside the method; if it sees the same third, the loss is in
        # replay or transport and no method parameter will touch it.
        # And the nodes' OWN accounting. rtabmap_sync publishes, per node, how
        # many synchronised sets it delivered and how many topics it dropped,
        # on /diagnostics. The last status of each named node is kept and lands
        # in timing.json, so "did the sync drop them or did the estimator" is
        # read off the run rather than argued.
        self.diagnostics = {}
        try:
            from diagnostic_msgs.msg import DiagnosticArray
            self.create_subscription(DiagnosticArray, "/diagnostics", self.on_diag, 50)
        except ImportError:
            pass
        self.seen = 0
        if count_topic:
            from sensor_msgs.msg import Image
            self.create_subscription(Image, count_topic, self.on_witness,
                                     QoSProfile(depth=100, reliability=ReliabilityPolicy.RELIABLE))
        if map_topic:
            from sensor_msgs.msg import PointCloud2
            self.create_subscription(PointCloud2, map_topic, self.on_map, 1)

    def on_pose(self, msg):
        self.fh.write(_row(msg.header, msg.pose))
        self.n += 1
        # Flushed often, for the same reason: what is on disk at any moment
        # should be what has been computed. 50 poses is ~3 s at 14.9 Hz.
        if self.n % 50 == 0:
            self.fh.flush()

    def on_diag(self, msg):
        for st in msg.status:
            lvl = st.level
            self.diagnostics[st.name] = {
                # DiagnosticStatus.level is a ROS `byte`: a 1-byte bytes object in rclpy
                "level": lvl[0] if isinstance(lvl, (bytes, bytearray)) else int(lvl),
                "message": st.message,
                **{kv.key: kv.value for kv in st.values}}

    def on_witness(self, msg):
        self.seen += 1

    def on_path(self, msg):
        # Keep only the latest. Every republish is the whole graph re-optimised,
        # so the last one is the method's final answer and the earlier ones are
        # snapshots of it mid-run.
        self.path = msg
        # And WRITE IT NOW, rather than at finish(). A container that is killed
        # rather than asked to stop -- a hung node, an out-of-memory, a Ctrl-C at
        # the wrong second -- used to lose a completed 156 s run entirely, with
        # every pose already computed and none of them on disk. Written through a
        # temporary file and renamed, so a kill mid-write leaves the previous
        # good graph rather than half of the new one.
        self._write_path()

    def _write_path(self):
        tmp = self.out / "trajectory.tum.tmp"
        with tmp.open("w") as f:
            f.write(HEADER)
            for ps in self.path.poses:
                f.write(_row(ps.header, ps.pose))
        tmp.replace(self.out / "trajectory.tum")

    def on_map(self, msg):
        self.cloud = msg                       # keep only the latest; that is the map

    def finish(self):
        self.fh.close()
        odom = self.out / "odometry.tum"
        traj = self.out / "trajectory.tum"
        if self.path is not None and self.path.poses:
            self._write_path()
            source, n = "optimised_graph", len(self.path.poses)
        else:
            # No graph: the odometry IS the estimate. Copied rather than moved so
            # the two files mean the same thing in every run directory.
            traj.write_text(odom.read_text())
            source, n = "odometry", self.n
        if self.cloud is not None:
            try:
                self._write_ply(self.cloud, self.out / "map.ply")
            except Exception as e:                           # noqa: BLE001
                # The map is a bonus; the trajectory is the deliverable and is
                # already on disk. Do not let a missing sensor_msgs_py take
                # timing.json down with it -- say so and carry on.
                print(f"map.ply not written: {e}", file=sys.stderr)
        ru = resource.getrusage(resource.RUSAGE_SELF)
        (self.out / "timing.json").write_text(json.dumps({
            "frames": n, "odometry_poses": self.n,
            "input_frames_witnessed": self.seen,
            "diagnostics": self.diagnostics,
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


def parse(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    """Split OUR arguments from ROS's.

    The entrypoint launches this with `--ros-args -p use_sim_time:=true` on the
    end, because that override is the only way to put a node on the bag's
    clock. argparse does not know those flags and `parse_args()` exits 2 on
    them -- which is a crash on the first line, backgrounded, invisible. So the
    unknown remainder is handed to rclpy.init instead of rejected.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--pose-topic", required=True)
    ap.add_argument("--map-topic", default="")
    ap.add_argument("--path-topic", default="",
                    help="the OPTIMISED graph, if the method publishes one. It "
                         "becomes trajectory.tum and the odometry is kept beside it.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--count-topic", default="",
                    help="an input image topic to count, as a witness to what the bag "
                         "delivered. Reported as input_frames_witnessed.")
    ap.add_argument("--pose-type", default="nav_msgs/msg/Odometry",
                    help="ROS message type on --pose-topic, e.g. nav_msgs/msg/Odometry "
                         "or geometry_msgs/msg/PoseStamped. The image declares it.")
    args, ros_args = ap.parse_known_args(argv[1:])
    return args, [argv[0], *ros_args]


def main():
    args, ros_argv = parse(sys.argv)

    rclpy.init(args=ros_argv)
    node = Recorder(args.pose_topic, args.map_topic, Path(args.out), args.path_topic,
                    args.pose_type, args.count_topic)
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
