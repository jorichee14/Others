#!/usr/bin/env python3
"""
STAGE 09 - publish the best pose per robot as a ROS 2 mcap bag.

For every robot in the "09_publish" block the chosen stage-08 trajectory
(TUM, camera OPTICAL frame in `map`) is converted to the robot BODY frame and
written as:

  /<robot>/global_pose   geometry_msgs/PoseStamped, frame_id "map"
                         the anchored map frame shared by all robots; the
                         series starts at the trajectory's own first pose
  /<robot>/local_pose    geometry_msgs/PoseStamped, frame_id "<robot>/start"
                         the robot's own start pose is the origin, so the
                         first message is the identity - an odometry-like
                         frame that does not drift
  /tf                    map -> <robot>/<body_frame> (optional, for RViz)

Timestamps are the original bag stamps, so the output can be played next to
the raw bag.

    python3 09_publish_poses.py [pipeline_config.json]      # writes the bag
    python3 09_publish_poses.py pipeline_config.json --dry  # counts only, no ROS needed

Needs ROS 2 (rclpy, rosbag2_py, geometry_msgs, tf2_msgs) and the mcap
storage plugin (ros-<distro>-rosbag2-storage-mcap).

CONFIG
"09_publish": {
  "output_bag": "map_stages_20260828_outputs/coop2_best_poses",   <- a directory; mcap inside
  "map_frame": "map",
  "write_tf": true,
  "rate_hz": 0,                      <- 0 = every pose in the file; else decimate
  "robots": [
    { "name": "mobile_1",
      "traj": "map_stages_20260828_outputs/reference_coop2_mobile1_all/traj_mobile_1_zed_C_joint.tum",
      "cam_extrinsic_xyzquat": [-0.010, 0.060, 0.015, -0.5, 0.5, -0.5, 0.5],
      "body_frame": "zed_camera_link" },
    { "name": "mobile_2",
      "traj": "map_stages_20260828_outputs/reference_coop2_mobile2/traj_mobile_2_rs_C_joint.tum",
      "cam_extrinsic_xyzquat": [0.000308, 0.059191, -0.000162, 0.499583, -0.497446, 0.501716, -0.501244],
      "body_frame": "camera_link" }
  ]
}
"""
import json
import os
import shutil
import sys
import numpy as np
from scipy.spatial.transform import Rotation as Rot


def Rt(R, t):
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = t; return T


def inv(T):
    R = T[:3, :3]; o = np.eye(4); o[:3, :3] = R.T; o[:3, 3] = -R.T @ T[:3, 3]; return o


def make_T_xyzq(v):
    return Rt(Rot.from_quat(v[3:7]).as_matrix(), np.asarray(v[0:3], float))


def read_tum(path):
    A = np.loadtxt(path)
    if A.ndim == 1:
        A = A[None]
    A = A[np.argsort(A[:, 0])]
    ts = A[:, 0]
    Ts = np.array([Rt(Rot.from_quat(r[4:8]).as_matrix(), r[1:4]) for r in A])
    return ts, Ts


def decimate(ts, Ts, rate_hz):
    if not rate_hz or rate_hz <= 0:
        return ts, Ts
    keep, t_last = [], -1e18
    for i, t in enumerate(ts):
        if t - t_last >= (1.0 / rate_hz) * 0.9:
            keep.append(i); t_last = t
    return ts[keep], Ts[keep]


def robot_poses(rb, rate_hz):
    """-> ts, T_map_body, T_start_body (the same poses relative to the first)."""
    ts, Tc = read_tum(rb["traj"])
    ts, Tc = decimate(ts, Tc, rate_hz)
    X = make_T_xyzq(rb["cam_extrinsic_xyzquat"]) if rb.get("cam_extrinsic_xyzquat") \
        else np.eye(4)
    # camera optical -> body: T_map_body = T_map_cam @ inv(X)
    Tb = np.array([T @ inv(X) for T in Tc])
    T0 = Tb[0]
    Tl = np.array([inv(T0) @ T for T in Tb])
    path = float(np.sum(np.linalg.norm(np.diff(Tb[:, :3, 3], axis=0), axis=1)))
    print("  %s: %d poses, %.1f s, path %.1f m, body frame '%s'; start in map "
          "xyz=%s" % (rb["name"], len(ts), ts[-1] - ts[0], path, rb.get("body_frame", "?"),
                      np.round(T0[:3, 3], 3).tolist()))
    return ts, Tb, Tl


def main():
    cfg_path = next((a for a in sys.argv[1:] if not a.startswith("--")), "pipeline_config.json")
    dry = "--dry" in sys.argv
    cfg = json.load(open(cfg_path))
    s = cfg.get("09_publish")
    if s is None:
        raise SystemExit("add a '09_publish' block to %s (sample in this file)" % cfg_path)
    map_frame = s.get("map_frame", "map")
    rate_hz = float(s.get("rate_hz", 0) or 0)
    write_tf = bool(s.get("write_tf", True))
    robots = []
    for rb in s["robots"]:
        if not os.path.exists(rb["traj"]):
            raise SystemExit("%s: trajectory %s not found" % (rb["name"], rb["traj"]))
        robots.append((rb, *robot_poses(rb, rate_hz)))
    if dry:
        print("dry run: %d robots, %d messages" % (
            len(robots), sum(len(r[1]) * (3 if write_tf else 2) for r in robots)))
        return

    import rosbag2_py
    from rclpy.serialization import serialize_message
    from rclpy.time import Time
    from geometry_msgs.msg import PoseStamped, TransformStamped
    from tf2_msgs.msg import TFMessage

    out = s["output_bag"]
    if os.path.isdir(out):
        shutil.rmtree(out)
    w = rosbag2_py.SequentialWriter()
    w.open(rosbag2_py.StorageOptions(uri=out, storage_id="mcap"),
           rosbag2_py.ConverterOptions("", ""))
    topics = []
    for rb, ts, Tb, Tl in robots:
        for nm in ("global_pose", "local_pose"):
            topics.append("/%s/%s" % (rb["name"], nm))
    for tp in topics:
        w.create_topic(rosbag2_py.TopicMetadata(
            name=tp, type="geometry_msgs/msg/PoseStamped", serialization_format="cdr"))
    if write_tf:
        w.create_topic(rosbag2_py.TopicMetadata(
            name="/tf", type="tf2_msgs/msg/TFMessage", serialization_format="cdr"))

    def stamp(t):
        sec = int(t); nsec = int(round((t - sec) * 1e9))
        if nsec >= 1_000_000_000:
            sec += 1; nsec -= 1_000_000_000
        return sec, nsec

    def pose_msg(t, T, frame):
        m = PoseStamped()
        m.header.stamp.sec, m.header.stamp.nanosec = stamp(t)
        m.header.frame_id = frame
        q = Rot.from_matrix(T[:3, :3]).as_quat()
        m.pose.position.x, m.pose.position.y, m.pose.position.z = map(float, T[:3, 3])
        m.pose.orientation.x, m.pose.orientation.y, m.pose.orientation.z, \
            m.pose.orientation.w = map(float, q)
        return m

    # interleave all robots by time so the bag plays in order
    events = []
    for rb, ts, Tb, Tl in robots:
        for i, t in enumerate(ts):
            events.append((t, rb, Tb[i], Tl[i]))
    events.sort(key=lambda e: e[0])
    n = 0
    for t, rb, Tm, Tloc in events:
        t_ns = int(round(t * 1e9))
        w.write("/%s/global_pose" % rb["name"],
                serialize_message(pose_msg(t, Tm, map_frame)), t_ns)
        w.write("/%s/local_pose" % rb["name"],
                serialize_message(pose_msg(t, Tloc, "%s/start" % rb["name"])), t_ns)
        n += 2
        if write_tf:
            tfm = TFMessage(); tr = TransformStamped()
            tr.header.stamp.sec, tr.header.stamp.nanosec = stamp(t)
            tr.header.frame_id = map_frame
            tr.child_frame_id = "%s/%s" % (rb["name"], rb.get("body_frame", "base"))
            q = Rot.from_matrix(Tm[:3, :3]).as_quat()
            tr.transform.translation.x, tr.transform.translation.y, \
                tr.transform.translation.z = map(float, Tm[:3, 3])
            tr.transform.rotation.x, tr.transform.rotation.y, tr.transform.rotation.z, \
                tr.transform.rotation.w = map(float, q)
            tfm.transforms.append(tr)
            w.write("/tf", serialize_message(tfm), t_ns)
            n += 1
    del w
    print("wrote %s: %d messages on %s%s" % (out, n, ", ".join(topics),
                                            ", /tf" if write_tf else ""))
    print("  global_pose: body frame in '%s'; local_pose: body frame relative to "
          "the robot's first pose (frame '<robot>/start')" % map_frame)


if __name__ == "__main__":
    main()
