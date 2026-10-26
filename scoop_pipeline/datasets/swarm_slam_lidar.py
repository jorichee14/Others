#!/usr/bin/env python3
"""A multi-robot pass as input for Swarm-SLAM (github.com/MISTLab/Swarm-SLAM,
the cslam packages), LiDAR mode on every robot, with the odometry taken from
outside: any method's trajectory (KISS-ICP, FAST-LIO2, GLIM's odometry, ...)
as a TUM file, or the ground truth.

    python scoop_pipeline/datasets/swarm_slam_lidar.py <finalized merged bag> [<out dir>]
        [--agents mobile_1 mobile_2] [--odom gt | --odom-tum MACHINE TUM [FRAME] ...]
        [--start 0] [--duration 0] [--rate 10] [--min-range 0.5] [--voxel 0.5]
        [--ref-dir DIR | --gt-tum MACHINE TUM ...] [--start-at-gt]

Writes <out> (default data/processed/<date>/<pass>/datasets/swarm_slam_lidar/<tag>_<odom>):

  bag/             one ROS 2 bag, robot i under /r<i> (the topic names of
                   cslam_experiments' ouster_lidar.yaml):
                     /r<i>/pointcloud  sensor_msgs/PointCloud2, x y z float32 in
                                       the LiDAR frame, no-return points and
                                       those within --min-range (the robot, the
                                       operator) left out; not deskewed
                     /r<i>/odom        nav_msgs/Odometry: the LiDAR's pose in the
                                       robot's odom frame at the scan stamp
                   stamped as recorded (play it with --clock)
  scoop_lidar.yaml the cslam config: ouster_lidar.yaml's, with logs on
                   (results/) and the visualization off
  gt_r<i>.tum      the LiDAR in map at every scan written, for
                   analysis/slam/cslam_eval.py (which reads this export as it
                   does datasets/swarm_slam.py's)
  report.txt

Ground truth: stage 04's LiDAR track, T_map_lidar of the points topic's frame
(data/processed/<date>/<pass>/odometry/reference_<tag>/traj_<machine>_lidar.tum,
or --ref-dir, or --gt-tum per robot).

Odometry (Swarm-SLAM takes it from outside, matched to the clouds by stamp
within pointcloud_odom_approx_time_sync_s):
  --odom gt                  the ground truth itself (a check of the setup: no
                             drift to correct)
  --odom-tum MACHINE TUM [FRAME]
                             a robot's odometry from any method, 't x y z qx qy
                             qz qw' (s) of FRAME in the method's own odom frame,
                             interpolated to every scan stamp. FRAME defaults to
                             the LiDAR's (KISS-ICP); another frame (FAST-LIO2's
                             IMU, os_imu_mobile1) is moved to the LiDAR through
                             /tf_static. It must be odometry, not a loop-closed
                             trajectory: a correction jump would be fed to
                             cslam's graph as motion
A robot without --odom-tum takes --odom (default gt).

Start: each robot's odometry in its own odom frame, as a robot would have it;
--start-at-gt puts it in map at the ground truth of its first scan written
(the initial pose only: cslam_eval's in-map errors are then drift from a
known start).

--start / --duration (s, from the first scan of any robot): a slice of the
pass, for a small test.
"""
import argparse
import heapq
import logging
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import mcap_convert as mc                                           # noqa: E402
import swarm_slam as SS                                             # noqa: E402

log = logging.getLogger("swarm_slam_lidar")

POINTS = {"mobile_1": "/mobile_1/ouster/points",
          "mobile_2": "/mobile_2/ouster/points"}
XYZ = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4")])
# cslam_experiments/config/ouster_lidar.yaml, but: evaluation logs on, with the pose
# timestamps (cslam_eval); visualization off; max_waiting_time_sec 60 (as
# datasets/swarm_slam.py: the run ends 30 s after the bag); voxel_size from --voxel
CONFIG = """/**:
  ros__parameters:
    frontend:
      odom_topic: "odom"
      sensor_type: "lidar"
      pointcloud_topic: "pointcloud"
      voxel_size: %(voxel)g
      registration_min_inliers: 60
      inter_robot_loop_closure_budget: 1
      inter_robot_detection_period_sec: 5
      enable_sparsification: true
      max_queue_size: 10
      similarity_threshold: 0.8
      global_descriptors_topic: "global_descriptors"
      intra_loop_min_inbetween_keyframes: 20
      detection_publication_period_sec: 0.5
      detection_publication_max_elems_per_msg: 100
      enable_intra_robot_loop_closures: true
      keyframe_generation_ratio_distance: 0.5
      pointcloud_odom_approx_time_sync_s: 0.1
      use_vertex_cover_selection: true
      map_manager_process_period_ms: 100
      global_descriptor_technique: "scancontext"
    neighbor_management:
      enable_neighbor_monitoring: true
      max_heartbeat_delay_sec: 5.0
      init_delay_sec: 5.0
      heartbeat_period_sec: 1.0
    backend:
      pose_graph_optimization_start_period_ms: 10000
      pose_graph_optimization_loop_period_ms: 100
      max_waiting_time_sec: 60
      enable_broadcast_tf_frames: true
    evaluation:
      enable_logs: true
      log_folder: "%(logs)s"
      enable_gps_recording: false
      gps_topic: "gps/fix"
      enable_pose_timestamps_recording: true
      enable_sparsification_comparison: false
    visualization:
      enable: false
      publishing_period_ms: 1000
"""


def pass_paths(bag):
    """(processed/<date>/<pass>, tag) of a bag in the data layout, or (None, None)."""
    dp = mc.pass_of(bag)
    if dp is None:
        return None, None
    date, pas = dp
    p = os.path.abspath(os.path.expanduser(bag[0] if isinstance(bag, (list, tuple)) else bag))
    root = p[:p.rindex(os.sep + date + os.sep + pas)]
    root = root[:root.rindex(os.sep)]
    return os.path.join(root, "processed", date, pas), "%s_%s" % (pas.replace("/", "_"), date)


def default_out(bag, odom_label):
    proc, tag = pass_paths(bag)
    if proc is None:
        raise mc.ConvertError(f"{bag} is not below data/{{raw,work,processed}}/<date>/<pass>/: "
                              "give the output folder")
    return os.path.join(proc, "datasets", "swarm_slam_lidar", "%s_%s" % (tag, odom_label))


def default_gt(bag, machine, ref_dir=None):
    if ref_dir is None:
        proc, tag = pass_paths(bag)
        if proc is None:
            return None
        ref_dir = os.path.join(proc, "odometry", "reference_" + tag)
    return os.path.join(ref_dir, "traj_%s_lidar.tum" % machine)


def scan(msg, min_range):
    """A PointCloud2's points as XYZ records, no returns and those within
    min_range (m) of the sensor left out."""
    P, _ = mc.cloud_points(msg)
    if min_range > 0:
        P = P[np.einsum("ij,ij->i", P, P) >= min_range * min_range]
    a = np.empty(len(P), XYZ)
    a["x"], a["y"], a["z"] = P[:, 0], P[:, 1], P[:, 2]
    return a


class Robot:
    def __init__(self, i, machine, paths, edges, gt_tum, odom_tum=None, odom_frame=None):
        self.i, self.machine, self.topic = i, machine, POINTS[machine]
        first = mc.first_msg(paths, self.topic)
        if first is None:
            raise mc.ConvertError(f"{machine}: no messages on {self.topic}")
        self.lidar = first.header.frame_id.lstrip("/")
        if not gt_tum or not os.path.exists(gt_tum):
            raise mc.ConvertError(f"{machine}: no ground truth {gt_tum} (stage 04's "
                                  f"traj_{machine}_lidar.tum): give --ref-dir or --gt-tum")
        self.gt, self.gt_src = mc.PoseInterp.from_tum(gt_tum), gt_tum
        if odom_tum is None:
            self.odom, self.T_fl, self.odom_src = self.gt, np.eye(4), "ground truth (%s)" % gt_tum
        else:
            frame = (odom_frame or self.lidar).lstrip("/")
            self.T_fl = np.eye(4) if frame == self.lidar else mc.lookup_static(edges, frame, self.lidar)
            if self.T_fl is None:
                raise mc.ConvertError(f"{machine}: the odometry's frame {frame} and the LiDAR frame "
                                      f"{self.lidar} are not connected in /tf_static")
            self.odom = mc.PoseInterp.from_tum(odom_tum)
            self.odom_src = "%s (%s -> %s)" % (odom_tum, frame, self.lidar)
        self.T_mo, self.start = np.eye(4), "own odom frame"
        self.last = None
        self.gt_lines, self.n, self.points = [], 0, 0
        self.drop = {"no odom": 0, "no ground truth": 0}

    def frame(self, t, msg, min_range, start_at_gt):
        """[(t, topic, msg)] for one scan, or [] (no odometry or ground truth there)."""
        To, G = self.odom.at(t), self.gt.at(t)
        if To is None or G is None:
            self.drop["no odom" if To is None else "no ground truth"] += 1
            return []
        To = To @ self.T_fl
        if start_at_gt and self.n == 0:
            self.T_mo = G @ np.linalg.inv(To)
            self.start = "map, at the ground truth of its first scan (t %.3f)" % (t * 1e-9)
        pts = scan(msg, min_range)
        ns, lidar = "/r%d" % self.i, "r%d/lidar" % self.i
        self.gt_lines.append(mc.tum_line(t, G))
        self.n += 1
        self.points += len(pts)
        return [(t, ns + "/pointcloud", SS.T_msg().pointcloud2(pts, t, lidar)),
                (t, ns + "/odom", SS.odometry(self.T_mo @ To, "r%d/odom" % self.i, lidar, t))]


def convert(bag, out=None, *, agents=("mobile_1", "mobile_2"), odom="gt", odom_tums=None,
            gt_tums=None, ref_dir=None, start=0.0, duration=0.0, rate=10.0, min_range=0.5,
            voxel=0.5, start_at_gt=False, reorder_s=3.0):
    """odom_tums: {machine: (tum, frame or None)}; a robot without one takes
    `odom` ("gt")."""
    from scoop.bagwrite import BagWriter
    odom_tums, gt_tums = odom_tums or {}, gt_tums or {}
    if odom != "gt":
        raise mc.ConvertError(f"--odom {odom}: only gt (give a method's odometry with --odom-tum)")
    paths = mc._paths(bag)
    label = "+".join(sorted({os.path.splitext(os.path.basename(odom_tums[m][0]))[0]
                             if m in odom_tums else "gt" for m in agents}))
    out = os.path.abspath(os.path.expanduser(out or default_out(bag, label)))
    bag_out = os.path.join(out, "bag")
    if os.path.exists(bag_out):
        raise mc.ConvertError(f"{bag_out} exists: remove it first")
    edges = mc.build_tf_static(paths)
    robots = [Robot(i, m, paths, edges, gt_tums.get(m) or default_gt(bag, m, ref_dir),
                    *(odom_tums.get(m) or (None, None)))
              for i, m in enumerate(agents)]
    os.makedirs(out, exist_ok=True)
    for r in robots:
        log.info("r%d = %s: %s (%s); odometry %s" % (r.i, r.machine, r.topic, r.lidar, r.odom_src))
    by_topic = {r.topic: r for r in robots}
    period = int(round(1e9 / rate))
    t_first = t_lo = t_hi = None
    w = BagWriter(bag_out)
    heap, seq, top, last_out = [], 0, -1, -1
    for m in mc.read(paths, list(by_topic)):
        r, t = by_topic[m.channel.topic], mc.stamp_ns(m.ros_msg)
        if t_first is None:
            t_first = t
            t_lo = t_first + int(round(start * 1e9))
            t_hi = t_lo + int(round(duration * 1e9)) if duration > 0 else None
        if t < t_lo or (t_hi is not None and t > t_hi):
            continue
        if r.last is not None and t - r.last < 0.9 * period:
            continue
        recs = r.frame(t, m.ros_msg, min_range, start_at_gt)
        if not recs:
            continue
        r.last = t
        for tr, topic, msg in recs:
            heapq.heappush(heap, (tr, seq, topic, msg))
            seq += 1
        top = max(top, t)
        while heap and heap[0][0] <= top - int(reorder_s * 1e9):
            tr, _, topic, msg = heapq.heappop(heap)
            if tr >= last_out:
                w.write(topic, msg, tr)
                last_out = tr
        if sum(x.n for x in robots) % 1000 == 0:
            log.info("  %s scans" % " + ".join(str(x.n) for x in robots))
    while heap:
        tr, _, topic, msg = heapq.heappop(heap)
        if tr >= last_out:
            w.write(topic, msg, tr)
            last_out = tr
    w.close()
    for r in robots:
        mc.write_tum(os.path.join(out, "gt_r%d.tum" % r.i), r.gt_lines)
    open(os.path.join(out, "scoop_lidar.yaml"), "w").write(
        CONFIG % {"voxel": voxel, "logs": os.path.join(out, "results")})
    lines = ["bag                 : %s" % bag_out,
             "robots              : %s" % ", ".join("r%d = %s" % (r.i, r.machine) for r in robots),
             "slice               : %s" % ("whole pass" if not (start or duration) else
                                           "from %g s, %s" % (start, "%g s" % duration if duration
                                                              else "to the end")),
             "min range, voxel    : %g m, %g m" % (min_range, voxel)]
    for r in robots:
        lines.append("r%d %-16s: %d scans at <= %g Hz (%s, %s), %.0f points per scan; odometry %s, "
                     "starting in %s; ground truth %s; dropped %s"
                     % (r.i, r.machine, r.n, rate, r.topic, r.lidar, r.points / max(r.n, 1),
                        r.odom_src, r.start, r.gt_src, r.drop))
    report = "\n".join(lines) + "\n"
    open(os.path.join(out, "report.txt"), "w").write(report)
    log.info("\n" + report)
    if not all(r.n for r in robots):
        raise mc.ConvertError("a robot has no scans in the slice with both odometry and ground "
                              "truth: see %s" % os.path.join(out, "report.txt"))
    return {"out": out, "frames": [r.n for r in robots], "robots": robots}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bag", help="the finalized merged bag (has the Ouster clouds and /tf_static)")
    ap.add_argument("out", nargs="?", default=None)
    ap.add_argument("--agents", nargs="+", default=["mobile_1", "mobile_2"], choices=sorted(POINTS))
    ap.add_argument("--odom", choices=["gt"], default="gt",
                    help="the odometry of a robot without --odom-tum")
    ap.add_argument("--odom-tum", nargs="+", action="append", default=[],
                    metavar="MACHINE TUM [FRAME]",
                    help="a robot's odometry: TUM of FRAME (default the LiDAR's) in its odom frame")
    ap.add_argument("--ref-dir", default=None,
                    help="stage 04's folder (default data/processed/<date>/<pass>/odometry/reference_<tag>/)")
    ap.add_argument("--gt-tum", nargs=2, action="append", default=[], metavar=("MACHINE", "TUM"),
                    help="a robot's ground truth, T_map_lidar (default <ref-dir>/traj_<machine>_lidar.tum)")
    ap.add_argument("--start", type=float, default=0.0, help="s from the first scan")
    ap.add_argument("--duration", type=float, default=0.0, help="s (0: to the end)")
    ap.add_argument("--rate", type=float, default=10.0, help="most scans per second per robot")
    ap.add_argument("--min-range", type=float, default=0.5, help="points closer are left out (m)")
    ap.add_argument("--voxel", type=float, default=0.5,
                    help="cslam's frontend.voxel_size (m; ouster_lidar.yaml's 0.5)")
    ap.add_argument("--start-at-gt", action="store_true",
                    help="each robot's odometry starts in map at its ground truth")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    odom_tums = {}
    for v in a.odom_tum:
        if len(v) not in (2, 3) or v[0] not in POINTS:
            ap.error("--odom-tum MACHINE TUM [FRAME], MACHINE one of %s: got %s" % (sorted(POINTS), v))
        odom_tums[v[0]] = (v[1], v[2] if len(v) == 3 else None)
    try:
        convert(a.bag, a.out, agents=a.agents, odom=a.odom, odom_tums=odom_tums,
                gt_tums=dict(a.gt_tum), ref_dir=a.ref_dir, start=a.start, duration=a.duration,
                rate=a.rate, min_range=a.min_range, voxel=a.voxel, start_at_gt=a.start_at_gt)
    except mc.ConvertError as e:
        sys.exit(str(e))


if __name__ == "__main__":
    main()
