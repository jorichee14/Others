#!/usr/bin/env python3
"""A multi-robot pass as input for Swarm-SLAM (github.com/MISTLab/Swarm-SLAM,
the cslam packages), RGB-D mode on every robot.

    python scoop_pipeline/datasets/swarm_slam.py <finalized merged bag> [<out dir>]
        [--agents mobile_1 mobile_2] [--odom onboard|gt] [--rate 10] [--width 960]

Writes <out> (default data/processed/<date>/<pass>/datasets/swarm_slam/<pass>_<date>):

  bag/            one ROS 2 bag, robot i under /r<i> (the topic names of
                  cslam_experiments' realsense_rgbd.yaml):
                    /r<i>/color/image_raw                  bgr8, rectified
                    /r<i>/color/camera_info                its intrinsics, no distortion
                    /r<i>/aligned_depth_to_color/image_raw 16UC1 mm, in the colour camera
                    /r<i>/odom                             nav_msgs/Odometry: the colour
                                                           camera's pose in the robot's odom
                  stamped as recorded (play it with --clock)
  scoop_rgbd.yaml the cslam config: realsense_rgbd.yaml's, with logs on (results/)
                  and the CosPlace weights at <out>/resnet18_64.pth
  gt_r<i>.tum     the colour camera in map at every frame written (10's
                  /<robot>/global_pose), for the evaluation
  report.txt

Per robot (AGENTS of replica_multiagent, plus ODOM): the colour image is
undistorted with its CameraInfo and scaled to --width; the depth is put in
that camera (the ZED's depth_registered is resampled, the RealSense's is
back-projected and moved through /tf_static, or the D455 default when the bag
has no link). Frames: a colour/depth pair at most every 1 / --rate s.

Odometry (Swarm-SLAM takes it from outside, matched to the images by stamp):
  onboard  what the robot ran: the ZED's /mobile_1/zed/odom, the RealSense
           robot's cuVSLAM /mobile_2/visual_slam/tracking/odometry, moved to
           the colour camera through /tf_static and interpolated to the image
           stamp; a robot whose odometry was not recorded takes the offline
           cuVSLAM of processing/vslam.py (data/work/<date>/<pass>/<machine>/
           vslam/traj_vslam.txt, of camera_link), when it is there
  gt       the ground truth itself (a check of the setup: no drift to correct)
"""
import argparse
import heapq
import logging
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import mcap_convert as mc                                           # noqa: E402
import replica_multiagent as RM                                     # noqa: E402

log = logging.getLogger("swarm_slam")

ODOM = {"mobile_1": "/mobile_1/zed/odom",
        "mobile_2": "/mobile_2/visual_slam/tracking/odometry"}
VSLAM_BASE = "camera_link"           # processing/vslam.py's base_frame
CONFIG = """/**:
  ros__parameters:
    frontend:
      color_image_topic: "color/image_raw"
      color_camera_info_topic: "color/camera_info"
      depth_image_topic: "aligned_depth_to_color/image_raw"
      odom_topic: "odom"
      sensor_type: "rgbd"
      sensor_base_frame_id: ""
      inter_robot_loop_closure_budget: 1
      inter_robot_detection_period_sec: 1
      max_queue_size: 100
      similarity_threshold: 0.8
      global_descriptors_topic: "global_descriptors"
      image_crop_size: %(crop)d
      intra_loop_min_inbetween_keyframes: 8
      pnp_min_inliers: 100
      detection_publication_period_sec: 1.0
      detection_publication_max_elems_per_msg: 10
      enable_intra_robot_loop_closures: true
      keyframe_generation_ratio_threshold: 0.5
      use_vertex_cover_selection: true
      map_manager_process_period_ms: 100
      enable_sparsification: true
      global_descriptor_technique: "cosplace"
      nn_checkpoint: "%(ckpt)s"
      cosplace:
        backbone: "resnet18"
        descriptor_dim: 64
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
      enable_pose_timestamps_recording: true
      enable_sparsification_comparison: false
    visualization:
      enable: false
      publishing_period_ms: 500
      voxel_size: 0.2
      max_range: 4.0
"""


def T_msg():
    from scoop import rosmsg
    return rosmsg


def camera_info(tgt, frame, t):
    R = T_msg()
    T = R.msg_type
    return T("sensor_msgs/msg/CameraInfo")(
        header=R.header(t, frame), height=int(tgt.height), width=int(tgt.width),
        distortion_model="plumb_bob", d=np.zeros(5),
        k=np.array([tgt.fx, 0, tgt.cx, 0, tgt.fy, tgt.cy, 0, 0, 1.0]), r=np.eye(3).reshape(-1),
        p=np.array([tgt.fx, 0, tgt.cx, 0, 0, tgt.fy, tgt.cy, 0, 0, 0, 1.0, 0]),
        binning_x=0, binning_y=0,
        roi=T("sensor_msgs/msg/RegionOfInterest")(x_offset=0, y_offset=0, height=0, width=0,
                                                  do_rectify=False))


def image(arr, enc, frame, t):
    R = T_msg()
    H, W = arr.shape[:2]
    ch = arr.shape[2] if arr.ndim == 3 else 1
    return R.msg_type("sensor_msgs/msg/Image")(
        header=R.header(t, frame), height=H, width=W, encoding=enc, is_bigendian=0,
        step=W * ch * arr.dtype.itemsize, data=np.frombuffer(np.ascontiguousarray(arr).tobytes(), np.uint8))


def odometry(T_oc, frame, child, t):
    R = T_msg()
    T = R.msg_type
    q = mc.T_to_xyzq(T_oc)
    V = T("geometry_msgs/msg/Vector3")
    return T("nav_msgs/msg/Odometry")(
        header=R.header(t, frame), child_frame_id=child,
        pose=T("geometry_msgs/msg/PoseWithCovariance")(
            pose=T("geometry_msgs/msg/Pose")(
                position=T("geometry_msgs/msg/Point")(x=q[0], y=q[1], z=q[2]),
                orientation=T("geometry_msgs/msg/Quaternion")(x=q[3], y=q[4], z=q[5], w=q[6])),
            covariance=np.zeros(36)),
        twist=T("geometry_msgs/msg/TwistWithCovariance")(
            twist=T("geometry_msgs/msg/Twist")(linear=V(x=0.0, y=0.0, z=0.0),
                                               angular=V(x=0.0, y=0.0, z=0.0)),
            covariance=np.zeros(36)))


class Robot:
    def __init__(self, i, machine, paths, edges, width, odom, depth_extrinsic=None, work_dir=None):
        spec = dict(RM.AGENTS[machine])
        self.a = RM.Agent(i, machine, spec, paths, paths, edges, depth_extrinsic)
        self.i, self.machine, self.spec = i, machine, spec
        K, _ = RM.k_d(self.a.info)
        s = width / float(self.a.info.width)
        H = int(round(self.a.info.height * s))
        self.tgt = mc.Intrinsics(K[0, 0] * s, K[1, 1] * s, (K[0, 2] + 0.5) * s - 0.5,
                                 (K[1, 2] + 0.5) * s - 0.5, int(width), H, [0.0] * 4, None)
        self.cmap = RM.colour_map(self.a.info, self.tgt)
        self.dmap = RM.colour_map(self.a.dinfo, self.tgt) if np.allclose(self.a.T_cd, np.eye(4)) else None
        self.gt = self.a.poses                         # /<robot>/global_pose: colour camera in map
        cf = self.a.frames[0]
        if odom == "gt":
            self.odom, self.T_bc, self.odom_src = self.gt, np.eye(4), spec["pose"] + " (ground truth)"
        else:
            topic = ODOM[machine]
            msg = mc.first_msg(paths, topic)
            vs = os.path.join(work_dir or "", machine, "vslam", "traj_vslam.txt")
            if msg is not None:
                ts, pos, quat, frames = mc.load_poses(paths, topic)
                child = msg.child_frame_id.lstrip("/")
                self.odom = mc.PoseInterp(ts, pos, quat)
                src = topic
            elif work_dir and os.path.exists(vs):
                # not recorded: the offline cuVSLAM of processing/vslam.py (T_odom_base)
                self.odom, child, src = mc.PoseInterp.from_tum(vs), VSLAM_BASE, vs
                log.info(f"{machine}: no {topic} in the bag: odometry from {vs}")
            else:
                raise mc.ConvertError(f"{machine}: no messages on {topic} and no {vs} "
                                      "(processing/vslam.py), or use --odom gt")
            self.T_bc = np.eye(4) if child == cf else mc.lookup_static(edges, child, cf)
            if self.T_bc is None:
                raise mc.ConvertError(f"{machine}: the odometry's frame {child} and the colour frame {cf} "
                                      "are not connected in /tf_static (try --odom gt to check the rest)")
            self.odom_src = f"{src} ({child} -> {cf})"
        self.last = None
        self.gt_lines, self.n, self.drop = [], 0, {"no odom": 0}

    def frame(self, ct, cm, dm, dmin, dmax):
        """[(t, topic, msg)] for one colour/depth pair, or [] (no odometry)."""
        To = self.odom.at(ct)
        if To is None:
            self.drop["no odom"] += 1
            return []
        bgr = mc.to_bgr(mc.decode_image(cm), cm.encoding)
        bgr = cv2.remap(bgr, self.cmap[0], self.cmap[1], cv2.INTER_LINEAR)
        d, _ = mc.depth_to_metres(mc.decode_image(dm), dm.encoding, 0.001, dmin, dmax)
        D = RM.register_depth(d, self.a.dinfo, self.a.T_cd, self.tgt, dmin, dmax, self.dmap)
        mm = np.clip(np.round(D * 1000.0), 0, 65535).astype(np.uint16)
        ns, cam = "/r%d" % self.i, "r%d/camera" % self.i
        G = self.gt.at(ct)
        if G is not None:
            self.gt_lines.append(mc.tum_line(ct, G))
        self.n += 1
        return [(ct, ns + "/color/image_raw", image(bgr, "bgr8", cam, ct)),
                (ct, ns + "/color/camera_info", camera_info(self.tgt, cam, ct)),
                (ct, ns + "/aligned_depth_to_color/image_raw", image(mm, "16UC1", cam, ct)),
                (ct, ns + "/odom", odometry(To @ self.T_bc, "r%d/odom" % self.i, cam, ct))]


def pairs(paths, robots, tol_ns):
    """(robot, t, colour msg, depth msg) over every robot, in bag order: each
    robot's colour and depth matched by stamp within tol (as iter_synced)."""
    by_topic = {}
    for r in robots:
        by_topic[r.spec["color"]] = (r, 0)
        by_topic[r.spec["depth"]] = (r, 1)
    bufs = {r.i: ([], []) for r in robots}
    for m in mc.read(paths, list(by_topic)):
        r, k = by_topic[m.channel.topic]
        a, b = bufs[r.i]
        (a if k == 0 else b).append((mc.stamp_ns(m.ros_msg), m.ros_msg))
        while a and b:
            ta, ma = a[0]
            tb, mb = b[0]
            if ta - tb > tol_ns:
                b.pop(0)
            elif tb - ta > tol_ns:
                a.pop(0)
            else:
                a.pop(0)
                b.pop(0)
                yield r, ta, ma, mb


def default_out(bag):
    return os.path.join(os.path.dirname(os.path.dirname(RM.default_out(bag))), "swarm_slam",
                        os.path.basename(RM.default_out(bag)))


def convert(bag, out=None, *, agents=("mobile_1", "mobile_2"), odom="onboard", rate=10.0, width=960,
            depth_min=0.2, depth_max=10.0, sync_tol_ms=20.0, checkpoint=None, depth_extrinsics=None,
            reorder_s=3.0):
    from scoop.bagwrite import BagWriter
    paths = mc._paths(bag)
    out = os.path.abspath(os.path.expanduser(out or default_out(bag)))
    bag_out = os.path.join(out, "bag")
    if os.path.exists(bag_out):
        raise mc.ConvertError(f"{bag_out} exists: remove it first")
    os.makedirs(out, exist_ok=True)
    edges = mc.build_tf_static(paths)
    b = os.path.abspath(bag[0] if isinstance(bag, (list, tuple)) else bag)
    work_dir = os.path.dirname(os.path.dirname(b) if os.path.isfile(b) else b)   # data/work/<date>/<pass>
    robots = [Robot(i, m, paths, edges, width, odom, (depth_extrinsics or {}).get(m), work_dir)
              for i, m in enumerate(agents)]
    for r in robots:
        log.info("r%d = %s: %dx%d f %.1f; odometry %s" % (r.i, r.machine, r.tgt.width, r.tgt.height,
                                                         r.tgt.fx, r.odom_src))
    period = int(round(1e9 / rate))
    w = BagWriter(bag_out)
    heap, seq, top, last_out = [], 0, -1, -1
    for r, ct, cm, dm in pairs(paths, robots, int(sync_tol_ms * 1e6)):
        if r.last is not None and ct - r.last < 0.9 * period:
            continue
        recs = r.frame(ct, cm, dm, depth_min, depth_max)
        if not recs:
            continue
        r.last = ct
        for t, topic, msg in recs:
            heapq.heappush(heap, (t, seq, topic, msg))
            seq += 1
        top = max(top, ct)
        while heap and heap[0][0] <= top - int(reorder_s * 1e9):
            t, _, topic, msg = heapq.heappop(heap)
            if t >= last_out:
                w.write(topic, msg, t)
                last_out = t
        if sum(x.n for x in robots) % 1000 == 0:
            log.info("  %s frames" % " + ".join(str(x.n) for x in robots))
    while heap:
        t, _, topic, msg = heapq.heappop(heap)
        if t >= last_out:
            w.write(topic, msg, t)
            last_out = t
    w.close()
    for r in robots:
        with open(os.path.join(out, "gt_r%d.tum" % r.i), "w") as fh:
            fh.write("\n".join(r.gt_lines) + "\n")
    crop = min(r.tgt.height for r in robots)
    ckpt = os.path.abspath(checkpoint) if checkpoint else os.path.join(out, "resnet18_64.pth")
    open(os.path.join(out, "scoop_rgbd.yaml"), "w").write(
        CONFIG % {"crop": min(crop, 376), "ckpt": ckpt,
                  "logs": os.path.join(out, "results")})
    lines = ["bag                 : %s" % bag_out,
             "robots              : %s" % ", ".join("r%d = %s" % (r.i, r.machine) for r in robots),
             "odometry            : %s" % odom]
    for r in robots:
        lines.append("r%d %-16s: %d frames at <= %g Hz, %dx%d fx %.2f fy %.2f cx %.1f cy %.1f; "
                     "odometry %s; dropped %s; %d ground-truth poses"
                     % (r.i, r.machine, r.n, rate, r.tgt.width, r.tgt.height, r.tgt.fx, r.tgt.fy,
                        r.tgt.cx, r.tgt.cy, r.odom_src, r.drop, len(r.gt_lines)))
    report = "\n".join(lines) + "\n"
    open(os.path.join(out, "report.txt"), "w").write(report)
    log.info("\n" + report)
    return {"out": out, "frames": [r.n for r in robots], "robots": robots}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bag", help="the finalized merged bag (has /<robot>/global_pose)")
    ap.add_argument("out", nargs="?", default=None)
    ap.add_argument("--agents", nargs="+", default=["mobile_1", "mobile_2"], choices=sorted(ODOM))
    ap.add_argument("--odom", choices=["onboard", "gt"], default="onboard")
    ap.add_argument("--rate", type=float, default=10.0, help="most frames per second per robot")
    ap.add_argument("--width", type=int, default=960, help="image width (px), every robot")
    ap.add_argument("--checkpoint", default=None,
                    help="CosPlace ResNet-18/64 weights (default: <out>/resnet18_64.pth, which "
                         "slam/run_swarm_slam.sh fetches)")
    ap.add_argument("--depth-extrinsic", nargs=8, action="append", default=[],
                    metavar=("MACHINE", "X", "Y", "Z", "QX", "QY", "QZ", "QW"))
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        convert(a.bag, a.out, agents=a.agents, odom=a.odom, rate=a.rate, width=a.width,
                checkpoint=a.checkpoint,
                depth_extrinsics={e[0]: [float(v) for v in e[1:]] for e in a.depth_extrinsic})
    except mc.ConvertError as e:
        sys.exit(str(e))


if __name__ == "__main__":
    main()
