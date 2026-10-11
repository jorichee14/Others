#!/usr/bin/env python3
"""A robot's visual-inertial odometry, offline: its stereo pair and IMU out of
the merged pass bag, through Isaac ROS cuVSLAM (processing/run_vslam.sh, here
or in the isaac_ros container), for a robot whose VSLAM was not recorded.

    python scoop_pipeline/processing/vslam.py <pipeline config> [--machine mobile_1|mobile_2]
        [--redo] [--rate R] [--imu] [--ground]

--machine: that robot's preset (MACHINES: mobile_1 the ZED's stereo + IMU,
mobile_2 the RealSense's) instead of the config's "vslam" block. Images that
cuVSLAM does not take (the ZED's bgra8) go through a small converter to mono8
first (to_mono.py, written next to the launch file), header kept.

--imu turns IMU fusion on for this run (the config's enable_imu_fusion
otherwise); --ground keeps the pose on the ground plane (a ground robot:
cuVSLAM's enable_ground_constraint_in_odometry / _in_slam), which holds the
IMU's drift through a camera dropout to the floor's plane; --rate plays the
bag slower (0.5: half speed), so the replay itself drops no frames. --redo
keeps the run before it as <machine>/vslam_prev/ (replacing an older one).

The "vslam" block of the pipeline config says what to play (defaults: mobile_2's
RealSense D435i, as isaac_ros_visual_slam's RealSense example):

  "vslam": {"machine": "mobile_2",
            "images":       ["/mobile_2/infra1/image_rect_raw", "/mobile_2/infra2/image_rect_raw"],
            "camera_infos": ["/mobile_2/infra1/camera_info", "/mobile_2/infra2/camera_info"],
            "imu": "/mobile_2/imu",
            "base_frame": "camera_link",
            "camera_optical_frames": ["camera_infra1_optical_frame", "camera_infra2_optical_frame"],
            "imu_frame": "camera_gyro_optical_frame",
            "enable_imu_fusion": false, "rate": 1.0,
            "params": {...}}        # any other cuVSLAM parameter

Writes data/work/<date>/<pass>/<machine>/vslam/:
  odometry/          the recording (/visual_slam/tracking/odometry, status), logs
  vslam.launch.py    the launch file it ran
  traj_vslam.txt     T_odom_base per odometry message (TUM, base = base_frame,
                     the robot's root frame), on the images' header stamps
  summary.json       poses, duration, path, gaps

Stage 04 registers it to the reference map ("04_reference" "tracks").
"""
import argparse
import json
import os
import shutil
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "map_stages"))

RUN = os.path.join(ROOT, "processing", "run_vslam.sh")
ODOM = "/visual_slam/tracking/odometry"
DEFAULTS = {
    "machine": "mobile_2",
    "images": ["/mobile_2/infra1/image_rect_raw", "/mobile_2/infra2/image_rect_raw"],
    "camera_infos": ["/mobile_2/infra1/camera_info", "/mobile_2/infra2/camera_info"],
    "imu": "/mobile_2/imu",
    "base_frame": "camera_link",
    "camera_optical_frames": ["camera_infra1_optical_frame", "camera_infra2_optical_frame"],
    "imu_frame": "camera_gyro_optical_frame",
    "enable_imu_fusion": False,
    "rate": 1.0,
    "params": {},
}
# per robot: what --machine selects
MACHINES = {
    "mobile_2": dict(DEFAULTS),
    "mobile_1": dict(DEFAULTS, machine="mobile_1",
                     images=["/mobile_1/zed/left/image_rect_color",
                             "/mobile_1/zed/right/image_rect_color"],
                     camera_infos=["/mobile_1/zed/left/camera_info",
                                   "/mobile_1/zed/right/camera_info"],
                     imu="/mobile_1/zed/imu/data",
                     base_frame="zed_camera_link",
                     camera_optical_frames=["zed_left_camera_optical_frame",
                                            "zed_right_camera_optical_frame"],
                     imu_frame="zed_imu_link"),
}
# what cuVSLAM takes as is; anything else is converted to mono8
CUVSLAM_ENCODINGS = ("mono8", "rgb8")
# isaac_ros_visual_slam's RealSense example (D435i IMU noise)
IMU_NOISE = {"gyro_noise_density": 0.000244, "gyro_random_walk": 0.000019393,
             "accel_noise_density": 0.001862, "accel_random_walk": 0.003,
             "calibration_frequency": 200.0}

LAUNCH = '''# written by scoop_pipeline/processing/vslam.py
import os

from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode

PARAMS = %r
REMAP = %r
CONVERT = %r      # [(topic in, mono8 topic out)]: to_mono.py, next to this file


def generate_launch_description():
    node = ComposableNode(name="visual_slam_node", package="isaac_ros_visual_slam",
                          plugin="nvidia::isaac_ros::visual_slam::VisualSlamNode",
                          parameters=[PARAMS], remappings=REMAP)
    conv = [ExecuteProcess(cmd=["python3", os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                                        "to_mono.py")] + [t for p in CONVERT for t in p],
                           output="screen")] if CONVERT else []
    return LaunchDescription(conv + [ComposableNodeContainer(
        name="visual_slam_launch_container", namespace="", package="rclcpp_components",
        executable="component_container", composable_node_descriptions=[node],
        output="screen")])
'''

TO_MONO = '''# written by scoop_pipeline/processing/vslam.py: images -> mono8, header kept
#   python3 to_mono.py <in> <out> [<in> <out> ...]
import sys

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

CH = {"mono8": 1, "bgr8": 3, "rgb8": 3, "bgra8": 4, "rgba8": 4}


class ToMono(Node):
    def __init__(self, pairs):
        super().__init__("vslam_to_mono")
        self.subs = []
        for src, dst in pairs:
            pub = self.create_publisher(Image, dst, 10)
            self.subs.append(self.create_subscription(Image, src, lambda m, p=pub: self.cb(m, p), 10))

    def cb(self, m, pub):
        n = CH[m.encoding]
        a = np.frombuffer(bytes(m.data), np.uint8).reshape(m.height, m.step)[:, :m.width * n]
        a = a.reshape(m.height, m.width, n).astype(np.float32)
        if n == 1:
            y = a[..., 0]
        else:
            r, b = (a[..., 2], a[..., 0]) if m.encoding.startswith("bgr") else (a[..., 0], a[..., 2])
            y = 0.299 * r + 0.587 * a[..., 1] + 0.114 * b
        out = Image(header=m.header, height=m.height, width=m.width, encoding="mono8",
                    is_bigendian=0, step=m.width)
        out.data = np.clip(y + 0.5, 0, 255).astype(np.uint8).tobytes()
        pub.publish(out)


def main():
    a = sys.argv[1:]
    rclpy.init()
    rclpy.spin(ToMono(list(zip(a[0::2], a[1::2]))))


if __name__ == "__main__":
    main()
'''


def settings(P):
    c = dict(DEFAULTS)
    c.update(P.cfg.get("vslam") or {})
    return c


def out_dir(P, machine=None):
    """data/work/<date>/<pass>/<machine>/vslam (next to the merged bag)."""
    bag = os.path.normpath(os.path.expanduser(P.dataset["bag"]))
    return os.path.join(os.path.dirname(bag), machine or settings(P)["machine"], "vslam")


def traj_path(P, machine=None):
    return os.path.join(out_dir(P, machine), "traj_vslam.txt")


def node_params(c):
    """cuVSLAM's parameters, as the launch the robot's own replay uses
    (isaac_ros_visual_slam 3.x): stereo infra pair, no IMU fusion, a jitter
    threshold that lets dropped frames pass (300 ms: the D435i drops some, and
    a tight threshold makes cuVSLAM lose track at each), no GXF memory pool."""
    p = {"use_sim_time": True,
         "enable_image_denoising": False,
         "rectified_images": True,
         "enable_imu_fusion": bool(c["enable_imu_fusion"]),
         "image_jitter_threshold_ms": 300.0,
         "map_frame": "map_vslam", "odom_frame": "odom_vslam",
         "base_frame": c["base_frame"],
         "publish_map_to_odom_tf": True, "publish_odom_to_base_tf": True,
         "imu_frame": c["imu_frame"],
         "enable_slam_visualization": False, "enable_landmarks_view": False,
         "enable_observations_view": False,
         "camera_optical_frames": list(c["camera_optical_frames"]),
         "use_vslam_gxf_memory_pool": False, "disable_gxf_memory_pool": True}
    p.update(IMU_NOISE)
    p.update(c.get("params") or {})
    return p


def mono_topic(t):
    return "/vslam_mono" + t


def image_encoding(bag, topic):
    """The encoding of the first image on `topic`."""
    from scoop import bag as sbag
    for _, _, _, msg in sbag.iter_mcap_records(sbag.open_bag(bag), [topic]):
        c = sbag._Cdr(msg.data)
        c.header_ns()
        c.u32()
        c.u32()
        return c.string().lower()
    raise SystemExit("no messages on %s in %s" % (topic, bag))


def remaps(c, convert=()):
    conv = dict(convert)
    (i0, i1), (c0, c1) = [conv.get(t, t) for t in c["images"]], c["camera_infos"]
    return [("visual_slam/image_0", i0), ("visual_slam/camera_info_0", c0),
            ("visual_slam/image_1", i1), ("visual_slam/camera_info_1", c1),
            ("visual_slam/imu", c["imu"])]


def check(bag, c):
    """The topics and frames cuVSLAM needs are in the bag."""
    from scoop import bag as sbag, tftree
    have = sbag.open_bag(bag).topics()
    want = list(c["images"]) + list(c["camera_infos"]) + ([c["imu"]] if c["enable_imu_fusion"] else [])
    miss = [t for t in want if t not in have]
    if miss:
        raise SystemExit("not in %s: %s" % (bag, ", ".join(miss)))
    edges = tftree.tf_edges([bag], seconds=1.0)
    frames = set(edges) | {e.parent for e in edges.values()}
    need = [c["base_frame"]] + list(c["camera_optical_frames"]) + \
        ([c["imu_frame"]] if c["enable_imu_fusion"] else [])
    miss = [f for f in need if f not in frames]
    if miss:
        raise SystemExit("frames not in the bag's /tf_static: %s (have %s)"
                         % (", ".join(miss), ", ".join(sorted(frames))))
    roots = {tftree._root(f, edges) for f in need}
    if len(roots) != 1:
        raise SystemExit("%s are not in one tree (roots %s)" % (need, sorted(roots)))
    print("  frames: %s, all below %s" % (", ".join(need), roots.pop()))


def to_tum(odom_bag, path, base_frame):
    """The odometry recording -> TUM (T_odom_base) and a summary."""
    from scoop import tftree
    poses, parent, child = tftree.read_pose_topic([odom_bag], ODOM)
    if child and child != base_frame:
        raise SystemExit("odometry is of %s, not %s" % (child, base_frame))
    poses.sort(key=lambda p: p[0])
    t = np.array([p[0] for p in poses], float) * 1e-9
    xyz = np.array([p[1][:3, 3] for p in poses])
    with open(path, "w") as fh:
        for t_ns, T in poses:
            tq = tftree.to_tq(T)
            fh.write("%.9f %.6f %.6f %.6f %.9f %.9f %.9f %.9f\n"
                     % ((t_ns * 1e-9,) + tuple(tq[0]) + tuple(tq[1])))
    dt = np.diff(t)
    step = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
    gaps = [(float(t[i] - t[0]), float(dt[i])) for i in np.flatnonzero(dt > 0.5)]
    jumps = [(float(t[i + 1] - t[0]), float(step[i])) for i in np.flatnonzero(step > 0.3)]
    return {"poses": len(poses), "frame": parent, "base_frame": child or base_frame,
            "t0": float(t[0]), "duration_s": float(t[-1] - t[0]),
            "rate_hz": float(len(t) / max(t[-1] - t[0], 1e-9)),
            "path_m": float(step.sum()), "gaps_over_0.5s": gaps, "jumps_over_0.3m": jumps}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--machine", choices=sorted(MACHINES), default=None,
                    help="that robot's preset instead of the config's vslam block")
    ap.add_argument("--redo", action="store_true", help="run cuVSLAM again")
    ap.add_argument("--rate", type=float, default=None, help="ros2 bag play --rate")
    ap.add_argument("--imu", action="store_true", help="IMU fusion on (gyro + accel)")
    ap.add_argument("--ground", action="store_true", help="pose kept on the ground plane")
    a = ap.parse_args()
    from pipeline_common import load_pipeline
    P = load_pipeline(a.config)
    c = dict(MACHINES[a.machine]) if a.machine else settings(P)
    if a.imu:
        c["enable_imu_fusion"] = True
    if a.ground:
        c["params"] = dict(c.get("params") or {}, enable_ground_constraint_in_odometry=True,
                           enable_ground_constraint_in_slam=True)
    bag = os.path.expanduser(P.dataset["bag"])
    out = out_dir(P, c["machine"])
    rec = os.path.join(out, "odometry")
    traj = traj_path(P, c["machine"])
    ground = bool((c.get("params") or {}).get("enable_ground_constraint_in_odometry"))
    print("bag:  %s\nout:  %s\nIMU fusion: %s, ground constraint: %s"
          % (bag, out, "on" if c["enable_imu_fusion"] else "off", "on" if ground else "off"))
    if a.redo and os.path.isdir(out):
        prev = out + "_prev"
        if os.path.isdir(prev):
            shutil.rmtree(prev)
        shutil.move(out, prev)
        print("the run before: kept as %s" % prev)
    if not os.path.isfile(os.path.join(rec, "metadata.yaml")):
        check(bag, c)
        part = os.path.join(os.path.dirname(out), ".partial", "vslam")
        if os.path.isdir(part):
            shutil.rmtree(part)
        os.makedirs(part)
        launch = os.path.join(part, "vslam.launch.py")
        enc = image_encoding(bag, c["images"][0])
        convert = [] if enc in CUVSLAM_ENCODINGS else [(t, mono_topic(t)) for t in c["images"]]
        if convert:
            print("  images are %s: converted to mono8 for cuVSLAM (%s)"
                  % (enc, ", ".join(d for _, d in convert)))
            with open(os.path.join(part, "to_mono.py"), "w") as fh:
                fh.write(TO_MONO)
        with open(launch, "w") as fh:
            fh.write(LAUNCH % (node_params(c), remaps(c, convert), convert))
        topics = list(c["images"]) + list(c["camera_infos"]) + [c["imu"], "/tf_static"]
        rate = a.rate if a.rate is not None else float(c["rate"])
        code = subprocess.call([RUN, bag, os.path.join(part, "odometry"), launch, str(rate)]
                               + topics)
        if code != 0 or not os.path.isfile(os.path.join(part, "odometry", "metadata.yaml")):
            raise SystemExit("cuVSLAM failed (exit %d); what there is stays in %s" % (code, part))
        if os.path.isdir(out):
            shutil.rmtree(out)
        shutil.move(part, out)
    else:
        print("[skip] cuVSLAM: %s is there (--redo runs it again)" % rec)
    s = to_tum(rec, traj, c["base_frame"])
    json.dump(s, open(os.path.join(out, "summary.json"), "w"), indent=2)
    print("%s: %d poses of %s in %s, %.1f s at %.1f Hz, path %.1f m"
          % (traj, s["poses"], s["base_frame"], s["frame"], s["duration_s"], s["rate_hz"],
             s["path_m"]))
    for at, d in s["gaps_over_0.5s"][:10]:
        print("  ! no odometry for %.1f s at %.1f s (tracking lost?)" % (d, at))
    for at, d in s["jumps_over_0.3m"][:10]:
        print("  ! jump of %.2f m at %.1f s" % (d, at))


if __name__ == "__main__":
    main()
