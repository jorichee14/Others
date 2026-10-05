#!/usr/bin/env python3
"""A robot's visual-inertial odometry, offline: its stereo pair and IMU out of
the merged pass bag, through Isaac ROS cuVSLAM (processing/run_vslam.sh, here
or in the isaac_ros container), for a robot whose VSLAM was not recorded.

    python scoop_pipeline/processing/vslam.py <pipeline config> [--redo] [--rate R] [--imu]

--imu turns IMU fusion on for this run (the config's enable_imu_fusion
otherwise); --rate plays the bag slower (0.5: half speed), so the replay
itself drops no frames.

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
# isaac_ros_visual_slam's RealSense example (D435i IMU noise)
IMU_NOISE = {"gyro_noise_density": 0.000244, "gyro_random_walk": 0.000019393,
             "accel_noise_density": 0.001862, "accel_random_walk": 0.003,
             "calibration_frequency": 200.0}

LAUNCH = '''# written by scoop_pipeline/processing/vslam.py
from launch import LaunchDescription
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode

PARAMS = %r
REMAP = %r


def generate_launch_description():
    node = ComposableNode(name="visual_slam_node", package="isaac_ros_visual_slam",
                          plugin="nvidia::isaac_ros::visual_slam::VisualSlamNode",
                          parameters=[PARAMS], remappings=REMAP)
    return LaunchDescription([ComposableNodeContainer(
        name="visual_slam_launch_container", namespace="", package="rclcpp_components",
        executable="component_container", composable_node_descriptions=[node],
        output="screen")])
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


def remaps(c):
    (i0, i1), (c0, c1) = c["images"], c["camera_infos"]
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
    ap.add_argument("--redo", action="store_true", help="run cuVSLAM again")
    ap.add_argument("--rate", type=float, default=None, help="ros2 bag play --rate")
    ap.add_argument("--imu", action="store_true", help="IMU fusion on (gyro + accel)")
    a = ap.parse_args()
    from pipeline_common import load_pipeline
    P = load_pipeline(a.config)
    c = settings(P)
    if a.imu:
        c["enable_imu_fusion"] = True
    bag = os.path.expanduser(P.dataset["bag"])
    out = out_dir(P, c["machine"])
    rec = os.path.join(out, "odometry")
    traj = traj_path(P, c["machine"])
    print("bag:  %s\nout:  %s\nIMU fusion: %s" % (bag, out, "on" if c["enable_imu_fusion"] else "off"))
    if a.redo and os.path.isdir(out):
        shutil.rmtree(out)
    if not os.path.isfile(os.path.join(rec, "metadata.yaml")):
        check(bag, c)
        part = os.path.join(os.path.dirname(out), ".partial", "vslam")
        if os.path.isdir(part):
            shutil.rmtree(part)
        os.makedirs(part)
        launch = os.path.join(part, "vslam.launch.py")
        with open(launch, "w") as fh:
            fh.write(LAUNCH % (node_params(c), remaps(c)))
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
