#!/usr/bin/env python3
"""A scenario from a real coop pass: its two robots with their real sensors
and mounts, driving their real trajectories through the scanned building,
plus the actors you add; and the converter config that turns the synthetic
bag into OPV2V frames.

    python3 isaac_coop_sim/make_scenario.py <coop pipeline config> --scene scene.usd --out DIR
        [--walker "x0,y0;x1,y1;..." --walker-speed 1.2]...   # people, looping along the points
        [--chair x,y,yaw_deg]...                              # static furniture, labelled
        [--robot-path mobile_2="x0,y0;x1,y1" --robot-speed 0.6]  # instead of its real trajectory
        [--lidar-config mobile_1=OS1_REV6_64ch10hz1024res] [--duration S] [--seed N]

Writes <out>/scenario.yaml (scenario.py's schema), <out>/opv2v_sim.yaml
(rosbag_to_opv2v's config for the bag run_isaac.py records: the robots as
CAVs with their lidar frame as base, the cameras' extrinsics, the labels
file) and prints the plan. The sensors and mounts are the 20261015
rigs' (SENSORS below): mobile_1's Ouster OS1-64 and ZED 2i at 960 x 540,
mobile_2's Ouster OS1-128 and RealSense D455 at 848 x 480, the mounts from
configs/static_tf.yaml.
"""
import argparse
import json
import os
import sys

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "scoop_pipeline", "map_stages"))
sys.path.insert(0, os.path.join(ROOT, "scoop_pipeline"))

# the rigs as recorded on 20261015: the mounts are configs/static_tf.yaml's, the
# intrinsics the bags' CameraInfo (the ZED rectified, the RealSense undistorted here:
# the renderer makes pinhole images, D = 0)
SENSORS = {
    "mobile_1": {
        "lidar_frame": "os_lidar_mobile1", "lidar_topic": "/mobile_1/ouster/points",
        "lidar_config": "OS1_REV6_64ch10hz1024res", "lidar_beams": 64,
        "box": [0.62, 0.55, 1.25], "box_z": 0.55,               # a cart with the ZED on a mast
        "cameras": [{"name": "zed", "frame": "zed_left_camera_optical_frame",
                     "topic": "/mobile_1/zed/left/image_rect_color",
                     "info_topic": "/mobile_1/zed/left/camera_info",
                     "width": 960, "height": 540, "fx": 544.983, "fy": 544.983, "cx": 489.396, "cy": 266.657,
                     "mount": {"xyz": [-0.074928, -0.066971, -0.091627],
                               "qxyzw": [-0.497829, -0.498035, 0.501789, 0.502329]}}],
    },
    "mobile_2": {
        "lidar_frame": "os_lidar_mobile2", "lidar_topic": "/mobile_2/ouster/points",
        "lidar_config": "OS1_REV6_128ch10hz1024res", "lidar_beams": 128,
        "box": [0.62, 0.55, 1.05], "box_z": 0.45,
        "cameras": [{"name": "realsense", "frame": "camera_color_optical_frame",
                     "topic": "/mobile_2/color/image_raw",
                     "info_topic": "/mobile_2/color/camera_info",
                     "width": 848, "height": 480, "fx": 427.703, "fy": 427.108, "cx": 423.132, "cy": 239.735,
                     "mount": {"xyz": [-0.273322, 0.029175, 0.004183],
                               "qxyzw": [0.552208, 0.540455, -0.450592, -0.447148]}}],
    },
}


def rpy_block(xyz, qxyzw):
    """rosbag_to_opv2v's {x, y, z, roll, pitch, yaw} (degrees, scipy 'xyz')."""
    from scipy.spatial.transform import Rotation
    r, p, y = Rotation.from_quat(qxyzw).as_euler("xyz", degrees=True)
    return {"x": float(xyz[0]), "y": float(xyz[1]), "z": float(xyz[2]),
            "roll": float(r), "pitch": float(p), "yaw": float(y)}


def parse_points(s):
    return [[float(v) for v in p.split(",")] for p in s.split(";") if p.strip()]


def build(cfg_path, scene_usd, out, walkers=(), walker_speed=1.2, chairs=(), robot_paths=None,
          robot_speed=0.6, lidar_configs=None, duration=None, seed=0, log=print):
    from pipeline_common import load_pipeline
    P = load_pipeline(cfg_path)
    out = os.path.expanduser(out)
    os.makedirs(out, exist_ok=True)
    scene_usd = os.path.abspath(os.path.expanduser(scene_usd))
    info_p = os.path.splitext(scene_usd)[0] + ".json"
    floor_z = json.load(open(info_p))["floor_z"] if os.path.exists(info_p) else 0.0
    robot_paths = dict(robot_paths or {})
    lidar_configs = dict(lidar_configs or {})
    robots, dur = [], 0.0
    for name in ("mobile_1", "mobile_2"):
        S = SENSORS[name]
        if name in robot_paths:
            traj = {"waypoints": parse_points(robot_paths[name]), "speed": robot_speed}
            n = len(traj["waypoints"])
            log(f"{name}: a scripted path of {n} points at {robot_speed} m/s")
        else:
            cands = [os.path.join(P.reference_dir(), f"traj_{name}_lidar.tum")]
            tum = next((c for c in cands if os.path.exists(c)), None)
            if tum is None:
                log(f"{name}: no trajectory ({cands[0]} missing: run the pass's 04 first, or --robot-path)")
                continue
            traj = {"tum": tum}
            t = np.loadtxt(tum, usecols=0)
            dur = max(dur, float(t[-1] - t[0]))
            log(f"{name}: replays {tum} ({len(t)} poses, {t[-1] - t[0]:.0f} s)")
        robots.append({"name": name, "base_frame": S["lidar_frame"], "trajectory": traj,
                       "box": S["box"], "box_z": S["box_z"],
                       "lidar": {"config": lidar_configs.get(name, S["lidar_config"]), "beams": S["lidar_beams"],
                                 "frame": S["lidar_frame"], "topic": S["lidar_topic"],
                                 "mount": {"xyz": [0, 0, 0], "qxyzw": [0, 0, 0, 1]}},
                       "cameras": [dict(c) for c in S["cameras"]]})
    actors = []
    for k, w in enumerate(walkers):
        actors.append({"name": f"person_{k + 1}", "kind": "walker", "asset": None, "box": [0.5, 0.5, 1.75],
                       "path": {"waypoints": parse_points(w), "speed": walker_speed, "loop": True, "start_s": 0},
                       "yaw_follows_path": True})
    for k, c in enumerate(chairs):
        x, y, yaw = (parse_points(c)[0] + [0.0])[:3]
        actors.append({"name": f"chair_{k + 1}", "kind": "chair", "asset": None, "box": [0.55, 0.55, 0.9],
                       "path": {"waypoints": [[x, y, yaw]], "speed": 0.0, "loop": True}})
    if duration is None:
        duration = dur if dur > 0 else 60.0
    scen = {"duration_s": round(float(duration), 1), "seed": int(seed),
            "rates": {"lidar_hz": 10, "camera_hz": 15, "sim_hz": 60},
            "scene": {"usd": scene_usd, "floor_z": floor_z},
            "robots": robots, "actors": actors,
            "_note": f"from {os.path.abspath(cfg_path)}: the robots' real trajectories in the map frame "
                     f"(the scene USD must be in it: export_usd.py --frame), the 20261015 rigs' sensors"}
    sp = os.path.join(out, "scenario.yaml")
    yaml.safe_dump(scen, open(sp, "w"), sort_keys=False)
    cp = os.path.join(out, "opv2v_sim.yaml")
    yaml.safe_dump(converter_config(scen, out), open(cp, "w"), sort_keys=False)
    log(f"wrote {sp}\n      {cp}: {len(robots)} robots, {len(actors)} actors, {scen['duration_s']} s, floor z {floor_z}")
    return sp, cp


def converter_config(scen, out):
    """rosbag_to_opv2v's config for the bag run_isaac.py writes: each robot a
    CAV whose base is its lidar frame (pose topic /<robot>/global_pose of
    that frame), cloud extrinsic identity, cameras at their mounts, the
    labels file run_isaac.py writes, time grid on the first robot's lidar."""
    agents = []
    for k, rb in enumerate(scen["robots"]):
        cams = []
        for c in rb["cameras"]:
            cams.append({"topic": c["topic"], "camera_info_topic": c["info_topic"],
                         "extrinsic": rpy_block(c["mount"]["xyz"], c["mount"]["qxyzw"]), "optical_frame": True})
        agents.append({"name": rb["name"], "id": k + 1, "role": "cav", "host": rb["name"], "enabled": True,
                       "required": True,
                       "pose": {"source": "pose", "topic": f"/{rb['name']}/global_pose", "interpolation": "linear",
                                "max_gap_ms": 200,
                                "align": {"x": 0, "y": 0, "z": 0, "roll": 0, "pitch": 0, "yaw": 0},
                                "child_to_base": {"x": 0, "y": 0, "z": 0, "roll": 0, "pitch": 0, "yaw": 0}},
                       "cloud": {"kind": "pointcloud2", "topic": rb["lidar"]["topic"],
                                 "extrinsic": {"x": 0, "y": 0, "z": 0, "roll": 0, "pitch": 0, "yaw": 0},
                                 "intensity": {"field": "intensity", "scale": 1.0, "default": 0.5},
                                 "deskew": False, "max_points": 100000, "ground_lift": 0.0, "min_points": 1},
                       "cameras": cams,
                       "object": {"emit": False}})
    return {"bag": os.path.join(out, "sim_bag"),
            "output": {"root": os.path.join(out, "OPV2V_sim"), "split": "train", "scenario_name": "sim",
                       "frames_per_scenario": 0, "frame_stride": 1, "include_self_in_vehicles": False,
                       "write_provenance": True, "labels_file": os.path.join(out, "labels_dynamic.json")},
            "time": {"stamp_source": "header", "master_agent": scen["robots"][0]["name"] if scen["robots"] else "mobile_1",
                     "rate_hz": 0, "match_tolerance_ms": 40, "drop_incomplete_frames": True,
                     "start_offset_s": 0.0, "duration_s": 0.0},
            "clock": {"enabled": False},
            "world": {"frame_id": "map"},
            "agents": agents,
            "_note": "synthetic run (isaac_coop_sim): one clock, no sync to verify; the labels are per frame "
                     "(labels_dynamic.json, at the lidar stamps), the robots included as objects"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", help="the coop pass's pipeline config")
    ap.add_argument("--scene", required=True, help="scene.usd from export_usd.py (its .json beside it)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--walker", action="append", default=[], metavar="X,Y;X,Y;...")
    ap.add_argument("--walker-speed", type=float, default=1.2)
    ap.add_argument("--chair", action="append", default=[], metavar="X,Y[,YAW]")
    ap.add_argument("--robot-path", action="append", default=[], metavar="ROBOT=X,Y;X,Y;...")
    ap.add_argument("--robot-speed", type=float, default=0.6)
    ap.add_argument("--lidar-config", action="append", default=[], metavar="ROBOT=NAME")
    ap.add_argument("--duration", type=float, default=None)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    build(a.config, a.scene, a.out, a.walker, a.walker_speed, a.chair,
          dict(p.split("=", 1) for p in a.robot_path), a.robot_speed,
          dict(p.split("=", 1) for p in a.lidar_config), a.duration, a.seed)


if __name__ == "__main__":
    main()
