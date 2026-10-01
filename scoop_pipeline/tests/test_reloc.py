#!/usr/bin/env python
"""Self-test for map_stages/lidar_reloc.py through 06_init_from_boards.py.

A 20 x 14 x 3.5 m room with five boxes placed so that no rotation maps it
onto itself. The reference pass (mapping_A) has the room's surfaces as its
anchored map; the run (survey_1) is a robot parked for 6 s, then driving an
ellipse, scanned by a 32 x 360 lidar at 10 Hz (ray-cast, so walls behind the
boxes are hidden). Its GLIM trajectory is in its own world: the map moved by
6.3 m and turned 137 deg. Two cases:

  room with boxes   06 must find the start pose (cm / tenth of a degree of
                    the truth) and pass the fit, unique and repeat checks;
  bare room         symmetric under a half turn: 06 must refuse.

    python scoop_pipeline/tests/test_reloc.py
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
STAGES = os.path.join(ROOT, "map_stages")

from scoop import rosmsg                                            # noqa: E402
from scoop.bagwrite import BagWriter                                # noqa: E402

S = 10**9
T0 = 1_790_000_000
ROOM = np.array([[-10.0, -7.0, 0.0], [10.0, 7.0, 3.5]])
BOXES = np.array([
    [[-8.0, 4.0, 0.0], [-6.5, 6.0, 1.2]],
    [[3.0, -6.0, 0.0], [3.6, -3.5, 3.5]],
    [[6.0, 2.0, 0.0], [8.5, 3.0, 0.8]],
    [[-2.0, -1.0, 0.0], [-1.2, -0.2, 2.0]],
    [[1.0, 5.2, 0.0], [4.5, 7.0, 2.4]],
])
H, W, DUR, PARK, Z = 32, 360, 30.0, 6.0, 0.6


def rotz(deg, t=(0, 0, 0)):
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    T = np.eye(4)
    T[:2, :2] = [[c, -s], [s, c]]
    T[:3, 3] = t
    return T


def true_pose(t):
    """Lidar in the map at t (s): parked PARK s, then an ellipse."""
    a = 2 * np.pi * max(t - PARK, 0.0) / (2 * (DUR - PARK))
    yaw = np.arctan2(3.6 * np.cos(a), -5.5 * np.sin(a))
    return rotz(np.degrees(yaw), (5.5 * np.cos(a), 3.6 * np.sin(a) - 0.4, Z))


T_MAP_RUN = rotz(137.0, (6.3, -2.1, 0.15))          # map <- the run's GLIM world


def cast(o, d, boxes):
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = 1.0 / d
        t1, t2 = (ROOM[0] - o) * inv, (ROOM[1] - o) * inv
        r = np.min(np.maximum(t1, t2), axis=1)
        for lo, hi in boxes:
            a, b = (lo - o) * inv, (hi - o) * inv
            tn = np.max(np.minimum(a, b), axis=1)
            tf = np.min(np.maximum(a, b), axis=1)
            hit = (tn <= tf) & (tn > 0)
            r = np.where(hit & (tn < r), tn, r)
    return r


CLOUD = np.dtype({"names": ["x", "y", "z", "t"], "formats": ["<f4", "<f4", "<f4", "<u4"],
                  "offsets": [0, 4, 8, 12], "itemsize": 16})


def scan(T, rng, stamp_ns, boxes):
    az = np.linspace(np.pi, -np.pi, W, endpoint=False)
    el = np.radians(np.linspace(22.5, -22.5, H))
    d = np.stack([np.cos(el)[:, None] * np.cos(az), np.cos(el)[:, None] * np.sin(az),
                  np.repeat(np.sin(el)[:, None], W, 1)], -1).reshape(-1, 3)
    r = cast(T[:3, 3], d @ T[:3, :3].T, boxes) + rng.normal(0, 0.005, len(d))
    a = np.zeros(len(d), CLOUD)
    a["x"], a["y"], a["z"] = (d * r[:, None]).T
    return rosmsg.pointcloud2(a, stamp_ns, "os_lidar", height=H, width=W)


def surfaces(boxes, step=0.03):
    out = []
    for lo, hi in [ROOM] + list(boxes):
        for ax in range(3):
            u, v = [k for k in range(3) if k != ax]
            U, V = np.meshgrid(np.arange(lo[u], hi[u] + 1e-9, step),
                               np.arange(lo[v], hi[v] + 1e-9, step))
            for c in (lo[ax], hi[ax]):
                p = np.zeros((U.size, 3))
                p[:, u], p[:, v], p[:, ax] = U.ravel(), V.ravel(), c
                out.append(p)
    return np.concatenate(out)


def write_tum(path, ts, Ts):
    from scipy.spatial.transform import Rotation
    q = Rotation.from_matrix(np.array([T[:3, :3] for T in Ts])).as_quat()
    np.savetxt(path, np.c_[ts, [T[:3, 3] for T in Ts], q], fmt="%.9f")


def T_of(rec):
    from scipy.spatial.transform import Rotation
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat(rec["qxyzw"]).as_matrix()
    T[:3, 3] = rec["xyz"]
    return T


def build(tmp, boxes):
    """data/ with mapping_A's anchored map + anchor frame, survey_1's bag and
    GLIM trajectory, and the run's config. -> config path."""
    import open3d as o3d
    rng = np.random.default_rng(5)
    data = os.path.join(tmp, "data")
    ref = os.path.join(data, "processed", "20260101", "mapping_A")
    os.makedirs(os.path.join(ref, "mapping"))
    os.makedirs(os.path.join(ref, "frames"))
    o3d.io.write_point_cloud(
        os.path.join(ref, "mapping", "map_final_mapping_A_20260101_anchored.pcd"),
        o3d.geometry.PointCloud(o3d.utility.Vector3dVector(surfaces(boxes))))
    json.dump({"map_frame": "map", "board_axes": "ros", "board_origin": "center",
               "T_N_world": np.eye(4).tolist(),
               "boards": {"anchor": {"parent": "map", "child": "board", "frame": "board",
                                     "design": "anchor", "xyz": [9.99, 0.0, 1.0],
                                     "qxyzw": [0, 0, 1, 0]}}},
              open(os.path.join(ref, "frames", "anchor_frame.json"), "w"))

    work = os.path.join(data, "work", "20260101", "survey_1")
    bag = os.path.join(work, "x_survey_1_20260101_merged")
    T = rosmsg.msg_type
    w = BagWriter(bag)
    T_lc = rotz(0.0, (0.07, 0.0, -0.09))
    T_lc[:3, :3] = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], float)   # optical in lidar
    q = rosmsg.tf_message([("os_lidar", "zed_left_camera_optical_frame", T_lc[:3, 3],
                            [-0.5, 0.5, -0.5, 0.5])], T0 * S)
    w.write("/tf_static", q, T0 * S)
    info = T("sensor_msgs/msg/CameraInfo")(
        header=rosmsg.header(T0 * S, "zed_left_camera_optical_frame"), height=540, width=960,
        distortion_model="plumb_bob", d=np.zeros(5),
        k=np.array([545.0, 0, 480, 0, 545.0, 270, 0, 0, 1]), r=np.eye(3).reshape(-1),
        p=np.array([545.0, 0, 480, 0, 0, 545.0, 270, 0, 0, 0, 1, 0]), binning_x=0, binning_y=0,
        roi=T("sensor_msgs/msg/RegionOfInterest")(x_offset=0, y_offset=0, height=0, width=0,
                                                  do_rectify=False))
    w.write("/mobile_1/zed/left/camera_info", info, T0 * S + 1)
    stamps = np.arange(0.05, DUR, 0.1)
    for t in stamps:
        ns = T0 * S + int(round(t * S))
        w.write("/mobile_1/ouster/points", scan(true_pose(t), rng, ns, boxes), ns)
    w.close()
    glim = os.path.join(work, "mobile_1", "glim")
    os.makedirs(glim)
    T_run_map = np.linalg.inv(T_MAP_RUN)
    write_tum(os.path.join(glim, "traj_lidar.txt"), T0 + stamps,
              [T_run_map @ true_pose(t) for t in stamps])

    cfg = os.path.join(tmp, "pipeline_config_survey_1.json")
    json.dump({
        "dataset": {"bag": bag, "machine": "mobile_1", "reference_pass": "mapping_A"},
        "boards": {"anchor": {"frame": "board", "squares_x": 9, "squares_y": 7,
                              "square_len": 0.02, "marker_len": 0.015,
                              "dictionary": "DICT_4X4_50", "min_corners": 8,
                              "max_reproj": 1.5, "min_ambiguity_ratio": 1.5}},
        "01_build_map": {"time_tol": 0.04, "lidar_min": 0.3, "lidar_max": 30.0,
                         "deskew": False},
        "06_init": {"anchor_frame": "anchor_frame.json", "output": "session_anchor.json",
                    "map_frame": "map",
                    "sensors": [{"name": "zed", "board": "anchor", "source": "camera",
                                 "cam_frame": "zed_left_camera_optical_frame",
                                 "image_topic": "/mobile_1/zed/left/image_rect_color",
                                 "camera_info_topic": "/mobile_1/zed/left/camera_info",
                                 "rectified": True}],
                    "relocalize": {"enabled": True, "window_s": 3.0, "check_after_s": 14.0}}},
        open(cfg, "w"))
    return cfg, data, T_lc


def run06(cfg):
    return subprocess.run([sys.executable, os.path.join(STAGES, "06_init_from_boards.py"), cfg],
                          capture_output=True, text=True, cwd=STAGES)


def board_check_case():
    """The board dwell compared with the LiDAR start: 0 when they agree,
    the offset when a board fix is off by 3 cm."""
    sys.path.insert(0, STAGES)
    from lidar_reloc import board_check
    from pipeline_common import traj_quats
    ts = T0 + np.arange(0.05, 8.0, 0.1)
    T_run_map = np.linalg.inv(T_MAP_RUN)
    tr_T = np.array([T_run_map @ true_pose(t - T0) for t in ts])
    r = {"tr": (ts, tr_T, traj_quats(tr_T)), "T_map_glim": T_MAP_RUN}
    T_lc = rotz(0.0, (0.07, 0.0, -0.09))
    T_lc[:3, :3] = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], float)
    hits = [{"t": t, "T_map_cam": true_pose(t - T0) @ T_lc} for t in ts[5:40]]
    cm, deg, n = board_check(r, hits, T_lc)
    off = [dict(h, T_map_cam=rotz(0, (0.03, 0, 0)) @ h["T_map_cam"]) for h in hits]
    cm2, _, _ = board_check(r, off, T_lc)
    print(f"board check: agreeing {cm:.3f} cm / {deg:.4f} deg (n={n}); 3 cm off -> {cm2:.2f} cm")
    return [] if (cm < 0.01 and deg < 0.01 and abs(cm2 - 3.0) < 0.01) else \
        [f"board check: {cm} {deg} {cm2}"]


def main():
    failed = board_check_case()
    tmp = tempfile.mkdtemp()
    try:
        cfg, data, T_lc = build(os.path.join(tmp, "boxes"), BOXES)
        out = run06(cfg)
        print(out.stdout[-4000:])
        if out.returncode:
            failed.append("room with boxes: 06 failed\n" + out.stdout[-1500:] + out.stderr[-2000:])
        else:
            sa = json.load(open(os.path.join(data, "processed", "20260101", "survey_1",
                                             "frames", "session_anchor.json")))
            z = sa["cameras"]["zed"]
            T_l = T_of(z["map_to_lidar"])
            G = true_pose(z["dwell_t_end"] - T0)
            dcm = np.linalg.norm(T_l[:3, 3] - G[:3, 3]) * 100
            ddeg = np.degrees(np.arccos(np.clip((np.trace(T_l[:3, :3].T @ G[:3, :3]) - 1) / 2,
                                                -1, 1)))
            Tc = T_of(z["map_to_cam"])
            cerr = np.linalg.norm(Tc[:3, 3] - (G @ T_lc)[:3, 3]) * 100
            print(f"start pose vs truth: lidar {dcm:.2f} cm / {ddeg:.3f} deg, camera {cerr:.2f} cm")
            if not (dcm < 2.0 and ddeg < 0.2 and cerr < 2.5):
                failed.append(f"start pose off: {dcm:.2f} cm / {ddeg:.3f} deg (camera {cerr:.2f})")
            ch = z["checks"]
            if not (ch["fitness"] > 0.8 and ch.get("repeat_cm", 99) < 3.0):
                failed.append(f"checks: {ch}")
            if not os.path.exists(z["look"]):
                failed.append("no look ply")

        cfg, data, _ = build(os.path.join(tmp, "bare"), [])
        out = run06(cfg)
        print(out.stdout[-2500:])
        refused = out.returncode != 0 and "refused" in (out.stdout + out.stderr)
        print("bare room: %s" % ("refused" if refused else "ACCEPTED"))
        if not refused:
            failed.append("bare (symmetric) room was not refused")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("\n".join(failed) or "relocalization ok")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
