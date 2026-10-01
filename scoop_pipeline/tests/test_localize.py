#!/usr/bin/env python
"""Self-test for map_stages/08_localize.py + 01a_refine_poses.py with a
reference pass (a run localized in another pass's map).

A 20 x 14 x 3.5 m room with five boxes of different sizes placed so that no
rotation of the room maps it onto itself. The reference pass's map is the
room's surfaces; the run is a 40 s loop scanned by a 32 x 360 lidar at 10 Hz
(ray-cast, so walls behind boxes are hidden). Its "GLIM" trajectory is in its
own world -- the reference world moved by 6.3 m and turned 137 deg -- and
drifts 0.4 deg and 4 cm over the run. 08 must find the world-to-world
transform; 01a, seeded from it, must put every scan back on the true pose.

    python scoop_pipeline/tests/test_localize.py
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
BOXES = np.array([                                   # [min], [max]
    [[-8.0, 4.0, 0.0], [-6.5, 6.0, 1.2]],
    [[3.0, -6.0, 0.0], [3.6, -3.5, 3.5]],
    [[6.0, 2.0, 0.0], [8.5, 3.0, 0.8]],
    [[-2.0, -1.0, 0.0], [-1.2, -0.2, 2.0]],
    [[1.0, 5.2, 0.0], [4.5, 7.0, 2.4]],
])
H, W, DUR, Z = 32, 360, 40.0, 0.6


def true_pose(t):
    """Lidar pose in the reference world at t (s): an ellipse around the room."""
    a = 2 * np.pi * t / DUR
    x, y = 5.5 * np.cos(a), 3.6 * np.sin(a) - 0.4
    yaw = np.arctan2(3.6 * np.cos(a), -5.5 * np.sin(a))
    T = np.eye(4)
    T[:2, :2] = [[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]]
    T[:3, 3] = [x, y, Z]
    return T


def rotz(deg, t=(0, 0, 0)):
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    T = np.eye(4)
    T[:2, :2] = [[c, -s], [s, c]]
    T[:3, 3] = t
    return T


T_REF_RUN = rotz(137.0, (6.3, -2.1, 0.15))          # reference world <- run world


def cast(o, d):
    """Range along unit rays d from o: inside the room, outside the boxes."""
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = 1.0 / d
        t1, t2 = (ROOM[0] - o) * inv, (ROOM[1] - o) * inv
        r = np.min(np.maximum(t1, t2), axis=1)                    # room exit
        for lo, hi in BOXES:
            a, b = (lo - o) * inv, (hi - o) * inv
            tn = np.max(np.minimum(a, b), axis=1)
            tf = np.min(np.maximum(a, b), axis=1)
            hit = (tn <= tf) & (tn > 0)
            r = np.where(hit & (tn < r), tn, r)
    return r


CLOUD = np.dtype({"names": ["x", "y", "z", "t"], "formats": ["<f4", "<f4", "<f4", "<u4"],
                  "offsets": [0, 4, 8, 12], "itemsize": 16})


def scan(T, rng, stamp_ns):
    az = np.linspace(np.pi, -np.pi, W, endpoint=False)
    el = np.radians(np.linspace(22.5, -22.5, H))
    d = np.stack([np.cos(el)[:, None] * np.cos(az), np.cos(el)[:, None] * np.sin(az),
                  np.repeat(np.sin(el)[:, None], W, 1)], -1).reshape(-1, 3)
    r = cast(T[:3, 3], d @ T[:3, :3].T) + rng.normal(0, 0.005, len(d))
    a = np.zeros(len(d), CLOUD)
    a["x"], a["y"], a["z"] = (d * r[:, None]).T
    return rosmsg.pointcloud2(a, stamp_ns, "os_lidar", height=H, width=W)


def surfaces(step=0.03):
    """Points on the room's and the boxes' faces (the reference map)."""
    out = []
    for lo, hi in [ROOM] + list(BOXES):
        for ax in range(3):
            u, v = [k for k in range(3) if k != ax]
            gu = np.arange(lo[u], hi[u] + 1e-9, step)
            gv = np.arange(lo[v], hi[v] + 1e-9, step)
            U, V = np.meshgrid(gu, gv)
            for c in (lo[ax], hi[ax]):
                p = np.zeros((U.size, 3))
                p[:, u], p[:, v], p[:, ax] = U.ravel(), V.ravel(), c
                out.append(p)
    return np.concatenate(out)


def write_tum(path, ts, Ts):
    from scipy.spatial.transform import Rotation
    q = Rotation.from_matrix(np.array([T[:3, :3] for T in Ts])).as_quat()
    np.savetxt(path, np.c_[ts, [T[:3, 3] for T in Ts], q], fmt="%.9f")


def read_tum(path):
    from scipy.spatial.transform import Rotation
    d = np.loadtxt(path)
    Ts = np.tile(np.eye(4), (len(d), 1, 1))
    Ts[:, :3, :3] = Rotation.from_quat(d[:, 4:8]).as_matrix()
    Ts[:, :3, 3] = d[:, 1:4]
    return d[:, 0], Ts


def errors(ts, Ts):
    pe, ye = [], []
    for t, T in zip(ts, Ts):
        G = true_pose(t - T0)
        pe.append(np.linalg.norm(T[:3, 3] - G[:3, 3]))
        dR = T[:3, :3] @ G[:3, :3].T
        ye.append(np.degrees(np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1))))
    return np.array(pe) * 100, np.array(ye)


def run(script, cfg):  # noqa: D103
    out = subprocess.run([sys.executable, os.path.join(STAGES, script), cfg],
                         capture_output=True, text=True, cwd=STAGES)
    print(out.stdout[-3000:])
    if out.returncode:
        raise AssertionError(out.stdout[-2000:] + out.stderr[-3000:])
    return out.stdout


def main():
    import open3d as o3d
    rng = np.random.default_rng(3)
    tmp = tempfile.mkdtemp()
    data = os.path.join(tmp, "data")
    ref = os.path.join(data, "processed", "20260101", "mapping_A", "mapping")
    os.makedirs(ref)
    pts = surfaces() + rng.normal(0, 0.003, (1, 3))
    o3d.io.write_point_cloud(os.path.join(ref, "denoised_mapping_A_20260101.pcd"),
                             o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts)))

    work = os.path.join(data, "work", "20260101", "survey_1")
    bag = os.path.join(work, "x_survey_1_20260101_merged")
    stamps = np.arange(0.0, DUR, 0.1)
    w = BagWriter(bag)
    for t in stamps:
        ns = (T0 * S) + int(round(t * S))
        w.write("/mobile_1/ouster/points", scan(true_pose(t), rng, ns), ns)
    w.close()
    glim = os.path.join(work, "mobile_1", "glim")
    os.makedirs(glim)
    T_run_ref = np.linalg.inv(T_REF_RUN)
    drift = [rotz(0.4 * t / DUR, (0.04 * t / DUR, 0, 0)) for t in stamps]
    write_tum(os.path.join(glim, "traj_lidar.txt"), T0 + stamps,
              [T_run_ref @ D @ true_pose(t) for D, t in zip(drift, stamps)])

    cfg = os.path.join(tmp, "pipeline_config_survey_1.json")
    json.dump({"dataset": {"bag": bag, "traj": "traj_lidar_seed.txt",
                           "machine": "mobile_1", "reference_pass": "mapping_A"},
               "01_build_map": {"time_tol": 0.04, "lidar_min": 0.3, "lidar_max": 30.0,
                                "deskew": False},
               "01a_refine": {"rounds": 2}}, open(cfg, "w"))
    failed = []
    try:
        run("08_localize.py", cfg)
        od = os.path.join(data, "processed", "20260101", "survey_1", "odometry", "mobile_1")
        rec = json.load(open(os.path.join(data, "processed", "20260101", "survey_1",
                                          "frames", "localize_survey_1_20260101.json")))
        T = np.array(rec["T_ref_run"])
        dt = np.linalg.norm(T[:3, 3] - T_REF_RUN[:3, 3]) * 100
        dR = T[:3, :3] @ T_REF_RUN[:3, :3].T
        da = np.degrees(np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1)))
        print(f"08: world-to-world off by {dt:.1f} cm / {da:.2f} deg")
        if not (dt < 15 and da < 1.0):
            failed.append(f"08 transform off by {dt:.1f} cm / {da:.2f} deg")
        ts, Ts = read_tum(os.path.join(od, "traj_lidar_seed.txt"))
        pe, ye = errors(ts, Ts)
        print(f"seed:   position err median {np.median(pe):.2f} cm, max {pe.max():.2f} cm; "
              f"rotation max {ye.max():.3f} deg")

        run("01a_refine_poses.py", cfg)
        ts, Ts = read_tum(os.path.join(od, "traj_lidar_refined.txt"))
        pe, ye = errors(ts, Ts)
        print(f"01a:    position err median {np.median(pe):.2f} cm, p95 "
              f"{np.percentile(pe, 95):.2f} cm, max {pe.max():.2f} cm; rotation median "
              f"{np.median(ye):.3f} deg, max {ye.max():.3f} deg")
        if not (np.median(pe) < 1.0 and np.percentile(pe, 95) < 2.0 and np.median(ye) < 0.1):
            failed.append(f"01a refined poses off: median {np.median(pe):.2f} cm, "
                          f"p95 {np.percentile(pe, 95):.2f} cm, rot {np.median(ye):.3f} deg")
    except AssertionError as e:
        failed.append(str(e)[-1500:])
    shutil.rmtree(tmp, ignore_errors=True)
    print("\n".join(failed) or "localization ok")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
