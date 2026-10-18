#!/usr/bin/env python
"""Self-test for check.py walls (analysis/maps.py): a synthetic 8 x 6 m room
scanned from a robot driving a circle in it (10 Hz, 40 s), placed by stage
01's own iter_world_scans. Clean points: the scans as placed are as thin as
the map. Points with a 2 % range scale error: where a scan puts a wall
moves away from the sensor with distance, which the distance table must
show. Needs numpy, scipy, open3d, rosbags, mcap.

    python tests/test_wall_build.py
"""
import contextlib
import io
import json
import os
import shutil
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "map_stages"))
sys.path.insert(0, os.path.join(ROOT, "analysis"))

import open3d as o3d                                                   # noqa: E402

import pipeline_common as pc                                           # noqa: E402
import maps                                                            # noqa: E402
from scoop import rosmsg                                               # noqa: E402
from scoop.bagwrite import BagWriter                                   # noqa: E402

TOPIC = "/mobile_2/ouster/points"
T0 = 1_800_000_000


def walls(n, rng):
    """n points on the room's four walls (x 0..8, y 0..6, z 0.2..2.3)."""
    side = rng.integers(0, 4, n)
    u, z = rng.random(n), 0.2 + 2.1 * rng.random(n)
    x = np.where(side == 0, 0.0, np.where(side == 1, 8.0, 8 * u))
    y = np.where(side == 2, 0.0, np.where(side == 3, 6.0, 6 * u))
    return np.c_[x, y, z]


def pose(s):
    a = 2 * np.pi * s / 20.0                        # one circle per 20 s
    t = np.array([4 + 1.5 * np.cos(a), 3 + 1.5 * np.sin(a), 0.5])
    yaw = a + np.pi / 2
    R = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    return R, t, yaw


def run(d, scale):
    rng = np.random.default_rng(0)
    bag = os.path.join(d, "bag_%g" % scale)
    traj = os.path.join(d, "traj_%g.txt" % scale)
    placed = []
    with BagWriter(bag) as w, open(traj, "w") as ft:
        for k in range(400):
            s = k / 10
            R, t, yaw = pose(s)
            P = walls(6000, rng)
            ps = (P - t) @ R                                 # world -> sensor
            r = np.linalg.norm(ps, axis=1, keepdims=True)
            ps = ps * (1 + scale) + ps / r * rng.normal(0, 0.005, (len(ps), 1))
            placed.append(ps @ R.T + t)
            pts = np.zeros(len(ps), [("x", "<f4"), ("y", "<f4"), ("z", "<f4")])
            pts["x"], pts["y"], pts["z"] = ps.T
            ns = (T0 + k // 10) * 10**9 + (k % 10) * 10**8
            w.write(TOPIC, rosmsg.pointcloud2(pts, ns, "os_lidar_mobile2"), ns)
            q = (0, 0, np.sin(yaw / 2), np.cos(yaw / 2))
            ft.write("%.9f %.6f %.6f %.6f %.6f %.6f %.6f %.6f\n" % (ns * 1e-9, *t, *q))
    m = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.vstack(placed)))
    mp = os.path.join(d, "map_%g.pcd" % scale)
    o3d.io.write_point_cloud(mp, m.voxel_down_sample(0.02))
    cfg = pc.load_config(os.path.join(ROOT, "map_stages", "pipeline_config_mapping_20261015.json"))
    cfg.pop("extends", None)
    cfg["dataset"].update(bag=bag, traj=traj, points_topic=TOPIC, out_dir=os.path.join(d, "out"))
    cp = os.path.join(d, "cfg_%g.json" % scale)
    json.dump(cfg, open(cp, "w"))
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        r = maps.measure(cp, mp, n_probes=60)
    return r, out.getvalue()


def main():
    d = tempfile.mkdtemp(prefix="wall_build_")
    fails = []
    try:
        r, log = run(d, 0.0)
        line = (f"clean: one scan {r['single'] * 100:.1f}, whole run {r['whole'] * 100:.1f}, "
                f"map {r['map'] * 100:.1f} cm")
        if not (r["whole"] < 0.03 and abs(r["map"] - r["whole"]) < 0.02
                and "adds nothing" in log):
            fails.append(line + "\n" + log)
        else:
            print("  ok    " + line)
        r, log = run(d, 0.02)
        bd = r["by_dist"]
        trend = bd[-1][2] - bd[0][2]
        line = (f"2 % range error: whole run {r['whole'] * 100:.1f} cm, wall placed "
                f"{trend * 100:+.1f} cm further at {bd[-1][0]:g}+ m than at {bd[0][0]:g}-{bd[0][1]:g} m")
        if not (trend > 0.03 and r["whole"] > 0.05):
            fails.append(line + "\n" + log)
        else:
            print("  ok    " + line)
    finally:
        shutil.rmtree(d, ignore_errors=True)
    for f in fails:
        print("  FAIL  " + f)
    print("wall build ok" if not fails else "wall build FAILED")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
