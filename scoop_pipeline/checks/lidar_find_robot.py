#!/usr/bin/env python3
"""Where is mobile_2 in mobile_1's LiDAR? Every `--every` s over the whole run:
mobile_1's scan in map, the points the reference map does not have, at robot
height (the floor under mobile_2's start up to `--top` above its camera),
clustered; the clusters of at least `--min-points` points within `--range` m
of mobile_1, as map positions with their size and point count -- next to
mobile_2's track pose where it has one. A robot is a large cluster that moves
between rows; a chair stays put; a person is tall (dropped).

    python scoop_pipeline/checks/lidar_find_robot.py <pipeline config> [--every 5]
"""
import argparse
import importlib
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "map_stages"))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--every", type=float, default=5.0)
    ap.add_argument("--range", type=float, default=15.0)
    ap.add_argument("--min-points", type=int, default=40)
    ap.add_argument("--top", type=float, default=0.3)
    ap.add_argument("--track1", default="mobile_1_lidar")
    ap.add_argument("--track2", default="mobile_2_depth")
    a = ap.parse_args()
    from pipeline_common import load_pipeline, iter_scans
    from scipy.spatial import cKDTree
    import open3d as o3d
    import lidar_sightings as LS
    A = importlib.import_module("02_refine_poses")
    P = load_pipeline(a.config)
    od = P.reference_dir()
    t1, T1 = A.load_traj(os.path.join(od, "traj_%s.tum" % a.track1))
    p2 = os.path.join(od, "traj_%s.tum" % a.track2)
    t2, T2 = A.load_traj(p2) if os.path.exists(p2) else (np.zeros(0), None)
    ref_map = P.ref_file("anchored_map") if P.reference else P.outp("map_final_{tag}_anchored.pcd")
    M = np.asarray(o3d.io.read_point_cloud(ref_map).voxel_down_sample(0.05).points)
    tree = cKDTree(M)
    sa = __import__("json").load(open(P.outp("session_anchor.json")))
    cam = sa["cameras"]["realsense"]["map_to_cam"]["xyz"]
    near = np.linalg.norm(M[:, :2] - np.array(cam[:2]), axis=1) < 1.5
    z = M[near & (M[:, 2] < cam[2]) & (M[:, 2] > cam[2] - 3.0), 2]
    h, e = np.histogram(z, bins=np.arange(z.min(), z.max() + 0.1, 0.05))
    floor = float(e[np.argmax(h)] + 0.025)
    band = dict(LS.DEFAULTS, z_lo=floor + 0.05, z_hi=cam[2] + a.top, max_size_m=1.5,
                min_points=a.min_points)
    print("floor %.2f m, mobile_2's camera %.2f m: robot height %.2f..%.2f m; clusters of %d+ "
          "points within %.0f m of mobile_1" % (floor, cam[2], band["z_lo"], band["z_hi"],
                                                a.min_points, a.range))
    t0 = (t2[0] if len(t2) else t1[0]) - 21.0
    want = set()
    last = -1e9
    for i, t in enumerate(t1):
        if t - last >= a.every:
            want.add(i)
            last = t
    s01 = P.cfg.get("01_build_map", {})
    bag = os.path.expanduser(P.dataset["bag"])
    print("\n%7s  %-14s %-16s  %s" % ("t s", "mobile_1", "mobile_2 track", "clusters: (x, y) size m, points"))
    for j1, W in iter_scans(bag, "/mobile_1/ouster/points", t1, float(s01.get("time_tol", 0.04)),
                            0.8, a.range, T1, bool(s01.get("deskew", True)),
                            int(s01.get("deskew_bins") or 100), frame="world", verbose=False):
        if j1 not in want:
            continue
        dd, _ = tree.query(W, workers=-1, distance_upper_bound=0.15)
        off = W[~np.isfinite(dd)]
        cl = LS.clusters(off, band)
        m2 = "-"
        if len(t2):
            k = int(np.argmin(np.abs(t2 - t1[j1])))
            if abs(t2[k] - t1[j1]) < 0.5:
                m2 = "(%.1f, %.1f)" % tuple(T2[k][:2, 3])
        cl.sort(key=lambda x: -x[2])
        print("%7.1f  %-14s %-16s  %s" % (t1[j1] - t0, "(%.1f, %.1f)" % tuple(T1[j1][:2, 3]), m2,
                                          "  ".join("(%.1f, %.1f) %d" % (x, y, n) for x, y, n in cl[:5])
                                          or "-"))


if __name__ == "__main__":
    main()
