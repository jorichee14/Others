#!/usr/bin/env python3
"""Does mobile_1's LiDAR see mobile_2? At times when both are placed (04's
tracks) and 1.5-15 m apart, mobile_1's scan in map, and around mobile_2's
pose: the points the reference map does not have (no height band, no
clustering) -- how many, at what height, how far from mobile_2's camera --
and whether the map stands between them (the share of the line of sight that
passes through mapped surfaces). Also the floor under mobile_2 (the densest
layer of map points below its camera).

    python scoop_pipeline/checks/lidar_sees_robot.py <pipeline config> [--samples 40]
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
    ap.add_argument("--samples", type=int, default=40)
    ap.add_argument("--track1", default="mobile_1_lidar")
    ap.add_argument("--track2", default="mobile_2_depth")
    ap.add_argument("--radius", type=float, default=1.5)
    a = ap.parse_args()
    from pipeline_common import load_pipeline, iter_scans
    from scipy.spatial import cKDTree
    import open3d as o3d
    A = importlib.import_module("02_refine_poses")
    P = load_pipeline(a.config)
    od = P.reference_dir()
    t1, T1 = A.load_traj(os.path.join(od, "traj_%s.tum" % a.track1))
    t2, T2 = A.load_traj(os.path.join(od, "traj_%s.tum" % a.track2))
    ref_map = P.ref_file("anchored_map") if P.reference else P.outp("map_final_{tag}_anchored.pcd")
    pc = o3d.io.read_point_cloud(ref_map).voxel_down_sample(0.05)
    M = np.asarray(pc.points)
    tree = cKDTree(M)
    print("map %s: %d points (0.05 m)" % (ref_map, len(M)))

    # the floor under mobile_2: the densest 5 cm layer below its camera
    c0 = T2[0][:3, 3]
    near = np.linalg.norm(M[:, :2] - c0[:2], axis=1) < 1.5
    z = M[near & (M[:, 2] < c0[2]) & (M[:, 2] > c0[2] - 3.0), 2]
    if len(z):
        h, e = np.histogram(z, bins=np.arange(z.min(), z.max() + 0.05, 0.05))
        print("mobile_2's camera at z=%.2f; map points below it within 1.5 m: densest layer "
              "z=%.2f (%d pts), lowest %.2f" % (c0[2], e[np.argmax(h)] + 0.025, h.max(), z.min()))

    # times both are placed and 1.5-15 m apart
    want = []
    for t in t2[::5]:
        i = int(np.argmin(np.abs(t1 - t)))
        if abs(t1[i] - t) > 0.1:
            continue
        j = int(np.argmin(np.abs(t2 - t)))
        d = float(np.linalg.norm(T1[i][:2, 3] - T2[j][:2, 3]))
        if 1.5 <= d <= 15.0:
            want.append((t1[i], i, j, d))
    if not want:
        raise SystemExit("no time with both placed 1.5-15 m apart")
    pick = [want[k] for k in np.linspace(0, len(want) - 1, min(a.samples, len(want))).astype(int)]
    by_i = {p[1]: p for p in pick}
    s01 = P.cfg.get("01_build_map", {})
    bag = os.path.expanduser(P.dataset["bag"])
    print("\n%8s %6s %7s %8s %15s %10s %8s" % ("t s", "dist m", "offmap", "near2 n", "z range", "nearest m",
                                              "blocked"))
    rows = []
    t0 = t2[0] - 21.0
    for j1, W in iter_scans(bag, "/mobile_1/ouster/points", t1, float(s01.get("time_tol", 0.04)),
                            0.8, 25.0, T1, bool(s01.get("deskew", True)),
                            int(s01.get("deskew_bins") or 100), frame="world", verbose=False):
        if j1 not in by_i:
            continue
        t, i, j, d = by_i.pop(j1)
        p2 = T2[j][:3, 3]
        dd, _ = tree.query(W, workers=-1, distance_upper_bound=0.15)
        off = W[~np.isfinite(dd)]
        r = np.linalg.norm(off[:, :2] - p2[:2], axis=1)
        nb = off[r <= a.radius]
        # line of sight mobile_1 -> mobile_2 at the camera's height: through mapped surfaces?
        o = T1[i][:3, 3]
        s = np.linspace(0.1, 0.9, 40)[:, None]
        L = o + s * (np.r_[p2[:2], p2[2]] - o)
        ld, _ = tree.query(L, distance_upper_bound=0.1)
        blocked = float(np.isfinite(ld).mean())
        rows.append((len(nb), blocked))
        print("%8.1f %6.1f %7d %8d %15s %10s %7.0f%%"
              % (t - t0, d, len(off), len(nb),
                 "%.2f..%.2f" % (nb[:, 2].min(), nb[:, 2].max()) if len(nb) else "-",
                 "%.2f" % r.min() if len(r) else "-", 100 * blocked))
        if not by_i:
            break
    n = np.array([r[0] for r in rows])
    b = np.array([r[1] for r in rows])
    print("\noff-map points within %.1f m of mobile_2: in %d of %d scans (median %d); line of "
          "sight through the map in %d of them" % (a.radius, int((n > 0).sum()), len(n),
                                                   int(np.median(n)), int((b > 0.05).sum())))


if __name__ == "__main__":
    main()
