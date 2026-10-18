#!/usr/bin/env python3
"""Does one map cover what another covers? Completeness and accuracy of a map
B against a reference A, the usual reconstruction metrics:

  completeness   share of A's points with a point of B within --dist
                 (what B is missing)
  accuracy       share of B's points with a point of A within --dist
                 (what B has that A has not: new surfaces, or junk)

  python3 analysis/mapping/compare_maps.py A.pcd B.pcd [--dist 0.10]
      [--traj traj.txt] [--save missing.ply] [--sample 5000000]

    e.g. A = the union map (mapping/union_ba/denoised_...pcd),
         B = the TSDF map  (mapping/denoised_...pcd)

--dist    within this of each other counts as covered (m). A union map's
          surfaces are a band several cm wide; 0.10 holds the band.
--traj    also split completeness by distance from the robot's path and by
          height, to see WHERE B is missing what A has
--save    A's points B is missing, as a PLY (scalar_dist_cm: to the nearest
          point of B; scalar_range_m: to the path, with --traj)
--sample  measure on at most this many points of each map (random)
Writes nothing else.
"""
import argparse

import numpy as np
from scipy.spatial import cKDTree


def load(path):
    import open3d as o3d
    return np.asarray(o3d.io.read_point_cloud(path).points)


def sample(P, n, rng):
    return P if len(P) <= n else P[rng.choice(len(P), n, replace=False)]


def coverage(A, B, dist=0.10, n=5_000_000, seed=0):
    """(completeness, accuracy, As, dA, comp): the share of A's points with B
    within dist and of B's with A within dist, on at most n points of each;
    As the points of A measured, dA their distance to B, comp the mask."""
    rng = np.random.default_rng(seed)
    As, Bs = sample(A, n, rng), sample(B, n, rng)
    dA, _ = cKDTree(B).query(As, distance_upper_bound=1.0)
    dB, _ = cKDTree(A).query(Bs, distance_upper_bound=1.0)
    comp = np.isfinite(dA) & (dA <= dist)
    acc = np.isfinite(dB) & (dB <= dist)
    return float(comp.mean()), float(acc.mean()), As, dA, comp


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--dist", type=float, default=0.10)
    ap.add_argument("--traj")
    ap.add_argument("--save")
    ap.add_argument("--sample", type=int, default=5_000_000)
    a = ap.parse_args()

    A, B = load(a.a), load(a.b)
    print(f"A {a.a}: {len(A)} points, extent {np.round(A.max(0) - A.min(0), 1)} m")
    print(f"B {a.b}: {len(B)} points, extent {np.round(B.max(0) - B.min(0), 1)} m")
    c, acc, As, dA, comp = coverage(A, B, a.dist, a.sample)
    print(f"\nwithin {a.dist * 100:g} cm:")
    print(f"  completeness  {c * 100:5.1f} %  of A's points have B nearby "
          f"(B misses {100 - c * 100:.1f} %)")
    print(f"  accuracy      {acc * 100:5.1f} %  of B's points have A nearby")

    rng_m = None
    if a.traj:
        pos = np.loadtxt(a.traj, comments="#", ndmin=2)[:, 1:4]
        rng_m, _ = cKDTree(pos).query(As)
        print("\ncompleteness by distance from the path (m):")
        for lo, hi in ((0, 2), (2, 4), (4, 7), (7, 12), (12, 1e9)):
            m = (rng_m >= lo) & (rng_m < hi)
            if m.any():
                print(f"  {lo:4g}-{'' if hi > 1e8 else f'{hi:g}':4s} {comp[m].mean() * 100:5.1f} %  "
                      f"({m.mean() * 100:4.1f} % of A)")
        z = As[:, 2] - pos[:, 2].min()
        print("completeness by height above the path's lowest point (m):")
        for lo, hi in ((-1e9, -1), (-1, 0), (0, 1), (1, 2), (2, 3), (3, 5), (5, 1e9)):
            m = (z >= lo) & (z < hi)
            if m.any():
                lab = f"{'' if lo < -1e8 else f'{lo:g}'} .. {'' if hi > 1e8 else f'{hi:g}'}"
                print(f"  {lab:10s} {comp[m].mean() * 100:5.1f} %  ({m.mean() * 100:4.1f} % of A)")

    if a.save:
        miss = ~comp
        cols = [As[miss], np.where(np.isfinite(dA[miss]), dA[miss], 1.0)[:, None] * 100]
        names = ["scalar_dist_cm"]
        if rng_m is not None:
            cols.append(rng_m[miss][:, None])
            names.append("scalar_range_m")
        M = np.hstack(cols)
        with open(a.save, "w") as fh:
            fh.write("ply\nformat ascii 1.0\nelement vertex %d\n" % len(M))
            for f in ["x", "y", "z"] + names:
                fh.write("property float %s\n" % f)
            fh.write("end_header\n")
            np.savetxt(fh, M, fmt="%.3f")
        print(f"\nmissing points -> {a.save} ({len(M)})")


if __name__ == "__main__":
    main()
