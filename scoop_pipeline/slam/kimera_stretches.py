#!/usr/bin/env python3
"""The stretches of a datasets/euroc.py export Kimera-VIO can run through:
no IMU, camera or ground-truth gap longer than --gap seconds inside.

    python3 scoop_pipeline/slam/kimera_stretches.py <export>/<robot> [--gap 0.5] [--min 10]
        [--margin 0.3] [--initial-k 50]

Prints one stretch per line, "k0 k1 t0 t1": Kimera-VIO's --initial_k and
--final_k (stereo pair indices in mav0/cam0/data.csv, k1 exclusive) and the
seconds after the first pair. A stretch starts --margin seconds after a gap
(IMU before its first frame, the ground truth Kimera-VIO starts from within
10 ms of it: euroc.py writes it at every pair away from a gap) and ends at the
last pair with IMU after it; shorter than --min seconds, it is left out. The
first starts at pair --initial-k at the earliest (Kimera-VIO's own default).
mobile_2 of 20260924: the RealSense dropped out 15 times (0.7-8.7 s, its
images, IMU and so the ground truth together).
"""
import argparse
import os
import sys

import numpy as np


def first_column(path):
    return np.loadtxt(path, delimiter=",", comments="#", dtype=np.int64, usecols=0, ndmin=1)


def stretches(robot_dir, gap=0.5, min_s=10.0, margin=0.3, initial_k=50):
    m0 = os.path.join(robot_dir, "mav0")
    c = first_column(os.path.join(m0, "cam0", "data.csv"))
    u = first_column(os.path.join(m0, "imu0", "data.csv"))
    g = first_column(os.path.join(m0, "state_groundtruth_estimate0", "data.csv"))
    G = int(gap * 1e9)
    # holes: every gap longer than --gap in any of the three, as [from, to] ns
    holes = []
    for x in (c, u, g):
        d = np.diff(x)
        holes += [(int(x[i]), int(x[i + 1])) for i in np.flatnonzero(d > G)]
    lo, hi = max(u[0], g[0]), min(u[-1], g[-1])
    bounds, cur = [], lo
    for a, b in sorted(holes):
        if a > cur:
            bounds.append((cur, a))
        cur = max(cur, b)
    if hi > cur:
        bounds.append((cur, hi))
    out = []
    M = int(margin * 1e9)
    for a, b in bounds:
        # frames from margin after the hole to the last one with IMU (50 ms) after it
        k0 = int(np.searchsorted(c, a + M))
        k1 = int(np.searchsorted(c, b - 50_000_000))      # exclusive
        k0 = max(k0, initial_k if not out else 0)
        if k1 - k0 < 2 or (c[k1 - 1] - c[k0]) * 1e-9 < min_s:
            continue
        out.append((k0, k1, (c[k0] - c[0]) * 1e-9, (c[k1 - 1] - c[0]) * 1e-9))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("robot_dir")
    ap.add_argument("--gap", type=float, default=0.5)
    ap.add_argument("--min", type=float, default=10.0, dest="min_s")
    ap.add_argument("--margin", type=float, default=0.3)
    ap.add_argument("--initial-k", type=int, default=50)
    a = ap.parse_args()
    s = stretches(a.robot_dir, a.gap, a.min_s, a.margin, a.initial_k)
    if not s:
        sys.exit("no stretch of %g s or more" % a.min_s)
    for k0, k1, t0, t1 in s:
        print("%d %d %.1f %.1f" % (k0, k1, t0, t1))


if __name__ == "__main__":
    main()
