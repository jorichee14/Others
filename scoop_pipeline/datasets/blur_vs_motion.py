#!/usr/bin/env python3
"""Is the blur the robot's motion, or just plain walls?

    python3 scoop_pipeline/datasets/blur_vs_motion.py <replica export> [--fps 30]

Reads sharpness.csv (frame_sharpness.py) and traj.txt of the export, takes
each frame's turning rate and speed from the poses around it, and prints the
median sharpness by turning-rate band and by speed band, with the rank
correlation. Sharpness that falls as the turning rate rises is motion blur;
sharpness unrelated to motion is content (a bare wall scores low whether the
camera moved or not).
"""
import argparse
import csv
import os
import sys

import numpy as np


def rank_corr(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path")
    ap.add_argument("--fps", type=float, default=30.0, help="frame rate of the export")
    a = ap.parse_args()
    path = os.path.expanduser(a.path)
    T = np.loadtxt(os.path.join(path, "traj.txt"), ndmin=2).reshape(-1, 4, 4)
    rows = list(csv.DictReader(open(os.path.join(path, "sharpness.csv"))))
    sharp = np.array([float(r["sharpness"]) for r in rows])
    if len(sharp) != len(T):
        sys.exit(f"{len(sharp)} sharpness rows, {len(T)} poses")
    # motion over the two frames around each one: translation (m/s) and rotation (deg/s)
    R, t = T[:, :3, :3], T[:, :3, 3]
    i0, i1 = np.clip(np.arange(len(T)) - 1, 0, len(T) - 1), np.clip(np.arange(len(T)) + 1, 0, len(T) - 1)
    dt = (i1 - i0) / a.fps
    speed = np.linalg.norm(t[i1] - t[i0], axis=1) / dt
    dR = np.einsum("nij,nik->njk", R[i0], R[i1])                 # R0^T R1
    ang = np.degrees(np.arccos(np.clip((np.trace(dR, axis1=1, axis2=2) - 1) / 2, -1, 1))) / dt
    print(f"{len(T)} frames; sharpness median {np.median(sharp):.0f}; "
          f"speed median {np.median(speed):.2f} m/s, turning median {np.median(ang):.1f} deg/s")
    print(f"rank correlation of sharpness with turning rate {rank_corr(sharp, ang):+.2f}, with speed {rank_corr(sharp, speed):+.2f}")
    for name, v, edges in (("turning rate (deg/s)", ang, [0, 5, 10, 20, 40, 80, 1e9]),
                           ("speed (m/s)", speed, [0, 0.1, 0.2, 0.4, 0.6, 0.8, 1e9])):
        print(f"by {name}:")
        for lo, hi in zip(edges[:-1], edges[1:]):
            sel = (v >= lo) & (v < hi)
            if sel.any():
                print(f"  {lo:>5g} .. {hi if hi < 1e9 else '':<5}  {int(sel.sum()):6d} frames  sharpness median {np.median(sharp[sel]):6.0f}  "
                      f"under half the overall median {100 * np.mean(sharp[sel] < 0.5 * np.median(sharp)):5.1f}%")
    blurred = sharp < 0.5 * np.median(sharp)
    print(f"of the {int(blurred.sum())} blurred frames, {100 * np.mean(ang[blurred] > 20):.0f}% were turning over 20 deg/s "
          f"and {100 * np.mean((ang[blurred] <= 20) & (speed[blurred] <= 0.2)):.0f}% were near still (the content, not the motion)")


if __name__ == "__main__":
    main()
