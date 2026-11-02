#!/usr/bin/env python3
"""How blurred the photographs of an export are, and what picking the sharpest
frame of each window would change.

    python3 scoop_pipeline/datasets/frame_sharpness.py <replica export or image folder> [--every 6]

Per frame: the variance of the Laplacian on a 320-px grey copy. Prints the
median over all frames, over every Nth (the scene as built), and over the
sharpest of each window of N (the scene --pick sharpest would build); the
share of frames under half the median; and the ten most blurred frames with
their neighbours' sharpness, to look at.
"""
import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lod3dgs import frame_sharpness, sharpest_frames  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", help="a replica export (results/frame*.jpg) or a folder of images")
    ap.add_argument("--every", type=int, default=6)
    ap.add_argument("--out", default=None, help="csv of frame,sharpness (default <path>/sharpness.csv)")
    a = ap.parse_args()
    path = os.path.expanduser(a.path)
    rgb = sorted(glob.glob(os.path.join(path, "results", "frame*.jpg"))) or sorted(
        p for p in glob.glob(os.path.join(path, "*")) if p.lower().endswith((".jpg", ".png")))
    if not rgb:
        sys.exit(f"{path}: no frames")
    print(f"{len(rgb)} frames", flush=True)
    keep, sharp = sharpest_frames(rgb, a.every, print)
    out = a.out or os.path.join(path, "sharpness.csv")
    with open(out, "w") as fh:
        fh.write("frame,sharpness\n")
        fh.writelines(f"{os.path.basename(p)},{v:.2f}\n" for p, v in zip(rgb, sharp))
    print(f"wrote {out}")
    worst = np.argsort(sharp)[:10]
    print("most blurred (frame: sharpness, neighbours):")
    for i in sorted(worst):
        lo, hi = max(0, i - 3), min(len(sharp), i + 4)
        print(f"  {os.path.basename(rgb[i])}: {sharp[i]:.1f}   " + " ".join(f"{v:.0f}" for v in sharp[lo:hi]))


if __name__ == "__main__":
    main()
