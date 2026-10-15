#!/usr/bin/env python3
"""Field of view of calibrated cameras, from their intrinsics (results/intrinsics/cam_N.json).

  python fov.py                                  # every camera in results/intrinsics
  python fov.py results/intrinsics/cam_3.json    # one file (also the March 2026 cam_N.json or a camera_info YAML)

pinhole:  2 atan(w / 2 fx) etc., ignoring the lens distortion (what spec sheets usually give)
real:     the image edges undistorted with the distortion coefficients: what the camera actually sees
          (wider than pinhole for barrel distortion). Use this one for coverage planning.
"""
import glob
import os
import re
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from calib.common import read_intrinsics_file  # noqa: E402


def fov(K, d, size, model="full"):
    """(pinhole h, v, real h, v, real diagonal) in degrees."""
    w, h = size
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    ph = np.degrees(np.arctan(cx / fx) + np.arctan((w - cx) / fx))
    pv = np.degrees(np.arctan(cy / fy) + np.arctan((h - cy) / fy))
    edges = np.array([[0, cy], [w, cy], [cx, 0], [cx, h], [0, 0], [w, h]], float).reshape(-1, 1, 2)
    n = (cv2.fisheye.undistortPoints(edges, K, np.asarray(d, float).reshape(4, 1)) if model == "fisheye"
         else cv2.undistortPoints(edges, K, np.asarray(d, float))).reshape(-1, 2)
    rh = np.degrees(np.arctan(-n[0, 0]) + np.arctan(n[1, 0]))
    rv = np.degrees(np.arctan(-n[2, 1]) + np.arctan(n[3, 1]))
    a, b = (np.r_[p, 1] / np.linalg.norm(np.r_[p, 1]) for p in n[4:6])
    return ph, pv, rh, rv, np.degrees(np.arccos(np.clip(a @ b, -1, 1)))


def main():
    files = sys.argv[1:] or sorted(glob.glob(os.path.join(HERE, "results", "intrinsics", "cam_*.json")),
                                   key=lambda p: [int(x) for x in re.findall(r"\d+", os.path.basename(p))])
    if not files:
        sys.exit("no calibrations found: give a cam_N.json, or calibrate cameras first")
    print("%-16s %-10s %-22s %s" % ("file", "image", "pinhole h x v (deg)", "real h x v, diagonal (deg)"))
    for f in files:
        K, d, size, model = read_intrinsics_file(f)
        if size is None:
            print("%-16s no image size in the file" % os.path.basename(f))
            continue
        ph, pv, rh, rv, rd = fov(K, d, size, model)
        print("%-16s %-10s %5.1f x %4.1f            %5.1f x %4.1f, %5.1f" % (
            os.path.basename(f), "%dx%d" % tuple(size), ph, pv, rh, rv, rd))


if __name__ == "__main__":
    main()
