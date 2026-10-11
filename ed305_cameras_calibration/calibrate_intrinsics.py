#!/usr/bin/env python3
"""Intrinsics per camera from the frames saved by capture.py --mode intr.

  python calibrate_intrinsics.py --data data --cams 1 2 3 4 13 14
  python calibrate_intrinsics.py --data data            # every camera that has frames

Reads every <data>/<person>/intr/cam_N/*.json (all people merged), calibrates with the stored
ChArUco corners, drops the worst views once, and writes results/intrinsics/cam_N.json.
"""
import argparse
import glob
import os

import cv2
import numpy as np

from calib.common import load_boards, read_json, write_json

IMAGE_SIZE = (1920, 1080)


def load_views(data_dir, cam):
    views = []
    for path in sorted(glob.glob(os.path.join(data_dir, "*", "intr", "cam_%d" % cam, "*.json"))):
        j = read_json(path)
        if j.get("charuco") and len(j["charuco"]["ids"]) >= 6:
            views.append((j["board"], np.array(j["charuco"]["corners"], np.float32),
                          np.array(j["charuco"]["ids"], np.int32), path))
    return views


def calibrate(views, boards, flags=0):
    obj, img = [], []
    for board_name, corners, ids, _ in views:
        o, i = boards[board_name].board.matchImagePoints(corners.reshape(-1, 1, 2), ids.reshape(-1, 1))
        obj.append(o.reshape(-1, 3).astype(np.float32))
        img.append(i.reshape(-1, 2).astype(np.float32))
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(obj, img, IMAGE_SIZE, None, None, flags=flags)
    per_view = []
    for o, i, r, t in zip(obj, img, rvecs, tvecs):
        p, _ = cv2.projectPoints(o, r, t, K, dist)
        per_view.append(float(np.sqrt(np.mean(np.sum((p.reshape(-1, 2) - i) ** 2, axis=1)))))
    return rms, K, dist, per_view


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data")
    ap.add_argument("--results", default="results")
    ap.add_argument("--cams", nargs="*", type=int, default=list(range(1, 17)))
    ap.add_argument("--rational", action="store_true", help="8-coefficient distortion model")
    args = ap.parse_args()
    boards = load_boards()
    flags = cv2.CALIB_RATIONAL_MODEL if args.rational else 0

    for cam in args.cams:
        views = load_views(args.data, cam)
        if len(views) < 10:
            print("cam %2d: only %d usable frames, need 10+ (skipped)" % (cam, len(views)))
            continue
        rms, K, dist, per_view = calibrate(views, boards, flags)
        # drop clear outliers once (views with error > 2x median) and recalibrate
        med = np.median(per_view)
        keep = [v for v, e in zip(views, per_view) if e <= max(2 * med, 0.5)]
        dropped = len(views) - len(keep)
        if dropped and len(keep) >= 10:
            rms, K, dist, per_view = calibrate(keep, boards, flags)
            views = keep
        write_json(os.path.join(args.results, "intrinsics", "cam_%d.json" % cam), {
            "camera": cam, "image_size": IMAGE_SIZE, "K": K, "dist": dist.reshape(-1),
            "rms_px": rms, "views": len(views), "dropped_views": dropped,
            "per_view_rms_px": per_view, "frames": [v[3] for v in views],
        })
        flag = "OK" if rms < 0.6 else "CHECK (want < 0.6 px: more tilted/corner views, board sharp)"
        print("cam %2d: %3d views, rms %.3f px, dropped %d  %s  fx %.1f fy %.1f cx %.1f cy %.1f" % (
            cam, len(views), rms, dropped, flag, K[0, 0], K[1, 1], K[0, 2], K[1, 2]))


if __name__ == "__main__":
    main()
