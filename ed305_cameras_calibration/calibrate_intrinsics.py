#!/usr/bin/env python3
"""Intrinsics per camera from the frames saved by capture.py --mode intr.

  python calibrate_intrinsics.py --data data --cams 1 2 3 4 13 14
  python calibrate_intrinsics.py --data data            # every camera that has frames

Every 5th saved view of a camera is a TEST view: it is never used in the fit, only to measure the
result (rms_test_px). Reads <data>/<person>/intr/cam_N/*.json (all people merged) and writes
  results/intrinsics/cam_N.json      K, dist, errors, the views used
  results/intrinsics/views.csv       one row per view: split, used, rms, board x y z roll pitch yaw
  results/intrinsics/summary.csv     one row per camera: errors, K, distortion, coverage, status
"""
import argparse
import os

import cv2
import numpy as np

from calib.common import cam_name, latest_session, load_boards, write_csv, write_json
from calib.intr import DEFAULT_MODEL, FIELDS, IMAGE_SIZE, MODELS, fit_camera, load_views, summary_row, view_rows


def solve_camera(cam, data_dir, results_dir, boards, flags=0, session=None, model=DEFAULT_MODEL):
    """Fit one camera from the saved views of one calibration round (default: the latest) with a lens
    model (calib.intr.MODELS) and write cam_N.json (round 1) or cam_N_<round>.json. Returns the
    fit_camera result or None."""
    session = session or latest_session(data_dir, results_dir, cam)
    views = load_views(data_dir, cam, session)
    size = views[0]["image_size"] if views else IMAGE_SIZE
    res = fit_camera(views, boards, flags, size, model)
    if res is None:
        return None
    K, d, (w, h) = res["K"], res["dist"], size
    write_json(os.path.join(results_dir, "intrinsics", cam_name(cam, session) + ".json"), {
        "camera": cam, "session": session, "model": model, "model_text": MODELS[model],
        "image_size": size, "K": K, "dist": d,
        "rms_px": res["rms_calib"], "rms_test_px": res["rms_test"],
        "views": res["n_calib"], "test_views": res["n_test"], "dropped_views": res["dropped"],
        "coverage": res["coverage"],
        "frames": [v["path"] for v in res["views"] if v["used"]],
        "test_frames": [v["path"] for v in res["views"] if v["split"] == "test"],
        # the same keys as the March 2026 ED305 calibration files (calibration/data/outputs/results/cam_N.json
        # and camera_params/cam_N.json), so tools reading those read this too
        "camera_id": cam, "img_w": w, "img_h": h, "camera_matrix": K, "dist_coeffs": d,
        "fx": K[0, 0], "fy": K[1, 1], "cx": K[0, 2], "cy": K[1, 2], "rms": res["rms_calib"],
        "intrinsic": K.reshape(-1), "distortion": d,
    })
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data")
    ap.add_argument("--results", default="results")
    ap.add_argument("--cams", nargs="*", type=int, default=list(range(1, 17)))
    ap.add_argument("--model", default=DEFAULT_MODEL, choices=sorted(MODELS),
                    help="lens model: " + "; ".join("%s = %s" % kv for kv in MODELS.items()))
    ap.add_argument("--rational", action="store_true", help="with --model full: 8-coefficient distortion model")
    args = ap.parse_args()
    boards = load_boards()
    flags = cv2.CALIB_RATIONAL_MODEL if args.rational else 0

    rows, summary = [], []
    for cam in args.cams:
        res = solve_camera(cam, args.data, args.results, boards, flags, model=args.model)
        if res is None:
            n = len([v for v in load_views(args.data, cam, latest_session(args.data, args.results, cam))
                     if v["split"] == "calib"])
            print("cam %2d: only %d calibrate views, need 10+ (skipped)" % (cam, n))
            continue
        rows += view_rows(cam, res)
        summary.append(summary_row(cam, res))
        K, cov = res["K"], res["coverage"]
        print("cam %2d: %3d calib + %2d test views, rms calib %.3f  test %.3f px, dropped %d | "
              "fx %.1f fy %.1f cx %.1f cy %.1f | coverage %s | %s" % (
                  cam, res["n_calib"], res["n_test"], res["rms_calib"], res["rms_test"], res["dropped"],
                  K[0, 0], K[1, 1], K[0, 2], K[1, 2],
                  " ".join("%s %.0f%%" % (k, 100 * cov[k]) for k in FIELDS), summary[-1]["status"]))
    if rows:
        write_csv(os.path.join(args.results, "intrinsics", "views.csv"), rows)
        write_csv(os.path.join(args.results, "intrinsics", "summary.csv"), summary)
        print("written", os.path.join(args.results, "intrinsics", "summary.csv"))


if __name__ == "__main__":
    main()
