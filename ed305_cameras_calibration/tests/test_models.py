#!/usr/bin/env python3
"""Each lens model recovers a known camera from simulated board views (no cameras, no files):
  ed305    1920x1080, principal point at the centre, k1 k2 only (like the ED305 Basler cameras)
  full     848x480, principal point off-centre, k1 k2 p1 p2 k3 (like the RealSense colour camera)
  fisheye  1280x800, 190 deg class lens, equidistant k1..k4

  python tests/test_models.py
"""
import os
import sys
import tempfile

import cv2
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
TEST_BOARDS = os.path.join(tempfile.gettempdir(), "ed305_test_models_boards.yaml")
with open(TEST_BOARDS, "w") as _f:
    _f.write("boards:\n  A: {dictionary: DICT_4X4_50, squares_x: 9, squares_y: 7, square_m: 0.0245, marker_m: 0.018}\n"
             "floor_tags: {dictionary: DICT_APRILTAG_36h11, size_m: 0.200}\n")
os.environ["ED305_BOARDS"] = TEST_BOARDS
from calib.common import load_boards  # noqa: E402
from calib.intr import fit_camera, project_model  # noqa: E402

CASES = {
    "ed305": ((1920, 1080), np.array([[1750.0, 0, 960], [0, 1750.0, 540], [0, 0, 1]]), np.array([-0.35, 0.20, 0, 0, 0])),
    "full": ((848, 480), np.array([[427.8, 0, 423.1], [0, 427.2, 239.7], [0, 0, 1]]),
             np.array([-0.057, 0.065, -0.0005, 0.0006, -0.021])),
    "fisheye": ((1280, 800), np.array([[330.0, 0, 642], [0, 330.0, 398], [0, 0, 1]]), np.array([0.05, -0.02, 0.01, -0.003])),
}


def views_for(board, size, K, dist, model, n=60, seed=0):
    """Board views spread over the image, near and far, tilted, with 0.2 px corner noise."""
    rng = np.random.default_rng(seed)
    obj = board.object_points - board.object_points.mean(0)
    w, h = size
    views = []
    while len(views) < n:
        z = rng.uniform(0.12, 0.45)
        ray = cv2.undistortPoints(np.array([[[rng.uniform(0.1, 0.9) * w, rng.uniform(0.1, 0.9) * h]]]), K,
                                  None if model == "fisheye" else dist).reshape(2) if model != "fisheye" else \
            cv2.fisheye.undistortPoints(np.array([[[rng.uniform(0.15, 0.85) * w, rng.uniform(0.15, 0.85) * h]]]), K,
                                        dist.reshape(4, 1)).reshape(2)
        t = np.array([ray[0] * z, ray[1] * z, z])
        r = rng.normal(0, 0.45, 3) * [1, 1, 0.8]
        R, _ = cv2.Rodrigues(r)
        if (obj @ R.T + t)[:, 2].min() < 0.05:
            continue
        img = project_model(obj, r, t, K, dist, model) + rng.normal(0, 0.2, (len(obj), 2))
        ok = (img[:, 0] > 2) & (img[:, 0] < w - 2) & (img[:, 1] > 2) & (img[:, 1] < h - 2)
        if ok.sum() < 20:
            continue
        views.append({"board": "A", "corners": img[ok].astype(np.float32), "ids": np.where(ok)[0].astype(np.int32),
                      "path": "v%03d" % len(views), "person": "sim", "split": "test" if len(views) % 5 == 4 else "calib",
                      "image_size": size})
    return views


def main():
    boards = load_boards()
    for model, (size, K, dist) in CASES.items():
        res = fit_camera(views_for(boards["A"], size, K, dist, model), boards, 0, size, model)
        dK = np.abs(res["K"] - K).max()
        print("%-8s rms calib %.3f test %.3f px | K error %.2f px | dist %s (true %s)" % (
            model, res["rms_calib"], res["rms_test"], dK, np.round(res["dist"], 4), np.round(dist, 4)))
        assert res["rms_test"] < 0.4, model + ": test error too high"
        assert dK < 3.0, model + ": K off"
        assert len(res["dist"]) == (4 if model == "fisheye" else 5)
        if model == "ed305":
            assert res["K"][0, 2] == size[0] / 2 and res["K"][1, 2] == size[1] / 2, "principal point not at the centre"
            assert np.all(res["dist"][2:] == 0), "ed305 model must have p1 = p2 = k3 = 0"
    print("PASS")


if __name__ == "__main__":
    main()
