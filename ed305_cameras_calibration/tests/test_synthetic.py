#!/usr/bin/env python3
"""End-to-end check without cameras: a synthetic ED305 (16 cameras around a 6x6 m room looking at
the centre), synthetic intr frames and still-board events written in capture.py's file layout,
then calibrate_intrinsics.py, calibrate_extrinsics.py and evaluate.py are run and compared with
the truth (camera positions, focal lengths, test-set errors, CSV outputs).

  python tests/test_synthetic.py
"""
import os
import shutil
import subprocess
import sys
import tempfile

import cv2
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
from calib.common import FloorTags, T_to_rt, inv_T, load_boards, project, read_csv, read_json, rt_to_T, write_json  # noqa: E402
from calib.intr import coverage, view_features  # noqa: E402

rng = np.random.default_rng(0)
W, H = 1920, 1080


def look_at(pos, target):
    z = target - pos
    z /= np.linalg.norm(z)
    x = np.cross(z, np.array([0, 0, 1.0]))
    if np.linalg.norm(x) < 1e-6:
        x = np.array([1.0, 0, 0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R_wc = np.stack([x, y, z], axis=1)  # camera axes in world
    T = np.eye(4)
    T[:3, :3] = R_wc.T
    T[:3, 3] = -R_wc.T @ pos
    return T  # world -> camera


def truth():
    centre = np.array([3.0, 3.0, 0.0])
    cams = {}
    for i in range(1, 17):
        if i <= 4:
            pos = np.array([0.2, 1.0 + (i - 1) * 1.3, 2.6])
        elif i <= 8:
            pos = np.array([5.8, 1.0 + (i - 5) * 1.3, 2.6])
        elif i <= 12:
            pos = np.array([1.2 + (i - 9) * 1.2, 4.5, 2.7])
        else:
            pos = np.array([1.2 + (i - 13) * 1.2, 1.5, 2.7])
        T = look_at(pos, centre + rng.normal(0, 0.15, 3) * [1, 1, 0])
        f = 1400 + rng.normal(0, 30)
        K = np.array([[f, 0, 960 + rng.normal(0, 5)], [0, f, 540 + rng.normal(0, 5)], [0, 0, 1]])
        dist = np.array([-0.12, 0.05, 0.0005, -0.0003, 0.0])
        cams[i] = (T, K, dist)
    return cams


def visible(img, obj_cam):
    ok = (obj_cam[:, 2] > 0.3) & (img[:, 0] > 5) & (img[:, 0] < W - 5) & (img[:, 1] > 5) & (img[:, 1] < H - 5)
    return ok


def main():
    data = tempfile.mkdtemp(prefix="ed305_")
    results = os.path.join(data, "results")
    boards = load_boards()
    tags = FloorTags()
    world_tags = tags.world_corners()
    cams = truth()
    noise = 0.3

    # --- intrinsics frames: board placed in front of each camera at many poses
    for cam, (T_cw, K, dist) in cams.items():
        board = boards["A"]
        obj = board.object_points
        n = 0
        while n < 45:
            # board pose in camera frame: 1.2-3 m ahead, random tilt, random offset
            rvec = rng.normal(0, 0.35, 3)
            R, _ = cv2.Rodrigues(rvec)
            t = np.array([rng.uniform(-0.9, 0.5), rng.uniform(-0.6, 0.3), rng.uniform(1.2, 3.0)])
            pc = obj @ R.T + t
            img = project(K, dist, np.zeros(3), np.zeros(3), pc) + rng.normal(0, noise, (len(obj), 2))
            ok = visible(img, pc)
            if ok.sum() < 20:
                continue
            write_json(os.path.join(data, "P", "intr", "cam_%d" % cam, "f%03d.json" % n), {
                "camera": cam, "board": "A", "time": 0, "event": None,
                "charuco": {"ids": np.where(ok)[0], "corners": img[ok]}, "tags": {}})
            n += 1

    # --- extrinsic events: still board at random spots, seen by whoever sees it, plus floor tags
    ev = 0
    tag_obs = 0
    while ev < 120:
        board = boards[["A", "B", "C"][ev % 3]]
        obj = board.object_points
        pos = np.array([rng.uniform(0.8, 5.2), rng.uniform(0.8, 5.2), rng.uniform(0.3, 1.8)])
        R, _ = cv2.Rodrigues(np.array([rng.uniform(-1.6, -0.6), rng.normal(0, 0.4), rng.uniform(-3.1, 3.1)]))
        pw = obj @ R.T + pos
        seen = {}
        for cam, (T_cw, K, dist) in cams.items():
            r, t = T_to_rt(T_cw)
            pc = pw @ T_cw[:3, :3].T + T_cw[:3, 3]
            img = project(K, dist, r, t, pw) + rng.normal(0, noise, (len(obj), 2))
            ok = visible(img, pc)
            if ok.sum() >= 12:
                tag_det = {}
                for tid, wc in world_tags.items():
                    ti = project(K, dist, r, t, wc) + rng.normal(0, noise, (4, 2))
                    tc = wc @ T_cw[:3, :3].T + T_cw[:3, 3]
                    if visible(ti, tc).all() and rng.uniform() < 0.5:
                        tag_det[str(tid)] = ti
                        tag_obs += 1
                seen[cam] = {"ids": np.where(ok)[0], "corners": img[ok]}, tag_det
        if len(seen) < 2:
            continue
        name = "ev%04d" % ev
        for cam, (ch, tag_det) in seen.items():
            write_json(os.path.join(data, "P", "extr", name, "cam_%d.json" % cam), {
                "camera": cam, "board": board.name, "time": 0, "event": name, "charuco": ch, "tags": tag_det})
        ev += 1
    print("synthetic data in", data, "tag observations", tag_obs)

    env = dict(os.environ, PYTHONPATH=HERE)
    subprocess.run([sys.executable, os.path.join(HERE, "calibrate_intrinsics.py"), "--data", data,
                    "--results", results], check=True, env=env)
    subprocess.run([sys.executable, os.path.join(HERE, "calibrate_extrinsics.py"), "--data", data,
                    "--results", results], check=True, env=env)

    subprocess.run([sys.executable, os.path.join(HERE, "evaluate.py"), "--data", data,
                    "--results", results], check=True, env=env)

    out = read_json(os.path.join(results, "extrinsics", "cameras.json"))["cameras"]
    worst_pos, worst_f, worst_ang = 0, 0, 0
    for cam, (T_cw, K, dist) in cams.items():
        o = out[str(cam)]
        pos_true = -T_cw[:3, :3].T @ T_cw[:3, 3]
        worst_pos = max(worst_pos, np.linalg.norm(np.array(o["position_world"]) - pos_true))
        worst_f = max(worst_f, abs(np.array(o["K"])[0, 0] - K[0, 0]))
        dR = np.array(o["R_cw"]) @ T_cw[:3, :3].T
        worst_ang = max(worst_ang, np.degrees(np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1))))
    print("worst camera position error %.1f mm, rotation error %.3f deg, focal error %.1f px" % (
        worst_pos * 1000, worst_ang, worst_f))
    assert worst_pos < 0.03, "position error too large"
    assert worst_ang < 0.2, "rotation error too large"
    assert worst_f < 8, "focal error too large"

    # calibrate/test split and the CSVs
    views = read_csv(os.path.join(results, "intrinsics", "views.csv"))
    summ = read_csv(os.path.join(results, "intrinsics", "summary.csv"))
    assert len(summ) == 16 and all(int(r["n_test"]) >= 8 for r in summ), "test views missing"
    assert all(r["used"] == "0" for r in views if r["split"] == "test"), "a test view was used in the fit"
    assert all(float(r["rms_test_px"]) < 1.0 for r in summ), "intrinsics test error too high"
    assert all(0.5 < float(r["z"]) < 4.0 and abs(float(r["pitch"])) < 90 for r in views if r["z"]), "bad view features"
    ev = read_csv(os.path.join(results, "extrinsics", "events.csv"))
    cams_csv = read_csv(os.path.join(results, "extrinsics", "cameras.csv"))
    assert sum(r["split"] == "test" for r in ev) >= 20, "test events missing"
    assert all(float(r["rms_test_px"]) < 1.0 for r in cams_csv if r["rms_test_px"]), "extrinsics test error too high"
    assert all(float(r["floor_err_test_cm"]) < 2.0 for r in cams_csv if r["floor_err_test_cm"]), "floor error too high"
    assert all(2.5 < float(r["z"]) < 2.8 for r in cams_csv), "camera height wrong"
    ev_dirs = sorted(os.listdir(os.path.join(results, "evaluation")))
    ev_rows = read_csv(os.path.join(results, "evaluation", ev_dirs[-1], "cameras.csv"))
    assert len(ev_rows) == 16 and all(r["status"] == "OK" for r in ev_rows), "evaluate.py flagged a camera"
    print("intrinsics test rms: worst %.3f px; extrinsics test rms: worst %.3f px; floor error: worst %.1f cm" % (
        max(float(r["rms_test_px"]) for r in summ), max(float(r["rms_test_px"]) for r in cams_csv if r["rms_test_px"]),
        max(float(r["floor_err_test_cm"]) for r in cams_csv if r["floor_err_test_cm"])))

    # coverage bars: a set of views that spans the wanted ranges must fill every bar
    board = boards["A"]
    K = cams[1][1]
    feats = []
    for rv, t in [((0, 0, 0), (-0.9, -0.5, 1.5)), ((0, 0, 0), (1.0, 0.6, 2.5)), ((0, 0, 0), (0.3, 0.1, 4.0)), ((0, 0, 1.2), (0, 0, 2)),
                  ((0.6, 0, 0), (0, 0, 2)), ((-0.6, 0, 0), (0, 0, 2)), ((0, 0.6, 0), (0, 0, 2)), ((0, -0.6, 0), (0, 0, 2))]:
        R, _ = cv2.Rodrigues(np.array(rv, float))
        pc = board.object_points @ R.T + np.array(t)
        img = project(K, cams[1][2], np.zeros(3), np.zeros(3), pc).astype(np.float32)
        feats.append(view_features(board, img, np.arange(len(img)), K, cams[1][2]))
    cov = coverage(feats)
    assert all(v >= 1.0 for v in cov.values()), "coverage bars not full: %s" % cov
    shutil.rmtree(data)
    print("PASS")


if __name__ == "__main__":
    main()
