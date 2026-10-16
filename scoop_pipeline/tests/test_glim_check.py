#!/usr/bin/env python
"""Self-test for GLIM's loop-closure check: map_stages/05_anchor.py --traj
--check on a synthetic mobile_2 bag. The rs_anchor board is rendered into
RealSense colour images at the start (0-4 s) and the end (20-24 s) of a loop
that ends where it began; the trajectory given to the check drifted 15 mm by
the end. The check must find the board twice and report ~15 mm, write its
report, and write nothing else. Needs opencv-contrib, open3d, rosbags, mcap.

    python tests/test_glim_check.py
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STAGES = os.path.join(ROOT, "map_stages")
sys.path.insert(0, ROOT)
sys.path.insert(0, STAGES)

import cv2                                                             # noqa: E402

from scoop import rosmsg, tftree                                       # noqa: E402
from scoop.bagwrite import BagWriter                                   # noqa: E402

W, H = 848, 480
K = np.array([[427.7, 0, 423.1], [0, 427.1, 239.7], [0, 0, 1]])
DRIFT = np.array([0.015, 0.0, 0.0])                    # the trajectory's error at the end
CAL = ([-0.273322, 0.029175, 0.004183], [0.552208, 0.540455, -0.450592, -0.447148])


def camera_info(stamp):
    T = rosmsg.typestore().types
    return T["sensor_msgs/msg/CameraInfo"](
        header=rosmsg.header(stamp, "camera_color_optical_frame"), height=H, width=W,
        distortion_model="plumb_bob", d=np.zeros(5), k=K.ravel(), r=np.eye(3).ravel(),
        p=np.hstack([K, np.zeros((3, 1))]).ravel(), binning_x=0, binning_y=0,
        roi=T["sensor_msgs/msg/RegionOfInterest"](x_offset=0, y_offset=0, height=0,
                                                   width=0, do_rectify=False))


def render(board_img, m_per_px, T_cam_board, rng):
    """The board (OpenCV board frame: origin top-left, z into the board) seen
    from the camera, on a white background, with a little pixel noise."""
    R, t = T_cam_board[:3, :3], T_cam_board[:3, 3]
    Hm = K @ np.column_stack([R[:, 0], R[:, 1], t])          # board metres -> pixels
    S = np.diag([m_per_px, m_per_px, 1.0])                  # board pixels -> metres
    img = cv2.warpPerspective(board_img, Hm @ S, (W, H), flags=cv2.INTER_AREA,
                              borderValue=255)
    noisy = np.clip(img.astype(np.float32) + rng.normal(0, 2.0, img.shape), 0, 255)
    return np.repeat(noisy.astype(np.uint8)[:, :, None], 3, axis=2)


def image(stamp, rgb):
    T = rosmsg.typestore().types
    return T["sensor_msgs/msg/Image"](header=rosmsg.header(stamp, "camera_color_optical_frame"),
                                      height=H, width=W, encoding="rgb8", is_bigendian=0,
                                      step=W * 3, data=rgb.ravel())


def main():
    import pipeline_boards as pb
    import pipeline_common as pc
    d = tempfile.mkdtemp(prefix="glim_check_")
    fails = []
    try:
        cfg = pc.load_config(os.path.join(STAGES, "pipeline_config_mapping_20261015.json"))
        design = pb.Board("rs_anchor", cfg["boards"]["rs_anchor"])
        sq_px = 200
        bw, bh = design.sx * sq_px, design.sy * sq_px
        board_img = design.board.generateImage((bw, bh)) if hasattr(design.board, "generateImage") \
            else design.board.draw((bw, bh))
        # the board 0.6 m ahead of the start camera, centred, tilted as a real
        # one is (seen exactly face-on its pose is ambiguous, and 05 refuses it)
        R = cv2.Rodrigues(np.array([0.25, 0.45, 0.05]))[0]
        half = np.array([design.sx * design.square_len / 2, design.sy * design.square_len / 2, 0])
        T_cam_board = np.eye(4)
        T_cam_board[:3, :3] = R
        T_cam_board[:3, 3] = np.array([0, 0, 0.6]) - R @ half
        rng = np.random.default_rng(3)
        bag = os.path.join(d, "data", "work", "20261015", "mapping", "mobile_2", "x_retimed")
        t0 = 1_800_000_000 * 10**9
        with BagWriter(bag) as w:
            w.write("/tf_static", rosmsg.tf_message(
                [("os_lidar_mobile2", "camera_color_optical_frame", *CAL)], t0), t0)
            pts = np.zeros(4, [("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("intensity", "<f4")])
            w.write("/mobile_2/ouster/points", rosmsg.pointcloud2(pts, t0, "os_lidar_mobile2"), t0)
            for k in range(int(24 * 15)):                     # 15 Hz camera
                ts = t0 + int(k / 15 * 1e9)
                s = k / 15
                if not (s <= 4.0 or s >= 20.0):               # the board only at start / end
                    continue
                w.write("/mobile_2/color/camera_info", camera_info(ts), ts)
                w.write("/mobile_2/color/image_raw",
                        image(ts, render(board_img, design.square_len / sq_px, T_cam_board, rng)), ts)
        # the trajectory: at the start pose, a loop away, back at the start pose,
        # and the end 15 mm off -- GLIM's drift for the check to find
        traj = os.path.join(d, "traj_lidar.txt")
        with open(traj, "w") as fh:
            for k in range(241):                              # 10 Hz, 24 s
                s = k / 10
                x = 0.0 if s <= 4 or s >= 20 else 3.0 * np.sin(np.pi * (s - 4) / 16)
                p = np.array([x, 0.0, 0.0]) + (DRIFT if s >= 20 else 0.0)
                fh.write("%.9f %.6f %.6f %.6f 0 0 0 1\n" % (t0 * 1e-9 + s, *p))
        cfg["dataset"]["bag"] = bag
        cfg.pop("extends", None)
        cfg_path = os.path.join(d, "cfg.json")
        json.dump(cfg, open(cfg_path, "w"))
        out = os.path.join(d, "check.json")
        proc = os.path.join(d, "data", "processed", "20261015", "mapping")
        before = sorted(os.listdir(proc)) if os.path.isdir(proc) else []
        p = subprocess.run([sys.executable, os.path.join(STAGES, "05_anchor.py"), cfg_path,
                            "--traj", traj, "--check", out], capture_output=True, text=True)
        if p.returncode != 0:
            raise AssertionError(p.stdout[-3000:] + p.stderr[-3000:])
        r = json.load(open(out))
        b = r["boards"]["rs_anchor"]
        lc = b["loop_closure"]
        if b["sections"] != 2 or lc is None:
            fails.append(f"expected 2 sections with a loop closure: {b}")
        elif abs(lc["mm"] - 15.0) > 2.0:
            fails.append(f"loop closure {lc['mm']} mm, expected ~15")
        if r["poses"] != 241:
            fails.append(f"poses {r['poses']}")
        after = sorted(os.listdir(proc)) if os.path.isdir(proc) else []
        for sub in after:                                     # nothing but empty folders
            for root, _, files in os.walk(os.path.join(proc, sub)):
                if files:
                    fails.append(f"--check wrote {root}: {files}")
        if not fails:
            print(f"  ok    loop closure {lc['mm']} mm / {lc['deg']} deg over "
                  f"{lc['dt_s']} s (noise {lc['noise_mm']} mm) -> "
                  f"{'DRIFT' if lc['significant'] else 'not significant'}; nothing else written")
        _ = before, tftree
        # the same through the GLIM step's wrapper: report + summary in glim.yaml
        import yaml
        from scoop import recording
        work = os.path.join(d, "data", "work", "20261015", "mapping", "mobile_2")
        rec = recording.Recording(recording.Path(d, "data", "raw", "20261015", "mapping",
                                                 "mobile_2"), recording.Path(bag), recording.Path(work))
        os.makedirs(rec.odometry, exist_ok=True)
        shutil.copy(traj, rec.odometry / "traj_lidar.txt")
        yaml.safe_dump({"files": {"traj_lidar.txt": {"poses": 241}}},
                       open(rec.odometry / "glim.yaml", "w"))
        logs = []
        summ = recording.glim_loop_check(rec, logs.append, config=cfg_path)
        info = yaml.safe_load(open(rec.odometry / "glim.yaml"))
        lc2 = (info.get("loop_closure") or {}).get("boards", {}).get("rs_anchor", {})
        if summ is None or abs((lc2.get("mm") or 0) - 15.0) > 2.0:
            fails.append(f"glim.yaml loop_closure: {info.get('loop_closure')} / {logs}")
        elif info["files"]["traj_lidar.txt"]["poses"] != 241:
            fails.append("glim.yaml lost its other values")
        elif not any("loop closure rs_anchor: s0->s1" in ln for ln in logs):
            fails.append(f"no summary line: {logs}")
        else:
            print("  ok    " + [ln for ln in logs if "loop closure rs_anchor" in ln][0].strip())
    finally:
        shutil.rmtree(d, ignore_errors=True)
    for f in fails:
        print("  FAIL  " + f)
    print("glim check ok" if not fails else "glim check FAILED")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
