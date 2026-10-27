#!/usr/bin/env python
"""Self-test for check.py board: a bag of a 1.00 x 0.75 m ChArUco board stood
at 2, 3, 4 and 5 m, turned 20 deg off square, with a wall 1.2 m behind it and
a floor. Every LiDAR point is ranged along its own beam with a known noise
(4 mm + 1.2 mm per m) and a known +10 mm bias, so the check must read back:

  sigma along the beam  6.4, 7.6, 8.8, 10.0 mm per station, a line 4 + 1.2 d
  bias                  +10 mm at every station, whatever the incidence
  thickness             3.29 x sigma about the board's normal

and, on a second bag of the board shimmed 5 mm at a time at 2 m, read as runs
(the stations are millimetres apart, so range alone cannot separate them):

  step                  +5.0 mm per shim, from the LiDAR's plane alone
  repeat                under a millimetre: 1300 points average the 6 mm of
                        range noise away, which is why a movement far under
                        sigma is still read

    python tests/test_board_noise.py
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "map_stages"))

import cv2                                                             # noqa: E402
from pipeline_boards import Board                                      # noqa: E402
from scoop import rosmsg                                               # noqa: E402
from scoop.bagwrite import BagWriter                                   # noqa: E402

IMAGE = "/mobile_1/zed/left/image_rect_color"
INFO = "/mobile_1/zed/left/camera_info"
LIDAR = "/mobile_1/ouster/points"
CAM_FRAME = "zed_left_camera_optical_frame"
LIDAR_FRAME = "os_lidar"

W, H = 1280, 720
K = np.array([[1100.0, 0, 640], [0, 1100.0, 360], [0, 0, 1]])
SQ_PX = 60                                 # the printed board's pixels per square
BOARD = {"squares_x": 8, "squares_y": 6, "square_len": 0.125, "marker_len": 0.094,
         "dictionary": "DICT_5X5_100", "min_corners": 8, "max_reproj": 1.0}
STATIONS = (2.0, 3.0, 4.0, 5.0)
SHIMS = (0.0, 0.005, 0.010)                # the step test: the board at 2 m, shimmed
YAW = np.radians(20.0)                     # off square, so PnP is not a mirror flip
SIG_A, SIG_B, BIAS = 0.004, 0.0012, 0.010  # m, m per m, m: the truth to read back
T0 = 1_800_000_000


def T_lidar_cam():
    """The camera optical frame in the LiDAR's: 7 cm ahead, 9 cm below, z
    forward / y down against the LiDAR's x forward / z up."""
    T = np.eye(4)
    T[:3, :3] = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], float)
    T[:3, 3] = (0.07, 0.0, -0.09)
    return T


def T_lidar_board(d, shim=0.0):
    """The board d m ahead of the LiDAR (shim m further along its own normal), centred on its x axis, 0.1 m up, its
    printed face toward the sensor and turned YAW about the vertical. Board
    axes are OpenCV's: x across, y down, z into the face."""
    # columns: the board's axes in the LiDAR's frame. Facing the sensor from
    # ahead: z (into the face) = +x, y (down the face) = -z, x = y cross z = -y
    R = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
    c, s = np.cos(YAW), np.sin(YAW)
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])                    # about the LiDAR's z
    w, h = BOARD["squares_x"] * BOARD["square_len"], BOARD["squares_y"] * BOARD["square_len"]
    T = np.eye(4)
    T[:3, :3] = Rz @ R
    T[:3, 3] = np.array([d, 0.0, 0.1]) - T[:3, :3] @ np.array([w / 2, h / 2, 0.0])
    T[:3, 3] += shim * T[:3, :3][:, 2]                 # along the board's normal, away
    return T


def beams(n_az=1024, n_el=48, fov=np.radians(45.0)):
    """Unit directions of an Ouster-like grid, the sector the board is in."""
    az = np.radians(np.linspace(-22.0, 22.0, n_az // 8))
    el = np.linspace(-fov / 2, fov / 2, n_el)
    A, E = np.meshgrid(az, el)
    A, E = A.ravel(), E.ravel()
    return np.column_stack([np.cos(E) * np.cos(A), np.cos(E) * np.sin(A), np.sin(E)])


def plane_hit(u, nrm, p0):
    """Range along each beam u to the plane (n, p0), inf where it misses."""
    den = u @ nrm
    t = np.where(np.abs(den) > 1e-9, (p0 @ nrm) / np.where(den == 0, 1e-9, den), np.inf)
    return np.where(t > 0, t, np.inf)


def scan(d, rng, u, shim=0.0):
    """One scan: the board at d m, a wall 1.2 m behind it, the floor 0.9 m
    below, each point ranged along its own beam with the noise and bias."""
    T = T_lidar_board(d, shim)
    R, t = T[:3, :3], T[:3, 3]
    w, h = BOARD["squares_x"] * BOARD["square_len"], BOARD["squares_y"] * BOARD["square_len"]
    nb = R[:, 2]                                     # the board's normal in the LiDAR frame
    rb = plane_hit(u, nb, t)
    P = u * rb[:, None]
    Q = (P - t) @ R                                  # the hit in board coordinates
    on = np.isfinite(rb) & (Q[:, 0] > 0) & (Q[:, 0] < w) & (Q[:, 1] > 0) & (Q[:, 1] < h)
    rw = plane_hit(u, np.array([-1.0, 0, 0]), np.array([d + 1.2, 0, 0]))     # wall behind
    rf = plane_hit(u, np.array([0, 0, 1.0]), np.array([0, 0, -0.9]))         # floor
    r = np.where(on, rb, np.minimum(rw, rf))
    ok = np.isfinite(r) & (r < 60.0)
    r, u = r[ok], u[ok]
    r = r + BIAS + rng.normal(0, SIG_A + SIG_B * r)
    return u * r[:, None]


def board_image(design, T_cam_board, rng):
    """The board rendered as the camera sees it at that pose."""
    big = design.board.generateImage((design.sx * SQ_PX, design.sy * SQ_PX)) \
        if hasattr(design.board, "generateImage") else design.board.draw(
            (design.sx * SQ_PX, design.sy * SQ_PX))
    R, t = T_cam_board[:3, :3], T_cam_board[:3, 3]
    Hm = K @ np.column_stack([R[:, 0], R[:, 1], t])
    S = np.diag([design.square_len / SQ_PX] * 2 + [1.0])
    img = cv2.warpPerspective(big, Hm @ S, (W, H), flags=cv2.INTER_AREA, borderValue=140)
    return np.clip(img.astype(np.float32) + rng.normal(0, 2.0, img.shape), 0,
                   255).astype(np.uint8)


def build(tmp, stations=None, name="board_bag"):
    """A bag with the camera, its info, /tf_static and the LiDAR. `stations`
    is [(range m, shim m)], five seconds standing at each."""
    stations = [(d, 0.0) for d in STATIONS] if stations is None else list(stations)
    design = Board("test_board", BOARD)
    T_lc = T_lidar_cam()
    T_cl = np.linalg.inv(T_lc)
    rng = np.random.default_rng(3)
    u = beams()
    S = 10 ** 9
    bag = os.path.join(tmp, name)
    MT = rosmsg.msg_type
    with BagWriter(bag) as w:
        ns0 = T0 * S
        w.write("/tf_static", rosmsg.tf_message(
            [(LIDAR_FRAME, CAM_FRAME, T_lc[:3, 3], [-0.5, 0.5, -0.5, 0.5])], ns0), ns0)
        info = MT("sensor_msgs/msg/CameraInfo")(
            header=rosmsg.header(ns0, CAM_FRAME), height=H, width=W,
            distortion_model="plumb_bob", d=np.zeros(5), k=K.reshape(-1),
            r=np.eye(3).reshape(-1),
            p=np.array([K[0, 0], 0, K[0, 2], 0, 0, K[1, 1], K[1, 2], 0, 0, 0, 1, 0]),
            binning_x=0, binning_y=0,
            roi=MT("sensor_msgs/msg/RegionOfInterest")(x_offset=0, y_offset=0, height=0,
                                                       width=0, do_rectify=False))
        w.write(INFO, info, ns0 + 1)
        k = 0
        for d, shim in stations:
            T_cb = T_cl @ T_lidar_board(d, shim)
            img = board_image(design, T_cb, rng)
            for _ in range(5):                       # 5 s standing at each station
                ns = ns0 + int(k * 1.0 * S)
                k += 1
                im = MT("sensor_msgs/msg/Image")(
                    header=rosmsg.header(ns, CAM_FRAME), height=H, width=W,
                    encoding="mono8", is_bigendian=0, step=W,
                    data=np.clip(img.astype(np.float32)
                                 + rng.normal(0, 1.0, img.shape), 0, 255).astype(np.uint8).ravel())
                w.write(IMAGE, im, ns)
                ps = scan(d, rng, u, shim)
                pts = np.zeros(len(ps), [("x", "<f4"), ("y", "<f4"), ("z", "<f4")])
                pts["x"], pts["y"], pts["z"] = ps.T
                w.write(LIDAR, rosmsg.pointcloud2(pts, ns, LIDAR_FRAME), ns)
    return bag


def main():
    fails = []
    tmp = tempfile.mkdtemp(prefix="board_noise_")
    try:
        bag = build(tmp)
        out = os.path.join(tmp, "out")
        r = subprocess.run([sys.executable, os.path.join(ROOT, "check.py"), "board", bag,
                            "--squares", "8x6", "--square-len", "0.125",
                            "--marker-len", "0.094", "--dict", "DICT_5X5_100",
                            "--period", "0.5", "--out", out],
                           capture_output=True, text=True)
        print(r.stdout[-3000:] + r.stderr[-2000:])
        if r.returncode:
            fails.append("check.py board exited %d" % r.returncode)
        rep = json.load(open(os.path.join(out, "board_noise.json")))
        st = rep["stations"]
        if len(st) != len(STATIONS):
            fails.append("%d stations, want %d" % (len(st), len(STATIONS)))
        for s, d in zip(st, STATIONS):
            want = (SIG_A + SIG_B * d) * 1e3
            if abs(s["range_m"] - d) > 0.2:
                fails.append("station at %.2f m, want %.1f" % (s["range_m"], d))
            if abs(s["beam_mm"] - want) > 0.25 * want:
                fails.append("%.0f m: %.1f mm along the beam, want %.1f"
                             % (d, s["beam_mm"], want))
            if abs(s["thick_mm"] / s["sigma_mm"] - 3.29) > 0.6:
                fails.append("%.0f m: thickness / sigma %.2f, want 3.29"
                             % (d, s["thick_mm"] / s["sigma_mm"]))
            if abs(s["bias_mm"] - BIAS * 1e3) > 6:
                fails.append("%.0f m: bias %+.1f mm, want %+.0f" % (d, s["bias_mm"], BIAS * 1e3))
            if abs(s["inc_deg"] - 20) > 6:
                fails.append("%.0f m: incidence %.1f deg, want 20" % (d, s["inc_deg"]))
            if s["points"] < 50:
                fails.append("%.0f m: %d points on the board" % (d, s["points"]))
        # the board shimmed 5 mm at a time at one distance: stations millimetres
        # apart, read as runs, and the step read back from the plane alone
        step_bag = build(tmp, [(2.0, h) for h in SHIMS], name="step_bag")
        sout = os.path.join(tmp, "step")
        r2 = subprocess.run([sys.executable, os.path.join(ROOT, "check.py"), "board", step_bag,
                             "--squares", "8x6", "--square-len", "0.125",
                             "--marker-len", "0.094", "--dict", "DICT_5X5_100",
                             "--period", "0.5", "--runs", "--station-gap", "0.003",
                             "--out", sout], capture_output=True, text=True)
        print(r2.stdout[-1800:] + r2.stderr[-800:])
        if r2.returncode:
            fails.append("check.py board --runs exited %d" % r2.returncode)
        step = json.load(open(os.path.join(sout, "board_noise.json")))["stations"]
        if len(step) != len(SHIMS):
            fails.append("%d shimmed stations, want %d" % (len(step), len(SHIMS)))
        for st_, want in zip(step[1:], np.diff(SHIMS) * 1e3):
            if st_["step_mm"] is None or abs(st_["step_mm"] - want) > 1.0:
                fails.append("step %s mm, want %+.0f" % (st_["step_mm"], want))
        if any(s["repeat_mm"] is None or s["repeat_mm"] > 1.0 for s in step):
            fails.append("plane repeats to %s mm standing still, want under 1"
                         % [s["repeat_mm"] for s in step])

        fit = rep["fit"]
        if not fit or abs(fit["a_mm"] - SIG_A * 1e3) > 2.0 \
                or abs(fit["b_mm_per_m"] - SIG_B * 1e3) > 0.5:
            fails.append("fit %s, want %.1f + %.1f d mm" % (fit, SIG_A * 1e3, SIG_B * 1e3))
        for f in ("board_noise.json", "board_noise.md", "board_noise.png"):
            if not os.path.isfile(os.path.join(out, f)):
                fails.append("%s not written" % f)
        if not fails:
            print("  ok    %s" % "; ".join(
                "%.0f m: %.1f mm (%.0f pts, %.0f deg, %+.1f mm)"
                % (s["range_m"], s["beam_mm"], s["points"], s["inc_deg"], s["bias_mm"])
                for s in st))
            print("  ok    sigma = %.1f + %.2f d mm (truth %.1f + %.1f), r2 %.2f"
                  % (fit["a_mm"], fit["b_mm_per_m"], SIG_A * 1e3, SIG_B * 1e3, fit["r2"]))
            print("  ok    shimmed 5 mm at a time at 2 m: steps %s mm, the plane repeating "
                  "to %.2f mm scan to scan"
                  % (", ".join("%+.1f" % s["step_mm"] for s in step[1:]),
                     max(s["repeat_mm"] for s in step)))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    for f in fails:
        print("  FAIL  " + f)
    print("board_noise ok" if not fails else "board_noise FAILED")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
