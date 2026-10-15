#!/usr/bin/env python3
"""The dashboard's extrinsics mode with a reference camera (--ref, --ref-pose), end to end without
cameras or ROS: server.py runs for real; its camera streams are rendered views of a simulated room
and the robot's zed_pose_server.py is a small HTTP server here.

Room: cam 5 on a wall, cam 10 on the ceiling (fixed, known poses in the map); cam 20 = the ZED on a
robot that drives to a new spot for every board placement and reports its pose like the robot would.
Checks over the dashboard's own API that both cameras get placed within 10 mm / 0.3 deg of the truth,
that the files are written, and that the page renders (screenshots if Playwright is installed).
Every stream arrives with its own delay (ZED 0.1 s, cam 5 0.4 s, cam 10 0.25 s, pose 0.05 s), as over
RTSP: there is no common clock, the pairing must still be right.

  python tests/test_ref_extrinsics.py
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as Rot

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TMP = tempfile.mkdtemp(prefix="refextr_")
SQ = 0.10
with open(os.path.join(TMP, "boards.yaml"), "w") as f:
    f.write("boards:\n  A: {dictionary: DICT_4X4_50, squares_x: 9, squares_y: 7, square_m: %.3f, marker_m: %.3f}\n"
            "floor_tags: {dictionary: DICT_APRILTAG_36h11, size_m: 0.2}\n" % (SQ, SQ * 0.75))
os.environ["ED305_BOARDS"] = os.path.join(TMP, "boards.yaml")
sys.path.insert(0, HERE)
PORT, POSE_PORT = 8099, 8765
W, H = 1280, 720
K = {5: np.array([[1000.0, 0, 640], [0, 1000.0, 360], [0, 0, 1]]),
     10: np.array([[1000.0, 0, 640], [0, 1000.0, 360], [0, 0, 1]]),
     20: np.array([[700.0, 0, 640], [0, 700.0, 360], [0, 0, 1]])}


def look_at(pos, target, up=(0, 0, 1)):
    z = np.asarray(target, float) - pos
    z /= np.linalg.norm(z)
    x = np.cross(z, up)
    if np.linalg.norm(x) < 1e-6:
        x = np.cross(z, (0, 1, 0))
    x /= np.linalg.norm(x)
    T = np.eye(4)
    T[:3, :3] = np.stack([x, np.cross(z, x), z], 1)
    T[:3, 3] = pos
    return T


TRUTH = {5: look_at(np.array([0.0, -3.0, 2.0]), [0.0, 0.0, 0.6]),
         10: look_at(np.array([0.3, 0.2, 2.7]), [0.0, 0.0, 0.0], up=(0, 1, 0))}
WORLD = {"board": np.eye(4), "robot": look_at(np.array([2.0, 0, 1.0]), [0, 0, 0.5])}
DELAY = {20: 0.10, 5: 0.40, 10: 0.25, "pose": 0.05}     # seconds each stream lags behind the room
HISTORY = []                                            # (time, board, robot) every time something moved


def world_at(t):
    """The room as it was at time t (what a stream with that delay shows now)."""
    past = [h for h in HISTORY if h[0] <= t]
    return past[-1] if past else HISTORY[0]


def move(board, robot):
    WORLD["board"], WORLD["robot"] = board, robot
    HISTORY.append((time.time(), board, robot))
BOARD_IMG = None


def render(cam):
    _, board, robot = world_at(time.time() - DELAY[cam])
    T_wc = robot if cam == 20 else TRUTH[cam]
    T = np.linalg.inv(T_wc) @ board
    s = BOARD_IMG.shape[1] / (9 * SQ)
    Hm = K[cam] @ np.column_stack([T[:3, 0], T[:3, 1], T[:3, 3]]) @ np.diag([1 / s, 1 / s, 1])
    g = cv2.warpPerspective(BOARD_IMG, Hm, (W, H), borderValue=110) if T[2, 3] > 0 else np.full((H, W), 110, np.uint8)
    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)


class WorldStream:
    def __init__(self, cam):
        self.cam, self.ok, self.n = cam, True, 0

    def latest(self):
        return render(self.cam), time.time()

    def stop(self):
        pass


class Pose(BaseHTTPRequestHandler):
    def do_GET(self):
        data = json.dumps({"ok": True, "T": world_at(time.time() - DELAY["pose"])[2].tolist(), "world_frame": "map_zed",
                           "camera_frame": "zed_left_camera_optical_frame"}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass


def api(path, body=None):
    req = urllib.request.Request("http://127.0.0.1:%d%s" % (PORT, path), data=None if body is None else json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=5) as r:
        return json.loads(r.read().decode())


def main():
    global BOARD_IMG
    os.makedirs(os.path.join(TMP, "results", "intrinsics"))
    for c, k in K.items():
        with open(os.path.join(TMP, "results", "intrinsics", "cam_%d.json" % c), "w") as f:
            json.dump({"camera": c, "K": k.tolist(), "dist": [0, 0, 0, 0, 0], "model": "full", "image_size": [W, H]}, f)
    move(WORLD["board"], WORLD["robot"])
    threading.Thread(target=ThreadingHTTPServer(("127.0.0.1", POSE_PORT), Pose).serve_forever, daemon=True).start()

    import server
    from calib.common import load_boards
    BOARD_IMG = load_boards(["A"])["A"].board.generateImage((900, 700), marginSize=0)
    server.open_streams = lambda urls: {c: WorldStream(c) for c in urls}
    sys.argv = ["server.py", "--cams", "5", "10", "20", "--url", "5=sim://5", "--url", "10=sim://10", "--url", "20=sim://zed",
                "--ref", "20", "--ref-pose", "http://127.0.0.1:%d/pose" % POSE_PORT, "--out", os.path.join(TMP, "data"),
                "--results", os.path.join(TMP, "results"), "--port", str(PORT), "--host", "127.0.0.1", "--hz", "5"]
    threading.Thread(target=server.main, daemon=True).start()
    for _ in range(50):
        try:
            api("/api/state")
            break
        except Exception:
            time.sleep(0.2)
    api("/api/mode", {"mode": "extr"})

    rng = np.random.default_rng(3)
    t_end, placements = time.time() + 150, 0
    while time.time() < t_end:
        net = api("/api/state")["net"]
        if all(v["locked"] for k, v in net["cams"].items() if k != "20"):
            break
        centre = np.array([rng.uniform(-0.4, 0.4), rng.uniform(-0.4, 0.4), rng.uniform(0.3, 0.9)])
        R = Rot.from_euler("xyz", [rng.uniform(140, 220), rng.uniform(-25, 25), rng.uniform(-30, 30)], degrees=True).as_matrix()
        Tb = np.eye(4)
        Tb[:3, :3], Tb[:3, 3] = R, centre - R @ np.array([4.5 * SQ, 3.5 * SQ, 0])
        ang = rng.uniform(0, 2 * np.pi)
        pos = centre + np.array([np.cos(ang), np.sin(ang), 0]) * rng.uniform(1.5, 2.5) + [0, 0, rng.uniform(-0.2, 0.3)]
        move(Tb, look_at(pos, centre))                                # robot drives there, board placed
        placements += 1
        time.sleep(3.5)                                               # hold still
    st = api("/api/state")
    net = st["net"]
    print("placements:", placements, "| pose:", net["pose"], "| frame:", net["frame"])
    ok = True
    for c in (5, 10):
        rec = json.load(open(os.path.join(TMP, "results", "extrinsics_ref", "cam_%d.json" % c)))
        T = np.array(rec["T"])
        dt = np.linalg.norm(T[:3, 3] - TRUTH[c][:3, 3]) * 1000
        dr = np.degrees(np.linalg.norm(Rot.from_matrix(T[:3, :3].T @ TRUTH[c][:3, :3]).as_rotvec()))
        x = net["cams"][str(c)]
        print("cam %d: locked %s, %d samples, %d placements, spread %.1f mm | error vs truth %.1f mm / %.3f deg" % (
            c, x["locked"], x["samples"], x["holds"], x["spread_mm"], dt, dr))
        ok &= x["locked"] and dt < 10 and dr < 0.3 and rec["stage"] == "locked" and rec["parent_frame"] == "map_zed"
    netjson = json.load(open(os.path.join(TMP, "results", "extrinsics_ref", "network.json")))
    print("network.json:", sorted(netjson), "| log:", [l for l in st["log"].splitlines() if "PLACED" in l])
    assert ok and sorted(netjson) == ["10", "20", "5"]

    shots = os.environ.get("SHOTS")       # optional: directory for screenshots (needs Playwright via node)
    if shots and shutil.which("node"):
        js = os.path.join(TMP, "shot.js")
        with open(js, "w") as f:
            f.write("""const { chromium } = require(process.env.PW);
(async () => { const b = await chromium.launch(); const p = await b.newPage({ viewport: { width: 1300, height: 900 } });
  const errs = []; p.on('pageerror', e => errs.push(e.message));
  await p.goto('http://127.0.0.1:%d/'); await p.waitForTimeout(3500); await p.screenshot({ path: '%s/extr_overview.png' });
  await p.goto('http://127.0.0.1:%d/#cam-5'); await p.waitForTimeout(3500); await p.screenshot({ path: '%s/extr_cam5.png' });
  console.log('page errors', JSON.stringify(errs)); await b.close(); })();""" % (PORT, shots, PORT, shots))
        root = subprocess.run(["npm", "root", "-g"], capture_output=True, text=True).stdout.strip()
        print(subprocess.run(["node", js], capture_output=True, text=True, env=dict(os.environ, PW=root + "/playwright")).stdout.strip())
    print("PASS")


if __name__ == "__main__":
    main()
