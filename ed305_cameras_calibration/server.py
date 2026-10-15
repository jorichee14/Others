#!/usr/bin/env python3
"""Calibration dashboard: ONE PC reads all 16 streams, everyone works from a browser.

  python server.py                      # then open http://<this pc>:8080 on any phone or laptop
  python server.py --fake               # no cameras: synthetic moving boards, to try the dashboard
  python server.py --cams 17 --url 17=rtsp://<robot-ip>:8554/mobile_2/color
                                        # any RTSP stream as camera 17 (not in config/cameras.yaml)
  python server.py --intrinsics 17=camera_info.yaml
                                        # camera already calibrated: rectified view right away

Anyone can tap a camera to watch it (live, rectified once calibrated, or both). Recording frames
for a camera needs a name and a board (Join), then "Record" on that camera; "Stop" releases it.
The Frames panel shows every saved frame with its detected corners (green) and, once the camera is
calibrated, the reprojected corners (red) and its error; a bad frame can be deleted there.

Each person opens the page, types their name, picks their board (A, B or C) and a camera.
Intrinsics mode: the PC detects that person's board in that camera and saves a frame by itself
  whenever the board has moved (data/<person>/intr/cam_N, every 5th frame a TEST frame). The page
  shows the frame count and the six coverage bars (x y z roll pitch yaw). "Calibrate" fits the
  camera and shows the calibrate/test error; results/intrinsics/cam_N.json, summary.csv and
  views.csv are written.
Extrinsics mode: every camera watches every active board; a board held still and seen by 2+
  cameras saves an event (data/<person>/extr/<event>). "Solve extrinsics" runs
  calibrate_extrinsics.py and the page shows cameras.csv (x y z roll pitch yaw, errors).
Everything is saved on this PC: nothing to merge.
"""
import argparse
import glob
import shutil
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from flask import Flask, Response, jsonify, request, send_from_directory

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from calib.common import (FloorTags, cam_name, camera_urls, latest_session, load_boards, now_tag,  # noqa: E402
                          read_csv, read_intrinsics_file, read_json, write_csv)
from calib.intr import (DEFAULT_MODEL, FIELDS, MODELS, board_pose, coverage, nominal_K, project_model, ready,  # noqa: E402
                        split_for, view_features)
from calib.network import BoardPose, Pair, Still, finite, pose_dict  # noqa: E402
from calib.streams import open_streams  # noqa: E402
from calibrate_intrinsics import solve_camera  # noqa: E402
from capture import save_frame  # noqa: E402
from calib.intr import summary_row, view_rows  # noqa: E402

MIN_CORNERS = 12
ACTIVE_S = 60          # a person whose browser has not polled for this long is inactive
STILL_PX, MOVE_PX = 2.0, 25.0
STILL_S = 0.8          # the board must have been within STILL_PX for this long to count as still
VIEW_S = 4.0           # a camera someone has looked at within this many seconds is refreshed every tick

app = Flask(__name__, static_folder=os.path.join(HERE, "web"), static_url_path="/web")
app.json.sort_keys = False
S = {"mode": "intr", "people": {}, "armed": {}, "events": {}, "log": [], "busy": ""}
LOCK = threading.Lock()
CAMS, STREAMS, BOARDS, TAGS, RES, ARGS = {}, {}, {}, None, {}, None
NET, REFPOSE = {}, None          # extrinsics against a reference camera (--ref): per camera state, the ZED pose


def log(msg):
    """Keep the last lines shown on the page (events, calibrations, solver output)."""
    print(msg)
    with LOCK:
        S["log"] = (S["log"] + msg.rstrip().split("\n"))[-25:]


class Cam:
    def __init__(self, cam, url=""):
        self.cam = cam
        self.url = url
        self.label = url.split("://")[-1].split("/", 1)[-1] if url else ""
        self.lock = threading.Lock()
        self.person = None                        # who records frames on this camera (None: nobody)
        self.viewed, self.rect_wanted = 0.0, 0.0  # last time a browser fetched its frame / rectified frame
        self.jpeg_rect = None
        self.size = None                          # (w, h) of the stream
        self.calibrated, self.source = False, ""  # real K known (and where it came from)
        self.session = 1                          # calibration round: frames in cam_N, then cam_N_2, ...
        self.calibrating = False                  # a Calibrate is running (status text is overwritten by the live view)
        self._maps = None
        self.count, self.feats, self.last_saved = 0, [], None
        self.rms, self.status, self.last_board = "", "idle", ""
        self.jpeg, self.jpeg_t = None, 0.0
        self.still, self.still_frame = {}, {}     # board -> recent centres / (frame, corners, ids, tags, t, person)
        self.K, self.dist = nominal_K(), None
        self.model = "full"                       # lens model of the current K/dist (calib.intr.MODELS)

    def lens(self):
        """(K, dist, model) to use now: the calibration, or the nominal pinhole before there is one."""
        return (self.K, self.dist, self.model) if self.calibrated else (self.K, None, "full")

    def set_intrinsics(self, K, dist, source, size=None, model="full"):
        """Real K and distortion (rectified view from now on). size: the image size K belongs to;
        K is scaled when the stream turns out to have another resolution."""
        with self.lock:
            self.K, self.dist, self.model = np.array(K, float), np.array(dist, float).reshape(-1), model
            self.k_size = tuple(size) if size else None
            self.calibrated, self.source, self._maps = True, source, None

    def check_size(self, w, h):
        """Called with every frame: nominal K for the real resolution, or the real K scaled to it."""
        if self.size == (w, h):
            return
        self.size = (w, h)
        with self.lock:
            if not self.calibrated:
                self.K = nominal_K((w, h))
            elif self.k_size and self.k_size != (w, h):
                sx, sy = w / self.k_size[0], h / self.k_size[1]
                self.K = np.diag([sx, sy, 1.0]) @ self.K
                log("cam %d: intrinsics were for %dx%d, stream is %dx%d: K scaled" % (self.cam, *self.k_size, w, h))
                self.k_size = (w, h)
            self._maps = None

    def rectify(self, frame):
        """Undistorted frame (same K for the output, cropped to valid pixels), or None if not calibrated."""
        if not self.calibrated:
            return None
        h, w = frame.shape[:2]
        with self.lock:
            if self._maps is None or self._maps[0] != (w, h):
                if self.model == "fisheye":
                    D = self.dist.reshape(4, 1)
                    newK = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(self.K, D, (w, h), np.eye(3), balance=0.0)
                    m1, m2 = cv2.fisheye.initUndistortRectifyMap(self.K, D, np.eye(3), newK, (w, h), cv2.CV_16SC2)
                else:
                    newK, _ = cv2.getOptimalNewCameraMatrix(self.K, self.dist, (w, h), 0)
                    m1, m2 = cv2.initUndistortRectifyMap(self.K, self.dist, None, newK, (w, h), cv2.CV_16SC2)
                self._maps = ((w, h), m1, m2)
            _, m1, m2 = self._maps
        return cv2.remap(frame, m1, m2, cv2.INTER_LINEAR)

    def load_existing(self, data_dir):
        """Counts and bar features of frames already on disk (so a restart keeps the bars)."""
        self.count, self.feats = 0, []
        for path in sorted(glob.glob(os.path.join(data_dir, "*", "intr", cam_name(self.cam, self.session), "*.json"))):
            j = read_json(path)
            self.count += 1
            if j.get("split") == "calib" and j.get("features"):
                self.feats.append(j["features"])


# ---------------------------------------------------------------- fake cameras (no lab needed)
class FakeStream:
    """A synthetic pinhole camera (f = 1400 px, no distortion): the three boards move in 3D in front
    of it and pause for 3 s every 8 s (so extr events fire). A calibration of it should give
    fx = fy = 1400, cx = 960, cy = 540 and an rms well under 1 px."""
    F = 1400.0

    def __init__(self, cam, boards):
        self.cam = cam
        self.t0 = time.time()
        self.imgs, self.sizes = {}, {}
        for n, b in boards.items():
            self.imgs[n] = cv2.cvtColor(b.board.generateImage((770, 560), marginSize=0), cv2.COLOR_GRAY2BGR)
            self.sizes[n] = b.board.getChessboardSize()[0] * b.board.getSquareLength(), \
                b.board.getChessboardSize()[1] * b.board.getSquareLength()

    def latest(self):
        t = time.time() - self.t0
        ph = t % 8.0
        te = t - ph + min(ph, 5.0)
        frame = np.full((1080, 1920, 3), 90, np.uint8)
        cv2.putText(frame, "FAKE cam %d" % self.cam, (40, 1040), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (200, 200, 200), 3)
        K = np.array([[self.F, 0, 960], [0, self.F, 540], [0, 0, 1]])
        for k, (n, img) in enumerate(self.imgs.items()):
            if (self.cam + k) % 3 != 0 and np.sin(0.05 * te + k) < 0.3:
                continue
            bw, bh = self.sizes[n]
            z = 2.2 + 1.2 * np.sin(0.09 * te + k + self.cam)
            tx = 0.35 * z * np.sin(0.21 * te + k * 2.1 + self.cam)
            ty = 0.2 * z * np.cos(0.17 * te + k * 1.3 + self.cam)
            rvec = np.array([0.5 * np.sin(0.13 * te + k), 0.5 * np.cos(0.19 * te + k + 1), 0.6 * np.sin(0.08 * te + k)])
            R, _ = cv2.Rodrigues(rvec)
            corners3 = np.array([[0, 0, 0], [bw, 0, 0], [bw, bh, 0], [0, bh, 0]], float) - [bw / 2, bh / 2, 0]
            pc = corners3 @ R.T + [tx, ty, z]
            if (pc[:, 2] < 0.3).any():
                continue
            dst = (pc @ K.T)
            dst = (dst[:, :2] / dst[:, 2:]).astype(np.float32)
            h, w = img.shape[:2]
            H = cv2.getPerspectiveTransform(np.float32([[0, 0], [w, 0], [w, h], [0, h]]), dst)
            warped = cv2.warpPerspective(img, H, (1920, 1080), borderValue=(90, 90, 90))
            mask = cv2.warpPerspective(np.full((h, w), 255, np.uint8), H, (1920, 1080))
            frame[mask > 0] = warped[mask > 0]
        return frame, time.time()

    def stop(self):
        pass


# ---------------------------------------------------------------- processing
def active_people():
    now = time.time()
    with LOCK:
        return {n: dict(p) for n, p in S["people"].items() if now - p["seen"] < ACTIVE_S}


def board_users(active):
    """{board: one person using it}: copies of the same board are one board for extrinsics."""
    out = {}
    for person, info in sorted(active.items()):
        out.setdefault(info["board"], person)
    return out


def tag(img, text, color=(255, 255, 255), bottom=False):
    """Small label on a dark strip in a corner, sized to the image (the page shows the rest)."""
    h, w = img.shape[:2]
    fs = max(0.55, w / 1300)
    th = max(1, int(round(fs * 1.6)))
    (tw, tht), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, th)
    pad = int(6 * fs + 2)
    y0 = h - tht - base - 2 * pad if bottom else 0
    cv2.rectangle(img, (0, y0), (tw + 2 * pad, y0 + tht + base + 2 * pad), (20, 20, 20), -1)
    cv2.putText(img, text, (pad, y0 + pad + tht), cv2.FONT_HERSHEY_SIMPLEX, fs, color, th, cv2.LINE_AA)


def dots(img, corners, tag_det=None, color=(0, 255, 0)):
    """Detected board corners (and floor tags) as dots sized to the image."""
    r = max(3, img.shape[1] // 300)
    if corners is not None:
        for c in corners.reshape(-1, 2):
            cv2.circle(img, (int(c[0]), int(c[1])), r, color, -1, cv2.LINE_AA)
    for tid, c in (tag_det or {}).items():
        cv2.polylines(img, [c.astype(int)], True, (255, 0, 255), max(2, r // 2))


def encode(frame, width=960):
    h, w = frame.shape[:2]
    small = cv2.resize(frame, (width, int(round(h * width / w))), interpolation=cv2.INTER_AREA) if w > width else frame
    return cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 70])[1].tobytes()


def publish(cs, raw, shown):
    """JPEGs for the page: the annotated frame, and the rectified raw frame when someone watches it."""
    cs.jpeg, cs.jpeg_t = encode(shown), time.time()
    if cs.calibrated and time.time() - cs.rect_wanted < VIEW_S:
        rect = cs.rectify(raw)
        tag(rect, "rectified", (255, 220, 0))
        cs.jpeg_rect = encode(rect)


def process(cam, mode, active):
    cs = CAMS[cam]
    frame, t = STREAMS[cam].latest()
    if frame is None:
        cs.status = "no stream"
        return
    h, w = frame.shape[:2]
    cs.check_size(w, h)
    raw = frame.copy() if cs.calibrated else frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    tag_det = {}                      # floor tags are slow to detect: only on the frames that get saved
    if mode == "intr":
        person = cs.person
        if person not in active:
            cs.status = "idle"
            return
        b = active[person]["board"]
        corners, ids = BOARDS[b].detect(gray)
        seen = corners is not None and len(corners) >= MIN_CORNERS
        label = "REC  %d saved" % cs.count
        cs.status = "board seen" if seen else "no board"
        if seen:
            centre, size = corners.mean(0), np.ptp(corners, axis=0).max()
            prev = cs.last_saved
            moved = cs.person is not None and (prev is None or np.linalg.norm(centre - prev[0]) > 0.06 * w or abs(size - prev[1]) > 0.12 * prev[1])
            if moved:
                tag_det = TAGS.detect(gray)
                with cs.lock:
                    name = now_tag()
                    split = split_for(cs.count)
                    K, dist, model = cs.lens()
                    f = view_features(BOARDS[b], corners, ids, K, dist, (w, h), model)
                    out_root = os.path.join(ARGS.out, person, "intr")
                    save_frame(os.path.join(out_root, cam_name(cam, cs.session)), name, frame, cam, b, corners, ids, tag_det, t,
                               split=split, features=f)
                    write_csv(os.path.join(out_root, "views.csv"), [{
                        "person": person, "board": b, "camera": cam, "file": name, "time": t, "split": split,
                        "corners": len(corners), **{k: (round(f[k], 3) if f else "") for k in FIELDS}}], append=True)
                    cs.count += 1
                    if split == "calib" and f:
                        cs.feats.append(f)
                    cs.last_saved = (centre, size)
                label += "  SAVED " + split
        dots(frame, corners, tag_det)
        tag(frame, label if seen else label + "  no board", (0, 255, 0) if seen else (0, 200, 255))
    else:
        found = []
        for b, person in board_users(active).items():
            corners, ids = BOARDS[b].detect(gray)
            seen = corners is not None and len(corners) >= MIN_CORNERS
            if seen:
                hist = cs.still.setdefault(b, [])
                c0 = corners.mean(0)
                hist.append((t, c0))
                hist[:] = [x for x in hist if t - x[0] <= 2.5]
                near = [x for x in hist if np.linalg.norm(c0 - x[1]) < STILL_PX]
                is_still = len(near) >= 2 and t - min(x[0] for x in near) >= STILL_S
                if len(hist) >= 2 and np.linalg.norm(c0 - hist[-2][1]) > MOVE_PX:
                    with LOCK:
                        S["armed"][b] = True
                if is_still:
                    cs.still_frame[b] = (frame.copy(), corners, ids, gray, t, person)
                else:
                    cs.still_frame.pop(b, None)
                found.append("%s %s" % (b, "STILL" if is_still else "moving"))
                dots(frame, corners, None, (0, 255, 0) if is_still else (0, 200, 255))
            else:
                cs.still[b] = []
                cs.still_frame.pop(b, None)
        cs.status = ", ".join(found) if found else "no board"
        tag(frame, "cam %d  %s" % (cam, cs.status), (0, 255, 0))
    publish(cs, raw, frame)


def extr_events(active):
    now = time.time()
    for b, person in board_users(active).items():
        static = {c: cs.still_frame[b] for c, cs in CAMS.items() if b in cs.still_frame and now - cs.still_frame[b][4] < 1.5}
        with LOCK:
            armed = S["armed"].get(b, True)
        if armed and len(static) >= 2:
            with LOCK:
                S["armed"][b] = False
                S["events"][b] = S["events"].get(b, 0) + 1
                n = S["events"][b]
            log("event %d for board %s: cams %s" % (n, b, sorted(static)))
            threading.Thread(target=save_event, args=(person, b, now_tag(), static), daemon=True).start()


def save_event(person, b, ev, static):
    """Write one still-board event (frames of every camera that saw it still), tags detected now."""
    for cam, (frame, corners, ids, gray, t, who) in static.items():
        save_frame(os.path.join(ARGS.out, person, "extr", ev), "cam_%d" % cam, frame, cam, b,
                   corners, ids, TAGS.detect(gray), t, event=ev)


# ---------------------------------------------------------------- extrinsics against a reference camera
# (--ref N: camera N is the ZED; every other calibrated camera is placed relative to it, or in the ZED map
#  when --ref-pose gives the ZED's pose. Same method as radar_camera_calibration/network_charuco.py.)
class RefPose(threading.Thread):
    """Polls the robot's zed_pose_server.py (GET /pose) and tells whether the robot is still."""

    def __init__(self, url):
        super().__init__(daemon=True)
        self.url, self.T, self.t, self.error, self.world = url, None, 0.0, "not polled yet", ""
        self.still = Still(ARGS.still_s, 0.005, 0.3)
        self.start()

    def run(self):
        import json
        import urllib.request
        while True:
            try:
                with urllib.request.urlopen(self.url, timeout=1.0) as r:
                    j = json.loads(r.read().decode())
                if j.get("ok"):
                    self.T, self.t, self.error = np.array(j["T"], float), time.time(), ""
                    self.world = j.get("world_frame", "map")
                    self.still.add(self.t, self.T)
                else:
                    self.T, self.error = None, j.get("error", "no pose")
                    self.still.add(time.time(), None)
            except Exception as e:  # server down, robot off the network
                self.T, self.error = None, str(e)
                self.still.add(time.time(), None)
            time.sleep(0.1)

    def now(self):
        """T_world_ref if the pose is fresh and the robot has been still, else None."""
        if self.T is None or time.time() - self.t > 1.0 or not self.still.still:
            return None
        return self.T.copy()

    def status(self):
        if self.T is None:
            return {"ok": False, "text": "no pose: " + self.error}
        return {"ok": True, "still": self.still.still, "text": "robot still" if self.still.still else "robot moving",
                "world": self.world}


class NetCam:
    """One camera in reference-camera extrinsics: board pose, stillness, pooled pose, round."""

    def __init__(self, cam, board):
        self.cam, self.board = cam, board
        self.pose = BoardPose(board, 8)
        self.still = Still(ARGS.still_s, 0.003, 0.3)
        self.pair = self.new_pair()
        self.T, self.reproj, self.n, self.t, self.sees = None, None, 0, 0.0, "no stream"
        self.last_sample, self.session, self.saved = 0.0, 1, None

    @staticmethod
    def new_pair():
        return Pair("cam", conv_window=15, conv_hold=10, conv_t_std=0.003, conv_rot_std=0.2, conv_pool=30, min_holds=3)


def net_dir():
    return os.path.join(ARGS.results, "extrinsics_ref")


def net_parent():
    return (REFPOSE.world or "map") if REFPOSE is not None else "cam_%d" % ARGS.ref


def net_record(ns):
    """What is saved for a camera (same keys as network_charuco.py's cam_N.json)."""
    X = ns.pair.locked if ns.pair.locked is not None else ns.pair.X
    if X is None:
        return ns.saved
    st = ns.pair.status()
    return {"parent_frame": net_parent(), "child_frame": "cam_%d" % ns.cam, "reference_camera": ARGS.ref,
            "camera": ns.cam, "session": ns.session, **pose_dict(X),
            "stage": "locked" if ns.pair.locked is not None else "live", "reproj_a_px": NET[ARGS.ref].reproj,
            "reproj_b_px": st["reproj_b_px"], "samples": st["samples"], "board_placements": st["holds"],
            "spread_mm": st["spread_mm"], "spread_deg": st["spread_deg"], "t_std_mm": st["t_std_mm"],
            "rot_std_deg": st["rot_std_deg"], "time": time.strftime("%Y-%m-%d %H:%M:%S")}


def net_save(ns):
    import json
    rec = net_record(ns)
    if rec is None:
        return
    os.makedirs(net_dir(), exist_ok=True)
    with open(os.path.join(net_dir(), cam_name(ns.cam, ns.session) + ".json"), "w") as f:
        json.dump(rec, f, indent=2)
    allc = {str(c): net_record(n) for c, n in NET.items() if c != ARGS.ref and net_record(n) is not None}
    allc[str(ARGS.ref)] = {"camera": ARGS.ref, "reference": True, "parent_frame": net_parent()}
    with open(os.path.join(net_dir(), "network.json"), "w") as f:
        json.dump(allc, f, indent=2)


def net_load():
    """Cameras locked in an earlier run (newest round of each) count as done after a restart."""
    import json
    import re
    d = net_dir()
    if not os.path.isdir(d):
        return
    for fn in os.listdir(d):
        m = re.match(r"^cam_(\d+)(?:_(\d+))?\.json$", fn)
        if not m or int(m.group(1)) not in NET or int(m.group(1)) == ARGS.ref:
            continue
        ns, k = NET[int(m.group(1))], int(m.group(2) or 1)
        if k < ns.session:
            continue
        ns.session = k
        j = json.load(open(os.path.join(d, fn)))
        ns.saved = j if j.get("stage") == "locked" else None
        ns.pair = NetCam.new_pair()
        if ns.saved:
            ns.pair.locked = np.array(j["T"], float)


def net_detect(cam):
    """Board pose in the newest frame of one camera; the live view shows the corners."""
    cs, ns = CAMS[cam], NET[cam]
    frame, _ = STREAMS[cam].latest()
    if frame is None:
        cs.status, ns.sees, ns.T = "no stream", "no stream", None
        ns.still.add(time.time(), None)
        return
    cs.check_size(frame.shape[1], frame.shape[0])
    K, dist, model = cs.lens()
    T = None
    if not cs.calibrated or model == "fisheye":
        ns.sees, ns.n = "needs intrinsics" if not cs.calibrated else "fisheye not supported", 0
    else:
        T, r, n = ns.pose(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), K, np.asarray(dist, float))
        if T is not None and r > 1.5:
            T = None
        ns.reproj, ns.n = r, n
    ns.T, ns.t = T, time.time()
    still = ns.still.add(ns.t, T)
    if T is not None:
        ns.sees = "board still" if still else "board moving"
    elif cs.calibrated and model != "fisheye":
        ns.sees = "no board"
    raw = frame.copy() if cs.calibrated else frame
    if ns.pose.corners is not None:
        dots(frame, ns.pose.corners, None, (0, 255, 0) if still else (0, 200, 255))
    label = "REFERENCE" if cam == ARGS.ref else ("PLACED" if ns.pair.locked is not None else "%d samples" % len(ns.pair.pool))
    tag(frame, "cam %d  %s  %s" % (cam, label, ns.sees), (0, 255, 0) if still else (0, 200, 255))
    cs.status = ns.sees
    publish(cs, raw, frame)


def net_pair():
    """Every camera that sees the board still together with the reference camera gets a sample."""
    ref, now = NET[ARGS.ref], time.time()
    if ref.T is None or not ref.still.still or now - ref.t > 1.0:
        return
    W = None
    if REFPOSE is not None:
        W = REFPOSE.now()
        if W is None:
            return
    for cam, ns in NET.items():
        if cam == ARGS.ref or ns.pair.locked is not None or ns.T is None or not ns.still.still:
            continue
        if now - ns.t > 1.0 or now - ns.last_sample < 0.2:
            continue
        X = ref.T @ np.linalg.inv(ns.T)                # T_ref_cam
        if W is not None:
            X = W @ X                                  # T_world_cam
        ns.last_sample = now
        if ns.pair.add(X, (ref.still.hold, ns.still.hold), ns.reproj):
            net_save(ns)
            st = ns.pair.status()
            log("cam %d PLACED in %s: %d samples, %d placements, spread %.1f mm / %.2f deg -> %s" % (
                cam, net_parent(), st["samples"], st["holds"], st["spread_mm"], st["spread_deg"],
                cam_name(cam, ns.session) + ".json"))


def net_state():
    if ARGS.ref is None:
        return None
    cams = {}
    for c, ns in NET.items():
        st = ns.pair.status()
        rec = ns.saved if (ns.pair.locked is not None and ns.saved and not ns.pair.pool) else None
        X = ns.pair.locked if ns.pair.locked is not None else ns.pair.X
        cams[c] = {"reference": c == ARGS.ref, "sees": ns.sees, "locked": ns.pair.locked is not None,
                   "samples": rec["samples"] if rec else st["samples"], "holds": rec["board_placements"] if rec else st["holds"],
                   "spread_mm": rec["spread_mm"] if rec else st["spread_mm"],
                   "spread_deg": rec["spread_deg"] if rec else st["spread_deg"],
                   "t_std_mm": st["t_std_mm"], "rot_std_deg": st["rot_std_deg"], "session": ns.session,
                   "position": None if X is None else [round(float(v), 3) for v in X[:3, 3]]}
    return finite({"ref": ARGS.ref, "frame": net_parent(), "pose": REFPOSE.status() if REFPOSE is not None else None,
                   "cams": {str(c): v for c, v in cams.items()}})


def loop():
    pool = ThreadPoolExecutor(max_workers=ARGS.workers)
    period = 1.0 / ARGS.hz
    while True:
        t0 = time.time()
        with LOCK:
            mode = S["mode"]
        active = active_people()
        if mode == "intr":
            jobs = [c for c in CAMS if CAMS[c].person in active]
        else:
            jobs = list(CAMS) if active else []
        if mode == "extr" and ARGS.ref is not None:      # extrinsics against the reference camera (the ZED)
            jobs = list(CAMS)
            list(pool.map(net_detect, jobs))
            net_pair()
        else:
            list(pool.map(lambda c: process(c, mode, active), jobs))
        if mode == "extr" and ARGS.ref is None:
            extr_events(active)
        for c, cs in CAMS.items():      # other cameras: raw view every tick if watched, else once a second
            watched = time.time() - cs.viewed < VIEW_S
            if c not in jobs and (watched or time.time() - cs.jpeg_t > 1.0):
                frame, _ = STREAMS[c].latest()
                if frame is not None:
                    cs.check_size(frame.shape[1], frame.shape[0])
                    raw = frame.copy() if cs.calibrated else frame
                    tag(frame, "cam %d" % c, (200, 200, 200))
                    publish(cs, raw, frame)
                    if cs.person is None:
                        cs.status = "live"
                else:
                    cs.status = "no stream"
        time.sleep(max(0.0, period - (time.time() - t0)))


# ---------------------------------------------------------------- solving
def write_summaries():
    """summary.csv / views.csv for every camera solved in this server (keeping rows of other cameras)."""
    d = os.path.join(ARGS.results, "intrinsics")
    old_s = [r for r in read_csv(os.path.join(d, "summary.csv")) if int(r["camera"]) not in RES]
    old_v = [r for r in read_csv(os.path.join(d, "views.csv")) if int(r["camera"]) not in RES]
    rows_s = [summary_row(c, RES[c]) for c in sorted(RES)]
    rows_v = sum((view_rows(c, RES[c]) for c in sorted(RES)), [])
    if rows_s:
        write_csv(os.path.join(d, "summary.csv"), old_s + rows_s, list(rows_s[0].keys()))
        write_csv(os.path.join(d, "views.csv"), old_v + rows_v, list(rows_v[0].keys()))


def calibrate_thread(cam, model):
    cs = CAMS[cam]
    try:
        res = solve_camera(cam, ARGS.out, ARGS.results, BOARDS, session=cs.session, model=model)
        if res is None:
            cs.rms = "fewer than 10 calibrate views"
        else:
            cs.set_intrinsics(res["K"], res["dist"], "calibrated here, %s.json" % cam_name(cam, cs.session),
                              res["views"][0]["image_size"], model)
            cs.rms = "rms calib %.2f test %.2f px (%d+%d views) %s" % (
                res["rms_calib"], res["rms_test"], res["n_calib"], res["n_test"], summary_row(cam, res)["status"])
            locked = None
            with LOCK:
                RES[cam] = res
                try:
                    write_summaries()
                except PermissionError as e:   # Windows: the CSV is open in Excel
                    locked = os.path.basename(e.filename or "a CSV file")
            if locked:
                log("cam %d: calibration saved in %s.json, but %s could not be written (open in Excel? "
                    "close it, then press Calibrate again to refresh the CSV files)" % (cam, cam_name(cam, cs.session), locked))
    except Exception as e:  # keep the server alive
        cs.rms = "error: %s" % e
    cs.calibrating = False
    log("cam %d: %s" % (cam, cs.rms))


def extrinsics_thread():
    with LOCK:
        S["busy"] = "solving extrinsics..."
    p = subprocess.run([sys.executable, os.path.join(HERE, "calibrate_extrinsics.py"), "--data", ARGS.out,
                        "--results", ARGS.results], capture_output=True, text=True)
    with LOCK:
        S["busy"] = ""
    log("--- calibrate_extrinsics.py ---\n" + "\n".join((p.stdout + p.stderr).rstrip().split("\n")[-14:]))


# ---------------------------------------------------------------- web
@app.route("/")
def index():
    return send_from_directory(os.path.join(HERE, "web"), "dashboard.html")


@app.route("/frame/<int:cam>.jpg")
def frame(cam):
    """Latest view of a camera; ?view=rect for the rectified one (calibrated cameras only)."""
    cs = CAMS.get(cam)
    if cs is None:
        return Response(status=404)
    rect = request.args.get("view") == "rect"
    if request.args.get("thumb") != "1":       # overview thumbnails: the once-a-second refresh is enough
        cs.viewed = time.time()
    if rect:
        cs.rect_wanted = time.time()
    jpeg = cs.jpeg_rect if rect else cs.jpeg
    if jpeg is None:
        return Response(status=204)
    return Response(jpeg, mimetype="image/jpeg", headers={"Cache-Control": "no-store"})


# ---------------------------------------------------------------- saved frames
def view_files(cam):
    """Saved intr frames of a camera, newest first: (id 'person/name', json path)."""
    paths = glob.glob(os.path.join(ARGS.out, "*", "intr", cam_name(cam, CAMS[cam].session), "*.json"))
    out = [("%s/%s" % (p.split(os.sep)[-4], os.path.basename(p)[:-5]), p) for p in paths]
    return sorted(out, key=lambda x: x[0].split("/")[1], reverse=True)


def view_errors(cam):
    """{json path: (rms_px, used)} from the last fit of this camera (this run, or views.csv on disk)."""
    if cam in RES:
        return {v["path"]: (v["rms_px"], v["used"]) for v in RES[cam]["views"]}
    out = {}
    for r in read_csv(os.path.join(ARGS.results, "intrinsics", "views.csv")):
        if r.get("camera") and int(r["camera"]) == cam:
            out[r["file"]] = (float(r["rms_px"]) if r["rms_px"] else float("nan"), r["used"] == "1")
    return out


def view_path(cam, vid):
    """json path of a saved frame from its id, refusing anything outside this camera's folders."""
    person, _, name = vid.partition("/")
    if not person or not name or os.sep in name or ".." in vid or "/" in name:
        return None
    p = os.path.join(ARGS.out, person, "intr", cam_name(cam, CAMS[cam].session), name + ".json")
    return p if os.path.exists(p) else None


def result_file(cam):
    """The newest result file of a camera (its current round or an earlier one), or None."""
    cs = CAMS[cam]
    for k in range(cs.session, 0, -1):
        p = os.path.join(ARGS.results, "intrinsics", cam_name(cam, k) + ".json")
        if os.path.exists(p):
            return p
    return None


@app.route("/api/result/<int:cam>")
def result(cam):
    """This camera's calibration as stored on disk (K, distortion, errors, views), for the camera page."""
    if cam not in CAMS:
        return jsonify({"error": "unknown camera"}), 404
    p = result_file(cam)
    if p is None:
        cs = CAMS[cam]
        return jsonify({"file": None, "source": cs.source, "K": cs.K.tolist() if cs.calibrated else None,
                        "dist": cs.dist.tolist() if cs.calibrated and cs.dist is not None else None})
    j = read_json(p)
    j.pop("frames", None)
    j.pop("test_frames", None)
    j["file"] = os.path.relpath(p, HERE) if os.path.abspath(p).startswith(HERE) else p
    j["name"] = os.path.basename(p)
    j["saved"] = time.strftime("%Y-%m-%d %H:%M", time.localtime(os.path.getmtime(p)))
    return jsonify(j)


@app.route("/download/<int:cam>.json")
def download(cam):
    p = result_file(cam) if cam in CAMS else None
    if p is None:
        return Response(status=404)
    return send_from_directory(os.path.dirname(os.path.abspath(p)), os.path.basename(p), as_attachment=True)


@app.route("/api/views/<int:cam>")
def views(cam):
    err = view_errors(cam)
    out = []
    for vid, p in view_files(cam):
        j = read_json(p)
        e = err.get(p) or err.get(os.path.relpath(p)) or (float("nan"), None)
        out.append({"id": vid, "split": j.get("split") or "", "board": j.get("board"),
                    "corners": len((j.get("charuco") or {}).get("ids") or []),
                    "rms_px": None if not np.isfinite(e[0]) else round(e[0], 3), "used": e[1]})
    return jsonify({"camera": cam, "views": out})


@app.route("/view/<int:cam>/<path:vid>.jpg")
def view_image(cam, vid):
    """A saved frame with its detected corners (green) and, if calibrated, the reprojected ones (red)."""
    p = view_path(cam, vid)
    cs = CAMS.get(cam)
    if p is None or cs is None:
        return Response(status=404)
    img = cv2.imread(p[:-5] + ".jpg")
    j = read_json(p)
    ch = j.get("charuco")
    if img is None:
        return Response(status=404)
    full = request.args.get("full") == "1"
    r = max(2, img.shape[1] // 400)
    if ch:
        corners, ids = np.array(ch["corners"], np.float32).reshape(-1, 2), np.array(ch["ids"], int).reshape(-1)
        for c in corners:
            cv2.circle(img, tuple(np.round(c).astype(int)), r + 2, (0, 255, 0), 2)
        b = BOARDS.get(j.get("board"))
        if cs.calibrated and b is not None:
            K, dist, model = cs.lens()
            pose = board_pose(b, corners, ids, K, dist, model)
            if pose is not None:
                obj = b.object_points[ids]
                pr = project_model(obj, pose[0], pose[1], K, dist, model)
                for c in pr:
                    cv2.circle(img, tuple(np.round(c).astype(int)), r, (0, 0, 255), -1)
                e = np.sqrt(np.mean(np.sum((pr - corners) ** 2, axis=1)))
                tag(img, "%.2f px" % e, (80, 80, 255), bottom=True)
    tag(img, "%s  %s" % (j.get("split") or "", vid.split("/")[1]), (0, 255, 255))
    return Response(encode(img, img.shape[1] if full else 480), mimetype="image/jpeg")


@app.route("/coverage/<int:cam>.jpg")
def coverage_image(cam):
    """Every corner of every saved frame on one image: where in the image the board has been."""
    cs = CAMS.get(cam)
    if cs is None:
        return Response(status=404)
    w, h = cs.size or (1920, 1080)
    s = 480.0 / w
    img = np.full((int(h * s), 480, 3), 24, np.uint8)
    for k in range(1, 4):
        cv2.line(img, (int(k * 120), 0), (int(k * 120), img.shape[0]), (60, 60, 60), 1)
        cv2.line(img, (0, int(k * img.shape[0] / 4)), (480, int(k * img.shape[0] / 4)), (60, 60, 60), 1)
    for _, p in view_files(cam):
        j = read_json(p)
        ch = j.get("charuco")
        if not ch:
            continue
        col = (0, 160, 255) if j.get("split") == "test" else (80, 220, 80)
        for c in np.array(ch["corners"], float).reshape(-1, 2) * s:
            cv2.circle(img, (int(c[0]), int(c[1])), 1, col, -1)
    return Response(encode(img, 480), mimetype="image/jpeg", headers={"Cache-Control": "no-store"})


@app.route("/api/delete_view", methods=["POST"])
def delete_view():
    """Move a saved frame to cam_N/deleted/ (kept on disk, no longer used); recalibrate afterwards."""
    j = request.get_json(force=True)
    cam = j.get("camera")
    p = view_path(cam, j.get("id") or "") if cam in CAMS else None
    if p is None:
        return jsonify({"error": "no such frame"}), 404
    d = os.path.join(os.path.dirname(p), "deleted")
    os.makedirs(d, exist_ok=True)
    for ext in (".json", ".jpg"):
        if os.path.exists(p[:-5] + ext):
            shutil.move(p[:-5] + ext, os.path.join(d, os.path.basename(p[:-5]) + ext))
    with CAMS[cam].lock:
        CAMS[cam].load_existing(ARGS.out)
    log("cam %d: frame %s deleted by %s (press Calibrate again)" % (cam, j["id"], j.get("name") or "?"))
    return jsonify({"ok": True})


@app.route("/api/state")
def state():
    name = request.args.get("name")
    with LOCK:
        if name and name in S["people"]:
            S["people"][name]["seen"] = time.time()
        people = {n: {"board": p["board"], "camera": p.get("camera"), "active": time.time() - p["seen"] < ACTIVE_S}
                  for n, p in S["people"].items()}
        mode, events, lines, busy = S["mode"], dict(S["events"]), list(S["log"]), S["busy"]
    cams = {}
    for c, cs in CAMS.items():
        cov = coverage(cs.feats)
        K = cs.K if cs.calibrated else None
        cams[c] = {"person": cs.person, "count": cs.count, "calib": len(cs.feats), "coverage": cov,
                   "ready": ready(cs.feats), "rms": cs.rms, "status": "calibrating" if cs.calibrating else cs.status, "label": cs.label, "url": cs.url,
                   "size": cs.size, "calibrated": cs.calibrated, "source": cs.source, "session": cs.session,
                   "model": cs.model if cs.calibrated else None,
                   "round_done": os.path.exists(os.path.join(ARGS.results, "intrinsics", cam_name(c, cs.session) + ".json")),
                   "K": None if K is None else {"fx": round(K[0, 0], 1), "fy": round(K[1, 1], 1),
                                                "cx": round(K[0, 2], 1), "cy": round(K[1, 2], 1)}}
    return jsonify({"mode": mode, "people": people, "cams": cams, "events": events, "log": "\n".join(lines), "busy": busy,
                    "fields": FIELDS, "boards": sorted(BOARDS), "models": MODELS, "default_model": ARGS.model,
                    "net": net_state()})


@app.route("/api/join", methods=["POST"])
def join():
    j = request.get_json(force=True)
    name, board = (j.get("name") or "").strip(), j.get("board")
    if not name or board not in BOARDS:
        return jsonify({"error": "name and board (%s) needed" % "/".join(sorted(BOARDS))}), 400
    with LOCK:   # several people may use copies of the same board (one camera each when recording)
        p = S["people"].setdefault(name, {"camera": None})
        p.update({"board": board, "seen": time.time()})
    return jsonify({"ok": True})


@app.route("/api/record", methods=["POST"])
def record():
    """on: this person records frames on this camera (one person per camera); off: stop."""
    j = request.get_json(force=True)
    name, cam, on = j.get("name"), j.get("camera"), j.get("on", True)
    if not on:
        with LOCK:
            if cam in CAMS and CAMS[cam].person == name:
                CAMS[cam].person, CAMS[cam].last_saved = None, None
            if name in S["people"] and S["people"][name].get("camera") == cam:
                S["people"][name]["camera"] = None
        return jsonify({"ok": True})
    if name not in S["people"] or cam not in CAMS:
        return jsonify({"error": "join first (name and board) / unknown camera"}), 400
    with LOCK:
        other = CAMS[cam].person
        if other and other != name and other in S["people"] and time.time() - S["people"][other]["seen"] < ACTIVE_S:
            return jsonify({"error": "camera %d is used by %s" % (cam, other)}), 409
        old = S["people"][name].get("camera")
        if old in CAMS and CAMS[old].person == name:
            CAMS[old].person = None
        CAMS[cam].person, CAMS[cam].last_saved = name, None
        S["people"][name]["camera"] = cam
        S["people"][name]["seen"] = time.time()
    return jsonify({"ok": True})


@app.route("/api/mode", methods=["POST"])
def mode():
    m = request.get_json(force=True).get("mode")
    if m not in ("intr", "extr"):
        return jsonify({"error": "mode intr or extr"}), 400
    with LOCK:
        S["mode"] = m
    return jsonify({"ok": True})


@app.route("/api/calibrate", methods=["POST"])
def calibrate():
    j = request.get_json(force=True)
    cam, model = j.get("camera"), j.get("model") or ARGS.model
    if cam not in CAMS or model not in MODELS:
        return jsonify({"error": "unknown camera or lens model"}), 400
    with LOCK:
        if CAMS[cam].calibrating:
            return jsonify({"error": "already calibrating"}), 409
        CAMS[cam].calibrating = True
    threading.Thread(target=calibrate_thread, args=(cam, model), daemon=True).start()
    return jsonify({"ok": True})


@app.route("/api/recalibrate", methods=["POST"])
def recalibrate():
    """Start a new calibration round for a camera: new frames go to cam_N_<round>, the next Calibrate
    writes cam_N_<round>.json. Earlier frames and results stay on disk untouched; the rectified view
    keeps using the previous calibration until the new one is done."""
    j = request.get_json(force=True)
    cam = j.get("camera")
    if cam not in CAMS:
        return jsonify({"error": "unknown camera"}), 400
    cs = CAMS[cam]
    if cs.calibrating:
        return jsonify({"error": "camera %d is calibrating" % cam}), 409
    with cs.lock:
        cs.session = max(cs.session, latest_session(ARGS.out, ARGS.results, cam)) + 1
        cs.count, cs.feats, cs.last_saved, cs.rms = 0, [], None, ""
    RES.pop(cam, None)
    log("cam %d: recalibration round %d started by %s (frames go to %s, earlier rounds kept)" % (
        cam, cs.session, j.get("name") or "?", cam_name(cam, cs.session)))
    return jsonify({"ok": True, "session": cs.session})


@app.route("/api/net/reset", methods=["POST"])
def net_reset():
    """Collect a camera's extrinsic again as a new round (cam_N_2.json, ...); earlier files are kept."""
    j = request.get_json(force=True)
    cam = j.get("camera")
    if ARGS.ref is None or cam not in NET or cam == ARGS.ref:
        return jsonify({"error": "no such camera in reference extrinsics"}), 400
    ns = NET[cam]
    if ns.pair.locked is not None or ns.pair.pool:
        ns.session += 1
    ns.pair, ns.saved = NetCam.new_pair(), None
    log("cam %d: extrinsics collected again (round %d) by %s" % (cam, ns.session, j.get("name") or "?"))
    return jsonify({"ok": True})


@app.route("/api/solve_extrinsics", methods=["POST"])
def solve_extrinsics():
    with LOCK:
        if S["busy"]:
            return jsonify({"error": S["busy"]}), 409
        S["busy"] = "starting"
    threading.Thread(target=extrinsics_thread, daemon=True).start()
    return jsonify({"ok": True})


@app.route("/api/results")
def results():
    d = ARGS.results
    rep = os.path.join(d, "extrinsics", "report.md")
    ref_rows = []
    net = os.path.join(d, "extrinsics_ref", "network.json")
    if os.path.exists(net):
        for c, r in sorted(read_json(net).items(), key=lambda kv: int(kv[0])):
            if r and not r.get("reference"):
                ref_rows.append({"camera": c, "stage": r.get("stage"), "frame": r.get("parent_frame"),
                                 "x": round(r["translation"][0], 3), "y": round(r["translation"][1], 3),
                                 "z": round(r["translation"][2], 3), "roll": round(r["rpy_deg"][0], 2),
                                 "pitch": round(r["rpy_deg"][1], 2), "yaw": round(r["rpy_deg"][2], 2),
                                 "samples": r.get("samples"), "placements": r.get("board_placements"),
                                 "spread_mm": None if r.get("spread_mm") is None else round(r["spread_mm"], 1),
                                 "round": r.get("session", 1)})
    return jsonify({"intrinsics": read_csv(os.path.join(d, "intrinsics", "summary.csv")),
                    "extrinsics": read_csv(os.path.join(d, "extrinsics", "cameras.csv")),
                    "extrinsics_ref": ref_rows,
                    "report": open(rep).read() if os.path.exists(rep) else ""})


def main():
    global ARGS, TAGS, BOARDS, REFPOSE
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--cams", nargs="*", type=int, default=None)
    ap.add_argument("--out", default="data")
    ap.add_argument("--results", default="results")
    ap.add_argument("--hz", type=float, default=4.0, help="processing rate per camera")
    ap.add_argument("--workers", type=int, default=max(2, (os.cpu_count() or 4) - 1))
    ap.add_argument("--fake", action="store_true", help="synthetic cameras, to try the dashboard")
    ap.add_argument("--model", default=DEFAULT_MODEL, choices=sorted(MODELS),
                    help="lens model preselected on the page (each Calibrate can pick another): "
                         + "; ".join("%s = %s" % kv for kv in MODELS.items()))
    ap.add_argument("--ref", type=int, default=None, metavar="N",
                    help="extrinsics mode: camera N (the ZED, e.g. streamed with ros_image_to_rtsp.py) is the reference; "
                         "every other calibrated camera is placed relative to it")
    ap.add_argument("--ref-pose", default="", metavar="URL",
                    help="with --ref: http://<robot>:8765/pose from robot/zed_pose_server.py; cameras are then placed "
                         "in the ZED map and the robot may move between board placements")
    ap.add_argument("--ref-board", default="", metavar="NAME",
                    help="with --ref: the board (name in config/boards.yaml) held up for extrinsics, e.g. E "
                         "(default: the first board)")
    ap.add_argument("--still-s", type=float, default=0.6, metavar="S",
                    help="with --ref: the board (and the robot) must be still this long in every view before a sample "
                         "is taken; streams arrive with different delays, so raise it (e.g. 1.0) on a laggy network")
    ap.add_argument("--url", action="append", default=[], metavar="N=URL",
                    help="camera N reads this stream (adds to / overrides config/cameras.yaml); repeatable")
    ap.add_argument("--intrinsics", action="append", default=[], metavar="N=FILE",
                    help="camera N is already calibrated: K and distortion from FILE (this repo's cam_N.json, "
                         "the wiki's results/cam_N.json, or a ROS camera_info YAML); repeatable")
    ARGS = ap.parse_args()
    BOARDS, TAGS = load_boards(), FloorTags()
    extra = {int(n): u for n, _, u in (a.partition("=") for a in ARGS.url)}
    urls = {**camera_urls([c for c in (ARGS.cams or []) if c not in extra] if ARGS.cams else None), **extra}
    if ARGS.cams:
        urls = {c: urls[c] for c in ARGS.cams if c in urls}
    STREAMS.update({c: FakeStream(c, BOARDS) for c in urls} if ARGS.fake else open_streams(urls))
    given = {int(n): f for n, _, f in (a.partition("=") for a in ARGS.intrinsics)}
    for c in sorted(urls):
        CAMS[c] = Cam(c, "" if ARGS.fake else urls[c])
        CAMS[c].session = latest_session(ARGS.out, ARGS.results, c)
        CAMS[c].load_existing(ARGS.out)
        done = [os.path.join(ARGS.results, "intrinsics", cam_name(c, k) + ".json") for k in range(CAMS[c].session, 0, -1)]
        mine = next((p for p in done if os.path.exists(p)), None)     # newest round that has a result
        for path, src in ((mine, "results/intrinsics"), (given.get(c), "given")):
            if path and os.path.exists(path):
                K, dist, size, model = read_intrinsics_file(path)
                CAMS[c].set_intrinsics(K, dist, "%s (%s)" % (src, os.path.basename(path)), size, model)
            elif path and src == "given":
                sys.exit("--intrinsics %d=%s: file not found" % (c, path))
    if ARGS.ref is not None:
        if ARGS.ref not in CAMS:
            sys.exit("--ref %d: no such camera (add it with --url %d=rtsp://...)" % (ARGS.ref, ARGS.ref))
        if ARGS.ref_board and ARGS.ref_board not in BOARDS:
            sys.exit("--ref-board %s: not in config/boards.yaml (%s)" % (ARGS.ref_board, ", ".join(sorted(BOARDS))))
        board = BOARDS[ARGS.ref_board or sorted(BOARDS)[0]].board
        NET.update({c: NetCam(c, board) for c in CAMS})
        REFPOSE = RefPose(ARGS.ref_pose) if ARGS.ref_pose else None
        net_load()
    threading.Thread(target=loop, daemon=True).start()
    print("dashboard: http://%s:%d  (saving to %s, results in %s)" % (ARGS.host, ARGS.port, ARGS.out, ARGS.results))
    app.run(host=ARGS.host, port=ARGS.port, threaded=True, debug=False)


if __name__ == "__main__":
    main()
