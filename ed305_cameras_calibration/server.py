#!/usr/bin/env python3
"""Calibration dashboard: ONE PC reads all 16 streams, everyone works from a browser.

  python server.py                      # then open http://<this pc>:8080 on any phone or laptop
  python server.py --fake               # no cameras: synthetic moving boards, to try the dashboard

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
from calib.common import FloorTags, camera_urls, load_boards, load_intrinsics, now_tag, read_csv, read_json, write_csv  # noqa: E402
from calib.intr import FIELDS, coverage, nominal_K, ready, split_for, view_features  # noqa: E402
from calib.streams import open_streams  # noqa: E402
from calibrate_intrinsics import solve_camera  # noqa: E402
from capture import draw, draw_bars, save_frame  # noqa: E402
from calib.intr import summary_row, view_rows  # noqa: E402

MIN_CORNERS = 12
ACTIVE_S = 60          # a person whose browser has not polled for this long is inactive
STILL_PX, MOVE_PX = 2.0, 25.0
STILL_S = 0.8          # the board must have been within STILL_PX for this long to count as still

app = Flask(__name__, static_folder=os.path.join(HERE, "web"), static_url_path="/web")
app.json.sort_keys = False
S = {"mode": "intr", "people": {}, "armed": {}, "events": {}, "log": [], "busy": ""}
LOCK = threading.Lock()
CAMS, STREAMS, BOARDS, TAGS, RES, ARGS = {}, {}, {}, None, {}, None


def log(msg):
    """Keep the last lines shown on the page (events, calibrations, solver output)."""
    print(msg)
    with LOCK:
        S["log"] = (S["log"] + msg.rstrip().split("\n"))[-25:]


class Cam:
    def __init__(self, cam):
        self.cam = cam
        self.lock = threading.Lock()
        self.person = None
        self.count, self.feats, self.last_saved = 0, [], None
        self.rms, self.status, self.last_board = "", "idle", ""
        self.jpeg, self.jpeg_t = None, 0.0
        self.still, self.still_frame = {}, {}     # board -> recent centres / (frame, corners, ids, tags, t, person)
        self.K, self.dist = nominal_K(), None

    def load_existing(self, data_dir):
        """Counts and bar features of frames already on disk (so a restart keeps the bars)."""
        for path in sorted(glob.glob(os.path.join(data_dir, "*", "intr", "cam_%d" % self.cam, "*.json"))):
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


def encode(frame):
    small = cv2.resize(frame, (960, 540))
    return cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 70])[1].tobytes()


def process(cam, mode, active):
    cs = CAMS[cam]
    frame, t = STREAMS[cam].latest()
    if frame is None:
        cs.status = "no stream"
        return
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    tag_det = {}                      # floor tags are slow to detect: only on the frames that get saved
    h, w = frame.shape[:2]
    if mode == "intr":
        person = cs.person
        if person not in active:
            cs.status = "idle"
            return
        b = active[person]["board"]
        corners, ids = BOARDS[b].detect(gray)
        seen = corners is not None and len(corners) >= MIN_CORNERS
        label = "cam %d  %s  board %s  %d saved" % (cam, person, b, cs.count)
        cs.status = "board seen" if seen else "no board"
        if seen:
            centre, size = corners.mean(0), np.ptp(corners, axis=0).max()
            prev = cs.last_saved
            moved = prev is None or np.linalg.norm(centre - prev[0]) > 0.06 * w or abs(size - prev[1]) > 0.12 * prev[1]
            if moved:
                tag_det = TAGS.detect(gray)
                with cs.lock:
                    name = now_tag()
                    split = split_for(cs.count)
                    f = view_features(BOARDS[b], corners, ids, cs.K, cs.dist, (w, h))
                    out_root = os.path.join(ARGS.out, person, "intr")
                    save_frame(os.path.join(out_root, "cam_%d" % cam), name, frame, cam, b, corners, ids, tag_det, t,
                               split=split, features=f)
                    write_csv(os.path.join(out_root, "views.csv"), [{
                        "person": person, "board": b, "camera": cam, "file": name, "time": t, "split": split,
                        "corners": len(corners), **{k: (round(f[k], 3) if f else "") for k in FIELDS}}], append=True)
                    cs.count += 1
                    if split == "calib" and f:
                        cs.feats.append(f)
                    cs.last_saved = (centre, size)
                label += "  SAVED " + split
        draw(frame, corners, ids, tag_det, label, (0, 255, 0) if seen else (0, 200, 255))
        draw_bars(frame, cs.feats, len(cs.feats), cs.rms)
    else:
        found = []
        for person, info in active.items():
            b = info["board"]
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
                draw(frame, corners, ids, tag_det, "", (0, 255, 0) if is_still else (0, 200, 255))
            else:
                cs.still[b] = []
                cs.still_frame.pop(b, None)
        cs.status = ", ".join(found) if found else "no board"
        cv2.putText(frame, "cam %d  %s" % (cam, cs.status), (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 255, 0), 4)
    cs.jpeg, cs.jpeg_t = encode(frame), time.time()


def extr_events(active):
    now = time.time()
    for person, info in active.items():
        b = info["board"]
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
        list(pool.map(lambda c: process(c, mode, active), jobs))
        if mode == "extr":
            extr_events(active)
        for c, cs in CAMS.items():      # idle cameras: a raw thumbnail once a second
            if c not in jobs and time.time() - cs.jpeg_t > 1.0:
                frame, _ = STREAMS[c].latest()
                if frame is not None:
                    cv2.putText(frame, "cam %d" % c, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (200, 200, 200), 4)
                    cs.jpeg, cs.jpeg_t = encode(frame), time.time()
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


def calibrate_thread(cam):
    cs = CAMS[cam]
    cs.status = "calibrating"
    try:
        res = solve_camera(cam, ARGS.out, ARGS.results, BOARDS)
        if res is None:
            cs.rms = "fewer than 10 calibrate views"
        else:
            with LOCK:
                RES[cam] = res
                write_summaries()
            cs.K, cs.dist = res["K"], res["dist"]
            cs.rms = "rms calib %.2f test %.2f px (%d+%d views) %s" % (
                res["rms_calib"], res["rms_test"], res["n_calib"], res["n_test"], summary_row(cam, res)["status"])
    except Exception as e:  # keep the server alive
        cs.rms = "error: %s" % e
    cs.status = "calibrated"
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
    cs = CAMS.get(cam)
    if cs is None or cs.jpeg is None:
        return Response(status=204)
    return Response(cs.jpeg, mimetype="image/jpeg", headers={"Cache-Control": "no-store"})


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
        cams[c] = {"person": cs.person, "count": cs.count, "calib": len(cs.feats), "coverage": cov,
                   "ready": ready(cs.feats), "rms": cs.rms, "status": cs.status,
                   "calibrated": c in RES or os.path.exists(os.path.join(ARGS.results, "intrinsics", "cam_%d.json" % c))}
    return jsonify({"mode": mode, "people": people, "cams": cams, "events": events, "log": "\n".join(lines), "busy": busy,
                    "fields": FIELDS})


@app.route("/api/join", methods=["POST"])
def join():
    j = request.get_json(force=True)
    name, board = (j.get("name") or "").strip(), j.get("board")
    if not name or board not in BOARDS:
        return jsonify({"error": "name and board (A/B/C) needed"}), 400
    with LOCK:
        for n, p in S["people"].items():
            if n != name and p["board"] == board and time.time() - p["seen"] < ACTIVE_S:
                return jsonify({"error": "board %s is used by %s" % (board, n)}), 409
        p = S["people"].setdefault(name, {"camera": None})
        p.update({"board": board, "seen": time.time()})
    return jsonify({"ok": True})


@app.route("/api/select", methods=["POST"])
def select():
    j = request.get_json(force=True)
    name, cam = j.get("name"), j.get("camera")
    if name not in S["people"] or cam not in CAMS:
        return jsonify({"error": "join first / unknown camera"}), 400
    with LOCK:
        other = CAMS[cam].person
        if other and other != name and other in S["people"] and time.time() - S["people"][other]["seen"] < ACTIVE_S:
            return jsonify({"error": "camera %d is used by %s" % (cam, other)}), 409
        old = S["people"][name].get("camera")
        if old in CAMS and CAMS[old].person == name:
            CAMS[old].person = None
        CAMS[cam].person = name
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
    cam = request.get_json(force=True).get("camera")
    if cam not in CAMS:
        return jsonify({"error": "unknown camera"}), 400
    if CAMS[cam].status == "calibrating":
        return jsonify({"error": "already calibrating"}), 409
    threading.Thread(target=calibrate_thread, args=(cam,), daemon=True).start()
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
    return jsonify({"intrinsics": read_csv(os.path.join(d, "intrinsics", "summary.csv")),
                    "extrinsics": read_csv(os.path.join(d, "extrinsics", "cameras.csv")),
                    "report": open(rep).read() if os.path.exists(rep) else ""})


def main():
    global ARGS, TAGS, BOARDS
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--cams", nargs="*", type=int, default=None)
    ap.add_argument("--out", default="data")
    ap.add_argument("--results", default="results")
    ap.add_argument("--hz", type=float, default=4.0, help="processing rate per camera")
    ap.add_argument("--workers", type=int, default=max(2, (os.cpu_count() or 4) - 1))
    ap.add_argument("--fake", action="store_true", help="synthetic cameras, to try the dashboard")
    ARGS = ap.parse_args()
    BOARDS, TAGS = load_boards(), FloorTags()
    urls = camera_urls(ARGS.cams)
    STREAMS.update({c: FakeStream(c, BOARDS) for c in urls} if ARGS.fake else open_streams(urls))
    intr = load_intrinsics(ARGS.results)
    for c in sorted(urls):
        CAMS[c] = Cam(c)
        CAMS[c].load_existing(ARGS.out)
        if c in intr:
            CAMS[c].K, CAMS[c].dist = intr[c]
    threading.Thread(target=loop, daemon=True).start()
    print("dashboard: http://%s:%d  (saving to %s, results in %s)" % (ARGS.host, ARGS.port, ARGS.out, ARGS.results))
    app.run(host=ARGS.host, port=ARGS.port, threaded=True, debug=False)


if __name__ == "__main__":
    main()
