#!/usr/bin/env python3
"""Capture calibration frames from the ED305 RTSP streams.

Three people can run this at the same time on three laptops: each person has their own board
(A, B or C) and the tool only reacts to that board, so the others' boards are ignored.

  Intrinsics round (one person per camera group, see config/assignments.yaml):
    python capture.py --person Jorich --board A --mode intr --cams 1 2 3 4 13 14
  Extrinsics round (every laptop watches all 16 cameras, each person walks their own board):
    python capture.py --person Jorich --board A --mode extr

intr: a frame is saved for a camera when the board is well seen and has moved since the last
      saved frame. Six bars per camera (like the X Y Size Skew bars of the ROS tool) show how
      much of the wanted range of board poses is covered: x, y (where in the image), z (near
      to far), roll, pitch, yaw (how tilted). When all bars are green and 20+ frames are saved
      the camera says READY. Press c to calibrate right away (fit on the calibrate views, error
      on the held-out test views = every 5th saved frame); keep moving the board to add more
      and press c again. Every saved frame is logged in data/<person>/intr/views.csv.
extr: a frame SET is saved when the board is held STILL for ~1 s and at least two cameras see
      it (hold, wait for the beep text, move on). Floor tags seen in any frame are stored too.

Everything goes to --out/<person>/..., one folder per person, so the folders can simply be
copied together afterwards (see README).
"""
import argparse
import os
import time

import cv2
import numpy as np

from calib.common import FloorTags, camera_urls, load_boards, load_intrinsics, now_tag, write_csv, write_json
from calib.intr import FIELDS, MIN_VIEWS, coverage, nominal_K, ready, split_for, view_features
from calib.streams import open_streams, tile


def detections(board, tags, frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids = board.detect(gray)
    return corners, ids, tags.detect(gray)


def save_frame(out_dir, name, frame, cam, board, corners, ids, tag_det, t, event=None, split=None, features=None):
    os.makedirs(out_dir, exist_ok=True)
    cv2.imwrite(os.path.join(out_dir, name + ".jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    write_json(os.path.join(out_dir, name + ".json"), {
        "camera": cam, "board": board, "time": t, "event": event, "split": split, "features": features,
        "charuco": None if corners is None else {"ids": ids, "corners": corners},
        "tags": {str(k): v for k, v in tag_det.items()},
    })


def draw(frame, corners, ids, tag_det, text, color=(0, 255, 0)):
    if corners is not None:
        cv2.aruco.drawDetectedCornersCharuco(frame, corners.reshape(-1, 1, 2), None, color)
    for tid, c in tag_det.items():
        cv2.polylines(frame, [c.astype(int)], True, (255, 0, 255), 3)
        cv2.putText(frame, "tag %d" % tid, tuple(c[0].astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 255), 3)
    cv2.putText(frame, text, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.6, color, 4)


def draw_bars(frame, feats, count, rms_text):
    """Six coverage bars along the bottom of the frame, green when full; READY when enough."""
    h, w = frame.shape[:2]
    cov = coverage(feats)
    bw, bh, gap, y0 = 250, 44, 18, h - 70
    for k, name in enumerate(FIELDS):
        x0 = 20 + k * (bw + gap)
        full = cov[name] >= 1.0
        col = (0, 200, 0) if full else (0, 200, 255)
        cv2.rectangle(frame, (x0, y0), (x0 + bw, y0 + bh), (255, 255, 255), 2)
        cv2.rectangle(frame, (x0, y0), (x0 + int(bw * cov[name]), y0 + bh), col, -1)
        cv2.putText(frame, "%s %d%%" % (name, 100 * cov[name]), (x0 + 8, y0 + 32), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (0, 0, 0) if cov[name] > 0.35 else (255, 255, 255), 2)
    x0 = 20 + 6 * (bw + gap)
    if ready(feats):
        cv2.putText(frame, "READY: press c", (x0, y0 + 34), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    else:
        cv2.putText(frame, "%d/%d" % (count, MIN_VIEWS), (x0, y0 + 34), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
    if rms_text:
        cv2.putText(frame, rms_text, (20, y0 - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)


def calibrate_now(cams, data_root, results_dir, boards_all):
    """Fit every listed camera from what is saved so far (press c). Returns {cam: text}."""
    from calibrate_intrinsics import solve_camera
    out = {}
    for cam in cams:
        res = solve_camera(cam, data_root, results_dir, boards_all)
        if res is None:
            out[cam] = "cam %d: fewer than 10 calibrate views" % cam
            continue
        out[cam] = "rms calib %.2f test %.2f px (%d+%d views) %s" % (
            res["rms_calib"], res["rms_test"], res["n_calib"], res["n_test"],
            "OK" if res["rms_test"] < 1.0 and res["rms_calib"] < 0.6 else "CHECK")
        print("cam %2d: %s  fx %.1f fy %.1f cx %.1f cy %.1f" % (
            cam, out[cam], res["K"][0, 0], res["K"][1, 1], res["K"][0, 2], res["K"][1, 2]))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--person", required=True)
    ap.add_argument("--board", required=True, choices=["A", "B", "C"])
    ap.add_argument("--mode", required=True, choices=["intr", "extr"])
    ap.add_argument("--cams", nargs="*", type=int, default=None, help="camera numbers; default all 16")
    ap.add_argument("--out", default="data")
    ap.add_argument("--min-corners", type=int, default=12)
    ap.add_argument("--target", type=int, default=40, help="intr: frames wanted per camera")
    ap.add_argument("--hz", type=float, default=5.0, help="processing rate")
    ap.add_argument("--results", default="results", help="intr: where c writes intrinsics/cam_N.json")
    args = ap.parse_args()

    urls = camera_urls(args.cams)
    boards_all = load_boards()
    board = boards_all[args.board]
    tags = FloorTags()
    streams = open_streams(urls)
    cams = sorted(urls)
    out_root = os.path.join(args.out, args.person, args.mode)
    print("saving to", out_root)

    counts = {c: 0 for c in cams}
    feats = {c: [] for c in cams}                 # intr: board pose features of the saved views
    intr = load_intrinsics(args.results)          # intr: real K if known, else a nominal one, for the bars
    Kd = {c: intr.get(c, (nominal_K(), None)) for c in cams}
    rms_text = {c: "" for c in cams}
    last_saved = {c: None for c in cams}          # (centre, size) of the last saved board view
    still = {c: [] for c in cams}                 # extr: recent corner positions per camera
    armed = True                                  # extr: wait for movement before next event
    events = 0
    period = 1.0 / args.hz
    win = "capture %s board %s (%s)  %sq=quit" % (args.person, args.board, args.mode,
                                                  "c=calibrate  " if args.mode == "intr" else "")

    while True:
        t0 = time.time()
        tiles, moving_any, static_cams = [], False, {}
        for cam in cams:
            frame, t = streams[cam].latest()
            if frame is None:
                tiles.append(None)
                continue
            corners, ids, tag_det = detections(board, tags, frame)
            seen = corners is not None and len(corners) >= args.min_corners
            label = "cam %d  %d saved" % (cam, counts[cam])

            if args.mode == "intr":
                if seen:
                    h, w = frame.shape[:2]
                    centre, size = corners.mean(0), np.ptp(corners, axis=0).max()
                    prev = last_saved[cam]
                    moved = prev is None or np.linalg.norm(centre - prev[0]) > 0.06 * w or abs(size - prev[1]) > 0.12 * prev[1]
                    if moved and counts[cam] < args.target:
                        name = now_tag()
                        split = split_for(counts[cam])
                        f = view_features(board, corners, ids, Kd[cam][0], Kd[cam][1], (w, h))
                        save_frame(os.path.join(out_root, "cam_%d" % cam), name, frame, cam, args.board,
                                   corners, ids, tag_det, t, split=split, features=f)
                        write_csv(os.path.join(out_root, "views.csv"), [{
                            "person": args.person, "board": args.board, "camera": cam, "file": name, "time": t,
                            "split": split, "corners": len(corners),
                            **{k: (round(f[k], 3) if f else "") for k in FIELDS}}], append=True)
                        counts[cam] += 1
                        if split == "calib" and f:
                            feats[cam].append(f)
                        last_saved[cam] = (centre, size)
                        label += "  SAVED " + split
                draw(frame, corners, ids, tag_det, label, (0, 255, 0) if seen else (0, 200, 255))
                draw_bars(frame, feats[cam], len(feats[cam]), rms_text[cam])
            else:  # extr: collect per-camera stillness
                if seen:
                    hist = still[cam]
                    hist.append(corners.mean(0))
                    del hist[:-4]
                    is_still = len(hist) >= 3 and max(np.linalg.norm(hist[-1] - p) for p in hist[:-1]) < 2.0
                    if len(hist) >= 2 and np.linalg.norm(hist[-1] - hist[-2]) > 25:
                        moving_any = True
                    if is_still:
                        static_cams[cam] = (frame, corners, ids, tag_det, t)
                    label = "cam %d  %s" % (cam, "STILL" if is_still else "board")
                else:
                    still[cam] = []
                    label = "cam %d" % cam
                draw(frame, corners, ids, tag_det, label, (0, 255, 0) if seen else (0, 200, 255))
            tiles.append(frame)

        if args.mode == "extr":
            if moving_any:
                armed = True
            if armed and len(static_cams) >= 2:
                ev = now_tag()
                for cam, (frame, corners, ids, tag_det, t) in static_cams.items():
                    save_frame(os.path.join(out_root, ev), "cam_%d" % cam, frame, cam, args.board,
                               corners, ids, tag_det, t, event=ev)
                events += 1
                armed = False
                print("event %d saved: cams %s" % (events, sorted(static_cams)))

        canvas = tile(tiles, cols=4 if len(cams) > 6 else 3)
        status = "%s | %s | events %d | move the board to re-arm" % (args.mode, args.person, events) if args.mode == "extr" \
            else "%s | %s | %d frames" % (args.mode, args.person, sum(counts.values()))
        cv2.putText(canvas, status, (10, canvas.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.imshow(win, canvas)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("c") and args.mode == "intr":
            cv2.putText(canvas, "calibrating... see terminal", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
            cv2.imshow(win, canvas)
            cv2.waitKey(1)
            rms_text.update(calibrate_now([c for c in cams if counts[c] >= 10], args.out, args.results, boards_all))
            intr = load_intrinsics(args.results)
            Kd = {c: intr.get(c, (nominal_K(), None)) for c in cams}
        time.sleep(max(0.0, period - (time.time() - t0)))

    for s in streams.values():
        s.stop()
    if args.mode == "intr":
        print("done. saved per camera:", {c: counts[c] for c in cams})
        for c in cams:
            cov = coverage(feats[c])
            print("cam %2d coverage: %s  %s" % (c, " ".join("%s %3.0f%%" % (k, 100 * cov[k]) for k in FIELDS),
                                              "READY" if ready(feats[c]) else "more views needed"))
    else:
        print("done. %d events" % events)


if __name__ == "__main__":
    main()
