#!/usr/bin/env python3
"""Capture calibration frames from the ED305 RTSP streams.

Three people can run this at the same time on three laptops: each person has their own board
(A, B or C) and the tool only reacts to that board, so the others' boards are ignored.

  Intrinsics round (one person per camera group, see config/assignments.yaml):
    python capture.py --person Jorich --board A --mode intr --cams 1 2 3 4 13 14
  Extrinsics round (every laptop watches all 16 cameras, each person walks their own board):
    python capture.py --person Jorich --board A --mode extr

intr: a frame is saved for a camera when the board is well seen and has moved since the last
      saved frame; the window shows how many frames each camera has and which image regions
      are still empty. Target: 40 per camera, board in every region, near and far, tilted.
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

from calib.common import FloorTags, camera_urls, load_boards, now_tag, write_json
from calib.streams import open_streams, tile

GRID = 3  # coverage grid per image (3x3)


def detections(board, tags, frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids = board.detect(gray)
    return corners, ids, tags.detect(gray)


def save_frame(out_dir, name, frame, cam, board, corners, ids, tag_det, t, event=None):
    os.makedirs(out_dir, exist_ok=True)
    cv2.imwrite(os.path.join(out_dir, name + ".jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    write_json(os.path.join(out_dir, name + ".json"), {
        "camera": cam, "board": board, "time": t, "event": event,
        "charuco": None if corners is None else {"ids": ids, "corners": corners},
        "tags": {str(k): v for k, v in tag_det.items()},
    })


def draw(frame, corners, ids, tag_det, text, color=(0, 255, 0)):
    if corners is not None:
        cv2.aruco.drawDetectedCornersCharuco(frame, corners.reshape(-1, 1, 2), ids.reshape(-1, 1), color)
    for tid, c in tag_det.items():
        cv2.polylines(frame, [c.astype(int)], True, (255, 0, 255), 3)
        cv2.putText(frame, "tag %d" % tid, tuple(c[0].astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 255), 3)
    cv2.putText(frame, text, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.6, color, 4)


def coverage_text(cov):
    return "".join("".join("#" if cov[r][c] else "." for c in range(GRID)) + "|" for r in range(GRID))


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
    args = ap.parse_args()

    urls = camera_urls(args.cams)
    board = load_boards([args.board])[args.board]
    tags = FloorTags()
    streams = open_streams(urls)
    cams = sorted(urls)
    out_root = os.path.join(args.out, args.person, args.mode)
    print("saving to", out_root)

    counts = {c: 0 for c in cams}
    cov = {c: [[0] * GRID for _ in range(GRID)] for c in cams}
    last_saved = {c: None for c in cams}          # (centre, size) of the last saved board view
    still = {c: [] for c in cams}                 # extr: recent corner positions per camera
    armed = True                                  # extr: wait for movement before next event
    events = 0
    period = 1.0 / args.hz
    win = "capture %s board %s (%s)  q=quit" % (args.person, args.board, args.mode)

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
            label = "cam %d  %d/%d  %s" % (cam, counts[cam], args.target, coverage_text(cov[cam]))

            if args.mode == "intr":
                if seen:
                    h, w = frame.shape[:2]
                    centre, size = corners.mean(0), np.ptp(corners, axis=0).max()
                    prev = last_saved[cam]
                    moved = prev is None or np.linalg.norm(centre - prev[0]) > 0.06 * w or abs(size - prev[1]) > 0.12 * prev[1]
                    if moved and counts[cam] < args.target:
                        name = now_tag()
                        save_frame(os.path.join(out_root, "cam_%d" % cam), name, frame, cam, args.board,
                                   corners, ids, tag_det, t)
                        counts[cam] += 1
                        last_saved[cam] = (centre, size)
                        gr, gc = min(GRID - 1, int(centre[1] / h * GRID)), min(GRID - 1, int(centre[0] / w * GRID))
                        cov[cam][gr][gc] += 1
                        label += "  SAVED"
                draw(frame, corners, ids, tag_det, label, (0, 255, 0) if seen else (0, 200, 255))
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
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        time.sleep(max(0.0, period - (time.time() - t0)))

    for s in streams.values():
        s.stop()
    print("done.", {c: counts[c] for c in cams} if args.mode == "intr" else "%d events" % events)


if __name__ == "__main__":
    main()
