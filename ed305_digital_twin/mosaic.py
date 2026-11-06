#!/usr/bin/env python3
"""All cameras in one video, frame-aligned on the common clock.

    python mosaic.py ~/Downloads/yolo_20261105_10 --out all.mp4
    python mosaic.py rec/ --model runs/detect/ed305_v1/weights/best.pt --out all_det.mp4
    python mosaic.py rec/ --model runs/detect/ed305_v1/weights/best.pt --track --start 60 --seconds 120 --out all_track.mp4

Uses cam_N_times.csv next to the videos when there (motion_sync.py or
flash_sync.py), so every tile shows the same moment; without them each video's
own time is used. Also the quickest check of a sync: someone walking from one
camera's view into the next should move continuously across tiles.

--model draws the detector's boxes; --track adds per-camera tracking IDs
(BoT-SORT, configs/botsort_ed305.yaml; each camera has its own IDs. One ID per
person for the whole room is room_track.py --video all).
"""
import argparse
import math
from pathlib import Path

import cv2
import numpy as np

from multiview_label import SyncedVideo, find_videos

HERE = Path(__file__).resolve().parent


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("rec", type=Path, help="folder with one cam_N video per camera")
    ap.add_argument("--out", type=Path, required=True, help="output .mp4")
    ap.add_argument("--fps", type=float, default=10.0, help="output frames per second (default 10; 30 = every frame)")
    ap.add_argument("--start", type=float, help="first time on the common clock (s)")
    ap.add_argument("--seconds", type=float, help="how long (default: to the end of the shortest recording)")
    ap.add_argument("--tile", default="480x270", help="size of each camera tile (default 480x270)")
    ap.add_argument("--cols", type=int, help="tiles per row (default: about square)")
    ap.add_argument("--model", help="draw this detector's person boxes")
    ap.add_argument("--track", action="store_true", help="with --model: per-camera tracking IDs")
    ap.add_argument("--tracker", default=str(HERE / "configs/botsort_ed305.yaml"))
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--device", default=None)
    return ap.parse_args()


PALETTE = [(230, 159, 0), (86, 180, 233), (0, 158, 115), (240, 228, 66), (0, 114, 178), (213, 94, 0),
           (204, 121, 167), (120, 120, 255), (255, 120, 120), (60, 200, 60), (170, 110, 40), (128, 0, 128),
           (0, 200, 200), (255, 0, 255), (90, 90, 200), (200, 200, 90)]          # RGB, clearly different


def color(key):
    """The same colour for the same id everywhere; ids like p7 / 7 take palette entry 7."""
    import re
    import zlib
    k = key[-1] if isinstance(key, tuple) else key
    m = re.fullmatch(r"p?(\d+)", str(k))
    i = int(m.group(1)) if m and not isinstance(key, tuple) else zlib.crc32(str(key).encode())
    r, g, b = PALETTE[i % len(PALETTE)]
    return (b, g, r)


def draw_boxes(img, boxes, labels=None, colors=None, scale=1.0):
    """boxes in full-frame pixels; img already scaled by `scale`."""
    lw = max(2, int(round(3 * scale * 2)))
    for k, b in enumerate(boxes):
        x0, y0, x1, y1 = (int(v * scale) for v in b)
        c = colors[k] if colors else (60, 200, 60)
        cv2.rectangle(img, (x0, y0), (x1, y1), c, lw)
        if labels and labels[k]:
            cv2.putText(img, str(labels[k]), (x0 + 2, max(14, y0 - 4)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, c, 2, cv2.LINE_AA)
    return img


def tile(frame, size, title):
    if frame is None:
        img = np.full((size[1], size[0], 3), 30, np.uint8)
        cv2.putText(img, "no frame", (10, size[1] // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (160, 160, 160), 1)
    else:
        img = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
    cv2.rectangle(img, (0, 0), (size[0], 22), (0, 0, 0), -1)
    cv2.putText(img, title, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def grid(tiles, cols, size, extra=None):
    """Tiles in rows; `extra` (an image as tall as the grid) is put on the right."""
    while len(tiles) % cols:
        tiles.append(np.full((size[1], size[0], 3), 0, np.uint8))
    rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
    img = np.vstack(rows)
    if extra is not None:
        h = img.shape[0]
        e = cv2.resize(extra, (int(extra.shape[1] * h / extra.shape[0]), h), interpolation=cv2.INTER_AREA)
        img = np.hstack([img, e])
    return img


def even(img):
    """Video encoders want even sizes."""
    h, w = img.shape[:2]
    return img[: h - h % 2, : w - w % 2]


def main():
    a = parse()
    size = tuple(int(v) for v in a.tile.lower().split("x"))
    vids = find_videos(a.rec)
    names = sorted(vids, key=lambda n: (len(n), n))
    videos = {n: SyncedVideo(vids[n]) for n in names}
    synced = [n for n in names if videos[n].synced]
    print(f"{len(names)} cameras; {len(synced)} with a times file"
          + ("" if len(synced) == len(names) else " (the others: their own time; run motion_sync.py)"))
    t0 = max(v.t[0] for v in videos.values()) if a.start is None else a.start
    t1 = min(v.t[-1] for v in videos.values())
    if a.seconds:
        t1 = min(t1, t0 + a.seconds)
    steps = np.arange(t0, t1, 1.0 / a.fps)
    cols = a.cols or math.ceil(math.sqrt(len(names)))
    models = {}
    if a.model:
        from ultralytics import YOLO
        if a.track:
            models = {n: YOLO(a.model) for n in names}       # one tracker state per camera
        else:
            models = {"all": YOLO(a.model)}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    vw = None
    for k, T in enumerate(steps):
        frames = {n: videos[n].at(T) for n in names}
        boxes = {n: ([], None) for n in names}
        if a.model and not a.track:
            have = [n for n in names if frames[n] is not None]
            if have:
                m = models["all"]
                res = m.predict([frames[n] for n in have], conf=a.conf, imgsz=a.imgsz, device=a.device,
                                classes=[i for i, v in m.names.items() if v == "person"], verbose=False)
                for n, r in zip(have, res):
                    boxes[n] = (r.boxes.xyxy.cpu().numpy(), None)
        elif a.track:
            for n in names:
                if frames[n] is None:
                    continue
                m = models[n]
                r = m.track(frames[n], persist=True, tracker=a.tracker, conf=a.conf, imgsz=a.imgsz, device=a.device,
                            classes=[i for i, v in m.names.items() if v == "person"], verbose=False)[0]
                ids = r.boxes.id.int().tolist() if r.boxes.id is not None else [None] * len(r.boxes)
                boxes[n] = (r.boxes.xyxy.cpu().numpy(), ids)
        tiles = []
        for n in names:
            f = frames[n]
            img = tile(f, size, f"{n}   t {T:7.2f} s")
            if f is not None and len(boxes[n][0]):
                s = size[0] / f.shape[1]
                ids = boxes[n][1]
                draw_boxes(img, boxes[n][0], labels=ids, colors=[color((n, i)) for i in ids] if ids else None, scale=s)
            tiles.append(img)
        out = even(grid(tiles, cols, size))
        if vw is None:
            vw = cv2.VideoWriter(str(a.out), cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (out.shape[1], out.shape[0]))
        vw.write(out)
        print(f"  {k + 1}/{len(steps)}  t {T:.1f} s", end="\r")
    if vw is not None:
        vw.release()
    print(f"\n{a.out}  ({len(steps)} frames, {len(steps) / a.fps:.0f} s)")


if __name__ == "__main__":
    main()
