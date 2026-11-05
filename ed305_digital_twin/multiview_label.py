#!/usr/bin/env python3
"""Person labels for every camera from all 16 views at once, no hand labelling.

    python multiview_label.py rec/ --calib results/extrinsics/cameras.json --out data/overhead
    python multiview_label.py rec/ --calib results/extrinsics/cameras.json --out data/overhead --preview 20

rec/ holds one recording per camera, cam_N.mkv (or .mp4), recorded at the
same time, and the flash_sync.py output cam_N_times.csv (frame, pts_s, t_s:
every camera's frames on one clock, from the lights going off and on). Without
a times file a camera's frames are timed from its own start (only right if
every recording started at the same moment: a warning says so).

--calib: the calibration's results/extrinsics/cameras.json (room frame from
the floor tags, z up, floor z = 0), or a cameras YAML of this repo in a frame
with the floor at z = 0.

Every --every seconds on the common clock: the nearest frame of every camera,
a pretrained detector on each at a low threshold, then ed305/multiview.py:
people are the floor points that several cameras agree on; per camera the
boxes of those people are kept (also the ones the detector was unsure of),
boxes no other camera agrees with are dropped unless confident and not
refuted, and people a camera should have found but did not are "missed".
A frame with a miss is left unlabelled (--add-missed: the expected box is
written instead), so every label file written can be trained on as is.

Writes to --out (the layout make_dataset.py and train_yolo.py read):
  images/cam_N_t<ms>.jpg      a camera's frame, saved when it changed since its last one
  labels/cam_N_t<ms>.txt      person boxes (only for frames without a miss)
  frames.csv                  image, camera, video, t (common clock)
  multiview.csv               per frame: confirmed / single / dropped / missed boxes, status
  people.csv                  per time step: every person's floor position, shape, views
  preview/step_<k>.jpg        with --preview: all cameras + a floor view, boxes coloured by kind
"""
import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

from ed305.camera import load_any
from ed305.multiview import SHAPES, Grid, associate, label_cameras, support
from ed305.yolo_data import append_frames_csv, camera_of, change_fraction, thumb, yolo_line

COLORS = {"confirmed": (60, 200, 60), "single": (230, 180, 40), "added": (0, 200, 255),
          "dropped": (60, 60, 230), "missed": (0, 220, 255)}
SHAPE_NAMES = ("sitting", "mid", "standing")


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("rec", type=Path, help="folder with cam_N.mkv/.mp4 (+ cam_N_times.csv from flash_sync.py)")
    ap.add_argument("--calib", type=Path, required=True, help="cameras.json (calibration) or cameras YAML")
    ap.add_argument("--out", type=Path, required=True, help="dataset folder")
    ap.add_argument("--every", type=float, default=1.0, help="seconds between time steps (default 1)")
    ap.add_argument("--start", type=float, help="first time on the common clock (s)")
    ap.add_argument("--end", type=float, help="last time on the common clock (s)")
    ap.add_argument("--bounds", help="floor area x0,x1,y0,y1 in metres (default: around the cameras)")
    ap.add_argument("--cell", type=float, default=0.1, help="floor grid step in metres (default 0.1)")
    ap.add_argument("--model", default="yolo26x.pt", help="detector (default yolo26x.pt, COCO)")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--device", default=None)
    ap.add_argument("--det-conf", type=float, default=0.05, help="detector threshold (default 0.05: low on purpose)")
    ap.add_argument("--min-views", type=int, default=2, help="cameras that must agree on a person (default 2)")
    ap.add_argument("--min-score", type=float, default=0.8, help="sum of their scores (default 0.8)")
    ap.add_argument("--iou", type=float, default=0.4, help="box vs expected box to agree (default 0.4)")
    ap.add_argument("--single-conf", type=float, default=0.6,
                    help="keep a box only one camera has if its score is this high (default 0.6)")
    ap.add_argument("--refute-views", type=int, default=3,
                    help="...unless this many other cameras see that spot clearly (default 3; 0 = never)")
    ap.add_argument("--add-missed", action="store_true", help="write expected boxes for misses instead of leaving the frame out")
    ap.add_argument("--min-change", type=float, default=0.005, help="save a camera's frame only if this much changed")
    ap.add_argument("--preview", type=int, default=0, help="write preview images for the first N time steps")
    return ap.parse_args()


class SyncedVideo:
    """A recording read forward in time, frames picked by time on the common clock."""

    def __init__(self, path):
        self.path, self.cap = path, cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            raise SystemExit(f"cannot open {path}")
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.fps = fps if 1 <= fps <= 240 else 30.0
        n = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        times = path.with_name(path.stem + "_times.csv")
        if times.exists():
            with open(times, newline="") as f:
                self.t = np.array([float(r["t_s"]) for r in csv.DictReader(f)])
            self.synced = True
        else:
            self.t = np.arange(max(n, 1)) / self.fps
            self.synced = False
        self.size = (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self.i = -1                       # index of the frame last grabbed

    def at(self, T):
        """The frame nearest time T (within half a frame period), or None."""
        j = int(np.clip(np.searchsorted(self.t, T), 1, len(self.t) - 1))
        j = j - 1 if abs(self.t[j - 1] - T) <= abs(self.t[j] - T) else j
        if abs(self.t[j] - T) > 0.6 / self.fps or j <= self.i:
            return None
        while self.i < j:
            if not self.cap.grab():
                return None
            self.i += 1
        ok, frame = self.cap.retrieve()
        return frame if ok else None


def find_videos(rec):
    vids = {}
    for p in sorted(rec.iterdir()):
        if p.suffix.lower() in (".mkv", ".mp4", ".avi", ".mov") and p.name.lower().startswith("cam"):
            vids[camera_of(p)] = p
    return vids


def bev(grid, people, bounds, tile):
    x0, x1, y0, y1 = bounds
    s = 0.95 * min(tile[0] / (x1 - x0), tile[1] / (y1 - y0))
    img = np.full((int((y1 - y0) * s) + 1, int((x1 - x0) * s) + 1, 3), 245, np.uint8)
    px = lambda x, y: (int((x - x0) * s), int((y1 - y) * s))
    for cam in grid.cams:
        cv2.circle(img, px(*cam.center[:2]), 5, (90, 90, 90), -1)
        cv2.putText(img, cam.name.replace("cam_", ""), px(*cam.center[:2]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (40, 40, 40), 1)
    for p in people:
        cv2.circle(img, px(*p.xy), max(3, int(0.25 * s)), (60, 200, 60), -1)
        cv2.putText(img, str(p.views), px(*p.xy), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    return img


def preview(path, grid, frames, labels, people, bounds, tile=(480, 270)):
    tiles = []
    for c, cam in enumerate(grid.cams):
        f = frames[c]
        if f is None:
            tiles.append(np.full((tile[1], tile[0], 3), 40, np.uint8))
            continue
        img = f.copy()
        L = labels[c]
        for b, kind, _ in L.keep:
            cv2.rectangle(img, tuple(map(int, b[:2])), tuple(map(int, b[2:])), COLORS[kind], 4)
        for b, _, _ in L.dropped:
            cv2.rectangle(img, tuple(map(int, b[:2])), tuple(map(int, b[2:])), COLORS["dropped"], 2)
        for b in L.missed:
            cv2.rectangle(img, tuple(map(int, b[:2])), tuple(map(int, b[2:])), COLORS["missed"], 4, cv2.LINE_4)
        img = cv2.resize(img, tile, interpolation=cv2.INTER_AREA)
        cv2.putText(img, f"{cam.name}  keep {len(L.keep)} drop {len(L.dropped)} miss {len(L.missed)}",
                    (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        tiles.append(img)
    b = bev(grid, people, bounds, tile)[:tile[1], :tile[0]]
    t = np.full((tile[1], tile[0], 3), 255, np.uint8)
    t[:b.shape[0], :b.shape[1]] = b
    cv2.putText(t, "floor: people (views)", (6, tile[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (60, 60, 60), 1)
    tiles.append(t)
    while len(tiles) % 4:
        tiles.append(np.full((tile[1], tile[0], 3), 255, np.uint8))
    rows = [np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]
    cv2.imwrite(str(path), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])


def main():
    a = parse()
    from ultralytics import YOLO

    vids = find_videos(a.rec)
    cams_all = load_any(a.calib)
    cams = [c for c in cams_all if c.name in vids]
    if len(cams) < 2:
        raise SystemExit(f"recordings {sorted(vids)} and calibration {[c.name for c in cams_all]}: "
                         "need 2+ cameras in both")
    left = sorted(set(vids) - {c.name for c in cams})
    if left:
        print(f"no calibration for {left}: not used")
    videos = [SyncedVideo(vids[c.name]) for c in cams]
    for cam, v in zip(cams, videos):
        if v.size != (cam.width, cam.height):
            raise SystemExit(f"{cam.name}: video {v.size}, calibration {cam.width}x{cam.height}")
    unsynced = [c.name for c, v in zip(cams, videos) if not v.synced]
    if unsynced:
        print(f"WARNING no *_times.csv for {unsynced}: assuming they started recording at the same moment")

    if a.bounds:
        bounds = tuple(float(v) for v in a.bounds.split(","))
    else:
        xy = np.array([c.center[:2] for c in cams])
        bounds = (xy[:, 0].min(), xy[:, 0].max(), xy[:, 1].min(), xy[:, 1].max())
    print(f"{len(cams)} cameras, floor {bounds[0]:.1f}..{bounds[1]:.1f} x {bounds[2]:.1f}..{bounds[3]:.1f} m; "
          "expected boxes per floor cell ...")
    grid = Grid(cams, bounds, a.cell)

    out = a.out
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(exist_ok=True)
    cls_file = out / "classes.txt"
    if cls_file.exists() and cls_file.read_text().splitlines() != ["person"]:
        raise SystemExit(f"{cls_file} has {cls_file.read_text().splitlines()}: these labels have person only, "
                         "and would teach the model the other classes are not there. Use a new --out.")
    cls_file.write_text("person\n")

    model = YOLO(a.model)
    person = [k for k, v in model.names.items() if v == "person"]
    if not person:
        raise SystemExit(f"{a.model} has no 'person' class")

    t0 = max(v.t[0] for v in videos) if a.start is None else a.start
    t1 = min(v.t[-1] for v in videos) if a.end is None else a.end
    steps = np.arange(t0, t1, a.every)
    print(f"{len(steps)} time steps, {t0:.1f} .. {t1:.1f} s")

    mv_path, ppl_path = out / "multiview.csv", out / "people.csv"
    mv_new, ppl_new = not mv_path.exists(), not ppl_path.exists()
    mv = open(mv_path, "a", newline="")
    ppl = open(ppl_path, "a", newline="")
    mvw = csv.DictWriter(mv, ["image", "camera", "t", "confirmed", "single", "added", "dropped", "missed", "status"])
    pw = csv.DictWriter(ppl, ["t", "x", "y", "shape", "views", "score"])
    if mv_new:
        mvw.writeheader()
    if ppl_new:
        pw.writeheader()
    if a.preview:
        (out / "preview").mkdir(exist_ok=True)

    last = [None] * len(cams)
    totals = dict(steps=0, people=0, saved=0, labelled=0, confirmed=0, single=0, dropped=0, missed=0)
    try:
        for k, T in enumerate(steps):
            frames = [v.at(T) for v in videos]
            present = [f is not None for f in frames]
            if sum(present) < 2:
                continue
            idx = [c for c in range(len(cams)) if present[c]]
            res = model.predict([frames[c] for c in idx], conf=a.det_conf, iou=0.6, imgsz=a.imgsz,
                                classes=person, device=a.device, verbose=False)
            boxes = [np.zeros((0, 4)) for _ in cams]
            confs = [np.zeros(0) for _ in cams]
            for c, r in zip(idx, res):
                boxes[c] = r.boxes.xyxy.cpu().numpy().astype(float)
                confs[c] = r.boxes.conf.cpu().numpy().astype(float)
            sup = support(grid, boxes, a.iou)
            people = associate(grid, boxes, confs, a.min_views, a.min_score, a.iou, sup)
            labels = label_cameras(grid, boxes, confs, people, sup, a.single_conf, a.refute_views, present=present)
            totals["steps"] += 1
            totals["people"] += len(people)
            for p in people:
                pw.writerow({"t": f"{T:.3f}", "x": f"{p.xy[0]:.2f}", "y": f"{p.xy[1]:.2f}",
                             "shape": SHAPE_NAMES[p.shape], "views": p.views, "score": f"{p.score:.2f}"})
            if k < a.preview:
                preview(out / "preview" / f"step_{k:04d}.jpg", grid, frames, labels, people, bounds)

            rows = []
            for c in idx:
                th = thumb(frames[c])
                if last[c] is not None and change_fraction(th, last[c]) < a.min_change:
                    continue
                last[c] = th
                cam, L = cams[c], labels[c]
                name = f"{cam.name}_t{round(T * 1000):08d}"
                cv2.imwrite(str(out / "images" / f"{name}.jpg"), frames[c], [cv2.IMWRITE_JPEG_QUALITY, 95])
                keep = list(L.keep)
                if a.add_missed:
                    keep += [(b, "added", 0.0) for b in L.missed]
                ok = not L.missed or a.add_missed
                if ok:
                    (out / "labels" / f"{name}.txt").write_text(
                        "".join(yolo_line(0, b, cam.width, cam.height) + "\n" for b, _, _ in keep))
                kinds = [kd for _, kd, _ in keep]
                mvw.writerow({"image": f"{name}.jpg", "camera": cam.name, "t": f"{T:.3f}",
                              "confirmed": kinds.count("confirmed"), "single": kinds.count("single"),
                              "added": kinds.count("added"), "dropped": len(L.dropped), "missed": len(L.missed),
                              "status": "labelled" if ok else "left out (missed)"})
                rows.append({"image": f"{name}.jpg", "camera": cam.name, "video": videos[c].path.name, "t": f"{T:.3f}"})
                totals["saved"] += 1
                totals["labelled"] += ok
                totals["confirmed"] += kinds.count("confirmed")
                totals["single"] += kinds.count("single")
                totals["dropped"] += len(L.dropped)
                totals["missed"] += len(L.missed)
            append_frames_csv(out, rows)
            print(f"  t {T:8.1f} s: {len(people)} people, {len(rows)} frames saved", end="\r")
    finally:
        mv.close()
        ppl.close()
    n = max(totals["steps"], 1)
    print(f"\n{totals['steps']} time steps, {totals['people'] / n:.1f} people per step on average")
    print(f"{totals['saved']} frames saved, {totals['labelled']} labelled, {totals['saved'] - totals['labelled']} "
          f"left out for a miss (multiview.csv)")
    print(f"boxes: {totals['confirmed']} confirmed by 2+ cameras, {totals['single']} single-view, "
          f"{totals['dropped']} dropped, {totals['missed']} missed")


if __name__ == "__main__":
    main()
