#!/usr/bin/env python3
"""One identity per person for the whole room, from any sensors (see TRACKING.md).

Simulated room with known identities (try it first, prints the scores):
    python room_track.py --sim dense --out out/sim_dense
    python room_track.py --sim sparse --out out/sim_sparse      # cameras with a blind band
    python room_track.py --sim fused --out out/sim_fused        # cameras + radar

Any sensors, already as measurements (one CSV per sensor, format in ed305/tracking/adapters.py):
    python room_track.py --measurements radar.csv robot_1.csv --sensors configs/sensors_example.yaml --out out/run

The 16 cameras (calibrated, synced recordings, as multiview_label.py), plus other sensors:
    python room_track.py --rec rec/ --calib results/extrinsics/cameras.json \\
        --model runs/detect/ed305_v1/weights/best.pt --reid yolo26s-reid.onnx --out out/run
    python room_track.py --rec rec/ --calib cameras.json --measurements radar.csv \\
        --sensors configs/sensors_example.yaml --video cam_3 --out out/run

Writes to --out:
  tracks.csv        track, t, x, y, vx, vy: every person's path in the room frame
  measurements.csv  every measurement with the track it was given (empty: clutter)
  boxes.csv         (cameras) t, camera, x0, y0, x1, y1, track: each camera box with the room identity
  <cam>_ids.mp4     (--video) that camera with the room identities drawn
  summary.md        tracks, how long, which sensors; with --sim also the identity scores
"""
import argparse
import csv
import zlib
from pathlib import Path

import cv2
import numpy as np

from ed305.tracking import TrackerConfig, id_metrics, load_config, track
from ed305.tracking.adapters import (camera_sensor, people_to_measurements, read_measurements_csv,
                                     write_measurements_csv)
from ed305.tracking.metrics import tracks_to_dict
from ed305.tracking.model import Zones


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--sim", choices=["dense", "sparse", "fused"], help="simulated room instead of data")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--measurements", nargs="*", type=Path, default=[], help="measurement CSV files (any sensors)")
    ap.add_argument("--sensors", type=Path, help="sensor models, entry zones, tracker settings (YAML)")
    ap.add_argument("--rec", type=Path, help="folder with cam_N recordings (+ cam_N_times.csv)")
    ap.add_argument("--calib", type=Path, help="cameras.json or cameras YAML (room frame, floor z = 0)")
    ap.add_argument("--model", default="yolo26s.pt", help="person detector (your fine-tuned best.pt)")
    ap.add_argument("--reid", help="appearance model for the cameras, e.g. yolo26s-reid.onnx (default: none)")
    ap.add_argument("--rate", type=float, default=10.0, help="camera time steps per second (default 10)")
    ap.add_argument("--start", type=float)
    ap.add_argument("--end", type=float)
    ap.add_argument("--bounds", help="floor area x0,x1,y0,y1 (default: around the cameras)")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--device", default=None)
    ap.add_argument("--det-conf", type=float, default=0.05)
    ap.add_argument("--min-views", type=int, default=2)
    ap.add_argument("--video", help="also write this camera with the room identities, e.g. cam_3")
    ap.add_argument("--no-link", action="store_true", help="stop after stage 1 (tracklets), for comparison")
    return ap.parse_args()


def camera_measurements(a, out):
    """Recordings -> camera measurements (one per person per time step) and the cameras' SensorModel."""
    from ultralytics import YOLO

    from ed305.camera import load_any
    from ed305.multiview import Grid, associate, support
    from multiview_label import SyncedVideo, find_videos

    vids = find_videos(a.rec)
    cams = [c for c in load_any(a.calib) if c.name in vids]
    if len(cams) < 2:
        raise SystemExit("need 2+ cameras with both a recording and a calibration")
    videos = [SyncedVideo(vids[c.name]) for c in cams]
    if not all(v.synced for v in videos):
        print("WARNING some cameras have no *_times.csv: assuming they started together")
    if a.bounds:
        bounds = tuple(float(v) for v in a.bounds.split(","))
    else:
        xy = np.array([c.center[:2] for c in cams])
        bounds = (xy[:, 0].min(), xy[:, 0].max(), xy[:, 1].min(), xy[:, 1].max())
    print(f"{len(cams)} cameras; expected boxes per floor cell ...")
    grid = Grid(cams, bounds, 0.1)
    model = YOLO(a.model)
    person = [k for k, v in model.names.items() if v == "person"]
    enc = None
    if a.reid:
        from ultralytics.trackers.utils.reid import ReID
        enc = ReID(a.reid, device=a.device)
    t0 = max(v.t[0] for v in videos) if a.start is None else a.start
    t1 = min(v.t[-1] for v in videos) if a.end is None else a.end
    steps = np.arange(t0, t1, 1.0 / a.rate)
    meas = []
    for k, T in enumerate(steps):
        frames = [v.at(T) for v in videos]
        idx = [c for c, f in enumerate(frames) if f is not None]
        if len(idx) < 2:
            continue
        res = model.predict([frames[c] for c in idx], conf=a.det_conf, iou=0.6, imgsz=a.imgsz,
                            classes=person, device=a.device, verbose=False)
        boxes = [np.zeros((0, 4)) for _ in cams]
        confs = [np.zeros(0) for _ in cams]
        for c, r in zip(idx, res):
            boxes[c] = r.boxes.xyxy.cpu().numpy().astype(float)
            confs[c] = r.boxes.conf.cpu().numpy().astype(float)
        sup = support(grid, boxes, 0.4)
        people = associate(grid, boxes, confs, a.min_views, 0.8, 0.4, sup)
        feats = None
        if enc is not None:
            feats = [None] * len(cams)
            for c in idx:
                used = sorted({d for p in people for cc, d in p.dets.items() if cc == c})
                if used:
                    b = boxes[c][used]
                    xywh = np.c_[(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2, b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]]
                    fs = enc(frames[c], xywh)
                    feats[c] = {d: f for d, f in zip(used, fs)}
            feats = [{} if f is None else f for f in feats]
            feats = [[f.get(d) for d in range(len(boxes[c]))] for c, f in enumerate(feats)]
        meas += people_to_measurements(float(T), people, cams, boxes, confs, feats)
        print(f"  t {T:8.1f} s: {len(people)} people", end="\r")
    print()
    area = (bounds[1] - bounds[0]) * (bounds[3] - bounds[2])
    return meas, camera_sensor(grid, a.min_views, clutter=0.2 / area), cams, videos, steps


def write_outputs(a, meas, tracks, owner, sensors):
    out = a.out
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "tracks.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["track", "t", "x", "y", "vx", "vy"])
        for tk in tracks:
            t, x = tk.samples()
            for ti, xi in zip(t, x):
                w.writerow([tk.id, f"{ti:.3f}", f"{xi[0]:.3f}", f"{xi[1]:.3f}", f"{xi[2]:.3f}", f"{xi[3]:.3f}"])
    write_measurements_csv(out / "measurements.csv", meas, owner)
    cam_boxes = [(k, m) for k, m in enumerate(meas) if m.meta.get("boxes")]
    if cam_boxes:
        with open(out / "boxes.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t", "camera", "x0", "y0", "x1", "y1", "track"])
            for k, m in cam_boxes:
                for cam, b in m.meta["boxes"].items():
                    w.writerow([f"{m.t:.3f}", cam, *(f"{v:.1f}" for v in b), owner.get(k, "")])


def summary(tracks, meas, owner):
    lines = ["| track | from (s) | to (s) | tracklets | measurements | sensors |", "|---|---|---|---|---|---|"]
    for tk in sorted(tracks, key=lambda tk: tk.t0):
        sens = sorted(set().union(*[tr.sensors for tr in tk.tracklets]))
        lines.append(f"| {tk.id} | {tk.t0:.1f} | {tk.t1:.1f} | {len(tk.tracklets)} | {len(tk.measurements)} | "
                     f"{', '.join(sens)} |")
    n_clutter = len(meas) - len(owner)
    return "\n".join(lines) + f"\n\n{len(tracks)} tracks; {n_clutter} of {len(meas)} measurements not in any track.\n"


def color(tid):
    h = zlib.crc32(str(tid).encode()) % 180
    return tuple(int(v) for v in cv2.cvtColor(np.uint8([[[h, 200, 230]]]), cv2.COLOR_HSV2BGR)[0, 0])


def render(a, cams, videos, steps, meas, owner):
    names = [c.name for c in cams]
    if a.video not in names:
        raise SystemExit(f"--video {a.video}: not one of {names}")
    c = names.index(a.video)
    from multiview_label import SyncedVideo
    v = SyncedVideo(videos[c].path)
    by_t = {}
    for k, m in enumerate(meas):
        b = m.meta.get("boxes", {}).get(a.video)
        if b is not None:
            by_t.setdefault(round(m.t, 3), []).append((b, owner.get(k)))
    path = a.out / f"{a.video}_ids.mp4"
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), a.rate, v.size)
    for T in steps:
        f = v.at(T)
        if f is None:
            continue
        for b, tid in by_t.get(round(float(T), 3), []):
            col = color(tid) if tid else (150, 150, 150)
            cv2.rectangle(f, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), col, 3)
            cv2.putText(f, tid or "?", (int(b[0]), int(b[1]) - 8), cv2.FONT_HERSHEY_SIMPLEX, 1.0, col, 2)
        vw.write(f)
    vw.release()
    print(f"video: {path}")


def main():
    a = parse()
    sensors, zones, overrides = ({}, Zones(), {}) if a.sensors is None else load_config(a.sensors)
    cfg = TrackerConfig(**overrides)
    meas, gt, cams = [], None, None
    if a.sim:
        from ed305.tracking.sim import simulate
        meas, sensors, zones, gt = simulate(a.sim, seed=a.seed)
    if a.rec:
        if not a.calib:
            raise SystemExit("--rec needs --calib")
        cm, cam_model, cams, videos, steps = camera_measurements(a, a.out)
        meas += cm
        sensors.setdefault("cams", cam_model)
    for p in a.measurements:
        m = read_measurements_csv(p)
        meas += m
        for name in {x.sensor for x in m} - set(sensors):
            print(f"WARNING no sensor model for '{name}' in --sensors: using defaults (p_detect 0.9, clutter 1e-3)")
    if not meas:
        raise SystemExit("no measurements: give --sim, --rec or --measurements")
    tracks, owner = track(meas, sensors, zones, cfg, link=not a.no_link)
    write_outputs(a, meas, tracks, owner, sensors)
    text = summary(tracks, meas, owner)
    if gt is not None:
        m = id_metrics(gt, tracks_to_dict(tracks))
        base = id_metrics(gt, tracks_to_dict(track(meas, sensors, zones, cfg, link=False)[0]))
        text = (f"simulated '{a.sim}', seed {a.seed}: {m['gt_ids']} people/robots\n\n"
                f"| | IDF1 | IDP | IDR | switches | tracks |\n|---|---|---|---|---|---|\n"
                f"| tracklets only (stage 1) | {base['IDF1']:.3f} | {base['IDP']:.3f} | {base['IDR']:.3f} | "
                f"{base['switches']} | {base['pred_ids']} |\n"
                f"| room tracker (stage 1 + 2) | {m['IDF1']:.3f} | {m['IDP']:.3f} | {m['IDR']:.3f} | "
                f"{m['switches']} | {m['pred_ids']} |\n\n") + text
    (a.out / "summary.md").write_text(text)
    print(text)
    if a.video:
        if cams is None:
            raise SystemExit("--video needs --rec")
        render(a, cams, videos, steps, meas, owner)


if __name__ == "__main__":
    main()
