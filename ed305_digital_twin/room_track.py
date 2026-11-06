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
  persons.csv       (cameras) which tracks are one person (tracks staying at the same spot are merged)
  persons_state.csv per person, every 1/--rate s: x, y, vx, vy, position covariance (sxx sxy syy;
                    sigma_m its largest std), seen, age_s since the last report, views (cameras
                    agreeing), id_conf (how sure the id is the same person since the track began),
                    link_conf (how sure the most recent identity decision was): for planners
  boxes.csv         (cameras) t, camera, x0, y0, x1, y1, person: every detection, labelled from the
                    persons' positions (the same id in every camera)
  <cam>_ids.mp4     (--video cam_3) that camera with the room identities drawn
  all_ids.mp4       (--video all) every camera in one grid + the floor plan with everyone's path
  summary.md        tracks, how long, which sensors; with --sim also the identity scores
"""
import argparse
import csv
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
    ap.add_argument("--video", help="also write a video with the room identities: a camera (cam_3), or 'all' "
                                    "(every camera in one grid + the floor plan)")
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
        dev = a.device
        if dev is not None and str(dev).split(",")[0].strip().isdigit():
            dev = f"cuda:{str(dev).split(',')[0].strip()}"         # the detector takes 0, the ReID model cuda:0
        enc = ReID(a.reid, device=dev)
    t0 = max(v.t[0] for v in videos) if a.start is None else a.start
    t1 = min(v.t[-1] for v in videos) if a.end is None else a.end
    steps = np.arange(t0, t1, 1.0 / a.rate)
    meas = []
    a.detections = {}
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
        a.detections[round(float(T), 3)] = (boxes, confs)
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
    a.bounds_used = bounds
    a.grid = grid
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


def summary(tracks, meas, owner):
    lines = ["| track | from (s) | to (s) | tracklets | measurements | sensors |", "|---|---|---|---|---|---|"]
    for tk in sorted(tracks, key=lambda tk: tk.t0):
        sens = sorted(set().union(*[tr.sensors for tr in tk.tracklets]))
        lines.append(f"| {tk.id} | {tk.t0:.1f} | {tk.t1:.1f} | {len(tk.tracklets)} | {len(tk.measurements)} | "
                     f"{', '.join(sens)} |")
    n_clutter = len(meas) - len(owner)
    return "\n".join(lines) + f"\n\n{len(tracks)} tracks; {n_clutter} of {len(meas)} measurements not in any track.\n"


def color(tid):
    from mosaic import color as c
    return c(tid)


def floor_plan(tracks, cams, bounds, T, h=864, seen_within=0.5):
    """The room from above at time T, to scale: cameras, and every person's id tag at their spot
    (grey when not seen in the last `seen_within` s: position carried over a gap)."""
    x0, x1, y0, y1 = bounds
    pad = 0.4
    w = int(h * (x1 - x0 + 2 * pad) / (y1 - y0 + 2 * pad))
    w = min(w, int(h * 1.3))
    s = min((h - 30) / (y1 - y0 + 2 * pad), w / (x1 - x0 + 2 * pad))
    w = int((x1 - x0 + 2 * pad) * s)
    img = np.full((h, w, 3), 248, np.uint8)
    top = max(26, int((h - (y1 - y0 + 2 * pad) * s) / 2))                 # the room centred vertically
    px = lambda x, y: (int((x - x0 + pad) * s), int((y1 + pad - y) * s) + top)
    for gx in np.arange(np.ceil(x0), x1 + 1e-9):
        cv2.line(img, px(gx, y0), px(gx, y1), (232, 232, 232), 1)
    for gy in np.arange(np.ceil(y0), y1 + 1e-9):
        cv2.line(img, px(x0, gy), px(x1, gy), (232, 232, 232), 1)
    cv2.rectangle(img, px(x0, y1), px(x1, y0), (150, 150, 150), 1)
    for c in cams:
        cv2.drawMarker(img, px(*c.center[:2]), (120, 120, 120), cv2.MARKER_TRIANGLE_UP, 8, 1)
        cv2.putText(img, c.name.replace("cam_", ""), px(c.center[0] + 0.08, c.center[1] + 0.08),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (120, 120, 120), 1, cv2.LINE_AA)
    n_here, placed = 0, []
    font, fs, th = cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
    items = []
    for tk in tracks:
        if not (tk.t0 <= T <= tk.t1):
            continue
        P = tk.position_at([T])
        P = P[~np.isnan(P[:, 0])]
        if not len(P):
            continue
        t_upd = tk.samples()[0]
        seen = bool(np.any((t_upd <= T) & (t_upd > T - seen_within)))
        items.append((tk.id, P, seen))
    for tid, P, seen in items:
        col = color(tid)
        x, y = px(*P[-1])
        cv2.circle(img, (x, y), 2, col, -1, cv2.LINE_AA)
        (tw, tht), _ = cv2.getTextSize(tid, font, fs, th)
        box = None
        offsets = [(6, -6), (6, tht + 6), (-tw - 6, -6), (-tw - 6, tht + 6), (6, -tht - 12), (-tw // 2, -14),
                   (-tw // 2, tht + 14)]
        offsets += [(int(r * np.cos(a)) - tw // 2, int(r * np.sin(a)) + tht // 2)          # further out, all around
                    for r in (28, 42, 56) for a in np.linspace(0, 2 * np.pi, 12, endpoint=False)]
        for dx, dy in offsets:
            b = (x + dx - 2, y + dy - tht - 2, x + dx + tw + 2, y + dy + 4)
            if not any(b[0] < q[2] and q[0] < b[2] and b[1] < q[3] and q[1] < b[3] for q in placed):
                box = b
                break
        box = box or (x + 6 - 2, y - 6 - tht - 2, x + 6 + tw + 2, y - 6 + 4)
        placed.append(box)
        cx, cy = (box[0] + box[2]) // 2, (box[1] + box[3]) // 2
        if abs(cx - x) > tw or abs(cy - y) > 2 * tht:              # tag moved away: a thin line to its spot
            cv2.line(img, (x, y), (min(max(x, box[0]), box[2]), min(max(y, box[1]), box[3])), col, 1, cv2.LINE_AA)
        cv2.rectangle(img, box[:2], box[2:], (255, 255, 255) if seen else (240, 240, 240), -1)
        cv2.rectangle(img, box[:2], box[2:], col, 1)
        cv2.putText(img, tid, (box[0] + 2, box[3] - 4), font, fs, col if seen else (150, 150, 150), th, cv2.LINE_AA)
        n_here += seen
    cv2.putText(img, f"t {T:.1f} s   seen now: {n_here}", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 80), 1,
                cv2.LINE_AA)
    return img


def labelled_boxes(a, cams, persons, min_conf=0.4):
    """{(t, camera): [(box, person id or None, label confidence, detector score)]}: every detection
    labelled from the persons' positions (unlabelled ones only if the detector was fairly sure)."""
    from ed305.tracking.relabel import label_boxes
    by = {}
    for T, (boxes, confs) in a.detections.items():
        labels, lconf = label_boxes(a.grid, persons, T, boxes, with_conf=True)
        for c, cam in enumerate(cams):
            for b, cf, lab, lc in zip(boxes[c], confs[c], labels[c], lconf[c]):
                if lab is not None or cf >= min_conf:
                    by.setdefault((T, cam.name), []).append((b, lab, lc, cf))
    return by


def write_boxes(a, by):
    with open(a.out / "boxes.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "camera", "x0", "y0", "x1", "y1", "person", "label_conf", "det_score"])
        for (T, cam), items in sorted(by.items()):
            for b, lab, lc, cf in items:
                w.writerow([f"{T:.3f}", cam, *(f"{v:.1f}" for v in b), lab or "", f"{lc:.2f}", f"{cf:.2f}"])


def write_person_states(a, persons, meas, times, cfg):
    """persons_state.csv: per person, on a regular time grid, where they are and how far to trust it."""
    with open(a.out / "persons_state.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["person", "t", "x", "y", "vx", "vy", "sxx", "sxy", "syy", "sigma_m", "seen", "age_s",
                    "views", "id_conf", "link_conf"])
        for p in persons:
            ts = times[(times >= p.t0) & (times <= p.t1)]
            if not len(ts):
                continue
            S = p.state_at(ts, cfg)
            vt, vn = [], []                                   # views of each report: cameras agreeing (or 1)
            for m in getattr(p, "members", [p]):
                for k in m.measurements:
                    vt.append(meas[k].t)
                    vn.append(len((meas[k].meta or {}).get("boxes") or {}) or 1)
            order = np.argsort(vt)
            vt, vn = np.array(vt)[order], np.array(vn)[order]
            for i, t in enumerate(ts):
                if np.isnan(S["xy"][i, 0]):
                    continue
                j = np.searchsorted(vt, t, side="right") - 1
                views = int(vn[j]) if j >= 0 and t - vt[j] <= 0.5 else 0
                C = S["cov"][i]
                w.writerow([p.id, f"{t:.3f}", f"{S['xy'][i, 0]:.3f}", f"{S['xy'][i, 1]:.3f}", f"{S['v'][i, 0]:.3f}",
                            f"{S['v'][i, 1]:.3f}", f"{C[0, 0]:.5f}", f"{C[0, 1]:.5f}", f"{C[1, 1]:.5f}",
                            f"{np.sqrt(np.linalg.eigvalsh(C).max()):.3f}", int(S["seen"][i]), f"{S['age'][i]:.2f}",
                            views, f"{S['id_conf'][i]:.3f}", f"{S['link_conf'][i]:.3f}"])


def render_all(a, cams, videos, steps, by, persons):
    """Every camera in one grid with the room identities, and the floor plan on the right."""
    import math
    from mosaic import draw_boxes, even, grid, tile
    from multiview_label import SyncedVideo
    size = (480, 270)
    vs = [SyncedVideo(v.path) for v in videos]
    cols = math.ceil(math.sqrt(len(cams)))
    path = a.out / "all_ids.mp4"
    vw = None
    for T in steps:
        tiles = []
        for c, v in zip(cams, vs):
            f = v.at(T)
            img = tile(f, size, f"{c.name}   t {T:7.2f} s")
            items = by.get((round(float(T), 3), c.name), [])
            if f is not None and items:
                draw_boxes(img, [it[0] for it in items], labels=[it[1] or "?" for it in items],
                           colors=[color(it[1]) if it[1] else (150, 150, 150) for it in items], scale=size[0] / f.shape[1])
            tiles.append(img)
        g = grid(tiles, cols, size)
        out = even(np.hstack([g, floor_plan(persons, cams, a.bounds_used, float(T), h=g.shape[0])]))
        if vw is None:
            vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), a.rate, (out.shape[1], out.shape[0]))
        vw.write(out)
    if vw is not None:
        vw.release()
    print(f"video: {path}")


def render(a, cams, videos, steps, by):
    names = [c.name for c in cams]
    if a.video not in names:
        raise SystemExit(f"--video {a.video}: not one of {names}")
    c = names.index(a.video)
    from multiview_label import SyncedVideo
    v = SyncedVideo(videos[c].path)
    by_t = {T: items for (T, cam), items in by.items() if cam == a.video}
    path = a.out / f"{a.video}_ids.mp4"
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), a.rate, v.size)
    for T in steps:
        f = v.at(T)
        if f is None:
            continue
        for b, tid, *_ in by_t.get(round(float(T), 3), []):
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
    from ed305.tracking.relabel import Person, merge_colocated
    if cams is not None:
        persons = merge_colocated(tracks, meas)
        with open(a.out / "persons.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["track", "person"])
            for p in persons:
                for m in p.members:
                    w.writerow([m.id, p.id])
        by = labelled_boxes(a, cams, persons)
        write_boxes(a, by)
        text += (f"{len(persons)} persons after merging tracks that stay at the same spot "
                 f"(persons.csv); camera boxes labelled from the persons' positions (boxes.csv).\n")
    else:
        persons = [Person(tk.id, [tk]) for tk in tracks]
    t_lo = min(m.t for m in meas)
    t_hi = max(m.t for m in meas)
    write_person_states(a, persons, meas, np.round(np.arange(t_lo, t_hi + 1e-9, 1.0 / a.rate), 3), cfg)
    text += "per-person state with trust values: persons_state.csv\n"
    (a.out / "summary.md").write_text(text)
    print(text)
    if a.video:
        if cams is None:
            raise SystemExit("--video needs --rec")
        if a.video == "all":
            render_all(a, cams, videos, steps, by, persons)
        else:
            render(a, cams, videos, steps, by)


if __name__ == "__main__":
    main()
