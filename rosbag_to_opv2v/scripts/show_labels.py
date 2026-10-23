#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Show the labelled boxes in the robots' own camera images, so a box can be
checked by eye before it becomes ground truth.

    python3 scripts/show_labels.py --dataset <root>/train --labels labels.json --out <dir>
        [--seed=X,Y --seed=X,Y ... --seed-z Z]   # unfitted spots as 0.6 x 0.6 x 1.0 m boxes
        [--per-agent 3] [--max-range 8] [--camera 0]

For every label and every agent, the frames whose camera sees the box centre
(in front of the camera, inside the image, within --max-range) are ranked by
distance and centring, --per-agent of them at least 3 s apart are drawn with
the box's twelve edges, name and distance, and tiled into <out>/<name>.png.
<out>/overview_<agent>.png is the one frame per agent that sees the most
labels, all of them drawn. The yaml's camera block is used as the converter
wrote it: `cords` is the camera's optical frame in the map, `intrinsic` the
matrix of the written image. A box that lands beside its object in several
frames of both robots is in the wrong place or the wrong size; a box that is
right in one robot's frames and off in the other's says the two robots
disagree, which is a calibration or anchoring question, not a labelling one.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Dict, List, Optional

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ros2opv2v.geometry import invert, x_to_world                     # noqa: E402
from ros2opv2v.statics import box_corners                               # noqa: E402

EDGES = [(0, 1), (0, 2), (1, 3), (2, 3), (4, 5), (4, 6), (5, 7), (6, 7),
         (0, 4), (1, 5), (2, 6), (3, 7)]                 # box_corners' sign order


def project(points_world: np.ndarray, cords, intrinsic) -> np.ndarray:
    """(N, 3) map points -> (N, 3) [u, v, depth] through a camera block."""
    T_wc = x_to_world(cords)
    K = np.asarray(intrinsic, dtype=np.float64)
    P = np.c_[points_world, np.ones(len(points_world))] @ invert(T_wc).T
    z = P[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = K[0, 0] * P[:, 0] / z + K[0, 2]
        v = K[1, 1] * P[:, 1] / z + K[1, 2]
    return np.c_[u, v, z]


def locate(frame: dict, u: float, v: float, floor_z: float):
    """The map x, y where the pixel (u, v) of this frame's camera meets the
    floor plane z = floor_z, and the distance; None if the ray never does."""
    T_wc = x_to_world(frame["cords"])
    K = np.asarray(frame["intrinsic"], dtype=np.float64)
    d_cam = np.array([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], 1.0])
    d = T_wc[:3, :3] @ d_cam
    o = T_wc[:3, 3]
    if abs(d[2]) < 1e-9:
        return None
    t = (floor_z - o[2]) / d[2]
    if t <= 0:
        return None
    p = o + t * d
    return float(p[0]), float(p[1]), float(np.linalg.norm(t * d))


def load_frames(split_dir: str, camera: int) -> Dict[str, List[dict]]:
    """{agent: [{key, dir, cords, intrinsic, png, t_ns}]} for frames with the image."""
    out: Dict[str, List[dict]] = {}
    for scenario in sorted(glob.glob(os.path.join(split_dir, "*"))):
        if not os.path.isdir(scenario):
            continue
        for agent in sorted(os.listdir(scenario)):
            adir = os.path.join(scenario, agent)
            if not os.path.isdir(adir):
                continue
            try:
                int(agent)
            except ValueError:
                continue
            for y in sorted(glob.glob(os.path.join(adir, "*.yaml"))):
                key = os.path.basename(y)[:-5]
                png = os.path.join(adir, f"{key}_camera{camera}.png")
                if not os.path.exists(png):
                    continue
                with open(y) as handle:
                    p = yaml.safe_load(handle) or {}
                cam = p.get(f"camera{camera}")
                if not cam or not cam.get("intrinsic") or cam.get("image_present") is False:
                    continue
                out.setdefault(agent, []).append({
                    "key": key, "dir": adir, "cords": cam["cords"], "intrinsic": cam["intrinsic"],
                    "png": png, "t_ns": int(p.get("ros_frame_stamp_ns", 0))})
    return out


def image_size(frames: List[dict]) -> Optional[tuple]:
    import cv2
    im = cv2.imread(frames[0]["png"])
    return None if im is None else (im.shape[1], im.shape[0])


def sees(frame: dict, box: dict, size, max_range: float, margin: float = 0.05):
    """(depth, centring 0..1) if the camera sees the box centre, else None."""
    uvz = project(np.asarray([box["location"]], dtype=np.float64), frame["cords"], frame["intrinsic"])[0]
    u, v, z = uvz
    W, H = size
    if not (0.3 < z <= max_range) or not (margin * W <= u <= (1 - margin) * W) \
            or not (margin * H <= v <= (1 - margin) * H):
        return None
    centring = 1.0 - 2.0 * abs(u / W - 0.5)
    return float(z), float(centring)


def pick(frames: List[dict], box: dict, size, max_range: float, n: int, apart_s: float = 3.0):
    cands = []
    for fr in frames:
        s = sees(fr, box, size, max_range)
        if s is not None:
            z, c = s
            cands.append((z - 2.0 * c, z, fr))
    cands.sort(key=lambda t: t[0])
    chosen = []
    for score, z, fr in cands:
        if all(abs(fr["t_ns"] - o[1]["t_ns"]) >= apart_s * 1e9 for o in chosen):
            chosen.append((z, fr))
        if len(chosen) >= n:
            break
    return chosen, len(cands)


def draw_box(im, box: dict, cords, intrinsic, colour=(0, 220, 255), text: str = ""):
    import cv2
    uvz = project(box_corners(box), cords, intrinsic)
    if (uvz[:, 2] <= 0.1).any():
        return False
    pts = uvz[:, :2].round().astype(int)
    for a, b in EDGES:
        cv2.line(im, tuple(pts[a]), tuple(pts[b]), colour, 2, cv2.LINE_AA)
    if text:
        x, y = int(pts[:, 0].min()), int(pts[:, 1].min()) - 6
        cv2.putText(im, text, (max(2, x), max(14, y)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(im, text, (max(2, x), max(14, y)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 1, cv2.LINE_AA)
    return True


def tile(images, width: int = 640):
    import cv2
    rows = []
    for row in images:
        if not row:
            continue
        scaled = []
        for im in row:
            h = int(round(im.shape[0] * width / im.shape[1]))
            scaled.append(cv2.resize(im, (width, h)))
        hmax = max(s.shape[0] for s in scaled)
        rows.append(np.hstack([np.pad(s, ((0, hmax - s.shape[0]), (0, 0), (0, 0))) for s in scaled]))
    if not rows:
        return None
    wmax = max(r.shape[1] for r in rows)
    return np.vstack([np.pad(r, ((0, 0), (0, wmax - r.shape[1]), (0, 0))) for r in rows])


def show(split_dir: str, labels: List[dict], out: str, per_agent: int = 3, max_range: float = 8.0,
         camera: int = 0, log=print) -> dict:
    import cv2
    frames = load_frames(split_dir, camera)
    if not frames:
        raise SystemExit(f"no frames with camera{camera} images under {split_dir}")
    sizes = {a: image_size(f) for a, f in frames.items()}
    os.makedirs(out, exist_ok=True)
    report = {}
    for box in labels:
        name = str(box.get("name") or box["id"])
        rows, rep = [], {}
        for agent, fr in sorted(frames.items()):
            chosen, n_seen = pick(fr, box, sizes[agent], max_range, per_agent)
            rep[agent] = {"frames_seeing": n_seen, "shown": [f["key"] for _, f in chosen],
                          "nearest_m": round(min((z for z, _ in chosen), default=float("nan")), 2)}
            row = []
            for z, f in chosen:
                im = cv2.imread(f["png"])
                draw_box(im, box, f["cords"], f["intrinsic"], text=f"{name} {z:.1f} m")
                cv2.putText(im, f"agent {agent}  frame {f['key']}", (8, im.shape[0] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
                row.append(im)
            rows.append(row)
            log(f"  {name:<18} agent {agent}: seen in {n_seen} frames, showing {[f['key'] for _, f in chosen]}")
        sheet = tile(rows)
        if sheet is not None:
            cv2.imwrite(os.path.join(out, f"{name}.png"), sheet)
            rep["sheet"] = os.path.join(out, f"{name}.png")
        else:
            log(f"  {name:<18} no camera sees it within {max_range} m")
        report[name] = rep
    # the overview: per agent, the frame that sees the most labels
    for agent, fr in sorted(frames.items()):
        best, best_n = None, 0
        for f in fr:
            n = sum(1 for b in labels if sees(f, b, sizes[agent], max_range) is not None)
            if n > best_n:
                best, best_n = f, n
        if best is None:
            continue
        im = cv2.imread(best["png"])
        for b in labels:
            if sees(best, b, sizes[agent], max_range) is not None:
                draw_box(im, b, best["cords"], best["intrinsic"], text=str(b.get("name") or b["id"]))
        cv2.imwrite(os.path.join(out, f"overview_{agent}.png"), im)
        log(f"  overview agent {agent}: frame {best['key']} sees {best_n} label(s)")
        report[f"overview_{agent}"] = {"key": best["key"], "labels": best_n}
    with open(os.path.join(out, "show_labels.json"), "w") as handle:
        json.dump(report, handle, indent=2)
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, help="a converted split folder (holds the scenarios)")
    ap.add_argument("--labels", help="labels json (label_static.py --out)")
    ap.add_argument("--seed", action="append", default=[], metavar="X,Y",
                    help="an unfitted spot, drawn as a 0.6 x 0.6 x 1.0 m box standing at --seed-z")
    ap.add_argument("--seed-z", type=float, default=None,
                    help="floor z for --seed boxes (default: the agents' camera height minus 1.0 m)")
    ap.add_argument("--out", help="folder for the sheets (not needed with --locate)")
    ap.add_argument("--per-agent", type=int, default=3)
    ap.add_argument("--max-range", type=float, default=8.0)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--locate", action="append", default=[], metavar="AGENT:KEY:U,V",
                    help="a pixel of a frame's image (full size, as written) -> the map x, y "
                         "where it meets the floor: the base of an object, for a --manual box "
                         "or a seed. Floor z from --floor-z, else the labels' bottoms, else "
                         "the camera height minus 1 m")
    ap.add_argument("--floor-z", type=float, default=None)
    a = ap.parse_args()
    if a.locate:
        frames = load_frames(os.path.expanduser(a.dataset), a.camera)
        floor = a.floor_z
        if floor is None and a.labels:
            with open(os.path.expanduser(a.labels)) as handle:
                lab = json.load(handle)
            floor = float(np.median([b["location"][2] - b["extent"][2] for b in lab])) if lab else None
        if floor is None:
            zs = [f["cords"][2] for v in frames.values() for f in v[:50]]
            floor = (float(np.median(zs)) if zs else 0.0) - 1.0
        print("floor z %+.3f m" % floor)
        for item in a.locate:
            agent, key, uv = item.split(":")
            u, v = (float(x) for x in uv.split(","))
            fr = next((f for f in frames.get(agent, []) if f["key"] == key), None)
            if fr is None:
                print("  %s: no frame %s with camera%d for agent %s" % (item, key, a.camera, agent))
                continue
            hit = locate(fr, u, v, floor)
            if hit is None:
                print("  %s: the ray does not reach the floor (above the horizon)" % item)
            else:
                print("  %s -> x %.2f  y %.2f   (%.1f m from the camera)   --seed=%.2f,%.2f"
                      % (item, hit[0], hit[1], hit[2], hit[0], hit[1]))
        return 0
    labels: List[dict] = []
    if a.labels:
        with open(os.path.expanduser(a.labels)) as handle:
            labels = json.load(handle)
    if a.seed:
        z = a.seed_z
        if z is None:
            fr = load_frames(os.path.expanduser(a.dataset), a.camera)
            zs = [f["cords"][2] for v in fr.values() for f in v[:50]]
            z = (float(np.median(zs)) if zs else 0.0) - 1.0
        for k, item in enumerate(a.seed):
            x, y = (float(v) for v in item.split(","))
            labels.append({"id": 9000 + k, "name": f"seed_{k + 1}", "location": [x, y, z + 0.5],
                           "extent": [0.3, 0.3, 0.5], "angle": [0.0, 0.0, 0.0]})
    if not labels:
        print("nothing to show: give --labels or --seed", file=sys.stderr)
        return 2
    if not a.out:
        print("--out is needed to write the sheets", file=sys.stderr)
        return 2
    show(os.path.expanduser(a.dataset), labels, os.path.expanduser(a.out), a.per_agent, a.max_range, a.camera)
    print(f"wrote {a.out}/<name>.png, overview_<agent>.png, show_labels.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
