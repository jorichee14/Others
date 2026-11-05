#!/usr/bin/env python3
"""The mapping camera's own frames as a video with the object registry on
them: every object the camera sees masked, boxed and named, frame by frame,
the same id wherever it shows up.

    python3 scoop_pipeline/assets/camera_tour.py <replica export> <registry> [<out .mp4>]
        [--every 1] [--fps 15] [--start 0] [--frames 0] [--scale 1.0]
        [--min-visible 0.10] [--min-pixels 400] [--max-dist 8] [--alpha 0.35]
        [--skip-label wall,floor,ceiling]

The export is the one the registry was built from (objects.py, views.py:
mcap_convert.py --format replica, depth rendered from the map): its colour
frames (undistorted onto the depth's camera), map depth and camera poses.
The registry is assets/ after consolidate. Default out: <registry>/camera_tour.mp4.

Per frame, each object's points projected with the frame's pose (views.view_of):
the ones the map's depth does not hide (5 cm + 1 % of the range) are what
the camera sees of it. With --min-visible of its points seen, a mask of
--min-pixels or more and nearer than --max-dist: its mask tinted in its
label's colour (--alpha), its box drawn, named by its id ("on <id>" when it
stands on another), its range in the corner. A panel counts what is in view.
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "datasets"))

from tour_objects import colour_of, draw                                # noqa: E402
from views import in_front, read_ply_xyz, view_of                       # noqa: E402


def frame_detections(objs, pts, depth, K, T_wc, tol=0.05, voxel=0.02, min_visible=0.10,
                     min_pixels=400, max_dist=8.0):
    """[{id, label, on, z, box, mask}] of the objects seen in one frame."""
    out = []
    for o, P in zip(objs, pts):
        if "aabb_min" in o and not in_front(o["aabb_min"], o["aabb_max"], T_wc):
            continue
        v = view_of(P, depth, K, T_wc, tol, voxel)
        if not v or v["visible"] < min_visible or v["area"] < min_pixels or v["range_m"] > max_dist:
            continue
        out.append({"id": o["id"], "label": o["label"], "on": o.get("on"), "z": v["range_m"],
                    "box": tuple(v["bbox"]), "mask": v["_mask"]})
    return out


def tint(bgr, dets, alpha):
    """Each detection's mask tinted in its label's colour, its outline drawn."""
    over = bgr.copy()
    for d in sorted(dets, key=lambda d: -d["z"]):
        c = colour_of(d["label"])
        over[d["mask"]] = c
    out = cv2.addWeighted(over, alpha, bgr, 1 - alpha, 0)
    for d in dets:
        cs, _ = cv2.findContours(d["mask"].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cs, -1, colour_of(d["label"]), 1, cv2.LINE_AA)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__.split("\n\n", 1)[1])
    ap.add_argument("export")
    ap.add_argument("registry")
    ap.add_argument("out", nargs="?", default=None)
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--fps", type=float, default=15.0)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--frames", type=int, default=0, help="at most this many (0: all)")
    ap.add_argument("--scale", type=float, default=1.0, help="the video's size against the frames'")
    ap.add_argument("--tol", type=float, default=0.05)
    ap.add_argument("--min-visible", type=float, default=0.10)
    ap.add_argument("--min-pixels", type=int, default=400)
    ap.add_argument("--max-dist", type=float, default=8.0)
    ap.add_argument("--alpha", type=float, default=0.35)
    ap.add_argument("--max-points", type=int, default=5000, help="per object, for speed")
    ap.add_argument("--skip-label", default="wall,floor,ceiling")
    a = ap.parse_args(argv)
    from objects import Export
    from lod3dgs import video_writer
    ex = Export(a.export)
    reg = os.path.expanduser(a.registry)
    out = os.path.expanduser(a.out or os.path.join(reg, "camera_tour.mp4"))
    skip = {s for s in a.skip_label.split(",") if s}
    objs = [o for o in json.load(open(os.path.join(reg, "objects.json")))["objects"] if o["label"] not in skip]
    for lab in sorted({o["label"] for o in objs}):                       # colours fixed for the whole video
        colour_of(lab)
    rng = np.random.default_rng(0)
    pts = []
    for o in objs:
        P = read_ply_xyz(os.path.join(reg, o["points"]))
        if len(P) > a.max_points:
            P = P[rng.choice(len(P), a.max_points, replace=False)]
        pts.append(P)
    idx = list(range(a.start, len(ex), max(1, a.every)))
    if a.frames:
        idx = idx[:a.frames]
    W, H = int(round(ex.W * a.scale)), int(round(ex.H * a.scale))
    print("%s: %d frames, %d used (every %d) at %.0f fps (%.0f s); %d objects (%s left out) -> %s"
          % (a.export, len(ex), len(idx), a.every, a.fps, len(idx) / a.fps, len(objs),
             ",".join(sorted(skip)) or "none", out))
    vw = video_writer(out, a.fps, (W, H))
    seen = {}
    for n, i in enumerate(idx):
        bgr = np.ascontiguousarray(ex.image(i)[:, :, ::-1])
        dets = frame_detections(objs, pts, ex.depth(i), ex.K, ex.poses[i], a.tol, 0.02,
                                a.min_visible, a.min_pixels, a.max_dist)
        img = tint(bgr, dets, a.alpha)
        if a.scale != 1.0:
            img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
            for d in dets:
                d["box"] = tuple(int(round(v * a.scale)) for v in d["box"])
        vw.write(draw(img, dets, n / a.fps))
        for d in dets:
            seen[d["id"]] = d["label"]
        if (n + 1) % 200 == 0:
            print("  %d / %d frames, %d objects seen so far" % (n + 1, len(idx), len(seen)))
    vw.release()
    labels = {}
    for lab in seen.values():
        labels[lab] = labels.get(lab, 0) + 1
    print("-> %s: %d frames; %d of %d objects shown (%s)" % (
        out, len(idx), len(seen), len(objs),
        ", ".join("%s %d" % kv for kv in sorted(labels.items(), key=lambda kv: -kv[1]))))


if __name__ == "__main__":
    sys.exit(main())
