#!/usr/bin/env python3
"""Every view of every object in the registry, before SAM 3D: not only the
frames where the segmenter found it, but every frame of the export in which
its measured points are in sight and not hidden behind something else.

    python3 scoop_pipeline/assets/views.py <replica export> <registry dir>
        [--every 1] [--tol 0.05] [--min-visible 0.10] [--min-pixels 400]
        [--best 5] [--max-points 20000]

Per frame (every --every-th; 1 = all), per object whose box is in front of
the camera:
  * its points projected into the image (the export's pinhole camera);
  * a point is VISIBLE where the map's depth at its pixel is not nearer than
    the point by more than --tol + 1 % of the range (something in front
    hides it) and there is depth at all;
  * the view: share of the object's points visible, share cut off by the
    image edge, its visible pixels as a mask (points splatted at their own
    size, holes closed) with area and box, mean range, and whether the
    segmenter found it in this frame (objects.json's views).
A view counts when at least --min-visible of the object is visible and its
mask holds --min-pixels pixels.

For SAM 3D, each object's views are ranked by visible share x sqrt(mask
area) x (1 - cut-off share): the most complete, largest, uncut views first.
The --best of them get their masks written (masks/<id>_<frame>_proj.png; a
view the segmenter also found keeps its own mask beside it).

objects.json gains per object:
  views_all     every view: frame, visible, cut, area, bbox, range_m, segmented
  best_for_3d   the ranked --best views (frame, rank score, mask file)
  seen          frames with a view; segmented: the frames SAM 3 found it in
                (segmented / seen is the segmenter's recall on this object)
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from objects import Export                                               # noqa: E402


def read_ply_xyz(path):
    """xyz of a binary little-endian ply written by objects.write_ply (or any
    with float x, y, z first and uchar rgb)."""
    with open(path, "rb") as fh:
        n, props = 0, []
        while True:
            line = fh.readline().decode().strip()
            if line.startswith("element vertex"):
                n = int(line.split()[-1])
            elif line.startswith("property"):
                props.append(line.split()[1:])
            elif line == "end_header":
                break
        sizes = {"float": "<f4", "double": "<f8", "uchar": "u1", "uint8": "u1",
                 "int": "<i4", "uint": "<u4", "short": "<i2", "ushort": "<u2"}
        dt = np.dtype([(p[1], sizes[p[0]]) for p in props])
        rec = np.frombuffer(fh.read(n * dt.itemsize), dtype=dt, count=n)
    return np.c_[rec["x"], rec["y"], rec["z"]].astype(np.float64)


def view_of(P, depth, K, T_wc, tol, voxel):
    """One object in one frame: dict, or None when it is not in front."""
    import cv2
    Tcw = np.linalg.inv(T_wc)
    pc = P @ Tcw[:3, :3].T + Tcw[:3, 3]
    front = pc[:, 2] > 0.1
    if not front.any():
        return None
    H, W = depth.shape
    z = pc[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = np.round(K[0, 0] * pc[:, 0] / z + K[0, 2]).astype(np.int64)
        v = np.round(K[1, 1] * pc[:, 1] / z + K[1, 2]).astype(np.int64)
    inside = front & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    n = len(P)
    cut = float((front & ~inside).sum()) / max(int(front.sum()), 1)
    if not inside.any():
        return None
    d = depth[v[inside], u[inside]]
    zi = z[inside]
    vis_in = (d > 0) & (d >= zi - (tol + 0.01 * zi))
    vis = np.zeros(n, bool)
    vis[np.flatnonzero(inside)[vis_in]] = True
    if not vis.any():
        return {"visible": 0.0, "cut": round(cut, 4), "area": 0}
    mask = np.zeros((H, W), np.uint8)
    mask[v[vis], u[vis]] = 1
    r = int(np.ceil(K[0, 0] * voxel / float(np.median(z[vis])))) + 1     # a point's own size
    k = np.ones((2 * r + 1, 2 * r + 1), np.uint8)
    mask = cv2.morphologyEx(cv2.dilate(mask, k), cv2.MORPH_CLOSE, k)
    ys, xs = np.nonzero(mask)
    return {"visible": round(float(vis.sum()) / n, 4), "cut": round(cut, 4),
            "area": int(mask.sum()), "bbox": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
            "range_m": round(float(np.median(z[vis])), 3), "_mask": mask.astype(bool)}


def rank(view):
    return view["visible"] * np.sqrt(view["area"]) * (1.0 - view["cut"])


def in_front(aabb_min, aabb_max, T_wc):
    """Any corner of the box in front of the camera."""
    c = np.array([[x, y, z] for x in (aabb_min[0], aabb_max[0])
                  for y in (aabb_min[1], aabb_max[1]) for z in (aabb_min[2], aabb_max[2])])
    Tcw = np.linalg.inv(T_wc)
    return bool(((c @ Tcw[:3, :3].T + Tcw[:3, 3])[:, 2] > 0.1).any())


def all_views(export, reg_dir, every=1, tol=0.05, min_visible=0.10, min_pixels=400,
              best=5, max_points=20000, voxel=0.02, log=print):
    import cv2
    path = os.path.join(reg_dir, "objects.json")
    reg = json.load(open(path))
    objs = reg["objects"]
    rng = np.random.default_rng(0)
    pts = {}
    for o in objs:
        P = read_ply_xyz(os.path.join(reg_dir, o["points"]))
        if len(P) > max_points:
            P = P[rng.choice(len(P), max_points, replace=False)]
        pts[o["id"]] = P
        o["views_all"], o["_masks"] = [], {}
    frames = list(range(0, len(export), max(1, every)))
    for n, i in enumerate(frames):
        T = export.poses[i]
        cand = [o for o in objs if in_front(o["aabb_min"], o["aabb_max"], T)]
        if not cand:
            continue
        depth = export.depth(i)
        for o in cand:
            v = view_of(pts[o["id"]], depth, export.K, T, tol, voxel)
            if v is None or v["visible"] < min_visible or v["area"] < min_pixels:
                continue
            seg = {s["frame"] for s in o["views"]}
            v["frame"], v["segmented"] = i, i in seg
            v["rank"] = round(float(rank(v)), 3)
            o["_masks"][i] = v.pop("_mask")
            o["views_all"].append(v)
        if (n + 1) % 200 == 0:
            log("    %d / %d frames" % (n + 1, len(frames)))
    for o in objs:
        views = sorted(o["views_all"], key=lambda v: -v["rank"])
        top = []
        for v in views[:best]:
            f = "masks/%s_%06d_proj.png" % (o["id"], v["frame"])
            cv2.imwrite(os.path.join(reg_dir, f), o["_masks"][v["frame"]].astype(np.uint8) * 255)
            own = "masks/%s_%06d.png" % (o["id"], v["frame"])
            top.append({"frame": v["frame"], "rank": v["rank"], "visible": v["visible"],
                        "cut": v["cut"], "mask": f,
                        "segmenter_mask": own if os.path.isfile(os.path.join(reg_dir, own)) else None})
        o["best_for_3d"] = top
        o["seen"] = len(o["views_all"])
        o["segmented"] = len({s["frame"] for s in o["views"]} & {v["frame"] for v in o["views_all"]})
        del o["_masks"]
    reg["views_settings"] = {"every": every, "tol": tol, "min_visible": min_visible,
                             "min_pixels": min_pixels, "frames_checked": len(frames)}
    json.dump(reg, open(path, "w"), indent=1)
    return objs, len(frames)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__.split("\n\n", 1)[1])
    ap.add_argument("export")
    ap.add_argument("registry")
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--tol", type=float, default=0.05)
    ap.add_argument("--min-visible", type=float, default=0.10)
    ap.add_argument("--min-pixels", type=int, default=400)
    ap.add_argument("--best", type=int, default=5)
    ap.add_argument("--max-points", type=int, default=20000)
    a = ap.parse_args(argv)
    ex = Export(a.export)
    reg = os.path.expanduser(a.registry)
    objs, nf = all_views(ex, reg, a.every, a.tol, a.min_visible, a.min_pixels, a.best,
                         a.max_points)
    print("%d frames checked, %d objects:" % (nf, len(objs)))
    for o in objs:
        b = o["best_for_3d"][0] if o["best_for_3d"] else None
        print("  %-24s seen in %4d frames, segmented in %4d (%3.0f%%)  best: %s" % (
            o["id"], o["seen"], o["segmented"], 100.0 * o["segmented"] / max(o["seen"], 1),
            "frame %d, %.0f%% visible, %.0f%% cut" % (b["frame"], 100 * b["visible"], 100 * b["cut"])
            if b else "-"))
    print("-> %s" % os.path.join(reg, "objects.json"))


if __name__ == "__main__":
    sys.exit(main())
