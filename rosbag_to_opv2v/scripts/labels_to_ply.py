#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The labelled boxes drawn INTO the cloud, as a PLY a viewer opens: the box
edges as coloured points beside the returns they are supposed to hold.

    python3 scripts/labels_to_ply.py --pcd cloud_in_ref.pcd --labels labels.json
        --out /tmp/gt.ply [--around 2.5] [--ceiling 2.0] [--step 0.015]
        [--name chair_1 --name chair_2]

A projection into a camera image shows a box against what the camera saw; this
shows it against what the LiDAR measured, which is what the box was fitted to
and what a detector will be trained on. A box that looks right in both is
right.

Edges are drawn as dense points (--step apart) because PLY edges are ignored by
most viewers; every point of a box carries that box's colour, the cloud around
it is grey, and the points INSIDE each box are tinted with it, so a box holding
nothing is obvious without rotating the view.

--around crops to that radius round each box (0 keeps the whole cloud, which
is usually too much to orbit). --ceiling cuts what stands more than that over
the box, so a ceiling or a beam does not hide the object from above.
"""
import argparse
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ros2opv2v.statics import points_in_box, read_cloud_xyz, write_ply     # noqa: E402

# one colour per box, in order; grey is the cloud
WHEEL = [(255, 64, 64), (64, 160, 255), (96, 220, 96), (255, 176, 48),
         (208, 112, 255), (64, 224, 208), (255, 112, 160), (180, 180, 60)]


def corners(box):
    """The box's 8 corners in the map frame."""
    ex, ey, ez = (float(v) for v in box["extent"])
    unit = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float)
    local = unit * np.array([ex, ey, ez])
    yaw = math.radians(float(box["angle"][1]))
    cos, sin = math.cos(yaw), math.sin(yaw)
    R = np.array([[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]])
    return local @ R.T + np.asarray(box["location"], float)


def edges(box, step=0.015):
    """Points along the box's 12 edges, `step` apart."""
    c = corners(box)
    # corner index = (sx, sy, sz) bits, in the order corners() builds them
    pairs = [(0, 1), (2, 3), (4, 5), (6, 7),          # along z
             (0, 2), (1, 3), (4, 6), (5, 7),          # along y
             (0, 4), (1, 5), (2, 6), (3, 7)]          # along x
    out = []
    for a, b in pairs:
        n = max(int(np.linalg.norm(c[b] - c[a]) / step), 2)
        out.append(c[a] + np.linspace(0, 1, n)[:, None] * (c[b] - c[a]))
    return np.vstack(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pcd", required=True, help="the cloud the boxes were fitted in (.pcd/.ply)")
    ap.add_argument("--labels", required=True, help="labels json (label_static.py --out)")
    ap.add_argument("--out", required=True, help="the .ply to write")
    ap.add_argument("--name", action="append", default=[], help="only these boxes (repeatable)")
    ap.add_argument("--around", type=float, default=2.5,
                    help="crop to this radius round each box (0: the whole cloud)")
    ap.add_argument("--ceiling", type=float, default=2.0,
                    help="drop what stands more than this over a box's top (0: keep it)")
    ap.add_argument("--step", type=float, default=0.015, help="spacing along an edge (m)")
    ap.add_argument("--margin", type=float, default=0.0,
                    help="count a point inside when it is within this of the box")
    ap.add_argument("--max-points", type=int, default=20000000)
    a = ap.parse_args(argv)

    labels = json.load(open(os.path.expanduser(a.labels)))
    if a.name:
        labels = [l for l in labels if l.get("name") in a.name]
    if not labels:
        print("no labels to draw")
        return 2
    cloud = read_cloud_xyz(os.path.expanduser(a.pcd), max_points=a.max_points)
    print("%d points, %d box(es)" % (len(cloud), len(labels)))

    keep = np.zeros(len(cloud), bool) if a.around > 0 else np.ones(len(cloud), bool)
    if a.around > 0:
        for box in labels:
            c = np.asarray(box["location"], float)
            near = np.linalg.norm(cloud[:, :2] - c[:2], axis=1) <= a.around
            if a.ceiling > 0:
                near &= cloud[:, 2] <= c[2] + float(box["extent"][2]) + a.ceiling
            keep |= near
    P = cloud[keep]
    colour = np.full((len(P), 3), 150, np.uint8)

    pts, cols = [P], [colour]
    for i, box in enumerate(labels):
        rgb = WHEEL[i % len(WHEEL)]
        inside = points_in_box(P, box, a.margin)
        colour[inside] = rgb                       # the returns this box claims
        e = edges(box, a.step)
        pts.append(e)
        cols.append(np.tile(rgb, (len(e), 1)))
        l, w, h = (2 * float(v) for v in box["extent"])
        print("  %-16s %-13s %5d points inside | %.2f x %.2f x %.2f m | rgb %s"
              % (box.get("name", "?"), box.get("obj_type", "?"), int(inside.sum()),
                 l, w, h, rgb))
    xyz = np.vstack(pts)
    write_ply(os.path.expanduser(a.out), xyz, np.vstack(cols))
    print("wrote %s: %d points (%d of the cloud, %d of edges)"
          % (a.out, len(xyz), len(P), len(xyz) - len(P)))
    print("  grey: the cloud; each box's colour: its edges AND the points it holds.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
