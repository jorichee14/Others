#!/usr/bin/env python3
"""A room's map without what lies outside its walls, and without stray specks.

    python3 scoop_pipeline/processing/crop_room.py <map .pcd/.ply> [<out>]
        [--margin 0.08] [--peak-frac 0.3] [--yaw DEG] [--bounds XMIN XMAX YMIN YMAX]
        [--radius 0.05] [--min-neighbors 6]

e.g. the ceiling-cut copy of a mapping pass (06_cut):
    python3 scoop_pipeline/processing/crop_room.py \\
        <data>/processed/20261103/mapping/mapping/map_final_mapping_20261103_anchored_noceil.pcd
    -> map_final_mapping_20261103_anchored_noceil_room.ply next to it

1. The walls' direction: on a 5 cm copy of the map, each point's normal from
   its neighbours; the vertical, planar ones vote for the room's axes (angles
   folded modulo 90 deg), refined to the yaw at which the walls stack up
   sharpest, so a rectangular room's yaw comes out whatever the map frame's
   heading. --yaw sets it instead. (The 5 cm copy only finds the walls; the
   output keeps every point of the input.)
2. The walls: in that rotated frame, points whose normal runs along x (y)
   are histogrammed along x (y); the outermost peaks holding at least
   --peak-frac of the largest are the room's walls. What the LiDAR saw through
   a door or window (a corridor wall, the outside) is a smaller peak further
   out and is left out. --bounds sets the rectangle instead (in the rotated
   frame the tool prints).
3. The crop: every point inside that rectangle, grown by --margin (m) so the
   walls' own scatter (a few cm) stays.
4. Clean up: a point with fewer than --min-neighbors others within --radius
   (m) is a speck and goes.

The output is in the input's frame (the anchored map's), colours kept;
nothing is rotated. Floor and ceiling are 06_cut's job. Refuses to overwrite.
"""
import argparse
import os
import sys

import numpy as np


def rot_z(deg):
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def voxel_thin(xyz, voxel):
    """One point per voxel (the first)."""
    v = np.floor(xyz / voxel).astype(np.int64)
    _, idx = np.unique(v, axis=0, return_index=True)
    return xyz[np.sort(idx)]


def normals(P, k=16):
    """(normals, planarity) by PCA over each point's k nearest neighbours."""
    from scipy.spatial import cKDTree
    _, nn = cKDTree(P).query(P, k=min(k, len(P)), workers=-1)
    Q = P[nn] - P[nn].mean(1, keepdims=True)
    w, V = np.linalg.eigh(np.einsum("nki,nkj->nij", Q, Q))
    planar = (w[:, 1] - w[:, 0]) / np.maximum(w[:, 2], 1e-12)
    return V[:, :, 0], planar


def wall_frame(xyz, voxel=0.05):
    """(yaw deg, thinned points rotated by -yaw, their normals, wall mask): the
    room's axes from its vertical planar surfaces."""
    P = voxel_thin(xyz, voxel)
    lo, hi = np.percentile(P[:, 2], [10, 90])          # walls, not floor / ceiling
    P = P[(P[:, 2] > lo) & (P[:, 2] < hi)]
    n, planar = normals(P)
    wall = (np.abs(n[:, 2]) < 0.15) & (planar > 0.5)
    if wall.sum() < 50:
        raise SystemExit("too few wall points to find the room's axes; give --yaw")
    th = np.arctan2(n[wall, 1], n[wall, 0])
    yaw = np.degrees(np.angle(np.exp(4j * th).mean()) / 4.0)       # (-45, 45]
    yaw = refine_yaw(P[wall], yaw)
    R = rot_z(-yaw)
    return yaw, P @ R.T, n @ R.T, wall


def refine_yaw(W, yaw, span=3.0, step=0.05, bin_m=0.02):
    """The yaw within +-span deg of the normals' estimate at which the wall
    points stack up sharpest along both axes (sum of squared 2 cm histogram
    counts): the normals' vote is a few tenths of a degree off on its own."""
    best, best_yaw = -1.0, yaw
    for y in np.arange(yaw - span, yaw + span + step / 2, step):
        Wr = W[:, :2] @ rot_z(-y)[:2, :2].T
        score = 0.0
        for ax in (0, 1):
            h = np.bincount(np.floor((Wr[:, ax] - Wr[:, ax].min()) / bin_m).astype(np.int64))
            score += float((h.astype(np.float64) ** 2).sum())
        if score > best:
            best, best_yaw = score, float(y)
    return best_yaw


def wall_peaks(coord, frac, bin_m=0.02):
    """Peaks of a histogram of wall points along one axis: [(position, count)],
    and the outermost two holding >= frac of the largest."""
    edges = np.arange(coord.min() - bin_m, coord.max() + 2 * bin_m, bin_m)
    h, edges = np.histogram(coord, edges)
    hs = np.convolve(h, np.ones(5) / 5.0, mode="same")
    c = 0.5 * (edges[:-1] + edges[1:])
    pk = [(float(c[i]), float(hs[i])) for i in range(1, len(hs) - 1)
          if hs[i] >= hs[i - 1] and hs[i] > hs[i + 1] and hs[i] > 0]
    if not pk:
        raise SystemExit("no wall found along an axis; give --bounds")
    top = max(p[1] for p in pk)
    strong = [p for p in pk if p[1] >= frac * top]
    return pk, strong[0][0], strong[-1][0]


def room_bounds(xyz, frac=0.3, yaw=None):
    """(yaw, (xmin, xmax, ymin, ymax) in the frame rotated by -yaw, report lines)."""
    y_est, Pr, nr, wall = wall_frame(xyz)
    if yaw is None:
        yaw = y_est
    else:                                               # a given yaw: re-rotate
        R = rot_z(-(yaw - y_est))
        Pr, nr = Pr @ R.T, nr @ R.T
    lines = ["walls' yaw %.2f deg%s" % (yaw, "" if yaw == y_est else " (given; found %.2f)" % y_est)]
    out = []
    for ax, name in ((0, "x"), (1, "y")):
        sel = wall & (np.abs(nr[:, ax]) > 0.9)
        if sel.sum() < 20:
            raise SystemExit("too few walls facing %s; give --bounds" % name)
        pk, a, b = wall_peaks(Pr[sel, ax], frac)
        top = max(p[1] for p in pk)
        big = sorted(pk, key=lambda p: -p[1])[:6]
        lines.append("  walls along %s: %.2f .. %.2f m  (largest peaks: %s)" % (
            name, a, b, ", ".join("%.2f (%d%%)" % (p[0], 100 * p[1] / top)
                                  for p in sorted(big))))
        out += [a, b]
    return yaw, tuple(out), lines


def crop_mask(xyz, yaw, bounds, margin):
    xr = xyz @ rot_z(-yaw).T
    x0, x1, y0, y1 = bounds
    return ((xr[:, 0] >= x0 - margin) & (xr[:, 0] <= x1 + margin)
            & (xr[:, 1] >= y0 - margin) & (xr[:, 1] <= y1 + margin))


def speck_mask(xyz, radius, min_neighbors):
    """True where a point has at least min_neighbors others within radius."""
    from scipy.spatial import cKDTree
    n = cKDTree(xyz).query_ball_point(xyz, radius, return_length=True, workers=-1)
    return n - 1 >= min_neighbors


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__.split("\n\n", 1)[1])
    ap.add_argument("src")
    ap.add_argument("out", nargs="?")
    ap.add_argument("--margin", type=float, default=0.08)
    ap.add_argument("--peak-frac", type=float, default=0.3)
    ap.add_argument("--yaw", type=float, default=None)
    ap.add_argument("--bounds", type=float, nargs=4, default=None,
                    metavar=("XMIN", "XMAX", "YMIN", "YMAX"))
    ap.add_argument("--radius", type=float, default=0.05)
    ap.add_argument("--min-neighbors", type=int, default=6)
    a = ap.parse_args(argv)
    import open3d as o3d
    src = os.path.expanduser(a.src)
    out = os.path.expanduser(a.out) if a.out else os.path.splitext(src)[0] + "_room.ply"
    if os.path.exists(out):
        raise SystemExit("%s exists; remove it or name another output" % out)
    pcd = o3d.io.read_point_cloud(src)
    xyz = np.asarray(pcd.points)
    if not len(xyz):
        raise SystemExit("no points in %s" % src)
    print("%s: %d points" % (src, len(xyz)))
    if a.bounds is not None:
        yaw = a.yaw if a.yaw is not None else 0.0
        bounds = tuple(a.bounds)
        print("walls given: yaw %.2f deg, x %.2f .. %.2f, y %.2f .. %.2f m" % ((yaw,) + bounds))
    else:
        yaw, bounds, lines = room_bounds(xyz, a.peak_frac, a.yaw)
        print("\n".join(lines))
    inside = crop_mask(xyz, yaw, bounds, a.margin)
    print("crop (+%.2f m): kept %d, dropped %d outside the walls"
          % (a.margin, inside.sum(), (~inside).sum()))
    idx = np.flatnonzero(inside)
    ok = speck_mask(xyz[idx], a.radius, a.min_neighbors)
    print("clean: dropped %d specks (< %d neighbours within %.2f m)"
          % ((~ok).sum(), a.min_neighbors, a.radius))
    keep = pcd.select_by_index(idx[ok].tolist())
    o3d.io.write_point_cloud(out, keep)
    print("wrote %s (%d points, colours kept, same frame as the input)" % (out, len(keep.points)))
    print("rerun with --yaw %.2f --bounds %.2f %.2f %.2f %.2f to adjust" % ((yaw,) + bounds))


if __name__ == "__main__":
    sys.exit(main())
