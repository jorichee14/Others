#!/usr/bin/env python3
"""From how many directions was each patch of the building photographed?

A splat is constrained only along the directions its surface was seen from.
A forward-facing pass sees every wall from one direction; the model is then
free along every other, and from any of them it streaks. This measures that
directly, per patch of the LiDAR surface and per capture session: which
viewing directions actually reached it, occlusion included.

    python3 scoop_pipeline/datasets/view_coverage.py <scene> [--out coverage]
        [--stride 10] [--sessions 1:oct_zed,2:sep_realsense]

The scene is an LOD-3DGS one (sparse/0, depths/): the surface is its
points3D.ply, the cameras are its images.txt, and the depth map beside each
image -- the LiDAR map projected into that view -- says whether a point was
in front of the wall or behind it. Sessions are told apart by camera id,
which is how the scene builder tells two exports apart.

Directions are binned on the sphere (12 in azimuth x 3 in elevation), and
two things are read off the bins that reached a patch: its SPREAD, the
widest angle between any two of them -- under 30 degrees is one direction
whatever the count, since a camera sliding past a wall fans only a little
-- and how HEAD-ON it was ever seen, the smallest angle any of them makes
with the surface normal. A corridor pass scores wide on spread yet never
head-on: every view is a grazing one from the same side. Writes a summary,
a json, and a PLY coloured by spread for each session and for all of them
together, so the gain from a second robot is a picture as well as a number.
"""
import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from virtual_views import read_images                           # noqa: E402

AZIMUTH_BINS, ELEVATION_BINS = 12, 3


def read_cameras(path):
    cams = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            v = line.split()
            if len(v) >= 8 and v[0].isdigit():
                cams[int(v[0])] = (int(v[2]), int(v[3])) + tuple(float(x) for x in v[4:8])
    return cams


def read_points(path, stride):
    """x y z of a PLY: the scene builder's binary layout, or anything open3d reads."""
    with open(path, "rb") as handle:
        header, props = [], []
        while True:
            line = handle.readline().decode("ascii", "replace").strip()
            header.append(line)
            if line.startswith("property"):
                props.append(line.split()[1:])
            if line == "end_header":
                break
        n = int(next(h for h in header if h.startswith("element vertex")).split()[-1])
        if "format binary_little_endian 1.0" in header and props[0][1] == "x":
            kinds = {"float": "<f4", "uchar": "u1", "double": "<f8", "int": "<i4", "uint": "<u4"}
            dtype = np.dtype([(name, kinds[kind]) for kind, name in props])
            raw = np.frombuffer(handle.read(n * dtype.itemsize), dtype=dtype, count=n)
            return np.column_stack([raw["x"], raw["y"], raw["z"]]).astype(np.float64)[::stride]
    import open3d as o3d
    return np.asarray(o3d.io.read_point_cloud(path).points, dtype=np.float64)[::stride]


def unwarp(grey):
    """The scene builder's depth encoding back to metres."""
    w = grey.astype(np.float64) / 255.0
    return np.where(w < 0.5, 20.0 * w, 5.0 / np.maximum(1.0 - w, 1e-6))


def direction_bin(d):
    """Which of the sphere's bins a unit direction (toward the camera) falls in."""
    azimuth = np.arctan2(d[:, 1], d[:, 0])
    elevation = np.arcsin(np.clip(d[:, 2], -1, 1))
    a = ((azimuth + np.pi) / (2 * np.pi) * AZIMUTH_BINS).astype(int) % AZIMUTH_BINS
    e = np.clip(((elevation + np.pi / 2) / np.pi * ELEVATION_BINS).astype(int), 0, ELEVATION_BINS - 1)
    return e * AZIMUTH_BINS + a


def coverage_of(points, cams, frames, scene, log=print):
    """Per point, a bitmask of the direction bins that saw it, per camera id."""
    masks = {}
    ones = np.ones((len(points), 1))
    homogeneous = np.hstack([points, ones])
    for n, (name, cid, c2w) in enumerate(frames):
        W, H, fx, fy, cx, cy = cams[cid]
        depth_png = cv2.imread(os.path.join(scene, "depths", os.path.splitext(name)[0] + ".png"),
                               cv2.IMREAD_UNCHANGED)
        if depth_png is None:
            continue
        depth = unwarp(depth_png[:, :, 0])
        valid = depth_png[:, :, 3] > 0 if depth_png.ndim == 3 and depth_png.shape[2] == 4 else depth > 0
        w2c = np.linalg.inv(c2w)
        local = homogeneous @ w2c.T
        z = local[:, 2]
        front = z > 0.1
        u = np.full(len(points), -1.0)
        v = np.full(len(points), -1.0)
        u[front] = fx * local[front, 0] / z[front] + cx
        v[front] = fy * local[front, 1] / z[front] + cy
        inside = front & (u >= 0) & (u < W - 1) & (v >= 0) & (v < H - 1)
        ui, vi = u[inside].astype(int), v[inside].astype(int)
        seen_depth = depth[vi, ui]
        ok = valid[vi, ui] & (np.abs(z[inside] - seen_depth) <= 0.06 + 0.02 * z[inside])
        idx = np.flatnonzero(inside)[ok]
        if len(idx):
            toward = c2w[:3, 3] - points[idx]
            toward /= np.linalg.norm(toward, axis=1, keepdims=True)
            bins = direction_bin(toward)
            mask = masks.setdefault(cid, np.zeros(len(points), dtype=np.uint64))
            mask[idx] |= (np.uint64(1) << bins.astype(np.uint64))
        if n % 500 == 0:
            log(f"  {n}/{len(frames)} frames")
    return masks


def bin_centres():
    """The unit direction at the middle of each bin, in the bins' own order."""
    e = (np.arange(ELEVATION_BINS) + 0.5) / ELEVATION_BINS * np.pi - np.pi / 2
    a = (np.arange(AZIMUTH_BINS) + 0.5) / AZIMUTH_BINS * 2 * np.pi - np.pi
    el, az = np.meshgrid(e, a, indexing="ij")
    return np.column_stack([np.cos(el.ravel()) * np.cos(az.ravel()),
                            np.cos(el.ravel()) * np.sin(az.ravel()), np.sin(el.ravel())])


def bits_of(mask):
    """(N, bins) boolean: which bins each point's mask holds."""
    total = AZIMUTH_BINS * ELEVATION_BINS
    return ((mask[:, None] >> np.arange(total, dtype=np.uint64)) & np.uint64(1)).astype(bool)


def spread_of(mask):
    """The widest angle (degrees) between any two directions that saw each point."""
    bits = bits_of(mask)
    centres = bin_centres()
    pairwise = np.degrees(np.arccos(np.clip(centres @ centres.T, -1, 1)))
    widest = np.zeros(len(mask))
    for i in range(len(centres)):
        has = bits[:, i]
        if not has.any():
            continue
        # for points holding bin i, the farthest other bin they also hold
        others = bits[has] * pairwise[i][None, :]
        widest[has] = np.maximum(widest[has], others.max(axis=1))
    return widest


def head_on_of(mask, normals):
    """The smallest angle (degrees) between a seeing direction and the surface normal."""
    bits = bits_of(mask)
    centres = bin_centres()
    angle = np.degrees(np.arccos(np.clip(np.abs(normals @ centres.T), 0, 1)))   # (N, bins)
    angle[~bits] = 180.0
    best = angle.min(axis=1)
    best[~bits.any(axis=1)] = np.nan
    return best


def normals_of(points, k=16):
    """A normal per point from the plane through its neighbours (sign unfixed)."""
    from scipy.spatial import cKDTree
    _, idx = cKDTree(points).query(points, k=k, workers=-1)
    normals = np.zeros_like(points)
    for start in range(0, len(points), 50000):
        block = idx[start:start + 50000]
        nb = points[block] - points[block].mean(axis=1, keepdims=True)
        cov = np.einsum("nki,nkj->nij", nb, nb)
        _, vecs = np.linalg.eigh(cov)
        normals[start:start + 50000] = vecs[:, :, 0]
    return normals


def popcount(mask):
    counts = np.zeros(len(mask), dtype=np.int32)
    m = mask.copy()
    while m.any():
        counts += (m & np.uint64(1)).astype(np.int32)
        m >>= np.uint64(1)
    return counts


def write_coloured(path, points, seen, spread):
    """A PLY coloured by spread: unseen grey, one direction red, a fan amber, wide green."""
    colours = np.full((len(points), 3), 160, dtype=np.uint8)
    colours[seen & (spread < 30)] = (220, 60, 50)
    colours[seen & (spread >= 30) & (spread < 90)] = (240, 170, 40)
    colours[seen & (spread >= 90)] = (60, 170, 80)
    dt = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    rows = np.zeros(len(points), dt)
    rows["x"], rows["y"], rows["z"] = points.T.astype(np.float32)
    rows["red"], rows["green"], rows["blue"] = colours.T
    with open(path, "wb") as handle:
        handle.write(("ply\nformat binary_little_endian 1.0\n"
                      f"element vertex {len(points)}\n"
                      "property float x\nproperty float y\nproperty float z\n"
                      "property uchar red\nproperty uchar green\nproperty uchar blue\n"
                      "end_header\n").encode("ascii"))
        handle.write(rows.tobytes())


def report(label, mask, normals, log=print):
    seen = mask != 0
    spread, head_on = spread_of(mask), head_on_of(mask, normals)
    sp, ho = spread[seen], head_on[seen]
    if seen.any():
        log(f"  {label:<14} seen {100 * seen.mean():5.1f}%   spread: one direction (<30) "
            f"{100 * (sp < 30).mean():5.1f}%, fan (30-90) {100 * ((sp >= 30) & (sp < 90)).mean():5.1f}%, "
            f"wide (>90) {100 * (sp >= 90).mean():5.1f}%   ever head-on (<30 to normal) "
            f"{100 * (ho < 30).mean():5.1f}%")
    else:
        log(f"  {label:<14} seen   0.0%")
    return {"label": label, "seen": float(seen.mean()),
            "one_direction": float((sp < 30).mean()) if seen.any() else 0.0,
            "fan": float(((sp >= 30) & (sp < 90)).mean()) if seen.any() else 0.0,
            "wide": float((sp >= 90).mean()) if seen.any() else 0.0,
            "head_on": float((ho < 30).mean()) if seen.any() else 0.0,
            "mean_spread_deg": float(sp.mean()) if seen.any() else 0.0}, seen, spread


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scene")
    ap.add_argument("--out", default=None, help="prefix for the outputs (default <scene>/coverage)")
    ap.add_argument("--stride", type=int, default=10, help="every Nth surface point")
    ap.add_argument("--sessions", default="", help="camera id -> name, e.g. 1:oct_zed,2:sep_realsense")
    a = ap.parse_args()

    scene = os.path.expanduser(a.scene)
    out = os.path.expanduser(a.out) if a.out else os.path.join(scene, "coverage")
    names = {int(k): v for k, v in (p.split(":") for p in a.sessions.split(",") if ":" in p)}
    cams = read_cameras(os.path.join(scene, "sparse", "0", "cameras.txt"))
    frames = read_images(os.path.join(scene, "sparse", "0", "images.txt"))
    frames = [(name, cid, c2w) for _, cid, name, c2w in frames]
    points = read_points(os.path.join(scene, "sparse", "0", "points3D.ply"), a.stride)
    print(f"{len(points):,} surface points (every {a.stride}th), {len(frames)} cameras, "
          f"{len(cams)} camera model(s)")

    masks = coverage_of(points, cams, frames, scene)
    normals = normals_of(points)
    print("\nhow each surface point was seen:")
    results = []
    union = np.zeros(len(points), dtype=np.uint64)
    first_seen = first_spread = None
    for cid in sorted(masks):
        label = names.get(cid, f"camera_{cid}")
        summary, seen, spread = report(label, masks[cid], normals)
        results.append(summary)
        write_coloured(f"{out}_{label}.ply", points, seen, spread)
        if first_seen is None:
            first_seen, first_spread = seen, spread
        union |= masks[cid]
    if len(masks) > 1:
        summary, seen, spread = report("all sessions", union, normals)
        results.append(summary)
        write_coloured(f"{out}_all.ply", points, seen, spread)
        new_points = (seen & ~first_seen).mean()
        widened = ((spread - first_spread) >= 30)[first_seen].mean()
        print(f"\n  beyond the first session: {100 * new_points:.1f}% of the surface newly seen, "
              f"and of what it had seen, {100 * widened:.1f}% now seen from a direction "
              f"at least 30 degrees away")
    import json
    with open(f"{out}.json", "w", encoding="utf-8") as handle:
        json.dump({"scene": scene, "stride": a.stride, "bins": AZIMUTH_BINS * ELEVATION_BINS,
                   "sessions": results}, handle, indent=1)
    print(f"\nwrote {out}.json and {out}_<session>.ply (grey unseen, red one direction, "
          "amber a fan under 90 degrees, green wider)")


if __name__ == "__main__":
    main()
