#!/usr/bin/env python3
"""How far each trained surfel moved from the LiDAR point it started on.

    python3 scoop_pipeline/nest/surfel_drift.py <scene>/sparse/0/points3D.ply \\
        <model>/point_cloud/iteration_N/point_cloud.ply [--out drift.ply] [--far 0.10]

Per trained surfel: the distance to the nearest starting point (the scene's
points3D, sampled from the cleaned mesh). The starting points are the map,
accurate to a few centimetres, so a surfel more than --far from every one of
them has left the surface: a floater the training made, not the LiDAR. The
histogram says how many; drift.ply holds the surfels that moved, coloured by
distance (yellow 10 cm, red 1 m and more), to open with the map and see where
they gather (in front of glass and windows, where no LiDAR depth holds them).
"""
import argparse
import os

import numpy as np
from scipy.spatial import cKDTree

TYPES = {"char": "i1", "uchar": "u1", "int8": "i1", "uint8": "u1", "short": "<i2", "ushort": "<u2",
         "int16": "<i2", "uint16": "<u2", "int": "<i4", "uint": "<u4", "int32": "<i4", "uint32": "<u4",
         "float": "<f4", "float32": "<f4", "double": "<f8", "float64": "<f8"}


def read_xyz(path, with_normals=False):
    """x y z of a PLY's vertices (binary little endian or ascii; other properties skipped);
    with_normals: also nx ny nz, or None when the file has none."""
    with open(path, "rb") as fh:
        header, props, n, fmt, in_vertex = [], [], 0, "", False
        while True:
            line = fh.readline().decode("ascii", "ignore").strip()
            header.append(line)
            w = line.split()
            if not w:
                continue
            if w[0] == "format":
                fmt = w[1]
            elif w[0] == "element":
                in_vertex = w[1] == "vertex"
                if in_vertex:
                    n = int(w[2])
            elif w[0] == "property" and in_vertex:
                if w[1] == "list":
                    raise SystemExit(f"{path}: list property in vertex element")
                props.append((w[2], TYPES[w[1]]))
            elif w[0] == "end_header":
                break
        if fmt == "ascii":
            a = np.loadtxt(fh, max_rows=n, ndmin=2)
            names = [p for p, _ in props]
            xyz = a[:, [names.index("x"), names.index("y"), names.index("z")]].astype(np.float64)
            if not with_normals:
                return xyz
            N = a[:, [names.index(k) for k in ("nx", "ny", "nz")]] if "nx" in names else None
            return xyz, N
        if fmt != "binary_little_endian":
            raise SystemExit(f"{path}: {fmt} not handled")
        rec = np.frombuffer(fh.read(n * np.dtype(props).itemsize), dtype=np.dtype(props), count=n)
    xyz = np.stack([rec["x"], rec["y"], rec["z"]], 1).astype(np.float64)
    if not with_normals:
        return xyz
    names = rec.dtype.names
    N = np.stack([rec[k] for k in ("nx", "ny", "nz")], 1).astype(np.float64) if "nx" in names else None
    return xyz, N


def write_coloured(path, P, rgb):
    v = np.empty(len(P), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    v["x"], v["y"], v["z"] = P[:, 0], P[:, 1], P[:, 2]
    v["red"], v["green"], v["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    with open(path, "wb") as fh:
        fh.write((f"ply\nformat binary_little_endian 1.0\nelement vertex {len(P)}\n"
                  "property float x\nproperty float y\nproperty float z\n"
                  "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n").encode())
        fh.write(v.tobytes())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("start", help="<scene>/sparse/0/points3D.ply")
    ap.add_argument("trained", help="<model>/point_cloud/iteration_N/point_cloud.ply")
    ap.add_argument("--out", default=None, help="the moved surfels, coloured (default beside the trained ply)")
    ap.add_argument("--far", type=float, default=0.10, help="metres from every starting point to count as moved")
    a = ap.parse_args()
    S, N = read_xyz(os.path.expanduser(a.start), with_normals=True)
    T = read_xyz(os.path.expanduser(a.trained))
    print(f"{len(S)} starting points, {len(T)} trained surfels")
    if len(S) == len(T):
        # nothing grown or pruned: surfel k started on point k, so the move is exact
        D = T - S
        d = np.linalg.norm(D, axis=1)
        print("same count: each surfel against its own starting point")
        if N is not None and np.isfinite(N).all() and (np.linalg.norm(N, axis=1) > 0.5).mean() > 0.9:
            n = N / np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-9)
            off = np.abs((D * n).sum(1))                    # off the surface: thickening, floaters
            along = np.linalg.norm(D - (D * n).sum(1, keepdims=True) * n, axis=1)   # sliding along it
            for name, v in (("off the surface (along its normal)", off), ("along the surface", along)):
                print(f"  {name}: median {np.median(v) * 100:.1f} cm, 90% {np.percentile(v, 90) * 100:.1f} cm, "
                      f"99% {np.percentile(v, 99) * 100:.1f} cm, over 10 cm {100 * np.mean(v > 0.10):.2f}%")
            d = off                                         # what makes a floater is leaving the surface
            print("the histogram and drift.ply below are the move OFF the surface")
    else:
        d, _ = cKDTree(S).query(T, k=1, workers=-1)
        print("different counts: each surfel against the nearest starting point")
    edges = [0, 0.01, 0.02, 0.05, 0.10, 0.25, 0.5, 1.0, 2.0, np.inf]
    print("distance from the nearest starting point:")
    for lo, hi in zip(edges[:-1], edges[1:]):
        k = int(((d >= lo) & (d < hi)).sum())
        print(f"  {lo * 100:6.0f} .. {hi * 100 if np.isfinite(hi) else float('inf'):>6} cm  {k:9d}  {100 * k / len(d):6.2f}%")
    moved = d >= a.far
    print(f"moved {a.far * 100:.0f} cm or more: {int(moved.sum())} surfels ({100 * moved.mean():.2f}%), "
          f"median of those {np.median(d[moved]) if moved.any() else 0:.2f} m")
    out = a.out or os.path.join(os.path.dirname(os.path.expanduser(a.trained)), "drift.ply")
    if moved.any():
        t = np.clip((np.log10(d[moved]) - np.log10(a.far)) / (0 - np.log10(a.far)), 0, 1)   # far .. 1 m
        rgb = np.stack([np.full_like(t, 255), 255 * (1 - t), np.zeros_like(t)], 1).astype(np.uint8)
        write_coloured(out, T[moved], rgb)
        print(f"wrote {out}: open it with the map; yellow moved {a.far * 100:.0f} cm, red 1 m or more")


if __name__ == "__main__":
    main()
