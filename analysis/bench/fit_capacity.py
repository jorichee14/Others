#!/usr/bin/env python3
"""Predict link capacity from the robot's own LiDAR map.

The question this answers: indoors, the structures that attenuate Wi-Fi are the
same ones the robot already maps for navigation. Is what it has mapped enough
to say how much bandwidth it will have, at a place it has never measured?

The model is the established indoor one -- COST 231 multi-wall -- fitted from
geometry instead of a survey:

    capacity_dBish = A - 10 n log10(d) - sum_i k_i L_i

where d is the robot-to-AP distance, k_i the number of obstructions of class i
that the straight ray crosses, and L_i their fitted attenuation. The classes
come from the map: a "wall" is a run of occupied voxels the ray passes
through, and its thickness is how far the ray travels inside it. Everything
here is read off the anchored point cloud, so it transfers to a room where no
radio measurement was ever taken.

    python bench/fit_capacity.py --selftest
    python bench/fit_capacity.py --rows rows.parquet --map-dir maps/ --ap ap.json

`rows.parquet` is one row per (agent, second): area, agent, t_s, x, y, z and
the measured capacity_mbps. Splits are by area, never by row: a model fitted
where it is tested proves nothing.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

FEATURES = ["log_d", "n_walls", "wall_path_m", "min_clearance_m", "dz"]


def occupancy_grid(points: np.ndarray, voxel: float = 0.10):
    """Occupied voxel set of a point cloud, as a dict for O(1) lookup."""
    keys = np.floor(points / voxel).astype(np.int64)
    return set(map(tuple, keys)), voxel


def ray_features(grid, voxel: float, p0: np.ndarray, p1: np.ndarray, step_frac: float = 0.5):
    """Geometry of the straight path from p0 (robot) to p1 (AP).

    Returns the features a propagation model needs and a map can supply:
    distance, how many separate obstructions the ray enters, how far it travels
    inside them, and how close it passes to the nearest occupied voxel off-axis
    (a crude first-Fresnel-zone clearance). A ray that grazes a wall is not the
    same as one with a clear metre either side, and the radio knows it."""
    d = float(np.linalg.norm(p1 - p0))
    if d <= 0:
        return dict(log_d=0.0, n_walls=0.0, wall_path_m=0.0, min_clearance_m=0.0, dz=0.0, d_m=0.0)
    step = voxel * step_frac
    n = max(int(d / step), 1)
    ts = np.linspace(0.0, 1.0, n + 1)[:, None]
    pts = p0[None, :] + ts * (p1 - p0)[None, :]
    keys = np.floor(pts / voxel).astype(np.int64)
    occ = np.array([tuple(k) in grid for k in keys])
    # a wall is a run of occupied samples; count the runs, not the samples
    entries = int(np.sum(occ[1:] & ~occ[:-1])) + int(occ[0])
    wall_path = float(occ.sum() * step)
    # clearance: nearest occupied voxel to the free part of the ray, probed on a
    # small perpendicular cross; cheap stand-in for the Fresnel radius
    free = pts[~occ]
    clearance = 3.0
    if len(free):
        axis = (p1 - p0) / d
        perp = np.cross(axis, [0.0, 0.0, 1.0])
        if np.linalg.norm(perp) < 1e-6:
            perp = np.array([1.0, 0.0, 0.0])
        perp /= np.linalg.norm(perp)
        probe = free[:: max(len(free) // 64, 1)]
        for r in np.arange(voxel, 3.0, 4 * voxel):
            hit = False
            for sgn in (1, -1):
                q = probe + sgn * r * perp[None, :]
                k = np.floor(q / voxel).astype(np.int64)
                if any(tuple(x) in grid for x in k):
                    hit = True
                    break
            if hit:
                clearance = float(r)
                break
    return dict(log_d=float(np.log10(max(d, 1e-3))), n_walls=float(entries),
                wall_path_m=wall_path, min_clearance_m=clearance,
                dz=float(abs(p1[2] - p0[2])), d_m=d)


def build_features(rows: pd.DataFrame, grids: dict, aps: dict) -> pd.DataFrame:
    out = []
    for area, g in rows.groupby("area"):
        grid, voxel = grids[area]
        ap = np.asarray(aps[area], float)
        for r in g.itertuples():
            f = ray_features(grid, voxel, np.array([r.x, r.y, r.z], float), ap)
            f.update(area=area, agent=r.agent, t_s=r.t_s, capacity_mbps=r.capacity_mbps)
            out.append(f)
    return pd.DataFrame(out)


def fit_linear(X: np.ndarray, y: np.ndarray):
    """Least squares with an intercept -- the multi-wall model is linear in its
    terms, so this is the model, not an approximation of it."""
    A = np.column_stack([np.ones(len(X)), X])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return coef


def predict_linear(coef: np.ndarray, X: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(len(X)), X]) @ coef


def evaluate(feat: pd.DataFrame, feature_sets: dict) -> pd.DataFrame:
    """Leave-one-area-out: fit on the other areas, predict the held-out one."""
    rows = []
    areas = sorted(feat["area"].unique())
    for held in areas:
        tr, te = feat[feat["area"] != held], feat[feat["area"] == held]
        if not len(tr) or not len(te):
            continue
        y_te = te["capacity_mbps"].to_numpy()
        for name, cols in feature_sets.items():
            if name == "constant":
                pred = np.full(len(te), tr["capacity_mbps"].mean())
            else:
                coef = fit_linear(tr[cols].to_numpy(), tr["capacity_mbps"].to_numpy())
                pred = predict_linear(coef, te[cols].to_numpy())
            rows.append({"held_out_area": held, "estimator": name,
                         "mae_mbps": float(np.abs(pred - y_te).mean()),
                         "rel_err": float(np.abs(pred - y_te).mean() / max(y_te.mean(), 1e-9)),
                         "n_test": len(te)})
    return pd.DataFrame(rows)


ESTIMATORS = {
    "constant": [],
    "distance only": ["log_d"],
    "distance + walls": ["log_d", "n_walls"],
    "full geometry": FEATURES,
}


def selftest() -> int:
    """A synthetic building where the truth is known.

    Three rooms, each with walls in different places, an AP in a corner, and a
    capacity that follows a multi-wall law. If the fit cannot recover a law it
    was given, it will not find one that is only approximately true."""
    rng = np.random.default_rng(0)
    grids, aps, rows = {}, {}, []
    for ai, area in enumerate(["area1", "area2", "area3"]):
        pts = []
        for wx in (3.0 + ai, 7.0):                    # two walls, placed differently per area
            ys = np.arange(0, 12, 0.05)
            zs = np.arange(0, 2.5, 0.05)
            Y, Z = np.meshgrid(ys, zs)
            w = np.column_stack([np.full(Y.size, wx), Y.ravel(), Z.ravel()])
            pts.append(w)
        P = np.vstack(pts)
        grids[area] = occupancy_grid(P, 0.10)
        aps[area] = [0.5, 6.0, 2.0]
        for i in range(400):
            p = np.array([rng.uniform(0.5, 11.0), rng.uniform(0.5, 11.5), 0.6])
            f = ray_features(*grids[area], p, np.asarray(aps[area], float))
            cap = 95.0 - 22.0 * f["log_d"] - 14.0 * f["n_walls"] + rng.normal(0, 2.0)
            rows.append({"area": area, "agent": f"r{i%2+1}", "t_s": float(i),
                         "x": p[0], "y": p[1], "z": p[2], "capacity_mbps": max(cap, 1.0)})
    rows = pd.DataFrame(rows)
    feat = build_features(rows, grids, aps)
    res = evaluate(feat, ESTIMATORS)
    table = res.groupby("estimator")["mae_mbps"].mean().sort_values()
    print("\nleave-one-area-out MAE (Mbps), synthetic building with a known law")
    print(table.to_string())
    ok = table["distance + walls"] < table["distance only"] < table["constant"]
    print(f"\n{'ok  ' if ok else 'FAIL'}  walls beat distance beat constant")
    print(f"{'ok  ' if table.min() < 4.0 else 'FAIL'}  best MAE {table.min():.2f} < 4.0 Mbps "
          f"(noise floor is 2.0)")
    return 0 if ok and table.min() < 4.0 else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--rows", type=Path, help="parquet: area, agent, t_s, x, y, z, capacity_mbps")
    ap.add_argument("--map-dir", type=Path, help="one <area>.pcd per area")
    ap.add_argument("--ap", type=Path, help='json: {"area1": [x, y, z], ...}')
    ap.add_argument("--voxel", type=float, default=0.10)
    ap.add_argument("--out", type=Path, default=Path("results/capacity"))
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    if not (args.rows and args.map_dir and args.ap):
        raise SystemExit("need --rows, --map-dir and --ap (or --selftest)")

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from common import read_pcd_xy  # noqa: PLC0415  (kept for the 2-D fallback)

    rows = pd.read_parquet(args.rows)
    aps = json.loads(args.ap.read_text())
    grids = {}
    for area in sorted(rows["area"].unique()):
        try:
            import open3d as o3d  # noqa: PLC0415
            P = np.asarray(o3d.io.read_point_cloud(str(args.map_dir / f"{area}.pcd")).points)
        except Exception:
            xy = read_pcd_xy(args.map_dir / f"{area}.pcd")
            if xy is None:
                raise SystemExit(f"cannot read {args.map_dir / f'{area}.pcd'}; install open3d")
            P = np.column_stack([xy, np.zeros(len(xy))])
        grids[area] = occupancy_grid(P, args.voxel)

    feat = build_features(rows, grids, aps)
    res = evaluate(feat, ESTIMATORS)
    args.out.mkdir(parents=True, exist_ok=True)
    feat.to_parquet(args.out / "capacity_features.parquet", index=False)
    res.to_csv(args.out / "capacity_baselines.csv", index=False)
    print(res.pivot_table(index="estimator", columns="held_out_area",
                          values="mae_mbps").round(2).to_string())
    print(f"\nwrote {args.out}/capacity_features.parquet, capacity_baselines.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
