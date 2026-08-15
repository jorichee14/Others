#!/usr/bin/env python3
"""
Why are there no walls? Answer it in minutes, without re-running detect.

Two causes produce the identical `0 wall` line and want opposite responses:

  PLANE BUDGET   walls exist in the cloud, but RANSAC peels largest-first and
                 floors, ceilings and table tops consumed every max_planes slot
                 before one was reached. Fix: raise structure.max_planes.

  NO RETURNS     the walls are glass, polished or simply never scanned, so
                 there is nothing to fit. No setting recovers this -- where the
                 LiDAR never measured the surface, nothing can.

This peels far more planes than a normal run would and reports where the first
wall appears in the ranking. A wall at rank 40 means the budget; no wall at any
rank means the surface is not in the data.

Runs on a downsampled copy: plane orientation is determined to well under the
sensor noise by an 8 cm sample, and a 187 M-point cloud is otherwise minutes of
RANSAC per plane.

    python3 diagnose_structure.py pipeline_config.json [cloud.pcd]
        [--planes 150] [--voxel 0.08] [--dist 0.06] [--tol-deg 15]
"""

import argparse
import os
import sys
import numpy as np
import open3d as o3d

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import pipeline_detect as pdet          # noqa: E402
from pipeline_common import load_pipeline   # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", nargs="?", default="pipeline_config.json")
    ap.add_argument("cloud", nargs="?", default=None,
                    help="default: the colorize output, else denoised.pcd")
    ap.add_argument("--planes", type=int, default=150)
    ap.add_argument("--voxel", type=float, default=0.08)
    ap.add_argument("--dist", type=float, default=None)
    ap.add_argument("--tol-deg", type=float, default=None)
    ap.add_argument("--min-wall-height", type=float, default=None)
    ap.add_argument("--min-wall-area", type=float, default=None)
    a = ap.parse_args()

    P = load_pipeline(a.config)
    s = P.stage("01_build_map")
    st = s.get("detect", {}).get("structure", {})
    dist = a.dist if a.dist is not None else float(st.get("dist", 0.04))
    tol = a.tol_deg if a.tol_deg is not None else float(st.get("normal_tol_deg", 15.0))
    minh = (a.min_wall_height if a.min_wall_height is not None
            else float(st.get("min_wall_height", 0.8)))
    mina = (a.min_wall_area if a.min_wall_area is not None
            else float(st.get("min_wall_area", 2.0)))

    path = a.cloud
    if path is None:
        for cand in (s["colorize"].get("output", "colored.pcd"),
                     "denoised.pcd", "static.pcd", "merged.pcd"):
            if os.path.exists(P.outp(cand)):
                path = P.outp(cand)
                break
    if path is None or not os.path.exists(path):
        raise SystemExit("no cloud found; pass one explicitly")

    print(f"reading {path}")
    pcd = o3d.io.read_point_cloud(path)
    n0 = len(pcd.points)
    if n0 == 0:
        raise SystemExit("empty cloud")
    small = pcd.voxel_down_sample(a.voxel) if a.voxel > 0 else pcd
    pts = np.asarray(small.points)
    del pcd
    lo, hi = pts.min(0), pts.max(0)
    print(f"  {n0} pts -> {len(pts)} at {a.voxel} m")
    print(f"  extent {np.round(hi - lo, 1).tolist()} m")
    print(f"  settings: dist={dist} tol={tol} deg  min_wall_height={minh} m"
          f"  min_wall_area={mina} m2"
          f"  (run config says max_planes={st.get('max_planes', 12)})\n")

    print(f"peeling up to {a.planes} planes...", flush=True)
    models = pdet.fit_plane_models(pts, dist=dist, min_points=1,
                                   max_planes=a.planes, fit_voxel=0.0)
    print(f"  {len(models)} planes fitted\n")

    # classify each plane on its own so the RANK is visible -- Structure()
    # classifies the set, and the set is not what the question is about
    up = np.array([0.0, 0.0, 1.0])
    ct, stt = np.cos(np.deg2rad(tol)), np.sin(np.deg2rad(tol))
    rows = []
    unclaimed = np.ones(len(pts), bool)
    for r, m in enumerate(models):
        res = np.abs(pts @ m[:3] + m[3])
        idx = np.flatnonzero((res < dist) & unclaimed)
        if idx.size < 50:
            continue
        unclaimed[idx] = False
        p = pts[idx]
        c = abs(float(m[:3] @ up))
        tilt = float(np.degrees(np.arccos(min(1.0, c))))    # 0 = horizontal
        zext = float(p[:, 2].max() - p[:, 2].min())
        area = pdet.plane_area(p, m)
        if c >= ct:
            kind = "horizontal"
        elif c <= stt:
            # area AND height: a chair back is vertical and 0.9 m tall
            kind = ("WALL" if (zext >= minh and area >= mina)
                    else ("vertical(small)" if zext >= minh
                          else "vertical(short)"))
        else:
            kind = "tilted"
        rows.append({"rank": r, "kind": kind, "n": int(idx.size),
                     "tilt": tilt, "zext": zext,
                     "area": area, "z": float(np.median(p[:, 2]))})

    walls = [x for x in rows if x["kind"] == "WALL"]
    print(f"{'rank':>4} {'kind':<15} {'points':>9} {'area m2':>8} "
          f"{'tilt':>6} {'z-ext':>6} {'z':>7}")
    for x in rows[:40]:
        print(f"{x['rank']:>4} {x['kind']:<15} {x['n']:>9} {x['area']:>8.1f} "
              f"{x['tilt']:>5.1f}d {x['zext']:>5.2f}m {x['z']:>6.2f}m")
    if len(rows) > 40:
        print(f"  ... {len(rows) - 40} more")

    print("\n--- verdict ---")
    kinds = {}
    for x in rows:
        kinds[x["kind"]] = kinds.get(x["kind"], 0) + 1
    print("  " + ", ".join(f"{v} {k}" for k, v in sorted(kinds.items())))
    cfg_max = int(st.get("max_planes", 12))
    if walls:
        first = walls[0]["rank"]
        print(f"  first wall at rank {first} "
              f"({walls[0]['area']:.0f} m2, {walls[0]['n']} pts)")
        print(f"  {len(walls)} walls total, {sum(w['area'] for w in walls):.0f} m2")
        if first >= cfg_max:
            print(f"\n  -> PLANE BUDGET. The walls are in the cloud but the "
                  f"first one ranks {first}, beyond structure.max_planes="
                  f"{cfg_max}.\n     Set structure.max_planes to at least "
                  f"{int((walls[-1]['rank'] + 1) * 1.3)} and re-run detect.")
        else:
            print(f"\n  -> walls are reachable within max_planes={cfg_max}. If "
                  f"the run still reported none, the difference is dist / "
                  f"normal_tol_deg / min_wall_height, not the budget.")
    else:
        vert = [x for x in rows if x["kind"].startswith("vertical")]
        print(f"  NO WALL at any rank up to {a.planes}.")
        if vert:
            print(f"  {len(vert)} vertical planes found, but none clears both "
                  f"gates: tallest {max(v['zext'] for v in vert):.2f} m "
                  f"(need {minh}), largest {max(v['area'] for v in vert):.1f} "
                  f"m2 (need {mina}).")
            print("\n  -> these are furniture faces and wall fragments, not "
                  "architecture. Lower min_wall_area/min_wall_height only if "
                  "you have checked they really are walls.")
        else:
            print("\n  -> NO RETURNS. There is no near-vertical surface in this "
                  "cloud to fit.\n     On a reflective site this is the "
                  "expected outcome: glass returns nothing at oblique\n"
                  "     incidence, and free-space carving erases what little it "
                  "does return.\n     No detect setting recovers this. Options: "
                  "rebuild with remove_dynamic.carve\n     disabled (or "
                  "free_ratio 3+), or accept that wall_contact() stays inert.")

    unc = int(unclaimed.sum())
    print(f"\n  {unc} of {len(pts)} sampled points ({100 * unc / len(pts):.0f}%) "
          f"lie on no plane at all")


if __name__ == "__main__":
    main()
