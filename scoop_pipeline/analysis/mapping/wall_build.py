#!/usr/bin/env python3
"""Is the map's wall thickness made by stage 01, or already in the scans as
placed? Places the scans with 01's own code (iter_world_scans: the same
deskew, poses, range limits) and measures the same wall spots of the map
before and after 01's merge / dynamic removal / denoise.

  python3 scoop_pipeline/analysis/mapping/wall_build.py <pipeline config> \\
      [--map <data>/processed/<date>/<pass>/mapping/denoised_<pass>_<date>.pcd] \\
      [--probes 300] [--stride 1]

--map     the map to compare with (default: 01's denoised cloud of the config)
--probes  wall spots measured (planar wall patches of the map, 0.25 m radius)
--stride  every Nth scan (faster, fewer views)

For each spot, every scan that hit it: where that scan puts the wall along
the wall's normal. Thickness is the 5-95 % spread, as map_quality.py:

  one scan            the sensor's noise floor at that spot
  1 s ... whole run   the scans of that span around the spot's best view,
                      together: how the thickness grows with time and with
                      the places it was seen from
  the map             the same spot in --map

  whole run ~ map    01's merge / dynamic removal / denoise add nothing:
                     the thickness is in the scans as placed (poses, deskew,
                     the points themselves), not in how 01 builds the map
  map >> whole run   01 adds it

and where each scan puts the wall by its distance to the wall, signed away
from the sensor: a trend with distance is a range error in the points.
Reads the bag once, writes nothing.
"""
import argparse
import importlib.util
import os
import sys

from types import SimpleNamespace

import numpy as np
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
STAGES = os.path.join(os.path.dirname(os.path.dirname(HERE)), "map_stages")
sys.path.insert(0, STAGES)
sys.path.insert(0, HERE)

import map_quality as mq                                               # noqa: E402
from pipeline_common import DEFAULT_TOPICS                             # noqa: E402

KEEP = 20                         # points kept per scan and spot for the spans
SPANS = (1.0, 10.0, 60.0, 300.0)
DIST = (0.0, 2.0, 4.0, 7.0, 12.0, 1e9)


def stage01():
    spec = importlib.util.spec_from_file_location("build_map",
                                                  os.path.join(STAGES, "01_build_map.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def wall_probes(M, n, seed=0):
    """Up to n planar one-layer wall patches of the map: centre, normal,
    thickness."""
    rng = np.random.default_rng(seed)
    tree = cKDTree(M)
    out = []
    for i in rng.choice(len(M), size=min(len(M), 20 * n), replace=False):
        nb = tree.query_ball_point(M[i], mq.R)
        if len(nb) < mq.MIN_PTS:
            continue
        c = M[nb].mean(0)
        Q = M[nb] - c
        _, sv, vt = np.linalg.svd(Q, full_matrices=False)
        nrm = vt[2]
        if abs(nrm[2]) > np.cos(np.radians(60)) or sv[2] / sv[1] > mq.PLANARITY:
            continue                                  # not a wall, or not planar
        d = Q @ nrm
        if mq.split_layers(d) is not None:
            continue                                  # two layers: not one surface
        out.append((c, nrm, mq.spread(d)))
        if len(out) >= n:
            break
    return out


def measure(config, map_path=None, n_probes=300, stride=1):
    """Prints the tables; returns their medians (m): single, spans {s: m},
    whole, map, by_dist [(lo, hi, median offset)]."""
    b = stage01()
    P = b.load_pipeline(config)
    s = P.stage("01_build_map")
    P.traj = b.load_traj_cached(P)
    # 01 takes the points topic from P.sensor, which is dataset.points_topic
    # when set; this one does not need the camera P.sensor also reads
    S = SimpleNamespace(points_topic=P.dataset.get("points_topic")
                        or DEFAULT_TOPICS["points_topic"])
    mpath = map_path or P.outp(P.pcd("denoised"))
    import open3d as o3d
    M = np.asarray(o3d.io.read_point_cloud(mpath).points)
    print(f"map {mpath}: {len(M)} points")
    probes = wall_probes(M, n_probes)
    del M
    if not probes:
        raise SystemExit("no planar wall patches in the map")
    C = np.array([p[0] for p in probes])
    N = np.array([p[1] for p in probes])
    map_thick = np.array([p[2] for p in probes])
    print(f"{len(probes)} wall spots, map thickness there median {np.median(map_thick) * 100:.1f} cm")
    print(f"placing the scans with 01's iter_world_scans ({len(P.traj[0])} poses, "
          f"deskew {'on' if s.get('deskew', True) else 'off'}, "
          f"range {s['lidar_min']}-{s['lidar_max']} m), every {stride} scan(s)")

    ptree = cKDTree(C)
    rng = np.random.default_rng(1)
    hits = [[] for _ in probes]      # per spot: (t, n, spread, median d, dist, sign, kept d)
    n_scan = 0
    for k, (t, wp, origin) in enumerate(b.iter_world_scans(P, S, s)):
        if k % stride:
            continue
        n_scan += 1
        wp = np.asarray(b.as_cpu(wp), dtype=np.float64)
        o = np.asarray(b.as_cpu(origin), dtype=np.float64)
        dd, ii = ptree.query(wp, distance_upper_bound=mq.R)
        ok = np.isfinite(dd)
        if not ok.any():
            continue
        ii, pts = ii[ok], wp[ok]
        order = np.argsort(ii, kind="stable")
        ii, pts = ii[order], pts[order]
        cuts = np.flatnonzero(np.diff(ii)) + 1
        for grp, q in zip(np.split(ii, cuts), np.split(pts, cuts)):
            j = int(grp[0])
            if len(q) < 10:
                continue
            d = (q - C[j]) @ N[j]
            away = 1.0 if (C[j] - o) @ N[j] > 0 else -1.0     # +d = further from the sensor
            kept = d if len(d) <= KEEP else rng.choice(d, KEEP, replace=False)
            hits[j].append((t, len(d), mq.spread(d) if len(d) >= mq.MIN_PTS else np.nan,
                            float(np.median(d)), float(np.linalg.norm(C[j] - o)), away,
                            kept.astype(np.float32)))
        if n_scan % 1000 == 0:
            print(f"    {n_scan} scans", flush=True)
    print(f"    {n_scan} scans read")

    single, span_th = [], {w: [] for w in SPANS + (None,)}
    mapt, by_dist = [], [[] for _ in DIST[:-1]]
    for j, h in enumerate(hits):
        if len(h) < 3:
            continue
        ts = np.array([x[0] for x in h])
        cnt = np.array([x[1] for x in h])
        sp = np.array([x[2] for x in h])
        single += list(sp[np.isfinite(sp)])
        t0 = ts[np.argmax(cnt)]
        for w in span_th:
            sel = np.ones(len(h), bool) if w is None else np.abs(ts - t0) <= w / 2
            d = np.concatenate([h[i][6] for i in np.flatnonzero(sel)])
            if len(d) >= mq.MIN_PTS:
                span_th[w].append(mq.spread(d))
        mapt.append(map_thick[j])
        med = np.array([x[3] for x in h])
        away = np.array([x[5] for x in h])
        dist = np.array([x[4] for x in h])
        off = (med - np.median(med)) * away            # + = this scan puts the wall further away
        for bi in range(len(DIST) - 1):
            m = (dist >= DIST[bi]) & (dist < DIST[bi + 1])
            by_dist[bi] += list(off[m])

    def cm(v):
        return f"{np.median(v) * 100:5.1f}" if len(v) else "    -"
    print(f"\nwall thickness at the same {len(mapt)} spots, median cm (spots)")
    print(f"  one scan          {cm(single)}   (noise floor)")
    for w in SPANS:
        print(f"  {w:5g} s together  {cm(span_th[w])}   ({len(span_th[w])})")
    print(f"  whole run         {cm(span_th[None])}   ({len(span_th[None])})")
    print(f"  the map (01)      {cm(mapt)}")

    print("\nwhere a scan puts the wall by its distance to it, cm, + = further from the "
          "sensor than the spot's median (scans)")
    dist_med = []
    for bi in range(len(DIST) - 1):
        v = np.array(by_dist[bi])
        hi = "+" if DIST[bi + 1] > 1e8 else f"-{DIST[bi + 1]:g}"
        if len(v):
            dist_med.append((DIST[bi], DIST[bi + 1], float(np.median(v))))
            print(f"  {DIST[bi]:4g}{hi:5s} m   median {np.median(v) * 100:+5.1f}   "
                  f"p10-p90 {np.percentile(v, 10) * 100:+5.1f} .. {np.percentile(v, 90) * 100:+5.1f}"
                  f"   ({len(v)})")

    whole, mp = np.median(span_th[None]) if span_th[None] else np.nan, np.median(mapt)
    print()
    if np.isfinite(whole) and mp > whole + 0.02:
        print(f"READING: the map is {100 * (mp - whole):.1f} cm thicker than the scans "
              f"as placed: stage 01 adds it (merge / dynamic removal / denoise)")
    elif np.isfinite(whole):
        print(f"READING: the scans as placed are already {whole * 100:.1f} cm thick here "
              f"(map {mp * 100:.1f}): stage 01's merge adds nothing; the spans above "
              f"show when it appears, the distance table whether it is a range error")
    med = lambda v: float(np.median(v)) if len(v) else float("nan")       # noqa: E731
    return dict(single=med(single), spans={w: med(span_th[w]) for w in SPANS},
                whole=float(whole), map=float(mp), by_dist=dist_med)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--map")
    ap.add_argument("--probes", type=int, default=300)
    ap.add_argument("--stride", type=int, default=1)
    a = ap.parse_args()
    measure(a.config, a.map, a.probes, a.stride)


if __name__ == "__main__":
    main()
