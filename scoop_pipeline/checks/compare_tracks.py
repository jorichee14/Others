#!/usr/bin/env python3
"""Two runs of one robot's track compared (e.g. the pose graph with and
without the depth frames' map fixes: 04 --track mobile_2_depth --as
mobile_2_depth_nomap --set pgo_map=false).

    python scoop_pipeline/checks/compare_tracks.py <pipeline config>
        [--a mobile_2_depth] [--b mobile_2_depth_nomap] [--bin 30]

Prints, frame by frame where both have a pose: how far apart they are (xy);
each one's uncertainty (cov_<track>.csv: the xy ellipse's major semi-axis);
each one's distance to the board sightings (summary_<track>.json); how often
mobile_1's LiDAR sees a robot within 1 m of it (lidar_sightings cache), and
the same per time bin. Writes odometry/reference_<tag>/compare_<a>_<b>.png:
both on the floor plan, and their uncertainty and separation over time.
"""
import argparse
import importlib
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "map_stages"))


def lidar_check(od, track_t, track_xy, t1, T1, cache):
    """mobile_1 1.5-15 m away: share of poses with a LiDAR candidate within
    1 m, and the median distance to it (as depth_track reports)."""
    if not os.path.exists(cache) or T1 is None:
        return None
    scans = json.load(open(cache))["scans"]
    st = np.array([s["t"] for s in scans])
    looked, d = 0, []
    for t, p in zip(track_t[::3], track_xy[::3]):
        i1 = int(np.argmin(np.abs(t1 - t)))
        if abs(t1[i1] - t) > 0.2 or not 1.5 <= np.linalg.norm(T1[i1][:2, 3] - p) <= 15.0:
            continue
        looked += 1
        k = int(np.argmin(np.abs(st - t)))
        if abs(st[k] - t) > 0.15 or not scans[k]["c"]:
            continue
        dm = float(np.min(np.linalg.norm(np.array([c[:2] for c in scans[k]["c"]]) - p, axis=1)))
        if dm <= 1.0:
            d.append(dm)
    return looked, len(d), (float(np.median(d)) if d else float("nan"))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--a", default="mobile_2_depth")
    ap.add_argument("--b", default="mobile_2_depth_nomap")
    ap.add_argument("--bin", type=float, default=30.0, help="seconds per row of the table")
    ap.add_argument("--other", default="mobile_1_lidar")
    a = ap.parse_args()
    from pipeline_common import load_pipeline
    A = importlib.import_module("02_refine_poses")
    P = load_pipeline(a.config)
    od = P.reference_dir()

    run = {}
    for name in (a.a, a.b):
        p = os.path.join(od, "traj_%s.tum" % name)
        if not os.path.exists(p):
            raise SystemExit("no %s: run 04 for %s first" % (p, name))
        t, T = A.load_traj(p)
        pc = os.path.join(od, "cov_%s.csv" % name)
        cv = np.loadtxt(pc, delimiter=",", skiprows=1, ndmin=2) if os.path.exists(pc) else None
        ps = os.path.join(od, "summary_%s.json" % name)
        sm = json.load(open(ps)) if os.path.exists(ps) else {}
        run[name] = (t, T, cv, sm)
    ta, Ta, ca, sa = run[a.a]
    tb, Tb, cb, sb = run[a.b]
    # frame by frame where both have one (same depth frames: matched by time)
    ib = np.clip(np.searchsorted(tb, ta), 0, len(tb) - 1)
    ib = np.where(np.abs(tb[ib] - ta) <= np.abs(tb[np.clip(ib - 1, 0, None)] - ta), ib,
                  np.clip(ib - 1, 0, None))
    both = np.abs(tb[ib] - ta) <= 0.01
    ia = np.flatnonzero(both)
    ib = ib[both]
    d = np.linalg.norm(Ta[ia, :2, 3] - Tb[ib, :2, 3], axis=1)
    t0 = ta[0]

    def sig(cv, tt):
        if cv is None:
            return np.full(len(tt), np.nan)
        k = np.clip(np.searchsorted(cv[:, 0], tt), 0, len(cv) - 1)
        return np.where(np.abs(cv[k, 0] - tt) <= 0.01, cv[k, 5], np.nan)

    sga, sgb = sig(ca, ta[ia]), sig(cb, tb[ib])
    print("%s vs %s: %d frames with both" % (a.a, a.b, len(ia)))
    print("  apart (xy): median %.2f m, p95 %.2f m, max %.2f m"
          % (np.median(d), np.percentile(d, 95), d.max()))
    for name, sg in ((a.a, sga), (a.b, sgb)):
        if np.isfinite(sg).any():
            print("  %-28s sigma xy (major semi-axis): median %6.1f cm, p95 %6.1f cm, max %6.1f cm"
                  % (name, 100 * np.nanmedian(sg), 100 * np.nanpercentile(sg, 95), 100 * np.nanmax(sg)))
    for name, sm in ((a.a, sa), (a.b, sb)):
        r = sm.get("track_minus_board_cm") or {}
        if r.get("median") is not None:
            print("  %-28s track - board sightings: median %.1f cm, p95 %.1f cm, max %.1f cm"
                  % (name, r["median"], r["p95"], r["max"]))
    po = os.path.join(od, "traj_%s.tum" % a.other)
    t1, T1 = A.load_traj(po) if os.path.exists(po) else (np.zeros(0), None)
    for name, (t, T, _, sm) in run.items():
        cache = os.path.join(od, "lidar_sightings_%s.json" % a.a)
        r = lidar_check(od, t, T[:, :2, 3], t1, T1, cache)
        if r and r[0]:
            print("  %-28s mobile_1's LiDAR, mobile_2 placed 1.5-15 m from it: a candidate within 1 m "
                  "in %d of %d frames (%.0f %%), %.0f cm off (median)"
                  % (name, r[1], r[0], 100.0 * r[1] / r[0], 100 * r[2]))
    print("\nper %.0f s (A = %s, B = %s; sigma: the xy ellipse's major semi-axis, median)"
          % (a.bin, a.a, a.b))
    print("%14s %7s %12s %12s %12s %12s" % ("s from start", "frames", "apart med m", "apart max m",
                                           "sigma A cm", "sigma B cm"))
    tt = ta[ia] - t0
    for lo in np.arange(0, tt.max() + 1e-9, a.bin):
        m = (tt >= lo) & (tt < lo + a.bin)
        if not m.any():
            continue
        print("%6.0f-%6.0f %7d %12.2f %12.2f %12.1f %12.1f"
              % (lo, lo + a.bin, m.sum(), np.median(d[m]), d[m].max(),
                 100 * np.nanmedian(sga[m]) if np.isfinite(sga[m]).any() else np.nan,
                 100 * np.nanmedian(sgb[m]) if np.isfinite(sgb[m]).any() else np.nan))

    # the figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import open3d as o3d
    ref_map = P.ref_file("anchored_map") if P.reference else P.outp("map_final_{tag}_anchored.pcd")
    M = np.asarray(o3d.io.read_point_cloud(ref_map).voxel_down_sample(0.05).points)
    c0 = Ta[0][:3, 3]
    near = np.linalg.norm(M[:, :2] - c0[:2], axis=1) < 1.5
    z = M[near & (M[:, 2] < c0[2]) & (M[:, 2] > c0[2] - 3.0), 2]
    floor = c0[2] - 0.8
    if len(z):
        h, e = np.histogram(z, bins=np.arange(z.min(), z.max() + 0.1, 0.05))
        floor = float(e[np.argmax(h)] + 0.025)
    W = M[(M[:, 2] > floor + 0.3) & (M[:, 2] < floor + 1.6)]
    allxy = np.vstack([Ta[:, :2, 3], Tb[:, :2, 3]])
    lo, hi = allxy.min(0) - 3.0, allxy.max(0) + 3.0
    W = W[(W[:, 0] > lo[0]) & (W[:, 0] < hi[0]) & (W[:, 1] > lo[1]) & (W[:, 1] < hi[1])]
    fig = plt.figure(figsize=(12, 12))
    gs = fig.add_gridspec(2, 1, height_ratios=[5, 2], hspace=0.2)
    ax = fig.add_subplot(gs[0])
    if len(W):
        cell = 0.08
        H, xe, ye = np.histogram2d(W[:, 0], W[:, 1], bins=[np.arange(lo[0], hi[0] + cell, cell),
                                                             np.arange(lo[1], hi[1] + cell, cell)])
        ax.imshow(np.ma.masked_less((H > 2).T.astype(float), 0.5), origin="lower",
                  extent=(xe[0], xe[-1], ye[0], ye[-1]), cmap="Greys", vmin=0, vmax=1.6,
                  interpolation="nearest", zorder=0)
    for (t, T, _, _), col, lab in ((run[a.b], "#8a4fbf", a.b), (run[a.a], "#1f5fae", a.a)):
        cut = np.r_[0, np.flatnonzero(np.diff(t) > 1.0) + 1, len(t)]
        for k, (s0, s1) in enumerate(zip(cut[:-1], cut[1:])):
            ax.plot(T[s0:s1, 0, 3], T[s0:s1, 1, 3], color=col, lw=1.4, alpha=0.9,
                    label=lab if k == 0 else None)
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_aspect("equal")
    ax.set_xlabel("map x (m)")
    ax.set_ylabel("map y (m)")
    ax.legend(loc="best", fontsize=9)
    ax.grid(color="#e6e6e6", lw=0.5, zorder=-1)
    ax.set_title("%s vs %s: apart median %.2f m, max %.2f m" % (a.a, a.b, np.median(d), d.max()),
                 fontsize=11)
    ux = fig.add_subplot(gs[1])
    for sg, col, lab in ((sgb, "#8a4fbf", "sigma xy, " + a.b), (sga, "#1f5fae", "sigma xy, " + a.a)):
        if np.isfinite(sg).any():
            ux.plot(tt, 100 * sg, ".", ms=2, color=col, label=lab)
    ux.plot(tt, 100 * d, ".", ms=2, color="#d9653b", label="apart (xy)")
    ux.set_yscale("log")
    ux.set_xlabel("s from the start")
    ux.set_ylabel("cm")
    ux.legend(loc="upper left", fontsize=8, markerscale=4)
    ux.grid(color="#e6e6e6", lw=0.5, which="both")
    out = os.path.join(od, "compare_%s_%s.png" % (a.a, a.b))
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print("\nwrote %s" % out)


if __name__ == "__main__":
    main()
