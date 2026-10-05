#!/usr/bin/env python3
"""A robot's track on the map, and where it has no pose.

    python scoop_pipeline/checks/plot_track.py <pipeline config> [--track mobile_2_depth]
        [--other mobile_1_lidar] [--gap 1.0] [--no-vslam]

Top: the floor plan (the reference map's walls), the track (04's
traj_<track>.tum) coloured by time, each stretch without a pose as a dashed
line from the last pose before it to the first after it (labelled with its
start and length), the other robot's track in grey, the boards, and the raw
cuVSLAM odometry (processing/vslam.py's traj_vslam.txt) in purple, put on the
map at the track's first pose (the board seed) and never corrected after:
where it and the track part, one of them is wrong. Time marks every 50 s on both.
Bottom: the timeline -- when the track has a pose, when not.
Writes odometry/reference_<tag>/track_<track>.png and lists the stretches.
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

MISSING = "#d9653b"          # the stretches without a pose
OTHER = "#9a9a9a"            # the other robot
VSLAM = "#8a4fbf"            # raw cuVSLAM, placed at the start


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--track", default="mobile_2_depth")
    ap.add_argument("--other", default="mobile_1_lidar")
    ap.add_argument("--gap", type=float, default=1.0, help="a stretch without a pose: over this many s")
    ap.add_argument("--no-vslam", action="store_true", help="leave the raw cuVSLAM path out")
    a = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    import open3d as o3d
    from pipeline_common import load_pipeline
    A = importlib.import_module("02_refine_poses")
    P = load_pipeline(a.config)
    od = P.reference_dir()
    t2, T2 = A.load_traj(os.path.join(od, "traj_%s.tum" % a.track))
    po = os.path.join(od, "traj_%s.tum" % a.other)
    t1, T1 = A.load_traj(po) if a.other and os.path.exists(po) else (np.zeros(0), None)
    # the run's clock from mobile_2's start (as the track's log does: its first odometry - 21 s
    # is not known here; seconds from the earlier of the two tracks' starts)
    t0 = min([t2[0]] + ([t1[0]] if len(t1) else []))
    xy = T2[:, :2, 3]

    # raw cuVSLAM, put on the map at the track's first pose
    tv, Vm = np.zeros(0), None
    if not a.no_vslam and P.cfg.get("vslam"):
        sys.path.insert(0, os.path.join(ROOT, "processing"))
        import vslam
        pv = vslam.traj_path(P)
        if os.path.exists(pv):
            tv, V = A.load_traj(pv)
            k = int(np.argmin(np.abs(tv - t2[0])))
            Vm = np.einsum("ij,njk->nik", T2[0] @ np.linalg.inv(V[k]), V)
            print("raw cuVSLAM %s: %d poses, placed at the track's first pose (%.2f s apart)"
                  % (pv, len(tv), abs(tv[k] - t2[0])))

    # the floor plan: map points 0.3-1.6 m above the floor under the track's start
    ref_map = P.ref_file("anchored_map") if P.reference else P.outp("map_final_{tag}_anchored.pcd")
    M = np.asarray(o3d.io.read_point_cloud(ref_map).voxel_down_sample(0.05).points)
    c0 = T2[0][:3, 3]
    near = np.linalg.norm(M[:, :2] - c0[:2], axis=1) < 1.5
    z = M[near & (M[:, 2] < c0[2]) & (M[:, 2] > c0[2] - 3.0), 2]
    floor = c0[2] - 0.8
    if len(z):
        h, e = np.histogram(z, bins=np.arange(z.min(), z.max() + 0.1, 0.05))
        floor = float(e[np.argmax(h)] + 0.025)
    W = M[(M[:, 2] > floor + 0.3) & (M[:, 2] < floor + 1.6)]
    allxy = np.vstack([xy] + ([T1[:, :2, 3]] if len(t1) else []) + ([Vm[:, :2, 3]] if len(tv) else []))
    lo, hi = allxy.min(0) - 3.0, allxy.max(0) + 3.0
    W = W[(W[:, 0] > lo[0]) & (W[:, 0] < hi[0]) & (W[:, 1] > lo[1]) & (W[:, 1] < hi[1])]

    # stretches without a pose
    dt = np.diff(t2)
    gi = np.flatnonzero(dt > a.gap)
    gaps = [(t2[i], t2[i + 1], xy[i], xy[i + 1]) for i in gi]
    span = (t2[0] - t0, t2[-1] - t0)

    fig = plt.figure(figsize=(12, 10.5))
    gs = fig.add_gridspec(2, 1, height_ratios=[6, 1], hspace=0.18)
    ax = fig.add_subplot(gs[0])
    ax.set_facecolor("white")
    if len(W):
        cell = 0.08
        H, xe, ye = np.histogram2d(W[:, 0], W[:, 1], bins=[np.arange(lo[0], hi[0] + cell, cell),
                                                             np.arange(lo[1], hi[1] + cell, cell)])
        ax.imshow(np.ma.masked_less((H > 2).T.astype(float), 0.5), origin="lower",
                  extent=(xe[0], xe[-1], ye[0], ye[-1]), cmap="Greys", vmin=0, vmax=1.6,
                  interpolation="nearest", zorder=0)
    if len(t1):
        ax.plot(T1[:, 0, 3], T1[:, 1, 3], color=OTHER, lw=1.0, alpha=0.8, zorder=1,
                label=a.other.replace("_lidar", "") + " (for reference)")
    if len(tv):
        vxy = Vm[:, :2, 3]
        vc = np.r_[0, np.flatnonzero(np.diff(tv) > a.gap) + 1, len(tv)]
        for k, (s0, s1) in enumerate(zip(vc[:-1], vc[1:])):
            ax.plot(vxy[s0:s1, 0], vxy[s0:s1, 1], color=VSLAM, lw=1.2, alpha=0.85, zorder=2,
                    label="raw cuVSLAM, placed at the start" if k == 0 else None)
    # time marks every 50 s on the track and on raw cuVSLAM
    for tt, pts, col in [(t2, xy, "#1f3d7a")] + ([(tv, Vm[:, :2, 3], VSLAM)] if len(tv) else []):
        for m in np.arange(50.0, tt[-1] - t0, 50.0):
            k = int(np.argmin(np.abs(tt - t0 - m)))
            if abs(tt[k] - t0 - m) > 2.0:
                continue
            ax.plot(*pts[k], "o", ms=3.5, color=col, zorder=6)
            ax.annotate("%.0f" % m, pts[k], fontsize=7, color=col, xytext=(3, 3),
                        textcoords="offset points", zorder=6)
    # the track, cut at the gaps, coloured by time
    cut = np.r_[0, gi + 1, len(t2)]
    norm = plt.Normalize(0, t2[-1] - t0)
    cmap = plt.get_cmap("Blues")
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("b", cmap(np.linspace(0.35, 1.0, 256)))
    lc = None
    for s0, s1 in zip(cut[:-1], cut[1:]):
        if s1 - s0 < 2:
            ax.plot(xy[s0:s1, 0], xy[s0:s1, 1], "o", ms=3, color=cmap(norm(t2[s0] - t0)), zorder=3)
            continue
        seg = np.stack([xy[s0:s1 - 1], xy[s0 + 1:s1]], axis=1)
        lc = LineCollection(seg, cmap=cmap, norm=norm, linewidths=2.0, zorder=3)
        lc.set_array(t2[s0:s1 - 1] - t0)
        ax.add_collection(lc)
    for k, (ta, tb, pa, pb) in enumerate(gaps):
        ax.plot([pa[0], pb[0]], [pa[1], pb[1]], ls=(0, (4, 3)), color=MISSING, lw=1.6, zorder=4,
                label="no pose (straight line between the ends)" if k == 0 else None)
        ax.plot([pa[0], pb[0]], [pa[1], pb[1]], "o", ms=4, color=MISSING, zorder=5)
        if tb - ta >= 5.0:
            m = (pa + pb) / 2
            ax.annotate("%.0f s, %.0f s" % (ta - t0, tb - ta), m, fontsize=8, color="#5a2a14",
                        xytext=(4, 4), textcoords="offset points", zorder=6)
    ax.plot(*xy[0], "s", ms=8, color=cmap(0.0), mec="black", mew=0.6, zorder=6, label="start")
    ax.plot(*xy[-1], "D", ms=7, color=cmap(1.0), mec="black", mew=0.6, zorder=6, label="end")
    # boards
    af = json.load(open(P.anchor_frame()))
    boards = dict(af.get("boards") or {})
    boards.update({k: v for k, v in P.extra_boards().items() if k not in boards})
    for n, r in boards.items():
        bx, by = r["xyz"][:2]
        if lo[0] <= bx <= hi[0] and lo[1] <= by <= hi[1]:
            ax.plot(bx, by, "^", ms=7, color="#2b2b2b", zorder=6)
            ax.annotate(n, (bx, by), fontsize=8, color="#2b2b2b", xytext=(5, -10),
                        textcoords="offset points")
    if lc is not None:
        cb = fig.colorbar(lc, ax=ax, fraction=0.03, pad=0.01)
        cb.set_label("s from the start")
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_aspect("equal")
    ax.set_xlabel("map x (m)")
    ax.set_ylabel("map y (m)")
    have = len(t2)
    ax.set_title("%s: %d poses over %.0f-%.0f s, %d stretches without a pose (%.0f s in all)"
                 % (a.track, have, span[0], span[1], len(gaps), sum(tb - ta for ta, tb, _, _ in gaps)),
                 fontsize=11)
    ax.legend(loc="best", fontsize=8, frameon=True)
    ax.grid(color="#e6e6e6", lw=0.5, zorder=-1)

    # timeline
    tx = fig.add_subplot(gs[1])
    tx.broken_barh([(span[0], span[1] - span[0])], (0, 1), color=cmap(0.6))
    for ta, tb, _, _ in gaps:
        tx.broken_barh([(ta - t0, tb - ta)], (0, 1), color=MISSING)
    tx.set_yticks([])
    tx.set_xlim(0, (max(t2[-1], t1[-1] if len(t1) else 0) - t0))
    tx.set_xlabel("s from the start  (blue: a pose, orange: none)")
    for sp in ("top", "right", "left"):
        tx.spines[sp].set_visible(False)

    out = os.path.join(od, "track_%s.png" % a.track)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print("wrote %s" % out)
    print("stretches without a pose (over %.1f s):" % a.gap)
    for ta, tb, pa, pb in gaps:
        print("  %6.1f - %6.1f s (%5.1f s): from (%.1f, %.1f) to (%.1f, %.1f), %.1f m apart"
              % (ta - t0, tb - t0, tb - ta, pa[0], pa[1], pb[0], pb[1], np.linalg.norm(pb - pa)))


if __name__ == "__main__":
    main()
