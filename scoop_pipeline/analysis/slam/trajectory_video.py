#!/usr/bin/env python3
"""Top-view videos of a pass's two robots: their odometry alone, and
Swarm-SLAM's merged graph with its loop closures, over the reference.

    python scoop_pipeline/analysis/slam/trajectory_video.py odometry \\
        --run <odometry run dir> --run <odometry run dir> --out odometry.mp4 [--map PCD]
    python scoop_pipeline/analysis/slam/trajectory_video.py swarm <export dir> \\
        --run <export dir>/runs/<channel> --out swarm.mp4 [--map PCD]

    common: --speed 10 (bag s per video s), --fps 30, --size 1280x720, --zband 0.3 1.8,
            --title TEXT

odometry   one run of slam/odometry/run_odometry.sh per robot (run.json names
           the robot, the bag and the frame of trajectory.tum). Each estimate
           is put in the LiDAR frame and placed on the reference at its own
           first pose: the robots do not know where they started relative to
           each other, the start is given for the picture, and what grows
           from it is each one's drift. The reference (stage 04's
           traj_<machine>_lidar.tum) is drawn in grey under it.
swarm      a slam/run_swarm_slam.sh run of a datasets/swarm_slam_lidar.py
           export (cslam_eval.py reads the same files). Swarm-SLAM writes its
           graph once, at the end: the final optimized keyframes are drawn in
           the order they were made, all moved by one rigid fit onto the
           reference (both robots together, so their relative placement is
           Swarm-SLAM's own). Until the first inter-robot loop closure, robot 1
           is drawn faded where the graph before optimization had it, in its
           own frame; from then on, in the merged map. A loop closure is drawn
           once both its keyframes exist (Swarm-SLAM may have found it later):
           green between the robots, purple within one.

--map: a cloud of the site in the reference's frame (e.g. the pass's
mapping/cloud_in_ref_<tag>.pcd), its points between --zband m drawn as the
floor plan.
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "analysis"))
sys.path.insert(0, os.path.join(ROOT, "slam", "odometry"))
sys.path.insert(0, os.path.join(ROOT, "datasets"))
sys.path.insert(0, os.path.join(ROOT, "map_stages"))

COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
INTER, INTRA = "#2ca02c", "#9467bd"


class Track:
    """One robot's poses over time, in the drawing's frame (the reference's)."""

    def __init__(self, label, color, t, P, faded=None):
        o = np.argsort(t)
        self.label, self.color, self.t, self.P = label, color, np.asarray(t)[o], np.asarray(P)[o]
        self.faded = faded             # (t_until, t, P): drawn faded until t_until, then this track


def floor_plan(path, zband, cell=0.05):
    """Points of a cloud between zband, one per cell of `cell` m."""
    import mcap_convert as mc
    P = mc.read_cloud(path)
    P = P[(P[:, 2] >= zband[0]) & (P[:, 2] <= zband[1])]
    _, k = np.unique(np.floor(P[:, :2] / cell).astype(np.int64), axis=0, return_index=True)
    return P[k, :2]


def render(out, tracks, refs, edges, title, speed=10.0, fps=30, size=(1280, 720), plan=None,
           notes=None, merged_at=None):
    """Write the video. refs: [(label, P)] drawn whole in grey; edges: [(t,
    p_a, p_b, kind)] drawn from t on; notes: lines under the title."""
    import cv2
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    ts = [tr.t for tr in tracks] + [tr.faded[1] for tr in tracks if tr.faded]
    t0, t1 = min(t[0] for t in ts), max(t[-1] for t in ts)
    pts = np.vstack([tr.P[:, :2] for tr in tracks] + [P[:, :2] for _, P in refs]
                    + [tr.faded[2][:, :2] for tr in tracks if tr.faded])
    lo, hi = pts.min(0), pts.max(0)
    pad = 0.08 * max(hi - lo) + 0.5
    lo, hi = lo - pad, hi + pad

    w, h = size
    fig = plt.figure(figsize=(w / 100.0, h / 100.0), dpi=100)
    ax = fig.add_axes([0.05, 0.07, 0.62, 0.83])
    side = fig.add_axes([0.70, 0.07, 0.28, 0.83])
    side.axis("off")
    ax.set_aspect("equal")
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.grid(True, lw=0.3, alpha=0.4)
    fig.suptitle(title, fontsize=13, x=0.05, ha="left")
    if plan is not None and len(plan):
        ax.scatter(plan[:, 0], plan[:, 1], s=0.3, c="0.80", lw=0, zorder=0)
    for i, (lab, P) in enumerate(refs):
        ax.plot(P[:, 0], P[:, 1], color="0.55", lw=1.0, ls="--", zorder=1,
                label="reference" if i == 0 else None)
    lines, heads, fades = [], [], []
    for tr in tracks:
        lines.append(ax.plot([], [], color=tr.color, lw=2.0, zorder=3, label=tr.label)[0])
        heads.append(ax.plot([], [], "o", color=tr.color, ms=8, mec="k", zorder=5)[0])
        fades.append(ax.plot([], [], color=tr.color, lw=1.5, alpha=0.35, ls=":", zorder=2)[0]
                     if tr.faded else None)
    lc_inter = LineCollection([], colors=INTER, lw=1.2, alpha=0.9, zorder=4)
    lc_intra = LineCollection([], colors=INTRA, lw=1.2, alpha=0.9, zorder=4)
    ax.add_collection(lc_inter)
    ax.add_collection(lc_intra)
    if edges:
        ax.plot([], [], color=INTER, lw=1.2, label="loop closure, between robots")
        if any(e[3] == "intra" for e in edges):
            ax.plot([], [], color=INTRA, lw=1.2, label="loop closure, one robot")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.85)
    info = side.text(0.0, 1.0, "", va="top", ha="left", fontsize=10, family="monospace",
                     transform=side.transAxes)
    et = np.array([e[0] for e in edges]) if edges else np.zeros(0)

    vw = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not vw.isOpened():
        raise SystemExit("cannot write %s (OpenCV's mp4v writer)" % out)
    n = max(1, int(np.ceil((t1 - t0) / speed * fps)) + fps)       # a second of the end held
    for f in range(n):
        now = min(t0 + f * speed / fps, t1)
        txt = ["t = %6.1f s" % (now - t0), ""]
        for tr, ln, hd, fd in zip(tracks, lines, heads, fades):
            if tr.faded and now < tr.faded[0]:
                k = np.searchsorted(tr.faded[1], now, side="right")
                fd.set_data(tr.faded[2][:k, 0], tr.faded[2][:k, 1])
                ln.set_data([], [])
                hd.set_data(tr.faded[2][k - 1:k, 0], tr.faded[2][k - 1:k, 1])
                hd.set_alpha(0.35)
                txt.append("%-10s own frame, not merged" % tr.label.split()[0])
                continue
            if fd is not None:
                fd.set_data([], [])
            k = np.searchsorted(tr.t, now, side="right")
            ln.set_data(tr.P[:k, 0], tr.P[:k, 1])
            hd.set_data(tr.P[k - 1:k, 0], tr.P[k - 1:k, 1])
            hd.set_alpha(1.0)
            txt.append("%-10s %5.1f m" % (tr.label.split()[0],
                                          float(np.linalg.norm(np.diff(tr.P[:k, :2], axis=0), axis=1).sum())
                                          if k > 1 else 0.0))
        if len(et):
            on = et <= now
            lc_inter.set_segments([(e[1][:2], e[2][:2]) for e, o in zip(edges, on) if o and e[3] == "inter"])
            lc_intra.set_segments([(e[1][:2], e[2][:2]) for e, o in zip(edges, on) if o and e[3] == "intra"])
            ni = sum(1 for e, o in zip(edges, on) if o and e[3] == "inter")
            txt += ["", "loop closures", "  between robots %3d" % ni,
                    "  one robot      %3d" % sum(1 for e, o in zip(edges, on) if o and e[3] == "intra")]
            if merged_at is not None:
                txt.append("merged at %.1f s" % (merged_at - t0) if now >= merged_at else "not merged yet")
        if notes:
            txt += [""] + notes
        info.set_text("\n".join(txt))
        fig.canvas.draw()
        img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        vw.write(np.ascontiguousarray(img[:, :, ::-1]))
    vw.release()
    plt.close(fig)
    return n


# ------------------------------------------------------------------ odometry

def odometry_tracks(runs, ref_override=None):
    import make_config as MK
    import score as SC
    import trajs
    from pipeline_common import load_traj
    tracks, refs, notes = [], [], []
    for i, run in enumerate(runs):
        rj = json.load(open(os.path.join(run, "run.json")))
        paths = MK.mc._paths(rj["bag"])
        T_fl, _ = MK.to_lidar(paths, MK.mc.build_tf_static(paths), rj["machine"], rj["tum_frame"])
        t, T = load_traj(os.path.join(run, "trajectory.tum"))
        E = np.einsum("nij,jk->nik", T, T_fl)
        ref = (ref_override or {}).get(rj["machine"]) or SC.default_ref(rj["bag"], rj["machine"])
        tr, Tr = load_traj(ref)
        m = (t >= tr[0]) & (t <= tr[-1])
        t, E = t[m], E[m]
        A = trajs.interp(tr, Tr, t[:1])[0] @ np.linalg.inv(E[0])   # on the reference at its first pose
        P = np.einsum("ij,nj->ni", A[:3, :3], E[:, :3, 3]) + A[:3, 3]
        tracks.append(Track("%s %s" % (rj["machine"], rj["method"]), COLORS[i % len(COLORS)], t, P))
        refs.append((rj["machine"], Tr[(tr >= t[0]) & (tr <= t[-1]), :3, 3]))
        acc = os.path.join(run, "accuracy.json")
        if os.path.exists(acc):
            a = json.load(open(acc))
            notes.append("%s ATE %.1f cm" % (rj["machine"], a["ate"]["position_cm"]["rmse"]))
    return tracks, refs, notes


# ------------------------------------------------------------------ swarm

def swarm_tracks(export, run):
    import cslam_eval as CE
    import score as SC
    results = os.path.join(run, "results")
    st = CE.stamps(results)
    for k, v in CE.keyframe_stamps(run).items():
        st.setdefault(k, v)
    graphs = [s for _, s in CE.runs(results) if os.path.exists(os.path.join(s, "optimized_global_pose_graph.g2o"))]
    if not graphs:
        raise SystemExit("no optimized_global_pose_graph.g2o under %s" % results)
    best = max(graphs, key=lambda s: (len(CE.read_g2o(os.path.join(s, "optimized_global_pose_graph.g2o"))[0]),
                                      graphs.index(s)))
    V, E = CE.read_g2o(os.path.join(best, "optimized_global_pose_graph.g2o"))
    ip = os.path.join(best, "initial_global_pose_graph.g2o")
    V0 = CE.read_g2o(ip)[0] if os.path.exists(ip) else {}
    gts = {}
    for r in sorted({CE.key_robot(k)[0] for k in V}):
        p = os.path.join(export, "gt_r%d.tum" % r)
        if os.path.exists(p):
            gts[r] = CE.read_tum(p)

    def gt(k):
        r = CE.key_robot(k)[0]
        if r not in gts or k not in st:
            return None
        t, T = gts[r]
        i = int(np.argmin(np.abs(t - st[k])))
        return T[i] if abs(t[i] - st[k]) <= 0.01 else None

    keys = [k for k in V if k in st]
    pairs = [(V[k], gt(k)) for k in keys if gt(k) is not None]
    if len(pairs) < 3:
        raise SystemExit("fewer than 3 keyframes with ground truth: no fit onto the reference")
    Rf, tf = SC.pose_fit(np.array([a for a, _ in pairs]), np.array([b for _, b in pairs]))

    def to_ref(T):
        return Rf @ T[:3, 3] + tf

    inter = [(a, b) for a, b, _ in E if CE.key_robot(a)[0] != CE.key_robot(b)[0] and a in st and b in st]
    merged_at = min(max(st[a], st[b]) for a, b in inter) if inter else None
    tracks = []
    for r in sorted({CE.key_robot(k)[0] for k in keys}):
        ks = sorted((k for k in keys if CE.key_robot(k)[0] == r), key=lambda k: st[k])
        t = np.array([st[k] for k in ks])
        P = np.array([to_ref(V[k]) for k in ks])
        faded = None
        if r != 0 and V0:
            k0 = [k for k in ks if k in V0]
            if k0:
                faded = (merged_at if merged_at is not None else np.inf,
                         np.array([st[k] for k in k0]), np.array([to_ref(V0[k]) for k in k0]))
        tracks.append(Track("%s (r%d)" % (["mobile_1", "mobile_2"][r] if r < 2 else "robot_%d" % r, r),
                            COLORS[r % len(COLORS)], t, P, faded))
    edges = []
    for a, b, _ in E:
        (ra, ia), (rb, ib) = CE.key_robot(a), CE.key_robot(b)
        if a not in st or b not in st or a not in V or b not in V:
            continue
        if ra != rb:
            kind = "inter"
        elif abs(int(ib) - int(ia)) != 1:
            kind = "intra"
        else:
            continue
        edges.append((max(st[a], st[b]), to_ref(V[a]), to_ref(V[b]), kind))
    refs = [("r%d" % r, gts[r][1][:, :3, 3]) for r in sorted(gts)]
    notes = []
    ej = os.path.join(run, "cslam_eval.json")
    if os.path.exists(ej):
        e = json.load(open(ej))
        o = (e.get("optimized") or {}).get("joint_ate_rmse_m")
        if o is not None:
            notes.append("joint ATE %.1f cm" % (o * 100))
    return tracks, refs, edges, merged_at, notes


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["odometry", "swarm"])
    ap.add_argument("export", nargs="?", help="swarm: the export folder")
    ap.add_argument("--run", action="append", default=[], help="odometry: a run dir per robot; swarm: the run")
    ap.add_argument("--out", required=True)
    ap.add_argument("--map", default=None)
    ap.add_argument("--ref", action="append", default=[], metavar="MACHINE=TUM",
                    help="odometry: a robot's reference (default stage 04's for the run's bag)")
    ap.add_argument("--zband", nargs=2, type=float, default=[0.3, 1.8])
    ap.add_argument("--speed", type=float, default=10.0)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--size", default="1280x720")
    ap.add_argument("--title", default=None)
    a = ap.parse_args()
    size = tuple(int(v) for v in a.size.lower().split("x"))
    plan = floor_plan(a.map, a.zband) if a.map else None
    if a.mode == "odometry":
        if not a.run:
            ap.error("odometry: give --run <odometry run dir> per robot")
        tracks, refs, notes = odometry_tracks(a.run, dict(v.split("=", 1) for v in a.ref))
        title = a.title or "Odometry alone: each robot placed on the reference at its own first pose"
        n = render(a.out, tracks, refs, [], title, a.speed, a.fps, size, plan, notes)
    else:
        if not a.export or len(a.run) != 1:
            ap.error("swarm: <export dir> --run <export>/runs/<channel>")
        tracks, refs, edges, merged_at, notes = swarm_tracks(a.export, a.run[0])
        title = a.title or "Swarm-SLAM: final graph in time order, one rigid fit for both robots"
        n = render(a.out, tracks, refs, edges, title, a.speed, a.fps, size, plan, notes, merged_at)
    print("wrote %s (%d frames, %d fps, x%g)" % (a.out, n, a.fps, a.speed))


if __name__ == "__main__":
    main()
