#!/usr/bin/env python3
"""The relative pose between two robots over a Swarm-SLAM run, as collaborative
perception would use it: at every --step s of the run, where the robots think
they are with respect to each other, against the ground truth.

    python scoop_pipeline/analysis/slam/relative_pose.py <export dir> --run <run> [--run <run> ...]
        [--step 1.0] [--range 20] [--usable 0.5 2.0] [--out <png>]

At a time t of the run, the estimate is what the robots had then: the latest
global graph cslam had optimized with both robots connected (its time: its
newest keyframe's stamp, as cslam_eval's time to merge), each robot's pose at
t carried from its last keyframe in that graph by its own odometry (the
export's /r<i>/odom):

    P_i(t) = V[k_i] · O_i(t_k)^-1 · O_i(t)

and the relative pose P_0(t)^-1 P_1(t) is compared with the ground truth's
G_0(t)^-1 G_1(t) (gt_r<i>.tum): the translation error (m) and the rotation
error (deg) of where robot 0 places robot 1. Before the first such graph
there is no relative pose at all: the robots are not in one frame.

Per run:
  available      share of the run with a relative pose (from the first merged
                 graph to the end), and when it starts
  error          translation and rotation error, median / p95 / max over the
                 available time
  at --range m   where a point --range m from robot 1 lands in robot 0's
                 frame, the bound translation + range × rotation (rad): what
                 a fused detection that far away is moved by
  usable         share of the run with a relative pose within --usable
                 (m, deg) (default 0.5 m, 2 deg)
  before / after the first loop closure
                 the same, split at the first graph with an inter-robot loop
                 closure (an edge other than the board edge between the
                 robots' first keyframes): what the boards alone gave, and
                 what the loop closures made of it

Writes relative_pose.json and relative_pose.csv (t, error per step) into
each run, and with --out one plot of the error over time for every run given
(with and without boards side by side).
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cslam_eval as CE                                             # noqa: E402


def interp(export, r):
    """The export's /r<i>/odom of robot r as a pose interpolator."""
    mc = CE._mc()
    return mc.PoseInterp.from_topic(mc._paths(os.path.join(export, "bag")), "/r%d/odom" % r, max_gap_s=1.0)


def merged_graphs(results, st):
    """[(t, {key: 4x4}, has a loop closure)] of the optimized global graphs
    with an inter-robot edge, t = their newest keyframe's stamp, in time order."""
    out = []
    for _, s in CE.runs(results):
        p = os.path.join(s, "optimized_global_pose_graph.g2o")
        if not os.path.exists(p):
            continue
        V, E = CE.read_g2o(p)
        inter = [(CE.key_robot(a), CE.key_robot(b)) for a, b, _ in E if CE.key_robot(a)[0] != CE.key_robot(b)[0]]
        if not inter:
            continue
        ts = [st[k] for k in V if k in st]
        if ts:
            # a loop closure: an inter-robot edge that is not the board edge (keyframe 0 to keyframe 0)
            out.append((max(ts), V, any(not (ka[1] == 0 and kb[1] == 0) for ka, kb in inter)))
    out.sort(key=lambda g: g[0])
    return out


def series(export, run, step=1.0, robots=(0, 1)):
    """([(t, translation error m, rotation error deg)] over the run (NaN where
    there is no relative pose), the run's length, and the time of the first
    loop closure (s into the run, or None))."""
    mc = CE._mc()
    results = os.path.join(run, "results")
    st = CE.stamps(results)
    for k, v in CE.keyframe_stamps(run).items():
        st.setdefault(k, v)
    graphs = merged_graphs(results, st)
    gts = {r: mc.PoseInterp.from_tum(os.path.join(export, "gt_r%d.tum" % r), max_gap_s=1.0) for r in robots}
    odo = {r: interp(export, r) for r in robots}
    t0 = max(gts[r].ts[0] for r in robots) * 1e-9
    t1 = min(gts[r].ts[-1] for r in robots) * 1e-9
    # each graph's keyframes per robot, by stamp
    kf = []
    for tg, V, _ in graphs:
        per = {}
        for k in V:
            r, _ = CE.key_robot(k)
            if r in robots and k in st:
                per.setdefault(r, []).append((st[k], k))
        kf.append({r: sorted(v) for r, v in per.items()})
    out = []
    gi = -1
    for t in np.arange(t0, t1, step):
        while gi + 1 < len(graphs) and graphs[gi + 1][0] <= t:
            gi += 1
        row = [float(t - t0), np.nan, np.nan]
        if gi >= 0:
            V, per = graphs[gi][1], kf[gi]
            P = {}
            for r in robots:
                ks = [x for x in per.get(r, []) if x[0] <= t]
                if not ks:
                    break
                tk, k = ks[-1]
                Ok, Ot = odo[r].at(int(round(tk * 1e9))), odo[r].at(int(round(t * 1e9)))
                if Ok is None or Ot is None:
                    break
                P[r] = V[k] @ np.linalg.inv(Ok) @ Ot
            G = {r: gts[r].at(int(round(t * 1e9))) for r in robots}
            if len(P) == len(robots) and all(g is not None for g in G.values()):
                a, b = robots
                D = np.linalg.inv(np.linalg.inv(G[a]) @ G[b]) @ (np.linalg.inv(P[a]) @ P[b])
                row[1], row[2] = float(np.linalg.norm(D[:3, 3])), CE.rot_deg(D[:3, :3])
        out.append(row)
    lc = [g[0] for g in graphs if g[2]]
    return np.array(out), t1 - t0, (float(lc[0] - t0) if lc else None)


def summarize(S, dur, rng=20.0, usable=(0.5, 2.0), lc=None):
    rep = _summary(S, dur, rng, usable)
    rep["first_loop_closure_s"] = None if lc is None else round(lc, 1)
    if lc is not None:
        for name, m in (("before_loop_closure", S[:, 0] < lc), ("after_loop_closure", S[:, 0] >= lc)):
            part = _summary(S[m], dur * m.mean(), rng, usable)
            part["seconds"] = int(m.sum())
            rep[name] = part
    return rep


def _summary(S, dur, rng, usable):
    ok = ~np.isnan(S[:, 1])
    rep = {"duration_s": round(float(dur), 1), "steps": int(len(S)),
           "available_pct": round(100.0 * ok.mean(), 1) if len(S) else 0.0,
           "available_from_s": round(float(S[ok, 0][0]), 1) if ok.any() and len(S) else None}
    if ok.any():
        tr, rot = S[ok, 1], S[ok, 2]
        at = tr + rng * np.radians(rot)

        def st(v, nd):
            return {"median": round(float(np.median(v)), nd), "p95": round(float(np.percentile(v, 95)), nd),
                    "max": round(float(v.max()), nd)}
        rep.update({"translation_m": st(tr, 3), "rotation_deg": st(rot, 2), "at_range_m": st(at, 3),
                    "range_m": rng})
        good = ok.copy()
        good[ok] = (tr <= usable[0]) & (rot <= usable[1])
        rep["usable"] = {"within_m": usable[0], "within_deg": usable[1],
                         "pct_of_run": round(100.0 * good.mean(), 1)}
    return rep


def report(name, rep):
    lines = ["%s: relative pose available %.1f %% of %.0f s (from %s s)"
             % (name, rep["available_pct"], rep["duration_s"],
                "-" if rep["available_from_s"] is None else "%.1f" % rep["available_from_s"])]
    if "translation_m" in rep:
        t, r, a, u = rep["translation_m"], rep["rotation_deg"], rep["at_range_m"], rep["usable"]
        lines += ["  robot-to-robot position median %.3f / p95 %.3f / max %.3f m; rotation median %.2f / p95 %.2f / max %.2f deg"
                  % (t["median"], t["p95"], t["max"], r["median"], r["p95"], r["max"]),
                  "  a point %g m away moved by: median %.3f / p95 %.3f m"
                  % (rep["range_m"], a["median"], a["p95"]),
                  "  usable (within %g m, %g deg): %.1f %% of the run"
                  % (u["within_m"], u["within_deg"], u["pct_of_run"])]
    if rep.get("first_loop_closure_s") is not None:
        lines.append("  first loop closure at %.1f s" % rep["first_loop_closure_s"])
        for name in ("before_loop_closure", "after_loop_closure"):
            p = rep[name]
            if "translation_m" in p:
                lines.append("    %-20s %4d s: available %5.1f %%, position %.3f / %.3f m, rotation %.2f / %.2f deg "
                             "(median / p95), usable %.1f %%"
                             % (name.replace("_", " "), p["seconds"], p["available_pct"], p["translation_m"]["median"],
                                p["translation_m"]["p95"], p["rotation_deg"]["median"], p["rotation_deg"]["p95"],
                                p["usable"]["pct_of_run"]))
            else:
                lines.append("    %-20s %4d s: no shared frame" % (name.replace("_", " "), p["seconds"]))
    else:
        lines.append("  no inter-robot loop closure in the run")
    return "\n".join(lines)


def plot(curves, path, lcs=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for i, (name, S) in enumerate(curves):
        l0 = ax[0].plot(S[:, 0], S[:, 1], label=name, lw=1.5)[0]
        ax[1].plot(S[:, 0], S[:, 2], label=name, lw=1.5, color=l0.get_color())
        lc = (lcs or [None] * len(curves))[i]
        if lc is not None:
            for a in ax:
                a.axvline(lc, color=l0.get_color(), ls=":", lw=1.2)
    ax[0].set_ylabel("translation error (m)")
    ax[1].set_ylabel("rotation error (deg)")
    ax[1].set_xlabel("time into the run (s); no line = no shared frame yet; dotted = first loop closure")
    for a in ax:
        a.grid(alpha=0.3)
        a.set_ylim(bottom=0)
    ax[0].legend(loc="upper right")
    ax[0].set_title("Relative pose between the robots, against the ground truth")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("export")
    ap.add_argument("--run", action="append", required=True, help="a run of slam/run_swarm_slam.sh (repeat to compare)")
    ap.add_argument("--step", type=float, default=1.0, help="s between evaluated times")
    ap.add_argument("--range", type=float, default=20.0, help="m: distance of the point whose displacement is reported")
    ap.add_argument("--usable", type=float, nargs=2, default=[0.5, 2.0], metavar=("M", "DEG"))
    ap.add_argument("--out", default=None, help="plot of every run's error over time (png)")
    a = ap.parse_args()
    curves, lcs = [], []
    for run in a.run:
        S, dur, lc = series(a.export, run, a.step)
        rep = summarize(S, dur, a.range, tuple(a.usable), lc)
        lcs.append(lc)
        name = os.path.basename(os.path.normpath(run))
        print(report(name, rep))
        json.dump(rep, open(os.path.join(run, "relative_pose.json"), "w"), indent=1)
        np.savetxt(os.path.join(run, "relative_pose.csv"), S, delimiter=",", fmt="%.4f",
                   header="t_s,translation_m,rotation_deg", comments="")
        curves.append((name, S))
    if a.out:
        plot(curves, a.out, lcs)
        print("wrote", a.out)


if __name__ == "__main__":
    main()
