#!/usr/bin/env python3
"""Swarm-SLAM (cslam) against the ground truth: a run of
slam/run_swarm_slam.sh on a datasets/swarm_slam.py export.

    python scoop_pipeline/analysis/slam/cslam_eval.py <export dir> [--run runs/<channel>] [--results DIR]
        [--cov ROBOT FILE] [--sigma-max 0.15] [--lc-ok 0.5 10]

Reads <export>/results/<..._experiment_robot_<i>>/<time>/ (cslam's logs,
evaluation.enable_logs): optimized_global_pose_graph.g2o and
initial_global_pose_graph.g2o (keyframe poses of the colour camera, gtsam keys
'g', 'A' + robot, keyframe), pose_timestamps<i>.csv (key -> stamp), log.csv;
and gt_r<i>.tum (the colour camera in map at every frame exported).

Reported, from the last global graph (the one with the most keyframes):
  ATE per robot   each robot's keyframes aligned to the ground truth on their
                  own (rigid, no scale): RMSE and median of the position error
  joint ATE       one alignment for every robot: the merged map's consistency,
                  which per-robot alignment hides
  initial         the same for the graph before optimization (what the
                  inter-robot loop closures and optimization changed)
  loop closures   every inter-robot edge against the ground-truth relative
                  pose of its two keyframes: translation / rotation error,
                  correct within --lc-ok (m, deg)
  time to merge   the first global graph with an inter-robot edge: the latest
                  keyframe stamp in it, s from the run's start
  communication   log.csv's front-end bytes, matches and vertices per robot
  odometry        the export's /r<i>/odom (what cslam was given) against the
                  ground truth at every frame, aligned the same way: the
                  baseline the optimized graph should beat
Keyframes whose ground truth is not known within 10 ms are left out, and so
are those where --cov (04's cov_<track>.csv: t, ..., sigma_xy_major_m) says
the ground truth's own uncertainty is over --sigma-max m.

--run: a run of slam/run_swarm_slam.sh (<export>/runs/<channel>): its results/
are read and the outputs go there. Writes cslam_eval.json and cslam_eval.png
(top view: ground truth, aligned keyframes per robot, inter-robot loop
closures) in the run (or <export> for --results / the default results/).
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
from scipy.spatial.transform import Rotation

HERE = os.path.dirname(os.path.abspath(__file__))
SYM_MASK = (1 << 48) - 1
# gtsam's writeG2o drops a LabeledSymbol's top byte (the 'g'); pose_timestamps keeps it
KEY_MASK = (1 << 56) - 1


def key_robot(k):
    """gtsam LabeledSymbol('g', 'A' + robot, index) -> (robot, index)."""
    k = int(k)
    return ((k >> 48) & 0xFF) - ord("A"), k & SYM_MASK


def pose(v):
    """[x y z qx qy qz qw] -> 4x4."""
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat(v[3:7]).as_matrix()
    T[:3, 3] = v[:3]
    return T


def read_g2o(path):
    """({key: 4x4}, [(k1, k2, 4x4)]) of a gtsam writeG2o file (Pose3)."""
    V, E = {}, []
    for line in open(path):
        p = line.split()
        if not p:
            continue
        if p[0].startswith("VERTEX_SE3"):
            V[int(p[1]) & KEY_MASK] = pose([float(x) for x in p[2:9]])
        elif p[0].startswith("EDGE_SE3"):
            E.append((int(p[1]) & KEY_MASK, int(p[2]) & KEY_MASK, pose([float(x) for x in p[3:10]])))
    return V, E


def read_tum(path):
    a = np.loadtxt(path, ndmin=2)
    return a[:, 0], np.array([pose(r[1:8]) for r in a])


def umeyama(src, dst):
    """Rigid (no scale) T with dst ~ T src, points (N, 3)."""
    ms, md = src.mean(0), dst.mean(0)
    U, _, Vt = np.linalg.svd((dst - md).T @ (src - ms))
    S = np.eye(3)
    S[2, 2] = np.sign(np.linalg.det(U @ Vt))
    R = U @ S @ Vt
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, md - R @ ms
    return T


def ate(est, gt):
    """(rmse, median, T) of positions after the rigid alignment est -> gt."""
    if len(est) < 3:
        return None
    T = umeyama(est, gt)
    e = np.linalg.norm(est @ T[:3, :3].T + T[:3, 3] - gt, axis=1)
    return float(np.sqrt(np.mean(e ** 2))), float(np.median(e)), T


def rot_deg(R):
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def runs(results):
    """[(robot, folder)] of every log written, oldest first."""
    out = []
    for d in glob.glob(os.path.join(results, "*_experiment_robot_*")):
        r = int(d.rsplit("_", 1)[1])
        for sub in glob.glob(os.path.join(d, "*")):
            if os.path.isdir(sub):
                out.append((os.path.getmtime(sub), r, sub))
    return [(r, s) for _, r, s in sorted(out)]


def stamps(results):
    """{key: stamp s} from every robot's latest pose_timestamps<i>.csv."""
    latest = {}
    for r, sub in runs(results):
        p = os.path.join(sub, "pose_timestamps%d.csv" % r)
        if os.path.exists(p):
            latest[r] = p
    out = {}
    for p in latest.values():
        for line in open(p).readlines()[1:]:
            k, s, ns = line.strip().split(",")
            out[int(k) & KEY_MASK] = int(s) + int(ns) * 1e-9
    return out


def read_log(path):
    out = {}
    for line in open(path):
        p = line.strip().split(",")
        if len(p) == 2 and p[0] != "error":
            try:
                out[p[0]] = float(p[1])
            except ValueError:
                pass
    return out


def odometry(export, gts, ok):
    """{robot: (stamps s, positions (N, 3))} of the export's /r<i>/odom at
    the frames whose ground truth ok(robot, stamp) gives, with that ground
    truth; {} without the export's bag."""
    bag = os.path.join(export, "bag")
    if not os.path.isdir(bag):
        return {}
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "datasets"))
    import mcap_convert as mc
    est = {r: [] for r in gts}
    for m in mc.read(mc._paths(bag), ["/r%d/odom" % r for r in gts]):
        P = m.ros_msg.pose.pose.position
        est[int(m.channel.topic.split("/")[1][1:])].append((mc.stamp_ns(m.ros_msg) * 1e-9, P.x, P.y, P.z))
    out = {}
    for r, rows in est.items():
        e, g = [], []
        for t, x, y, z in rows:
            G = ok(r, t)
            if G is not None:
                e.append((x, y, z))
                g.append(G[:3, 3])
        if e:
            out[r] = (np.array(e), np.array(g))
    return out


def evaluate(export, results=None, cov=None, sigma_max=0.15, lc_ok=(0.5, 10.0), plot=True, out=None):
    results = results or os.path.join(export, "results")
    out = out or export
    gts = {}
    for p in sorted(glob.glob(os.path.join(export, "gt_r*.tum"))):
        gts[int(os.path.basename(p)[4:-4])] = read_tum(p)
    if not gts:
        raise SystemExit("no gt_r<i>.tum in %s (datasets/swarm_slam.py writes them)" % export)
    st = stamps(results)
    logs = runs(results)
    graphs = [(r, s) for r, s in logs if os.path.exists(os.path.join(s, "optimized_global_pose_graph.g2o"))]
    if not graphs:
        raise SystemExit("no optimized_global_pose_graph.g2o under %s (cslam with "
                         "evaluation.enable_logs: true)" % results)
    bad_cov = {}
    for r, f in (cov or {}).items():
        c = np.loadtxt(f, delimiter=",", skiprows=1, ndmin=2)
        bad_cov[int(r)] = c[c[:, 5] > sigma_max, 0]

    def gt_at(r, s):
        if r not in gts:
            return None
        t, T = gts[r]
        i = int(np.argmin(np.abs(t - s)))
        if abs(t[i] - s) > 0.01:
            return None
        if r in bad_cov and len(bad_cov[r]) and np.min(np.abs(bad_cov[r] - s)) <= 0.01:
            return None
        return T[i]

    def gt_of(k):
        return gt_at(key_robot(k)[0], st[k]) if k in st else None

    t_start = min(g[0][0] for g in gts.values())
    # time to merge: the first global graph with an inter-robot edge
    merged_at = None
    for r, s in graphs:
        V, E = read_g2o(os.path.join(s, "optimized_global_pose_graph.g2o"))
        if any(key_robot(a)[0] != key_robot(b)[0] for a, b, _ in E):
            ts = [st[k] for k in V if k in st]
            merged_at = (max(ts) - t_start) if ts else None
            break
    # the last global graph with the most keyframes
    best = max(graphs, key=lambda g: (len(read_g2o(os.path.join(g[1], "optimized_global_pose_graph.g2o"))[0]),
                                      graphs.index(g)))
    rep = {"export": export, "graph": best[1], "time_to_merge_s": merged_at, "robots": {}}
    out_plot = {}
    for which in ("optimized", "initial"):
        path = os.path.join(best[1], "%s_global_pose_graph.g2o" % which)
        if not os.path.exists(path):
            continue
        V, E = read_g2o(path)
        keys = [k for k in V if gt_of(k) is not None]
        if not keys:
            ks = [k for k in V if k in st]
            for r in sorted({key_robot(k)[0] for k in V}):
                kr = [st[k] for k in ks if key_robot(k)[0] == r]
                print("%s graph, r%d: %d keyframes, %d with a stamp %s; ground truth %s" % (
                    which, r, sum(key_robot(k)[0] == r for k in V), len(kr),
                    "%.3f..%.3f" % (min(kr), max(kr)) if kr else "",
                    "%.3f..%.3f (%d)" % (gts[r][0][0], gts[r][0][-1], len(gts[r][0])) if r in gts else "none"))
            print("  first keys in the graph: %s; in pose_timestamps: %s" % (
                [hex(k) for k in list(V)[:3]], [hex(k) for k in list(st)[:3]]))
            raise SystemExit("no keyframe of %s matched to the ground truth (stamps within 10 ms)" % path)
        est = np.array([V[k][:3, 3] for k in keys])
        gt = np.array([gt_of(k)[:3, 3] for k in keys])
        rob = np.array([key_robot(k)[0] for k in keys])
        j = ate(est, gt)
        rep[which] = {"keyframes": len(V), "with_ground_truth": len(keys),
                      "joint_ate_rmse_m": j[0] if j else None, "joint_ate_median_m": j[1] if j else None}
        for r in sorted(set(rob)):
            a = ate(est[rob == r], gt[rob == r])
            rr = rep["robots"].setdefault(int(r), {})
            rr[which] = {"keyframes": int((rob == r).sum()),
                         "ate_rmse_m": a[0] if a else None, "ate_median_m": a[1] if a else None}
        if which == "optimized":
            lcs = []
            for a, b, Tm in E:
                ra, rb = key_robot(a)[0], key_robot(b)[0]
                if ra == rb:
                    continue
                Ga, Gb = gt_of(a), gt_of(b)
                if Ga is None or Gb is None:
                    lcs.append({"robots": [ra, rb], "checked": False})
                    continue
                D = np.linalg.inv(Tm) @ np.linalg.inv(Ga) @ Gb
                et, er = float(np.linalg.norm(D[:3, 3])), rot_deg(D[:3, :3])
                lcs.append({"robots": [ra, rb], "checked": True, "t_err_m": et, "r_err_deg": er,
                            "correct": bool(et <= lc_ok[0] and er <= lc_ok[1]),
                            "stamps": [st.get(a), st.get(b)]})
            ck = [x for x in lcs if x["checked"]]
            rep["inter_robot_loop_closures"] = {
                "total": len(lcs), "checked": len(ck), "correct": sum(x["correct"] for x in ck),
                "t_err_median_m": float(np.median([x["t_err_m"] for x in ck])) if ck else None,
                "r_err_median_deg": float(np.median([x["r_err_deg"] for x in ck])) if ck else None,
                "edges": lcs}
            out_plot = {"est": est, "gt": gt, "rob": rob, "T": j[2] if j else np.eye(4),
                        "edges": [(V[a][:3, 3], V[b][:3, 3]) for a, b, _ in E
                                  if key_robot(a)[0] != key_robot(b)[0]]}
    for r, (e, g) in odometry(export, gts, gt_at).items():
        a = ate(e, g)
        rep["robots"].setdefault(r, {})["odometry"] = {
            "frames": len(e), "ate_rmse_m": a[0] if a else None, "ate_median_m": a[1] if a else None}
    for r, s in logs:
        p = os.path.join(s, "log.csv")
        if os.path.exists(p):
            rep["robots"].setdefault(r, {})["log"] = read_log(p)        # the latest wins
    json.dump(rep, open(os.path.join(out, "cslam_eval.json"), "w"), indent=1)
    if plot and out_plot:
        draw(out_plot, gts, os.path.join(out, "cslam_eval.png"))
    return rep


def draw(d, gts, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 8))
    for r, (t, T) in sorted(gts.items()):
        ax.plot(T[:, 0, 3], T[:, 1, 3], color="0.75", lw=1.0, label="ground truth" if r == 0 else None)
    A = d["est"] @ d["T"][:3, :3].T + d["T"][:3, 3]
    for r in sorted(set(d["rob"])):
        m = d["rob"] == r
        ax.plot(A[m, 0], A[m, 1], ".", ms=3, label="r%d keyframes (joint alignment)" % r)
    for k, (a, b) in enumerate(d["edges"]):
        a, b = d["T"][:3, :3] @ a + d["T"][:3, 3], d["T"][:3, :3] @ b + d["T"][:3, 3]
        ax.plot([a[0], b[0]], [a[1], b[1]], "r-", lw=0.6, label="inter-robot loop closure" if k == 0 else None)
    ax.set_aspect("equal")
    ax.set_xlabel("map x (m)")
    ax.set_ylabel("map y (m)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def fmt(v, nd=3):
    return "-" if v is None else "%.*f" % (nd, v)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("export")
    ap.add_argument("--run", default=None, help="a run folder (runs/<channel>)")
    ap.add_argument("--results", default=None)
    ap.add_argument("--cov", nargs=2, action="append", default=[], metavar=("ROBOT", "FILE"))
    ap.add_argument("--sigma-max", type=float, default=0.15)
    ap.add_argument("--lc-ok", type=float, nargs=2, default=[0.5, 10.0], metavar=("M", "DEG"))
    a = ap.parse_args()
    export = os.path.abspath(os.path.expanduser(a.export))
    run = os.path.abspath(os.path.expanduser(a.run)) if a.run else None
    if run and not os.path.isdir(run):
        run = os.path.join(export, "runs", a.run)
    rep = evaluate(export, a.results or (os.path.join(run, "results") if run else None),
                   {int(r): f for r, f in a.cov}, a.sigma_max, tuple(a.lc_ok), out=run)
    print("graph: %s" % rep["graph"])
    for w in ("optimized", "initial"):
        if w in rep:
            g = rep[w]
            print("%-9s joint ATE rmse %s m, median %s m (%d of %d keyframes with ground truth)"
                  % (w, fmt(g["joint_ate_rmse_m"]), fmt(g["joint_ate_median_m"]), g["with_ground_truth"],
                     g["keyframes"]))
    for r, v in sorted(rep["robots"].items()):
        o, i, lg = v.get("optimized", {}), v.get("initial", {}), v.get("log", {})
        od = v.get("odometry", {})
        print("  r%d  ATE rmse %s m (initial %s m, odometry %s m over %d frames), %d keyframes; "
              "front-end bytes %s, matches %s"
              % (r, fmt(o.get("ate_rmse_m")), fmt(i.get("ate_rmse_m")), fmt(od.get("ate_rmse_m")),
                 od.get("frames", 0), o.get("keyframes", 0),
                 fmt(lg.get("total_front_end_cumulative_communication_bytes"), 0),
                 fmt(lg.get("total_nb_successful_matches"), 0)))
    lc = rep.get("inter_robot_loop_closures") or {}
    print("inter-robot loop closures: %s, %s checked against the ground truth, %s correct "
          "(median %s m, %s deg)" % (lc.get("total", 0), lc.get("checked", 0), lc.get("correct", 0),
                                     fmt(lc.get("t_err_median_m")), fmt(lc.get("r_err_median_deg"), 1)))
    print("time to merge: %s s into the run" % fmt(rep["time_to_merge_s"], 1))
    print("wrote %s/cslam_eval.json, cslam_eval.png" % (run or export))


if __name__ == "__main__":
    main()
