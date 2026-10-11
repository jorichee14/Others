#!/usr/bin/env python3
"""Hydra against the ground truth, the way analysis/slam/cslam_eval.py does
Swarm-SLAM: a datasets/hydra.py export and the output Hydra saved for each
robot (log_path; slam/run_hydra.sh puts it in runs/<channel>/).

    python scoop_pipeline/analysis/slam/hydra_eval.py <hydra export dir> [--run runs/<channel>]
        [--cov ROBOT FILE] [--sigma-max 0.15]

Reads, per robot i: <run>/hydra_out_r<i>/backend/trajectory.csv (Hydra's
optimized trajectory: timestamp[ns],x,y,z,qw,qx,qy,qz of r<i>/base, which
sits at the colour camera, in map) and loop_closures.csv next to it;
<export>/gt_r<i>.tum (the colour camera in map, /<robot>/global_pose); and
<export>/bag_r<i>'s /tf (odom -> base: the odometry Hydra was given).
<run> defaults to the export itself (hydra_out_r<i> next to the bags).

Reported, with cslam_eval's alignment and ground-truth matching (a pose
counts when the ground truth has a frame within 10 ms, and --cov's sigma
there is at most --sigma-max):
  ATE per robot   Hydra's poses aligned to the ground truth on their own
                  (rigid, no scale): RMSE and median of the position error
  joint ATE       one alignment for both robots: whether their maps agree
  in map          every error also without any alignment: each robot's
                  odometry starts in map at its session anchor
  odometry        the same for the odometry Hydra was given, at Hydra's own
                  poses (the like-for-like baseline: Hydra keeps a pose every
                  few seconds, and an ATE over fewer poses differs by itself)
                  and over every frame
  loop closures   how many Hydra found (backend/loop_closures.csv)
Hydra keeps one map per robot (no inter-robot loop closures: that is
Hydra-Multi), so there is no time to merge.

Writes hydra_eval.json and hydra_eval.png (top view: ground truth, Hydra
per robot, jointly aligned) in <run>.
"""
import argparse
import csv
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cslam_eval as CE                                             # noqa: E402


def read_trajectory(path):
    """(stamps s, positions (N, 3)) of Hydra's backend/trajectory.csv."""
    rows = list(csv.reader(open(path)))
    if not rows:
        return np.zeros(0), np.zeros((0, 3))
    body = [r for r in rows[1:] if len(r) >= 4]
    t = np.array([int(float(r[0])) * 1e-9 for r in body])
    p = np.array([[float(r[1]), float(r[2]), float(r[3])] for r in body]).reshape(-1, 3)
    return t, p


def count_rows(path):
    if not os.path.exists(path):
        return None
    return max(0, sum(1 for ln in open(path) if ln.strip()) - 1)


def tf_odometry(bag, i):
    """(stamps s, positions (N, 3)) of r<i>/odom -> r<i>/base in a bag's /tf."""
    sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
    from scoop import bag as B, rosmsg
    ts = rosmsg.typestore()
    t, p = [], []
    for _, _, _, msg in B.iter_mcap_records(B.open_bag(bag), topics=["/tf"]):
        for x in ts.deserialize_cdr(msg.data, "tf2_msgs/msg/TFMessage").transforms:
            if x.header.frame_id == "r%d/odom" % i and x.child_frame_id == "r%d/base" % i:
                s = x.header.stamp
                t.append(s.sec + s.nanosec * 1e-9)
                tr = x.transform.translation
                p.append((tr.x, tr.y, tr.z))
    return np.array(t), np.array(p).reshape(-1, 3)


def evaluate(export, run=None, cov=None, sigma_max=0.15, plot=True):
    export = os.path.abspath(os.path.expanduser(export))
    run = os.path.abspath(os.path.expanduser(run)) if run else export
    gts = {int(os.path.basename(p)[4:-4]): CE.read_tum(p)
           for p in sorted(glob.glob(os.path.join(export, "gt_r*.tum")))}
    if not gts:
        raise SystemExit("no gt_r<i>.tum in %s (datasets/hydra.py copies them)" % export)
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

    def matched(r, t, p):
        e, g = [], []
        for s, x in zip(t, p):
            G = gt_at(r, s)
            if G is not None:
                e.append(x)
                g.append(G[:3, 3])
        return np.array(e).reshape(-1, 3), np.array(g).reshape(-1, 3)

    rep = {"export": export, "run": run, "robots": {}}
    est_all, gt_all, rob_all = [], [], []
    for r in sorted(gts):
        rr = rep["robots"][r] = {}
        traj = os.path.join(run, "hydra_out_r%d" % r, "backend", "trajectory.csv")
        bag = os.path.join(export, "bag_r%d" % r)
        to, po = tf_odometry(bag, r) if os.path.isdir(bag) else (np.zeros(0), np.zeros((0, 3)))
        if os.path.exists(traj):
            t, p = read_trajectory(traj)
            # the odometry at Hydra's poses (frame stamps: nearest within 10 ms)
            keep, po_at = [], []
            for k, s in enumerate(t):
                if len(to):
                    j = int(np.argmin(np.abs(to - s)))
                    if abs(to[j] - s) <= 0.01:
                        keep.append(k)
                        po_at.append(po[j])
            if keep:
                e, g = matched(r, t[keep], np.array(po_at))
                a, am = CE.ate(e, g), CE.map_err(e, g)
                rr["odometry_at_hydra_poses"] = {
                    "poses": len(e), "ate_rmse_m": a[0] if a else None, "ate_median_m": a[1] if a else None,
                    "map_rmse_m": am[0] if am else None, "map_median_m": am[1] if am else None}
            e, g = matched(r, t, p)
            a, am = CE.ate(e, g), CE.map_err(e, g)
            rr["hydra"] = {"poses": len(t), "with_ground_truth": len(e),
                           "ate_rmse_m": a[0] if a else None, "ate_median_m": a[1] if a else None,
                           "map_rmse_m": am[0] if am else None, "map_median_m": am[1] if am else None,
                           "loop_closures": count_rows(os.path.join(os.path.dirname(traj), "loop_closures.csv"))}
            est_all.append(e)
            gt_all.append(g)
            rob_all.append(np.full(len(e), r))
        else:
            rr["hydra"] = None
        if len(to):
            e, g = matched(r, to, po)
            a, am = CE.ate(e, g), CE.map_err(e, g)
            rr["odometry"] = {"frames": len(e), "ate_rmse_m": a[0] if a else None,
                              "ate_median_m": a[1] if a else None,
                              "map_rmse_m": am[0] if am else None, "map_median_m": am[1] if am else None}
    if not est_all:
        raise SystemExit("no hydra_out_r<i>/backend/trajectory.csv under %s (Hydra saves it when it stops)" % run)
    est, gt, rob = np.vstack(est_all), np.vstack(gt_all), np.concatenate(rob_all)
    j, jm = CE.ate(est, gt), CE.map_err(est, gt)
    rep["joint"] = {"poses": len(est), "ate_rmse_m": j[0] if j else None, "ate_median_m": j[1] if j else None,
                    "map_rmse_m": jm[0] if jm else None, "map_median_m": jm[1] if jm else None}
    json.dump(rep, open(os.path.join(run, "hydra_eval.json"), "w"), indent=1)
    if plot and j:
        draw(est, rob, j[2], gts, os.path.join(run, "hydra_eval.png"))
    return rep


def draw(est, rob, T, gts, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 8))
    for r, (t, G) in sorted(gts.items()):
        ax.plot(G[:, 0, 3], G[:, 1, 3], color="0.75", lw=1.0, label="ground truth" if r == min(gts) else None)
    A = est @ T[:3, :3].T + T[:3, 3]
    for r, c in zip(sorted(set(rob)), ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")):
        m = rob == r
        ax.plot(A[m, 0], A[m, 1], color=c, lw=2, label="r%d Hydra (joint alignment)" % r)
    ax.set_aspect("equal")
    ax.set_xlabel("map x (m)")
    ax.set_ylabel("map y (m)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def report(rep):
    f = CE.fmt
    j = rep["joint"]
    lines = ["joint ATE rmse %s m, median %s m; in map rmse %s m (%d poses with ground truth)"
             % (f(j["ate_rmse_m"]), f(j["ate_median_m"]), f(j["map_rmse_m"]), j["poses"])]
    for r, d in sorted(rep["robots"].items()):
        h, o, oh = d.get("hydra"), d.get("odometry") or {}, d.get("odometry_at_hydra_poses") or {}
        if not h:
            lines.append("  r%d  no Hydra trajectory" % r)
            continue
        lines.append("  r%d  ATE rmse %s m (odometry at the same %s poses %s m; over all %s frames %s m); "
                     "in map %s m (odometry %s m); loop closures %s"
                     % (r, f(h["ate_rmse_m"]), oh.get("poses", "-"), f(oh.get("ate_rmse_m")), o.get("frames", "-"),
                        f(o.get("ate_rmse_m")), f(h["map_rmse_m"]), f(oh.get("map_rmse_m")),
                        h["loop_closures"] if h["loop_closures"] is not None else "-"))
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("export", help="a datasets/hydra.py export (bag_r<i>, gt_r<i>.tum)")
    ap.add_argument("--run", default=None, help="a run folder (runs/<channel>; default: the export)")
    ap.add_argument("--cov", nargs=2, action="append", default=[], metavar=("ROBOT", "FILE"))
    ap.add_argument("--sigma-max", type=float, default=0.15)
    a = ap.parse_args()
    export = os.path.abspath(os.path.expanduser(a.export))
    run = a.run
    if run and not os.path.isdir(run):
        run = os.path.join(export, "runs", run)
    rep = evaluate(export, run, {int(r): f for r, f in a.cov}, a.sigma_max)
    print(report(rep))
    print("  wrote %s" % os.path.join(rep["run"], "hydra_eval.json"))


if __name__ == "__main__":
    main()
