#!/usr/bin/env python3
"""Kimera-VIO (slam/run_kimera.sh) against the ground truth of its
datasets/euroc.py export: one run over the whole data, or one per stretch
without a dropout (STRETCHES=1).

    python3 scoop_pipeline/analysis/slam/kimera_eval.py <export>/<robot>

Reads <robot>/out/ (or out/stretch_<n>/ each, with out/stretches.txt):
traj_vio.csv (odometry) and traj_pgo.csv (with its loop closures), both
"#timestamp,x,y,z,qw,qx,qy,qz,vx,..." of the IMU in Kimera-VIO's world, and
output_lcd_result.csv; and mav0/state_groundtruth_estimate0 (the IMU in map),
interpolated at every pose that has ground truth on both sides within 0.5 s.

Per run and trajectory:
  ATE    rigid alignment of that run alone (rmse, m)
  in map no alignment at all: the run starts from the ground truth at its first
         frame (euroc.py's autoInitialize 0), so this is its drift from there
  path   its path length against the ground truth's, over the same poses
Over the stretches: the rmse of every pose of every run (each aligned on its
own), the median of the runs' ATE, how many ran away (ATE over 5 m) and the
seconds covered. Writes out/kimera_eval.json.
"""
import argparse
import glob
import json
import os

import numpy as np

RUNAWAY_M = 5.0            # a run's ATE above this: it ran away (one such hides every other in the rmse)


def load(path):
    if not os.path.exists(path):
        return None
    a = np.loadtxt(path, delimiter=",", comments="#", ndmin=2)
    return a if len(a) else None


def gt_at(g, t, max_gap=0.5e9):
    """Ground-truth positions at stamps t, and which have ground truth either side within max_gap."""
    i = np.searchsorted(g[:, 0], t)
    ok = (i > 0) & (i < len(g))
    ii = np.clip(i, 1, len(g) - 1)
    ok &= (g[ii, 0] - g[ii - 1, 0] <= max_gap) | (g[ii, 0] == t) | (g[ii - 1, 0] == t)
    P = np.stack([np.interp(t, g[:, 0], g[:, k]) for k in (1, 2, 3)], 1)
    return P, ok


def rigid(E, G):
    me, mg = E.mean(0), G.mean(0)
    U, _, Vt = np.linalg.svd((E - me).T @ (G - mg))
    R = (U @ np.diag([1, 1, np.linalg.det(U @ Vt)]) @ Vt).T
    return np.linalg.norm((E - me) @ R.T + mg - G, axis=1)


def path(P):
    return float(np.sum(np.linalg.norm(np.diff(P, axis=0), axis=1))) if len(P) > 1 else 0.0


def score(traj, g, t0):
    G, ok = gt_at(g, traj[:, 0])
    E, G, t = traj[ok, 1:4], G[ok], traj[ok, 0]
    if len(E) < 3:
        return None
    e, m = rigid(E, G), np.linalg.norm(E - G, axis=1)
    return {"poses": int(len(traj)), "with_ground_truth": int(len(E)),
            "from_s": float((t[0] - t0) * 1e-9), "to_s": float((t[-1] - t0) * 1e-9),
            "ate_rmse_m": float(np.sqrt(np.mean(e ** 2))), "in_map_rmse_m": float(np.sqrt(np.mean(m ** 2))),
            "in_map_end_m": float(m[-1]), "path_m": path(E), "gt_path_m": path(G), "_err": e}


def evaluate(robot_dir):
    m0 = os.path.join(robot_dir, "mav0")
    t0 = int(np.loadtxt(os.path.join(m0, "cam0", "data.csv"), delimiter=",", comments="#",
                        dtype=np.int64, usecols=0, ndmin=1)[0])
    g = np.loadtxt(os.path.join(m0, "state_groundtruth_estimate0", "data.csv"), delimiter=",", comments="#",
                   ndmin=2)
    out = os.path.join(robot_dir, "out")
    dirs = sorted(glob.glob(os.path.join(out, "stretch_*")), key=lambda p: int(p.rsplit("_", 1)[1]))
    runs = [(os.path.basename(d), d) for d in dirs] or [("whole run", out)]
    total = (int(np.loadtxt(os.path.join(m0, "cam0", "data.csv"), delimiter=",", comments="#", dtype=np.int64,
                            usecols=0, ndmin=1)[-1]) - t0) * 1e-9
    rep = {"robot_dir": os.path.abspath(robot_dir), "data_s": total, "runs": []}
    errs = {"traj_vio": [], "traj_pgo": []}
    for name, d in runs:
        r = {"name": name, "failed": False, "loop_closures": None}
        log = os.path.join(d, "kimera.log")
        if os.path.exists(log):
            r["failed"] = "Variable has type" in open(log, errors="replace").read()
        lcd = os.path.join(d, "output_lcd_result.csv")
        if os.path.exists(lcd):
            r["loop_closures"] = sum(1 for ln in open(lcd)
                                    if not ln.startswith("#") and ln.strip().split(",")[3:4] == ["1"])
        for k in ("traj_vio", "traj_pgo"):
            tr = load(os.path.join(d, k + ".csv"))
            s = score(tr, g, t0) if tr is not None else None
            if s:
                errs[k].append(s.pop("_err"))
            r[k] = s
        rep["runs"].append(r)
    for k, e in errs.items():
        if e:
            e = np.concatenate(e)
            per = [r[k]["ate_rmse_m"] for r in rep["runs"] if r.get(k)]
            rep[k] = {"poses": int(len(e)), "ate_rmse_m": float(np.sqrt(np.mean(e ** 2))),
                      "median_run_ate_m": float(np.median(per)), "runaways": int(sum(a > RUNAWAY_M for a in per)),
                      "runs": len(per),
                      "covered_s": float(sum(r[k]["to_s"] - r[k]["from_s"] for r in rep["runs"] if r.get(k)))}
    if os.path.isdir(out):
        json.dump(rep, open(os.path.join(out, "kimera_eval.json"), "w"), indent=1)
    return rep


def report(rep):
    f = lambda x: "-" if x is None else "%.2f" % x
    lines = ["%-10s %-13s %6s  %8s  %8s  %8s  %9s  %s" % ("run", "s", "poses", "ATE m", "in map", "path m",
                                                          "gt path", "loop closures")]
    for r in rep["runs"]:
        v = r.get("traj_vio")
        if not v:
            lines.append("%-10s no odometry%s" % (r["name"], " (backend failed)" if r["failed"] else ""))
            continue
        p = r.get("traj_pgo") or {}
        lines.append("%-10s %5.1f-%5.1f %6d  %8s  %8s  %8.1f  %9.1f  %s%s" % (
            r["name"], v["from_s"], v["to_s"], v["poses"], f(v["ate_rmse_m"]), f(v["in_map_rmse_m"]), v["path_m"],
            v["gt_path_m"], r["loop_closures"] if r["loop_closures"] is not None else "-",
            "  (backend failed)" if r["failed"] else ""))
        if p:
            lines.append("%-10s %13s %6s  %8s  %8s   with loop closures" % ("", "", "", f(p["ate_rmse_m"]),
                                                                          f(p["in_map_rmse_m"])))
    for k, label in (("traj_vio", "odometry"), ("traj_pgo", "with loop closures")):
        if k in rep:
            x = rep[k]
            lines.append("%s over all runs: ATE rmse %.2f m (%d poses, %.0f of %.0f s covered); median of the "
                         "runs %.2f m, %d of %d ran away (over %g m)"
                         % (label, x["ate_rmse_m"], x["poses"], x["covered_s"], rep["data_s"], x["median_run_ate_m"],
                            x["runaways"], x["runs"], RUNAWAY_M))
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("robot_dir", help="<euroc export>/<robot> (with mav0/ and out/)")
    a = ap.parse_args()
    print(report(evaluate(a.robot_dir)))


if __name__ == "__main__":
    main()
