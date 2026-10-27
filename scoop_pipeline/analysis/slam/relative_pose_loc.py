#!/usr/bin/env python3
"""The relative pose between two robots that each localize in the same map
(slam/odometry/run_odometry.sh fast2lamaa_loc, one run per robot on the same
pass), the way relative_pose.py measures it for Swarm-SLAM, so the two compare
line for line:

    python scoop_pipeline/analysis/slam/relative_pose_loc.py <run robot 0> <run robot 1>
        [--step 1.0] [--range 20] [--usable 0.5 2.0] [--out <png>]

Each run's trajectory.tum is already in map (run.json "world": "map"): at a
time t, P_i(t) is robot i's latest pose there (its LiDAR, through
make_config.to_lidar), held by nothing else, and P_0(t)^-1 P_1(t) is compared
with the reference's G_0(t)^-1 G_1(t) (stage 04's tracks, as score.py uses):
the translation error (m) and rotation error (deg) of where robot 0 places
robot 1. No alignment, no shared graph: the map is the shared frame from the
first scan.

Prints and writes <run robot 0>/relative_pose_loc.json and .csv: available
(share of the run with both poses), error median / p95 / max, a point --range
m away, usable (within --usable m, deg), the same fields as relative_pose.py.
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "slam", "odometry"))
import relative_pose as RP                                          # noqa: E402


def track(run, max_gap_s=0.5):
    """(the run's estimate as a PoseInterp of T_map_lidar, its reference TUM, its machine)."""
    import make_config as MK
    rj = json.load(open(os.path.join(run, "run.json")))
    if rj.get("world") != "map":
        raise SystemExit(f"{run}: not a run in map (run.json world {rj.get('world')!r}): "
                         f"relative_pose_loc compares localizer runs")
    paths = MK.mc._paths(rj["bag"])
    T_fl, _ = MK.to_lidar(paths, MK.mc.build_tf_static(paths), rj["machine"], rj["tum_frame"])
    a = np.loadtxt(os.path.join(run, "trajectory.tum"), ndmin=2)
    from scipy.spatial.transform import Rotation
    T = np.tile(np.eye(4), (len(a), 1, 1))
    T[:, :3, :3] = Rotation.from_quat(a[:, 4:8]).as_matrix()
    T[:, :3, 3] = a[:, 1:4]
    E = T @ T_fl
    ts = np.round(a[:, 0] * 1e9).astype(np.int64)
    o = np.argsort(ts)
    ts, E = ts[o], E[o]
    keep = np.concatenate([[True], np.diff(ts) > 0])
    ts, E = ts[keep], E[keep]
    est = MK.mc.PoseInterp(ts, E[:, :3, 3], Rotation.from_matrix(E[:, :3, :3]).as_quat(), max_gap_s=max_gap_s)
    ref = MK.default_ref(rj["bag"], rj["machine"])
    return est, ref, rj["machine"]


def series(est, gts, step=1.0):
    """[(t, translation error m, rotation error deg)] from the reference's
    common span (NaN where either robot has no pose), and its length."""
    t0 = max(g.ts[0] for g in gts) * 1e-9
    t1 = min(g.ts[-1] for g in gts) * 1e-9
    out = []
    for t in np.arange(t0, t1, step):
        n = int(round(t * 1e9))
        P = [e.at(n) for e in est]
        G = [g.at(n) for g in gts]
        row = [float(t - t0), np.nan, np.nan]
        if all(x is not None for x in P + G):
            D = np.linalg.inv(np.linalg.inv(G[0]) @ G[1]) @ (np.linalg.inv(P[0]) @ P[1])
            row[1], row[2] = float(np.linalg.norm(D[:3, 3])), RP.CE.rot_deg(D[:3, :3])
        out.append(row)
    return np.array(out), t1 - t0


def report(name, rep):
    lines = ["%s: relative pose available %.1f %% of %.0f s (from %s s)"
             % (name, rep["available_pct"], rep["duration_s"],
                "-" if rep["available_from_s"] is None else "%.1f" % rep["available_from_s"])]
    if "translation_m" in rep:
        t, r, a, u = rep["translation_m"], rep["rotation_deg"], rep["at_range_m"], rep["usable"]
        lines += ["  robot-to-robot position median %.3f / p95 %.3f / max %.3f m; rotation median %.2f / p95 %.2f / "
                  "max %.2f deg" % (t["median"], t["p95"], t["max"], r["median"], r["p95"], r["max"]),
                  "  a point %g m away moved by: median %.3f / p95 %.3f m" % (rep["range_m"], a["median"], a["p95"]),
                  "  usable (within %g m, %g deg): %.1f %% of the run"
                  % (u["within_m"], u["within_deg"], u["pct_of_run"])]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run0")
    ap.add_argument("run1")
    ap.add_argument("--step", type=float, default=1.0)
    ap.add_argument("--range", type=float, default=20.0)
    ap.add_argument("--usable", type=float, nargs=2, default=[0.5, 2.0], metavar=("M", "DEG"))
    ap.add_argument("--out", default=None, help="plot of the error over time (png)")
    a = ap.parse_args()
    tracks = [track(r) for r in (a.run0, a.run1)]
    mc = RP.CE._mc()
    gts = [mc.PoseInterp.from_tum(ref, max_gap_s=1.0) for _, ref, _ in tracks]
    S, dur = series([e for e, _, _ in tracks], gts, a.step)
    rep = RP._summary(S, dur, a.range, tuple(a.usable))
    rep["robots"] = [m for _, _, m in tracks]
    name = "%s + %s, localized in map" % tuple(rep["robots"])
    print(report(name, rep))
    json.dump(rep, open(os.path.join(a.run0, "relative_pose_loc.json"), "w"), indent=1)
    np.savetxt(os.path.join(a.run0, "relative_pose_loc.csv"), S, delimiter=",", fmt="%.4f",
               header="t_s,translation_m,rotation_deg", comments="")
    if a.out:
        RP.plot([(name, S)], a.out)
        print("wrote", a.out)


if __name__ == "__main__":
    main()
