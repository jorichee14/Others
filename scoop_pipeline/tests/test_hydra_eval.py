#!/usr/bin/env python
"""Self-test for analysis/slam/hydra_eval.py on test_hydra_export's two-robot
export, with ground truth 1 m off the odometry (a pure shift) and Hydra's
saved trajectories: r0 on the ground truth with a small wobble, r1 on it
shifted 0.3 m, two loop closures for r0.

The odometry must read ATE 0 (a shift aligns away) and 1.0 m in map; r1's
Hydra ATE 0 and 0.3 m in map; r0's ATE the wobble; the loop closures
counted; a pose whose ground truth --cov marks uncertain left out; the
report, json and figure written in the run folder.

    python scoop_pipeline/tests/test_hydra_eval.py
"""
import json
import os
import shutil
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "datasets"))
sys.path.insert(0, os.path.join(ROOT, "analysis", "slam"))

import hydra as H                                                   # noqa: E402
import hydra_eval as HE                                             # noqa: E402
import mcap_convert as mc                                           # noqa: E402
import test_hydra_export as THE                                     # noqa: E402

SHIFT = np.array([1.0, 0.0, 0.0])


def gt_pose(i, k):
    T = THE.cam_pose(i, k).copy()
    T[:3, 3] += SHIFT
    return T


def stamp(i, k):
    return THE.T0 + k * 100_000_000 + i * 1_000_000


def main():
    f = []
    tmp = tempfile.mkdtemp()
    try:
        exp = os.path.join(tmp, "datasets", "swarm_slam", "p")
        THE.make_export(exp)
        for i in (0, 1):
            with open(os.path.join(exp, "gt_r%d.tum" % i), "w") as fh:
                for k in range(THE.N):
                    fh.write(mc.tum_line(stamp(i, k), gt_pose(i, k)) + "\n")
        out = H.convert(exp)["out"]
        run = os.path.join(out, "runs", "ideal")
        wob = 0.02
        for i in (0, 1):
            b = os.path.join(run, "hydra_out_r%d" % i, "backend")
            os.makedirs(b)
            with open(os.path.join(b, "trajectory.csv"), "w") as fh:
                fh.write("timestamp[ns],x,y,z,qw,qx,qy,qz\n")
                for k in range(THE.N):
                    p = gt_pose(i, k)[:3, 3] + ([0.0, wob * (-1) ** k, 0.0] if i == 0 else [0.0, 0.3, 0.0])
                    fh.write("%d,%f,%f,%f,1,0,0,0\n" % (stamp(i, k), p[0], p[1], p[2]))
            if i == 0:
                open(os.path.join(b, "loop_closures.csv"), "w").write(
                    "time_from_ns,time_to_ns,x,y,z,qw,qx,qy,qz,type,level\n1,2,0,0,0,1,0,0,0,0,0\n"
                    "3,4,0,0,0,1,0,0,0,0,0\n")
        # r1's ground truth uncertain at frame 3 (cov_<track>.csv: t, ..., sigma_xy_major_m in column 5)
        cov = os.path.join(tmp, "cov.csv")
        with open(cov, "w") as fh:
            fh.write("t,a,b,c,d,sigma_xy_major_m\n")
            for k in range(THE.N):
                fh.write("%.6f,0,0,0,0,%f\n" % (stamp(1, k) * 1e-9, 0.5 if k == 3 else 0.02))
        rep = HE.evaluate(out, run, {1: cov}, 0.15)
        r0, r1 = rep["robots"][0], rep["robots"][1]
        o0 = r0["odometry"]
        if not (o0["ate_rmse_m"] < 1e-4 and abs(o0["map_rmse_m"] - 1.0) < 1e-4 and o0["frames"] == THE.N):
            f.append("odometry r0 %s" % o0)
        if not (r1["hydra"]["ate_rmse_m"] < 1e-4 and abs(r1["hydra"]["map_rmse_m"] - 0.3) < 1e-4):
            f.append("hydra r1 %s" % r1["hydra"])
        if r1["hydra"]["with_ground_truth"] != THE.N - 1 or r1["odometry"]["frames"] != THE.N - 1:
            f.append("--cov did not leave out r1's uncertain frame: %s / %s" % (r1["hydra"], r1["odometry"]))
        if not (0.005 < r0["hydra"]["ate_rmse_m"] <= wob + 1e-6 and r0["hydra"]["loop_closures"] == 2):
            f.append("hydra r0 %s" % r0["hydra"])
        if r1["hydra"]["loop_closures"] is not None:
            f.append("r1 loop closures %s without a file" % r1["hydra"]["loop_closures"])
        if rep["joint"]["poses"] != 2 * THE.N - 1:
            f.append("joint %s" % rep["joint"])
        for n in ("hydra_eval.json", "hydra_eval.png"):
            if not os.path.exists(os.path.join(run, n)):
                f.append("no %s in the run" % n)
        if json.load(open(os.path.join(run, "hydra_eval.json")))["robots"]["0"]["hydra"]["loop_closures"] != 2:
            f.append("json")
        txt = HE.report(rep)
        oh = r1["odometry_at_hydra_poses"]
        if oh["poses"] != THE.N - 1 or not oh["ate_rmse_m"] < 1e-4 or abs(oh["map_rmse_m"] - 1.0) > 1e-4:
            f.append("odometry at hydra poses r1 %s" % oh)
        if "loop closures 2" not in txt or "r1  ATE rmse 0.000 m (odometry at the same 11 poses 0.000 m" not in txt:
            f.append("report:\n" + txt)
    except Exception:                                              # noqa: BLE001
        import traceback
        f.append(traceback.format_exc())
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("\n".join(f) or "hydra eval ok")
    sys.exit(1 if f else 0)


if __name__ == "__main__":
    main()
