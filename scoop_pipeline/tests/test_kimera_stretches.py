#!/usr/bin/env python
"""Self-test for slam/kimera_stretches.py and analysis/slam/kimera_eval.py on a
made-up export: pairs at 14 Hz, IMU at 200 Hz and ground truth at 10 Hz for
100 s, all three out at 30-35 s (a dropout) and the IMU alone at 60-62 s.

  stretches  three: from pair 50 to just before 30 s, from 35.3 s to just
             before 60 s, from 62.3 s to the end (each ending 50 ms before the
             first of the three to stop); with --min 30 only the last
  eval       stretch 1 on the ground truth (ATE 0, in map 0), stretch 2
             shifted 0.5 m (ATE 0, in map 0.5), stretch 3's backend failed
             with no odometry; over all runs: the two, 0.25 m in map ignored
             by the ATE (each run aligned on its own), json written; the
             whole-run layout (out/ itself) read as one run

    python scoop_pipeline/tests/test_kimera_stretches.py
"""
import json
import os
import shutil
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "slam"))
sys.path.insert(0, os.path.join(ROOT, "analysis", "slam"))
import kimera_stretches as KS                                       # noqa: E402
import kimera_eval as KE                                            # noqa: E402

T0 = 1_790_000_000_000_000_000


def ns(t):
    return (T0 + np.round(np.asarray(t) * 1e9)).astype(np.int64)


def gt_pos(t):
    t = np.asarray(t, float)
    return np.c_[0.5 * t, np.sin(0.1 * t), 0 * t]


def make(d):
    for s in ("cam0", "imu0", "state_groundtruth_estimate0"):
        os.makedirs(os.path.join(d, "mav0", s))
    tc = np.arange(0, 100, 1 / 14)
    tc = tc[(tc < 30) | (tc >= 35)]
    tu = np.arange(0, 100, 0.005)
    tu = tu[((tu < 30) | (tu >= 35)) & ((tu < 60) | (tu >= 62))]
    tg = np.arange(0, 100, 0.1)
    tg = tg[(tg < 30) | (tg >= 35)]
    with open(os.path.join(d, "mav0", "cam0", "data.csv"), "w") as fh:
        fh.write("#timestamp [ns],filename\n")
        for t in ns(tc):
            fh.write("%d,%d.png\n" % (t, t))
    np.savetxt(os.path.join(d, "mav0", "imu0", "data.csv"), np.c_[ns(tu), np.zeros((len(tu), 6))],
               fmt=["%d"] + ["%.1f"] * 6, delimiter=",", header="t")
    p = gt_pos(tg)
    np.savetxt(os.path.join(d, "mav0", "state_groundtruth_estimate0", "data.csv"),
               np.c_[ns(tg), p, np.tile([1, 0, 0, 0], (len(tg), 1)), np.zeros((len(tg), 9))],
               fmt=["%d"] + ["%.6f"] * 16, delimiter=",", header="t")
    return tc


def write_traj(path, t, P):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        fh.write("#timestamp,x,y,z,qw,qx,qy,qz,vx,vy,vz,bgx,bgy,bgz,bax,bay,baz\n")
        for a, p in zip(ns(t), P):
            fh.write("%d,%f,%f,%f,1,0,0,0,0,0,0,0,0,0,0,0,0\n" % (a, p[0], p[1], p[2]))


def main():
    f = []
    tmp = tempfile.mkdtemp()
    try:
        d = os.path.join(tmp, "mobile_2")
        tc = make(d)
        s = KS.stretches(d)
        want = [(50 / 14, 30), (35.3, 60), (62.3, 100)]
        if len(s) != 3 or any(abs(t0 - a) > 0.08 or not (b - 0.25 < t1 < b - 0.05) for (_, _, t0, t1), (a, b)
                              in zip(s, want)):
            f.append("stretches %s, want about %s" % (s, want))
        if s and abs(tc[s[1][0]] - s[1][2]) > 1e-6:
            f.append("stretch 2's k0 is not the pair at its t0")
        if len(KS.stretches(d, min_s=30)) != 1:
            f.append("--min 30: %s" % KS.stretches(d, min_s=30))

        out = os.path.join(d, "out")
        t1 = np.arange(4, 29, 0.2)
        write_traj(os.path.join(out, "stretch_1", "traj_vio.csv"), t1, gt_pos(t1))
        t2 = np.arange(36, 59, 0.2)
        write_traj(os.path.join(out, "stretch_2", "traj_vio.csv"), t2, gt_pos(t2) + [0.5, 0, 0])
        open(os.path.join(out, "stretch_2", "output_lcd_result.csv"), "w").write(
            "#timestamp_kf,a,b,isLoop\n1,2,3,1\n1,2,3,0\n1,2,3,1\n")
        os.makedirs(os.path.join(out, "stretch_3"))
        open(os.path.join(out, "stretch_3", "kimera.log"), "w").write("E... ERROR: Variable has type 'x' and index 4\n")
        rep = KE.evaluate(d)
        r1, r2, r3 = rep["runs"]
        if not (r1["traj_vio"]["ate_rmse_m"] < 1e-4 and r1["traj_vio"]["in_map_rmse_m"] < 1e-4):
            f.append("stretch 1 %s" % r1["traj_vio"])
        if not (r2["traj_vio"]["ate_rmse_m"] < 1e-4 and abs(r2["traj_vio"]["in_map_rmse_m"] - 0.5) < 1e-4
                and r2["loop_closures"] == 2):
            f.append("stretch 2 %s, loop closures %s" % (r2["traj_vio"], r2["loop_closures"]))
        if not (r3["failed"] and r3["traj_vio"] is None):
            f.append("stretch 3 %s" % r3)
        if rep["traj_vio"]["poses"] != len(t1) + len(t2) or rep["traj_vio"]["ate_rmse_m"] > 1e-4 \
                or rep["traj_vio"]["runaways"] != 0 or rep["traj_vio"]["runs"] != 2:
            f.append("over all runs %s" % rep["traj_vio"])
        txt = KE.report(rep)
        if "backend failed" not in txt or "over all runs" not in txt:
            f.append("report:\n" + txt)
        if not os.path.exists(os.path.join(out, "kimera_eval.json")) or \
                json.load(open(os.path.join(out, "kimera_eval.json")))["runs"][1]["loop_closures"] != 2:
            f.append("json")
        shutil.rmtree(out)
        write_traj(os.path.join(out, "traj_vio.csv"), t1, gt_pos(t1) + [0, 0.2, 0])
        rep = KE.evaluate(d)
        if len(rep["runs"]) != 1 or rep["runs"][0]["name"] != "whole run" \
                or abs(rep["runs"][0]["traj_vio"]["in_map_rmse_m"] - 0.2) > 1e-4:
            f.append("whole run %s" % rep["runs"])
    except Exception:                                              # noqa: BLE001
        import traceback
        f.append(traceback.format_exc())
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("\n".join(f) or "kimera stretches ok")
    sys.exit(1 if f else 0)


if __name__ == "__main__":
    main()
