#!/usr/bin/env python
"""Self-test for analysis/slam/relative_pose.py on a synthetic Swarm-SLAM run:
two robots moving 40 s, each robot's odometry perfect in an odom frame of its
own, keyframes every second. cslam's logs: a single-robot graph (no
inter-robot edge, so not merged), a merged graph whose newest keyframe is at
10 s with robot 1 placed 0.3 m off, and an exact merged graph at 20 s.

Before 10 s there must be no relative pose; from 10 to 20 s an error of
0.3 m and 0 deg (the odometry carries the poses between keyframes exactly);
from 20 s none. So: available 75 % of the run from 10 s, usable within 0.5 m
75 %, within 0.2 m 50 %.

    python scoop_pipeline/tests/test_relative_pose.py
"""
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
import test_cslam_eval as TC                                        # noqa: E402
import relative_pose as RP                                          # noqa: E402
import mcap_convert as mc                                           # noqa: E402
import swarm_slam as SS                                             # noqa: E402
from scoop.bagwrite import BagWriter                                # noqa: E402

S = 1_000_000_000


def truth(r, t):
    """Robot r's LiDAR in map at t s into the run."""
    return TC.rz(0.05 * t, [0.3 * t, 2.0 * r + 0.1 * t, 0.0])


def fixture(tmp):
    W = {0: TC.rz(0.4, [3.0, -1.0, 0.0]), 1: TC.rz(-1.1, [-2.0, 5.0, 0.2])}    # each robot's odom frame
    w = BagWriter(os.path.join(tmp, "bag"))
    for k in range(400):
        t = k * 0.1
        ns = int(round((TC.T0 + t) * S))
        for r in (0, 1):
            w.write("/r%d/odom" % r, SS.odometry(np.linalg.inv(W[r]) @ truth(r, t), "r%d/odom" % r,
                                                 "r%d/lidar" % r, ns), ns + r)
    w.close()
    for r in (0, 1):
        mc.write_tum(os.path.join(tmp, "gt_r%d.tum" % r),
                     [mc.tum_line(int(round((TC.T0 + k * 0.1) * S)), truth(r, k * 0.1)) for k in range(400)])
    G = TC.rz(0.9, [10.0, 4.0, 1.0])                               # the graph's frame
    shift = TC.rz(0.0, [0.3, 0.0, 0.0])

    def graph(upto, off):
        V = {TC.key(r, i): (off if r == 1 else np.eye(4)) @ np.linalg.inv(G) @ truth(r, i)
             for r in (0, 1) for i in range(upto + 1)}
        E = [(TC.key(r, i), TC.key(r, i + 1), np.eye(4)) for r in (0, 1) for i in range(upto)]
        E.append((TC.key(0, upto), TC.key(1, upto), np.eye(4)))
        return V, E

    run = os.path.join(tmp, "runs", "ideal")
    d0 = os.path.join(run, "results", "x_experiment_robot_0")
    for n, (name, upto, off, merged) in enumerate([("g1", 4, np.eye(4), False), ("g2", 10, shift, True),
                                                    ("g3", 20, np.eye(4), True)]):
        d = os.path.join(d0, name)
        os.makedirs(d)
        V, E = graph(upto, off)
        if not merged:
            V = {k: v for k, v in V.items() if CE_robot(k) == 0}
            E = [e for e in E if CE_robot(e[0]) == 0 and CE_robot(e[1]) == 0]
        TC.write_g2o(os.path.join(d, "optimized_global_pose_graph.g2o"), V, E)
        os.utime(d, (1000 + n, 1000 + n))
    for r in (0, 1):
        d = os.path.join(run, "results", "x_experiment_robot_%d" % r, "stamps")
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "pose_timestamps%d.csv" % r), "w") as fh:
            fh.write("vertice_id,sec,nanosec\n")
            for i in range(40):
                t = TC.T0 + i
                fh.write("%d,%d,%d\n" % (TC.key(r, i), int(t), round((t - int(t)) * 1e9)))
    return run


def CE_robot(k):
    return RP.CE.key_robot(k & RP.CE.KEY_MASK)[0]


def main():
    tmp = tempfile.mkdtemp()
    f = []
    try:
        run = fixture(tmp)
        Sx, dur = RP.series(tmp, run, step=1.0)
        t, tr, rot = Sx[:, 0], Sx[:, 1], Sx[:, 2]
        if not np.isnan(tr[t < 9.5]).all():
            f.append("a relative pose before the first merged graph")
        mid = (t >= 10) & (t < 19.5)
        if np.abs(tr[mid] - 0.3).max() > 1e-3 or rot[mid].max() > 1e-3:
            f.append("10-20 s: %.4f m / %.4f deg, want 0.3 m / 0" % (np.nanmax(np.abs(tr[mid] - 0.3)), np.nanmax(rot[mid])))
        late = t >= 20
        if np.nanmax(tr[late]) > 1e-3 or np.nanmax(rot[late]) > 1e-3:
            f.append("after 20 s: %.4f m / %.4f deg, want 0" % (np.nanmax(tr[late]), np.nanmax(rot[late])))
        rep = RP.summarize(Sx, dur, 20.0, (0.5, 2.0))
        if rep["available_pct"] != 75.0 or rep["available_from_s"] != 10.0 or rep["usable"]["pct_of_run"] != 75.0:
            f.append("summary %s" % rep)
        if RP.summarize(Sx, dur, 20.0, (0.2, 2.0))["usable"]["pct_of_run"] != 50.0:
            f.append("usable within 0.2 m: want 50 %")
        out = os.path.join(tmp, "rp.png")
        RP.plot([("ideal", Sx)], out)
        if not os.path.getsize(out):
            f.append("no plot")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    if f:
        print("FAIL\n  " + "\n  ".join(f))
        sys.exit(1)
    print("relative pose ok")


if __name__ == "__main__":
    main()
