#!/usr/bin/env python
"""Self-test for analysis/slam/cslam_eval.py on synthetic cslam logs.

Two robots' ground truth (gt_r<i>.tum); cslam's global graph is the same
keyframes in another frame with 2 cm of noise, except robot 1's frame is off
by a further 1 m in the initial graph (before the inter-robot loop closure).
Two inter-robot edges: one true, one 2 m wrong. Two logs: an earlier graph
with no inter-robot edge and a later one with them.

It must: decode gtsam keys to robots; find per-robot and joint ATE of about
2 cm for the optimized graph and a joint ATE near 0.5 m for the initial one;
flag exactly one of the two loop closures as correct; date the merge by the
latest keyframe of the first graph with an inter-robot edge; drop the
keyframe whose ground truth uncertainty is over --sigma-max.

    python scoop_pipeline/tests/test_cslam_eval.py
"""
import os
import shutil
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "analysis", "slam"))
import cslam_eval as CE                                             # noqa: E402

T0 = 1_790_000_000.0


def key(r, i):
    return (ord("g") << 56) | ((ord("A") + r) << 48) | i


def xyzq(T):
    from scipy.spatial.transform import Rotation
    return list(T[:3, 3]) + list(Rotation.from_matrix(T[:3, :3]).as_quat())


def rz(a, t):
    T = np.eye(4)
    T[:3, :3] = [[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]]
    T[:3, 3] = t
    return T


def write_g2o(path, V, E):
    """As gtsam's writeG2o: keys without the symbol's 'g' (its top byte)."""
    m = (1 << 56) - 1
    with open(path, "w") as f:
        for k, T in V.items():
            f.write("VERTEX_SE3:QUAT %d %s\n" % (k & m, " ".join("%.9f" % v for v in xyzq(T))))
        for a, b, T in E:
            f.write("EDGE_SE3:QUAT %d %d %s %s\n" % (a & m, b & m, " ".join("%.9f" % v for v in xyzq(T)),
                                                      " ".join(["1"] + ["0"] * 20)))


def main():
    rng = np.random.default_rng(3)
    tmp = tempfile.mkdtemp()
    f = []
    try:
        gt, st = {}, {}
        for r in (0, 1):
            lines = []
            for i in range(40):
                t = T0 + i * 0.5
                T = rz(0.05 * i, [0.3 * i, 2.0 * r + 0.1 * i, 0.0])
                gt[(r, i)] = T
                st[key(r, i)] = t
                lines.append("%.6f %s" % (t, " ".join("%.9f" % v for v in xyzq(T))))
            open(os.path.join(tmp, "gt_r%d.tum" % r), "w").write("\n".join(lines) + "\n")
        W = rz(0.7, [5.0, -3.0, 1.0])                     # cslam's frame in map
        shift = rz(0.0, [1.0, 0.0, 0.0])

        def graph(r_max, initial, with_lc):
            V, E = {}, []
            for (r, i), T in gt.items():
                if i >= r_max:
                    continue
                N = rz(0.0, rng.normal(0, 0.02, 3))
                Tr = (shift if (initial and r == 1) else np.eye(4)) @ np.linalg.inv(W) @ T @ N
                V[key(r, i)] = Tr
            if with_lc:
                good = np.linalg.inv(gt[(0, 10)]) @ gt[(1, 12)]
                bad = good @ rz(0.0, [2.0, 0.0, 0.0])
                E += [(key(0, 10), key(1, 12), good), (key(0, 20), key(1, 22), bad)]
            E += [(key(0, i), key(0, i + 1), np.eye(4)) for i in range(r_max - 1)]
            return V, E

        res = os.path.join(tmp, "results")
        for n, (r_max, lc) in enumerate(((15, False), (40, True))):
            for r in (0, 1):
                d = os.path.join(res, "2026-10-10_experiment_robot_%d" % r, "t%d" % n)
                os.makedirs(d)
                with open(os.path.join(d, "pose_timestamps%d.csv" % r), "w") as fh:
                    fh.write("vertice_id,sec,nanosec\n")
                    for i in range(r_max):
                        t = st[key(r, i)]
                        fh.write("%d,%d,%d\n" % (key(r, i), int(t), round((t - int(t)) * 1e9)))
                open(os.path.join(d, "log.csv"), "w").write(
                    "robot_id,%d\ntotal_nb_successful_matches,%d\n"
                    "total_front_end_cumulative_communication_bytes,%d\n" % (r, n + 1, 1000 * (n + 1)))
                if r == n % 2:                               # the robot that optimized
                    write_g2o(os.path.join(d, "optimized_global_pose_graph.g2o"), *graph(r_max, False, lc))
                    write_g2o(os.path.join(d, "initial_global_pose_graph.g2o"), *graph(r_max, True, lc))
                time.sleep(0.02)
        cov = os.path.join(tmp, "cov_r1.csv")
        with open(cov, "w") as fh:
            fh.write("t,sx,sy,syaw,corr,sigma_xy_major_m\n")
            for i in range(40):
                fh.write("%.6f,0,0,0,0,%.3f\n" % (T0 + i * 0.5, 0.5 if i == 5 else 0.03))
        rep = CE.evaluate(tmp, cov={1: cov}, sigma_max=0.15, plot=False)
        o, ini = rep["optimized"], rep["initial"]
        if o["keyframes"] != 80 or o["with_ground_truth"] != 79:
            f.append("keyframes %d, with ground truth %d (want 80, 79: one over sigma-max)"
                     % (o["keyframes"], o["with_ground_truth"]))
        if not 0.01 < o["joint_ate_rmse_m"] < 0.05:
            f.append("joint ATE %.3f m, want ~0.03 (2 cm noise per axis)" % o["joint_ate_rmse_m"])
        for r in (0, 1):
            a = rep["robots"][r]["optimized"]["ate_rmse_m"]
            if not 0.01 < a < 0.05:
                f.append("r%d ATE %.3f m, want ~0.03" % (r, a))
        if not 0.4 < ini["joint_ate_rmse_m"] < 0.6:
            f.append("initial joint ATE %.3f m, want ~0.5 (robot 1 1 m off)" % ini["joint_ate_rmse_m"])
        lc = rep["inter_robot_loop_closures"]
        if (lc["total"], lc["checked"], lc["correct"]) != (2, 2, 1):
            f.append("loop closures %s" % {k: lc[k] for k in ("total", "checked", "correct")})
        if abs(rep["time_to_merge_s"] - 39 * 0.5) > 1e-3:
            f.append("time to merge %s s, want 19.5" % rep["time_to_merge_s"])
        if rep["robots"][1]["log"]["total_front_end_cumulative_communication_bytes"] != 2000:
            f.append("log.csv not the latest: %s" % rep["robots"][1]["log"])
    except Exception:                                              # noqa: BLE001
        import traceback
        f.append(traceback.format_exc())
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("\n".join(f) or "cslam eval ok")
    sys.exit(1 if f else 0)


if __name__ == "__main__":
    main()
