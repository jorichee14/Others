#!/usr/bin/env python
"""Self-test for analysis/slam/trajectory_video.py.

swarm: a synthetic Swarm-SLAM run (cslam's logs, as test_cslam_eval.py writes
them): two robots' keyframes in cslam's frame, robot 1 a further 1 m off in
the graph before optimization, one inter-robot loop closure at keyframes
(0,10)-(1,12) and one intra-robot at (0,2)-(0,30). One loop-closure event, at the merge, held 2 s.
The tracks must land on the
ground truth after the one fit, robot 1 be faded (its own frame) until the
merge, which is the later of the two keyframes' stamps; the edges sorted into
inter and intra. odometry: test_odometry.py's bag, FAST-LIO2's IMU-frame
estimate in an odom frame of its own: placed on the reference at its first
pose, it must lie on the reference. Both videos are written and hold the
frames the speed asks for. The loop closure's scans come from an export bag
of every keyframe seeing the same points: robot 1's, placed by the loop
closure, must land on them as robot 0 sees them; placed with no shared frame,
not.

    python scoop_pipeline/tests/test_trajectory_video.py
"""
import os
import shutil
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "analysis", "slam"))
sys.path.insert(0, os.path.join(ROOT, "slam", "odometry"))
sys.path.insert(0, os.path.join(ROOT, "datasets"))
import test_cslam_eval as TC                                        # noqa: E402
import trajectory_video as TV                                       # noqa: E402


def swarm_fixture(tmp):
    gt, st = {}, {}
    for r in (0, 1):
        lines = []
        for i in range(40):
            t = TC.T0 + i * 0.5
            T = TC.rz(0.05 * i, [0.3 * i, 2.0 * r + 0.1 * i, 0.0])
            gt[(r, i)], st[TC.key(r, i)] = T, t
            lines.append("%.6f %s" % (t, " ".join("%.9f" % v for v in TC.xyzq(T))))
        open(os.path.join(tmp, "gt_r%d.tum" % r), "w").write("\n".join(lines) + "\n")
    W = TC.rz(0.7, [5.0, -3.0, 1.0])
    shift = TC.rz(0.0, [1.0, 0.0, 0.0])

    def graph(initial):
        V = {TC.key(r, i): (shift if (initial and r == 1) else np.eye(4)) @ np.linalg.inv(W) @ T
             for (r, i), T in gt.items()}
        E = [(TC.key(0, 10), TC.key(1, 12), np.linalg.inv(gt[(0, 10)]) @ gt[(1, 12)]),
             (TC.key(0, 2), TC.key(0, 30), np.linalg.inv(gt[(0, 2)]) @ gt[(0, 30)])]
        E += [(TC.key(0, i), TC.key(0, i + 1), np.eye(4)) for i in range(39)]
        return V, E

    # the export's bag: every keyframe's scan of the same points, in its LiDAR frame
    import swarm_slam as SS
    import swarm_slam_lidar as SL
    from scoop.bagwrite import BagWriter
    rng = np.random.default_rng(3)
    world = np.c_[rng.uniform(-3, 12, (3000, 2)), np.full(3000, 0.5)]
    w = BagWriter(os.path.join(tmp, "bag"))
    for (r, i), T in sorted(gt.items(), key=lambda x: st[TC.key(*x[0])]):
        P = (world - T[:3, 3]) @ T[:3, :3]
        a = np.empty(len(P), SL.XYZ)
        a["x"], a["y"], a["z"] = P[:, 0], P[:, 1], P[:, 2]
        ns = int(round(st[TC.key(r, i)] * 1e9))
        w.write("/r%d/pointcloud" % r, SS.T_msg().pointcloud2(a, ns, "r%d/lidar" % r), ns)
    w.close()

    run = os.path.join(tmp, "runs", "ideal")
    for r in (0, 1):
        d = os.path.join(run, "results", "x_experiment_robot_%d" % r, "t0")
        os.makedirs(d)
        with open(os.path.join(d, "pose_timestamps%d.csv" % r), "w") as fh:
            fh.write("vertice_id,sec,nanosec\n")
            for i in range(40):
                t = st[TC.key(r, i)]
                fh.write("%d,%d,%d\n" % (TC.key(r, i), int(t), round((t - int(t)) * 1e9)))
        if r == 0:
            TC.write_g2o(os.path.join(d, "optimized_global_pose_graph.g2o"), *graph(False))
            TC.write_g2o(os.path.join(d, "initial_global_pose_graph.g2o"), *graph(True))
    return run, gt, st, world


def frames(path):
    import cv2
    c = cv2.VideoCapture(path)
    n = int(c.get(cv2.CAP_PROP_FRAME_COUNT))
    c.release()
    return n


def main():
    tmp = tempfile.mkdtemp()
    f = []
    try:
        run, gt, st, world = swarm_fixture(tmp)
        tracks, refs, edges, merged_at, _, events = TV.swarm_tracks(tmp, run)
        for r, tr in enumerate(tracks):
            want = np.array([gt[(r, i)][:3, 3] for i in range(40)])
            if np.abs(tr.P - want).max() > 1e-6:
                f.append("r%d not on the ground truth after the fit: %.3g m" % (r, np.abs(tr.P - want).max()))
        if abs(merged_at - st[TC.key(1, 12)]) > 1e-6:
            f.append("merged at %s, want the stamp of (1,12)" % merged_at)
        if tracks[0].faded is not None or tracks[1].faded is None \
                or np.abs(np.linalg.norm(tracks[1].faded[2] - tracks[1].P, axis=1) - 1.0).max() > 1e-6:
            f.append("robot 1 not faded 1 m off before the merge (robot 0 not at all)")
        if sorted(e[3] for e in edges) != ["inter", "intra"]:
            f.append("edges %s" % [e[3] for e in edges])
        out = os.path.join(tmp, "videos", "swarm.mp4")
        if len(events) != 1 or abs(events[0]["t"] - merged_at) > 1e-9:
            f.append("loop-closure events %s" % [e["t"] for e in events])
        # scans: robot b's placed by the loop closure lands on the points as robot a
        # sees them; with no shared frame, it does not
        from scipy.spatial import cKDTree
        e = events[0]
        Ta = gt[(0, 10)]
        seen = cKDTree(((world - Ta[:3, 3]) @ Ta[:3, :3])[:, :2])
        if e.get("cloud_a") is None:
            f.append("no scans for the loop closure")
        else:
            d_a, d_after, d_before = (np.median(seen.query(e[k])[0]) for k in ("cloud_a", "cloud_b_after", "cloud_b_before"))
            if max(d_a, d_after) > 1e-6 or d_before < 0.05:
                f.append("scans: robot a %.3g m, b after %.3g m, b before %.3g m off" % (d_a, d_after, d_before))
        n = TV.render(out, tracks, refs, edges, "test", speed=4.0, fps=10, size=(320, 240), merged_at=merged_at,
                      events=events, hold=3, hold_s=2.0)
        if frames(out) != n or n != int(np.ceil(19.5 / 4.0 * 10)) + 10 + 20:
            f.append("swarm video: %d frames written, %d rendered" % (frames(out), n))

        # odometry: test_odometry's bag, FAST-LIO2's estimate in its IMU frame
        import test_odometry as TO
        import make_config as MK
        bag = os.path.join(tmp, "obag")
        TO.write_bag(bag)
        orun = os.path.join(tmp, "odo")
        MK.make("fast_lio", bag, "mobile_1", orun)
        tr, Tr = TO.ref_traj()
        T_off = TO.T_of([1.0, -2.0, 0.3], TO.Rotation.from_euler("z", 40, degrees=True).as_quat())
        T_imu_lidar = np.linalg.inv(TO.T_of(*TO.T_SENSOR_IMU)) @ TO.T_of(*TO.T_SENSOR_LIDAR)
        TO.write_tum(os.path.join(orun, "trajectory.tum"), tr,
                     np.einsum("ij,njk,kl->nil", np.linalg.inv(T_off), Tr, np.linalg.inv(T_imu_lidar)))
        ref = os.path.join(tmp, "ref.tum")
        TO.write_tum(ref, tr, Tr)
        tracks, refs, _ = TV.odometry_tracks([orun], {"mobile_1": ref})
        if np.abs(tracks[0].P - Tr[:, :3, 3]).max() > 1e-6:
            f.append("odometry not on the reference from its first pose: %.3g m"
                     % np.abs(tracks[0].P - Tr[:, :3, 3]).max())
        out = os.path.join(tmp, "odo.mp4")
        n = TV.render(out, tracks, refs, [], "test", speed=20.0, fps=10, size=(320, 240))
        if frames(out) != n:
            f.append("odometry video: %d frames written, %d rendered" % (frames(out), n))
    except Exception:                                              # noqa: BLE001
        import traceback
        f.append(traceback.format_exc())
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("\n".join(f) or "trajectory video ok")
    sys.exit(1 if f else 0)


if __name__ == "__main__":
    main()
