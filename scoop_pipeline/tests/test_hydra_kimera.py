#!/usr/bin/env python
"""Self-test for datasets/hydra.py --poses kimera on test_hydra_export's
export (robot 0, mobile_1, with its ground truth per frame) and a made-up
Kimera-VIO run in a datasets/euroc.py layout: the IMU 5 cm behind the colour
camera and turned to its body axes; Kimera-VIO's odometry the ground truth
0.2 m along x, at frames 1-10; two loop closures: frame 10 on frame 2 as the
ground truth has it, and frame 11 on frame 1 claiming 3 m.

It must put the camera where Kimera-VIO's IMU puts it (0.2 m along x of the
ground truth) at every frame 1-10, interpolated between keyframes, and no
/tf at frames 0 and 11; send the right closure on /hydra/external_loop_closures
(keys the two stamps, the query's base frame in the match's) and leave the
other out; and report both, with the ground truth's verdict.

    python scoop_pipeline/tests/test_hydra_kimera.py
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
import hydra as H                                                   # noqa: E402
import mcap_convert as mc                                           # noqa: E402
import test_hydra_export as THE                                     # noqa: E402
from scoop import bag as B, rosmsg                                  # noqa: E402

T_IC = np.eye(4)                                                    # the colour camera in the IMU frame
T_IC[:3, :3] = H.R_BODY_OPTICAL
T_IC[:3, 3] = [0.05, 0.0, 0.02]
SHIFT = np.array([0.2, 0.0, 0.0])


def stamp(k):
    return THE.T0 + k * 100_000_000


def kstamp(k):
    """Kimera-VIO's keyframe stamp near frame k: a few ns off, as no float64 holds it exactly."""
    return stamp(k) + (-7 if k == 1 else 7)


def imu_gt(k):
    return THE.gt_pose(0, k) @ np.linalg.inv(T_IC)


def wxyz(T):
    q = mc.T_to_xyzq(T)
    return [q[6], q[3], q[4], q[5]]


def make_kimera(d):
    m0 = os.path.join(d, "mobile_1", "mav0")
    os.makedirs(os.path.join(m0, "state_groundtruth_estimate0"))
    with open(os.path.join(m0, "state_groundtruth_estimate0", "data.csv"), "w") as fh:
        fh.write("#timestamp,p,q,v,b\n")
        for k in range(THE.N):
            T = imu_gt(k)
            fh.write("%d,%s,%s,0,0,0,0,0,0,0,0,0\n" % (stamp(k), ",".join("%.9f" % x for x in T[:3, 3]),
                                                      ",".join("%.9f" % x for x in wxyz(T))))
    o = os.path.join(d, "mobile_1", "out")
    os.makedirs(o)
    with open(os.path.join(o, "traj_vio.csv"), "w") as fh:
        fh.write("#timestamp,x,y,z,qw,qx,qy,qz,vx,vy,vz,bgx,bgy,bgz,bax,bay,baz\n")
        for k in (1, 2, 4, 10):                          # keyframes; the rest interpolated, 4-10 0.6 s
            T = imu_gt(k)
            fh.write("%d,%s,%s,0,0,0,0,0,0,0,0,0\n" % (kstamp(k), ",".join("%.9f" % x for x in T[:3, 3] + SHIFT),
                                                      ",".join("%.9f" % x for x in wxyz(T))))
    right = np.linalg.inv(imu_gt(2)) @ imu_gt(10)
    wrong = np.eye(4)
    wrong[:3, 3] = [3.0, 0.0, 0.0]
    with open(os.path.join(o, "output_lcd_result.csv"), "w") as fh:
        fh.write("#timestamp_kf,timestamp_query,timestamp_match,isLoop,matchKfId,queryKfId,x,y,z,qw,qx,qy,qz\n")
        for tq, tm, T in ((10, 2, right), (10, 1, wrong)):
            fh.write("%d,%d,%d,1,0,0,%s,%s\n" % (kstamp(tq), kstamp(tq), kstamp(tm),
                                                ",".join("%.9f" % x for x in T[:3, 3]),
                                                ",".join("%.9f" % x for x in wxyz(T))))
        fh.write("%d,0,0,0,0,0,0,0,0,1,0,0,0\n" % stamp(5))


def main():
    f = []
    tmp = tempfile.mkdtemp()
    try:
        exp = os.path.join(tmp, "datasets", "swarm_slam", "pass_x")
        THE.make_export(exp)
        with open(os.path.join(exp, "gt_r0.tum"), "w") as fh:
            for k in range(THE.N):
                fh.write(mc.tum_line(stamp(k), THE.gt_pose(0, k)) + "\n")
        kim = os.path.join(tmp, "datasets", "euroc", "pass_x")
        make_kimera(kim)
        res = H.convert(exp, robots=[0], poses="kimera", kimera=kim)
        out = res["out"]
        if not out.endswith(os.path.join("hydra", "pass_x_kimera")):
            f.append("default out %s" % out)
        ts = rosmsg.typestore()
        rosmsg._register_pgt()
        tf, lcs = {}, []
        for _, _, ch, msg in B.iter_mcap_records(B.open_bag(os.path.join(out, "bag_r0"))):
            if ch.topic == "/tf":
                m = ts.deserialize_cdr(msg.data, "tf2_msgs/msg/TFMessage")
                T = np.eye(4)
                for x in m.transforms:
                    tr, q = x.transform.translation, x.transform.rotation
                    T = T @ mc.xyzq_to_T([tr.x, tr.y, tr.z, q.x, q.y, q.z, q.w])
                s = m.transforms[0].header.stamp
                tf[s.sec * 1_000_000_000 + s.nanosec] = T
            elif ch.topic == H.LC_TOPIC:
                lcs.append((msg.log_time, ts.deserialize_cdr(msg.data, "pose_graph_tools_msgs/msg/PoseGraph")))
        want = {stamp(k): k for k in range(1, 11)}
        if sorted(tf) != sorted(want):
            f.append("/tf at frames %s, want 1-10" % sorted((t - THE.T0) // 100_000_000 for t in tf))
        for t, k in want.items():
            if t in tf:
                g = THE.gt_pose(0, k)
                tol = 1e-5 if k in (1, 2, 4, 10) else 3e-3       # between keyframes: chord vs arc
                if not (np.allclose(tf[t][:3, 3], g[:3, 3] + SHIFT, atol=tol)
                        and np.allclose(tf[t][:3, :3], g[:3, :3], atol=tol)):
                    f.append("frame %d: camera at %s, want %s" % (k, tf[t][:3, 3], g[:3, 3] + SHIFT))
                    break
        if len(lcs) != 1:
            f.append("%d loop closure messages, want 1" % len(lcs))
        else:
            lt, pg = lcs[0]
            e = pg.edges[0]
            base = lambda k: THE.gt_pose(0, k) @ np.linalg.inv(H.body_T_camera())
            rel = np.linalg.inv(base(2)) @ base(10)
            p, q = e.pose.position, e.pose.orientation
            got = mc.xyzq_to_T([p.x, p.y, p.z, q.x, q.y, q.z, q.w])
            if (e.key_from, e.key_to, e.robot_from, e.robot_to, e.type) != (kstamp(10), kstamp(2), 0, 0, 1) \
                    or lt != kstamp(10) + H.LC_DELAY_NS or not np.allclose(got, rel, atol=1e-5):
                f.append("closure: keys %s %s at %s, pose\n%s\nwant\n%s" % (e.key_from, e.key_to, lt, got, rel))
        # a robot standing still: two keyframes 20 s apart, the pose known in between
        ko = os.path.join(tmp, "still", "out")
        os.makedirs(ko)
        with open(os.path.join(ko, "traj_vio.csv"), "w") as fh:
            fh.write("#timestamp,x,y,z,qw,qx,qy,qz\n%d,1,2,0,1,0,0,0\n%d,1,2,0,1,0,0,0\n" % (THE.T0, THE.T0 + 20 * 10 ** 9))
        mid = H.KimeraOdometry(os.path.join(tmp, "still")).at(THE.T0 + 10 * 10 ** 9)
        if mid is None or not np.allclose(mid[:3, 3], [1, 2, 0]):
            f.append("no pose between keyframes 20 s apart: %s" % mid)
        rep = open(os.path.join(out, "report.txt")).read()
        if "2 from Kimera-VIO; 1 agree with its odometry" not in rep or "(1 of 1 right by the ground truth), " \
                "1 left out (0 of 1 right)" not in rep or "2 frames without a pose" not in rep:
            f.append("report:\n" + rep)
    except Exception:                                              # noqa: BLE001
        import traceback
        f.append(traceback.format_exc())
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("\n".join(f) or "hydra kimera ok")
    sys.exit(1 if f else 0)


if __name__ == "__main__":
    main()
