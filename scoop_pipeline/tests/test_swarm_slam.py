#!/usr/bin/env python
"""Self-test for datasets/swarm_slam.py on test_replica_multiagent's two-robot
bag, with each robot's onboard odometry added: of a body frame (not the
camera) in an odom frame of its own (not map), linked to the colour camera
through /tf_static. Then again with mobile_2's odometry not recorded: it
must come from <work>/mobile_2/vslam/traj_vslam.txt (processing/vslam.py's
offline cuVSLAM, of camera_link) the same.

It must write one bag with /r0 and /r1 in cslam's RGB-D layout: colour at
--width with matching intrinsics and no distortion, depth registered to it
(3 m to the wall for both robots), and odometry that is the colour camera's
pose in the robot's odom frame at the image stamp; frames at most --rate;
ground truth (map) per frame; the cslam config.

    python scoop_pipeline/tests/test_swarm_slam.py
"""
import os
import shutil
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "datasets"))

import mcap_convert as mc                                           # noqa: E402
import swarm_slam as SS                                             # noqa: E402
import test_replica_multiagent as RMT                               # noqa: E402
from scoop import rosmsg, merge                                     # noqa: E402
from scoop.bagwrite import BagWriter                                # noqa: E402

S = RMT.S
# body -> colour optical (x fwd, y left, z up -> optical), and each odom frame in map
R_BO = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], float)
T_BC = {"mobile_1": np.eye(4), "mobile_2": np.eye(4)}
T_BC["mobile_1"][:3, :3] = R_BO
T_BC["mobile_1"][:3, 3] = [0.1, 0.06, 0.3]
T_BC["mobile_2"][:3, :3] = R_BO
T_BC["mobile_2"][:3, 3] = [0.0, -0.02, 0.5]
T_MO = {"mobile_1": np.eye(4), "mobile_2": np.eye(4)}
T_MO["mobile_1"][:3, 3] = [1.0, -2.0, 0.0]
T_MO["mobile_2"][:3, 3] = [-3.0, 0.5, 0.2]
BODY = {"mobile_1": "m1_base", "mobile_2": "m2_link"}
CAM = {"mobile_1": "m1_left", "mobile_2": "m2_color"}


def odom_bag(path, skip=()):
    T = rosmsg.msg_type
    recs = []
    tfs = []
    for m in SS.ODOM:
        q = mc.T_to_xyzq(T_BC[m])
        tfs.append((BODY[m], CAM[m], q[:3], q[3:]))
    recs.append((RMT.T0, "/tf_static", rosmsg.tf_message(tfs, RMT.T0)))
    for m, topic in SS.ODOM.items():
        if m in skip:
            continue
        for k in range(int(RMT.DUR * 50) + 1):                     # 50 Hz
            t = RMT.T0 + k * S // 50
            Tb = np.linalg.inv(T_MO[m]) @ RMT.pose(m, t) @ np.linalg.inv(T_BC[m])
            recs.append((t, topic, SS.odometry(Tb, m + "_odom", BODY[m], t)))
    w = BagWriter(path)
    for t, topic, msg in sorted(recs, key=lambda r: r[0]):
        w.write(topic, msg, t)
    w.close()


def main():
    tmp = tempfile.mkdtemp()
    f = []
    try:
        b1, b2 = os.path.join(tmp, "cams"), os.path.join(tmp, "odom")
        RMT.write_bag(b1)
        odom_bag(b2)
        bag = os.path.join(tmp, "merged")
        merge.merge_bags([b1, b2], bag, log=lambda *_: None)
        out = os.path.join(tmp, "swarm")
        r = SS.convert(bag, out, rate=10.0, width=100)
        p = mc._paths(os.path.join(out, "bag"))
        msgs = {}
        for m in mc.read(p, None):
            msgs.setdefault(m.channel.topic, []).append(m.ros_msg)
        for i, m in enumerate(SS.ODOM):
            ns = "/r%d" % i
            topics = [ns + x for x in ("/color/image_raw", "/color/camera_info",
                                       "/aligned_depth_to_color/image_raw", "/odom")]
            n = [len(msgs.get(t, [])) for t in topics]
            if len(set(n)) != 1 or n[0] == 0:
                f.append("%s: messages per topic %s" % (ns, dict(zip(topics, n))))
                continue
            st = np.array([mc.stamp_ns(x) for x in msgs[topics[0]]]) / S
            if np.diff(st).min() < 0.09 * 0.99:
                f.append("%s: frames %.0f ms apart (want >= 90)" % (ns, 1e3 * np.diff(st).min()))
            c = RMT.CAMS[m]
            img, info = msgs[topics[0]][3], msgs[topics[1]][3]
            s = 100.0 / c["W"]
            if img.width != 100 or abs(info.k[0] - c["f"] * s) > 1e-6 or any(info.d):
                f.append("%s: image %dx%d, fx %.3f (want 100 wide, fx %.3f, no distortion)"
                         % (ns, img.width, img.height, info.k[0], c["f"] * s))
            d = mc.decode_image(msgs[topics[2]][3]).astype(float) / 1000.0
            # mobile_2's colour sees wider than its depth imager: its outer columns
            # (tan 0.85 of 1.105 each side) have none
            cover = 0.9 if m == "mobile_1" else 0.85 / (99.5 / 90.0) - 0.02
            if (d > 0).mean() < cover or abs(np.median(d[d > 0]) - RMT.WALL) > 0.003:
                f.append("%s: depth %.3f m on %.0f %% of pixels (want %.1f m)"
                         % (ns, np.median(d[d > 0]) if (d > 0).any() else -1, 100 * (d > 0).mean(),
                            RMT.WALL))
            err = 0.0
            for om in msgs[topics[3]]:
                t = mc.stamp_ns(om)
                P = om.pose.pose
                Tg = mc.xyzq_to_T([P.position.x, P.position.y, P.position.z, P.orientation.x,
                                   P.orientation.y, P.orientation.z, P.orientation.w])
                want = np.linalg.inv(T_MO[m]) @ RMT.pose(m, t)
                err = max(err, np.abs(Tg - want).max())
            if err > 1e-3:
                f.append("%s: odometry off the colour camera in its odom frame by %.4f" % (ns, err))
            gt = np.loadtxt(os.path.join(out, "gt_r%d.tum" % i), ndmin=2)
            if len(gt) == 0 or np.abs(gt[:, 1] - np.array([RMT.pose(m, int(round(t * S)))[0, 3]
                                                          for t in gt[:, 0]])).max() > 1e-3:
                f.append("%s: ground truth not the camera in map" % ns)
        # mobile_2's odometry not recorded: the offline cuVSLAM file next to the bag
        w2 = os.path.join(tmp, "w2")
        os.makedirs(os.path.join(w2, "mobile_2", "vslam"))
        BODY["mobile_2"] = SS.VSLAM_BASE
        b3 = os.path.join(tmp, "odom1")
        odom_bag(b3, skip=("mobile_2",))
        merge.merge_bags([b1, b3], os.path.join(w2, "merged"), log=lambda *_: None)
        with open(os.path.join(w2, "mobile_2", "vslam", "traj_vslam.txt"), "w") as fh:
            for k in range(int(RMT.DUR * 50) + 1):
                t = RMT.T0 + k * S // 50
                Tb = np.linalg.inv(T_MO["mobile_2"]) @ RMT.pose("mobile_2", t) @ np.linalg.inv(T_BC["mobile_2"])
                fh.write(mc.tum_line(t, Tb) + "\n")
        out2 = os.path.join(tmp, "swarm2")
        SS.convert(os.path.join(w2, "merged"), out2, rate=10.0, width=100)
        err = 0.0
        for m_ in mc.read(mc._paths(os.path.join(out2, "bag")), ["/r1/odom"]):
            P = m_.ros_msg.pose.pose
            Tg = mc.xyzq_to_T([P.position.x, P.position.y, P.position.z, P.orientation.x,
                               P.orientation.y, P.orientation.z, P.orientation.w])
            want = np.linalg.inv(T_MO["mobile_2"]) @ RMT.pose("mobile_2", mc.stamp_ns(m_.ros_msg))
            err = max(err, np.abs(Tg - want).max())
        if not err < 1e-3:
            f.append("offline cuVSLAM odometry: off by %.4f" % err)
        y = open(os.path.join(out, "scoop_rgbd.yaml")).read()
        for k in ('sensor_type: "rgbd"', "enable_logs: true", 'odom_topic: "odom"'):
            if k not in y:
                f.append("config lacks %r" % k)
    except Exception:                                              # noqa: BLE001
        import traceback
        f.append(traceback.format_exc())
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("\n".join(f) or "swarm slam ok")
    sys.exit(1 if f else 0)


if __name__ == "__main__":
    main()
