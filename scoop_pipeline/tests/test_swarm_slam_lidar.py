#!/usr/bin/env python
"""Self-test for datasets/swarm_slam_lidar.py on a synthetic two-robot bag in
the data layout (data/work/20261015/coop/site_1/<bag>): each robot an Ouster
cloud at 10 Hz (x y z t, no-return points and points on the robot itself), the
LiDAR below an IMU frame on /tf_static, and stage 04's reference track
(processed/.../odometry/reference_coop_site_1_20261015/traj_<machine>_lidar.tum).

mobile_1 takes its odometry from a TUM file of its IMU frame in an odom frame
of its own (as FAST-LIO2 writes it), mobile_2 the ground truth. It must write
one bag with /r0 and /r1 in cslam's LiDAR layout: as many clouds as odometry
messages, at most --rate, only in the --start/--duration slice; clouds of x y z
float32 in the LiDAR frame without the no-returns and the points within
--min-range; odometry that is the LiDAR's pose in the robot's odom frame at
the scan stamp (with --start-at-gt: the ground truth itself); the ground truth
per scan; the cslam config. mobile_2's odometry in its Ouster IMU frame, which
/tf_static does not link, must go to the LiDAR through its GLIM config's
T_lidar_imu. A TUM whose frame nothing links must be refused.

Boards: a session anchor (03's) with each robot's IMU frame as its camera,
mobile_2's 2 deg off: board_start.json must put mobile_1's odom frame where
its first scans' ground truth is, and report mobile_2 2 deg off. Then
slam/board_rendezvous.py's measurement between the first keyframes, through
its quaternion conversion, must be inv(P_0) P_1.

    python scoop_pipeline/tests/test_swarm_slam_lidar.py
"""
import os
import shutil
import sys
import tempfile

import numpy as np
from scipy.spatial.transform import Rotation

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "datasets"))

import mcap_convert as mc                                           # noqa: E402
import swarm_slam_lidar as SL                                       # noqa: E402
from scoop import rosmsg                                            # noqa: E402
from scoop.bagwrite import BagWriter                                # noqa: E402

S = 1_000_000_000
T0 = 1_760_000_000 * S
DUR = 20.0
LIDAR = {"mobile_1": "os_lidar_mobile1", "mobile_2": "os_lidar_mobile2"}
IMU = {"mobile_1": "os_imu_mobile1", "mobile_2": "os_imu_mobile2"}
# the LiDAR in the IMU frame, and mobile_1's odom frame in map
T_IL = np.eye(4)
T_IL[:3, :3] = Rotation.from_euler("z", 180, degrees=True).as_matrix()
T_IL[:3, 3] = [0.006, -0.012, 0.03]
T_MO = np.eye(4)
T_MO[:3, :3] = Rotation.from_euler("z", 30, degrees=True).as_matrix()
T_MO[:3, 3] = [2.0, -1.0, 0.1]
CLOUD = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("t", "<u4"), ("intensity", "<f4")])


def lidar_pose(m, t):
    """T_map_lidar of each robot: a circle each, opposite ways."""
    s = (t - T0) / S
    a = 0.2 * s * (1 if m == "mobile_1" else -1)
    c = (0.0, 0.0) if m == "mobile_1" else (6.0, 1.0)
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler("z", a + np.pi / 2).as_matrix()
    T[:3, 3] = [c[0] + 2.0 * np.cos(a), c[1] + 2.0 * np.sin(a), 0.5]
    return T


def room():
    """Points on the four walls of a 16 x 10 m room, 0..3 m high."""
    g = np.linspace(0, 1, 40)
    u, v = np.meshgrid(g, g)
    u, v = u.ravel(), v.ravel()
    z = 3.0 * v
    walls = [np.c_[-5 + 16 * u, np.full_like(u, -4), z], np.c_[-5 + 16 * u, np.full_like(u, 6), z],
             np.c_[np.full_like(u, -5), -4 + 10 * u, z], np.c_[np.full_like(u, 11), -4 + 10 * u, z]]
    return np.vstack(walls)


def cloud(m, t, P_map):
    T = np.linalg.inv(lidar_pose(m, t))
    P = P_map @ T[:3, :3].T + T[:3, 3]
    near = np.array([[0.1, 0.0, 0.0], [0.0, -0.3, 0.1], [0.2, 0.2, -0.1]])      # the robot itself
    P = np.vstack([P, near, np.zeros((5, 3))])                                  # + no returns
    a = np.zeros(len(P), CLOUD)
    a["x"], a["y"], a["z"] = P[:, 0], P[:, 1], P[:, 2]
    a["t"] = np.arange(len(P)) % 100_000
    return rosmsg.pointcloud2(a, t, LIDAR[m])


def write_pass(root, linked=("mobile_1", "mobile_2")):
    """The bag and stage 04's tracks in data/{work,processed}/20261015/coop/site_1/.
    `linked`: the robots whose IMU /tf_static links to their LiDAR; the others
    have an IMU topic and no link (mobile_2's Ouster in the 20261015 bags)."""
    bag = os.path.join(root, "work", "20261015", "coop", "site_1", "merged_final")
    ref = os.path.join(root, "processed", "20261015", "coop", "site_1", "odometry",
                       "reference_coop_site_1_20261015")
    os.makedirs(ref)
    P_map = room()
    q = mc.T_to_xyzq(T_IL)
    recs = [(T0, "/tf_static", rosmsg.tf_message(
        [(IMU[m], LIDAR[m], q[:3], q[3:]) for m in LIDAR if m in linked], T0))]
    for m in LIDAR:
        if m not in linked:
            recs.append((T0, SL.IMU[m], rosmsg.imu(T0, IMU[m], [0, 0, 9.81], [0, 0, 0])))
    gt = {m: [] for m in LIDAR}
    for k in range(int(DUR * 10)):
        for j, m in enumerate(LIDAR):
            t = T0 + k * S // 10 + j * 37_000_000                  # the two LiDARs out of phase
            recs.append((t, SL.POINTS[m], cloud(m, t, P_map)))
    for k in range(int(DUR * 20) + 1):                             # the tracks at 20 Hz
        t = T0 + k * S // 20
        for m in LIDAR:
            gt[m].append(mc.tum_line(t, lidar_pose(m, t)))
    for m in LIDAR:
        mc.write_tum(os.path.join(ref, "traj_%s_lidar.tum" % m), gt[m])
    w = BagWriter(bag)
    for t, topic, msg in sorted(recs, key=lambda r: r[0]):
        w.write(topic, msg, t)
    w.close()
    return bag


def glim_T_lidar_imu(machine):
    """T_lidar_imu from configs/glim/<machine>/config_sensors.json: [x y z qx qy qz qw]."""
    import re
    v = [float(x) for x in re.search(r'"T_lidar_imu"\s*:\s*\[([^\]]*)\]',
                                     open(SL.MK.GLIM_CONFIG % machine).read()).group(1).split(",")]
    return mc.xyzq_to_T(v)


def write_odom(path):
    """mobile_1's odometry: its IMU frame in its own odom frame, 50 Hz."""
    lines = []
    for k in range(int(DUR * 50) + 1):
        t = T0 + k * S // 50
        lines.append(mc.tum_line(t, np.linalg.inv(T_MO) @ lidar_pose("mobile_1", t) @ np.linalg.inv(T_IL)))
    mc.write_tum(path, lines)


def read_out(out):
    msgs = {}
    for m in mc.read(mc._paths(os.path.join(out, "bag")), None):
        msgs.setdefault(m.channel.topic, []).append(m.ros_msg)
    return msgs


def pose_of(om):
    P = om.pose.pose
    return mc.xyzq_to_T([P.position.x, P.position.y, P.position.z, P.orientation.x,
                         P.orientation.y, P.orientation.z, P.orientation.w])


def check(out, msgs, f, *, start, duration, start_at_gt):
    t_lo = T0 + start * S
    for i, m in enumerate(LIDAR):
        ns = "/r%d" % i
        pcs, ods = msgs.get(ns + "/pointcloud", []), msgs.get(ns + "/odom", [])
        if not pcs or len(pcs) != len(ods):
            f.append("%s: %d clouds, %d odometry messages" % (ns, len(pcs), len(ods)))
            continue
        st = np.array([mc.stamp_ns(x) for x in pcs])
        if st.min() < t_lo or st.max() > T0 + 37_000_000 + (start + duration) * S:
            f.append("%s: scans %.2f..%.2f s, want within the slice %g + %g s"
                     % (ns, (st.min() - T0) / S, (st.max() - T0) / S, start, duration))
        if np.diff(st).min() < 0.09 * S:
            f.append("%s: scans %.0f ms apart (want >= 90)" % (ns, np.diff(st).min() / 1e6))
        want_n = int(duration * 10)
        if abs(len(pcs) - want_n) > 1:
            f.append("%s: %d scans in %g s at 10 Hz (want ~%d)" % (ns, len(pcs), duration, want_n))
        pc = pcs[len(pcs) // 2]
        if [x.name for x in pc.fields] != ["x", "y", "z"] or pc.header.frame_id != "r%d/lidar" % i:
            f.append("%s: cloud fields %s in %s" % (ns, [x.name for x in pc.fields], pc.header.frame_id))
        P, _ = mc.cloud_points(pc)
        r = np.linalg.norm(P, axis=1)
        if len(P) != len(room()) or r.min() < 0.5:
            f.append("%s: %d points, nearest %.2f m (want %d, none within 0.5 m)"
                     % (ns, len(P), r.min(), len(room())))
        # the cloud lands on the room at the ground-truth pose
        T = lidar_pose(m, mc.stamp_ns(pc))
        Pm = P @ T[:3, :3].T + T[:3, 3]
        lo, hi = Pm.min(axis=0), Pm.max(axis=0)
        if np.abs(lo[:2] - [-5, -4]).max() > 1e-3 or np.abs(hi[:2] - [11, 6]).max() > 1e-3:
            f.append("%s: cloud in map spans %s..%s, want the room" % (ns, lo[:2], hi[:2]))
        if ods[0].child_frame_id != "r%d/lidar" % i or ods[0].header.frame_id != "r%d/odom" % i:
            f.append("%s: odometry %s -> %s" % (ns, ods[0].header.frame_id, ods[0].child_frame_id))
        # odometry: the LiDAR in the robot's odom frame (map for the ground truth, and
        # for every robot with --start-at-gt)
        T_mo = T_MO if (m == "mobile_1" and not start_at_gt) else np.eye(4)
        err = max(np.abs(pose_of(om) - np.linalg.inv(T_mo) @ lidar_pose(m, mc.stamp_ns(om))).max()
                  for om in ods)
        if err > 1e-4:
            f.append("%s: odometry off the LiDAR's pose by %.2g" % (ns, err))
        gt = np.loadtxt(os.path.join(out, "gt_r%d.tum" % i), ndmin=2)
        if len(gt) != len(pcs) or np.abs(gt[:, 0] - st / S).max() > 1e-6:
            f.append("%s: %d ground-truth poses for %d scans" % (ns, len(gt), len(pcs)))
        else:
            e = max(np.abs(gt[k, 1:4] - lidar_pose(m, st[k])[:3, 3]).max() for k in range(len(gt)))
            if e > 1e-4:                     # linear between 20 Hz samples on a 2 m circle: ~2e-5
                f.append("%s: ground truth off by %.2g m" % (ns, e))
    cfg = open(os.path.join(out, "scoop_lidar.yaml")).read()
    for want in ('sensor_type: "lidar"', 'pointcloud_topic: "pointcloud"', "voxel_size: 0.5",
                 'log_folder: "%s"' % os.path.join(out, "results"), "enable_logs: true"):
        if want not in cfg:
            f.append("config: no %r" % want)


def boards(tmp, bag, out, f):
    import json
    sys.path.insert(0, os.path.join(ROOT, "slam"))
    import board_rendezvous as BR
    fr = os.path.join(tmp, "data", "processed", "20261015", "coop", "site_1", "frames")
    os.makedirs(fr, exist_ok=True)
    off = np.eye(4)
    off[:3, :3] = Rotation.from_euler("z", 2.0, degrees=True).as_matrix()
    cams = {}
    for m, nm, t_a, E in (("mobile_1", "zed", T0 + 6 * S, np.eye(4)), ("mobile_2", "realsense", T0 + 8 * S, off)):
        q = mc.T_to_xyzq(lidar_pose(m, t_a) @ E @ np.linalg.inv(T_IL))
        cams[nm] = {"cam_frame": IMU[m], "dwell_t_end": t_a / S,
                    "map_to_cam": {"xyz": list(q[:3]), "qxyzw": list(q[3:])}}
    json.dump({"cameras": cams}, open(os.path.join(fr, "session_anchor.json"), "w"))
    path, lines = SL.board_start(bag, out)
    rec = {r["robot_id"]: r for r in json.load(open(path))["robots"]}
    v0, v1 = rec[0]["vs_ground_truth"], rec[1]["vs_ground_truth"]
    if v0["t_m"] > 1e-3 or v0["r_deg"] > 0.01:
        f.append("boards: mobile_1 off its ground truth by %.4f m %.3f deg" % (v0["t_m"], v0["r_deg"]))
    if abs(v1["r_deg"] - 2.0) > 0.01 or abs(abs(v1["yaw_deg"]) - 2.0) > 0.01:
        f.append("boards: mobile_2 %.3f deg (yaw %.3f) off, want 2" % (v1["r_deg"], v1["yaw_deg"]))
    # the rendezvous measurement, through its quaternions
    P0, P1 = lidar_pose("mobile_1", T0), lidar_pose("mobile_2", T0 + 3 * S)
    M = BR.measurement(P0, P1)
    M2 = BR.T_of(M[:3, 3], BR.quat_of(M[:3, :3]))
    if np.abs(P0 @ M2 - P1).max() > 1e-9:
        f.append("board_rendezvous: inv(P_0) P_1 does not take P_0 to P_1")
    for ang in (0.0, 179.9, -120.0):
        R = Rotation.from_euler("xyz", [ang, 30, ang / 2], degrees=True).as_matrix()
        if np.abs(BR.T_of([0, 0, 0], BR.quat_of(R))[:3, :3] - R).max() > 1e-9:
            f.append("board_rendezvous: quaternion round trip at %g deg" % ang)


def main():
    tmp = tempfile.mkdtemp()
    f = []
    try:
        bag = write_pass(os.path.join(tmp, "data"))
        odom = os.path.join(tmp, "traj_fastlio.tum")
        write_odom(odom)
        tums = {"mobile_1": (odom, IMU["mobile_1"])}
        r = SL.convert(bag, None, odom_tums=tums, start=4.0, duration=10.0)
        if r["out"] != os.path.join(tmp, "data", "processed", "20261015", "coop", "site_1", "datasets",
                                    "swarm_slam_lidar", "coop_site_1_20261015_gt+traj_fastlio"):
            f.append("default out %s" % r["out"])
        check(r["out"], read_out(r["out"]), f, start=4.0, duration=10.0, start_at_gt=False)
        out = os.path.join(tmp, "at_gt")
        check(out, read_out(SL.convert(bag, out, odom_tums=tums, start=4.0, duration=10.0,
                                       start_at_gt=True)["out"]),
              f, start=4.0, duration=10.0, start_at_gt=True)
        # mobile_2's Ouster IMU, which /tf_static does not link: its GLIM config's T_lidar_imu
        bag2 = write_pass(os.path.join(tmp, "data2"), linked=("mobile_1",))
        T_li = glim_T_lidar_imu("mobile_2")
        odom2 = os.path.join(tmp, "traj_fastlio_m2.tum")
        mc.write_tum(odom2, [mc.tum_line(T0 + k * S // 50, lidar_pose("mobile_2", T0 + k * S // 50) @ T_li)
                             for k in range(int(DUR * 50) + 1)])
        out = os.path.join(tmp, "m2_glim")
        check(out, read_out(SL.convert(bag2, out, odom_tums=dict(tums, mobile_2=(odom2, IMU["mobile_2"])),
                                       start=4.0, duration=10.0, start_at_gt=True)["out"]),
              f, start=4.0, duration=10.0, start_at_gt=True)
        boards(tmp, bag, r["out"], f)
        try:
            SL.convert(bag, os.path.join(tmp, "bad"), odom_tums={"mobile_1": (odom, "nowhere")})
            f.append("an odometry frame not in /tf_static was accepted")
        except mc.ConvertError:
            pass
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    if f:
        print("FAIL\n  " + "\n  ".join(f))
        sys.exit(1)
    print("swarm slam lidar ok")


if __name__ == "__main__":
    main()
