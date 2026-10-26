#!/usr/bin/env python
"""Self-test for slam/odometry/make_config.py and slam/monitor/summarize.py,
no ROS: a synthetic two-robot bag (mobile_1: a 64-beam Ouster and its IMU,
the LiDAR below an IMU frame on /tf_static; mobile_2: the D455's infrared
pair, depth in the left camera, its IMU) and hand-made probe logs.

make_config must read every frame, the beams and the transforms off the bag:
FAST-LIO2's extrinsic is the LiDAR in the IMU frame, RKO-LIO's the IMU in the
LiDAR (= base) frame, OpenVINS's T_cam_imu per camera; every LiDAR method
leaves out the first 1.0 m. Where the bag does not link a robot's
LiDAR and IMU (mobile_2), T_lidar_imu comes from its GLIM config. LiDAR
methods get the sweep relay. It must refuse OpenVINS without an IMU noise
file, a camera method on mobile_1, RTAB-Map when the depth is not in the
left camera, a key topic under 90 % of its nominal rate, and a run folder
that holds a run.

score must put an estimate in the LiDAR frame and find ATE 0 for a perfect
one in another odom frame, ~1 % drift for a 1 % scale error, a gap and a
jump where they are. summarize must time latency between publishers (the send times, not the
probe's own late arrival), match each output to its own input: one stamped at the
scan's last point (FAST-LIO2) also when the next scan has arrived before it,
one stamped a few ms before its image (OpenVINS, with its tolerance); count
drops and repeats; decide "kept up"; and window the resources to the play.

    python scoop_pipeline/tests/test_odometry.py
"""
import json
import os
import shutil
import sys
import tempfile

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "slam", "odometry"))
sys.path.insert(0, os.path.join(ROOT, "slam", "monitor"))

import make_config as MK                                            # noqa: E402
import summarize as SM                                              # noqa: E402
from scoop import rosmsg                                            # noqa: E402
from scoop.bagwrite import BagWriter                                # noqa: E402

S = 1_000_000_000
T0 = 1_760_000_000 * S
DUR = 6.0
OUSTER = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("intensity", "<f4"), ("t", "<u4"),
                   ("reflectivity", "<u2"), ("ring", "<u2"), ("ambient", "<u2"), ("range", "<u4")])
# os_sensor -> os_lidar (180 deg yaw, 36 mm up) and os_sensor -> os_imu, as an OS1's metadata has them
Q_LIDAR = Rotation.from_euler("z", 180, degrees=True).as_quat()
T_SENSOR_LIDAR = ([0.0, 0.0, 0.03618], Q_LIDAR)
T_SENSOR_IMU = ([0.006253, -0.011775, 0.007645], [0.0, 0.0, 0.0, 1.0])
# camera_link -> the D455's frames
Q_OPT = Rotation.from_matrix([[0, 0, 1], [-1, 0, 0], [0, -1, 0]]).as_quat()   # optical in link
CAM_TF = [("camera_link", "camera_infra1_optical_frame", [0.0, 0.0, 0.0], Q_OPT),
          ("camera_link", "camera_infra2_optical_frame", [0.0, -0.095, 0.0], Q_OPT),
          ("camera_link", "camera_depth_optical_frame", [0.0, 0.0, 0.0], Q_OPT),
          ("camera_link", "camera_imu_optical_frame", [-0.0302, 0.0074, 0.016], Q_OPT)]
K = [425.0, 0.0, 424.0, 0.0, 425.0, 240.0, 0.0, 0.0, 1.0]


def T_of(t, q):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat(q).as_matrix()
    T[:3, 3] = t
    return T


def info(t, frame, k=K, w=848, h=480, tx=0.0):
    T = rosmsg.msg_type
    return T("sensor_msgs/msg/CameraInfo")(
        header=rosmsg.header(t, frame), height=h, width=w, distortion_model="plumb_bob",
        d=np.zeros(5), k=np.array(k, float), r=np.eye(3).reshape(-1),
        p=np.array([k[0], 0, k[2], tx, 0, k[4], k[5], 0, 0, 0, 1, 0], float), binning_x=0, binning_y=0,
        roi=T("sensor_msgs/msg/RegionOfInterest")(x_offset=0, y_offset=0, height=0, width=0, do_rectify=False))


def image(t, frame, enc="mono8", w=8, h=6):
    return rosmsg.msg_type("sensor_msgs/msg/Image")(
        header=rosmsg.header(t, frame), height=h, width=w, encoding=enc, is_bigendian=0,
        step=w * (2 if enc == "16UC1" else 1), data=np.zeros(w * h * (2 if enc == "16UC1" else 1), np.uint8))


def write_bag(path, cam_hz=30.0, depth_k=K):
    recs = [(T0, "/tf_static", rosmsg.tf_message(
        [("os_sensor_mobile1", "os_lidar_mobile1") + T_SENSOR_LIDAR,
         ("os_sensor_mobile1", "os_imu_mobile1") + T_SENSOR_IMU] + CAM_TF, T0))]
    for k in range(int(DUR * 10)):
        t = T0 + k * S // 10
        a = np.zeros(64 * 16, OUSTER)
        a["x"] = 5.0
        recs.append((t, "/mobile_1/ouster/points", rosmsg.pointcloud2(a, t, "os_lidar_mobile1", height=64)))
        # mobile_2's Ouster, which the pass bag's /tf_static does not link to its IMU
        recs.append((t, "/mobile_2/ouster/points", rosmsg.pointcloud2(a, t, "os_lidar_mobile2", height=64)))
    for k in range(int(DUR * 100)):
        t = T0 + k * S // 100
        recs.append((t, "/mobile_1/ouster/imu", rosmsg.imu(t, "os_imu_mobile1", [0, 0, 9.81], [0, 0, 0])))
        recs.append((t, "/mobile_2/ouster/imu", rosmsg.imu(t, "os_imu_mobile2", [0, 0, 9.81], [0, 0, 0])))
    for k in range(int(DUR * 200)):
        t = T0 + k * S // 200
        recs.append((t, "/mobile_2/imu", rosmsg.imu(t, "camera_imu_optical_frame", [0, -9.81, 0], [0, 0, 0])))
    for k in range(int(DUR * cam_hz)):
        t = T0 + int(k * S / cam_hz)
        recs += [(t, "/mobile_2/infra1/image_rect_raw", image(t, "camera_infra1_optical_frame")),
                 (t, "/mobile_2/infra1/camera_info", info(t, "camera_infra1_optical_frame")),
                 (t, "/mobile_2/infra2/image_rect_raw", image(t, "camera_infra2_optical_frame")),
                 (t, "/mobile_2/infra2/camera_info", info(t, "camera_infra2_optical_frame", tx=-40.4)),
                 (t, "/mobile_2/depth/image_rect_raw", image(t, "camera_depth_optical_frame", "16UC1")),
                 (t, "/mobile_2/depth/camera_info", info(t, "camera_depth_optical_frame", k=depth_k))]
    w = BagWriter(path)
    for t, topic, msg in sorted(recs, key=lambda r: r[0]):
        w.write(topic, msg, t)
    w.close()


def refused(f, what, fn, *a, **kw):
    try:
        fn(*a, **kw)
    except MK.Refused:
        return
    f.append("%s: not refused" % what)


def check_config(tmp, f):
    bag = os.path.join(tmp, "bag")
    write_bag(bag)
    run = lambda name: os.path.join(tmp, "runs", name)                 # noqa: E731
    T_sl, T_si = T_of(*T_SENSOR_LIDAR), T_of(*T_SENSOR_IMU)
    T_imu_lidar = np.linalg.inv(T_si) @ T_sl

    r = MK.make("kiss_icp", bag, "mobile_1", run("kiss"))
    c = yaml.safe_load(open(os.path.join(run("kiss"), "config", "kiss_icp.yaml")))
    d = c["kiss_icp_node"]["ros__parameters"]["data"]
    if d["min_range"] != 1.0 or d["deskew"] is not True:
        f.append("kiss_icp: data %s" % d)
    if r["tum_frame"] != "os_lidar_mobile1" or r["play_topics"] != ["/mobile_1/ouster/points"]:
        f.append("kiss_icp: frame %s, play %s" % (r["tum_frame"], r["play_topics"]))
    if abs(r["key_rate"]["hz"] - 10.0) > 0.2:
        f.append("kiss_icp: key rate %s" % r["key_rate"])

    r = MK.make("fast_lio", bag, "mobile_1", run("fastlio"))
    p = yaml.safe_load(open(os.path.join(run("fastlio"), "config", "fast_lio.yaml")))["/**"]["ros__parameters"]
    Te = np.eye(4)
    Te[:3, :3] = np.array(p["mapping"]["extrinsic_R"]).reshape(3, 3)
    Te[:3, 3] = p["mapping"]["extrinsic_T"]
    if np.abs(Te - T_imu_lidar).max() > 1e-6:
        f.append("fast_lio: extrinsic\n%s\nwant\n%s" % (Te, T_imu_lidar))
    if p["preprocess"]["scan_line"] != 64 or p["preprocess"]["blind"] != 1.0 or r["tum_frame"] != "os_imu_mobile1":
        f.append("fast_lio: preprocess %s, frame %s" % (p["preprocess"], r["tum_frame"]))
    if p["publish"]["scan_publish_en"] or p["pcd_save"]["pcd_save_en"]:
        f.append("fast_lio: publishes more than its odometry")

    MK.make("rko_lio", bag, "mobile_1", run("rko"))
    c = yaml.safe_load(open(os.path.join(run("rko"), "config", "rko_lio.yaml")))
    v = c["extrinsic_imu2base_quat_xyzw_xyz"]
    Tr = T_of(v[4:], v[:4])
    if np.abs(Tr - np.linalg.inv(T_imu_lidar)).max() > 1e-6 or c["base_frame"] != "os_lidar_mobile1":
        f.append("rko_lio: imu2base\n%s\nwant\n%s, base %s" % (Tr, np.linalg.inv(T_imu_lidar), c["base_frame"]))

    rl = MK.make("fast_lio", bag, "mobile_1", run("fastlio_relay"))["relay"]
    if rl != {"in": "/replay/mobile_1/ouster/points", "out": "/mobile_1/ouster/points", "hold_s": 0.1}:
        f.append("fast_lio: relay %s" % rl)
    # mobile_2: no /tf_static link between its Ouster's LiDAR and IMU -> the GLIM config's T_lidar_imu
    r2 = MK.make("rko_lio", bag, "mobile_2", run("rko_m2"))
    c = yaml.safe_load(open(os.path.join(run("rko_m2"), "config", "rko_lio.yaml")))
    v = c["extrinsic_imu2base_quat_xyzw_xyz"]
    import re as _re
    g = [float(x) for x in _re.search(r'"T_lidar_imu"\s*:\s*\[([^\]]*)\]',
                                      open(MK.GLIM_CONFIG % "mobile_2").read()).group(1).split(",")]
    if np.abs(T_of(v[4:], v[:4]) - T_of(g[:3], g[3:7])).max() > 1e-9 \
            or "glim" not in r2["settings"]["extrinsic_imu2base"][0]:
        f.append("rko_lio mobile_2: imu2base %s from %s, want GLIM's %s"
                 % (v, r2["settings"]["extrinsic_imu2base"][0], g))
    refused(f, "openvins without --imu-noise", MK.make, "openvins", bag, "mobile_2", run("ov0"))
    noise = os.path.join(tmp, "noise.yaml")
    yaml.safe_dump({k: 1e-3 for k in MK.NOISE_KEYS}, open(noise, "w"))
    r = MK.make("openvins", bag, "mobile_2", run("ov"), imu_noise=noise)
    text = open(os.path.join(run("ov"), "config", "kalibr_imucam_chain.yaml")).read()
    cams = yaml.safe_load(text.replace("%YAML:1.0", ""))
    links = {c[1]: T_of(c[2], c[3]) for c in CAM_TF}
    for cam, fr in (("cam0", "camera_infra1_optical_frame"), ("cam1", "camera_infra2_optical_frame")):
        want = np.linalg.inv(links[fr]) @ links["camera_imu_optical_frame"]
        got = np.array(cams[cam]["T_cam_imu"])
        if np.abs(got - want).max() > 1e-9:
            f.append("openvins %s: T_cam_imu\n%s\nwant\n%s" % (cam, got, want))
        if cams[cam]["intrinsics"] != [425.0, 425.0, 424.0, 240.0] or cams[cam]["resolution"] != [848, 480]:
            f.append("openvins %s: intrinsics %s" % (cam, cams[cam]["intrinsics"]))
    imu = yaml.safe_load(open(os.path.join(run("ov"), "config", "kalibr_imu_chain.yaml")).read()
                         .replace("%YAML:1.0", ""))["imu0"]
    if abs(imu["update_rate"] - 200.0) > 0.5 or imu["rostopic"] != "/mobile_2/imu":
        f.append("openvins: imu %s Hz on %s" % (imu["update_rate"], imu["rostopic"]))
    if r["tum_frame"] != "camera_imu_optical_frame" or abs(r["stamp_tol_s"] - 0.5 / 30) > 1e-9:
        f.append("openvins: frame %s, tol %s" % (r["tum_frame"], r["stamp_tol_s"]))

    r = MK.make("rtabmap_rgbd", bag, "mobile_2", run("rtab"))
    if "frame_id:=camera_infra1_optical_frame" not in r["launch"] or r["key_topic"] != "/mobile_2/infra1/image_rect_raw":
        f.append("rtabmap_rgbd: launch %s" % r["launch"])

    refused(f, "a camera method on mobile_1", MK.make, "rtabmap_rgbd", bag, "mobile_1", run("zed"))
    refused(f, "a run folder holding a run", MK.make, "kiss_icp", bag, "mobile_1", run("kiss"))
    bad = os.path.join(tmp, "bag_other_depth")
    write_bag(bad, depth_k=[430.0] + K[1:])
    refused(f, "rtabmap_rgbd with depth outside the left camera", MK.make, "rtabmap_rgbd", bad, "mobile_2",
            run("rtab_bad"))
    slow = os.path.join(tmp, "bag_1fps")
    write_bag(slow, cam_hz=1.0)
    refused(f, "a 1 fps camera", MK.make, "rtabmap_rgbd", slow, "mobile_2", run("slow"))


def msgs(rows):
    by = {}
    for w, role, topic, stamp, b in rows:
        by.setdefault(topic, {"role": role, "m": []})["m"].append((w, stamp, b))
    for t in by.values():
        t["m"].sort()
    return by


def check_summary(tmp, f):
    ms = 1_000_000
    # FAST-LIO2: 10 scans, out stamped 99.9 ms after the scan; scan 3 answered 150 ms late
    # (scan 4 has arrived by then), scan 6 never
    rows = []
    for k in range(10):
        st, w = T0 + k * 100 * ms, T0 + k * 100 * ms + 5 * ms
        rows.append((w, "in", "/pts", st, 1000))
        if k == 6:
            continue
        rows.append((w + (150 if k == 3 else 20) * ms, "out", "/odo", st + 99_900_000, 100))
    t = SM.latency(msgs(rows), "/pts", "/odo", 0.0)
    if t["dropped"] != 1 or t["answered"] != 9 or t["repeats"] or t["unmatched_outputs"]:
        f.append("fast-lio-like: %s" % t)
    if abs(t["latency_ms"]["max"] - 150.0) > 1e-6 or abs(t["latency_ms"]["median"] - 20.0) > 1e-6:
        f.append("fast-lio-like latency %s (want 20 median, 150 max)" % t["latency_ms"])
    if t["kept_up"]:
        f.append("fast-lio-like: a 10 % drop kept up")
    # OpenVINS: out stamped 3 ms before its image; tol 16.7 ms
    rows = []
    for k in range(30):
        st = T0 + int(k * S / 30)
        rows += [(st + 2 * ms, "in", "/img", st, 400),
                 (st + 12 * ms, "out", "/pose", st - 3 * ms, 100)]
    t = SM.latency(msgs(rows), "/img", "/pose", 0.5 / 30)
    if t["dropped"] or not t["kept_up"] or abs(t["latency_ms"]["max"] - 10.0) > 1e-6:
        f.append("openvins-like: %s" % t)
    t0 = SM.latency(msgs(rows), "/img", "/pose", 0.0)                 # without the tolerance:
    if t0["answered"] == 30:                                           # each matched the image before
        f.append("openvins-like without tolerance matched every image to itself")
    # repeats: two outputs for one input
    rows = [(T0, "in", "/a", T0, 1), (T0 + 10 * ms, "out", "/b", T0, 1), (T0 + 20 * ms, "out", "/b", T0, 1)]
    t = SM.latency(msgs(rows), "/a", "/b", 0.0)
    if t["repeats"] != 1 or t["latency_ms"]["max"] != 10.0:
        f.append("repeats: %s" % t)
    # resources windowed to the play; summary.json written
    run = os.path.join(tmp, "sum")
    os.makedirs(run)
    with open(os.path.join(run, "messages.csv"), "w") as fh:
        fh.write("wall_ns,role,topic,stamp_ns,bytes\n")
        for k in range(10):
            fh.write("%d,in,/pts,%d,2097152\n" % (T0 + k * S, T0 + k * S))
            fh.write("%d,out,/odo,%d,100\n" % (T0 + k * S + 30 * ms, T0 + k * S))
    with open(os.path.join(run, "resources.csv"), "w") as fh:
        fh.write("wall_ns,n_procs,threads,method_cpu_pct,method_rss_mb,system_cpu_pct,mem_used_mb,"
                 "gpu_util_pct,gpu_mem_mb,power_mw\n")
        fh.write("%d,1,4,900.0,9000.0,90.0,1000,,,\n" % (T0 - 5 * S))       # before the play
        for k in range(9):
            fh.write("%d,1,4,%.1f,%.1f,10.0,1000,,,\n" % (T0 + k * S + S // 2, 50.0 + k, 100.0 + k))
    json.dump({"method": "x", "machine": "y", "key_topic": "/pts", "out_topic": "/odo"},
              open(os.path.join(run, "run.json"), "w"))
    s = SM.summarize(run)
    r = s["resources"]
    if r["samples"] != 9 or r["method_cpu_pct"]["max"] != 58.0 or r["method_rss_mb_peak"] != 108.0:
        f.append("resources not windowed to the play: %s" % r)
    if abs(s["topics"]["/pts"]["mb_per_s"] - 10 * 2.0 / 9) > 1e-3 or abs(s["topics"]["/pts"]["hz"] - 1.0) > 1e-9:
        f.append("topic rate %s" % s["topics"]["/pts"])
    if not s["timing"]["kept_up"] or not os.path.exists(os.path.join(run, "summary.json")):
        f.append("summary: %s" % s["timing"])
    SM.report(s)
    # send times, when logged, time the latency: the probe got each scan 40 ms late
    # (its own delivery), the method answered 25 ms after the player sent it
    run2 = os.path.join(tmp, "sent")
    os.makedirs(run2)
    with open(os.path.join(run2, "messages.csv"), "w") as fh:
        fh.write("wall_ns,role,topic,stamp_ns,bytes,src_ns\n")
        for k in range(10):
            sent = T0 + k * 100 * ms
            fh.write("%d,in,/pts,%d,3000000,%d\n" % (sent + 40 * ms, sent, sent))
            fh.write("%d,out,/odo,%d,100,%d\n" % (sent + 26 * ms, sent, sent + 25 * ms))
    json.dump({"key_topic": "/pts", "out_topic": "/odo"}, open(os.path.join(run2, "run.json"), "w"))
    t = SM.summarize(run2)["timing"]
    if t["answered"] != 10 or abs(t["latency_ms"]["median"] - 25.0) > 1e-6:
        f.append("send-time latency: %s (want 10 answered, 25 ms)" % t)


def ref_traj(n=600, dt=0.1):
    """A reference T_map_lidar: a 3 m circle at 0.5 m/s, 10 Hz."""
    t = T0 / S + np.arange(n) * dt
    a = 0.5 / 3.0 * (t - t[0])
    T = np.tile(np.eye(4), (n, 1, 1))
    T[:, :3, :3] = Rotation.from_euler("z", (a + np.pi / 2)[:, None]).as_matrix()
    T[:, 0, 3], T[:, 1, 3], T[:, 2, 3] = 3 * np.cos(a), 3 * np.sin(a), 0.4
    return t, T


def write_tum(path, t, T):
    with open(path, "w") as fh:
        for ti, Ti in zip(t, T):
            q = Rotation.from_matrix(Ti[:3, :3]).as_quat()
            fh.write("%.9f %.9f %.9f %.9f %.9f %.9f %.9f %.9f\n" % ((ti,) + tuple(Ti[:3, 3]) + tuple(q)))


def check_score(tmp, f):
    import score as SC
    tr, Tr = ref_traj()
    T_off = T_of([1.0, -2.0, 0.3], Rotation.from_euler("z", 40, degrees=True).as_quat())
    E = np.einsum("ij,njk->nik", np.linalg.inv(T_off), Tr)          # the same motion in its own odom frame
    a = SC.score(tr, E, tr, Tr)
    if a["ate"]["position_cm"]["rmse"] > 1e-6 or a["start_aligned"]["max_cm"] > 1e-6 \
            or abs(a["path"]["ratio"] - 1) > 1e-9 or a["coverage"]["pct"] != 100.0 or a["jumps"]["count"]:
        f.append("score, a perfect estimate in its own odom frame: %s" % a)
    # 1 % long: over 10 m of a 3 m circle the motion is 1 % of the chord, divided by the 10 m
    # travelled (KITTI's drift): 1 % x 6 sin(10/6) / 10
    Es = E.copy()
    Es[:, :3, 3] *= 1.01
    a = SC.score(tr, Es, tr, Tr)
    d = a.get("drift_10m_pct")
    want = 1.0 * 6 * np.sin(10.0 / 6) / 10
    if not d or abs(d["median"] - want) > 0.01 or abs(a["path"]["ratio"] - 1.01) > 1e-6 or a["ate"]["position_cm"]["rmse"] < 1:
        f.append("score, 1 %% scale: drift %s, ratio %s, ATE %s" % (d, a["path"]["ratio"], a["ate"]["position_cm"]))
    # a 2 s gap and a 1 m jump
    keep = np.ones(len(tr), bool)
    keep[100:120] = False
    Ej = E.copy()
    Ej[300:, 0, 3] += 1.0
    a = SC.score(tr[keep], Ej[keep], tr, Tr)
    if a["jumps"]["count"] != 1 or len(a["coverage"]["gaps_over_0.5s"]) != 1 or a["coverage"]["pct"] > 97:
        f.append("score, a gap and a jump: coverage %s, jumps %s" % (a["coverage"], a["jumps"]))
    # a whole run: FAST-LIO2's IMU-frame estimate through run.json, the reference by --ref
    bag = os.path.join(tmp, "bag")
    run = os.path.join(tmp, "runs", "score_fastlio")
    MK.make("fast_lio", bag, "mobile_1", run)
    T_imu_lidar = np.linalg.inv(T_of(*T_SENSOR_IMU)) @ T_of(*T_SENSOR_LIDAR)
    write_tum(os.path.join(run, "trajectory.tum"), tr, np.einsum("nij,jk->nik", E, np.linalg.inv(T_imu_lidar)))
    ref = os.path.join(tmp, "ref.tum")
    write_tum(ref, tr, Tr)
    a = SC.run_score(run, ref)
    if "error" in a or a["ate"]["position_cm"]["rmse"] > 1e-3 or a["ate"]["rotation_deg"]["rmse"] > 1e-3:
        f.append("score, FAST-LIO2's IMU frame not put in the LiDAR's: %s" % a)
    SC.report(a)


def main():
    tmp = tempfile.mkdtemp()
    f = []
    try:
        check_config(tmp, f)
        check_summary(tmp, f)
        check_score(tmp, f)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    if f:
        print("FAIL\n  " + "\n  ".join(f))
        sys.exit(1)
    print("odometry ok")


if __name__ == "__main__":
    main()
