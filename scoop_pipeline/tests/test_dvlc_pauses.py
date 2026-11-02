#!/usr/bin/env python
"""Self-tests for processing/dvlc_pauses.py. Needs mcap and rosbags.

A synthetic merged bag (15 s): Ouster points, VSLAM odometry, a RealSense
IMU, a colour image and its camera_info, plus a topic that is not cut.
The robot drives, stands still from 4 to 11 s, drives again; the script must
find that one pause and cut a bag with points, images and camera_info inside
it only.

    python tests/test_dvlc_pauses.py
"""
import contextlib
import importlib.util
import io
import os
import re
import shutil
import sys
import tempfile
import traceback
import warnings

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.simplefilter("ignore", FutureWarning)

from scoop import bag, rosmsg                                          # noqa: E402

TS = rosmsg.typestore()
SCRIPT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "processing", "dvlc_pauses.py")
STILL = (4.0, 11.0)                 # s from the bag's start
T0, T1 = 1_700_000_000 * 10**9, 1_700_000_015 * 10**9
FAILED = []


def _stamp(ns):
    return TS.types["builtin_interfaces/msg/Time"](sec=ns // 10**9, nanosec=ns % 10**9)


def _hdr(ns, frame):
    return TS.types["std_msgs/msg/Header"](stamp=_stamp(ns), frame_id=frame)


def make_recording(d):
    """The merged bag as a rosbag2 folder; returns (folder, t0 ns)."""
    from mcap.writer import CompressionType, Writer
    t0, t1 = T0, T1

    V3 = TS.types["geometry_msgs/msg/Vector3"]
    Q = TS.types["geometry_msgs/msg/Quaternion"]
    P = TS.types["geometry_msgs/msg/Point"]
    rng = np.random.default_rng(1)

    def moving(ns):
        s = (ns - t0) * 1e-9
        return not (STILL[0] <= s <= STILL[1])

    extra = []                                       # (topic, type, log ns, bytes)
    pts = np.zeros(50, [("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("intensity", "<f4")])
    pts["x"] = np.arange(50)
    for ns in range(t0, t1, 100_000_000):            # points 10 Hz
        m = rosmsg.pointcloud2(pts, ns, "os_lidar")
        extra.append(("/mobile_2/ouster/points", "sensor_msgs/msg/PointCloud2", ns,
                      rosmsg.serialize(m)))
        m = rosmsg.string("not cut")
        extra.append(("/mobile_2/other", "std_msgs/msg/String", ns, rosmsg.serialize(m)))
    x = 0.0
    for ns in range(t0, t1, 33_333_333):             # odometry 30 Hz
        if moving(ns):
            x += 0.5 / 30                            # 0.5 m/s
        G = TS.types
        pose = G["geometry_msgs/msg/PoseWithCovariance"](
            pose=G["geometry_msgs/msg/Pose"](position=P(x=x, y=0.0, z=0.0),
                                             orientation=Q(x=0.0, y=0.0, z=0.0, w=1.0)),
            covariance=np.zeros(36))
        tw = G["geometry_msgs/msg/TwistWithCovariance"](
            twist=G["geometry_msgs/msg/Twist"](linear=V3(x=0.0, y=0.0, z=0.0),
                                               angular=V3(x=0.0, y=0.0, z=0.0)),
            covariance=np.zeros(36))
        m = G["nav_msgs/msg/Odometry"](header=_hdr(ns, "odom"), child_frame_id="base",
                                       pose=pose, twist=tw)
        extra.append(("/mobile_2/visual_slam/tracking/odometry", "nav_msgs/msg/Odometry", ns,
                      bytes(TS.serialize_cdr(m, "nav_msgs/msg/Odometry"))))
    for ns in range(t0, t1, 5_000_000):              # IMU 200 Hz
        k = 0.3 if moving(ns) else 0.01
        g = rng.normal(0, k * 0.2, 3)
        a = np.array([0.0, 0.0, 9.81]) + rng.normal(0, k, 3)
        m = TS.types["sensor_msgs/msg/Imu"](
            header=_hdr(ns, "imu"), orientation=Q(x=0.0, y=0.0, z=0.0, w=1.0),
            orientation_covariance=np.zeros(9),
            angular_velocity=V3(x=g[0], y=g[1], z=g[2]), angular_velocity_covariance=np.zeros(9),
            linear_acceleration=V3(x=a[0], y=a[1], z=a[2]),
            linear_acceleration_covariance=np.zeros(9))
        extra.append(("/mobile_2/imu", "sensor_msgs/msg/Imu", ns,
                      bytes(TS.serialize_cdr(m, "sensor_msgs/msg/Imu"))))
    for ns in range(t0, t1, 66_666_666):             # colour 15 Hz
        img = TS.types["sensor_msgs/msg/Image"](
            header=_hdr(ns, "color"), height=4, width=6, encoding="rgb8", is_bigendian=0,
            step=18, data=np.full(72, 7, np.uint8))
        ci = TS.types["sensor_msgs/msg/CameraInfo"](
            header=_hdr(ns, "color"), height=4, width=6, distortion_model="plumb_bob",
            d=np.zeros(5), k=np.array([5.0, 0, 3, 0, 5, 2, 0, 0, 1]), r=np.eye(3).ravel(),
            p=np.array([5.0, 0, 3, 0, 0, 5, 2, 0, 0, 0, 1, 0]), binning_x=0, binning_y=0,
            roi=TS.types["sensor_msgs/msg/RegionOfInterest"](
                x_offset=0, y_offset=0, height=0, width=0, do_rectify=False))
        extra.append(("/mobile_2/color/image_raw", "sensor_msgs/msg/Image", ns,
                      bytes(TS.serialize_cdr(img, "sensor_msgs/msg/Image"))))
        extra.append(("/mobile_2/color/camera_info", "sensor_msgs/msg/CameraInfo", ns,
                      bytes(TS.serialize_cdr(ci, "sensor_msgs/msg/CameraInfo"))))

    rec = os.path.join(d, "rec")
    os.makedirs(rec)
    with open(os.path.join(rec, "rec_0.mcap"), "wb") as fh:
        w = Writer(fh, compression=CompressionType.NONE)
        w.start("ros2", "test")
        chans, schemas, allm = {}, {}, []
        for topic, typ, ns, data in extra:
            if topic not in chans:
                if typ not in schemas:
                    text, _ = TS.generate_msgdef(typ, ros_version=2)
                    schemas[typ] = w.register_schema(typ, "ros2msg", text.encode())
                chans[topic] = w.register_channel(topic, "cdr", schemas[typ])
            allm.append((ns, chans[topic], data))
        for i, (ns, ch, data) in enumerate(sorted(allm, key=lambda r: r[0])):
            w.add_message(ch, log_time=ns, publish_time=ns, data=data, sequence=i)
        w.finish()
    return rec, t0


_spec = importlib.util.spec_from_file_location("dvlc_pauses", SCRIPT)
dvlc_pauses = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dvlc_pauses)


def run(*args, fail=False):
    """The script's main() on these arguments: its stdout, or with
    fail=True the message it exited with."""
    out, argv = io.StringIO(), sys.argv
    sys.argv = [SCRIPT, *args]
    try:
        with contextlib.redirect_stdout(out):
            dvlc_pauses.main()
    except SystemExit as e:
        if fail:
            return str(e.code)
        raise AssertionError(f"{out.getvalue()}exit: {e.code}")
    finally:
        sys.argv = argv
    if fail:
        raise AssertionError("expected the script to exit with an error")
    return out.getvalue()


def check(name, fn):
    try:
        fn()
        print("  ok    %s" % name)
    except Exception:
        FAILED.append(name)
        print("  FAIL  %s" % name)
        traceback.print_exc()


def main():
    d = tempfile.mkdtemp(prefix="dvlc_pauses_")
    try:
        rec, t0 = make_recording(d)
        common = []

        def test_list_finds_the_one_pause():
            out = run(rec, os.path.join(d, "unused"), "--list", *common)
            rows = [ln.split() for ln in out.splitlines()
                    if re.match(r"\s+\d+\s+[\d.]+\s+[\d.]+\s+[\d.]+", ln)]
            assert len(rows) == 1, out
            s, e = float(rows[0][1]), float(rows[0][2])
            # still 4..11 s, +-0.5 s of windows, then 1 s trimmed off each end
            assert 4.5 <= s <= 6.0 and 9.0 <= e <= 10.5, (s, e)
            assert not os.path.exists(os.path.join(d, "unused"))

        def test_pause_bag_holds_points_images_info_inside_the_pause():
            out_dir = os.path.join(d, "out")
            run(rec, out_dir, *common)
            meta = yaml.safe_load(open(os.path.join(out_dir, "pauses.yaml")))
            assert [p["name"] for p in meta["pauses"]] == ["pause_00"]
            p = meta["pauses"][0]
            lo, hi = t0 + int(p["start_s"] * 1e9), t0 + int(p["end_s"] * 1e9)
            r = bag.open_bag(os.path.join(out_dir, "pause_00"))
            tp = r.topics()
            assert set(tp) == {"/mobile_2/ouster/points", "/mobile_2/color/image_raw",
                               "/mobile_2/color/camera_info"}, set(tp)
            n_scans = tp["/mobile_2/ouster/points"].count
            assert abs(n_scans - (p["end_s"] - p["start_s"]) * 10) <= 2, n_scans
            assert tp["/mobile_2/color/image_raw"].count >= 4 * 15
            for topic, log_ns, payload, decode in r.iter_raw():
                assert lo <= log_ns <= hi, (topic, (log_ns - t0) * 1e-9)
            src = {(t, ns): b for t, ns, b, _ in bag.open_bag(rec).iter_raw()}
            for topic, log_ns, payload, _ in r.iter_raw():
                assert bytes(payload) == bytes(src[(topic, log_ns)]), topic   # byte for byte

        def test_intervals_and_only():
            out_dir = os.path.join(d, "manual")
            run(rec, out_dir, "--intervals", "1-2,6-8", "--only", "1", *common)
            meta = yaml.safe_load(open(os.path.join(out_dir, "pauses.yaml")))
            assert [p["name"] for p in meta["pauses"]] == ["pause_01"]
            assert sorted(os.listdir(out_dir)) == ["pause_01", "pauses.yaml"]

        def test_db3_bag_cut_by_intervals():
            # a .db3 calibration bag: no odometry or IMU, no stored definitions
            from rosbags.rosbag2 import Writer
            db3 = os.path.join(d, "calib_db3")
            src = bag.open_bag(rec)
            keep = {t: i.msgtype for t, i in src.topics().items()
                    if t in ("/mobile_2/ouster/points", "/mobile_2/color/image_raw",
                             "/mobile_2/color/camera_info")}
            with Writer(db3) as w:
                conns = {t: w.add_connection(t, ty, typestore=TS) for t, ty in keep.items()}
                for topic, log_ns, payload, _ in src.iter_raw(list(keep)):
                    w.write(conns[topic], log_ns, bytes(payload))
            out_dir = os.path.join(d, "db3_cut")
            run(db3, out_dir, "--intervals", "2-5", *common)
            r = bag.open_bag(os.path.join(out_dir, "pause_00"))
            assert set(r.topics()) == set(keep), set(r.topics())
            got = list(r.iter_raw())
            assert got and all(t0 + 2 * 10**9 <= ns <= t0 + 5 * 10**9 for _, ns, _, _ in got)
            orig = {(t, ns): bytes(b) for t, ns, b, _ in bag.open_bag(db3).iter_raw()}
            for topic, log_ns, payload, _ in got:
                assert bytes(payload) == orig[(topic, log_ns)], topic
            frames = list(bag.iter_images(r, "/mobile_2/color/image_raw"))
            assert frames and frames[0].image.shape == (4, 6, 3)

        def test_refuses_a_full_output_folder():
            msg = run(rec, os.path.join(d, "out"), *common, fail=True)
            assert "not empty" in msg, msg

        for name, fn in sorted(locals().items()):
            if name.startswith("test_"):
                check(name, fn)
    finally:
        shutil.rmtree(d, ignore_errors=True)
    n = 5
    print(f"\n{n - len(FAILED)}/{n} passed")
    sys.exit(1 if FAILED else 0)


if __name__ == "__main__":
    main()
