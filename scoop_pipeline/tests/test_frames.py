#!/usr/bin/env python
"""Self-tests for scoop.frames (renaming TF frames in a bag) and
decode_ouster's frame_id. Needs mcap and rosbags (+ ouster-sdk for the last).

    python tests/test_frames.py
"""
import os
import shutil
import sys
import tempfile
import traceback
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.simplefilter("ignore", FutureWarning)

from scoop import bag, frames, rosmsg                                  # noqa: E402
from scoop.bag import BagError, TopicSchema                            # noqa: E402
from scoop.bagwrite import BagWriter                                   # noqa: E402

TS = rosmsg.typestore()
FAILED = []
MAP = {"os_lidar": "os_lidar_mobile2"}


def schema(name):
    return TopicSchema(name, "ros2msg", rosmsg.msgdef(name).encode(), "cdr")


def cloud(frame="os_lidar"):
    pts = np.zeros(7, [("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("intensity", "<f4")])
    pts["x"] = np.arange(7)
    return rosmsg.pointcloud2(pts, 123_000_000_456, frame)


def imu(frame="os_lidar"):
    return rosmsg.imu(5, frame, [1, 2, 3], [4, 5, 6], accel_cov=0.01)


def odom(frame="odom", child="os_lidar"):
    G = TS.types
    V3, Q = G["geometry_msgs/msg/Vector3"], G["geometry_msgs/msg/Quaternion"]
    return G["nav_msgs/msg/Odometry"](
        header=rosmsg.header(7, frame), child_frame_id=child,
        pose=G["geometry_msgs/msg/PoseWithCovariance"](
            pose=G["geometry_msgs/msg/Pose"](position=G["geometry_msgs/msg/Point"](x=1.0, y=2.0, z=3.0),
                                             orientation=Q(x=0.0, y=0.0, z=0.0, w=1.0)),
            covariance=np.arange(36, dtype=float)),
        twist=G["geometry_msgs/msg/TwistWithCovariance"](
            twist=G["geometry_msgs/msg/Twist"](linear=V3(x=0.0, y=0.0, z=0.0),
                                               angular=V3(x=0.0, y=0.0, z=0.0)),
            covariance=np.zeros(36)))


def ser(m):
    return rosmsg.serialize(m)


def de(b, name):
    return TS.deserialize_cdr(bytes(b), name)


def test_header_splice_is_what_a_serializer_writes():
    for m in (cloud(), imu()):
        name = m.__msgtype__
        out = frames.rename_payload(schema(name), ser(m), MAP)
        # the same message built with the new name, serialized from scratch
        want = ser(cloud("os_lidar_mobile2") if name.endswith("PointCloud2")
                   else imu("os_lidar_mobile2"))
        assert out == want, name


def test_other_lengths_go_through_the_decoder():
    for new in ("l", "os_lidar_m2", "a_much_longer_lidar_frame_name"):
        out = frames.rename_payload(schema("sensor_msgs/msg/Imu"), ser(imu()), {"os_lidar": new})
        m = de(out, "sensor_msgs/msg/Imu")
        assert m.header.frame_id == new
        assert list(m.angular_velocity.__dict__.values())[:3] == [4.0, 5.0, 6.0]
        assert m.linear_acceleration_covariance[0] == 0.01


def test_untouched_messages_come_back_none():
    assert frames.rename_payload(schema("sensor_msgs/msg/Imu"), ser(imu("os_imu")), MAP) is None
    s = schema("std_msgs/msg/String")
    assert frames.rename_payload(s, ser(rosmsg.string("os_lidar")), MAP) is None   # no header


def test_tf_and_odometry_child_frames():
    tf = rosmsg.tf_message([("os_sensor", "os_lidar", [0, 0, 0.036], [0, 0, 0, 1]),
                            ("os_lidar", "zed", [1, 2, 3], [0, 0, 0, 1]),
                            ("map", "odom", [0, 0, 0], [0, 0, 0, 1])], 9)
    m = de(frames.rename_payload(schema("tf2_msgs/msg/TFMessage"), ser(tf), MAP),
           "tf2_msgs/msg/TFMessage")
    assert [(t.header.frame_id, t.child_frame_id) for t in m.transforms] == \
        [("os_sensor", "os_lidar_mobile2"), ("os_lidar_mobile2", "zed"), ("map", "odom")]
    assert m.transforms[1].transform.translation.y == 2.0
    o = de(frames.rename_payload(schema("nav_msgs/msg/Odometry"), ser(odom()), MAP),
           "nav_msgs/msg/Odometry")
    assert (o.header.frame_id, o.child_frame_id) == ("odom", "os_lidar_mobile2")
    assert o.pose.covariance[35] == 35.0
    assert frames.rename_payload(schema("nav_msgs/msg/Odometry"),
                                 ser(odom(child="base")), MAP) is None


def test_unknown_type_with_odd_length_change_is_refused():
    s = TopicSchema("my_pkg/msg/Thing", "ros2msg", b"std_msgs/Header header\nfloat64 v", "cdr")
    payload = ser(imu())                                   # any header-first bytes
    try:
        frames.rename_payload(s, payload, {"os_lidar": "x"})
    except BagError as e:
        assert "my_pkg/msg/Thing" in str(e)
    else:
        raise AssertionError("expected a refusal")
    out = frames.rename_payload(s, payload, MAP)           # +8 bytes: spliced, no decoder
    assert b"os_lidar_mobile2\0" in out


def test_suffix_names_all_three_ouster_frames(d):
    from scoop import ouster
    assert ouster.frame_names("") == {}
    m = ouster.frame_names("_mobile2")
    assert m == {"os_lidar": "os_lidar_mobile2", "os_sensor": "os_sensor_mobile2",
                 "os_imu": "os_imu_mobile2"}
    src, out = os.path.join(d, "src"), os.path.join(d, "out")
    with BagWriter(src) as w:
        w.write("/points", cloud(), 1)
        w.write("/imu", imu("os_imu"), 2)
        w.write("/tf_static", rosmsg.tf_message(
            [("os_sensor", "os_lidar", [0, 0, 0], [0, 0, 0, 1]),
             ("os_sensor", "os_imu", [0, 0, 0], [0, 0, 0, 1])], 0), 3)
    frames.rename_bag(src, out, m, log=lambda *_: None)
    got = {t: dec(p) for t, _, p, dec in bag.open_bag(out).iter_raw()}
    assert got["/points"].header.frame_id == "os_lidar_mobile2"
    assert got["/imu"].header.frame_id == "os_imu_mobile2"
    assert [(t.header.frame_id, t.child_frame_id) for t in got["/tf_static"].transforms] == \
        [("os_sensor_mobile2", "os_lidar_mobile2"), ("os_sensor_mobile2", "os_imu_mobile2")]


def test_parse_renames():
    assert frames.parse_renames(["a:=b", " c := d "]) == {"a": "b", "c": "d"}
    for bad in ("a", "a:=", ":=b", "a:=a"):
        try:
            frames.parse_renames([bad])
        except ValueError:
            continue
        raise AssertionError(bad)


def test_rename_bag_end_to_end(d):
    src, out = os.path.join(d, "src"), os.path.join(d, "out")
    msgs = [("/points", cloud(), 10), ("/imu", imu("os_imu"), 11),
            ("/tf_static", rosmsg.tf_message([("os_sensor", "os_lidar", [0, 0, 0], [0, 0, 0, 1])], 0), 12),
            ("/points", cloud(), 20)]
    with BagWriter(src) as w:
        w.set_qos("/tf_static", "- durability: 1")
        for t, m, ns in msgs:
            w.write(t, m, ns)
    counts = frames.rename_bag(src, out, MAP, log=lambda *_: None)
    assert counts == {"/points": 2, "/tf_static": 1}, counts
    r = bag.open_bag(out)
    got = [(t, ns) for t, ns, _, _ in r.iter_raw()]
    assert got == [("/points", 10), ("/imu", 11), ("/tf_static", 12), ("/points", 20)], got
    for t, ns, payload, dec in r.iter_raw():
        m = dec(payload)
        if t == "/points":
            assert m.header.frame_id == "os_lidar_mobile2"
        elif t == "/imu":
            assert m.header.frame_id == "os_imu"
        else:
            assert m.transforms[0].child_frame_id == "os_lidar_mobile2"
    assert "durability: 1" in bag.topic_qos(r)["/tf_static"]


def test_decode_frame_suffix_names_the_cloud_imu_and_the_copied_tf(d):
    try:
        import test_ouster as to
    except ImportError as e:                                  # no ouster-sdk
        print(f"        (skipped: {e})")
        return
    from mcap.reader import make_reader
    from mcap.writer import Writer
    from scoop import ouster, replay
    pk = os.path.join(d, "pk.mcap")
    to.make_bag(pk, n_frames=12)
    # the same bag plus a /tf_static naming os_lidar, as the driver publishes
    src = os.path.join(d, "pk_tf.mcap")
    with open(pk, "rb") as fi, open(src, "wb") as fo:
        recs = list(make_reader(fi).iter_messages())
        w = Writer(fo)
        w.start("ros2", "test")
        ch = {}
        for sc, c, m in recs:
            if c.topic not in ch:
                ch[c.topic] = w.register_channel(c.topic, "cdr",
                                                 w.register_schema(sc.name, sc.encoding, sc.data))
        tfs = w.register_schema("tf2_msgs/msg/TFMessage", "ros2msg",
                                rosmsg.msgdef("tf2_msgs/msg/TFMessage").encode())
        ctf = w.register_channel("/tf_static", "cdr", tfs)
        t0 = recs[0][2].log_time
        w.add_message(ctf, log_time=t0, publish_time=t0, data=ser(rosmsg.tf_message(
            [("os_sensor", "os_lidar", [0, 0, 0.036], [0, 0, 1, 0])], t0)))
        for sc, c, m in recs:
            w.add_message(ch[c.topic], log_time=m.log_time, publish_time=m.log_time, data=m.data)
        w.finish()
    out = os.path.join(d, "dec")
    replay.decode_ouster_bag(src, out, packets_ns=to.NS, out_ns="/mobile_2/ouster",
                             metadata=to.meta(), copy_all=True, frame_suffix="_mobile2",
                             log=lambda *_: None)
    r = bag.open_bag(out)
    pts = next(r.iter_messages(["/mobile_2/ouster/points"]))[2]
    assert pts.header.frame_id == "os_lidar_mobile2"
    tf = next(r.iter_messages(["/tf_static"]))[2]
    assert (tf.transforms[0].header.frame_id, tf.transforms[0].child_frame_id) == \
        ("os_sensor_mobile2", "os_lidar_mobile2")
    imu_msg = next(r.iter_messages(["/mobile_2/ouster/imu"]))[2]
    assert imu_msg.header.frame_id == "os_imu_mobile2"
    # without a suffix: the ouster_ros names, as before
    out2 = os.path.join(d, "dec2")
    replay.decode_ouster_bag(src, out2, packets_ns=to.NS, out_ns="/mobile_2/ouster",
                             metadata=to.meta(), copy_all=True, log=lambda *_: None)
    r2 = bag.open_bag(out2)
    assert next(r2.iter_messages(["/mobile_2/ouster/points"]))[2].header.frame_id == "os_lidar"
    assert next(r2.iter_messages(["/tf_static"]))[2].transforms[0].child_frame_id == "os_lidar"
    assert next(r2.iter_messages(["/mobile_2/ouster/imu"]))[2].header.frame_id == \
        ouster.ROS_IMU_FRAME_ID


def check(name, fn, *args):
    try:
        fn(*args)
        print("  ok    %s" % name)
    except Exception:
        FAILED.append(name)
        print("  FAIL  %s" % name)
        traceback.print_exc()


def main():
    d = tempfile.mkdtemp(prefix="frames_")
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    try:
        for name, fn in tests:
            args = []
            if fn.__code__.co_argcount == 1:            # a test that wants a folder
                args = [os.path.join(d, name)]
                os.makedirs(args[0])
            check(name, fn, *args)
    finally:
        shutil.rmtree(d, ignore_errors=True)
    print(f"\n{len(tests) - len(FAILED)}/{len(tests)} passed")
    sys.exit(1 if FAILED else 0)


if __name__ == "__main__":
    main()
