#!/usr/bin/env python
"""Self-tests for per-machine settings (configs/recording.yaml machines:):
a mobile_2 folder is processed with its own namespace and frame suffix,
stops after retimed, and the merge takes its retimed bag. Needs ouster-sdk,
mcap and rosbags.

    python tests/test_machines.py
"""
import os
import shutil
import sys
import tempfile
import traceback
import warnings

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.simplefilter("ignore", FutureWarning)

import test_ouster as to                                               # noqa: E402
from scoop import bag, ouster, recording, rosmsg, session              # noqa: E402

FAILED = []
NS2 = "/mobile_2/ouster"


def test_settings_for_merges_a_machine_over_the_top_level():
    s = recording.load_settings()
    m1 = recording.settings_for(s, "mobile_1")                 # no block: the top level
    assert m1["ouster"]["packets_ns"] == "/ouster" and "machines" not in m1
    assert m1["ouster"]["frame_suffix"] == "_mobile1"
    m2 = recording.settings_for(s, "mobile_2")
    o = m2["ouster"]
    assert (o["packets_ns"], o["out_ns"], o["frame_suffix"]) == (NS2, NS2, "_mobile2")
    assert o["remap"] == [] and o["imu_sync"] == []           # lists replaced, not merged
    assert o["copy_all"] is True and o["driver_min_range"] == s["ouster"]["driver_min_range"]
    assert m2["until"] == "retimed" and m2["glim"] == s["glim"]
    assert recording.settings_for(s, None)["ouster"] == s["ouster"]


def _mobile2_tree(root):
    """data/raw/20261015/mapping/mobile_2/<packets bag with /tf_static>/."""
    from mcap.reader import make_reader
    from mcap.writer import Writer
    pk = os.path.join(root, "pk.mcap")
    to.NS = NS2
    try:
        to.make_bag(pk, n_frames=120)
    finally:
        to.NS = "/ouster"
    raw = os.path.join(root, "data", "raw", "20261015", "mapping", "mobile_2")
    b = os.path.join(raw, "mirc_dataset_mapping_run_20261015_mobile_2")
    os.makedirs(b)
    name = os.path.basename(b) + "_0.mcap"
    with open(pk, "rb") as fi, open(os.path.join(b, name), "wb") as fo:
        recs = list(make_reader(fi).iter_messages())
        w = Writer(fo)
        w.start("ros2", "test")
        ch = {}
        for sc, c, m in recs:
            if c.topic not in ch:
                ch[c.topic] = w.register_channel(c.topic, "cdr",
                                                 w.register_schema(sc.name, sc.encoding, sc.data))
        ctf = w.register_channel("/tf_static", "cdr", w.register_schema(
            "tf2_msgs/msg/TFMessage", "ros2msg", rosmsg.msgdef("tf2_msgs/msg/TFMessage").encode()))
        t0 = recs[0][2].log_time
        w.add_message(ctf, log_time=t0, publish_time=t0, data=rosmsg.serialize(rosmsg.tf_message(
            [("os_sensor", "os_lidar", [0, 0, 0.036], [0, 0, 1, 0]),
             ("camera_link", "camera_color_optical_frame", [0, 0.015, 0], [-0.5, 0.5, -0.5, 0.5])],
            t0)))
        for sc, c, m in recs:
            w.add_message(ch[c.topic], log_time=m.log_time, publish_time=m.log_time, data=m.data)
        w.finish()
    yaml.safe_dump({"rosbag2_bagfile_information": {
        "version": 5, "storage_identifier": "mcap", "relative_file_paths": [name]}},
        open(os.path.join(b, "metadata.yaml"), "w"))
    return raw, b


def test_mobile_2_folder_processed_with_its_own_settings(d):
    raw, b = _mobile2_tree(d)
    s = recording.settings_for(recording.load_settings(), os.path.basename(raw))
    rec = recording.find_recording(raw, None, s["ouster"]["packets_ns"],
                                   recording.zed_bags(s["zed"]))
    assert rec.bag == recording.Path(b).resolve()
    assert str(rec.work).endswith(os.path.join("work", "20261015", "mapping", "mobile_2"))
    saved = ouster.sensor_info
    ouster.sensor_info = lambda src: to.meta()         # SDK 1.0 and the synthetic JSON
    try:
        st = recording.process(rec, s, log=lambda *_: None)
    finally:
        ouster.sensor_info = saved
    assert st["decoded"] and st["retimed"] and not st["glim"], st   # until: retimed
    assert not os.path.exists(rec.step("glim"))
    r = bag.open_bag(rec.step("retimed"))
    topics = r.topics()
    assert f"{NS2}/points" in topics and f"{NS2}/imu" in topics
    assert f"{NS2}/lidar_packets" not in topics and f"{NS2}/imu_packets" not in topics
    pts = next(r.iter_messages([f"{NS2}/points"]))[2]
    imu = next(r.iter_messages([f"{NS2}/imu"]))[2]
    tf = next(r.iter_messages(["/tf_static"]))[2]
    assert pts.header.frame_id == "os_lidar_mobile2"
    assert imu.header.frame_id == "os_imu_mobile2"
    assert [(t.header.frame_id, t.child_frame_id) for t in tf.transforms] == \
        [("os_sensor_mobile2", "os_lidar_mobile2"), ("camera_link", "camera_color_optical_frame")]

    # the merge takes mobile_2's retimed bag, checked against its own topics
    plan = session.RecordPlan(None)
    u = session._unit("mobile_2", recording.Path(raw), [recording.Path(b)], plan,
                      recording.load_settings(), None, None)
    assert u.processed and u.bags == [rec.step("retimed")], u.bags
    assert f"{NS2}/points" in u.expected and f"{NS2}/lidar_packets" not in u.expected

    # static_tf.yaml's calibration hangs from os_lidar_mobile2 in that tree
    from scoop import tftree
    cals = tftree.load_calibrations(yaml.safe_load(open(os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs", "static_tf.yaml"))))
    mine = [c for c in cals if c.parent == "os_lidar_mobile2"]
    assert [c.child for c in mine] == ["camera_color_optical_frame"]
    edges = tftree.tf_edges([rec.step("retimed")])
    added, notes = tftree.attach(mine, edges)
    assert added, notes


def check(name, fn, *args):
    try:
        fn(*args)
        print("  ok    %s" % name)
    except Exception:
        FAILED.append(name)
        print("  FAIL  %s" % name)
        traceback.print_exc()


def main():
    d = tempfile.mkdtemp(prefix="machines_")
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    try:
        for name, fn in tests:
            args = []
            if fn.__code__.co_argcount == 1:
                args = [os.path.join(d, name)]
                os.makedirs(args[0])
            check(name, fn, *args)
    finally:
        shutil.rmtree(d, ignore_errors=True)
    print(f"\n{len(tests) - len(FAILED)}/{len(tests)} passed")
    sys.exit(1 if FAILED else 0)


if __name__ == "__main__":
    main()
