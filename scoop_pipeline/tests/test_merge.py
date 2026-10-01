#!/usr/bin/env python
"""Self-tests for scoop.session and scoop.merge (the topic check and the
merge of a pass). No ROS, no real bags.

A synthetic pass: mobile_1 processed (retimed + two ZED bags in data/work),
mobile_2 raw with a topic missing, infra_1 raw and a repeat (_r2).

    python tests/test_merge.py
"""
import os
import shutil
import subprocess
import sys
import tempfile
import traceback

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np                                                  # noqa: E402

from scoop import bag, merge, recording, rosmsg, session, tftree    # noqa: E402
from scoop.bagwrite import BagWriter                                # noqa: E402

FAILED = []
S = 1_000_000_000
T0 = 1_790_218_481 * S
P = "mirc_dataset_mapping_A_20260924"
LATCHED = "- history: 1\n  depth: 1\n  reliability: 1\n  durability: 1\n"

PLAN = {"tf_recorded_by": "mobile_1",
        "check": {"skip": ["/tf_static"],
                  "modes": {"mapping": {"skip": ["/*/wifi/*"]}, "coop": {}}},
        "machines": {
    "mobile_1": {"topics": ["/ouster/lidar_packets", "/ouster/imu_packets", "/ouster/metadata",
                            "/zed/zed_node/imu/data", "/mobile_1/diagnostics"]},
    "mobile_2": {"topics": ["/mobile_2/imu", "/mobile_2/color/image_raw",
                            "/mobile_2/wifi/ping"]},                 # no wifi when mapping
    "infra_1": {"topics": ["/infra_1/radar/points_all"]},
    "mobile_1/sniffer": {"topics": ["/mobile_1/sniffer/infra_1/csi"]}}}


def quiet(*_):
    pass


def check(name, fn):
    try:
        fn()
        print("  ok    %s" % name)
    except Exception:
        FAILED.append(name)
        print("  FAIL  %s" % name)
        traceback.print_exc()


def write_bag(path, topics, n=20, t0=T0, step=S // 10, qos=None):
    """A bag with n String messages per topic, interleaved in log-time order."""
    with BagWriter(path) as w:
        for t, q in (qos or {}).items():
            w.set_qos(t, q)
        for i in range(n):
            for k, t in enumerate(topics):
                ts = t0 + i * step + k
                msg = (rosmsg.tf_message([("base", f"{t[1:]}_child", [0, 0, 0], [0, 0, 0, 1])], ts)
                       if t in ("/tf", "/tf_static") else rosmsg.string(f"{t} {i}"))
                w.write(t, msg, ts)


class Pass:
    """data/raw/20260924/mapping_A/{mobile_1,mobile_2,infra_1} + data/work."""

    def __init__(self):
        self.root = tempfile.mkdtemp()
        self.raw = os.path.join(self.root, "data", "raw", "20260924", "mapping_A")
        self.work = os.path.join(self.root, "data", "work", "20260924", "mapping_A")
        m1, m2, i1 = (os.path.join(self.raw, m) for m in ("mobile_1", "mobile_2", "infra_1"))
        for d in (m1, m2, i1):
            os.makedirs(d)
        raw1 = PLAN["machines"]["mobile_1"]["topics"] + ["/tf", "/tf_static"]
        write_bag(os.path.join(m1, f"{P}_mobile_1"), raw1)
        open(os.path.join(m1, "cam.svo2"), "w").write("svo")
        w1 = os.path.join(self.work, "mobile_1")
        settings = self.settings()
        rec = recording.Recording(recording.Path(m1), recording.Path(m1, f"{P}_mobile_1"),
                                  recording.Path(w1), recording.Path(m1, "cam.svo2"),
                                  tuple(recording.zed_bags(settings["zed"])))
        proc = session.processed_topics(raw1, settings, rec)
        zed_topics = [t for t in proc if t.startswith("/mobile_1/zed/") and "imu" not in t]
        right = [t for t in zed_topics if "/right/" in t]
        write_bag(os.path.join(w1, f"{P}_mobile_1_retimed"),
                  [t for t in proc if t not in zed_topics],
                  qos={"/tf_static": LATCHED}, t0=T0 + 3)
        write_bag(os.path.join(w1, "cam_zed"), [t for t in zed_topics if t not in right],
                  t0=T0 + 5, step=S // 15)
        write_bag(os.path.join(w1, "cam_zed_right"), right, t0=T0 + 7, step=S // 15)
        write_bag(os.path.join(m2, f"{P}_mobile_2"), ["/mobile_2/color/image_raw"],
                  t0=T0 + 11)                                      # /mobile_2/imu missing
        write_bag(os.path.join(i1, f"{P}_infra_1_r2"), ["/infra_1/radar/points_all"],
                  t0=T0 + 13, step=S // 3)
        self.plan_path = os.path.join(self.root, "record.yaml")
        yaml.safe_dump(PLAN, open(self.plan_path, "w"))
        self.plan = session.RecordPlan(self.plan_path)

    @staticmethod
    def settings():
        s = recording.load_settings()
        s["zed"]["bags"] = {"zed": ["/zed/zed_node/left/image_rect_color",
                                    "/zed/zed_node/odom"],
                            "zed_right": ["/zed/zed_node/right/image_rect_color"]}
        return s

    def close(self):
        shutil.rmtree(self.root, ignore_errors=True)


def test_machine_of():
    p = session.RecordPlan(os.path.join(ROOT, "configs", "record.yaml"))
    assert p.machine_of(f"{P}_mobile_1") == "mobile_1"
    assert p.machine_of(f"{P}_mobile_1_r2") == "mobile_1"
    assert p.machine_of(f"{P}_mobile_1_sniffer") == "mobile_1/sniffer"
    assert p.machine_of(f"{P}_mobile_12") is None
    assert p.machine_of(f"{P}_infra_1") == "infra_1"
    assert "/tf" in p.topics("mobile_1", "survey") and "/tf" not in p.topics("mobile_2")
    assert "/tf" in p.topics("mobile_1", "mapping")


def test_modes():
    t = Pass()
    try:
        p = t.plan
        assert p.mode_of("mapping_A") == "mapping"
        assert p.mode_of("run3", f"{P}_coop_2_mobile_1") == "coop"
        assert p.mode_of("unmapped_B") is None                   # a word, not a substring
        assert "/tf" in p.topics("mobile_1") and "/tf_static" not in p.topics("mobile_1")
        assert "/mobile_2/wifi/ping" not in p.topics("mobile_2", "mapping")
        assert "/mobile_2/wifi/ping" in p.topics("mobile_2", "coop")
        u = {x.machine: x for x in session.find_units(t.raw, p, t.settings(), mode="coop",
                                                      log=quiet)}
        assert u["mobile_2"].check().missing == ["/mobile_2/imu", "/mobile_2/wifi/ping"]
        real = session.RecordPlan(os.path.join(ROOT, "configs", "record.yaml"))
        m1 = real.topics("mobile_1", "mapping")
        assert "/tf" in m1 and "/mobile_1/wifi/status" not in m1 and \
            "/mobile_1/ntp/events" not in m1 and "/ouster/lidar_packets" in m1
        assert "/mobile_1/wifi/status" in real.topics("mobile_1", "survey")
        assert "/mobile_1/diagnostics" not in m1
        assert "/mobile_1/diagnostics" in real.topics("mobile_1", "survey")
    finally:
        t.close()


def test_processed_topics():
    t = Pass()
    try:
        u = {x.machine: x for x in session.find_units(t.raw, t.plan, t.settings(),
                                                      mode="mapping", log=quiet)}
        m1 = u["mobile_1"]
        assert m1.processed
        assert [b.name for b in m1.bags] == [f"{P}_mobile_1_retimed", "cam_zed", "cam_zed_right"]
        for want in ("/mobile_1/ouster/points", "/mobile_1/ouster/imu", "/mobile_1/ouster/metadata",
                     "/mobile_1/ouster/lidar_packets", "/mobile_1/zed/imu/data", "/tf",
                     "/mobile_1/diagnostics", "/mobile_1/zed/left/image_rect_color",
                     "/mobile_1/zed/right/image_rect_color"):
            assert want in m1.expected, (want, m1.expected)
        assert "/ouster/lidar_packets" not in m1.expected       # renamed
        assert m1.check().ok, m1.check()
        assert not u["mobile_2"].processed
        c = u["mobile_2"].check()
        assert c.missing == ["/mobile_2/imu"] and not c.ok
        assert u["infra_1"].check().ok                          # found by name, _r2 and all
        assert session.session_name(list(u.values()), t.plan) == P
    finally:
        t.close()


def test_merge_bags():
    t = Pass()
    try:
        units = session.find_units(t.raw, t.plan, t.settings(), log=quiet)
        inputs = [b for u in units for b in u.bags]
        out = os.path.join(t.root, "merged")
        res = merge.merge_bags(inputs, out, log=quiet)
        want = {}
        for b in inputs:
            for tp, info in bag.open_bag(b).topics().items():
                want[tp] = want.get(tp, 0) + info.count
        want["/tf_static"] = 1                                   # combined into one message
        assert res.counts == want, (res.counts, want)
        r = bag.open_bag(out)
        logs = [m.log_time for _, _, _, m in bag.iter_mcap_records(r)]
        assert logs == sorted(logs) and len(logs) == sum(want.values())
        retimed = next(b for b in inputs if b.name.endswith("_retimed"))
        a = [bytes(m.data) for _, _, _, m in bag.iter_mcap_records(bag.open_bag(retimed), ["/tf"])]
        assert a
        b = [bytes(m.data) for _, _, _, m in bag.iter_mcap_records(r, ["/tf"])]
        assert a == b                                            # byte for byte
        assert "durability: 1" in bag.topic_qos(r)["/tf_static"]   # latched
        from mcap.reader import make_reader
        with open(os.path.join(out, "merged_0.mcap"), "rb") as fh:       # default: zstd
            chunks = make_reader(fh).get_summary().chunk_indexes
            assert chunks and all(c.compression == "zstd" for c in chunks)
        outz = os.path.join(t.root, "mz")
        merge.merge_bags(inputs, outz, compression="none", log=quiet)
        with open(os.path.join(outz, "mz_0.mcap"), "rb") as fh:
            chunks = make_reader(fh).get_summary().chunk_indexes
            assert chunks and all(c.compression == "" for c in chunks)
        assert bag.open_bag(outz).topics()["/tf"].count == want["/tf"]   # reads back the same

        with BagWriter(os.path.join(t.root, "clash2")) as w:
            w.write("/tf", rosmsg.imu(T0, "x", [0, 0, 0], [0, 0, 0]), T0)
        try:
            merge.merge_bags([retimed, os.path.join(t.root, "clash2")],
                             os.path.join(t.root, "bad"), log=quiet)
        except bag.BagError:
            pass
        else:
            raise AssertionError("one topic with two types must be refused")
        try:
            merge.merge_bags(inputs, os.path.join(t.root, "full"), min_free_gb=1e9, log=quiet)
        except bag.BagError:
            pass
        else:
            raise AssertionError("a nearly full disk must stop the merge")
        assert not os.path.exists(os.path.join(t.root, "full")), "partial bag left behind"
    finally:
        t.close()


CAL = tftree.Calibration("os_lidar", "zed_left_camera_optical_frame",
                         [-0.074928, -0.066971, -0.091627],
                         [-0.497829, -0.498035, 0.501789, 0.502329])
OPTICAL_Q = [-0.5, 0.5, -0.5, 0.5]


def _tree_bag(path):
    """/tf_static: os_sensor->os_lidar, the ZED's camera chain; /tf: odom."""
    with BagWriter(path) as w:
        w.write("/tf_static", rosmsg.tf_message([
            ("os_sensor", "os_lidar", [0, 0, 0.036], [0, 0, 1, 0]),
            ("zed_camera_link", "zed_left_camera_frame", [0, 0.06, 0], [0, 0, 0, 1]),
            ("zed_left_camera_frame", "zed_left_camera_optical_frame", [0, 0, 0], OPTICAL_Q)],
            T0), T0)
        for i in range(10):
            w.write("/tf", rosmsg.tf_message([("map_zed", "odom_zed", [0, 0, 0], [0, 0, 0, 1]),
                                              ("odom_zed", "zed_camera_link", [i, 0, 0],
                                               [0, 0, 0, 1])], T0 + i * S), T0 + i * S)


def test_tf_math_and_attach():
    M = tftree.matrix(CAL.translation, CAL.rotation_xyzw)
    t, q = tftree.to_tq(M)
    assert np.allclose(t, CAL.translation) and np.allclose(np.abs(q), np.abs(CAL.rotation_xyzw),
                                                             atol=1e-5)
    np.testing.assert_allclose(tftree.matrix(t, q), M, atol=1e-9)
    # a child without a parent: added as given
    pub, notes = tftree.attach([CAL], {})
    assert [(p, c) for p, c, _, _ in pub] == [("os_lidar", "zed_left_camera_optical_frame")]
    d = tempfile.mkdtemp()
    try:
        _tree_bag(os.path.join(d, "b"))
        edges = tftree.tf_edges([os.path.join(d, "b")])
        assert edges["os_lidar"].parent == "os_sensor" and edges["os_lidar"].static
        assert edges["zed_camera_link"].parent == "odom_zed" and not edges["zed_camera_link"].static
        # the optical frame has a parent: the Ouster tree is hung below it instead
        pub, notes = tftree.attach([CAL], edges)
        assert [(p, c) for p, c, _, _ in pub] == [("zed_left_camera_optical_frame", "os_sensor")]
        assert "already has a parent" in notes[0]
        T_opt_sensor = tftree.matrix(pub[0][2], pub[0][3])
        T_sensor_lidar = edges["os_lidar"].T
        T_lidar_opt = np.linalg.inv(T_opt_sensor @ T_sensor_lidar)   # a TF lookup, by hand
        np.testing.assert_allclose(T_lidar_opt, M, atol=1e-9)        # = the calibration
        edges2 = dict(edges)
        edges2["os_sensor"] = tftree.Edge("zed_left_camera_optical_frame", "os_sensor", True,
                                          T_opt_sensor)
        try:
            tftree.attach([CAL], edges2)                           # already connected
        except bag.BagError:
            pass
        else:
            raise AssertionError("a calibration inside one tree must be refused (loop)")
    finally:
        shutil.rmtree(d)


def test_merge_adds_static_tf():
    d = tempfile.mkdtemp()
    try:
        write_bag(os.path.join(d, "a"), ["/x"], t0=T0 + 5)
        _tree_bag(os.path.join(d, "b"))
        pub, _ = tftree.attach([CAL], tftree.tf_edges([os.path.join(d, "b")]))
        out = os.path.join(d, "m")
        merge.merge_bags([os.path.join(d, "a"), os.path.join(d, "b")], out, static_tf=pub,
                         log=quiet)
        r = bag.open_bag(out)
        msgs = [(lt, dec(pl)) for _, lt, pl, dec in r.iter_raw(["/tf_static"])]
        assert len(msgs) == 1 and msgs[0][0] == T0            # ONE message, at the start
        pairs = {(tr.header.frame_id, tr.child_frame_id) for tr in msgs[0][1].transforms}
        assert pairs == {("os_sensor", "os_lidar"),           # the inputs' static transforms
                         ("zed_camera_link", "zed_left_camera_frame"),
                         ("zed_left_camera_frame", "zed_left_camera_optical_frame"),
                         ("zed_left_camera_optical_frame", "os_sensor")}, pairs   # + ours
        # two static messages in the inputs (Ouster's and the ZED's) -> still one
        with BagWriter(os.path.join(d, "second")) as w:
            w.write("/tf_static", rosmsg.tf_message([("os_sensor", "os_imu", [0, 0, 0.03],
                                                      [0, 0, 0, 1])], T0 + 2), T0 + 2)
        out3 = os.path.join(d, "m3b")
        merge.merge_bags([os.path.join(d, "b"), os.path.join(d, "second")], out3, log=quiet)
        m3 = [dec(pl) for _, _, pl, dec in bag.open_bag(out3).iter_raw(["/tf_static"])]
        assert len(m3) == 1 and len(m3[0].transforms) == 4, m3
        edges = tftree.tf_edges([out])                          # the merged tree
        assert edges["os_sensor"].parent == "zed_left_camera_optical_frame"
        T = np.linalg.inv(edges["os_sensor"].T @ edges["os_lidar"].T)
        np.testing.assert_allclose(T, tftree.matrix(CAL.translation, CAL.rotation_xyzw),
                                   atol=1e-6)                   # float64 through CDR
        write_bag(os.path.join(d, "s"), ["/x"])                  # /tf_static of another type
        with BagWriter(os.path.join(d, "odd")) as w:
            w.write("/tf_static", rosmsg.string("not tf"), T0)
        try:
            merge.merge_bags([os.path.join(d, "odd")], os.path.join(d, "m3"), static_tf=pub,
                             log=quiet)
        except bag.BagError as e:
            assert "TFMessage" in str(e), e
        else:
            raise AssertionError("a /tf_static that is not TFMessage must be refused")
        out2 = os.path.join(d, "m2")                            # no /tf_static in the inputs
        merge.merge_bags([os.path.join(d, "a")], out2, static_tf=pub, log=quiet)
        assert "durability: 1" in bag.topic_qos(bag.open_bag(out2))["/tf_static"]
    finally:
        shutil.rmtree(d)


def test_select_topics():
    av = ["/a/x", "/a/y", "/b/x", "/tf"]
    assert merge.select_topics(av) == sorted(av)
    assert merge.select_topics(av, ["/a/*", "/tf"]) == ["/a/x", "/a/y", "/tf"]
    assert merge.select_topics(av, ["/a/*"], ["*/y"]) == ["/a/x"]
    for inc, exc in ((["/nope"], []), ([], ["*"])):
        try:
            merge.select_topics(av, inc, exc)
        except bag.BagError:
            continue
        raise AssertionError((inc, exc))


def test_merge_bags_cli():
    d = tempfile.mkdtemp()
    try:
        a, b = os.path.join(d, "a"), os.path.join(d, "b")
        write_bag(a, ["/cam/left", "/cam/right", "/tf_static"], t0=T0)
        write_bag(b, ["/lidar/points", "/radar/points", "/tf"], t0=T0 + 3)
        cli = [sys.executable, os.path.join(ROOT, "processing", "merge_bags.py"), a, b]
        r = subprocess.run(cli + ["--list"], capture_output=True, text=True)
        assert r.returncode == 0 and "/radar/points" in r.stdout and "  6  " in r.stdout, r.stdout
        out = os.path.join(d, "ab")
        r = subprocess.run(cli + ["-o", out, "--topics", "/cam/*", "/lidar/points", "/tf_static",
                                  "--exclude", "*/right"], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        tp = bag.open_bag(out).topics()
        assert sorted(tp) == ["/cam/left", "/lidar/points", "/tf_static"], sorted(tp)
        assert tp["/cam/left"].count == 20 and tp["/tf_static"].count == 1
        logs = [m.log_time for _, _, _, m in bag.iter_mcap_records(bag.open_bag(out))]
        assert logs == sorted(logs)
        out2 = os.path.join(d, "picked")                           # --pick: numbers + a pattern
        r = subprocess.run(cli + ["-o", out2, "--pick"], input="1-2, /radar/*\n",
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        assert sorted(bag.open_bag(out2).topics()) == ["/cam/left", "/cam/right", "/radar/points"]
        # a topic in both bags: refused, unless --source picks the bag
        write_bag(os.path.join(d, "c2"), ["/cam/left", "/cam/right2"], t0=T0 + 7)
        cli2 = [sys.executable, os.path.join(ROOT, "processing", "merge_bags.py"), a,
                os.path.join(d, "c2")]
        r = subprocess.run(cli2 + ["-o", os.path.join(d, "dup"), "--topics", "/cam/*"],
                           capture_output=True, text=True)
        assert r.returncode != 0 and "/cam/left: a, c2" in r.stderr, r.stderr
        out3 = os.path.join(d, "src")
        r = subprocess.run(cli2 + ["-o", out3, "--topics", "/cam/*", "--source", "/cam/left=2"],
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        tp3 = bag.open_bag(out3).topics()
        assert tp3["/cam/left"].count == 20 and sorted(tp3) == ["/cam/left", "/cam/right",
                                                                "/cam/right2"]
        left = [m.log_time for _, _, _, m in bag.iter_mcap_records(bag.open_bag(out3),
                                                                   ["/cam/left"])]
        assert left[0] == T0 + 7                                  # from bag 2, not bag 1
        out4 = os.path.join(d, "all_cam_from_2")              # /cam/right is only in bag 1
        r = subprocess.run(cli2 + ["-o", out4, "--topics", "/cam/*", "--source", "/cam/*=2"],
                           capture_output=True, text=True)
        assert r.returncode == 0 and "/cam/right is not in c2" in r.stdout, (r.stdout, r.stderr)
        assert sorted(bag.open_bag(out4).topics()) == ["/cam/left", "/cam/right", "/cam/right2"]
        r = subprocess.run(cli2 + ["-o", os.path.join(d, "x"), "--topics", "/cam/*",
                                   "--source", "/cam/right=2"], capture_output=True, text=True)
        assert r.returncode != 0 and "no chosen topic matching it is in c2" in r.stderr, r.stderr
        r = subprocess.run(cli + ["-o", out2, "--topics", "/x"], capture_output=True, text=True)
        assert r.returncode != 0 and "exists" in r.stderr          # never overwritten
        r = subprocess.run(cli + ["-o", os.path.join(d, "c"), "--topics", "/typo"],
                           capture_output=True, text=True)
        assert r.returncode != 0 and "matches no topic" in r.stderr, r.stderr
    finally:
        shutil.rmtree(d)


def test_tf_edit():
    """tf_edit.py: show the tree, add static transforms, drop the ZED
    odometry and hang the tree on a new trajectory of os_lidar."""
    d = tempfile.mkdtemp()
    try:
        b = os.path.join(d, "b")
        _tree_bag(b)
        tool = [sys.executable, os.path.join(ROOT, "processing", "tf_edit.py")]
        r = subprocess.run(tool + [b, "--show"], capture_output=True, text=True)
        assert r.returncode == 0 and "~~ odom_zed" in r.stdout and "── os_lidar" in r.stdout, \
            r.stdout
        rng = np.random.default_rng(0)
        poses = []
        with open(os.path.join(d, "traj.txt"), "w") as fh:
            for i in range(5):
                t = rng.normal(size=3)
                q = rng.normal(size=4)
                q /= np.linalg.norm(q)
                ts = (T0 + i * S + S // 2) * 1e-9
                fh.write(f"{ts:.9f} {t[0]} {t[1]} {t[2]} {q[0]} {q[1]} {q[2]} {q[3]}\n")
                poses.append(tftree.matrix(t, q))
        out = os.path.join(d, "m")
        add = ["--add", CAL.parent, CAL.child] + [str(v) for v in CAL.translation] + \
              [str(v) for v in CAL.rotation_xyzw]
        cli = tool + [b, "-o", out] + add + ["--drop", "map_zed->odom_zed", "odom_zed->*",
                                             "--odom-tum", os.path.join(d, "traj.txt"),
                                             "--odom-frame", "os_lidar", "--odom-parent", "map"]
        r = subprocess.run(cli, capture_output=True, text=True)
        assert r.returncode == 0, (r.stdout, r.stderr)
        assert "~~ zed_camera_link" in r.stdout and "odom_zed" not in r.stdout.split(
            "TF tree of the new bag:")[1], r.stdout
        edges = tftree.tf_edges([out], seconds=1e6)
        assert "odom_zed" not in edges and edges["zed_camera_link"].parent == "map"
        T_link_lidar = tftree._static_pose("zed_camera_link", "os_lidar", edges)
        T_lidar_opt = np.linalg.inv(tftree._static_pose("zed_camera_link", "os_lidar", edges)) \
            @ tftree._static_pose("zed_camera_link", "zed_left_camera_optical_frame", edges)
        np.testing.assert_allclose(T_lidar_opt, tftree.matrix(CAL.translation, CAL.rotation_xyzw),
                                   atol=1e-6)                     # the calibration, exactly
        ts = rosmsg.typestore()
        got = []
        for _, _, payload, _ in bag.open_bag(out).iter_raw(["/tf"]):
            for tr in ts.deserialize_cdr(bytes(payload), "tf2_msgs/msg/TFMessage").transforms:
                assert tr.header.frame_id == "map", tr.header.frame_id   # ZED odom gone
                v, q = tr.transform.translation, tr.transform.rotation
                got.append(tftree.matrix([v.x, v.y, v.z], [q.x, q.y, q.z, q.w]))
        assert len(got) == 5
        for T_map_link, want in zip(got, poses):
            np.testing.assert_allclose(T_map_link @ T_link_lidar, want, atol=1e-6)
        assert bag.open_bag(out).topics()["/tf_static"].count == 1
        # the odometry while the old one is still attached: refused
        r = subprocess.run(tool + [b, "-o", os.path.join(d, "m2")] + add +
                           ["--odom-tum", os.path.join(d, "traj.txt"), "--odom-frame",
                            "os_lidar", "--odom-parent", "map"], capture_output=True, text=True)
        assert r.returncode != 0 and "not static" in r.stderr, r.stderr
        r = subprocess.run(tool + [b, "-o", os.path.join(d, "m3"), "--drop", "nope->x"],
                           capture_output=True, text=True)
        assert r.returncode != 0 and "matches no TF edge" in r.stderr
        r = subprocess.run(tool + [b, "-o", os.path.join(d, "m4")], capture_output=True, text=True)
        assert r.returncode != 0 and "nothing to change" in r.stderr
    finally:
        shutil.rmtree(d)


def _cli(t, *args):
    return subprocess.run([sys.executable, os.path.join(ROOT, "processing", "merge_session.py"),
                           t.raw, "--record", t.plan_path, *args],
                          capture_output=True, text=True, env=dict(os.environ))


def test_merge_session_cli():
    t = Pass()
    try:
        # the zed bags in these settings are the Pass's, not configs/recording.yaml's
        sp = os.path.join(t.root, "settings.yaml")
        yaml.safe_dump(t.settings(), open(sp, "w"))
        r = _cli(t, "--settings", sp, "--check")
        assert r.returncode == 0, r.stderr
        assert any(l.split() == ["MISSING", "/mobile_2/imu"] for l in r.stdout.splitlines()), r.stdout
        assert "no bags for: mobile_1/sniffer" in r.stdout, r.stdout
        assert not os.path.exists(t.work + f"/{P}_merged")
        r = _cli(t, "--settings", sp)
        assert r.returncode != 0 and "--allow-missing" in r.stderr, (r.stdout, r.stderr)
        r = _cli(t, "--settings", sp, "--allow-missing")
        assert r.returncode == 0, (r.stdout, r.stderr)
        final = os.path.join(t.work, f"{P}_merged")
        assert os.path.isfile(os.path.join(final, "metadata.yaml")), os.listdir(t.work)
        assert not os.path.exists(os.path.join(t.work, ".partial"))
        assert "/mobile_1/zed/right/image_rect_color" in bag.open_bag(final).topics()
        r = _cli(t, "--settings", sp, "--allow-missing")
        assert r.returncode != 0 and "exists" in r.stderr         # never overwritten
    finally:
        t.close()


if __name__ == "__main__":
    tests = [(k, v) for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    for name, fn in tests:
        check(name, fn)
    print("\n%d/%d passed" % (len(tests) - len(FAILED), len(tests)))
    sys.exit(1 if FAILED else 0)
