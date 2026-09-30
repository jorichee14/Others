#!/usr/bin/env python
"""Self-tests for scoop.session and scoop.merge (the topic check and the
merge of a pass). No ROS, no real bags.

A synthetic pass: mobile_1 processed (retimed + two ZED bags in data/work),
mobile_2 raw with a topic missing, infra_1 raw and a repeat (_r2).

    python scripts/test_merge.py
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

from scoop import bag, merge, recording, rosmsg, session            # noqa: E402
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
                w.write(t, rosmsg.string(f"{t} {i}"), t0 + i * step + k)


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
    assert "/tf" not in p.topics("mobile_1", "mapping")          # mapping records no tf


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
        assert "/tf" not in m1 and "/mobile_1/wifi/status" not in m1 and \
            "/mobile_1/ntp/events" not in m1 and "/ouster/lidar_packets" in m1
        assert "/mobile_1/wifi/status" in real.topics("mobile_1", "survey")
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
        assert res.counts == want, (res.counts, want)
        r = bag.open_bag(out)
        logs = [m.log_time for _, _, _, m in bag.iter_mcap_records(r)]
        assert logs == sorted(logs) and len(logs) == sum(want.values())
        retimed = next(b for b in inputs if b.name.endswith("_retimed"))
        a = [bytes(m.data) for _, _, _, m in bag.iter_mcap_records(bag.open_bag(retimed), ["/tf"])]
        assert a
        b = [bytes(m.data) for _, _, _, m in bag.iter_mcap_records(r, ["/tf"])]
        assert a == b                                            # byte for byte
        assert bag.topic_qos(r)["/tf_static"] == LATCHED
        with open(os.path.join(out, f"merged_0.mcap"), "rb") as fh:
            from mcap.reader import make_reader
            chunks = make_reader(fh).get_summary().chunk_indexes
            assert chunks and all(c.compression == "zstd" for c in chunks)

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


def _cli(t, *args):
    return subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "merge_session.py"),
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
