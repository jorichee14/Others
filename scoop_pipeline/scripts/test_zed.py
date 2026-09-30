#!/usr/bin/env python
"""Self-tests for scoop.zed and scripts/run_zed.sh. No ZED SDK, no ROS, no GPU.

A fake `ros2` stands in for the recorder and the ZED wrapper: its recorder
writes a bag like `ros2 bag record` would after an SVO replay at twice real
time -- header stamps = capture time, log times = replay time, depth arriving
later than the image -- and its launch says "SVO reached the end." and waits
for Ctrl+C. The real run_zed.sh drives both.

    python scripts/test_zed.py
"""
import os
import shutil
import stat
import sys
import tempfile
import traceback

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from scoop import bag, recording, rosmsg, zed                       # noqa: E402
from scoop.bag import TopicSchema                                   # noqa: E402
from scoop.bagwrite import BagWriter                                # noqa: E402

FAILED = []
MS = 1_000_000
S0 = 1_790_218_481_000_000_000          # capture time of the first frame
W0 = 1_790_900_000_000_000_000          # when the replay happens (days later)
FRAME = 66 * MS                         # 15 Hz
N = 60
IMAGE, INFO, DEPTH, STATUS = ("/zed/zed_node/left/image_rect_color",
                              "/zed/zed_node/left/camera_info",
                              "/zed/zed_node/depth/depth_registered",
                              "/zed/zed_node/pose/status")
LATENCY = {IMAGE: 2 * MS, INFO: 3 * MS, DEPTH: 150 * MS}     # replay -> recorder
LATCHED = "- history: 3\n  depth: 0\n  reliability: 1\n  durability: 1\n"


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


def write_recording(out, topics=(IMAGE, INFO, DEPTH, STATUS)):
    """What `ros2 bag record` leaves after the replay: header stamps are the
    SVO's, log times the replay's (twice real time, per-topic latency)."""
    items = []
    for i in range(N):
        stamp, sent = S0 + i * FRAME, W0 + i * FRAME // 2
        for t in topics:
            if t == STATUS:                                   # no header
                items.append((sent + 5 * MS, t, rosmsg.string(f"ok {i}")))
            else:
                items.append((sent + LATENCY[t], t, rosmsg.imu(stamp, "zed_left", [0, 0, 9.8],
                                                               [0, 0, 0])))
    items.sort(key=lambda x: x[0])
    with BagWriter(out, library="rosbag2") as w:
        for t in topics:
            w.set_qos(t, LATCHED if t == INFO else f"qos of {t}\n")
        for log_ns, t, msg in items:
            w.write(t, msg, log_ns)


def test_has_header():
    sc = lambda text: TopicSchema("x/msg/Y", "ros2msg", text.encode(), "cdr")
    assert zed.has_header(sc("# a comment\nstd_msgs/Header header\nfloat64 x\n"))
    assert zed.has_header(sc("Header header\n"))
    assert zed.has_header(sc("uint8 OK=0\n\nstd_msgs/msg/Header header\n"))
    assert not zed.has_header(sc("string data\n"))
    assert not zed.has_header(sc("uint8 status\n====\nMSG: std_msgs/Header\n"))
    assert not zed.has_header(TopicSchema("x", "ros2idl", b"Header header", "cdr"))


def test_prefix_remap_and_settings():
    got = zed.prefix_remap([IMAGE, "/zed/zed_nodeX/a", "/tf"], "/zed/zed_node/", "/mobile_1/zed")
    assert got == {IMAGE: "/mobile_1/zed/left/image_rect_color"}, got
    args = recording.zed_args(recording.load_settings()["zed"])
    assert args["camera_model"] == "zed2i" and not args["realtime"]
    assert all(v.startswith("/mobile_1/zed/") for v in args["remap"].values())
    assert set(args["remap"]) == set(args["topics"])
    p = zed.wrapper_params({"svo": {"replay_rate": 0.5}, "depth": {"depth_mode": "NEURAL"}})
    ros = p["/**"]["ros__parameters"]
    assert ros["svo"] == {"use_svo_timestamps": True, "svo_loop": False,
                          "svo_realtime": False, "replay_rate": 0.5}, ros
    assert ros["depth"] == {"depth_mode": "NEURAL"}


def test_find_svo():
    d = tempfile.mkdtemp()
    try:
        assert zed.find_svo(d) is None
        open(os.path.join(d, "a.svo2"), "w").close()
        assert zed.find_svo(d).name == "a.svo2"
        open(os.path.join(d, "b.SVO"), "w").close()
        try:
            zed.find_svo(d)
        except bag.BagError:
            pass
        else:
            raise AssertionError("two SVOs must be refused")
    finally:
        shutil.rmtree(d)


def _check_restamped(out, remap):
    r = bag.open_bag(out)
    tp = r.topics()
    for t in (IMAGE, INFO, DEPTH):
        name = remap.get(t, t)
        assert tp[name].count == N, (name, tp[name].count)
        recs = [m for _, _, _, m in bag.iter_mcap_records(r, [name])]
        assert [m.log_time for m in recs] == [S0 + i * FRAME for i in range(N)], name
        assert all(m.publish_time == m.log_time for m in recs)
    status = [m.log_time for _, _, _, m in bag.iter_mcap_records(r, [remap.get(STATUS, STATUS)])]
    assert len(status) == N
    for i, lt in enumerate(status):          # sent 5 ms after frame i, replayed 2x fast
        assert S0 + i * FRAME <= lt < S0 + (i + 1) * FRAME, (i, lt - S0 - i * FRAME)
    logs = [m.log_time for _, _, _, m in bag.iter_mcap_records(r)]
    assert logs == sorted(logs)
    qos = bag.topic_qos(r)
    assert qos[remap.get(INFO, INFO)] == LATCHED
    return tp


def test_restamp_bag():
    d = tempfile.mkdtemp()
    try:
        rec, out = os.path.join(d, "rec"), os.path.join(d, "out")
        write_recording(rec)
        remap = zed.prefix_remap([IMAGE, INFO, DEPTH, STATUS, "/zed/zed_node/odom"],
                                 "/zed/zed_node", "/mobile_1/zed")    # odom: never recorded
        res = zed.restamp_bag(rec, out, remap, log=quiet)
        tp = _check_restamped(out, remap)
        assert not any(t.startswith("/zed/") for t in tp), list(tp)
        assert res.moved == {"/mobile_1/zed/pose/status": N}, res.moved
        a = [bytes(m.data) for _, _, _, m in bag.iter_mcap_records(bag.open_bag(rec), [DEPTH])]
        b = [bytes(m.data) for _, _, _, m in
             bag.iter_mcap_records(bag.open_bag(out), [remap[DEPTH]])]
        assert a == b, "payloads must be copied untouched"
        try:
            zed.restamp_bag(rec, os.path.join(d, "tight"), remap, reorder_s=0.05, log=quiet)
        except bag.BagError:
            pass
        else:
            raise AssertionError("depth arrives 150 ms late: a 50 ms window must fail")
    finally:
        shutil.rmtree(d)


FAKE_ROS2 = r"""#!/usr/bin/env bash
case "$1 $2" in
    "pkg prefix") [ "$3" = zed_wrapper ] && { echo /opt/fake; exit 0; }; exit 1 ;;
    "bag record") shift 2; exec python3 "$FAKE_DIR/recorder.py" "$@" ;;
    "launch zed_wrapper") shift 3; exec python3 "$FAKE_DIR/launch.py" "$@" ;;
esac
exit 1
"""

FAKE_RECORDER = r"""
import os, sys, time
sys.path.insert(0, os.environ["FAKE_ROOT"]); sys.path.insert(0, os.environ["FAKE_DIR"])
args = sys.argv[1:]
out = args[args.index("-o") + 1]
topics = [a for a in args if a.startswith("/") and a != out]
print("[INFO] [rosbag2_recorder]: Listening for topics...", flush=True)
try:
    while True:
        time.sleep(0.1)
except KeyboardInterrupt:                    # Ctrl+C, as ros2 bag record gets it
    import test_zed
    test_zed.write_recording(out, [t for t in topics if t != "/zed/zed_node/odom"])
    open(os.path.join(os.environ["FAKE_DIR"], "recorder_topics.txt"), "w").write("\n".join(topics))
"""

FAKE_LAUNCH = r"""
import os, sys, time
d = os.environ["FAKE_DIR"]
open(os.path.join(d, "launch_args.txt"), "w").write("\n".join(sys.argv[1:]))
params = [a.split(":=", 1)[1] for a in sys.argv[1:] if a.startswith("ros_params_override_path:=")]
open(os.path.join(d, "params_seen.yaml"), "w").write(open(params[0]).read())
print("[zed_node]: opening the SVO", flush=True)
time.sleep(0.5)
if os.environ.get("FAKE_ZED_FAIL"):
    print("[zed_node]: CUDA error", flush=True); sys.exit(1)
print("[WARN] [zed_node]: SVO reached the end.", flush=True)
try:
    while True:
        time.sleep(0.1)
except KeyboardInterrupt:
    print("[zed_node]: stopped", flush=True)
"""


class FakeRos:
    """A PATH with the fake ros2 on it."""

    def __init__(self):
        self.dir = tempfile.mkdtemp()
        for name, text in (("ros2", FAKE_ROS2), ("recorder.py", FAKE_RECORDER),
                           ("launch.py", FAKE_LAUNCH)):
            p = os.path.join(self.dir, name)
            open(p, "w").write(text)
            os.chmod(p, os.stat(p).st_mode | stat.S_IEXEC)
        shutil.copy(os.path.abspath(__file__), os.path.join(self.dir, "test_zed.py"))
        self._env = dict(os.environ)
        os.environ.update(PATH=self.dir + os.pathsep + os.environ["PATH"], FAKE_DIR=self.dir,
                          FAKE_ROOT=ROOT, ZED_DRAIN_S="0")

    def read(self, name):
        return open(os.path.join(self.dir, name)).read()

    def close(self):
        os.environ.clear()
        os.environ.update(self._env)
        shutil.rmtree(self.dir, ignore_errors=True)


def _settings():
    return recording.zed_args({"camera_model": "zed2i",
                               "rename": ["/zed/zed_node", "/mobile_1/zed"],
                               "topics": [IMAGE, INFO, DEPTH, STATUS, "/zed/zed_node/odom"],
                               "params": {"depth": {"depth_mode": "NEURAL"}}})


def test_svo_to_bag_through_run_zed():
    fake, d = FakeRos(), tempfile.mkdtemp()
    try:
        svo = os.path.join(d, "run1_mobile_1.svo2")
        open(svo, "w").write("svo")
        out = os.path.join(d, "run1_mobile_1_zed")
        logs = []
        zed.svo_to_bag(svo, out, log=logs.append, **_settings())
        args = fake.read("launch_args.txt").splitlines()
        assert "camera_model:=zed2i" in args and f"svo_path:={svo}" in args, args
        seen = yaml.safe_load(fake.read("params_seen.yaml"))["/**"]["ros__parameters"]
        assert seen["svo"]["use_svo_timestamps"] and not seen["svo"]["svo_loop"], seen
        assert seen["depth"]["depth_mode"] == "NEURAL"
        assert fake.read("recorder_topics.txt").splitlines() == _settings()["topics"]
        _check_restamped(out, _settings()["remap"])
        assert sorted(os.listdir(os.path.join(out, "zed_logs"))) == \
            ["params.yaml", "record.log", "zed.log"], os.listdir(os.path.join(out, "zed_logs"))
        assert "SVO reached the end" in open(os.path.join(out, "zed_logs", "zed.log")).read()
        assert sorted(os.listdir(d)) == ["run1_mobile_1.svo2", "run1_mobile_1_zed"], \
            "the intermediate recording and params file must be gone"
        assert any("never published" in l and "odom" in l for l in logs), logs

        os.environ["FAKE_ZED_FAIL"] = "1"
        try:
            zed.svo_to_bag(svo, os.path.join(d, "fails"), log=quiet, **_settings())
        except bag.BagError:
            pass
        else:
            raise AssertionError("a wrapper that stops early must fail the step")
        assert not os.path.exists(os.path.join(d, "fails"))
    finally:
        fake.close()
        shutil.rmtree(d)


FAKE_DOCKER = r"""#!/usr/bin/env bash
[ "$1" = ps ] && { echo other; echo isaac_ros_dev-x86_64-container; exit 0; }
[ "$1" = exec ] || exit 1
shift; envs=()
while [ "$1" = -i ] || [ "$1" = -e ]; do
    if [ "$1" = -e ]; then envs+=("$2"); shift 2; else shift; fi
done
echo "$1" > "$FAKE_DIR/container.txt"; shift
echo "$@" > "$FAKE_DIR/docker_args.txt"
exec env "${envs[@]}" PATH="$FAKE_DIR:/usr/bin:/bin" "$@"
"""


def test_run_zed_in_container():
    """No ros2 here: run_zed.sh re-runs itself in the container, with the
    paths translated to the container's view of the workspace."""
    fake, d = FakeRos(), tempfile.mkdtemp()
    try:
        host_bin = os.path.join(d, "host_bin")                 # docker, but no ros2
        os.makedirs(host_bin)
        p = os.path.join(host_bin, "docker")
        open(p, "w").write(FAKE_DOCKER)
        os.chmod(p, 0o755)
        ws = os.path.join(d, "isaac_ros-dev")
        os.makedirs(os.path.join(ws, "data", "raw"))
        os.symlink(ws, os.path.join(d, "container_ws"))        # the container's mount
        svo = os.path.join(ws, "data", "raw", "run1.svo2")
        open(svo, "w").write("svo")
        os.environ.update(PATH=host_bin + os.pathsep + "/usr/bin:/bin", ISAAC_ROS_WS=ws,
                          ZED_CONTAINER_WS=os.path.join(d, "container_ws"))
        out = os.path.join(ws, "data", "work", "run1_zed")
        zed.svo_to_bag(svo, out, log=quiet, **_settings())
        assert fake.read("container.txt").strip() == "isaac_ros_dev-x86_64-container"
        args = fake.read("docker_args.txt").split()
        assert args[:3] == ["bash", "-s", "--"], args
        assert args[3] == os.path.join(d, "container_ws", "data", "raw", "run1.svo2"), args
        assert f"svo_path:={args[3]}" in fake.read("launch_args.txt").splitlines()
        _check_restamped(out, _settings()["remap"])

        outside = os.path.join(d, "elsewhere.svo2")
        open(outside, "w").write("svo")
        try:
            zed.svo_to_bag(outside, os.path.join(ws, "data", "work", "x_zed"), log=quiet,
                           **_settings())
        except bag.BagError:
            pass
        else:
            raise AssertionError("an SVO the container cannot see must be refused")
    finally:
        fake.close()
        shutil.rmtree(d)


def test_process_recording_zed_step():
    fake, root = FakeRos(), tempfile.mkdtemp()
    try:
        raw = os.path.join(root, "data", "raw", "20260924", "survey_1")
        work = os.path.join(root, "data", "work", "20260924", "survey_1")
        os.makedirs(raw)
        open(os.path.join(raw, "run1_mobile_1.svo2"), "w").write("svo")
        rec = recording.Recording(recording.Path(raw), recording.Path(raw, "the_bag"),
                                  recording.Path(work), zed.find_svo(raw))
        for s in ("decoded", "retimed"):                 # the Ouster steps: already done
            os.makedirs(rec.step(s))
            open(os.path.join(rec.step(s), "metadata.yaml"), "w").write("x")
        os.makedirs(rec.step("glim"))
        open(os.path.join(rec.step("glim"), "traj_lidar.txt"), "w").write("x")
        settings = recording.load_settings()
        settings["zed"]["topics"] = [IMAGE, INFO, DEPTH, STATUS]
        logs = []
        st = recording.process(rec, settings, log=logs.append)
        assert st == {"decoded": True, "retimed": True, "glim": True, "zed": True}, st
        assert rec.step("zed").name == "run1_mobile_1_zed"
        tp = bag.open_bag(rec.step("zed")).topics()
        assert "/mobile_1/zed/left/image_rect_color" in tp, list(tp)
        assert not os.path.exists(os.path.join(work, ".partial"))
        mt = os.path.getmtime(os.path.join(rec.step("retimed"), "metadata.yaml"))
        logs = []
        recording.process(rec, settings, redo="zed", log=logs.append)
        ran = [l.split()[2] for l in logs if l.startswith("[run ]")]
        assert ran == ["zed"], logs                       # zed alone, nothing upstream
        assert os.path.getmtime(os.path.join(rec.step("retimed"), "metadata.yaml")) == mt
    finally:
        fake.close()
        shutil.rmtree(root)


if __name__ == "__main__":
    tests = [(k, v) for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    for name, fn in tests:
        check(name, fn)
    print("\n%d/%d passed" % (len(tests) - len(FAILED), len(tests)))
    sys.exit(1 if FAILED else 0)
