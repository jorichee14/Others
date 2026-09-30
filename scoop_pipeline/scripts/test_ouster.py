#!/usr/bin/env python
"""Self-tests for scoop.ouster. Needs ouster-sdk and rosbags; no real bag.

Builds a synthetic packets bag the way ouster_ros records one: sensor
metadata on <ns>/metadata, lidar packets (made by the SDK's own encoder) on
<ns>/lidar_packets, each logged at a host time = known offset + drift +
positive receive jitter. Then checks the clock fit recovers that mapping and
that every decoded point sits where the geometry says, with the right
per-point time, in the destaggered row-major order.

    python scripts/test_ouster.py
"""
import os
import shutil
import struct
import sys
import tempfile
import traceback
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.simplefilter("ignore", FutureWarning)

from rosbags.typesys import Stores, get_typestore, get_types_from_msg  # noqa: E402

from scoop import bag, ouster, replay, retime, rosmsg                  # noqa: E402

TS = get_typestore(Stores.ROS2_HUMBLE)
TS.register(get_types_from_msg("uint8[] buf", "ouster_sensor_msgs/msg/PacketMsg"))
S = ouster.sdk()
NS = "/ouster"
OFFSET = 1_700_000_000.25        # host - sensor at s0, seconds
DRIFT = 20e-6                     # 20 ppm
FAILED = []
IMU_TRUTH = {}
ZED_COUNT = {}


def check(name, fn):
    try:
        fn()
        print("  ok    %s" % name)
    except Exception:
        FAILED.append(name)
        print("  FAIL  %s" % name)
        traceback.print_exc()


def make_bag(path, n_frames=100, dead_cols=None, seed=0, s0=1000 * 10**9):
    """Returns (info, frames): the truth the decoder is checked against."""
    from mcap.writer import CompressionType, Writer
    rng = np.random.default_rng(seed)
    info = default_info()
    pf = ouster._packet_format(info)
    W = info.w
    frames = []
    packets = []                                           # (host ns, bytes)
    for k in range(n_frames):
        f = ouster._new_frame(info)
        r = f.field(S.ChanField.RANGE)
        r[:] = rng.integers(300, 40000, r.shape)
        r[rng.random(r.shape) < 0.05] = 0                  # no return
        ts = s0 + k * 100_000_000 + np.arange(W, dtype=np.uint64) * 97_656
        dead = (dead_cols or {}).get(k, 0)
        if dead:
            ts[:dead] = 0
            r[:, :dead] = 0
        f.timestamp[:] = ts
        f.status[:] = (ts > 0).astype(np.uint32)
        f.measurement_id[:] = np.arange(W)
        f.frame_id = k
        frames.append({"range": r.copy(), "ts": ts.copy()})
        for p in S.mod.frame_to_packets(f, pf, info.init_id, int(info.sn)):
            if not isinstance(p, S.LidarPacket):
                continue
            buf = bytes(p.buf)
            t_s = int(pf.packet_header(S.ColHeader.TIMESTAMP, np.frombuffer(buf, np.uint8))[0])
            if t_s == 0:
                t_s = int(ts[np.flatnonzero(ts)[0]])
            sec = t_s * 1e-9
            host = sec + OFFSET + DRIFT * (sec - s0 * 1e-9)
            jitter = rng.exponential(30e-6) + (0.02 if rng.random() < 0.02 else 0.0)
            # a receive queue delays packets but never reorders them
            h_ns = int(round((host + jitter) * 1e9))
            if packets:
                h_ns = max(h_ns, packets[-1][0] + 1)
            packets.append((h_ns, buf))

    imu = []                                               # (host ns, bytes, truth)
    for i in range(n_frames * 10):                         # 100 Hz
        g_ns = s0 + i * 10_000_000 + 3_000
        la = (0.01 * (i % 7), -0.02, 1.0)                  # g
        av = (1.5, -2.0, 0.25 * (i % 5))                   # deg/s
        b = struct.pack("<QQQffffff", g_ns - 5_000, g_ns - 1_000, g_ns, *la, *av)
        sec = g_ns * 1e-9
        h = int(round((sec + OFFSET + DRIFT * (sec - s0 * 1e-9) + 50e-6) * 1e9))
        imu.append((h, b, (g_ns, la, av)))
    IMU_TRUTH[path] = [x[2] for x in imu]
    ZED_COUNT[path] = n_frames * 5

    zed = []                                               # a topic that is only copied
    Imu = TS.types["sensor_msgs/msg/Imu"]
    for i in range(n_frames * 5):                          # 50 Hz
        h = int(round((s0 * 1e-9 + OFFSET + i * 0.02 + 0.001) * 1e9))
        m = Imu(header=TS.types["std_msgs/msg/Header"](
                    stamp=TS.types["builtin_interfaces/msg/Time"](sec=h // 10**9, nanosec=h % 10**9),
                    frame_id="zed_imu_link"),
                orientation=TS.types["geometry_msgs/msg/Quaternion"](x=0.0, y=0.0, z=0.0, w=1.0),
                orientation_covariance=np.zeros(9),
                angular_velocity=TS.types["geometry_msgs/msg/Vector3"](x=0.0, y=0.0, z=float(i)),
                angular_velocity_covariance=np.zeros(9),
                linear_acceleration=TS.types["geometry_msgs/msg/Vector3"](x=0.0, y=0.0, z=9.8),
                linear_acceleration_covariance=np.zeros(9))
        zed.append((h, bytes(TS.serialize_cdr(m, "sensor_msgs/msg/Imu"))))

    PM = TS.types["ouster_sensor_msgs/msg/PacketMsg"]
    Str = TS.types["std_msgs/msg/String"]
    with open(path, "wb") as fh:
        w = Writer(fh, compression=CompressionType.NONE)
        w.start("ros2", "test")
        sm = w.register_schema("std_msgs/msg/String", "ros2msg", b"string data")
        sp = w.register_schema("ouster_sensor_msgs/msg/PacketMsg", "ros2msg", b"uint8[] buf")
        cm = w.register_channel(f"{NS}/metadata", "cdr", sm)
        cp = w.register_channel(f"{NS}/lidar_packets", "cdr", sp)
        ci = w.register_channel(f"{NS}/imu_packets", "cdr", sp)
        text, _ = TS.generate_msgdef("sensor_msgs/msg/Imu", ros_version=2)
        cz = w.register_channel("/zed/zed_node/imu/data", "cdr",
                                w.register_schema("sensor_msgs/msg/Imu", "ros2msg", text.encode()))
        meta = bytes(TS.serialize_cdr(Str(data=info.to_json_string()), "std_msgs/msg/String"))
        t0 = min(h for h, _ in packets) - 10**6
        w.add_message(cm, log_time=t0, publish_time=t0, data=meta)
        def pkt(b):
            return bytes(TS.serialize_cdr(PM(buf=np.frombuffer(b, np.uint8).copy()),
                                          "ouster_sensor_msgs/msg/PacketMsg"))
        allm = ([(h, cp, pkt(b)) for h, b in packets] + [(h, ci, pkt(b)) for h, b, _ in imu]
                + [(h, cz, d) for h, d in zed])
        for i, (h, ch, data) in enumerate(sorted(allm, key=lambda x: x[0])):
            w.add_message(ch, log_time=h, publish_time=h, data=data, sequence=i)
        w.finish()
    return info, frames, s0


def default_info():
    """The SDK's default OS-1-64. SDK 1.0 cannot parse its own default JSON
    back (the packet profile comes out OFF), so tests hand the decoder this
    object; real sensor metadata is read from the bag."""
    info = S.SensorInfo.from_default(S.mod.LidarMode._1024x10)
    info.format.udp_profile_lidar = S.mod.UDPProfileLidar.LEGACY
    return info


INFO = None


def meta():
    global INFO
    INFO = INFO or default_info()
    return INFO


def truth_host_ns(sensor_ns, s0):
    s = sensor_ns * 1e-9
    return (s + OFFSET + DRIFT * (s - s0 * 1e-9)) * 1e9


class Fixture:
    _cache = {}

    @classmethod
    def get(cls, dead=None):
        key = str(dead)
        if key not in cls._cache:
            d = tempfile.mkdtemp()
            p = os.path.join(d, "packets.mcap")
            info, frames, s0 = make_bag(p, dead_cols=dead)
            cls._cache[key] = (d, p, info, frames, s0)
        return cls._cache[key]

    @classmethod
    def cleanup(cls):
        for d, *_ in cls._cache.values():
            shutil.rmtree(d, ignore_errors=True)


def quiet(*_):
    pass


# --------------------------------------------------------------------------- #
def test_metadata_read_and_off_profile_refused():
    _, p, info, _, _ = Fixture.get()
    js = ouster.read_metadata(bag.open_bag(p), NS)
    assert S.SensorInfo(js).w == info.w
    try:
        ouster.sensor_info(js)            # SDK 1.0 parses this as profile OFF
    except bag.BagError as e:
        assert "OFF" in str(e)
    else:
        assert str(S.SensorInfo(js).format.udp_profile_lidar).endswith("LEGACY")


def test_clock_fit_recovers_offset_and_drift():
    _, p, _, frames, s0 = Fixture.get()
    cm = ouster.fit_clock(bag.open_bag(p), NS, metadata=meta(), log=quiet)
    assert abs(cm.drift_ppm - DRIFT * 1e6) < 2.0, cm
    for f in (frames[0], frames[len(frames) // 2], frames[-1]):
        t = int(f["ts"][0])
        err_us = abs(cm.host_ns(t) - truth_host_ns(t, s0)) * 1e-3
        assert err_us < 100, (err_us, cm)                  # jitter scale is 30 us


def test_scans_geometry_time_and_order():
    _, p, info, frames, s0 = Fixture.get()
    reader = bag.open_bag(p)
    cm = ouster.fit_clock(reader, NS, metadata=meta(), log=quiet)
    st = {}
    scans = list(ouster.iter_ouster_scans(reader, NS, clock=cm, stats=st, metadata=meta()))
    assert len(scans) == len(frames) - 1, (len(scans), st)   # last frame never closes
    shift = np.asarray(info.format.pixel_shift_by_row)
    H, W = info.h, info.w
    li = S.SensorInfo(info.to_json_string())
    li.format.udp_profile_lidar = info.format.udp_profile_lidar
    li.lidar_to_sensor_transform = np.eye(4)
    lut = S.mod.XYZLut(li)                                   # double precision
    for s in scans:
        f = frames[s.key]
        assert s.stamp_ns == cm.host_ns(int(f["ts"][0]))
        # destaggered (u, v) comes from staggered column (v - shift[u]) % W
        u, v = np.mgrid[0:H, 0:W]
        src = (v - shift[:, None]) % W
        r_d = f["range"][u, src].reshape(-1)
        keep = r_d > 0
        exp_t = (f["ts"][src].astype(np.int64) - int(f["ts"][0])).reshape(-1)[keep] * 1e-9
        xyz_s = lut(f["range"])                              # staggered
        exp_xyz = xyz_s[u, src].reshape(-1, 3)[keep]
        assert s.xyz.shape == exp_xyz.shape, (s.xyz.shape, exp_xyz.shape)
        np.testing.assert_allclose(s.xyz, exp_xyz, atol=2e-4)
        np.testing.assert_allclose(s.t, exp_t, atol=1e-12)
        # geometry: lidar-frame points are range from a beam origin ~16 mm out
        dist = np.linalg.norm(exp_xyz, axis=1)
        assert np.all(np.abs(dist - r_d[keep] * 1e-3) < 0.02)


def test_first_dead_columns_move_the_stamp():
    _, p, info, frames, _ = Fixture.get(dead={5: 40})
    reader = bag.open_bag(p)
    scans = {s.key: s for s in ouster.iter_ouster_scans(reader, NS, metadata=meta())}   # sensor time
    f = frames[5]
    first = int(f["ts"][40])
    assert scans[5].stamp_ns == first, (scans[5].stamp_ns, first)
    assert scans[5].t.min() >= 0.0


def test_driver_min_range_and_frames():
    _, p, info, frames, _ = Fixture.get()
    reader = bag.open_bag(p)
    a = next(ouster.iter_ouster_scans(reader, NS, frame="lidar", metadata=meta()))
    b = next(ouster.iter_ouster_scans(reader, NS, frame="sensor", metadata=meta()))
    c = next(ouster.iter_ouster_scans(reader, NS, driver_min_range=1.30, metadata=meta()))
    r = frames[0]["range"]
    assert len(c.xyz) == int(((r >= 1300)).sum()), (len(c.xyz), int((r >= 1300).sum()))
    # os_sensor = Rz(180) * os_lidar + [0, 0, 36.18 mm] for the default metadata
    T = np.asarray(info.lidar_to_sensor_transform)
    exp = a.xyz @ T[:3, :3].T + T[:3, 3] * 1e-3
    np.testing.assert_allclose(b.xyz, exp, atol=2e-4)


def test_select_before_projection():
    _, p, _, frames, _ = Fixture.get()
    st = {}
    got = [s.key for s in ouster.iter_ouster_scans(
        bag.open_bag(p), NS, select=lambda ns: None if (ns // 100_000_000) % 2 else "even",
        stats=st, metadata=meta())]
    assert got == ["even"] * st["selected"] and 0 < st["selected"] < st["frames"], st


def test_clock_save_load():
    d = tempfile.mkdtemp()
    try:
        cm = ouster.ClockMap(1.5, 2e-5, 1000.0, 7, 0.01)
        assert ouster.ClockMap.load(cm.save(os.path.join(d, "c.json"))) == cm
        assert ouster.ClockMap.identity().host_ns(123456789) == 123456789
    finally:
        shutil.rmtree(d, ignore_errors=True)


def _decoded(tag="dec", remap=None):
    d, p, info, frames, s0 = Fixture.get()
    out = os.path.join(d, tag)
    if not os.path.exists(out):
        replay.decode_ouster_bag(p, out, NS, "/mobile_1/ouster", driver_min_range=1.30,
                                 metadata=meta(), remap=remap, log=quiet)
    return out, p, frames, s0


def test_decoded_bag_points_match_decoder():
    out, p, frames, _ = _decoded()
    r = bag.open_bag(out)                                  # via metadata.yaml
    tp = r.topics()
    assert tp["/mobile_1/ouster/points"].count == len(frames) - 1, tp
    got = list(bag.iter_scans(r, "/mobile_1/ouster/points", with_time=True, log=quiet))
    ref = list(ouster.iter_ouster_scans(bag.open_bag(p), NS, metadata=meta(),
                                        driver_min_range=1.30))
    assert len(got) == len(ref)
    for g, e in zip(got, ref):
        assert g.stamp_ns == e.stamp_ns                    # sensor time, first valid column
        np.testing.assert_array_equal(g.xyz, e.xyz)
        np.testing.assert_allclose(g.t, e.t, atol=1e-9)


def test_decoded_bag_layout_and_imu():
    out, p, frames, _ = _decoded()
    r = bag.open_bag(out)
    H, W = meta().h, meta().w
    msg = None
    for topic, _, payload, decode in r.iter_raw(["/mobile_1/ouster/points"]):
        msg = decode(payload)                               # generic ROS 2 decoder
        break
    assert (msg.height, msg.width, msg.point_step) == (H, W, 48)
    assert [f.name for f in msg.fields] == ["x", "y", "z", "intensity", "t",
                                            "reflectivity", "ring", "ambient", "range"]
    a = np.frombuffer(bytes(msg.data), ouster.ROS_POINT_DTYPE)
    assert np.array_equal(a["ring"].reshape(H, W)[:, 0], np.arange(H))
    nan = ~np.isfinite(a["x"])
    assert np.all((a["range"][~nan] >= 1300)), "a kept point is below min_range"
    assert msg.header.frame_id == "os_lidar"

    truth = IMU_TRUTH[p]
    imus = [decode(pl) for _, _, pl, decode in r.iter_raw(["/mobile_1/ouster/imu"])]
    assert len(imus) == len(truth), (len(imus), len(truth))
    for m, (g_ns, la, av) in zip(imus, truth):
        assert m.header.stamp.sec * 10**9 + m.header.stamp.nanosec == g_ns
        np.testing.assert_allclose([m.linear_acceleration.x, m.linear_acceleration.y,
                                    m.linear_acceleration.z],
                                   np.array(la, np.float32) * 9.80665, rtol=1e-6)
        np.testing.assert_allclose([m.angular_velocity.x, m.angular_velocity.y,
                                    m.angular_velocity.z],
                                   np.radians(np.array(av, np.float32)), rtol=1e-6)
        assert m.header.frame_id == "os_imu"
    meta_msgs = list(r.iter_raw(["/mobile_1/ouster/metadata"]))
    assert len(meta_msgs) == 1


def test_decoded_bag_metadata_yaml_for_retime():
    import yaml
    out, *_ = _decoded()
    info = yaml.safe_load(open(os.path.join(out, "metadata.yaml")))["rosbag2_bagfile_information"]
    for k in ("relative_file_paths", "message_count", "starting_time", "duration",
              "topics_with_message_count", "files", "storage_identifier"):
        assert k in info, k                                # the keys retime_bag.py rewrites
    names = {t["topic_metadata"]["name"] for t in info["topics_with_message_count"]}
    assert names == {"/mobile_1/ouster/points", "/mobile_1/ouster/imu",
                     "/mobile_1/ouster/metadata"}, names
    total = sum(t["message_count"] for t in info["topics_with_message_count"])
    assert total == info["message_count"]
    logs = [lt for _, lt, _, _ in bag.open_bag(out).iter_raw()]
    assert logs == sorted(logs), "log times must be non-decreasing"


REMAP = {"/ouster/metadata": "/mobile_1/ouster/metadata",
         "/ouster/lidar_packets": "/mobile_1/ouster/lidar_packets",
         "/zed/zed_node/imu/data": "/mobile_1/zed/imu/data"}


def test_remap_copies_topics_verbatim():
    out, p, frames, _ = _decoded("remapped", remap=REMAP)
    r = bag.open_bag(out)
    tp = r.topics()
    src = bag.open_bag(p)
    for s_topic, d_topic in REMAP.items():
        a = [(lt, bytes(pl)) for _, lt, pl, _ in src.iter_raw([s_topic])]
        b = [(lt, bytes(pl)) for _, lt, pl, _ in r.iter_raw([d_topic])]
        assert a == b, (s_topic, len(a), len(b))             # same bytes, same times
        assert tp[d_topic].msgtype == src.topics()[s_topic].msgtype
    assert tp["/mobile_1/zed/imu/data"].count == ZED_COUNT[p]
    assert tp["/mobile_1/ouster/points"].count == len(frames) - 1   # still decoded
    assert tp["/mobile_1/ouster/metadata"].count == 1                # copied, not doubled
    m = next(dec(pl) for _, _, pl, dec in r.iter_raw(["/mobile_1/zed/imu/data"]))
    assert m.linear_acceleration.z == 9.8                            # decodes as sensor_msgs/Imu
    logs = [lt for _, lt, _, _ in r.iter_raw()]
    assert logs == sorted(logs)


def test_parse_remaps():
    assert replay.parse_remaps(["/a:=/b", " /c := /c "]) == {"/a": "/b", "/c": "/c"}
    for bad in (["/a=/b"], ["a:=/b"], ["/a:=/b", "/a:=/c"], ["/a:=/x", "/b:=/x"]):
        try:
            replay.parse_remaps(bad)
        except ValueError:
            continue
        raise AssertionError(bad)
    _, p, _, _, _ = Fixture.get()
    for bad in ({"/zed/zed_node/imu/data": "/mobile_1/ouster/points"}, {"/nope": "/x"}):
        try:
            replay.decode_ouster_bag(p, os.path.join(Fixture.get()[0], "bad"), NS,
                                     metadata=meta(), remap=bad, log=quiet)
        except (ValueError, bag.BagError):
            continue
        raise AssertionError(bad)


def test_header_stamp_in_bytes():
    m = rosmsg.imu(1_700_000_000_000_000_123, "imu", [0, 0, 9.8], [0, 0, 1])
    b = rosmsg.serialize(m)
    assert retime.header_stamp_ns(b) == 1_700_000_000_000_000_123
    b2 = retime.with_header_stamp(b, 1_700_000_005_000_000_007)
    m2 = TS.deserialize_cdr(b2, "sensor_msgs/msg/Imu")
    assert (m2.header.stamp.sec, m2.header.stamp.nanosec) == (1_700_000_005, 7)
    assert b2[12:] == b[12:]                               # nothing else touched


def test_retime_decoded_bag():
    out, p, frames, s0 = _decoded("remapped", remap=REMAP)
    dst = os.path.join(Fixture.get()[0], "retimed")
    if not os.path.exists(dst):
        retime.retime_bag(p, out, dst, NS, "/mobile_1/ouster", metadata=meta(), log=quiet)
    cm = ouster.fit_clock(bag.open_bag(p), NS, metadata=meta(), log=quiet)
    src = {ch.topic: [] for _, _, ch, _ in bag.iter_mcap_records(bag.open_bag(out))}
    for _, _, ch, m in bag.iter_mcap_records(bag.open_bag(out)):
        src[ch.topic].append(m)
    got = {t: [] for t in src}
    for _, _, ch, m in bag.iter_mcap_records(bag.open_bag(dst)):
        got[ch.topic].append(m)
    # points / imu: stamp mapped by the clock fit, log = publish = new stamp
    for t in ("/mobile_1/ouster/points", "/mobile_1/ouster/imu"):
        assert len(got[t]) == len(src[t])
        for a, b in zip(src[t], got[t]):
            new = cm.host_ns(retime.header_stamp_ns(a.data))
            assert retime.header_stamp_ns(b.data) == new == b.log_time == b.publish_time
            assert bytes(b.data)[12:] == bytes(a.data)[12:]
    # everything else: bytes untouched, times shifted by one constant
    shifts = set()
    for t in ("/mobile_1/zed/imu/data", "/mobile_1/ouster/lidar_packets"):
        for a, b in zip(src[t], got[t]):
            assert bytes(a.data) == bytes(b.data)
            shifts.add(b.log_time - a.log_time)
    assert len(shifts) == 1, shifts
    logs = [m.log_time for _, _, _, m in bag.iter_mcap_records(bag.open_bag(dst))]
    assert logs == sorted(logs)
    # output lands on the capture clock the fit describes
    first = cm.host_ns(int(frames[0]["ts"][0]))
    assert abs(first - truth_host_ns(int(frames[0]["ts"][0]), s0)) < 100_000


if __name__ == "__main__":
    tests = [(k, v) for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    try:
        for name, fn in tests:
            check(name, fn)
    finally:
        Fixture.cleanup()
    print("\n%d/%d passed" % (len(tests) - len(FAILED), len(tests)))
    sys.exit(1 if FAILED else 0)
