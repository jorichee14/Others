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
import sys
import tempfile
import traceback
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.simplefilter("ignore", FutureWarning)

from rosbags.typesys import Stores, get_typestore, get_types_from_msg  # noqa: E402

from scoop import bag, ouster                                          # noqa: E402

TS = get_typestore(Stores.ROS2_HUMBLE)
TS.register(get_types_from_msg("uint8[] buf", "ouster_sensor_msgs/msg/PacketMsg"))
S = ouster.sdk()
NS = "/ouster"
OFFSET = 1_700_000_000.25        # host - sensor at s0, seconds
DRIFT = 20e-6                     # 20 ppm
FAILED = []


def check(name, fn):
    try:
        fn()
        print("  ok    %s" % name)
    except Exception:
        FAILED.append(name)
        print("  FAIL  %s" % name)
        traceback.print_exc()


def make_bag(path, n_frames=60, dead_cols=None, seed=0):
    """Returns (info, frames): the truth the decoder is checked against."""
    from mcap.writer import CompressionType, Writer
    rng = np.random.default_rng(seed)
    info = default_info()
    pf = ouster._packet_format(info)
    W = info.w
    s0 = 1000 * 10**9                                     # sensor clock, ns
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

    PM = TS.types["ouster_sensor_msgs/msg/PacketMsg"]
    Str = TS.types["std_msgs/msg/String"]
    with open(path, "wb") as fh:
        w = Writer(fh, compression=CompressionType.NONE)
        w.start("ros2", "test")
        sm = w.register_schema("std_msgs/msg/String", "ros2msg", b"string data")
        sp = w.register_schema("ouster_sensor_msgs/msg/PacketMsg", "ros2msg", b"uint8[] buf")
        cm = w.register_channel(f"{NS}/metadata", "cdr", sm)
        cp = w.register_channel(f"{NS}/lidar_packets", "cdr", sp)
        meta = bytes(TS.serialize_cdr(Str(data=info.to_json_string()), "std_msgs/msg/String"))
        t0 = min(h for h, _ in packets) - 10**6
        w.add_message(cm, log_time=t0, publish_time=t0, data=meta)
        for i, (h, b) in enumerate(packets):
            data = bytes(TS.serialize_cdr(PM(buf=np.frombuffer(b, np.uint8).copy()),
                                          "ouster_sensor_msgs/msg/PacketMsg"))
            w.add_message(cp, log_time=h, publish_time=h, data=data, sequence=i)
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
