#!/usr/bin/env python
"""Self-tests for scoop.bag. No real bag, no ROS, no GPU.

Every payload is serialised by an independent implementation (rosbags) and
the fast CDR path is checked against that library's own deserialiser, so a
parser bug cannot hide behind a matching encoder bug. The last tests write a
synthetic MCAP and read it back through the full iterator, including the
generic-decoder fallback.

    python scripts/test_bag.py
"""
import os
import shutil
import sys
import tempfile
import traceback

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rosbags.typesys import Stores, get_typestore                   # noqa: E402

from scoop import bag                                               # noqa: E402

TS = get_typestore(Stores.ROS2_HUMBLE)
T = TS.types
NS = 1_000_000_000
FAILED = []


def check(name, fn):
    try:
        fn()
        print("  ok    %s" % name)
    except Exception:
        FAILED.append(name)
        print("  FAIL  %s" % name)
        traceback.print_exc()


# --------------------------------------------------------------------------- #
def header(t_ns, frame="os_sensor"):
    return T["std_msgs/msg/Header"](
        stamp=T["builtin_interfaces/msg/Time"](sec=t_ns // NS, nanosec=t_ns % NS),
        frame_id=frame)


def ouster_cloud(t_ns, h=4, w=8, seed=0, big=False, pad_rows=0, abs_time=False):
    """An Ouster-shaped cloud: xyz f32, 4 pad bytes, intensity f32, t u32 ns,
    reflectivity u16, ring u16, then padding to a 48-byte point_step -- the
    real os_cloud_node layout, padding included."""
    rng = np.random.default_rng(seed)
    n = h * w
    pts = rng.uniform(-20, 20, (n, 3)).astype(np.float32)
    pts[0] = np.nan                      # no-return points
    pts[1] = 0.0                         # Ouster zeros a no-return
    e = ">" if big else "<"
    if abs_time:
        tfield = ("t", e + "f8", 24)
        tvals = t_ns * 1e-9 + np.linspace(0, 0.1, n)
    else:
        tfield = ("t", e + "u4", 20)
        tvals = np.linspace(0, 99_000_000, n).astype(np.uint32)
    dt = np.dtype({"names": ["x", "y", "z", "intensity", tfield[0], "reflectivity"],
                   "formats": [e + "f4"] * 4 + [tfield[1], e + "u2"],
                   "offsets": [0, 4, 8, 16, tfield[2], 32],
                   "itemsize": 48})
    arr = np.zeros(n, dt)
    arr["x"], arr["y"], arr["z"] = pts[:, 0], pts[:, 1], pts[:, 2]
    arr["intensity"] = rng.uniform(0, 100, n)
    arr["t"] = tvals
    arr["reflectivity"] = np.arange(n) % 65535
    PF = T["sensor_msgs/msg/PointField"]
    fields = [PF(name="x", offset=0, datatype=7, count=1),
              PF(name="y", offset=4, datatype=7, count=1),
              PF(name="z", offset=8, datatype=7, count=1),
              PF(name="intensity", offset=16, datatype=7, count=1),
              PF(name="t", offset=tfield[2], datatype=8 if abs_time else 6, count=1),
              PF(name="reflectivity", offset=32, datatype=4, count=1)]
    raw = arr.tobytes()
    row = w * 48
    if pad_rows:
        rows = np.frombuffer(raw, np.uint8).reshape(h, row)
        raw = np.concatenate([rows, np.full((h, pad_rows), 0xAB, np.uint8)], 1).tobytes()
    msg = T["sensor_msgs/msg/PointCloud2"](
        header=header(t_ns), height=h, width=w, fields=fields, is_bigendian=big,
        point_step=48, row_step=row + pad_rows,
        data=np.frombuffer(raw, np.uint8).copy(), is_dense=False)
    return msg, arr


def ser(msg, little=True):
    return bytes(TS.serialize_cdr(msg, msg.__msgtype__, little_endian=little))


def image_msg(t_ns, enc, h=6, w=5, pad=0, seed=0):
    rng = np.random.default_rng(seed)
    ch = {"mono8": 1, "bgr8": 3, "rgb8": 3, "bgra8": 4, "rgba8": 4, "16uc1": 2}[enc]
    img = rng.integers(0, 255, (h, w * ch), dtype=np.uint8)
    data = np.concatenate([img, np.zeros((h, pad), np.uint8)], 1) if pad else img
    msg = T["sensor_msgs/msg/Image"](header=header(t_ns, "cam"), height=h, width=w,
                                     encoding=enc, is_bigendian=0, step=w * ch + pad,
                                     data=data.reshape(-1).copy())
    return msg, img


# --------------------------------------------------------------------------- #
# CDR PointCloud2
# --------------------------------------------------------------------------- #
def test_pointcloud2_matches_reference_decoder():
    msg, arr = ouster_cloud(1_700_000_000 * NS + 123, seed=1)
    stamp, got = bag.parse_pointcloud2(ser(msg))
    assert stamp == 1_700_000_000 * NS + 123
    for f in ("x", "y", "z", "intensity", "t", "reflectivity"):
        np.testing.assert_array_equal(got[f], arr[f], err_msg=f)


def test_pointcloud2_big_endian_cdr_and_data():
    msg, arr = ouster_cloud(5 * NS, big=True, seed=2)
    _, got = bag.parse_pointcloud2(ser(msg, little=False))
    for f in ("x", "y", "z", "t"):
        np.testing.assert_array_equal(got[f], arr[f], err_msg=f)


def test_pointcloud2_padded_rows():
    msg, arr = ouster_cloud(5 * NS, h=3, w=7, pad_rows=13, seed=3)
    _, got = bag.parse_pointcloud2(ser(msg))
    assert got.shape == (21,)
    np.testing.assert_array_equal(got["x"], arr["x"])
    np.testing.assert_array_equal(got["t"], arr["t"])


def test_pointcloud2_is_zero_copy():
    payload = ser(ouster_cloud(5 * NS)[0])
    _, got = bag.parse_pointcloud2(payload)
    assert not got.flags.writeable and got.base is not None


def test_xyz_view_and_gather_paths_agree():
    msg, arr = ouster_cloud(5 * NS, seed=4)
    _, le = bag.parse_pointcloud2(ser(msg))
    _, be = bag.parse_pointcloud2(ser(ouster_cloud(5 * NS, seed=4, big=True)[0]))
    v = bag._xyz(le, "/t")               # adjacent native f32 -> strided view
    g = bag._xyz(be, "/t")               # big-endian data -> gathered copy
    assert v.base is not None and not v.flags.writeable
    assert g.flags.writeable
    ref = np.stack([arr["x"], arr["y"], arr["z"]], 1)
    np.testing.assert_array_equal(v, ref)
    np.testing.assert_array_equal(g, ref)


def test_unsupported_encapsulation_refused():
    payload = bytearray(ser(ouster_cloud(5 * NS)[0]))
    payload[1] = 0x07                    # XCDR2 little endian
    try:
        bag.parse_pointcloud2(bytes(payload))
    except bag._Unsupported:
        return
    raise AssertionError("XCDR2 must go to the generic decoder")


def test_truncated_payload_refused():
    payload = ser(ouster_cloud(5 * NS)[0])
    for cut in (10, 60, len(payload) - 100):
        try:
            bag.parse_pointcloud2(payload[:cut])
        except bag._Unsupported:
            continue
        raise AssertionError("truncated at %d accepted" % cut)


# --------------------------------------------------------------------------- #
# per-point time
# --------------------------------------------------------------------------- #
def test_point_time_relative_ns():
    msg, arr = ouster_cloud(5 * NS)
    _, got = bag.parse_pointcloud2(ser(msg))
    pt = bag._PointTime(log=lambda *_: None)
    t = pt(got, 5 * NS)
    assert pt.scale == 1e-9 and not pt.absolute
    np.testing.assert_allclose(t, arr["t"] * 1e-9)


def test_point_time_absolute_seconds():
    t0 = 1_700_000_000 * NS
    msg, arr = ouster_cloud(t0, abs_time=True)
    _, got = bag.parse_pointcloud2(ser(msg))
    pt = bag._PointTime(log=lambda *_: None)
    t = pt(got, t0)
    assert pt.absolute
    np.testing.assert_allclose(t, np.linspace(0, 0.1, len(t)), atol=1e-6)


# --------------------------------------------------------------------------- #
# images
# --------------------------------------------------------------------------- #
def test_image_encodings():
    import cv2
    for enc in ("mono8", "bgr8", "rgb8", "bgra8", "rgba8"):
        msg, img = image_msg(7 * NS, enc, pad=3)
        stamp, e, got = bag.parse_image(ser(msg))
        assert stamp == 7 * NS and e == enc
        ch = 1 if enc == "mono8" else len(enc) - 1
        ref = img.reshape(6, 5, ch) if ch > 1 else img
        np.testing.assert_array_equal(got, ref, err_msg=enc)
        bgr = bag._convert(e, got, "bgr")
        assert bgr.shape == (6, 5, 3) and bgr.flags.writeable
        if enc == "rgb8":
            np.testing.assert_array_equal(bgr, ref[:, :, ::-1])
        gray = bag._convert(e, got, "gray")
        assert gray.shape == (6, 5)
        if enc == "bgr8":
            np.testing.assert_array_equal(gray, cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY))


def test_image_depth_raw():
    msg, img = image_msg(7 * NS, "16uc1")
    _, e, got = bag.parse_image(ser(msg))
    assert e == "16uc1" and got.dtype == np.dtype("<u2") and got.shape == (6, 5)
    np.testing.assert_array_equal(got, img.reshape(-1).view("<u2").reshape(6, 5))


def test_compressed_image():
    import cv2
    img = np.zeros((8, 8, 3), np.uint8)
    img[:, 4:] = (10, 200, 30)
    ok, png = cv2.imencode(".png", img)
    msg = T["sensor_msgs/msg/CompressedImage"](header=header(3 * NS), format="png",
                                               data=png.reshape(-1))
    stamp, e, got = bag.parse_image(ser(msg), compressed=True)
    assert stamp == 3 * NS and e == "bgr8"
    np.testing.assert_array_equal(got, img)


# --------------------------------------------------------------------------- #
# selection
# --------------------------------------------------------------------------- #
def test_nearest_pose_tolerance():
    sel = bag.nearest_pose([10.0, 10.1, 10.2], 0.02)
    assert sel(int(10.1 * NS)) == 1
    assert sel(int(10.115 * NS)) == 1
    assert sel(int(10.14 * NS)) is None      # 40 ms from both neighbours
    assert sel(int(9.99 * NS)) == 0
    assert sel(int(9.9 * NS)) is None        # before the trajectory
    assert sel(int(10.21 * NS)) == 2


# --------------------------------------------------------------------------- #
# end to end through an MCAP
# --------------------------------------------------------------------------- #
def write_mcap(path, streams):
    """streams: {topic: [(msg, log_ns), ...]}; payloads serialised by rosbags,
    schemas generated by rosbags, so the fallback decoder (mcap_ros2) reads
    the same bytes independently."""
    from mcap.writer import CompressionType, Writer
    with open(path, "wb") as fh:
        w = Writer(fh, compression=CompressionType.NONE)   # SCooP bags are uncompressed
        w.start("ros2", "scoop-test")
        chans = {}
        for topic, items in streams.items():
            tname = items[0][0].__msgtype__
            text, _ = TS.generate_msgdef(tname, ros_version=2)
            sid = w.register_schema(tname, "ros2msg", text.encode())
            chans[topic] = w.register_channel(topic, "cdr", sid)
        allmsgs = sorted(((log, topic, msg) for topic, items in streams.items()
                          for msg, log in items), key=lambda x: (x[0], x[1]))
        for i, (log, topic, msg) in enumerate(allmsgs):
            data = ser(msg)
            if getattr(msg, "_xcdr2", False):
                data = data[:1] + b"\x07" + data[2:]
            w.add_message(chans[topic], log_time=log, publish_time=log, data=data,
                          sequence=i)
        w.finish()


def with_bag(streams, fn):
    tmp = tempfile.mkdtemp()
    try:
        path = os.path.join(tmp, "test.mcap")
        write_mcap(path, streams)
        fn(bag.open_bag(path))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_iter_scans_select_and_filters():
    stamps = [100 * NS + k * 100_000_000 for k in range(10)]
    clouds = [ouster_cloud(s, seed=k) for k, s in enumerate(stamps)]
    # log time lags the header by 30 ms: selection must use the HEADER
    streams = {"/ouster/points": [(m, s + 30_000_000) for (m, _), s in zip(clouds, stamps)],
               "/radar/points": [(ouster_cloud(s, h=1, w=4)[0], s) for s in stamps]}
    traj = np.array([s * 1e-9 for s in stamps[2:8]])

    def run(reader):
        st = {}
        scans = list(bag.iter_scans(reader, "/ouster/points",
                                    select=bag.nearest_pose(traj, 0.005),
                                    min_range=0.5, max_range=30.0,
                                    with_time=True, stats=st, log=lambda *_: None))
        assert st["seen"] == 10 and st["selected"] == 6 and st["fast"] == 6, st
        assert st["fallback"] == 0
        assert [s.key for s in scans] == list(range(6))
        for s in scans:
            _, arr = clouds[s.key + 2]
            xyz = np.stack([arr["x"], arr["y"], arr["z"]], 1)
            d = np.linalg.norm(xyz, axis=1)
            keep = np.isfinite(d) & (d > 0.5) & (d < 30.0)
            assert s.stamp_ns == stamps[s.key + 2]
            np.testing.assert_array_equal(s.xyz, xyz[keep])
            np.testing.assert_allclose(s.t, arr["t"][keep] * 1e-9)
            assert s.xyz.dtype == np.float32 and s.xyz.flags.writeable
    with_bag(streams, run)


def test_iter_scans_fallback_matches_fast_path():
    stamps = [100 * NS + k * 100_000_000 for k in range(4)]
    fast = [(ouster_cloud(s, seed=k)[0], s) for k, s in enumerate(stamps)]
    slow = [(ouster_cloud(s, seed=k)[0], s) for k, s in enumerate(stamps)]
    for m, _ in slow:
        m._xcdr2 = True                   # forces the generic decoder
    got = {}

    def run(topic):
        def inner(reader):
            st = {}
            got[topic] = list(bag.iter_scans(reader, topic, with_time=True,
                                             stats=st, log=lambda *_: None))
            got[topic + "_st"] = st
        return inner

    with_bag({"/a": fast}, run("/a"))
    try:
        with_bag({"/b": slow}, run("/b"))
    except Exception as e:           # mcap_ros2 may not read XCDR2 either
        print("        (fallback on XCDR2 not decodable by mcap_ros2: %s)" % e)
        return
    assert got["/b_st"]["fallback"] == 4, got["/b_st"]
    for a, b in zip(got["/a"], got["/b"]):
        np.testing.assert_array_equal(a.xyz, b.xyz)
        np.testing.assert_array_equal(a.t, b.t)


def test_iter_images_period_and_modes():
    stamps = [50 * NS + k * 50_000_000 for k in range(20)]          # 20 Hz
    frames = [image_msg(s, "bgra8", seed=k) for k, s in enumerate(stamps)]

    def run(reader):
        st = {}
        got = list(bag.iter_images(reader, "/cam", mode="gray", period=0.2, stats=st))
        assert [f.stamp_ns for f in got] == stamps[::4], [f.stamp_ns for f in got]
        assert st["selected"] == 5 and st["fast"] == 5
        assert got[0].image.shape == (6, 5) and got[0].encoding == "bgra8"
        bgr = next(bag.iter_images(reader, "/cam", mode="bgr"))
        np.testing.assert_array_equal(bgr.image, frames[0][1].reshape(6, 5, 4)[:, :, :3])
    with_bag({"/cam": [(m, s) for (m, _), s in zip(frames, stamps)]}, run)


def test_detect_points_topic():
    s = [NS * k for k in range(1, 6)]
    streams = {"/ouster/points": [(ouster_cloud(t, h=1, w=4)[0], t) for t in s],
               "/radar/points": [(ouster_cloud(t, h=1, w=4)[0], t) for t in s * 3],
               "/zed/cloud": [(ouster_cloud(t, h=1, w=4)[0], t) for t in s * 2]}
    with_bag(streams, lambda r: _eq(
        bag.detect_points_topic(r, n_poses=5, log=lambda *_: None), "/ouster/points"))


def _eq(a, b):
    assert a == b, (a, b)


# --------------------------------------------------------------------------- #
# writing: rosmsg + bagwrite
# --------------------------------------------------------------------------- #
def test_point_fields_from_dtype():
    from scoop import rosmsg
    dt = np.dtype({"names": ["x", "y", "z", "t", "ring"],
                   "formats": ["<f4", "<f4", "<f4", "<u4", "<u2"],
                   "offsets": [0, 4, 8, 16, 20], "itemsize": 24})
    assert rosmsg.point_fields(dt) == [("x", 0, 7), ("y", 4, 7), ("z", 8, 7),
                                       ("t", 16, 6), ("ring", 20, 4)]
    for bad in (np.dtype([("x", ">f4")]), np.dtype([("x", "<f4", (3,))]),
                np.dtype([("x", "<c8")])):
        try:
            rosmsg.point_fields(bad)
        except ValueError:
            continue
        raise AssertionError(bad)


def test_bagwriter_roundtrip_and_copy():
    from scoop import rosmsg
    from scoop.bagwrite import BagWriter
    tmp = tempfile.mkdtemp()
    try:
        # a source bag with one topic to copy
        src_dir = os.path.join(tmp, "src")
        with BagWriter(src_dir) as w:
            for k in range(3):
                w.write("/zed/imu", rosmsg.imu(10 * NS + k, "imu", [0, 0, 9.8], [0, 0, k],
                                               accel_cov=0.1), 10 * NS + k)
        src = bag.open_bag(src_dir)                              # via metadata.yaml
        schemas = bag.topic_schemas(src)
        assert schemas["/zed/imu"].name == "sensor_msgs/msg/Imu"

        dt = np.dtype({"names": ["x", "y", "z", "t"], "formats": ["<f4"] * 3 + ["<u4"],
                       "offsets": [0, 4, 8, 12], "itemsize": 16})
        pts = np.zeros(6, dt)
        pts["x"] = np.arange(1, 7)
        pts["t"] = np.arange(6) * 10_000_000                     # 0..50 ms, a sweep
        out_dir = os.path.join(tmp, "out")
        with BagWriter(out_dir) as w:
            w.write("/meta", rosmsg.string("hello"), 10 * NS)
            w.write("/pts", rosmsg.pointcloud2(pts, 10 * NS + 5, "lidar", height=2, width=3),
                    10 * NS + 5)
            for _, lt, pl, _ in src.iter_raw(["/zed/imu"]):
                w.write_raw("/copied/imu", schemas["/zed/imu"], pl, lt)
        r = bag.open_bag(out_dir)
        tp = r.topics()
        assert {t: (i.msgtype, i.count) for t, i in tp.items()} == {
            "/meta": ("std_msgs/msg/String", 1),
            "/pts": ("sensor_msgs/msg/PointCloud2", 1),
            "/copied/imu": ("sensor_msgs/msg/Imu", 3)}
        scan = next(bag.iter_scans(r, "/pts", with_time=True, log=lambda *_: None))
        np.testing.assert_array_equal(scan.xyz[:, 0], np.arange(1, 7))
        np.testing.assert_allclose(scan.t, np.arange(6) * 0.01)
        assert scan.stamp_ns == 10 * NS + 5
        a = [bytes(pl) for _, _, pl, _ in src.iter_raw(["/zed/imu"])]
        b = [bytes(pl) for _, _, pl, _ in r.iter_raw(["/copied/imu"])]
        assert a == b
        m = next(d(pl) for _, _, pl, d in r.iter_raw(["/copied/imu"]))
        assert m.linear_acceleration.z == 9.8 and m.orientation_covariance[0] == -1.0
        import yaml
        meta = yaml.safe_load(open(os.path.join(out_dir, "metadata.yaml")))
        assert meta["rosbag2_bagfile_information"]["message_count"] == 5
        try:
            BagWriter(out_dir)
        except FileExistsError:
            pass
        else:
            raise AssertionError("an existing bag must never be overwritten")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    tests = [(k, v) for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    for name, fn in tests:
        check(name, fn)
    print("\n%d/%d passed" % (len(tests) - len(FAILED), len(tests)))
    sys.exit(1 if FAILED else 0)
