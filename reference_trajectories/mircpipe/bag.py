"""One place that knows how to read a rosbag2 mcap.

Every stage and analysis goes through `iter_topic`; nothing else in the
pipeline opens a bag. ROS 2 imports are deferred to call time so that the
maths modules stay importable without a sourced workspace.
"""
import numpy as np

from .se3 import Rt, path_length


def _reader(path, topics=None):
    import rosbag2_py
    r = rosbag2_py.SequentialReader()
    r.open(rosbag2_py.StorageOptions(uri=str(path), storage_id="mcap"),
           rosbag2_py.ConverterOptions("", ""))
    types = {t.name: t.type for t in r.get_all_topics_and_types()}
    if topics:
        f = rosbag2_py.StorageFilter(); f.topics = list(topics); r.set_filter(f)
    return r, types


def topic_types(path):
    """{topic: type_string} for everything in the bag."""
    r, types = _reader(path)
    del r
    return types


def iter_topic(path, topic, stride=1, limit=None):
    """Yield (t, msg) for one topic. `t` is the header stamp in seconds when
    the message has a header, else the bag receive time."""
    from rclpy.serialization import deserialize_message
    try:
        from rosidl_runtime_py.utilities import get_message
    except ImportError:
        from rosidl_runtime_py.utility import get_message
    r, types = _reader(path, [topic])
    if topic not in types:
        raise KeyError("%s not in bag; have %s..." % (topic, sorted(types)[:8]))
    cls = get_message(types[topic])
    i = n = 0
    while r.has_next():
        _, data, t_bag = r.read_next()
        if i % stride:
            i += 1; continue
        i += 1
        m = deserialize_message(data, cls)
        h = getattr(m, "header", None)
        t = (h.stamp.sec + h.stamp.nanosec * 1e-9) if h is not None else t_bag * 1e-9
        yield t, m
        n += 1
        if limit and n >= limit:
            break


def topic_frame(path, topic):
    """header.frame_id of the first message - the frame the data is IN."""
    for _, m in iter_topic(path, topic, limit=1):
        h = getattr(m, "header", None)
        return h.frame_id if h is not None else "?"
    return "?"


# ------------------------------------------------------------------ odometry
def read_odom(path, topic, printer=print):
    """nav_msgs/Odometry -> (ts, Ts, child_frame_id)."""
    from scipy.spatial.transform import Rotation as Rot
    ts, Ts, child, parent = [], [], None, None
    for t, m in iter_topic(path, topic):
        p = m.pose.pose.position; o = m.pose.pose.orientation
        if child is None:
            child = getattr(m, "child_frame_id", "") or "?"
            parent = m.header.frame_id or "?"
        ts.append(t)
        Ts.append(Rt(Rot.from_quat([o.x, o.y, o.z, o.w]).as_matrix(),
                     np.array([p.x, p.y, p.z])))
    if not ts:
        raise SystemExit("no odometry on %s" % topic)
    ts = np.array(ts); Ts = np.array(Ts)
    if printer:
        printer("  odom %s: %d poses, %.1f s, path %.1f m, frame '%s' -> "
                "child_frame_id '%s'" % (topic, len(ts), ts[-1] - ts[0],
                                         path_length(Ts), parent, child))
    return ts, Ts, child


def read_pose(path, topic):
    """geometry_msgs/PoseStamped (or PoseWithCovarianceStamped) -> (ts, Ts)."""
    from scipy.spatial.transform import Rotation as Rot
    ts, Ts = [], []
    for t, m in iter_topic(path, topic):
        ps = m.pose.pose if hasattr(m.pose, "pose") else m.pose
        p, o = ps.position, ps.orientation
        ts.append(t)
        Ts.append(Rt(Rot.from_quat([o.x, o.y, o.z, o.w]).as_matrix(),
                     np.array([p.x, p.y, p.z])))
    return np.array(ts), np.array(Ts)


def read_imu(path, topic, printer=print):
    """sensor_msgs/Imu -> (ts, gyro (N,3), accel (N,3), frame_id)."""
    ts, gyr, acc, frame = [], [], [], None
    for t, m in iter_topic(path, topic):
        if frame is None:
            frame = m.header.frame_id
        ts.append(t)
        gyr.append([m.angular_velocity.x, m.angular_velocity.y, m.angular_velocity.z])
        acc.append([m.linear_acceleration.x, m.linear_acceleration.y,
                    m.linear_acceleration.z])
    if not ts:
        raise SystemExit("no IMU on %s" % topic)
    ts = np.array(ts)
    if printer:
        printer("  imu %s: %d samples, %.0f Hz, frame '%s'"
                % (topic, len(ts), len(ts) / max(ts[-1] - ts[0], 1e-9), frame))
    return ts, np.array(gyr), np.array(acc), frame


def tf_static_rot(path, target, source):
    """R_target_source from /tf_static, chained through intermediate frames.
    None when the two frames are not connected (or there is no /tf_static)."""
    from scipy.spatial.transform import Rotation as Rot
    edges = {}
    try:
        for _, m in iter_topic(path, "/tf_static"):
            for tr in m.transforms:
                q = tr.transform.rotation
                R = Rot.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
                p, c = tr.header.frame_id, tr.child_frame_id
                edges.setdefault(p, []).append((c, R))
                edges.setdefault(c, []).append((p, R.T))
    except KeyError:
        return None
    seen = {target: np.eye(3)}; queue = [target]
    while queue:
        x = queue.pop(0)
        if x == source:
            return seen[x]
        for y, R_xy in edges.get(x, []):
            if y not in seen:
                seen[y] = seen[x] @ R_xy; queue.append(y)
    return None


# ------------------------------------------------------------- point clouds
_DT = {1: np.int8, 2: np.uint8, 3: np.int16, 4: np.uint16,
       5: np.int32, 6: np.uint32, 7: np.float32, 8: np.float64}


def pc2_xyzt(msg):
    """PointCloud2 -> (xyz float32 (N,3), per-point time offsets (N,) or
    None). Handles row padding and nanosecond time fields."""
    n = msg.width * msg.height
    raw = np.frombuffer(msg.data, np.uint8)
    if msg.row_step != msg.width * msg.point_step and msg.height > 1:
        raw = raw.reshape(msg.height, msg.row_step)[:, :msg.width * msg.point_step]
        raw = raw.reshape(-1)
    buf = raw[:n * msg.point_step].reshape(n, msg.point_step)
    off = {f.name: (f.offset, _DT[f.datatype]) for f in msg.fields}

    def col(name):
        o, dt = off[name]
        return buf[:, o:o + np.dtype(dt).itemsize].copy().view(dt).ravel()

    xyz = np.column_stack([col("x"), col("y"), col("z")]).astype(np.float32)
    ok = np.isfinite(xyz).all(1) & (np.abs(xyz) < 1e4).all(1)
    tr = None
    for name in ("t", "time", "timestamp", "time_offset"):
        if name in off:
            tv = col(name).astype(np.float64)[ok]
            if tv.size and tv.max() > 1e6:          # nanoseconds
                tv = tv * 1e-9
            tr = tv - tv.min() if tv.size else None
            break
    return xyz[ok], tr


# ------------------------------------------------------------------- images
def img_gray(m):
    """sensor_msgs/Image -> uint8 grayscale."""
    a = np.frombuffer(m.data, np.uint8)
    enc = m.encoding.lower()
    if enc in ("mono8", "8uc1"):
        return a.reshape(m.height, m.step)[:, :m.width].copy()
    if enc in ("bgr8", "rgb8", "bgra8", "rgba8"):
        nch = 3 if enc in ("bgr8", "rgb8") else 4
        im = a.reshape(m.height, m.step)[:, :m.width * nch]
        im = im.reshape(m.height, m.width, nch)
        w = (0.114, 0.587, 0.299) if enc.startswith("bgr") else (0.299, 0.587, 0.114)
        return (im[..., 0] * w[0] + im[..., 1] * w[1] + im[..., 2] * w[2]).astype(np.uint8)
    if enc in ("mono16", "16uc1"):
        return (a.view(np.uint16).reshape(m.height, m.step // 2)[:, :m.width]
                >> 8).astype(np.uint8)
    raise SystemExit("unsupported image encoding %r" % m.encoding)


def img_depth(m):
    """Depth image -> metres. 16UC1 is millimetres (D455), 32FC1 is metres
    with NaN/inf for invalid (ZED)."""
    a = np.frombuffer(m.data, np.uint8)
    if m.encoding in ("16UC1", "mono16"):
        return (a.view(np.uint16).reshape(m.height, m.step // 2)[:, :m.width]
                .astype(np.float32) * 0.001)
    if m.encoding == "32FC1":
        z = a.view(np.float32).reshape(m.height, m.step // 4)[:, :m.width].copy()
        z[~np.isfinite(z)] = 0
        return z
    raise SystemExit("unsupported depth encoding %r" % m.encoding)


def camera_K(path, info_topic):
    """3x3 intrinsics from the first CameraInfo on a topic."""
    for _, ci in iter_topic(path, info_topic, limit=1):
        return np.array(ci.k).reshape(3, 3)
    raise SystemExit("no CameraInfo on %s" % info_topic)


def read_map_xyz(path):
    """Point cloud file (pcd/ply) -> (N,3). Needs open3d."""
    import open3d as o3d
    P = np.asarray(o3d.io.read_point_cloud(str(path)).points)
    if len(P) == 0:
        raise SystemExit("no points in %s" % path)
    return P
