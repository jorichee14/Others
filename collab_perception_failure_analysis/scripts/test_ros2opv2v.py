#!/usr/bin/env python
"""Self-tests for the ros2opv2v converter. No bag, no ROS, no GPU required.

Covers the four things that silently produce a wrong-but-plausible dataset:

  * the pose parameterisation (angles that OpenCOOD's x_to_world must reproduce),
  * the intensity encoding (OpenCOOD reads it out of the PCD colour channel),
  * PointCloud2 / depth decoding (field offsets, padding, endianness, optical frame),
  * frame synchronisation (an agent missing from one frame is a KeyError later).

The last test writes a synthetic three-agent MCAP with known geometry, runs the
full conversion over it, and checks the boxes land where the geometry says they
must — so the whole pipeline is exercised end to end.

    python scripts/test_ros2opv2v.py
"""
import os
import shutil
import sys
import tempfile

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ros2opv2v import config as cfgmod                            # noqa: E402
from ros2opv2v.bagreader import stamp_from_cdr                    # noqa: E402
from ros2opv2v.convert import convert                             # noqa: E402
from ros2opv2v.geometry import (invert, make_transform,           # noqa: E402
                                matrix_to_opencood_pose, matrix_to_rpy_config,
                                quat_to_matrix, matrix_to_quat, transform_points,
                                x_to_world)
from ros2opv2v.labels import agent_box, vehicles_for_viewer       # noqa: E402
from ros2opv2v.pointclouds import (cloud_from_depth_image,        # noqa: E402
                                   cloud_from_pointcloud2, pointcloud2_to_array)
from ros2opv2v.pointclouds import deskew_cloud                     # noqa: E402
from ros2opv2v.sync import (PoseTrack, StampIndex,                # noqa: E402
                            build_frame_table, frame_times, tightness_curve)
from ros2opv2v import clock as clockmod                            # noqa: E402
from ros2opv2v.writers import read_pcd, write_pcd                 # noqa: E402

FAILED = []
NS = 1_000_000_000


def check(name, fn):
    try:
        fn()
        print('[PASS] %s' % name)
    except Exception as e:  # noqa: BLE001
        FAILED.append(name)
        import traceback
        print('[FAIL] %s — %s: %s' % (name, type(e).__name__, e))
        if os.environ.get('R2O_TRACE'):
            traceback.print_exc()


# --------------------------------------------------------------------- geometry

def test_pose_roundtrip_matches_opencood():
    """Every rotation must be expressible in x_to_world's own (roll, yaw, pitch)."""
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(2000):
        q = rng.normal(size=4)
        T = quat_to_matrix(*q)
        T[:3, 3] = rng.uniform(-100, 100, 3)
        pose = matrix_to_opencood_pose(T)
        worst = max(worst, float(np.abs(x_to_world(pose) - T).max()))
    assert worst < 1e-9, 'worst reconstruction error %g' % worst


def test_pose_roundtrip_at_gimbal_lock():
    for sign in (+1, -1):
        T = make_transform(1.0, 2.0, 3.0, roll=0.0, pitch=sign * 90.0, yaw=37.0)
        pose = matrix_to_opencood_pose(T)
        assert np.abs(x_to_world(pose) - T).max() < 1e-9


def test_relative_transform_is_convention_free():
    """x1_to_x2 built from our angles must equal the true relative transform.

    This is the property the whole converter rests on: OpenCOOD never sees our
    world frame, only differences of poses within it.
    """
    rng = np.random.default_rng(3)
    for _ in range(300):
        a = quat_to_matrix(*rng.normal(size=4)); a[:3, 3] = rng.uniform(-20, 20, 3)
        b = quat_to_matrix(*rng.normal(size=4)); b[:3, 3] = rng.uniform(-20, 20, 3)
        expected = invert(b) @ a
        got = np.linalg.inv(x_to_world(matrix_to_opencood_pose(b))) @ \
            x_to_world(matrix_to_opencood_pose(a))
        assert np.abs(expected - got).max() < 1e-9


def test_quaternion_roundtrip():
    rng = np.random.default_rng(11)
    for _ in range(500):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        back = matrix_to_quat(quat_to_matrix(*q))
        assert min(np.abs(back - q).max(), np.abs(back + q).max()) < 1e-9


# ---------------------------------------------------------------------- writers

def test_pcd_roundtrip_selfread():
    rng = np.random.default_rng(1)
    cloud = np.hstack([rng.uniform(-60, 60, (500, 3)),
                       rng.uniform(0, 1, (500, 1))]).astype(np.float32)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, 'a.pcd')
        assert write_pcd(path, cloud) == 500
        back = read_pcd(path)
        assert np.abs(back[:, :3] - cloud[:, :3]).max() == 0.0
        assert np.abs(back[:, 3] - cloud[:, 3]).max() <= 1.0 / 255


def test_pcd_readable_by_opencood_path():
    """OpenCOOD reads intensity from pcd.colors[:, 0] — assert that path works."""
    try:
        import open3d as o3d
    except ImportError:
        print('       (open3d absent — skipped the open3d half)')
        return
    rng = np.random.default_rng(2)
    cloud = np.hstack([rng.uniform(-60, 60, (400, 3)),
                       rng.uniform(0, 1, (400, 1))]).astype(np.float32)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, 'b.pcd')
        write_pcd(path, cloud)
        pcd = o3d.io.read_point_cloud(path)
        xyz = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)
        assert xyz.shape == (400, 3), xyz.shape
        assert colors.shape == (400, 3), 'no colour channel: intensity would be lost'
        assert np.abs(xyz - cloud[:, :3]).max() < 1e-5
        assert np.abs(colors[:, 0] - cloud[:, 3]).max() <= 1.0 / 255
        assert np.allclose(colors[:, 0], colors[:, 2])


def test_pcd_empty_cloud():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, 'e.pcd')
        assert write_pcd(path, np.zeros((0, 4), dtype=np.float32)) == 0
        assert read_pcd(path).shape == (0, 4)


# ------------------------------------------------------------------ point cloud

class _Field:
    def __init__(self, name, offset, datatype, count=1):
        self.name, self.offset, self.datatype, self.count = name, offset, datatype, count


class _Cloud:
    def __init__(self, fields, data, width, point_step, bigendian=False, dense=True):
        self.fields, self.data, self.width = fields, data, width
        self.height, self.point_step = 1, point_step
        self.row_step = point_step * width
        self.is_bigendian, self.is_dense = bigendian, dense


def _make_cloud(xyz, intensity, pad=4, dtype='<f4'):
    """Build a PointCloud2 payload with padding between xyz and intensity."""
    step = 12 + pad + 4
    buf = bytearray(step * len(xyz))
    for i, (point, value) in enumerate(zip(xyz, intensity)):
        off = i * step
        buf[off:off + 12] = np.asarray(point, dtype=dtype).tobytes()
        buf[off + 12 + pad:off + 16 + pad] = np.asarray([value], dtype=dtype).tobytes()
    fields = [_Field('x', 0, 7), _Field('y', 4, 7), _Field('z', 8, 7),
              _Field('intensity', 12 + pad, 7)]
    return _Cloud(fields, bytes(buf), len(xyz), step,
                  bigendian=dtype.startswith('>'))


def test_pointcloud2_respects_offsets_and_padding():
    xyz = np.array([[1, 2, 3], [4, 5, 6], [-7, 8, -9]], dtype=np.float32)
    intensity = np.array([0.25, 0.5, 0.75], dtype=np.float32)
    msg = _make_cloud(xyz, intensity)
    cloud = cloud_from_pointcloud2(msg, cfgmod.IntensityConfig.parse(None))
    assert np.abs(cloud[:, :3] - xyz).max() == 0.0
    assert np.abs(cloud[:, 3] - intensity).max() == 0.0


def test_pointcloud2_bigendian():
    xyz = np.array([[1.5, -2.5, 3.5]], dtype=np.float32)
    msg = _make_cloud(xyz, np.array([0.5], dtype=np.float32), dtype='>f4')
    cloud = cloud_from_pointcloud2(msg, cfgmod.IntensityConfig.parse(None))
    assert np.abs(cloud[:, :3] - xyz).max() == 0.0


def test_pointcloud2_drops_non_finite():
    xyz = np.array([[1, 1, 1], [np.nan, 0, 0], [np.inf, 0, 0], [2, 2, 2]],
                   dtype=np.float32)
    msg = _make_cloud(xyz, np.zeros(4, dtype=np.float32))
    cloud = cloud_from_pointcloud2(msg, cfgmod.IntensityConfig.parse(None))
    assert cloud.shape[0] == 2, cloud.shape


def test_intensity_scaling_and_missing_field():
    xyz = np.array([[1, 0, 0], [2, 0, 0]], dtype=np.float32)
    raw = np.array([1000.0, 60000.0], dtype=np.float32)
    msg = _make_cloud(xyz, raw)
    scaled = cloud_from_pointcloud2(
        msg, cfgmod.IntensityConfig.parse({'field': 'intensity', 'scale': 1e-4}))
    assert abs(scaled[0, 3] - 0.1) < 1e-6
    assert scaled[1, 3] == 1.0, 'must clip to 1.0'

    # a radar cloud with no intensity field falls back to the configured constant
    bare = _Cloud([_Field('x', 0, 7), _Field('y', 4, 7), _Field('z', 8, 7)],
                  np.asarray(xyz, dtype='<f4').tobytes(), 2, 12)
    fallback = cloud_from_pointcloud2(
        bare, cfgmod.IntensityConfig.parse({'field': 'intensity', 'default': 0.3}))
    assert np.allclose(fallback[:, 3], 0.3)


class _Image:
    def __init__(self, array, encoding, step=None):
        self.height, self.width = array.shape[:2]
        self.encoding = encoding
        self.data = array.tobytes()
        self.step = step or array.strides[0]
        self.is_bigendian = False


class _Info:
    def __init__(self, fx, fy, cx, cy, width=8, height=6):
        self.k = [fx, 0, cx, 0, fy, cy, 0, 0, 1]
        self.width, self.height = width, height


def test_depth_reprojection_geometry():
    """A pixel at the principal point at depth d must land d metres ahead."""
    depth = np.zeros((6, 8), dtype=np.uint16)
    depth[3, 4] = 2000                       # 2.0 m at 16UC1 mm scale
    msg = _Image(depth, '16uc1')
    info = _Info(fx=400.0, fy=400.0, cx=4.0, cy=3.0)
    cloud_cfg = cfgmod.CloudConfig.parse(
        {'kind': 'depth_image', 'topic': '/d', 'camera_info_topic': '/i',
         'extrinsic': {}, 'pixel_stride': 1, 'min_depth': 0.1, 'max_depth': 10.0},
        'test')
    cloud = cloud_from_depth_image(msg, info, cloud_cfg)
    assert cloud.shape[0] == 1, cloud.shape
    # optical (0, 0, 2) -> body (x fwd 2, y 0, z 0)
    assert np.allclose(cloud[0, :3], [2.0, 0.0, 0.0], atol=1e-6), cloud[0, :3]


def test_depth_optical_axes():
    """A pixel right-of-centre and above-centre must go to -y and +z in the body."""
    depth = np.zeros((6, 8), dtype=np.uint16)
    depth[1, 6] = 1000                       # right of cx, above cy
    msg = _Image(depth, '16uc1')
    info = _Info(fx=100.0, fy=100.0, cx=4.0, cy=3.0)
    cloud_cfg = cfgmod.CloudConfig.parse(
        {'kind': 'depth_image', 'topic': '/d', 'camera_info_topic': '/i',
         'extrinsic': {}, 'pixel_stride': 1}, 'test')
    point = cloud_from_depth_image(msg, info, cloud_cfg)[0, :3]
    assert point[0] > 0, 'depth must be forward'
    assert point[1] < 0, 'right of centre must be -y in FLU'
    assert point[2] > 0, 'above centre must be +z in FLU'


def test_depth_range_gating():
    depth = np.array([[0, 500, 3000, 30000]], dtype=np.uint16)
    msg = _Image(depth, '16uc1')
    info = _Info(fx=100.0, fy=100.0, cx=2.0, cy=0.0, width=4, height=1)
    cloud_cfg = cfgmod.CloudConfig.parse(
        {'kind': 'depth_image', 'topic': '/d', 'camera_info_topic': '/i',
         'extrinsic': {}, 'pixel_stride': 1, 'min_depth': 1.0, 'max_depth': 10.0},
        'test')
    cloud = cloud_from_depth_image(msg, info, cloud_cfg)
    assert cloud.shape[0] == 1, 'only the 3.0 m pixel is in [1, 10]'


# ------------------------------------------------------------------------- sync

def test_stamp_index_tolerance():
    index = StampIndex([0, 100, 200, 300])
    assert index.nearest(140, 60) == 100
    assert index.nearest(151, 60) == 200
    assert index.nearest(500, 60) is None
    assert index.nearest(500) == 300


def test_frame_table_drops_incomplete():
    fast = StampIndex([i * 100_000_000 for i in range(10)])         # 10 Hz
    slow = StampIndex([i * 100_000_000 for i in range(10) if i % 3])  # gaps
    table = build_frame_table(
        times=fast.stamps, cloud_indices={'a': fast, 'b': slow},
        required={'a': True, 'b': True}, tolerance_ns=20_000_000,
        master='a', drop_incomplete=True)
    assert len(table) == 6, len(table)
    assert all(set(f.cloud_stamps) == {'a', 'b'} for f in table.frames)
    assert sum(table.dropped.values()) == 4


def test_frame_table_optional_agent_survives():
    fast = StampIndex([i * 100_000_000 for i in range(6)])
    sparse = StampIndex([0, 500_000_000])
    table = build_frame_table(
        times=fast.stamps, cloud_indices={'a': fast, 'b': sparse},
        required={'a': True, 'b': False}, tolerance_ns=20_000_000, master='a')
    assert len(table) == 6
    assert sum(1 for f in table.frames if 'b' in f.cloud_stamps) == 2


def test_frame_times_fixed_rate():
    stamps = [int(i * NS / 30) for i in range(300)]        # 30 Hz source
    times = frame_times(stamps, rate_hz=10.0)
    assert abs(len(times) - 100) <= 1, len(times)
    assert times[1] - times[0] == NS // 10


def test_pose_track_interpolation():
    track = PoseTrack()
    track.add(0, make_transform(0, 0, 0, yaw=0.0))
    track.add(NS, make_transform(10, 0, 0, yaw=90.0))
    track.finish()
    T, _ = track.lookup(NS // 2, mode='linear')
    assert abs(T[0, 3] - 5.0) < 1e-9
    assert abs(np.degrees(np.arctan2(T[1, 0], T[0, 0])) - 45.0) < 1e-6
    T_near, _ = track.lookup(NS // 2 + 1, mode='nearest')
    assert abs(T_near[0, 3] - 10.0) < 1e-9


def test_pose_track_refuses_to_extrapolate():
    track = PoseTrack()
    track.add(0, np.identity(4))
    track.add(NS, np.identity(4))
    track.finish()
    assert track.lookup(5 * NS, max_gap_ns=NS // 5) is None


def test_pose_track_sorts_unordered_input():
    track = PoseTrack()
    for t in (2, 0, 1):
        track.add(t * NS, make_transform(t, 0, 0))
    track.finish()
    assert track.stamps == [0, NS, 2 * NS]
    assert track.lookup(NS)[0][0, 3] == 1.0


# ----------------------------------------------------------------------- labels

def _project_world_objects(vehicles, lidar_pose):
    """Replica of opencood.utils.box_utils.project_world_objects (centres only)."""
    out = {}
    for object_id, content in vehicles.items():
        pose = [content['location'][i] + content['center'][i] for i in range(3)] + \
            list(content['angle'])
        object_to_lidar = np.linalg.inv(x_to_world(lidar_pose)) @ x_to_world(pose)
        out[object_id] = object_to_lidar[:3, 3]
    return out


def test_agent_box_lands_where_geometry_says():
    """An agent 5 m ahead of the ego must appear at x=+5 in the ego lidar frame."""
    ego = make_transform(10.0, -3.0, 0.0, yaw=30.0)
    other = ego @ make_transform(5.0, 0.0, 0.0)
    vehicles = vehicles_for_viewer(
        {'ego': ego, 'other': other},
        {'ego': {'object_id': 1, 'extent': [0.4, 0.3, 0.3], 'center': [0, 0, 0]},
         'other': {'object_id': 2, 'extent': [0.4, 0.3, 0.3], 'center': [0, 0, 0]}},
        viewer='ego')
    assert set(vehicles) == {2}, 'the viewer must not label itself'
    centres = _project_world_objects(vehicles, matrix_to_opencood_pose(ego))
    assert np.allclose(centres[2], [5.0, 0.0, 0.0], atol=1e-9), centres[2]


def test_box_centre_offset_is_rotated_into_world():
    """A body-frame centre offset must follow the robot's heading, not the world's."""
    pose = make_transform(0.0, 0.0, 0.0, yaw=90.0)
    box = agent_box(pose, extent=[0.4, 0.3, 0.25], center=[1.0, 0.0, 0.2])
    assert np.allclose(box['location'], [0.0, 1.0, 0.2], atol=1e-9), box['location']
    assert box['center'] == [0.0, 0.0, 0.0]


# ----------------------------------------------------------------------- config

def test_config_rejects_null_alignment():
    raised = False
    try:
        cfgmod._transform_from(None, 'agent[x].align')
    except cfgmod.ConfigError as error:
        raised = 'null' in str(error)
    assert raised, 'a null transform must be refused, never treated as identity'


def test_config_rejects_positive_rsu_id():
    raised = False
    try:
        cfgmod.AgentConfig.parse({
            'name': 'infra', 'id': 3, 'role': 'rsu',
            'pose': {'source': 'static', 'world_pose': {}},
            'cloud': {'kind': 'pointcloud2', 'topic': '/p', 'extrinsic': {}}})
    except cfgmod.ConfigError as error:
        raised = 'negative' in str(error)
    assert raised


def test_cdr_stamp_extraction():
    import struct
    payload = struct.pack('<BBHiI', 0, 1, 0, 1787899802, 217921000) + b'\x00' * 32
    assert stamp_from_cdr(payload) == 1787899802 * NS + 217921000
    assert stamp_from_cdr(b'\x00\x01') is None


# ------------------------------------------------------------------ end to end

def _write_synthetic_bag(path, n_frames=12, two_clock_error_ns=0, transit_ns=None,
                         ntp_topic=None, ntp_offset_unit='s', ego_point_times=False,
                         two_speed=0.0, optical_pose_X=None):
    """Three agents on a 10 Hz grid with exactly known geometry.

    ego (agent 1) drives along +x; agent 2 sits 6 m ahead of the ego's start;
    the RSU is static at (3, 4). Each carries one cloud so the converter has to
    exercise both the PointCloud2 and depth paths.

    The optional arguments inject the failure modes a single-host bag cannot
    exhibit, with known ground truth:

    ``two_clock_error_ns``  agent 2's clock reads this far off, so its
        ``header.stamp`` values are wrong by exactly that much while ``log_time``
        stays on the recorder's clock — the situation clock reconciliation exists
        to detect.
    ``transit_ns``          per-agent network delay added to ``log_time`` only.
        Its *asymmetry* between hosts is the error term of the delivery-floor
        estimate, so the tests can check that bound is honest.
    ``ntp_topic``           writes an NtpStatus-shaped topic for agent 2 carrying
        its true offset, in the host-minus-reference convention.
    ``ntp_offset_unit``     whether that topic reports seconds or milliseconds —
        the message never says which, and the two differ by a factor of 1000.
    ``ego_point_times``     adds a per-point ``t`` field to the ego's cloud so the
        sweep spans 90 ms instead of being instantaneous, which is what deskew
        needs and what a real spinning LiDAR always has.
    ``two_speed``           moves agent 2 along +y at this speed, so an error in
        *when* its pose is evaluated becomes a visible error in *where* it is.
    ``optical_pose_X``      also publish ``/ego/pose`` as a PoseStamped of the
        CAMERA OPTICAL frame, i.e. the odometry pose composed with this
        ``T_child_cam``. That is the shape an offline pipeline republishes a
        corrected trajectory in, and it has no twist and no child_frame_id.
    """
    transit_ns = transit_ns or {}
    from mcap_ros2.writer import Writer

    time_def = 'int32 sec\nuint32 nanosec'
    header_def = ('builtin_interfaces/Time stamp\nstring frame_id\n'
                  '================================================================================\n'
                  'MSG: builtin_interfaces/Time\n' + time_def)
    cloud_def = (
        'std_msgs/Header header\nuint32 height\nuint32 width\n'
        'sensor_msgs/PointField[] fields\nbool is_bigendian\nuint32 point_step\n'
        'uint32 row_step\nuint8[] data\nbool is_dense\n'
        '================================================================================\n'
        'MSG: std_msgs/Header\n' + header_def.split('\n=====')[0] +
        '\n================================================================================\n'
        'MSG: builtin_interfaces/Time\n' + time_def +
        '\n================================================================================\n'
        'MSG: sensor_msgs/PointField\nstring name\nuint32 offset\nuint8 datatype\n'
        'uint32 count')
    odom_def = (
        'std_msgs/Header header\nstring child_frame_id\n'
        'geometry_msgs/PoseWithCovariance pose\n'
        'geometry_msgs/TwistWithCovariance twist\n'
        '================================================================================\n'
        'MSG: std_msgs/Header\nbuiltin_interfaces/Time stamp\nstring frame_id\n'
        '================================================================================\n'
        'MSG: builtin_interfaces/Time\n' + time_def +
        '\n================================================================================\n'
        'MSG: geometry_msgs/PoseWithCovariance\ngeometry_msgs/Pose pose\n'
        'float64[36] covariance\n'
        '================================================================================\n'
        'MSG: geometry_msgs/Pose\ngeometry_msgs/Point position\n'
        'geometry_msgs/Quaternion orientation\n'
        '================================================================================\n'
        'MSG: geometry_msgs/Point\nfloat64 x\nfloat64 y\nfloat64 z\n'
        '================================================================================\n'
        'MSG: geometry_msgs/Quaternion\nfloat64 x\nfloat64 y\nfloat64 z\nfloat64 w\n'
        '================================================================================\n'
        'MSG: geometry_msgs/TwistWithCovariance\ngeometry_msgs/Twist twist\n'
        'float64[36] covariance\n'
        '================================================================================\n'
        'MSG: geometry_msgs/Twist\ngeometry_msgs/Vector3 linear\n'
        'geometry_msgs/Vector3 angular\n'
        '================================================================================\n'
        'MSG: geometry_msgs/Vector3\nfloat64 x\nfloat64 y\nfloat64 z')
    image_def = (
        'std_msgs/Header header\nuint32 height\nuint32 width\nstring encoding\n'
        'uint8 is_bigendian\nuint32 step\nuint8[] data\n'
        '================================================================================\n'
        'MSG: std_msgs/Header\nbuiltin_interfaces/Time stamp\nstring frame_id\n'
        '================================================================================\n'
        'MSG: builtin_interfaces/Time\n' + time_def)
    info_def = (
        'std_msgs/Header header\nuint32 height\nuint32 width\n'
        'string distortion_model\nfloat64[] d\nfloat64[9] k\nfloat64[9] r\n'
        'float64[12] p\nuint32 binning_x\nuint32 binning_y\n'
        'sensor_msgs/RegionOfInterest roi\n'
        '================================================================================\n'
        'MSG: std_msgs/Header\nbuiltin_interfaces/Time stamp\nstring frame_id\n'
        '================================================================================\n'
        'MSG: builtin_interfaces/Time\n' + time_def +
        '\n================================================================================\n'
        'MSG: sensor_msgs/RegionOfInterest\nuint32 x_offset\nuint32 y_offset\n'
        'uint32 height\nuint32 width\nbool do_rectify')

    def stamp(t_ns):
        return {'sec': t_ns // NS, 'nanosec': t_ns % NS}

    def header(t_ns, frame):
        return {'stamp': stamp(t_ns), 'frame_id': frame}

    def cloud_msg(t_ns, frame, points, point_times_ns=None):
        fields = [{'name': n, 'offset': 4 * i, 'datatype': 7, 'count': 1}
                  for i, n in enumerate(('x', 'y', 'z', 'intensity'))]
        if point_times_ns is None:
            data = np.zeros((len(points), 4), dtype='<f4')
            data[:, :3] = points
            data[:, 3] = 0.5
            step = 16
            payload = data.tobytes()
        else:
            record = np.zeros(len(points), dtype=np.dtype(
                [('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('intensity', '<f4'),
                 ('t', '<u4')]))
            record['x'], record['y'], record['z'] = (points[:, 0], points[:, 1],
                                                     points[:, 2])
            record['intensity'] = 0.5
            record['t'] = np.asarray(point_times_ns, dtype=np.uint32)
            fields.append({'name': 't', 'offset': 16, 'datatype': 6, 'count': 1})
            step = 20
            payload = record.tobytes()
        return {'header': header(t_ns, frame), 'height': 1, 'width': len(points),
                'fields': fields, 'is_bigendian': False, 'point_step': step,
                'row_step': step * len(points), 'data': payload,
                'is_dense': True}

    def odom_msg(t_ns, frame, child, position, speed):
        return {'header': header(t_ns, frame), 'child_frame_id': child,
                'pose': {'pose': {'position': {'x': position[0], 'y': position[1],
                                               'z': position[2]},
                                  'orientation': {'x': 0.0, 'y': 0.0, 'z': 0.0,
                                                  'w': 1.0}},
                         'covariance': [0.0] * 36},
                'twist': {'twist': {'linear': {'x': speed, 'y': 0.0, 'z': 0.0},
                                    'angular': {'x': 0.0, 'y': 0.0, 'z': 0.0}},
                          'covariance': [0.0] * 36}}

    with open(path, 'wb') as handle:
        writer = Writer(handle)
        cloud_schema = writer.register_msgdef('sensor_msgs/msg/PointCloud2', cloud_def)
        odom_schema = writer.register_msgdef('nav_msgs/msg/Odometry', odom_def)
        image_schema = writer.register_msgdef('sensor_msgs/msg/Image', image_def)
        info_schema = writer.register_msgdef('sensor_msgs/msg/CameraInfo', info_def)
        pose_schema = None
        if optical_pose_X is not None:
            pose_def = (
                'std_msgs/Header header\ngeometry_msgs/Pose pose\n'
                '================================================================================'
                '\nMSG: std_msgs/Header\nbuiltin_interfaces/Time stamp\nstring frame_id\n'
                '================================================================================'
                '\nMSG: builtin_interfaces/Time\n' + time_def +
                '\n================================================================================'
                '\nMSG: geometry_msgs/Pose\ngeometry_msgs/Point position\n'
                'geometry_msgs/Quaternion orientation\n'
                '================================================================================'
                '\nMSG: geometry_msgs/Point\nfloat64 x\nfloat64 y\nfloat64 z\n'
                '================================================================================'
                '\nMSG: geometry_msgs/Quaternion\nfloat64 x\nfloat64 y\nfloat64 z\n'
                'float64 w')
            pose_schema = writer.register_msgdef('geometry_msgs/msg/PoseStamped',
                                                 pose_def)
        ntp_schema = None
        if ntp_topic:
            ntp_def = ('std_msgs/Header header\nfloat64 offset\nfloat64 jitter\n'
                       'bool synchronized\n'
                       '================================================================================'
                       '\nMSG: std_msgs/Header\nbuiltin_interfaces/Time stamp\n'
                       'string frame_id\n'
                       '================================================================================'
                       '\nMSG: builtin_interfaces/Time\n' + time_def)
            ntp_schema = writer.register_msgdef('ntp_monitor_msgs/msg/NtpStatus', ntp_def)

        t0 = 1_700_000_000 * NS
        # A 4x4 depth image whose single valid pixel sits at the principal point.
        depth = np.zeros((4, 4), dtype='<u2')
        depth[2, 2] = 3000
        ego_transit = int(transit_ns.get('ego', 0))
        two_transit = int(transit_ns.get('two', 0))
        ego_points = np.array([[1.0, 0.0, 0.0], [2.0, 1.0, -0.5], [3.0, -1.0, 0.2]])
        # A 90 ms sweep: first point at the stamp, last 90 ms later.
        ego_times = [0, 45_000_000, 90_000_000] if ego_point_times else None
        for i in range(n_frames):
            t = t0 + i * (NS // 10)
            writer.write_message('/ego/points', cloud_schema,
                                 cloud_msg(t, 'ego_lidar', ego_points, ego_times),
                                 log_time=t + ego_transit, publish_time=t)
            writer.write_message('/ego/odom', odom_schema,
                                 odom_msg(t, 'ego_odom', 'ego_base',
                                          (0.5 * i, 0.0, 0.0), 5.0),
                                 log_time=t + ego_transit, publish_time=t)
            if pose_schema is not None:
                T_oc = np.identity(4); T_oc[0, 3] = 0.5 * i
                T_cam = T_oc @ optical_pose_X
                q = matrix_to_quat(T_cam[:3, :3])
                writer.write_message(
                    '/ego/pose', pose_schema,
                    {'header': header(t, 'map'),
                     'pose': {'position': {'x': float(T_cam[0, 3]),
                                           'y': float(T_cam[1, 3]),
                                           'z': float(T_cam[2, 3])},
                              'orientation': {'x': float(q[0]), 'y': float(q[1]),
                                              'z': float(q[2]), 'w': float(q[3])}}},
                    log_time=t + ego_transit, publish_time=t)
            # agent 2 publishes slightly off-grid: the synchroniser must match it
            t2_true = t + 15_000_000
            t2 = t2_true + int(two_clock_error_ns)     # what agent 2's clock says
            if ntp_schema is not None:
                writer.write_message(
                    ntp_topic, ntp_schema,
                    {'header': header(t2, 'ntp'),
                     'offset': (two_clock_error_ns / 1e9 if ntp_offset_unit == 's'
                                else two_clock_error_ns / 1e6),  # host minus reference
                     'jitter': 0.0002, 'synchronized': True},
                    log_time=t2_true + two_transit, publish_time=t2_true)
            writer.write_message('/two/depth', image_schema,
                                 {'header': header(t2, 'two_depth_optical'),
                                  'height': 4, 'width': 4, 'encoding': '16UC1',
                                  'is_bigendian': 0, 'step': 8,
                                  'data': depth.tobytes()},
                                 log_time=t2_true + two_transit, publish_time=t2_true)
            writer.write_message('/two/depth/camera_info', info_schema,
                                 {'header': header(t2, 'two_depth_optical'),
                                  'height': 4, 'width': 4,
                                  'distortion_model': 'plumb_bob', 'd': [],
                                  'k': [200.0, 0, 2.0, 0, 200.0, 2.0, 0, 0, 1.0],
                                  'r': [1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0],
                                  'p': [200.0, 0, 2.0, 0, 0, 200.0, 2.0, 0,
                                        0, 0, 1.0, 0],
                                  'binning_x': 0, 'binning_y': 0,
                                  'roi': {'x_offset': 0, 'y_offset': 0, 'height': 0,
                                          'width': 0, 'do_rectify': False}},
                                 log_time=t2_true + two_transit, publish_time=t2_true)
            writer.write_message('/two/odom', odom_schema,
                                 odom_msg(t2, 'two_odom', 'two_base',
                                          (6.0, two_speed * (t2_true - t0) / NS, 0.0),
                                          two_speed),
                                 log_time=t2_true + two_transit, publish_time=t2_true)
            writer.write_message('/infra/radar', cloud_schema,
                                 cloud_msg(t, 'infra_radar',
                                           np.array([[4.0, 0.5, 0.1]])),
                                 log_time=t + int(transit_ns.get('infra', 0)),
                                 publish_time=t)
        writer.finish()


def _synthetic_config(bag_path, out_root):
    return {
        'bag': bag_path,
        'output': {'root': out_root, 'split': 'test',
                   'scenario_name': 'synthetic', 'frames_per_scenario': 0},
        'time': {'stamp_source': 'header', 'match_tolerance_ms': 60},
        'agents': [
            {'name': 'ego', 'id': 1, 'role': 'cav',
             'pose': {'source': 'odometry', 'topic': '/ego/odom',
                      'align': {'x': 0, 'y': 0, 'z': 0}},
             'cloud': {'kind': 'pointcloud2', 'topic': '/ego/points',
                       'extrinsic': {'x': 0, 'y': 0, 'z': 0.5},
                       'intensity': {'field': 'intensity', 'scale': 1.0}},
             'object': {'emit': True, 'extent': [0.4, 0.3, 0.25],
                        'center': [0, 0, 0.25], 'object_id': 101}},
            {'name': 'two', 'id': 2, 'role': 'cav',
             'pose': {'source': 'odometry', 'topic': '/two/odom',
                      'align': {'x': 0, 'y': 0, 'z': 0}},
             'cloud': {'kind': 'depth_image', 'topic': '/two/depth',
                       'camera_info_topic': '/two/depth/camera_info',
                       'extrinsic': {'x': 0.1, 'y': 0, 'z': 0.3},
                       'depth_scale': 0.001, 'min_depth': 0.1, 'max_depth': 10.0,
                       'pixel_stride': 1},
             'object': {'emit': True, 'extent': [0.35, 0.3, 0.25],
                        'center': [0, 0, 0.25], 'object_id': 102}},
            {'name': 'infra', 'id': -1, 'role': 'rsu',
             'pose': {'source': 'static',
                      'world_pose': {'x': 3.0, 'y': 4.0, 'z': 2.0, 'yaw': 180.0}},
             'cloud': {'kind': 'pointcloud2', 'topic': '/infra/radar',
                       'extrinsic': {'x': 0, 'y': 0, 'z': 0},
                       'intensity': {'field': 'none', 'default': 0.4}},
             'object': {'emit': False}},
        ],
    }


def test_end_to_end_conversion():
    tmp = tempfile.mkdtemp(prefix='r2o_')
    try:
        bag = os.path.join(tmp, 'synthetic.mcap')
        _write_synthetic_bag(bag, n_frames=12)

        config_path = os.path.join(tmp, 'cfg.yaml')
        out_root = os.path.join(tmp, 'out')
        with open(config_path, 'w') as handle:
            yaml.safe_dump(_synthetic_config(bag, out_root), handle)

        cfg = cfgmod.load_config(config_path)
        report = convert(cfg, overwrite=True)

        assert report.frames_written == 12, report.frames_written
        scenario = os.path.join(out_root, 'test', 'synthetic')
        assert sorted(os.listdir(scenario)) == ['-1', '1', '2'], os.listdir(scenario)

        for agent in ('1', '2', '-1'):
            files = os.listdir(os.path.join(scenario, agent))
            assert sum(f.endswith('.pcd') for f in files) == 12, (agent, files)
            assert sum(f.endswith('.yaml') for f in files) == 12, (agent, files)

        # --- ego frame 000004: pose, cloud and GT must all agree with the source
        with open(os.path.join(scenario, '1', '000004.yaml')) as handle:
            ego_params = yaml.safe_load(handle)
        # ego drives 0.5 m per frame, lidar sits 0.5 m up
        assert np.allclose(ego_params['lidar_pose'][:3], [2.0, 0.0, 0.5], atol=1e-6), \
            ego_params['lidar_pose']
        assert abs(ego_params['ego_speed'] - 18.0) < 1e-6, 'm/s must become km/h'

        # agent 2 is static at x=6; from the ego's lidar it must sit 4 m ahead
        centres = _project_world_objects(ego_params['vehicles'],
                                         ego_params['lidar_pose'])
        assert set(centres) == {102}, centres
        assert np.allclose(centres[102], [4.0, 0.0, -0.25], atol=1e-6), centres[102]

        # the RSU's yaml must carry both robots as objects
        with open(os.path.join(scenario, '-1', '000004.yaml')) as handle:
            rsu_params = yaml.safe_load(handle)
        assert set(rsu_params['vehicles']) == {101, 102}

        cloud = read_pcd(os.path.join(scenario, '1', '000004.pcd'))
        assert cloud.shape == (3, 4), cloud.shape
        assert np.allclose(cloud[0, :3], [1.0, 0.0, 0.0], atol=1e-6)
        assert abs(cloud[0, 3] - 0.5) <= 1.0 / 255

        # the depth agent's single valid pixel: 3 m ahead of its camera
        depth_cloud = read_pcd(os.path.join(scenario, '2', '000004.pcd'))
        assert depth_cloud.shape == (1, 4), depth_cloud.shape
        assert np.allclose(depth_cloud[0, :3], [3.0, 0.0, 0.0], atol=1e-4), depth_cloud

        # the radar RSU used the intensity fallback, not the missing field
        radar = read_pcd(os.path.join(scenario, '-1', '000004.pcd'))
        assert abs(radar[0, 3] - 0.4) <= 1.0 / 255

        # sync accounting: agent two is 15 ms off the grid, inside tolerance
        assert report.sync['two']['max_offset_ms'] <= 16.0, report.sync['two']
        assert report.sync['ego']['max_offset_ms'] == 0.0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_end_to_end_rejects_missing_topic():
    tmp = tempfile.mkdtemp(prefix='r2o_')
    try:
        bag = os.path.join(tmp, 'synthetic.mcap')
        _write_synthetic_bag(bag, n_frames=4)
        raw = _synthetic_config(bag, os.path.join(tmp, 'out'))
        raw['agents'][0]['cloud']['topic'] = '/ego/does_not_exist'
        config_path = os.path.join(tmp, 'cfg.yaml')
        with open(config_path, 'w') as handle:
            yaml.safe_dump(raw, handle)
        cfg = cfgmod.load_config(config_path)
        raised = False
        try:
            convert(cfg, overwrite=True)
        except Exception as error:                      # noqa: BLE001
            raised = 'not in the bag' in str(error)
        assert raised, 'a typo in a topic name must fail before anything is written'
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_ground_lift_preserves_world_geometry():
    """Lifting the sensor must move points and pose together, leaving GT alignment."""
    tmp = tempfile.mkdtemp(prefix='r2o_')
    try:
        bag = os.path.join(tmp, 'synthetic.mcap')
        _write_synthetic_bag(bag, n_frames=6)
        raw = _synthetic_config(bag, os.path.join(tmp, 'out'))
        raw['agents'][0]['cloud']['ground_lift'] = 1.4
        config_path = os.path.join(tmp, 'cfg.yaml')
        with open(config_path, 'w') as handle:
            yaml.safe_dump(raw, handle)
        convert(cfgmod.load_config(config_path), overwrite=True)

        scenario = os.path.join(tmp, 'out', 'test', 'synthetic')
        with open(os.path.join(scenario, '1', '000002.yaml')) as handle:
            params = yaml.safe_load(handle)
        cloud = read_pcd(os.path.join(scenario, '1', '000002.pcd'))
        # pose up by 1.4, points down by 1.4 => the world position is unchanged
        assert abs(params['lidar_pose'][2] - 1.9) < 1e-6, params['lidar_pose']
        assert abs(cloud[0, 2] - (-1.4)) < 1e-5, cloud[0]
        world = transform_points(cloud[:1, :3], x_to_world(params['lidar_pose']))
        assert abs(world[0, 2] - 0.5) < 1e-4, world
    finally:
        shutil.rmtree(tmp, ignore_errors=True)



# ----------------------------------------------------------------------- clocks
# A multi-host bag has a failure mode a single-host bag cannot: a constant clock
# offset biases every stream on one machine in the same direction, which is
# indistinguishable from that agent being uniformly late — and this study's own
# results say a uniform latency is the most damaging impairment there is.

def test_offset_track_holds_flat_outside_its_range():
    """An NTP offset is a controlled variable, not a trajectory. Extrapolating a
    control loop's error invents a correction nobody measured."""
    track = clockmod.OffsetTrack([0, 10 * NS], [1_000_000, 3_000_000]).finish()
    assert track.at(5 * NS) == 2_000_000, track.at(5 * NS)
    assert track.at(-99 * NS) == 1_000_000
    assert track.at(999 * NS) == 3_000_000


def test_offset_unit_is_inferred_conservatively():
    """Reading milliseconds as seconds inflates a correction a thousandfold, so a
    weak guess must be reported as weak rather than acted on."""
    assert clockmod.infer_unit([0.001, 0.002])[0] == 1.0
    assert clockmod.infer_unit([1.5, 2.5])[0] == 1e-3
    assert clockmod.infer_unit([0.0, 0.0])[1] == 'all_zero'
    assert clockmod.infer_unit([1e7])[1] == 'low'


def test_delivery_floor_recovers_a_known_offset():
    """min(log - stamp) is transit + offset, so differencing two hosts cancels the
    recorder's clock and leaves the offset."""
    clocks = clockmod.HostClocks('ego')
    for value in (2_000_000, 4_000_000, 10_000_000):
        clocks.delivery.setdefault('ego', clockmod.DeliveryStats('ego')).add(value)
    for value in (32_000_000, 34_000_000, 50_000_000):
        clocks.delivery.setdefault('two', clockmod.DeliveryStats('two')).add(value)
    assert clocks.delivery_floor_correction_ns('two') == 30_000_000
    correction, source = clocks.correction_ns('two', 0)
    assert (correction, source) == (30_000_000, 'delivery_floor')
    assert clocks.correction_ns('ego', 0) == (0, 'reference')


def test_ntp_sign_is_chosen_from_the_delivery_floor():
    """ntpq and chrony disagree about the sign of 'offset'. Guessing wrong doubles
    the error instead of removing it, so it is decided by the data."""
    clocks = clockmod.HostClocks('ego')
    for value in (2_000_000, 4_000_000):
        clocks.delivery.setdefault('ego', clockmod.DeliveryStats('ego')).add(value)
    for value in (32_000_000, 34_000_000):
        clocks.delivery.setdefault('two', clockmod.DeliveryStats('two')).add(value)
    clocks.ntp['two'] = clockmod.OffsetTrack([0, NS], [-30_000_000] * 2).finish()
    sign, detail = clockmod.choose_sign(clocks)
    assert sign == 1.0, detail
    assert detail['verdict'] == 'decided', detail
    assert clocks.correction_ns('two', 0) == (30_000_000, 'ntp')
    assert clocks.cross_check('two', 20_000_000)[0] == 'agree'


def test_residual_is_the_offsets_spread_not_their_magnitude():
    """A perfectly constant 30 ms offset leaves nothing behind once corrected;
    calling it a 30 ms residual would be exactly backwards."""
    clocks = clockmod.HostClocks('ego')
    clocks.ntp['two'] = clockmod.OffsetTrack([0, NS], [30_000_000] * 2).finish()
    residual, source = clocks.residual_ns('two')
    assert residual == 0.0 and source == 'ntp_spread', (residual, source)
    clocks.ntp['three'] = clockmod.OffsetTrack(
        [0, NS, 2 * NS], [0, 4_000_000, 8_000_000]).finish()
    assert clocks.residual_ns('three')[0] > 1_000_000


def test_clock_disagreement_is_reported_not_averaged():
    clocks = clockmod.HostClocks('ego')
    for value in (2_000_000,):
        clocks.delivery.setdefault('ego', clockmod.DeliveryStats('ego')).add(value)
    for value in (32_000_000,):
        clocks.delivery.setdefault('two', clockmod.DeliveryStats('two')).add(value)
    clocks.ntp['two'] = clockmod.OffsetTrack([0, NS], [-300_000_000] * 2).finish()
    verdict, detail = clocks.cross_check('two', 20_000_000)
    assert verdict == 'DISAGREE', detail
    assert abs(detail['difference_ms'] - 270.0) < 1e-6, detail


# ------------------------------------------------------------------- tightness

def test_stamp_index_keeps_corrected_and_raw_stamps_apart():
    """Matching compares corrected stamps; the write pass must ask the bag for the
    raw ones. Conflating them silently produces either uncorrected matching or a
    write pass that selects nothing."""
    index = StampIndex([100, 200, 300], raw=[90, 190, 290])
    assert index.nearest_pair(205) == (200, 190)
    assert index.nearest(205) == 200
    assert index.nearest_pair(205, tolerance_ns=1) is None


def test_half_period_is_the_structural_floor():
    """No matching strategy beats half a publication period, so a tolerance below
    it rejects frames for a reason no processing can fix."""
    stamps = [i * (NS // 10) for i in range(11)]        # 10 Hz
    assert abs(StampIndex(stamps).half_period_ns() - 0.05 * NS) < 1e3


def test_tightness_curve_is_monotone_and_counts_the_clock_residual():
    fast = StampIndex([i * (NS // 10) for i in range(5)])
    table = build_frame_table([i * (NS // 10) for i in range(5)],
                              {'a': fast}, {'a': True}, NS // 20, 'a')
    clean = tightness_curve(table, ['a'], grid_ms=(1, 10))
    assert clean['curve'][0]['frames'] == 5, clean
    # 5 ms of un-correctable clock uncertainty pushes every frame past a 1 ms budget
    noisy = tightness_curve(table, ['a'], {'a': 5.0}, grid_ms=(1, 10))
    assert noisy['curve'][0]['frames'] == 0 and noisy['curve'][1]['frames'] == 5, noisy
    assert all(noisy['curve'][i]['frames'] <= noisy['curve'][i + 1]['frames']
               for i in range(len(noisy['curve']) - 1))


# --------------------------------------------------------------------- deskew

def _straight_line_track(speed_mps=5.0, samples=4):
    track = PoseTrack()
    for k in range(samples):
        matrix = np.identity(4)
        matrix[0, 3] = speed_mps * k * 0.1
        track.add(int(k * NS // 10), matrix, speed_mps)
    return track.finish()


def test_deskew_moves_points_by_the_distance_travelled():
    """A 10 Hz sweep observes over ~100 ms — the same order as the whole
    inter-agent budget — and the smear is azimuth-dependent, so it is not noise."""
    track = _straight_line_track()
    cloud = np.zeros((3, 4), dtype=np.float32)
    offsets = np.array([0, 45_000_000, 90_000_000])
    out, info = deskew_cloud(cloud, offsets, stamp_ns=NS // 10, t_ref_ns=NS // 10,
                             track=track, sensor_from_base=np.identity(4),
                             max_gap_ns=NS // 5)
    assert info['applied'], info
    # 5 m/s. The last point was observed 90 ms after the frame instant, by which
    # time the sensor had travelled 0.45 m; expressed in the sensor frame at the
    # frame instant, that measurement therefore sits 0.45 m ahead. Leaving it at 0
    # would place three returns from the same physical surface half a metre apart.
    #
    # Points are bucketed in time and one transform is applied per bucket, so each
    # value carries at most half a bucket of travel: 90 ms / 64 buckets / 2 at
    # 5 m/s is 3.5 mm. Asserting against that bound rather than a round tolerance
    # keeps the test sensitive to a real regression.
    bucket_residual = 0.5 * (0.090 / 64) * 5.0
    for got, want in zip(out[:, 0], (0.0, 0.225, 0.45)):
        assert abs(got - want) <= bucket_residual + 1e-6, (out[:, 0], bucket_residual)


def test_deskew_targets_the_frame_time_not_the_message_stamp():
    """Deskewing to the frame time absorbs the agent's selection skew as well as
    the sweep's own duration."""
    track = _straight_line_track()
    cloud = np.zeros((1, 4), dtype=np.float32)
    out, info = deskew_cloud(cloud, np.array([0]), stamp_ns=NS // 10,
                             t_ref_ns=NS // 10 + 20_000_000, track=track,
                             sensor_from_base=np.identity(4), max_gap_ns=NS // 5)
    # Observed 20 ms before the frame instant, i.e. 0.1 m earlier along the path.
    assert info['applied'], info
    assert abs(out[0, 0] + 0.1) < 1e-3, out[:, 0]


def test_deskew_declines_rather_than_half_correcting():
    """A partially corrected cloud is worse than an uncorrected one: it is no
    longer internally consistent."""
    track = _straight_line_track()
    cloud = np.ones((3, 4), dtype=np.float32)
    out, info = deskew_cloud(cloud, np.array([0, 45_000_000, 90_000_000]),
                             stamp_ns=NS // 10, t_ref_ns=99 * NS, track=track,
                             sensor_from_base=np.identity(4), max_gap_ns=NS // 50)
    assert not info['applied'] and np.allclose(out, cloud), info
    out2, info2 = deskew_cloud(cloud, None, NS // 10, NS // 10, track,
                               np.identity(4), NS)
    assert not info2['applied'] and np.allclose(out2, cloud), info2


def test_deskew_is_free_of_the_alignment_transform():
    """Only relative motion matters, so a wrong operator-supplied ``align`` — the
    one input the converter cannot check — cannot corrupt the correction."""
    track = _straight_line_track()
    align = make_transform(x=17.0, y=-3.0, yaw=37.0, degrees=True)
    shifted = PoseTrack()
    for stamp, matrix in zip(track.stamps, track.transforms):
        shifted.add(stamp, align @ matrix, 0.0)
    shifted.finish()
    cloud = np.zeros((3, 4), dtype=np.float32)
    offsets = np.array([0, 45_000_000, 90_000_000])
    a, _ = deskew_cloud(cloud, offsets, NS // 10, NS // 10, track,
                        np.identity(4), NS // 5)
    b, _ = deskew_cloud(cloud, offsets, NS // 10, NS // 10, shifted,
                        np.identity(4), NS // 5)
    assert np.allclose(a, b, atol=1e-5), (a, b)


def test_config_refuses_deskew_without_a_time_field():
    base = {'kind': 'pointcloud2', 'topic': '/a', 'extrinsic': {'x': 0}}
    try:
        cfgmod.CloudConfig.parse(dict(base, deskew=True), 'cloud')
    except cfgmod.ConfigError as exc:
        assert 'point_time_field' in str(exc), exc
    else:
        raise AssertionError('deskew without point_time_field must be refused')
    ok = cfgmod.CloudConfig.parse(dict(base, deskew=True, point_time_field='t'), 'cloud')
    assert ok.deskew and ok.point_time_field == 't'


# --------------------------------------------------- end to end, multi-host

def _multi_host_config(bag_path, out_root, clock_enabled, deskew=False):
    """The synthetic config with agents on *named hosts* and clock reconciliation."""
    cfg = _synthetic_config(bag_path, out_root)
    for agent in cfg['agents']:
        agent['host'] = {'ego': 'host_a', 'two': 'host_b', 'infra': 'host_a'}[agent['name']]
    cfg['clock'] = {
        'enabled': clock_enabled,
        'reference_host': 'host_a',
        'ntp_topics': {'host_b': '/two/ntp'},
        'cross_check_tolerance_ms': 20.0,
    }
    if deskew:
        ego = next(a for a in cfg['agents'] if a['name'] == 'ego')
        ego['cloud']['deskew'] = True
        ego['cloud']['point_time_field'] = 't'
    return cfg


def _convert_multi_host(tmp, clock_enabled, deskew=False, offset_unit=None,
                        **bag_kwargs):
    bag = os.path.join(tmp, 'mh.mcap')
    if not os.path.exists(bag):
        _write_synthetic_bag(bag, n_frames=12, **bag_kwargs)
    out_root = os.path.join(tmp, 'out_clock' if clock_enabled else 'out_raw')
    config_path = os.path.join(tmp, 'cfg_%s.yaml' % ('clock' if clock_enabled else 'raw'))
    config = _multi_host_config(bag, out_root, clock_enabled, deskew)
    if offset_unit:
        config['clock']['offset_unit'] = offset_unit
    with open(config_path, 'w') as handle:
        yaml.safe_dump(config, handle)
    cfg = cfgmod.load_config(config_path)
    report = convert(cfg, overwrite=True)
    scenario = os.path.join(out_root, 'test', 'synthetic')
    return report, scenario


def test_end_to_end_recovers_a_multi_host_clock_offset():
    """Agent 2's clock is 50 ms slow. Both independent estimators must find it.

    This is the failure a single-host converter cannot have and a multi-robot one
    always can: nothing in the data looks wrong, the frames all match inside
    tolerance, and one agent's every message is 50 ms stale.
    """
    tmp = tempfile.mkdtemp(prefix='r2o_clock_')
    try:
        report, scenario = _convert_multi_host(
            tmp, clock_enabled=True, two_clock_error_ns=-50_000_000,
            transit_ns={'ego': 2_000_000, 'two': 8_000_000, 'infra': 2_000_000},
            ntp_topic='/two/ntp')

        host_b = report.clocks['host_b']
        assert host_b['correction_source'] == 'ntp', host_b
        assert abs(host_b['correction_ms'] - 50.0) < 0.5, host_b
        assert report.clocks['_meta']['sign'] == 1.0, report.clocks['_meta']

        # The delivery floor sees the same offset, up to the transit asymmetry it
        # cannot separate from it: transit was 8 ms on host_b and 2 ms on host_a,
        # so the estimate must land 6 ms high and the cross-check must accept it.
        detail = host_b['cross_check_detail']
        assert host_b['cross_check'] == 'agree', detail
        assert abs(detail['delivery_floor_correction_ms'] - 56.0) < 0.5, detail

        assert report.clocks['host_a']['correction_source'] == 'reference'

        # Agent 2 publishes 15 ms after the grid; once corrected that is all that
        # is left of its skew.
        with open(os.path.join(scenario, '2', '000004.yaml')) as handle:
            params = yaml.safe_load(handle)
        sync = params['ros_sync']
        assert abs(sync['cloud_dt_ms'] - 15.0) < 0.5, sync
        assert sync['host'] == 'host_b' and sync['clock_correction_source'] == 'ntp'
        assert sync['total_ms'] >= abs(sync['cloud_dt_ms'])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_uncorrected_clock_offset_is_visible_in_the_frame_yaml():
    """With reconciliation off the same bag still converts — and the frame yaml
    says how stale it is, instead of the dataset looking clean.

    Agent 2 is 15 ms late on the grid and 50 ms early on the clock, so its
    contribution sits 35 ms before the frame time. Nothing else in the output
    would reveal that.
    """
    tmp = tempfile.mkdtemp(prefix='r2o_raw_')
    try:
        report, scenario = _convert_multi_host(
            tmp, clock_enabled=False, two_clock_error_ns=-50_000_000,
            transit_ns={'ego': 2_000_000, 'two': 8_000_000, 'infra': 2_000_000},
            ntp_topic='/two/ntp')
        assert report.clocks == {}, 'reconciliation was off; nothing should be claimed'
        with open(os.path.join(scenario, '2', '000004.yaml')) as handle:
            params = yaml.safe_load(handle)
        sync = params['ros_sync']
        assert abs(sync['cloud_dt_ms'] + 35.0) < 0.5, sync
        assert sync['clock_correction_source'] == 'disabled', sync
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_report_carries_the_tightness_curve_and_structural_floor():
    tmp = tempfile.mkdtemp(prefix='r2o_tight_')
    try:
        report, _ = _convert_multi_host(
            tmp, clock_enabled=True, two_clock_error_ns=0,
            transit_ns={'ego': 2_000_000, 'two': 2_000_000, 'infra': 2_000_000},
            ntp_topic='/two/ntp')
        tight = report.tightness
        assert tight['complete_frames'] == 12, tight
        frames = [row['frames'] for row in tight['curve']]
        assert frames == sorted(frames), tight
        assert frames[-1] == 12, tight
        # every agent publishes at 10 Hz, so no frame can be tighter than 50 ms
        assert abs(tight['structural_floor_ms'] - 50.0) < 5.0, tight
        assert report.sync['two']['half_period_ms'] > 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_end_to_end_deskews_the_ego_sweep():
    """The ego's 90 ms sweep, uncorrected, smears its own motion into its cloud."""
    tmp = tempfile.mkdtemp(prefix='r2o_deskew_')
    try:
        _, plain = _convert_multi_host(tmp, clock_enabled=False, deskew=False,
                                       ego_point_times=True)
        raw = read_pcd(os.path.join(plain, '1', '000004.pcd'))
        shutil.rmtree(os.path.join(tmp, 'out_raw'))
        os.remove(os.path.join(tmp, 'cfg_raw.yaml'))

        report, fixed = _convert_multi_host(tmp, clock_enabled=False, deskew=True,
                                            ego_point_times=True)
        moved = read_pcd(os.path.join(fixed, '1', '000004.pcd'))

        assert report.deskew['ego']['applied'] == 12, report.deskew
        assert abs(report.deskew['ego']['sweep_span_ms'] - 90.0) < 1e-6, report.deskew
        # 5 m/s: the point observed 90 ms into the sweep moves 0.45 m, the one at
        # the sweep's start does not move at all.
        shift = moved[:, 0] - raw[:, 0]
        assert abs(shift[0]) < 0.01, shift
        assert abs(shift[2] - 0.45) < 0.02, shift
        # y and z are untouched by straight-line motion along x
        assert np.allclose(moved[:, 1:3], raw[:, 1:3], atol=1e-5)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_ntp_unit_is_arbitrated_by_the_delivery_floor():
    """An NtpStatus message never says whether ``offset`` is seconds or
    milliseconds, and the two differ by a factor of 1000. Here the topic reports
    milliseconds, which the magnitude alone would read as a 50-second error.

    The delivery floor is an independent measurement of the same quantity, so it
    settles the unit the same way it settles the sign — and a rescale is only
    taken when it wins decisively, since "the estimates disagree" is a more useful
    output than a confidently rescaled wrong number.
    """
    tmp = tempfile.mkdtemp(prefix='r2o_unit_')
    try:
        # A *well* synchronised host is where the magnitude test fails: 0.4 ms
        # reported in milliseconds reads as a plausible 0.4-second offset, and
        # acting on it would inject 400 ms of latency into that agent.
        report, _ = _convert_multi_host(
            tmp, clock_enabled=True, two_clock_error_ns=-400_000,
            transit_ns={'ego': 2_000_000, 'two': 8_000_000, 'infra': 2_000_000},
            ntp_topic='/two/ntp', ntp_offset_unit='ms')
        meta = report.clocks['_meta']
        assert meta['unit_rescale'] == 1e-3, meta
        assert abs(report.clocks['host_b']['correction_ms'] - 0.4) < 0.1, report.clocks
        assert any('rescaled' in w for w in report.warnings), report.warnings
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_declared_offset_unit_is_not_second_guessed():
    """An explicit clock.offset_unit is an instruction, not a hint: the estimator
    may still choose the sign but must not override the unit."""
    tmp = tempfile.mkdtemp(prefix='r2o_declared_')
    try:
        report, _ = _convert_multi_host(
            tmp, clock_enabled=True, offset_unit='ms', two_clock_error_ns=-400_000,
            transit_ns={'ego': 2_000_000, 'two': 8_000_000, 'infra': 2_000_000},
            ntp_topic='/two/ntp', ntp_offset_unit='ms')
        assert report.clocks['_meta']['unit_rescale'] == 1.0, report.clocks['_meta']
        assert abs(report.clocks['host_b']['correction_ms'] - 0.4) < 0.1, report.clocks
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_marginal_rescale_is_rejected_rather_than_applied():
    """When no rescale wins decisively the parsed unit stands and the cross-check
    is left to flag the disagreement."""
    clocks = clockmod.HostClocks('ego')
    for value in (5_000_000, 6_000_000):
        clocks.delivery.setdefault('ego', clockmod.DeliveryStats('ego')).add(value)
    for value in (5_000_000, 6_000_000):
        clocks.delivery.setdefault('two', clockmod.DeliveryStats('two')).add(value)
    # A microsecond offset: rescaling it makes the residual "1000x smaller" than a
    # number that was already negligible, which is not evidence of anything.
    clocks.ntp['two'] = clockmod.OffsetTrack([0, NS], [1_000] * 2).finish()
    sign, scale, detail = clockmod.choose_form(clocks)
    assert scale == 1.0, detail
    assert sign in (1.0, -1.0)


def test_pose_tracks_are_clock_corrected_too():
    """A corrected cloud matched against an uncorrected pose is the subtlest form
    of the offset bug: the agent's data and the pose it is placed at come from
    instants tens of milliseconds apart, and the result is a rigid position error
    that nothing downstream attributes to timing.

    Agent 2 moves at 2 m/s and its clock is 50 ms slow, so getting this wrong
    displaces it by 0.1 m — larger than the box it is supposed to be.
    """
    tmp = tempfile.mkdtemp(prefix='r2o_posetime_')
    try:
        report, scenario = _convert_multi_host(
            tmp, clock_enabled=True, two_clock_error_ns=-50_000_000, two_speed=2.0,
            transit_ns={'ego': 2_000_000, 'two': 8_000_000, 'infra': 2_000_000},
            ntp_topic='/two/ntp')
        assert abs(report.clocks['host_b']['correction_ms'] - 50.0) < 0.5, report.clocks

        # Frame 4 is at t0 + 0.4 s on the reference clock. Agent 2's own message is
        # stamped 15 ms later in true time, and its pose at that true instant is
        # y = 2.0 * 0.415 = 0.83 m. Interpolated at the frame time it is 0.80 m;
        # evaluated on the uncorrected timeline it would be 0.70 m.
        with open(os.path.join(scenario, '2', '000004.yaml')) as handle:
            params = yaml.safe_load(handle)
        y = params['true_ego_pos'][1]
        assert abs(y - 0.80) < 0.01, (y, 'pose evaluated on the wrong timeline')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ------------------------------------------- poses already in the world frame
# An offline mapping pipeline republishes its corrected trajectory as a
# PoseStamped in the anchored map frame, and the pose it publishes is often the
# CAMERA OPTICAL frame rather than a body frame. Both facts change the config:
# `align` becomes a true identity (the agents are in one world because the map
# anchoring put them there, not because an operator declared it), and the
# agent's `base` becomes that optical frame, so every sensor extrinsic is
# measured from it.

def test_posestamped_is_accepted_as_a_pose_source():
    """PoseStamped has no twist and no child_frame_id. The extractor must take
    it, and the speed must then be derived from motion rather than left at the
    zero a missing twist would otherwise imply."""
    from ros2opv2v.convert import _pose_and_speed, _fill_speed_from_motion

    class _P:
        pass

    msg = _P(); msg.pose = _P()
    msg.pose.position = _P(); msg.pose.orientation = _P()
    msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = 1.0, 2.0, 3.0
    msg.pose.orientation.x = msg.pose.orientation.y = msg.pose.orientation.z = 0.0
    msg.pose.orientation.w = 1.0
    T, speed = _pose_and_speed(msg)
    assert np.allclose(T[:3, 3], [1, 2, 3]) and speed == 0.0, (T, speed)

    track = PoseTrack()
    for k in range(4):
        M = np.identity(4); M[0, 3] = 2.0 * k       # 2 m per 0.1 s = 20 m/s
        track.add(int(k * NS // 10), M, 0.0)
    track.finish()
    assert _fill_speed_from_motion(track), 'a twistless pose source must get a derived speed'
    assert abs(track.speeds[1] - 20.0) < 1e-6, track.speeds


def test_optical_frame_pose_source_matches_the_odometry_route():
    """The substitution itself, end to end.

    Route A: odometry of a body frame, with `child_to_base` identity and the
             sensor mounted at Z on that body.
    Route B: a PoseStamped of the CAMERA OPTICAL frame (odometry composed with
             X), `align` and `child_to_base` identity, and the same sensor
             re-expressed from the optical frame as inv(X) @ Z.

    Both describe the same physical rig, so every `lidar_pose` must agree. If
    they do not, the optical-frame substitution is silently rotating the sensor
    — which would not show up as a conversion failure, only as a dataset whose
    clouds are consistently in the wrong place.
    """
    X = make_transform(x=0.02, y=-0.05, z=0.15, roll=-90.0, pitch=0.0, yaw=-90.0,
                       degrees=True)                       # body -> optical
    tmp = tempfile.mkdtemp(prefix='r2o_optical_')
    try:
        bag = os.path.join(tmp, 'optical.mcap')
        _write_synthetic_bag(bag, n_frames=8, optical_pose_X=X)

        def run(tag, mutate):
            cfg = _synthetic_config(bag, os.path.join(tmp, 'out_' + tag))
            cfg['agents'] = [a for a in cfg['agents'] if a['name'] == 'ego']
            cfg['agents'][0]['object']['emit'] = False
            mutate(cfg['agents'][0])
            path = os.path.join(tmp, 'cfg_%s.yaml' % tag)
            with open(path, 'w') as h:
                yaml.safe_dump(cfg, h)
            convert(cfgmod.load_config(path), overwrite=True)
            out = []
            for i in range(8):
                f = os.path.join(tmp, 'out_' + tag, 'test', 'synthetic', '1',
                                 '%06d.yaml' % i)
                if os.path.exists(f):
                    with open(f) as h:
                        out.append(yaml.safe_load(h)['lidar_pose'])
            return np.array(out)

        Z = make_transform(z=0.5)                          # body -> sensor
        opt_extrinsic = invert(X) @ Z                      # optical -> sensor
        rpy = matrix_to_opencood_pose(opt_extrinsic)

        A = run('body', lambda a: None)

        def to_optical(a):
            a['pose'] = {'source': 'pose', 'topic': '/ego/pose',
                         'interpolation': 'linear', 'max_gap_ms': 200,
                         'align': {'x': 0, 'y': 0, 'z': 0,
                                   'roll': 0, 'pitch': 0, 'yaw': 0},
                         'child_to_base': {'x': 0, 'y': 0, 'z': 0,
                                           'roll': 0, 'pitch': 0, 'yaw': 0}}
            # x_to_world's (roll, yaw, pitch) triple is not the config block's
            # (roll, pitch, yaw), so go through the matrix rather than the name.
            a['cloud']['extrinsic'] = matrix_to_rpy_config(opt_extrinsic)

        B = run('optical', to_optical)
        assert len(A) and len(A) == len(B), (len(A), len(B))
        d = np.linalg.norm(A[:, :3] - B[:, :3], axis=1)
        assert d.max() < 1e-6, (d.max(), A[0], B[0], rpy)
        # and the orientations, through the matrices they generate
        for pa, pb in zip(A, B):
            assert np.allclose(x_to_world(pa), x_to_world(pb), atol=1e-9), (pa, pb)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_matrix_to_rpy_config_round_trips():
    """Every operator-supplied extrinsic starts as a 4x4 in a calibration file;
    hand-converting it to the config's RPY block is where a sign flips."""
    rng = np.random.default_rng(3)
    for _ in range(500):
        T = make_transform(*rng.uniform(-2, 2, 3),
                           *rng.uniform(-180, 180, 3), degrees=True)
        back = make_transform(**matrix_to_rpy_config(T))
        assert np.allclose(back, T, atol=1e-9), (T, back)
    # gimbal lock (pitch = +-90) must still round-trip the MATRIX
    for p_ in (90.0, -90.0):
        T = make_transform(x=1.0, roll=25.0, pitch=p_, yaw=40.0)
        assert np.allclose(make_transform(**matrix_to_rpy_config(T)), T, atol=1e-9)


# ------------------------------------- the shipped MIRC config's provenance
# Every extrinsic in configs/mirc_coop2.yaml was derived from transforms the
# dataset publishes. Hand-editing one of them is a silent geometric error — the
# conversion still succeeds and every cloud is in the wrong place — so the
# derivation is pinned here rather than living only in a comment.

MIRC_PUBLISHED = {
    # name: (translation, quaternion xyzw) for "A -> B" = the pose of B in A
    'map__arducam': ([-5.508181, -2.590753, 1.998232],
                     [-0.374458, 0.769398, -0.461975, 0.233208]),
    'oslidar__zedopt': ([-0.074928, -0.066971, -0.091627],
                        [-0.497829, -0.498035, 0.501789, 0.502329]),
    'coloropt__depthopt': ([0.059190, -0.000010, -0.000406],
                           [-0.002966, 0.000832, 0.001305, 0.999994]),
    'arducam__infra1': ([0.017773, -0.208023, -0.155788],
                        [0.571431, 0.563856, 0.432969, -0.409965]),
    'cameralink__coloropt': ([0.000308, 0.059191, -0.000162],
                             [0.499583, -0.497446, 0.501716, -0.501244]),
}


def _published(name):
    t, q = MIRC_PUBLISHED[name]
    T = quat_to_matrix(*q)
    T[:3, 3] = t
    return T


def _mirc_config():
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        'configs', 'mirc_coop2.yaml')
    with open(path) as handle:
        return yaml.safe_load(handle)


def _agent(cfg, name):
    return next(a for a in cfg['agents'] if a['name'] == name)


def test_mirc_extrinsics_match_the_published_transforms():
    """Each config extrinsic must reproduce the transform it claims to be."""
    cfg = _mirc_config()
    cases = [
        # (agent, block, expected 4x4, why)
        ('mobile_1', _agent(cfg, 'mobile_1')['cloud']['extrinsic'],
         invert(_published('oslidar__zedopt')),
         'the dataset publishes os_lidar -> ZED optical; the base is the camera, '
         'so the config needs the inverse'),
        ('mobile_2', _agent(cfg, 'mobile_2')['cloud']['extrinsic'],
         _published('coloropt__depthopt'),
         'colour optical -> depth optical, as published'),
        ('infra_1', _agent(cfg, 'infra_1')['cloud']['extrinsic'],
         _published('arducam__infra1'),
         'arducam optical -> radar, as published'),
        ('infra_1 world', _agent(cfg, 'infra_1')['pose']['world_pose'],
         _published('map__arducam'),
         'map -> arducam optical, as published'),
    ]
    for label, block, expected, why in cases:
        assert block is not None, (label, 'still null')
        got = make_transform(**block)
        assert np.allclose(got, expected, atol=2e-5), (label, why, got, expected)


def test_mirc_optical_frame_flags_match_the_optical_bases():
    """Every agent's base is a camera optical frame, so a depth cloud must NOT be
    rotated into ROS body convention on the way out.

    `optical_frame: true` applies OPTICAL_TO_FLU to the reprojected points. With
    an optical base and an optical->optical extrinsic that rotation is pure
    error, and it is invisible: the conversion succeeds and the whole agent is
    turned 90 degrees. On a 3 m return it displaces the point by 4.2 m.
    """
    from ros2opv2v.geometry import OPTICAL_TO_FLU
    cfg = _mirc_config()
    for name in ('mobile_2', 'mobile_1_zed'):
        cloud = _agent(cfg, name)['cloud']
        if cloud['kind'] != 'depth_image':
            continue
        assert cloud.get('optical_frame') is False, \
            (name, 'depth cloud on an optical base must set optical_frame: false')
    # and the magnitude of the mistake, so the assertion above has a stated cost
    T = _published('coloropt__depthopt')
    p3 = np.array([0.0, 0.0, 3.0, 1.0])
    displaced = np.linalg.norm((T @ p3)[:3] - (T @ invert(OPTICAL_TO_FLU) @ p3)[:3])
    assert displaced > 4.0, displaced


def test_realsense_camera_link_is_the_depth_frame():
    """camera_link -> colour -> depth must close on itself. It is the one
    consistency check available between the two RealSense transforms, and it is
    what licenses treating camera_link as the depth body frame."""
    chain = _published('cameralink__coloropt') @ _published('coloropt__depthopt')
    assert np.linalg.norm(chain[:3, 3]) < 1e-5, chain[:3, 3]


def test_mirc_config_has_no_unresolved_geometry():
    """The only value the dataset does not supply is the robots' physical size.
    Everything else must be filled, so a null anywhere else is a regression."""
    cfg = _mirc_config()
    missing = []
    for a in cfg['agents']:
        if not a.get('enabled', True):
            continue
        if a['pose']['source'] == 'static' and a['pose'].get('world_pose') is None:
            missing.append('%s.pose.world_pose' % a['name'])
        if a['cloud'].get('extrinsic') is None:
            missing.append('%s.cloud.extrinsic' % a['name'])
        for i, cam in enumerate(a.get('cameras') or []):
            if cam.get('extrinsic') is None:
                missing.append('%s.cameras[%d].extrinsic' % (a['name'], i))
        if a['pose']['source'] != 'static' and a['pose'].get('align') is None:
            missing.append('%s.pose.align' % a['name'])
    assert not missing, missing


if __name__ == '__main__':
    tests = [(k, v) for k, v in sorted(globals().items())
             if k.startswith('test_') and callable(v)]
    for name, fn in tests:
        check(name, fn)
    print('\n%d/%d passed' % (len(tests) - len(FAILED), len(tests)))
    sys.exit(1 if FAILED else 0)
