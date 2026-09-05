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
import json
import math
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
    clocks = clockmod.HostClocks('ego', apply_corrections=True)
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
    clocks = clockmod.HostClocks('ego', apply_corrections=True)
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
    """CORRECT mode: a perfectly constant 30 ms offset leaves nothing behind once
    applied; calling it a 30 ms residual would be exactly backwards."""
    clocks = clockmod.HostClocks('ego', apply_corrections=True)
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
                        mode='correct', **bag_kwargs):
    bag = os.path.join(tmp, 'mh.mcap')
    if not os.path.exists(bag):
        _write_synthetic_bag(bag, n_frames=12, **bag_kwargs)
    out_root = os.path.join(tmp, 'out_clock' if clock_enabled else 'out_raw')
    config_path = os.path.join(tmp, 'cfg_%s.yaml' % ('clock' if clock_enabled else 'raw'))
    config = _multi_host_config(bag, out_root, clock_enabled, deskew)
    config['clock']['mode'] = mode
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


def test_mirc_depth_clouds_are_written_in_body_axes():
    """Every agent's base is a camera optical frame, so a depth cloud is optical
    on both sides of its extrinsic and nothing ever turns it.

    That was once resolved by NOT rotating, which kept points and pose agreeing
    with each other and disagreeing with every detector: the written cloud had
    z forward and y down, so a BEV model's ground plane was the image plane.
    The real dataset showed it plainly — the ego's cloud measured z 0.65..8.74 m
    with nothing below zero, no floor anywhere. The fix rotates BOTH halves, so
    the points are body-axes and the pose beside them says so.
    """
    cfg = _mirc_config()
    for name in ('mobile_2', 'mobile_1_zed'):
        cloud = _agent(cfg, name)['cloud']
        if cloud['kind'] != 'depth_image':
            continue
        assert cloud.get('optical_frame') is True, \
            (name, 'a depth cloud must be rotated into body axes for a BEV model')
    # a PointCloud2 is used in its header's frame; the flag would be a promise
    # the converter does not keep, and the Ouster carried it meaninglessly
    for name in ('mobile_1', 'infra_1'):
        cloud = _agent(cfg, name)['cloud']
        assert cloud['kind'] == 'pointcloud2' and 'optical_frame' not in cloud, name

    loaded = cfgmod.load_config(os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'configs', 'mirc_coop2.yaml'))
    ego = next(a for a in loaded.active_agents if a.name == 'mobile_2')
    assert ego.cloud.points_rotated_to_body()
    assert not np.allclose(ego.cloud.frame_extrinsic(), ego.cloud.extrinsic)



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


# --------------------------------------------- describing the box sensibly
# The agent-derived boxes are the only labels a bag can produce, and their frame
# is whatever the pose source uses. With an optical base that is [right, down,
# forward], where a box with its length and height swapped still looks like a
# box — so `object.extrinsic` lets the operator describe the robot in ROS body
# axes instead, and these tests pin what that composition actually does.

def test_object_extrinsic_puts_the_box_in_body_axes():
    from ros2opv2v.geometry import OPTICAL_TO_FLU
    from ros2opv2v.labels import agent_box

    # An optical frame at (2, 3, 1) looking along world +x, upright:
    # optical z (forward) = world x, optical x (right) = world -y,
    # optical y (down) = world -z.
    world_from_base = np.identity(4)
    world_from_base[:3, :3] = np.column_stack([[0, -1, 0], [0, 0, -1], [1, 0, 0]])
    world_from_base[:3, 3] = [2.0, 3.0, 1.0]

    box_frame = invert(OPTICAL_TO_FLU)          # optical -> FLU at the same point
    composed = world_from_base @ box_frame
    assert np.allclose(composed[:3, :3], np.identity(3), atol=1e-12), composed
    # so in that frame x really is world-forward, y left, z up

    extent = [0.30, 0.25, 0.25]                 # 0.6 x 0.5 x 0.5 m platform
    centre = [-0.05, 0.0, -0.35]                # camera 35 cm up, 5 cm ahead
    box = agent_box(world_from_base, extent, centre, box_frame)
    assert np.allclose(box["location"], [1.95, 3.0, 0.65], atol=1e-12), box["location"]
    assert box["extent"] == extent
    # OpenCOOD reads `angle` back through x_to_world; that must be the box frame
    assert np.allclose(x_to_world([0, 0, 0] + box["angle"])[:3, :3],
                       np.identity(3), atol=1e-9), box["angle"]


def test_object_extrinsic_defaults_to_the_base_frame():
    """Omitting it must leave existing configs untouched."""
    from ros2opv2v.labels import agent_box
    world_from_base = make_transform(x=1.0, y=2.0, z=3.0, yaw=30.0)
    a = agent_box(world_from_base, [0.3, 0.2, 0.1], [0.0, 0.0, 0.0])
    b = agent_box(world_from_base, [0.3, 0.2, 0.1], [0.0, 0.0, 0.0], np.identity(4))
    assert a == b, (a, b)
    assert np.allclose(a["location"], [1.0, 2.0, 3.0])


def test_box_centre_offset_is_rotated_by_the_box_frame():
    """`center` must be read in the box frame, not the base frame — otherwise the
    extrinsic would fix the axes of `extent` and silently leave `center` wrong."""
    from ros2opv2v.geometry import OPTICAL_TO_FLU
    from ros2opv2v.labels import agent_box
    world_from_base = np.identity(4)            # optical axes aligned with world
    up_in_body = agent_box(world_from_base, [0.3, 0.2, 0.1], [0.0, 0.0, 1.0],
                           invert(OPTICAL_TO_FLU))
    # +z in the body frame is 'up', which in an identity-posed OPTICAL base is
    # world -y (optical y points down)
    assert np.allclose(up_in_body["location"], [0.0, -1.0, 0.0], atol=1e-12), \
        up_in_body["location"]


def test_mirc_boxes_are_declared_in_body_axes():
    """Both mobile agents must carry the optical->body box frame even while
    `emit` is off.

    The boxes are disabled because this dataset is being annotated by hand, but
    the frame has to stay declared: someone turning `emit` back on later would
    otherwise supply cart dimensions that get read as
    [half-width, half-height, half-length], and a box with its length and height
    swapped still looks like a box.
    """
    from ros2opv2v.geometry import OPTICAL_TO_FLU
    cfg = _mirc_config()
    for name in ('mobile_1', 'mobile_2'):
        obj = _agent(cfg, name)['object']
        got = make_transform(**obj['extrinsic'])
        assert np.allclose(got, invert(OPTICAL_TO_FLU), atol=1e-9), (name, got)


def test_converts_with_no_ground_truth_at_all():
    """A dataset built for MANUAL annotation carries no boxes, and that has to be
    a supported state rather than a degenerate one: it must convert, it must
    validate, and every frame must keep enough provenance to tie a label drawn
    later back to the exact source message."""
    tmp = tempfile.mkdtemp(prefix='r2o_nogt_')
    try:
        bag = os.path.join(tmp, 'b.mcap')
        _write_synthetic_bag(bag, n_frames=8)
        cfg = _synthetic_config(bag, os.path.join(tmp, 'out'))
        for agent in cfg['agents']:
            agent['object'] = {'emit': False}
        path = os.path.join(tmp, 'c.yaml')
        with open(path, 'w') as handle:
            yaml.safe_dump(cfg, handle)
        report = convert(cfgmod.load_config(path), overwrite=True)
        assert report.frames_written == 8, report.frames_written
        with open(os.path.join(tmp, 'out', 'test', 'synthetic', '1',
                               '000004.yaml')) as handle:
            params = yaml.safe_load(handle)
        assert params['vehicles'] == {}, params['vehicles']
        for key in ('ros_stamp_ns', 'ros_frame_stamp_ns', 'ros_sync'):
            assert key in params, (key, sorted(params))
        # the pose is still there, so an annotation in map coordinates can be
        # projected into any agent's frame afterwards
        assert np.isfinite(params['lidar_pose']).all()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_mirc_config_is_ready_to_run():
    """The shipped config must load with nothing left for an operator to supply."""
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        'configs', 'mirc_coop2.yaml')
    cfg = cfgmod.load_config(path)                       # raises if anything is null
    names = sorted(a.name for a in cfg.active_agents)
    assert names == ['infra_1', 'mobile_1', 'mobile_2'], names
    # The ego is the narrow-FOV depth cart ON PURPOSE: the Ouster sees both chairs
    # alone in ~95% of frames, so with it as ego there is no collaboration gap to
    # measure. The clock reference stays the Ouster host: it is the frame master.
    ego = cfgmod.ego_agent(cfg)
    assert ego.name == 'mobile_2' and ego.cav_id == 1, (ego.name, ego.cav_id)
    assert next(a for a in cfg.agents if a.name == 'mobile_1').cav_id == 2
    assert cfg.clock.enabled and cfg.clock.reference_host == 'mobile_1'
    assert all(not a.obj.emit for a in cfg.active_agents), \
        'this dataset is converted for manual annotation; no agent-derived boxes'


# --------------------------------------------- the pose source's start frame
# A trajectory republished at a body frame under the sensor has the same shape
# and timing as the optical-frame one and sits a metre lower. It is the one
# frame error the converter cannot detect from the poses alone, so the config
# can declare where the base must start and the converter refuses otherwise.

def test_expected_start_accepts_the_right_frame():
    tmp = tempfile.mkdtemp(prefix='r2o_start_ok_')
    try:
        bag = os.path.join(tmp, 'b.mcap')
        _write_synthetic_bag(bag, n_frames=6)
        cfg = _synthetic_config(bag, os.path.join(tmp, 'out'))
        for agent in cfg['agents']:
            agent['object'] = {'emit': False}
        ego = _agent(cfg, 'ego')
        ego['pose']['expected_start'] = [0.0, 0.0, 0.0]   # ego odometry starts at origin
        path = os.path.join(tmp, 'c.yaml')
        with open(path, 'w') as handle:
            yaml.safe_dump(cfg, handle)
        report = convert(cfgmod.load_config(path), overwrite=True)
        assert report.pose_stats['ego']['start_gap_m'] < 0.01, report.pose_stats['ego']
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_expected_start_refuses_a_pose_source_a_metre_low():
    """The failure this exists for: same trajectory, republished 1.2 m lower."""
    from ros2opv2v.convert import ConversionError
    tmp = tempfile.mkdtemp(prefix='r2o_start_bad_')
    try:
        bag = os.path.join(tmp, 'b.mcap')
        _write_synthetic_bag(bag, n_frames=6)
        cfg = _synthetic_config(bag, os.path.join(tmp, 'out'))
        for agent in cfg['agents']:
            agent['object'] = {'emit': False}
        ego = _agent(cfg, 'ego')
        ego['pose']['expected_start'] = [0.0, 0.0, 1.2]   # the camera is up here
        path = os.path.join(tmp, 'c.yaml')
        with open(path, 'w') as handle:
            yaml.safe_dump(cfg, handle)
        try:
            convert(cfgmod.load_config(path), overwrite=True)
        except ConversionError as exc:
            message = str(exc)
            assert 'ego' in message and 'dz -1.20' in message, message
            assert 'BODY frame' in message, 'a purely vertical gap must be diagnosed as such'
        else:
            raise AssertionError('a 1.2 m vertical mismatch must refuse to convert')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_mirc_config_declares_the_published_anchors():
    cfg = _mirc_config()
    assert _agent(cfg, 'mobile_1')['pose']['expected_start'] == [0.697952, -0.062696, 0.193334]
    assert _agent(cfg, 'mobile_2')['pose']['expected_start'] == [7.704074, -14.308017, 0.165040]


# ------------------------------------------------ verify mode (the default)
# On a host running chrony the system clock is already disciplined: every
# header.stamp is on the corrected clock, and what the NTP status reports is the
# residual the daemon believes remains. Applying that again would be redundant
# at best and, with the sign guessed, twice wrong. So the default is to read the
# NTP topics as evidence, shift nothing, and carry the reported residual.

def test_verify_mode_shifts_nothing_and_carries_the_reported_offset():
    clocks = clockmod.HostClocks('ego')                       # default: verify
    assert clocks.apply_corrections is False
    for value in (2_000_000, 4_000_000):
        clocks.delivery.setdefault('ego', clockmod.DeliveryStats('ego')).add(value)
    for value in (32_000_000, 34_000_000):
        clocks.delivery.setdefault('two', clockmod.DeliveryStats('two')).add(value)
    clocks.ntp['two'] = clockmod.OffsetTrack([0, NS], [-30_000_000] * 2).finish()
    # the estimate is still made, and still cross-checked...
    assert clocks.estimate_ns('two', 0) == (30_000_000, 'ntp')
    assert clocks.cross_check('two', 20_000_000)[0] == 'agree'
    # ...but nothing is applied, and the residual IS the daemon's reported offset
    assert clocks.correction_ns('two', 0) == (0, 'verify:ntp')
    residual, source = clocks.residual_ns('two')
    assert residual == 30_000_000 and source == 'ntp_reported_offset', (residual, source)


def test_verify_mode_end_to_end_leaves_stamps_alone_and_warns():
    """Same bag as the correct-mode test: agent 2's daemon reports 50 ms. In
    verify mode the frames keep their uncorrected skew (-35 ms), the 50 ms rides
    along as clock_residual_ms, and the report says the residual is large."""
    tmp = tempfile.mkdtemp(prefix='r2o_verify_')
    try:
        report, scenario = _convert_multi_host(
            tmp, clock_enabled=True, mode='verify', two_clock_error_ns=-50_000_000,
            transit_ns={'ego': 2_000_000, 'two': 8_000_000, 'infra': 2_000_000},
            ntp_topic='/two/ntp')
        host_b = report.clocks['host_b']
        assert host_b['mode'] == 'verify'
        assert host_b['correction_ms'] == 0.0, host_b
        assert abs(host_b['estimated_offset_ms'] - 50.0) < 0.5, host_b
        assert abs(host_b['residual_ms'] - 50.0) < 0.5, host_b
        assert any('residual offset of 50.0 ms' in w for w in report.warnings), report.warnings
        with open(os.path.join(scenario, '2', '000004.yaml')) as handle:
            sync = yaml.safe_load(handle)['ros_sync']
        assert abs(sync['cloud_dt_ms'] + 35.0) < 0.5, sync          # uncorrected
        assert abs(sync['clock_residual_ms'] - 50.0) < 0.5, sync    # but declared
        assert sync['clock_correction_source'] == 'verify:ntp', sync
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_verify_mode_is_quiet_when_the_daemons_agree():
    """A well-disciplined fleet must produce no clock warnings at all."""
    tmp = tempfile.mkdtemp(prefix='r2o_verify_ok_')
    try:
        report, _ = _convert_multi_host(
            tmp, clock_enabled=True, mode='verify', two_clock_error_ns=-400_000,
            transit_ns={'ego': 2_000_000, 'two': 2_000_000, 'infra': 2_000_000},
            ntp_topic='/two/ntp')
        clock_warnings = [w for w in report.warnings if w.startswith('clock:')]
        assert not clock_warnings, clock_warnings
        assert abs(report.clocks['host_b']['residual_ms'] - 0.4) < 0.05, report.clocks
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_mirc_config_verifies_rather_than_corrects():
    cfg = _mirc_config()
    assert cfg['clock']['mode'] == 'verify', cfg['clock']
    assert set(cfg['clock']['events_topics']) == {'infra_1', 'mobile_2'}


# ----------------------------------------------------- daemon event reading
# The first real dry run produced ten "STEP" warnings for infra_1, all at
# t=4.2 s, all under 5 ms, with monotonic timestamps spanning 26 minutes: a
# latched topic replaying chrony's routine adjustments from before the
# recording. Warning about those is worse than silence — it teaches the reader
# to ignore the warning that matters.

def _events_report(parsed, bag_span_s=156.0, max_residual_ms=10.0):
    from ros2opv2v.convert import ConversionReport, _classify_clock_events
    report = ConversionReport()
    _classify_clock_events('h', parsed, bag_span_s, max_residual_ms, report)
    return report.warnings


def test_clock_event_parser_reads_delta_and_mono():
    from ros2opv2v.convert import _parse_clock_event
    e = _parse_clock_event(4_200_000_000, 'STEP host=x delta=4.144ms mono=18326.509', 0)
    assert e['t_rel_s'] == 4.2 and abs(e['delta_ms'] - 4.144) < 1e-9
    assert abs(e['mono_s'] - 18326.509) < 1e-9
    e = _parse_clock_event(0, 'STEP delta=-2us', 0)
    assert abs(e['delta_ms'] + 0.002) < 1e-9


def test_latched_backlog_is_recognised_not_alarmed():
    parsed = [{'t_rel_s': 4.2, 'text': 'STEP delta=%.3fms mono=%.1f' % (d, m),
               'delta_ms': d, 'mono_s': m}
              for d, m in zip((0.35, 0.19, -0.31, 1.77, 4.14), (17418, 17675, 17806, 18065, 18326))]
    warnings = _events_report(parsed)
    assert len(warnings) == 1 and 'backlog' in warnings[0], warnings
    assert 'STEPPED' not in warnings[0]


def test_small_adjustments_during_the_run_are_routine():
    parsed = [{'t_rel_s': 46.0, 'text': 'STEP delta=0.671ms mono=12010.5',
               'delta_ms': 0.671, 'mono_s': 12010.5}]
    warnings = _events_report(parsed)
    assert len(warnings) == 1 and 'routine' in warnings[0], warnings


def test_a_real_step_and_a_lost_source_still_alarm():
    parsed = [{'t_rel_s': 46.0, 'text': 'STEP delta=250ms mono=12010.5',
               'delta_ms': 250.0, 'mono_s': 12010.5},
              {'t_rel_s': 80.0, 'text': 'source unreachable'}]
    warnings = _events_report(parsed)
    assert any('STEPPED by +250.0 ms' in w for w in warnings), warnings
    assert any('unreachable' in w and 'free-running' in w for w in warnings), warnings


def test_fastest_moves_flag_steps_across_dropped_frames():
    """A 0.26 m step over 1.9 s of dropped frames is a walk, not a jump."""
    from ros2opv2v.convert import _fastest_moves
    rows = _fastest_moves(steps=[0.10, 0.26, 0.10], times=[1.0, 3.0, 3.1],
                          dts=[0.1, 1.9, 0.1])
    by_t = {r['t_s']: r for r in rows}
    assert by_t[3.0]['gap'] is True and abs(by_t[3.0]['speed_mps'] - 0.137) < 1e-3, rows
    assert by_t[1.0]['gap'] is False and abs(by_t[1.0]['speed_mps'] - 1.0) < 1e-9
    assert rows[0]['t_s'] in (1.0, 3.1), 'ordered by speed, so the walk comes last'


# -------------------------------------------------- the real NtpStatus schema
# ntp_monitor_msgs/NtpStatus, as shown by `ros2 interface show`: every timing
# field is float64 SECONDS, jitter is `jitter_seconds`, and the message carries
# its own health verdicts. The first resolver pass had no `jitter_seconds`
# candidate and fell through to `root_dispersion` — a formal worst-case bound
# several times the realised jitter — and reported it as the residual.

class _NtpStatus:
    """A message shaped like the real schema, without ROS."""

    def __init__(self, offset_s, jitter_s=0.0002, root_disp_s=0.005, synced=True,
                 stepped=False, delta_s=0.0, reach=100, stratum=2, warnings=()):
        self.role = "client"; self.hostname = "h"
        self.synchronized = synced; self.sync_source = "10.0.0.1"
        self.stratum_level = str(stratum); self.stratum = stratum
        self.offset_seconds = offset_s; self.delay_seconds = 0.001
        self.jitter_seconds = jitter_s; self.root_delay = 0.002
        self.root_dispersion = root_disp_s; self.frequency_error_ppm = -3.1
        self.poll_interval_seconds = 64; self.reach_register = 255 if reach == 100 else 127
        self.reachability_percent = reach; self.connected_clients = -1
        self.leap_indicator = "none"; self.warnings = list(warnings)
        self.seq = 0; self.monotonic_seconds = 100.0
        self.clock_stepped = stepped; self.offset_delta_seconds = delta_s


def test_ntpstatus_fields_are_read_as_the_schema_says():
    rows = [(i * NS, _NtpStatus(offset_s=0.0004)) for i in range(20)]
    track, meta = clockmod.build_offset_track(rows, offset_field='offset_seconds',
                                              offset_unit='s', source='/x/ntp/status')
    assert meta['jitter_field'] == 'jitter_seconds', meta
    assert meta['bound_field'] == 'root_dispersion', meta
    assert abs(meta['bound_p95_ms'] - 5.0) < 1e-6, meta
    stats = track.stats()
    assert abs(stats['p95_abs_ms'] - 0.4) < 1e-6, stats
    assert abs(stats['reported_jitter_p95_ms'] - 0.2) < 1e-6, stats
    # verify-mode residual is the realised error, not the formal bound
    clocks = clockmod.HostClocks('ref')
    clocks.ntp['h'] = track
    residual, source = clocks.residual_ns('h')
    assert abs(residual / 1e6 - 0.4) < 1e-6 and source == 'ntp_reported_offset', (residual, source)


def test_ntpstatus_health_flags_are_surfaced():
    from ros2opv2v.convert import ConversionReport, _health_warnings
    rows = [(i * NS, _NtpStatus(offset_s=0.0003)) for i in range(10)]
    rows[3] = (3 * NS, _NtpStatus(offset_s=0.0003, synced=False, reach=50))
    rows[7] = (7 * NS, _NtpStatus(offset_s=0.0003, stepped=True, delta_s=0.120,
                                  warnings=('source switched',)))
    _, meta = clockmod.build_offset_track(rows, offset_field='offset_seconds',
                                          offset_unit='s')
    health = meta['health']
    assert health['unsynced_samples'] == 1 and health['reachability_min_pct'] == 50, health
    assert len(health['steps']) == 1 and abs(health['steps'][0]['delta_ms'] - 120.0) < 1e-9
    assert health['daemon_warnings'] == {'source switched': 1}
    report = ConversionReport()
    _health_warnings('h', health, 10.0, 0, report)
    texts = "\n".join(report.warnings)
    assert 'synchronized=false in 1 of 10' in texts, texts
    assert 'clock_stepped with a +120.0 ms delta at t=7.0 s' in texts, texts
    assert 'reachability dropped to 50%' in texts, texts
    assert "daemon warning x1: 'source switched'" in texts, texts


def test_ntpstatus_healthy_stream_is_quiet():
    from ros2opv2v.convert import ConversionReport, _health_warnings
    rows = [(i * NS, _NtpStatus(offset_s=0.0003)) for i in range(10)]
    _, meta = clockmod.build_offset_track(rows, offset_field='offset_seconds', offset_unit='s')
    report = ConversionReport()
    _health_warnings('h', meta['health'], 10.0, 0, report)
    assert report.warnings == [], report.warnings


def test_mirc_config_pins_the_ntpstatus_fields():
    cfg = _mirc_config()
    assert cfg['clock']['offset_field'] == 'offset_seconds'
    assert cfg['clock']['offset_unit'] == 's'


# ----------------------------------------------- static labels from the map
# Two chairs that never move are two boxes, not 2 x 1330 hand-drawn ones. The
# geometry has to be exact, because a box placed by hand IS the ground truth —
# nothing downstream can correct it.

def test_fit_box_recovers_a_known_oriented_box():
    from ros2opv2v.statics import fit_box, points_in_box
    rng = np.random.default_rng(0)
    yaw = math.radians(30.0)
    rot = np.array([[math.cos(yaw), -math.sin(yaw), 0.0],
                    [math.sin(yaw), math.cos(yaw), 0.0], [0.0, 0.0, 1.0]])
    local = (rng.random((4000, 3)) - 0.5) * np.array([0.6, 0.4, 0.9])
    pts = local @ rot.T + np.array([3.0, -2.0, -0.55])
    box = fit_box(pts)
    assert np.allclose(box["location"], [3.0, -2.0, -0.55], atol=2e-3), box
    assert np.allclose(box["extent"], [0.30, 0.20, 0.45], atol=2e-3), box
    assert abs(box["angle"][1] - 30.0) < 0.1, box
    assert points_in_box(pts, box).mean() > 0.99


def test_fit_box_puts_the_long_axis_first_whatever_the_hull_edge():
    """`extent` must read as [half-length, half-width, half-height] regardless of
    which hull edge won the rotating-calipers search."""
    from ros2opv2v.statics import fit_box
    rng = np.random.default_rng(1)
    for yaw_deg in (0.0, 37.0, 91.0, 175.0, -120.0):
        yaw = math.radians(yaw_deg)
        rot = np.array([[math.cos(yaw), -math.sin(yaw), 0.0],
                        [math.sin(yaw), math.cos(yaw), 0.0], [0.0, 0.0, 1.0]])
        pts = (rng.random((3000, 3)) - 0.5) * np.array([1.0, 0.4, 0.8]) @ rot.T
        box = fit_box(pts)
        assert box["extent"][0] >= box["extent"][1], (yaw_deg, box["extent"])
        assert np.allclose(sorted(box["extent"][:2]), [0.2, 0.5], atol=5e-3), (yaw_deg, box)


def test_fit_box_extends_to_the_floor():
    """A LiDAR sees a chair's seat and back and almost none of its legs, so a box
    fitted to the returns alone floats and every IoU against it is wrong the same
    way."""
    from ros2opv2v.statics import fit_box
    rng = np.random.default_rng(2)
    seat = rng.random((800, 3)) * np.array([0.5, 0.5, 0.1]) + np.array([0, 0, 0.45])
    floating = fit_box(seat)
    grounded = fit_box(seat, ground_z=0.0)
    assert floating["fit"]["z_range"][0] > 0.4, floating
    assert grounded["fit"]["z_range"][0] == 0.0 and grounded["fit"]["extended_to_ground"]
    assert grounded["extent"][2] > floating["extent"][2] * 2


def test_ground_level_finds_the_floor_not_the_ceiling():
    from ros2opv2v.statics import ground_level
    rng = np.random.default_rng(3)
    floor = np.column_stack([rng.uniform(-5, 5, 8000), rng.uniform(-5, 5, 8000),
                             rng.normal(-1.05, 0.01, 8000)])
    ceiling = np.column_stack([rng.uniform(-5, 5, 12000), rng.uniform(-5, 5, 12000),
                               rng.normal(1.60, 0.01, 12000)])
    z, info = ground_level(np.vstack([floor, ceiling]))
    assert abs(z + 1.05) < 0.05, (z, info)


def test_cluster_at_picks_the_object_not_its_neighbour():
    from ros2opv2v.statics import cluster_at
    rng = np.random.default_rng(4)
    a = rng.random((900, 3)) * np.array([0.4, 0.4, 0.8]) + np.array([0.0, 0.0, 0.05])
    b = rng.random((900, 3)) * np.array([0.4, 0.4, 0.8]) + np.array([1.5, 0.0, 0.05])
    cluster, info = cluster_at(np.vstack([a, b]), np.array([0.2, 0.2, 0.4]), radius=1.2)
    assert info["points_in_cluster"] == 900, info
    assert cluster[:, 0].max() < 0.5, 'the neighbour 1.5 m away must not join'


def test_points_in_box_respects_yaw():
    from ros2opv2v.statics import points_in_box
    box = {"location": [0.0, 0.0, 0.0], "extent": [1.0, 0.2, 0.5], "angle": [0.0, 90.0, 0.0]}
    # yawed 90 deg: the long axis now runs along world y
    assert points_in_box(np.array([[0.0, 0.9, 0.0]]), box)[0]
    assert not points_in_box(np.array([[0.9, 0.0, 0.0]]), box)[0]


def test_static_labels_land_in_every_frame_of_every_agent():
    """The point of static labels: one box in the file, correct in all frames."""
    tmp = tempfile.mkdtemp(prefix='r2o_statics_')
    try:
        bag = os.path.join(tmp, 'b.mcap')
        _write_synthetic_bag(bag, n_frames=8)
        labels_path = os.path.join(tmp, 'statics.json')
        with open(labels_path, 'w') as handle:
            json.dump([{"id": 7, "name": "chair_1", "location": [2.0, 1.0, 0.4],
                        "extent": [0.3, 0.3, 0.45], "angle": [0.0, 15.0, 0.0]}], handle)
        cfg = _synthetic_config(bag, os.path.join(tmp, 'out'))
        for agent in cfg['agents']:
            agent['object'] = {'emit': False}
        cfg['output']['labels_file'] = labels_path
        path = os.path.join(tmp, 'c.yaml')
        with open(path, 'w') as handle:
            yaml.safe_dump(cfg, handle)
        report = convert(cfgmod.load_config(path), overwrite=True)
        assert any('static object' in w for w in report.warnings), report.warnings
        for agent in ('1', '2', '-1'):
            for key in ('000000', '000004', '000007'):
                with open(os.path.join(tmp, 'out', 'test', 'synthetic', agent,
                                       key + '.yaml')) as handle:
                    vehicles = yaml.safe_load(handle)['vehicles']
                assert set(vehicles) == {7}, (agent, key, vehicles)
                assert np.allclose(vehicles[7]['location'], [2.0, 1.0, 0.4]), vehicles
                assert abs(vehicles[7]['angle'][1] - 15.0) < 1e-9
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_static_labels_coexist_with_agent_boxes_and_reject_id_clashes():
    from ros2opv2v.labels import merge_external_labels, vehicles_for_viewer
    poses = {'ego': make_transform(x=1.0), 'two': make_transform(x=6.0)}
    objects = {'two': {'object_id': 10002, 'extent': [0.3, 0.3, 0.3], 'center': [0, 0, 0]}}
    vehicles = vehicles_for_viewer(poses, objects, viewer='ego')
    merged = merge_external_labels(dict(vehicles), [
        {"id": 7, "location": [2.0, 1.0, 0.4], "extent": [0.3, 0.3, 0.45]}])
    assert set(merged) == {10002, 7}, merged
    try:
        merge_external_labels(dict(vehicles), [
            {"id": 10002, "location": [0, 0, 0], "extent": [1, 1, 1]}])
    except ValueError as exc:
        assert 'collides' in str(exc)
    else:
        raise AssertionError('an id colliding with an agent box must be refused')


def test_missing_labels_file_is_refused_at_config_load():
    tmp = tempfile.mkdtemp(prefix='r2o_nolabels_')
    try:
        cfg = _synthetic_config('/nonexistent.mcap', os.path.join(tmp, 'out'))
        cfg['output']['labels_file'] = os.path.join(tmp, 'not_there.json')
        path = os.path.join(tmp, 'c.yaml')
        with open(path, 'w') as handle:
            yaml.safe_dump(cfg, handle)
        try:
            cfgmod.load_config(path)
        except cfgmod.ConfigError as exc:
            assert 'label_static.py' in str(exc), exc
        else:
            raise AssertionError('a missing labels_file must fail at load, not at write time')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_frame_check_does_not_fail_on_static_infrastructure():
    """A static node legitimately sits outside the mapped floor and looks into it
    — the MIRC Arducam is 5.5 m outside it, 2 m up. Only a MOVING agent's
    trajectory is evidence about the frame."""
    import importlib
    label_static = importlib.import_module('label_static')
    rng = np.random.default_rng(0)
    cloud = np.column_stack([rng.uniform(0, 8, 5000), rng.uniform(-14, 0, 5000),
                             rng.normal(-1.0, 0.01, 5000)])
    moving = [{"lidar_pose": [x, -x, 0.2, 0, 0, 0]} for x in np.linspace(0.5, 7.5, 40)]
    static_outside = [{"lidar_pose": [-5.5, -2.6, 2.0, 0, 0, 0]}] * 40
    assert label_static.verify_frame(cloud, {"1": moving, "-1": static_outside})
    # but a moving agent outside it is a real frame error
    shifted = [{"lidar_pose": [x + 100.0, -x, 0.2, 0, 0, 0]} for x in np.linspace(0.5, 7.5, 40)]
    assert not label_static.verify_frame(cloud, {"1": shifted})



def _room_cloud(rng):
    """A 10 x 10 m room: a long wall along x = 5, one chair in the open, one
    chair pushed against the wall. The wall is 2.5 m tall, the chairs 0.9 m."""
    parts = [np.column_stack([rng.uniform(-5, 5, 40000), rng.uniform(-5, 5, 40000),
                              np.full(40000, -0.66)])]                      # floor
    wall_y = rng.uniform(-5, 5, 30000)
    parts.append(np.column_stack([np.full(30000, 5.0) + rng.normal(0, 0.01, 30000),
                                  wall_y, rng.uniform(-0.66, 1.84, 30000)]))
    for cx, cy in ((0.0, 0.0), (4.75, 2.0)):                                # chairs
        n = 4000
        parts.append(np.column_stack([
            cx + rng.uniform(-0.25, 0.25, n), cy + rng.uniform(-0.25, 0.25, n),
            rng.uniform(-0.66, 0.24, n)]))
    return np.vstack(parts)


class _ProposeArgs(object):
    ceiling = 2.0
    max_wall_contact = 1.0
    cell = 0.10
    min_height = 0.30
    max_height = 1.30
    min_footprint = 0.15
    max_footprint = 1.50
    min_points = 80
    max_proposals = 25
    map_scale = 3


def test_propose_ranks_a_free_standing_chair_above_a_wall_foot():
    """The wall is excluded by height, but its base cells survive the band and
    look like objects. What separates them from furniture is how much of the
    blob's outline is structure — a chair is open, a wall foot is not."""
    import importlib
    label_static = importlib.import_module('label_static')
    cloud = _room_cloud(np.random.default_rng(3))
    rows = label_static.propose(cloud, -0.66, _ProposeArgs())
    assert rows, 'the chairs should be proposed'
    first = rows[0]
    assert abs(first['centre'][0]) < 0.25 and abs(first['centre'][1]) < 0.25, first
    assert first['wall_contact'] == 0.0, first
    assert 0.8 <= first['top_m'] <= 1.0, first
    # nothing reported may be the wall itself: it is taller than the band
    for row in rows:
        assert max(row['footprint_m']) <= 1.50, row


def test_propose_still_finds_a_chair_pushed_against_a_wall():
    """Excluding structure must not exclude the furniture standing next to it."""
    import importlib
    label_static = importlib.import_module('label_static')
    cloud = _room_cloud(np.random.default_rng(5))
    rows = label_static.propose(cloud, -0.66, _ProposeArgs())
    near_wall = [r for r in rows if abs(r['centre'][0] - 4.75) < 0.4
                 and abs(r['centre'][1] - 2.0) < 0.4]
    assert near_wall, 'the chair against the wall was lost with the wall'
    assert near_wall[0]['wall_contact'] > 0.0, near_wall[0]
    assert near_wall[0]['wall_contact'] < 0.5, (
        'a chair touches structure on one side, not on most of its outline')


def test_height_map_bounds_and_grid_line_up_with_the_cloud():
    import importlib
    label_static = importlib.import_module('label_static')
    cloud = _room_cloud(np.random.default_rng(7))
    hmap = label_static.HeightMap(cloud, -0.66, 0.10)
    x0, y0, x1, y1 = hmap.bounds()
    assert x0 <= 0.0 <= x1 and y0 <= 0.0 <= y1
    grid = hmap.as_image_grid()
    assert grid.shape == (hmap.depth, hmap.width)
    # the wall column is the tallest thing in the picture
    col = int((5.0 - x0) / 0.10)
    assert grid[:, min(col, grid.shape[1] - 1)].max() > 2.0


def test_png_writer_emits_a_file_a_decoder_accepts():
    """No PIL on the robot, so the PNG is written by hand; it still has to be a
    PNG — right magic, right dimensions, a decodable IDAT."""
    import struct
    import zlib
    from ros2opv2v.preview import Canvas
    tmp = tempfile.mkdtemp()
    try:
        path = os.path.join(tmp, 'map.png')
        canvas = Canvas(0.0, 0.0, 3.0, 2.0, 0.1, scale=4)
        canvas.box(1.0, 0.5, 2.0, 1.5, (220, 0, 0))
        canvas.text(1.0, 1.4, '12', (0, 0, 0))
        canvas.save(path)
        blob = open(path, 'rb').read()
        assert blob[:8] == b'\x89PNG\r\n\x1a\n'
        width, height, depth, colour = struct.unpack('>IIBB', blob[16:26])
        assert (width, height) == (120, 80), (width, height)
        assert (depth, colour) == (8, 2)
        idat = blob[blob.index(b'IDAT') + 4:]
        raw = zlib.decompressobj().decompress(idat)
        assert len(raw) >= height * (1 + width * 3)
        # the drawn box really is red in the decoded pixels
        rows = [raw[i * (1 + width * 3) + 1:(i + 1) * (1 + width * 3)] for i in range(height)]
        assert any(b'\xdc\x00\x00' in row for row in rows), 'no red box in the image'
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_render_map_draws_the_searched_area_without_a_plotting_stack():
    import importlib
    label_static = importlib.import_module('label_static')
    cloud = _room_cloud(np.random.default_rng(11))
    args = _ProposeArgs()
    hmap = label_static.HeightMap(cloud, -0.66, args.cell)
    rows = label_static.propose(cloud, -0.66, args, None, hmap)
    tmp = tempfile.mkdtemp()
    try:
        path = os.path.join(tmp, 'top_down.png')
        poses = {'1': [{'xyz': [x, 0.0, 0.0]} for x in np.linspace(-4, 4, 20)]}
        label_static.render_map(path, hmap, rows, args, poses)
        assert os.path.getsize(path) > 200
    finally:
        shutil.rmtree(tmp, ignore_errors=True)



def test_a_scanned_ceiling_hides_every_object_until_it_is_cut_off():
    """The map is "tallest return per cell". A ceiling is the tallest return over
    every cell of the floor it covers, so with it in the map the whole room reads
    as structure and the chairs standing there are never proposed — not filtered
    out, never seen. Capping the height restores them, and the wall stays
    structure because it is still taller than the object band."""
    import importlib
    label_static = importlib.import_module('label_static')
    rng = np.random.default_rng(13)
    room = _room_cloud(rng)
    ceiling = np.column_stack([rng.uniform(-5, 5, 30000), rng.uniform(-5, 5, 30000),
                               np.full(30000, -0.66 + 2.60)])
    roofed = np.vstack([room, ceiling])

    args = _ProposeArgs()
    args.ceiling = None
    blind = label_static.propose(roofed, -0.66, args)
    assert not blind, 'a scanned ceiling should swallow the whole floor'
    blind_map = label_static.HeightMap(roofed, -0.66, 0.10, ceiling=None)
    assert blind_map.structure_fraction(1.30) > 0.9

    args.ceiling = 2.0
    hmap = label_static.HeightMap(roofed, -0.66, 0.10, ceiling=2.0)
    assert hmap.overhead > 0
    assert hmap.structure_fraction(1.30) < 0.3
    rows = label_static.propose(roofed, -0.66, args, None, hmap)
    assert rows, 'the chairs are back once the ceiling is cut off'
    assert abs(rows[0]['centre'][0]) < 0.25 and abs(rows[0]['centre'][1]) < 0.25
    # the wall is still excluded: nothing proposed sits on it
    assert all(row['centre'][0] < 4.9 for row in rows), rows


def test_max_wall_contact_keeps_furniture_and_drops_wall_feet():
    import importlib
    label_static = importlib.import_module('label_static')
    cloud = _room_cloud(np.random.default_rng(17))
    args = _ProposeArgs()
    args.max_wall_contact = 0.3
    for row in label_static.propose(cloud, -0.66, args):
        assert row['wall_contact'] <= 0.3, row


def test_start_roi_is_the_box_between_where_the_agents_began():
    import importlib
    label_static = importlib.import_module('label_static')
    poses = {
        '1': [{'lidar_pose': [1.0, -2.0, 0, 0, 0, 0]}]
             + [{'lidar_pose': [x, -2.0, 0, 0, 0, 0]} for x in np.linspace(1, 9, 20)],
        '2': [{'lidar_pose': [5.0, -8.0, 0, 0, 0, 0]}]
             + [{'lidar_pose': [5.0, y, 0, 0, 0, 0]} for y in np.linspace(-8, 0, 20)],
        '-1': [{'lidar_pose': [-5.5, -2.6, 2.0, 0, 0, 0]}] * 5,   # static: not a corner
    }
    roi = label_static.start_roi(poses, 1.0)
    assert np.allclose(roi, [0.0, -9.0, 6.0, -1.0]), roi
    # the whole path reaches further, which is exactly the difference
    whole = label_static.trajectory_roi(poses, 1.0)
    assert whole[2] > roi[2]


def test_render_map_accepts_dataset_poses_as_they_are_loaded():
    """load_dataset_poses yields lidar_pose, not xyz — the picture must draw the
    path from the same rows every other stage reads."""
    import importlib
    label_static = importlib.import_module('label_static')
    cloud = _room_cloud(np.random.default_rng(19))
    args = _ProposeArgs()
    hmap = label_static.HeightMap(cloud, -0.66, args.cell, ceiling=args.ceiling)
    rows = label_static.propose(cloud, -0.66, args, None, hmap)
    poses = {'1': [{'lidar_pose': [x, 0.0, 0.1, 0, 0, 0]} for x in np.linspace(-4, 4, 20)]}
    tmp = tempfile.mkdtemp()
    try:
        path = os.path.join(tmp, 'top_down.png')
        label_static.render_map(path, hmap, rows, args, poses)
        assert os.path.getsize(path) > 200
    finally:
        shutil.rmtree(tmp, ignore_errors=True)



def test_footprint_corners_walk_the_rectangle_not_a_bow_tie():
    """box_corners is in sign order, where consecutive entries share an x face.
    Walked as a polygon that draws two crossing triangles, so the footprint has
    to be reordered before anything draws it."""
    from ros2opv2v.statics import footprint_corners
    box = {"location": [1.0, 2.0, 0.0], "extent": [0.5, 0.25, 0.45],
           "angle": [0.0, 30.0, 0.0]}
    corners = footprint_corners(box)
    assert corners.shape == (4, 2)
    sides = [float(np.linalg.norm(corners[(i + 1) % 4] - corners[i])) for i in range(4)]
    assert np.allclose(sorted(sides), [0.5, 0.5, 1.0, 1.0]), sides
    # opposite corners are the diagonals, so the walk is cyclic and not crossed
    diag = float(np.linalg.norm(corners[2] - corners[0]))
    assert abs(diag - math.hypot(1.0, 0.5)) < 1e-9, diag
    # and the polygon is centred on the box
    assert np.allclose(corners.mean(axis=0), [1.0, 2.0])


def test_seeding_a_chair_under_a_ceiling_does_not_swallow_the_ceiling():
    """A chair and the ceiling above it are separated by empty air, which the
    flood fill cannot cross — until anything vertical beside the chair bridges
    it, and then the cluster is the whole roof. Gating the cluster height keeps
    the box on the furniture."""
    from ros2opv2v.statics import cluster_at, fit_box
    rng = np.random.default_rng(23)
    ground = -0.66
    chair = np.column_stack([rng.uniform(-0.25, 0.25, 3000),
                             rng.uniform(-0.25, 0.25, 3000),
                             rng.uniform(ground, ground + 0.90, 3000)])
    pole = np.column_stack([rng.uniform(0.24, 0.30, 4000), rng.uniform(-0.03, 0.03, 4000),
                            rng.uniform(ground, ground + 2.60, 4000)])
    ceiling = np.column_stack([rng.uniform(-1.0, 1.0, 12000),
                               rng.uniform(-1.0, 1.0, 12000),
                               np.full(12000, ground + 2.60) + rng.normal(0, 0.01, 12000)])
    cloud = np.vstack([chair, pole, ceiling])
    seed = np.array([0.0, 0.0, ground + 0.5])

    def footprint(box):
        return max(2 * box["extent"][0], 2 * box["extent"][1])

    loose, _ = cluster_at(cloud, seed, radius=1.2, voxel=0.06, z_min=ground + 0.03)
    assert footprint(fit_box(loose, ground_z=ground)) > 1.5, (
        'the pole should carry the cluster up into the ceiling')

    gated, info = cluster_at(cloud, seed, radius=1.2, voxel=0.06,
                             z_min=ground + 0.03, z_max=ground + 2.0)
    box = fit_box(gated, ground_z=ground)
    assert footprint(box) < 0.7, box
    assert abs(box["location"][0]) < 0.35 and abs(box["location"][1]) < 0.35, box
    assert info["points_in_cluster"] < len(cloud)



def _chair_on_a_floor(rng, cx=0.0, cy=0.0, ground=-0.66, with_chair=True):
    """A thick, noisy floor — real floors are centimetres deep — and optionally a
    chair standing on it."""
    n = 40000
    parts = [np.column_stack([cx + rng.uniform(-2, 2, n), cy + rng.uniform(-2, 2, n),
                              ground + rng.uniform(0.0, 0.06, n)])]
    if with_chair:
        m = 4000
        parts.append(np.column_stack([cx + rng.uniform(-0.25, 0.25, m),
                                      cy + rng.uniform(-0.22, 0.22, m),
                                      rng.uniform(ground, ground + 0.90, m)]))
    return np.vstack(parts)


def test_clustering_does_not_spread_across_a_thick_floor():
    """A floor is not a plane, it is a slab centimetres deep with returns
    throughout. Gate the cluster only 3 cm above the ground and the chair's legs
    connect it to the whole floor, and the fitted box is the radius, 2.4 m
    across and 10 cm tall."""
    from ros2opv2v.statics import cluster_at, fit_box
    ground = -0.66
    cloud = _chair_on_a_floor(np.random.default_rng(29), ground=ground)
    seed = np.array([0.0, 0.0, ground + 0.65])

    grazing, _ = cluster_at(cloud, seed, radius=1.2, voxel=0.06, z_min=ground + 0.03)
    flat = fit_box(grazing, ground_z=ground)
    assert max(2 * flat["extent"][0], 2 * flat["extent"][1]) > 1.5, (
        'the floor should swallow the cluster at a 3 cm gate')

    gated, _ = cluster_at(cloud, seed, radius=1.2, voxel=0.06, z_min=ground + 0.15)
    box = fit_box(gated, ground_z=ground)
    assert max(2 * box["extent"][0], 2 * box["extent"][1]) < 0.7, box
    assert 0.8 < 2 * box["extent"][2] < 1.0, box
    assert abs(box["location"][0]) < 0.1 and abs(box["location"][1]) < 0.1, box


def test_seed_report_says_when_the_object_is_not_in_this_cloud():
    """The failure that produced a floor-shaped box: a seed read off one cloud
    and clustered in another. The height profile at the seed shows it — an empty
    column above the floor means the object is not in this file, and no amount
    of tuning will fit a box to it."""
    from ros2opv2v.statics import seed_report
    ground = -0.66
    rng = np.random.default_rng(31)
    seed = np.array([0.0, 0.0, ground + 0.65])

    here = seed_report(_chair_on_a_floor(rng, ground=ground), seed, ground)
    standing = sum(c for lo, _hi, c in here["bands"] if lo >= 0.15)
    assert standing > 500, here
    assert here["tallest_m"] > 0.8
    assert here["nearest_m"] < 0.1

    missing = seed_report(_chair_on_a_floor(rng, ground=ground, with_chair=False),
                          seed, ground)
    assert missing["points"] > 0, 'the floor is still there'
    assert sum(c for lo, _hi, c in missing["bands"] if lo >= 0.15) == 0, missing
    assert missing["tallest_m"] < 0.15
    assert missing["nearest_m"] > 0.5, 'the nearest return is a floor point below'
    assert missing["seed_height_m"] == 0.65



def test_probe_names_the_stage_that_still_has_the_furniture():
    """A map pipeline leaves a directory of stages that all look alike and differ
    in whether the furniture survived. Probing has to separate them by what
    stands at the seeds, not by size or bounds — which are nearly identical."""
    import importlib
    import subprocess
    label_static = importlib.import_module('label_static')
    from ros2opv2v.writers import write_pcd
    rng = np.random.default_rng(43)
    ground = -0.659
    n = 60000
    floor = np.column_stack([rng.uniform(-6, 6, n), rng.uniform(-6, 6, n),
                             ground + rng.uniform(0.0, 0.06, n)])

    def chair(cx, cy, m=6000):
        return np.column_stack([cx + rng.uniform(-0.26, 0.26, m),
                                cy + rng.uniform(-0.22, 0.22, m),
                                rng.uniform(ground, ground + 0.90, m)])

    tmp = tempfile.mkdtemp()
    try:
        stages = os.path.join(tmp, 'stages')
        os.makedirs(stages)
        for name, arr in (('a_raw.pcd', np.vstack([floor, chair(1.0, -2.0), chair(3.5, 1.0)])),
                          ('b_filtered.pcd', floor),
                          ('c_anchored.pcd', np.vstack([floor, chair(1.0, -2.0)]))):
            write_pcd(os.path.join(stages, name),
                      np.column_stack([arr, np.zeros(len(arr))]).astype(np.float32))
        script = os.path.join(os.path.dirname(os.path.abspath(label_static.__file__)),
                              'label_static.py')
        run = subprocess.run([sys.executable, script, '--pcd', stages,
                              '--seed', '1.0,-2.0,-0.01', '--seed', '3.5,1.0,-0.01'],
                             capture_output=True, text=True)
        assert run.returncode == 0, run.stderr[-800:]
        report = {}
        current = None
        for line in run.stdout.splitlines():
            if line.endswith('.pcd'):
                current = line.strip()
                report[current] = []
            elif current and line.strip().startswith('seed '):
                report[current].append('OK' if line.rstrip().endswith('OK') else 'empty')
        assert report['a_raw.pcd'] == ['OK', 'OK'], report
        assert report['b_filtered.pcd'] == ['empty', 'empty'], report
        assert report['c_anchored.pcd'] == ['OK', 'empty'], report
    finally:
        shutil.rmtree(tmp, ignore_errors=True)



def _write_binary_pcd(path, xyz, rgb=True):
    """A PCL-style binary PCD with x y z rgb, written by hand so a test can
    truncate it."""
    n = len(xyz)
    fields = "x y z rgb" if rgb else "x y z"
    size = "4 4 4 4" if rgb else "4 4 4"
    kind = "F F F U" if rgb else "F F F"
    header = (
        "# .PCD v0.7 - Point Cloud Data file format\nVERSION 0.7\n"
        "FIELDS %s\nSIZE %s\nTYPE %s\nCOUNT %s\n"
        "WIDTH %d\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS %d\nDATA binary\n"
        % (fields, size, kind, " ".join(["1"] * (4 if rgb else 3)), n, n))
    cols = [xyz[:, 0].astype("<f4"), xyz[:, 1].astype("<f4"), xyz[:, 2].astype("<f4")]
    if rgb:
        cols.append(np.zeros(n, dtype="<u4"))
    body = np.empty(n, dtype=[(f, c.dtype) for f, c in
                              zip(fields.split(), cols)])
    for f, c in zip(fields.split(), cols):
        body[f] = c
    with open(path, "wb") as handle:
        handle.write(header.encode("ascii"))
        handle.write(body.tobytes())
    return header


def test_a_truncated_pcd_is_an_error_not_a_smaller_cloud():
    """The failure this comes from: a file whose header declared 11.6 M points
    while its payload held 1.19 M. Read silently, that is a map with 90% of its
    contents missing — and a missing chair looks exactly like a chair that was
    filtered out, so the operator hunts the wrong problem for hours."""
    from ros2opv2v.statics import StaticsError, read_pcd_xyz
    rng = np.random.default_rng(47)
    xyz = rng.uniform(-5, 5, size=(5000, 3))
    tmp = tempfile.mkdtemp()
    try:
        path = os.path.join(tmp, 'full.pcd')
        _write_binary_pcd(path, xyz)
        assert len(read_pcd_xyz(path)) == 5000

        cut = os.path.join(tmp, 'cut.pcd')
        blob = open(path, 'rb').read()
        keep = blob.index(b'DATA binary\n') + len(b'DATA binary\n')
        open(cut, 'wb').write(blob[:keep] + blob[keep:keep + 500 * 16])
        try:
            read_pcd_xyz(cut)
        except StaticsError as exc:
            message = str(exc)
            assert '5000' in message and '500' in message, message
            assert 'truncated' in message, message
        else:
            raise AssertionError('a truncated PCD must not read as a smaller cloud')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)



def test_refitting_a_named_object_replaces_it_instead_of_duplicating():
    """Re-running the same seeds with the same --out must leave two labels, not
    four. A duplicate box is a phantom object in every frame of the dataset, and
    a phantom scores as a miss against every detector that gets it right."""
    import importlib
    import subprocess
    label_static = importlib.import_module('label_static')
    from ros2opv2v.writers import write_pcd
    rng = np.random.default_rng(53)
    ground = -0.719
    n = 60000
    floor = np.column_stack([rng.uniform(-6, 6, n), rng.uniform(-6, 6, n),
                             ground + rng.uniform(0.0, 0.06, n)])

    def chair(cx, cy, m=6000):
        return np.column_stack([cx + rng.uniform(-0.26, 0.26, m),
                                cy + rng.uniform(-0.22, 0.22, m),
                                rng.uniform(ground, ground + 0.90, m)])

    tmp = tempfile.mkdtemp()
    try:
        pcd = os.path.join(tmp, 'map.pcd')
        arr = np.vstack([floor, chair(1.0, -2.0), chair(3.5, 1.0)])
        write_pcd(pcd, np.column_stack([arr, np.zeros(len(arr))]).astype(np.float32))
        out = os.path.join(tmp, 'statics.json')
        script = os.path.join(os.path.dirname(os.path.abspath(label_static.__file__)),
                              'label_static.py')
        argv = [sys.executable, script, '--pcd', pcd, '--out', out,
                '--seed', '1.0,-2.0,-0.05', '--name', 'chair_1',
                '--seed', '3.5,1.0,-0.05', '--name', 'chair_2']
        for _ in range(3):
            run = subprocess.run(argv, capture_output=True, text=True)
            assert run.returncode == 0, run.stdout[-1500:] + run.stderr[-800:]
        labels = json.load(open(out))
        assert len(labels) == 2, [l['name'] for l in labels]
        assert sorted(l['name'] for l in labels) == ['chair_1', 'chair_2']
        assert sorted(l['id'] for l in labels) == [1, 2], labels
        # a third, differently named object still gets its own id
        run = subprocess.run(argv[:6] + ['--seed', '1.0,-2.0,-0.05', '--name', 'other'],
                             capture_output=True, text=True)
        assert run.returncode == 0, run.stderr[-800:]
        labels = json.load(open(out))
        assert len(labels) == 3, [l['name'] for l in labels]
        assert 'within 0.25 m' in run.stdout, 'two labels on one object should be flagged'
    finally:
        shutil.rmtree(tmp, ignore_errors=True)



def _tiny_dataset(root, agents=('1', '2', '-1'), ego='1', frames=3):
    """A minimal OPV2V tree, enough for the validator to read."""
    from ros2opv2v.writers import write_pcd
    scenario = os.path.join(root, 'scen')
    for agent in agents:
        folder = os.path.join(scenario, agent)
        os.makedirs(folder)
        for i in range(frames):
            key = '%06d' % i
            params = {
                'ego': agent == ego,
                'lidar_pose': [float(i), 0.0, 0.0, 0.0, 0.0, 0.0],
                'ego_speed': 0.0,
                'vehicles': {1: {'location': [3.0, -1.0, -0.2], 'center': [0, 0, 0],
                                 'extent': [0.4, 0.37, 0.48], 'angle': [0.0, 158.2, 0.0]}},
            }
            with open(os.path.join(folder, key + '.yaml'), 'w') as handle:
                yaml.safe_dump(params, handle)
            pts = np.zeros((10, 4), dtype=np.float32)
            pts[:, 0] = np.arange(10)
            write_pcd(os.path.join(folder, key + '.pcd'), pts)
    return scenario


def test_validator_predicts_the_ego_opencood_will_actually_pick():
    """OpenCOOD's basedataset (checked against main) sorts the agent folders,
    moves a LEADING negative id to the end, and takes the first remaining folder
    as ego; it never reads the `ego` flag. So '-1', '1', '2' with ego '1' is
    fine, the same tree with ego '2' is a silent mis-assignment, and a second
    negative id is a vehicle in OpenCOOD's eyes."""
    import importlib
    validate = importlib.import_module('validate_opv2v')

    class Args:
        sample = 3
        read_pcd = False
        with_open3d = False
        range = validate.DEFAULT_RANGE

    def run(agents, ego):
        tmp = tempfile.mkdtemp()
        try:
            root = os.path.join(tmp, 'test')
            os.makedirs(root)
            _tiny_dataset(root, agents=agents, ego=ego)
            findings = validate.Findings()
            validate.check_scenario(os.path.join(root, 'scen'), findings, Args())
            assert not findings.errors, findings.errors
            return findings.warnings
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    # the conventional layout: RSU at -1, ego is the smallest non-negative id
    assert not any('ego' in w for w in run(('1', '2', '-1'), ego='1'))
    # the flag says 2 but OpenCOOD will take 1
    assert any('OpenCOOD will take 1' in w for w in run(('1', '2', '-1'), ego='2'))
    # two RSUs: only the first negative id is moved
    assert any('negative ids' in w for w in run(('1', '-1', '-2'), ego='1'))


def test_with_opencood_builds_the_real_dataset_and_reports_failures():
    """--with-opencood used to check only that the import worked, while calling
    itself the definitive check. Stub OpenCOOD to prove it now constructs the
    dataset, reads samples, and reports what the loader says rather than what we
    believe it would say."""
    import importlib
    import types
    validate = importlib.import_module('validate_opv2v')

    built = {}

    def install_stub(behaviour):
        pkg = types.ModuleType('opencood')
        data_utils = types.ModuleType('opencood.data_utils')
        datasets = types.ModuleType('opencood.data_utils.datasets')
        late = types.ModuleType('opencood.data_utils.datasets.late_fusion_dataset')
        hypes = types.ModuleType('opencood.hypes_yaml')
        yaml_utils = types.ModuleType('opencood.hypes_yaml.yaml_utils')

        class LateFusionDataset(object):
            def __init__(self, params, visualize, train):
                built['params'] = params
                if behaviour == 'refuse':
                    raise KeyError('validate_dir')

            def __len__(self):
                return 0 if behaviour == 'empty' else 12

            def __getitem__(self, index):
                if behaviour == 'read_error':
                    raise RuntimeError('bad pcd at %d' % index)
                mask = np.zeros(10) if behaviour == 'no_gt' else np.ones(10)
                return {'ego': {'object_bbx_mask': mask}}

        late.LateFusionDataset = LateFusionDataset
        yaml_utils.load_yaml = lambda path: {'train_params': {'max_cav': 5}}
        for name, module in (('opencood', pkg), ('opencood.data_utils', data_utils),
                             ('opencood.data_utils.datasets', datasets),
                             ('opencood.data_utils.datasets.late_fusion_dataset', late),
                             ('opencood.hypes_yaml', hypes),
                             ('opencood.hypes_yaml.yaml_utils', yaml_utils)):
            sys.modules[name] = module

    def clear_stub():
        for name in [n for n in sys.modules if n == 'opencood' or n.startswith('opencood.')]:
            del sys.modules[name]

    tmp = tempfile.mkdtemp()
    try:
        config = os.path.join(tmp, 'hypes.yaml')
        open(config, 'w').write('{}\n')

        install_stub('ok')
        try:
            findings = validate.Findings()
            validate.check_with_opencood('/data/test', findings, config)
            assert not findings.errors, findings.errors
            assert any('12 samples' in n for n in findings.notes), findings.notes
            assert any('accepts this dataset' in n for n in findings.notes), findings.notes
            # the tree under test must be what got loaded, not the config's own dirs
            assert built['params']['validate_dir'] == '/data/test'
            assert built['params']['root_dir'] == '/data/test'

            # no config: refuse to pretend, rather than guess the preprocessor
            quiet = validate.Findings()
            validate.check_with_opencood('/data/test', quiet, None)
            assert not quiet.errors
            assert any('--opencood-config' in w for w in quiet.warnings), quiet.warnings
        finally:
            clear_stub()

        for behaviour, needle in (('refuse', 'refused this tree'),
                                  ('empty', 'length 0'),
                                  ('read_error', 'failed reading sample')):
            install_stub(behaviour)
            try:
                findings = validate.Findings()
                validate.check_with_opencood('/data/test', findings, config)
                assert any(needle in e for e in findings.errors), (behaviour, findings.errors)
            finally:
                clear_stub()

        install_stub('no_gt')
        try:
            findings = validate.Findings()
            validate.check_with_opencood('/data/test', findings, config)
            assert any('no GT boxes' in w for w in findings.warnings), findings.warnings
        finally:
            clear_stub()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)



def test_rotating_a_depth_cloud_rotates_the_pose_written_beside_it():
    """`optical_frame` has to move the points AND the frame they are declared
    to be in. Moving only the points is self-consistent nowhere but validates
    cleanly: the pcd is body-axes while lidar_pose describes an optical frame,
    so every box projected through that pose lands 90 degrees out."""
    from ros2opv2v.geometry import OPTICAL_TO_FLU, invert, make_transform
    base = cfgmod.CloudConfig('depth_image', '/d', make_transform(
        x=0.06, y=0.0, z=0.0, roll=0.0, pitch=0.0, yaw=0.0, degrees=True))

    base.optical_frame = True
    assert base.points_rotated_to_body()
    assert np.allclose(base.frame_extrinsic(), base.extrinsic @ invert(OPTICAL_TO_FLU))

    base.optical_frame = False
    assert not base.points_rotated_to_body()
    assert np.allclose(base.frame_extrinsic(), base.extrinsic)

    # a PointCloud2 is used in the frame its header names, whatever the flag says
    pc = cfgmod.CloudConfig('pointcloud2', '/p', np.identity(4))
    pc.optical_frame = True
    assert not pc.points_rotated_to_body()
    assert np.allclose(pc.frame_extrinsic(), pc.extrinsic)


def test_a_depth_agents_written_cloud_has_the_floor_below_it():
    """The end-to-end property, stated the way the bug showed up in the data:
    with a camera looking level at a room, the written cloud must put the floor
    at negative z and what is ahead at positive x. In optical axes the forward
    axis is z, which is what the real dataset showed (z 0.65..8.74, nothing
    below zero) before the flag moved the pose too."""
    from ros2opv2v.geometry import OPTICAL_TO_FLU, invert, transform_points
    cloud = cfgmod.CloudConfig('depth_image', '/d', np.identity(4))
    cloud.optical_frame = True

    # points as a depth camera produces them: x right, y down, z forward.
    # a floor 0.9 m below, 2..6 m ahead
    rng = np.random.default_rng(61)
    n = 2000
    optical = np.column_stack([rng.uniform(-2, 2, n), np.full(n, 0.9),
                               rng.uniform(2, 6, n)])
    body = transform_points(optical, OPTICAL_TO_FLU)
    assert body[:, 2].max() < -0.85, 'floor must sit below the sensor'
    assert body[:, 0].min() > 1.9, 'what is ahead must be +x'

    # and the declared frame turns by the same rotation, so world positions hold
    world_from_base = np.identity(4)
    world_from_sensor_points = world_from_base @ cloud.frame_extrinsic()
    world_from_sensor_raw = world_from_base @ cloud.extrinsic
    assert np.allclose(transform_points(body, world_from_sensor_points),
                       transform_points(optical, world_from_sensor_raw), atol=1e-9), \
        'rotating the points must not move them in the world'



def test_the_anchor_check_is_skipped_on_a_window_from_mid_bag():
    """pose.expected_start names where an agent stood at the START of the bag.
    Convert a window from further in and the agent has legitimately driven away,
    so the check measures the drive, not the frame — it refused a good 6 s slice
    at t=40 s over 7.00 m, which is how far that cart had walked."""
    import importlib
    convertmod = importlib.import_module('ros2opv2v.convert')
    cfg = cfgmod.load_config(os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'configs', 'mirc_coop2.yaml'))
    report = convertmod.ConversionReport(bag=cfg.bag)
    report.pose_stats = {a.name: {'start_m': [99.0, 99.0, 99.0]}
                         for a in cfg.active_agents}

    cfg.time.start_offset_s = 0.0
    try:
        convertmod._check_expected_starts(cfg, report)
    except convertmod.ConversionError:
        pass
    else:
        raise AssertionError('a wrong start at t=0 is a real frame error')

    cfg.time.start_offset_s = 40.0
    before = len(report.warnings)
    convertmod._check_expected_starts(cfg, report)      # must not raise
    assert len(report.warnings) == before + 1
    assert 'not checked' in report.warnings[-1]



def test_obj_type_reaches_the_frame_yaml():
    """InCoP's loader reads a class from obj_type and falls back to id 0 --
    potted_plant -- without one, so an unlabelled chair is not unclassified, it
    is confidently a plant."""
    from ros2opv2v.labels import merge_external_labels
    out = merge_external_labels({}, [
        {"id": 1, "location": [3.1, -1.4, -0.24], "extent": [0.4, 0.37, 0.48],
         "angle": [0.0, 158.2, 0.0], "obj_type": "chair"},
        {"id": 2, "location": [4.4, -8.1, -0.28], "extent": [0.34, 0.3, 0.44],
         "angle": [0.0, 2.0, 0.0]},
    ])
    assert out[1]["obj_type"] == "chair"
    assert "obj_type" not in out[2], 'an unset class must not be invented'


def test_incop_sidecars_cover_every_scenario_and_agent():
    import importlib
    sidecars = importlib.import_module('make_incop_sidecars')
    tmp = tempfile.mkdtemp()
    try:
        for split, agents in (('train', ('0', '1')), ('validate', ('0', '1'))):
            for agent in agents:
                os.makedirs(os.path.join(tmp, split, 'case_0', agent))
        found = sidecars.scenarios(tmp)
        assert found == {'case_0': ['0', '1']}, found

        # the same scenario with different agents in two splits cannot be
        # described by one modality map, and silently picking one would put a
        # KeyError deep inside the loader instead of here
        os.makedirs(os.path.join(tmp, 'test', 'case_0', '2'))
        try:
            sidecars.scenarios(tmp)
        except SystemExit as exc:
            assert 'cannot cover both' in str(exc)
        else:
            raise AssertionError('mismatched agent sets must be refused')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_incop_config_tracks_the_opv2v_one_where_it_matters():
    """Two configs, one bag. Everything geometric must be identical; only the
    packaging differs. A drift in an extrinsic between them would show up as a
    difference between the OPV2V and InCoP arms that is not about the models."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    base = cfgmod.load_config(os.path.join(root, 'configs', 'mirc_coop2.yaml'))
    incop = cfgmod.load_config(os.path.join(root, 'configs', 'mirc_coop2_incop.yaml'))

    assert base.bag == incop.bag
    assert cfgmod.ego_agent(incop).name == 'mobile_2'
    assert cfgmod.ego_agent(incop).cav_id == 0
    assert sorted(a.cav_id for a in incop.active_agents) == [0, 1]
    assert all(a.name != 'infra_1' for a in incop.active_agents)
    assert incop.output.scenario_name == 'case_0'

    for name in ('mobile_1', 'mobile_2'):
        a = next(x for x in base.active_agents if x.name == name)
        b = next(x for x in incop.active_agents if x.name == name)
        assert np.allclose(a.cloud.extrinsic, b.cloud.extrinsic), name
        assert np.allclose(a.cloud.frame_extrinsic(), b.cloud.frame_extrinsic()), name
        assert a.cloud.optical_frame == b.cloud.optical_frame, name
        assert np.allclose(a.pose.align, b.pose.align), name
        assert len(a.cameras) == len(b.cameras), name
        for ca, cb in zip(a.cameras, b.cameras):
            assert np.allclose(ca.extrinsic, cb.extrinsic), name



def test_the_incop_output_path_selects_left_hand_visualisation():
    """inference_isaac.py picks BEV handedness from the dataset PATH: left_hand
    is true when it contains OPV2V, V2XSET or REAL_WORLD, and this data follows
    that lidar convention. A path matching none of them mirrors every box in the
    video while the metrics stay correct — right numbers, wrong picture."""
    cfg = cfgmod.load_config(os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'configs', 'mirc_coop2_incop.yaml'))
    upper = cfg.output.root.upper()
    assert any(key in upper for key in ('OPV2V', 'V2XSET', 'REAL_WORLD')), cfg.output.root



def test_camera_images_are_written_rgb_not_rgba():
    """The ZED publishes bgra8. Carrying the alpha through gives a 4-channel PNG,
    and every model that reads it normalises with a 3-channel mean and std, so it
    dies with a tensor shape mismatch deep inside torchvision — which looks like
    a model problem and is not one."""
    from ros2opv2v.writers import image_to_array

    class Msg:
        def __init__(self, encoding, data, height, width, step):
            self.encoding, self.data = encoding, data
            self.height, self.width, self.step = height, width, step

    height, width = 3, 4
    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    rgba[..., 0], rgba[..., 1], rgba[..., 2], rgba[..., 3] = 10, 20, 30, 255

    out = image_to_array(Msg('rgba8', rgba.tobytes(), height, width, width * 4))
    assert out.shape == (height, width, 3), out.shape
    assert out[0, 0].tolist() == [10, 20, 30]

    bgra = rgba[:, :, [2, 1, 0, 3]]
    out = image_to_array(Msg('bgra8', bgra.tobytes(), height, width, width * 4))
    assert out.shape == (height, width, 3), out.shape
    assert out[0, 0].tolist() == [10, 20, 30], 'bgra must come back as rgb'

    rgb = np.dstack([np.full((height, width), v, np.uint8) for v in (10, 20, 30)])
    out = image_to_array(Msg('rgb8', rgb.tobytes(), height, width, width * 3))
    assert out.shape == (height, width, 3) and out[0, 0].tolist() == [10, 20, 30]



def test_a_frame_missing_its_camera_is_dropped_like_a_missing_cloud():
    """Cameras were matched AFTER the completeness decision, so a frame could be
    written with a cloud and a yaml but no image. Anything LiDAR-only loads it
    happily; anything multimodal dies 278 frames into a run on a bare
    FileNotFoundError, with nothing to say the dataset was built that way."""
    from ros2opv2v.sync import StampIndex, build_frame_table

    ns = 1_000_000_000
    times = [i * ns // 10 for i in range(10)]          # 10 Hz grid
    clouds = {'ego': StampIndex(times), 'partner': StampIndex(times)}

    # the ego's camera drops out for frames 4 and 5
    gap = [t for i, t in enumerate(times) if i not in (4, 5)]
    cameras = {('ego', 0): StampIndex(gap), ('partner', 0): StampIndex(times)}

    table = build_frame_table(times, clouds, {'ego': True, 'partner': True},
                              ns // 40, 'ego', drop_incomplete=True,
                              camera_indices=cameras)
    assert len(table.frames) == 8, len(table.frames)
    assert any('camera0' in key for key in table.dropped), table.dropped
    # every surviving frame carries an image for every agent that has a camera
    for frame in table.frames:
        assert set(frame.camera_stamps) == {('ego', 0), ('partner', 0)}, frame.camera_stamps

    # an OPTIONAL agent's missing camera must not drop the frame
    table = build_frame_table(times, clouds, {'ego': False, 'partner': True},
                              ns // 40, 'ego', drop_incomplete=True,
                              camera_indices=cameras)
    assert len(table.frames) == 10, len(table.frames)



def test_a_chair_against_a_wall_is_separable_by_column_height_only():
    """A chair touching a wall is ONE connected blob with it at every voxel size,
    and no radius excludes the wall without clipping the chair. What differs is
    the column: the chair's tops out below the object band, the wall's carries
    past it. The real case measured 1.33 x 0.73 x 2.00 m — a slab of wall."""
    from ros2opv2v.statics import cluster_at, fit_box, without_structure
    rng = np.random.default_rng(67)
    ground = -0.719
    # a wall in the x = 0.0..0.1 slab, floor to 3.3 m
    wall = np.column_stack([rng.uniform(0.0, 0.10, 30000), rng.uniform(-2, 2, 30000),
                            rng.uniform(ground, ground + 3.3, 30000)])
    # a chair standing against it, 0.10..0.75 in x
    chair = np.column_stack([rng.uniform(0.10, 0.75, 8000),
                             rng.uniform(-0.35, 0.35, 8000),
                             rng.uniform(ground, ground + 0.90, 8000)])
    floor = np.column_stack([rng.uniform(-1, 3, 20000), rng.uniform(-2, 2, 20000),
                             ground + rng.uniform(0, 0.05, 20000)])
    cloud = np.vstack([wall, chair, floor])
    seed = np.array([0.45, 0.0, ground + 0.57])

    def fit(points):
        blob, _ = cluster_at(points, seed, radius=0.7, voxel=0.06,
                             z_min=ground + 0.15, z_max=ground + 2.0)
        return fit_box(blob, ground_z=ground)

    swallowed = fit(cloud)
    assert 2 * swallowed["extent"][2] > 1.8, 'the wall should reach the height gate'

    kept = without_structure(cloud, ground, 1.30, 0.10)
    assert len(kept) < len(cloud), 'the wall columns must be dropped'
    box = fit(kept)
    assert 2 * box["extent"][2] < 1.1, box
    assert max(2 * box["extent"][0], 2 * box["extent"][1]) < 0.9, box
    assert box["location"][0] > 0.15, 'the box must sit on the chair, not the wall'


def test_dropping_tall_columns_leaves_a_free_standing_object_untouched():
    from ros2opv2v.statics import without_structure
    rng = np.random.default_rng(71)
    ground = -0.66
    chair = np.column_stack([rng.uniform(-0.3, 0.3, 4000), rng.uniform(-0.3, 0.3, 4000),
                             rng.uniform(ground, ground + 0.9, 4000)])
    assert len(without_structure(chair, ground, 1.30, 0.10)) == len(chair)



def test_resized_images_carry_a_matching_intrinsic():
    """A lift-splat branch resizes whatever it is handed to the size ITS config
    declares, then corrects the intrinsic by ONE scalar. That correction is only
    true when the image already had the declared shape: 1280x720 into a config
    expecting 1280x800 leaves the vertical focal length 11% wrong, and 960x540
    leaves it 48% wrong, with nothing raised anywhere."""
    from ros2opv2v.writers import resize_image, scale_intrinsic

    rng = np.random.default_rng(73)
    image = rng.integers(0, 256, size=(720, 1280, 3), dtype=np.uint8)
    out = resize_image(image, 1280, 800)
    assert out.shape == (800, 1280, 3) and out.dtype == np.uint8
    assert np.allclose(resize_image(image, 1280, 720), image), 'a no-op must not resample'

    # a corner stays a corner, and a flat image stays flat
    flat = np.full((540, 960), 77, dtype=np.uint8)
    assert (resize_image(flat, 1280, 800) == 77).all()

    k = [[640.0, 0.0, 630.0], [0.0, 645.0, 350.0], [0.0, 0.0, 1.0]]
    scaled = scale_intrinsic(k, [1280, 720], [1280, 800])
    assert scaled[0][0] == 640.0 and scaled[0][2] == 630.0, 'width unchanged'
    assert abs(scaled[1][1] - 645.0 * 800 / 720) < 1e-9
    assert abs(scaled[1][2] - 350.0 * 800 / 720) < 1e-9

    # the property that matters: after the model's uniform 0.65, a point at the
    # principal point still maps to the principal point of the resized image
    for source in ([1280, 720], [960, 540]):
        s = scale_intrinsic(k, source, [1280, 800])
        assert abs(s[0][2] / 1280 - k[0][2] / source[0]) < 1e-9
        assert abs(s[1][2] / 800 - k[1][2] / source[1]) < 1e-9


def test_the_incop_config_pins_every_camera_to_the_models_declared_size():
    cfg = cfgmod.load_config(os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'configs', 'mirc_coop2_incop.yaml'))
    cameras = [(a.name, c) for a in cfg.active_agents for c in a.cameras]
    assert cameras, 'the multimodal models need camera images'
    for name, camera in cameras:
        assert camera.output_size == [1280, 800], (name, camera.output_size)



def test_the_label_class_survives_into_every_frame_yaml():
    """The class has to reach the frame yaml, not just the labels file. A reader
    that finds no class does not treat the object as unclassified — it falls
    back to class id 0. On this dataset that made 730 chairs load as
    potted_plant, so every chair prediction was scored against the wrong label
    and the chair AP was undefined rather than bad."""
    import importlib
    convertmod = importlib.import_module('ros2opv2v.convert')
    from ros2opv2v.labels import merge_external_labels

    tmp = tempfile.mkdtemp()
    try:
        path = os.path.join(tmp, 'statics.json')
        json.dump([
            {"id": 1, "name": "chair_1", "location": [3.1, -1.4, -0.24],
             "extent": [0.4, 0.37, 0.48], "angle": [0, 158.2, 0], "obj_type": "chair"},
            {"id": 2, "name": "chair_2", "location": [4.4, -8.1, -0.28],
             "extent": [0.34, 0.3, 0.44], "angle": [0, 2.0, 0]},
        ], open(path, 'w'))

        cfg = cfgmod.load_config(os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'configs', 'mirc_coop2.yaml'))
        cfg.output.labels_file = path
        report = convertmod.ConversionReport(bag=cfg.bag)
        loaded = convertmod.load_static_labels(cfg, report)

        assert loaded[0]["obj_type"] == "chair", loaded[0]
        assert "obj_type" not in loaded[1], 'an unset class must not be invented'
        assert any('no obj_type' in w and 'chair_2' in w for w in report.warnings), \
            report.warnings

        vehicles = merge_external_labels({}, loaded)
        assert vehicles[1]["obj_type"] == "chair"
        assert "obj_type" not in vehicles[2]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == '__main__':
    tests = [(k, v) for k, v in sorted(globals().items())
             if k.startswith('test_') and callable(v)]
    for name, fn in tests:
        check(name, fn)
    print('\n%d/%d passed' % (len(tests) - len(FAILED), len(tests)))
    sys.exit(1 if FAILED else 0)
