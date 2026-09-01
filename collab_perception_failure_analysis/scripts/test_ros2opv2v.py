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
                                matrix_to_opencood_pose, quat_to_matrix,
                                matrix_to_quat, transform_points, x_to_world)
from ros2opv2v.labels import agent_box, vehicles_for_viewer       # noqa: E402
from ros2opv2v.pointclouds import (cloud_from_depth_image,        # noqa: E402
                                   cloud_from_pointcloud2, pointcloud2_to_array)
from ros2opv2v.sync import (PoseTrack, StampIndex,                # noqa: E402
                            build_frame_table, frame_times)
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

def _write_synthetic_bag(path, n_frames=12):
    """Three agents on a 10 Hz grid with exactly known geometry.

    ego (agent 1) drives along +x; agent 2 sits 6 m ahead of the ego's start;
    the RSU is static at (3, 4). Each carries one cloud so the converter has to
    exercise both the PointCloud2 and depth paths.
    """
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

    def cloud_msg(t_ns, frame, points):
        data = np.zeros((len(points), 4), dtype='<f4')
        data[:, :3] = points
        data[:, 3] = 0.5
        fields = [{'name': n, 'offset': 4 * i, 'datatype': 7, 'count': 1}
                  for i, n in enumerate(('x', 'y', 'z', 'intensity'))]
        return {'header': header(t_ns, frame), 'height': 1, 'width': len(points),
                'fields': fields, 'is_bigendian': False, 'point_step': 16,
                'row_step': 16 * len(points), 'data': data.tobytes(),
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

        t0 = 1_700_000_000 * NS
        # A 4x4 depth image whose single valid pixel sits at the principal point.
        depth = np.zeros((4, 4), dtype='<u2')
        depth[2, 2] = 3000
        for i in range(n_frames):
            t = t0 + i * (NS // 10)
            writer.write_message('/ego/points', cloud_schema,
                                 cloud_msg(t, 'ego_lidar',
                                           np.array([[1.0, 0.0, 0.0], [2.0, 1.0, -0.5],
                                                     [3.0, -1.0, 0.2]])),
                                 log_time=t, publish_time=t)
            writer.write_message('/ego/odom', odom_schema,
                                 odom_msg(t, 'ego_odom', 'ego_base',
                                          (0.5 * i, 0.0, 0.0), 5.0),
                                 log_time=t, publish_time=t)
            # agent 2 publishes slightly off-grid: the synchroniser must match it
            t2 = t + 15_000_000
            writer.write_message('/two/depth', image_schema,
                                 {'header': header(t2, 'two_depth_optical'),
                                  'height': 4, 'width': 4, 'encoding': '16UC1',
                                  'is_bigendian': 0, 'step': 8,
                                  'data': depth.tobytes()},
                                 log_time=t2, publish_time=t2)
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
                                 log_time=t2, publish_time=t2)
            writer.write_message('/two/odom', odom_schema,
                                 odom_msg(t2, 'two_odom', 'two_base',
                                          (6.0, 0.0, 0.0), 0.0),
                                 log_time=t2, publish_time=t2)
            writer.write_message('/infra/radar', cloud_schema,
                                 cloud_msg(t, 'infra_radar',
                                           np.array([[4.0, 0.5, 0.1]])),
                                 log_time=t, publish_time=t)
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


if __name__ == '__main__':
    tests = [(k, v) for k, v in sorted(globals().items())
             if k.startswith('test_') and callable(v)]
    for name, fn in tests:
        check(name, fn)
    print('\n%d/%d passed' % (len(tests) - len(FAILED), len(tests)))
    sys.exit(1 if FAILED else 0)
