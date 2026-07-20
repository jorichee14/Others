#!/usr/bin/env python3
"""
Radar point-cloud MERGE  —  deployment-time fusion, NO calibration rig
======================================================================
You already calibrated each radar against the SAME camera and saved a
`T_cam_radar` per radar (the JSON/YAML that `radar_camera_calib.py` writes).
This node just USES those extrinsics: it transforms every radar's cloud into
one shared frame and concatenates them. No ChArUco board, no corner reflector,
no solving — calibration is done; this is the runtime step.

    p_cam = R_i · p_radar_i + t_i            (T_cam_radar_i, from radar i's calib)

To merge into a common `target_frame`:

    T_cam_target  = I                if target_frame == the shared camera (parent)
                  = T_cam_radar_j    if target_frame == radar j's link
    T_target_radar_i = inv(T_cam_target) · T_cam_radar_i

Each radar's points are pushed through `T_target_radar_i` and stacked. The
merged cloud keeps `intensity` (SNR) and `doppler`, and adds a `source` field
(the radar index) so a consumer can tell which radar a point came from — which
also disambiguates `doppler`, since Doppler is a RADIAL velocity measured along
each radar's own line of sight and stays relative to that radar after the
rigid transform (the rotation moves the position, not the scalar range-rate).

Why this needs no rig
─────────────────────
Calibration recovered the geometry once; merging is pure composition of known
rigid transforms. The only thing that can drift is the physical mounting — if a
radar is bumped, re-run `radar_camera_calib.py` for that unit and drop in the
new JSON. Nothing here estimates anything.

Two radars mounted ORTHOGONALLY (one rolled ~90° about boresight) complement
each other: radar A resolves azimuth well / elevation poorly, radar B the
reverse. Merged in a common frame you get points that together cover both the
azimuth and elevation FoV — the elevation-weakness of a single IWR6843ISK
(±20° FoV) is filled by the other unit's strong axis. This node does the
geometric merge; if you later want a single de-duplicated 3-D point per target
(azimuth from A, elevation from B) that is a measurement-level association step
on top of this merged cloud.

Usage
─────
    python3 radar_merge.py --ros-args \
      -p extrinsic_files:="['/path/extrinsic_zed_left__radar1.json','/path/extrinsic_zed_left__radar2.json']" \
      -p radar_topics:="['/radar1/radar/points_all','/radar2/radar/points_all']" \
      -p target_frame:=zed_left_camera_optical_frame \
      -p output_topic:=/radar_merged/points

`extrinsic_files` and `radar_topics` are index-matched (file[i] calibrates the
radar publishing topic[i]). `target_frame` empty → the shared camera/parent
frame from the first file. A `.yaml` path auto-prefers its `.json` sibling.

Deps:  numpy scipy  +  ROS2 rclpy sensor_msgs_py tf2_ros
"""
import ast
import json
import os
import numpy as np
from scipy.spatial.transform import Rotation as Rot
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
import tf2_ros
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2, PointField
from geometry_msgs.msg import TransformStamped

try:
    from sensor_msgs_py import point_cloud2 as pc2
    _HAVE_PC2 = True
except Exception:
    _HAVE_PC2 = False


def load_extrinsic(path):
    """Load a calib result (from radar_camera_calib.py) → dict with a 4×4
    T_cam_radar, frames, and any ingest range correction.

    Prefers the .json sibling (proper JSON); falls back to parsing the flat
    `key: value` .yaml the tool also writes (values are Python literals)."""
    jpath = path
    if path.endswith('.yaml') and os.path.exists(path[:-5] + '.json'):
        jpath = path[:-5] + '.json'
    with open(jpath) as f:
        text = f.read()
    try:
        d = json.loads(text)
    except Exception:
        # flat "key: value" fallback — values are Python reprs
        d = {}
        for line in text.splitlines():
            if ':' not in line:
                continue
            k, _, v = line.partition(':')
            v = v.strip()
            try:
                d[k.strip()] = ast.literal_eval(v)
            except Exception:
                d[k.strip()] = v

    t = np.asarray(d['T_cam_radar_translation'], float)
    q = np.asarray(d['T_cam_radar_quaternion_xyzw'], float)  # x,y,z,w
    T = np.eye(4)
    T[:3, :3] = Rot.from_quat(q).as_matrix()
    T[:3, 3] = t
    return {
        'T_cam_radar': T,
        'parent_frame': d.get('parent_frame', ''),
        'child_frame': d.get('child_frame', ''),
        'range_scale': float(d.get('radar_range_scale', 1.0)),
        'range_bias': float(d.get('radar_range_bias_m', 0.0)),
        'path': jpath,
    }


def _range_correct(xyz, scale, bias):
    """Same ingest correction the calibrator applied, so merged points match
    the frame the extrinsic was solved in."""
    if scale == 1.0 and bias == 0.0:
        return xyz
    xyz = xyz * float(scale)
    if bias != 0.0:
        r = np.linalg.norm(xyz, axis=1, keepdims=True)
        r = np.where(r < 1e-6, 1e-6, r)
        xyz = xyz - bias * (xyz / r)
    return xyz


class RadarMerge(Node):
    def __init__(self, overrides=None):
        super().__init__('radar_merge')
        gp = self._gp
        if overrides:
            for k, v in overrides.items():
                if not self.has_parameter(k):
                    self.declare_parameter(k, v)

        files = gp('extrinsic_files', [])
        topics = gp('radar_topics', [])
        if not files or not topics:
            raise RuntimeError(
                "set extrinsic_files and radar_topics (index-matched lists)")
        if len(files) != len(topics):
            raise RuntimeError(
                f"extrinsic_files ({len(files)}) and radar_topics "
                f"({len(topics)}) must be the same length")

        # field names (IWR6843ISK points_all: x,y,z,doppler,intensity)
        self.fx = gp('pc_field_x', 'x'); self.fy = gp('pc_field_y', 'y')
        self.fz = gp('pc_field_z', 'z')
        self.fsnr = gp('pc_field_snr', 'intensity')
        self.fdop = gp('pc_field_doppler', 'doppler')

        self.output_topic = gp('output_topic', '/radar_merged/points')
        self.publish_rate = float(gp('publish_rate_hz', 15.0))
        self.max_age = float(gp('max_age_s', 0.25))
        self.add_source = bool(gp('add_source_field', True))
        self.publish_tf = bool(gp('publish_tf', True))

        # load every extrinsic, resolve the target frame
        self.radars = [load_extrinsic(f) for f in files]
        parents = {r['parent_frame'] for r in self.radars}
        if len(parents) > 1:
            self.get_logger().warn(
                f"radars were calibrated against different parent frames "
                f"{parents} — merging assumes ONE shared camera; check your files")
        parent = self.radars[0]['parent_frame']

        target = gp('target_frame', '') or parent
        # T_cam_target : identity if target is the camera(parent), else that radar's extrinsic
        T_cam_target = np.eye(4)
        if target != parent:
            hit = [r for r in self.radars if r['child_frame'] == target]
            if not hit:
                raise RuntimeError(
                    f"target_frame '{target}' is neither the parent '{parent}' "
                    f"nor any radar child {[r['child_frame'] for r in self.radars]}")
            T_cam_target = hit[0]['T_cam_radar']
        T_target_cam = np.linalg.inv(T_cam_target)
        self.target_frame = target

        # per-radar: precompute T_target_radar (R,t), cache slot, subscription
        self.latest = [None] * len(self.radars)          # (structured pts, stamp)
        self.stamps = [0.0] * len(self.radars)
        self._subs = []
        for i, (r, topic) in enumerate(zip(self.radars, topics)):
            T = T_target_cam @ r['T_cam_radar']
            r['R'] = T[:3, :3].astype(np.float64)
            r['t'] = T[:3, 3].astype(np.float64)
            r['topic'] = topic
            self._subs.append(self.create_subscription(
                PointCloud2, topic,
                lambda msg, idx=i: self._on_radar(msg, idx),
                qos_profile_sensor_data))
            self.get_logger().info(
                f"radar[{i}] {topic}  ({r['child_frame']}) → {self.target_frame}  "
                f"|t|={np.linalg.norm(r['t']):.3f} m  src={r['path']}")

        self.pub = self.create_publisher(PointCloud2, self.output_topic, 10)

        if self.publish_tf:
            self._static_tf = tf2_ros.StaticTransformBroadcaster(self)
            self._broadcast_static(parent)

        if self.publish_rate > 0:
            self.create_timer(1.0 / self.publish_rate, self._publish_merged)
            mode = f"timer {self.publish_rate:g} Hz"
        else:
            mode = "on every incoming cloud"
        self.get_logger().info(
            f"merging {len(self.radars)} radars → {self.output_topic} "
            f"in '{self.target_frame}' ({mode}, max_age {self.max_age:g}s)")

    def _gp(self, name, default):
        if not self.has_parameter(name):
            self.declare_parameter(name, default)
        return self.get_parameter(name).value

    def _now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _broadcast_static(self, parent):
        """Publish parent→child for each radar so the raw per-radar clouds also
        display correctly in RViz alongside the merged cloud."""
        tfs = []
        stamp = self.get_clock().now().to_msg()
        for r in self.radars:
            T = r['T_cam_radar']
            q = Rot.from_matrix(T[:3, :3]).as_quat()
            m = TransformStamped()
            m.header.stamp = stamp
            m.header.frame_id = r['parent_frame'] or parent
            m.child_frame_id = r['child_frame']
            m.transform.translation.x = float(T[0, 3])
            m.transform.translation.y = float(T[1, 3])
            m.transform.translation.z = float(T[2, 3])
            m.transform.rotation.x = float(q[0]); m.transform.rotation.y = float(q[1])
            m.transform.rotation.z = float(q[2]); m.transform.rotation.w = float(q[3])
            tfs.append(m)
        if tfs:
            self._static_tf.sendTransform(tfs)

    def _read(self, msg):
        """PointCloud2 → (xyz Nx3, snr N, doppler N), applying no correction."""
        if not _HAVE_PC2:
            return None, None, None
        names = [f.name for f in msg.fields]
        want = [self.fx, self.fy, self.fz]
        has_snr = self.fsnr in names; has_dop = self.fdop in names
        if has_snr:
            want.append(self.fsnr)
        if has_dop:
            want.append(self.fdop)
        try:
            arr = list(pc2.read_points(msg, field_names=want, skip_nans=True))
        except Exception as e:
            self.get_logger().warn(f"read_points failed on {want}: {e}")
            return None, None, None
        if not arr:
            return None, None, None
        arr = np.array([tuple(a) for a in arr], float)
        xyz = arr[:, :3]
        col = 3
        snr = arr[:, col] if has_snr else np.zeros(len(arr)); col += int(has_snr)
        dop = arr[:, col] if has_dop else np.zeros(len(arr))
        return xyz, snr, dop

    def _on_radar(self, msg, idx):
        xyz, snr, dop = self._read(msg)
        if xyz is None or len(xyz) == 0:
            self.latest[idx] = None
            return
        r = self.radars[idx]
        xyz = _range_correct(xyz, r['range_scale'], r['range_bias'])
        pts = (xyz @ r['R'].T) + r['t']                 # → target frame
        n = len(pts)
        block = np.empty((n, 6), np.float32)
        block[:, 0:3] = pts
        block[:, 3] = snr
        block[:, 4] = dop
        block[:, 5] = idx                                # source
        self.latest[idx] = block
        self.stamps[idx] = self._now()
        if self.publish_rate <= 0:
            self._publish_merged()

    def _publish_merged(self):
        now = self._now()
        blocks = [b for i, b in enumerate(self.latest)
                  if b is not None and (now - self.stamps[i]) <= self.max_age]
        if not blocks:
            return
        data = np.vstack(blocks)

        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1),
            PointField(name='doppler', offset=16, datatype=PointField.FLOAT32, count=1),
        ]
        if self.add_source:
            fields.append(PointField(name='source', offset=20,
                                     datatype=PointField.FLOAT32, count=1))
            rows = data
        else:
            rows = data[:, :5]

        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = self.target_frame
        cloud = pc2.create_cloud(header, fields, rows.tolist())
        self.pub.publish(cloud)


def main(overrides=None):
    rclpy.init()
    node = RadarMerge(overrides)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
