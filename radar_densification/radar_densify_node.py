#!/usr/bin/env python3
"""
Dense static point-cloud mapping from mmWave radar DETECTIONS
=============================================================

The point-cloud-domain port of DREAM-PCD (arXiv:2309.15374). Their pipeline
runs on raw ADC data; ours starts from what the IWR6843's on-chip CFAR
already emitted (x, y, z, snr, doppler), so each of their stages is replaced
by its detection-domain counterpart:

  their stage                      here
  ------------------------------   -------------------------------------------
  ego-motion compensation          ego velocity / full 6-DOF ego twist fitted
                                   to the Doppler field (RANSAC over static
                                   returns) — "remove ego speed"
  static point extraction          Doppler residual vs the fitted ego motion
  Non-Coherent Accumulation        static points from every frame, transformed
                                   into one world frame, accumulated on a
                                   voxel EVIDENCE grid; a voxel must be seen
                                   in >= min_frames distinct frames to be
                                   emitted (integration gain -> noise dies,
                                   structure persists)
  Synthetic Aperture Accumulation  impossible without phase. Two substitutes:
                                   (a) all THREE calibrated radars fuse into
                                   one map = a physically larger, sparse
                                   aperture; (b) per-detection anisotropic
                                   covariance (sharp range, wide cross-range)
                                   fused in information form, so crossing
                                   viewpoints intersect their error ellipsoids
                                   and cross-range error shrinks with motion
  learned denoiser                 evidence threshold + 26-neighbour radius
                                   outlier filter (classical, no training)

Ego pose source (`ego_mode`):
  tf       lookup map_frame <- base_frame from TF (GLIM / lidar odometry) —
           most accurate, use it when the Ouster is running
  odom     a nav_msgs/Odometry topic (pose of base_frame in map_frame)
  doppler  radar-only dead reckoning from the fitted ego twist. With all
           three radars the joint solve observes angular rate too (the lever
           arms t_k make w visible); with one radar it is translation-only.
           Drift grows with time — fine for short sweeps, prefer tf.
  static   rig does not move; identity pose

Run (defaults are this rig: three IWR6843ISK in the Ouster os_lidar frame,
extrinsics = the 2026-08-19 radar<->lidar solves):

  python3 radar_densify_node.py --ros-args \
    -p ego_mode:=tf -p map_frame:=odom -p base_frame:=os_lidar \
    -p pc_field_snr:=intensity

  ros2 topic pub -1 /radar_densify/save  std_msgs/msg/Empty "{}"
  ros2 topic pub -1 /radar_densify/reset std_msgs/msg/Empty "{}"

Topics out: ~/map (the dense static cloud, republished every
map_publish_period), ~/static_points and ~/dynamic_points (this frame's
classification, map frame, for RViz sanity checks).

Doppler sign: TI reports range-rate positive for a RECEDING target, so a
static world point obeys d = -(u . v_sensor); `doppler_sign:=-1` (default)
encodes that. Classification is sign-invariant — the sign only matters for
the direction of dead-reckoned motion in ego_mode:=doppler.
"""

import math

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2 as pc2
from std_msgs.msg import Empty, Header
from nav_msgs.msg import Odometry

try:
    import tf2_ros
except ImportError:              # ego_mode != tf works without tf2
    tf2_ros = None

from densify_core import (VoxelEvidenceMap, detection_information,
                          estimate_base_twist, estimate_sensor_velocity,
                          quat_to_R, radius_outlier_filter, rotate_information,
                          save_ply, so3_exp)

# 2026-08-19 solved extrinsics, T_os_lidar_radarN (see sessions/ report)
DEFAULT_RADARS = {
    "radar1": {"topic": "/radar1/radar/points_all",
               "t": [0.033393, 0.140555, -0.168519],
               "q": [0.123377, 0.001261, 0.992298, -0.010988]},
    "radar2": {"topic": "/radar2/radar/points_all",
               "t": [0.036705, -0.120823, -0.139143],
               "q": [0.013827, 0.736618, 0.675767, 0.023290]},
    "radar3": {"topic": "/radar3/radar/points_all",
               "t": [0.041839, -0.101587, 0.161252],
               "q": [-0.007406, 0.986026, 0.007861, 0.166243]},
}


def _stamp_to_sec(stamp):
    return stamp.sec + stamp.nanosec * 1e-9


class RadarDensifyNode(Node):

    def __init__(self):
        super().__init__("radar_densify")
        p = self.declare_parameter

        # radars: enable any subset; topic + extrinsic per radar
        self.radars = {}
        for name, d in DEFAULT_RADARS.items():
            if not p(f"{name}_enable", True).value:
                continue
            self.radars[name] = {
                "topic": p(f"{name}_topic", d["topic"]).value,
                "R": quat_to_R(np.array(
                    p(f"{name}_quat_xyzw", d["q"]).value, float)),
                "t": np.array(p(f"{name}_t_xyz", d["t"]).value, float),
                "last": None,      # latest gated frame, for the joint twist
            }

        # point-cloud fields (this rig publishes snr as 'intensity')
        self.f_x = p("pc_field_x", "x").value
        self.f_y = p("pc_field_y", "y").value
        self.f_z = p("pc_field_z", "z").value
        self.f_snr = p("pc_field_snr", "intensity").value
        self.f_dop = p("pc_field_doppler", "doppler").value

        # gates on the raw detections
        self.min_range = p("min_range", 0.3).value
        self.max_range = p("max_range", 25.0).value
        self.min_snr = p("min_snr", 0.0).value

        # ego motion
        self.ego_mode = p("ego_mode", "tf").value        # tf|odom|doppler|static
        self.map_frame = p("map_frame", "odom").value
        self.base_frame = p("base_frame", "os_lidar").value
        self.odom_topic = p("odom_topic", "/odom").value
        self.doppler_sign = p("doppler_sign", -1.0).value
        self.static_thresh = p("static_doppler_thresh", 0.15).value
        self.twist_sync_s = p("twist_sync_s", 0.10).value

        # radar noise model (same numbers the calibration used)
        self.sigma_r = p("sigma_range_m", 0.05).value
        self.sigma_az = p("sigma_az_deg", 3.0).value
        self.sigma_el = p("sigma_el_deg", 8.0).value

        # accumulation / evidence
        self.map = VoxelEvidenceMap(p("voxel_m", 0.10).value)
        self.min_frames = p("min_frames", 3).value
        self.min_hits = p("min_hits", 0).value
        self.min_neighbors = p("min_neighbors", 2).value
        self.output_path = p("output_path", "radar_dense_map.ply").value

        # state
        self.frame_id = 0
        self.T_wb = (np.eye(3), np.zeros(3))   # doppler dead-reckoned pose
        self.last_twist_stamp = None
        self.odom_buf = []                     # (t, R, p) recent odometry

        qos = QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT)
        for name, r in self.radars.items():
            self.create_subscription(
                PointCloud2, r["topic"],
                lambda msg, n=name: self.on_cloud(n, msg), qos)
        if self.ego_mode == "odom":
            self.create_subscription(Odometry, self.odom_topic,
                                     self.on_odom, 20)
        if self.ego_mode == "tf":
            if tf2_ros is None:
                raise RuntimeError("ego_mode=tf needs tf2_ros")
            self.tf_buf = tf2_ros.Buffer()
            self.tf_listener = tf2_ros.TransformListener(self.tf_buf, self)

        self.pub_map = self.create_publisher(PointCloud2, "~/map", 1)
        self.pub_static = self.create_publisher(PointCloud2, "~/static_points", 5)
        self.pub_dynamic = self.create_publisher(PointCloud2, "~/dynamic_points", 5)
        self.create_subscription(Empty, "~/save", self.on_save, 1)
        self.create_subscription(Empty, "~/reset", self.on_reset, 1)
        self.create_timer(p("map_publish_period", 2.0).value, self.publish_map)

        self.get_logger().info(
            f"densify: radars={list(self.radars)} ego_mode={self.ego_mode} "
            f"voxel={self.map.voxel} m, min_frames={self.min_frames}")

    # ---------------------------------------------------------------- ego

    def on_odom(self, msg):
        q = msg.pose.pose.orientation
        pos = msg.pose.pose.position
        self.odom_buf.append((
            _stamp_to_sec(msg.header.stamp),
            quat_to_R([q.x, q.y, q.z, q.w]),
            np.array([pos.x, pos.y, pos.z])))
        if len(self.odom_buf) > 400:
            del self.odom_buf[:200]

    def pose_at(self, t_sec):
        """T_map_base at time t, per ego_mode. None = not available yet."""
        if self.ego_mode == "static":
            return np.eye(3), np.zeros(3)
        if self.ego_mode == "doppler":
            return self.T_wb
        if self.ego_mode == "odom":
            if not self.odom_buf:
                return None
            i = int(np.argmin([abs(t - t_sec) for t, _, _ in self.odom_buf]))
            t, R, p0 = self.odom_buf[i]
            if abs(t - t_sec) > 0.25:
                return None
            return R, p0
        # tf
        try:
            tr = self.tf_buf.lookup_transform(
                self.map_frame, self.base_frame, rclpy.time.Time())
        except Exception:
            return None
        q = tr.transform.rotation
        tt = tr.transform.translation
        return (quat_to_R([q.x, q.y, q.z, q.w]),
                np.array([tt.x, tt.y, tt.z]))

    def update_doppler_pose(self, t_sec):
        """Joint 6-DOF twist from the freshest frame of every radar, then
        integrate. Falls back to per-radar 3-DOF (translation only)."""
        frames = [r["last"] for r in self.radars.values() if r["last"]]
        frames = [f for f in frames if abs(f["t"] - t_sec) <= self.twist_sync_s]
        if not frames:
            return
        per_radar = [{"pts": f["pts"], "doppler": f["dop"],
                      "R": f["Rx"], "t": f["tx"]} for f in frames]
        v, w, _ = estimate_base_twist(per_radar, thresh=self.static_thresh)
        if v is None:
            # translation-only fallback from the first radar's 3-DOF fit
            f = frames[0]
            v3, _ = estimate_sensor_velocity(f["pts"], f["dop"],
                                             thresh=self.static_thresh)
            if v3 is None:
                return
            v, w = f["Rx"] @ v3, np.zeros(3)
        v = self.doppler_sign * v
        w = self.doppler_sign * w
        if self.last_twist_stamp is not None:
            dt = t_sec - self.last_twist_stamp
            if 0.0 < dt < 0.5:
                Rwb, twb = self.T_wb
                twb = twb + Rwb @ (v * dt)
                Rwb = Rwb @ so3_exp(w * dt)
                self.T_wb = (Rwb, twb)
        self.last_twist_stamp = t_sec

    # ------------------------------------------------------------- clouds

    def on_cloud(self, name, msg):
        r = self.radars[name]
        want = [self.f_x, self.f_y, self.f_z]
        have = {f.name for f in msg.fields}
        opt = [f for f in (self.f_snr, self.f_dop) if f in have]
        pts_iter = pc2.read_points(msg, field_names=want + opt,
                                   skip_nans=True)
        rows = np.array([tuple(pt) for pt in pts_iter], float)
        if rows.size == 0:
            return
        pts = rows[:, 0:3]
        snr = rows[:, 3 + opt.index(self.f_snr)] if self.f_snr in opt \
            else np.zeros(len(rows))
        dop = rows[:, 3 + opt.index(self.f_dop)] if self.f_dop in opt \
            else np.zeros(len(rows))

        rng = np.linalg.norm(pts, axis=1)
        keep = (rng >= self.min_range) & (rng <= self.max_range)
        if self.min_snr > 0:
            keep &= snr >= self.min_snr
        pts, snr, dop = pts[keep], snr[keep], dop[keep]
        if len(pts) == 0:
            return

        t_sec = _stamp_to_sec(msg.header.stamp)
        r["last"] = {"pts": pts, "dop": dop, "t": t_sec,
                     "Rx": r["R"], "tx": r["t"]}
        if self.ego_mode == "doppler":
            self.update_doppler_pose(t_sec)

        # 1+2: remove ego speed, split static/dynamic
        if self.f_dop in have:
            _, static = estimate_sensor_velocity(pts, dop,
                                                 thresh=self.static_thresh)
        else:
            static = np.ones(len(pts), bool)   # no doppler field: keep all

        pose = self.pose_at(t_sec)
        if pose is None:
            return                              # no ego pose yet — skip frame
        Rwb, twb = pose
        Rwr = Rwb @ r["R"]                      # radar -> map rotation
        twr = Rwb @ r["t"] + twb

        pw_static = pts[static] @ Rwr.T + twr
        pw_dynamic = pts[~static] @ Rwr.T + twr

        # 3+4: anisotropic information, accumulate on the evidence grid
        if len(pw_static):
            Lam = detection_information(pts[static], self.sigma_r,
                                        self.sigma_az, self.sigma_el,
                                        snr=snr[static])
            Lam_w = rotate_information(Lam, Rwr)
            self.map.add(pw_static, Lam_w, snr[static], self.frame_id)
        self.frame_id += 1

        self._publish_xyzi(self.pub_static, pw_static, snr[static],
                           msg.header.stamp)
        self._publish_xyzi(self.pub_dynamic, pw_dynamic, snr[~static],
                           msg.header.stamp)

    # ------------------------------------------------------------- output

    def _extract(self):
        pts, attrs = self.map.extract(self.min_frames, self.min_hits)
        return radius_outlier_filter(pts, attrs, self.map.voxel,
                                     self.min_neighbors)

    def _publish_xyzi(self, pub, pts, inten, stamp):
        if pub.get_subscription_count() == 0:
            return
        header = Header(stamp=stamp, frame_id=self.map_frame)
        fields = [PointField(name=n, offset=4 * i,
                             datatype=PointField.FLOAT32, count=1)
                  for i, n in enumerate(("x", "y", "z", "intensity"))]
        data = [(float(p[0]), float(p[1]), float(p[2]), float(s))
                for p, s in zip(pts, inten)]
        pub.publish(pc2.create_cloud(header, fields, data))

    def publish_map(self):
        pts, attrs = self._extract()
        header = Header(stamp=self.get_clock().now().to_msg(),
                        frame_id=self.map_frame)
        fields = [PointField(name=n, offset=4 * i,
                             datatype=PointField.FLOAT32, count=1)
                  for i, n in enumerate(("x", "y", "z", "intensity",
                                         "frames", "hits"))]
        data = [(float(p[0]), float(p[1]), float(p[2]),
                 float(attrs["snr"][i]), float(attrs["frames"][i]),
                 float(attrs["hits"][i])) for i, p in enumerate(pts)]
        self.pub_map.publish(pc2.create_cloud(header, fields, data))
        self.get_logger().info(
            f"map: {len(pts)} pts emitted / {len(self.map)} voxels touched "
            f"/ {self.frame_id} frames", throttle_duration_sec=10.0)

    def on_save(self, _):
        pts, attrs = self._extract()
        save_ply(self.output_path, pts, attrs)
        self.get_logger().info(f"saved {len(pts)} pts -> {self.output_path}")

    def on_reset(self, _):
        self.map = VoxelEvidenceMap(self.map.voxel)
        self.frame_id = 0
        self.T_wb = (np.eye(3), np.zeros(3))
        self.last_twist_stamp = None
        self.get_logger().info("map reset")


def main():
    rclpy.init()
    node = RadarDensifyNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
