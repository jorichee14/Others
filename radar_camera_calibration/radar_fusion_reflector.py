#!/usr/bin/env python3
"""
Two-radar reflector fusion + display
====================================
Locate the trihedral corner reflector accurately by FUSING two calibrated radars
whose good axes are orthogonal, and draw the result on the ZED image.

Why fusion helps here
─────────────────────
A single IWR6843 measures RANGE precisely, one ANGLE moderately (azimuth), the
other ANGLE poorly (elevation, few antennas). We mounted radar2 rolled ~90° vs
radar1, so their poor axes are PERPENDICULAR in the camera frame:
  • radar1 : sharp horizontal, soft vertical
  • radar2 : sharp vertical,   soft horizontal
Each radar's detection is turned into a 3-D point + an anisotropic covariance
(σ_range along the radial, range·σ_az / range·σ_el across it), rotated into the
camera frame by that radar's calibrated extrinsic. A covariance-weighted (BLUE /
information-filter) fusion then automatically takes the horizontal from radar1
and the vertical from radar2 → a tight 3-D reflector position.

    p_i   = R_i q_i + t_i                         (each radar → camera frame)
    Σ_i   = R_i · diag_local(σ_r², (r·σ_az)², (r·σ_el)²) · R_iᵀ
    Σ_f   = (Σ1⁻¹ + Σ2⁻¹)⁻¹
    p_f   = Σ_f (Σ1⁻¹ p1 + Σ2⁻¹ p2)

Display
───────
  • radar1 point (cyan)  with a bar along its blind axis (≈ vertical)
  • radar2 point (orange) with a bar along its blind axis (≈ horizontal)
  • fused point (green) at the intersection — the accurate reflector location
Single-radar frames still draw that radar's point + blind-axis bar.

Extrinsics default to the values solved by radar_camera_calib.py for this rig;
override with params. Run:
  ros2 run wicoms_utils radar_fusion_reflector -p ...params...
"""
import numpy as np
from scipy.spatial.transform import Rotation as Rot
import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from geometry_msgs.msg import PointStamped
from cv_bridge import CvBridge

try:
    from sensor_msgs_py import point_cloud2 as pc2
    _HAVE_PC2 = True
except Exception:
    _HAVE_PC2 = False


def project(pt_cam, K, D):
    if pt_cam[2] <= 0:
        return None
    uv, _ = cv2.projectPoints(pt_cam.reshape(1, 3), np.zeros(3), np.zeros(3), K, D)
    return int(round(uv[0, 0, 0])), int(round(uv[0, 0, 1]))


def radar_cov_cam(q, R, sr, saz, sel):
    """3×3 covariance of a radar point in the CAMERA frame. Local basis: radial,
    azimuth-tangent (horizontal in the radar's XY), elevation-tangent."""
    r = float(np.linalg.norm(q))
    if r < 1e-6:
        r = 1e-6
    er = q / r
    eaz = np.array([-q[1], q[0], 0.0])                 # radar X=fwd, Y=left → az tangent
    n = np.linalg.norm(eaz); eaz = eaz / n if n > 1e-6 else np.array([0., 1., 0.])
    eel = np.cross(er, eaz)
    n = np.linalg.norm(eel); eel = eel / n if n > 1e-6 else np.array([0., 0., 1.])
    S = (sr ** 2) * np.outer(er, er) \
        + (r * saz) ** 2 * np.outer(eaz, eaz) \
        + (r * sel) ** 2 * np.outer(eel, eel)
    return R @ S @ R.T, R @ eel * (r * sel)            # cov + blind-axis vector (cam frame)


class RadarFusionReflector(Node):
    def __init__(self):
        super().__init__('radar_fusion_reflector')
        dp = self.declare_parameter
        dp('image_topic', '/zed/zed_node/left/image_rect_color')
        dp('info_topic',  '/zed/zed_node/left/camera_info')
        dp('radar1_topic', '/radar1/radar/points_all')
        dp('radar2_topic', '/radar2/radar/points_all')
        dp('pc_field_x', 'x'); dp('pc_field_y', 'y'); dp('pc_field_z', 'z')
        dp('pc_field_snr', 'intensity')
        # per-radar extrinsics T_cam_radar (defaults = this rig's solved values)
        dp('r1_t_xyz', [0.2218, -0.0067, -0.1721])
        dp('r1_quat_xyzw', [-0.5345, 0.5853, -0.4196, -0.4424])
        dp('r2_t_xyz', [-0.0999, -0.0124, -0.0011])
        dp('r2_quat_xyzw', [0.7882, -0.0406, 0.6121, 0.0499])
        dp('r1_range_scale', 1.039); dp('r2_range_scale', 1.026)
        # radar noise (same chip for both) — drives the fusion weighting
        dp('sigma_range_m', 0.05); dp('sigma_az_deg', 3.0); dp('sigma_el_deg', 8.0)
        # reflector selection
        dp('min_range', 0.3); dp('max_range', 6.0); dp('min_snr', 100.0)
        dp('assoc_gate_m', 0.6)      # fuse only if the two camera-frame points agree within this
        dp('stale_s', 0.3)           # ignore a radar detection older than this
        dp('publish_point', True)    # publish fused reflector as PointStamped in the camera frame
        dp('show_window', True)
        dp('debug_image_topic', '/radar_fusion/debug_image')

        g = lambda n: self.get_parameter(n).value
        self.image_topic = g('image_topic')
        self.fx, self.fy, self.fz = g('pc_field_x'), g('pc_field_y'), g('pc_field_z')
        self.fsnr = g('pc_field_snr')
        self.R1 = Rot.from_quat(g('r1_quat_xyzw')).as_matrix(); self.t1 = np.array(g('r1_t_xyz'), float)
        self.R2 = Rot.from_quat(g('r2_quat_xyzw')).as_matrix(); self.t2 = np.array(g('r2_t_xyz'), float)
        self.s1 = g('r1_range_scale'); self.s2 = g('r2_range_scale')
        self.sr = g('sigma_range_m')
        self.saz = np.radians(g('sigma_az_deg')); self.sel = np.radians(g('sigma_el_deg'))
        self.min_range = g('min_range'); self.max_range = g('max_range'); self.min_snr = g('min_snr')
        self.assoc_gate = g('assoc_gate_m'); self.stale = g('stale_s')
        self.show_window = bool(g('show_window'))
        self.window = 'radar_fusion — radar1(cyan) radar2(orange) fused(green)'
        self._win_ok = None; self._last = None

        self.bridge = CvBridge(); self.K = None; self.D = None
        # latest reflector detection per radar: (p_cam, cov, blind_vec, snr, stamp_s)
        self.det = {1: None, 2: None}

        self.create_subscription(CameraInfo, g('info_topic'), self._info, qos_profile_sensor_data)
        self.create_subscription(Image, self.image_topic, self._image, qos_profile_sensor_data)
        self.create_subscription(PointCloud2, g('radar1_topic'),
                                 lambda m: self._radar(m, 1), qos_profile_sensor_data)
        self.create_subscription(PointCloud2, g('radar2_topic'),
                                 lambda m: self._radar(m, 2), qos_profile_sensor_data)
        self.dbg_pub = self.create_publisher(Image, g('debug_image_topic'), 1)
        self.pt_pub = (self.create_publisher(PointStamped, '/radar_fusion/reflector', 1)
                       if bool(g('publish_point')) else None)
        if self.show_window:
            self.create_timer(0.05, self._gui)
        self.get_logger().info(
            f"[radar_fusion] radar1 {g('radar1_topic')}  radar2 {g('radar2_topic')}\n"
            f"  fusing into {g('image_topic')} — cyan=radar1, orange=radar2, green=fused")

    def _info(self, msg):
        if self.K is None:
            self.K = np.array(msg.k).reshape(3, 3)
            self.D = np.array(msg.d) if len(msg.d) else np.zeros(5)
            self.get_logger().info(f"intrinsics locked ({msg.width}x{msg.height})")

    def _now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _radar(self, msg, which):
        """Select the reflector (brightest gated return) and store its camera-frame
        point + covariance for this radar."""
        if not _HAVE_PC2:
            return
        names = [f.name for f in msg.fields]
        has_snr = self.fsnr in names
        want = [self.fx, self.fy, self.fz] + ([self.fsnr] if has_snr else [])
        arr = list(pc2.read_points(msg, field_names=want, skip_nans=True))
        if not arr:
            self.det[which] = None; return
        arr = np.array([tuple(a) for a in arr], float)
        scale = self.s1 if which == 1 else self.s2
        xyz = arr[:, :3] * float(scale)
        snr = arr[:, 3] if has_snr else np.zeros(len(arr))
        rng = np.linalg.norm(xyz, axis=1)
        keep = (rng >= self.min_range) & (rng <= self.max_range)
        if not keep.any():
            self.det[which] = None; return
        xg, sg = xyz[keep], snr[keep]
        i = int(np.argmax(sg))
        if self.min_snr > 0 and sg[i] < self.min_snr:
            self.det[which] = None; return
        q = xg[i]
        R, t = (self.R1, self.t1) if which == 1 else (self.R2, self.t2)
        p = R @ q + t
        cov, blind = radar_cov_cam(q, R, self.sr, self.saz, self.sel)
        self.det[which] = (p, cov, blind, float(sg[i]), self._now())

    def _fresh(self, which):
        d = self.det[which]
        if d is None or (self._now() - d[4]) > self.stale:
            return None
        return d

    def _image(self, msg):
        if self.K is None:
            return
        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().warn(f"cv_bridge: {e}"); return
        h, w = bgr.shape[:2]
        d1, d2 = self._fresh(1), self._fresh(2)

        def draw_pt(p, blind, color, label):
            uv = project(p, self.K, self.D)
            if uv and 0 <= uv[0] < w and 0 <= uv[1] < h:
                cv2.circle(bgr, uv, 6, color, -1)
                a = project(p - blind, self.K, self.D); b = project(p + blind, self.K, self.D)
                if a and b:
                    cv2.line(bgr, a, b, color, 1)              # blind-axis uncertainty bar
                cv2.putText(bgr, label, (uv[0] + 8, uv[1] - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            return uv

        if d1:
            draw_pt(d1[0], d1[2], (255, 255, 0), f"r1 {np.linalg.norm(d1[0]):.2f}m")
        if d2:
            draw_pt(d2[0], d2[2], (0, 165, 255), f"r2 {np.linalg.norm(d2[0]):.2f}m")

        fused = None
        if d1 and d2 and np.linalg.norm(d1[0] - d2[0]) <= self.assoc_gate:
            I1 = np.linalg.inv(d1[1]); I2 = np.linalg.inv(d2[1])
            Sf = np.linalg.inv(I1 + I2)
            fused = Sf @ (I1 @ d1[0] + I2 @ d2[0])
            sig = np.sqrt(np.clip(np.diag(Sf), 0, None)) * 1000
            uv = project(fused, self.K, self.D)
            if uv and 0 <= uv[0] < w and 0 <= uv[1] < h:
                cv2.circle(bgr, uv, 10, (0, 255, 0), 2)
                cv2.line(bgr, (uv[0]-16, uv[1]), (uv[0]+16, uv[1]), (0, 255, 0), 1)
                cv2.line(bgr, (uv[0], uv[1]-16), (uv[0], uv[1]+16), (0, 255, 0), 1)
                cv2.putText(bgr, f"FUSED {np.linalg.norm(fused):.2f}m  1sig[{sig[0]:.0f},{sig[1]:.0f},{sig[2]:.0f}]mm",
                            (uv[0]+14, uv[1]+18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        elif d1 and not d2:
            fused = d1[0]
        elif d2 and not d1:
            fused = d2[0]

        status = f"r1:{'OK' if d1 else '--'} r2:{'OK' if d2 else '--'}  " \
                 + ("FUSED" if (d1 and d2 and fused is not None) else "single" if fused is not None else "no reflector")
        cv2.putText(bgr, status, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        if fused is not None and self.pt_pub is not None:
            m = PointStamped(); m.header = msg.header
            m.point.x, m.point.y, m.point.z = map(float, fused)
            self.pt_pub.publish(m)

        self._last = bgr
        try:
            self.dbg_pub.publish(self.bridge.cv2_to_imgmsg(bgr, 'bgr8'))
        except Exception:
            pass

    def _gui(self):
        if not self.show_window or self._last is None or self._win_ok is False:
            return
        try:
            if self._win_ok is None:
                cv2.namedWindow(self.window, cv2.WINDOW_NORMAL); self._win_ok = True
            cv2.imshow(self.window, self._last); cv2.waitKey(1)
        except Exception as e:
            self._win_ok = False
            self.get_logger().warn(f"show_window failed ({e}) — use rqt_image_view {self.get_parameter('debug_image_topic').value}")


def main():
    rclpy.init(); node = RadarFusionReflector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
