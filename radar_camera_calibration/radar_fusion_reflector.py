#!/usr/bin/env python3
"""
Two-radar reflector fusion + tracking + display
===============================================
Locate the trihedral corner reflector accurately by FUSING two calibrated radars
whose good axes are orthogonal, and draw a SMOOTH tracked estimate on the ZED
image.

Why fusion helps here
─────────────────────
A single IWR6843 measures RANGE precisely, one ANGLE moderately (azimuth), the
other ANGLE poorly (elevation, few antennas). radar2 is mounted rolled ~90° vs
radar1, so their poor axes are PERPENDICULAR in the camera frame:
  • radar1 : sharp horizontal, soft vertical
  • radar2 : sharp vertical,   soft horizontal

Each detection becomes a 3-D point + an anisotropic covariance
(σ_range along the radial, range·σ_az / range·σ_el across it), rotated into the
camera frame by that radar's calibrated extrinsic.

Why a TRACKER, not a per-frame combine
──────────────────────────────────────
A memoryless per-frame BLUE is only as steady as the raw detections, which hop
between multipath returns → a jumpy output. Instead we run a constant-velocity
Kalman filter in the camera frame. Both radars update it ASYNCHRONOUSLY with
their full 3-D covariance, so every axis is constrained by whichever radar sees
it sharply AND the estimate is smoothed over time:

    predict:  x⁻ = F x,   P⁻ = F P Fᵀ + Q(σ_a)
    update  :  y = z_i − H x⁻,   S = H P⁻ Hᵀ + R_i,   K = P⁻ Hᵀ S⁻¹
               x = x⁻ + K y,     P = (I − K H) P⁻
    R_i     = R_model_i (anisotropic, from the extrinsic)  +  Cov(recent
              innovations of radar i)          ← "uncertainty from errors before"

A radar that has recently been landing far from the track gets its R inflated and
is automatically down-weighted. Each measurement is Mahalanobis-gated so a clutter
jump is rejected before it can move the estimate.

Display
───────
  • radar1 raw point (cyan)  + bar along its blind axis (≈ vertical)
  • radar2 raw point (orange)+ bar along its blind axis (≈ horizontal)
  • TRACKED fused point (green) with a short trail + per-axis 1σ from P
Falls back to whichever radar is fresh; coasts briefly on prediction if both drop.

Extrinsics default to the values solved by radar_camera_calib.py for this rig;
override with params.  Run:
  ros2 run wicoms_utils radar_fusion_reflector -p ...params...
"""
from collections import deque
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
    """3×3 measurement covariance of a radar point in the CAMERA frame. Local
    basis: radial, azimuth-tangent (horizontal in the radar XY), elevation-tangent.
    Returns (cov, blind_axis_vector) both in the camera frame."""
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


class TrackKF:
    """Constant-velocity Kalman filter, state = [x,y,z, vx,vy,vz] in the camera
    frame. Asynchronous 3-D position updates (one per radar detection)."""

    def __init__(self, sigma_accel, v0_sigma=1.0):
        self.qa = float(sigma_accel)            # process accel std (m/s²)
        self.v0 = float(v0_sigma)
        self.x = None                           # (6,)
        self.P = None                           # (6,6)
        self.t = None                           # last-touched time (s)
        self.H = np.zeros((3, 6)); self.H[0, 0] = self.H[1, 1] = self.H[2, 2] = 1.0

    def valid(self):
        return self.x is not None

    def init(self, z, R, t):
        self.x = np.zeros(6); self.x[:3] = z
        self.P = np.zeros((6, 6))
        self.P[:3, :3] = R
        self.P[3:, 3:] = np.eye(3) * (self.v0 ** 2)
        self.t = t

    def _FQ(self, dt):
        F = np.eye(6)
        F[0, 3] = F[1, 4] = F[2, 5] = dt
        q = self.qa ** 2
        dt2, dt3 = dt * dt, dt * dt * dt
        Q = np.zeros((6, 6))
        Q[:3, :3] = np.eye(3) * (dt3 / 3.0 * q)
        Q[:3, 3:] = Q[3:, :3] = np.eye(3) * (dt2 / 2.0 * q)
        Q[3:, 3:] = np.eye(3) * (dt * q)
        return F, Q

    def peek(self, t):
        """Predicted (x, P) at time t WITHOUT mutating (for display)."""
        if self.x is None:
            return None, None
        dt = max(0.0, t - self.t)
        F, Q = self._FQ(dt)
        return F @ self.x, F @ self.P @ F.T + Q

    def predict_to(self, t):
        dt = max(0.0, t - self.t)
        if dt == 0.0:
            return
        F, Q = self._FQ(dt)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        self.t = t

    def update(self, z, R, t, gate_chi2):
        """Predict to t then Kalman-update with measurement z, cov R. Returns the
        innovation y if accepted, or None if gated out."""
        self.predict_to(t)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + R
        try:
            Si = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return None
        d2 = float(y @ Si @ y)
        if d2 > gate_chi2:
            return None
        K = self.P @ self.H.T @ Si
        self.x = self.x + K @ y
        self.P = self.P - K @ S @ K.T
        self.P = 0.5 * (self.P + self.P.T)      # keep symmetric
        return y


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
        # radar noise (same chip for both) — the MODEL floor for the fusion weighting
        dp('sigma_range_m', 0.05); dp('sigma_az_deg', 3.0); dp('sigma_el_deg', 8.0)
        # reflector selection
        dp('min_range', 0.3); dp('max_range', 6.0); dp('min_snr', 100.0)
        dp('select_radius_m', 0.5)   # SNR-weighted centroid of points within this of prediction
        # tracker
        dp('process_accel', 2.0)     # KF process noise: reflector accel std (m/s²). ↑ = follows faster/jumpier
        dp('innov_gate_chi2', 11.35) # Mahalanobis gate, 3-DOF 99% = 11.34
        dp('adapt_window', 12)       # # recent innovations used to inflate R per radar
        dp('adapt_max_scale', 4.0)   # cap adaptive R inflation at this × the model
        dp('reinit_gap_s', 1.0)      # gap with no update longer than this ⇒ hard reinit
        dp('coast_s', 0.5)           # keep drawing the tracked point up to this long after last update
        dp('trail_len', 25)
        dp('publish_point', True)    # publish tracked reflector as PointStamped in the camera frame
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
        self.select_radius = g('select_radius_m')
        self.gate_chi2 = g('innov_gate_chi2'); self.adapt_win = int(g('adapt_window'))
        self.adapt_max = g('adapt_max_scale'); self.reinit_gap = g('reinit_gap_s'); self.coast = g('coast_s')
        self.show_window = bool(g('show_window'))
        self.window = 'radar_fusion — radar1(cyan) radar2(orange) tracked(green)'
        self._win_ok = None; self._last = None

        self.bridge = CvBridge(); self.K = None; self.D = None
        self.kf = TrackKF(g('process_accel'))
        # per-radar: latest RAW detection (p_cam, blind_vec, snr, stamp) for display,
        # and a ring buffer of recent innovations for the adaptive R.
        self.raw = {1: None, 2: None}
        self.innov = {1: deque(maxlen=self.adapt_win), 2: deque(maxlen=self.adapt_win)}
        self.trail = deque(maxlen=int(g('trail_len')))

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
            f"  constant-velocity tracker, adaptive R — cyan=radar1, orange=radar2, green=tracked")

    def _info(self, msg):
        if self.K is None:
            self.K = np.array(msg.k).reshape(3, 3)
            self.D = np.array(msg.d) if len(msg.d) else np.zeros(5)
            self.get_logger().info(f"intrinsics locked ({msg.width}x{msg.height})")

    def _now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _select(self, xyz, snr, R, t):
        """Pick the reflector detection in one radar's cloud. Prefer the SNR-weighted
        centroid of points within select_radius of the current track prediction (in
        this radar's frame); fall back to the brightest gated return when there is no
        track yet. Returns (q_radar_frame, snr) or None."""
        rng = np.linalg.norm(xyz, axis=1)
        keep = (rng >= self.min_range) & (rng <= self.max_range)
        if self.min_snr > 0:
            keep &= snr >= self.min_snr
        if not keep.any():
            return None
        xg, sg = xyz[keep], snr[keep]
        # gated centroid around the predicted apex (stabilises selection frame-to-frame)
        if self.kf.valid():
            x_pred, _ = self.kf.peek(self._now())
            q_pred = R.T @ (x_pred[:3] - t)                    # predicted apex in radar frame
            near = np.linalg.norm(xg - q_pred, axis=1) <= self.select_radius
            if near.any():
                w = np.clip(sg[near], 1e-3, None)
                q = (xg[near] * w[:, None]).sum(0) / w.sum()
                return q, float(sg[near].max())
        # bootstrap / no nearby points: brightest
        i = int(np.argmax(sg))
        return xg[i], float(sg[i])

    def _adaptive_R(self, which, R_model):
        """R_model inflated by the sample covariance of this radar's recent
        innovations, capped at adapt_max × the model (in trace)."""
        buf = self.innov[which]
        if len(buf) < 3:
            return R_model
        C = np.cov(np.array(buf).T)
        if C.shape != (3, 3) or not np.all(np.isfinite(C)):
            return R_model
        ta, tm = np.trace(C), (self.adapt_max - 1.0) * np.trace(R_model)
        if ta > tm and ta > 0:                                 # cap the inflation
            C = C * (tm / ta)
        return R_model + C

    def _radar(self, msg, which):
        if not _HAVE_PC2 or self.K is None:
            return
        names = [f.name for f in msg.fields]
        has_snr = self.fsnr in names
        want = [self.fx, self.fy, self.fz] + ([self.fsnr] if has_snr else [])
        arr = list(pc2.read_points(msg, field_names=want, skip_nans=True))
        if not arr:
            return
        arr = np.array([tuple(a) for a in arr], float)
        scale = self.s1 if which == 1 else self.s2
        xyz = arr[:, :3] * float(scale)
        snr = arr[:, 3] if has_snr else np.zeros(len(arr))
        sel = self._select(xyz, snr, *( (self.R1, self.t1) if which == 1 else (self.R2, self.t2) ))
        if sel is None:
            return
        q, s = sel
        R, t = (self.R1, self.t1) if which == 1 else (self.R2, self.t2)
        p = R @ q + t                                          # detection in camera frame
        R_model, blind = radar_cov_cam(q, R, self.sr, self.saz, self.sel)
        now = self._now()
        self.raw[which] = (p, blind, s, now)

        # (re)initialise the track on the first detection or after a long gap
        if (not self.kf.valid()) or (now - self.kf.t) > self.reinit_gap:
            self.kf.init(p, R_model, now)
            self.innov[which].clear()
            return
        R_use = self._adaptive_R(which, R_model)
        y = self.kf.update(p, R_use, now, self.gate_chi2)      # gated CV-KF update
        if y is not None:
            self.innov[which].append(y)

    def _fresh_raw(self, which):
        d = self.raw[which]
        if d is None or (self._now() - d[3]) > self.coast:
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
        now = self._now()
        d1, d2 = self._fresh_raw(1), self._fresh_raw(2)

        def draw_pt(p, blind, color, label):
            uv = project(p, self.K, self.D)
            if uv and 0 <= uv[0] < w and 0 <= uv[1] < h:
                cv2.circle(bgr, uv, 6, color, -1)
                a = project(p - blind, self.K, self.D); b = project(p + blind, self.K, self.D)
                if a and b:
                    cv2.line(bgr, a, b, color, 1)              # blind-axis uncertainty bar
                cv2.putText(bgr, label, (uv[0] + 8, uv[1] - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        if d1:
            draw_pt(d1[0], d1[1], (255, 255, 0), f"r1 {np.linalg.norm(d1[0]):.2f}m")
        if d2:
            draw_pt(d2[0], d2[1], (0, 165, 255), f"r2 {np.linalg.norm(d2[0]):.2f}m")

        tracked = None
        if self.kf.valid() and (now - self.kf.t) <= self.coast:
            x_pred, P_pred = self.kf.peek(now)                 # smooth estimate at image time
            tracked = x_pred[:3]
            sig = np.sqrt(np.clip(np.diag(P_pred)[:3], 0, None)) * 1000
            spd = np.linalg.norm(x_pred[3:]) * 100             # cm/s
            self.trail.append(tracked.copy())
            pts = [project(p, self.K, self.D) for p in self.trail]
            pts = [q for q in pts if q and 0 <= q[0] < w and 0 <= q[1] < h]
            for a, b in zip(pts[:-1], pts[1:]):
                cv2.line(bgr, a, b, (0, 140, 0), 1)            # fading trail
            uv = project(tracked, self.K, self.D)
            if uv and 0 <= uv[0] < w and 0 <= uv[1] < h:
                cv2.circle(bgr, uv, 10, (0, 255, 0), 2)
                cv2.line(bgr, (uv[0]-16, uv[1]), (uv[0]+16, uv[1]), (0, 255, 0), 1)
                cv2.line(bgr, (uv[0], uv[1]-16), (uv[0], uv[1]+16), (0, 255, 0), 1)
                cv2.putText(bgr, f"TRACK {np.linalg.norm(tracked):.2f}m 1sig[{sig[0]:.0f},{sig[1]:.0f},{sig[2]:.0f}]mm {spd:.0f}cm/s",
                            (uv[0]+14, uv[1]+18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        both = d1 and d2
        status = f"r1:{'OK' if d1 else '--'} r2:{'OK' if d2 else '--'}  " \
                 + ("TRACKING (2-radar)" if (both and tracked is not None)
                    else "TRACKING (1-radar)" if tracked is not None else "acquiring…")
        cv2.putText(bgr, status, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        if tracked is not None and self.pt_pub is not None:
            m = PointStamped(); m.header = msg.header
            m.point.x, m.point.y, m.point.z = map(float, tracked)
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
