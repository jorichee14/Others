#!/usr/bin/env python3
"""
RADAR ↔ LIDAR extrinsic calibration (solves T_cam_radar, lidar as reference)
============================================================================

Pairs with `lidar_reflector_detector.py` (run BOTH nodes). The detector finds
the corner-reflector apex in the lidar cloud and publishes it in the CAMERA
frame (via the Koide T_cam_lidar). This node:

  1. gates the radar cloud around that apex (background subtraction, a
     rotation-invariant RANGE gate, |doppler|≈0, then a 3-D gate around the
     predicted point once a solve or prior exists),
  2. shows a LIVE AIM line so you can see, before triggering, whether the
     reflector is aimed well enough at the radar for the capture to pass:
        radar: best 1240 (norm 1870) @ 2.31 m  OK
        radar: best 41 (norm 62) @ 2.28 m  RE-AIM
        radar: no return near lidar range
     `norm` is snr·(r/1.5m)^4 — received power falls as 1/r^4, so a raw SNR
     threshold would wrongly refuse a well-aimed reflector at 4 m,
  3. on ~/capture stores one atomic (p_cam, p_radar) pair — BOTH sides must
     pass their gates or the capture is REFUSED with the reason logged,
  4. solves T_cam_radar with the same measurement-space ML solver as the
     camera pipeline (imported from radar_camera_calib — Huber, reject_sigma,
     per-axis reject, covariance, LOO). The lidar apex enters as
     board_R = I, board_t = p_cam, apex offset pinned at 0 — no offset,
     no board, nothing hand-measured on the target,
  5. draws Stage B on top of the detector's Stage-A overlay: magenta dot =
     radar's pick projected through the current solve, line + Δmm to the
     lidar crosshair. The final validation is unchanged: carry the reflector
     around the FoV (especially UP and DOWN) and the dot must stay glued.

Control topics (this node forwards ~/background to the detector, so ONE
command pools both sensors):
  ~/background  reflector OFF, tripod in place, you out of the radar view
  ~/capture     reflector ON, aimed (aim line OK), you out of the scene
  ~/solve ~/reset ~/save

Run
---
  ros2 run wicoms_utils radar_lidar_calib --ros-args \
    -p radar_topic:=/radar1/radar/points_all -p pc_field_snr:=intensity \
    -p prior_t_xyz:="[0.20, 0.0, 0.0]" -p prior_rpy_deg:="[-90.0, -90.0, 0.0]" \
    -p radar_name:=radar1 -p child_frame:=radar1_link

The session json is compatible with sessions/solve_from_poses_joint.py
(board pose = identity/apex, offset = 0) for offline audits.
"""
import json
import os
import time
from collections import deque

import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2, Image, CameraInfo
from geometry_msgs.msg import PointStamped, TransformStamped
from std_msgs.msg import Empty
from cv_bridge import CvBridge
from scipy.spatial.transform import Rotation as Rot
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster

try:                                             # flat-module or package layout
    from radar_camera_calib import (robust_ml_calibrate, loo_cross_val,
                                    condition_number, cluster_points,
                                    cart_to_raz, _wrap)
except ImportError:
    from wicoms_utils.radar_camera_calib import (robust_ml_calibrate, loo_cross_val,
                                                 condition_number, cluster_points,
                                                 cart_to_raz, _wrap)

_DT = {1: np.int8, 2: np.uint8, 3: np.int16, 4: np.uint16,
       5: np.int32, 6: np.uint32, 7: np.float32, 8: np.float64}


def cloud_fields(msg, names):
    """PointCloud2 → dict of named columns (float32). Missing names → None."""
    n = msg.width * msg.height
    if n == 0:
        return {k: None for k in names}
    step = msg.point_step
    buf = np.frombuffer(bytes(msg.data), np.uint8, count=n * step).reshape(n, step)
    offs = {f.name: (f.offset, f.datatype) for f in msg.fields}
    out = {}
    for name in names:
        if name in offs:
            off, dt = offs[name]
            typ = _DT.get(dt)
            w = np.dtype(typ).itemsize
            out[name] = buf[:, off:off + w].copy().view(typ).ravel().astype(np.float32)
        else:
            out[name] = None
    return out


class RadarLidarCalib(Node):
    def __init__(self):
        super().__init__('radar_lidar_calib')
        dp = self.declare_parameter
        # ── inputs ──
        dp('radar_topic', '/radar1/radar/points_all')
        dp('pc_field_x', 'x'); dp('pc_field_y', 'y'); dp('pc_field_z', 'z')
        dp('pc_field_snr', 'intensity'); dp('pc_field_doppler', 'doppler')
        dp('apex_topic', '/lidar_reflector/apex_cam')       # from the detector
        dp('base_image_topic', '/lidar_reflector/debug_image')  # Stage-A overlay
        dp('image_topic', '/zed/zed_node/left/image_rect_color')  # fallback base
        dp('info_topic', '/zed/zed_node/left/camera_info')
        dp('lidar_background_topic', '/lidar_reflector/background')  # forwarded
        # ── radar gating ──
        dp('min_range', 0.3); dp('max_range', 8.0)
        dp('range_gate_margin_m', 0.5)      # |r_radar − r_expected| gate (rotation-invariant)
        dp('max_abs_doppler', 0.15)         # tripod target is truly static
        dp('bg_frames', 15); dp('bg_match_dist', 0.2)
        dp('cluster_eps', 0.20); dp('min_cluster_size', 1)
        dp('gate_radius', 0.40)             # 3-D gate around prediction (once solve/prior)
        dp('cluster_strict', True)          # no blob near prediction → no selection
        dp('min_snr', 100.0)                # threshold on snr·(r/1.5m)^4
        dp('snr_ref_range', 1.5)
        dp('radar_range_scale', 1.0)        # ingest correction; redo §0c vs the LIDAR
        dp('radar_range_bias_m', 0.0)
        # ── noise model / solver (radar-dominated; lidar apex ~1 cm ≪ these) ──
        dp('sigma_range_m', 0.05); dp('sigma_az_deg', 3.0); dp('sigma_el_deg', 8.0)
        dp('huber_f_scale', 1.5); dp('reject_sigma', 4.0); dp('reject_axis_sigma', 3.5)
        # ── extrinsic prior (tape + nominal mounting; camera frame) ──
        dp('use_extrinsic_prior', True)
        dp('prior_t_xyz', [0.0, 0.0, 0.0]); dp('prior_rpy_deg', [0.0, 0.0, 0.0])
        dp('prior_t_sigma_m', 0.10); dp('prior_rot_sigma_deg', 5.0)
        # ── capture ──
        dp('capture_frames', 5)             # radar selections averaged per ~/capture
        dp('capture_timeout_s', 6.0)
        dp('lidar_std_mm', 15.0)            # apex jitter gate over the window
        dp('radar_std_m', 0.10)             # radar point jitter gate over the window
        dp('min_points', 12)                # solve after this many pairs
        dp('min_baseline', 0.10)
        # ── validation / output ──
        dp('measured_baseline_m', -1.0)     # tape |t| for the report (>0 to enable)
        dp('camera_frame', 'zed_left_camera_optical_frame')
        dp('child_frame', 'radar1_link'); dp('radar_name', 'radar1')
        dp('camera_name', 'zed_left')
        dp('output_path', ''); dp('publish_tf', True)
        dp('show_window', True); dp('debug_scale', 1.0)

        g = lambda k: self.get_parameter(k).value
        self.fx, self.fy, self.fz = g('pc_field_x'), g('pc_field_y'), g('pc_field_z')
        self.fsnr, self.fdop = g('pc_field_snr'), g('pc_field_doppler')
        self.min_range, self.max_range = float(g('min_range')), float(g('max_range'))
        self.rmargin = float(g('range_gate_margin_m'))
        self.max_dop = float(g('max_abs_doppler'))
        self.bg_n, self.bg_dist = int(g('bg_frames')), float(g('bg_match_dist'))
        self.ceps, self.cmin = float(g('cluster_eps')), int(g('min_cluster_size'))
        self.gate_r, self.strict = float(g('gate_radius')), bool(g('cluster_strict'))
        self.min_snr, self.snr_r0 = float(g('min_snr')), float(g('snr_ref_range'))
        self.rscale, self.rbias = float(g('radar_range_scale')), float(g('radar_range_bias_m'))
        self.sig_r = float(g('sigma_range_m'))
        self.sig_az = np.radians(float(g('sigma_az_deg')))
        self.sig_el = np.radians(float(g('sigma_el_deg')))
        self.huber, self.rej = float(g('huber_f_scale')), float(g('reject_sigma'))
        self.rej_axis = float(g('reject_axis_sigma'))
        self.use_prior = bool(g('use_extrinsic_prior'))
        self.t_prior = np.array(g('prior_t_xyz'), float)
        self.R_prior = Rot.from_euler('xyz', g('prior_rpy_deg'), degrees=True).as_matrix()
        self.t_psig, self.r_psig = float(g('prior_t_sigma_m')), np.radians(float(g('prior_rot_sigma_deg')))
        self.cap_n, self.cap_to = int(g('capture_frames')), float(g('capture_timeout_s'))
        self.lstd, self.rstd = float(g('lidar_std_mm')), float(g('radar_std_m'))
        self.min_points, self.min_base = int(g('min_points')), float(g('min_baseline'))
        self.meas_base = float(g('measured_baseline_m'))
        self.camera_frame, self.child_frame = g('camera_frame'), g('child_frame')
        self.radar_name, self.camera_name = g('radar_name'), g('camera_name')
        self.out_path = g('output_path') or f'extrinsic_{g("camera_name")}__{self.radar_name}_lidar'
        self.publish_tf, self.show_window = bool(g('publish_tf')), bool(g('show_window'))
        self.dscale = float(g('debug_scale'))

        self.bridge = CvBridge()
        self.K = self.D = None
        self.base_img = None; self.base_t = 0.0
        self.raw_img = None
        self.apex = None; self.apex_t = 0.0     # latest lidar apex (camera frame)
        self.bg_pts = None; self.bg_accum = []; self.bg_want = 0
        self.sel = None                          # latest radar selection dict
        self.aim = ('no lidar apex yet', (0, 0, 255))
        self.cap_deadline = 0.0; self.cap_lidar = []; self.cap_radar = []
        self.captures = []
        self.solution = None
        self.tfb = StaticTransformBroadcaster(self) if self.publish_tf else None

        qs = qos_profile_sensor_data
        self.create_subscription(PointCloud2, g('radar_topic'), self._radar, qs)
        self.create_subscription(PointStamped, g('apex_topic'), self._apex, 20)
        self.create_subscription(Image, g('base_image_topic'), self._base, qs)
        self.create_subscription(Image, g('image_topic'), self._raw, qs)
        self.create_subscription(CameraInfo, g('info_topic'), self._info, qs)
        self.create_subscription(Empty, '~/background', lambda _: self._bg_start(), 1)
        self.create_subscription(Empty, '~/capture', lambda _: self._arm(), 1)
        self.create_subscription(Empty, '~/solve', lambda _: self._solve(force=True), 1)
        self.create_subscription(Empty, '~/reset', lambda _: self._reset(), 1)
        self.create_subscription(Empty, '~/save', lambda _: self._save(), 1)
        self.fwd_bg = self.create_publisher(Empty, g('lidar_background_topic'), 1)
        self.pub_dbg = self.create_publisher(Image, '~/debug_image', 2)
        self.create_timer(0.05, self._gui)
        self.get_logger().info(
            f'radar_lidar_calib ready — solving T_{self.camera_frame}_{self.child_frame} '
            f'from lidar apexes. 1) ~/background (reflector OFF)  2) mount+aim '
            f'(watch the aim line)  3) ~/capture  4) repeat 12-20 placements')

    # ── plumbing ──
    def _info(self, m):
        if self.K is None:
            self.K = np.array(m.k).reshape(3, 3)
            self.D = np.array(m.d) if len(m.d) else np.zeros(5)

    def _base(self, m):
        self.base_img = self.bridge.imgmsg_to_cv2(m, 'bgr8'); self.base_t = time.time()

    def _raw(self, m):
        self.raw_img = self.bridge.imgmsg_to_cv2(m, 'bgr8')

    def _apex(self, m):
        self.apex = np.array([m.point.x, m.point.y, m.point.z]); self.apex_t = time.time()
        if self.cap_deadline > time.time():
            self.cap_lidar.append(self.apex.copy())

    # ── control ──
    def _bg_start(self):
        self.bg_accum, self.bg_want, self.bg_pts = [], self.bg_n, None
        self.fwd_bg.publish(Empty())                     # one trigger pools BOTH sensors
        self.get_logger().info(f'pooling radar background ({self.bg_n} frames) '
                               f'+ forwarded to lidar detector — reflector OFF, stay clear')

    def _arm(self):
        if self.bg_pts is None:
            self.get_logger().warn('capture refused: no radar background — ~/background first')
            return
        if self.apex is None or time.time() - self.apex_t > 1.0:
            self.get_logger().warn('capture refused: no live lidar apex (reflector on? detector running?)')
            return
        self.cap_lidar, self.cap_radar = [], []
        self.cap_deadline = time.time() + self.cap_to
        self.get_logger().info(f'capture armed: pairing next {self.cap_n} radar selections '
                               f'(timeout {self.cap_to:.0f} s)')

    def _reset(self):
        self.captures, self.solution = [], None
        self.get_logger().info('captures cleared')

    # ── the extrinsic used for prediction/overlay: solve first, else prior ──
    def _current_T(self):
        if self.solution is not None:
            return self.solution['R'], self.solution['t']
        if self.use_prior:
            return self.R_prior, self.t_prior
        return None, None

    # ── radar pipeline, one cloud at a time ──
    def _radar(self, msg):
        f = cloud_fields(msg, [self.fx, self.fy, self.fz, self.fsnr, self.fdop])
        x, y, z = f[self.fx], f[self.fy], f[self.fz]
        if x is None:
            return
        xyz = np.stack([x, y, z if z is not None else np.zeros_like(x)], 1)
        snr = f[self.fsnr] if f[self.fsnr] is not None else np.ones(len(xyz))
        dop = f[self.fdop]
        if self.rscale != 1.0 or self.rbias != 0.0:      # ingest range correction
            r = np.linalg.norm(xyz, axis=1)
            ok = r > 1e-6
            xyz[ok] *= ((self.rscale * r[ok] + self.rbias) / r[ok])[:, None]
        r = np.linalg.norm(xyz, axis=1)
        keep = np.isfinite(r) & (r > self.min_range) & (r < self.max_range)

        if self.bg_want > 0:                             # background pooling mode
            self.bg_accum.append(xyz[keep])
            self.bg_want -= 1
            if self.bg_want == 0:
                self.bg_pts = np.concatenate(self.bg_accum) if self.bg_accum else np.zeros((0, 3))
                self.get_logger().info(f'radar background ready: {len(self.bg_pts)} points')
            return
        if self.bg_pts is not None and len(self.bg_pts) and keep.any():
            d = np.linalg.norm(xyz[keep][:, None, :] - self.bg_pts[None, :, :], axis=2).min(1)
            kk = np.where(keep)[0]
            keep[kk[d <= self.bg_dist]] = False

        self.sel = None
        if self.apex is None or time.time() - self.apex_t > 1.0:
            self.aim = ('no lidar apex — mount reflector / check detector', (0, 0, 255))
            return
        R, t = self._current_T()
        r_exp = np.linalg.norm(self.apex - (t if t is not None else 0.0))
        keep &= np.abs(r - r_exp) <= self.rmargin        # rotation-invariant range gate
        if self.max_dop > 0 and dop is not None:
            keep &= np.abs(dop) <= self.max_dop
        pts, s = xyz[keep], snr[keep]
        if len(pts) == 0:
            self.aim = ('radar: no return near lidar range — RE-AIM or unblock', (0, 0, 255))
            return

        pred = R.T @ (self.apex - t) if R is not None else None
        if pred is not None:                             # 3-D gate around prediction
            near = np.linalg.norm(pts - pred, axis=1) <= self.gate_r
            if near.any():
                pts, s = pts[near], s[near]
            elif self.strict and self.solution is not None:
                self.aim = (f'radar: nothing within {self.gate_r:.2f} m of prediction', (0, 100, 255))
                return
        clusters = cluster_points(pts, self.ceps, self.cmin)
        if not clusters:
            self.aim = ('radar: no cluster after gating', (0, 100, 255))
            return
        if pred is not None:
            cent = [pts[c].mean(0) for c in clusters]
            ci = int(np.argmin([np.linalg.norm(c - pred) for c in cent]))
        else:
            ci = int(np.argmax([s[c].max() for c in clusters]))
        c = clusters[ci]
        w = s[c] / max(s[c].sum(), 1e-9)
        p_sel = (pts[c] * w[:, None]).sum(0)             # SNR-weighted blob centroid
        snr_sel = float(s[c].max())
        r_sel = float(np.linalg.norm(p_sel))
        snr_norm = snr_sel * (r_sel / self.snr_r0) ** 4  # aim metric, range-fair
        ok = snr_norm >= self.min_snr
        self.sel = dict(p=p_sel, snr=snr_sel, snr_norm=snr_norm, r=r_sel, n=len(c))
        self.aim = (f'radar: best {snr_sel:.0f} (norm {snr_norm:.0f}) @ {r_sel:.2f} m  '
                    + ('OK' if ok else 'RE-AIM'),
                    (0, 220, 0) if ok else (0, 165, 255))

        if self.cap_deadline > time.time() and ok:
            self.cap_radar.append(p_sel.copy())
            if len(self.cap_radar) >= self.cap_n:
                self.cap_deadline = 0.0
                self._finish_capture()
        elif self.cap_deadline and time.time() > self.cap_deadline:
            self.cap_deadline = 0.0
            self.get_logger().warn(
                f'capture REFUSED: timeout — only {len(self.cap_radar)}/{self.cap_n} passing radar '
                f'frames in {self.cap_to:.0f} s (last aim: {self.aim[0]})')

    # ── atomic pair: both sides must pass ──
    def _finish_capture(self):
        if len(self.cap_lidar) < 3:
            self.get_logger().warn('capture REFUSED: too few lidar apex updates in window')
            return
        L, Rr = np.stack(self.cap_lidar), np.stack(self.cap_radar)
        lstd = float(np.linalg.norm(L.std(0)) * 1000)
        rstd = float(np.linalg.norm(Rr.std(0)))
        if lstd > self.lstd:
            self.get_logger().warn(f'capture REFUSED: lidar apex std {lstd:.1f} mm > {self.lstd}')
            return
        if rstd > self.rstd:
            self.get_logger().warn(f'capture REFUSED: radar point std {rstd*100:.0f} cm > '
                                   f'{self.rstd*100:.0f} (multipath flicker? re-aim/move slightly)')
            return
        p_cam, p_radar = L.mean(0), Rr.mean(0)
        for i, cp in enumerate(self.captures):
            if np.linalg.norm(np.array(cp['p_cam']) - p_cam) < self.min_base:
                self.get_logger().warn(f'note: close to capture #{i+1} — move the tripod more')
                break
        self.captures.append(dict(
            idx=len(self.captures) + 1, stamp=time.time(),
            p_cam=[round(float(v), 4) for v in p_cam],
            p_radar=[round(float(v), 4) for v in p_radar],
            board_R_quat_xyzw=[0.0, 0.0, 0.0, 1.0],      # identity: apex IS p_cam
            board_t=[round(float(v), 4) for v in p_cam], # (solve_from_poses compat)
            snr=round(float(self.sel['snr']), 1), lidar_std_mm=round(lstd, 1),
            radar_std_mm=round(rstd * 1000, 1)))
        self.get_logger().info(
            f'*** CAPTURED #{len(self.captures)}  cam [{p_cam[0]:.3f}, {p_cam[1]:.3f}, '
            f'{p_cam[2]:.3f}]  radar [{p_radar[0]:.3f}, {p_radar[1]:.3f}, {p_radar[2]:.3f}]  '
            f'snr {self.sel["snr"]:.0f} ***')
        if len(self.captures) >= self.min_points:
            self._solve()
        self._save(quiet=True)

    # ── solve + report (imported measurement-space ML; offset pinned at 0) ──
    def _solve(self, force=False):
        n = len(self.captures)
        if n < (4 if force else self.min_points):
            self.get_logger().info(f'{n} captures — solving at {self.min_points} (~/solve to force)')
            return
        P = np.array([c['p_radar'] for c in self.captures])
        Q = np.array([c['p_cam'] for c in self.captures])
        I = np.repeat(np.eye(3)[None], n, axis=0)
        res = robust_ml_calibrate(
            P, I, Q, np.zeros(3), self.sig_r, self.sig_az, self.sig_el,
            use_elevation=True, solve_offset=False,
            R_prior=self.R_prior if self.use_prior else None,
            t_prior=self.t_prior if self.use_prior else None,
            rot_prior_sigma=self.r_psig if self.use_prior else None,
            t_prior_sigma=self.t_psig if self.use_prior else None,
            huber=self.huber, reject_sigma=self.rej, reject_axis_sigma=self.rej_axis)
        R, t, mask = res['R'], res['t'], res['inlier_mask']
        self.solution = res
        Pin, Qin = P[mask], Q[mask]
        cov = res['cov']
        sig = np.sqrt(np.clip(np.diag(cov), 0, None))
        rot1s, t1s = np.degrees(sig[:3]), sig[3:] * 1000
        pred = (R @ Pin.T).T + t
        err = pred - Qin
        bias = err.mean(0) * 1000
        rms = np.sqrt((err ** 2).mean(0)) * 1000
        loo = loo_cross_val(P[mask], I[mask], Q[mask], np.zeros(3),
                            (self.sig_r, self.sig_az, self.sig_el), True)
        cond = condition_number(Pin)
        raz = np.array([cart_to_raz(p) for p in Pin])
        spread = (raz[:, 0].ptp(), np.degrees(raz[:, 1].ptp()), np.degrees(raz[:, 2].ptp()))
        q = Rot.from_matrix(R).as_quat()
        rpy = Rot.from_matrix(R).as_euler('xyz', degrees=True)
        cam_r, rad_r = np.linalg.norm(Qin - t, axis=1), np.linalg.norm(Pin, axis=1)
        a, b = np.linalg.lstsq(np.vstack([rad_r, np.ones_like(rad_r)]).T, cam_r, rcond=None)[0]
        lines = [
            f'=== T_{self.camera_frame}_{self.child_frame}  (camera <- radar, LIDAR reference) ===',
            f'  captures {n}   inliers {res["n_in"]}/{n}   residual {res["rms_sigma"]:.2f} s   cond {cond:.1f}',
            f'  xyz (m) : {t[0]:+.4f} {t[1]:+.4f} {t[2]:+.4f}   |t| {np.linalg.norm(t)*100:.1f} cm',
            f'  quat    : {q[0]:+.4f} {q[1]:+.4f} {q[2]:+.4f} {q[3]:+.4f}',
            f'  rpy(deg): {rpy[0]:+.2f} {rpy[1]:+.2f} {rpy[2]:+.2f}   (gimbal-locked near pitch -90: compare quats)',
            f'  1s rot  : {rot1s[0]:.2f} {rot1s[1]:.2f} {rot1s[2]:.2f} deg   '
            f'1s t: {t1s[0]:.1f} {t1s[1]:.1f} {t1s[2]:.1f} mm',
            f'  spread  : range {spread[0]*100:.0f} cm, az {spread[1]:.0f} deg, el {spread[2]:.0f} deg',
            f'  bias mm : X {bias[0]:+.0f} Y {bias[1]:+.0f} Z {bias[2]:+.0f}   '
            f'3-D RMS mm: {rms[0]:.0f} {rms[1]:.0f} {rms[2]:.0f}',
        ]
        if loo:
            lines.append(f'  LOO CV  : {loo[0]:.2f} s (max {loo[1]:.2f})')
        if abs(a - 1) > 0.02 or abs(b) > 0.05:
            lines.append(f'  range fit: cam_r = {a:.3f}*radar_r {b:+.3f} m (want a~1) -> '
                         f'set radar_range_scale={a*self.rscale:.4f}')
        if self.meas_base > 0:
            d = abs(np.linalg.norm(t) - self.meas_base)
            lines.append(f'  baseline: |t| {np.linalg.norm(t)*100:.1f} vs tape '
                         f'{self.meas_base*100:.1f} cm -> D {d*100:.1f} cm '
                         f'[{"OK" if d <= 0.05 else "MISMATCH"}]')
        gates = [('residual~1s', res['rms_sigma'] <= 1.5), ('cond<=5', cond <= 5),
                 ('rot1s<=4deg', rot1s.max() <= 4), ('bias<=50mm', np.abs(bias).max() <= 50)]
        lines.append('  GATES   : ' + '  '.join(f'{k}[{"P" if v else "F"}]' for k, v in gates)
                     + '   + overlay up/down test before ~/save')
        self.get_logger().info('\n' + '\n'.join(lines))
        if self.tfb is not None:
            tf = TransformStamped()
            tf.header.stamp = self.get_clock().now().to_msg()
            tf.header.frame_id = self.camera_frame
            tf.child_frame_id = self.child_frame
            tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z = map(float, t)
            (tf.transform.rotation.x, tf.transform.rotation.y,
             tf.transform.rotation.z, tf.transform.rotation.w) = map(float, q)
            self.tfb.sendTransform(tf)

    def _save(self, quiet=False):
        g = lambda k: self.get_parameter(k).value
        out = dict(kind='radar_lidar_session', stamp=time.time(),
                   parent_frame=self.camera_frame, child_frame=self.child_frame,
                   camera_name=self.camera_name, radar_name=self.radar_name,
                   params=dict(
                       sigma_range_m=self.sig_r, sigma_az_deg=np.degrees(self.sig_az),
                       sigma_el_deg=np.degrees(self.sig_el),
                       reflector_offset_x=0.0, reflector_offset_y=0.0, reflector_offset_z=0.0,
                       use_extrinsic_prior=self.use_prior,
                       prior_t_xyz=[float(v) for v in self.t_prior],
                       prior_rpy_deg=list(g('prior_rpy_deg')),
                       prior_t_sigma_m=self.t_psig,
                       prior_rot_sigma_deg=float(np.degrees(self.r_psig)),
                       radar_range_scale=self.rscale, radar_range_bias_m=self.rbias,
                       reject_sigma=self.rej, reject_axis_sigma=self.rej_axis),
                   captures=self.captures)
        if self.solution is not None:
            R, t = self.solution['R'], self.solution['t']
            q = Rot.from_matrix(R).as_quat()
            out['result'] = dict(
                T_cam_radar_translation=[float(v) for v in t],
                T_cam_radar_quaternion_xyzw=[float(v) for v in q],
                n_inliers=int(self.solution['n_in']),
                residual_rms_sigma=float(self.solution['rms_sigma']),
                static_tf_cmd=('ros2 run tf2_ros static_transform_publisher '
                               + ' '.join(f'{v:.6f}' for v in t) + ' '
                               + ' '.join(f'{v:.6f}' for v in q)
                               + f' {self.camera_frame} {self.child_frame}'))
        path = self.out_path + '_session.json'
        with open(path, 'w') as f:
            json.dump(out, f, indent=1)
        if not quiet:
            self.get_logger().info(f'saved {len(self.captures)} captures -> {os.path.abspath(path)}')

    # ── Stage-B overlay on top of the detector's Stage-A image ──
    def _project(self, p):
        p = np.asarray(p, np.float64)
        if p[2] <= 0.05:
            return None
        uv, _ = cv2.projectPoints(p.reshape(1, 1, 3), np.zeros(3), np.zeros(3), self.K, self.D)
        return int(uv[0, 0, 0]), int(uv[0, 0, 1])

    def _gui(self):
        base = self.base_img if (self.base_img is not None and time.time() - self.base_t < 1.0) \
            else self.raw_img
        if base is None or self.K is None:
            return
        im = base.copy()
        h = im.shape[0]

        def bline(i, txt, col):
            y = h - 12 - 22 * i
            cv2.putText(im, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 0, 0), 3)
            cv2.putText(im, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX, .55, col, 1)

        bline(0, self.aim[0], self.aim[1])
        if self.solution is not None:
            s = self.solution
            bline(1, f'solve: n={len(self.captures)} inl={s["n_in"]} residual {s["rms_sigma"]:.2f}s',
                  (0, 220, 0) if s['rms_sigma'] <= 1.5 else (0, 165, 255))
        elif self.bg_pts is None:
            bline(1, 'radar: NO BACKGROUND - ~/background first', (0, 0, 255))
        else:
            bline(1, f'pairs {len(self.captures)}/{self.min_points} before first solve', (240, 240, 240))
        if self.cap_deadline > time.time():
            bline(2, f'CAPTURING... {len(self.cap_radar)}/{self.cap_n} radar frames', (0, 255, 255))

        # Stage B: the radar's pick, through the current solve, vs the lidar apex
        R, t = self._current_T()
        if self.sel is not None and R is not None:
            p_cam_radar = R @ self.sel['p'] + t
            uv = self._project(p_cam_radar)
            if uv:
                cv2.circle(im, uv, 7, (255, 0, 255), 2)
                if self.apex is not None and time.time() - self.apex_t < 1.0:
                    av = self._project(self.apex)
                    if av:
                        cv2.line(im, uv, av, (255, 0, 255), 1)
                        dmm = np.linalg.norm(p_cam_radar - self.apex) * 1000
                        tag = 'prior' if self.solution is None else 'solved'
                        cv2.putText(im, f'D {dmm:.0f} mm ({tag})', (uv[0] + 10, uv[1] + 16),
                                    cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 0), 3)
                        cv2.putText(im, f'D {dmm:.0f} mm ({tag})', (uv[0] + 10, uv[1] + 16),
                                    cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 0, 255), 1)

        if self.dscale != 1.0:
            im = cv2.resize(im, None, fx=self.dscale, fy=self.dscale)
        self.pub_dbg.publish(self.bridge.cv2_to_imgmsg(im, 'bgr8'))
        if self.show_window:
            cv2.imshow('radar_lidar_calib (Stage A+B)', im)
            cv2.waitKey(1)


def main():
    rclpy.init()
    node = RadarLidarCalib()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node._save()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
