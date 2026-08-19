#!/usr/bin/env python3
"""
RADAR ↔ LIDAR extrinsic calibration  —  solves T_lidar_radar
=============================================================

ONE node, no board, no hand-measured offsets. The whole target is a corner
reflector on a tripod. The SOLVE lives entirely in the LIDAR frame:

    p_lidar = R · p_radar + t          (X = T_lidar_radar)

The camera is OPTIONAL and never enters the solve. Given the GLIM lidar↔camera
transform it is used for two things only: composing the deployable
T_cam_radar = T_cam_lidar · T_lidar_radar, and drawing the ZED image overlay.
So an error in the lidar↔camera calibration shows up in the composed output and
the overlay but cannot corrupt the radar↔lidar result — and re-running GLIM
later lets you recompose without recollecting any radar data.

How the reflector is found
--------------------------
LIDAR — it does not look for a "reflector". It looks for what is NEW:
  1. ~/background memorises the empty scene on a voxel grid (walls, floor AND
     the tripod — the reflector must be OFF the tripod for this),
  2. every later cloud is background-subtracted, so the only surviving points
     are the object that was not there before = the reflector,
  3. the survivors are clustered, and the APEX is localised by RANSAC-fitting
     the three plates and intersecting them (analytic corner, nothing
     measured). Fallback when the plates are too sparse: the farthest point
     along the viewing ray, which for a reflector aimed at the sensor IS the
     corner.
  The premise is "nothing else changed since the background" — hence the
  per-placement loop: background → mount reflector → step out → capture.

RADAR — background subtraction, then a ROTATION-INVARIANT range gate around
  the lidar's range (you do not know R yet, but |p| is the same in any
  orientation), |doppler|≈0 (a tripod target is truly static), clustering, and
  the SNR-weighted centroid of the best blob. Once a solve exists it tightens
  to a 3-D gate around the predicted point.

Aim feedback (before you trigger)
---------------------------------
The status marker / log shows continuously:
    radar: best 1240 (norm 1870) @ 2.31 m  OK        → capture will pass
    radar: best 41 (norm 62) @ 2.28 m  RE-AIM        → fix the aim first
    radar: no return near lidar range                → aimed way off / blocked
`norm` = snr·(r/1.5 m)^4. Received power falls as 1/r^4, so a raw SNR
threshold would wrongly refuse a well-aimed reflector at 4 m.

A capture is ATOMIC: both sensors must pass their gates or it is REFUSED with
the reason logged. Nothing half-good is ever stored.

RViz verification (plus the ZED image overlay when the camera is configured)
--------------------------------------------------
Fixed Frame = your lidar frame. Add the lidar PointCloud2 and a MarkerArray on
~/markers:
  cyan points   the foreground cluster        → must sit on the reflector only
  green sphere  the detected apex             → must sit on its corner
  amber spheres captured apexes, numbered     → your coverage map
  magenta       the radar's pick through the current solve (after first solve)
  magenta line  radar pick ↔ lidar apex, labelled with the gap in mm
  text          aim status + capture/solve state
FINAL CHECK: carry the reflector around — near/far, left/right, UP/DOWN — and
the magenta sphere must stay on the green one. If it mirrors when you raise the
reflector, the rotation is in the wrong branch: do not save.

Run
---
  ros2 run wicoms_utils radar_lidar_calib --ros-args \
    -p lidar_topic:=/ouster/points \
    -p radar_topic:=/radar1/radar/points_all -p pc_field_snr:=intensity \
    -p radar_name:=radar1 -p child_frame:=radar1_link

  ros2 topic pub -1 /radar_lidar_calib/background std_msgs/msg/Empty "{}"
  ros2 topic pub -1 /radar_lidar_calib/capture    std_msgs/msg/Empty "{}"
  ros2 topic pub -1 /radar_lidar_calib/solve      std_msgs/msg/Empty "{}"
  ros2 topic pub -1 /radar_lidar_calib/save       std_msgs/msg/Empty "{}"
  ros2 topic pub -1 /radar_lidar_calib/reset      std_msgs/msg/Empty "{}"

Camera parameters default to the GLIM result for this rig
(`lidar_camera_xyz` / `lidar_camera_quat_xyzw`, direction `lidar_camera`,
i.e. the given transform maps CAMERA points into the LIDAR frame and is
inverted internally). Set `show_image_overlay:=false` to run headless, or
override `camera_transform_is:=camera_lidar` if your file stores the other
direction.

Structure: [A] cloud tools · [B] apex locator · [C] node.
"""
import json
import os
import time
from collections import deque

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2, Image, CameraInfo
from geometry_msgs.msg import PointStamped, TransformStamped, Point
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Empty, ColorRGBA
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as Rot
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster

try:                                    # optional: only for the image overlay
    import cv2
    from cv_bridge import CvBridge
    _HAVE_CV = True
except ImportError:
    _HAVE_CV = False

try:                                    # flat-module or installed-package layout
    from radar_camera_calib import (robust_ml_calibrate, loo_cross_val,
                                    condition_number, cluster_points as radar_cluster,
                                    cart_to_raz)
except ImportError:
    from wicoms_utils.radar_camera_calib import (robust_ml_calibrate, loo_cross_val,
                                                 condition_number,
                                                 cluster_points as radar_cluster,
                                                 cart_to_raz)


# ────────────────────────────── [A] cloud tools ──────────────────────────────
_DT = {1: np.int8, 2: np.uint8, 3: np.int16, 4: np.uint16,
       5: np.int32, 6: np.uint32, 7: np.float32, 8: np.float64}


def cloud_fields(msg, names):
    """PointCloud2 → dict of named float32 columns (missing name → None).
    Parsed straight from the buffer; the read_points generator is far too slow
    for a 64×1024 Ouster at 10-20 Hz."""
    n = msg.width * msg.height
    if n == 0:
        return {k: None for k in names}
    step = msg.point_step
    buf = np.frombuffer(bytes(msg.data), np.uint8, count=n * step).reshape(n, step)
    offs = {f.name: (f.offset, f.datatype) for f in msg.fields}
    out = {}
    for name in names:
        if name in offs and _DT.get(offs[name][1]) is not None:
            off, dt = offs[name]
            typ = _DT[dt]
            w = np.dtype(typ).itemsize
            out[name] = buf[:, off:off + w].copy().view(typ).ravel().astype(np.float32)
        else:
            out[name] = None
    return out


def cloud_xyz(msg):
    f = cloud_fields(msg, ['x', 'y', 'z'])
    if f['x'] is None or f['y'] is None or f['z'] is None:
        return np.zeros((0, 3), np.float32)
    return np.stack([f['x'], f['y'], f['z']], 1)


# Voxel background set: keys are packed int64s, membership tested against the
# 27-neighbourhood so "within ~one voxel of any background point" counts as
# background. Vectorised with sort + searchsorted (no per-point python).
_OFF = 1 << 20
_SX, _SY = 1 << 42, 1 << 21
_NEIGH = np.array([dx * _SX + dy * _SY + dz
                   for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)],
                  dtype=np.int64)


def voxel_keys(xyz, voxel):
    idx = np.floor(xyz / voxel).astype(np.int64) + _OFF
    return idx[:, 0] * _SX + idx[:, 1] * _SY + idx[:, 2]


def foreground_mask(xyz, bg_sorted, voxel):
    keys = voxel_keys(xyz, voxel)
    hit = np.zeros(len(keys), bool)
    for d in _NEIGH:
        k = keys + d
        i = np.clip(np.searchsorted(bg_sorted, k), 0, len(bg_sorted) - 1)
        hit |= (bg_sorted[i] == k)
    return ~hit


def lidar_cluster(P, eps, min_size, cap=20000):
    """Connected components within `eps` via cKDTree. Returns point arrays,
    largest first. The foreground should be tiny (just the reflector); the cap
    is a defensive decimation for when the background went stale."""
    if len(P) == 0:
        return []
    if len(P) > cap:
        P = P[np.random.choice(len(P), cap, replace=False)]
    tree = cKDTree(P)
    lab = np.full(len(P), -1, int)
    out, cid = [], 0
    for i in range(len(P)):
        if lab[i] >= 0:
            continue
        stack, members = [i], [i]
        lab[i] = cid
        while stack:
            j = stack.pop()
            for k in tree.query_ball_point(P[j], eps):
                if lab[k] < 0:
                    lab[k] = cid
                    stack.append(k)
                    members.append(k)
        if len(members) >= min_size:
            out.append(np.array(members))
        cid += 1
    out.sort(key=len, reverse=True)
    return [P[m] for m in out]


def grow_from_seed(raw, seed, eps, max_r):
    """Recover the WHOLE object from a partial detection.

    Background subtraction erases everything within ~bg_voxel of a memorised
    point, so a reflector bolted to a memorised tripod head keeps only its top.
    Those surviving points are still a reliable SEED: region-grow from them back
    through the RAW cloud (single-linkage at `eps`) to pull in the erased body.

    Bounded to `max_r` of the seed centroid so the growth cannot run down the
    tripod legs — it stops after the reflector plus, at worst, the head, which
    the plane fit then rejects as outliers.
    """
    c0 = seed.mean(0)
    near = raw[np.linalg.norm(raw - c0, axis=1) <= max_r]
    if len(near) < len(seed):
        return seed
    tree = cKDTree(near)
    seen = np.zeros(len(near), bool)
    stack = []
    for p in seed:
        for i in tree.query_ball_point(p, eps):
            if not seen[i]:
                seen[i] = True
                stack.append(i)
    while stack:
        for k in tree.query_ball_point(near[stack.pop()], eps):
            if not seen[k]:
                seen[k] = True
                stack.append(k)
    return near[seen] if seen.sum() >= len(seed) else seed


# ────────────────────────────── [B] apex locator ─────────────────────────────
def ransac_planes(P, tol, iters, min_pts, rng):
    """Sequentially RANSAC up to three planes; each refined by SVD on its
    inliers, whose points are removed before the next fit."""
    planes, pts = [], P.copy()
    for _ in range(3):
        if len(pts) < min_pts:
            break
        best = None
        for _ in range(iters):
            a, b, c = pts[rng.choice(len(pts), 3, replace=False)]
            n = np.cross(b - a, c - a)
            nn = np.linalg.norm(n)
            if nn < 1e-9:
                continue
            n = n / nn
            inl = np.abs(pts @ n - n @ a) < tol
            if best is None or inl.sum() > best[0]:
                best = (int(inl.sum()), inl)
        if best is None or best[0] < min_pts:
            break
        Q = pts[best[1]]
        c0 = Q.mean(0)
        n = np.linalg.svd(Q - c0)[2][2]
        planes.append((n, float(n @ c0)))
        pts = pts[~best[1]]
    return planes


def locate_apex(P, tol, iters, min_pts, perp_tol_deg, rng):
    """Trihedral apex from its point cluster.
    'planes3'  three mutually ~perpendicular planes intersected → exact corner.
    'deepest'  farthest points along the viewing ray (the corner of a reflector
               aimed at the sensor is its deepest point). Used when the plates
               are too sparse to fit — i.e. at longer range."""
    planes = ransac_planes(P, tol, iters, min_pts, rng)
    if len(planes) == 3:
        budget = np.cos(np.radians(90.0 - perp_tol_deg))
        if all(abs(planes[i][0] @ planes[j][0]) < budget
               for i in range(3) for j in range(i + 1, 3)):
            N = np.stack([p[0] for p in planes])
            d = np.array([p[1] for p in planes])
            try:
                apex = np.linalg.solve(N, d)
                if np.min(np.linalg.norm(P - apex, axis=1)) < 0.10:   # must touch the cluster
                    return apex, 'planes3'
            except np.linalg.LinAlgError:
                pass
    u = P.mean(0)
    u = u / (np.linalg.norm(u) + 1e-9)
    k = min(8, len(P))
    return P[np.argsort(P @ u)[-k:]].mean(0), 'deepest'


# ─────────────────────────────── [C] the node ────────────────────────────────
CYAN = ColorRGBA(r=0.1, g=0.85, b=0.95, a=1.0)
GREEN = ColorRGBA(r=0.1, g=1.0, b=0.2, a=1.0)
AMBER = ColorRGBA(r=1.0, g=0.75, b=0.1, a=0.9)
MAGENTA = ColorRGBA(r=1.0, g=0.1, b=0.9, a=1.0)
WHITE = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)


class RadarLidarCalib(Node):
    def __init__(self):
        super().__init__('radar_lidar_calib')
        dp = self.declare_parameter
        # ── topics ──
        dp('lidar_topic', '/ouster/points')
        dp('radar_topic', '/radar1/radar/points_all')
        dp('pc_field_x', 'x'); dp('pc_field_y', 'y'); dp('pc_field_z', 'z')
        dp('pc_field_snr', 'intensity'); dp('pc_field_doppler', 'doppler')
        # ── lidar detection ──
        dp('lidar_min_range', 0.3); dp('lidar_max_range', 8.0)
        dp('bg_frames_lidar', 10)
        dp('bg_voxel', 0.05)                 # background match distance (m). Anything
                                             # within this of a background point is
                                             # erased (up to 3.5x that in the worst
                                             # case), so keep it well under the gap
                                             # between reflector and tripod head.
        dp('cluster_eps', 0.12)              # foreground clustering radius (m)
        dp('min_cluster_size', 8)
        dp('grow_radius', 0.30)              # region-grow the seed back into the raw
                                             # cloud out to this radius (0 = off), to
                                             # recover the part of the reflector that
                                             # background subtraction erased
        dp('grow_eps', 0.06)                 # connectivity for that growth (m)
        dp('plane_tol', 0.015)               # RANSAC inlier distance (m)
        dp('plane_iters', 250)
        dp('min_plane_pts', 12)              # per plate — sets max planes3 range
        dp('perp_tol_deg', 25.0)
        # ── radar detection ──
        dp('radar_min_range', 0.3); dp('radar_max_range', 8.0)
        dp('bg_frames_radar', 15); dp('bg_match_dist', 0.2)
        dp('range_gate_margin_m', -1.0)      # <=0 = OFF (default). The reflector is
                                             # the brightest NEW thing in a
                                             # background-subtracted scene, so max-SNR
                                             # needs no range assumption. Enable only
                                             # with a real prior_t_xyz: the gate is
                                             # centred on |apex - t_radar|, so a
                                             # zeroed guess on a rig with any baseline
                                             # rejects the genuine return.
        dp('max_abs_doppler', 0.15)          # tripod target is genuinely static
        dp('radar_cluster_eps', 0.20); dp('radar_min_cluster_size', 3)
        # A single radar frame is not a reliable pick: detections flicker between
        # the reflector and multipath. Pool the last N frames and cluster the
        # accumulation instead — the static reflector lands in the same place
        # every frame (dense, persistent cluster), noise appears once and moves.
        dp('radar_accum_frames', 10)
        dp('radar_min_frames', 3)            # cluster must appear in >= this many frames
        dp('gate_radius', 0.40)              # 3-D gate once a solve exists
        dp('cluster_strict', True)
        dp('min_snr', 100.0)                 # threshold on snr·(r/ref)^4
        dp('snr_ref_range', 1.5)
        dp('radar_range_scale', 1.0)         # ingest correction; tune until a≈1
        dp('radar_range_bias_m', 0.0)
        # ── noise model + solver (radar-dominated; lidar apex ~1 cm ≪ these) ──
        dp('sigma_range_m', 0.05); dp('sigma_az_deg', 3.0); dp('sigma_el_deg', 8.0)
        dp('huber_f_scale', 1.5); dp('reject_sigma', 4.0); dp('reject_axis_sigma', 3.5)
        # ── radar position guess + optional solver prior ──
        #   prior_t_xyz is ALWAYS used to predict the radar's range to the target
        #   (r_exp = |apex - t|). That is pure gating, not regularisation: get it
        #   wrong by more than range_gate_margin_m and the real return is thrown
        #   away. use_extrinsic_prior controls only whether it also biases the SOLVE.
        dp('use_extrinsic_prior', False)
        dp('prior_t_xyz', [0.0, 0.0, 0.0]); dp('prior_rpy_deg', [0.0, 0.0, 0.0])
        dp('prior_t_sigma_m', 0.15); dp('prior_rot_sigma_deg', 15.0)
        # ── capture ──
        dp('capture_frames', 5)              # radar selections averaged per capture
        dp('capture_timeout_s', 6.0)
        dp('lidar_std_mm', 15.0)             # apex jitter gate
        dp('radar_std_m', 0.10)              # radar point jitter gate
        dp('min_points', 12)                 # first solve at this many pairs
        dp('min_baseline', 0.15)
        # ── output ──
        dp('measured_baseline_m', -1.0)      # tape lidar→radar distance (check only)
        dp('child_frame', 'radar1_link'); dp('radar_name', 'radar1')
        dp('lidar_name', 'ouster')
        dp('output_path', ''); dp('publish_tf', True)
        dp('status_marker_xyz', [2.0, 0.0, 1.0])
        # ── OPTIONAL camera (verification + composed output only; the SOLVE never
        #    uses it, so an error here cannot corrupt the radar calibration) ──
        dp('camera_frame', 'zed_left_camera_optical_frame')
        dp('image_topic', '/zed/zed_node/left/image_rect_color')
        dp('info_topic', '/zed/zed_node/left/camera_info')
        dp('show_image_overlay', True)     # build/publish the ZED overlay
        dp('show_window', True)            # ALSO pop a native cv2 window for it
        dp('debug_scale', 1.0)             # shrink the published/shown overlay
        # GLIM output, os_lidar -> zed_left_camera_optical_frame (T_lidar_camera)
        dp('lidar_camera_xyz', [-0.074928, -0.066971, -0.091627])
        dp('lidar_camera_quat_xyzw', [-0.497829, -0.498035, 0.501789, 0.502329])
        dp('camera_transform_is', 'lidar_camera')   # 'lidar_camera' | 'camera_lidar'

        g = lambda k: self.get_parameter(k).value
        self.fx, self.fy, self.fz = g('pc_field_x'), g('pc_field_y'), g('pc_field_z')
        self.fsnr, self.fdop = g('pc_field_snr'), g('pc_field_doppler')
        self.lmin, self.lmax = float(g('lidar_min_range')), float(g('lidar_max_range'))
        self.bgl_n, self.bg_voxel = int(g('bg_frames_lidar')), float(g('bg_voxel'))
        self.ceps, self.cmin = float(g('cluster_eps')), int(g('min_cluster_size'))
        self.grow_r, self.grow_eps = float(g('grow_radius')), float(g('grow_eps'))
        self.ptol, self.piters = float(g('plane_tol')), int(g('plane_iters'))
        self.pmin, self.perp = int(g('min_plane_pts')), float(g('perp_tol_deg'))
        self.rmin, self.rmax = float(g('radar_min_range')), float(g('radar_max_range'))
        self.bgr_n, self.bg_dist = int(g('bg_frames_radar')), float(g('bg_match_dist'))
        self.rmargin = float(g('range_gate_margin_m'))
        self.max_dop = float(g('max_abs_doppler'))
        self.rceps, self.rcmin = float(g('radar_cluster_eps')), int(g('radar_min_cluster_size'))
        self.acc_n, self.min_frames = int(g('radar_accum_frames')), int(g('radar_min_frames'))
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
        self.t_psig = float(g('prior_t_sigma_m'))
        self.r_psig = np.radians(float(g('prior_rot_sigma_deg')))
        self.cap_n, self.cap_to = int(g('capture_frames')), float(g('capture_timeout_s'))
        self.lstd_max, self.rstd_max = float(g('lidar_std_mm')), float(g('radar_std_m'))
        self.min_points, self.min_base = int(g('min_points')), float(g('min_baseline'))
        self.meas_base = float(g('measured_baseline_m'))
        self.child_frame, self.radar_name = g('child_frame'), g('radar_name')
        self.lidar_name = g('lidar_name')
        self.lidar_topic, self.radar_topic = g('lidar_topic'), g('radar_topic')
        self.out_path = g('output_path') or f'extrinsic_{self.lidar_name}__{self.radar_name}'
        self.publish_tf = bool(g('publish_tf'))
        self.status_xyz = list(g('status_marker_xyz'))
        self.camera_frame = g('camera_frame')

        # ── camera transform: store as T_cam_lidar (p_cam = R_cl·p_lidar + t_cl) ──
        Rg = Rot.from_quat(list(g('lidar_camera_quat_xyzw'))).as_matrix()
        tg = np.array(g('lidar_camera_xyz'), float)
        if str(g('camera_transform_is')) == 'lidar_camera':      # given maps cam -> lidar
            self.R_cl, self.t_cl = Rg.T, -Rg.T @ tg
        else:                                                    # given already maps lidar -> cam
            self.R_cl, self.t_cl = Rg, tg
        ax = {'lidar +X': self.R_cl @ [1, 0, 0], 'lidar +Y': self.R_cl @ [0, 1, 0],
              'lidar +Z': self.R_cl @ [0, 0, 1]}
        self.get_logger().info(
            'T_cam_lidar (for the composed output / overlay only): t=['
            + ' '.join(f'{v:+.4f}' for v in self.t_cl) + '] m  |  '
            + '  '.join(f'{k}->[{v[0]:+.2f} {v[1]:+.2f} {v[2]:+.2f}]' for k, v in ax.items()))

        self.rng = np.random.default_rng(0)
        self.lidar_frame = None
        self.bg_lidar = None; self.bgl_accum = []; self.bgl_want = 0
        self.bg_radar = None; self.bgr_accum = []; self.bgr_want = 0
        self.det = None                  # latest lidar detection
        self.det_t = 0.0
        self.sel = None                  # latest radar selection
        self.aim = ('starting up — no sensor data yet', 'red')
        self.lidar_stat = 'lidar: waiting'
        self.lidar_msg_t = 0.0
        self.radar_msg_t = 0.0
        self.acc = deque(maxlen=int(g('radar_accum_frames')))   # rolling radar frames
        self.frame_n = 0
        self.cap_deadline = 0.0; self.cap_lidar = []; self.cap_radar = []
        self.captures = []
        self.solution = None
        self.tfb = StaticTransformBroadcaster(self) if self.publish_tf else None

        qs = qos_profile_sensor_data
        self.create_subscription(PointCloud2, g('lidar_topic'), self._lidar, qs)
        self.create_subscription(PointCloud2, g('radar_topic'), self._radar, qs)
        self.create_subscription(Empty, '~/background', lambda _: self._bg_start(), 1)
        self.create_subscription(Empty, '~/capture', lambda _: self._arm(), 1)
        self.create_subscription(Empty, '~/solve', lambda _: self._solve(force=True), 1)
        self.create_subscription(Empty, '~/reset', lambda _: self._reset(), 1)
        self.create_subscription(Empty, '~/save', lambda _: self._save(), 1)
        self.pub_apex = self.create_publisher(PointStamped, '~/apex', 5)
        self.pub_mk = self.create_publisher(MarkerArray, '~/markers', 2)
        self.create_timer(0.1, self._markers)
        self.create_timer(1.0, self._heartbeat)

        # optional image overlay (verification only)
        self.K = self.D = self.img = None
        self.bridge = CvBridge() if _HAVE_CV else None
        self.overlay_on = bool(g('show_image_overlay')) and _HAVE_CV
        self.show_window = bool(g('show_window'))
        self.dscale = float(g('debug_scale'))
        if self.overlay_on:
            self.create_subscription(Image, g('image_topic'),
                                     lambda m: setattr(self, 'img',
                                                       self.bridge.imgmsg_to_cv2(m, 'bgr8')), qs)
            self.create_subscription(CameraInfo, g('info_topic'), self._info, qs)
            self.pub_img = self.create_publisher(Image, '~/debug_image', 2)
            self.create_timer(0.05, self._overlay)
        self.get_logger().info(
            'radar_lidar_calib ready — solves T_lidar_radar; camera used only for '
            'the composed T_cam_radar'
            + (' + image overlay.\n' if self.overlay_on else ' (overlay off).\n')
            + '  RViz: Fixed Frame = your lidar frame, add the cloud + MarkerArray '
            'on ~/markers\n'
            '  per placement: reflector OFF -> ~/background | reflector ON, aim, '
            'step out -> ~/capture')

    # ── control ──
    def _bg_start(self):
        self.bgl_accum, self.bgl_want, self.bg_lidar = [], self.bgl_n, None
        self.bgr_accum, self.bgr_want, self.bg_radar = [], self.bgr_n, None
        self.acc.clear()
        self.get_logger().info(
            f'pooling background: lidar {self.bgl_n} + radar {self.bgr_n} frames — '
            f'reflector OFF the tripod, stay out of view')

    def _arm(self):
        if self.bg_lidar is None or self.bg_radar is None:
            self.get_logger().warn('capture refused: background not pooled — ~/background first')
            return
        if self.det is None or time.time() - self.det_t > 1.0:
            self.get_logger().warn('capture refused: no live lidar detection '
                                   '(reflector mounted? in range? background stale?)')
            return
        self.cap_lidar, self.cap_radar = [], []
        self.cap_deadline = time.time() + self.cap_to
        self.get_logger().info(f'capture armed: pairing next {self.cap_n} radar frames '
                               f'(timeout {self.cap_to:.0f} s)')

    def _reset(self):
        self.captures, self.solution = [], None
        self.get_logger().info('captures cleared')

    def _current_T(self):
        if self.solution is not None:
            return self.solution['R'], self.solution['t']
        if self.use_prior:
            return self.R_prior, self.t_prior
        return None, None

    def _heartbeat(self):
        """Name a silent topic rather than letting the status line go stale —
        'waiting for X' is otherwise indistinguishable from 'X never arrived'."""
        now = time.time()
        dead = []
        if now - self.lidar_msg_t > 3.0:
            dead.append(f'LIDAR silent ({self.lidar_topic})')
        if now - self.radar_msg_t > 3.0:
            dead.append(f'RADAR silent ({self.radar_topic})')
        if dead:
            self.aim = (' | '.join(dead) + ' — check the topic name and that it is publishing',
                        'red')
        if self.det is None:
            self.get_logger().info(self.lidar_stat, throttle_duration_sec=2.0)
        self.get_logger().info(self.aim[0], throttle_duration_sec=2.0)

    # ── lidar: background-subtract → cluster → apex ──
    def _lidar(self, msg):
        self.lidar_msg_t = time.time()
        self.lidar_frame = msg.header.frame_id
        xyz = cloud_xyz(msg)
        if len(xyz) == 0:
            return
        r = np.linalg.norm(xyz, axis=1)
        xyz = xyz[np.isfinite(r) & (r > self.lmin) & (r < self.lmax)]

        if self.bgl_want > 0:
            self.bgl_accum.append(voxel_keys(xyz, self.bg_voxel))
            self.bgl_want -= 1
            if self.bgl_want == 0:
                self.bg_lidar = np.unique(np.concatenate(self.bgl_accum))
                self.bgl_accum = []
                self.get_logger().info(f'lidar background ready: {len(self.bg_lidar)} voxels')
            return
        if self.bg_lidar is None:
            self.det = None
            self.lidar_stat = f'lidar: {len(xyz)} pts in range — no background yet'
            return

        fg = xyz[foreground_mask(xyz, self.bg_lidar, self.bg_voxel)]
        clusters = lidar_cluster(fg, self.ceps, self.cmin)
        if not clusters:
            # the counts say WHICH stage lost it: no foreground at all means the
            # background is eating the target (reflector too close to something
            # memorised, e.g. mounted straight on the tripod head — lower
            # bg_voxel or raise it off the head); foreground but no cluster means
            # min_cluster_size is too high or cluster_eps too tight.
            self.det = None
            self.lidar_stat = (f'lidar: {len(xyz)} in range -> {len(fg)} new -> '
                               + ('NO new points (background is eating it — lower '
                                  'bg_voxel / raise the reflector off the tripod)'
                                  if len(fg) == 0 else
                                  f'no cluster >= {self.cmin} pts (lower min_cluster_size '
                                  f'or raise cluster_eps)'))
            return
        P = clusters[0]
        n_seed = len(P)
        if self.grow_r > 0:                 # seed -> whole reflector
            P = grow_from_seed(xyz, P, self.grow_eps, self.grow_r)
        apex, method = locate_apex(P, self.ptol, self.piters, self.pmin, self.perp, self.rng)
        self.det = dict(apex=apex, cluster=P, method=method,
                        n_fg=len(fg), n_extra=len(clusters) - 1)
        self.det_t = time.time()
        self.lidar_stat = (f'lidar: {len(fg)} new -> seed {n_seed} -> grown {len(P)}, {method}'
                           + (f', +{len(clusters)-1} EXTRA' if len(clusters) > 1 else ''))
        if self.cap_deadline > time.time():
            self.cap_lidar.append(apex.copy())
        ps = PointStamped()
        ps.header = msg.header
        ps.point.x, ps.point.y, ps.point.z = map(float, apex)
        self.pub_apex.publish(ps)

    # ── radar: background-subtract → range gate → cluster → SNR centroid ──
    def _radar(self, msg):
        self.radar_msg_t = time.time()
        f = cloud_fields(msg, [self.fx, self.fy, self.fz, self.fsnr, self.fdop])
        if f[self.fx] is None:
            have = ', '.join(fl.name for fl in msg.fields)
            self.aim = (f'radar: no field "{self.fx}" — cloud has: {have}', 'red')
            return
        z = f[self.fz] if f[self.fz] is not None else np.zeros_like(f[self.fx])
        xyz = np.stack([f[self.fx], f[self.fy], z], 1)
        snr = f[self.fsnr] if f[self.fsnr] is not None else np.ones(len(xyz))
        dop = f[self.fdop]
        if self.rscale != 1.0 or self.rbias != 0.0:
            rr = np.linalg.norm(xyz, axis=1)
            ok = rr > 1e-6
            xyz[ok] *= ((self.rscale * rr[ok] + self.rbias) / rr[ok])[:, None]
        r = np.linalg.norm(xyz, axis=1)
        keep = np.isfinite(r) & (r > self.rmin) & (r < self.rmax)

        if self.bgr_want > 0:
            self.bgr_accum.append(xyz[keep])
            self.bgr_want -= 1
            if self.bgr_want == 0:
                self.bg_radar = (np.concatenate(self.bgr_accum) if self.bgr_accum
                                 else np.zeros((0, 3)))
                self.bgr_accum = []
                self.get_logger().info(f'radar background ready: {len(self.bg_radar)} points')
            return
        if self.bg_radar is None:
            self.aim = ('NO BACKGROUND — reflector OFF the tripod, then ~/background', 'red')
            return
        if len(self.bg_radar) and keep.any():
            idx = np.where(keep)[0]
            d = np.linalg.norm(xyz[idx][:, None, :] - self.bg_radar[None, :, :], axis=2).min(1)
            keep[idx[d <= self.bg_dist]] = False

        self.sel = None
        if self.det is None or time.time() - self.det_t > 1.0:
            self.aim = ('no lidar detection — mount the reflector / re-do background', 'red')
            return
        apex = self.det['apex']
        R, t = self._current_T()
        # Rotation-invariant range gate: |p_radar| = |apex − t_radar| for ANY R, so
        # only the radar's POSITION matters here. Use the solved t once available,
        # otherwise the guess — never 0, or a metre of baseline rejects every real
        # return (the symptom is 'no return near lidar range' while a strong,
        # correct return sits one baseline away).
        t_gate = t if t is not None else self.t_prior
        r_exp = np.linalg.norm(apex - t_gate)
        if self.rmargin > 0:
            keep &= np.abs(r - r_exp) <= self.rmargin
        if self.max_dop > 0 and dop is not None:
            keep &= np.abs(dop) <= self.max_dop
        # accumulate this frame, then work on the pooled cloud
        self.acc.append((xyz[keep], snr[keep], self.frame_n))
        self.frame_n += 1
        pts = np.concatenate([a[0] for a in self.acc]) if self.acc else np.zeros((0, 3))
        sr = np.concatenate([a[1] for a in self.acc]) if self.acc else np.zeros(0)
        fid = (np.concatenate([np.full(len(a[0]), a[2]) for a in self.acc])
               if self.acc else np.zeros(0, int))
        n_frames = len(self.acc)
        if len(pts) == 0:
            self.aim = ('radar: nothing new after background (%d frames pooled) — '
                        'RE-AIM / re-do background' % n_frames, 'red')
            return

        pred = R.T @ (apex - t) if R is not None else None
        if pred is not None:
            near = np.linalg.norm(pts - pred, axis=1) <= self.gate_r
            if near.any():
                pts, sr, fid = pts[near], sr[near], fid[near]
            elif self.strict and self.solution is not None:
                self.aim = (f'radar: nothing within {self.gate_r:.2f} m of prediction', 'orange')
                return
        clusters = radar_cluster(pts, self.rceps, self.rcmin)
        # persistence: how many DISTINCT frames contributed to each cluster. The
        # reflector scores ~n_frames; a one-off multipath spike scores 1.
        persist = [len(np.unique(fid[c])) for c in clusters]
        good = [i for i, k in enumerate(persist) if k >= min(self.min_frames, n_frames)]
        if not good:
            best = max(persist) if persist else 0
            self.aim = (f'radar: no persistent cluster (best {best}/{n_frames} frames, '
                        f'need {self.min_frames}) — flickering / re-aim', 'orange')
            return
        if pred is not None:
            ci = good[int(np.argmin([np.linalg.norm(pts[clusters[i]].mean(0) - pred)
                                     for i in good]))]
        else:
            ci = good[int(np.argmax([sr[clusters[i]].max() for i in good]))]
        c = clusters[ci]
        w = sr[c] / max(sr[c].sum(), 1e-9)
        p_sel = (pts[c] * w[:, None]).sum(0)
        snr_sel = float(sr[c].max())
        n_seen = persist[ci]
        r_sel = float(np.linalg.norm(p_sel))
        snr_norm = snr_sel * (r_sel / self.snr_r0) ** 4
        ok = snr_norm >= self.min_snr
        self.sel = dict(p=p_sel, snr=snr_sel, snr_norm=snr_norm, r=r_sel, n=len(c),
                        seen=n_seen, frames=n_frames)
        # Range agreement is reported, not enforced, until a solve exists: before
        # then the baseline is unknown, so a mismatch is uninformative. After the
        # solve the 3-D prediction gate above is already doing the real work.
        gap = abs(r_sel - r_exp)
        self.aim = (f'radar: best {snr_sel:.0f} (norm {snr_norm:.0f}) @ {r_sel:.2f} m'
                    f' [{n_seen}/{n_frames} frames]'
                    + (f' (lidar {r_exp:.2f}, d {gap*100:.0f} cm)' if self.solution else '')
                    + '  ' + ('OK' if ok else 'RE-AIM'), 'green' if ok else 'orange')

        if self.cap_deadline > time.time() and ok:
            self.cap_radar.append(p_sel.copy())
            if len(self.cap_radar) >= self.cap_n:
                self.cap_deadline = 0.0
                self._finish_capture()
        elif self.cap_deadline and time.time() > self.cap_deadline:
            self.cap_deadline = 0.0
            self.get_logger().warn(
                f'capture REFUSED: timeout — only {len(self.cap_radar)}/{self.cap_n} passing '
                f'radar frames in {self.cap_to:.0f} s   (last: {self.aim[0]})')

    # ── atomic capture: both sensors must pass ──
    def _finish_capture(self):
        if len(self.cap_lidar) < 3:
            self.get_logger().warn('capture REFUSED: too few lidar detections in the window')
            return
        L, Rr = np.stack(self.cap_lidar), np.stack(self.cap_radar)
        lstd = float(np.linalg.norm(L.std(0)) * 1000)
        rstd = float(np.linalg.norm(Rr.std(0)))
        if lstd > self.lstd_max:
            self.get_logger().warn(f'capture REFUSED: lidar apex std {lstd:.1f} mm > '
                                   f'{self.lstd_max:.0f} (something still moving)')
            return
        if rstd > self.rstd_max:
            self.get_logger().warn(f'capture REFUSED: radar point std {rstd*100:.0f} cm > '
                                   f'{self.rstd_max*100:.0f} (multipath flicker — nudge or re-aim)')
            return
        p_lidar, p_radar = L.mean(0), Rr.mean(0)
        for i, cp in enumerate(self.captures):
            if np.linalg.norm(np.array(cp['p_lidar']) - p_lidar) < self.min_base:
                self.get_logger().warn(f'note: {np.linalg.norm(np.array(cp["p_lidar"])-p_lidar)*100:.0f}'
                                       f' cm from capture #{i+1} — move the tripod further')
                break
        self.captures.append(dict(
            idx=len(self.captures) + 1, stamp=time.time(),
            p_lidar=[round(float(v), 4) for v in p_lidar],
            p_radar=[round(float(v), 4) for v in p_radar],
            method=self.det['method'], snr=round(float(self.sel['snr']), 1),
            radar_frames_seen=int(self.sel['seen']), radar_frames_pooled=int(self.sel['frames']),
            lidar_std_mm=round(lstd, 1), radar_std_mm=round(rstd * 1000, 1),
            # solve_from_poses_* compatibility: identity pose, apex offset zero
            board_R_quat_xyzw=[0.0, 0.0, 0.0, 1.0],
            board_t=[round(float(v), 4) for v in p_lidar]))
        self.get_logger().info(
            f'*** CAPTURED #{len(self.captures)}  lidar [{p_lidar[0]:.3f} {p_lidar[1]:.3f} '
            f'{p_lidar[2]:.3f}]  radar [{p_radar[0]:.3f} {p_radar[1]:.3f} {p_radar[2]:.3f}]  '
            f'{self.det["method"]}  snr {self.sel["snr"]:.0f} ***')
        if len(self.captures) >= self.min_points:
            self._solve()
        self._save(quiet=True)

    # ── solve: measurement-space ML, offset pinned at zero ──
    def _solve(self, force=False):
        n = len(self.captures)
        if n < (4 if force else self.min_points):
            self.get_logger().info(f'{n} captures — first solve at {self.min_points} '
                                   f'(~/solve to force)')
            return
        P = np.array([c['p_radar'] for c in self.captures])
        Q = np.array([c['p_lidar'] for c in self.captures])
        I = np.repeat(np.eye(3)[None], n, axis=0)
        res = robust_ml_calibrate(
            P, I, Q, np.zeros(3), self.sig_r, self.sig_az, self.sig_el,
            use_elevation=True, solve_offset=False,
            R_prior=self.R_prior if self.use_prior else None,
            t_prior=self.t_prior if self.use_prior else None,
            rot_prior_sigma=self.r_psig if self.use_prior else None,
            t_prior_sigma=self.t_psig if self.use_prior else None,
            huber=self.huber, reject_sigma=self.rej, reject_axis_sigma=self.rej_axis)
        self.solution = res
        R, t, mask = res['R'], res['t'], res['inlier_mask']
        Pin, Qin = P[mask], Q[mask]
        sig = np.sqrt(np.clip(np.diag(res['cov']), 0, None))
        rot1s, t1s = np.degrees(sig[:3]), sig[3:] * 1000
        err = ((R @ Pin.T).T + t) - Qin
        bias, rms = err.mean(0) * 1000, np.sqrt((err ** 2).mean(0)) * 1000
        loo = loo_cross_val(Pin, I[mask], Qin, np.zeros(3),
                            (self.sig_r, self.sig_az, self.sig_el), True)
        cond = condition_number(Pin)
        raz = np.array([cart_to_raz(p) for p in Pin])
        q = Rot.from_matrix(R).as_quat()
        lid_r, rad_r = np.linalg.norm(Qin - t, axis=1), np.linalg.norm(Pin, axis=1)
        a, b = np.linalg.lstsq(np.vstack([rad_r, np.ones_like(rad_r)]).T, lid_r, rcond=None)[0]
        axes = {k: R @ v for k, v in (('X fwd', [1, 0, 0]), ('Y left', [0, 1, 0]),
                                      ('Z up', [0, 0, 1]))}
        L = [f'=== T_{self.lidar_frame or "lidar"}_{self.child_frame}  (lidar <- radar) ===',
             f'  captures {n}   inliers {res["n_in"]}/{n}   residual {res["rms_sigma"]:.2f} s'
             f'   cond {cond:.1f}',
             f'  xyz (m) : {t[0]:+.4f} {t[1]:+.4f} {t[2]:+.4f}   |t| {np.linalg.norm(t)*100:.1f} cm',
             f'  quat    : {q[0]:+.4f} {q[1]:+.4f} {q[2]:+.4f} {q[3]:+.4f}',
             f'  1s rot  : {rot1s[0]:.2f} {rot1s[1]:.2f} {rot1s[2]:.2f} deg'
             f'   1s t: {t1s[0]:.1f} {t1s[1]:.1f} {t1s[2]:.1f} mm',
             f'  spread  : range {raz[:,0].ptp()*100:.0f} cm  az {np.degrees(raz[:,1].ptp()):.0f} deg'
             f'  el {np.degrees(raz[:,2].ptp()):.0f} deg',
             f'  bias mm : {bias[0]:+.0f} {bias[1]:+.0f} {bias[2]:+.0f}'
             f'   3-D RMS mm: {rms[0]:.0f} {rms[1]:.0f} {rms[2]:.0f}',
             '  radar axes in lidar frame: '
             + '  '.join(f'{k}->[{v[0]:+.2f} {v[1]:+.2f} {v[2]:+.2f}]' for k, v in axes.items())]
        if loo:
            L.append(f'  LOO CV  : {loo[0]:.2f} s (max {loo[1]:.2f})')
        if abs(a - 1) > 0.02 or abs(b) > 0.05:
            L.append(f'  range fit: lidar_r = {a:.3f}*radar_r {b:+.3f} m (want a~1) -> '
                     f'set radar_range_scale={a*self.rscale:.4f}')
        if self.meas_base > 0:
            d = abs(np.linalg.norm(t) - self.meas_base)
            L.append(f'  baseline: |t| {np.linalg.norm(t)*100:.1f} vs tape '
                     f'{self.meas_base*100:.1f} cm -> {d*100:.1f} cm '
                     f'[{"OK" if d <= 0.05 else "MISMATCH"}]')
        gates = [('residual~1s', res['rms_sigma'] <= 1.5), ('cond<=5', cond <= 5),
                 ('rot1s<=4deg', rot1s.max() <= 4), ('bias<=50mm', np.abs(bias).max() <= 50)]
        L.append('  GATES   : ' + '  '.join(f'{k}[{"P" if v else "F"}]' for k, v in gates)
                 + '   + RViz up/down check before ~/save')
        Rcr, tcr = self._compose_cam_radar(R, t)          # T_cam_radar, for deployment
        qc = Rot.from_matrix(Rcr).as_quat()
        L += [f'  --- composed T_cam_radar = T_cam_lidar * T_lidar_radar ---',
              f'  xyz (m) : {tcr[0]:+.4f} {tcr[1]:+.4f} {tcr[2]:+.4f}'
              f'   |t| {np.linalg.norm(tcr)*100:.1f} cm',
              f'  quat    : {qc[0]:+.4f} {qc[1]:+.4f} {qc[2]:+.4f} {qc[3]:+.4f}',
              '  radar axes in CAMERA frame: '
              + '  '.join(f'{k}->[{v[0]:+.2f} {v[1]:+.2f} {v[2]:+.2f}]'
                          for k, v in (('X fwd', Rcr @ [1, 0, 0]), ('Y left', Rcr @ [0, 1, 0]),
                                       ('Z up', Rcr @ [0, 0, 1])))]
        self.get_logger().info('\n' + '\n'.join(L))
        if self.tfb is not None and self.lidar_frame:
            tf = TransformStamped()
            tf.header.stamp = self.get_clock().now().to_msg()
            tf.header.frame_id = self.lidar_frame
            tf.child_frame_id = self.child_frame
            (tf.transform.translation.x, tf.transform.translation.y,
             tf.transform.translation.z) = map(float, t)
            (tf.transform.rotation.x, tf.transform.rotation.y,
             tf.transform.rotation.z, tf.transform.rotation.w) = map(float, q)
            self.tfb.sendTransform(tf)

    # ── save ──
    def _save(self, quiet=False):
        g = lambda k: self.get_parameter(k).value
        out = dict(kind='radar_lidar_session', stamp=time.time(),
                   parent_frame=self.lidar_frame or 'lidar',
                   child_frame=self.child_frame,
                   lidar_name=self.lidar_name, radar_name=self.radar_name,
                   note='T_lidar_radar. Compose later: T_cam_radar = T_cam_lidar * T_lidar_radar',
                   params=dict(
                       sigma_range_m=self.sig_r, sigma_az_deg=float(np.degrees(self.sig_az)),
                       sigma_el_deg=float(np.degrees(self.sig_el)),
                       reflector_offset_x=0.0, reflector_offset_y=0.0, reflector_offset_z=0.0,
                       use_extrinsic_prior=self.use_prior,
                       prior_t_xyz=[float(v) for v in self.t_prior],
                       prior_rpy_deg=list(g('prior_rpy_deg')),
                       prior_t_sigma_m=self.t_psig,
                       prior_rot_sigma_deg=float(np.degrees(self.r_psig)),
                       radar_range_scale=self.rscale, radar_range_bias_m=self.rbias,
                       reject_sigma=self.rej, reject_axis_sigma=self.rej_axis,
                       min_snr=self.min_snr, gate_radius=self.gate_r),
                   captures=self.captures)
        if self.solution is not None:
            R, t = self.solution['R'], self.solution['t']
            q = Rot.from_matrix(R).as_quat()
            Rcr, tcr = self._compose_cam_radar(R, t)
            qc = Rot.from_matrix(Rcr).as_quat()
            out['result'] = dict(
                T_lidar_radar_translation=[float(v) for v in t],
                T_lidar_radar_quaternion_xyzw=[float(v) for v in q],
                n_inliers=int(self.solution['n_in']),
                residual_rms_sigma=float(self.solution['rms_sigma']),
                static_tf_cmd=('ros2 run tf2_ros static_transform_publisher '
                               + ' '.join(f'{v:.6f}' for v in t) + ' '
                               + ' '.join(f'{v:.6f}' for v in q) + ' '
                               + f'{self.lidar_frame or "lidar"} {self.child_frame}'),
                # composed with the GLIM lidar<->camera transform; this is what
                # radar_fusion_reflector.py consumes (r1_t_xyz / r1_quat_xyzw)
                T_cam_radar_translation=[float(v) for v in tcr],
                T_cam_radar_quaternion_xyzw=[float(v) for v in qc],
                T_cam_lidar_translation=[float(v) for v in self.t_cl],
                T_cam_lidar_quaternion_xyzw=[float(v) for v in
                                             Rot.from_matrix(self.R_cl).as_quat()],
                static_tf_cmd_cam=('ros2 run tf2_ros static_transform_publisher '
                                   + ' '.join(f'{v:.6f}' for v in tcr) + ' '
                                   + ' '.join(f'{v:.6f}' for v in qc) + ' '
                                   + f'{self.camera_frame} {self.child_frame}'))
        path = self.out_path + '_session.json'
        with open(path, 'w') as f:
            json.dump(out, f, indent=1)
        if not quiet:
            self.get_logger().info(f'saved {len(self.captures)} captures -> '
                                   f'{os.path.abspath(path)}')

    # ── camera composition + image overlay (verification / deployment only) ──
    def _compose_cam_radar(self, R_lr, t_lr):
        """T_cam_radar = T_cam_lidar · T_lidar_radar. The solve itself never
        touches the camera, so a wrong GLIM transform shows up here and in the
        overlay but leaves the radar↔lidar result intact — and a re-run of the
        lidar↔camera calibration can be recomposed without recollecting radar."""
        return self.R_cl @ R_lr, self.R_cl @ t_lr + self.t_cl

    def _info(self, m):
        if self.K is None:
            self.K = np.array(m.k).reshape(3, 3)
            self.D = np.array(m.d) if len(m.d) else np.zeros(5)

    def _proj(self, pts_lidar):
        """lidar-frame points → pixels, via T_cam_lidar and the ZED intrinsics."""
        P = np.atleast_2d(np.asarray(pts_lidar, float))
        Pc = (self.R_cl @ P.T).T + self.t_cl
        uv = np.full((len(Pc), 2), np.nan)
        ok = Pc[:, 2] > 0.05
        if ok.any():
            p, _ = cv2.projectPoints(Pc[ok].reshape(-1, 1, 3), np.zeros(3), np.zeros(3),
                                     self.K, self.D)
            uv[ok] = p.reshape(-1, 2)
        return uv

    def _overlay(self):
        if self.img is None or self.K is None:
            return
        im = self.img.copy()

        def txt(p, s, col, sc=.55):
            cv2.putText(im, s, p, cv2.FONT_HERSHEY_SIMPLEX, sc, (0, 0, 0), 3)
            cv2.putText(im, s, p, cv2.FONT_HERSHEY_SIMPLEX, sc, col, 1)

        # background is never drawn — only the live foreground + apex
        if self.det is not None and time.time() - self.det_t < 1.0:
            d = self.det
            for (u, v) in self._proj(d['cluster']):
                if np.isfinite(u):
                    cv2.circle(im, (int(u), int(v)), 2, (255, 220, 40), -1)
            au, av = self._proj(d['apex'])[0]
            if np.isfinite(au):
                au, av = int(au), int(av)
                cv2.drawMarker(im, (au, av), (0, 255, 0), cv2.MARKER_CROSS, 26, 2)
                txt((au + 12, av - 10), f'{np.linalg.norm(d["apex"]):.2f} m {d["method"]}',
                    (0, 255, 0))
        for c in self.captures:                      # pinned coverage map
            u, v = self._proj(np.array(c['p_lidar']))[0]
            if np.isfinite(u):
                cv2.circle(im, (int(u), int(v)), 6, (255, 200, 0), 2)
                txt((int(u) + 7, int(v) + 5), str(c['idx']), (255, 200, 0), .45)

        R, t = self._current_T()                     # radar pick through the solve
        if self.sel is not None and R is not None:
            p_l = R @ self.sel['p'] + t
            uv = self._proj(p_l)[0]
            if np.isfinite(uv[0]):
                u, v = int(uv[0]), int(uv[1])
                cv2.circle(im, (u, v), 7, (255, 0, 255), 2)
                if self.det is not None and time.time() - self.det_t < 1.0:
                    a = self._proj(self.det['apex'])[0]
                    if np.isfinite(a[0]):
                        cv2.line(im, (u, v), (int(a[0]), int(a[1])), (255, 0, 255), 1)
                        txt((u + 10, v + 16),
                            f'D {np.linalg.norm(p_l - self.det["apex"])*1000:.0f} mm', (255, 0, 255), .5)

        col = {'green': (0, 220, 0), 'orange': (0, 165, 255), 'red': (0, 0, 255)}[self.aim[1]]
        h = im.shape[0]
        txt((10, h - 56), self.lidar_stat,
            (200, 255, 200) if self.det is not None else (0, 165, 255))
        txt((10, h - 34), self.aim[0], col)
        state = ('NO BACKGROUND - ~/background first' if self.bg_lidar is None
                 else f'captures {len(self.captures)}'
                      + (f' | residual {self.solution["rms_sigma"]:.2f}s inl {self.solution["n_in"]}'
                         if self.solution else f'/{self.min_points} to first solve'))
        txt((10, h - 12), state, (240, 240, 240))
        if self.dscale != 1.0:
            im = cv2.resize(im, None, fx=self.dscale, fy=self.dscale)
        self.pub_img.publish(self.bridge.cv2_to_imgmsg(im, 'bgr8'))
        if self.show_window:
            cv2.imshow('radar_lidar_calib', im)
            cv2.waitKey(1)

    # ── RViz markers (the verification layer) ──
    def _mk(self, ns, mid, typ, scale, color):
        m = Marker()
        m.header.frame_id = self.lidar_frame or 'lidar'
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns, m.id, m.type, m.action = ns, mid, typ, Marker.ADD
        m.pose.orientation.w = 1.0
        m.scale.x = m.scale.y = m.scale.z = scale
        m.color = color
        return m

    def _markers(self):
        if self.lidar_frame is None:
            return
        arr = MarkerArray()

        if self.det is not None and time.time() - self.det_t < 1.0:
            d = self.det
            pc = self._mk('cluster', 0, Marker.POINTS, 0.02, CYAN)
            pc.points = [Point(x=float(p[0]), y=float(p[1]), z=float(p[2])) for p in d['cluster']]
            arr.markers.append(pc)
            ap = self._mk('apex', 1, Marker.SPHERE, 0.07, GREEN)
            ap.pose.position.x, ap.pose.position.y, ap.pose.position.z = map(float, d['apex'])
            arr.markers.append(ap)
            lab = self._mk('apex_label', 2, Marker.TEXT_VIEW_FACING, 0.09, GREEN)
            lab.pose.position.x, lab.pose.position.y = float(d['apex'][0]), float(d['apex'][1])
            lab.pose.position.z = float(d['apex'][2]) + 0.15
            lab.text = (f'{np.linalg.norm(d["apex"]):.2f} m  {d["method"]}'
                        + (f'  +{d["n_extra"]} EXTRA CLUSTER' if d['n_extra'] else ''))
            arr.markers.append(lab)

        if self.captures:
            cap = self._mk('captures', 3, Marker.SPHERE_LIST, 0.06, AMBER)
            cap.points = [Point(x=float(c['p_lidar'][0]), y=float(c['p_lidar'][1]),
                                z=float(c['p_lidar'][2])) for c in self.captures]
            arr.markers.append(cap)
            for c in self.captures:
                tm = self._mk('capture_ids', 100 + c['idx'], Marker.TEXT_VIEW_FACING, 0.07, AMBER)
                tm.pose.position.x, tm.pose.position.y = float(c['p_lidar'][0]), float(c['p_lidar'][1])
                tm.pose.position.z = float(c['p_lidar'][2]) - 0.12
                tm.text = str(c['idx'])
                arr.markers.append(tm)

        # the radar's pick, mapped into the lidar frame by the current solve
        R, t = self._current_T()
        if self.sel is not None and R is not None:
            p = R @ self.sel['p'] + t
            rm = self._mk('radar_pick', 4, Marker.SPHERE, 0.07, MAGENTA)
            rm.pose.position.x, rm.pose.position.y, rm.pose.position.z = map(float, p)
            arr.markers.append(rm)
            if self.det is not None and time.time() - self.det_t < 1.0:
                ln = self._mk('delta', 5, Marker.LINE_LIST, 0.012, MAGENTA)
                ln.points = [Point(x=float(p[0]), y=float(p[1]), z=float(p[2])),
                             Point(x=float(self.det['apex'][0]), y=float(self.det['apex'][1]),
                                   z=float(self.det['apex'][2]))]
                arr.markers.append(ln)
                dt = self._mk('delta_label', 6, Marker.TEXT_VIEW_FACING, 0.08, MAGENTA)
                mid = (p + self.det['apex']) / 2
                dt.pose.position.x, dt.pose.position.y, dt.pose.position.z = map(float, mid)
                dt.text = (f'D {np.linalg.norm(p - self.det["apex"])*1000:.0f} mm '
                           f'({"solved" if self.solution else "prior"})')
                arr.markers.append(dt)

        st = self._mk('status', 7, Marker.TEXT_VIEW_FACING, 0.12, WHITE)
        st.pose.position.x, st.pose.position.y, st.pose.position.z = map(float, self.status_xyz)
        if self.bg_lidar is None or self.bg_radar is None:
            st.text = 'NO BACKGROUND — reflector OFF, then ~/background'
            st.color = ColorRGBA(r=1.0, g=0.3, b=0.2, a=1.0)
        else:
            head = (f'captures {len(self.captures)}'
                    + (f'  |  residual {self.solution["rms_sigma"]:.2f}s '
                       f'inl {self.solution["n_in"]}' if self.solution else
                       f'/{self.min_points} to first solve'))
            st.text = head + '\n' + self.lidar_stat + '\n' + self.aim[0]
            st.color = {'green': GREEN, 'orange': AMBER,
                        'red': ColorRGBA(r=1.0, g=0.3, b=0.2, a=1.0)}[self.aim[1]]
        if self.cap_deadline > time.time():
            st.text += f'\nCAPTURING {len(self.cap_radar)}/{self.cap_n}'
        arr.markers.append(st)
        self.pub_mk.publish(arr)


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
