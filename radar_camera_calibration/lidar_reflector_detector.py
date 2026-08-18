#!/usr/bin/env python3
"""
LIDAR corner-reflector detector + Stage-A verification overlay
==============================================================

Front end for the lidar↔radar extrinsic calibration. This node does the LIDAR
side only — no radar anywhere in this file. It:

  1. pools a lidar BACKGROUND with the reflector OFF the tripod (~/background),
  2. background-subtracts each live cloud → the foreground IS the reflector,
  3. clusters the foreground and localises the APEX (RANSAC 3-plane
     intersection, with a deepest-point fallback),
  4. maps the apex into the CAMERA frame via your Koide calibration json
     (T_cam_lidar) and projects it onto the ZED image — the Stage-A check:
     the crosshair must sit on the physical reflector corner,
  5. on ~/capture (manual — you trigger it once you are out of the scene) it
     averages the next few detections, quality-gates them, and appends the
     capture to a json record that the radar solve step will consume.

Logically this is three modules kept in ONE file on purpose:
  [A] cloud tools    — PointCloud2 parsing, voxel background set, clustering
  [B] apex locator   — RANSAC planes → intersection; deepest-point fallback
  [C] node           — ROS wiring, Stage-A overlay, capture/save logic

Workflow per tripod placement
-----------------------------
  move tripod (reflector OFF)
    → ros2 topic pub -1 /lidar_reflector/background std_msgs/msg/Empty "{}"
  mount reflector (quick-release), walk out of the scene
    → ros2 topic pub -1 /lidar_reflector/capture   std_msgs/msg/Empty "{}"
  check the overlay: crosshair on the corner, dots only on the reflector.
  repeat. ~/save writes the json (also auto-written after every capture).

Run
---
  ros2 run wicoms_utils lidar_reflector_detector --ros-args \
    -p lidar_topic:=/ouster/points \
    -p image_topic:=/zed/zed_node/left/image_rect_color \
    -p info_topic:=/zed/zed_node/left/camera_info \
    -p calib_json:=/path/to/koide/calib.json \
    -p show_window:=true

Koide calib.json note
---------------------
direct_visual_lidar_calibration stores results.T_lidar_camera as
[tx,ty,tz,qx,qy,qz,qw] mapping CAMERA-frame points INTO the LIDAR frame, so we
INVERT it to get T_cam_lidar (p_cam = T_cam_lidar · p_lidar). 4x4 matrices and
a plain "T_cam_lidar" key are also accepted. If the projection looks like a
uniform smear instead of dots-on-objects, the direction is wrong — flip
`invert_calib`. The Stage-A overlay itself is the test.

Display rule: the BACKGROUND is never reprojected — only the foreground
cluster + apex live, and after each capture the captured apexes stay pinned
(numbered circles) so you can see your pose spread build up on the image.
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
from geometry_msgs.msg import PointStamped
from std_msgs.msg import Empty
from cv_bridge import CvBridge
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as Rot


# ────────────────────────────── [A] cloud tools ──────────────────────────────
_DT = {1: np.int8, 2: np.uint8, 3: np.int16, 4: np.uint16,
       5: np.int32, 6: np.uint32, 7: np.float32, 8: np.float64}


def cloud_to_xyz_i(msg):
    """PointCloud2 → (N,3) float32 xyz + (N,) intensity (or None).

    Parses straight from the buffer (fast enough for a 64x1024 Ouster in
    numpy; the generator API is not). Organized clouds carry zero-points for
    no-return pixels — the caller filters them with the range gate.
    """
    n = msg.width * msg.height
    if n == 0:
        return np.zeros((0, 3), np.float32), None
    step = msg.point_step
    buf = np.frombuffer(bytes(msg.data), dtype=np.uint8, count=n * step).reshape(n, step)
    offs = {f.name: (f.offset, f.datatype) for f in msg.fields}

    def col(name):
        if name not in offs:
            return None
        off, dt = offs[name]
        typ = _DT.get(dt)
        if typ is None:
            return None
        w = np.dtype(typ).itemsize
        return buf[:, off:off + w].copy().view(typ).ravel().astype(np.float32)

    x, y, z = col('x'), col('y'), col('z')
    if x is None or y is None or z is None:
        return np.zeros((0, 3), np.float32), None
    xyz = np.stack([x, y, z], axis=1)
    inten = col('intensity')
    if inten is None:
        inten = col('reflectivity')       # Ouster fallback (uint16)
    if inten is None:
        inten = col('signal')
    return xyz, inten


# Voxel background set. Keys are packed int64s; membership is tested against
# the 27-neighbourhood so a live point within one voxel (~bg_voxel metres) of
# ANY background point counts as background. Vectorised via sorted-searchsorted.
_OFF = 1 << 20        # centre the (possibly negative) voxel indices
_SX, _SY = 1 << 42, 1 << 21


def voxel_keys(xyz, voxel):
    idx = np.floor(xyz / voxel).astype(np.int64) + _OFF
    return idx[:, 0] * _SX + idx[:, 1] * _SY + idx[:, 2]


_NEIGH = np.array([dx * _SX + dy * _SY + dz
                   for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)],
                  dtype=np.int64)


def foreground_mask(xyz, bg_sorted, voxel):
    keys = voxel_keys(xyz, voxel)
    hit = np.zeros(len(keys), bool)
    for d in _NEIGH:
        k = keys + d
        i = np.searchsorted(bg_sorted, k)
        i = np.clip(i, 0, len(bg_sorted) - 1)
        hit |= (bg_sorted[i] == k)
    return ~hit


def cluster_points(P, eps, min_size, cap=20000):
    """Greedy BFS connected components within `eps` (cKDTree). Returns list of
    index arrays, largest first. Foreground should be small (just the
    reflector); decimate defensively if something went wrong upstream."""
    if len(P) == 0:
        return []
    if len(P) > cap:
        sel = np.random.choice(len(P), cap, replace=False)
        P = P[sel]
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


# ────────────────────────────── [B] apex locator ─────────────────────────────
def ransac_planes(P, tol, iters, min_pts, rng):
    """Sequentially RANSAC up to 3 planes. Each plane refined by SVD on its
    inliers; inliers removed before fitting the next. Returns [(n, d, count)]."""
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
        n = np.linalg.svd(Q - c0)[2][2]           # smallest singular vector
        planes.append((n, float(n @ c0), len(Q)))
        pts = pts[~best[1]]
    return planes


def locate_apex(P, plane_tol, plane_iters, min_plane_pts, perp_tol_deg, rng):
    """Apex of a trihedral from its point cluster.

    planes3 : intersect three mutually ~perpendicular RANSAC planes — analytic
              apex, no measured offset anywhere. Preferred.
    deepest : the apex of a reflector aimed at the sensor is its farthest
              point along the viewing ray; mean of the top-8 deepest points.
    Returns (apex_xyz, method) or (None, reason).
    """
    planes = ransac_planes(P, plane_tol, plane_iters, min_plane_pts, rng)
    if len(planes) == 3:
        perp = np.cos(np.radians(90.0 - perp_tol_deg))     # |n_i·n_j| budget
        ok = all(abs(planes[i][0] @ planes[j][0]) < perp
                 for i in range(3) for j in range(i + 1, 3))
        if ok:
            N = np.stack([p[0] for p in planes])
            d = np.array([p[1] for p in planes])
            try:
                apex = np.linalg.solve(N, d)
                # sanity: the apex must live at the cluster, not off in space
                if np.min(np.linalg.norm(P - apex, axis=1)) < 0.10:
                    return apex, 'planes3'
            except np.linalg.LinAlgError:
                pass
    u = P.mean(0)
    u = u / (np.linalg.norm(u) + 1e-9)                     # viewing ray
    proj = P @ u
    k = min(8, len(P))
    apex = P[np.argsort(proj)[-k:]].mean(0)
    return apex, 'deepest'


# ─────────────────────────────── Koide json ──────────────────────────────────
def load_T_cam_lidar(path, invert_override):
    """Accepts koide3 calib.json (results.T_lidar_camera, 7-vec [t,quat]) or a
    plain {"T_cam_lidar": 4x4}. Returns (4x4 T_cam_lidar, description)."""
    d = json.load(open(path))

    def to_T(v):
        v = np.asarray(v, float)
        if v.size == 16:
            return v.reshape(4, 4)
        if v.size == 7:
            T = np.eye(4)
            T[:3, :3] = Rot.from_quat(v[3:7]).as_matrix()
            T[:3, 3] = v[:3]
            return T
        raise ValueError(f'transform in {path} has {v.size} values (want 7 or 16)')

    candidates = [('results', 'T_lidar_camera'), (None, 'T_lidar_camera'),
                  (None, 'T_cam_lidar'), (None, 'T_camera_lidar'),
                  ('results', 'init_T_lidar_camera')]
    for grp, key in candidates:
        src = d.get(grp, {}) if grp else d
        if isinstance(src, dict) and key in src:
            T = to_T(src[key])
            # 'lidar_camera' maps camera→lidar ⇒ invert to get cam←lidar
            inv = 'lidar_camera' in key if invert_override == 'auto' \
                else (invert_override == 'true')
            if inv:
                T = np.linalg.inv(T)
            return T, f'{key}{" (inverted)" if inv else ""}'
    raise ValueError(f'no known transform key in {path}: {list(d.keys())}')


# ─────────────────────────────── [C] the node ────────────────────────────────
class LidarReflectorDetector(Node):
    def __init__(self):
        super().__init__('lidar_reflector')
        dp = self.declare_parameter
        dp('lidar_topic', '/ouster/points')
        dp('image_topic', '/zed/zed_node/left/image_rect_color')
        dp('info_topic', '/zed/zed_node/left/camera_info')
        dp('calib_json', '')                 # Koide output (REQUIRED)
        dp('invert_calib', 'auto')           # 'auto' | 'true' | 'false'
        dp('camera_frame', 'zed_left_camera_optical_frame')

        dp('min_range', 0.3)                 # lidar range prefilter (m)
        dp('max_range', 8.0)
        dp('bg_frames', 10)                  # clouds pooled on ~/background
        dp('bg_voxel', 0.10)                 # voxel size = bg match distance (m)

        dp('cluster_eps', 0.12)              # foreground clustering radius (m)
        dp('min_cluster_size', 15)           # fewer points → detection rejected

        dp('plane_tol', 0.015)               # RANSAC inlier distance (m)
        dp('plane_iters', 250)
        dp('min_plane_pts', 12)              # per plate; sets max usable range
        dp('perp_tol_deg', 25.0)             # plates must be 90°±this apart

        dp('capture_frames', 5)              # detections averaged per ~/capture
        dp('capture_max_std_mm', 15.0)       # reject capture if apex jitters more
        dp('min_baseline', 0.10)             # warn if closer than this to old capture

        dp('output_path', 'lidar_apex_captures.json')
        dp('show_window', True)
        dp('show_cluster', True)             # draw the cluster dots, not just apex
        dp('debug_scale', 1.0)

        g = lambda k: self.get_parameter(k).value
        self.min_range, self.max_range = float(g('min_range')), float(g('max_range'))
        self.bg_frames_n, self.bg_voxel = int(g('bg_frames')), float(g('bg_voxel'))
        self.cluster_eps, self.min_cluster = float(g('cluster_eps')), int(g('min_cluster_size'))
        self.plane_tol, self.plane_iters = float(g('plane_tol')), int(g('plane_iters'))
        self.min_plane_pts, self.perp_tol = int(g('min_plane_pts')), float(g('perp_tol_deg'))
        self.cap_frames, self.cap_std = int(g('capture_frames')), float(g('capture_max_std_mm'))
        self.min_baseline = float(g('min_baseline'))
        self.out_path = g('output_path')
        self.show_window, self.show_cluster = bool(g('show_window')), bool(g('show_cluster'))
        self.dscale = float(g('debug_scale'))
        self.camera_frame = g('camera_frame')

        if not g('calib_json'):
            raise SystemExit('set -p calib_json:=/path/to/koide/calib.json')
        self.T_cam_lidar, desc = load_T_cam_lidar(g('calib_json'), str(g('invert_calib')))
        t = self.T_cam_lidar[:3, 3]
        rpy = Rot.from_matrix(self.T_cam_lidar[:3, :3]).as_euler('xyz', degrees=True)
        self.get_logger().info(
            f'T_cam_lidar from {desc}: t=[{t[0]:+.3f} {t[1]:+.3f} {t[2]:+.3f}] m  '
            f'rpy=[{rpy[0]:+.1f} {rpy[1]:+.1f} {rpy[2]:+.1f}] deg — '
            f'VERIFY with the overlay (dots must land on the reflector)')

        self.rng = np.random.default_rng(0)
        self.bridge = CvBridge()
        self.K = self.D = None
        self.img = None
        self.bg_sorted = None
        self.bg_accum, self.bg_want = [], 0

        # live state
        self.last = None            # dict: apex_l, apex_c, cluster, method, n
        self.recent = deque(maxlen=32)   # (stamp, apex_lidar, method) for capture averaging
        self.cap_armed = 0          # >0 → collecting this many detections
        self.cap_buf = []
        self.captures = []

        qs = qos_profile_sensor_data
        self.create_subscription(PointCloud2, g('lidar_topic'), self._cloud, qs)
        self.create_subscription(Image, g('image_topic'), self._image, qs)
        self.create_subscription(CameraInfo, g('info_topic'), self._info, qs)
        self.create_subscription(Empty, '~/background', lambda _: self._start_bg(), 1)
        self.create_subscription(Empty, '~/capture', lambda _: self._arm(), 1)
        self.create_subscription(Empty, '~/reset', lambda _: self._reset(), 1)
        self.create_subscription(Empty, '~/save', lambda _: self._save(), 1)
        self.pub_apex = self.create_publisher(PointStamped, '~/apex_cam', 5)
        self.pub_dbg = self.create_publisher(Image, '~/debug_image', 2)
        self.create_timer(0.05, self._gui)   # always: ~/debug_image feeds radar_lidar_calib
        self._frame = None
        self.get_logger().info(
            'ready — 1) reflector OFF, tripod in place → ~/background   '
            '2) reflector ON, step out → ~/capture   3) ~/save')

    # ── camera plumbing ──
    def _info(self, m):
        if self.K is None:
            self.K = np.array(m.k).reshape(3, 3)
            self.D = np.array(m.d) if len(m.d) else np.zeros(5)

    def _image(self, m):
        self.img = self.bridge.imgmsg_to_cv2(m, 'bgr8')

    # ── control ──
    def _start_bg(self):
        self.bg_accum, self.bg_want = [], self.bg_frames_n
        self.bg_sorted = None
        self.get_logger().info(f'pooling background over {self.bg_want} clouds '
                               f'(reflector must be OFF, tripod IN place)')

    def _arm(self):
        if self.bg_sorted is None:
            self.get_logger().warn('no background pooled — ~/background first')
            return
        self.cap_armed, self.cap_buf = self.cap_frames, []
        self.get_logger().info(f'capture armed: averaging next {self.cap_frames} detections')

    def _reset(self):
        self.captures = []
        self.get_logger().info('captures cleared')

    # ── main pipeline, one cloud at a time ──
    def _cloud(self, msg):
        xyz, inten = cloud_to_xyz_i(msg)
        if len(xyz) == 0:
            return
        r = np.linalg.norm(xyz, axis=1)
        keep = np.isfinite(r) & (r > self.min_range) & (r < self.max_range)
        xyz = xyz[keep]
        inten = inten[keep] if inten is not None else None

        # background pooling mode
        if self.bg_want > 0:
            self.bg_accum.append(voxel_keys(xyz, self.bg_voxel))
            self.bg_want -= 1
            if self.bg_want == 0:
                self.bg_sorted = np.unique(np.concatenate(self.bg_accum))
                self.bg_accum = []
                self.get_logger().info(f'background ready: {len(self.bg_sorted)} voxels')
            return
        if self.bg_sorted is None:
            self.last = None
            return

        fg = xyz[foreground_mask(xyz, self.bg_sorted, self.bg_voxel)]
        clusters = cluster_points(fg, self.cluster_eps, self.min_cluster)
        if not clusters:
            self.last = None
            return
        P = clusters[0]                       # reflector = only (largest) new object
        apex_l, method = locate_apex(P, self.plane_tol, self.plane_iters,
                                     self.min_plane_pts, self.perp_tol, self.rng)
        apex_c = (self.T_cam_lidar @ np.append(apex_l, 1.0))[:3]
        self.last = dict(apex_l=apex_l, apex_c=apex_c, cluster=P, method=method,
                         n_fg=len(fg), n_extra=len(clusters) - 1)
        self.recent.append((time.time(), apex_l, method))

        ps = PointStamped()
        ps.header.stamp = msg.header.stamp
        ps.header.frame_id = self.camera_frame
        ps.point.x, ps.point.y, ps.point.z = map(float, apex_c)
        self.pub_apex.publish(ps)

        if self.cap_armed > 0:
            self.cap_buf.append(apex_l)
            self.cap_armed -= 1
            if self.cap_armed == 0:
                self._finish_capture(method, len(P))

    def _finish_capture(self, method, n_cluster):
        A = np.stack(self.cap_buf)
        std_mm = float(np.linalg.norm(A.std(0)) * 1000)
        if std_mm > self.cap_std:
            self.get_logger().warn(
                f'capture REJECTED: apex std {std_mm:.1f} mm > {self.cap_std} mm '
                f'(rig or scene still moving — re-trigger)')
            return
        apex_l = A.mean(0)
        apex_c = (self.T_cam_lidar @ np.append(apex_l, 1.0))[:3]
        for i, c in enumerate(self.captures):
            d = np.linalg.norm(np.array(c['p_cam']) - apex_c)
            if d < self.min_baseline:
                self.get_logger().warn(
                    f'note: only {d*100:.0f} cm from capture #{i+1} — move more for diversity')
                break
        self.captures.append(dict(
            idx=len(self.captures) + 1, stamp=time.time(),
            p_lidar=[round(float(v), 4) for v in apex_l],
            p_cam=[round(float(v), 4) for v in apex_c],
            method=method, n_cluster=int(n_cluster), std_mm=round(std_mm, 1)))
        self.get_logger().info(
            f'*** CAPTURED #{len(self.captures)}  lidar [{apex_l[0]:.3f}, {apex_l[1]:.3f}, '
            f'{apex_l[2]:.3f}]  cam [{apex_c[0]:.3f}, {apex_c[1]:.3f}, {apex_c[2]:.3f}]  '
            f'{method}  σ {std_mm:.1f} mm ***')
        self._save()

    def _save(self):
        rec = dict(kind='lidar_apex_captures',
                   stamp=time.time(),
                   camera_frame=self.camera_frame,
                   T_cam_lidar=[[round(float(v), 6) for v in row] for row in self.T_cam_lidar],
                   params={p.name: (p.value if not isinstance(p.value, (bytes,)) else str(p.value))
                           for p in self.get_parameters(
                               [d.name for d in self._parameters.values()])},
                   captures=self.captures)
        with open(self.out_path, 'w') as f:
            json.dump(rec, f, indent=1)
        self.get_logger().info(f'saved {len(self.captures)} captures → {os.path.abspath(self.out_path)}')

    # ── Stage-A overlay ──
    def _project(self, pts):
        pts = np.atleast_2d(np.asarray(pts, np.float64))
        ok = pts[:, 2] > 0.05
        uv = np.full((len(pts), 2), np.nan)
        if ok.any():
            p, _ = cv2.projectPoints(pts[ok].reshape(-1, 1, 3), np.zeros(3), np.zeros(3),
                                     self.K, self.D)
            uv[ok] = p.reshape(-1, 2)
        return uv

    def _gui(self):
        if self.img is None or self.K is None:
            return
        im = self.img.copy()
        H = 22

        def line(i, txt, col=(240, 240, 240)):
            cv2.putText(im, txt, (10, H * (i + 1)), cv2.FONT_HERSHEY_SIMPLEX, .52, (0, 0, 0), 3)
            cv2.putText(im, txt, (10, H * (i + 1)), cv2.FONT_HERSHEY_SIMPLEX, .52, col, 1)

        if self.bg_want > 0:
            line(0, f'POOLING BACKGROUND ({self.bg_want} left) - reflector OFF', (0, 200, 255))
        elif self.bg_sorted is None:
            line(0, 'NO BACKGROUND - reflector OFF, then ~/background', (0, 0, 255))
        else:
            line(0, f'bg {len(self.bg_sorted)} vox | captures {len(self.captures)}')

        # live detection: cluster dots (depth-coloured) + apex crosshair.
        # The background is never drawn — foreground only, per design.
        if self.last is not None and self.bg_sorted is not None:
            L = self.last
            Pc = (self.T_cam_lidar[:3, :3] @ L['cluster'].T).T + self.T_cam_lidar[:3, 3]
            if self.show_cluster:
                uv = self._project(Pc)
                z = Pc[:, 2]
                zn = np.clip((z - z.min()) / max(z.ptp(), 1e-6), 0, 1)
                for (u, v), t in zip(uv, zn):
                    if np.isfinite(u):
                        c = (int(255 * (1 - t)), 80, int(255 * t))   # near=blue → far=red
                        cv2.circle(im, (int(u), int(v)), 2, c, -1)
            au, av = self._project(L['apex_c'])[0]
            if np.isfinite(au):
                au, av = int(au), int(av)
                cv2.drawMarker(im, (au, av), (0, 255, 0), cv2.MARKER_CROSS, 26, 2)
                cv2.circle(im, (au, av), 9, (0, 255, 0), 1)
                rng = np.linalg.norm(L['apex_l'])
                cv2.putText(im, f'{rng:.2f} m {L["method"]}', (au + 12, av - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 0, 0), 3)
                cv2.putText(im, f'{rng:.2f} m {L["method"]}', (au + 12, av - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 255, 0), 1)
            warn = ' | +%d extra cluster(s)!' % L['n_extra'] if L['n_extra'] else ''
            line(1, f'fg {L["n_fg"]} pts | cluster {len(L["cluster"])} | {L["method"]}{warn}',
                 (0, 200, 255) if (L['n_extra'] or L['method'] != 'planes3') else (200, 255, 200))
        elif self.bg_sorted is not None:
            line(1, 'no foreground cluster (reflector on? in view? in range?)', (0, 200, 255))

        # pinned captured apexes — your pose spread building up on the image
        for c in self.captures:
            u, v = self._project(np.array(c['p_cam']))[0]
            if np.isfinite(u):
                cv2.circle(im, (int(u), int(v)), 6, (255, 200, 0), 2)
                cv2.putText(im, str(c['idx']), (int(u) + 7, int(v) + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 200, 0), 1)
        if self.cap_armed:
            line(2, f'CAPTURING... {self.cap_armed} frames left', (0, 255, 255))

        if self.dscale != 1.0:
            im = cv2.resize(im, None, fx=self.dscale, fy=self.dscale)
        self.pub_dbg.publish(self.bridge.cv2_to_imgmsg(im, 'bgr8'))
        self._frame = im
        if self.show_window:
            cv2.imshow('lidar_reflector (Stage A)', im)
            cv2.waitKey(1)


def main():
    rclpy.init()
    node = LidarReflectorDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node._save()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
