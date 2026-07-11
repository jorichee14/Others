#!/usr/bin/env python3
"""
Radar ↔ Camera Extrinsic Calibration  —  ChArUco board + trihedral reflector
============================================================================
ONE rigid target: a ChArUco board with a trihedral corner reflector bolted to
it at a KNOWN, FIXED offset. The reflector apex is the single brightest radar
return; because it is rigidly tied to the board, the CAMERA can locate the same
apex from the board pose.

    camera : detect ChArUco board → T_cam_board (6-DOF) → p_cam = T_cam_board·apex_board
    radar  : reflector = strongest gated return → p_radar   (a 3-D point)

A radar point has no orientation, so ONE view can't give a 6-DOF transform.
Move the rig to N ≥ 3 non-collinear spots, collect corresponding point pairs
{(p_cam^i, p_radar^i)}, and solve the rigid registration (Kabsch/Umeyama):

    min_{R,t} Σ ‖ R·p_radar^i + t − p_cam^i ‖²   ⇒   X = T_cam_radar
    p_cam = R·p_radar + t        (X maps a radar point into the camera frame)

WHY THE CAMERA SIDE USES THE BOARD, NOT DEPTH
─────────────────────────────────────────────
A bare corner reflector is specular — stereo/ToF depth on bare metal is
unreliable, and a hand-click on it is ~10 px noisy. The ChArUco board pose is
metric (from the known square size), sub-mm, and fully automatic. The apex
offset is measured once, in the board frame (see README).

RADAR SIDE (inspired by the click-based v8 tool, improved)
──────────────────────────────────────────────────────────
  • BACKGROUND SUBTRACTION — pool N frames of the static scene (rig absent); a
    live point is "new" only if far from every background point. Kills clutter.
  • GATING — range window, optional |doppler| window (the held-still reflector
    is ~0 doppler, so moving people are rejected), and — once an extrinsic
    exists — proximity to the camera-predicted apex.
  • HIGHEST-SNR SELECTION — among the surviving points, take argmax(SNR). The
    trihedral is engineered to be the strongest reflector, so this is the apex.
    (No clustering: one bright, background-subtracted point is the target.)

VALIDATION (carried over from v8)
─────────────────────────────────
  • Leave-one-out cross-validation (honest generalisation on a small set).
  • Per-axis SIGNED residual bias (RMS hides a constant lean; a mean residual
    on an axis exposes leftover range bias or a small axis error).
  • Reprojection error in pixels (project the radar-predicted apex through K).
  • Point-set condition number (warns on collinear/planar poses).
  • Range scale/bias diagnostic (radar range vs camera range).
  • Optional tape-measured |t| baseline check.
  • LIVE reprojection overlay published as an image: the whole radar cloud
    projected onto the camera feed via the solved X. Dots staying glued to the
    reflector everywhere in the FoV is the single most trustworthy check.

Coordinate conventions
──────────────────────
  radar  : X=forward, Y=left,  Z=up     (automotive)
  camera : X=right,   Y=down,  Z=forward (optical / pinhole)
  Raw radar points are fed straight into Kabsch, so the solved R absorbs the
  full ~90° frame difference AND the mounting rotation — no separate remap
  step to get wrong. A large-looking R (≈90°) is therefore EXPECTED and correct.

Control (std_msgs/Empty topics)
───────────────────────────────
  ~/background  pool background frames (rig OUT of scene)
  ~/capture     grab one pose now (manual mode / on demand)
  ~/solve       force a solve + full validation report
  ~/reset       clear all captures
  ~/save        write YAML/JSON now

Deps:  numpy scipy opencv-contrib-python  +  ROS2 rclpy cv_bridge sensor_msgs_py
"""
import json
import numpy as np
from scipy.spatial.transform import Rotation as Rot
import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
import message_filters
import tf2_ros
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from std_msgs.msg import Empty
from geometry_msgs.msg import TransformStamped
from cv_bridge import CvBridge

try:
    from sensor_msgs_py import point_cloud2 as pc2
    _HAVE_PC2 = True
except Exception:
    _HAVE_PC2 = False


_DICT_MAP = {
    'DICT_4X4_50': cv2.aruco.DICT_4X4_50, 'DICT_4X4_100': cv2.aruco.DICT_4X4_100,
    'DICT_4X4_250': cv2.aruco.DICT_4X4_250, 'DICT_5X5_50': cv2.aruco.DICT_5X5_50,
    'DICT_5X5_100': cv2.aruco.DICT_5X5_100, 'DICT_6X6_50': cv2.aruco.DICT_6X6_50,
    'DICT_6X6_250': cv2.aruco.DICT_6X6_250,
}


def build_board(sx, sy, square, marker, dict_name):
    dname = _DICT_MAP.get(dict_name, cv2.aruco.DICT_4X4_50)
    d = (cv2.aruco.getPredefinedDictionary(dname)
         if hasattr(cv2.aruco, "getPredefinedDictionary")
         else cv2.aruco.Dictionary_get(dname))
    if hasattr(cv2.aruco, "CharucoBoard"):
        try:
            b = cv2.aruco.CharucoBoard((sx, sy), square, marker, d)
        except Exception:
            b = cv2.aruco.CharucoBoard_create(sx, sy, square, marker, d)
    else:
        b = cv2.aruco.CharucoBoard_create(sx, sy, square, marker, d)
    new = hasattr(cv2.aruco, "CharucoDetector")
    det = cv2.aruco.CharucoDetector(b) if new else None
    obj = b.getChessboardCorners() if hasattr(b, "getChessboardCorners") else b.chessboardCorners
    return d, b, det, new, obj


# ─────────────── maths ───────────────
def kabsch(P, Q):
    """Rigid transform with q = R p + t, least squares (Kabsch/Umeyama, no scale).
    P,Q: (N,3) source(radar), target(camera). Returns R,t."""
    muP, muQ = P.mean(0), Q.mean(0)
    H = (P - muP).T @ (Q - muQ)
    U, _, Vt = np.linalg.svd(H)
    D = np.diag([1., 1., float(np.sign(np.linalg.det(Vt.T @ U.T)))])
    R = Vt.T @ D @ U.T
    t = muQ - R @ muP
    return R, t


def rms_3d(P, Q, R, t):
    pred = (P @ R.T) + t
    return float(np.sqrt(((pred - Q) ** 2).sum(1).mean()))


def loo_cross_val(P, Q):
    """Leave-one-out CV: refit on N-1, predict the held-out point. Returns
    (rms_m, max_m) of held-out 3-D errors, or None if too few points."""
    n = len(P)
    if n < 4:
        return None
    errs = []
    idx = np.arange(n)
    for i in range(n):
        m = idx != i
        Ri, ti = kabsch(P[m], Q[m])
        errs.append(float(np.linalg.norm((Ri @ P[i] + ti) - Q[i])))
    errs = np.array(errs)
    return float(np.sqrt((errs ** 2).mean())), float(errs.max())


def condition_number(P):
    Pc = P - P.mean(0)
    if len(Pc) < 3:
        return float('inf')
    return float(np.linalg.cond(Pc))


def project(pt_cam, K, D):
    if pt_cam[2] <= 0:
        return None
    uv, _ = cv2.projectPoints(pt_cam.reshape(1, 3), np.zeros(3), np.zeros(3), K, D)
    return int(round(uv[0, 0, 0])), int(round(uv[0, 0, 1]))


class RadarCameraCalib(Node):
    def __init__(self):
        super().__init__('radar_camera_calib')
        dp = self.declare_parameter
        # --- camera ---
        dp('image_topic', '/zed/zed_node/left/image_rect_color')
        dp('info_topic',  '/zed/zed_node/left/camera_info')
        # --- board (ChArUco) ---
        dp('squares_x', 9); dp('squares_y', 7)
        dp('square_len', 0.020); dp('marker_len', 0.015)
        dp('dictionary', 'DICT_4X4_50')
        dp('min_corners', 8); dp('max_reproj_px', 1.5)
        # --- reflector apex offset in the BOARD frame (metres) — MEASURE THIS ---
        dp('reflector_offset_x', 0.0)
        dp('reflector_offset_y', 0.0)
        dp('reflector_offset_z', 0.0)
        # --- radar (/points_all: x,y,z,snr,doppler) ---
        dp('radar_topic', '/points_all')
        dp('pc_field_x', 'x'); dp('pc_field_y', 'y'); dp('pc_field_z', 'z')
        dp('pc_field_snr', 'snr'); dp('pc_field_doppler', 'doppler')
        dp('select_by', 'snr')          # 'snr' (recommended) | 'nearest'
        dp('min_range', 0.3); dp('max_range', 20.0)
        dp('max_abs_doppler', -1.0)     # >0 → keep |doppler| below this (still rig ≈0); <=0 disables
        dp('gate_radius', 1.0)          # m; once X exists, radar pt must be within this of predicted apex
        # --- background subtraction ---
        dp('bg_accum_frames', 15)       # radar frames pooled on ~/background
        dp('bg_match_dist', 0.2)        # m; live pt is "new" if farther than this from all bg pts
        dp('require_background', False) # if True, refuse to capture until background pooled
        # --- radar range correction (applied at ingest, before everything) ---
        dp('radar_range_scale', 1.0)    # multiply xyz (fixes proportional/units error)
        dp('radar_range_bias_m', 0.0)   # subtract along radial dir (fixes constant offset)
        # --- capture / convergence ---
        dp('capture_mode', 'auto')      # 'auto' | 'manual'
        dp('stable_window', 12)
        dp('stable_std', 0.01)          # m; per-pose jitter (cam & radar) to call it "still"
        dp('min_baseline', 0.15)        # m; min move between auto-captures
        dp('min_points', 6)
        dp('sync_slop', 0.06)
        # --- validation thresholds / verdict ---
        dp('val_pass_reproj_px', 20.0)
        dp('val_pass_3d_mm', 150.0)
        dp('val_pass_bias_mm', 50.0)
        dp('measured_baseline_m', -1.0) # >0 → compare |t| against tape measure
        dp('baseline_tol_m', 0.03)
        # --- frames / output ---
        dp('parent_frame', 'zed_left_camera_optical_frame')
        dp('child_frame',  'radar_link')
        dp('camera_name', 'zed_left'); dp('radar_name', 'radar')
        dp('output_path', '')
        dp('publish_tf', True)
        dp('debug_image', True)
        dp('debug_image_topic', '/radar_camera_calib/debug_image')
        dp('radar_watchdog_s', 3.0)

        g = lambda n: self.get_parameter(n).value
        self.min_corners = int(g('min_corners')); self.max_reproj = g('max_reproj_px')
        self.dict, self.board, self.det, self.new_api, self.obj = build_board(
            int(g('squares_x')), int(g('squares_y')),
            g('square_len'), g('marker_len'), g('dictionary'))
        self.apex_board = np.array([g('reflector_offset_x'),
                                    g('reflector_offset_y'),
                                    g('reflector_offset_z')], float)
        self.bridge = CvBridge(); self.K = None; self.D = None

        self.radar_topic = g('radar_topic')
        self.fx, self.fy, self.fz = g('pc_field_x'), g('pc_field_y'), g('pc_field_z')
        self.fsnr, self.fdop = g('pc_field_snr'), g('pc_field_doppler')
        self.select_by = g('select_by')
        self.min_range = g('min_range'); self.max_range = g('max_range')
        self.max_abs_doppler = g('max_abs_doppler')
        self.gate_radius = g('gate_radius')
        self.bg_accum_frames = int(g('bg_accum_frames')); self.bg_match_dist = g('bg_match_dist')
        self.require_background = bool(g('require_background'))
        self.range_scale = g('radar_range_scale'); self.range_bias = g('radar_range_bias_m')

        self.capture_mode = g('capture_mode')
        self.stable_window = int(g('stable_window')); self.stable_std = g('stable_std')
        self.min_baseline = g('min_baseline'); self.min_points = int(g('min_points'))

        self.val_px = g('val_pass_reproj_px'); self.val_3d = g('val_pass_3d_mm')
        self.val_bias = g('val_pass_bias_mm')
        self.measured_baseline = g('measured_baseline_m'); self.baseline_tol = g('baseline_tol_m')

        self.parent_frame, self.child_frame = g('parent_frame'), g('child_frame')
        self.camera_name, self.radar_name = g('camera_name'), g('radar_name')
        op = g('output_path')
        self.output_path = op if op else f"extrinsic_{self.camera_name}__{self.radar_name}.yaml"
        self.publish_tf = bool(g('publish_tf'))
        self.want_debug = bool(g('debug_image'))
        self.watchdog_s = g('radar_watchdog_s')

        # state
        self.win = []                 # rolling [(p_cam, p_radar), ...] for stability
        self.captures = []            # accepted [(p_cam, p_radar, snr, doppler), ...]
        self.last_capture_cam = None
        self.manual_capture_req = False
        self.bg_radar = None          # (M,3) pooled background points (raw radar frame)
        self.bg_accum = None          # accumulation buffer while pooling
        self.X = None; self.rms = None
        self.last_radar_stamp = 0.0

        self.create_subscription(CameraInfo, g('info_topic'), self._info, qos_profile_sensor_data)
        img = message_filters.Subscriber(self, Image, g('image_topic'),
                                         qos_profile=qos_profile_sensor_data)
        radar = message_filters.Subscriber(self, PointCloud2, self.radar_topic,
                                           qos_profile=qos_profile_sensor_data)
        self.sync = message_filters.ApproximateTimeSynchronizer([img, radar], 30, g('sync_slop'))
        self.sync.registerCallback(self._pair)
        # raw radar tap for background pooling + watchdog
        self.create_subscription(PointCloud2, self.radar_topic, self._radar_raw, qos_profile_sensor_data)

        self.create_subscription(Empty, '~/background', lambda _: self._start_background(), 1)
        self.create_subscription(Empty, '~/capture',    lambda _: self._on_capture(), 1)
        self.create_subscription(Empty, '~/solve',      lambda _: self._solve(force=True), 1)
        self.create_subscription(Empty, '~/reset',      lambda _: self._reset(), 1)
        self.create_subscription(Empty, '~/save',       lambda _: self._save_now(), 1)

        self.tf_static = tf2_ros.StaticTransformBroadcaster(self)
        self.dbg_pub = (self.create_publisher(Image, g('debug_image_topic'), 1)
                        if self.want_debug else None)
        self.create_timer(1.0, self._watchdog)

        self.get_logger().info(
            f"[radar_camera_calib] OpenCV {cv2.__version__}, {'new' if self.new_api else 'old'} aruco\n"
            f"  camera : {g('image_topic')}  (parent/optical = {self.parent_frame})\n"
            f"  radar  : {self.radar_topic}  (child = {self.child_frame})\n"
            f"           fields x/y/z={self.fx}/{self.fy}/{self.fz} snr={self.fsnr} doppler={self.fdop}\n"
            f"           select_by={self.select_by}  range [{self.min_range},{self.max_range}] m\n"
            + (f"           range correction scale={self.range_scale} bias={self.range_bias} m\n"
               if (self.range_scale != 1.0 or self.range_bias != 0.0) else "")
            + f"  board  : {int(g('squares_x'))}x{int(g('squares_y'))} sq {g('square_len')} "
            f"mk {g('marker_len')} {g('dictionary')}\n"
            f"  apex_in_board (m): {self.apex_board.tolist()}\n"
            f"  capture: {self.capture_mode}  min_points {self.min_points}  "
            f"min_baseline {self.min_baseline} m\n"
            f"  output -> {self.output_path}\n"
            f"  1) rig OUT of scene → publish Empty on ~/background\n"
            f"  2) bring rig in, move it around the shared FoV, pausing at each spot\n"
            f"  3) auto-captures when still+moved; ~/solve, ~/save, ~/reset any time")

    # ── intrinsics / watchdog ──
    def _info(self, msg):
        if self.K is None:
            self.K = np.array(msg.k).reshape(3, 3)
            self.D = np.array(msg.d) if len(msg.d) else np.zeros(5)
            self.get_logger().info(f"intrinsics locked ({msg.width}x{msg.height})")

    def _watchdog(self):
        if self.last_radar_stamp == 0.0:
            return
        age = self.get_clock().now().nanoseconds * 1e-9 - self.last_radar_stamp
        if age > self.watchdog_s:
            self.get_logger().warn(
                f"no radar on {self.radar_topic} for {age:.0f}s — wrong topic / node down?",
                throttle_duration_sec=5.0)

    # ── radar parsing ──
    def _radar_correct(self, xyz):
        if self.range_scale == 1.0 and self.range_bias == 0.0:
            return xyz
        xyz = xyz * float(self.range_scale)
        if self.range_bias != 0.0:
            r = np.linalg.norm(xyz, axis=1, keepdims=True)
            r = np.where(r < 1e-6, 1e-6, r)
            xyz = xyz - self.range_bias * (xyz / r)
        return xyz

    def _read_radar(self, msg):
        """Return (xyz Nx3 corrected, snr N, doppler N) or (None,None,None)."""
        if not _HAVE_PC2:
            return None, None, None
        names = [f.name for f in msg.fields]
        want = [self.fx, self.fy, self.fz]
        has_snr = self.fsnr in names; has_dop = self.fdop in names
        if has_snr: want.append(self.fsnr)
        if has_dop: want.append(self.fdop)
        arr = list(pc2.read_points(msg, field_names=want, skip_nans=True))
        if not arr:
            return None, None, None
        arr = np.array([tuple(a) for a in arr], float)
        xyz = self._radar_correct(arr[:, :3])
        col = 3
        snr = arr[:, col] if has_snr else np.zeros(len(arr)); col += has_snr
        dop = arr[:, col] if has_dop else np.zeros(len(arr))
        return xyz, snr, dop

    def _radar_raw(self, msg):
        self.last_radar_stamp = self.get_clock().now().nanoseconds * 1e-9
        if self.bg_accum is None:
            return
        xyz, _, _ = self._read_radar(msg)
        if xyz is not None:
            self.bg_accum.append(xyz)
            if len(self.bg_accum) >= self.bg_accum_frames:
                self.bg_radar = np.vstack(self.bg_accum)
                n = len(self.bg_radar); self.bg_accum = None
                self.get_logger().info(
                    f"background pooled: {self.bg_accum_frames} frames, {n} pts "
                    f"(match dist {self.bg_match_dist} m). Bring the rig in.")

    def _start_background(self):
        self.bg_accum = []
        self.get_logger().info(
            f"pooling {self.bg_accum_frames} background frames — keep the rig OUT of the scene…")

    def _select_radar(self, xyz, snr, dop, predicted=None):
        """Background-subtract + gate, then pick the reflector return."""
        if xyz is None or len(xyz) == 0:
            return None, None, None, 0
        keep = np.ones(len(xyz), bool)
        rng = np.linalg.norm(xyz, axis=1)
        keep &= (rng >= self.min_range) & (rng <= self.max_range)
        if self.max_abs_doppler > 0:
            keep &= (np.abs(dop) <= self.max_abs_doppler)
        if self.bg_radar is not None and len(self.bg_radar):
            diff = xyz[:, None, :] - self.bg_radar[None, :, :]
            mind = np.sqrt((diff ** 2).sum(2)).min(1)
            keep &= (mind > self.bg_match_dist)
        if predicted is not None:
            keep &= (np.linalg.norm(xyz - predicted, axis=1) <= self.gate_radius)
        n_gated = int(keep.sum())
        if n_gated == 0:
            return None, None, None, 0
        xyz, snr, dop = xyz[keep], snr[keep], dop[keep]
        if predicted is not None:
            idx = int(np.argmin(np.linalg.norm(xyz - predicted, axis=1)))
        elif self.select_by == 'snr' and np.any(np.isfinite(snr)) and snr.max() > 0:
            idx = int(np.argmax(snr))            # ← highest-SNR return = the trihedral
        else:
            idx = int(np.argmin(np.linalg.norm(xyz, axis=1)))
        return xyz[idx], float(snr[idx]), float(dop[idx]), n_gated

    # ── camera: board pose → apex in camera frame ──
    def _apex_in_camera(self, gray):
        if self.new_api:
            cc, cid, _, _ = self.det.detectBoard(gray)
        else:
            mc, mids, _ = cv2.aruco.detectMarkers(gray, self.dict)
            if mids is None or len(mids) == 0:
                return None, None, 0, None
            _, cc, cid = cv2.aruco.interpolateCornersCharuco(mc, mids, gray, self.board)
        n = 0 if cid is None else len(cid)
        if n < self.min_corners:
            return None, None, n, None
        objp = self.obj[cid.flatten()]
        ok, rvec, tvec = cv2.solvePnP(objp, cc, self.K, self.D, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            return None, None, n, None
        proj, _ = cv2.projectPoints(objp, rvec, tvec, self.K, self.D)
        reproj = float(cv2.norm(cc, proj, cv2.NORM_L2) / len(proj))
        if reproj > self.max_reproj:
            return None, reproj, n, None
        R = cv2.Rodrigues(rvec)[0]
        p_cam = (R @ self.apex_board + tvec[:, 0])
        return p_cam, reproj, n, (rvec, tvec)

    # ── synced pair ──
    def _pair(self, a_img, radar_msg):
        if self.K is None:
            return
        try:
            bgr = self.bridge.imgmsg_to_cv2(a_img, 'bgr8')
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        except Exception as e:
            self.get_logger().warn(f"cv_bridge: {e}"); return
        p_cam, reproj, n, pose = self._apex_in_camera(gray)
        xyz, snr, dop = self._read_radar(radar_msg)
        predicted = None
        if self.X is not None and p_cam is not None:
            predicted = self.X[:3, :3].T @ (p_cam - self.X[:3, 3])   # camera → radar
        p_radar, snr_i, dop_i, n_gated = self._select_radar(xyz, snr, dop, predicted)

        self._publish_debug(bgr, pose, p_cam, xyz, n, reproj,
                            p_radar is not None, n_gated)

        if p_cam is None:
            self.get_logger().info(f"no board (n={n}, reproj={reproj})", throttle_duration_sec=2.0)
            return
        if self.require_background and self.bg_radar is None:
            self.get_logger().info("waiting for ~/background (require_background=True)",
                                   throttle_duration_sec=3.0)
            return
        if p_radar is None:
            self.get_logger().info(f"no gated radar return (of {0 if xyz is None else len(xyz)})",
                                   throttle_duration_sec=2.0)
            return

        self.win.append((p_cam, p_radar, snr_i, dop_i))
        if len(self.win) > self.stable_window:
            self.win.pop(0)
        stable = False
        if len(self.win) >= self.stable_window:
            cams = np.array([w[0] for w in self.win]); rads = np.array([w[1] for w in self.win])
            cstd = float(np.linalg.norm(cams.std(0))); rstd = float(np.linalg.norm(rads.std(0)))
            stable = (cstd < self.stable_std and rstd < self.stable_std)
            self.get_logger().info(
                f"apex cam {p_cam.round(3).tolist()} radar {p_radar.round(3).tolist()} "
                f"snr {snr_i:.1f} still? {stable} (cam σ {cstd*1000:.0f}mm) captures {len(self.captures)}",
                throttle_duration_sec=1.0)

        if self.capture_mode == 'auto' and stable:
            self._accept()
        elif self.manual_capture_req and len(self.win) >= max(3, self.stable_window // 2):
            self.manual_capture_req = False
            self._accept(force=True)

    def _accept(self, force=False):
        p_cam = np.mean([w[0] for w in self.win], 0)
        p_radar = np.mean([w[1] for w in self.win], 0)
        snr_i = float(np.mean([w[2] for w in self.win]))
        dop_i = float(np.mean([w[3] for w in self.win]))
        if (not force and self.last_capture_cam is not None and
                np.linalg.norm(p_cam - self.last_capture_cam) < self.min_baseline):
            return
        self.captures.append((p_cam, p_radar, snr_i, dop_i))
        self.last_capture_cam = p_cam; self.win.clear()
        self.get_logger().info(
            f"*** CAPTURED #{len(self.captures)}  cam {p_cam.round(3).tolist()}  "
            f"radar {p_radar.round(3).tolist()}  snr {snr_i:.1f} ***")
        self._solve()

    def _on_capture(self):
        self.manual_capture_req = True
        self.get_logger().info("manual capture requested — hold the rig still")

    # ── solve + validation ──
    def _solve(self, force=False):
        need = 3 if force else self.min_points
        if len(self.captures) < max(3, need):
            if force:
                self.get_logger().warn(f"need ≥3 captures (have {len(self.captures)})")
            return
        P = np.array([c[1] for c in self.captures])   # radar (source)
        Q = np.array([c[0] for c in self.captures])   # camera (target)
        R, t = kabsch(P, Q)
        self.X = np.eye(4); self.X[:3, :3] = R; self.X[:3, 3] = t
        self.rms = rms_3d(P, Q, R, t)

        cond = condition_number(P)
        span = np.linalg.svd(P - P.mean(0), compute_uv=False)
        planar = span[2] < max(1e-3, 0.02 * span[0])
        q = Rot.from_matrix(R).as_quat(); rpy = Rot.from_matrix(R).as_euler('xyz', degrees=True)
        tmag = float(np.linalg.norm(t))

        lines = [
            f"\n=== T_{self.parent_frame}_{self.child_frame}  (camera ← radar) ===",
            f"  captures {len(self.captures)}   in-sample RMS {self.rms*1000:.1f} mm   cond {cond:.1f}",
            f"  xyz (m) : {t[0]:+.4f} {t[1]:+.4f} {t[2]:+.4f}   |t| {tmag*100:.1f} cm",
            f"  quat    : {q[0]:+.4f} {q[1]:+.4f} {q[2]:+.4f} {q[3]:+.4f}",
            f"  rpy(deg): {rpy[0]:+.2f} {rpy[1]:+.2f} {rpy[2]:+.2f}",
        ]
        loo = loo_cross_val(P, Q)
        if loo is not None:
            loo_rms, loo_max = loo
            infl = loo_rms / self.rms if self.rms > 1e-9 else float('inf')
            lines.append(f"  LOO CV  : {loo_rms*1000:.1f} mm (max {loo_max*1000:.1f})"
                         + (f"  !! {infl:.1f}× in-sample — overfit/too few/clustered" if infl > 3 else ""))
        if planar:
            lines.append("  !! radar points PLANAR/collinear — out-of-plane rotation weak; "
                         "vary height & range")
        elif cond > 200:
            lines.append(f"  !! high condition ({cond:.0f}) — spread poses more in X/Y/Z")

        # per-axis signed bias + reproj px
        res = (P @ R.T + t) - Q
        bias_mm = res.mean(0) * 1000; std_mm = res.std(0) * 1000
        lines.append("  signed residual (pred − cam), mm:  "
                     f"X {bias_mm[0]:+.0f}±{std_mm[0]:.0f}  Y {bias_mm[1]:+.0f}±{std_mm[1]:.0f}  "
                     f"Z {bias_mm[2]:+.0f}±{std_mm[2]:.0f}")
        errs_px = []
        for i in range(len(P)):
            uv_t = project(Q[i], self.K, self.D); uv_p = project(P[i] @ R.T + t, self.K, self.D)
            if uv_t and uv_p:
                errs_px.append(float(np.linalg.norm(np.subtract(uv_p, uv_t))))
        mean_px = float(np.mean(errs_px)) if errs_px else float('nan')
        mean_3d = float(np.linalg.norm(res, axis=1).mean()) * 1000
        lines.append(f"  reproj  : {mean_px:.1f} px   mean 3-D {mean_3d:.1f} mm")

        # range diagnostic (camera range vs radar range)
        cam_r = np.linalg.norm(Q, axis=1); rad_r = np.linalg.norm(P, axis=1)
        if len(P) >= 2:
            A = np.vstack([rad_r, np.ones_like(rad_r)]).T
            (a, b), *_ = np.linalg.lstsq(A, cam_r, rcond=None)
            if abs(a - 1) > 0.03 or abs(b) > 0.05:
                lines.append(f"  range fit: cam_r = {a:.3f}·radar_r {b:+.3f} m  → "
                             f"try radar_range_scale={1/a:.4f} bias={-b:+.4f}")

        # baseline check + verdict
        if self.measured_baseline > 0:
            d = abs(tmag - self.measured_baseline)
            lines.append(f"  baseline: |t| {tmag*100:.1f} cm vs measured "
                         f"{self.measured_baseline*100:.1f} cm → Δ {d*100:.1f} cm "
                         f"[{'OK' if d <= self.baseline_tol else 'MISMATCH'}]")
        checks = [("reproj_px", mean_px, self.val_px), ("3d_mm", mean_3d, self.val_3d),
                  ("bias_mm", float(np.abs(bias_mm).max()), self.val_bias)]
        verdict = all(v <= lim for _, v, lim in checks)
        lines.append("  VERDICT : " + ("✔ GOOD  " if verdict else "✗ SUSPECT  ")
                     + "  ".join(f"{k} {v:.1f}/{lim:.0f}[{'P' if v<=lim else 'F'}]"
                                 for k, v, lim in checks))
        self.get_logger().info("\n".join(lines))

        if self.publish_tf:
            self._broadcast()
        self._save(mean_px, mean_3d, bias_mm, cond, loo, planar, verdict)

    def _reset(self):
        self.captures.clear(); self.win.clear(); self.last_capture_cam = None
        self.X = None; self.rms = None
        self.get_logger().info("reset — captures cleared (background kept)")

    def _save_now(self):
        if self.X is not None:
            self._solve(force=True)
        else:
            self.get_logger().warn("nothing to save yet")

    # ── outputs ──
    def _broadcast(self):
        m = TransformStamped()
        m.header.stamp = self.get_clock().now().to_msg()
        m.header.frame_id = self.parent_frame; m.child_frame_id = self.child_frame
        X = self.X
        m.transform.translation.x = float(X[0, 3]); m.transform.translation.y = float(X[1, 3])
        m.transform.translation.z = float(X[2, 3])
        q = Rot.from_matrix(X[:3, :3]).as_quat()
        (m.transform.rotation.x, m.transform.rotation.y,
         m.transform.rotation.z, m.transform.rotation.w) = map(float, q)
        self.tf_static.sendTransform(m)

    def _save(self, mean_px, mean_3d, bias_mm, cond, loo, planar, verdict):
        X = self.X; Xinv = np.linalg.inv(X)
        q = Rot.from_matrix(X[:3, :3]).as_quat(); rpy = Rot.from_matrix(X[:3, :3]).as_euler('xyz', degrees=True)
        qi = Rot.from_matrix(Xinv[:3, :3]).as_quat()
        data = {
            'parent_frame': self.parent_frame, 'child_frame': self.child_frame,
            'camera_name': self.camera_name, 'radar_name': self.radar_name,
            'n_captures': len(self.captures),
            'verdict_pass': bool(verdict), 'planar_warning': bool(planar),
            'in_sample_rms_mm': float(self.rms) * 1000,
            'loo_cv_rms_mm': (loo[0] * 1000 if loo else None),
            'mean_reproj_px': float(mean_px), 'mean_3d_mm': float(mean_3d),
            'residual_bias_mm_xyz': [float(v) for v in bias_mm],
            'condition_number': float(cond),
            'radar_range_scale': self.range_scale, 'radar_range_bias_m': self.range_bias,
            # T_cam_radar : p_cam = R p_radar + t
            'T_cam_radar_translation': [float(v) for v in X[:3, 3]],
            'T_cam_radar_quaternion_xyzw': [float(v) for v in q],
            'T_cam_radar_rpy_deg': [float(v) for v in rpy],
            # inverse
            'T_radar_cam_translation': [float(v) for v in Xinv[:3, 3]],
            'T_radar_cam_quaternion_xyzw': [float(v) for v in qi],
            'apex_offset_in_board_m': [float(v) for v in self.apex_board],
            'static_tf_cmd': (f"ros2 run tf2_ros static_transform_publisher "
                              f"{X[0,3]:.6f} {X[1,3]:.6f} {X[2,3]:.6f} "
                              f"{q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f} "
                              f"{self.parent_frame} {self.child_frame}"),
            'stamp': self.get_clock().now().nanoseconds}
        try:
            with open(self.output_path, 'w') as f:
                for k, v in data.items():
                    f.write(f"{k}: {v}\n")
            with open(self.output_path.replace('.yaml', '.json'), 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            self.get_logger().warn(f"save failed: {e}")

    def _publish_debug(self, bgr, pose, p_cam, radar_xyz, n, reproj, got_radar, n_gated):
        if self.dbg_pub is None:
            return
        try:
            h, w = bgr.shape[:2]
            # live overlay: whole radar cloud projected via X, coloured by depth
            if self.X is not None and radar_xyz is not None and len(radar_xyz):
                Pc = (radar_xyz @ self.X[:3, :3].T) + self.X[:3, 3]
                for p in Pc:
                    uv = project(p, self.K, self.D)
                    if uv and 0 <= uv[0] < w and 0 <= uv[1] < h:
                        f = max(0.0, min(1.0, (p[2] - 0.5) / 7.5))
                        cv2.circle(bgr, uv, 4, (int(255 * f), 60, int(255 * (1 - f))), -1)
            if pose is not None:
                rvec, tvec = pose
                cv2.drawFrameAxes(bgr, self.K, self.D, rvec, tvec, 0.05)
                uv = project(p_cam, self.K, self.D)
                if uv:
                    col = (0, 255, 0) if got_radar else (0, 165, 255)
                    cv2.circle(bgr, uv, 9, col, 2)
                    cv2.putText(bgr, "apex", (uv[0] + 10, uv[1]),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
            txt = f"corners {n} reproj {reproj if reproj else 0:.2f}px  gated {n_gated}  captures {len(self.captures)}"
            if self.rms is not None:
                txt += f"  RMS {self.rms*1000:.0f}mm"
            if self.bg_radar is None:
                txt += "  [no bg]"
            cv2.putText(bgr, txt, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            self.dbg_pub.publish(self.bridge.cv2_to_imgmsg(bgr, 'bgr8'))
        except Exception as e:
            self.get_logger().warn(f"debug image: {e}", throttle_duration_sec=5.0)


def main():
    rclpy.init(); node = RadarCameraCalib()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        if node.X is not None:
            node.get_logger().info(f"Final extrinsic in {node.output_path}")
    finally:
        node.destroy_node(); rclpy.shutdown()


if __name__ == '__main__':
    main()
