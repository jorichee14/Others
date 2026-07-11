#!/usr/bin/env python3
"""
Radar <-> Camera Extrinsic Calibration via a ChArUco board + trihedral reflector
================================================================================
You have ONE rigid target:  a ChArUco board with a trihedral corner reflector
bolted to it at a KNOWN, FIXED offset.  The reflector's apex is the single point
that the radar sees as its brightest return, and its position relative to the
board is fixed, so the CAMERA can also predict where that apex is.

    camera  : detect the ChArUco board  ->  T_cam_board (full 6-DOF pose)
              apex in camera frame:  p_cam = T_cam_board @ apex_in_board
    radar   : the reflector is the strongest return -> p_radar (a 3-D point)

A radar point has NO orientation, so a single view cannot give the 6-DOF
extrinsic.  Instead we move the rig to many (>=3, non-collinear) positions,
collect corresponding point pairs {(p_cam^i, p_radar^i)}, and solve the rigid
point-set registration (Kabsch / Umeyama, scale fixed at 1):

    min_{R,t}  sum_i || R p_radar^i + t - p_cam^i ||^2      =>   T_cam_radar
    p_cam = R p_radar + t              (X = T_cam_radar maps radar -> camera)

The tool auto-detects planar (2-D) radar (all z ~ 0) and warns, because
out-of-plane rotation is then unobservable.

------------------------------------------------------------------------------
INPUTS (ROS 2 topics)
  camera image + camera_info    (this camera's intrinsics + ChArUco detection)
  radar detections              sensor_msgs/PointCloud2  OR  radar_msgs/RadarScan
OUTPUTS
  extrinsic_<cam>__<radar>.yaml / .json   (T_cam_radar and its inverse)
  static TF   parent_frame -> child_frame (camera_optical -> radar)
  optional annotated debug image (board axes + projected reflector apex)

WORKFLOW
  1. Measure the reflector apex offset in the BOARD frame (see README) and set
     reflector_offset_{x,y,z}.
  2. Launch the node; watch the debug image: the drawn apex dot should sit on
     the real reflector.  If not, fix the offset signs (README has the recipe).
  3. Move the rig around the shared field of view, pausing at each spot.  In
     auto mode a pair is captured when both sensors are stable and the rig has
     moved >= min_baseline from every previous capture.  Or trigger captures
     manually on ~/capture.
  4. After min_points captures the tool solves and keeps refining.  Publish
     Empty on ~/solve to force a solve, ~/reset to clear, ~/save to write out.
------------------------------------------------------------------------------
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
except Exception:                       # older distros / minimal installs
    _HAVE_PC2 = False

try:
    from radar_msgs.msg import RadarScan
    _HAVE_RADAR_MSGS = True
except Exception:
    _HAVE_RADAR_MSGS = False


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


def umeyama_no_scale(src, dst):
    """Rigid transform (R,t) with p_dst = R p_src + t, least-squares over all
    correspondences (Kabsch / Umeyama, scale fixed to 1). src,dst: (N,3).
    Returns (T 4x4, rms residual in metres, singular values of the covariance)."""
    src = np.asarray(src, float); dst = np.asarray(dst, float)
    cs = src.mean(0); cd = dst.mean(0)
    S = src - cs; D = dst - cd
    H = S.T @ D / len(src)
    U, Sig, Vt = np.linalg.svd(H)
    Dsign = np.eye(3)
    Dsign[2, 2] = np.sign(np.linalg.det(Vt.T @ U.T))   # reflection guard
    R = Vt.T @ Dsign @ U.T
    t = cd - R @ cs
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = t
    res = (dst - (src @ R.T + t))
    rms = float(np.sqrt((res ** 2).sum(1).mean()))
    return T, rms, Sig


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
        # --- reflector apex offset, expressed in the BOARD frame (metres) ---
        #   board origin = ChArUco origin (first inner chessboard corner);
        #   +x along squares_x, +y along squares_y, +z out of the board plane.
        dp('reflector_offset_x', 0.0)
        dp('reflector_offset_y', 0.0)
        dp('reflector_offset_z', 0.0)
        # --- radar ---
        dp('radar_topic', '/radar/points')
        dp('radar_type', 'pointcloud2')            # 'pointcloud2' | 'radarscan'
        # PointCloud2 field names (edit to match your driver)
        dp('pc_field_x', 'x'); dp('pc_field_y', 'y'); dp('pc_field_z', 'z')
        dp('pc_field_intensity', 'intensity')      # or 'rcs'; '' to disable
        # pick the reflector return by highest intensity/RCS (True) or nearest
        # to the camera-predicted apex once an estimate exists (always used as
        # a gate).  Range/azimuth gating keeps clutter out.
        dp('pick_by_intensity', True)
        dp('min_range', 0.3); dp('max_range', 20.0)
        dp('gate_radius', 1.0)     # m; once we have p_cam, radar pt must be
                                   # within this of the predicted apex (needs a
                                   # rough X or is skipped until first solve)
        # --- capture / convergence ---
        dp('capture_mode', 'auto')                 # 'auto' | 'manual'
        dp('stable_window', 12)                    # frames to average per pose
        dp('stable_t_std', 0.004)                  # m; jitter to call it "still"
        dp('min_baseline', 0.12)                   # m; min move between captures
        dp('min_points', 4)                        # captures before first solve
        dp('sync_slop', 0.06)
        # --- frames / output ---
        dp('parent_frame', 'zed_left_camera_optical_frame')   # camera optical
        dp('child_frame',  'radar_link')                      # radar
        dp('camera_name', 'zed_left'); dp('radar_name', 'radar')
        dp('output_path', '')
        dp('publish_tf', True)
        dp('debug_image', True)
        dp('debug_image_topic', '/radar_camera_calib/debug_image')

        g = lambda n: self.get_parameter(n).value
        self.min_corners = int(g('min_corners')); self.max_reproj = g('max_reproj_px')
        self.dict, self.board, self.det, self.new_api, self.obj = build_board(
            int(g('squares_x')), int(g('squares_y')),
            g('square_len'), g('marker_len'), g('dictionary'))
        self.apex_board = np.array([g('reflector_offset_x'),
                                    g('reflector_offset_y'),
                                    g('reflector_offset_z')], float)
        self.bridge = CvBridge(); self.K = None; self.D = None

        self.radar_type = g('radar_type')
        self.pc_fx, self.pc_fy, self.pc_fz = g('pc_field_x'), g('pc_field_y'), g('pc_field_z')
        self.pc_fi = g('pc_field_intensity')
        self.pick_by_intensity = bool(g('pick_by_intensity'))
        self.min_range = g('min_range'); self.max_range = g('max_range')
        self.gate_radius = g('gate_radius')

        self.capture_mode = g('capture_mode')
        self.stable_window = int(g('stable_window')); self.stable_t_std = g('stable_t_std')
        self.min_baseline = g('min_baseline'); self.min_points = int(g('min_points'))

        self.parent_frame, self.child_frame = g('parent_frame'), g('child_frame')
        self.camera_name, self.radar_name = g('camera_name'), g('radar_name')
        op = g('output_path')
        self.output_path = op if op else f"extrinsic_{self.camera_name}__{self.radar_name}.yaml"
        self.publish_tf = bool(g('publish_tf'))
        self.want_debug = bool(g('debug_image'))

        # state
        self.win = []                 # rolling (p_cam, p_radar) for stability
        self.captures = []            # accepted [(p_cam, p_radar), ...]
        self.last_capture_cam = None  # p_cam of the last accepted capture
        self.manual_capture_req = False
        self.X = None                 # latest T_cam_radar
        self.rms = None

        self.create_subscription(CameraInfo, g('info_topic'), self._info, qos_profile_sensor_data)
        img = message_filters.Subscriber(self, Image, g('image_topic'),
                                         qos_profile=qos_profile_sensor_data)
        if self.radar_type == 'radarscan':
            if not _HAVE_RADAR_MSGS:
                raise RuntimeError("radar_type=radarscan but radar_msgs not importable")
            radar = message_filters.Subscriber(self, RadarScan, g('radar_topic'),
                                               qos_profile=qos_profile_sensor_data)
        else:
            radar = message_filters.Subscriber(self, PointCloud2, g('radar_topic'),
                                               qos_profile=qos_profile_sensor_data)
        self.sync = message_filters.ApproximateTimeSynchronizer([img, radar], 30, g('sync_slop'))
        self.sync.registerCallback(self._pair)

        # control topics
        self.create_subscription(Empty, '~/capture', lambda _: self._on_capture(), 1)
        self.create_subscription(Empty, '~/solve',   lambda _: self._solve(force=True), 1)
        self.create_subscription(Empty, '~/reset',   lambda _: self._reset(), 1)
        self.create_subscription(Empty, '~/save',    lambda _: self._save_if_have(), 1)

        self.tf_static = tf2_ros.StaticTransformBroadcaster(self)
        self.dbg_pub = (self.create_publisher(Image, g('debug_image_topic'), 1)
                        if self.want_debug else None)

        self.get_logger().info(
            f"[radar_camera_calib] OpenCV {cv2.__version__}, {'new' if self.new_api else 'old'} aruco API\n"
            f"  camera : {g('image_topic')}  (parent/optical = {self.parent_frame})\n"
            f"  radar  : {g('radar_topic')}  [{self.radar_type}]  (child = {self.child_frame})\n"
            f"  board  : {int(g('squares_x'))}x{int(g('squares_y'))} sq {g('square_len')} "
            f"mk {g('marker_len')} {g('dictionary')}\n"
            f"  apex_in_board (m): {self.apex_board.tolist()}\n"
            f"  capture: {self.capture_mode}  min_points {self.min_points}  "
            f"min_baseline {self.min_baseline} m\n"
            f"  output -> {self.output_path}\n"
            f"  Move the rig around the shared FOV, pausing at each spot. "
            f"Publish Empty on ~/solve, ~/save, ~/reset, ~/capture.")

    # ---------- intrinsics ----------
    def _info(self, msg):
        if self.K is None:
            self.K = np.array(msg.k).reshape(3, 3)
            self.D = np.array(msg.d) if len(msg.d) else np.zeros(5)
            self.get_logger().info(f"intrinsics locked ({msg.width}x{msg.height})")

    # ---------- camera: board pose -> apex in camera frame ----------
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
        p_cam = (R @ self.apex_board + tvec[:, 0])       # apex in camera frame
        return p_cam, reproj, n, (rvec, tvec)

    # ---------- radar: strongest / gated return ----------
    def _radar_point(self, msg, predicted=None):
        pts = self._radar_points_xyz_i(msg)
        if pts is None or len(pts) == 0:
            return None
        xyz = pts[:, :3]; inten = pts[:, 3]
        rng = np.linalg.norm(xyz, axis=1)
        keep = (rng >= self.min_range) & (rng <= self.max_range)
        if predicted is not None:
            keep &= (np.linalg.norm(xyz - predicted, axis=1) <= self.gate_radius)
        if not np.any(keep):
            return None
        xyz = xyz[keep]; inten = inten[keep]
        if predicted is not None:
            idx = int(np.argmin(np.linalg.norm(xyz - predicted, axis=1)))
        elif self.pick_by_intensity and np.any(np.isfinite(inten)):
            idx = int(np.argmax(inten))
        else:
            idx = int(np.argmin(np.linalg.norm(xyz, axis=1)))   # nearest
        return xyz[idx]

    def _radar_points_xyz_i(self, msg):
        if self.radar_type == 'radarscan':
            out = []
            for r in msg.returns:
                az, el, rng = r.azimuth, r.elevation, r.range
                x = rng * np.cos(el) * np.cos(az)
                y = rng * np.cos(el) * np.sin(az)
                z = rng * np.sin(el)
                out.append((x, y, z, r.amplitude))
            return np.array(out, float) if out else None
        # PointCloud2
        fields = [self.pc_fx, self.pc_fy, self.pc_fz]
        has_i = bool(self.pc_fi) and any(f.name == self.pc_fi for f in msg.fields)
        if has_i:
            fields.append(self.pc_fi)
        if _HAVE_PC2:
            arr = list(pc2.read_points(msg, field_names=fields, skip_nans=True))
            if not arr:
                return None
            arr = np.array([tuple(a) for a in arr], float)
        else:
            return None
        if not has_i:
            arr = np.hstack([arr, np.zeros((len(arr), 1))])
        return arr

    # ---------- synced pair ----------
    def _pair(self, a_img, radar_msg):
        if self.K is None:
            return
        try:
            gray = cv2.cvtColor(self.bridge.imgmsg_to_cv2(a_img, 'bgr8'), cv2.COLOR_BGR2GRAY)
            bgr = self.bridge.imgmsg_to_cv2(a_img, 'bgr8')
        except Exception as e:
            self.get_logger().warn(f"cv_bridge: {e}"); return
        p_cam, reproj, n, pose = self._apex_in_camera(gray)
        if p_cam is None:
            self.get_logger().info(f"no board (n={n}, reproj={reproj})", throttle_duration_sec=2.0)
            self._publish_debug(bgr, None, None, n, reproj)
            return
        # predict radar location if we already have an extrinsic (tighter gate)
        predicted = None
        if self.X is not None:
            Rr = self.X[:3, :3]; tr = self.X[:3, 3]
            predicted = Rr.T @ (p_cam - tr)          # camera -> radar
        p_radar = self._radar_point(radar_msg, predicted)
        self._publish_debug(bgr, pose, p_cam, n, reproj, got_radar=p_radar is not None)
        if p_radar is None:
            self.get_logger().info("no gated radar return", throttle_duration_sec=2.0)
            return

        # stability window
        self.win.append((p_cam, p_radar))
        if len(self.win) > self.stable_window:
            self.win.pop(0)
        stable = False
        if len(self.win) >= self.stable_window:
            cams = np.array([w[0] for w in self.win])
            rads = np.array([w[1] for w in self.win])
            t_std = float(np.linalg.norm(cams.std(0)))
            r_std = float(np.linalg.norm(rads.std(0)))
            stable = (t_std < self.stable_t_std and r_std < self.stable_t_std)
            self.get_logger().info(
                f"apex cam {p_cam.round(3).tolist()}  radar {p_radar.round(3).tolist()}  "
                f"still? {stable} (cam-std {t_std*1000:.1f}mm) captures {len(self.captures)}",
                throttle_duration_sec=1.0)

        if self.capture_mode == 'auto' and stable:
            self._try_accept(np.mean([w[0] for w in self.win], 0),
                             np.mean([w[1] for w in self.win], 0))
        elif self.manual_capture_req and len(self.win) >= max(3, self.stable_window // 2):
            self.manual_capture_req = False
            self._try_accept(np.mean([w[0] for w in self.win], 0),
                             np.mean([w[1] for w in self.win], 0), force=True)

    def _try_accept(self, p_cam, p_radar, force=False):
        if (not force and self.last_capture_cam is not None and
                np.linalg.norm(p_cam - self.last_capture_cam) < self.min_baseline):
            return                                   # hasn't moved far enough yet
        self.captures.append((p_cam, p_radar))
        self.last_capture_cam = p_cam
        self.win.clear()
        self.get_logger().info(
            f"*** CAPTURED pose #{len(self.captures)}  "
            f"cam {p_cam.round(3).tolist()}  radar {p_radar.round(3).tolist()} ***")
        self._solve()

    def _on_capture(self):
        self.manual_capture_req = True
        self.get_logger().info("manual capture requested — hold the rig still")

    # ---------- solve ----------
    def _solve(self, force=False):
        if len(self.captures) < max(3, self.min_points if not force else 3):
            if force:
                self.get_logger().warn(f"need >=3 captures to solve (have {len(self.captures)})")
            return
        src = np.array([c[1] for c in self.captures])   # radar
        dst = np.array([c[0] for c in self.captures])   # camera
        # non-collinearity / planarity check
        span = np.linalg.svd(src - src.mean(0), compute_uv=False)
        planar = span[2] < 1e-3 or span[2] < 0.02 * span[0]
        X, rms, sig = umeyama_no_scale(src, dst)
        self.X = X; self.rms = rms
        t = X[:3, 3]; q = Rot.from_matrix(X[:3, :3]).as_quat()
        rpy = Rot.from_matrix(X[:3, :3]).as_euler('xyz', degrees=True)
        warn = ""
        if planar:
            warn = ("  !! radar points look PLANAR/collinear — out-of-plane "
                    "rotation is poorly observed. Add poses at different "
                    "heights/ranges (or treat radar as 2-D).\n")
        self.get_logger().info(
            f"\n=== T_{self.parent_frame}_{self.child_frame}  (camera <- radar) ===\n"
            f"  captures: {len(self.captures)}   RMS residual: {rms*1000:.1f} mm\n"
            f"  xyz (m) : {t[0]:+.4f} {t[1]:+.4f} {t[2]:+.4f}\n"
            f"  quat    : {q[0]:+.4f} {q[1]:+.4f} {q[2]:+.4f} {q[3]:+.4f}\n"
            f"  rpy(deg): {rpy[0]:+.2f} {rpy[1]:+.2f} {rpy[2]:+.2f}\n"
            f"  singval : {sig.round(3).tolist()}\n{warn}")
        if self.publish_tf:
            self._broadcast(X)
        self._save(X, rms, planar)

    def _reset(self):
        self.captures.clear(); self.win.clear(); self.last_capture_cam = None
        self.X = None; self.rms = None
        self.get_logger().info("reset — all captures cleared")

    def _save_if_have(self):
        if self.X is not None:
            self._save(self.X, self.rms, False)
            self.get_logger().info(f"saved to {self.output_path}")
        else:
            self.get_logger().warn("nothing to save yet")

    # ---------- outputs ----------
    def _broadcast(self, X):
        m = TransformStamped()
        m.header.stamp = self.get_clock().now().to_msg()
        m.header.frame_id = self.parent_frame; m.child_frame_id = self.child_frame
        m.transform.translation.x = float(X[0, 3]); m.transform.translation.y = float(X[1, 3])
        m.transform.translation.z = float(X[2, 3])
        q = Rot.from_matrix(X[:3, :3]).as_quat()
        (m.transform.rotation.x, m.transform.rotation.y,
         m.transform.rotation.z, m.transform.rotation.w) = map(float, q)
        self.tf_static.sendTransform(m)

    def _save(self, X, rms, planar):
        Xinv = np.linalg.inv(X)
        q = Rot.from_matrix(X[:3, :3]).as_quat()
        rpy = Rot.from_matrix(X[:3, :3]).as_euler('xyz', degrees=True)
        qi = Rot.from_matrix(Xinv[:3, :3]).as_quat()
        data = {
            'parent_frame': self.parent_frame, 'child_frame': self.child_frame,
            'camera_name': self.camera_name, 'radar_name': self.radar_name,
            'n_captures': len(self.captures), 'rms_residual_m': float(rms) if rms else None,
            'planar_warning': bool(planar),
            # T_cam_radar : p_cam = R p_radar + t
            'T_cam_radar_translation': [float(v) for v in X[:3, 3]],
            'T_cam_radar_quaternion_xyzw': [float(v) for v in q],
            'T_cam_radar_rpy_deg': [float(v) for v in rpy],
            # inverse, radar <- camera
            'T_radar_cam_translation': [float(v) for v in Xinv[:3, 3]],
            'T_radar_cam_quaternion_xyzw': [float(v) for v in qi],
            'apex_offset_in_board_m': [float(v) for v in self.apex_board],
            'stamp': self.get_clock().now().nanoseconds}
        try:
            with open(self.output_path, 'w') as f:
                for k, v in data.items():
                    f.write(f"{k}: {v}\n")
            with open(self.output_path.replace('.yaml', '.json'), 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            self.get_logger().warn(f"save failed: {e}")

    def _publish_debug(self, bgr, pose, p_cam, n, reproj, got_radar=False):
        if self.dbg_pub is None:
            return
        try:
            if pose is not None:
                rvec, tvec = pose
                cv2.drawFrameAxes(bgr, self.K, self.D, rvec, tvec, 0.05)
                pt, _ = cv2.projectPoints(self.apex_board.reshape(1, 3),
                                          rvec, tvec, self.K, self.D)
                u, v = int(pt[0, 0, 0]), int(pt[0, 0, 1])
                col = (0, 255, 0) if got_radar else (0, 165, 255)
                cv2.circle(bgr, (u, v), 8, col, 2)
                cv2.putText(bgr, "apex", (u + 10, v),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
            txt = f"corners {n} reproj {reproj if reproj else 0:.2f}px  captures {len(self.captures)}"
            if self.rms is not None:
                txt += f"  RMS {self.rms*1000:.1f}mm"
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
            node._save(node.X, node.rms, False)
            node.get_logger().info(f"Final extrinsic saved to {node.output_path}")
    finally:
        node.destroy_node(); rclpy.shutdown()


if __name__ == '__main__':
    main()
