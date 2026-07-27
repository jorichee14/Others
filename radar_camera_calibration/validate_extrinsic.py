#!/usr/bin/env python3
"""
Validate a FIXED radar↔camera extrinsic on FRESH points (held-out; nothing is
solved or fit here). Same pipeline as the calibrator — ChArUco board pose for the
camera apex, cluster/SNR reflector pick for the radar, in-node rectification for a
RAW Arducam feed, and the same debug overlay — but instead of estimating T, it
APPLIES the T you pass in and reports the per-axis error:

    p_pred = R · p_radar + t        (radar point pushed into the camera frame)
    err    = p_pred − p_cam         (vs the ChArUco-derived apex)  → (ex, ey, ez)

Reports, in the CAMERA optical frame (X=right, Y=down, Z=forward):
    signed bias (mean err), RMS, and std per axis, plus 3-D and reprojection px.

Run (defaults are the r4 infra↔Arducam transform):

    ros2 run radar_node validate_extrinsic --ros-args \
      -p image_topic:=/arducam/image_raw -p info_topic:=/arducam/camera_info \
      -p radar_topic:=/radar1/radar/points_all -p pc_field_snr:=intensity \
      -p squares_x:=4 -p squares_y:=4 -p square_len:=0.12 -p marker_len:=0.09 \
      -p reflector_offset_x:=0.256 -p reflector_offset_y:=0.539 -p reflector_offset_z:=-0.020 \
      -p radar_range_scale:=0.963 -p rectify_image:=true \
      -p ext_t_xyz:="[-0.063211, 0.023014, -0.024619]" \
      -p ext_quat_xyzw:="[0.594510, 0.422341, 0.436492, -0.526935]" \
      -p min_captures:=20 -p show_window:=true

Controls (std_msgs/Empty):  ~/capture  (force one)   ~/report  (print now)   ~/reset
Auto-captures when the board+reflector are steady and you've moved ≥ min_baseline
from the last capture. Move around the shared FoV: near/far, left/right, HIGH/LOW.
"""
import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, CompressedImage
from std_msgs.msg import Empty
from cv_bridge import CvBridge
import message_filters
from scipy.spatial.transform import Rotation as Rot
try:
    from sensor_msgs_py import point_cloud2 as pc2
    _HAVE_PC2 = True
except Exception:
    _HAVE_PC2 = False

_DICT = {
    'DICT_4X4_50': cv2.aruco.DICT_4X4_50, 'DICT_4X4_100': cv2.aruco.DICT_4X4_100,
    'DICT_4X4_250': cv2.aruco.DICT_4X4_250, 'DICT_5X5_50': cv2.aruco.DICT_5X5_50,
    'DICT_5X5_100': cv2.aruco.DICT_5X5_100, 'DICT_6X6_50': cv2.aruco.DICT_6X6_50,
    'DICT_6X6_250': cv2.aruco.DICT_6X6_250,
}


def build_board(sx, sy, square, marker, dict_name):
    dname = _DICT.get(dict_name, cv2.aruco.DICT_4X4_50)
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


def cluster_points(xs, eps, min_size):
    """Greedy connected-components: points within eps join one cluster."""
    n = len(xs)
    if n == 0:
        return []
    D = np.sqrt(((xs[:, None, :] - xs[None, :, :]) ** 2).sum(2))
    seen = np.zeros(n, bool)
    out = []
    for i in range(n):
        if seen[i]:
            continue
        stack, comp = [i], []
        seen[i] = True
        while stack:
            j = stack.pop()
            comp.append(j)
            nb = np.where((D[j] <= eps) & (~seen))[0]
            for k in nb:
                seen[k] = True
                stack.append(int(k))
        if len(comp) >= min_size:
            out.append(np.array(comp))
    return out


class Validate(Node):
    def __init__(self):
        super().__init__('validate_extrinsic')
        g = lambda n, v: self.declare_parameter(n, v).value

        # ── topics / frames ──
        self.image_topic = g('image_topic', '/arducam/image_raw')
        self.info_topic = g('info_topic', '/arducam/camera_info')
        self.radar_topic = g('radar_topic', '/radar1/radar/points_all')
        self.fx_, self.fy_, self.fz_ = g('pc_field_x', 'x'), g('pc_field_y', 'y'), g('pc_field_z', 'z')
        self.fsnr = g('pc_field_snr', 'intensity')

        # ── board ──
        self.dict, self.board, self.det, self.new_api, self.obj = build_board(
            int(g('squares_x', 4)), int(g('squares_y', 4)),
            g('square_len', 0.12), g('marker_len', 0.09), g('dictionary', 'DICT_4X4_50'))
        self.min_corners = int(g('min_corners', 4))
        self.max_reproj = g('max_reproj_px', 1.5)
        self.apex = np.array([g('reflector_offset_x', 0.256),
                              g('reflector_offset_y', 0.539),
                              g('reflector_offset_z', -0.020)], float)

        # ── the FIXED extrinsic under test (T_cam_radar) ──
        self.R = Rot.from_quat(np.array(g('ext_quat_xyzw',
                        [0.594510, 0.422341, 0.436492, -0.526935]), float)).as_matrix()
        self.t = np.array(g('ext_t_xyz', [-0.063211, 0.023014, -0.024619]), float)

        # ── radar ingest + gating (same knobs as the calibrator) ──
        self.range_scale = g('radar_range_scale', 0.963)
        self.range_bias = g('radar_range_bias_m', 0.0)
        self.min_range, self.max_range = g('min_range', 0.5), g('max_range', 2.5)
        self.range_gate_margin = g('range_gate_margin_m', 0.5)
        self.gate_radius = g('gate_radius', 0.5)
        self.min_snr = g('min_snr', 100.0)
        self.select_by = g('select_by', 'cluster')
        self.cluster_eps = g('cluster_eps', 0.20)
        self.min_cluster_size = int(g('min_cluster_size', 1))
        self.cluster_apex_radius = g('cluster_apex_radius', 0.40)

        # ── capture gating ──
        self.min_captures = int(g('min_captures', 20))
        self.stable_window = int(g('stable_window', 12))
        self.stable_std = g('stable_std', 0.02)
        self.stable_std_radar = g('stable_std_radar', 0.10)
        self.min_baseline = g('min_baseline', 0.12)
        self.sync_slop = g('sync_slop', 0.06)

        # ── rectification (RAW Arducam) ──
        self.rectify = bool(g('rectify_image', True))
        self.rectify_alpha = g('rectify_alpha', 0.0)
        self.map1 = self.map2 = None

        # ── output ──
        self.show_window = bool(g('show_window', True))
        self.dbg_topic = g('debug_image_topic', '/validate_extrinsic/debug_image')

        self.bridge = CvBridge()
        self.K = self.D = None
        self.win = []                # rolling (p_cam, p_radar)
        self.caps = []               # accepted (p_cam, p_radar, p_pred, err, reproj)
        self.last_cap = None
        self._last_dbg = None
        self._reported = False

        self.create_subscription(CameraInfo, self.info_topic, self._info, qos_profile_sensor_data)
        img = message_filters.Subscriber(self, Image, self.image_topic, qos_profile=qos_profile_sensor_data)
        rad = message_filters.Subscriber(self, PointCloud2, self.radar_topic, qos_profile=qos_profile_sensor_data)
        self.sync = message_filters.ApproximateTimeSynchronizer([img, rad], 30, self.sync_slop)
        self.sync.registerCallback(self._pair)
        self.create_subscription(Empty, '~/capture', lambda _: self._capture(force=True), 1)
        self.create_subscription(Empty, '~/report', lambda _: self._report(), 1)
        self.create_subscription(Empty, '~/reset', lambda _: self._reset(), 1)

        self.dbg_pub = self.create_publisher(Image, self.dbg_topic, 1)
        self.dbg_pub_c = self.create_publisher(CompressedImage, self.dbg_topic + '/compressed', 1)
        if self.show_window:
            self.create_timer(0.05, self._gui)

        rpy = np.round(Rot.from_matrix(self.R).as_euler('xyz', degrees=True), 2)
        self.get_logger().info(
            f"[validate_extrinsic] OpenCV {cv2.__version__}, {'new' if self.new_api else 'old'} aruco\n"
            f"  camera {self.image_topic}  radar {self.radar_topic}\n"
            f"  TESTING  t(cm)={np.round(100*self.t,2).tolist()}  rpy(deg)={rpy.tolist()}\n"
            f"  apex_in_board(m)={self.apex.tolist()}  range_scale={self.range_scale}\n"
            f"  need {self.min_captures} fresh captures — move the rig around the FoV (incl. HIGH/LOW).")

    # ── intrinsics + optional in-node rectify ──
    def _info(self, msg):
        if self.K is not None:
            return
        K = np.array(msg.k).reshape(3, 3)
        D = np.array(msg.d) if len(msg.d) else np.zeros(5)
        w, h = int(msg.width), int(msg.height)
        if self.rectify and w and h and np.any(np.abs(D) > 1e-9):
            newK, _ = cv2.getOptimalNewCameraMatrix(K, D, (w, h), self.rectify_alpha, (w, h))
            self.map1, self.map2 = cv2.initUndistortRectifyMap(K, D, None, newK, (w, h), cv2.CV_16SC2)
            self.K, self.D = newK, np.zeros(5)
            self.get_logger().info(f"intrinsics locked ({w}x{h}) — rectifying in-node")
        else:
            self.K, self.D = K, D
            self.get_logger().info(f"intrinsics locked ({w}x{h})")

    def _read_radar(self, msg):
        if not _HAVE_PC2:
            return None, None
        names = [f.name for f in msg.fields]
        want = [self.fx_, self.fy_, self.fz_]
        has = self.fsnr in names
        if has:
            want.append(self.fsnr)
        arr = list(pc2.read_points(msg, field_names=want, skip_nans=True))
        if not arr:
            return None, None
        a = np.array([tuple(x) for x in arr], float)
        xyz = a[:, :3] * float(self.range_scale)
        if self.range_bias != 0.0:
            r = np.linalg.norm(xyz, axis=1, keepdims=True)
            xyz = xyz - self.range_bias * (xyz / np.where(r < 1e-6, 1e-6, r))
        snr = a[:, 3] if has else np.ones(len(a))
        return xyz, snr

    def _apex_in_camera(self, gray):
        if self.new_api:
            cc, cid, _, _ = self.det.detectBoard(gray)
        else:
            mc, mids, _ = cv2.aruco.detectMarkers(gray, self.dict)
            if mids is None or len(mids) == 0:
                return None, None, None
            _, cc, cid = cv2.aruco.interpolateCornersCharuco(mc, mids, gray, self.board)
        if cid is None or len(cid) < self.min_corners:
            return None, None, None
        objp = self.obj[cid.flatten()]
        ok, rvec, tvec = cv2.solvePnP(objp, cc, self.K, self.D, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            return None, None, None
        proj, _ = cv2.projectPoints(objp, rvec, tvec, self.K, self.D)
        if float(cv2.norm(cc, proj, cv2.NORM_L2) / len(proj)) > self.max_reproj:
            return None, None, None
        R = cv2.Rodrigues(rvec)[0]
        p_cam = R @ self.apex + tvec[:, 0]
        return p_cam, (rvec, tvec), R

    def _select_radar(self, xyz, snr, predicted, cam_range):
        if xyz is None or len(xyz) == 0:
            return None
        rng = np.linalg.norm(xyz, axis=1)
        keep = (rng >= self.min_range) & (rng <= self.max_range)
        if cam_range is not None and self.range_gate_margin > 0:
            keep &= (np.abs(rng - cam_range) <= self.range_gate_margin)
        if predicted is not None:
            tight = keep & (np.linalg.norm(xyz - predicted, axis=1) <= self.gate_radius)
            if tight.any():
                keep = tight
        if not keep.any():
            return None
        xg, sg = xyz[keep], snr[keep]
        # cluster around the predicted apex → SNR-weighted centroid; else argmax SNR
        sel = None
        if self.select_by == 'cluster':
            xs, ss = xg, sg
            if predicted is not None and self.cluster_apex_radius > 0:
                near = np.linalg.norm(xg - predicted, axis=1) <= self.cluster_apex_radius
                if near.any():
                    xs, ss = xg[near], sg[near]
            cl = cluster_points(xs, self.cluster_eps, self.min_cluster_size)
            if cl:
                best = (min(cl, key=lambda c: np.linalg.norm(xs[c].mean(0) - predicted))
                        if predicted is not None else max(cl, key=lambda c: np.nanmax(ss[c])))
                w = np.clip(np.nan_to_num(ss[best]), 1e-6, None)
                sel = ((xs[best] * (w / w.sum())[:, None]).sum(0), float(np.nanmax(ss[best])))
        if sel is None:
            i = int(np.argmax(sg))
            sel = (xg[i], float(sg[i]))
        p, s = sel
        if self.min_snr > 0 and s < self.min_snr:
            return None
        return np.asarray(p, float)

    def _pair(self, img_msg, radar_msg):
        if self.K is None:
            return
        bgr = self.bridge.imgmsg_to_cv2(img_msg, 'bgr8')
        if self.map1 is not None:
            bgr = cv2.remap(bgr, self.map1, self.map2, cv2.INTER_LINEAR)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        p_cam, pose, _ = self._apex_in_camera(gray)
        p_pred = err = None
        if p_cam is not None:
            # where the extrinsic predicts the apex sits in the RADAR frame → gate the
            # radar cloud around it (proximity), plus the rotation-invariant range gate.
            pred_radar = self.R.T @ (p_cam - self.t)
            xyz, snr = self._read_radar(radar_msg)
            p_radar = self._select_radar(xyz, snr, pred_radar, float(np.linalg.norm(p_cam)))
            if p_radar is not None:
                p_pred = self.R @ p_radar + self.t
                err = p_pred - p_cam
                self.win.append((p_cam, p_radar, p_pred, err))
                if len(self.win) > self.stable_window:
                    self.win.pop(0)
                self._maybe_capture()
        self._draw(bgr, pose, p_cam, p_pred, err)

    def _maybe_capture(self):
        if len(self.win) < self.stable_window:
            return
        pc = np.array([w[0] for w in self.win])
        pr = np.array([w[1] for w in self.win])
        if pc.std(0).max() > self.stable_std or pr.std(0).max() > self.stable_std_radar:
            return
        p_cam = pc.mean(0)
        if self.last_cap is not None and np.linalg.norm(p_cam - self.last_cap) < self.min_baseline:
            return
        self._capture()

    def _capture(self, force=False):
        if not self.win:
            return
        pc = np.array([w[0] for w in self.win]); pr = np.array([w[1] for w in self.win])
        p_cam, p_radar = pc.mean(0), pr.mean(0)
        p_pred = self.R @ p_radar + self.t
        err = p_pred - p_cam
        proj_c, _ = cv2.projectPoints(p_cam.reshape(1, 3), np.zeros(3), np.zeros(3), self.K, self.D)
        proj_p, _ = cv2.projectPoints(p_pred.reshape(1, 3), np.zeros(3), np.zeros(3), self.K, self.D)
        reproj = float(np.linalg.norm(proj_c[0, 0] - proj_p[0, 0]))
        self.caps.append((p_cam, p_radar, p_pred, err, reproj))
        self.last_cap = p_cam
        self.get_logger().info(
            f"*** CAPTURE #{len(self.caps)}  err(mm) X {1000*err[0]:+.0f} Y {1000*err[1]:+.0f} "
            f"Z {1000*err[2]:+.0f}  reproj {reproj:.0f}px ***")
        if len(self.caps) >= self.min_captures and not self._reported:
            self._reported = True
            self._report()

    def _report(self):
        if len(self.caps) < 3:
            self.get_logger().warn(f"only {len(self.caps)} captures — collect more.")
            return
        E = np.array([c[3] for c in self.caps]) * 1000.0        # mm, camera frame
        rp = np.array([c[4] for c in self.caps])
        d3 = np.linalg.norm(E, axis=1)
        bias, rms, std = E.mean(0), np.sqrt((E ** 2).mean(0)), E.std(0)
        self.get_logger().info(
            "\n================= VALIDATION (held-out, T fixed) =================\n"
            f"  captures        : {len(self.caps)}\n"
            f"  frame           : camera optical  X=right  Y=down  Z=forward\n"
            f"  signed bias mm  : X {bias[0]:+7.1f}  Y {bias[1]:+7.1f}  Z {bias[2]:+7.1f}\n"
            f"  RMS error   mm  : X {rms[0]:7.1f}  Y {rms[1]:7.1f}  Z {rms[2]:7.1f}\n"
            f"  std (spread)mm  : X {std[0]:7.1f}  Y {std[1]:7.1f}  Z {std[2]:7.1f}\n"
            f"  3-D error   mm  : mean {d3.mean():.1f}   RMS {np.sqrt((d3**2).mean()):.1f}   "
            f"median {np.median(d3):.1f}\n"
            f"  reproj      px  : mean {rp.mean():.1f}   median {np.median(rp):.1f}\n"
            "  read: bias = systematic offset (should be ~0); RMS on Y = radar's weak\n"
            "        elevation (large is expected); X/Z small = good horizontal + range.\n"
            "==================================================================")

    def _reset(self):
        self.caps.clear(); self.win.clear(); self.last_cap = None; self._reported = False
        self.get_logger().info("reset — captures cleared.")

    def _draw(self, bgr, pose, p_cam, p_pred, err):
        if pose is not None:
            cv2.drawFrameAxes(bgr, self.K, self.D, pose[0], pose[1], 0.05)
        if p_cam is not None:
            pc, _ = cv2.projectPoints(p_cam.reshape(1, 3), np.zeros(3), np.zeros(3), self.K, self.D)
            u, v = int(pc[0, 0, 0]), int(pc[0, 0, 1])
            cv2.circle(bgr, (u, v), 7, (0, 255, 0), 2)                 # green = camera apex
            cv2.putText(bgr, "cam apex", (u + 8, v), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        if p_pred is not None:
            pp, _ = cv2.projectPoints(p_pred.reshape(1, 3), np.zeros(3), np.zeros(3), self.K, self.D)
            u, v = int(pp[0, 0, 0]), int(pp[0, 0, 1])
            cv2.circle(bgr, (u, v), 6, (255, 0, 255), 2)               # magenta = radar→cam
            cv2.putText(bgr, "radar", (u + 8, v + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
        cv2.putText(bgr, f"captures {len(self.caps)}/{self.min_captures}", (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        if err is not None:
            cv2.putText(bgr, f"err mm X{1000*err[0]:+.0f} Y{1000*err[1]:+.0f} Z{1000*err[2]:+.0f}",
                        (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        self._last_dbg = bgr
        try:
            self.dbg_pub.publish(self.bridge.cv2_to_imgmsg(bgr, 'bgr8'))
            ok, buf = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, 40])
            if ok:
                m = CompressedImage(); m.format = 'jpeg'; m.data = buf.tobytes()
                self.dbg_pub_c.publish(m)
        except Exception:
            pass

    def _gui(self):
        if self._last_dbg is not None:
            try:
                cv2.imshow('validate_extrinsic — green=cam apex  magenta=radar', self._last_dbg)
                cv2.waitKey(1)
            except Exception:
                self.show_window = False


def main():
    rclpy.init()
    node = Validate()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node._report()
    node.destroy_node(); rclpy.shutdown()


if __name__ == '__main__':
    main()
