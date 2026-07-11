#!/usr/bin/env python3
"""
General ChArUco Extrinsic Tool  —  one script, two roles
========================================================
Calibrates the extrinsic between two cameras (A=reference/parent, B=child)
that both see ONE shared ChArUco board simultaneously:
    T_A_B = T_A_board @ inv(T_B_board)        (board cancels)
X = T_A_B maps a point in B into A:  p_A = X p_B.

ROLE is set by `mode`:
  mode:=calibrator  -> detect THIS camera locally (as A), RECEIVE the other
                       camera's board pose over a topic (B), compose, pool,
                       converge, LATCH, save + broadcast the extrinsic.
  mode:=detector    -> detect THIS camera locally, PUBLISH its board pose
                       (T_cam_board) on a topic for a remote calibrator.

So: run mode:=detector on camera B's host (sends pose, never the image),
and mode:=calibrator on camera A's host (detects A, receives B's pose).
The SAME script, same board params, on both machines.

(For a fully-local pair you can still calibrate by running one detector and
one calibrator on the same host, or run two detectors into one calibrator.)
"""
import json, threading
import numpy as np
from scipy.spatial.transform import Rotation as Rot
import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
import message_filters
import tf2_ros
from tf2_ros import Buffer, TransformListener
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import TransformStamped, PoseWithCovarianceStamped
from cv_bridge import CvBridge


def transform_to_T(tr):
    t = tr.transform.translation; q = tr.transform.rotation
    T = np.eye(4)
    T[:3, :3] = Rot.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
    T[:3, 3] = [t.x, t.y, t.z]
    return T

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


def average_se3(Xs):
    ts = np.array([X[:3, 3] for X in Xs])
    qs = np.array([Rot.from_matrix(X[:3, :3]).as_quat() for X in Xs])
    qs = qs * np.sign(qs[:, 3:] + 1e-12)
    q = qs.mean(0); q /= np.linalg.norm(q)
    X = np.eye(4); X[:3, :3] = Rot.from_quat(q).as_matrix(); X[:3, 3] = ts.mean(0)
    return X


class GeneralCharuco(Node):
    def __init__(self):
        super().__init__('general_charuco')
        dp = self.declare_parameter
        dp('mode', 'calibrator')                 # 'calibrator' | 'detector'
        # this camera (the one detected locally in BOTH roles)
        dp('image_topic', '/zed/zed_node/left/image_rect_color')
        dp('info_topic',  '/zed/zed_node/left/camera_info')
        # board (must match on both machines)
        dp('squares_x', 9); dp('squares_y', 7)
        dp('square_len', 0.020); dp('marker_len', 0.015)
        dp('dictionary', 'DICT_4X4_50')
        dp('min_corners', 8); dp('max_reproj_px', 1.5)
        # --- detector role ---
        dp('pub_pose_topic', '/cam_b/board_pose')
        # --- calibrator role ---
        dp('recv_pose_topic', '/cam_b/board_pose')   # B's pose comes in here
        dp('parent_frame', 'zed_left_camera_optical_frame')   # this cam = A
        dp('child_frame',  'camera_color_optical_frame')      # the other cam = B
        dp('pool_cap', 400); dp('sync_slop', 0.08)
        dp('conv_window', 15); dp('conv_hold', 20)
        dp('conv_t_std', 0.003); dp('conv_rot_std', 0.2); dp('conv_pool', 60)
        dp('output_path', '')
        # --- pose-in-map recording (option 1): on lock, record B's pose in the
        #     world/ZED map by composing the ZED's live map pose with T_A_B ---
        dp('record_pose_in_map', True)
        dp('world_frame', 'map_zed')              # ZED's map (world origin)
        dp('cam_a_map_frame', 'zed_left_camera_optical_frame')  # A's frame in the world tree
        dp('camera_a_name', 'zed_left')
        dp('camera_b_name', 'realsense_color')
        # --- publish B into map_zed on lock ---
        #   'none'   : just save YAML (default, original behavior)
        #   'fixed'  : static  map_zed -> child_frame      (fixed node, no odom)
        #   'moving' : static  map_zed -> child_map_frame  (bridge; B has own cuVSLAM)
        dp('publish_to_map', 'none')
        dp('child_map_frame', 'map')   # B's own map root (used when publish_to_map=='moving')

        g = lambda n: self.get_parameter(n).value
        self.mode = g('mode')
        self.min_corners = int(g('min_corners')); self.max_reproj = g('max_reproj_px')
        self.dict, self.board, self.det, self.new_api, self.obj = build_board(
            int(g('squares_x')), int(g('squares_y')),
            g('square_len'), g('marker_len'), g('dictionary'))
        self.bridge = CvBridge(); self.K = None; self.D = None

        # camera_info (both roles need this camera's intrinsics)
        self.create_subscription(CameraInfo, g('info_topic'), self._info, qos_profile_sensor_data)

        if self.mode == 'detector':
            self.pub = self.create_publisher(
                PoseWithCovarianceStamped, g('pub_pose_topic'), 10)
            self.create_subscription(Image, g('image_topic'),
                                     self._detector_image, qos_profile_sensor_data)
            self.get_logger().info(
                f"[DETECTOR] OpenCV {cv2.__version__}, {'new' if self.new_api else 'old'} API\n"
                f"  detect {g('image_topic')} -> publish pose {g('pub_pose_topic')}\n"
                f"  board {int(g('squares_x'))}x{int(g('squares_y'))} "
                f"sq {g('square_len')} mk {g('marker_len')} {g('dictionary')}")
            return

        # ---- calibrator role ----
        self.parent_frame, self.child_frame = g('parent_frame'), g('child_frame')
        self.pool_cap = int(g('pool_cap'))
        self.conv_window = int(g('conv_window')); self.conv_hold = int(g('conv_hold'))
        self.conv_t_std = g('conv_t_std'); self.conv_rot_std = g('conv_rot_std')
        self.conv_pool = int(g('conv_pool'))
        op = g('output_path')
        self.output_path = op if op else f"extrinsic_{self.parent_frame}__{self.child_frame}.yaml"
        self.record_pose_in_map = bool(g('record_pose_in_map'))
        self.world_frame = g('world_frame')
        self.cam_a_map_frame = g('cam_a_map_frame')
        self.camera_a_name = g('camera_a_name')
        self.camera_b_name = g('camera_b_name')
        self.publish_to_map = g('publish_to_map')
        self.child_map_frame = g('child_map_frame')
        self.pool = []; self.est_buf = []; self.stable_count = 0
        self.locked_X = None; self.X = None

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        img = message_filters.Subscriber(self, Image, g('image_topic'),
                                         qos_profile=qos_profile_sensor_data)
        bp = message_filters.Subscriber(self, PoseWithCovarianceStamped, g('recv_pose_topic'))
        self.sync = message_filters.ApproximateTimeSynchronizer([img, bp], 20, g('sync_slop'))
        self.sync.registerCallback(self._calibrator_pair)
        self.tf_static = tf2_ros.StaticTransformBroadcaster(self)
        self.tf_static_map = tf2_ros.StaticTransformBroadcaster(self)
        self.get_logger().info(
            f"[CALIBRATOR] OpenCV {cv2.__version__}, {'new' if self.new_api else 'old'} API\n"
            f"  A(parent,local) = {self.parent_frame}  [{g('image_topic')}]\n"
            f"  B(child,pose)   = {self.child_frame}  [{g('recv_pose_topic')}]\n"
            f"  board {int(g('squares_x'))}x{int(g('squares_y'))} "
            f"sq {g('square_len')} mk {g('marker_len')} {g('dictionary')}\n"
            f"  output -> {self.output_path}\n"
            f"  ** Hold ONE board STILL, visible to BOTH cameras. **")

    def _info(self, msg):
        if self.K is None:
            self.K = np.array(msg.k).reshape(3, 3)
            self.D = np.array(msg.d) if len(msg.d) else np.zeros(5)
            self.get_logger().info(f"intrinsics locked ({msg.width}x{msg.height})")

    # ---- shared detection ----
    def _board_pose(self, gray):
        if self.new_api:
            cc, cid, _, _ = self.det.detectBoard(gray)
        else:
            mc, mids, _ = cv2.aruco.detectMarkers(gray, self.dict)
            if mids is None or len(mids) == 0:
                return None, None, 0
            _, cc, cid = cv2.aruco.interpolateCornersCharuco(mc, mids, gray, self.board)
        n = 0 if cid is None else len(cid)
        if n < self.min_corners:
            return None, None, n
        objp = self.obj[cid.flatten()]
        ok, rvec, tvec = cv2.solvePnP(objp, cc, self.K, self.D, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            return None, None, n
        proj, _ = cv2.projectPoints(objp, rvec, tvec, self.K, self.D)
        reproj = float(cv2.norm(cc, proj, cv2.NORM_L2) / len(proj))
        T = np.eye(4); T[:3, :3] = cv2.Rodrigues(rvec)[0]; T[:3, 3] = tvec[:, 0]
        return T, reproj, n

    # ---- DETECTOR role ----
    def _detector_image(self, msg):
        if self.K is None:
            return
        try:
            gray = cv2.cvtColor(self.bridge.imgmsg_to_cv2(msg, 'bgr8'), cv2.COLOR_BGR2GRAY)
        except Exception as e:
            self.get_logger().warn(f"cv_bridge: {e}"); return
        T, reproj, n = self._board_pose(gray)
        if T is None or reproj > self.max_reproj:
            self.get_logger().info(f"no good board (n={n}, reproj={reproj})",
                                   throttle_duration_sec=2.0)
            return
        q = Rot.from_matrix(T[:3, :3]).as_quat()
        out = PoseWithCovarianceStamped()
        out.header = msg.header
        out.pose.pose.position.x = float(T[0, 3]); out.pose.pose.position.y = float(T[1, 3])
        out.pose.pose.position.z = float(T[2, 3])
        (out.pose.pose.orientation.x, out.pose.pose.orientation.y,
         out.pose.pose.orientation.z, out.pose.pose.orientation.w) = map(float, q)
        cov = [0.0] * 36; cov[0] = reproj; cov[7] = float(n); out.pose.covariance = cov
        self.pub.publish(out)
        self.get_logger().info(f"pose sent: reproj {reproj:.2f}px, {n} corners",
                               throttle_duration_sec=2.0)

    # ---- CALIBRATOR role ----
    def _pose_msg_to_T(self, msg):
        p = msg.pose.pose.position; o = msg.pose.pose.orientation
        T = np.eye(4); T[:3, :3] = Rot.from_quat([o.x, o.y, o.z, o.w]).as_matrix()
        T[:3, 3] = [p.x, p.y, p.z]
        return T, msg.pose.covariance[0], int(msg.pose.covariance[7])

    def _calibrator_pair(self, a_img, b_pose):
        if self.K is None:
            return
        if self.locked_X is not None:
            self._broadcast(self.locked_X); return
        try:
            gA = cv2.cvtColor(self.bridge.imgmsg_to_cv2(a_img, 'bgr8'), cv2.COLOR_BGR2GRAY)
        except Exception as e:
            self.get_logger().warn(f"cv_bridge: {e}"); return
        TaB, rA, nA = self._board_pose(gA)
        TbB, rB, nB = self._pose_msg_to_T(b_pose)
        if TaB is None or TbB is None:
            self.get_logger().info(f"board not in both (A:{nA} B:{nB})", throttle_duration_sec=2.0)
            return
        if rA > self.max_reproj or rB > self.max_reproj:
            self.get_logger().info(f"reproj high A={rA:.2f} B={rB:.2f}", throttle_duration_sec=2.0)
            return

        X = TaB @ np.linalg.inv(TbB)
        self.pool.append(X)
        if len(self.pool) > self.pool_cap:
            self.pool = self.pool[-self.pool_cap:]
        Xbar = average_se3(self.pool)

        self.est_buf.append((Xbar[:3, 3].copy(), Rot.from_matrix(Xbar[:3, :3]).as_rotvec()))
        if len(self.est_buf) > self.conv_window:
            self.est_buf.pop(0)
        t_std = rot_std = float('inf')
        if len(self.est_buf) >= self.conv_window:
            ts = np.array([e[0] for e in self.est_buf]); rs = np.array([e[1] for e in self.est_buf])
            t_std = float(np.linalg.norm(ts.std(0))); rot_std = float(np.degrees(np.linalg.norm(rs.std(0))))
        stable = (t_std < self.conv_t_std and rot_std < self.conv_rot_std and len(self.pool) > self.conv_pool)
        self.stable_count = self.stable_count + 1 if stable else 0

        self.X = Xbar
        self._report(Xbar, rA, rB, nA, nB, len(self.pool), t_std, rot_std)
        if self.stable_count >= self.conv_hold:
            self.locked_X = Xbar.copy()
            pose_in_map = self._compute_pose_in_map(self.locked_X)
            self.get_logger().info(
                f"\n=== CONVERGED — LOCKED ===\n  stable {self.conv_hold} frames "
                f"(t-std {t_std*1000:.2f} mm, rot-std {rot_std:.3f} deg). "
                f"Saved to {self.output_path}. Ctrl-C to exit.")
            self._broadcast(self.locked_X)
            self._save(self.locked_X, rA, rB, 'locked', pose_in_map)
            self._publish_to_map(pose_in_map)
            return
        self._broadcast(Xbar); self._save(Xbar, rA, rB, 'live', None)

    def _compute_pose_in_map(self, X):
        """Record B's pose in the world frame (option 1):
           T_world_B = T_world_Aoptical(t_lock) · T_A_B.
           For RealSense this is its initial pose in map_zed; for a fixed node
           it is the node's (permanent) pose in map_zed. Returns 4x4 or None."""
        if not self.record_pose_in_map:
            return None
        try:
            tr = self.tf_buffer.lookup_transform(
                self.world_frame, self.cam_a_map_frame, rclpy.time.Time())
        except Exception as e:
            self.get_logger().warn(
                f"pose-in-map: TF {self.world_frame}->{self.cam_a_map_frame} "
                f"unavailable ({e}); skipping pose_in_map record")
            return None
        T_world_A = transform_to_T(tr)
        return T_world_A @ X            # T_world_B

    def _publish_to_map(self, pose_in_map):
        """On lock, place B into the world frame (map_zed) — fixed nodes only.
           fixed  : publish permanent static  world_frame -> child_frame.
           moving : record-only. The RealSense's pose in map_zed is saved to the
                    YAML (pose_in_map) as the 'init to map once' data; NOTHING is
                    published to TF, so the RealSense keeps riding its own cuVSLAM
                    with no double-parent. Use the YAML to seed it if/when needed."""
        if self.publish_to_map == 'fixed' and pose_in_map is not None:
            self._send_static(self.tf_static_map, self.world_frame,
                              self.child_frame, pose_in_map)
            self.get_logger().info(
                f"published FIXED static {self.world_frame} -> {self.child_frame} "
                f"(permanent placement)")
        elif self.publish_to_map == 'moving':
            self.get_logger().info(
                f"moving node: pose in {self.world_frame} recorded to YAML "
                f"(pose_in_map); nothing published to TF — {self.child_frame} "
                f"continues on its own odometry.")

    def _send_static(self, bc, parent, child, T):
        m = TransformStamped()
        m.header.stamp = self.get_clock().now().to_msg()
        m.header.frame_id = parent; m.child_frame_id = child
        m.transform.translation.x = float(T[0, 3]); m.transform.translation.y = float(T[1, 3])
        m.transform.translation.z = float(T[2, 3])
        q = Rot.from_matrix(T[:3, :3]).as_quat()
        (m.transform.rotation.x, m.transform.rotation.y,
         m.transform.rotation.z, m.transform.rotation.w) = map(float, q)
        bc.sendTransform(m)

    def _report(self, X, rA, rB, nA, nB, pool, t_std, rot_std):
        t = X[:3, 3]; q = Rot.from_matrix(X[:3, :3]).as_quat()
        rpy = Rot.from_matrix(X[:3, :3]).as_euler('xyz', degrees=True)
        conv = ""
        if np.isfinite(t_std):
            conv = (f"  stability: t-std {t_std*1000:.2f} mm, rot-std {rot_std:.3f} deg "
                    f"[stable {self.stable_count}/{self.conv_hold}]\n")
        self.get_logger().info(
            f"\n--- T_{self.parent_frame}_{self.child_frame} ---\n"
            f"  xyz (m) : {t[0]:+.4f} {t[1]:+.4f} {t[2]:+.4f}\n"
            f"  quat    : {q[0]:+.4f} {q[1]:+.4f} {q[2]:+.4f} {q[3]:+.4f}\n"
            f"  rpy(deg): {rpy[0]:+.2f} {rpy[1]:+.2f} {rpy[2]:+.2f}\n{conv}"
            f"  reproj  : A={rA:.2f}px (n={nA})  B={rB:.2f}px (n={nB})\n"
            f"  pool    : {pool}")

    def _broadcast(self, X):
        # When placing B into map_zed (fixed/moving), the map edge owns B's
        # placement. Publishing the optical->optical edge too would DOUBLE-PARENT
        # B's optical frame (it already has a parent in B's own tree) and lock it
        # to A's tree. So only publish optical->optical in 'none' mode.
        if self.publish_to_map != 'none':
            return
        m = TransformStamped()
        m.header.stamp = self.get_clock().now().to_msg()
        m.header.frame_id = self.parent_frame; m.child_frame_id = self.child_frame
        m.transform.translation.x = float(X[0, 3]); m.transform.translation.y = float(X[1, 3])
        m.transform.translation.z = float(X[2, 3])
        q = Rot.from_matrix(X[:3, :3]).as_quat()
        (m.transform.rotation.x, m.transform.rotation.y,
         m.transform.rotation.z, m.transform.rotation.w) = map(float, q)
        self.tf_static.sendTransform(m)

    def _save(self, X, rA, rB, stage, pose_in_map=None):
        q = Rot.from_matrix(X[:3, :3]).as_quat()
        rpy = Rot.from_matrix(X[:3, :3]).as_euler('xyz', degrees=True)
        data = {'parent_frame': self.parent_frame, 'child_frame': self.child_frame,
                'camera_a_name': self.camera_a_name, 'camera_b_name': self.camera_b_name,
                'translation': [float(v) for v in X[:3, 3]],
                'quaternion_xyzw': [float(v) for v in q], 'rpy_deg': [float(v) for v in rpy],
                'reproj_a_px': rA, 'reproj_b_px': rB, 'stage': stage,
                'stamp': self.get_clock().now().nanoseconds}
        if pose_in_map is not None:
            pq = Rot.from_matrix(pose_in_map[:3, :3]).as_quat()
            prpy = Rot.from_matrix(pose_in_map[:3, :3]).as_euler('xyz', degrees=True)
            data['world_frame'] = self.world_frame
            data['pose_in_map_translation'] = [float(v) for v in pose_in_map[:3, 3]]
            data['pose_in_map_quaternion_xyzw'] = [float(v) for v in pq]
            data['pose_in_map_rpy_deg'] = [float(v) for v in prpy]
        try:
            with open(self.output_path, 'w') as f:
                for k, v in data.items():
                    f.write(f"{k}: {v}\n")
            with open(self.output_path.replace('.yaml', '.json'), 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            self.get_logger().warn(f"save failed: {e}")


def main():
    rclpy.init(); node = GeneralCharuco()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        if getattr(node, 'X', None) is not None:
            node.get_logger().info(f"Final extrinsic in {node.output_path}")
    finally:
        node.destroy_node(); rclpy.shutdown()


if __name__ == '__main__':
    main()
