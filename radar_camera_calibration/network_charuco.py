#!/usr/bin/env python3
"""
Network ChArUco extrinsics  —  general_charuco.py for one ZED and MANY cameras
==============================================================================
Camera A = the ZED (ROS 2: rectified left image + camera_info; its pose in the ZED map from TF).
Cameras B = any number of RTSP cameras: the 16 ED305 cameras, the robot's other cameras streamed
with ros_image_to_rtsp.py, ... Their intrinsics come from the calibration dashboard
(ed305_cameras_calibration/results/intrinsics/cam_N.json), the board from its config/boards.yaml.

Hold ONE board STILL where the ZED and a camera B both see it. Every still moment gives a sample
    T_world_B = T_world_A · T_A_board · inv(T_B_board)        (world = the ZED map, map_zed)
Samples are pooled and averaged per camera like general_charuco.py; when stable (and taken from at
least 3 board placements) the camera LOCKS and its pose in the map is saved (and published to TF).
Because each sample is placed in the map with the ZED's own pose at that moment, the robot may move
between holds: drive to each camera, hold the board where both see it, move it, hold again.

Without a map (use_map:=false) samples are T_A_B instead: the ZED and B must then not move relative
to each other (e.g. two cameras on the same robot).

  ros2 run ... / python3 network_charuco.py --ros-args \\
      -p cameras:="1,2,3" \\                       # ED305 cameras from config/cameras.yaml
      -p urls:="17=rtsp://192.168.51.94:8554/mobile_2/color"   # and/or any other stream

Output (output_dir, default ./network_extrinsics): cam_N.json / .yaml per camera (same keys as
general_charuco.py's output, plus samples and spread) and network.json with every camera.
Board placements must be STILL; frames with no common timestamp (RTSP) are paired by stillness.
"""
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import rclpy
import tf2_ros
import yaml
from cv_bridge import CvBridge
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import Buffer, TransformListener

HERE = os.path.dirname(os.path.abspath(__file__))
CALIB_DIR = os.path.join(os.path.dirname(HERE), "ed305_cameras_calibration")
sys.path.insert(0, CALIB_DIR)
from calib.common import camera_urls, load_boards, load_intrinsics  # noqa: E402
from calib.network import BoardPose, Pair, Still, pose_dict  # noqa: E402
from calib.streams import Stream  # noqa: E402


def transform_to_T(tr):
    from scipy.spatial.transform import Rotation as Rot
    t, q = tr.transform.translation, tr.transform.rotation
    T = np.eye(4)
    T[:3, :3] = Rot.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
    T[:3, 3] = [t.x, t.y, t.z]
    return T


def nan(v):
    return float("nan") if v is None else v


class Cam:
    """One camera B: stream, intrinsics, board pose, stillness, pooled extrinsic."""

    def __init__(self, n, url, K, D, board, args):
        self.n, self.url, self.K, self.D = n, url, K, D
        self.stream = Stream(n, url)
        self.pose = BoardPose(board, args["min_corners"])
        self.still = Still(args["still_s"], args["still_mm"] / 1000.0, args["still_deg"])
        self.pair = Pair("cam_%d" % n, conv_window=args["conv_window"], conv_hold=args["conv_hold"],
                         conv_t_std=args["conv_t_std_mm"] / 1000.0, conv_rot_std=args["conv_rot_std"],
                         conv_pool=args["conv_pool"], min_holds=args["min_holds"])
        self.T, self.reproj, self.ncorners, self.t, self.last_frame_t = None, None, 0, 0.0, 0.0
        self.last_sample_t, self.saved_lock = 0.0, False

    def update(self, max_reproj):
        """Detect the board in the newest frame (if there is a new one)."""
        frame, t = self.stream.latest()
        if frame is None or t == self.last_frame_t:
            return
        self.last_frame_t = t
        T, reproj, n = self.pose(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), self.K, self.D)
        if T is not None and reproj > max_reproj:
            T = None
        self.T, self.reproj, self.ncorners, self.t = T, reproj, n, time.time()
        self.still.add(self.t, T)


class NetworkCharuco(Node):
    def __init__(self):
        super().__init__("network_charuco")
        dp = self.declare_parameter
        # camera A = the ZED (as general_charuco.py)
        dp("image_topic", "/zed/zed_node/left/image_rect_color")
        dp("info_topic", "/zed/zed_node/left/camera_info")
        dp("world_frame", "map_zed")                           # the ZED map (cuVSLAM)
        dp("cam_a_map_frame", "zed_left_camera_optical_frame")  # the ZED's frame in the TF tree
        dp("use_map", True)                                     # false: pool T_A_B (A and B must not move)
        # cameras B
        dp("cameras", "")       # "1,2,3": ED305 cameras from ed305_cameras_calibration/config/cameras.yaml
        dp("urls", "")          # "17=rtsp://...,18=rtsp://...": any other streams (number=url)
        dp("intrinsics_dir", os.path.join(CALIB_DIR, "results"))   # results/intrinsics/cam_N.json (newest round)
        dp("board", "A")        # board name in ed305_cameras_calibration/config/boards.yaml
        # detection and pairing
        dp("rate", 5.0); dp("min_corners", 8); dp("max_reproj_px", 1.5)
        dp("still_s", 0.6); dp("still_mm", 3.0); dp("still_deg", 0.3)
        # convergence (as general_charuco.py) + at least min_holds board placements
        dp("conv_window", 15); dp("conv_hold", 10); dp("conv_t_std_mm", 3.0); dp("conv_rot_std", 0.2)
        dp("conv_pool", 30); dp("min_holds", 3)
        # output
        dp("output_dir", "network_extrinsics")
        dp("publish_tf", True)                  # on lock: static world_frame -> <child_prefix>N
        dp("child_prefix", "ed305_cam_")
        dp("camera_a_name", "zed_left")
        g = lambda n: self.get_parameter(n).value  # noqa: E731
        self.args = {k: g(k) for k in ["rate", "min_corners", "max_reproj_px", "still_s", "still_mm", "still_deg",
                                       "conv_window", "conv_hold", "conv_t_std_mm", "conv_rot_std", "conv_pool",
                                       "min_holds", "use_map", "world_frame", "cam_a_map_frame", "output_dir",
                                       "publish_tf", "child_prefix", "camera_a_name"]}
        os.makedirs(self.args["output_dir"], exist_ok=True)

        board = load_boards([g("board")])[g("board")].board
        self.poseA = BoardPose(board, self.args["min_corners"])
        self.stillA = Still(self.args["still_s"], self.args["still_mm"] / 1000.0, self.args["still_deg"])
        self.KA = self.DA = None
        self.TA, self.TA_stamp, self.TA_t, self.rA, self.nA = None, None, 0.0, None, 0
        self.last_a = 0.0
        self.bridge = CvBridge()
        self.lock = threading.Lock()

        nums = [int(c) for c in str(g("cameras")).replace(" ", "").split(",") if c]
        urls = camera_urls(nums) if nums else {}
        for item in [u for u in str(g("urls")).split(",") if u.strip()]:
            n, _, u = item.strip().partition("=")
            urls[int(n)] = u
        if not urls:
            raise SystemExit("no cameras B: set cameras:=\"1,2\" and/or urls:=\"17=rtsp://...\"")
        intr = load_intrinsics(g("intrinsics_dir"))
        missing = [n for n in urls if n not in intr]
        if missing:
            raise SystemExit("no intrinsics for camera(s) %s in %s/intrinsics: calibrate them in the dashboard first"
                             % (missing, g("intrinsics_dir")))
        self.cams = {n: Cam(n, u, intr[n][0], intr[n][1], board, self.args) for n, u in sorted(urls.items())}
        self.pool = ThreadPoolExecutor(max_workers=max(2, min(len(self.cams), (os.cpu_count() or 4) - 1)))

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_static = tf2_ros.StaticTransformBroadcaster(self)
        self.static_msgs = []
        self.create_subscription(CameraInfo, g("info_topic"), self._info, qos_profile_sensor_data)
        self.create_subscription(Image, g("image_topic"), self._image, qos_profile_sensor_data)
        self.create_timer(1.0 / self.args["rate"], self._tick)
        self.create_timer(2.0, self._report)
        parent = self.args["world_frame"] if self.args["use_map"] else self.args["cam_a_map_frame"]
        self.get_logger().info(
            "[NETWORK] A = ZED %s; B = cameras %s; poses in %s; board %s; output %s\n"
            "  ** Hold the board STILL where the ZED and a camera both see it; move it, hold again. **"
            % (g("image_topic"), sorted(self.cams), parent, g("board"), self.args["output_dir"]))

    # ---- camera A (the ZED) ----
    def _info(self, msg):
        if self.KA is None:
            self.KA = np.array(msg.k, float).reshape(3, 3)
            self.DA = np.array(msg.d, float) if len(msg.d) else np.zeros(5)
            self.get_logger().info("ZED intrinsics locked (%dx%d)" % (msg.width, msg.height))

    def _image(self, msg):
        now = time.time()
        if self.KA is None or now - self.last_a < 1.0 / self.args["rate"]:
            return
        self.last_a = now
        try:
            gray = cv2.cvtColor(self.bridge.imgmsg_to_cv2(msg, "bgr8"), cv2.COLOR_BGR2GRAY)
        except Exception as e:
            self.get_logger().warn("cv_bridge: %s" % e, throttle_duration_sec=5.0)
            return
        T, r, n = self.poseA(gray, self.KA, self.DA)
        if T is not None and r > self.args["max_reproj_px"]:
            T = None
        with self.lock:
            self.TA, self.TA_stamp, self.TA_t, self.rA, self.nA = T, msg.header.stamp, now, r, n
            self.stillA.add(now, T)

    def _world_A(self, stamp):
        """T_world_A at the ZED image's time (None if the ZED pose is not known for it)."""
        try:
            return transform_to_T(self.tf_buffer.lookup_transform(
                self.args["world_frame"], self.args["cam_a_map_frame"], Time.from_msg(stamp)))
        except Exception:
            return None

    # ---- pairing ----
    def _tick(self):
        todo = [c for c in self.cams.values() if c.pair.locked is None]
        list(self.pool.map(lambda c: c.update(self.args["max_reproj_px"]), todo))
        with self.lock:
            TA, stamp, tA, stillA, holdA = self.TA, self.TA_stamp, self.TA_t, self.stillA.still, self.stillA.hold
        now = time.time()
        if TA is None or not stillA or now - tA > 1.0:
            return
        W = None
        if self.args["use_map"]:
            W = self._world_A(stamp)
            if W is None:
                self.get_logger().warn("no TF %s -> %s at the ZED image time: is the ZED tracking (cuVSLAM)?"
                                       % (self.args["world_frame"], self.args["cam_a_map_frame"]),
                                       throttle_duration_sec=5.0)
                return
        for c in todo:
            if c.T is None or not c.still.still or now - c.t > 1.0 or now - c.last_sample_t < 1.0 / self.args["rate"]:
                continue
            X = TA @ np.linalg.inv(c.T)                   # T_A_B
            if W is not None:
                X = W @ X                                 # T_world_B
            c.last_sample_t = now
            if c.pair.add(X, (holdA, c.still.hold), c.reproj):
                self._on_lock(c)

    # ---- output ----
    def _parent(self):
        return self.args["world_frame"] if self.args["use_map"] else self.args["cam_a_map_frame"]

    def _record(self, c):
        X = c.pair.locked if c.pair.locked is not None else c.pair.X
        if X is None:
            return None
        st = c.pair.status()
        return {"parent_frame": self._parent(), "child_frame": self.args["child_prefix"] + str(c.n),
                "camera_a_name": self.args["camera_a_name"], "camera_b_name": "cam_%d" % c.n, "url": c.url,
                **pose_dict(X), "stage": "locked" if c.pair.locked is not None else "live",
                "reproj_a_px": self.rA, "reproj_b_px": st["reproj_b_px"], "samples": st["samples"],
                "board_placements": st["holds"], "spread_mm": st["spread_mm"], "spread_deg": st["spread_deg"],
                "t_std_mm": st["t_std_mm"], "rot_std_deg": st["rot_std_deg"], "stamp": time.time()}

    def _save(self, c):
        rec = self._record(c)
        if rec is None:
            return
        base = os.path.join(self.args["output_dir"], "cam_%d" % c.n)
        with open(base + ".json", "w") as f:
            json.dump(rec, f, indent=2)
        with open(base + ".yaml", "w") as f:
            yaml.safe_dump(rec, f, sort_keys=False)

    def _on_lock(self, c):
        self._save(c)
        rec = self._record(c)
        self.get_logger().info(
            "\n=== cam %d LOCKED === %d samples from %d board placements, spread %.1f mm / %.2f deg\n"
            "  T_%s_cam%d xyz %s  rpy %s  -> %s/cam_%d.json"
            % (c.n, rec["samples"], rec["board_placements"], rec["spread_mm"], rec["spread_deg"], rec["parent_frame"],
               c.n, np.round(rec["translation"], 3).tolist(), np.round(rec["rpy_deg"], 2).tolist(),
               self.args["output_dir"], c.n))
        if self.args["publish_tf"]:
            from scipy.spatial.transform import Rotation as Rot
            X = c.pair.locked
            m = TransformStamped()
            m.header.stamp = self.get_clock().now().to_msg()
            m.header.frame_id, m.child_frame_id = rec["parent_frame"], rec["child_frame"]
            m.transform.translation.x, m.transform.translation.y, m.transform.translation.z = map(float, X[:3, 3])
            q = Rot.from_matrix(X[:3, :3]).as_quat()
            (m.transform.rotation.x, m.transform.rotation.y, m.transform.rotation.z, m.transform.rotation.w) = map(float, q)
            self.static_msgs.append(m)
            self.tf_static.sendTransform(self.static_msgs)   # all locked cameras (a static broadcaster keeps the last list)

    def _report(self):
        lines = ["ZED: %s" % ("board %d corners, %s" % (self.nA, "STILL" if self.stillA.still else "moving")
                              if self.TA is not None else "no board")]
        for c in self.cams.values():
            st = c.pair.status()
            seen = ("board %d corners %.2f px %s" % (c.ncorners, c.reproj, "STILL" if c.still.still else "moving")
                    if c.T is not None else ("no board" if c.stream.ok else "no stream"))
            lines.append("cam %-3d %-8s samples %3d placements %d  t-std %5.1f mm rot-std %5.2f deg | %s" % (
                c.n, "LOCKED" if st["locked"] else "", st["samples"], st["holds"], nan(st["t_std_mm"]), nan(st["rot_std_deg"]), seen))
            if c.pair.X is not None and c.pair.locked is None:
                self._save(c)
        net = {str(c.n): self._record(c) for c in self.cams.values() if c.pair.X is not None}
        with open(os.path.join(self.args["output_dir"], "network.json"), "w") as f:
            json.dump(net, f, indent=2)
        self.get_logger().info("\n" + "\n".join(lines))

    def stop(self):
        for c in self.cams.values():
            c.stream.stop()
            self._save(c)


def main():
    rclpy.init()
    node = NetworkCharuco()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
