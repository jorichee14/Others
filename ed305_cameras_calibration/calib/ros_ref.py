"""The reference camera (the ZED) read straight from ROS 2, for the dashboard on the robot.

One object that the dashboard uses both as the camera's stream (latest(), like calib.streams.Stream)
and as its map pose (now(), status(), like server.RefPose):
  image       sensor_msgs/Image (bgra8 / bgr8 / rgba8 / rgb8 / mono8), e.g. /zed/zed_node/left/image_rect_color
  intrinsics  sensor_msgs/CameraInfo of that image (K, D, size)
  pose        TF world_frame <- camera_frame at the capture time of the image the dashboard just used
Needs rclpy and tf2_ros (a sourced ROS 2); no cv_bridge.
"""
import threading
import time

import cv2
import numpy as np

from calib.network import Still

_ENC = {"bgra8": (4, cv2.COLOR_BGRA2BGR), "rgba8": (4, cv2.COLOR_RGBA2BGR), "bgr8": (3, None),
        "rgb8": (3, cv2.COLOR_RGB2BGR), "mono8": (1, cv2.COLOR_GRAY2BGR), "8UC1": (1, cv2.COLOR_GRAY2BGR)}


def image_to_bgr(msg):
    """sensor_msgs/Image -> BGR numpy image (copies, drops row padding)."""
    ch, conv = _ENC[msg.encoding]
    a = np.frombuffer(bytes(msg.data), np.uint8).reshape(msg.height, msg.step)[:, :msg.width * ch]
    a = a.reshape(msg.height, msg.width, ch) if ch > 1 else a.reshape(msg.height, msg.width)
    return cv2.cvtColor(a, conv) if conv is not None else a.copy()


def stamp_seconds(st):
    return st.sec + st.nanosec * 1e-9


class RosRef:
    def __init__(self, image_topic, info_topic, world_frame, camera_frame, still_s=0.6):
        import rclpy
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.node import Node
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import CameraInfo, Image
        from tf2_ros import Buffer, TransformListener
        from rclpy.time import Time
        self._Time = Time
        if not rclpy.ok():
            rclpy.init()
        self.node = Node("calibration_dashboard_ref")
        self.world, self.frame = world_frame, camera_frame
        self.image_topic, self.info_topic = image_topic, info_topic
        self.lock = threading.Lock()
        self.msg, self.msg_t, self.used_stamp = None, 0.0, None
        self.K = self.D = self.size = None
        self.ok, self.n = False, 0
        self.T, self.t, self.error = None, 0.0, "no image yet on " + image_topic
        self.still = Still(still_s, 0.005, 0.3)
        self.buffer = Buffer()
        self.listener = TransformListener(self.buffer, self.node)
        self.node.create_subscription(Image, image_topic, self._image, qos_profile_sensor_data)
        self.node.create_subscription(CameraInfo, info_topic, self._info, qos_profile_sensor_data)
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.node)
        threading.Thread(target=self.executor.spin, daemon=True).start()

    # ---- ROS callbacks
    def _image(self, msg):
        if msg.encoding not in _ENC:
            self.error = "unsupported image encoding %s" % msg.encoding
            return
        with self.lock:
            self.msg, self.msg_t, self.ok = msg, time.time(), True
            self.n += 1

    def _info(self, msg):
        if self.K is None:
            self.K = np.array(msg.k, float).reshape(3, 3)
            self.D = np.array(msg.d, float) if len(msg.d) else np.zeros(5)
            self.size = (int(msg.width), int(msg.height))

    # ---- as a stream (calib.streams.Stream)
    def latest(self):
        """(BGR frame, receive time) of the newest image; remembers its capture time for now()."""
        with self.lock:
            msg, t = self.msg, self.msg_t
        if msg is None or time.time() - t > 2.0:
            self.ok = False
            return None, 0.0
        self.ok = True
        self.used_stamp = msg.header.stamp
        return image_to_bgr(msg), t

    def stop(self):
        self.executor.shutdown()

    # ---- as the map pose (server.RefPose)
    def now(self):
        """T_world_camera at the capture time of the image last returned by latest(); None if TF has no
        pose for that time or the robot is not still."""
        st = self.used_stamp
        if st is None:
            return None
        try:
            tr = self.buffer.lookup_transform(self.world, self.frame, self._Time.from_msg(st))
        except Exception as e:
            self.T, self.error = None, str(e).splitlines()[0]
            self.still.add(time.time(), None)
            return None
        from scipy.spatial.transform import Rotation as Rot
        q, p = tr.transform.rotation, tr.transform.translation
        T = np.eye(4)
        T[:3, :3] = Rot.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
        T[:3, 3] = [p.x, p.y, p.z]
        self.T, self.t, self.error = T, time.time(), ""
        self.still.add(stamp_seconds(st), T)
        return T.copy() if self.still.still else None

    def status(self):
        if self.T is None:
            return {"ok": False, "text": "no ZED pose: " + self.error}
        return {"ok": True, "still": self.still.still, "text": "robot still" if self.still.still else "robot moving",
                "world": self.world}
