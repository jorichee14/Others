#!/usr/bin/env python3
"""Serve the ZED's pose in its map over HTTP, for the calibration dashboard's extrinsics mode.

Run on the robot next to the ZED wrapper (positional tracking on), with ROS 2 sourced:

  python3 zed_pose_server.py
  python3 zed_pose_server.py --ros-args -p port:=8765 -p world_frame:=map_zed \\
                                        -p camera_frame:=zed_left_camera_optical_frame

and start the dashboard with  --ref-pose http://<robot-ip>:8765/pose

GET /pose  ->  {"ok": true, "T": 4x4 (world <- camera optical frame, metres), "age_s": seconds since the
               ZED pose was last updated, "world_frame": ..., "camera_frame": ...}
               {"ok": false, "error": ...} while TF has no pose (tracking not started / lost)
camera_frame must be the frame of the image the dashboard sees (the rectified left image streamed
with ros_image_to_rtsp.py is zed_left_camera_optical_frame).
"""
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from scipy.spatial.transform import Rotation as Rot
from tf2_ros import Buffer, TransformListener


class ZedPoseServer(Node):
    def __init__(self):
        super().__init__("zed_pose_server")
        self.declare_parameter("world_frame", "map_zed")
        self.declare_parameter("camera_frame", "zed_left_camera_optical_frame")
        self.declare_parameter("port", 8765)
        self.world = self.get_parameter("world_frame").value
        self.camera = self.get_parameter("camera_frame").value
        self.buffer = Buffer()
        self.listener = TransformListener(self.buffer, self)
        self.lock = threading.Lock()
        self.pose = {"ok": False, "error": "no TF %s -> %s yet" % (self.world, self.camera)}
        self.updated = 0.0
        self.create_timer(0.05, self._update)
        port = int(self.get_parameter("port").value)
        server = ThreadingHTTPServer(("0.0.0.0", port), self._handler())
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.get_logger().info("serving http://<this-host>:%d/pose  (%s <- %s)" % (port, self.world, self.camera))

    def _update(self):
        try:
            tr = self.buffer.lookup_transform(self.world, self.camera, Time())
        except Exception as e:
            with self.lock:
                self.pose = {"ok": False, "error": str(e).splitlines()[0]}
            return
        t, q = tr.transform.translation, tr.transform.rotation
        T = np.eye(4)
        T[:3, :3] = Rot.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
        T[:3, 3] = [t.x, t.y, t.z]
        with self.lock:
            changed = self.pose.get("T") != T.tolist()
            self.pose = {"ok": True, "T": T.tolist(), "world_frame": self.world, "camera_frame": self.camera}
            if changed:
                self.updated = time.time()

    def _handler(self):
        node = self

        class H(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path.split("?")[0] != "/pose":
                    self.send_error(404)
                    return
                with node.lock:
                    body = dict(node.pose, age_s=round(time.time() - node.updated, 3) if node.updated else None)
                data = json.dumps(body).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *a):
                pass
        return H


def main():
    rclpy.init()
    node = ZedPoseServer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
