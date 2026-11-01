#!/usr/bin/env python3
"""board_pose_node.py end to end without ROS or cameras: rclpy and PoseStamped are small stand-ins,
the streams are simulated cameras showing board E at a known pose (cam 3 and 6) or no board (cam 8).
Checks that only cams 3 and 6 publish, every message names its camera, and the poses match the truth.

  python tests/test_board_pose_node.py
"""
import os
import sys
import tempfile
import threading
import time
import types

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as Rot

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import board_pose_node  # noqa: E402
from calib.common import load_boards, write_json  # noqa: E402

W, H = 1920, 1080
K = np.array([[1400.0, 0, W / 2], [0, 1400.0, H / 2], [0, 0, 1]])
B = load_boards(["E"])["E"]
SIZE = 4 * 0.120                                          # board E: 4 squares of 12 cm
IMG = B.board.generateImage((960, 960), marginSize=0)
TRUTH = {3: (Rot.from_euler("xyz", [20, -15, 5], degrees=True).as_matrix(), [-0.3, -0.2, 2.4]),
         6: (Rot.from_euler("xyz", [-10, 25, -30], degrees=True).as_matrix(), [0.1, -0.25, 2.8])}
RUNNING = [True]
SENT = []


def render(cam):
    if cam not in TRUTH:
        return np.full((H, W, 3), 110, np.uint8)
    R, t = TRUTH[cam]
    s = IMG.shape[1] / SIZE                               # board image pixels per metre
    Hm = K @ np.column_stack([R[:, 0], R[:, 1], t]) @ np.diag([1 / s, 1 / s, 1])
    return cv2.cvtColor(cv2.warpPerspective(IMG, Hm, (W, H), borderValue=110), cv2.COLOR_GRAY2BGR)


class FakeStream:
    def __init__(self, cam):
        self.cam, self.ok, self.n, self.img = cam, True, 0, render(cam)
        threading.Thread(target=self.run, daemon=True).start()

    def run(self):                                        # a new frame every 1/30 s
        while RUNNING[0]:
            self.n += 1
            time.sleep(1 / 30)

    def latest(self):
        return self.img.copy(), time.time()

    def stop(self):
        pass


def fake_ros():
    class Obj:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    class PoseStamped:
        def __init__(self):
            self.header = Obj(stamp=Obj(sec=0, nanosec=0), frame_id="")
            self.pose = Obj(position=Obj(x=0.0, y=0.0, z=0.0), orientation=Obj(x=0.0, y=0.0, z=0.0, w=1.0))

    class Node:
        def create_publisher(self, typ, topic, qos):
            assert topic == "/ed305/board_pose"
            return Obj(publish=SENT.append)

        def destroy_node(self):
            pass

    rclpy = types.ModuleType("rclpy")
    rclpy.init, rclpy.ok, rclpy.shutdown = (lambda: None), (lambda: RUNNING[0]), (lambda: None)
    rclpy.create_node = lambda name: Node()
    msg = types.ModuleType("geometry_msgs.msg")
    msg.PoseStamped = PoseStamped
    sys.modules.update({"rclpy": rclpy, "geometry_msgs": types.ModuleType("geometry_msgs"), "geometry_msgs.msg": msg})


def main():
    fake_ros()
    board_pose_node.open_streams = lambda urls: {c: FakeStream(c) for c in urls}
    tmp = tempfile.mkdtemp()
    args = ["--cams", "3", "6", "8", "--hz", "5", "--results", tmp]
    for c in (3, 6, 8):
        write_json(os.path.join(tmp, "cam_%d.json" % c), {"K": K.tolist(), "dist": [0, 0, 0, 0, 0]})
        args += ["--intrinsics", "%d=%s" % (c, os.path.join(tmp, "cam_%d.json" % c))]
    th = threading.Thread(target=board_pose_node.main, args=(args,), daemon=True)
    th.start()
    time.sleep(3)
    RUNNING[0] = False
    th.join(10)

    by_cam = {}
    for m in SENT:
        by_cam.setdefault(m.header.frame_id, []).append(m)
    print({k: len(v) for k, v in by_cam.items()})
    assert set(by_cam) == {"cam_3", "cam_6"}, by_cam.keys()
    for cam, (R, t) in TRUTH.items():
        for m in by_cam["cam_%d" % cam]:
            p, o = m.pose.position, m.pose.orientation
            dt = np.linalg.norm(np.array([p.x, p.y, p.z]) - t)
            dr = np.degrees((Rot.from_quat([o.x, o.y, o.z, o.w]) * Rot.from_matrix(R).inv()).magnitude())
            assert dt < 0.02 and dr < 1.0, (cam, dt, dr)
            assert abs(m.header.stamp.sec + m.header.stamp.nanosec * 1e-9 - time.time()) < 10
        print("cam_%d: %d poses, position error %.1f mm, rotation error %.2f deg"
              % (cam, len(by_cam["cam_%d" % cam]), dt * 1000, dr))
    print("OK")


if __name__ == "__main__":
    main()
