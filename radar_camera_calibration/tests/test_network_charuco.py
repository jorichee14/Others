#!/usr/bin/env python3
"""network_charuco.py end to end without ROS or cameras: ROS (rclpy, tf2, cv_bridge, messages) is
replaced by small stand-ins, the ZED images and the RTSP cameras by rendered views of one board.

A simulated room: two fixed cameras B (cam 5 on a wall, cam 10 on the ceiling) with known poses in
the ZED map; a robot whose ZED looks at the board from a different spot for every board placement
(so the robot moves between holds, as in the real procedure). The test checks that both cameras
LOCK and that their poses in the map match the truth.

  python tests/test_network_charuco.py
"""
import json
import os
import sys
import tempfile
import time
import types

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as Rot

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TMP = tempfile.mkdtemp(prefix="netcharuco_")
SQ = 0.10                       # a large board (9 x 7 squares of 10 cm) so the far camera sees it well
with open(os.path.join(TMP, "boards.yaml"), "w") as f:
    f.write("boards:\n  A: {dictionary: DICT_4X4_50, squares_x: 9, squares_y: 7, square_m: %.3f, marker_m: %.3f}\n"
            "floor_tags: {dictionary: DICT_APRILTAG_36h11, size_m: 0.2}\n" % (SQ, SQ * 0.75))
os.environ["ED305_BOARDS"] = os.path.join(TMP, "boards.yaml")
W, H = 1280, 720
K_ZED = np.array([[700.0, 0, 640], [0, 700.0, 360], [0, 0, 1]])
K_B = np.array([[1000.0, 0, 640], [0, 1000.0, 360], [0, 0, 1]])


# ---------------------------------------------------------------- ROS stand-ins
class Obj:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def make_ros():
    params = {"cameras": "", "urls": "5=sim://5,10=sim://10", "intrinsics_dir": os.path.join(TMP, "results"),
              "rate": 10.0, "output_dir": os.path.join(TMP, "out"), "still_s": 0.4}

    class Node:
        def __init__(self, name):
            self.subs, self.timers, self.logs = {}, [], []

        def declare_parameter(self, n, v):
            params.setdefault(n, v)

        def get_parameter(self, n):
            return Obj(value=params[n])

        def create_subscription(self, typ, topic, cb, qos):
            self.subs[typ.__name__] = cb

        def create_timer(self, period, cb):
            self.timers.append((period, cb))

        def get_logger(self):
            node = self

            class L:
                def info(self, m, **kw):
                    node.logs.append(m)

                warn = info
            return L()

        def get_clock(self):
            return Obj(now=lambda: Obj(to_msg=lambda: 0))

        def destroy_node(self):
            pass

    world = {"T_world_A": np.eye(4)}

    def T_to_tr(T):
        q = Rot.from_matrix(T[:3, :3]).as_quat()
        return Obj(transform=Obj(translation=Obj(x=T[0, 3], y=T[1, 3], z=T[2, 3]),
                                 rotation=Obj(x=q[0], y=q[1], z=q[2], w=q[3])))

    class Buffer:
        def lookup_transform(self, parent, child, t):
            return T_to_tr(world["T_world_A"])

    class TransformStamped:
        def __init__(self):
            self.header = Obj(stamp=None, frame_id="")
            self.child_frame_id = ""
            self.transform = Obj(translation=Obj(x=0, y=0, z=0), rotation=Obj(x=0, y=0, z=0, w=1))

    sent = []
    mods = {
        "rclpy": types.SimpleNamespace(init=lambda: None, spin=lambda n: None, shutdown=lambda: None),
        "rclpy.node": types.SimpleNamespace(Node=Node),
        "rclpy.qos": types.SimpleNamespace(qos_profile_sensor_data=None),
        "rclpy.time": types.SimpleNamespace(Time=types.SimpleNamespace(from_msg=lambda s: s)),
        "tf2_ros": types.SimpleNamespace(Buffer=Buffer, TransformListener=lambda b, n: None,
                                         StaticTransformBroadcaster=lambda n: Obj(sendTransform=sent.append)),
        "cv_bridge": types.SimpleNamespace(CvBridge=lambda: Obj(imgmsg_to_cv2=lambda m, enc: m.data)),
        "geometry_msgs": types.SimpleNamespace(), "geometry_msgs.msg": types.SimpleNamespace(TransformStamped=TransformStamped),
        "sensor_msgs": types.SimpleNamespace(),
        "sensor_msgs.msg": types.SimpleNamespace(CameraInfo=type("CameraInfo", (), {}), Image=type("Image", (), {})),
    }
    sys.modules.update(mods)
    return world, sent


# ---------------------------------------------------------------- rendering
def look_at(pos, target, up=(0, 0, 1)):
    """T_world_cam of an OpenCV camera (z forward, y down) at pos looking at target."""
    z = np.asarray(target, float) - pos
    z /= np.linalg.norm(z)
    x = np.cross(z, up)
    if np.linalg.norm(x) < 1e-6:
        x = np.cross(z, (0, 1, 0))
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    T = np.eye(4)
    T[:3, :3] = np.stack([x, y, z], 1)
    T[:3, 3] = pos
    return T


BOARD_IMG = None


def render(K, T_world_cam, T_world_board):
    """Grey image of the board seen by a pinhole camera."""
    T = np.linalg.inv(T_world_cam) @ T_world_board            # board -> camera
    s = BOARD_IMG.shape[1] / (9 * SQ)                         # board image px per metre
    H_ = K @ np.column_stack([T[:3, 0], T[:3, 1], T[:3, 3]]) @ np.diag([1 / s, 1 / s, 1])
    return cv2.warpPerspective(BOARD_IMG, H_, (W, H), borderValue=128)


def main():
    global BOARD_IMG
    world, sent = make_ros()
    os.makedirs(os.path.join(TMP, "results", "intrinsics"), exist_ok=True)
    for n in (5, 10):
        with open(os.path.join(TMP, "results", "intrinsics", "cam_%d.json" % n), "w") as f:
            json.dump({"camera": n, "K": K_B.tolist(), "dist": [0, 0, 0, 0, 0], "model": "full"}, f)
    sys.path.insert(0, HERE)
    import network_charuco as nc
    from calib.common import load_boards
    BOARD_IMG = load_boards(["A"])["A"].board.generateImage((900, 700), marginSize=0)

    truth = {5: look_at(np.array([0.0, -3.0, 2.0]), [0.0, 0.0, 0.6]),       # wall camera
             10: look_at(np.array([0.3, 0.2, 2.7]), [0.0, 0.0, 0.0], up=(0, 1, 0))}  # ceiling camera
    state = {"T_world_board": np.eye(4)}

    class FakeStream:
        """A fixed camera B in the room: renders whatever the board does."""
        def __init__(self, n, url):
            self.n, self.ok = n, True

        def latest(self):
            img = render(K_B, truth[self.n], state["T_world_board"])
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), time.time()

        def stop(self):
            pass

    nc.Stream = FakeStream
    node = nc.NetworkCharuco()
    node.subs["CameraInfo"](Obj(k=K_ZED.ravel().tolist(), d=[0.0] * 5, width=W, height=H))
    tick = [cb for p, cb in node.timers if p < 1.0][0]

    rng = np.random.default_rng(3)
    t_end = time.time() + 90
    placement = 0
    while time.time() < t_end and not all(c.pair.locked is not None for c in node.cams.values()):
        # a board placement somewhere the cameras look at, tilted towards them
        centre = np.array([rng.uniform(-0.4, 0.4), rng.uniform(-0.4, 0.4), rng.uniform(0.3, 0.9)])
        R = Rot.from_euler("xyz", [rng.uniform(140, 220), rng.uniform(-25, 25), rng.uniform(-30, 30)], degrees=True)
        Tb = np.eye(4)
        Tb[:3, :3] = R.as_matrix()
        Tb[:3, 3] = centre - R.as_matrix() @ np.array([4.5 * SQ, 3.5 * SQ, 0])
        state["T_world_board"] = Tb
        # the robot drives somewhere else and its ZED looks at the board from 1.5-2.5 m
        ang = rng.uniform(0, 2 * np.pi)
        pos = centre + np.array([np.cos(ang), np.sin(ang), 0]) * rng.uniform(1.5, 2.5) + [0, 0, rng.uniform(-0.2, 0.3)]
        world["T_world_A"] = look_at(pos, centre)
        placement += 1
        for _ in range(int(2.5 * 10)):             # hold still for 2.5 s at 10 Hz
            img = render(K_ZED, world["T_world_A"], Tb)
            node._image(Obj(data=cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), header=Obj(stamp=time.time())))
            tick()
            time.sleep(0.1)
    node._report()
    print("board placements used:", placement)
    ok = True
    for n, c in node.cams.items():
        st = c.pair.status()
        X = c.pair.locked if c.pair.locked is not None else c.pair.X
        dt = np.linalg.norm(X[:3, 3] - truth[n][:3, 3]) * 1000 if X is not None else float("nan")
        dr = np.degrees(np.linalg.norm(Rot.from_matrix(X[:3, :3].T @ truth[n][:3, :3]).as_rotvec())) if X is not None else float("nan")
        print("cam %d: locked %s, %d samples from %d placements, error vs truth %.1f mm / %.3f deg, spread %.1f mm" % (
            n, st["locked"], st["samples"], st["holds"], dt, dr, st["spread_mm"]))
        ok &= st["locked"] and dt < 10 and dr < 0.3
    out = json.load(open(os.path.join(TMP, "out", "network.json")))
    print("network.json cameras:", sorted(out), "| static TF published:", [m.child_frame_id for m in sent[-1]] if sent else [])
    assert ok, "a camera did not lock or is off"
    assert sorted(out) == ["10", "5"] and all(out[k]["stage"] == "locked" for k in out)
    print("PASS")


if __name__ == "__main__":
    main()
