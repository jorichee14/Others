#!/usr/bin/env python3
"""The dashboard with the reference camera read straight from ROS 2 (--ref-topic), end to end without
ROS: rclpy, tf2_ros and the message types are replaced by small stand-ins that deliver the simulated
ZED's images (with capture timestamps), its camera_info, and its map pose from "TF" at any timestamp.
The room, the other cameras and the stream delays are those of test_ref_extrinsics.py.

  python tests/test_ros_ref.py
"""
import importlib.util
import json
import os
import sys
import threading
import time
import types

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as Rot

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("base", os.path.join(HERE, "tests", "test_ref_extrinsics.py"))
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)              # the simulated room (sets ED305_BOARDS to its test board)
base.PORT = 8094


class Obj:
    def __init__(self, **kw):
        self.__dict__.update(kw)


SUBS = {}


def fake_ros():
    class Node:
        def __init__(self, name):
            pass

        def create_subscription(self, typ, topic, cb, qos):
            SUBS[typ.__name__] = cb

    class Buffer:
        def lookup_transform(self, world, frame, t):
            T = base.world_at(t)[2]                       # the robot's pose at that capture time
            q = Rot.from_matrix(T[:3, :3]).as_quat()
            return Obj(transform=Obj(translation=Obj(x=T[0, 3], y=T[1, 3], z=T[2, 3]),
                                     rotation=Obj(x=q[0], y=q[1], z=q[2], w=q[3])))

    class Executor:
        def add_node(self, n):
            pass

        def spin(self):
            pass

        def shutdown(self):
            pass

    sys.modules.update({
        "rclpy": types.SimpleNamespace(ok=lambda: True, init=lambda: None),
        "rclpy.executors": types.SimpleNamespace(SingleThreadedExecutor=Executor),
        "rclpy.node": types.SimpleNamespace(Node=Node),
        "rclpy.qos": types.SimpleNamespace(qos_profile_sensor_data=None),
        "rclpy.time": types.SimpleNamespace(Time=types.SimpleNamespace(from_msg=lambda st: st.sec + st.nanosec * 1e-9)),
        "sensor_msgs": types.SimpleNamespace(),
        "sensor_msgs.msg": types.SimpleNamespace(Image=type("Image", (), {}), CameraInfo=type("CameraInfo", (), {})),
        "tf2_ros": types.SimpleNamespace(Buffer=Buffer, TransformListener=lambda b, n: None),
    })


def publish_zed():
    """The ZED wrapper: images (bgra8, stamped with their capture time, 0.1 s late) and camera_info."""
    while "CameraInfo" not in SUBS or "Image" not in SUBS:
        time.sleep(0.05)
    SUBS["CameraInfo"](Obj(k=base.K[20].ravel().tolist(), d=[0.0] * 5, width=base.W, height=base.H))
    while True:
        t_cap = time.time() - base.DELAY[20]
        img = cv2.cvtColor(base.render(20), cv2.COLOR_BGR2BGRA)       # render() applies the same delay
        SUBS["Image"](Obj(encoding="bgra8", width=base.W, height=base.H, step=base.W * 4, data=img.tobytes(),
                          header=Obj(stamp=Obj(sec=int(t_cap), nanosec=int((t_cap % 1) * 1e9)))))
        time.sleep(1 / 15)


def main():
    fake_ros()
    os.makedirs(os.path.join(base.TMP, "results", "intrinsics"))
    for c in (5, 10):                                   # the ZED gets no file: its K comes from camera_info
        with open(os.path.join(base.TMP, "results", "intrinsics", "cam_%d.json" % c), "w") as f:
            json.dump({"camera": c, "K": base.K[c].tolist(), "dist": [0, 0, 0, 0, 0], "model": "full",
                       "image_size": [base.W, base.H]}, f)
    base.move(base.WORLD["board"], base.WORLD["robot"])
    sys.path.insert(0, HERE)
    import server
    from calib.common import load_boards
    base.BOARD_IMG = load_boards(["A"])["A"].board.generateImage((900, 700), marginSize=0)
    server.open_streams = lambda urls: {c: base.WorldStream(c) for c in urls}
    sys.argv = ["server.py", "--cams", "5", "10", "20", "--url", "5=sim://5", "--url", "10=sim://10",
                "--ref", "20", "--ref-topic", "/zed/zed_node/left/image_rect_color", "--out", os.path.join(base.TMP, "data"),
                "--results", os.path.join(base.TMP, "results"), "--port", str(base.PORT), "--host", "127.0.0.1", "--hz", "5"]
    threading.Thread(target=server.main, daemon=True).start()
    threading.Thread(target=publish_zed, daemon=True).start()
    for _ in range(50):
        try:
            base.api("/api/state")
            break
        except Exception:
            time.sleep(0.2)
    base.api("/api/mode", {"mode": "extr"})

    rng = np.random.default_rng(3)
    t_end, placements = time.time() + 150, 0
    while time.time() < t_end:
        net = base.api("/api/state")["net"]
        if all(v["locked"] for k, v in net["cams"].items() if k != "20"):
            break
        centre = np.array([rng.uniform(-0.4, 0.4), rng.uniform(-0.4, 0.4), rng.uniform(0.3, 0.9)])
        R = Rot.from_euler("xyz", [rng.uniform(140, 220), rng.uniform(-25, 25), rng.uniform(-30, 30)], degrees=True).as_matrix()
        Tb = np.eye(4)
        Tb[:3, :3], Tb[:3, 3] = R, centre - R @ np.array([4.5 * base.SQ, 3.5 * base.SQ, 0])
        ang = rng.uniform(0, 2 * np.pi)
        pos = centre + np.array([np.cos(ang), np.sin(ang), 0]) * rng.uniform(1.5, 2.5) + [0, 0, rng.uniform(-0.2, 0.3)]
        base.move(Tb, base.look_at(pos, centre))
        placements += 1
        time.sleep(3.5)
    st = base.api("/api/state")
    net = st["net"]
    ref_cam = st["cams"]["20"]
    print("placements:", placements, "| pose:", net["pose"], "| ZED intrinsics:", ref_cam["source"], ref_cam["K"])
    ok = ref_cam["calibrated"] and net["frame"] == "map_zed"
    for c in (5, 10):
        rec = json.load(open(os.path.join(base.TMP, "results", "extrinsics_ref", "cam_%d.json" % c)))
        T = np.array(rec["T"])
        dt = np.linalg.norm(T[:3, 3] - base.TRUTH[c][:3, 3]) * 1000
        dr = np.degrees(np.linalg.norm(Rot.from_matrix(T[:3, :3].T @ base.TRUTH[c][:3, :3]).as_rotvec()))
        x = net["cams"][str(c)]
        print("cam %d: locked %s, %d samples, %d placements, spread %.1f mm | error vs truth %.1f mm / %.3f deg" % (
            c, x["locked"], x["samples"], x["holds"], x["spread_mm"], dt, dr))
        ok &= x["locked"] and dt < 10 and dr < 0.3
    assert ok
    print("PASS")


if __name__ == "__main__":
    main()
