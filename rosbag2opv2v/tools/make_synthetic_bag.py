#!/usr/bin/env python3
"""Write a small synthetic MCAP bag that mimics the MIRC cooperative recording.

Same topic names, frames and message types as the real bag, but a scene we know
the ground truth of: two robots driving circles in a 12 x 10 m room watched by a
fixed infrastructure node.  Used by the test-suite (and handy for trying the
converter out) -- after converting it, every agent's cloud must land on the same
room walls and the ground-truth boxes must contain real points.

    python tools/make_synthetic_bag.py --out /tmp/synthetic.mcap --seconds 20
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rosbag2opv2v import transforms as tf  # noqa: E402

GEOMETRY = """================================================================================
MSG: geometry_msgs/Point
float64 x
float64 y
float64 z
================================================================================
MSG: geometry_msgs/Quaternion
float64 x
float64 y
float64 z
float64 w
================================================================================
MSG: geometry_msgs/Pose
geometry_msgs/Point position
geometry_msgs/Quaternion orientation
"""

MSGDEFS = {
    "sensor_msgs/msg/PointCloud2": """std_msgs/Header header
uint32 height
uint32 width
sensor_msgs/PointField[] fields
bool is_bigendian
uint32 point_step
uint32 row_step
uint8[] data
bool is_dense
================================================================================
MSG: std_msgs/Header
builtin_interfaces/Time stamp
string frame_id
================================================================================
MSG: builtin_interfaces/Time
int32 sec
uint32 nanosec
================================================================================
MSG: sensor_msgs/PointField
string name
uint32 offset
uint8 datatype
uint32 count
""",
    "sensor_msgs/msg/Image": """std_msgs/Header header
uint32 height
uint32 width
string encoding
uint8 is_bigendian
uint32 step
uint8[] data
================================================================================
MSG: std_msgs/Header
builtin_interfaces/Time stamp
string frame_id
================================================================================
MSG: builtin_interfaces/Time
int32 sec
uint32 nanosec
""",
    "sensor_msgs/msg/CameraInfo": """std_msgs/Header header
uint32 height
uint32 width
string distortion_model
float64[] d
float64[9] k
float64[9] r
float64[12] p
uint32 binning_x
uint32 binning_y
sensor_msgs/RegionOfInterest roi
================================================================================
MSG: std_msgs/Header
builtin_interfaces/Time stamp
string frame_id
================================================================================
MSG: builtin_interfaces/Time
int32 sec
uint32 nanosec
================================================================================
MSG: sensor_msgs/RegionOfInterest
uint32 x_offset
uint32 y_offset
uint32 height
uint32 width
bool do_rectify
""",
    "tf2_msgs/msg/TFMessage": """geometry_msgs/TransformStamped[] transforms
================================================================================
MSG: geometry_msgs/TransformStamped
std_msgs/Header header
string child_frame_id
geometry_msgs/Transform transform
================================================================================
MSG: std_msgs/Header
builtin_interfaces/Time stamp
string frame_id
================================================================================
MSG: builtin_interfaces/Time
int32 sec
uint32 nanosec
================================================================================
MSG: geometry_msgs/Transform
geometry_msgs/Vector3 translation
geometry_msgs/Quaternion rotation
================================================================================
MSG: geometry_msgs/Vector3
float64 x
float64 y
float64 z
================================================================================
MSG: geometry_msgs/Quaternion
float64 x
float64 y
float64 z
float64 w
""",
    "wifi_monitor_msgs/msg/WifiLinkStatus": """std_msgs/Header header
string ssid
float32 rssi_dbm
float32 tx_rate_mbps
uint32 retries
================================================================================
MSG: std_msgs/Header
builtin_interfaces/Time stamp
string frame_id
================================================================================
MSG: builtin_interfaces/Time
int32 sec
uint32 nanosec
""",
}

MSGDEFS["geometry_msgs/msg/PoseStamped"] = """std_msgs/Header header
geometry_msgs/Pose pose
================================================================================
MSG: std_msgs/Header
builtin_interfaces/Time stamp
string frame_id
================================================================================
MSG: builtin_interfaces/Time
int32 sec
uint32 nanosec
""" + GEOMETRY

ROOM_X, ROOM_Y, ROOM_Z = 6.0, 5.0, 2.6
T0 = 1787899802.0

# extrinsics baked into /tf_static (the config's `frame:` names point here)
STATIC_TF = [
    ("base_link", "os_sensor", [0.0, 0.0, 0.32], [0.0, 0.0, 0.0]),
    ("base_link", "zed_left_camera_optical_frame", [0.12, 0.06, 0.24],
     [-90.0, 0.0, -90.0]),
    ("mobile_2_base_link", "mobile_2_depth_optical_frame", [0.14, 0.0, 0.22],
     [-90.0, 0.0, -90.0]),
    ("mobile_2_base_link", "mobile_2_color_optical_frame", [0.14, 0.015, 0.22],
     [-90.0, 0.0, -90.0]),
    ("infra_1_base", "infra_1_radar", [0.0, 0.0, 2.45], [0.0, 25.0, 0.0]),
    ("infra_1_base", "infra_1_camera_optical", [0.0, 0.1, 2.45],
     [-90.0, 0.0, -90.0]),
]

ROBOT_A_EXTENT = np.array([0.35, 0.28, 0.35])
ROBOT_B_EXTENT = np.array([0.30, 0.25, 0.30])


def stamp(t: float) -> dict:
    return {"sec": int(t), "nanosec": int(round((t - int(t)) * 1e9))}


def robot_a_pose(t: float) -> np.ndarray:
    angle = 0.25 * t
    pos = [2.5 * math.cos(angle), 2.5 * math.sin(angle), 0.0]
    return tf.make_matrix(pos, tf.rpy_deg_to_matrix(
        0.0, 0.0, math.degrees(angle + math.pi / 2)))


def robot_b_pose(t: float) -> np.ndarray:
    angle = -0.18 * t + 1.2
    pos = [4.0 * math.cos(angle), 3.2 * math.sin(angle), 0.0]
    return tf.make_matrix(pos, tf.rpy_deg_to_matrix(
        0.0, 0.0, math.degrees(angle - math.pi / 2)))


def room_points(rng: np.random.Generator, count: int = 6000) -> np.ndarray:
    """Static scene: four walls plus a floor grid."""
    n = count // 5
    walls = [
        np.stack([np.full(n, ROOM_X), rng.uniform(-ROOM_Y, ROOM_Y, n),
                  rng.uniform(0.0, ROOM_Z, n)], axis=1),
        np.stack([np.full(n, -ROOM_X), rng.uniform(-ROOM_Y, ROOM_Y, n),
                  rng.uniform(0.0, ROOM_Z, n)], axis=1),
        np.stack([rng.uniform(-ROOM_X, ROOM_X, n), np.full(n, ROOM_Y),
                  rng.uniform(0.0, ROOM_Z, n)], axis=1),
        np.stack([rng.uniform(-ROOM_X, ROOM_X, n), np.full(n, -ROOM_Y),
                  rng.uniform(0.0, ROOM_Z, n)], axis=1),
        np.stack([rng.uniform(-ROOM_X, ROOM_X, n),
                  rng.uniform(-ROOM_Y, ROOM_Y, n), np.zeros(n)], axis=1),
    ]
    return np.concatenate(walls, axis=0).astype(np.float32)


def box_points(pose: np.ndarray, extent: np.ndarray, centre_z: float,
               rng: np.random.Generator, count: int = 400) -> np.ndarray:
    """Points on the surface of a robot's bounding box, in world coordinates."""
    local = rng.uniform(-1.0, 1.0, size=(count, 3))
    axis = rng.integers(0, 3, size=count)
    local[np.arange(count), axis] = np.sign(local[np.arange(count), axis])
    local = local * extent + np.array([0.0, 0.0, centre_z])
    return (local @ pose[:3, :3].T + pose[:3, 3]).astype(np.float32)


def pointcloud2(t: float, frame: str, xyz: np.ndarray,
                intensity: np.ndarray) -> dict:
    count = xyz.shape[0]
    buf = np.empty((count, 4), dtype=np.float32)
    buf[:, :3] = xyz
    buf[:, 3] = intensity
    fields = [{"name": name, "offset": 4 * i, "datatype": 7, "count": 1}
              for i, name in enumerate(["x", "y", "z", "intensity"])]
    return {
        "header": {"stamp": stamp(t), "frame_id": frame},
        "height": 1, "width": count, "fields": fields, "is_bigendian": False,
        "point_step": 16, "row_step": 16 * count,
        "data": buf.tobytes(), "is_dense": True,
    }


def camera_info(t: float, frame: str, width: int, height: int,
                fx: float) -> dict:
    k = [fx, 0.0, width / 2.0, 0.0, fx, height / 2.0, 0.0, 0.0, 1.0]
    return {
        "header": {"stamp": stamp(t), "frame_id": frame},
        "height": height, "width": width, "distortion_model": "plumb_bob",
        "d": [0.0] * 5, "k": k, "r": [1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0],
        "p": k[:3] + [0.0] + k[3:6] + [0.0] + k[6:] + [0.0],
        "binning_x": 0, "binning_y": 0,
        "roi": {"x_offset": 0, "y_offset": 0, "height": 0, "width": 0,
                "do_rectify": False},
    }


def rgb_image(t: float, frame: str, width: int, height: int,
              shade: int) -> dict:
    data = np.full((height, width, 3), shade, dtype=np.uint8)
    data[:, :, 1] = (np.arange(width, dtype=np.uint8) % 255)
    return {
        "header": {"stamp": stamp(t), "frame_id": frame},
        "height": height, "width": width, "encoding": "rgb8",
        "is_bigendian": 0, "step": width * 3, "data": data.tobytes(),
    }


def depth_image(t: float, frame: str, points_optical: np.ndarray,
                intrinsic: np.ndarray, width: int, height: int) -> dict:
    """Z-buffer the visible scene points into a 16UC1 millimetre depth image."""
    depth = np.zeros((height, width), dtype=np.uint16)
    z = points_optical[:, 2]
    keep = z > 0.3
    pts = points_optical[keep]
    z = z[keep]
    u = np.round(pts[:, 0] * intrinsic[0, 0] / z + intrinsic[0, 2]).astype(int)
    v = np.round(pts[:, 1] * intrinsic[1, 1] / z + intrinsic[1, 2]).astype(int)
    inside = (u >= 0) & (u < width) & (v >= 0) & (v < height) & (z < 8.0)
    u, v, z = u[inside], v[inside], z[inside]
    order = np.argsort(-z)               # nearest wins after the scatter
    depth[v[order], u[order]] = np.round(z[order] * 1000.0).astype(np.uint16)
    return {
        "header": {"stamp": stamp(t), "frame_id": frame},
        "height": height, "width": width, "encoding": "16UC1",
        "is_bigendian": 0, "step": width * 2, "data": depth.tobytes(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    from mcap_ros2.writer import Writer

    rng = np.random.default_rng(args.seed)
    scene = room_points(rng)
    static = {}
    for parent, child, xyz, rpy in STATIC_TF:
        static[child] = (parent, tf.make_matrix(
            xyz, tf.rpy_deg_to_matrix(rpy[0], rpy[1], rpy[2])))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "wb") as handle:
        writer = Writer(handle)
        schemas = {name: writer.register_msgdef(name, text)
                   for name, text in MSGDEFS.items()}

        def emit(topic, type_name, message, t):
            writer.write_message(topic, schemas[type_name], message,
                                 log_time=int(t * 1e9),
                                 publish_time=int(t * 1e9))

        # /tf_static, latched once at the start
        transforms = []
        for child, (parent, matrix) in static.items():
            quat = tf.matrix_to_quat(matrix[:3, :3])
            transforms.append({
                "header": {"stamp": stamp(T0), "frame_id": parent},
                "child_frame_id": child,
                "transform": {
                    "translation": dict(zip("xyz", matrix[:3, 3].tolist())),
                    "rotation": dict(zip("xyzw", quat)),
                }})
        emit("/tf_static", "tf2_msgs/msg/TFMessage",
             {"transforms": transforms}, T0)

        depth_k = np.array([[210.0, 0.0, 160.0], [0.0, 210.0, 120.0],
                            [0.0, 0.0, 1.0]])
        n_pose = int(args.seconds * 20)
        for i in range(n_pose):
            t = T0 + i / 20.0
            for topic, pose in (("/mobile_1/global_pose", robot_a_pose(t)),
                                ("/mobile_2/global_pose", robot_b_pose(t))):
                quat = tf.matrix_to_quat(pose[:3, :3])
                emit(topic, "geometry_msgs/msg/PoseStamped", {
                    "header": {"stamp": stamp(t), "frame_id": "map"},
                    "pose": {
                        "position": dict(zip("xyz", pose[:3, 3].tolist())),
                        "orientation": dict(zip("xyzw", quat)),
                    }}, t)

        # Ouster LiDAR on Robot A, ~9.7 Hz
        for i in range(int(args.seconds * 9.7)):
            t = T0 + i / 9.7
            pose_a, pose_b = robot_a_pose(t), robot_b_pose(t)
            world = np.concatenate([
                scene[rng.choice(len(scene), 3000, replace=False)],
                box_points(pose_b, ROBOT_B_EXTENT, 0.30, rng, 500)])
            sensor = pose_a @ static["os_sensor"][1]
            local = (world - sensor[:3, 3]) @ sensor[:3, :3]
            emit("/mobile_1/ouster/points", "sensor_msgs/msg/PointCloud2",
                 pointcloud2(t, "os_sensor", local,
                             rng.uniform(50.0, 900.0, len(local))), t)

        # RealSense depth + colour on Robot B, ~15 Hz
        for i in range(int(args.seconds * 15)):
            t = T0 + i / 15.0
            pose_a, pose_b = robot_a_pose(t), robot_b_pose(t)
            world = np.concatenate([
                scene[rng.choice(len(scene), 4000, replace=False)],
                box_points(pose_a, ROBOT_A_EXTENT, 0.35, rng, 600)])
            optical = pose_b @ static["mobile_2_depth_optical_frame"][1]
            local = (world - optical[:3, 3]) @ optical[:3, :3]
            emit("/mobile_2/depth/image_rect_raw", "sensor_msgs/msg/Image",
                 depth_image(t, "mobile_2_depth_optical_frame", local,
                             depth_k, 320, 240), t)
            emit("/mobile_2/depth/camera_info", "sensor_msgs/msg/CameraInfo",
                 camera_info(t, "mobile_2_depth_optical_frame", 320, 240,
                             210.0), t)

        # infrastructure radar, ~8.6 Hz, and its camera
        for i in range(int(args.seconds * 8.6)):
            t = T0 + i / 8.6
            pose_a, pose_b = robot_a_pose(t), robot_b_pose(t)
            world = np.concatenate([
                scene[rng.choice(len(scene), 400, replace=False)],
                box_points(pose_a, ROBOT_A_EXTENT, 0.35, rng, 60),
                box_points(pose_b, ROBOT_B_EXTENT, 0.30, rng, 60)])
            sensor = static["infra_1_radar"][1]
            local = (world - sensor[:3, 3]) @ sensor[:3, :3]
            emit("/infra_1/radar/points_all", "sensor_msgs/msg/PointCloud2",
                 pointcloud2(t, "infra_1_radar", local,
                             rng.uniform(1.0, 40.0, len(local))), t)

        for i in range(int(args.seconds * 10)):
            t = T0 + i / 10.0
            emit("/mobile_1/zed/left/image_rect_color",
                 "sensor_msgs/msg/Image",
                 rgb_image(t, "zed_left_camera_optical_frame", 320, 180, 90), t)
            emit("/mobile_1/zed/left/camera_info",
                 "sensor_msgs/msg/CameraInfo",
                 camera_info(t, "zed_left_camera_optical_frame", 320, 180,
                             260.0), t)
            emit("/mobile_2/color/image_raw", "sensor_msgs/msg/Image",
                 rgb_image(t, "mobile_2_color_optical_frame", 320, 240, 140), t)
            emit("/mobile_2/color/camera_info", "sensor_msgs/msg/CameraInfo",
                 camera_info(t, "mobile_2_color_optical_frame", 320, 240,
                             215.0), t)
            emit("/infra_1/image_raw", "sensor_msgs/msg/Image",
                 rgb_image(t, "infra_1_camera_optical", 320, 240, 60), t)
            emit("/infra_1/camera_info", "sensor_msgs/msg/CameraInfo",
                 camera_info(t, "infra_1_camera_optical", 320, 240, 300.0), t)

        for i in range(int(args.seconds * 2)):
            t = T0 + i / 2.0
            for topic, name in (("/mobile_1/wifi/status", "robot_a"),
                                ("/mobile_2/wifi/status", "robot_b")):
                emit(topic, "wifi_monitor_msgs/msg/WifiLinkStatus", {
                    "header": {"stamp": stamp(t), "frame_id": name},
                    "ssid": "mirc-5g",
                    "rssi_dbm": float(-45.0 - 12.0 * math.sin(0.3 * t)),
                    "tx_rate_mbps": float(430.0 + 40.0 * math.cos(0.2 * t)),
                    "retries": int(3 + 2 * math.sin(0.5 * t)),
                }, t)
        writer.finish()

    size = os.path.getsize(args.out)
    print("wrote %s (%.1f MB)" % (args.out, size / 1e6))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
