# -*- coding: utf-8 -*-
"""
Building ROS 2 messages without a ROS install: headers, PointCloud2, Imu,
String, and their CDR bytes and message definitions.

One place for every writer in the pipeline, so a bag written by any stage
carries the same message layout. Messages are ``rosbags`` types (Humble
definitions); :func:`serialize` gives the CDR bytes a rosbag2 file stores.
"""
from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

__all__ = ["typestore", "msg_type", "msgdef", "serialize", "header", "tf_message",
           "point_fields", "pointcloud2", "imu", "string"]

# numpy (kind, bytes) -> sensor_msgs/PointField datatype
_PF_CODE = {("i", 1): 1, ("u", 1): 2, ("i", 2): 3, ("u", 2): 4,
            ("i", 4): 5, ("u", 4): 6, ("f", 4): 7, ("f", 8): 8}

_TS = None


def typestore():
    """The rosbags type store (ROS 2 Humble message definitions), shared."""
    global _TS
    if _TS is None:
        from rosbags.typesys import Stores, get_typestore
        _TS = get_typestore(Stores.ROS2_HUMBLE)
    return _TS


def msg_type(name: str):
    """Message class, e.g. ``msg_type("sensor_msgs/msg/Imu")``."""
    return typestore().types[name]


def msgdef(name: str) -> str:
    """ros2msg definition text of a type (the schema a rosbag2 MCAP stores)."""
    text, _ = typestore().generate_msgdef(name, ros_version=2)
    return text


def serialize(msg) -> bytes:
    """Little-endian CDR bytes of a message."""
    return bytes(typestore().serialize_cdr(msg, msg.__msgtype__))


def header(stamp_ns: int, frame_id: str):
    Time = msg_type("builtin_interfaces/msg/Time")
    return msg_type("std_msgs/msg/Header")(
        stamp=Time(sec=int(stamp_ns // 1_000_000_000),
                   nanosec=int(stamp_ns % 1_000_000_000)),
        frame_id=frame_id)


def point_fields(dtype: np.dtype) -> List[Tuple[str, int, int]]:
    """(name, offset, PointField datatype) for every field of a structured
    little-endian dtype, in offset order -- the cloud layout IS the dtype."""
    out = []
    for name, (ft, off, *_) in dtype.fields.items():
        if ft.byteorder == ">" or (ft.byteorder == "=" and np.little_endian is False):
            raise ValueError(f"field {name!r} is big-endian")
        code = _PF_CODE.get((ft.kind, ft.itemsize))
        if code is None or ft.shape:
            raise ValueError(f"field {name!r}: no PointField type for {ft}")
        out.append((name, int(off), code))
    return sorted(out, key=lambda f: f[1])


def pointcloud2(points: np.ndarray, stamp_ns: int, frame_id: str,
                height: int = 1, width: Optional[int] = None,
                is_dense: bool = False):
    """sensor_msgs/PointCloud2 from a structured array (one record per point).
    ``height`` x ``width`` makes it organized (e.g. beams x columns)."""
    points = np.ascontiguousarray(points).reshape(-1)
    width = len(points) // height if width is None else width
    if height * width != len(points):
        raise ValueError(f"{len(points)} points do not fill {height} x {width}")
    PF = msg_type("sensor_msgs/msg/PointField")
    step = points.dtype.itemsize
    return msg_type("sensor_msgs/msg/PointCloud2")(
        header=header(stamp_ns, frame_id), height=height, width=width,
        fields=[PF(name=n, offset=o, datatype=c, count=1)
                for n, o, c in point_fields(points.dtype)],
        is_bigendian=False, point_step=step, row_step=step * width,
        data=np.frombuffer(points.tobytes(), np.uint8), is_dense=is_dense)


def _cov(diag: Sequence[float]) -> np.ndarray:
    return np.diag(np.asarray(diag, np.float64)).reshape(-1)


def imu(stamp_ns: int, frame_id: str, accel: Iterable[float], gyro: Iterable[float],
        accel_cov: float = 0.0, gyro_cov: float = 0.0, orientation=None):
    """sensor_msgs/Imu. Without ``orientation`` the message says so the ROS
    way: identity quaternion, orientation_covariance[0] = -1."""
    V = msg_type("geometry_msgs/msg/Vector3")
    Q = msg_type("geometry_msgs/msg/Quaternion")
    a = [float(v) for v in accel]
    g = [float(v) for v in gyro]
    ocov = np.zeros(9)
    if orientation is None:
        q = Q(x=0.0, y=0.0, z=0.0, w=1.0)
        ocov[0] = -1.0
    else:
        q = Q(x=float(orientation[0]), y=float(orientation[1]),
              z=float(orientation[2]), w=float(orientation[3]))
    return msg_type("sensor_msgs/msg/Imu")(
        header=header(stamp_ns, frame_id), orientation=q, orientation_covariance=ocov,
        angular_velocity=V(x=g[0], y=g[1], z=g[2]),
        angular_velocity_covariance=_cov([gyro_cov] * 3),
        linear_acceleration=V(x=a[0], y=a[1], z=a[2]),
        linear_acceleration_covariance=_cov([accel_cov] * 3))


def string(text: str):
    return msg_type("std_msgs/msg/String")(data=text)


def tf_message(transforms, stamp_ns: int):
    """tf2_msgs/TFMessage of ``(parent, child, translation xyz, rotation xyzw)``."""
    T = msg_type
    out = []
    for parent, child, t, q in transforms:
        out.append(T("geometry_msgs/msg/TransformStamped")(
            header=header(stamp_ns, parent), child_frame_id=child,
            transform=T("geometry_msgs/msg/Transform")(
                translation=T("geometry_msgs/msg/Vector3")(x=float(t[0]), y=float(t[1]), z=float(t[2])),
                rotation=T("geometry_msgs/msg/Quaternion")(x=float(q[0]), y=float(q[1]),
                                                           z=float(q[2]), w=float(q[3])))))
    return T("tf2_msgs/msg/TFMessage")(transforms=out)

