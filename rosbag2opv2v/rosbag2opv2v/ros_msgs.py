"""Decoding helpers for the ROS 2 messages the converter consumes."""

from __future__ import annotations

import struct
from typing import Dict, Optional, Tuple

import numpy as np

# sensor_msgs/PointField datatype -> numpy dtype string (endianness added later)
_PF_TYPES = {
    1: "i1", 2: "u1", 3: "i2", 4: "u2",
    5: "i4", 6: "u4", 7: "f4", 8: "f8",
}


def peek_header_stamp(data: bytes) -> Optional[Tuple[float, str]]:
    """Read ``header.stamp`` / ``header.frame_id`` out of a raw CDR payload.

    Valid for any message whose first field is a ``std_msgs/Header``; lets the
    indexing pass timestamp huge PointCloud2 / Image messages without paying to
    deserialise them.  Returns ``(seconds, frame_id)`` or ``None``.
    """
    if len(data) < 16:
        return None
    little = data[1] & 0x01 == 0x01
    fmt = "<" if little else ">"
    sec, nsec, name_len = struct.unpack_from(fmt + "iII", data, 4)
    frame_id = ""
    if 0 < name_len < 1024 and 16 + name_len <= len(data):
        frame_id = data[16:16 + name_len - 1].decode("utf-8", errors="replace")
    return sec + nsec * 1e-9, frame_id


def msg_stamp(msg) -> Optional[float]:
    """Seconds from ``msg.header.stamp`` when present."""
    header = getattr(msg, "header", None)
    stamp = getattr(header, "stamp", None) if header is not None else None
    if stamp is None:
        return None
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def pointcloud2_to_xyzi(msg,
                        intensity_field: Optional[str] = "intensity",
                        intensity_scale: float = 1.0,
                        keep_nan: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """sensor_msgs/PointCloud2 -> ((N,3) float32 xyz, (N,) float32 intensity).

    Intensity is scaled by ``intensity_scale`` and clipped into [0, 1] later by
    the PCD writer.  Missing intensity fields yield zeros.
    """
    endian = "<" if not msg.is_bigendian else ">"
    offsets: Dict[str, Tuple[int, str, int]] = {}
    for field in msg.fields:
        if field.datatype not in _PF_TYPES:
            continue
        offsets[field.name] = (field.offset, _PF_TYPES[field.datatype],
                               max(1, field.count))
    for axis in ("x", "y", "z"):
        if axis not in offsets:
            raise ValueError("PointCloud2 without an '%s' field" % axis)

    n_points = int(msg.width) * int(msg.height)
    point_step = int(msg.point_step)
    raw = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    usable = (raw.size // point_step) if point_step else 0
    n_points = min(n_points, usable)
    raw = raw[:n_points * point_step].reshape(n_points, point_step)

    def column(name: str) -> np.ndarray:
        offset, kind, _ = offsets[name]
        width = int(kind[1])
        chunk = raw[:, offset:offset + width]
        return np.frombuffer(np.ascontiguousarray(chunk).tobytes(),
                             dtype=np.dtype(endian + kind)).astype(np.float32)

    xyz = np.stack([column("x"), column("y"), column("z")], axis=1)
    if intensity_field and intensity_field in offsets:
        intensity = column(intensity_field) * float(intensity_scale)
    else:
        intensity = np.zeros(xyz.shape[0], dtype=np.float32)

    if not keep_nan:
        finite = np.isfinite(xyz).all(axis=1)
        xyz, intensity = xyz[finite], intensity[finite]
    return np.ascontiguousarray(xyz), np.ascontiguousarray(intensity)


def image_to_array(msg) -> np.ndarray:
    """sensor_msgs/Image -> numpy array shaped (H, W) or (H, W, C)."""
    encoding = msg.encoding.lower()
    dtype_map = {
        "rgb8": (np.uint8, 3), "bgr8": (np.uint8, 3),
        "rgba8": (np.uint8, 4), "bgra8": (np.uint8, 4),
        "mono8": (np.uint8, 1), "8uc1": (np.uint8, 1), "8uc3": (np.uint8, 3),
        "mono16": (np.uint16, 1), "16uc1": (np.uint16, 1),
        "32fc1": (np.float32, 1),
        "bayer_rggb8": (np.uint8, 1), "bayer_bggr8": (np.uint8, 1),
        "bayer_gbrg8": (np.uint8, 1), "bayer_grbg8": (np.uint8, 1),
    }
    if encoding not in dtype_map:
        raise ValueError("unsupported image encoding: %s" % msg.encoding)
    dtype, channels = dtype_map[encoding]
    dtype = np.dtype(dtype).newbyteorder(">" if msg.is_bigendian else "<")
    height, width, step = int(msg.height), int(msg.width), int(msg.step)
    arr = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    arr = arr[:height * step].reshape(height, step)
    arr = arr[:, :width * channels * dtype.itemsize]
    arr = np.frombuffer(np.ascontiguousarray(arr).tobytes(), dtype=dtype)
    arr = arr.reshape(height, width, channels)
    return arr[:, :, 0] if channels == 1 else arr


def image_to_rgb(msg) -> np.ndarray:
    """sensor_msgs/Image -> (H, W, 3) uint8 RGB, for PNG dumping."""
    arr = image_to_array(msg)
    encoding = msg.encoding.lower()
    if arr.ndim == 2:
        if arr.dtype != np.uint8:
            finite = arr[np.isfinite(arr)] if arr.dtype.kind == "f" else arr
            top = float(finite.max()) if finite.size else 1.0
            arr = (np.nan_to_num(arr.astype(np.float32)) /
                   max(top, 1e-6) * 255.0).astype(np.uint8)
        return np.repeat(arr[:, :, None], 3, axis=2)
    if encoding in ("bgr8", "bgra8"):
        arr = arr[:, :, [2, 1, 0]]
    return np.ascontiguousarray(arr[:, :, :3]).astype(np.uint8)


def camera_info_to_intrinsic(msg) -> np.ndarray:
    """sensor_msgs/CameraInfo -> 3x3 intrinsic matrix."""
    k = np.asarray(list(msg.k if hasattr(msg, "k") else msg.K), dtype=float)
    return k.reshape(3, 3)


def depth_to_points(depth: np.ndarray,
                    intrinsic: np.ndarray,
                    depth_scale: float = 1e-3,
                    stride: int = 1,
                    min_range: float = 0.0,
                    max_range: float = float("inf")) -> np.ndarray:
    """Depth image -> (N,3) points in the *optical* frame (x right, y down, z fwd)."""
    if depth.ndim != 2:
        raise ValueError("depth image must be single channel")
    stride = max(1, int(stride))
    sub = depth[::stride, ::stride]
    fx, fy = float(intrinsic[0, 0]), float(intrinsic[1, 1])
    cx, cy = float(intrinsic[0, 2]), float(intrinsic[1, 2])
    if fx == 0.0 or fy == 0.0:
        raise ValueError("camera_info has a zero focal length")

    rows = np.arange(0, depth.shape[0], stride, dtype=np.float32)
    cols = np.arange(0, depth.shape[1], stride, dtype=np.float32)
    uu, vv = np.meshgrid(cols, rows)
    z = sub.astype(np.float32) * float(depth_scale)
    valid = np.isfinite(z) & (z > min_range) & (z < max_range)
    z = z[valid]
    x = (uu[valid] - cx) * z / fx
    y = (vv[valid] - cy) * z / fy
    return np.stack([x, y, z], axis=1).astype(np.float32)


# Optical (z forward, x right, y down) -> ROS body (x forward, y left, z up).
OPTICAL_TO_BODY = np.array([
    [0.0, 0.0, 1.0, 0.0],
    [-1.0, 0.0, 0.0, 0.0],
    [0.0, -1.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
])


def transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Apply a 4x4 transform to (N,3) points."""
    if points.size == 0:
        return points
    return (points @ matrix[:3, :3].T + matrix[:3, 3]).astype(np.float32)


def msg_to_plain(msg, depth: int = 3):
    """Recursively convert a decoded ROS message into plain YAML-able types."""
    if depth < 0:
        return None
    if isinstance(msg, (bool, int, float, str)) or msg is None:
        return msg
    if isinstance(msg, (bytes, bytearray)):
        return None
    if isinstance(msg, np.ndarray):
        return [float(v) for v in msg.reshape(-1)[:64]]
    if isinstance(msg, (list, tuple)):
        out = [msg_to_plain(v, depth - 1) for v in msg[:64]]
        return [v for v in out if v is not None]
    slots = getattr(msg, "__slots__", None)
    if slots is None:
        return None
    out = {}
    for slot in slots:
        name = slot[1:] if slot.startswith("_") else slot
        value = msg_to_plain(getattr(msg, name, None), depth - 1)
        if value is not None and value != []:
            out[name] = value
    return out or None
