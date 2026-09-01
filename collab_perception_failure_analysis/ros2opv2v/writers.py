# -*- coding: utf-8 -*-
"""
On-disk writers for the OPV2V layout.

The one non-obvious constraint is intensity.  OpenCOOD reads a frame with

    pcd = o3d.io.read_point_cloud(path)
    pcd_np = np.hstack((np.asarray(pcd.points),
                        np.asarray(pcd.colors)[:, 0:1]))     # <- intensity

so intensity has to travel in the PCD's *colour* channel.  We therefore write
``FIELDS x y z rgb`` with r = g = b = round(intensity * 255), which is exactly how
OPV2V's own files are laid out (and why OPV2V intensity is 8-bit quantised).  The
round trip through open3d is asserted in ``scripts/test_ros2opv2v.py``.
"""

from __future__ import annotations

import os
import struct
import zlib
from typing import Optional

import numpy as np
import yaml

PCD_HEADER = """# .PCD v0.7 - Point Cloud Data file format
VERSION 0.7
FIELDS x y z rgb
SIZE 4 4 4 4
TYPE F F F F
COUNT 1 1 1 1
WIDTH {n}
HEIGHT 1
VIEWPOINT 0 0 0 1 0 0 0
POINTS {n}
DATA binary
"""


def write_pcd(path: str, cloud: np.ndarray) -> int:
    """Write an (N, 4) ``[x, y, z, intensity]`` array as an OpenCOOD-readable PCD.

    Returns the number of points written.  Intensity is clipped to [0, 1] and
    packed into the rgb field as an 8-bit grey level.
    """
    cloud = np.asarray(cloud, dtype=np.float32)
    if cloud.ndim != 2 or cloud.shape[1] < 3:
        raise ValueError(f"cloud must be (N, >=3), got {cloud.shape}")

    n = cloud.shape[0]
    xyz = cloud[:, :3].astype("<f4")
    if cloud.shape[1] >= 4:
        intensity = np.clip(cloud[:, 3], 0.0, 1.0)
    else:
        intensity = np.zeros(n, dtype=np.float32)

    grey = np.round(intensity * 255.0).astype(np.uint32)
    packed = (grey << 16) | (grey << 8) | grey            # 0x00RRGGBB, r = g = b
    rgb = packed.astype("<u4").view("<f4")

    payload = np.empty((n, 4), dtype="<f4")
    payload[:, :3] = xyz
    payload[:, 3] = rgb

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as handle:
        handle.write(PCD_HEADER.format(n=n).encode("ascii"))
        handle.write(payload.tobytes(order="C"))
    return n


def read_pcd(path: str) -> np.ndarray:
    """Read back a PCD written by :func:`write_pcd` (used by the validator).

    Deliberately minimal — it understands the exact header this module writes,
    and refuses anything else rather than half-parsing an arbitrary PCD.
    """
    with open(path, "rb") as handle:
        header, fields, size, types, count, n = {}, None, None, None, None, 0
        while True:
            line = handle.readline()
            if not line:
                raise ValueError(f"{path}: truncated PCD header")
            text = line.decode("ascii", errors="replace").strip()
            if text.startswith("#") or not text:
                continue
            key, _, value = text.partition(" ")
            header[key.upper()] = value.strip()
            if key.upper() == "DATA":
                break

        if header.get("DATA") != "binary":
            raise ValueError(f"{path}: expected DATA binary, got {header.get('DATA')!r}")
        fields = header.get("FIELDS", "").split()
        if fields != ["x", "y", "z", "rgb"]:
            raise ValueError(f"{path}: expected FIELDS x y z rgb, got {fields}")
        n = int(header.get("POINTS", header.get("WIDTH", 0)))

        raw = handle.read(n * 16)

    payload = np.frombuffer(raw, dtype="<f4", count=n * 4).reshape(n, 4)
    grey = (payload[:, 3].copy().view("<u4") & 0xFF).astype(np.float32) / 255.0
    out = np.empty((n, 4), dtype=np.float32)
    out[:, :3] = payload[:, :3]
    out[:, 3] = grey
    return out


class _Dumper(yaml.SafeDumper):
    """Plain-flow lists keep the frame yaml close to OPV2V's own formatting."""


def _represent_list(dumper, data):
    flow = all(isinstance(v, (int, float, str)) for v in data) and len(data) <= 16
    return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=flow)


_Dumper.add_representer(list, _represent_list)


def _plain(value):
    """Strip numpy scalars/arrays so the yaml stays loadable by plain PyYAML."""
    if isinstance(value, dict):
        return {(_plain(k) if not isinstance(k, str) else k): _plain(v)
                for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        return float(round(value, 8))
    return value


def write_frame_yaml(path: str, params: dict) -> None:
    """Write one frame's parameter file.

    OpenCOOD loads these with a Loader whose implicit float resolver is patched,
    so ordinary PyYAML output (including scientific notation) round-trips fine.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as handle:
        yaml.dump(_plain(params), handle, Dumper=_Dumper,
                  default_flow_style=False, sort_keys=False)


def write_png(path: str, image: np.ndarray) -> None:
    """Write an 8-bit greyscale or RGB PNG with no third-party dependency."""
    array = np.asarray(image)
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    if array.ndim == 2:
        array = array[:, :, None]
    height, width, channels = array.shape
    if channels not in (1, 3, 4):
        raise ValueError(f"PNG needs 1, 3 or 4 channels, got {channels}")
    colour_type = {1: 0, 3: 2, 4: 6}[channels]

    raw = b"".join(b"\x00" + array[row].tobytes() for row in range(height))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + tag + data +
                struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

    ihdr = struct.pack(">IIBBBBB", width, height, 8, colour_type, 0, 0, 0)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as handle:
        handle.write(b"\x89PNG\r\n\x1a\n")
        handle.write(chunk(b"IHDR", ihdr))
        handle.write(chunk(b"IDAT", zlib.compress(raw, 6)))
        handle.write(chunk(b"IEND", b""))


def image_to_array(msg) -> Optional[np.ndarray]:
    """Decode a ``sensor_msgs/Image`` into a uint8 array suitable for PNG export."""
    encoding = str(msg.encoding).lower()
    height, width = int(msg.height), int(msg.width)
    data = msg.data
    if isinstance(data, (bytes, bytearray, memoryview)):
        buf = np.frombuffer(bytes(data), dtype=np.uint8)
    else:
        buf = np.asarray(data, dtype=np.uint8)

    step = int(getattr(msg, "step", 0)) or buf.size // max(height, 1)
    rows = buf[:height * step].reshape(height, step)

    if encoding in ("rgb8", "bgr8"):
        out = rows[:, :width * 3].reshape(height, width, 3)
        return out[:, :, ::-1] if encoding == "bgr8" else out
    if encoding in ("rgba8", "bgra8"):
        out = rows[:, :width * 4].reshape(height, width, 4)
        return out[:, :, [2, 1, 0, 3]] if encoding == "bgra8" else out
    if encoding in ("mono8", "8uc1"):
        return rows[:, :width].reshape(height, width)
    if encoding in ("mono16", "16uc1"):
        view = np.ascontiguousarray(rows[:, :width * 2]).view(np.uint16).reshape(height, width)
        top = float(view.max()) or 1.0
        return (view.astype(np.float64) * (255.0 / top)).astype(np.uint8)
    return None
