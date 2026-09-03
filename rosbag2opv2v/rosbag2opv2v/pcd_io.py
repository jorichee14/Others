"""OPV2V-compatible PCD writing.

OpenCOOD reads point clouds with ``opencood.utils.pcd_utils.pcd_to_np``:

    pcd = o3d.io.read_point_cloud(pcd_file)
    xyz = np.asarray(pcd.points)
    intensity = np.expand_dims(np.asarray(pcd.colors)[:, 0], -1)

so the per-point intensity has to travel in the PCD ``rgb`` field, exactly the
way Open3D itself writes colours: one float32 whose bits hold a packed
0x00RRGGBB.  We replicate Open3D's own output (``FIELDS x y z rgb`` /
``TYPE F F F F``) instead of depending on Open3D, so the converter runs in a
plain ROS environment.  Note this quantises intensity to 8 bits -- same as real
OPV2V data, which is written by the same code path.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np

_HEADER = (
    "# .PCD v0.7 - Point Cloud Data file format\n"
    "VERSION 0.7\n"
    "FIELDS x y z rgb\n"
    "SIZE 4 4 4 4\n"
    "TYPE F F F F\n"
    "COUNT 1 1 1 1\n"
    "WIDTH {n}\n"
    "HEIGHT 1\n"
    "VIEWPOINT 0 0 0 1 0 0 0\n"
    "POINTS {n}\n"
    "DATA {mode}\n"
)


def _packed_rgb(intensity: Optional[np.ndarray], count: int) -> np.ndarray:
    if intensity is None:
        gray = np.zeros(count, dtype=np.uint32)
    else:
        clipped = np.clip(np.nan_to_num(intensity, nan=0.0), 0.0, 1.0)
        gray = np.round(clipped * 255.0).astype(np.uint32)
    return (gray << 16) | (gray << 8) | gray


def write_pcd(path: str,
              xyz: np.ndarray,
              intensity: Optional[np.ndarray] = None,
              binary: bool = True) -> int:
    """Write ``xyz`` (N,3) + optional ``intensity`` (N,) in [0, 1] as a PCD.

    Returns the number of points written.
    """
    xyz = np.asarray(xyz, dtype=np.float32).reshape(-1, 3)
    n = int(xyz.shape[0])
    if intensity is not None:
        intensity = np.asarray(intensity, dtype=np.float32).reshape(-1)
        if intensity.shape[0] != n:
            raise ValueError("intensity length does not match point count")
    packed = _packed_rgb(intensity, n)
    rgb_as_float = packed.astype(np.uint32).view(np.float32)

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    header = _HEADER.format(n=n, mode="binary" if binary else "ascii")
    if binary:
        buf = np.empty((n, 4), dtype=np.float32)
        buf[:, :3] = xyz
        buf[:, 3] = rgb_as_float
        with open(path, "wb") as handle:
            handle.write(header.encode("ascii"))
            handle.write(buf.tobytes())
    else:
        with open(path, "w") as handle:
            handle.write(header)
            for i in range(n):
                handle.write("%.6g %.6g %.6g %.9g\n" % (
                    xyz[i, 0], xyz[i, 1], xyz[i, 2], rgb_as_float[i]))
    return n


def read_pcd(path: str) -> np.ndarray:
    """Minimal reader for the format above -> (N, 4) [x, y, z, intensity].

    Mirrors what ``pcd_to_np`` gets out of Open3D, so the verifier can check the
    export without requiring Open3D to be installed.
    """
    with open(path, "rb") as handle:
        fields, size, types, count = [], [], [], []
        n_points, data_mode = 0, "ascii"
        while True:
            line = handle.readline()
            if not line:
                raise ValueError("unexpected end of PCD header: %s" % path)
            text = line.decode("ascii", errors="replace").strip()
            if not text or text.startswith("#"):
                continue
            key, _, rest = text.partition(" ")
            key = key.upper()
            if key == "FIELDS":
                fields = rest.split()
            elif key == "SIZE":
                size = [int(v) for v in rest.split()]
            elif key == "TYPE":
                types = rest.split()
            elif key == "COUNT":
                count = [int(v) for v in rest.split()]
            elif key == "POINTS":
                n_points = int(rest)
            elif key == "DATA":
                data_mode = rest.strip().lower()
                break
        if not count:
            count = [1] * len(fields)
        np_types = {("F", 4): "<f4", ("F", 8): "<f8", ("U", 1): "<u1",
                    ("U", 2): "<u2", ("U", 4): "<u4", ("I", 1): "<i1",
                    ("I", 2): "<i2", ("I", 4): "<i4"}
        dtype = np.dtype([
            (name, np_types[(t, s)], (c,) if c > 1 else ())
            for name, t, s, c in zip(fields, types, size, count)
        ])
        if data_mode == "binary":
            raw = handle.read(dtype.itemsize * n_points)
            arr = np.frombuffer(raw, dtype=dtype, count=n_points)
        elif data_mode == "ascii":
            text = handle.read().decode("ascii").split()
            values = np.asarray(text, dtype=np.float64)
            ncol = sum(count)
            values = values.reshape(-1, ncol)[:n_points]
            arr = np.empty(n_points, dtype=dtype)
            col = 0
            for name, c in zip(fields, count):
                block = values[:, col:col + c]
                if dtype[name].base.kind == "f" and dtype[name].base.itemsize == 4:
                    arr[name] = block.astype(np.float32).reshape(arr[name].shape)
                else:
                    arr[name] = block.astype(dtype[name].base).reshape(
                        arr[name].shape)
                col += c
        else:
            raise ValueError("unsupported PCD DATA mode: %s" % data_mode)

    xyz = np.stack([arr["x"], arr["y"], arr["z"]], axis=1).astype(np.float32)
    if "rgb" in fields:
        packed = np.ascontiguousarray(arr["rgb"]).astype(np.float32).view(np.uint32)
        intensity = ((packed >> 16) & 0xFF).astype(np.float32) / 255.0
    else:
        intensity = np.zeros(xyz.shape[0], dtype=np.float32)
    return np.hstack([xyz, intensity[:, None]]).astype(np.float32)
