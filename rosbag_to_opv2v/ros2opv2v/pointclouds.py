# -*- coding: utf-8 -*-
"""
Turning ROS 2 sensor messages into the (N, 4) ``x y z intensity`` arrays that
OPV2V stores per frame.

Two producers are supported, because only one of the three agents in a typical
robot testbed actually carries a LiDAR:

* ``pointcloud2`` — a ``sensor_msgs/PointCloud2`` passed through (Ouster, radar,
  stereo clouds). Field layout is read from the message, not assumed.
* ``depth_image`` — a ``sensor_msgs/Image`` reprojected through its
  ``CameraInfo`` intrinsics into a cloud (RealSense depth).

Both return points in the *sensor* frame; placing them on the robot is the
caller's job (see :mod:`ros2opv2v.convert`).
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .geometry import OPTICAL_TO_FLU, invert, transform_points

# sensor_msgs/PointField datatype enum -> numpy dtype
_PF_DTYPE = {
    1: np.int8, 2: np.uint8, 3: np.int16, 4: np.uint16,
    5: np.int32, 6: np.uint32, 7: np.float32, 8: np.float64,
}


class CloudError(RuntimeError):
    """Raised when a message cannot be interpreted as a point cloud."""


def _as_bytes(data) -> bytes:
    """Message payloads arrive as bytes, bytearray, memoryview or ndarray."""
    if isinstance(data, (bytes, bytearray, memoryview)):
        return bytes(data)
    arr = np.asarray(data)
    return arr.astype(np.uint8, copy=False).tobytes()


def pointcloud2_to_array(msg) -> np.ndarray:
    """Decode a ``sensor_msgs/PointCloud2`` into a structured numpy array.

    The dtype mirrors the message's own field layout (offsets and ``point_step``
    included), so padding bytes and exotic field orders are handled without
    guessing.
    """
    fields = []
    for pf in msg.fields:
        count = int(getattr(pf, "count", 1) or 1)
        dtype = _PF_DTYPE.get(int(pf.datatype))
        if dtype is None:
            continue                      # unknown datatype: skip that field
        shape = (count,) if count > 1 else ()
        fields.append((str(pf.name), dtype, shape, int(pf.offset)))

    if not fields:
        raise CloudError("PointCloud2 message declares no usable fields")

    names = [f[0] for f in fields]
    if len(set(names)) != len(names):     # duplicate names would break the dtype
        seen, unique = set(), []
        for name in names:
            base, i = name, 1
            while name in seen:
                name, i = f"{base}_{i}", i + 1
            seen.add(name)
            unique.append(name)
        names = unique

    point_step = int(msg.point_step)
    dtype = np.dtype({
        "names": names,
        "formats": [(f[1], f[2]) if f[2] else f[1] for f in fields],
        "offsets": [f[3] for f in fields],
        "itemsize": point_step,
    })
    if bool(getattr(msg, "is_bigendian", False)):
        dtype = dtype.newbyteorder(">")

    buf = _as_bytes(msg.data)
    n_points = int(msg.width) * int(msg.height)
    usable = min(n_points, len(buf) // point_step)
    if usable <= 0:
        return np.zeros(0, dtype=dtype)
    return np.frombuffer(buf, dtype=dtype, count=usable)


def cloud_from_pointcloud2(msg, intensity_cfg, time_field: Optional[str] = None):
    """``sensor_msgs/PointCloud2`` -> (N, 4) ``[x, y, z, intensity]`` float32.

    Non-finite points (``is_dense == False`` clouds are full of them) are dropped.

    With ``time_field`` set, returns ``(cloud, offsets_ns)`` instead: the per-point
    acquisition offsets from the message stamp, filtered by the *same* mask as the
    points. Recomputing that mask separately is how a deskew ends up applying the
    wrong correction to the wrong points, so the two are produced together or not
    at all. ``offsets_ns`` is ``None`` when the cloud carries no such field.
    """
    structured = pointcloud2_to_array(msg)
    if structured.size == 0:
        empty = np.zeros((0, 4), dtype=np.float32)
        return (empty, None) if time_field else empty

    names = structured.dtype.names
    for axis in ("x", "y", "z"):
        if axis not in names:
            raise CloudError(f"PointCloud2 has no '{axis}' field (has: {list(names)})")

    xyz = np.stack([structured["x"].astype(np.float64),
                    structured["y"].astype(np.float64),
                    structured["z"].astype(np.float64)], axis=-1)

    intensity = extract_intensity(structured, intensity_cfg, xyz.shape[0])

    finite = np.isfinite(xyz).all(axis=1) & np.isfinite(intensity)
    times = None
    if time_field and time_field in names:
        times = np.asarray(structured[time_field]).astype(np.float64)
        finite &= np.isfinite(times)
    xyz, intensity = xyz[finite], intensity[finite]
    if times is not None:
        times = times[finite]

    out = np.empty((xyz.shape[0], 4), dtype=np.float32)
    out[:, :3] = xyz
    out[:, 3] = np.clip(intensity, 0.0, 1.0)
    return (out, times) if time_field else out


def extract_intensity(structured: np.ndarray, cfg, n_points: int) -> np.ndarray:
    """Pull the configured attribute out of a structured cloud and scale it to [0, 1].

    Falls back to the configured constant whenever the field is absent — a radar
    cloud carrying only ``x y z`` is a normal input here, not an error.
    """
    name = getattr(cfg, "field_name", None)
    if not name:
        return np.full(n_points, float(cfg.default), dtype=np.float64)

    names = structured.dtype.names or ()
    if name not in names:
        # tolerate the usual spelling variations before giving up on the field
        lowered = {n.lower(): n for n in names}
        name = lowered.get(str(name).lower())
        if name is None:
            return np.full(n_points, float(cfg.default), dtype=np.float64)

    raw = structured[name]
    if raw.ndim > 1:
        raw = raw[:, 0]
    return raw.astype(np.float64) * float(cfg.scale) + float(cfg.offset)


def camera_intrinsics(camera_info) -> Tuple[float, float, float, float]:
    """``(fx, fy, cx, cy)`` from a ``sensor_msgs/CameraInfo``.

    Prefers ``k`` (the rectified intrinsic matrix); falls back to ``p`` for
    drivers that only populate the projection matrix.
    """
    k = getattr(camera_info, "k", None)
    if k is None:
        k = getattr(camera_info, "K", None)
    k = np.asarray(k, dtype=np.float64).reshape(-1) if k is not None else None

    if k is None or k.size < 9 or k[0] == 0.0:
        p = getattr(camera_info, "p", None) or getattr(camera_info, "P", None)
        if p is None:
            raise CloudError("CameraInfo has neither k nor p populated")
        p = np.asarray(p, dtype=np.float64).reshape(-1)
        return float(p[0]), float(p[5]), float(p[2]), float(p[6])

    return float(k[0]), float(k[4]), float(k[2]), float(k[5])


def depth_image_to_array(msg) -> np.ndarray:
    """``sensor_msgs/Image`` -> 2D float64 array of raw depth values."""
    encoding = str(msg.encoding).lower()
    height, width = int(msg.height), int(msg.width)
    buf = _as_bytes(msg.data)

    if encoding in ("16uc1", "mono16"):
        dtype, channels = np.uint16, 1
    elif encoding in ("32fc1",):
        dtype, channels = np.float32, 1
    elif encoding in ("mono8", "8uc1"):
        dtype, channels = np.uint8, 1
    else:
        raise CloudError(f"unsupported depth encoding {msg.encoding!r} "
                         f"(expected 16UC1, 32FC1, mono16 or mono8)")

    dtype = np.dtype(dtype)
    if bool(getattr(msg, "is_bigendian", False)):
        dtype = dtype.newbyteorder(">")

    step = int(getattr(msg, "step", 0)) or width * dtype.itemsize * channels
    rows = np.frombuffer(buf, dtype=np.uint8, count=height * step).reshape(height, step)
    rows = rows[:, :width * dtype.itemsize * channels]
    return np.ascontiguousarray(rows).view(dtype).reshape(height, width).astype(np.float64)


def cloud_from_depth_image(msg, camera_info, cfg) -> np.ndarray:
    """Reproject a depth image into a (N, 4) cloud in the *sensor body* frame.

    Depth cameras publish in the optical frame (x right, y down, z forward);
    with ``optical_frame: true`` the result is rotated into the ROS body
    convention (x forward, y left, z up) so a single body-frame extrinsic in the
    config places the sensor on the robot.
    """
    depth = depth_image_to_array(msg)
    fx, fy, cx, cy = camera_intrinsics(camera_info)
    if fx == 0.0 or fy == 0.0:
        raise CloudError("CameraInfo has zero focal length")

    stride = max(1, int(cfg.pixel_stride))
    depth = depth[::stride, ::stride]
    rows, cols = np.mgrid[0:depth.shape[0], 0:depth.shape[1]]
    u = cols * stride
    v = rows * stride

    encoding = str(msg.encoding).lower()
    metres = depth * (cfg.depth_scale if encoding in ("16uc1", "mono16", "mono8") else 1.0)

    valid = np.isfinite(metres) & (metres >= cfg.min_depth) & (metres <= cfg.max_depth)
    if not valid.any():
        return np.zeros((0, 4), dtype=np.float32)

    z = metres[valid]
    x = (u[valid] - cx) * z / fx
    y = (v[valid] - cy) * z / fy
    xyz = np.stack([x, y, z], axis=-1)

    if cfg.optical_frame:
        xyz = transform_points(xyz, OPTICAL_TO_FLU)

    out = np.empty((xyz.shape[0], 4), dtype=np.float32)
    out[:, :3] = xyz
    out[:, 3] = float(np.clip(cfg.intensity.default, 0.0, 1.0))
    return out


def apply_range_filter(cloud: np.ndarray, limits: Optional[list]) -> np.ndarray:
    """Crop a cloud to ``[xmin, ymin, zmin, xmax, ymax, zmax]`` in its own frame."""
    if limits is None or cloud.shape[0] == 0:
        return cloud
    x_min, y_min, z_min, x_max, y_max, z_max = [float(v) for v in limits]
    mask = ((cloud[:, 0] >= x_min) & (cloud[:, 0] <= x_max) &
            (cloud[:, 1] >= y_min) & (cloud[:, 1] <= y_max) &
            (cloud[:, 2] >= z_min) & (cloud[:, 2] <= z_max))
    return cloud[mask]


def subsample(cloud: np.ndarray, max_points: int) -> np.ndarray:
    """Deterministically thin a cloud to at most ``max_points`` points.

    Uses a fixed stride rather than random choice so a conversion is reproducible
    without carrying a seed around.
    """
    if max_points <= 0 or cloud.shape[0] <= max_points:
        return cloud
    idx = np.linspace(0, cloud.shape[0] - 1, num=max_points).astype(np.int64)
    return cloud[idx]


def deskew_cloud(cloud: np.ndarray, offsets_ns, stamp_ns: int, t_ref_ns: int,
                 track, sensor_from_base: np.ndarray, max_gap_ns: int,
                 buckets: int = 64):
    """Move every point to where it would have been observed at ``t_ref_ns``.

    ``docs/ROS2OPV2V.md`` used to list "no motion compensation" as a known
    limitation, on the grounds that OPV2V does not need it. OPV2V does not need it
    because its sweeps are simulated as instantaneous; a real 10 Hz spinning LiDAR
    observes over a ~100 ms window, which is the same order as the whole
    inter-agent synchronisation budget. Leaving it uncorrected means the ego's own
    cloud is smeared by its own motion, and the smear is *azimuth-dependent*, so it
    does not even look like noise.

    Only relative motion matters here, so the operator-supplied ``align`` transform
    cancels out and a wrong alignment cannot corrupt the deskew::

        p_ref = inv(odom_T_sensor(t_ref)) @ odom_T_sensor(t) @ p

    with ``odom_T_sensor(t) = track(t) @ sensor_from_base``.

    Points are bucketed by time and one rigid transform is applied per bucket: over
    a 100 ms sweep split 64 ways the bucketing residual is under 2 ms of ego motion
    (sub-millimetre at robot speeds), while a pose lookup per point would cost one
    per ~130,000 points.

    Returns ``(cloud, info)``. When the pose track cannot cover the sweep the cloud
    is returned **untouched** and ``info['applied']`` is False — a partially
    corrected cloud is worse than an uncorrected one, because it is no longer
    internally consistent.
    """
    if offsets_ns is None or cloud.shape[0] == 0:
        return cloud, {"applied": False, "reason": "no per-point time field"}

    times_ns = np.asarray(offsets_ns, dtype=np.float64) + float(stamp_ns)
    lo, hi = float(times_ns.min()), float(times_ns.max())
    span_ns = hi - lo
    if span_ns <= 0:
        # An instantaneous cloud still needs moving from its own stamp to the
        # frame time; only the per-point spread is absent. One bucket does that.
        buckets = 1

    reference = track.lookup(int(t_ref_ns), mode="linear", max_gap_ns=max_gap_ns)
    if reference is None:
        return cloud, {"applied": False, "reason": "no pose at the frame time"}
    odom_from_sensor_ref = reference[0] @ sensor_from_base
    ref_inverse = invert(odom_from_sensor_ref)

    edges = (np.linspace(lo, hi, buckets + 1) if span_ns > 0
             else np.array([lo, lo + 1.0]))
    index = np.clip(np.searchsorted(edges, times_ns, side="right") - 1, 0, buckets - 1)
    out = cloud.copy()
    used = 0
    for b in range(buckets):
        mask = index == b
        if not np.any(mask):
            continue
        found = track.lookup(int(0.5 * (edges[b] + edges[b + 1])), mode="linear",
                             max_gap_ns=max_gap_ns)
        if found is None:
            return cloud, {"applied": False, "reason": "pose gap inside the sweep"}
        relative = ref_inverse @ (found[0] @ sensor_from_base)
        out[mask, :3] = transform_points(cloud[mask, :3].astype(np.float64),
                                         relative).astype(np.float32)
        used += 1
    return out, {"applied": True, "buckets": used,
                 "sweep_span_ms": round(span_ns / 1e6, 3),
                 "bucket_ms": round(span_ns / 1e6 / buckets, 4)}
