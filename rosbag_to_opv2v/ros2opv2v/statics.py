# -*- coding: utf-8 -*-
"""
Static objects labelled once in the map frame, correct in every frame.

A bag carries no annotations, and hand-labelling 1330 frames to find two chairs
that never move is 1330 times the work the problem deserves. A static object has
ONE pose for the whole recording, so it is labelled once — in the shared world
frame — and the converter writes it into every agent's every frame, already
projected, because every frame's pose is known.

That also makes it the better ground truth than an agent-derived box. A box
derived from an agent's own pose is right by construction and therefore tests
nothing; a chair boxed by hand against the accumulated map is an INDEPENDENT
fact, so "do both agents' clouds land inside it" becomes a real check on the
extrinsics, the anchoring and the synchronisation at once.

This module is the geometry, with no I/O policy and no interactive dependency:

    read_pcd_xyz      a general PCD reader (ascii / binary / binary_compressed),
                      because the map cloud comes from the mapping pipeline and
                      is not written by this package
    ground_level      the dominant horizontal surface, which is also the number
                      `cloud.ground_lift` needs and that no pose can supply
    cluster_at        the points of the object standing at a seed position
    fit_box           the tightest ground-aligned oriented box around them
    points_in_box     the verification: whose returns actually fall inside

Everything is in the map frame and stays there. Projection into each agent is
the converter's job and is already exact.
"""

from __future__ import annotations

import math
import re
import struct
from typing import Dict, List, Optional, Tuple

import numpy as np


class StaticsError(RuntimeError):
    pass


# --------------------------------------------------------------------- reading
def _lzf_decompress(data: bytes, expected: int) -> bytes:
    """PCL writes `binary_compressed` with LZF, and neither numpy nor the stdlib
    can read it. Forty lines here beats telling the operator to convert the file
    with a tool they may not have."""
    out = bytearray()
    i = 0
    n = len(data)
    while i < n:
        ctrl = data[i]
        i += 1
        if ctrl < 32:                                   # literal run
            count = ctrl + 1
            out += data[i:i + count]
            i += count
        else:                                           # back reference
            length = ctrl >> 5
            if length == 7:
                length += data[i]
                i += 1
            ref = len(out) - ((ctrl & 0x1F) << 8) - data[i] - 1
            i += 1
            if ref < 0:
                raise StaticsError("corrupt LZF stream in the PCD")
            for _ in range(length + 2):
                out.append(out[ref])
                ref += 1
    if expected and len(out) != expected:
        raise StaticsError(f"LZF decompressed to {len(out)} bytes, header says {expected}")
    return bytes(out)


_PCD_NUMPY = {("F", 4): "<f4", ("F", 8): "<f8", ("U", 1): "<u1", ("U", 2): "<u2",
              ("U", 4): "<u4", ("U", 8): "<u8", ("I", 1): "<i1", ("I", 2): "<i2",
              ("I", 4): "<i4", ("I", 8): "<i8"}


def read_pcd_xyz(path: str, max_points: int = 0) -> np.ndarray:
    """Any PCD -> (N, 3) float64 xyz, non-finite points dropped.

    Handles ascii, binary and binary_compressed with arbitrary field layouts,
    since this file comes from the mapping pipeline rather than from this
    package's own writer.
    """
    header: Dict[str, str] = {}
    with open(path, "rb") as handle:
        while True:
            line = handle.readline()
            if not line:
                raise StaticsError(f"{path}: truncated PCD header")
            text = line.decode("ascii", errors="replace").strip()
            if not text or text.startswith("#"):
                continue
            key, _, value = text.partition(" ")
            header[key.upper()] = value.strip()
            if key.upper() == "DATA":
                break
        body = handle.read()

    fields = header.get("FIELDS", "").split()
    sizes = [int(v) for v in header.get("SIZE", "").split()]
    types = header.get("TYPE", "").split()
    counts = [int(v) for v in header.get("COUNT", " ".join("1" * len(fields))).split()] \
        or [1] * len(fields)
    n = int(header.get("POINTS") or (int(header.get("WIDTH", 0)) * int(header.get("HEIGHT", 1))))
    layout = header.get("DATA", "").lower()
    for axis in ("x", "y", "z"):
        if axis not in fields:
            raise StaticsError(f"{path}: PCD has no '{axis}' field (has {fields})")

    if layout == "ascii":
        rows = np.fromstring(body.decode("ascii", "replace").replace("nan", "nan"),
                             sep=" ") if False else np.array(
            [line.split() for line in body.decode("ascii", "replace").splitlines() if line.strip()],
            dtype=object)
        cols = {name: i for i, name in enumerate(fields)}
        xyz = np.array([[float(r[cols["x"]]), float(r[cols["y"]]), float(r[cols["z"]])]
                        for r in rows[:n]], dtype=np.float64)
    else:
        if layout == "binary_compressed":
            if len(body) < 8:
                raise StaticsError(f"{path}: truncated binary_compressed payload")
            compressed_size, uncompressed_size = struct.unpack_from("<II", body, 0)
            blob = _lzf_decompress(body[8:8 + compressed_size], uncompressed_size)
            # binary_compressed is COLUMN-major: all x, then all y, ...
            xyz_cols, offset = {}, 0
            for name, size, kind, count in zip(fields, sizes, types, counts):
                width = size * count * n
                if name in ("x", "y", "z"):
                    dtype = _PCD_NUMPY.get((kind.upper(), size))
                    if dtype is None:
                        raise StaticsError(f"{path}: unsupported field type {kind}{size}")
                    xyz_cols[name] = np.frombuffer(blob, dtype=dtype, count=n,
                                                   offset=offset).astype(np.float64)
                offset += width
            xyz = np.stack([xyz_cols["x"], xyz_cols["y"], xyz_cols["z"]], axis=1)
        else:
            dtype = np.dtype({
                "names": fields,
                "formats": [(_PCD_NUMPY[(k.upper(), s)], c) if c > 1
                            else _PCD_NUMPY[(k.upper(), s)]
                            for k, s, c in zip(types, sizes, counts)],
            })
            record = np.frombuffer(body, dtype=dtype, count=min(n, len(body) // dtype.itemsize))
            xyz = np.stack([record["x"].astype(np.float64).reshape(-1),
                            record["y"].astype(np.float64).reshape(-1),
                            record["z"].astype(np.float64).reshape(-1)], axis=1)

    xyz = xyz[np.isfinite(xyz).all(axis=1)]
    if max_points and len(xyz) > max_points:
        xyz = xyz[np.linspace(0, len(xyz) - 1, max_points).astype(int)]
    return xyz


# ---------------------------------------------------------------- ground level
def ground_level(points: np.ndarray, bin_m: float = 0.05,
                 percentile: float = 5.0) -> Tuple[float, dict]:
    """The z of the dominant horizontal surface — the floor.

    Worth having for two reasons. It is what separates an object from the floor
    it stands on, so every box below depends on it. And it is exactly the number
    `cloud.ground_lift` needs and that nothing else in the dataset supplies: an
    anchored map puts z = 0 at whatever anchored it (a surveyed board, here), not
    at the ground, so the floor's height is only knowable from the geometry.

    Taken as the mode of the z histogram within the lowest part of the cloud,
    which is robust to a ceiling that is also flat and also densely sampled.
    """
    z = points[:, 2]
    if z.size < 100:
        raise StaticsError("too few points to find a ground level")
    low = np.percentile(z, percentile)
    band = z[z < low + 1.0]
    edges = np.arange(band.min(), band.max() + bin_m, bin_m)
    if edges.size < 2:
        return float(np.median(band)), {"n": int(band.size), "method": "median"}
    hist, _ = np.histogram(band, bins=edges)
    peak = int(np.argmax(hist))
    centre = float(0.5 * (edges[peak] + edges[peak + 1]))
    return centre, {
        "n_in_band": int(band.size),
        "peak_count": int(hist[peak]),
        "z_p1": round(float(np.percentile(z, 1)), 3),
        "z_p99": round(float(np.percentile(z, 99)), 3),
        "method": "histogram mode of the lowest metre",
    }


# ------------------------------------------------------------------ clustering
def cluster_at(points: np.ndarray, seed: np.ndarray, radius: float = 1.2,
               voxel: float = 0.06, z_min: Optional[float] = None,
               z_max: Optional[float] = None) -> Tuple[np.ndarray, dict]:
    """The connected blob of points standing at `seed`.

    Voxel connected-components rather than a KD-tree DBSCAN: no scipy, exactly
    reproducible, and the voxel size is the same knob as the clustering
    tolerance. The seed only has to land somewhere on the object — the component
    it belongs to is what gets returned, so a click near the middle is enough.
    """
    seed = np.asarray(seed, dtype=np.float64).reshape(-1)
    keep = np.linalg.norm(points[:, :2] - seed[:2], axis=1) <= radius
    if z_min is not None:
        keep &= points[:, 2] >= z_min
    if z_max is not None:
        keep &= points[:, 2] <= z_max
    local = points[keep]
    if len(local) < 20:
        raise StaticsError(
            f"only {len(local)} points within {radius} m of {np.round(seed, 2).tolist()} "
            f"(after the height gate) — wrong seed, wrong frame, or the object is not there")

    grid = np.floor(local / voxel).astype(np.int64)
    grid -= grid.min(axis=0)
    keys = (grid[:, 0].astype(np.int64) << 42) | (grid[:, 1].astype(np.int64) << 21) | grid[:, 2]
    unique, inverse = np.unique(keys, return_inverse=True)
    lookup = {int(k): i for i, k in enumerate(unique)}
    coords = np.stack([(unique >> 42) & 0x1FFFFF, (unique >> 21) & 0x1FFFFF,
                       unique & 0x1FFFFF], axis=1)

    # 26-neighbourhood flood fill from the voxel nearest the seed
    offsets = [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)
               if (dx, dy, dz) != (0, 0, 0)]
    start = int(np.argmin(np.linalg.norm(
        coords * voxel + local.min(axis=0) - seed, axis=1)))
    seen = {start}
    stack = [start]
    while stack:
        current = stack.pop()
        cx, cy, cz = coords[current]
        for dx, dy, dz in offsets:
            key = int(((cx + dx) << 42) | ((cy + dy) << 21) | (cz + dz))
            nxt = lookup.get(key)
            if nxt is not None and nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    member = np.zeros(len(unique), dtype=bool)
    member[list(seen)] = True
    cluster = local[member[inverse]]
    return cluster, {
        "points_in_radius": int(len(local)),
        "points_in_cluster": int(len(cluster)),
        "voxels": int(member.sum()),
        "voxel_m": voxel,
    }


# --------------------------------------------------------------- box fitting
def _convex_hull_2d(pts: np.ndarray) -> np.ndarray:
    """Andrew's monotone chain, counter-clockwise, no repeated endpoint."""
    order = np.lexsort((pts[:, 1], pts[:, 0]))
    ordered = pts[order]
    if len(ordered) < 3:
        return ordered

    def half(seq):
        out: List[np.ndarray] = []
        for p in seq:
            while len(out) >= 2:
                a, b = out[-2], out[-1]
                if (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) <= 0:
                    out.pop()
                else:
                    break
            out.append(p)
        return out

    lower = half(ordered)
    upper = half(ordered[::-1])
    return np.array(lower[:-1] + upper[:-1])


def fit_box(cluster: np.ndarray, ground_z: Optional[float] = None,
            sit_on_ground: bool = True) -> dict:
    """The tightest ground-aligned oriented box around a cluster.

    Yaw comes from the minimum-area rectangle of the footprint (rotating
    calipers on the convex hull), which is exact rather than iterative. Roll and
    pitch are held at zero: a chair stands on the floor, and letting a
    least-squares fit tilt the box only lets sampling noise tilt it.

    With `ground_z`, the box is extended down to the floor. A LiDAR sees a chair's
    seat and back and almost none of its legs, so a box fitted to the returns
    alone floats, and every IoU against it is then wrong in the same direction.

    Returns the OPV2V-shaped fields: `location` (centre, map frame), `extent`
    (HALF dimensions) and `angle` ([roll, yaw, pitch] degrees, as OpenCOOD reads
    them).
    """
    if len(cluster) < 8:
        raise StaticsError(f"only {len(cluster)} points — too few to fit a box")
    hull = _convex_hull_2d(cluster[:, :2])
    best = None
    for i in range(len(hull)):
        edge = hull[(i + 1) % len(hull)] - hull[i]
        norm = float(np.hypot(*edge))
        if norm < 1e-9:
            continue
        axis = edge / norm
        rot = np.array([[axis[0], axis[1]], [-axis[1], axis[0]]])
        local = cluster[:, :2] @ rot.T
        lo, hi = local.min(axis=0), local.max(axis=0)
        area = float((hi[0] - lo[0]) * (hi[1] - lo[1]))
        if best is None or area < best[0]:
            best = (area, rot, lo, hi)
    if best is None:
        raise StaticsError("degenerate footprint — all points collinear")

    _area, rot, lo, hi = best
    centre_local = 0.5 * (lo + hi)
    centre_xy = centre_local @ rot                       # rot is orthonormal: R^T = R^-1
    size_xy = hi - lo
    yaw = math.degrees(math.atan2(rot[0, 1], rot[0, 0]))

    z_hi = float(cluster[:, 2].max())
    z_lo = float(cluster[:, 2].min())
    if ground_z is not None and sit_on_ground:
        z_lo = min(z_lo, float(ground_z))
    # The longer footprint axis is conventionally the box's x: it keeps `extent`
    # readable as [half-length, half-width, half-height] whichever hull edge won.
    if size_xy[1] > size_xy[0]:
        size_xy = size_xy[::-1]
        yaw += 90.0
    yaw = (yaw + 180.0) % 360.0 - 180.0

    return {
        "location": [round(float(centre_xy[0]), 4), round(float(centre_xy[1]), 4),
                     round(0.5 * (z_lo + z_hi), 4)],
        "extent": [round(float(size_xy[0]) / 2, 4), round(float(size_xy[1]) / 2, 4),
                   round((z_hi - z_lo) / 2, 4)],
        "angle": [0.0, round(yaw, 3), 0.0],
        "fit": {
            "points": int(len(cluster)),
            "footprint_m": [round(float(size_xy[0]), 3), round(float(size_xy[1]), 3)],
            "height_m": round(z_hi - z_lo, 3),
            "z_range": [round(z_lo, 3), round(z_hi, 3)],
            "extended_to_ground": bool(ground_z is not None and sit_on_ground
                                       and z_lo < cluster[:, 2].min() - 1e-9),
        },
    }


def box_corners(box: dict) -> np.ndarray:
    """The eight corners of a labelled box, map frame — for drawing and for tests."""
    yaw = math.radians(box["angle"][1])
    cos, sin = math.cos(yaw), math.sin(yaw)
    dx, dy, dz = box["extent"]
    signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)],
                     dtype=np.float64)
    local = signs * np.array([dx, dy, dz])
    rot = np.array([[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]])
    return local @ rot.T + np.asarray(box["location"], dtype=np.float64)


def points_in_box(points: np.ndarray, box: dict, margin: float = 0.0) -> np.ndarray:
    """Boolean mask of the points inside a labelled box (map frame).

    The verification the whole approach rests on: a hand-placed box is only
    ground truth if the agents' clouds actually fall inside it. If the ego's
    points land in it and a collaborator's do not while the collaborator is
    looking straight at it, the fault is upstream — extrinsics, anchoring or
    synchronisation — and this is what says so.
    """
    if len(points) == 0:
        return np.zeros(0, dtype=bool)
    yaw = math.radians(box["angle"][1])
    cos, sin = math.cos(yaw), math.sin(yaw)
    rel = np.asarray(points, dtype=np.float64) - np.asarray(box["location"], dtype=np.float64)
    local = np.stack([rel[:, 0] * cos + rel[:, 1] * sin,
                      -rel[:, 0] * sin + rel[:, 1] * cos,
                      rel[:, 2]], axis=1)
    limits = np.asarray(box["extent"], dtype=np.float64) + margin
    return (np.abs(local) <= limits).all(axis=1)
