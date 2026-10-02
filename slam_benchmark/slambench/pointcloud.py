"""Point clouds: PLY/PCD I/O, voxel downsampling, and a voxel-hash nearest
neighbour. numpy only — open3d is a large dependency to make a comparison
script wait on, and the three operations this harness needs are short.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

__all__ = ["load_cloud", "save_ply", "voxel_downsample", "crop_box",
           "nearest_distances", "transform_points"]


def load_cloud(path: str | Path) -> np.ndarray:
    """Read a .ply or .pcd into an (N, 3) float64 array of xyz. Other fields are
    dropped: nothing downstream scores intensity."""
    path = Path(path)
    if path.suffix.lower() == ".ply":
        return _load_ply(path)
    if path.suffix.lower() == ".pcd":
        return _load_pcd(path)
    raise ValueError(f"{path}: expected .ply or .pcd")


def _dtype_of(fields: list[tuple[str, str]], little: bool) -> np.dtype:
    m = {"float": "f4", "float32": "f4", "float64": "f8", "double": "f8",
         "uchar": "u1", "uint8": "u1", "char": "i1", "int8": "i1",
         "ushort": "u2", "uint16": "u2", "short": "i2", "int16": "i2",
         "uint": "u4", "uint32": "u4", "int": "i4", "int32": "i4"}
    pre = "<" if little else ">"
    return np.dtype([(n, pre + m[t]) for t, n in fields])


def _load_ply(path: Path) -> np.ndarray:
    raw = path.read_bytes()
    end = raw.find(b"end_header")
    if end < 0:
        raise ValueError(f"{path}: no end_header — not a PLY")
    header = raw[:end].decode("ascii", "replace")
    body = raw[raw.find(b"\n", end) + 1:]

    fmt = re.search(r"format\s+(\S+)", header)
    fmt = fmt.group(1) if fmt else "ascii"
    n = int(re.search(r"element\s+vertex\s+(\d+)", header).group(1))
    fields = re.findall(r"property\s+(\S+)\s+(\S+)", header)
    names = [f[1] for f in fields]
    for axis in ("x", "y", "z"):
        if axis not in names:
            raise ValueError(f"{path}: PLY vertex element has no '{axis}' property")
    if n == 0:
        return np.zeros((0, 3))

    if fmt == "ascii":
        rows = np.loadtxt(body.decode("ascii", "replace").splitlines()[:n], ndmin=2)
        idx = [names.index(a) for a in ("x", "y", "z")]
        return np.ascontiguousarray(rows[:, idx], dtype=np.float64)
    if fmt not in ("binary_little_endian", "binary_big_endian"):
        raise ValueError(f"{path}: unsupported PLY format {fmt!r}")
    dt = _dtype_of(fields, fmt.endswith("little_endian"))
    arr = np.frombuffer(body, dtype=dt, count=n)
    return np.stack([arr["x"], arr["y"], arr["z"]], axis=1).astype(np.float64)


def _load_pcd(path: Path) -> np.ndarray:
    raw = path.read_bytes()
    head_end = raw.find(b"DATA")
    line_end = raw.find(b"\n", head_end)
    header = raw[:line_end].decode("ascii", "replace")
    body = raw[line_end + 1:]
    get = lambda k: re.search(rf"^{k}\s+(.*)$", header, re.M).group(1).split()
    names, sizes, types = get("FIELDS"), get("SIZE"), get("TYPE")
    counts = get("COUNT") if re.search(r"^COUNT", header, re.M) else ["1"] * len(names)
    n = int(get("POINTS")[0]) if re.search(r"^POINTS", header, re.M) else \
        int(get("WIDTH")[0]) * int(get("HEIGHT")[0])
    data = get("DATA")[0]
    for axis in ("x", "y", "z"):
        if axis not in names:
            raise ValueError(f"{path}: PCD has no '{axis}' field")
    if n == 0:
        return np.zeros((0, 3))

    if data == "ascii":
        rows = np.loadtxt(body.decode("ascii", "replace").splitlines()[:n], ndmin=2)
        idx = [names.index(a) for a in ("x", "y", "z")]
        return np.ascontiguousarray(rows[:, idx], dtype=np.float64)
    if data != "binary":
        raise ValueError(f"{path}: PCD DATA {data!r} is not supported "
                         "(binary_compressed: convert with pcl_convert_pcd_ascii_binary)")
    kind = {"F": "f", "U": "u", "I": "i"}
    dt = np.dtype([(nm, f"<{kind[t]}{s}", int(c)) if int(c) > 1 else (nm, f"<{kind[t]}{s}")
                   for nm, s, t, c in zip(names, sizes, types, counts)])
    arr = np.frombuffer(body, dtype=dt, count=n)
    return np.stack([arr["x"], arr["y"], arr["z"]], axis=1).astype(np.float64)


def save_ply(points: np.ndarray, path: str | Path) -> None:
    points = np.ascontiguousarray(np.asarray(points, dtype=np.float32).reshape(-1, 3))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = ("ply\nformat binary_little_endian 1.0\n"
              f"element vertex {len(points)}\n"
              "property float x\nproperty float y\nproperty float z\n"
              "end_header\n").encode("ascii")
    path.write_bytes(header + points.tobytes())


def transform_points(points: np.ndarray, T: np.ndarray) -> np.ndarray:
    return points @ np.asarray(T)[:3, :3].T + np.asarray(T)[:3, 3]


def crop_box(points: np.ndarray, lo, hi) -> np.ndarray:
    """Keep points inside an axis-aligned box. The evaluation volume V."""
    lo, hi = np.asarray(lo, dtype=np.float64), np.asarray(hi, dtype=np.float64)
    m = np.all((points >= lo) & (points <= hi), axis=1)
    return points[m]


def voxel_downsample(points: np.ndarray, voxel_m: float) -> np.ndarray:
    """One point per occupied voxel, at the voxel's centroid.

    Both clouds are downsampled at the same resolution before any distance is
    computed. Without it, accuracy and completeness are density statistics: a
    method that returns ten times the points scores better on completeness for
    no geometric reason.
    """
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if voxel_m <= 0 or not len(points):
        return points
    key = np.floor(points / voxel_m).astype(np.int64)
    _, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    inv = inv.reshape(-1)
    out = np.zeros((len(cnt), 3), dtype=np.float64)
    np.add.at(out, inv, points)
    return out / cnt[:, None]


def nearest_distances(query: np.ndarray, target: np.ndarray, truncate_m: float,
                      ) -> tuple[np.ndarray, float]:
    """For each query point, the distance to the nearest target point, truncated.

    Returns (distances, truncated_fraction). Implemented on a voxel hash at
    edge `truncate_m`: any target point within `truncate_m` of a query lies in
    the query's own voxel or one of its 26 neighbours, so the 3x3x3 search is
    exact up to the truncation and nothing beyond it is searched for. The
    truncated fraction is returned rather than hidden, because on a partial
    reconstruction it IS the result: a mean distance computed over a cloud that
    is 40% truncated is not a distance, it is a coverage number in disguise.
    """
    query = np.asarray(query, dtype=np.float64).reshape(-1, 3)
    target = np.asarray(target, dtype=np.float64).reshape(-1, 3)
    if not len(query):
        return np.zeros(0), 0.0
    if not len(target):
        return np.full(len(query), truncate_m), 1.0

    s = float(truncate_m)
    tkey = np.floor(target / s).astype(np.int64)
    order = np.lexsort((tkey[:, 2], tkey[:, 1], tkey[:, 0]))
    tkey_s, tgt_s = tkey[order], target[order]
    cells, start = np.unique(tkey_s, axis=0, return_index=True)
    stop = np.append(start[1:], len(tgt_s))
    lut = {(int(a), int(b), int(c)): (int(i), int(j))
           for (a, b, c), i, j in zip(cells, start, stop)}

    qkey = np.floor(query / s).astype(np.int64)
    out = np.full(len(query), s, dtype=np.float64)
    offs = [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)]
    for i in range(len(query)):
        a, b, c = int(qkey[i, 0]), int(qkey[i, 1]), int(qkey[i, 2])
        best = s * s
        for dx, dy, dz in offs:
            span = lut.get((a + dx, b + dy, c + dz))
            if span is None:
                continue
            d2 = np.sum((tgt_s[span[0]:span[1]] - query[i]) ** 2, axis=1)
            m = d2.min()
            if m < best:
                best = float(m)
        out[i] = np.sqrt(best)
    return out, float(np.mean(out >= s - 1e-12))
