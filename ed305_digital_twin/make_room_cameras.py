#!/usr/bin/env python3
"""Cameras for the multi-camera tools, from a calibration in a LiDAR map.

    python make_room_cameras.py --extrinsics .../mapping/cameras_ed305/cameras.yaml \\
        --intrinsics .../raw/20261105/calibration --floor-pcd .../mapping/map.pcd
    python make_room_cameras.py --extrinsics cameras.yaml --intrinsics calib/ --floor-z -0.52

--extrinsics   T_map_cam per camera ("transforms: - parent: map, child: cam_N,
               translation, rotation_xyzw"), camera frame OpenCV (x right, y down, z forward)
--intrinsics   folder searched (recursively) for cam_N.json / cam_N.yaml with K and
               distortion (calibration dashboard results, the March 2026 files, or a
               ROS camera_info); of cam_N_2, cam_N_3 ... the newest round is used
--floor-map    the LiDAR map (.pcd or .ply, ascii or binary): the floor PLANE is fitted
               to it (RANSAC, then least squares), so a map that is slightly tilted is
               levelled too (recommended), or
--floor-z      the floor's height in the map as one number (assumes the map is level)

Writes configs/cameras_room.yaml (this repo's format: K, dist, size, T_world_cam):
the room frame is the map turned so the floor plane is level, and moved so the
floor is at z = 0 (x, y stay close to the map's). That file is --calib for
multiview_label.py and room_track.py. Prints per camera its height above the
floor, its tilt and where it looks at on the floor: check them against the room.
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--extrinsics", type=Path, required=True)
    ap.add_argument("--intrinsics", type=Path, required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--floor-z", type=float, help="floor height in the map frame (m)")
    g.add_argument("--floor-map", "--floor-pcd", dest="floor_map", type=Path,
                   help="LiDAR map (.pcd or .ply) to fit the floor plane in")
    ap.add_argument("--out", type=Path, default=HERE / "configs/cameras_room.yaml")
    return ap.parse_args()


def quat_to_R(q):
    x, y, z, w = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def read_intrinsics(path):
    """K (3x3), dist (5,), width, height from the formats the calibration tools write."""
    d = json.loads(path.read_text()) if path.suffix == ".json" else yaml.safe_load(path.read_text())

    def first(*keys):
        for k in keys:
            if k in d and d[k] is not None:
                return d[k]
        return None

    K = first("K", "camera_matrix", "k", "intrinsic")
    if isinstance(K, dict):
        K = K.get("data")
    if K is None and "fx" in d and "cx" in d:
        K = [[d["fx"], 0, d["cx"]], [0, d["fy"], d["cy"]], [0, 0, 1]]
    dist = first("dist", "dist_coeffs", "distortion_coefficients", "distortion", "d", "D")
    if isinstance(dist, dict):
        dist = dist.get("data")
    w = first("img_w", "width", "image_width")
    h = first("img_h", "height", "image_height")
    size = first("image_size", "img_size", "resolution")
    if size is not None and (w is None or h is None):
        w, h = (size["width"], size["height"]) if isinstance(size, dict) else (size[0], size[1])
    if K is None:
        raise ValueError(f"{path}: no K / camera_matrix")
    K = np.asarray(K, float).reshape(3, 3)
    dist = np.zeros(5) if dist is None else np.asarray(dist, float).ravel()
    if dist.size > 5:
        raise ValueError(f"{path}: {dist.size} distortion coefficients; the tools use k1 k2 p1 p2 k3 only")
    dist = np.r_[dist, np.zeros(5 - dist.size)]
    return K, dist, int(w) if w else None, int(h) if h else None


def find_intrinsics(folder, n):
    """cam_N.json/.yaml (or cam_N_<round>), newest round; not extrinsic files."""
    best, best_round = None, -1
    for p in folder.rglob("*"):
        if p.suffix not in (".json", ".yaml", ".yml") or "extrinsic" in p.name.lower():
            continue
        m = re.fullmatch(rf"cam_?{n}(?:_(\d+))?", p.stem, re.I)
        if m:
            r = int(m.group(1) or 1)
            if r > best_round:
                best, best_round = p, r
    return best


def read_pcd(path):
    """(N, 3) points of a .pcd (ascii or binary)."""
    raw = path.read_bytes()
    head, i = {}, 0
    while True:
        j = raw.index(b"\n", i)
        line = raw[i:j].decode("ascii", "replace").strip()
        i = j + 1
        if not line or line.startswith("#"):
            continue
        k, *v = line.split()
        head[k.upper()] = v
        if k.upper() == "DATA":
            break
    fields, sizes, types = head["FIELDS"], [int(s) for s in head["SIZE"]], head["TYPE"]
    counts = [int(c) for c in head.get("COUNT", ["1"] * len(fields))]
    n = int(head["POINTS"][0])
    mode = head["DATA"][0].lower()
    if mode == "ascii":
        a = np.loadtxt(raw[i:].decode().splitlines(), ndmin=2)
        cols = np.cumsum([0] + counts)
        return np.c_[[a[:, cols[fields.index(f)]] for f in "xyz"]].T
    if mode != "binary":
        raise SystemExit(f"{path}: DATA {mode} not supported; convert it to binary or ascii "
                         "(pcl_convert_pcd_ascii_binary map.pcd out.pcd 1)")
    tmap = {("F", 4): "f4", ("F", 8): "f8", ("I", 1): "i1", ("I", 2): "i2", ("I", 4): "i4", ("U", 1): "u1",
            ("U", 2): "u2", ("U", 4): "u4", ("I", 8): "i8", ("U", 8): "u8"}
    dt = np.dtype([(f, tmap[(t, s)], (c,)) if c > 1 else (f, tmap[(t, s)]) for f, s, t, c in zip(fields, sizes, types, counts)])
    a = np.frombuffer(raw, dtype=dt, count=n, offset=i)
    return np.c_[a["x"], a["y"], a["z"]].astype(float)


_PLY = {"float": "f4", "float32": "f4", "double": "f8", "float64": "f8", "uchar": "u1", "uint8": "u1",
        "char": "i1", "int8": "i1", "short": "i2", "int16": "i2", "ushort": "u2", "uint16": "u2",
        "int": "i4", "int32": "i4", "uint": "u4", "uint32": "u4"}


def read_ply(path):
    """(N, 3) vertices of a .ply (ascii or binary little endian)."""
    raw = path.read_bytes()
    end = raw.index(b"end_header") + len(b"end_header")
    end = raw.index(b"\n", end) + 1
    lines = raw[:end].decode("ascii", "replace").splitlines()
    fmt, props, n, in_vertex = None, [], 0, False
    for line in lines:
        w = line.split()
        if not w:
            continue
        if w[0] == "format":
            fmt = w[1]
        elif w[0] == "element":
            in_vertex = w[1] == "vertex"
            if in_vertex:
                n = int(w[2])
        elif w[0] == "property" and in_vertex:
            if w[1] == "list":
                raise SystemExit(f"{path}: list properties in vertices not supported")
            props.append((w[2], _PLY[w[1]]))
    names = [p[0] for p in props]
    if fmt == "ascii":
        a = np.loadtxt(raw[end:].decode().splitlines()[:n], ndmin=2)
        return a[:, [names.index(c) for c in "xyz"]].astype(float)
    if fmt != "binary_little_endian":
        raise SystemExit(f"{path}: {fmt} not supported")
    a = np.frombuffer(raw, dtype=np.dtype([(nm, "<" + t) for nm, t in props]), count=n, offset=end)
    return np.c_[a["x"], a["y"], a["z"]].astype(float)


def read_points(path):
    return read_ply(path) if path.suffix.lower() == ".ply" else read_pcd(path)


def floor_plane(P, max_tilt_deg=5.0, tol=0.02, iters=500, seed=0):
    """Floor plane of a room scan: (unit normal n with n_z > 0, d) with n.p + d = 0 on the floor,
    the inlier RMS (m) and the share of the scan on it.
    Candidates: points near the lowest large height layer (wide, so a tilted floor is still in it);
    RANSAC for a near-horizontal plane, then a least-squares fit to its inliers."""
    P = P[np.isfinite(P).all(1)]
    z0, _ = floor_from_points(P, window=0.04)
    C = P[np.abs(P[:, 2] - z0) < 0.3]
    rng = np.random.default_rng(seed)
    best, best_n = None, -1
    cos_max = np.cos(np.radians(max_tilt_deg))
    sub = C if len(C) < 200000 else C[rng.choice(len(C), 200000, replace=False)]
    for _ in range(iters):
        a, b, c = sub[rng.choice(len(sub), 3, replace=False)]
        n = np.cross(b - a, c - a)
        if np.linalg.norm(n) < 1e-9:
            continue
        n /= np.linalg.norm(n)
        if n[2] < 0:
            n = -n
        if n[2] < cos_max:
            continue
        cnt = int(np.sum(np.abs(sub @ n - n @ a) < tol))
        if cnt > best_n:
            best, best_n = (n, -n @ a), cnt
    n, d = best
    for _ in range(3):                                         # least squares on the inliers
        In = C[np.abs(C @ n + d) < tol]
        c = In.mean(0)
        n = np.linalg.eigh(np.cov((In - c).T))[1][:, 0]          # smallest-variance direction
        n = n if n[2] > 0 else -n
        d = -n @ c
    In = C[np.abs(C @ n + d) < tol]
    rms = float(np.sqrt(np.mean((In @ n + d) ** 2)))
    return n, float(d), rms, len(In) / len(P)


def levelling(n, d):
    """T (4x4) map -> room: turns the floor normal n to +z (smallest rotation) and puts the floor at z = 0."""
    z = np.array([0.0, 0.0, 1.0])
    v = np.cross(n, z)
    s, c = np.linalg.norm(v), float(n @ z)
    if s < 1e-12:
        R = np.eye(3)
    else:
        k = v / s
        Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
        R = np.eye(3) + s * Kx + (1 - c) * Kx @ Kx
    T = np.eye(4)
    T[:3, :3] = R
    T[2, 3] = d                       # a floor point p (n.p = -d) goes to z = (R p)_z = n.p = -d -> shift by +d
    return T


def floor_from_points(P, window=0.03):
    """The lowest large horizontal layer of a room scan: floor height (m) and how much of the scan is in it."""
    P = P[np.isfinite(P).all(1)]
    z = P[:, 2]
    lo, hi = np.percentile(z, [0.5, 99.5])
    bins = np.arange(lo, hi + 0.01, 0.01)
    hcount, edges = np.histogram(z, bins)
    smooth = np.convolve(hcount, np.ones(3) / 3, mode="same")
    peaks = [k for k in range(1, len(smooth) - 1)
             if smooth[k] >= smooth[k - 1] and smooth[k] >= smooth[k + 1] and smooth[k] > 0.2 * smooth.max()]
    k = min(peaks)
    zc = (edges[k] + edges[k + 1]) / 2
    near = np.abs(z - zc) < window
    return float(np.median(z[near])), float(near.mean())


def main():
    a = parse()
    doc = yaml.safe_load(a.extrinsics.read_text())
    if a.floor_map:
        n, d, rms, share = floor_plane(read_points(a.floor_map))
        T_room_map = levelling(n, d)
        tilt = np.degrees(np.arccos(n[2]))
        slope = np.hypot(n[0], n[1]) / n[2]
        print(f"floor plane: tilted {tilt:.2f} deg in the map ({100 * slope:.1f} cm per m, falling towards "
              f"({n[0] / max(np.hypot(n[0], n[1]), 1e-12):+.2f}, {n[1] / max(np.hypot(n[0], n[1]), 1e-12):+.2f}) in x, y); "
              f"height under the map origin {-d / n[2]:.3f} m; fit {1000 * rms:.0f} mm RMS over {100 * share:.0f}% of the scan")
        floor_note = f"floor plane n = {np.round(n, 5).tolist()}, d = {d:.4f} (map frame), levelled"
    else:
        T_room_map = np.eye(4)
        T_room_map[2, 3] = -a.floor_z
        floor_note = f"floor at z = {a.floor_z:.3f} in the map (no levelling)"
    cams, report = {}, []
    for tr in doc["transforms"]:
        name = tr["child"]
        n = int(re.search(r"(\d+)", name).group(1))
        if tr.get("quality", {}).get("trusted") is False:
            print(f"{name}: not trusted in the calibration, left out")
            continue
        ip = find_intrinsics(a.intrinsics, n)
        if ip is None:
            print(f"{name}: no intrinsics (cam_{n}.json / .yaml) under {a.intrinsics}, left out")
            continue
        K, dist, w, h = read_intrinsics(ip)
        T_map_cam = np.eye(4)
        T_map_cam[:3, :3] = quat_to_R(tr["rotation_xyzw"])
        T_map_cam[:3, 3] = np.asarray(tr["translation"], float)
        T = T_room_map @ T_map_cam
        cams[name] = {"K": K.round(4).tolist(), "dist": dist.round(6).tolist(), "width": w or 1920,
                      "height": h or 1080, "T_world_cam": T.round(6).tolist()}
        ax = T[:3, 2]
        tilt = np.degrees(np.arcsin(-ax[2]))
        hit = T[:3, 3] + ax * (T[2, 3] / -ax[2]) if ax[2] < 0 else None
        report.append((name, T[2, 3], tilt, hit, ip.relative_to(a.intrinsics), K[0, 0]))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    header = (f"# from {a.extrinsics}\n# intrinsics from {a.intrinsics}\n"
              f"# room frame = the LiDAR map, {floor_note}: z up, floor z = 0, x, y close to the map's\n"
              f"# T_room_map: {np.round(T_room_map, 6).tolist()}\n")
    a.out.write_text(header + yaml.safe_dump({"cameras": cams}, sort_keys=False, default_flow_style=None))
    print(f"\n{'camera':8s} {'height m':>8s} {'tilt deg':>8s} {'looks at floor (x, y)':>22s} {'fx':>7s}  intrinsics")
    for name, z, tilt, hit, ip, fx in report:
        at = f"({hit[0]:5.2f}, {hit[1]:5.2f})" if hit is not None else "above horizon"
        print(f"{name:8s} {z:8.2f} {tilt:8.0f} {at:>22s} {fx:7.0f}  {ip}")
    print(f"\n{len(cams)} cameras -> {a.out}\nCheck: heights should match the mounts (about 2.5-2.7 m), and each camera "
          "should look at the part of the room it shows.")


if __name__ == "__main__":
    main()
