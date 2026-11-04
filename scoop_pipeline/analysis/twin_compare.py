#!/usr/bin/env python3
"""A map against a digital twin's fixed geometry: aligned, then how complete
and how accurate it is, per part (floor, walls, ceiling, built-ins, each
appliance and table).

    python3 scoop_pipeline/analysis/twin_compare.py <map .pcd/.ply> <twin .ply> <out dir>
        --traj <TUM of the LiDAR (or camera) in the map's frame>
        [--voxel 0.03] [--pose-step 0.5] [--min-range 1.3] [--max-range 15]
        [--vfov 22.5] [--yaw-candidates 4]

<twin .ply> is twin_cloud.py's labelled cloud (its .json beside it names the
labels). The map is the room as built (with its ceiling: ed305_room_ceil.ply).

The twin's furniture is not where it was on the day of the map (and sofas,
chairs are not in it at all), so nothing but the room itself is used to
align and to measure the map; the fixed objects are found again wherever
they now stand.

1. Rough alignment: the map's walls' yaw (crop_room.wall_frame) onto the
   twin's axes, each quarter turn tried by cross-correlating top-down images
   of tall structure; Tukey-robust point-to-plane ICP on everything from each
   turn that scores near the best (a rectangular room fits half turned too),
   the one with the most of the map within 5 cm of the twin kept.
2. What the LiDAR could see: the twin rendered from poses every --pose-step m
   along --traj, a panoramic range image of +-vfov deg, --min-range ..
   --max-range (the map's own range gate). Outer wall faces, the slab under
   the floor and over a suspended ceiling, the inside of cabinets drop out.
3. The room's planes in the seen twin: four walls (crop_room.room_bounds),
   floor and ceiling (height peaks). The alignment redone on the map points
   near those planes only, onto the twin's points on them.
4. Room size: length, width, height from each cloud's own planes, map - twin.
   Per plane: the map's points within 10 cm of it, their offset (+ = inside
   the twin's room), the tilt of a plane fitted to them (bend, drift) and
   their scatter about it (noise); how much of the seen plane the map covers
   at 2 / 5 / 10 cm. The room's size is the map's scale error.
5. Fixed objects (twin_cloud's parts that are not floor, wall, ceiling,
   builtin): each searched for within 3 m of where the twin has it (every
   heading x offset of its footprint over the map's points not on the
   planes, then ICP): moved (cm), turned (deg), and the share of its seen
   surface with a map point within 5 cm there and in the twin's place;
   status in place / moved / not found. Built-ins (counters, cabinets)
   measured in place.

<out dir>: report.md, report.json, T_twin_map.json (the map -> twin 4x4),
map_error.ply (the map in the twin's frame, coloured by its distance: blue 0,
red >= 10 cm, grey > 20 cm = not in the twin), twin_coverage.ply (the visible
twin: green within 5 cm of the map, red not).
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "processing"))

from twin_cloud import read_labelled_ply                                 # noqa: E402


# ---------------------------------------------------------------- io
def load_cloud(path):
    """xyz of a .pcd / .ply (open3d), or of a binary .ply without it."""
    path = os.path.expanduser(path)
    try:
        import open3d as o3d
    except ImportError:
        if not path.endswith(".ply"):
            raise SystemExit("reading %s needs open3d" % path)
        sys.path.insert(0, os.path.join(os.path.dirname(HERE), "assets"))
        from views import read_ply_xyz
        return read_ply_xyz(path)
    return np.asarray(o3d.io.read_point_cloud(path).points, np.float64)


def write_rgb_ply(path, xyz, rgb):
    with open(path, "wb") as fh:
        fh.write(("ply\nformat binary_little_endian 1.0\nelement vertex %d\n"
                  "property float x\nproperty float y\nproperty float z\n"
                  "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
                  % len(xyz)).encode())
        rec = np.zeros(len(xyz), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                        ("r", "u1"), ("g", "u1"), ("b", "u1")])
        rec["x"], rec["y"], rec["z"] = xyz.T
        rec["r"], rec["g"], rec["b"] = rgb.T
        fh.write(rec.tobytes())


def read_tum(path):
    a = np.loadtxt(os.path.expanduser(path), comments="#", ndmin=2)
    return a[:, 0], a[:, 1:4]


# ---------------------------------------------------------------- helpers
def thin(P, voxel):
    k = np.floor(P / voxel).astype(np.int64)
    _, i = np.unique(k, axis=0, return_index=True)
    return P[np.sort(i)]


def rot_z(deg):
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def floor_z(P, bin_m=0.02):
    """The lowest height peak holding at least 30 % of the largest."""
    h, e = np.histogram(P[:, 2], np.arange(P[:, 2].min(), P[:, 2].max() + bin_m, bin_m))
    hs = np.convolve(h, np.ones(3) / 3, mode="same")
    c = 0.5 * (e[:-1] + e[1:])
    pk = [i for i in range(1, len(hs) - 1) if hs[i] >= hs[i - 1] and hs[i] > hs[i + 1]]
    top = max(hs[i] for i in pk)
    return float(c[min(i for i in pk if hs[i] >= 0.3 * top)])


def topdown(P, z0, cell, lo, hi):
    """2-D occupancy of tall structure (z0+0.3 .. z0+2.0) on a grid from lo to hi."""
    Q = P[(P[:, 2] > z0 + 0.3) & (P[:, 2] < z0 + 2.0)]
    n = np.ceil((hi - lo) / cell).astype(int) + 1
    ij = np.floor((Q[:, :2] - lo) / cell).astype(int)
    ok = (ij >= 0).all(1) & (ij < n).all(1)
    img = np.zeros(n)
    np.add.at(img, (ij[ok, 0], ij[ok, 1]), 1.0)
    return np.log1p(img)


def coarse(M, T, yaw_map, n_yaw=4, cell=0.05, log=print):
    """-> (4x4 map -> twin, score, per-candidate list)."""
    from scipy.signal import fftconvolve
    zt, zm = floor_z(T), floor_z(M)
    lo_t = T[:, :2].min(0) - 1.0
    hi_t = T[:, :2].max(0) + 1.0
    It = topdown(T, zt, cell, lo_t, hi_t)
    It = (It - It.mean()) / (It.std() + 1e-9)
    best, cands = None, []
    for k in range(n_yaw):
        yaw = k * 360.0 / n_yaw - yaw_map
        R = rot_z(yaw)
        Mr = M @ R.T
        lo_m = Mr[:, :2].min(0) - 1.0
        hi_m = Mr[:, :2].max(0) + 1.0
        Im = topdown(Mr, zm, cell, lo_m, hi_m)
        Im = (Im - Im.mean()) / (Im.std() + 1e-9)
        cc = fftconvolve(It, Im[::-1, ::-1], mode="full")
        i, j = np.unravel_index(int(np.argmax(cc)), cc.shape)
        score = float(cc[i, j]) / Im.size
        # cell (a, b) of Im lands on cell (a + i - (Im.shape-1)) of It
        sh = np.array([i - (Im.shape[0] - 1), j - (Im.shape[1] - 1)])
        t_xy = lo_t + sh * cell - lo_m
        Tm = np.eye(4)
        Tm[:3, :3] = R
        Tm[:3, 3] = [t_xy[0], t_xy[1], zt - zm]
        cands.append({"yaw_deg": round(yaw, 2), "score": round(score, 4),
                      "t": np.round(Tm[:3, 3], 3).tolist(), "_T": Tm})
        if best is None or score > best[1]:
            best = (Tm, score)
    for c in cands:
        log("    coarse yaw %7.2f deg: score %.4f, t %s" % (c["yaw_deg"], c["score"], c["t"]))
    return best[0], best[1], cands


def normals(P, k=16):
    from scipy.spatial import cKDTree
    _, nn = cKDTree(P).query(P, k=k, workers=-1)
    Q = P[nn] - P[nn].mean(1, keepdims=True)
    _, V = np.linalg.eigh(np.einsum("nki,nkj->nij", Q, Q))
    return V[:, :, 0]


def icp(S, T, Tn, T0, schedule=(0.25, 0.10, 0.05), iters=15, log=print):
    """Point-to-plane ICP of source S onto target T (normals Tn), Tukey weights.
    -> (4x4, inlier RMS m, inlier share)."""
    from scipy.spatial import cKDTree
    tree = cKDTree(T)
    X = T0.copy()
    for dmax in schedule:
        for _ in range(iters):
            P = S @ X[:3, :3].T + X[:3, 3]
            d, j = tree.query(P, distance_upper_bound=dmax, workers=-1)
            ok = np.isfinite(d)
            if ok.sum() < 100:
                break
            p, q, n = P[ok], T[j[ok]], Tn[j[ok]]
            r = np.einsum("ij,ij->i", p - q, n)
            c = dmax / 2.0
            w = np.where(np.abs(r) < c, (1 - (r / c) ** 2) ** 2, 0.0)     # Tukey
            A = np.c_[np.cross(p, n), n]
            H = (A * w[:, None]).T @ A
            g = (A * w[:, None]).T @ r
            try:
                x = -np.linalg.solve(H + 1e-9 * np.eye(6), g)
            except np.linalg.LinAlgError:
                break
            dR = rot_from(x[:3])
            D = np.eye(4)
            D[:3, :3], D[:3, 3] = dR, x[3:]
            X = D @ X
            if np.linalg.norm(x) < 1e-6:
                break
        P = S @ X[:3, :3].T + X[:3, 3]
        d, _ = tree.query(P, distance_upper_bound=dmax, workers=-1)
        ok = np.isfinite(d)
        log("    icp <= %.2f m: %.1f %% of the map within, rms %.1f mm"
            % (dmax, 100 * ok.mean(), 1000 * np.sqrt(np.mean(d[ok] ** 2)) if ok.any() else 0))
    return X, float(np.sqrt(np.mean(d[ok] ** 2))) if ok.any() else np.nan, float(ok.mean())


def rot_from(w):
    th = np.linalg.norm(w)
    if th < 1e-12:
        return np.eye(3)
    k = w / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def visible(T, poses, min_r, max_r, vfov, cell_deg=0.35, tol=0.03, spacing=0.02, log=print):
    """Twin points seen, nothing nearer in their panoramic cell, from any pose."""
    seen = np.zeros(len(T), bool)
    naz = int(round(360 / cell_deg))
    nel = int(np.ceil(2 * vfov / cell_deg))
    for k, o in enumerate(poses):
        v = T - o
        r = np.linalg.norm(v, axis=1)
        el = np.degrees(np.arcsin(np.clip(v[:, 2] / np.maximum(r, 1e-9), -1, 1)))
        inr = (r > 0.05) & (np.abs(el) <= vfov)
        idx = np.flatnonzero(inr)
        az = np.degrees(np.arctan2(v[idx, 1], v[idx, 0]))
        a = np.floor((az + 180) / cell_deg).astype(np.int64) % naz
        e = np.clip(np.floor((el[idx] + vfov) / cell_deg).astype(np.int64), 0, nel - 1)
        cell = a * nel + e
        zb = np.full(naz * nel, np.inf)
        np.minimum.at(zb, cell, r[idx])
        # a near point covers more than its cell (spacing / range wider than a
        # cell): spread over those, or what is behind shows through the gaps
        half = np.ceil(0.5 * spacing / (r[idx] * np.radians(cell_deg))).astype(int)
        for h in range(1, int(min(half.max(), 6)) + 1):
            g = np.flatnonzero(half >= h)
            for da in range(-h, h + 1):
                for de in (-h, h) if abs(da) < h else range(-h, h + 1):
                    np.minimum.at(zb, ((a[g] + da) % naz) * nel + np.clip(e[g] + de, 0, nel - 1), r[idx][g])
        ok = (r[idx] <= zb[cell] * 1.01 + tol) & (r[idx] >= min_r) & (r[idx] <= max_r)
        seen[idx[ok]] = True
        if (k + 1) % 50 == 0:
            log("    visibility: %d / %d poses, %.1f %% of the twin seen" % (k + 1, len(poses), 100 * seen.mean()))
    return seen


def poses_along(xyz, step):
    out, last = [xyz[0]], xyz[0]
    for p in xyz[1:]:
        if np.linalg.norm(p - last) >= step:
            out.append(p)
            last = p
    return np.array(out)


def ramp(d, lo=0.0, hi=0.10, far=0.20):
    t = np.clip((d - lo) / (hi - lo), 0, 1)
    rgb = np.c_[255 * t, 60 + 0 * t, 255 * (1 - t)].astype(np.uint8)
    rgb[d > far] = (150, 150, 150)
    return rgb


# ---------------------------------------------------------------- structure
SHELL_LABELS = ("floor", "wall", "ceiling", "builtin")


def level_peaks(z, floor_hint=None, bin_m=0.02):
    """(floor z, ceiling z): the lowest strong height peak in the lowest metre, and
    the biggest at least 2 m above it (a suspended ceiling, not the slab over
    it, when only the former is seen)."""
    h, e = np.histogram(z, np.arange(z.min(), z.max() + bin_m, bin_m))
    hs = np.convolve(h, np.ones(3) / 3, mode="same")
    c = 0.5 * (e[:-1] + e[1:])
    lo = np.flatnonzero(c <= (c[0] + 1.0 if floor_hint is None else floor_hint + 0.3))
    pk = [i for i in lo if 0 < i < len(hs) - 1 and hs[i] >= hs[i - 1] and hs[i] > hs[i + 1]] or list(lo)
    top = max(hs[i] for i in pk)
    f = float(c[min(i for i in pk if hs[i] >= 0.3 * top)])       # the lowest strong one: not a counter
    hi = c >= f + 2.0
    if not hi.any():
        raise SystemExit("no ceiling 2 m or more over the floor (z %.2f)" % f)
    return f, float(c[hi][np.argmax(hs[hi])])


def room_planes(P, yaw=None, frac=0.3):
    """The room's six planes from a cloud of what is seen of it: walls by
    crop_room.room_bounds (the outermost strong vertical planes along the
    room's axes), floor and ceiling by level_peaks. -> dict yaw, x0 x1 y0 y1
    (in the frame turned by -yaw), floor, ceiling."""
    import crop_room
    yaw, (x0, x1, y0, y1), _ = crop_room.room_bounds(P, frac=frac, yaw=yaw)
    f, c = level_peaks(P[:, 2])
    rp = {"yaw": float(yaw), "x0": float(x0), "x1": float(x1), "y0": float(y0), "y1": float(y1),
          "floor": f, "ceiling": c}
    # each plane where its own points are (the peaks are histogram bins)
    Pr = P @ rot_z(-yaw).T
    for tol in (0.08, 0.04, 0.02):
        pid = plane_of(Pr, rp, tol)
        for k, (_, ax, key, _) in enumerate(PLANES):
            if (pid == k).sum() >= 20:
                rp[key] = float(np.median(Pr[pid == k, ax]))
    return rp


PLANES = (("floor", 2, "floor", +1), ("ceiling", 2, "ceiling", -1),
          ("wall x-", 0, "x0", +1), ("wall x+", 0, "x1", -1),
          ("wall y-", 1, "y0", +1), ("wall y+", 1, "y1", -1))


def plane_of(Pr, rp, tol):
    """Index into PLANES of the plane each point (turned frame) lies on within
    tol, -1 none; 0.2 m clear of the other planes (corners, skirting)."""
    out = np.full(len(Pr), -1, np.int64)
    inside = ((Pr[:, 0] > rp["x0"] + 0.2) & (Pr[:, 0] < rp["x1"] - 0.2)
              & (Pr[:, 1] > rp["y0"] + 0.2) & (Pr[:, 1] < rp["y1"] - 0.2))
    mid = (Pr[:, 2] > rp["floor"] + 0.2) & (Pr[:, 2] < rp["ceiling"] - 0.2)
    for k, (_, ax, key, _) in enumerate(PLANES):
        near = np.abs(Pr[:, ax] - rp[key]) <= tol
        if ax == 2:
            sel = near & inside
        else:
            other = 1 - ax
            lo, hi = (rp["y0"], rp["y1"]) if other == 1 else (rp["x0"], rp["x1"])
            sel = near & mid & (Pr[:, other] > lo + 0.2) & (Pr[:, other] < hi - 0.2)
        out[sel & (out < 0)] = k
    return out


def plane_fit(Pr, pid, rp, k, centre=None):
    """Points of plane k against the plane rp has it at: signed distance (+ =
    into the room), a plane fitted to them (robust). -> dict: offset at centre
    (default: theirs), slope (2,), scatter about the fit, span, n, centre."""
    name, ax, key, sgn = PLANES[k]
    m = pid == k
    if m.sum() < 50:
        return None
    d = sgn * (Pr[m, ax] - rp[key])
    uv = Pr[m][:, [a for a in range(3) if a != ax]]
    c = uv.mean(0) if centre is None else np.asarray(centre)
    A = np.c_[np.ones(len(d)), uv - c]
    keep = np.ones(len(d), bool)
    for _ in range(3):
        x, *_ = np.linalg.lstsq(A[keep], d[keep], rcond=None)
        r = d - A @ x
        mad = 1.4826 * np.median(np.abs(r[keep] - np.median(r[keep]))) + 1e-4
        keep = np.abs(r) < 3 * mad
    return {"offset": float(x[0]), "slope": x[1:], "rms": float(np.sqrt(np.mean(r[keep] ** 2))),
            "span": float(np.ptp(uv[keep], axis=0).max()), "n": int(m.sum()), "centre": c,
            "median_abs": float(np.median(np.abs(d)))}


def plane_stats(Pr, pid, Tr, tpid, rp):
    """Per plane, the map's points on it against the twin's own points on it,
    both taken the same way (same band, same fit, same centre): what is near
    the plane in both (a light, a skirting board) cancels. Offset and tilt are
    map - twin; scatter is each one's about its fit."""
    rows = []
    for k, (name, *_r) in enumerate(PLANES):
        row = {"plane": name, "map_points": int((pid == k).sum())}
        t = plane_fit(Tr, tpid, rp, k)
        m = plane_fit(Pr, pid, rp, k, None if t is None else t["centre"])
        if t is None or m is None:
            rows.append(row)
            continue
        ds = m["slope"] - t["slope"]
        span = min(m["span"], t["span"])
        row.update({"offset_mm": round(1000 * (m["offset"] - t["offset"]), 1),
                    "tilt_deg": round(float(np.degrees(np.arctan(np.linalg.norm(ds)))), 3),
                    "tilt_mm_over_span": round(1000 * float(np.linalg.norm(ds)) * span, 1),
                    "span_m": round(span, 2),
                    "noise_rms_mm": round(1000 * m["rms"], 1),
                    "twin_rms_mm": round(1000 * t["rms"], 1),
                    "median_abs_mm": round(1000 * m["median_abs"], 1)})
        rows.append(row)
    return rows


# ---------------------------------------------------------------- objects
def xy_search(src, tgt, z0, cell=0.05, yaw_step=15.0, top=3):
    """Footprints (z0+0.1 .. z0+2.0) of src over tgt, every yaw, every offset:
    [(score, 4x4 moving src onto tgt)], best first, at least 0.5 m / 20 deg apart."""
    from scipy.signal import fftconvolve

    def occ(P, lo, n):
        ij = np.floor((P[:, :2] - lo) / cell).astype(int)
        ok = (ij >= 0).all(1) & (ij < n).all(1)
        I = np.zeros(n, np.float32)
        I[ij[ok, 0], ij[ok, 1]] = 1.0
        return I
    band = lambda P: P[(P[:, 2] > z0 + 0.1) & (P[:, 2] < z0 + 2.0)]       # noqa: E731
    S, T = band(src), band(tgt)
    if len(S) < 20 or len(T) < 20:
        return []
    from scipy import ndimage
    lo_t = T[:, :2].min(0) - 0.5
    It = ndimage.maximum_filter(occ(T, lo_t, np.ceil((T[:, :2].max(0) + 0.5 - lo_t) / cell).astype(int) + 1), 3)
    cs = S[:, :2].mean(0)
    peaks = []
    for yaw in np.arange(0.0, 360.0, yaw_step):
        R = rot_z(yaw)
        rel = (S[:, :2] - cs) @ R[:2, :2].T
        w0 = rel.min(0)
        W = occ(rel, w0, np.ceil((rel.max(0) - w0) / cell).astype(int) + 1)
        C = fftconvolve(It, W[::-1, ::-1], mode="full") / W.sum()
        k = np.unravel_index(int(np.argmax(C)), C.shape)
        shift = np.array(k) - (np.array(W.shape) - 1)
        X = np.eye(4)
        X[:3, :3] = R
        X[:2, 3] = -R[:2, :2] @ cs + lo_t - w0 + shift * cell
        peaks.append((float(C[k]), X))
    peaks.sort(key=lambda p: -p[0])
    out = []
    for sc, X in peaks:
        if all(np.linalg.norm(X[:2, 3] - Y[:2, 3]) > 0.5 or
               abs((np.degrees(np.arctan2(X[1, 0], X[0, 0]) - np.arctan2(Y[1, 0], Y[0, 0]) * 180 / np.pi)
                    + 180) % 360 - 180) > 20 for _, Y in out):
            out.append((sc, X))
        if len(out) >= top:
            break
    return out


def refit_object(O, Ov, M, z0, reach=3.0, near=0.05, keep_in_place=0.5, found=0.5, log=print):
    """A fixed object of the twin found again in the map (its points not on the
    room's planes, in the twin's frame) wherever it now stands within reach m:
    in place first, then every heading x offset of its footprint, then ICP.
    -> dict: in place and re-fitted share of its visible points with a map
    point within `near`, the move (cm, deg), the fit's median distance."""
    from scipy.spatial import cKDTree
    c = O.mean(0)
    ext = float(np.max(np.ptp(O[:, :2], axis=0))) / 2
    L = M[np.linalg.norm(M[:, :2] - c[:2], axis=1) < ext + reach]
    res = {"twin_points": int(len(O)), "visible": int(Ov.sum())}
    if len(L) < 50 or Ov.sum() < 20:
        res["status"] = "not in the map" if len(L) < 50 else "not visible"
        return res
    L = thin(L, 0.02)
    tree = cKDTree(L)
    Ln = normals(L)
    Os = thin(O[Ov], 0.02)

    def score(X):
        P = Os @ X[:3, :3].T + X[:3, 3]
        d, _ = tree.query(P, workers=-1)
        return float((d <= near).mean()), float(np.median(d))
    s0, m0 = score(np.eye(4))
    res.update({"in_place_share": round(s0, 3), "in_place_median_mm": round(1000 * m0, 1)})
    quiet = lambda *a, **k: None                                         # noqa: E731

    def fit(X0, sched):
        X, _, _ = icp(Os, L, Ln, X0, schedule=sched, iters=12, log=quiet)
        if np.degrees(np.arccos(np.clip(X[2, 2], -1, 1))) > 5.0:
            return None
        return (X,) + score(X)
    best = fit(np.eye(4), (0.10, 0.05, 0.03))                         # where the twin has it
    if best is None or best[1] < keep_in_place:
        for _, X0 in xy_search(Os, L, z0):
            f = fit(X0, (0.30, 0.10, 0.05))
            if f is not None and (best is None or f[1] > best[1] + 0.05):
                best = f
    if best is None:
        res["status"] = "not found"
        return res
    X, s, m = best
    cv = 0.5 * (O.min(0) + O.max(0))                     # the whole object's centre, not its seen side's
    mv = X[:3, :3] @ cv + X[:3, 3] - cv
    res.update({"refit_share": round(s, 3), "refit_median_mm": round(1000 * m, 1),
                "moved_cm": round(100 * float(np.linalg.norm(mv[:2])), 1),
                "moved_xy_m": np.round(mv[:2], 3).tolist(),
                "turned_deg": round(float(np.degrees(np.arctan2(X[1, 0], X[0, 0]))), 1)})
    res["status"] = "not found" if s < found else ("in place" if res["moved_cm"] < 10 else "moved")
    res["X"] = X
    return res


# ---------------------------------------------------------------- main
def compare(map_xyz, twin_xyz, twin_lab, names, traj_xyz, voxel=0.03, pose_step=0.5,
            min_r=1.3, max_r=15.0, vfov=22.5, n_yaw=4, plane_tol=0.10, log=print):
    from scipy.spatial import cKDTree
    import crop_room
    Ms = thin(map_xyz, voxel)
    Ts = thin(twin_xyz, voxel)
    yaw_m = crop_room.wall_frame(map_xyz)[0]
    log("  map: %d points (%d at %.2f m), walls' yaw %.2f deg; twin: %d points (%d)"
        % (len(map_xyz), len(Ms), voxel, yaw_m, len(twin_xyz), len(Ts)))
    # 1. rough: everything (Tukey ICP)
    T0, score, cands = coarse(Ms, Ts, yaw_m, n_yaw, log=log)
    # a rectangular room fits its walls turned half round as well: every
    # quarter turn that scores near the best is aligned, the one with the most
    # of the map on the twin (built-ins, fixed appliances) kept
    Tn_all = normals(Ts)
    tree_s = cKDTree(Ts)
    best = None
    for c in sorted(cands, key=lambda c: -c["score"]):
        if c["score"] < 0.7 * score:
            continue
        X, _, _ = icp(Ms, Ts, Tn_all, c["_T"], log=lambda *a: None)
        d, _ = tree_s.query(Ms @ X[:3, :3].T + X[:3, 3], workers=-1)
        c["on_twin_5cm"] = round(float((d <= 0.05).mean()), 4)
        log("    turn %7.2f deg aligned: %.1f %% of the map within 5 cm of the twin"
            % (c["yaw_deg"], 100 * c["on_twin_5cm"]))
        if best is None or c["on_twin_5cm"] > best[1]:
            best = (X, c["on_twin_5cm"])
    X0 = best[0]
    for c in cands:
        c.pop("_T", None)
    # 2. what of the twin the LiDAR saw, and the room's planes in it
    poses = poses_along(traj_xyz @ X0[:3, :3].T + X0[:3, 3], pose_step)
    log("  visibility from %d poses along the path (every %.2f m)" % (len(poses), pose_step))
    Tv = thin(twin_xyz, 0.02)
    lab_v = twin_lab[cKDTree(twin_xyz).query(Tv, workers=-1)[1]]
    seen = visible(Tv, poses, min_r, max_r, vfov, log=log)
    rp_t = room_planes(Tv[seen])
    Rz = rot_z(-rp_t["yaw"])
    tw_pid = plane_of(Tv @ Rz.T, rp_t, 0.03)
    struct_t = seen & (tw_pid >= 0)
    log("  twin's room (seen): yaw %.2f deg, x %.3f..%.3f, y %.3f..%.3f, floor %.3f, ceiling %.3f; "
        "%d points on its planes" % (rp_t["yaw"], rp_t["x0"], rp_t["x1"], rp_t["y0"], rp_t["y1"],
                                     rp_t["floor"], rp_t["ceiling"], int(struct_t.sum())))
    # 3. the alignment again, on the room's planes only (objects moved, sofas: left out)
    M0 = map_xyz @ X0[:3, :3].T + X0[:3, 3]
    dz = rp_t["floor"] - level_peaks(M0[:, 2])[0]          # the full twin's floor can be the slab's underside
    if abs(dz) > 0.01:
        log("  floors put level: map moved %+.3f m in z" % dz)
        X0[2, 3] += dz
        M0[:, 2] += dz
    on0 = plane_of(M0 @ Rz.T, rp_t, 0.15) >= 0
    Tst = thin(Tv[struct_t], voxel)
    log("  structure only: %d map points near the twin's planes" % int(on0.sum()))
    X, rms, share = icp(thin(map_xyz[on0], voxel), Tst, normals(Tst), X0,
                        schedule=(0.10, 0.05, 0.03), log=log)
    yaw = np.degrees(np.arctan2(X[1, 0], X[0, 0]))
    Mt = map_xyz @ X[:3, :3].T + X[:3, 3]
    Mr = Mt @ Rz.T
    pid = plane_of(Mr, rp_t, plane_tol)
    out = {"T_twin_map": X.round(6).tolist(), "yaw_deg": round(float(yaw), 3), "coarse": cands,
           "icp_rms_mm": round(1000 * rms, 2), "icp_inlier_share": round(share, 4), "twin_room": rp_t}
    # 4. each plane of the map against the twin's; the room's size from them
    Tr = Tv[seen] @ Rz.T
    out["planes"] = plane_stats(Mr, pid, Tr, plane_of(Tr, rp_t, plane_tol), rp_t)
    off = {p["plane"]: p.get("offset_mm") for p in out["planes"]}
    dims = []
    for name, a, b, pa, pb in (("length (x)", "x0", "x1", "wall x-", "wall x+"),
                               ("width (y)", "y0", "y1", "wall y-", "wall y+"),
                               ("height", "floor", "ceiling", "floor", "ceiling")):
        lt = rp_t[b] - rp_t[a]
        row = {"size": name, "twin_m": round(lt, 3)}
        if off[pa] is not None and off[pb] is not None:
            dmm = -(off[pa] + off[pb])                       # + offset = inside: the room smaller
            row.update({"map_m": round(lt + dmm / 1000, 3), "diff_mm": round(dmm, 1),
                        "diff_pct": round(100 * dmm / 1000 / lt, 3)})
            log("  %-10s twin %.3f m, map %.3f m: %+.1f mm (%+.2f %%)"
                % (name, lt, row["map_m"], dmm, row["diff_pct"]))
        dims.append(row)
    out["room"] = dims

    # 5. structure completeness: the twin's planes seen, covered by the map
    dt, _ = cKDTree(Mt).query(Tv, workers=-1)
    comp = []
    for k, (name, *_r) in enumerate(PLANES):
        v = seen & (tw_pid == k)
        comp.append({"plane": name, "visible_points": int(v.sum()),
                     **{"complete_%dcm" % c: (round(float((dt[v] <= c / 100).mean()), 4) if v.any() else None)
                        for c in (2, 5, 10)}})
    out["plane_completeness"] = comp
    # 6. fixed objects: found again wherever they stand, the biggest first; map
    # points the built-ins or an object already placed explain are taken out
    Mfree = Mt[(pid < 0) & (Mt[:, 2] > rp_t["floor"] + 0.03)]
    objs = []
    kb = names.index("builtin") if "builtin" in names else -1
    if kb >= 0:
        bv = seen & (lab_v == kb) & (tw_pid < 0)
        if bv.any():
            objs.append({"label": "builtin (in place)", "visible": int(bv.sum()), "status": "in place",
                         "in_place_share": round(float((dt[bv] <= 0.05).mean()), 3),
                         "in_place_median_mm": round(1000 * float(np.median(dt[bv])), 1)})
            d, _ = cKDTree(Tv[(lab_v == kb) & (tw_pid < 0)]).query(Mfree, workers=-1)
            Mfree = Mfree[d > 0.03]
    order = sorted((k for k, nm in enumerate(names) if nm not in SHELL_LABELS and (lab_v == k).any()),
                   key=lambda k: -int((lab_v == k).sum()))
    for k in order:
        nm = names[k]
        mv = lab_v == k
        O = Tv[mv]
        r = refit_object(O, seen[mv], Mfree, rp_t["floor"], log=log)
        r["label"] = nm
        X = r.pop("X", None)
        if X is not None and r["status"] != "not found":
            d, _ = cKDTree(O @ X[:3, :3].T + X[:3, 3]).query(Mfree, workers=-1)
            Mfree = Mfree[d > 0.03]
        objs.append(r)
        log("  %-18s %s" % (nm, r["status"] + ("" if "moved_cm" not in r else
            ": moved %.0f cm, turned %.0f deg; seen surface on the map %.0f %% (in place %.0f %%), median %.0f mm"
            % (r["moved_cm"], r["turned_deg"], 100 * r["refit_share"], 100 * r["in_place_share"],
               r["refit_median_mm"]))))
    out["objects"] = objs
    dm, _ = cKDTree(twin_xyz).query(Mt, workers=-1)
    out["overall"] = {
        "map_on_planes_share": round(float((pid >= 0).mean()), 4),
        "map_within_5cm_of_twin": round(float((dm <= 0.05).mean()), 4),
        "twin_visible_share": round(float(seen.mean()), 4)}
    return out, Mt, dm, Tv, seen, dt, pid


def report_md(r, map_path, twin_path):
    L = ["# Map against the digital twin", "",
         "Map `%s`; twin `%s` (its fixed parts; sofas and chairs left out)." % (map_path, twin_path), "",
         "Aligned on the room's planes only (floor, ceiling, four walls): yaw %.2f deg, ICP rms %.1f mm "
         "on %.1f %% of those map points. Objects play no part in it." % (
             r["yaw_deg"], r["icp_rms_mm"], 100 * r["icp_inlier_share"]), "",
         "## Room size", "", "| | twin | map | map - twin |", "|---|---|---|---|"]
    for d in r["room"]:
        if "map_m" not in d:
            L.append("| %s | %.3f m | - | - |" % (d["size"], d["twin_m"]))
            continue
        L.append("| %s | %.3f m | %.3f m | %+.1f mm (%+.2f %%) |"
                 % (d["size"], d["twin_m"], d["map_m"], d["diff_mm"], d["diff_pct"]))
    L += ["", "The twin's: between its planes as seen (inner faces); the map's: "
          "the twin's less the offsets of the map's two opposite planes (below), which the "
          "alignment cannot hide.", "",
          "## Planes", "",
          "| plane | map points | offset | tilt | over its span | scatter (RMS) map / twin | median abs |",
          "|---|---|---|---|---|---|---|"]
    for p in r["planes"]:
        if "offset_mm" not in p:
            L.append("| %s | %d | - | - | - | - | - |" % (p["plane"], p["map_points"]))
            continue
        L.append("| %s | %d | %+.1f mm | %.3f deg | %.1f mm / %.1f m | %.1f / %.1f mm | %.1f mm |" % (
            p["plane"], p["map_points"], p["offset_mm"], p["tilt_deg"], p["tilt_mm_over_span"],
            p["span_m"], p["noise_rms_mm"], p["twin_rms_mm"], p["median_abs_mm"]))
    L += ["", "The map's points within 10 cm of each twin plane (0.2 m clear of corners) and the "
          "seen twin's, each fitted with a plane the same way: offset of the map's at the plane's "
          "centre (+ = inside the twin's room) and its tilt against the twin's (bend, drift); "
          "scatter of each about its fit (the map's noise; the twin's: what else is that near).", "",
          "| plane | visible | complete 2 / 5 / 10 cm |", "|---|---|---|"]
    f = lambda x: "-" if x is None else "%.0f" % (100 * x)                 # noqa: E731
    for c in r["plane_completeness"]:
        L.append("| %s | %d | %s / %s / %s %% |" % (c["plane"], c["visible_points"], f(c["complete_2cm"]),
                                                    f(c["complete_5cm"]), f(c["complete_10cm"])))
    L += ["", "## Fixed objects", "",
          "| object | status | moved | turned | seen surface on the map (5 cm) | in place | median |",
          "|---|---|---|---|---|---|---|"]
    for o in r["objects"]:
        L.append("| %s | %s | %s | %s | %s | %s | %s |" % (
            o["label"], o["status"],
            "%.0f cm" % o["moved_cm"] if "moved_cm" in o else "-",
            "%.0f deg" % o["turned_deg"] if "turned_deg" in o else "-",
            "%.0f %%" % (100 * o["refit_share"]) if "refit_share" in o else "-",
            "%.0f %%" % (100 * o["in_place_share"]) if "in_place_share" in o else "-",
            "%.0f mm" % o.get("refit_median_mm", o.get("in_place_median_mm", float("nan")))
            if ("refit_median_mm" in o or "in_place_median_mm" in o) else "-"))
    o = r["overall"]
    L += ["", "Each object found again near where the twin has it (3 m around, every heading, then "
          "ICP): how far it moved, and how much of its surface the LiDAR could see has a map point "
          "within 5 cm there and in the twin's place. Under 50 %: not found (gone, or moved further); a symmetric object's turn is ambiguous by 180 deg.",
          "", "%.1f %% of the map is on the room's planes; %.1f %% within 5 cm of the twin as it "
          "stands (the rest: furniture moved or not in the twin, clutter)."
          % (100 * o["map_on_planes_share"], 100 * o["map_within_5cm_of_twin"])]
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__.split("\n\n", 1)[1])
    ap.add_argument("map")
    ap.add_argument("twin")
    ap.add_argument("out")
    ap.add_argument("--traj", required=True)
    ap.add_argument("--voxel", type=float, default=0.03)
    ap.add_argument("--pose-step", type=float, default=0.5)
    ap.add_argument("--min-range", type=float, default=1.3)
    ap.add_argument("--max-range", type=float, default=15.0)
    ap.add_argument("--vfov", type=float, default=22.5)
    ap.add_argument("--yaw-candidates", type=int, default=4)
    a = ap.parse_args(argv)
    names = json.load(open(os.path.splitext(os.path.expanduser(a.twin))[0] + ".json"))["labels"]
    twin_xyz, twin_lab = read_labelled_ply(a.twin)
    map_xyz = load_cloud(a.map)
    _, traj = read_tum(a.traj)
    print("%s vs %s" % (a.map, a.twin))
    r, Mt, dm, Tv, seen, dt, pid = compare(map_xyz, twin_xyz, twin_lab, names, traj, a.voxel, a.pose_step,
                                      a.min_range, a.max_range, a.vfov, a.yaw_candidates)
    out = os.path.expanduser(a.out)
    os.makedirs(out, exist_ok=True)
    json.dump(r, open(os.path.join(out, "report.json"), "w"), indent=1)
    json.dump({"T_twin_map": r["T_twin_map"]}, open(os.path.join(out, "T_twin_map.json"), "w"), indent=1)
    md = report_md(r, a.map, a.twin)
    open(os.path.join(out, "report.md"), "w").write(md)
    write_rgb_ply(os.path.join(out, "map_error.ply"), Mt, ramp(dm))
    col = np.where((dt[seen] <= 0.05)[:, None], [[40, 200, 40]], [[220, 40, 40]]).astype(np.uint8)
    write_rgb_ply(os.path.join(out, "twin_coverage.ply"), Tv[seen], col)
    print(md)
    print("-> %s (report.md, report.json, T_twin_map.json, map_error.ply, twin_coverage.ply)" % out)


if __name__ == "__main__":
    sys.exit(main())
