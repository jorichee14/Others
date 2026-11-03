#!/usr/bin/env python3
"""A map against a digital twin's fixed geometry: aligned, then how complete
and how accurate it is, per part (floor, walls, ceiling, built-ins, each
appliance and table).

    python3 scoop_pipeline/analysis/twin_compare.py <map .pcd/.ply> <twin .ply> <out dir>
        --traj <TUM of the LiDAR (or camera) in the map's frame>
        [--voxel 0.03] [--pose-step 0.5] [--min-range 1.3] [--max-range 15]
        [--vfov 22.5] [--yaw-candidates 4] [--no-scale]

<twin .ply> is twin_cloud.py's labelled cloud (its .json beside it names the
labels). The map is the room as built (with its ceiling: ed305_room_ceil.ply).

1. Coarse: the map's walls' yaw (crop_room.wall_frame) brings its walls onto
   the twin's axes; each of the four quarter turns is tried, and for each the
   top-down image of tall structure (points 0.3..2.0 m over the floor, 5 cm
   cells) is cross-correlated with the twin's: the best peak gives x, y; the
   floors (the lowest strong height peak) give z.
2. Fine: point-to-plane ICP of the map (--voxel) onto the twin, Tukey-robust
   (sofas, clutter and people are in the map, not in the twin), 25 -> 10 -> 5 cm.
3. Scale: a similarity fit (Umeyama) on the inlier pairs: the map's scale
   error, reported, not applied (--no-scale skips it).
4. What the LiDAR could see: the twin rendered from poses every --pose-step m
   along --traj (moved into the twin by the alignment), a panoramic range
   image of +-vfov deg, 0.35 deg cells, --min-range..--max-range (the map's own
   range gate): a twin point is visible from a pose where nothing of the twin
   is nearer in its cell. Outer wall faces, the slab under the floor and over
   the suspended ceiling, the inside of cabinets are never visible and are
   left out of completeness.
5. Per label, on the visible twin: completeness at 2 / 5 / 10 cm (share of its
   points with a map point that near); and accuracy: the map points whose
   nearest twin point is of that label and within 10 cm, their distance:
   median, mean, p90, RMS. Overall: the share of map points within 5 / 10 cm
   of the twin (the rest: movable furniture, clutter, people, errors).

Status: written, not yet run end to end (the synthetic test against the
real twin is the next step); twin_cloud.py has run on the ED305 twin.

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
                      "t": np.round(Tm[:3, 3], 3).tolist()})
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


def umeyama_scale(A, B):
    """Scale of the similarity taking A onto B (paired points)."""
    ma, mb = A.mean(0), B.mean(0)
    a, b = A - ma, B - mb
    U, S, Vt = np.linalg.svd(b.T @ a / len(A))
    D = np.eye(3)
    D[2, 2] = np.sign(np.linalg.det(U @ Vt))
    return float(np.trace(np.diag(S) @ D) / (a ** 2).sum(1).mean())


def visible(T, poses, min_r, max_r, vfov, cell_deg=0.35, tol=0.03, log=print):
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


# ---------------------------------------------------------------- main
def compare(map_xyz, twin_xyz, twin_lab, names, traj_xyz, voxel=0.03, pose_step=0.5,
            min_r=1.3, max_r=15.0, vfov=22.5, n_yaw=4, scale=True, log=print):
    from scipy.spatial import cKDTree
    import crop_room
    Ms = thin(map_xyz, voxel)
    Ts = thin(twin_xyz, voxel)
    yaw_m = crop_room.wall_frame(map_xyz)[0]
    log("  map: %d points (%d at %.2f m), walls' yaw %.2f deg; twin: %d points (%d)"
        % (len(map_xyz), len(Ms), voxel, yaw_m, len(twin_xyz), len(Ts)))
    T0, score, cands = coarse(Ms, Ts, yaw_m, n_yaw, log=log)
    Tn = normals(Ts)
    X, rms, share = icp(Ms, Ts, Tn, T0, log=log)
    yaw = np.degrees(np.arctan2(X[1, 0], X[0, 0]))
    out = {"T_twin_map": X.round(6).tolist(), "yaw_deg": round(float(yaw), 3),
           "coarse": cands, "icp_rms_mm": round(1000 * rms, 2), "icp_inlier_share": round(share, 4)}
    Mt = map_xyz @ X[:3, :3].T + X[:3, 3]
    tree_t = cKDTree(twin_xyz)
    dm, jm = tree_t.query(Mt, workers=-1)
    if scale:
        inl = dm < 0.05
        sub = np.flatnonzero(inl)[:: max(1, int(inl.sum() // 200000))]
        s = umeyama_scale(Mt[sub], twin_xyz[jm[sub]])
        out["scale"] = round(s, 6)
        out["scale_error_pct"] = round(100 * (s - 1), 3)
        log("  scale against the twin: %.4f (%+.2f %%)" % (s, 100 * (s - 1)))
    poses = poses_along(traj_xyz @ X[:3, :3].T + X[:3, 3], pose_step)
    log("  visibility from %d poses along the path (every %.2f m)" % (len(poses), pose_step))
    Tv = thin(twin_xyz, 0.02)
    lab_v = twin_lab[cKDTree(twin_xyz).query(Tv, workers=-1)[1]]
    seen = visible(Tv, poses, min_r, max_r, vfov, log=log)
    dt, _ = cKDTree(Mt).query(Tv, workers=-1)
    rows = []
    for k, nm in enumerate(names):
        m = lab_v == k
        v = m & seen
        a = (twin_lab[jm] == k) & (dm <= 0.10)
        row = {"label": nm, "twin_points": int(m.sum()), "visible": int(v.sum()),
               "visible_share": round(float(v.sum()) / max(int(m.sum()), 1), 4)}
        for d in (0.02, 0.05, 0.10):
            row["complete_%dcm" % round(d * 100)] = round(float((dt[v] <= d).mean()), 4) if v.any() else None
        if a.any():
            da = dm[a]
            row.update({"map_points": int(a.sum()), "median_mm": round(1000 * float(np.median(da)), 1),
                        "mean_mm": round(1000 * float(da.mean()), 1),
                        "p90_mm": round(1000 * float(np.percentile(da, 90)), 1),
                        "rms_mm": round(1000 * float(np.sqrt(np.mean(da ** 2))), 1)})
        rows.append(row)
    vis_all = seen
    out["per_label"] = rows
    out["overall"] = {
        "map_within_5cm": round(float((dm <= 0.05).mean()), 4),
        "map_within_10cm": round(float((dm <= 0.10).mean()), 4),
        "map_median_mm_within_10cm": round(1000 * float(np.median(dm[dm <= 0.10])), 1),
        "twin_visible_share": round(float(vis_all.mean()), 4),
        "complete_5cm": round(float((dt[vis_all] <= 0.05).mean()), 4),
        "complete_10cm": round(float((dt[vis_all] <= 0.10).mean()), 4)}
    return out, Mt, dm, Tv, seen, dt


def report_md(r, map_path, twin_path):
    o = r["overall"]
    L = ["# Map against the digital twin", "",
         "Map `%s`; twin `%s` (fixed parts only)." % (map_path, twin_path), "",
         "Alignment: yaw %.2f deg, ICP rms %.1f mm on %.1f %% of the map%s." % (
             r["yaw_deg"], r["icp_rms_mm"], 100 * r["icp_inlier_share"],
             "; scale %+.2f %%" % r["scale_error_pct"] if "scale" in r else ""), "",
         "Overall: %.1f %% of the map within 5 cm of the twin, %.1f %% within 10 cm (median %.1f mm); "
         "%.1f %% of the twin visible from the path, %.1f %% of it covered within 5 cm, %.1f %% within 10 cm."
         % (100 * o["map_within_5cm"], 100 * o["map_within_10cm"], o["map_median_mm_within_10cm"],
            100 * o["twin_visible_share"], 100 * o["complete_5cm"], 100 * o["complete_10cm"]), "",
         "| part | visible | complete 2 / 5 / 10 cm | map points | median | mean | p90 | RMS (mm) |",
         "|---|---|---|---|---|---|---|---|"]
    f = lambda x: "-" if x is None else "%.0f" % (100 * x)                 # noqa: E731
    for row in r["per_label"]:
        L.append("| %s | %.0f %% | %s / %s / %s %% | %s | %s | %s | %s | %s |" % (
            row["label"], 100 * row["visible_share"], f(row["complete_2cm"]), f(row["complete_5cm"]),
            f(row["complete_10cm"]), row.get("map_points", 0), row.get("median_mm", "-"),
            row.get("mean_mm", "-"), row.get("p90_mm", "-"), row.get("rms_mm", "-")))
    L += ["", "Completeness: share of the twin visible from the path (LiDAR range and field "
          "of view, nothing of the twin in front) with a map point that near. Accuracy: map "
          "points whose nearest twin point is of that part and within 10 cm. Movable furniture "
          "(sofas, chairs) is not in the twin: the map's points on it count against neither."]
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
    ap.add_argument("--no-scale", action="store_true")
    a = ap.parse_args(argv)
    names = json.load(open(os.path.splitext(os.path.expanduser(a.twin))[0] + ".json"))["labels"]
    twin_xyz, twin_lab = read_labelled_ply(a.twin)
    map_xyz = load_cloud(a.map)
    _, traj = read_tum(a.traj)
    print("%s vs %s" % (a.map, a.twin))
    r, Mt, dm, Tv, seen, dt = compare(map_xyz, twin_xyz, twin_lab, names, traj, a.voxel, a.pose_step,
                                      a.min_range, a.max_range, a.vfov, a.yaw_candidates, not a.no_scale)
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
