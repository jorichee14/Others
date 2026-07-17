#!/usr/bin/env python3
"""
STAGE 01 - build the map cloud from LiDAR scans placed by GLIM poses.

  bag (/ouster/points) + traj_lidar.txt  ->  merge -> [remove dynamic]
  -> denoise -> colorize -> [flatten] -> [anchor to camera start] -> map_final.pcd

Everything (paths, voxels, toggles, sensor calib, topics) comes from
pipeline_config.json + the calibration.json it points at. Nothing hardcoded.

Intermediate stages are written to out_dir so a re-run can resume from merge:
  merged.pcd  [static.pcd]  denoised.pcd  colored.pcd  [flattened.pcd]  [anchored.pcd]

DYNAMIC-OBJECT REMOVAL (moving people / vehicles), stage [1b], optional:
  Static structure is re-hit by many scans over a long time whenever it is in
  the LiDAR FOV; a moving object only deposits points in any given world voxel
  during the brief instant it passes through. So we bucket every scan's world
  points into a global voxel grid and, per voxel, keep the number of DISTINCT
  scans that hit it and the first/last hit time. A voxel is kept as static when
      (last_hit - first_hit) >= min_span_s   AND   hits >= min_hits
  and dropped as dynamic otherwise. The time SPAN is the real discriminator --
  static structure is observed across a long stretch of the run, a moving object
  leaves only a short contiguous burst in any one voxel; min_hits is just a
  noise guard so a lone stray return isn't its own "static" voxel. This reuses
  merge's exact scan<->pose
  association and, on a fresh run, is collected DURING the merge pass (no extra
  bag read); on a resume from merged.pcd it does one dedicated points pass.

  Config block (under 01_build_map, all optional -- omit to disable):
    "remove_dynamic": {
      "enable": true,
      "voxel": 0.15,       # decision grid size, metres (coarser = more aggressive)
      "min_span_s": 1.0,   # PRIMARY: seen across >= this many seconds -> static
      "min_hits": 2,       # noise guard: also need >= this many distinct scans
      "save": true         # write the cleaned cloud to static.pcd
    }
  Tuning: set min_span_s above a mover's dwell time in one voxel (a person walking
  past clears a 0.15 m voxel in well under a second) but below how long static
  structure stays observed. Tradeoff: genuinely static geometry only ever glimpsed
  in one short window (short span) can be dropped too, so prefer runs that
  revisit/sweep the scene. Slow-moving or temporarily-stationary objects that
  dwell in a voxel longer than min_span_s are, by construction, indistinguishable
  from static structure and will survive.

  python3 01_build_map.py [pipeline_config.json]
"""
import os
import sys
import shutil
import numpy as np
import open3d as o3d
import cv2
from pathlib import Path
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore

from pipeline_common import load_pipeline

TS = get_typestore(Stores.ROS2_HUMBLE)

# ---- global voxel-key packing for the dynamic filter -------------------------
# Pack a signed integer voxel index (vx, vy, vz) into one int64 so per-voxel
# stats live in flat numpy arrays. 20 bits/axis (+/-524287 voxels) stays well
# inside int64 and, at a 0.1 m grid, spans +/-52 km -- far beyond any real map.
_VOX_OFF = 1 << 19            # 524288, shifts indices non-negative
_VOX_BITS = 20               # 3 * 20 = 60 bits used


def pack_voxels(vox):
    """(N,3) int64 voxel indices -> (N,) int64 keys (bijective within range)."""
    v = vox.astype(np.int64) + _VOX_OFF
    return (v[:, 0] << (2 * _VOX_BITS)) | (v[:, 1] << _VOX_BITS) | v[:, 2]


def pc2_xyz(msg):
    off = {f.name: f.offset for f in msg.fields}
    raw = np.frombuffer(bytes(msg.data), np.uint8).reshape(-1, msg.point_step)
    def c(n): return raw[:, off[n]:off[n] + 4].copy().view(np.float32).ravel()
    return np.stack([c("x"), c("y"), c("z")], 1)


def decode_img(msg):
    enc = msg.encoding.lower()
    buf = np.frombuffer(bytes(msg.data), np.uint8)
    h, w = msg.height, msg.width
    if enc in ("bgra8", "rgba8"):
        img = buf.reshape(h, w, 4)[:, :, :3]
        if enc == "rgba8":
            img = img[:, :, ::-1]
    elif enc in ("bgr8", "rgb8"):
        img = buf.reshape(h, w, 3)
        if enc == "rgb8":
            img = img[:, :, ::-1]
    else:
        return None
    return np.ascontiguousarray(img)


def _pc(points):
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points)
    return pc


def nearest_pose_idx(tr_t, t):
    i = int(np.searchsorted(tr_t, t))
    if i <= 0:
        return 0
    if i >= len(tr_t):
        return len(tr_t) - 1
    return i if (tr_t[i] - t) < (t - tr_t[i - 1]) else i - 1


def iter_world_scans(P, S, s):
    """Yield (t, world_points) for every LiDAR scan that associates to a GLIM
    pose within time_tol, range-filtered. Single source of the scan<->pose
    placement so merge and the dynamic filter stay in lock-step."""
    tr_t, tr_T = P.traj
    lo, hi = s["lidar_min"], s["lidar_max"]; tol = s["time_tol"]
    with AnyReader([Path(P.dataset["bag"])], default_typestore=TS) as r:
        conns = [c for c in r.connections if c.topic == S.points_topic]
        for conn, _, raw in r.messages(connections=conns):
            msg = r.deserialize(raw, conn.msgtype)
            t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            j = nearest_pose_idx(tr_t, t)
            if abs(tr_t[j] - t) > tol:
                continue
            p = pc2_xyz(msg)
            d = np.linalg.norm(p, axis=1)
            p = p[np.isfinite(p).all(1) & (d > lo) & (d < hi)]
            if p.shape[0] == 0:
                continue
            Tw = tr_T[j]
            yield t, (Tw[:3, :3] @ p.T).T + Tw[:3, 3]


class DynStats:
    """Per-voxel occupancy accumulator for dynamic-object removal. add() one
    scan's world points at a time; dynamic_keys() returns the transient voxels."""
    def __init__(self, voxel):
        self.inv = 1.0 / float(voxel)
        self._keys = []   # per-scan unique voxel keys
        self._time = []   # matching scan time, broadcast per key

    def add(self, world_pts, t):
        vox = np.floor(world_pts * self.inv).astype(np.int64)
        u = np.unique(pack_voxels(vox))
        if u.size:
            self._keys.append(u)
            self._time.append(np.full(u.shape, float(t)))

    def dynamic_keys(self, min_hits, min_span_s):
        """Sorted int64 keys of voxels judged transient (moving objects)."""
        if not self._keys:
            return np.empty(0, np.int64), (0, 0)
        keys = np.concatenate(self._keys)
        times = np.concatenate(self._time)
        order = np.argsort(keys, kind="stable")
        keys = keys[order]; times = times[order]
        uniq, start = np.unique(keys, return_index=True)
        hits = np.diff(np.append(start, keys.size))         # distinct scans / voxel
        tmin = np.minimum.reduceat(times, start)
        tmax = np.maximum.reduceat(times, start)
        span = tmax - tmin
        # span is the real discriminator: static structure is observed across a
        # long stretch of the run; a moving object leaves only a short contiguous
        # burst in any one voxel. min_hits is just a noise guard (drop lone/stray
        # returns), so the test is AND, not OR -- a low hit count must NOT rescue a
        # short-span burst, which is precisely what a slow/large mover produces.
        static = (span >= float(min_span_s)) & (hits >= int(min_hits))
        dyn = uniq[~static]
        return dyn, (int(static.sum()), int((~static).sum()))


def drop_dynamic_points(pcd, dyn_keys, voxel):
    """Return pcd with points whose global voxel is in dyn_keys removed."""
    if len(pcd.points) == 0 or dyn_keys.size == 0:
        return pcd, 0
    pts = np.asarray(pcd.points)
    vox = np.floor(pts * (1.0 / float(voxel))).astype(np.int64)
    packed = pack_voxels(vox)
    # dyn_keys is sorted -> membership via searchsorted (fast, low memory)
    pos = np.searchsorted(dyn_keys, packed)
    pos = np.clip(pos, 0, dyn_keys.size - 1)
    is_dyn = dyn_keys[pos] == packed
    keep = np.where(~is_dyn)[0]
    out = pcd.select_by_index(keep)
    return out, int(is_dyn.sum())


def merge(P, S, s, dyn=None):
    print("[1] merge: LiDAR scans -> world cloud"
          + (" (+ dynamic-voxel stats)" if dyn is not None else ""))
    scan_voxel = s["scan_voxel"]; final_voxel = s["final_voxel"]
    flush = s["flush_every"]
    buf = []; compressed = None; n = 0

    def compress():
        nonlocal buf, compressed
        if not buf:
            return compressed
        stack = np.vstack(buf) if compressed is None else np.vstack([compressed] + buf)
        if scan_voxel > 0:
            stack = np.asarray(_pc(stack).voxel_down_sample(scan_voxel).points)
        return stack

    for t, wp in iter_world_scans(P, S, s):
        if dyn is not None:
            dyn.add(wp, t)                 # collect stats before downsampling
        buf.append(wp)
        n += 1
        if n % flush == 0:
            compressed = compress(); buf = []
            print(f"    {n} scans, {0 if compressed is None else len(compressed)} pts")
    compressed = compress()
    m = _pc(compressed if compressed is not None else np.empty((0, 3)))
    if final_voxel > 0:
        m = m.voxel_down_sample(final_voxel)
    print(f"    merged {n} scans -> {len(m.points)} pts")
    return m


def remove_dynamic(P, S, s, pcd, dyn):
    """Apply the dynamic filter using an already-populated DynStats (fresh run)
    or, if none was collected (resume), do one dedicated points pass first."""
    rd = s["remove_dynamic"]
    voxel = rd.get("voxel", 0.15)
    if dyn is None:
        print("[1b] remove-dynamic: replaying scans for voxel stats (resume path)")
        dyn = DynStats(voxel)
        for t, wp in iter_world_scans(P, S, s):
            dyn.add(wp, t)
    dyn_keys, (n_static, n_dyn) = dyn.dynamic_keys(rd.get("min_hits", 2),
                                                   rd.get("min_span_s", 1.0))
    out, removed = drop_dynamic_points(pcd, dyn_keys, voxel)
    print(f"[1b] remove-dynamic: {n_static} static / {n_dyn} dynamic voxels "
          f"(grid {voxel} m) -> dropped {removed} pts, kept {len(out.points)}")
    return out


def colorize(P, S, s, pcd):
    print("[3] colorize: best-view projection (KDTree-culled)")
    tr_t, tr_T = P.traj
    W, H = s["image_width"], s["image_height"]
    fx, fy, cx, cy = S.fx, S.fy, S.cx, S.cy
    c = s["colorize"]; stride = c["img_stride"]; max_range = c["max_range"]
    on_voxel = c["voxel"]; drop_gray = c["drop_gray"]

    if on_voxel > 0 and len(pcd.points) > 0:
        work = pcd.voxel_down_sample(on_voxel)
        print(f"    coloring downsampled copy: {len(work.points)} pts (from {len(pcd.points)})")
    else:
        work = pcd

    pts = np.asarray(work.points, dtype=np.float64)
    N = len(pts)
    colors = np.full((N, 3), 0.5)
    best = np.full(N, np.inf)
    kdt = o3d.geometry.KDTreeFlann(work)

    with AnyReader([Path(P.dataset["bag"])], default_typestore=TS) as r:
        conns = [cc for cc in r.connections if cc.topic == S.image_topic]
        n = 0
        for conn, _, raw in r.messages(connections=conns):
            n += 1
            if n % stride:
                continue
            msg = r.deserialize(raw, conn.msgtype)
            t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            j = nearest_pose_idx(tr_t, t)
            if abs(tr_t[j] - t) > s["time_tol"]:
                continue
            Twc = tr_T[j] @ S.T_lidar_camera
            cam = Twc[:3, 3]
            k, idx, _ = kdt.search_radius_vector_3d(cam, max_range)
            if k == 0:
                continue
            idx = np.asarray(idx); sub = pts[idx]
            Tcw = np.linalg.inv(Twc)
            Xc = (Tcw[:3, :3] @ sub.T).T + Tcw[:3, 3]
            z = Xc[:, 2]
            fr = z > 1e-3
            u = np.full(len(sub), -1.0); v = np.full(len(sub), -1.0)
            u[fr] = fx * Xc[fr, 0] / z[fr] + cx
            v[fr] = fy * Xc[fr, 1] / z[fr] + cy
            inb = fr & (u >= 0) & (u < W) & (v >= 0) & (v < H) & (z < max_range)
            if not inb.any():
                continue
            g_idx = idx[inb]; zc = z[inb]
            uu = u[inb].astype(np.int64); vv = v[inb].astype(np.int64)
            pix = vv * W + uu
            order = np.argsort(zc)
            _, first = np.unique(pix[order], return_index=True)
            keep = order[first]
            g_keep = g_idx[keep]; z_keep = zc[keep]
            uu_k = uu[keep]; vv_k = vv[keep]
            better = z_keep < best[g_keep]
            if not better.any():
                continue
            img = decode_img(msg)
            if img is None:
                continue
            if (img.shape[1], img.shape[0]) != (W, H):
                img = cv2.resize(img, (W, H))
            gb = g_keep[better]
            colors[gb] = img[vv_k[better], uu_k[better]][:, ::-1] / 255.0
            best[gb] = z_keep[better]
            if n % 750 == 0:
                print(f"    img {n}")

    seen = np.isfinite(best)
    print(f"    colored {seen.sum()}/{N} ({100 * seen.mean():.1f}%)")
    work.colors = o3d.utility.Vector3dVector(colors)

    if on_voxel > 0 and work is not pcd:
        print("    propagating colors to full-res via nearest neighbor")
        full = np.asarray(pcd.points)
        kdt2 = o3d.geometry.KDTreeFlann(work)
        wcol = np.asarray(work.colors)
        fcol = np.full((len(full), 3), 0.5)
        nn = np.empty(len(full), np.int64)
        for i in range(len(full)):
            _, ii, _ = kdt2.search_knn_vector_3d(full[i], 1)
            nn[i] = ii[0]
        fcol = wcol[nn]
        pcd.colors = o3d.utility.Vector3dVector(fcol)
        if drop_gray:
            pcd = pcd.select_by_index(np.where(seen[nn])[0])
            print(f"    dropped gray -> {len(pcd.points)}")
        return pcd

    if drop_gray:
        pcd = work.select_by_index(np.where(seen)[0])
        print(f"    dropped gray -> {len(pcd.points)}")
        return pcd
    return work


def flatten(s, pcd):
    print("[4] plane-flatten: RANSAC projection")
    f = s["flatten"]; dmin = f["min"]; dist = f["dist"]; maxp = f["max_planes"]
    has_col = pcd.has_colors(); rest = pcd; parts = []
    for _ in range(maxp):
        if len(rest.points) < dmin:
            break
        model, inl = rest.segment_plane(dist, 3, 1000)
        if len(inl) < dmin:
            break
        nrm = np.array(model[:3]); nn = nrm / np.linalg.norm(nrm); d = model[3] / np.linalg.norm(nrm)
        pl = rest.select_by_index(inl); pp = np.asarray(pl.points)
        proj = pp - np.outer(pp @ nn + d, nn)
        g = o3d.geometry.PointCloud(); g.points = o3d.utility.Vector3dVector(proj)
        if has_col:
            g.colors = pl.colors
        parts.append(g)
        rest = rest.select_by_index(inl, invert=True)
    out = o3d.geometry.PointCloud()
    allp = [np.asarray(g.points) for g in parts] + [np.asarray(rest.points)]
    out.points = o3d.utility.Vector3dVector(np.vstack(allp))
    if has_col:
        allc = [np.asarray(g.colors) for g in parts] + [np.asarray(rest.colors)]
        out.colors = o3d.utility.Vector3dVector(np.vstack(allc))
    print(f"    {len(parts)} planes flattened, {len(rest.points)} non-planar kept")
    return out


def save(P, pcd, name):
    path = P.outp(name)
    o3d.io.write_point_cloud(path, pcd)
    print(f"    saved {path}  ({len(pcd.points)} pts)")
    return path


def main():
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "pipeline_config.json"
    P = load_pipeline(cfg_path)
    S = P.sensor
    s = P.stage("01_build_map")
    P.traj = load_traj_cached(P)
    print(f"loaded {len(P.traj[0])} GLIM poses; outputs -> {P.out_dir}/")

    rd = s.get("remove_dynamic", {})
    rd_on = bool(rd.get("enable", False))

    merged_p = P.outp("merged.pcd")
    dyn = None
    if os.path.exists(merged_p):
        print("[resume] loading existing merged.pcd (delete to rebuild)")
        pcd = o3d.io.read_point_cloud(merged_p)
        print(f"    {len(pcd.points)} pts")
        # dyn stays None -> remove_dynamic() will replay one points pass if needed
    else:
        dyn = DynStats(rd.get("voxel", 0.15)) if rd_on else None
        pcd = merge(P, S, s, dyn); save(P, pcd, "merged.pcd")

    if rd_on:
        pcd = remove_dynamic(P, S, s, pcd, dyn)
        if rd.get("save", True):
            save(P, pcd, "static.pcd")

    if s["denoise"]["enable"]:
        print("[2] denoise: light statistical outlier removal")
        pcd, _ = pcd.remove_statistical_outlier(s["denoise"]["nb"], s["denoise"]["std"])
        save(P, pcd, "denoised.pcd")

    if s["colorize"]["enable"]:
        pcd = colorize(P, S, s, pcd); save(P, pcd, "colored.pcd")

    if s["flatten"]["enable"]:
        pcd = flatten(s, pcd); save(P, pcd, "flattened.pcd")

    if s["anchor_camera_start"]:
        print("[5] anchor: origin at camera start (z-up)")
        tr_t, tr_T = P.traj
        cam0 = (tr_T[0] @ S.T_lidar_camera)[:3, 3]
        pcd.translate(-cam0)
        print(f"    shift {(-cam0).round(3)}  (NOTE: stage 03 must know this via "
              f"01_build_map.anchor_camera_start)")
        save(P, pcd, "anchored.pcd")

    final = P.outp(s["output"])
    o3d.io.write_point_cloud(final, pcd)
    print(f"DONE -> {final}  ({len(pcd.points)} points)")


def load_traj_cached(P):
    from pipeline_common import load_traj
    return load_traj(P.dataset["traj"])


if __name__ == "__main__":
    main()
