#!/usr/bin/env python3
"""
STAGE 01 - build the map cloud from LiDAR scans placed by GLIM poses.

  bag (/ouster/points) + traj_lidar.txt  ->  merge -> [remove dynamic]
  -> denoise -> colorize -> [flatten] -> [anchor to camera start] -> map_final.pcd

Everything (paths, voxels, toggles, sensor calib, topics) comes from
pipeline_config.json + the calibration.json it points at. Nothing hardcoded.

Intermediate stages are written to out_dir so a re-run can resume from merge:
  merged.pcd  [static.pcd]  denoised.pcd  colored.pcd  [flattened.pcd]  [anchored.pcd]

DYNAMIC-OBJECT REMOVAL (moving people / vehicles), stage [1b], optional.
Two complementary tests, combined per voxel of a global decision grid:

  (i) TIME-SPAN test (cheap, catches fast movers):
  Static structure is re-hit by many scans over a long time whenever it is in
  the LiDAR FOV; a moving object only deposits points in any given world voxel
  during the brief instant it passes through. We bucket every scan's world
  points into the grid and, per voxel, keep the number of DISTINCT scans that
  hit it and the first/last hit time. A voxel passes as static when
      (last_hit - first_hit) >= min_span_s   AND   hits >= min_hits
  LIMITATION: anything that DWELLS in a voxel longer than min_span_s -- a
  person standing still, someone walking alongside the robot, a slow mover, a
  car that parks and later leaves -- is indistinguishable from structure by
  this test alone and survives. That is exactly the residue you see when "the
  output still has dynamic points".

  (ii) FREE-SPACE CARVING (visibility test, catches dwellers and slow movers):
  Every LiDAR return is also a statement that the ray from the sensor to the
  hit passed through EMPTY space. We march each (subsampled) ray through the
  decision grid and count, per voxel, in how many distinct scans it was seen
  THROUGH (= provably empty at that time). A voxel that has both occupancy
  hits AND many free-space observations held an object that was sometimes
  there and sometimes not -> dynamic, no matter how long it dwelled:
      dynamic if  free >= min_free  AND  free >= free_ratio * hits
  Genuinely static structure is (almost) never seen through -- rays stop at
  it -- so noise-level free counts are absorbed by min_free / free_ratio.
  This is the same principle as Removert / ERASOR / OctoMap occupancy.

  Both tests reuse merge's exact scan<->pose association and, on a fresh run,
  are collected DURING the merge pass (one bag read); on a resume from
  merged.pcd one dedicated points pass replays the scans.

  Config block (under 01_build_map, all optional -- omit to disable):
    "remove_dynamic": {
      "enable": true,
      "voxel": 0.15,       # decision grid size, metres (coarser = more aggressive)
      "min_span_s": 1.0,   # span test: seen across >= this many seconds -> static
      "min_hits": 2,       # noise guard: also need >= this many distinct scans
      "save": true,        # write the cleaned cloud to static.pcd
      "carve": {
        "enable": true,     # free-space carving (the dweller/slow-mover fix)
        "ray_stride": 4,    # march every k-th point of a scan (cost knob)
        "scan_stride": 2,   # carve every k-th scan (cost knob)
        "max_range": 20.0,  # ignore ray free-space beyond this (m)
        "endpoint_margin": 0.0,  # extra pull-back from the hit, on top of the
                                 # built-in 2-voxel guard band (m)
        "min_free": 3,      # voxel must be seen THROUGH in >= this many scans
        "free_ratio": 0.25  # ...and free >= this fraction of its hit count
      }
    }
  Tuning: with carving on, min_span_s only needs to catch fast movers (leave
  at ~1.0); carving handles everything that dwells. If thin static structure
  (railings, poles, foliage) starts disappearing, raise min_free / free_ratio
  or the endpoint_margin; if ghosts survive, lower free_ratio (e.g. 0.1) or
  scan_stride/ray_stride (denser evidence). Carving costs roughly one extra
  50-70%% of merge time at the default strides.

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
    """Yield (t, world_points, sensor_origin) for every LiDAR scan that
    associates to a GLIM pose within time_tol, range-filtered. Single source
    of the scan<->pose placement so merge and the dynamic filter (span stats
    AND ray carving) stay in lock-step."""
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
            yield t, (Tw[:3, :3] @ p.T).T + Tw[:3, 3], Tw[:3, 3].copy()


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

    def dynamic_keys(self, min_hits, min_span_s, carver=None,
                     min_free=3, free_ratio=0.25):
        """Sorted int64 keys of voxels judged transient (moving objects).
        Returns (keys, (n_static, n_dyn, n_carved))."""
        if not self._keys:
            return np.empty(0, np.int64), (0, 0, 0)
        keys = np.concatenate(self._keys)
        times = np.concatenate(self._time)
        order = np.argsort(keys, kind="stable")
        keys = keys[order]; times = times[order]
        uniq, start = np.unique(keys, return_index=True)
        hits = np.diff(np.append(start, keys.size))         # distinct scans / voxel
        tmin = np.minimum.reduceat(times, start)
        tmax = np.maximum.reduceat(times, start)
        span = tmax - tmin
        # span is the fast-mover discriminator: static structure is observed
        # across a long stretch of the run; a moving object leaves only a short
        # contiguous burst in any one voxel. min_hits is just a noise guard
        # (drop lone/stray returns), so the test is AND, not OR -- a low hit
        # count must NOT rescue a short-span burst.
        static = (span >= float(min_span_s)) & (hits >= int(min_hits))
        n_carved = 0
        if carver is not None:
            # visibility test: a voxel seen THROUGH by rays in min_free+ scans
            # was empty at those times; if it also has occupancy hits, whatever
            # sat there came and went -> dynamic even if it dwelled for minutes.
            free = carver.counts_for(uniq)
            carved = (free >= int(min_free)) & \
                     (free.astype(np.float64) >= float(free_ratio) * hits)
            n_carved = int((carved & static).sum())
            static &= ~carved
        dyn = uniq[~static]
        return dyn, (int(static.sum()), int((~static).sum()), n_carved)


class FreeSpaceCarver:
    """Free-space evidence accumulator on the SAME decision grid as DynStats.

    add() marches (subsampled) rays from the sensor origin toward each return,
    sampling one point per voxel-length, stopping a guard band short of the
    hit so surface voxels are never carved by their own rays. Per scan, the
    set of traversed voxels is deduplicated, so counts_for() reports "seen
    through in how many distinct scans". Buffered + periodically compacted so
    memory stays bounded on long runs."""

    def __init__(self, voxel, max_range=20.0, ray_stride=4, scan_stride=2,
                 endpoint_margin=0.0, chunk=8000, compact_at=20_000_000):
        self.voxel = float(voxel)
        self.inv = 1.0 / self.voxel
        self.step = self.voxel
        self.max_range = float(max_range)
        self.ray_stride = max(1, int(ray_stride))
        self.scan_stride = max(1, int(scan_stride))
        # never carve closer than 2 voxels to the hit: range noise + grazing
        # incidence would otherwise eat the surface itself
        self.margin = 2.0 * self.voxel + float(endpoint_margin)
        self.chunk = int(chunk)
        self.compact_at = int(compact_at)
        self.keys = np.empty(0, np.int64)
        self.counts = np.empty(0, np.int64)
        self._buf = []
        self._buf_n = 0
        self._n_scan = 0

    def add(self, origin, world_pts):
        self._n_scan += 1
        if (self._n_scan - 1) % self.scan_stride:
            return
        p = world_pts[::self.ray_stride]
        vec = p - origin[None, :]
        dist = np.linalg.norm(vec, axis=1)
        m = dist > (self.margin + self.step)
        if not m.any():
            return
        vec = vec[m]; dist = dist[m]
        end = np.minimum(dist - self.margin, self.max_range)
        scan_keys = []
        for a in range(0, len(vec), self.chunk):
            b = min(a + self.chunk, len(vec))
            d = dist[a:b]; e = end[a:b]
            dirs = vec[a:b] / d[:, None]
            n_steps = int(np.ceil(e.max() / self.step))
            if n_steps <= 0:
                continue
            t = (np.arange(n_steps) + 0.5) * self.step          # (S,)
            valid = t[None, :] < e[:, None]                     # (n,S)
            pts = origin[None, None, :] + dirs[:, None, :] * t[None, :, None]
            vox = np.floor(pts[valid] * self.inv).astype(np.int64)
            if vox.size:
                scan_keys.append(pack_voxels(vox))
        if not scan_keys:
            return
        u = np.unique(np.concatenate(scan_keys))
        self._buf.append(u)
        self._buf_n += u.size
        if self._buf_n >= self.compact_at:
            self._compact()

    def _compact(self):
        if not self._buf:
            return
        k = np.concatenate(self._buf)
        self._buf = []; self._buf_n = 0
        uniq, inv = np.unique(k, return_inverse=True)
        c = np.bincount(inv).astype(np.int64)
        if self.keys.size:
            comb = np.concatenate([self.keys, uniq])
            combc = np.concatenate([self.counts, c])
            uniq, inv = np.unique(comb, return_inverse=True)
            c = np.bincount(inv, weights=combc).astype(np.int64)
        self.keys, self.counts = uniq, c

    def counts_for(self, query):
        """Free-scan count per (sorted or unsorted) packed voxel key."""
        self._compact()
        if self.keys.size == 0:
            return np.zeros(query.shape, np.int64)
        pos = np.searchsorted(self.keys, query)
        pos = np.clip(pos, 0, self.keys.size - 1)
        return np.where(self.keys[pos] == query, self.counts[pos], 0)


def make_carver(rd, s):
    """Build a FreeSpaceCarver from the remove_dynamic config, or None."""
    cv = rd.get("carve", {})
    if not cv.get("enable", True):
        return None
    voxel = rd.get("voxel", 0.15)
    return FreeSpaceCarver(
        voxel,
        max_range=min(float(cv.get("max_range", 20.0)), float(s["lidar_max"])),
        ray_stride=cv.get("ray_stride", 4),
        scan_stride=cv.get("scan_stride", 2),
        endpoint_margin=cv.get("endpoint_margin", 0.0))


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


def merge(P, S, s, dyn=None, carver=None):
    extras = [x for x, on in (("dynamic-voxel stats", dyn is not None),
                              ("free-space carving", carver is not None)) if on]
    print("[1] merge: LiDAR scans -> world cloud"
          + (f" (+ {', '.join(extras)})" if extras else ""))
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

    for t, wp, origin in iter_world_scans(P, S, s):
        if dyn is not None:
            dyn.add(wp, t)                 # collect stats before downsampling
        if carver is not None:
            carver.add(origin, wp)
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


def remove_dynamic(P, S, s, pcd, dyn, carver):
    """Apply the dynamic filter using already-populated stats (fresh run) or,
    if none were collected (resume), do one dedicated points pass first."""
    rd = s["remove_dynamic"]
    voxel = rd.get("voxel", 0.15)
    cv = rd.get("carve", {})
    if dyn is None:
        print("[1b] remove-dynamic: replaying scans for voxel stats (resume path)")
        dyn = DynStats(voxel)
        carver = make_carver(rd, s)
        for t, wp, origin in iter_world_scans(P, S, s):
            dyn.add(wp, t)
            if carver is not None:
                carver.add(origin, wp)
    dyn_keys, (n_static, n_dyn, n_carved) = dyn.dynamic_keys(
        rd.get("min_hits", 2), rd.get("min_span_s", 1.0),
        carver=carver,
        min_free=cv.get("min_free", 3),
        free_ratio=cv.get("free_ratio", 0.25))
    out, removed = drop_dynamic_points(pcd, dyn_keys, voxel)
    carve_note = (f", {n_carved} of them span-static but carved by free-space"
                  if carver is not None else " (carving off)")
    print(f"[1b] remove-dynamic: {n_static} static / {n_dyn} dynamic voxels "
          f"(grid {voxel} m){carve_note} -> dropped {removed} pts, "
          f"kept {len(out.points)}")
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
        print("    propagating colors to full-res via nearest neighbor (batched cKDTree)")
        from scipy.spatial import cKDTree
        full = np.asarray(pcd.points)
        wpts = np.asarray(work.points)
        wcol = np.asarray(work.colors)
        tree = cKDTree(wpts)
        nn = np.empty(len(full), np.int64)
        chunk = 5_000_000  # bound peak memory on 100M+ clouds
        for a in range(0, len(full), chunk):
            b = min(a + chunk, len(full))
            _, nn[a:b] = tree.query(full[a:b], workers=-1)
            print(f"    propagated {b}/{len(full)}", flush=True)
        pcd.colors = o3d.utility.Vector3dVector(wcol[nn])
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

    # Resume from the furthest completed stage on disk. Delete a stage's .pcd
    # to force it (and everything after) to recompute.
    denoised_p = P.outp("denoised.pcd")
    static_p = P.outp("static.pcd")
    merged_p = P.outp("merged.pcd")
    dyn = None
    carver = None

    if s["denoise"]["enable"] and os.path.exists(denoised_p):
        print("[resume] loading existing denoised.pcd (delete to redo denoise/merge)")
        pcd = o3d.io.read_point_cloud(denoised_p)
        print(f"    {len(pcd.points)} pts")
    elif rd_on and os.path.exists(static_p):
        print("[resume] loading existing static.pcd (delete to redo dynamic/merge)")
        pcd = o3d.io.read_point_cloud(static_p)
        print(f"    {len(pcd.points)} pts")
        if s["denoise"]["enable"]:
            print("[2] denoise: light statistical outlier removal")
            pcd, _ = pcd.remove_statistical_outlier(s["denoise"]["nb"], s["denoise"]["std"])
            save(P, pcd, "denoised.pcd")
    else:
        if os.path.exists(merged_p):
            print("[resume] loading existing merged.pcd (delete to rebuild)")
            pcd = o3d.io.read_point_cloud(merged_p)
            print(f"    {len(pcd.points)} pts")
        else:
            if rd_on:
                dyn = DynStats(rd.get("voxel", 0.15))
                carver = make_carver(rd, s)
            pcd = merge(P, S, s, dyn, carver); save(P, pcd, "merged.pcd")

        if rd_on:
            pcd = remove_dynamic(P, S, s, pcd, dyn, carver)
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
