#!/usr/bin/env python3
"""
STAGE 01 - build the map cloud from LiDAR scans placed by GLIM poses.

  bag (/ouster/points) + traj_lidar.txt  ->  merge -> [remove dynamic]
  -> denoise -> colorize -> [detect objects] -> [flatten]
  -> [anchor to camera start] -> map_final.pcd

Everything (paths, voxels, toggles, sensor calib, topics) comes from
pipeline_config.json + the calibration.json it points at. Nothing hardcoded.

Intermediate stages are written to out_dir so a re-run can resume from merge:
  merged.pcd  [static.pcd]  denoised.pcd  colored.pcd  [flattened.pcd]  [anchored.pcd]

The optional detection stage adds a SEMANTIC LAYER on top of the geometry:
  background.pcd  objects.pcd  objects_by_instance.pcd  objects_inventory.yaml

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

OBJECT DETECTION + INVENTORY, stage [6], optional:
  A YOLO detector runs on the camera stream and its 2D masks/boxes are lifted
  onto the 3D map, giving a layered result instead of one undifferentiated
  cloud:

    layer 0  map_final.pcd            everything (geometry, as before)
    layer 1  background.pcd           map MINUS all detected objects -- the
                                      clean structural shell (best input for
                                      the mesher / Sionna)
    layer 2  objects.pcd              only object points, photo colours
             objects_by_instance.pcd  same points, one colour per instance
             objects/<id>_<label>.pcd optional per-object clouds
    layer 3  objects_inventory.yaml   the inventory: class, position, size,
                                      orientation, footprint, point count,
                                      colour, per-class totals

  How a 2D detection becomes a 3D object:
    1. For each (strided) image, the map points visible from that pose are
       found by radius cull + pinhole projection + z-buffer, so an occluded
       point can never receive a label meant for the surface in front of it.
    2. Each visible point inherits the class of the mask pixel it lands on
       (segmentation weights strongly preferred; plain boxes are shrunk by
       bbox_shrink because a box corner is background), and a depth-band gate
       around the detection's median depth stops a detection from labelling
       the wall behind it.
    3. Votes ACCUMULATE over every frame in which the point was visible. A
       label survives only with multi-view agreement -- min_votes absolute and
       min_ratio relative to how often that point was actually seen -- which
       is what kills single-frame false positives and silhouette bleed.
    4. Surviving points are DBSCAN-clustered per class in 3D; each cluster is
       one inventory entry, measured with a PCA footprint (yaw + extents +
       height, robust where a full 3D OBB degenerates on flat clusters).

  Config block (under 01_build_map, omit to disable):
    "detect_objects": {
      "enable": true,
      "model": "yolo11n-seg.pt",   # any ultralytics model; -seg strongly preferred
      "device": "cuda:0",          # omit for auto
      "conf": 0.35, "iou": 0.5, "imgsz": 640,
      "img_stride": 5,             # YOLO is the slow part -- stride the images
      "voxel": 0.05,               # voting resolution (labels propagate to full res)
      "max_range": 8.0,            # only label points closer than this (m)
      "classes": [],               # [] = every class the model knows
      "exclude": ["person"],       # never inventory these (dynamic anyway)
      "bbox_shrink": 0.12,         # box-only models: trim this fraction per side
      "mask_erode": 2,             # segmentation: erode masks by N px
      "depth_band": 1.0,           # keep points within +-this of median depth (m)
      "min_pixels": 60,            # ignore detections smaller than this on screen
      "min_votes": 3,              # label needs >= this many frames agreeing
      "min_ratio": 0.35,           # ...and >= this fraction of frames seen
      "cluster": { "eps": 0.12, "min_points": 60, "min_pts_keep": 120 },
      "structure_veto": {         # reject detections that landed on structure
        "enable": true,
        "voxel": 0.08,            # plane-fitting resolution (m)
        "plane_dist": 0.06,       # RANSAC inlier distance (m)
        "min_area": 4.0,          # only veto against planes this large (m^2)
        "max_planes": 40,
        "flush_tol": 0.05,        # voted points within this of a big plane
                                  #   are structure, not object
        "min_protrusion": 0.04    # ...and a whole cluster must stand off the
                                  #   nearest plane by at least this much
      },
      "save_layers": true, "save_per_object": false,
      "inventory": "objects_inventory.yaml"
    }
  On the structural veto: multi-view voting measures CONSISTENCY, not
  correctness, so a detector that fires on the same wall patch from every view
  ("tv" on a poster or a dark rectangle) earns a perfect vote ratio and fusion
  alone can never reject it. The veto answers that with geometry -- flush on a
  large plane means structure -- and it also trims the wall halo around
  genuinely wall-mounted objects, where the depth band cannot help because the
  wall behind sits at the same depth. Raise flush_tol/min_protrusion if wall
  fuzz still enters the inventory; LOWER them if thin wall-mounted objects
  (flat TVs, radiators) vanish, and raise min_area if large flat objects
  (counters, table tops) are being treated as structure.
  Needs `pip install ultralytics`. Run detection AFTER dynamic removal so
  people/movers are already gone -- the two stages are complementary: carving
  removes what moved, detection names what stayed.

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

# =============================================================================
# GPU BACKEND
# =============================================================================
# Stage 01's cost is concentrated in a handful of embarrassingly parallel
# kernels: the per-frame projection in colorize [3] and detect [6], the ray
# march in free-space carving [1b], and the very large int64 key sorts behind
# both voxel filters. Every one of them is plain array arithmetic, so the whole
# stage runs on numpy or cupy through ONE module handle -- there is no second
# code path to keep in sync, and anything the GPU cannot do falls back with a
# message instead of failing.
#
# Deliberately kept on the CPU: bag reading and message deserialization (I/O
# and Python bound, no kernel to run), Poisson/RANSAC/DBSCAN (Open3D has no
# CUDA path for them), and Open3D's voxel_down_sample, whose C++ implementation
# is already fast and whose GPU equivalent would need a second full-size index
# array -- the one place where VRAM, not time, is the binding constraint.
_GPU = {"xp": np, "on": False, "name": ""}


def init_gpu(want):
    """Bring up CuPy if requested and available. Returns True when live."""
    if not want:
        return False
    try:
        import cupy as cp
        if cp.cuda.runtime.getDeviceCount() == 0:
            raise RuntimeError("no CUDA device visible")
        cp.zeros(1).sum()                      # force context creation now
        props = cp.cuda.runtime.getDeviceProperties(0)
        name = props["name"]
        free, total = cp.cuda.runtime.memGetInfo()
        _GPU.update(xp=cp, on=True,
                    name=name.decode() if isinstance(name, bytes) else str(name))
        print(f"[gpu] {_GPU['name']}: {free / 2**30:.1f} of "
              f"{total / 2**30:.1f} GiB free")
        return True
    except Exception as e:
        print(f"[gpu] requested but unavailable ({type(e).__name__}: {e})"
              f" -> running on CPU")
        return False


def xp():
    """The active array module: numpy, or cupy when the GPU is live."""
    return _GPU["xp"]


def on_gpu():
    return _GPU["on"]


def as_cpu(a):
    """Device array -> numpy (no-op on the CPU backend)."""
    return _GPU["xp"].asnumpy(a) if _GPU["on"] else np.asarray(a)


def as_dev(a, dtype=None):
    """numpy -> device (no-op on the CPU backend)."""
    if _GPU["on"]:
        return _GPU["xp"].asarray(a, dtype=dtype)
    return np.asarray(a, dtype=dtype) if dtype is not None else a


def gpu_free():
    """Release cached device blocks between stages so the next one starts
    with the full card available."""
    if _GPU["on"]:
        _GPU["xp"].get_default_memory_pool().free_all_blocks()


def group_bounds(m, keys):
    """Sort keys and return (order, uniq, start, count).

    The backbone of every voxel reduction here. Uses only argsort, slicing and
    comparison, so it behaves identically on numpy and cupy -- unlike
    ufunc.reduceat, which cupy does not implement at all. The sort is STABLE,
    which is what lets callers read per-group first/last values straight out of
    the sorted array instead of running a separate min/max reduction."""
    order = m.argsort(keys, kind="stable")
    ks = keys[order]
    if ks.size == 0:
        return order, ks, ks, ks
    first = m.concatenate((m.ones(1, bool), ks[1:] != ks[:-1]))
    start = m.flatnonzero(first)
    end = m.concatenate((start[1:], m.array([ks.size], dtype=start.dtype)))
    return order, ks[start], start, end - start


def group_sum(m, vals, start, count):
    """Per-group sum via cumsum differences (reduceat-free, so cupy-safe)."""
    if start.size == 0:
        return vals[:0]
    cs = m.cumsum(vals)
    total = cs[start + count - 1]
    return total - m.concatenate((m.zeros(1, cs.dtype), total[:-1]))

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
            m = xp()
            p = as_dev(pc2_xyz(msg))
            dist = m.linalg.norm(p, axis=1)
            p = p[m.isfinite(p).all(1) & (dist > lo) & (dist < hi)]
            if p.shape[0] == 0:
                continue
            Tw = tr_T[j]
            R = as_dev(Tw[:3, :3], p.dtype); tv = as_dev(Tw[:3, 3], p.dtype)
            # world points stay on the device: the dynamic filter and the
            # carver consume them there, and only merge's accumulator needs
            # them back on the host
            yield t, p @ R.T + tv, as_dev(Tw[:3, 3].copy())


class DynStats:
    """Per-voxel occupancy accumulator for dynamic-object removal. add() one
    scan's world points at a time; dynamic_keys() returns the transient voxels.

    Runs on whichever backend is active. Partial results are compacted once the
    buffer passes compact_at entries, which bounds memory on long runs -- on
    the GPU that is the difference between finishing and an out-of-memory
    abort, since a 30-minute bag can deposit hundreds of millions of keys."""

    def __init__(self, voxel, compact_at=40_000_000):
        self.inv = 1.0 / float(voxel)
        self.compact_at = int(compact_at)
        self._keys = []   # per-scan unique voxel keys
        self._time = []   # matching scan time, broadcast per key
        self._n = 0
        # compacted state, always sorted by key and chronological within a key
        self.k = None; self.hits = None; self.tmin = None; self.tmax = None

    def add(self, world_pts, t):
        m = xp()
        vox = m.floor(as_dev(world_pts) * self.inv).astype(np.int64)
        u = m.unique(pack_voxels(vox))
        if u.size:
            self._keys.append(u)
            self._time.append(m.full(u.shape, float(t)))
            self._n += int(u.size)
            if self._n >= self.compact_at:
                self._compact()

    def _compact(self):
        """Fold the buffer into (key, hits, tmin, tmax) running totals."""
        m = xp()
        if not self._keys:
            return
        keys = m.concatenate(self._keys)
        times = m.concatenate(self._time)
        hits = m.ones(keys.size, np.int64)
        self._keys = []; self._time = []; self._n = 0
        if self.k is not None:
            # prior summaries FIRST: they cover earlier scans, so a stable sort
            # keeps every group chronological and first/last stay valid
            keys = m.concatenate((self.k, keys))
            hits = m.concatenate((self.hits, hits))
            times = m.concatenate((self.tmin, times))
            tmax_in = m.concatenate((self.tmax, times[self.tmin.size:]))
        else:
            tmax_in = times
        order, uniq, start, count = group_bounds(m, keys)
        self.k = uniq
        self.hits = group_sum(m, hits[order], start, count)
        self.tmin = times[order][start]
        self.tmax = tmax_in[order][start + count - 1]

    def dynamic_keys(self, min_hits, min_span_s, carver=None,
                     min_free=3, free_ratio=0.25):
        """Sorted int64 keys of voxels judged transient (moving objects).
        Returns (keys, (n_static, n_dyn, n_carved))."""
        m = xp()
        if not self._keys and self.k is None:
            return np.empty(0, np.int64), (0, 0, 0)
        self._compact()
        uniq = self.k; hits = self.hits
        span = self.tmax - self.tmin
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
        self.keys = None
        self.counts = None
        self._buf = []
        self._buf_n = 0
        self._n_scan = 0

    def add(self, origin, world_pts):
        self._n_scan += 1
        if (self._n_scan - 1) % self.scan_stride:
            return
        # The densest kernel in stage 01: every ray samples one point per voxel
        # length, so a single scan can generate tens of millions of samples.
        # It is also pure elementwise arithmetic over a (rays x steps) grid --
        # exactly what a GPU is for. Chunking bounds the peak allocation on
        # either backend; on the GPU raise carve.chunk to fill the card.
        m = xp()
        origin = as_dev(origin)
        p = as_dev(world_pts)[::self.ray_stride]
        vec = p - origin[None, :]
        dist = m.linalg.norm(vec, axis=1)
        keep = dist > (self.margin + self.step)
        if not bool(keep.any()):
            return
        vec = vec[keep]; dist = dist[keep]
        end = m.minimum(dist - self.margin, self.max_range)
        scan_keys = []
        for a in range(0, len(vec), self.chunk):
            b = min(a + self.chunk, len(vec))
            d = dist[a:b]; e = end[a:b]
            dirs = vec[a:b] / d[:, None]
            n_steps = int(np.ceil(float(e.max()) / self.step))
            if n_steps <= 0:
                continue
            t = (m.arange(n_steps) + 0.5) * self.step           # (S,)
            valid = t[None, :] < e[:, None]                     # (n,S)
            pts = origin[None, None, :] + dirs[:, None, :] * t[None, :, None]
            vox = m.floor(pts[valid] * self.inv).astype(np.int64)
            if vox.size:
                scan_keys.append(pack_voxels(vox))
        if not scan_keys:
            return
        u = m.unique(m.concatenate(scan_keys))
        self._buf.append(u)
        self._buf_n += int(u.size)
        if self._buf_n >= self.compact_at:
            self._compact()

    def _compact(self):
        m = xp()
        if not self._buf:
            return
        k = m.concatenate(self._buf)
        self._buf = []; self._buf_n = 0
        _, uniq, start, count = group_bounds(m, k)
        c = count.astype(np.int64)
        if self.keys is not None and self.keys.size:
            comb = m.concatenate([self.keys, uniq])
            combc = m.concatenate([self.counts, c])
            order, uniq, start, count = group_bounds(m, comb)
            c = group_sum(m, combc[order], start, count).astype(np.int64)
        self.keys, self.counts = uniq, c

    def counts_for(self, query):
        """Free-scan count per (sorted or unsorted) packed voxel key."""
        m = xp()
        self._compact()
        if self.keys is None or self.keys.size == 0:
            return m.zeros(query.shape, np.int64)
        pos = m.searchsorted(self.keys, query)
        pos = m.clip(pos, 0, self.keys.size - 1)
        return m.where(self.keys[pos] == query, self.counts[pos], 0)


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
        endpoint_margin=cv.get("endpoint_margin", 0.0),
        # the GPU eats far larger ray batches than the CPU wants to allocate
        chunk=int(cv.get("chunk", 65536 if on_gpu() else 8000)),
        compact_at=int(cv.get("compact_at", 20_000_000)))


def drop_dynamic_points(pcd, dyn_keys, voxel, chunk=20_000_000):
    """Return pcd with points whose global voxel is in dyn_keys removed."""
    if len(pcd.points) == 0 or dyn_keys.size == 0:
        return pcd, 0
    m = xp()
    dyn_keys = as_dev(dyn_keys)
    pts = np.asarray(pcd.points)
    flags = np.empty(len(pts), bool)
    for a in range(0, len(pts), chunk):       # chunked: bounds VRAM on 40M+
        b = min(a + chunk, len(pts))
        vox = m.floor(as_dev(pts[a:b]) * (1.0 / float(voxel))).astype(np.int64)
        packed = pack_voxels(vox)
        # dyn_keys is sorted -> membership via searchsorted (fast, low memory)
        pos = m.clip(m.searchsorted(dyn_keys, packed), 0, dyn_keys.size - 1)
        flags[a:b] = as_cpu(dyn_keys[pos] == packed)
    keep = np.where(~flags)[0]
    out = pcd.select_by_index(keep)
    return out, int(flags.sum())


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
        buf.append(as_cpu(wp))             # accumulator + Open3D live on host
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
    gpu_free()
    carve_note = (f", {n_carved} of them span-static but carved by free-space"
                  if carver is not None else " (carving off)")
    print(f"[1b] remove-dynamic: {n_static} static / {n_dyn} dynamic voxels "
          f"(grid {voxel} m){carve_note} -> dropped {removed} pts, "
          f"kept {len(out.points)}")
    return out


def _inv_se3(T):
    """Analytic SE(3) inverse (no general matrix inversion)."""
    Ti = np.eye(4)
    Ti[:3, :3] = T[:3, :3].T
    Ti[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return Ti


class BlockIndex:
    """Spatial index for camera culling, built ONCE per stage.

    Replaces the per-frame KDTree radius search, which was the dominant cost
    of colorize: FLANN is single-threaded and a 10 m ball on a 40 M-point map
    returns millions of indices per image. Instead points are bucketed into
    coarse blocks once, and each frame keeps only the blocks whose bounding
    sphere intersects the camera FRUSTUM. That is both cheaper (a few hundred
    plane tests instead of a tree descent) and tighter: a ball wastes ~5/6 of
    its points behind and beside a 94-degree camera.

    Culling is conservative -- the exact per-point bounds test still runs
    afterwards -- so no point that should be visible is lost. It is in fact a
    strict superset of the old behaviour: the sphere culled by DISTANCE while
    the per-point test accepts by DEPTH (z < max_range), so the radius search
    was silently dropping off-axis points near the image edges that were
    legitimately in range. Measured on a synthetic room, ~10% of colourable
    points were being lost that way."""

    def __init__(self, pts, block=2.0):
        key = pack_voxels(np.floor(pts / float(block)).astype(np.int64))
        self.order = np.argsort(key, kind="stable").astype(np.int32)
        ks = key[self.order]
        _, start = np.unique(ks, return_index=True)
        self.start = start
        self.count = np.diff(np.append(start, ks.size))
        Ps = pts[self.order]
        lo = np.minimum.reduceat(Ps, start)
        hi = np.maximum.reduceat(Ps, start)
        self.cen = 0.5 * (lo + hi)
        self.rad = 0.5 * np.linalg.norm(hi - lo, axis=1)

    def candidates(self, Tcw, S, W, H, max_range):
        """Indices of points in blocks overlapping the view frustum, or None."""
        C = (Tcw[:3, :3] @ self.cen.T).T + Tcw[:3, 3]
        r = self.rad
        ok = (C[:, 2] > -r) & ((max_range - C[:, 2]) > -r)
        l = -S.cx / S.fx; rr = (W - S.cx) / S.fx
        t = -S.cy / S.fy; b = (H - S.cy) / S.fy
        for nv in ((1.0, 0.0, -l), (-1.0, 0.0, rr),
                   (0.0, 1.0, -t), (0.0, -1.0, b)):
            n = np.array(nv)
            n /= np.linalg.norm(n)
            ok &= (C @ n) > -r
        bi = np.flatnonzero(ok)
        if bi.size == 0:
            return None
        return np.concatenate([self.order[a:a + c] for a, c
                               in zip(self.start[bi], self.count[bi])])


def project_visible(index, pts, Twc, S, W, H, max_range):
    """Map points visible from camera pose Twc, at most one per pixel.

    Projects through the pinhole intrinsics, then z-buffers so an occluded
    point never receives the pixel's colour or class label. Shared by colorize
    [3] and detect_objects [6] so both stages agree on exactly which point a
    pixel belongs to. Returns (global_idx, u, v, z) of the winners, or None.

    `index` is a BlockIndex on the CPU, or None on the GPU: transforming every
    point costs a couple of milliseconds there, less than the gather that
    culling would need, so the GPU path skips the index entirely."""
    m = xp()
    Tcw = _inv_se3(Twc)
    if index is None:
        sub = pts
        idx = None
    else:
        idx = index.candidates(Tcw, S, W, H, max_range)
        if idx is None:
            return None
        sub = pts[idx]
    R = as_dev(Tcw[:3, :3], sub.dtype)
    tvec = as_dev(Tcw[:3, 3], sub.dtype)
    Xc = sub @ R.T + tvec
    z = Xc[:, 2]
    fr = z > 1e-3
    zs = m.where(fr, z, 1.0)                   # branchless: no divide by ~0
    u = m.where(fr, S.fx * Xc[:, 0] / zs + S.cx, -1.0)
    v = m.where(fr, S.fy * Xc[:, 1] / zs + S.cy, -1.0)
    inb = fr & (u >= 0) & (u < W) & (v >= 0) & (v < H) & (z < max_range)
    if not bool(inb.any()):
        return None
    sel = m.flatnonzero(inb)
    g = sel if idx is None else idx[sel]       # GPU: mask index IS the global
    zc = z[sel]
    uu = u[sel].astype(np.int64); vv = v[sel].astype(np.int64)
    # ONE sort instead of two: pack pixel id and mm-quantised depth into a
    # single key so argsort orders by pixel then by depth. The old
    # argsort(z) + np.unique(pix) pair sorted the same data twice. A single
    # radix sort is also the one primitive a GPU accelerates best.
    zq = m.minimum((zc * 1000.0).astype(np.int64), (1 << 21) - 1)
    key = ((vv * W + uu) << 21) | zq
    order = m.argsort(key, kind="stable")
    ks = key[order] >> 21
    first = m.concatenate((m.ones(1, bool), ks[1:] != ks[:-1]))
    keep = order[first]
    return g[keep], uu[keep], vv[keep], zc[keep]


def pose_gate(c):
    """Frame selector that skips images taken from (almost) the same place.

    A fixed img_stride is a blunt instrument: standing still it burns full
    cost on identical views, and moving fast it skips the only views of a
    surface. Gating on actual camera motion adapts to both. Returns a
    predicate over the camera pose; min_baseline <= 0 disables it."""
    mb = float(c.get("min_baseline", 0.0))
    my = np.cos(np.deg2rad(float(c.get("min_rotation_deg", 0.0))))
    if mb <= 0 and c.get("min_rotation_deg", 0.0) <= 0:
        return lambda Twc: True
    last = {}

    def use(Twc):
        p = Twc[:3, 3]; f = Twc[:3, 2]
        if last:
            if (np.linalg.norm(p - last["p"]) < mb
                    and float(f @ last["f"]) > my):
                return False
        last["p"] = p.copy(); last["f"] = f.copy()
        return True
    return use


def denoise(s, pcd):
    """[2] statistical outlier removal, on the GPU when Open3D was built with
    CUDA. This is a k-NN over the whole cloud -- the one heavy step here with
    no CuPy expression, but Open3D's tensor API does carry a CUDA nearest-
    neighbour backend, so try it and fall back to the legacy CPU call."""
    nb, std = s["denoise"]["nb"], s["denoise"]["std"]
    if on_gpu():
        try:
            dev = o3d.core.Device("CUDA:0")
            t = o3d.t.geometry.PointCloud.from_legacy(pcd, o3d.core.float32,
                                                      dev)
            print("[2] denoise: statistical outlier removal (GPU)")
            t, _ = t.remove_statistical_outliers(nb, std)
            out = t.to_legacy()
            del t
            gpu_free()
            return out
        except Exception as e:
            print(f"    GPU denoise unavailable ({type(e).__name__}) -> CPU")
    print("[2] denoise: light statistical outlier removal")
    pcd, _ = pcd.remove_statistical_outlier(nb, std)
    return pcd


def colorize(P, S, s, pcd):
    print("[3] colorize: best-view projection (frustum-culled)")
    tr_t, tr_T = P.traj
    W, H = s["image_width"], s["image_height"]
    c = s["colorize"]; stride = c["img_stride"]; max_range = c["max_range"]
    on_voxel = c["voxel"]; drop_gray = c["drop_gray"]

    if on_voxel > 0 and len(pcd.points) > 0:
        work = pcd.voxel_down_sample(on_voxel)
        print(f"    coloring downsampled copy: {len(work.points)} pts (from {len(pcd.points)})")
    else:
        work = pcd

    m = xp()
    N = len(work.points)
    # uint8 colours + float32 depth: at 40 M points this is 200 MB instead of
    # 1.3 GB, and the inner loop is memory-bandwidth bound. float32 xyz is
    # accurate to ~10 um at building scale, far below a pixel footprint.
    if on_gpu():
        pts = m.asarray(np.asarray(work.points), dtype=m.float32)
        index = None                          # transform-all beats gather here
        print(f"    {N} pts resident on GPU "
              f"({pts.nbytes / 2**20:.0f} MiB), no cull needed")
    else:
        pts = np.asarray(work.points, dtype=np.float64)
        index = BlockIndex(pts, block=float(c.get("cull_block", 2.0)))
        print(f"    frustum index: {len(index.start)} blocks")
    colors = m.full((N, 3), 128, np.uint8)
    best = m.full(N, np.inf, np.float32)
    use_frame = pose_gate(c)

    with AnyReader([Path(P.dataset["bag"])], default_typestore=TS) as r:
        conns = [cc for cc in r.connections if cc.topic == S.image_topic]
        n = n_used = 0
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
            if not use_frame(Twc):
                continue
            vis = project_visible(index, pts, Twc, S, W, H, max_range)
            if vis is None:
                continue
            n_used += 1
            g_keep, uu_k, vv_k, z_keep = vis
            better = z_keep.astype(np.float32) < best[g_keep]
            if not bool(better.any()):
                continue
            img = decode_img(msg)
            if img is None:
                continue
            if (img.shape[1], img.shape[0]) != (W, H):
                img = cv2.resize(img, (W, H))
            gb = g_keep[better]
            # 640x360x3 is under a megabyte -- cheaper to push the image to the
            # device than to pull the (much larger) index arrays back
            img_d = as_dev(np.ascontiguousarray(img[:, :, ::-1]))
            colors[gb] = img_d[vv_k[better], uu_k[better]]
            best[gb] = z_keep[better]
            if n % 750 == 0:
                print(f"    img {n} ({n_used} used)", flush=True)

    colors = as_cpu(colors)
    seen = np.isfinite(as_cpu(best))
    del pts, best
    gpu_free()
    print(f"    {n_used} frames used; colored {seen.sum()}/{N} "
          f"({100 * seen.mean():.1f}%)")
    work.colors = o3d.utility.Vector3dVector(colors.astype(np.float64) / 255.0)

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


# =============================================================================
# [6] OBJECT DETECTION + INVENTORY (semantic layer)
# =============================================================================

def _yolo_model(d):
    try:
        from ultralytics import YOLO
    except ImportError:
        raise SystemExit(
            "detect_objects is enabled but ultralytics is missing.\n"
            "  pip install ultralytics\n"
            "Then point detect_objects.model at a segmentation checkpoint, "
            "e.g. yolo11n-seg.pt (masks give far cleaner 3D objects than boxes).")
    m = YOLO(d.get("model", "yolo11n-seg.pt"))
    return m, m.names


def _class_filter(d):
    """allowlist AND denylist by class NAME (empty allowlist = everything)."""
    allow = set(d.get("classes") or [])
    deny = set(d.get("exclude", ["person"]) or [])
    def ok(name):
        return (not allow or name in allow) and name not in deny
    return ok


def yolo_label_image(model, names, img, d, W, H, ok):
    """One frame -> (int16 image of per-instance ids, [(class_id, conf), ...]).

    Segmentation masks are used when the weights provide them and are eroded a
    couple of pixels, because the outermost mask ring straddles the silhouette
    and would label whatever is behind the object. Box-only weights fall back
    to the box shrunk by bbox_shrink per side, for the same reason."""
    res = model.predict(img, conf=d.get("conf", 0.35), iou=d.get("iou", 0.5),
                        imgsz=d.get("imgsz", 640), device=d.get("device"),
                        verbose=False)[0]
    lab = np.full((H, W), -1, np.int16)
    insts = []
    if res.boxes is None or len(res.boxes) == 0:
        return lab, insts
    cls = res.boxes.cls.cpu().numpy().astype(int)
    conf = res.boxes.conf.cpu().numpy()
    xyxy = res.boxes.xyxy.cpu().numpy()
    masks = None
    if getattr(res, "masks", None) is not None:
        masks = res.masks.data.cpu().numpy()
    shrink = float(d.get("bbox_shrink", 0.12))
    erode = int(d.get("mask_erode", 2))
    for i in range(len(cls)):
        nm = names[int(cls[i])]
        if not ok(nm):
            continue
        k = len(insts)
        if masks is not None and i < len(masks):
            m = cv2.resize(masks[i].astype(np.uint8), (W, H),
                           interpolation=cv2.INTER_NEAREST)
            if erode > 0:
                m = cv2.erode(m, np.ones((3, 3), np.uint8), iterations=erode)
            m = m > 0
            if not m.any():
                continue
            lab[m] = k
        else:
            x0, y0, x1, y1 = xyxy[i]
            dx = shrink * (x1 - x0); dy = shrink * (y1 - y0)
            x0 = int(max(0, x0 + dx)); x1 = int(min(W, x1 - dx))
            y0 = int(max(0, y0 + dy)); y1 = int(min(H, y1 - dy))
            if x1 <= x0 or y1 <= y0:
                continue
            lab[y0:y1, x0:x1] = k
        insts.append((int(cls[i]), float(conf[i])))
    return lab, insts


def fit_structure_planes(work_pts, obj_mask, voxel=0.08, dist=0.06,
                         min_area=4.0, max_planes=40):
    """Large planes of the map (walls, floor, ceiling) used to veto object
    votes that actually landed on structure.

    Multi-view voting measures CONSISTENCY, not correctness: a detector that
    fires "tv" on the same wall patch from every viewpoint produces a perfect
    vote ratio, so no amount of fusion can reject it. Geometry can. Fitting is
    done on a coarse copy for speed, then each candidate plane is re-evaluated
    analytically on the full voting cloud.

    A plane whose inliers are mostly voted-as-object is SKIPPED, so a large
    planar object (a long counter, a table top) can never veto itself."""
    pc = _pc(work_pts).voxel_down_sample(voxel)
    rest = pc
    planes = []
    cell = voxel * voxel
    for _ in range(max_planes):
        if len(rest.points) < 200:
            break
        model, inl = rest.segment_plane(dist, 3, 500)
        if len(inl) * cell < min_area:      # largest first -> rest are smaller
            break
        n = np.array(model[:3], float)
        L = np.linalg.norm(n)
        if L < 1e-9:
            break
        n /= L
        d0 = float(model[3] / L)
        m = np.abs(work_pts @ n + d0) < dist
        if m.any() and obj_mask[m].mean() < 0.5:
            planes.append((n, d0))
        rest = rest.select_by_index(inl, invert=True)
    return planes


def min_plane_distance(pts, planes, chunk=2_000_000):
    """Distance from each point to the NEAREST structural plane (inf if none).
    This doubles as the protrusion measure: a wall point sits at ~0, a real
    object stands off by its depth."""
    if not planes:
        return np.full(len(pts), np.inf)
    Nn = np.array([p[0] for p in planes]).T          # (3, K)
    dd = np.array([p[1] for p in planes])            # (K,)
    out = np.empty(len(pts))
    for a in range(0, len(pts), chunk):
        b = min(a + chunk, len(pts))
        out[a:b] = np.abs(pts[a:b] @ Nn + dd).min(axis=1)
    return out


def _footprint(Q):
    """PCA of the XY footprint -> (yaw_deg, [length, width]).

    Deliberately 2D: indoor objects stand upright, and a full 3D oriented box
    degenerates (or throws) on the flat, one-sided clusters LiDAR produces."""
    xy = Q[:, :2]
    c = xy.mean(axis=0)
    d = xy - c
    if len(xy) >= 3:
        w, V = np.linalg.eigh(d.T @ d / len(xy))
        e = V[:, -1]
    else:
        e = np.array([1.0, 0.0])
    R = np.array([[e[0], e[1]], [-e[1], e[0]]])
    loc = d @ R.T
    # a principal axis has no sign, so fold yaw into [-90, 90): 35 and -145
    # describe the same box and only one of them reads sensibly
    yaw = (np.degrees(np.arctan2(e[1], e[0])) + 90.0) % 180.0 - 90.0
    return float(yaw), loc.max(0) - loc.min(0)


def _instance_stats(Q, C, label, class_id, votes, seen, floor_z, det_conf,
                    protr=None):
    yaw, ext = _footprint(Q)
    lo = Q.min(axis=0); hi = Q.max(axis=0)
    cen = Q.mean(axis=0)
    agree = float(np.median(votes / np.maximum(seen, 1)))
    st = {
        "label": label,
        "class_id": int(class_id),
        "det_conf": round(float(det_conf), 3),
        "view_agreement": round(agree, 3),
        "n_points": int(len(Q)),
        "centroid": [round(float(x), 3) for x in cen],
        "aabb_min": [round(float(x), 3) for x in lo],
        "aabb_max": [round(float(x), 3) for x in hi],
        "size": [round(float(x), 3) for x in (hi - lo)],
        "footprint": {
            "yaw_deg": round(yaw, 1),
            "length": round(float(ext[0]), 3),
            "width": round(float(ext[1]), 3),
            "height": round(float(hi[2] - lo[2]), 3),
        },
        "base_z": round(float(lo[2]), 3),
        "height_above_floor": round(float(lo[2] - floor_z), 3),
    }
    if protr is not None and np.isfinite(protr).any():
        # how far this object stands off the nearest wall/floor/ceiling plane;
        # near zero means the "object" IS structure
        st["protrusion"] = round(float(np.median(protr)), 3)
    if C is not None:
        st["mean_rgb"] = [round(float(x), 3) for x in C.mean(axis=0)]
    return st


def detect_objects(P, S, s, pcd):
    """[6] fuse YOLO detections across views onto the map, cluster into
    instances. Returns a dict of layers + the inventory, or None."""
    d = s["detect_objects"]
    tr_t, tr_T = P.traj
    W, H = s["image_width"], s["image_height"]
    stride = max(1, int(d.get("img_stride", 5)))
    max_range = float(d.get("max_range", 8.0))
    dv = float(d.get("voxel", 0.05))
    band = float(d.get("depth_band", 1.0))
    min_px = int(d.get("min_pixels", 60))

    work = pcd.voxel_down_sample(dv) if dv > 0 else pcd
    pts = np.asarray(work.points)
    N = len(pts)
    if N == 0:
        print("[6] detect: empty cloud, skipping")
        return None
    print(f"[6] detect: voting on {N} pts at {dv} m "
          f"(from {len(pcd.points)}), every {stride}th image")
    m = xp()
    if on_gpu():
        ptsd = m.asarray(pts, dtype=m.float32)
        index = None
    else:
        ptsd = pts
        index = BlockIndex(pts, block=float(d.get("cull_block", 2.0)))
    model, names = _yolo_model(d)
    ok = _class_filter(d)
    use_frame = pose_gate(d)

    n_seen = m.zeros(N, np.int32)           # frames the point was visible in
    votes = {}                              # class_id -> per-point vote count
    conf_acc = {}                           # class_id -> [sum_conf, n_det]
    n_frames = n_det = 0

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
            if not use_frame(Twc):
                continue
            # cull BEFORE decoding/inferring: a frame with nothing in view
            # must not pay for a YOLO pass
            vis = project_visible(index, ptsd, Twc, S, W, H, max_range)
            if vis is None:
                continue
            img = decode_img(msg)
            if img is None:
                continue
            if (img.shape[1], img.shape[0]) != (W, H):
                img = cv2.resize(img, (W, H))
            g, uu, vv, z = vis
            n_frames += 1
            n_seen[m.unique(g)] += 1        # unique: one point may win >1 pixel
            lab_img, insts = yolo_label_image(model, names, img, d, W, H, ok)
            if not insts:
                continue
            lab = as_dev(lab_img)[vv, uu]
            for k, (c, cf) in enumerate(insts):
                sel = lab == k
                ns = int(sel.sum())
                if ns < min_px:
                    continue
                zk = z[sel]
                # depth band around the detection's median depth: a mask edge
                # that slips past the silhouette lands on the far wall, which
                # is metres behind -> rejected here rather than voted on
                inb = sel.copy()
                inb[sel] = m.abs(zk - m.median(zk)) <= band
                gi = m.unique(g[inb])
                if gi.size == 0:
                    continue
                if c not in votes:
                    votes[c] = m.zeros(N, np.int32)
                    conf_acc[c] = [0.0, 0]
                votes[c][gi] += 1
                conf_acc[c][0] += cf; conf_acc[c][1] += 1
                n_det += 1
            if n_frames % 200 == 0:
                print(f"    {n_frames} frames, {n_det} detections, "
                      f"{len(votes)} classes", flush=True)

    if not votes:
        print("[6] detect: no detections survived - check model/classes/conf")
        return None

    # voting is done; everything downstream (plane RANSAC, DBSCAN, instance
    # stats) is Open3D/CPU, so bring the accumulators back and free the card
    n_seen = as_cpu(n_seen)
    votes = {c: as_cpu(v) for c, v in votes.items()}
    del ptsd
    gpu_free()

    best_v = np.zeros(N, np.int32); best_c = np.full(N, -1, np.int32)
    for c, v in votes.items():
        gt = v > best_v
        best_v[gt] = v[gt]; best_c[gt] = c
    min_votes = int(d.get("min_votes", 3))
    min_ratio = float(d.get("min_ratio", 0.35))
    confident = (best_c >= 0) & (best_v >= min_votes) & \
                (best_v >= min_ratio * np.maximum(n_seen, 1))
    print(f"    {n_frames} frames, {n_det} detections -> "
          f"{int(confident.sum())} points pass multi-view agreement "
          f"(>= {min_votes} votes and >= {min_ratio:.0%} of views)")

    # ---- structural veto: the one failure voting cannot catch ---------------
    # A detector that fires on the same wall patch from every view (a poster
    # read as "tv", a dark rectangle, a plain hallucination) scores a PERFECT
    # vote ratio -- consistency is exactly what it has. It is rejected here on
    # geometry instead: points lying flush on a large structural plane are not
    # objects. This also trims the wall halo around genuinely wall-mounted
    # objects, where the depth-band gate is useless because the wall behind
    # sits at the same depth as the object.
    vt = d.get("structure_veto", {})
    protr = np.full(N, np.inf)
    min_protrusion = float(vt.get("min_protrusion", 0.04))
    if vt.get("enable", True):
        planes = fit_structure_planes(
            pts, confident,
            voxel=float(vt.get("voxel", 0.08)),
            dist=float(vt.get("plane_dist", 0.06)),
            min_area=float(vt.get("min_area", 4.0)),
            max_planes=int(vt.get("max_planes", 40)))
        protr = min_plane_distance(pts, planes)
        flush = protr < float(vt.get("flush_tol", 0.05))
        n_kill = int((confident & flush).sum())
        confident &= ~flush
        print(f"    structural veto: {len(planes)} large planes -> "
              f"{n_kill} voted pts dropped as flush-on-structure, "
              f"{int(confident.sum())} object pts remain")

    cl = d.get("cluster", {})
    eps = float(cl.get("eps", 0.12))
    min_points = int(cl.get("min_points", 60))
    keep_pts = int(cl.get("min_pts_keep", 120))
    cols = np.asarray(work.colors) if work.has_colors() else None
    floor_z = float(np.percentile(pts[:, 2], 1.0))
    inst_id = np.full(N, -1, np.int64)
    instances = []
    n_flat = 0
    for c in sorted(votes.keys()):
        sel = np.flatnonzero(confident & (best_c == c))
        if len(sel) < keep_pts:
            continue
        sub = work.select_by_index(sel)
        lb = np.asarray(sub.cluster_dbscan(eps=eps, min_points=min_points))
        det_conf = conf_acc[c][0] / max(conf_acc[c][1], 1)
        for L in range(int(lb.max()) + 1):
            m = lb == L
            if int(m.sum()) < keep_pts:
                continue
            gidx = sel[m]
            pr = protr[gidx]
            # second line of defence: a cluster that survived per-point vetoing
            # but still sits flat against structure (a whole wall segment read
            # as one object) is rejected as a body, not point by point
            if np.isfinite(pr).any() and float(np.median(pr)) < min_protrusion:
                n_flat += 1
                continue
            st = _instance_stats(
                pts[gidx], None if cols is None else cols[gidx],
                names[c], c, best_v[gidx], n_seen[gidx], floor_z, det_conf,
                protr=pr)
            st["id"] = len(instances) + 1
            inst_id[gidx] = len(instances)
            instances.append(st)

    if n_flat:
        print(f"    rejected {n_flat} wall-like clusters "
              f"(median protrusion < {min_protrusion} m)")
    print(f"[6] detect: {len(instances)} object instances "
          f"({sum(i['n_points'] for i in instances)} voting pts) in "
          f"{len(set(i['label'] for i in instances))} classes")
    return {"instances": instances, "inst_id": inst_id, "work": work,
            "floor_z": floor_z, "model": str(d.get("model", "")),
            "n_frames": n_frames}


def instance_palette(n):
    """Visually distinct per-instance colours (golden-ratio hue walk)."""
    if n == 0:
        return np.zeros((0, 3))
    h = ((np.arange(n) * 0.6180339887) % 1.0) * 179.0
    hsv = np.stack([h, np.full(n, 200.0), np.full(n, 245.0)], 1)
    hsv = hsv.astype(np.uint8)[None, :, :]
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0].astype(np.float64) / 255.0


def _yaml_scalar(v):
    if isinstance(v, str):
        return v if v.replace("_", "").replace("-", "").isalnum() else f'"{v}"'
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_yaml_scalar(x) for x in v) + "]"
    return str(v)


def _yaml_emit(o, f, ind=0):
    """Minimal YAML writer so stage 01 needs no pyyaml (used when it is
    absent). Only handles the shapes the inventory actually produces."""
    pad = "  " * ind
    if isinstance(o, dict):
        for k, v in o.items():
            if isinstance(v, dict) and v:
                f.write(f"{pad}{k}:\n"); _yaml_emit(v, f, ind + 1)
            elif isinstance(v, list) and v and isinstance(v[0], dict):
                f.write(f"{pad}{k}:\n"); _yaml_emit(v, f, ind + 1)
            else:
                f.write(f"{pad}{k}: {_yaml_scalar(v)}\n")
    elif isinstance(o, list):
        for v in o:
            if isinstance(v, dict):
                items = list(v.items())
                k0, v0 = items[0]
                f.write(f"{pad}- {k0}: {_yaml_scalar(v0)}\n")
                _yaml_emit(dict(items[1:]), f, ind + 1)
            else:
                f.write(f"{pad}- {_yaml_scalar(v)}\n")


def save_object_layers(P, s, res, pcd, shift):
    """Split the map into background / object layers and write the inventory.

    `shift` is the anchor translation already applied to the final cloud, so
    the inventory coordinates match whatever map_final.pcd is in."""
    d = s["detect_objects"]
    inst_id = res["inst_id"]; work = res["work"]; instances = res["instances"]
    dv = float(d.get("voxel", 0.05))

    if work is pcd:
        full_id = inst_id
    else:
        from scipy.spatial import cKDTree
        # the voting cloud lives in the pre-anchor frame; undo the shift on the
        # query points so the two clouds are compared in the same frame
        full = np.asarray(pcd.points) - np.asarray(shift, float)
        tree = cKDTree(np.asarray(work.points))
        full_id = np.empty(len(full), np.int64)
        chunk = 5_000_000
        for a in range(0, len(full), chunk):
            b = min(a + chunk, len(full))
            dd, nn = tree.query(full[a:b], workers=-1)
            fid = inst_id[nn].copy()
            fid[dd > 2.0 * dv] = -1   # never drag a label onto far-away points
            full_id[a:b] = fid
        print(f"    propagated instance ids to {len(full)} full-res pts")

    obj_idx = np.flatnonzero(full_id >= 0)
    bg_idx = np.flatnonzero(full_id < 0)
    objects = pcd.select_by_index(obj_idx)
    background = pcd.select_by_index(bg_idx)
    print(f"[6] layers: {len(bg_idx)} background pts / "
          f"{len(obj_idx)} object pts")

    if d.get("save_layers", True):
        save(P, background, "background.pcd")
        save(P, objects, "objects.pcd")
        pal = instance_palette(len(instances))
        by_inst = o3d.geometry.PointCloud(objects)
        by_inst.colors = o3d.utility.Vector3dVector(pal[full_id[obj_idx]])
        save(P, by_inst, "objects_by_instance.pcd")

    if d.get("save_per_object", False) and len(instances):
        odir = P.outp("objects")
        os.makedirs(odir, exist_ok=True)
        for i, st in enumerate(instances):
            sel = np.flatnonzero(full_id == i)
            if sel.size == 0:
                continue
            o3d.io.write_point_cloud(
                os.path.join(odir, f"{st['id']:04d}_{st['label']}.pcd"),
                pcd.select_by_index(sel))
        print(f"    wrote {len(instances)} per-object clouds -> {odir}/")

    # inventory coordinates follow the final cloud through the anchor shift
    if np.any(shift):
        sh = np.asarray(shift, float)
        for st in instances:
            for key in ("centroid", "aabb_min", "aabb_max"):
                st[key] = [round(float(a + b), 3) for a, b in zip(st[key], sh)]
            st["base_z"] = round(float(st["base_z"] + sh[2]), 3)

    counts = {}
    for st in instances:
        counts[st["label"]] = counts.get(st["label"], 0) + 1
    doc = {
        "map": {
            "source": os.path.basename(P.outp(s["output"])),
            "frame": "map_anchored" if np.any(shift) else "map",
            "anchor_shift": [round(float(x), 3) for x in np.asarray(shift)],
            "floor_z": round(float(res["floor_z"]), 3),
            "model": res["model"],
            "frames_used": int(res["n_frames"]),
            "background_points": int(len(bg_idx)),
            "object_points": int(len(obj_idx)),
        },
        "totals": dict(sorted(counts.items(), key=lambda kv: -kv[1])),
        "objects": [dict([("id", st["id"])] +
                         [(k, v) for k, v in st.items() if k != "id"])
                    for st in instances],
    }
    path = P.outp(d.get("inventory", "objects_inventory.yaml"))
    with open(path, "w") as f:
        try:
            import yaml
            yaml.safe_dump(doc, f, sort_keys=False, default_flow_style=None)
        except ImportError:
            _yaml_emit(doc, f)
    print(f"    saved {path}  ({len(instances)} objects)")
    for k, v in doc["totals"].items():
        print(f"      {v:4d}  {k}")
    return background, objects


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
    init_gpu(s.get("gpu", True))
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
            pcd = denoise(s, pcd)
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
            pcd = denoise(s, pcd)
            save(P, pcd, "denoised.pcd")

    if s["colorize"]["enable"]:
        pcd = colorize(P, S, s, pcd); save(P, pcd, "colored.pcd")

    # detection runs BEFORE flatten (which would distort object geometry) and
    # its layers are written after the anchor shift so every output shares one
    # coordinate frame with map_final.pcd
    det = None
    if s.get("detect_objects", {}).get("enable", False):
        det = detect_objects(P, S, s, pcd)

    if s["flatten"]["enable"]:
        pcd = flatten(s, pcd); save(P, pcd, "flattened.pcd")

    shift = np.zeros(3)
    if s["anchor_camera_start"]:
        print("[5] anchor: origin at camera start (z-up)")
        tr_t, tr_T = P.traj
        cam0 = (tr_T[0] @ S.T_lidar_camera)[:3, 3]
        shift = -cam0
        pcd.translate(shift)
        print(f"    shift {shift.round(3)}  (NOTE: stage 03 must know this via "
              f"01_build_map.anchor_camera_start)")
        save(P, pcd, "anchored.pcd")

    final = P.outp(s["output"])
    o3d.io.write_point_cloud(final, pcd)
    print(f"DONE -> {final}  ({len(pcd.points)} points)")

    if det is not None:
        # layers are cut from the same cloud that was just written, so the
        # background layer is exactly map_final minus the inventoried objects
        save_object_layers(P, s, det, pcd, shift)


def load_traj_cached(P):
    from pipeline_common import load_traj
    return load_traj(P.dataset["traj"])


if __name__ == "__main__":
    main()
