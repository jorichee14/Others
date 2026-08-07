#!/usr/bin/env python3
"""
STAGE 01 - build the map cloud from LiDAR scans placed by GLIM poses.

  bag (/ouster/points) + traj_lidar.txt  ->  merge -> [remove dynamic]
  -> denoise -> colorize -> [flatten] -> [anchor to camera start]
  -> map_final.pcd

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


def preload_cuda_libs():
    """Make the pip-installed CUDA libraries dlopen-able by soname.

    CuPy compiles its kernels at runtime and dlopen()s libnvrtc by plain
    soname. The pip CUDA wheels (pulled in by torch, or installed directly)
    put those .so files under site-packages/nvidia/*/lib, which is NOT on the
    loader path -- so a venv that physically contains every required library
    still fails with "libnvrtc.so.12: cannot open shared object file".

    Loading them by absolute path with RTLD_GLOBAL registers their SONAMEs in
    this process, and glibc then satisfies CuPy's later dlopen from the
    already-loaded set. Returns the number of libraries pinned."""
    import ctypes
    import glob
    import site
    want = ("nvrtc", "nvJitLink", "cudart", "cublas", "cufft", "curand",
            "cusolver", "cusparse")
    roots = list(site.getsitepackages())
    try:
        roots.append(site.getusersitepackages())
    except Exception:
        pass
    roots.append(os.path.dirname(os.path.dirname(np.__file__)))
    n = 0
    seen = set()
    for r in dict.fromkeys(roots):
        for so in glob.glob(os.path.join(r, "nvidia", "*", "lib", "lib*.so*")):
            base = os.path.basename(so)
            if base in seen or not any(w.lower() in base.lower() for w in want):
                continue
            try:
                ctypes.CDLL(so, mode=ctypes.RTLD_GLOBAL)
                seen.add(base)
                n += 1
            except OSError:
                pass
    return n


def init_gpu(want):
    """Bring up CuPy if requested and available. Returns True when live."""
    if not want:
        return False
    try:
        import cupy as cp
        if cp.cuda.runtime.getDeviceCount() == 0:
            raise RuntimeError("no CUDA device visible")
        try:
            _gpu_smoke_test(cp)
        except Exception:
            # most likely the NVRTC-not-on-the-loader-path case; pin the pip
            # CUDA libs and try once more before giving up
            n = preload_cuda_libs()
            if n:
                print(f"[gpu] pinned {n} CUDA libraries from the pip wheels")
            _gpu_smoke_test(cp)
        props = cp.cuda.runtime.getDeviceProperties(0)
        name = props["name"]
        free, total = cp.cuda.runtime.memGetInfo()
        _GPU.update(xp=cp, on=True,
                    name=name.decode() if isinstance(name, bytes) else str(name))
        print(f"[gpu] {_GPU['name']}: {free / 2**30:.1f} of "
              f"{total / 2**30:.1f} GiB free")
        return True
    except Exception as e:
        msg = str(e)
        print(f"[gpu] requested but unavailable ({type(e).__name__}: {msg})"
              f" -> running on CPU")
        if "nvrtc" in msg.lower():
            print("[gpu] CuPy JIT-compiles kernels and needs NVRTC. Install "
                  "the matching runtime wheel:\n"
                  "        pip install nvidia-cuda-nvrtc-cu12   # or -cu11")
        return False


def _gpu_smoke_test(cp):
    """Actually compile and run a kernel.

    Deliberately exercises an elementwise uint8 fill and a reduction, because
    those are the first things colorize does. A weaker probe (zeros().sum())
    can be served entirely from CuPy's on-disk kernel cache and prebuilt
    reductions, reporting a healthy GPU that then dies on the first real
    compile -- twenty minutes into a run, after the resume has already loaded
    a 120 M-point cloud."""
    a = cp.full((4, 3), 7, cp.uint8)
    b = cp.arange(12, dtype=cp.float32).reshape(4, 3)
    c = (a.astype(cp.float32) * b).sum() + cp.argsort(b[:, 0]).sum()
    float(c)                                   # forces a device sync


def xp():
    """The active array module: numpy, or cupy when the GPU is live."""
    return _GPU["xp"]


def mod_of(a):
    """The array module that OWNS `a`.

    Stage code must follow the data, not the global backend: colorize falls
    back to the host when a cloud will not fit in VRAM, and a shared helper
    that assumed xp() would then try to matmul a numpy array against a cupy
    one."""
    return _GPU["xp"] if _GPU["on"] and not isinstance(a, np.ndarray) else np


def gpu_free_bytes():
    if not _GPU["on"]:
        return 0
    try:
        return int(_GPU["xp"].cuda.runtime.memGetInfo()[0])
    except Exception:
        return 0


def gpu_fits(nbytes, headroom=2.0):
    """True when `nbytes` can be held resident with room for the per-frame
    temporaries (which need roughly as much again during projection)."""
    return _GPU["on"] and nbytes * headroom < gpu_free_bytes()


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
    """Per-group sum via cumsum differences (reduceat-free, so cupy-safe).
    Handles (N,) and (N,C) values -- the latter sums each column, which is how
    the merge accumulator carries xyz totals."""
    if start.size == 0:
        return vals[:0]
    cs = m.cumsum(vals, axis=0)              # axis matters: 2-D must not flatten
    total = cs[start + count - 1]
    pad = m.zeros((1,) + vals.shape[1:], cs.dtype)
    return total - m.concatenate((pad, total[:-1]))

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


def unpack_voxels(keys):
    """Inverse of pack_voxels: (N,) int64 keys -> (N,3) int64 indices."""
    m = np if isinstance(keys, np.ndarray) else xp()
    mask = (1 << _VOX_BITS) - 1
    return m.stack([(keys >> (2 * _VOX_BITS)) & mask,
                    (keys >> _VOX_BITS) & mask,
                    keys & mask], axis=1) - _VOX_OFF


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


def reduce_module(nbytes):
    """Pick numpy or cupy for one big reduction, by whether it comfortably
    fits in free VRAM (x3 covers the sort workspace and the output)."""
    if not on_gpu():
        return np
    try:
        free, _ = xp().cuda.runtime.memGetInfo()
        return xp() if nbytes * 3 < free else np
    except Exception:
        return np


class VoxelAccumulator:
    """Memory-bounded replacement for merge's grow-then-downsample buffer.

    The previous accumulator appended every scan and periodically ran Open3D's
    voxel_down_sample over the whole pile. That keeps the old array, the
    concatenated array and Open3D's internal copy alive at once -- roughly 4x
    the cloud -- while the cloud itself grows without bound at a fine voxel. A
    180 M-point run therefore peaked near 17 GB and was killed by the OS.

    This keeps ONE entry per occupied voxel, so memory tracks the mapped
    surface area rather than the number of scans, and no step ever needs a
    second full-size copy of the raw points.

    centroid=True reproduces voxel_down_sample's result (mean of the points
    falling in each voxel) at 36 bytes per voxel. centroid=False stores the
    voxel centre instead: 8 bytes per voxel, 4.5x leaner, displacing each point
    by at most half a voxel -- 5 mm on a 1 cm grid, far below the sensor's own
    range noise."""

    def __init__(self, voxel, centroid=True, flush_pts=20_000_000):
        self.voxel = float(voxel)
        self.inv = 1.0 / self.voxel
        self.centroid = bool(centroid)
        self.flush_pts = int(flush_pts)
        self.keys = None; self.sums = None; self.cnts = None
        self._buf = []; self._n = 0

    def add(self, pts):
        self._buf.append(np.asarray(as_cpu(pts), dtype=np.float32))
        self._n += len(self._buf[-1])
        if self._n >= self.flush_pts:
            self.flush()

    def flush(self):
        if not self._buf:
            return
        pts = np.concatenate(self._buf); self._buf = []; self._n = 0
        n_old = 0 if self.keys is None else self.keys.size
        m = reduce_module((n_old + len(pts)) * (32 if self.centroid else 8))
        k = pack_voxels(m.floor(m.asarray(pts, dtype=m.float64)
                                * self.inv).astype(np.int64))
        if self.keys is not None:
            k = m.concatenate([m.asarray(self.keys), k])
        if not self.centroid:
            self.keys = as_cpu_of(m, m.unique(k))
            del k
            return
        vals = m.asarray(pts, dtype=m.float64)
        ones = m.ones(len(pts), np.int64)
        if self.sums is not None:
            vals = m.concatenate([m.asarray(self.sums), vals])
            ones = m.concatenate([m.asarray(self.cnts), ones])
        order, uniq, start, count = group_bounds(m, k)
        del k
        self.keys = as_cpu_of(m, uniq)
        self.sums = as_cpu_of(m, group_sum(m, vals[order], start, count))
        self.cnts = as_cpu_of(m, group_sum(m, ones[order], start, count))

    def points(self):
        """The merged cloud, one point per occupied voxel."""
        self.flush()
        if self.keys is None or self.keys.size == 0:
            return np.empty((0, 3))
        if self.centroid:
            return self.sums / self.cnts[:, None]
        return (unpack_voxels(self.keys) + 0.5) * self.voxel

    def nbytes(self):
        return sum(a.nbytes for a in (self.keys, self.sums, self.cnts)
                   if a is not None)


def as_cpu_of(m, a):
    """as_cpu() for a reduction that may have run on a different module."""
    return m.asnumpy(a) if m is not np else np.asarray(a)


def merge(P, S, s, dyn=None, carver=None):
    extras = [x for x, on in (("dynamic-voxel stats", dyn is not None),
                              ("free-space carving", carver is not None)) if on]
    print("[1] merge: LiDAR scans -> world cloud"
          + (f" (+ {', '.join(extras)})" if extras else ""))
    scan_voxel = s["scan_voxel"]; final_voxel = s["final_voxel"]
    # the accumulator grid is the finer of the two: anything coarser would
    # throw away detail the final downsample is still asking for
    grid = min(v for v in (scan_voxel, final_voxel) if v > 0) \
        if (scan_voxel > 0 or final_voxel > 0) else 0.0
    if grid <= 0:
        raise SystemExit("merge needs scan_voxel or final_voxel > 0; an "
                         "un-voxelised merge cannot be memory-bounded")
    acc = VoxelAccumulator(grid,
                           centroid=bool(s.get("merge_centroid", True)),
                           flush_pts=int(s.get("merge_flush_pts", 20_000_000)))
    n = 0
    report = max(1, int(s["flush_every"]))
    for t, wp, origin in iter_world_scans(P, S, s):
        if dyn is not None:
            dyn.add(wp, t)                 # collect stats before downsampling
        if carver is not None:
            carver.add(origin, wp)
        acc.add(wp)
        n += 1
        if n % report == 0:
            acc.flush()
            print(f"    {n} scans, {0 if acc.keys is None else acc.keys.size} "
                  f"voxels ({acc.nbytes() / 2**20:.0f} MiB)", flush=True)
    m = _pc(acc.points())
    del acc
    gpu_free()
    if final_voxel > 0 and final_voxel > grid:
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


def projection_chunk(m, n_pts, bytes_per_point=44):
    """Points per projection slice, sized to what is actually free on the card.

    Each point in flight costs roughly Xc (12 B) + z/u/v (12 B) + the boolean
    masks and the gathered outputs, so ~44 B. Half the free pool is used, which
    leaves room for the sort that follows."""
    if m is np:
        return n_pts
    try:
        free, _ = m.cuda.runtime.memGetInfo()
    except Exception:
        free = 1 << 30
    return int(max(1_000_000, min(n_pts, free * 0.5 / bytes_per_point)))


def _project_chunk(m, sub, R, tvec, S, W, H, max_range):
    """Transform + in-bounds test for one slice. Returns local indices."""
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
    return sel, u[sel], v[sel], z[sel]


def project_visible(index, pts, Twc, S, W, H, max_range):
    """Map points visible from camera pose Twc, at most one per pixel.

    Projects through the pinhole intrinsics, then z-buffers so an occluded
    point never receives the pixel's colour. Used by colorize
    [3]. Returns (global_idx, u, v, z) of the winning points, or None.

    `index` is a BlockIndex on the CPU, or None on the GPU: transforming every
    point costs a couple of milliseconds there, less than the gather that
    culling would need, so the GPU path skips the index entirely."""
    m = mod_of(pts)                            # follow the data, not xp()
    Tcw = _inv_se3(Twc)
    R = m.asarray(Tcw[:3, :3], dtype=pts.dtype)
    tvec = m.asarray(Tcw[:3, 3], dtype=pts.dtype)
    if index is None:
        # GPU: no cull, but the transform is CHUNKED. sub @ R.T allocates a
        # second full (N,3) array, so a 124 M-point cloud needs 1.5 GB of
        # temporaries per frame on top of the 2.3 GB already resident -- which
        # is exactly how this used to die with OutOfMemoryError. Slicing keeps
        # the working set bounded while the resident cloud stays on the card.
        parts = []
        step = projection_chunk(m, len(pts))
        for a in range(0, len(pts), step):
            r = _project_chunk(m, pts[a:a + step], R, tvec, S, W, H, max_range)
            if r is not None:
                sel, uu_, vv_, z_ = r
                parts.append((sel + a, uu_, vv_, z_))
        if not parts:
            return None
        g = m.concatenate([p[0] for p in parts])
        u = m.concatenate([p[1] for p in parts])
        v = m.concatenate([p[2] for p in parts])
        zc = m.concatenate([p[3] for p in parts])
        del parts
    else:
        idx = index.candidates(Tcw, S, W, H, max_range)
        if idx is None:
            return None
        r = _project_chunk(m, pts[idx], R, tvec, S, W, H, max_range)
        if r is None:
            return None
        sel, u, v, zc = r
        g = idx[sel]
    uu = u.astype(np.int64); vv = v.astype(np.int64)
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

    N = len(work.points)
    # uint8 colours + float32 depth: at 40 M points this is 200 MB instead of
    # 1.3 GB, and the inner loop is memory-bandwidth bound. float32 xyz is
    # accurate to ~10 um at building scale, far below a pixel footprint.
    resident = N * (12 + 3 + 4)               # xyz f32 + rgb u8 + depth f32
    use_dev = on_gpu() and gpu_fits(resident)
    m = xp() if use_dev else np
    if use_dev:
        pts = m.asarray(np.asarray(work.points), dtype=m.float32)
        index = None                          # cull is unnecessary on device
        print(f"    {N} pts resident on GPU "
              f"({pts.nbytes / 2**20:.0f} MiB), no cull needed")
    else:
        if on_gpu():
            free = gpu_free_bytes()
            print(f"    cloud needs ~{resident / 2**30:.1f} GiB resident but "
                  f"only {free / 2**30:.1f} GiB is free on the card -> "
                  f"colorizing on the CPU.\n"
                  f"    set colorize.voxel (e.g. 0.04) to colour a "
                  f"downsampled copy and keep this on the GPU")
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
            img_d = m.asarray(np.ascontiguousarray(img[:, :, ::-1]))
            colors[gb] = img_d[vv_k[better], uu_k[better]]
            best[gb] = z_keep[better]
            if n % 750 == 0:
                print(f"    img {n} ({n_used} used)", flush=True)

    colors = np.asarray(colors if m is np else m.asnumpy(colors))
    seen = np.isfinite(np.asarray(best if m is np else m.asnumpy(best)))
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

    if s["flatten"]["enable"]:
        pcd = flatten(s, pcd); save(P, pcd, "flattened.pcd")

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


def load_traj_cached(P):
    from pipeline_common import load_traj
    return load_traj(P.dataset["traj"])


if __name__ == "__main__":
    main()
