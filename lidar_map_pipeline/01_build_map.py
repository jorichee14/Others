#!/usr/bin/env python3
"""
STAGE 01 - build the map cloud from LiDAR scans placed by GLIM poses.

  bag (/ouster/points) + traj_lidar.txt  ->  merge -> [remove dynamic]
  -> denoise -> colorize -> [detect] -> [synthesize] -> [flatten]
  -> [anchor to camera start] -> map_final.pcd

Everything (paths, voxels, toggles, sensor calib, topics) comes from
pipeline_config.json + the calibration.json it points at. Nothing hardcoded.

Intermediate stages are written to out_dir so a re-run can resume from merge:
  merged.pcd  [static.pcd]  denoised.pcd  colored.pcd  [labels.npz]
  [map_synth.pcd]  [flattened.pcd]  [anchored.pcd]

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
  with free_ratio ABOVE 1, so free-space evidence must dominate occupancy.
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
        "min_free": 5,      # voxel must be seen THROUGH in >= this many scans
        "free_ratio": 1.0,  # ...and free must at least MATCH its hit count
        "grazing_margin": 0.03  # extra ray pull-back per metre of range (m/m)
      }
    }
  Tuning: with carving on, min_span_s only needs to catch fast movers (leave
  at ~1.0); carving handles everything that dwells. If thin static structure
  (railings, poles, foliage) starts disappearing, raise min_free / free_ratio
  or the endpoint_margin; if ghosts survive, lower free_ratio (never below 1.0)
  or scan_stride/ray_stride (denser evidence). If the FLOOR or a whole wall
  section vanishes, the cause is grazing incidence: raise grazing_margin
  first, then free_ratio.
  A REFLECTIVE surface (glass, polished metal) is a hard limit rather than a
  tuning problem: when it returns nothing the rays pass through and hit what
  is behind, so the carver has genuine free-space evidence and no occupancy
  evidence to weigh against it. Where the LiDAR never measured the surface,
  no filter can preserve it -- set carve.enable false, or add those panels to
  the mesh by hand. Carving costs roughly one extra
  50-70%% of merge time at the default strides.

SEMANTICS: DETECT [6] and SYNTHESIZE [7], optional, a SIDE BRANCH.
Both run after colorize and write their own files. The main chain
(flatten -> anchor -> map_final.pcd) is byte-identical when they are off, so
turning them on can never change the map an existing downstream stage reads.
They keep the [6]/[7] numbering they have always had, which is about what they
are, not when they run.

  [6] DETECT lifts YOLO11-seg instance masks into the map. project_visible()
  already returns the map points visible from a camera pose, one per pixel and
  z-buffered against occlusion -- which is exactly the operator needed to turn
  a 2D mask into a 3D point set. So detect is colorize's loop with a different
  payload: instead of writing a colour it casts a class vote.

  A single frame's mask is never clean. Three guards, in order of how much
  they remove:
    * mask erosion strips the silhouette halo, where a boundary pixel sits on
      the object in one frame and on the wall 3 m behind it in the next;
    * a depth gate (median + MAD) drops the background genuinely visible
      THROUGH the mask -- between a chair's legs, between a plant's leaves;
    * multi-frame voting keeps only points many views agree on. Bleed lands on
      a different background point every frame because the camera moves, so it
      collects one vote; the object collects one per view.
  Instance identity is carried from the detector rather than rediscovered:
  two chairs side by side are one connected blob to any spatial clustering,
  but YOLO separated them in every image, and InstanceTracker keeps that.

  Walls and floors are NOT detected -- COCO has no class for them, and a
  second 2D semantic model would be more weights for a worse answer. They come
  from RANSAC planes classified by orientation against gravity. That is what
  makes "the TV is ON A WALL" computable, and it gives removal a surface to
  patch afterwards.

  Detect writes THREE things, and the third is the one a run is read for:
  labels.npz (per-point class / confidence / views / structure / instance),
  instances.json (what was found and WHERE -- centroid, extent, bbox, base
  height, view count), and semantic.pcd, the map coloured by class with
  structure muted underneath it. Arrays cannot be reviewed; the only way to
  know whether "chair" landed on the chair or on the wall behind it is to open
  the map with the labels painted on, so that file is not an extra, it is how
  the stage is checked.

  [7] SYNTHESIZE writes map_synth.pcd: the colorized map with a rules table
  applied per instance.
    remove   delete the points AND re-sample the plane behind them. The LiDAR
             never saw the wall behind a TV, so deleting it without filling
             leaves a TV-shaped void.
    replace  KEEP the measured points and add an asset fitted to them -- yaw
             from the instance's horizontal PCA, clamped anisotropic scale,
             ICP refinement projected back onto the gravity axis, base snapped
             to the classified support plane, and every asset point repainted
             from the measured colours around it. If the fit does not explain
             the measurement, no asset is emitted at all: a bad fit degrades
             to "unchanged", never to two overlapping chairs.
    keep     default.

  Config block (under 01_build_map):
    "detect": {
      "enable": true,
      "weights": "assets/models/yolo11x-seg.pt",   # relative to the CONFIG dir
      "conf": 0.35, "iou": 0.6, "imgsz": 960,
      "img_stride": 2, "min_baseline": 0.25, "min_rotation_deg": 10.0,
      "max_range": 8.0,            # ignore map points beyond this from a camera
      "classes": null,             # or ["chair", "tv", ...] to restrict
      "gpu_reserve_gib": 2.0,      # VRAM held back for the torch model
      "save_semantic": true,       # write semantic.pcd, coloured by class
      "semantic_output": "semantic.pcd",
      "vote": {"min_frames": 3, "min_ratio": 0.5, "mask_erode_px": 3,
               "depth_span": 0.6, "min_det_points": 25, "link_frac": 0.30},
      "cluster": {"eps": 0.12, "min_points": 60, "min_frames_seen": 2},
      "structure": {"enable": true, "dist": 0.04, "min_points": 20000,
                    "max_planes": 12, "fit_voxel": 0.05, "normal_tol_deg": 15.0,
                    "min_wall_height": 0.8}
    },
    "synthesize": {
      "enable": true,
      "assets": "assets",          # relative to the CONFIG dir
      "output": "map_synth.pcd",
      "density": 4000,             # asset points per m^2
      "min_coverage": 0.35,        # fit gate: measured points explained by it
      "scale_min": 0.75, "scale_max": 1.35,
      "dedupe_eps": 0.0,           # >0 thins measured points under the asset
      "fill_holes": true, "fill_spacing": 0.02, "fill_margin": 0.03,
      "rules": {
        "tv": {"on_wall": "remove", "default": "replace"},
        "clock": "remove", "person": "remove",
        "potted plant": "replace", "chair": "replace",
        "dining table": "replace", "couch": "replace"
      }
    }

  python3 01_build_map.py [pipeline_config.json]
"""
import os
import sys
import json
import time
import numpy as np
import open3d as o3d
import cv2
from pathlib import Path
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore

from pipeline_common import load_pipeline, pose_at_interp
import pipeline_detect as pdet
import pipeline_assets as past

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


def gpu_fits(nbytes, headroom=2.0, reserve=0):
    """True when `nbytes` can be held resident with room for the per-frame
    temporaries (which need roughly as much again during projection).

    `reserve` is VRAM this stage must NOT touch. Detect needs it: the torch
    model and its activations live on the same card, outside CuPy's pool and
    invisible to memGetInfo() until they are allocated, so a cloud sized to
    every free byte would fit right up until the first inference."""
    return _GPU["on"] and nbytes * headroom + int(reserve) < gpu_free_bytes()


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
                     min_free=5, free_ratio=1.0):
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
            # free must DOMINATE occupancy, not merely match a fraction of it.
            # A surface grazed by many rays -- a floor seen edge-on down a
            # corridor, a wall along the direction of travel -- collects large
            # free counts alongside large hit counts, so a threshold below 1.0
            # deletes exactly the structure it is meant to keep. A real mover
            # is the opposite: a handful of hits during its passage against
            # free evidence from the whole rest of the run.
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
                 endpoint_margin=0.0, grazing_margin=0.03, chunk=8000,
                 compact_at=20_000_000):
        self.voxel = float(voxel)
        self.inv = 1.0 / self.voxel
        self.step = self.voxel
        self.max_range = float(max_range)
        self.ray_stride = max(1, int(ray_stride))
        self.scan_stride = max(1, int(scan_stride))
        # never carve closer than 2 voxels to the hit: range noise would
        # otherwise eat the surface itself
        self.margin = 2.0 * self.voxel + float(endpoint_margin)
        # ...and pull back FURTHER in proportion to range. A ray striking a
        # surface at a shallow angle runs alongside it for metres before it
        # terminates, so a fixed guard protects only the last few centimetres
        # and the rest of the ray carves the very surface it just measured.
        # Pose error also grows with range. This is what keeps floors and
        # along-travel walls alive.
        self.grazing = float(grazing_margin)
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
        keep = dist * (1.0 - self.grazing) > (self.margin + self.step)
        if not bool(keep.any()):
            return
        vec = vec[keep]; dist = dist[keep]
        end = m.minimum(dist - (self.margin + self.grazing * dist),
                        self.max_range)
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
        # THE grazing fix: never carve a voxel that THIS SAME SCAN measured a
        # return in. A ray striking the floor 14 m away runs within one voxel
        # of it for the last two metres, passing over voxels where its sibling
        # rays terminate -- so without this the surface carves itself, and the
        # floor disappears. A real mover's voxels are absent from occ in every
        # scan after it leaves, so they still carve normally.
        # Distance-based guards cannot do this job: the offending samples are
        # metres from their own ray's endpoint while sitting millimetres above
        # the surface, and only the neighbouring returns know that.
        occ = m.unique(pack_voxels(
            m.floor(as_dev(world_pts) * self.inv).astype(np.int64)))
        if occ.size:
            pos = m.clip(m.searchsorted(occ, u), 0, occ.size - 1)
            u = u[occ[pos] != u]
        if u.size == 0:
            return
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
        grazing_margin=cv.get("grazing_margin", 0.03),
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
        min_free=cv.get("min_free", 5),
        free_ratio=cv.get("free_ratio", 1.0))
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
    point never receives the pixel's colour -- or, in detect [6], the pixel's
    class. Used by colorize [3] and detect [6]. Returns (global_idx, u, v, z)
    of the winning points, or None.

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


def residency(points, per_point_bytes, cull_block=2.0, reserve_gib=0.0):
    """Decide where a per-frame projection stage's cloud lives.

    Shared by colorize [3] and detect [6], which are the same loop with a
    different payload and must make the same call -- and, on a card that only
    fits one of them, must make it for the same reasons and print the same
    diagnosis. Returns (module, points, index); index is None when the cloud is
    resident on the device, because culling a device array costs more in the
    gather than the full transform does."""
    N = len(points)
    need = N * int(per_point_bytes)
    if on_gpu() and gpu_fits(need, reserve=int(float(reserve_gib) * 2**30)):
        m = xp()
        pts = m.asarray(np.asarray(points), dtype=m.float32)
        print(f"    {N} pts resident on GPU "
              f"({pts.nbytes / 2**20:.0f} MiB), no cull needed")
        return m, pts, None
    if on_gpu():
        extra = (f" (plus {reserve_gib:.1f} GiB reserved for the model)"
                 if reserve_gib else "")
        print(f"    cloud needs ~{need / 2**30:.1f} GiB resident{extra} but "
              f"only {gpu_free_bytes() / 2**30:.1f} GiB is free on the card -> "
              f"running this stage on the CPU.\n"
              f"    set colorize.voxel (e.g. 0.04) to work on a downsampled "
              f"copy and keep this on the GPU")
    pts = np.asarray(points, dtype=np.float64)
    index = BlockIndex(pts, block=float(cull_block))
    print(f"    frustum index: {len(index.start)} blocks")
    return np, pts, index


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
    m, pts, index = residency(work.points, 12 + 3 + 4,
                              cull_block=float(c.get("cull_block", 2.0)))
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


# =============================================================================
# [6] DETECT
# =============================================================================
def resolve_in(P, path):
    """Resolve a stage INPUT path against the config's directory.

    P.stage() resolves output filenames against out_dir, which is right for
    things this run produces and wrong for things it consumes: the asset
    library and the model weights live with the config, not with the outputs,
    and out_dir is frequently a scratch directory that gets wiped."""
    if os.path.isabs(path):
        return path
    return os.path.join(P.cfg_dir, path)


def sub_dir(P, *parts):
    """A directory under out_dir, created on demand.

    P.outp() cannot do this: it passes any name containing a directory
    separator straight through, so P.outp("layers/chair.pcd") resolves against
    the CURRENT WORKING DIRECTORY instead of out_dir and quietly scatters the
    layer clouds wherever the script was launched from."""
    d = os.path.join(P.out_dir, *parts)
    os.makedirs(d, exist_ok=True)
    return d


def save_subset(path, pts, cols, idx):
    """Write a subset of the map as its own cloud. Returns points written."""
    if idx is None or len(idx) == 0:
        return 0
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(pts[idx])
    if cols is not None:
        pc.colors = o3d.utility.Vector3dVector(cols[idx])
    o3d.io.write_point_cloud(path, pc)
    return len(idx)


def safe_name(s):
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in s)


def split_clouds(P, d, pts, cols, cls, struct_lab, instances, names):
    """Write the map split by what stage [6] decided each point is.

    Four groupings, because they answer different questions:

      background.pcd   the room WITHOUT its contents -- structure and anything
                       unclaimed. This is the one to re-mesh or hand to a
                       planner; objects are what makes a map non-reusable.
      objects.pcd      every detected object point, the complement of the above
      layers/<x>.pcd   one per class AND one per structure kind, so "just the
                       walls" or "just the chairs" is a file, not a filter
      objects/<n>.pcd  one per instance, for measuring or exporting a single
                       object

    Colours come from the map itself (colorize's), not the semantic palette --
    these are extracted geometry meant to be looked at and reused, and
    semantic.pcd already exists for the label view.
    """
    out = {"clouds": {}, "objects": {}}
    obj_mask = cls >= 0
    obj_idx = np.flatnonzero(obj_mask)
    bg_idx = np.flatnonzero(~obj_mask)

    if d.get("save_split", True):
        n = save_subset(P.outp("background.pcd"), pts, cols, bg_idx)
        print(f"    background.pcd  {n:9d} pts  (structure + unclaimed)")
        out["clouds"]["background"] = "background.pcd"
        n = save_subset(P.outp("objects.pcd"), pts, cols, obj_idx)
        print(f"    objects.pcd     {n:9d} pts  ({len(instances)} objects)")
        out["clouds"]["objects"] = "objects.pcd"

    if d.get("save_layers", True):
        ld = sub_dir(P, "layers")
        for code, nm in ((1, "floor"), (2, "wall"), (3, "ceiling"),
                         (4, "support")):
            idx = np.flatnonzero((struct_lab == code) & ~obj_mask)
            if idx.size:
                save_subset(os.path.join(ld, f"{nm}.pcd"), pts, cols, idx)
                print(f"    layers/{nm}.pcd  {idx.size:9d} pts")
                out["clouds"][nm] = f"layers/{nm}.pcd"
        for cid in np.unique(cls[obj_mask]):
            nm = safe_name(names.get(int(cid), str(int(cid))))
            idx = np.flatnonzero(cls == cid)
            save_subset(os.path.join(ld, f"{nm}.pcd"), pts, cols, idx)
            print(f"    layers/{nm}.pcd  {idx.size:9d} pts")
            out["clouds"][nm] = f"layers/{nm}.pcd"

    if d.get("save_per_object", False):
        od = sub_dir(P, "objects")
        for ins in instances:
            nm = safe_name(names.get(ins["cls_id"], str(ins["cls_id"])))
            rel = f"objects/{ins['instance']:03d}_{nm}.pcd"
            save_subset(os.path.join(od, os.path.basename(rel)),
                        pts, cols, ins["idx"])
            out["objects"][ins["instance"]] = rel
        print(f"    objects/        {len(instances)} per-object clouds")
    return out


def detect(P, S, s, pcd):
    """[6] YOLO-seg masks -> per-point labels, object instances, structure."""
    d = s["detect"]
    print("[6] detect: YOLO-seg instance masks lifted into the map")
    tr_t, tr_T = P.traj
    W, H = s["image_width"], s["image_height"]
    stride = int(d.get("img_stride", 2))
    max_range = float(d.get("max_range", 8.0))
    vote = d.get("vote", {})
    erode_px = int(vote.get("mask_erode_px", 3))
    depth_span = float(vote.get("depth_span", 0.6))
    min_det_pts = int(vote.get("min_det_points", 25))

    if not d.get("weights"):
        raise SystemExit("01_build_map.detect.enable is true but detect.weights "
                         "is not set -- point it at a YOLO -seg checkpoint "
                         "(see assets/README.md for fetching one)")
    detector = pdet.Detector(resolve_in(P, d["weights"]),
                             conf=d.get("conf", 0.35), iou=d.get("iou", 0.6),
                             imgsz=d.get("imgsz", 960),
                             classes=d.get("classes"),
                             exclude=d.get("exclude"),
                             device=d.get("device"))
    print(f"    {detector.describe()}")
    n_cls = max(detector.names) + 1

    pts_np = np.asarray(pcd.points)
    N = len(pts_np)
    m, pts, index = residency(pcd.points, 12,
                              cull_block=float(d.get("cull_block", 2.0)),
                              reserve_gib=float(d.get("gpu_reserve_gib", 2.0)))

    votes = pdet.VoteAccumulator(N, n_cls)
    tracker = pdet.InstanceTracker(N, link_frac=float(vote.get("link_frac", 0.30)))
    use_frame = pose_gate(d)

    with AnyReader([Path(P.dataset["bag"])], default_typestore=TS) as r:
        conns = [cc for cc in r.connections if cc.topic == S.image_topic]
        n = n_used = n_det = 0
        for conn, _, raw in r.messages(connections=conns):
            n += 1
            if n % stride:
                continue
            msg = r.deserialize(raw, conn.msgtype)
            t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            # interpolated, unlike colorize's nearest-pose lookup: at walking
            # pace a 0.1 s lever arm is tens of millimetres of camera error,
            # which is exactly what slides a mask off an object and onto the
            # surface behind it
            Twl, gap = pose_at_interp(tr_t, tr_T, t)
            if gap > s["time_tol"]:
                continue
            Twc = Twl @ S.T_lidar_camera
            if not use_frame(Twc):
                continue
            img = decode_img(msg)
            if img is None:
                continue
            if (img.shape[1], img.shape[0]) != (W, H):
                img = cv2.resize(img, (W, H))
            # inference BEFORE projection: a frame with nothing in it is the
            # common case outdoors, and skipping its projection is free
            dets = detector(img)
            if not dets:
                continue
            vis = project_visible(index, pts, Twc, S, W, H, max_range)
            if vis is None:
                continue
            g, uu, vv, zz = (as_cpu(x) for x in vis)
            n_used += 1
            for k in dets:
                mk = k["mask"]
                if mk.shape != (H, W):
                    mk = cv2.resize(mk.astype(np.uint8), (W, H),
                                    interpolation=cv2.INTER_NEAREST).astype(bool)
                mk = pdet.erode_mask(mk, erode_px)
                sel = mk[vv, uu]
                if not sel.any():
                    continue
                gi = g[sel]
                keep = pdet.depth_gate(zz[sel], depth_span)
                gi = gi[keep]
                if gi.size < min_det_pts:
                    continue
                votes.add(gi, k["cls_id"],
                          np.full(gi.size, k["conf"], np.float32))
                tracker.add(gi, k["cls_id"])
                n_det += 1
            if n % 500 == 0:
                print(f"    img {n} ({n_used} used, {n_det} detections)",
                      flush=True)

    del pts
    gpu_free()
    print(f"    {n_used} frames used, {n_det} detections, "
          f"{len(tracker.cls)} tracked instances")

    cls, conf, frames = votes.resolve(
        min_frames=int(vote.get("min_frames", 3)),
        min_ratio=float(vote.get("min_ratio", 0.5)))
    cl = d.get("cluster", {})
    instances = pdet.build_instances(
        pts_np, cls, conf, tracker.owner, tracker.cls,
        min_points=int(cl.get("min_points", 60)),
        eps=float(cl.get("eps", 0.12)),
        min_frames_seen=int(cl.get("min_frames_seen", 2)),
        hits=tracker.hits)
    print(f"    voted {int((cls >= 0).sum())}/{N} pts -> {len(instances)} "
          f"instances after clustering")

    st = d.get("structure", {})
    struct = None
    if st.get("enable", True):
        models = pdet.fit_plane_models(
            pts_np, dist=float(st.get("dist", 0.04)),
            min_points=int(st.get("min_points", 20000)),
            max_planes=int(st.get("max_planes", 12)),
            fit_voxel=float(st.get("fit_voxel", 0.05)))
        planes = pdet.assign_planes(pts_np, models,
                                    dist=float(st.get("dist", 0.04)),
                                    min_points=int(st.get("min_points", 20000)))
        struct = pdet.Structure(
            planes, pts_np, tol_deg=float(st.get("normal_tol_deg", 15.0)),
            min_wall_height=float(st.get("min_wall_height", 0.8)))
        print(f"    structure: {struct.summary()}")
        # structure exists now, so instances can be cleaned of the wall skirt
        # that no purely image-space or depth-space guard could reach
        trimmed = 0
        for ins in instances:
            new_idx, w = struct.trim_wall_skirt(ins["idx"], pts_np)
            if w is not None and new_idx.size >= int(cl.get("min_points", 60)):
                trimmed += ins["idx"].size - new_idx.size
                ins["idx"] = new_idx
        if trimmed:
            print(f"    wall-skirt trim: dropped {trimmed} bled wall points "
                  f"from instances standing proud of a wall")
        with open(P.outp("structure.json"), "w") as f:
            json.dump({"models": [p["model"].tolist() for p in struct.planes],
                       "kinds": [p["kind"] for p in struct.planes],
                       "dist": float(st.get("dist", 0.04)),
                       "min_points": int(st.get("min_points", 20000)),
                       "normal_tol_deg": float(st.get("normal_tol_deg", 15.0)),
                       "min_wall_height": float(st.get("min_wall_height", 0.8))},
                      f, indent=2)

    # instances are final now (clustered, and skirt-trimmed if structure ran),
    # so the per-point labels are made to agree with them before either is
    # written -- the two files must never disagree about what a point is
    cls = pdet.reconcile(cls, instances)
    struct_lab = (struct.labels(N) if struct is not None
                  else np.zeros(N, np.uint8))
    np.savez_compressed(P.outp("labels.npz"), cls=cls, conf=conf,
                        frames=frames, structure=struct_lab,
                        inst=pdet.instance_array(N, instances),
                        n_points=np.int64(N))
    geom = pdet.instance_geometry(pts_np, instances)
    pdet.dump_instances(P.outp("instances.json"), instances, detector.names, N,
                        extra=geom)

    # what was identified, and where -- the summary a run is actually read for
    print(f"    identified {len(instances)} objects "
          f"({int((cls >= 0).sum())} of {N} points):")
    by_cls = {}
    for ins in instances:
        by_cls.setdefault(ins["cls_id"], []).append(ins)
    for cid in sorted(by_cls, key=lambda c: -sum(i["idx"].size
                                                 for i in by_cls[c])):
        group = by_cls[cid]
        nm = detector.names.get(cid, str(cid))
        print(f"      {nm:<14} x{len(group):<3d} "
              f"{pdet.hexc(pdet.class_color(cid))}  "
              f"{sum(i['idx'].size for i in group):7d} pts")
        for ins in sorted(group, key=lambda i: -i["idx"].size):
            g = geom[ins["instance"]]
            print(f"        #{ins['instance']:<3d} at "
                  f"[{g['centroid'][0]:7.2f},{g['centroid'][1]:7.2f},"
                  f"{g['centroid'][2]:6.2f}]  "
                  f"{g['extent'][0]:.2f}x{g['extent'][1]:.2f}x"
                  f"{g['extent'][2]:.2f} m  {ins['idx'].size:6d} pts  "
                  f"conf {ins['conf']:.2f}  {ins['n_frames']} views")

    if d.get("save_semantic", True):
        col, legend = pdet.semantic_colors(cls, struct_lab, detector.names)
        sem = o3d.geometry.PointCloud()
        sem.points = o3d.utility.Vector3dVector(pts_np)
        sem.colors = o3d.utility.Vector3dVector(col)
        save(P, sem, d.get("semantic_output", "semantic.pcd"))
        print("    legend: " + "  ".join(f"{pdet.hexc(c)} {n}"
                                         for n, c, _ in legend))
        del sem, col

    # ---- the map, split by what each point turned out to be ----------------
    src_cols = np.asarray(pcd.colors) if pcd.has_colors() else None
    split = split_clouds(P, d, pts_np, src_cols, cls, struct_lab, instances,
                         detector.names)

    inv = d.get("inventory", "objects_inventory.yaml")
    if inv:
        ctx = pdet.object_context(pts_np, instances, struct)
        pdet.write_inventory(
            P.outp(inv), instances, detector.names, geom, ctx, cls, struct,
            pts_np,
            meta={"source_cloud": os.path.basename(
                      P.outp("colored.pcd") if s["colorize"]["enable"]
                      else P.outp("denoised.pcd")),
                  "model": os.path.basename(resolve_in(P, d["weights"])),
                  "conf": float(d.get("conf", 0.35)),
                  "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                 time.gmtime())},
            clouds=split["objects"])
        print(f"    inventory -> {P.outp(inv)}")

    print(f"    saved {P.outp('labels.npz')} and {P.outp('instances.json')}")
    return cls, conf, instances, struct, detector.names


def load_semantics(P, pcd):
    """Rebuild detect's output from disk, or None if it is absent/stale.

    The point-count check is not a formality. labels.npz indexes THIS cloud;
    if colorize's drop_gray setting changed, or the merge was rebuilt, the
    indices still resolve and every label silently lands on a different point.
    That is a wrong answer that looks like a right one, so it is refused
    outright rather than approximated."""
    lp, ip = P.outp("labels.npz"), P.outp("instances.json")
    if not (os.path.exists(lp) and os.path.exists(ip)):
        return None
    z = np.load(lp)
    N = len(pcd.points)
    if int(z["n_points"]) != N:
        print(f"[6] ! labels.npz was built for {int(z['n_points'])} points but "
              f"this cloud has {N}. Refusing to reuse it -- delete "
              f"{lp} to re-detect.")
        return None
    blob = json.load(open(ip))
    names = {int(k): v for k, v in blob.get("names", {}).items()}
    cls, conf, inst = z["cls"], z["conf"], z["inst"]
    instances = []
    for rec in blob["instances"]:
        idx = np.flatnonzero(inst == rec["instance"])
        if idx.size == 0:
            continue
        instances.append({"instance": rec["instance"],
                          "cls_id": rec["class_id"], "idx": idx,
                          "conf": rec["confidence"],
                          "n_frames": rec.get("n_frames", 0)})
    struct = None
    sp = P.outp("structure.json")
    if os.path.exists(sp):
        sj = json.load(open(sp))
        pts = np.asarray(pcd.points)
        planes = pdet.assign_planes(pts, [np.asarray(m) for m in sj["models"]],
                                    dist=sj.get("dist", 0.04),
                                    min_points=sj.get("min_points", 20000))
        struct = pdet.Structure(planes, pts,
                                tol_deg=sj.get("normal_tol_deg", 15.0),
                                min_wall_height=sj.get("min_wall_height", 0.8))
    print(f"[resume] loaded labels.npz: {len(instances)} instances"
          + (f", structure {struct.summary()}" if struct else ""))
    return cls, conf, instances, struct, names


# =============================================================================
# [7] SYNTHESIZE
# =============================================================================
def resolve_rule(rules, cls_name, on_wall):
    """Action for a class, optionally conditioned on what it is attached to.

    A plain string applies unconditionally; a dict keys on the support, which
    is what "a TV on a wall is fixture, a TV on a stand is furniture" needs.
    """
    r = rules.get(cls_name, rules.get("*", "keep"))
    if isinstance(r, dict):
        if on_wall and "on_wall" in r:
            return r["on_wall"]
        return r.get("default", "keep")
    return r


def synthesize(P, S, s, pcd, instances, struct, names):
    """[7] Apply the rules table -> map_synth.pcd + scene.json."""
    y = s["synthesize"]
    print("[7] synthesize: rules applied per instance")
    lib = past.AssetLibrary(resolve_in(P, y.get("assets", "assets")))
    print(f"    asset library: {lib.describe()}")
    rules = y.get("rules", {})
    density = float(y.get("density", 4000))
    min_cov = float(y.get("min_coverage", 0.35))
    scale_range = (float(y.get("scale_min", 0.75)),
                   float(y.get("scale_max", 1.35)))
    dedupe = float(y.get("dedupe_eps", 0.0))
    fill = bool(y.get("fill_holes", True))
    fill_sp = float(y.get("fill_spacing", 0.02))
    fill_mg = float(y.get("fill_margin", 0.03))

    pts = np.asarray(pcd.points)
    cols = (np.asarray(pcd.colors) if pcd.has_colors()
            else np.full((len(pts), 3), 0.5))
    remove = np.zeros(len(pts), bool)
    add_p, add_c = [], []
    scene = []
    n_rep = n_rem = n_keep = n_fill = 0

    for ins in instances:
        name = names.get(ins["cls_id"], str(ins["cls_id"]))
        idx = ins["idx"]
        ip, ic = pts[idx], cols[idx]
        wall, frac = (struct.wall_contact(ip) if struct is not None
                      else (None, 0.0))
        action = resolve_rule(rules, name, wall is not None)
        rec = {"instance": ins["instance"], "class": name,
               "confidence": round(float(ins["conf"]), 4),
               "n_points": int(idx.size),
               "on_wall": bool(wall is not None),
               "wall_fraction": round(float(frac), 3),
               "requested": action}

        if action == "remove":
            remove[idx] = True
            plane = wall
            if plane is None and struct is not None:
                plane, _ = struct.support_under(ip)
            if fill and plane is not None:
                # surviving surface only: a TV flush against a wall has its own
                # points assigned to that wall's plane, and feeding them back in
                # as "already measured" would suppress the very patch we need
                sidx = plane["idx"][~remove[plane["idx"]]]
                fp, fc = past.fill_plane_hole(plane, ip, pts[sidx], cols[sidx],
                                              spacing=fill_sp, margin=fill_mg)
                if len(fp):
                    add_p.append(fp); add_c.append(fc)
                    n_fill += len(fp)
                    rec["hole_filled_points"] = int(len(fp))
                    rec["hole_plane"] = plane["kind"]
            rec["action"] = "remove"
            n_rem += 1
            scene.append(rec)
            continue

        if action == "replace":
            ext = past.robust_extent(ip)
            meta = lib.pick(name, target_size=ext)
            if meta is None:
                rec["action"] = "keep"
                rec["reason"] = "no asset for this class"
                n_keep += 1
                scene.append(rec)
                continue
            base_z = float(np.percentile(ip[:, 2], 2.0))
            plane = None
            if struct is not None:
                plane, base_z = struct.support_under(ip, prefer=meta["support"])
            ap, ac = lib.points(meta, density)
            fit = past.fit_asset(ap, meta["size"], ip, meta["yaw_symmetry"],
                                 base_z, scale_range=scale_range,
                                 icp=bool(y.get("icp", True)))
            if fit is None or fit["coverage"] < min_cov:
                # the whole point of the gate: an asset that does not explain
                # the measurement would sit on top of it as a second object
                rec["action"] = "keep"
                rec["reason"] = ("fit rejected: coverage "
                                 f"{0.0 if fit is None else fit['coverage']:.2f}"
                                 f" < {min_cov}")
                n_keep += 1
                scene.append(rec)
                continue
            newc = past.repaint(fit["pts"], ac, ip, ic,
                                radius=float(y.get("repaint_radius", 0.08)),
                                tint=float(y.get("repaint_tint", 0.65)))
            add_p.append(fit["pts"]); add_c.append(newc)
            if dedupe > 0:
                from scipy.spatial import cKDTree
                dd, _ = cKDTree(fit["pts"]).query(ip, k=1)
                remove[idx[dd < dedupe]] = True
            rec.update({"action": "replace",
                        "asset": os.path.relpath(meta["path"], lib.root),
                        "support": meta["support"],
                        "support_plane": None if plane is None else plane["kind"],
                        "base_z": round(float(base_z), 4),
                        "yaw_deg": round(float(np.degrees(fit["yaw"])), 2),
                        "scale": [round(float(v), 4) for v in fit["scale"]],
                        "coverage": round(float(fit["coverage"]), 3),
                        "rmse_m": round(float(fit["rmse"]), 4),
                        "asset_points": int(len(fit["pts"])),
                        "pose": past.pose_matrix(fit["yaw"], fit["scale"],
                                                 fit["centre"]).tolist()})
            n_rep += 1
            scene.append(rec)
            continue

        rec["action"] = "keep"
        n_keep += 1
        scene.append(rec)

    keep_mask = ~remove
    out_p = [pts[keep_mask]] + add_p
    out_c = [cols[keep_mask]] + add_c
    out = o3d.geometry.PointCloud()
    out.points = o3d.utility.Vector3dVector(np.vstack(out_p))
    out.colors = o3d.utility.Vector3dVector(np.clip(np.vstack(out_c), 0, 1))
    with open(P.outp("scene.json"), "w") as f:
        json.dump({"source_points": int(len(pts)),
                   "removed_points": int(remove.sum()),
                   "added_points": int(sum(len(a) for a in add_p)),
                   "instances": scene}, f, indent=2)
    print(f"    {n_rep} replaced, {n_rem} removed, {n_keep} kept | "
          f"-{int(remove.sum())} pts, +{int(sum(len(a) for a in add_p))} pts "
          f"({n_fill} of them surface repair)")
    print(f"    saved {P.outp('scene.json')}")
    return out


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

    # ---- semantic side branch: never touches `pcd`, writes its own files ----
    synth = None
    if s.get("detect", {}).get("enable", False):
        sem = load_semantics(P, pcd)
        if sem is None:
            sem = detect(P, S, s, pcd)
        cls, conf, instances, struct, names = sem
        if s.get("synthesize", {}).get("enable", False):
            synth = synthesize(P, S, s, pcd, instances, struct, names)
            save(P, synth, s["synthesize"].get("output", "map_synth.pcd"))

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
        if synth is not None:
            # the synth cloud is a sibling of the map, not a descendant of it,
            # so it needs the same shift applied or the two stop overlaying
            synth.translate(shift)
            save(P, synth, s["synthesize"].get("output", "map_synth.pcd"))

    final = P.outp(s["output"])
    o3d.io.write_point_cloud(final, pcd)
    print(f"DONE -> {final}  ({len(pcd.points)} points)")


def load_traj_cached(P):
    from pipeline_common import load_traj
    return load_traj(P.dataset["traj"])


if __name__ == "__main__":
    main()
