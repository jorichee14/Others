"""GPU-backend refactor tests.

No CUDA here, so this does two things:
  1. verifies the CPU path still produces the SAME answers as before,
  2. exercises the GPU-only code branches (index=None whole-cloud projection,
     group_bounds/group_sum reductions, compaction) which the GPU path shares
     verbatim with the CPU one -- so a logic error would show up here even
     though the device does not.
"""
import sys, types
import numpy as np

for name in ("open3d", "cv2", "rosbags", "rosbags.highlevel",
             "rosbags.typesys", "pipeline_common"):
    sys.modules.setdefault(name, types.ModuleType(name))
sys.modules["rosbags.highlevel"].AnyReader = object
sys.modules["rosbags.typesys"].Stores = types.SimpleNamespace(ROS2_HUMBLE=0)
sys.modules["rosbags.typesys"].get_typestore = lambda *_: None
sys.modules["pipeline_common"].load_pipeline = lambda *_: None
for a in ("geometry", "utility", "io", "core", "t"):
    setattr(sys.modules["open3d"], a, types.SimpleNamespace())
sys.modules["cv2"].cvtColor = lambda *a: np.zeros((1, 1, 3), np.uint8)
sys.modules["cv2"].COLOR_HSV2RGB = 0

import importlib.util
spec = importlib.util.spec_from_file_location(
    "bm", "/home/user/Others/lidar_mapping/01_build_map.py")
bm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bm)

rng = np.random.default_rng(3)
S = types.SimpleNamespace(fx=300.0, fy=300.0, cx=320.0, cy=180.0)
W, H, MAXR = 640, 360, 10.0

# ---- group_bounds / group_sum vs numpy reference -------------------------- #
keys = rng.integers(0, 50, 4000).astype(np.int64)
vals = rng.integers(1, 5, 4000).astype(np.int64)
order, uniq, start, count = bm.group_bounds(np, keys)
assert np.array_equal(uniq, np.unique(keys))
ref_cnt = np.bincount(keys, minlength=50)[uniq]
assert np.array_equal(count, ref_cnt), "group_bounds count mismatch"
gs = bm.group_sum(np, vals[order], start, count)
ref_sum = np.bincount(keys, weights=vals, minlength=50)[uniq].astype(np.int64)
assert np.array_equal(gs, ref_sum), "group_sum mismatch"
print(f"group_bounds/group_sum: {len(uniq)} groups match numpy reference")

# stable ordering => per-group first/last are chronological (what DynStats
# relies on instead of a min/max reduction)
times = np.arange(4000, dtype=np.float64)
ts = times[order]
assert np.array_equal(ts[start], np.array(
    [times[keys == k].min() for k in uniq])), "first-of-group != tmin"
assert np.array_equal(ts[start + count - 1], np.array(
    [times[keys == k].max() for k in uniq])), "last-of-group != tmax"
print("stable-sort trick: first/last of each group == tmin/tmax")

# ---- DynStats: compaction must not change the verdict --------------------- #
def build(compact_at):
    d = bm.DynStats(0.15, compact_at=compact_at)
    r2 = np.random.default_rng(11)
    for i in range(60):
        wall = np.stack([np.full(500, 5.0), r2.uniform(-3, 3, 500),
                         r2.uniform(0, 3, 500)], 1)
        pts = wall
        if i < 20:                      # a transient blob for 2 s
            blob = np.stack([np.full(80, 2.5), r2.uniform(-.2, .2, 80),
                             r2.uniform(0, 1.7, 80)], 1)
            pts = np.vstack([wall, blob])
        d.add(pts, i * 0.1)
    return d.dynamic_keys(2, 3.0)

a_keys, a_stats = build(10**9)          # never compacts
b_keys, b_stats = build(500)            # compacts many times
assert np.array_equal(np.sort(a_keys), np.sort(b_keys)), \
    "compaction changed the dynamic set"
assert a_stats == b_stats, f"{a_stats} != {b_stats}"
print(f"DynStats: identical with and without compaction "
      f"({a_stats[0]} static / {a_stats[1]} dynamic voxels)")

# ---- carver: compaction invariance ---------------------------------------- #
def carve(compact_at):
    c = bm.FreeSpaceCarver(0.15, max_range=10.0, ray_stride=1, scan_stride=1,
                           compact_at=compact_at)
    r2 = np.random.default_rng(5)
    for i in range(25):
        wall = np.stack([np.full(400, 6.0), r2.uniform(-2, 2, 400),
                         r2.uniform(0, 2.5, 400)], 1)
        c.add(np.zeros(3), wall)
    q = np.unique(bm.pack_voxels(
        np.floor(np.stack([np.full(300, 3.0), r2.uniform(-1, 1, 300),
                           r2.uniform(0, 2, 300)], 1) / 0.15).astype(np.int64)))
    return q, c.counts_for(q)

q1, c1 = carve(10**9)
q2, c2 = carve(1000)
assert np.array_equal(q1, q2) and np.array_equal(c1, c2), \
    "carver compaction changed the free counts"
assert c1.max() > 0, "carving should see through empty space"
print(f"FreeSpaceCarver: compaction invariant (max free count {int(c1.max())})")

# ---- projection: GPU branch (index=None) == CPU branch (BlockIndex) ------- #
pts = np.concatenate([
    np.stack([rng.uniform(-10, 10, 200_000), rng.uniform(-10, 10, 200_000),
              rng.uniform(0, 3, 200_000)], 1),
    np.stack([rng.uniform(-1, 1, 20_000), rng.uniform(2, 4, 20_000),
              rng.uniform(0, 2, 20_000)], 1)])
index = bm.BlockIndex(pts, block=2.0)

def pose(k):
    r2 = np.random.default_rng(200 + k)
    yaw = r2.uniform(0, 2 * np.pi)
    T = np.eye(4)
    T[:3, 0] = [np.sin(yaw), -np.cos(yaw), 0]
    T[:3, 1] = [0, 0, -1]
    T[:3, 2] = [np.cos(yaw), np.sin(yaw), 0]
    T[:3, 3] = [r2.uniform(-5, 5), r2.uniform(-5, 5), 1.4]
    return T

n_ok = 0
for k in range(25):
    T = pose(k)
    a = bm.project_visible(index, pts, T, S, W, H, MAXR)       # CPU path
    b = bm.project_visible(None, pts, T, S, W, H, MAXR)        # GPU path
    if a is None and b is None:
        continue
    assert a is not None and b is not None, f"pose {k}: one path returned None"
    ma = dict(zip((a[2] * W + a[1]).tolist(), a[0].tolist()))
    mb = dict(zip((b[2] * W + b[1]).tolist(), b[0].tolist()))
    assert ma == mb, f"pose {k}: GPU branch disagrees with CPU branch"
    n_ok += 1
print(f"project_visible: whole-cloud (GPU) branch == frustum (CPU) branch "
      f"on {n_ok} poses")

# ---- backend plumbing ------------------------------------------------------ #
assert bm.xp() is np and not bm.on_gpu(), "should default to CPU"
assert bm.init_gpu(False) is False
assert bm.init_gpu(True) is False, "no CUDA here -> must report False, not throw"
assert bm.xp() is np, "failed GPU init must leave the CPU backend active"
x = np.arange(5)
assert bm.as_cpu(bm.as_dev(x)) is not None
bm.gpu_free()
print("backend: falls back to numpy cleanly when CUDA is absent")

print("\nALL GPU-BACKEND TESTS PASSED")
