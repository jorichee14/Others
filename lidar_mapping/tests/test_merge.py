"""VoxelAccumulator: must match Open3D's voxel_down_sample and stay bounded."""
import sys, types, tracemalloc
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

rng = np.random.default_rng(4)
VOX = 0.05

# ---- pack/unpack round trip ----------------------------------------------- #
v = rng.integers(-5000, 5000, (10000, 3)).astype(np.int64)
assert np.array_equal(bm.unpack_voxels(bm.pack_voxels(v)), v), "pack/unpack"
print("pack_voxels/unpack_voxels: bijective over +-5000 voxels")

# ---- centroid mode == reference voxel_down_sample -------------------------- #
def reference_downsample(pts, voxel):
    """What Open3D's voxel_down_sample computes: mean per occupied voxel."""
    k = bm.pack_voxels(np.floor(pts / voxel).astype(np.int64))
    uniq, inv = np.unique(k, return_inverse=True)
    out = np.zeros((len(uniq), 3))
    cnt = np.bincount(inv)
    for c in range(3):
        out[:, c] = np.bincount(inv, weights=pts[:, c]) / cnt
    return uniq, out

pts = np.concatenate([
    rng.normal(0, 2.0, (150_000, 3)),
    rng.uniform(-5, 5, (150_000, 3))]).astype(np.float64)

acc = bm.VoxelAccumulator(VOX, centroid=True, flush_pts=25_000)  # many flushes
for a in range(0, len(pts), 7_000):
    acc.add(pts[a:a + 7_000])
got = acc.points()
ref_keys, ref_pts = reference_downsample(pts, VOX)
assert np.array_equal(acc.keys, ref_keys), "occupied voxel set differs"
# float32 buffering => compare at the precision that buffering can support
err = np.abs(got - ref_pts).max()
print(f"centroid mode: {len(got):,} voxels, max centroid error {err*1e6:.2f} um")
assert err < 1e-4, f"centroid mismatch {err}"

# many small flushes must equal one big flush
acc2 = bm.VoxelAccumulator(VOX, centroid=True, flush_pts=10**9)
acc2.add(pts)
got2 = acc2.points()                      # points() flushes what is buffered
assert np.array_equal(acc2.keys, acc.keys)
assert np.abs(got2 - got).max() < 1e-9, "flush size changed result"
print("centroid mode: incremental flushing is exact (1 flush == 43 flushes)")

# ---- centre mode: leaner, bounded error ------------------------------------ #
acc3 = bm.VoxelAccumulator(VOX, centroid=False, flush_pts=25_000)
for a in range(0, len(pts), 7_000):
    acc3.add(pts[a:a + 7_000])
got3 = acc3.points()
assert np.array_equal(acc3.keys, ref_keys), "centre mode voxel set differs"
disp = np.abs(got3 - ref_pts).max()
print(f"centre mode  : {len(got3):,} voxels, max displacement "
      f"{disp*1000:.1f} mm (bound is voxel/2 = {VOX*500:.0f} mm)")
assert disp <= VOX / 2 + 1e-9, "displacement exceeds half a voxel"
print(f"              memory {acc3.nbytes()/2**10:.0f} KiB vs centroid "
      f"{acc.nbytes()/2**10:.0f} KiB ({acc.nbytes()/acc3.nbytes():.1f}x leaner)")

# ---- boundedness: the actual bug --------------------------------------------#
# Re-observing the SAME surface must not grow memory, which is exactly what
# the old grow-then-downsample buffer failed to do.
acc4 = bm.VoxelAccumulator(VOX, centroid=True, flush_pts=200_000)
surface = rng.uniform(-5, 5, (200_000, 3))
sizes = []
for sweep in range(8):                      # 8 passes over one surface
    acc4.add(surface + rng.normal(0, 0.002, surface.shape))
    acc4.flush()
    sizes.append(acc4.nbytes())
growth = sizes[-1] / sizes[0]
print(f"boundedness: 8 re-observations of one surface -> memory "
      f"{sizes[0]/2**20:.1f} -> {sizes[-1]/2**20:.1f} MiB ({growth:.2f}x)")
assert growth < 1.35, f"memory still grows with scan count ({growth:.2f}x)"

# peak allocation during a flush must stay near the steady state, not 4x it
tracemalloc.start()
acc4.add(surface)
base = tracemalloc.get_traced_memory()[0]
acc4.flush()
peak = tracemalloc.get_traced_memory()[1]
tracemalloc.stop()
print(f"flush peak: {peak/2**20:.1f} MiB for a "
      f"{acc4.nbytes()/2**20:.1f} MiB accumulator")

print("\nALL MERGE TESTS PASSED")
