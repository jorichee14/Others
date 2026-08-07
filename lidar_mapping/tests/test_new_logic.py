"""Synthetic tests for the new dynamic-removal carving + plane grid mesher."""
import sys, types
import numpy as np

# ---- stub heavy deps so 01_build_map imports -------------------------------
for name in ("open3d", "cv2", "rosbags", "rosbags.highlevel",
             "rosbags.typesys", "pipeline_common"):
    m = types.ModuleType(name)
    sys.modules.setdefault(name, m)
sys.modules["rosbags.highlevel"].AnyReader = object
sys.modules["rosbags.typesys"].Stores = types.SimpleNamespace(ROS2_HUMBLE=0)
sys.modules["rosbags.typesys"].get_typestore = lambda *_: None
sys.modules["pipeline_common"].load_pipeline = lambda *_: None
sys.modules["open3d"].geometry = types.SimpleNamespace()
sys.modules["open3d"].utility = types.SimpleNamespace()
sys.modules["open3d"].io = types.SimpleNamespace()

sys.path.insert(0, "/home/user/Others/lidar_mapping")
import importlib.util
spec = importlib.util.spec_from_file_location(
    "build_map", "/home/user/Others/lidar_mapping/01_build_map.py")
bm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bm)

rng = np.random.default_rng(0)

# ---- scene: a wall at x=5 (yz plane), sensor at origin ---------------------
# 200 scans over 20 s. A "person" (small blob) stands at x=2.5 for the first
# 100 scans (10 s dwell -> span test CANNOT catch it), then leaves.
def wall_points(n=2500):
    y = rng.uniform(-3, 3, n); z = rng.uniform(0, 6.0, n)
    return np.stack([np.full(n, 5.0), y, z], 1)

def person_points(n=150):
    y = rng.uniform(-0.2, 0.2, n); z = rng.uniform(0, 1.7, n)
    return np.stack([np.full(n, 2.5) + rng.uniform(-0.05, 0.05, n), y, z], 1)

voxel = 0.15
dyn = bm.DynStats(voxel)
carver = bm.FreeSpaceCarver(voxel, max_range=20.0, ray_stride=1, scan_stride=1)
origin = np.zeros(3)

for i in range(200):
    t = i * 0.1
    pts = wall_points()
    if i < 100:
        pts = np.vstack([pts, person_points()])
    dyn.add(pts, t)
    carver.add(origin, pts)

# span-only decision: person dwelled 10 s >> min_span_s=1.0 -> survives
dyn_keys_span, (ns1, nd1, nc1) = bm.DynStats.dynamic_keys(dyn, 2, 1.0)
# with carving: after the person leaves, rays to the wall pass through x=2.5
dyn_keys_carve, (ns2, nd2, nc2) = bm.DynStats.dynamic_keys(
    dyn, 2, 1.0, carver=carver, min_free=3, free_ratio=0.25)

person_vox = np.unique(bm.pack_voxels(
    np.floor(person_points(2000) / voxel).astype(np.int64)))
wall_vox = np.unique(bm.pack_voxels(
    np.floor(wall_points(4000) / voxel).astype(np.int64)))

surv_span = np.isin(person_vox, dyn_keys_span)
surv_carve = np.isin(person_vox, dyn_keys_carve)
wall_killed = np.isin(wall_vox, dyn_keys_carve)

print(f"span-only : person voxels flagged dynamic {surv_span.mean()*100:.0f}% "
      f"(expected ~0% -> demonstrates the blind spot)")
print(f"with carve: person voxels flagged dynamic {surv_carve.mean()*100:.0f}% "
      f"(expected ~100%)")
print(f"with carve: wall voxels wrongly flagged  {wall_killed.mean()*100:.1f}% "
      f"(expected ~0%)")
print(f"carve stats: {nc2} span-static voxels carved as dynamic")
assert surv_span.mean() < 0.1, "span test should MISS the dweller"
assert surv_carve.mean() > 0.9, "carving should catch the dweller"
assert wall_killed.mean() < 0.05, "carving must not eat the wall"

# drop_dynamic_points needs an o3d point cloud; test the key-membership math
pts = np.vstack([wall_points(1000), person_points(300)])
vox = np.floor(pts / voxel).astype(np.int64)
packed = bm.pack_voxels(vox)
pos = np.clip(np.searchsorted(dyn_keys_carve, packed), 0,
              max(dyn_keys_carve.size - 1, 0))
is_dyn = dyn_keys_carve[pos] == packed
print(f"membership: {is_dyn[:1000].mean()*100:.1f}% of wall pts dropped, "
      f"{is_dyn[1000:].mean()*100:.0f}% of person pts dropped")
assert is_dyn[1000:].mean() > 0.9 and is_dyn[:1000].mean() < 0.05

# ---- mesh_plane: L-shaped wall footprint with a doorway --------------------
from scipy import ndimage  # noqa: F401  (ensures scipy present)
spec2 = importlib.util.spec_from_file_location(
    "mesher", "/home/user/Others/lidar_mapping/02_pcd_to_mesh_sionna_v9.py")
ms = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(ms)

# vertical plane x=0, points spanning y in [0,6], z in [0,3], with a 1.2 m
# doorway gap at y in [2.4, 3.6], z < 2.1
n = 40000
y = rng.uniform(0, 6, n); z = rng.uniform(0, 3, n)
door = (y > 2.4) & (y < 3.6) & (z < 2.1)
y, z = y[~door], z[~door]
Q = np.stack([rng.normal(0, 0.01, len(y)), y, z], 1)
nrm = np.array([1.0, 0.0, 0.0]); d = 0.0
out = ms.mesh_plane(Q, nrm, d, cell=0.10, close_cells=3,
                    min_region_m2=0.5, seal_dilate=1)
assert out is not None
V, F = out
# all vertices exactly on the plane
assert np.abs(V @ nrm + d).max() < 1e-9, "grid mesh must be exactly flat"
# doorway must remain open: no vertex deep inside the gap
in_door = (V[:, 1] > 2.7) & (V[:, 1] < 3.3) & (V[:, 2] > 0.3) & (V[:, 2] < 1.6)
print(f"mesh_plane: {len(V)} verts, {len(F)} tris, "
      f"max off-plane {np.abs(V @ nrm).max():.2e}, "
      f"verts inside doorway: {int(in_door.sum())} (expected 0)")
assert in_door.sum() == 0, "doorway should stay open"
# triangles reference valid vertices
assert F.min() >= 0 and F.max() < len(V)

# plane extraction: two perpendicular walls + floor, unoriented normals
nA = 30000
w1 = np.stack([rng.normal(0, 0.01, nA), rng.uniform(0, 6, nA),
               rng.uniform(0, 3, nA)], 1)
w2 = np.stack([rng.uniform(0, 6, nA), rng.normal(0, 0.01, nA),
               rng.uniform(0, 3, nA)], 1)
fl = np.stack([rng.uniform(0, 6, nA), rng.uniform(0, 6, nA),
               rng.normal(0, 0.01, nA)], 1)
P = np.vstack([w1, w2, fl])
N = np.vstack([np.tile([1., 0, 0], (nA, 1)) * rng.choice([-1, 1], (nA, 1)),
               np.tile([0., 1, 0], (nA, 1)) * rng.choice([-1, 1], (nA, 1)),
               np.tile([0., 0, 1], (nA, 1)) * rng.choice([-1, 1], (nA, 1))])
planes, pid = ms.extract_planes_two_phase(P, N, 0.06, dist=0.08,
                                          ang_deg=35.0)
planes, pid, nm = ms.merge_coplanar(P, planes, pid)
print(f"extract_planes: {len(planes)} planes found (expected 3), "
      f"{(pid >= 0).mean()*100:.0f}% of points assigned")
assert len(planes) == 3
assert (pid >= 0).mean() > 0.95

print("\nALL TESTS PASSED")
