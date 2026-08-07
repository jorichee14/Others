"""Tests for the structural veto: protrusion math + per-point/per-cluster gates."""
import sys, types
import numpy as np

for name in ("open3d", "cv2", "rosbags", "rosbags.highlevel",
             "rosbags.typesys", "pipeline_common"):
    sys.modules.setdefault(name, types.ModuleType(name))
sys.modules["rosbags.highlevel"].AnyReader = object
sys.modules["rosbags.typesys"].Stores = types.SimpleNamespace(ROS2_HUMBLE=0)
sys.modules["rosbags.typesys"].get_typestore = lambda *_: None
sys.modules["pipeline_common"].load_pipeline = lambda *_: None
for a in ("geometry", "utility", "io"):
    setattr(sys.modules["open3d"], a, types.SimpleNamespace())

import importlib.util
spec = importlib.util.spec_from_file_location(
    "bm", "/home/user/Others/lidar_mapping/01_build_map.py")
bm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bm)

rng = np.random.default_rng(3)

# Scene: wall plane x = 0 (normal +x, d = 0), floor plane z = 0.
wall = (np.array([1.0, 0.0, 0.0]), 0.0)
floor = (np.array([0.0, 0.0, 1.0]), 0.0)
planes = [wall, floor]

def on_wall(n, spread=0.01, x0=0.0):
    """Points on/near the wall at standoff x0."""
    return np.stack([rng.normal(x0, spread, n),
                     rng.uniform(0, 4, n), rng.uniform(0, 2.5, n)], 1)

# --- min_plane_distance is the protrusion measure ------------------------- #
p_wall = on_wall(2000)                 # flush wall points
p_tv = on_wall(2000, 0.01, 0.07)       # a real TV: 7 cm off the wall
p_chair = np.stack([rng.uniform(1.0, 1.5, 2000), rng.uniform(0, 4, 2000),
                    rng.uniform(0.4, 0.9, 2000)], 1)   # free-standing

d_wall = bm.min_plane_distance(p_wall, planes)
d_tv = bm.min_plane_distance(p_tv, planes)
d_chair = bm.min_plane_distance(p_chair, planes)
print(f"protrusion  wall {np.median(d_wall):.3f} m | "
      f"tv {np.median(d_tv):.3f} m | chair {np.median(d_chair):.3f} m")
assert np.median(d_wall) < 0.02
assert 0.06 < np.median(d_tv) < 0.08
assert np.median(d_chair) > 0.3
# nearest-plane semantics: a point near the floor must measure against FLOOR
low = np.array([[2.0, 1.0, 0.02]])
assert abs(bm.min_plane_distance(low, planes)[0] - 0.02) < 1e-9
assert np.isinf(bm.min_plane_distance(low, [])[0]), "no planes -> inf"

# --- per-point veto: trims a wall halo, keeps the object ------------------ #
flush_tol = 0.05
# a "tv" detection that also voted a ring of surrounding wall
pts = np.vstack([p_tv, on_wall(3000)])
confident = np.ones(len(pts), bool)
protr = bm.min_plane_distance(pts, planes)
confident &= ~(protr < flush_tol)
kept_tv = confident[:len(p_tv)].mean()
kept_halo = confident[len(p_tv):].mean()
print(f"per-point veto: TV points kept {kept_tv:.0%}, "
      f"wall-halo points kept {kept_halo:.0%}")
assert kept_tv > 0.95 and kept_halo < 0.05

# --- per-cluster veto: whole wall patch detected as an object ------------- #
min_protrusion = 0.04
for name, cluster, expect_keep in (
        ("wall patch as 'tv'", on_wall(4000), False),
        ("real flat TV",       p_tv,          True),
        ("free-standing chair", p_chair,      True)):
    pr = bm.min_plane_distance(cluster, planes)
    keep = not (np.isfinite(pr).any() and float(np.median(pr)) < min_protrusion)
    print(f"cluster veto: {name:22s} median {np.median(pr):.3f} m -> "
          f"{'KEEP' if keep else 'REJECT'}")
    assert keep == expect_keep

# noise must not rescue a flat cluster: heavy tail, median still ~0
noisy = on_wall(4000, spread=0.06)
pr = bm.min_plane_distance(noisy, planes)
frac_above = (pr > flush_tol).mean()
keep = not (float(np.median(pr)) < min_protrusion)
print(f"cluster veto: noisy wall ({frac_above:.0%} of pts above flush_tol) "
      f"median {np.median(pr):.3f} -> {'KEEP' if keep else 'REJECT'}")
assert not keep, "median must resist a noisy tail"

# --- protrusion lands in the inventory ----------------------------------- #
st = bm._instance_stats(p_tv, None, "tv", 62, np.full(len(p_tv), 9),
                        np.full(len(p_tv), 10), floor_z=0.0, det_conf=0.8,
                        protr=bm.min_plane_distance(p_tv, planes))
print(f"inventory field: protrusion = {st['protrusion']} m")
assert 0.06 < st["protrusion"] < 0.08
st_noplanes = bm._instance_stats(p_tv, None, "tv", 62, np.full(len(p_tv), 9),
                                 np.full(len(p_tv), 10), 0.0, 0.8,
                                 protr=np.full(len(p_tv), np.inf))
assert "protrusion" not in st_noplanes, "no planes -> field omitted, not inf"

print("\nALL VETO TESTS PASSED")
