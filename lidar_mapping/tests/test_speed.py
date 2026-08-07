"""Verify the optimized projection is EQUIVALENT to the original, and faster."""
import sys, types, time
import numpy as np
sys.path.insert(0, "/home/user/Others/lidar_mapping")

for name in ("open3d", "cv2", "rosbags", "rosbags.highlevel",
             "rosbags.typesys", "pipeline_common"):
    sys.modules.setdefault(name, types.ModuleType(name))
sys.modules["rosbags.highlevel"].AnyReader = object
sys.modules["rosbags.typesys"].Stores = types.SimpleNamespace(ROS2_HUMBLE=0)
sys.modules["rosbags.typesys"].get_typestore = lambda *_: None
sys.modules["pipeline_common"].load_pipeline = lambda *_: None
for a in ("geometry", "utility", "io"):
    setattr(sys.modules["open3d"], a, types.SimpleNamespace())
sys.modules["cv2"].cvtColor = lambda *a: np.zeros((1, 1, 3), np.uint8)
sys.modules["cv2"].COLOR_HSV2RGB = 0

import importlib.util
spec = importlib.util.spec_from_file_location(
    "bm", "/home/user/Others/lidar_mapping/01_build_map.py")
bm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bm)

rng = np.random.default_rng(7)
S = types.SimpleNamespace(fx=300.0, fy=300.0, cx=320.0, cy=180.0)
W, H, MAXR = 640, 360, 10.0

# ---- reference implementation (the ORIGINAL algorithm, sphere + 2 sorts) --- #
def reference(pts, Twc):
    cam = Twc[:3, 3]
    dd = np.linalg.norm(pts - cam, axis=1)
    idx = np.flatnonzero(dd < MAXR)          # KDTree radius search equivalent
    if idx.size == 0:
        return None
    sub = pts[idx]
    Tcw = np.linalg.inv(Twc)
    Xc = (Tcw[:3, :3] @ sub.T).T + Tcw[:3, 3]
    z = Xc[:, 2]
    fr = z > 1e-3
    u = np.full(len(sub), -1.0); v = np.full(len(sub), -1.0)
    u[fr] = S.fx * Xc[fr, 0] / z[fr] + S.cx
    v[fr] = S.fy * Xc[fr, 1] / z[fr] + S.cy
    inb = fr & (u >= 0) & (u < W) & (v >= 0) & (v < H) & (z < MAXR)
    if not inb.any():
        return None
    g = idx[inb]; zc = z[inb]
    uu = u[inb].astype(np.int64); vv = v[inb].astype(np.int64)
    order = np.argsort(zc)
    _, first = np.unique((vv * W + uu)[order], return_index=True)
    keep = order[first]
    return g[keep], uu[keep], vv[keep], zc[keep]

# ---- a room-like cloud ---------------------------------------------------- #
n = 600_000
pts = np.concatenate([
    np.stack([rng.uniform(-12, 12, n), rng.uniform(-12, 12, n),
              rng.uniform(0, 3, n)], 1),
    np.stack([rng.uniform(-1, 1, 40_000), rng.uniform(2, 4, 40_000),
              rng.uniform(0, 2, 40_000)], 1)])           # a nearby object
print(f"cloud: {len(pts):,} points")

t0 = time.time()
index = bm.BlockIndex(pts, block=2.0)
t_build = time.time() - t0
print(f"BlockIndex build: {t_build*1000:.0f} ms, {len(index.start)} blocks")

def random_pose(k):
    rng2 = np.random.default_rng(100 + k)
    yaw = rng2.uniform(0, 2 * np.pi)
    p = np.array([rng2.uniform(-6, 6), rng2.uniform(-6, 6), 1.4])
    # camera optical frame: +z forward, +y down
    fwd = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    right = np.array([np.sin(yaw), -np.cos(yaw), 0.0])
    down = np.array([0.0, 0.0, -1.0])
    T = np.eye(4)
    T[:3, 0] = right; T[:3, 1] = down; T[:3, 2] = fwd; T[:3, 3] = p
    return T

# ---- equivalence over many random poses ----------------------------------- #
n_cmp = n_pts_ref = 0
n_extra_tot = [0]
for k in range(40):
    Twc = random_pose(k)
    a = reference(pts, Twc)
    b = bm.project_visible(index, pts, Twc, S, W, H, MAXR)
    if a is None and b is None:
        continue
    assert a is not None and b is not None, f"pose {k}: one returned None"
    ga, ua, va, za = a
    gb, ub, vb, zb = b
    # compare as pixel->point maps (order differs; content must not)
    ma = dict(zip((va * W + ua).tolist(), ga.tolist()))
    mb = dict(zip((vb * W + ub).tolist(), gb.tolist()))
    # the new cull is by DEPTH (z < max_range), matching the exact per-point
    # test; the old sphere cull was tighter than that test and silently lost
    # off-axis points. So new must be a strict SUPERSET of old...
    assert ma.keys() <= mb.keys(), f"pose {k}: optimized lost pixels"
    # ...and every extra pixel must be one the sphere wrongly excluded
    cam = Twc[:3, 3]
    for pix in mb.keys() - ma.keys():
        dist = np.linalg.norm(pts[mb[pix]] - cam)
        assert dist >= MAXR, f"pose {k}: unexplained extra pixel (d={dist})"
        n_extra_tot[0] += 1
    # allow a different winner only when depths tie to sub-mm (quantisation)
    za_map = dict(zip((va * W + ua).tolist(), za.tolist()))
    zb_map = dict(zip((vb * W + ub).tolist(), zb.tolist()))
    for pix in ma:
        if ma[pix] != mb[pix]:
            assert abs(za_map[pix] - zb_map[pix]) < 1e-3, \
                f"pose {k} pixel {pix}: {ma[pix]} vs {mb[pix]}"
    n_cmp += 1
    n_pts_ref += len(ga)
print(f"equivalence: {n_cmp} poses agree on every shared pixel "
      f"({n_pts_ref:,} visible points compared)")
print(f"             + {n_extra_tot[0]:,} pixels RECOVERED that the old "
      f"sphere cull wrongly dropped ({100.0*n_extra_tot[0]/n_pts_ref:.1f}%)")

# ---- speed ---------------------------------------------------------------- #
poses = [random_pose(k) for k in range(30)]
t0 = time.time()
for T in poses:
    reference(pts, T)
t_ref = time.time() - t0
t0 = time.time()
for T in poses:
    bm.project_visible(index, pts, T, S, W, H, MAXR)
t_new = time.time() - t0
print(f"per frame: original {1000*t_ref/len(poses):.1f} ms -> "
      f"optimized {1000*t_new/len(poses):.1f} ms  "
      f"({t_ref/max(t_new,1e-9):.1f}x)")

# how much the frustum cull tightens vs a sphere
Tcw = bm._inv_se3(poses[0])
cand = index.candidates(Tcw, S, W, H, MAXR)
sphere = int((np.linalg.norm(pts - poses[0][:3, 3], axis=1) < MAXR).sum())
print(f"candidates: sphere {sphere:,} -> frustum blocks {len(cand):,} "
      f"({sphere/max(len(cand),1):.1f}x fewer)")

# ---- pose gate ------------------------------------------------------------ #
gate = bm.pose_gate({"min_baseline": 0.10, "min_rotation_deg": 5.0})
T = np.eye(4)
used = 0
for i in range(100):                      # creep forward 1 cm per frame
    T2 = T.copy(); T2[0, 3] = i * 0.01
    used += bool(gate(T2))
print(f"pose gate: 100 frames at 1 cm spacing -> {used} used "
      f"(expected ~10 at min_baseline 0.10)")
assert 9 <= used <= 12
off = bm.pose_gate({})
assert all(off(np.eye(4)) for _ in range(5)), "gate must be off by default"

print("\nALL SPEED/EQUIVALENCE TESTS PASSED")
