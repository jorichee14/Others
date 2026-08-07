"""Carver invariants. Deliberately NOT tuned percentages -- the amount of
floor lost depends on scan density and scene geometry, which a synthetic
scene does not reproduce faithfully. These are properties that must hold
regardless of scene."""
import sys, types
import numpy as np
sys.path.insert(0, "/home/user/Others/lidar_mapping")
for n in ("open3d","cv2","rosbags","rosbags.highlevel","rosbags.typesys","pipeline_common"):
    sys.modules.setdefault(n, types.ModuleType(n))
sys.modules["rosbags.highlevel"].AnyReader=object
sys.modules["rosbags.typesys"].Stores=types.SimpleNamespace(ROS2_HUMBLE=0)
sys.modules["rosbags.typesys"].get_typestore=lambda *_: None
sys.modules["pipeline_common"].load_pipeline=lambda *_: None
for a in ("geometry","utility","io","core","t"): setattr(sys.modules["open3d"],a,types.SimpleNamespace())
sys.modules["cv2"].cvtColor=lambda *a: np.zeros((1,1,3),np.uint8); sys.modules["cv2"].COLOR_HSV2RGB=0
import importlib.util
spec=importlib.util.spec_from_file_location("bm","/home/user/Others/lidar_mapping/01_build_map.py")
bm=importlib.util.module_from_spec(spec); spec.loader.exec_module(bm)
VOX=0.15
rng=np.random.default_rng(21)

# ---- 1. a scan must never carve a voxel it measured a return in ----------- #
# This is the grazing fix: a ray hitting the floor 14 m out runs within a
# voxel of it for the last metres, over voxels its sibling rays terminate in.
car=bm.FreeSpaceCarver(VOX, max_range=20.0, ray_stride=1, scan_stride=1)
N=20000
r=rng.uniform(2,14,N); th=rng.uniform(-0.6,0.6,N)
floor=np.stack([r*np.cos(th), r*np.sin(th), np.zeros(N)],1)   # grazing floor
origin=np.array([0.,0.,1.0])
car.add(origin, floor)
car._compact()
occ=np.unique(bm.pack_voxels(np.floor(floor/VOX).astype(np.int64)))
carved=car.keys
overlap=np.intersect1d(occ, carved)
print(f"same-scan guard: {len(occ):,} measured voxels, {len(carved):,} carved, "
      f"{len(overlap)} overlap (must be 0)")
assert len(overlap)==0, "a scan carved a voxel it measured -- surface self-carving"

# ...and it must still carve genuinely empty space in front of the surface
mid=np.array([[3.0,0.0,0.55]])          # well above the floor, on a ray path
midk=bm.pack_voxels(np.floor(mid/VOX).astype(np.int64))
assert car.counts_for(midk)[0] > 0, "free space above the floor must carve"
print("             ...empty space above the surface still carves")

# ---- 2. free_ratio must be monotone: higher = strictly more conservative -- #
def run(free_ratio):
    dyn=bm.DynStats(VOX)
    c=bm.FreeSpaceCarver(VOX, max_range=15.0, ray_stride=2, scan_stride=1)
    r2=np.random.default_rng(3)
    for i in range(40):
        rr=r2.uniform(2,12,6000); tt=r2.uniform(-0.6,0.6,6000)
        f=np.stack([rr*np.cos(tt), rr*np.sin(tt), np.zeros(6000)],1)
        p=f if i>=15 else np.vstack([f, np.stack(
            [np.full(200,5.)+r2.uniform(-.1,.1,200), r2.uniform(-.2,.2,200),
             r2.uniform(.3,1.7,200)],1)])
        dyn.add(p, i*0.2); c.add(np.array([0.,0.,1.]), p)
    k,_=dyn.dynamic_keys(2, 1.0, carver=c, min_free=5, free_ratio=free_ratio)
    return set(k.tolist())
prev=None
counts=[]
for fr in (0.25, 0.5, 1.0, 2.0, 4.0):
    cur=run(fr); counts.append((fr,len(cur)))
    if prev is not None:
        assert cur <= prev, f"free_ratio {fr} deleted voxels a lower one kept"
    prev=cur
print("free_ratio monotonicity:", ", ".join(f"{fr}->{n}" for fr,n in counts))
print("             (higher free_ratio is always a subset -> safe to raise)")

# ---- 3. disabling carving must reproduce the span-only verdict exactly ---- #
def span_only():
    dyn=bm.DynStats(VOX)
    r2=np.random.default_rng(3)
    for i in range(40):
        rr=r2.uniform(2,12,6000); tt=r2.uniform(-0.6,0.6,6000)
        f=np.stack([rr*np.cos(tt), rr*np.sin(tt), np.zeros(6000)],1)
        p=f if i>=15 else np.vstack([f, np.stack(
            [np.full(200,5.)+r2.uniform(-.1,.1,200), r2.uniform(-.2,.2,200),
             r2.uniform(.3,1.7,200)],1)])
        dyn.add(p, i*0.2)
    k,st=dyn.dynamic_keys(2, 1.0, carver=None)
    return set(k.tolist()), st
so,st=span_only()
assert so <= run(0.25), "carving must only ADD to the span-only dynamic set"
print(f"carve off: {len(so)} dynamic voxels from the span test alone "
      f"({st[0]} static) -- a strict subset of any carved result")
print("             -> carve.enable false is a safe A/B to isolate the cause")

print("\nALL CARVER INVARIANT TESTS PASSED")
