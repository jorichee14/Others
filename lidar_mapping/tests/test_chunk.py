"""Chunked projection must equal unchunked, and a CPU fallback under an active
GPU backend must not mix array modules."""
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

rng=np.random.default_rng(9)
S=types.SimpleNamespace(fx=300.,fy=300.,cx=320.,cy=180.); W,H,R_=640,360,10.
pts=np.concatenate([
    np.stack([rng.uniform(-10,10,300_000),rng.uniform(-10,10,300_000),rng.uniform(0,3,300_000)],1),
    np.stack([rng.uniform(-1,1,30_000),rng.uniform(2,4,30_000),rng.uniform(0,2,30_000)],1)])
def pose(k):
    r=np.random.default_rng(300+k); y=r.uniform(0,2*np.pi); T=np.eye(4)
    T[:3,0]=[np.sin(y),-np.cos(y),0]; T[:3,1]=[0,0,-1]; T[:3,2]=[np.cos(y),np.sin(y),0]
    T[:3,3]=[r.uniform(-5,5),r.uniform(-5,5),1.4]; return T

# force several chunks by shrinking the slice size
orig = bm.projection_chunk
n_calls = [0]
def small(m, n, bytes_per_point=44):
    n_calls[0] += 1
    return 40_000                      # ~9 chunks over 330k points
bm.projection_chunk = small
ok=0
for k in range(20):
    T=pose(k)
    a=bm.project_visible(None, pts, T, S, W, H, R_)       # chunked
    bm.projection_chunk = lambda m,n,bytes_per_point=44: n
    b=bm.project_visible(None, pts, T, S, W, H, R_)       # single shot
    bm.projection_chunk = small
    if a is None and b is None: continue
    assert a is not None and b is not None, f"pose {k}"
    ma=dict(zip((a[2]*W+a[1]).tolist(), a[0].tolist()))
    mb=dict(zip((b[2]*W+b[1]).tolist(), b[0].tolist()))
    assert ma==mb, f"pose {k}: chunking changed the result"
    ok+=1
bm.projection_chunk = orig
print(f"chunked projection == single shot on {ok} poses ({n_calls[0]} chunked calls)")

# frustum (CPU index) path must still agree
index=bm.BlockIndex(pts, block=2.0)
for k in range(10):
    T=pose(k)
    a=bm.project_visible(index, pts, T, S, W, H, R_)
    b=bm.project_visible(None, pts, T, S, W, H, R_)
    if a is None or b is None: continue
    ma=dict(zip((a[2]*W+a[1]).tolist(), a[0].tolist()))
    mb=dict(zip((b[2]*W+b[1]).tolist(), b[0].tolist()))
    assert ma==mb
print("frustum path still agrees with whole-cloud path")

# mod_of / gpu_fits plumbing
assert bm.mod_of(np.zeros(3)) is np
assert bm.gpu_fits(10**6) is False, "no GPU here -> must not claim it fits"
assert bm.gpu_free_bytes() == 0
assert bm.projection_chunk(np, 12345) == 12345, "CPU must not chunk"
print("mod_of/gpu_fits/projection_chunk behave on the CPU backend")
print("\nALL CHUNK TESTS PASSED")
