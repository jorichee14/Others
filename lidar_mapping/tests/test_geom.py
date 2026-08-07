"""The detection-free variant must be self-contained and behave identically
to the full script on every shared code path."""
import sys, types, ast, inspect
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
def load(tag, path):
    sp=importlib.util.spec_from_file_location(tag, path)
    m=importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m
full=load("full","/home/user/Others/lidar_mapping/01_build_map.py")
geom=load("geom","/home/user/Others/lidar_mapping/01_build_map_geometry.py")

# ---- self-contained: no yolo_labels, no torch --------------------------------
tree=ast.parse(open("/home/user/Others/lidar_mapping/01_build_map_geometry.py").read())
tops=set()
for nd in tree.body:
    if isinstance(nd, ast.Import): tops.update(a.name.split(".")[0] for a in nd.names)
    elif isinstance(nd, ast.ImportFrom) and nd.module: tops.add(nd.module.split(".")[0])
assert "yolo_labels" not in tops, "must not need yolo_labels.py"
assert "ultralytics" not in tops and "torch" not in tops
print(f"self-contained: top-level imports {sorted(tops)}")
assert hasattr(geom, "decode_img"), "decode_img must be inlined"

# ---- detection API is gone ---------------------------------------------------
for gone in ("detect_objects","save_object_layers","print_funnel",
             "fit_structure_planes","instance_palette","detection_frames"):
    assert not hasattr(geom, gone), f"{gone} should be removed"
print("detection API removed (6 symbols)")

# ---- geometry API is intact --------------------------------------------------
keep = ("merge","remove_dynamic","denoise","colorize","flatten","save","main",
        "VoxelAccumulator","DynStats","FreeSpaceCarver","BlockIndex",
        "project_visible","projection_chunk","group_bounds","group_sum",
        "init_gpu","pose_gate","pack_voxels","unpack_voxels","drop_dynamic_points")
for k in keep:
    assert hasattr(geom, k), f"{k} missing from the geometry variant"
print(f"geometry API intact ({len(keep)} symbols)")

# ---- identical behaviour on shared paths -------------------------------------
rng=np.random.default_rng(12)
S=types.SimpleNamespace(fx=300.,fy=300.,cx=320.,cy=180.); W,H,R_=640,360,10.
pts=np.stack([rng.uniform(-8,8,200_000),rng.uniform(-8,8,200_000),
              rng.uniform(0,3,200_000)],1)
def pose(k):
    r=np.random.default_rng(50+k); y=r.uniform(0,2*np.pi); T=np.eye(4)
    T[:3,0]=[np.sin(y),-np.cos(y),0]; T[:3,1]=[0,0,-1]; T[:3,2]=[np.cos(y),np.sin(y),0]
    T[:3,3]=[r.uniform(-4,4),r.uniform(-4,4),1.4]; return T
ok=0
for k in range(15):
    T=pose(k)
    a=full.project_visible(full.BlockIndex(pts,2.0) if k==0 else IDXF, pts,T,S,W,H,R_) if k else None
    if k==0:
        IDXF=full.BlockIndex(pts,2.0); IDXG=geom.BlockIndex(pts,2.0)
        a=full.project_visible(IDXF,pts,T,S,W,H,R_)
    b=geom.project_visible(IDXG,pts,T,S,W,H,R_)
    if a is None and b is None: continue
    assert a is not None and b is not None
    assert np.array_equal(a[0],b[0]) and np.array_equal(a[3],b[3]), f"pose {k}"
    ok+=1
print(f"project_visible identical on {ok} poses")

# accumulator + dynamic filter must agree exactly
for mod,tag in ((full,"full"),(geom,"geom")):
    acc=mod.VoxelAccumulator(0.05,centroid=True,flush_pts=30_000)
    for a in range(0,len(pts),9000): acc.add(pts[a:a+9000])
    globals()[f"pts_{tag}"]=acc.points()
assert np.array_equal(pts_full,pts_geom), "VoxelAccumulator diverged"
print(f"VoxelAccumulator identical ({len(pts_full):,} voxels)")

def dyn(mod):
    d=mod.DynStats(0.15); r=np.random.default_rng(7)
    for i in range(40):
        w=np.stack([np.full(400,5.),r.uniform(-3,3,400),r.uniform(0,3,400)],1)
        p=w if i>=15 else np.vstack([w,np.stack([np.full(60,2.5),
            r.uniform(-.2,.2,60),r.uniform(0,1.7,60)],1)])
        d.add(p,i*0.1)
    return d.dynamic_keys(2,2.0)
k1,s1=dyn(full); k2,s2=dyn(geom)
assert np.array_equal(k1,k2) and s1==s2, "DynStats diverged"
print(f"DynStats identical ({s1[0]} static / {s1[1]} dynamic voxels)")

print("\nALL GEOMETRY-VARIANT TESTS PASSED")
