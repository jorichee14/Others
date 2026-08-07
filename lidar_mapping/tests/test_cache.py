"""Detection-cache round trip: what detect_cache.py writes is exactly what
01_build_map.py reads back, including the empty frames that carry the
denominator of the multi-view agreement test."""
import sys, types, json, os, tempfile
import numpy as np

sys.path.insert(0, "/home/user/Others/lidar_mapping")
import cv2  # real opencv needed for PNG round trip

for name in ("open3d", "rosbags", "rosbags.highlevel", "rosbags.typesys",
             "pipeline_common"):
    sys.modules.setdefault(name, types.ModuleType(name))
sys.modules["rosbags.highlevel"].AnyReader = object
sys.modules["rosbags.typesys"].Stores = types.SimpleNamespace(ROS2_HUMBLE=0)
sys.modules["rosbags.typesys"].get_typestore = lambda *_: None
sys.modules["pipeline_common"].load_pipeline = lambda *_: None
for a in ("geometry", "utility", "io", "core", "t"):
    setattr(sys.modules["open3d"], a, types.SimpleNamespace())

import importlib.util
spec = importlib.util.spec_from_file_location(
    "bm", "/home/user/Others/lidar_mapping/01_build_map.py")
bm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bm)
import yolo_labels

rng = np.random.default_rng(4)
W, H = 640, 360
tmp = tempfile.mkdtemp()
cache = os.path.join(tmp, "detections")
os.makedirs(cache)

# ---- write a cache exactly the way detect_cache.py does ------------------- #
truth = []
entries = []
for i in range(6):
    t = 100.0 + i * 0.5
    if i in (1, 4):                        # frames with NO detections
        entries.append({"n": i, "t": t, "insts": []})
        truth.append((t, None, []))
        continue
    lab = np.full((H, W), -1, np.int16)
    insts = []
    for k in range(rng.integers(1, 4)):
        x0, y0 = rng.integers(0, 400), rng.integers(0, 200)
        lab[y0:y0 + 80, x0:x0 + 100] = k
        insts.append((int(rng.integers(0, 79)), float(rng.uniform(.4, .95))))
    fn = f"{i:07d}.png"
    cv2.imwrite(os.path.join(cache, fn),
                np.clip(lab + 1, 0, 255).astype(np.uint8))
    entries.append({"n": i, "t": t,
                    "insts": [[c, round(cf, 4)] for c, cf in insts],
                    "png": fn})
    truth.append((t, lab, insts))

json.dump({"model": "yolo11n-seg.pt", "device": "cuda:0", "topic": "/img",
           "width": W, "height": H, "img_stride": 5, "conf": 0.35,
           "names": {str(i): f"class{i}" for i in range(80)},
           "frames": entries},
          open(os.path.join(cache, "index.json"), "w"))

# ---- read it back through 01's loader ------------------------------------- #
P = types.SimpleNamespace(outp=lambda n: os.path.join(tmp, n))
s = {"image_width": W, "image_height": H}
d = {"cache": "detections"}
gen, names = bm.cached_frames(P, s, d)
assert gen is not None, "cache should have been found"
got = list(gen)

assert len(got) == len(truth), f"{len(got)} frames back, expected {len(truth)}"
n_empty = 0
for (t_g, lab_g, ins_g), (t_t, lab_t, ins_t) in zip(got, truth):
    assert abs(t_g - t_t) < 1e-9, "timestamp drift"
    assert len(ins_g) == len(ins_t)
    for (cg, fg), (ct, ft) in zip(ins_g, ins_t):
        assert cg == ct and abs(fg - ft) < 1e-3
    if lab_t is None:
        assert lab_g is None and ins_g == []
        n_empty += 1
    else:
        assert lab_g.dtype == np.int16, f"label dtype {lab_g.dtype}"
        assert np.array_equal(lab_g, lab_t), "label image changed through PNG"
print(f"round trip: {len(got)} frames identical "
      f"(labels bit-exact through PNG, {n_empty} empty frames preserved)")
assert n_empty == 2, "empty frames must survive - they are the vote denominator"
assert names[0] == "class0" and len(names) == 80

# ---- guards ---------------------------------------------------------------- #
try:    # must fail fast at load, not part-way through the run
    bm.cached_frames(P, {"image_width": 320, "image_height": 180}, d)
    raise AssertionError("size mismatch should have raised")
except SystemExit as e:
    print(f"geometry guard: {str(e).splitlines()[0][:72]}...")

miss, _ = bm.cached_frames(P, s, {"cache": "nope"})
assert miss is None, "absent cache must report None so 'auto' can fall back"
try:
    bm.detection_frames(P, None, s, {"cache": "nope", "source": "cache"})
    raise AssertionError("source=cache with no cache should exit")
except SystemExit as e:
    assert "detect_cache.py" in str(e)
    print("source='cache' with no cache: clear build instruction given")

# ---- the shared module must stay torch-free ------------------------------- #
import ast, inspect
tree = ast.parse(inspect.getsource(yolo_labels))
top_imports = set()
for node in tree.body:                     # module level only, not nested
    if isinstance(node, ast.Import):
        top_imports.update(a.name.split(".")[0] for a in node.names)
    elif isinstance(node, ast.ImportFrom) and node.module:
        top_imports.add(node.module.split(".")[0])
assert top_imports == {"numpy", "cv2"}, f"yolo_labels imports {top_imports}"
assert "ultralytics" not in sys.modules, "importing yolo_labels pulled in torch"
# and the ultralytics import must be nested inside load_model
fn = next(n for n in tree.body
          if isinstance(n, ast.FunctionDef) and n.name == "load_model")
assert any(isinstance(x, ast.ImportFrom) and x.module == "ultralytics"
           for x in ast.walk(fn)), "ultralytics import must be INSIDE load_model"
print(f"yolo_labels: top-level imports are exactly {sorted(top_imports)}; "
      f"ultralytics is lazy")

print("\nALL CACHE TESTS PASSED")
