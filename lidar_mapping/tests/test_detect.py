"""Synthetic tests for the detection/inventory stage helpers (no YOLO needed)."""
import sys, types, io
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
# minimal cv2 stubs used by palette
def _cvt(hsv, code):
    h = hsv[0, :, 0].astype(np.float64) / 179.0
    i = (h * 6).astype(int) % 6
    f = h * 6 - (h * 6).astype(int)
    v = np.full(len(h), 245.0); s = 200.0 / 255.0
    p = v * (1 - s); q = v * (1 - f * s); t = v * (1 - (1 - f) * s)
    out = np.zeros((len(h), 3))
    for k in range(len(h)):
        out[k] = [(v[k], t[k], p[k]), (q[k], v[k], p[k]), (p[k], v[k], t[k]),
                  (p[k], q[k], v[k]), (t[k], p[k], v[k]),
                  (v[k], p[k], q[k])][i[k]]
    return out.astype(np.uint8)[None]
sys.modules["cv2"].cvtColor = _cvt
sys.modules["cv2"].COLOR_HSV2RGB = 0

import importlib.util
spec = importlib.util.spec_from_file_location(
    "bm", "/home/user/Others/lidar_mapping/01_build_map.py")
bm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bm)

rng = np.random.default_rng(1)

# ---- project_visible: occlusion must be respected ------------------------- #
S = types.SimpleNamespace(fx=300.0, fy=300.0, cx=320.0, cy=180.0)
# near point and far point on the SAME ray -> only the near one may survive
pts = np.array([[0.0, 0.0, 2.0], [0.0, 0.0, 6.0], [0.5, 0.2, 3.0]])
Twc = np.eye(4)
g, uu, vv, z = bm.project_visible(bm.BlockIndex(pts, block=2.0), pts, Twc,
                                  S, 640, 360, 10.0)
same_pixel = [i for i in range(len(g)) if (uu[i], vv[i]) == (320, 180)]
print(f"project_visible: {len(g)} winners; occluded far point present: "
      f"{1 in g} (expected False)")
assert 1 not in g, "z-buffer must drop the occluded point"
assert 0 in g and 2 in g
assert np.isclose(z[list(g).index(0)], 2.0)

# ---- _footprint: yaw + extents of a rotated rectangle ---------------------- #
L, Wd = 2.0, 0.6
q = np.stack([rng.uniform(-L / 2, L / 2, 4000),
              rng.uniform(-Wd / 2, Wd / 2, 4000),
              rng.uniform(0, 1.0, 4000)], 1)
th = np.deg2rad(35.0)
R = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0],
              [0, 0, 1]])
qr = q @ R.T + np.array([3.0, -2.0, 0.5])
yaw, ext = bm._footprint(qr)
print(f"_footprint: yaw {yaw:.1f} deg (expected 35), "
      f"extent {ext.round(2)} (expected ~[2.0 0.6])")
assert abs(((yaw - 35 + 90) % 180) - 90) < 2.0
assert abs(ext[0] - L) < 0.1 and abs(ext[1] - Wd) < 0.1

# ---- _instance_stats ------------------------------------------------------ #
st = bm._instance_stats(qr, np.tile([0.2, 0.4, 0.6], (len(qr), 1)),
                        "chair", 56, np.full(len(qr), 8),
                        np.full(len(qr), 10), floor_z=0.0, det_conf=0.72)
print(f"_instance_stats: size {st['size']}, agreement {st['view_agreement']}, "
      f"height_above_floor {st['height_above_floor']}")
assert st["view_agreement"] == 0.8
assert abs(st["footprint"]["height"] - 1.0) < 0.05
assert st["label"] == "chair" and st["n_points"] == len(qr)
assert st["mean_rgb"] == [0.2, 0.4, 0.6]

# ---- vote fusion logic (the multi-view agreement rule) -------------------- #
N = 6
n_seen = np.array([10, 10, 10, 10, 2, 10], np.int32)
votes = {56: np.array([8, 1, 0, 5, 2, 0], np.int32),
         60: np.array([0, 0, 9, 4, 0, 1], np.int32)}
best_v = np.zeros(N, np.int32); best_c = np.full(N, -1, np.int32)
for c, v in votes.items():
    m = v > best_v
    best_v[m] = v[m]; best_c[m] = c
conf = (best_c >= 0) & (best_v >= 3) & (best_v >= 0.35 * np.maximum(n_seen, 1))
print(f"vote fusion: confident={conf.astype(int)} classes={best_c}")
assert conf.tolist() == [True, False, True, True, False, False], conf
#   pt0 chair 8/10 keep | pt1 1 vote -> noise | pt2 table 9/10 keep
#   pt3 chair 5/10 wins tie-break by count | pt4 2 votes < min_votes
#   pt5 1 vote -> dropped

# ---- palette + YAML emitter ----------------------------------------------- #
pal = bm.instance_palette(5)
assert pal.shape == (5, 3) and pal.min() >= 0 and pal.max() <= 1
buf = io.StringIO()
bm._yaml_emit({"map": {"frame": "map", "floor_z": -1.47,
                       "anchor_shift": [0.0, 0.0, 0.0]},
               "totals": {"chair": 6, "table": 2},
               "objects": [{"id": 1, "label": "chair",
                            "centroid": [1.0, 2.0, 0.5],
                            "footprint": {"yaw_deg": 35.0, "length": 2.0}}]},
              buf)
out = buf.getvalue()
print("\n--- inventory YAML sample ---\n" + out + "----------------------------")
try:
    import yaml
    parsed = yaml.safe_load(out)
    assert parsed["objects"][0]["footprint"]["yaw_deg"] == 35.0
    assert parsed["totals"]["chair"] == 6
    assert parsed["map"]["anchor_shift"] == [0.0, 0.0, 0.0]
    print("fallback YAML parses correctly with pyyaml")
except ImportError:
    print("pyyaml absent - emitter output not cross-checked")

print("\nALL DETECTION TESTS PASSED")
