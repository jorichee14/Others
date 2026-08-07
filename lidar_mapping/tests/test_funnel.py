"""The funnel must name the stage that actually killed each class."""
import sys, types, io, contextlib
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

names={1:"chair",2:"tv",3:"sofa",4:"plant"}
funnel={
 1:{"voted":5000,"ratio":0.08,"agreed":0,"kept":0,"clusters":0,"small":0,"flat":0},
 2:{"voted":4000,"ratio":0.60,"agreed":3000,"kept":0,"clusters":0,"small":0,"flat":0},
 3:{"voted":900,"ratio":0.50,"agreed":800,"kept":700,"clusters":0,"small":3,"flat":0},
 4:{"voted":2000,"ratio":0.55,"agreed":1500,"kept":1200,"clusters":2,"small":0,"flat":0},
}
buf=io.StringIO()
with contextlib.redirect_stdout(buf):
    bm.print_funnel(funnel, names, 0.35, 120)
out=buf.getvalue()
print(out)
lines={l.split()[0]: l for l in out.splitlines() if l.strip() and "class" not in l}
assert "agreement" in lines["chair"] and "8%" in lines["chair"], lines["chair"]
assert "structural veto" in lines["tv"], lines["tv"]
assert "min_pts_keep" in lines["sofa"], lines["sofa"]
assert lines["plant"].rstrip().endswith("2"), "a class with objects needs no bottleneck"
print("funnel names the right bottleneck for each failure mode")
b=io.StringIO()
with contextlib.redirect_stdout(b): bm.print_funnel({}, names, 0.35, 120)
assert b.getvalue()==""
print("empty funnel prints nothing")
print("\nALL FUNNEL TESTS PASSED")
