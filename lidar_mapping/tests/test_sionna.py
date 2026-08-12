"""Sionna scene emission: the XML must satisfy Sionna's loading contract."""
import sys, types, os, tempfile, xml.etree.ElementTree as ET
sys.path.insert(0, "/home/user/Others/lidar_mapping")
o3d = types.ModuleType("open3d"); written = {}
o3d.io = types.SimpleNamespace(write_triangle_mesh=lambda p, m: written.setdefault(p, m))
o3d.geometry = types.SimpleNamespace()
sys.modules["open3d"] = o3d
import importlib.util
spec = importlib.util.spec_from_file_location("prep", "/home/user/Others/lidar_mapping/04_prep_sionna.py")
prep = importlib.util.module_from_spec(spec); spec.loader.exec_module(prep)

tmp = tempfile.mkdtemp()
root = ET.fromstring(open(prep.write_scene(
    {n: object() for n in ("walls","floor","ceiling","objects")}, tmp)).read())
assert root.tag == "scene" and root.get("version") == "2.1.0"
bsdfs = {b.get("id") for b in root.findall("bsdf")}
shapes = root.findall("shape")
assert len(shapes) == 4
for s in shapes:
    ref = s.find("ref").get("id")
    assert ref in bsdfs, f"dangling bsdf ref {ref}"
    assert ref.startswith("mat-itu_"), f"Sionna needs mat-<itu_*>: {ref}"
    assert s.get("id").startswith("mesh-")
    assert s.find("boolean[@name='face_normals']").get("value") == "true"
    assert s.find("string[@name='filename']").get("value").startswith("meshes/")
print(f"4 parts -> {len(bsdfs)} ITU materials, all refs resolve, "
      f"face_normals on, relative mesh paths")

# a single unsplit mesh must still emit a valid one-material scene
root2 = ET.fromstring(open(prep.write_scene({"mesh": object()}, tmp)).read())
assert len(root2.findall("shape")) == 1
assert root2.find("shape/ref").get("id") == "mat-itu_concrete"
print("single-mesh fallback emits a valid one-material scene")

# every configured material must be a real Sionna ITU name
SIONNA_ITU = {"itu_concrete","itu_brick","itu_plasterboard","itu_wood",
              "itu_glass","itu_ceiling_board","itu_chipboard","itu_floorboard",
              "itu_metal","itu_very_dry_ground","itu_medium_dry_ground",
              "itu_wet_ground","itu_marble"}
for part,(mat,rgb) in prep.MATERIALS.items():
    assert mat in SIONNA_ITU, f"{part} -> {mat} is not a Sionna ITU material"
    assert all(0.0 <= c <= 1.0 for c in rgb), f"{part} rgb out of range"
print(f"all {len(prep.MATERIALS)} configured materials are valid ITU names")
print("\nALL SIONNA PREP TESTS PASSED")
