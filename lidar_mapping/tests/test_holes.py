"""Floor holes: occlusion shadows must be floored, the courtyard must not."""
import sys, types
import numpy as np
sys.path.insert(0, "/home/user/Others/lidar_mapping")
sys.modules.setdefault("open3d", types.ModuleType("open3d"))
for a in ("geometry","utility","io","core","t"): setattr(sys.modules["open3d"],a,types.SimpleNamespace())
import importlib.util
spec=importlib.util.spec_from_file_location("ms","/home/user/Others/lidar_mapping/02_pcd_to_mesh_sionna_v9.py")
ms=importlib.util.module_from_spec(spec); spec.loader.exec_module(ms)

rng=np.random.default_rng(5)
CELL=0.10
# a 30 x 20 m floor slab (z=0) with three kinds of gap
n=1_200_000
x=rng.uniform(0,30,n); y=rng.uniform(0,20,n)
courtyard=(x>9)&(x<21)&(y>6)&(y<14)          # 12x8 = 96 m^2 atrium: KEEP OPEN
shadow   =(x>3)&(x<7)&(y>2)&(y<3.5)          # 4x1.5 = 6.0 m^2 pillar shadow
hairline =(x>24)&(x<25)&(y>1)&(y<19)         # 1 m wide seam between passes
drop = courtyard | shadow | hairline
x,y=x[~drop],y[~drop]
Q=np.stack([x,y,np.zeros(len(x))],1)
nrm=np.array([0.,0.,1.]); d=0.0

def occupied(V, px, py):
    """is the meshed surface present at (px,py)?"""
    return bool(((np.abs(V[:,0]-px)<0.12)&(np.abs(V[:,1]-py)<0.12)).any())

for fill, tag in ((0.0, "max_fill_m2=0 (old behaviour)"),
                  (8.0, "max_fill_m2=8.0 (new)")):
    V,F = ms.mesh_plane(Q, nrm, d, cell=CELL, close_cells=3,
                        min_region_m2=0.5, seal_dilate=1, max_fill_m2=fill)
    cy = occupied(V, 15.0, 10.0)      # centre of the courtyard
    sh = occupied(V, 5.0, 2.75)       # centre of the pillar shadow
    hl = occupied(V, 24.5, 10.0)      # centre of the 1 m seam
    fl = occupied(V, 1.0, 10.0)       # ordinary floor
    print(f"{tag}:")
    print(f"    ordinary floor meshed : {fl}   (want True)")
    print(f"    6.0 m^2 shadow filled : {sh}   (want True with fill)")
    print(f"    1 m seam filled       : {hl}")
    print(f"    96 m^2 courtyard open : {not cy}   (want True ALWAYS)")
    assert fl, "floor missing"
    assert not cy, "courtyard was sealed -- area cap failed"
    if fill == 0.0:
        assert not sh, "test scene should leave the shadow open without fill"
    else:
        assert sh, "shadow not filled"

# What protects the courtyard is the AREA CAP, not connectivity: an atrium is
# an enclosed hole like any other, so an unbounded cap fills it. Showing that
# explicitly is the point -- it is why the cap must stay well under the
# smallest open space you intend to keep.
V, F = ms.mesh_plane(Q, nrm, d, cell=CELL, close_cells=3, min_region_m2=0.5,
                     seal_dilate=1, max_fill_m2=1e6)
assert occupied(V, 15.0, 10.0), "unbounded cap should fill even the courtyard"
print("unbounded cap DOES fill the 96 m^2 courtyard -> the cap is the only")
print("               thing separating a shadow from an atrium; keep it well")
print("               below the smallest open space you want to preserve")

# ...but no cap may grow the surface past where points were measured
for cap in (0.0, 8.0, 1e6):
    V, _ = ms.mesh_plane(Q, nrm, d, cell=CELL, close_cells=3,
                         min_region_m2=0.5, seal_dilate=1, max_fill_m2=cap)
    assert V[:, 0].min() > -0.5 and V[:, 0].max() < 30.5, "grew in x"
    assert V[:, 1].min() > -0.5 and V[:, 1].max() < 20.5, "grew in y"
    assert np.abs(V @ nrm + d).max() < 1e-9, "no longer exactly planar"
print("every cap: surface stays within its measured extent and exactly planar")

print("\nALL HOLE-FILL TESTS PASSED")
