"""One sheet per wall + close_floor mechanics."""
import sys, types
import numpy as np
sys.path.insert(0, "/home/user/Others/lidar_mapping")
sys.modules.setdefault("open3d", types.ModuleType("open3d"))
for a in ("geometry","utility","io","core","t"):
    setattr(sys.modules["open3d"], a, types.SimpleNamespace())
import importlib.util
spec=importlib.util.spec_from_file_location("ms","/home/user/Others/lidar_mapping/02_pcd_to_mesh_sionna_v9.py")
ms=importlib.util.module_from_spec(spec); spec.loader.exec_module(ms)

rng=np.random.default_rng(8)
def sheet(x, ylo, yhi, n=6000, jitter=0.008):
    return np.stack([np.full(n,x)+rng.normal(0,jitter,n),
                     rng.uniform(ylo,yhi,n), rng.uniform(0,3,n)],1)

# the seven planes a drift-y double-sided scan produces
parts = [
    (sheet(0.00, 0, 6), ( 1,0,0),  0.00),   # 0 wall face A
    (sheet(0.20, 0, 6), (-1,0,0),  0.20),   # 1 face B (opposite normal)
    (sheet(0.12, 0, 6), ( 1,0,0), -0.12),   # 2 drift copy of the same wall
    (sheet(2.70, 0, 6), ( 1,0,0), -2.70),   # 3 distinct wall 2.7 m away
    (sheet(0.32, 20, 24), ( 1,0,0), -0.32), # 4 parallel, 12 cm off face B, but laterally DISJOINT
    (np.stack([rng.uniform(0,6,6000), rng.uniform(0,6,6000),
               rng.normal(0,0.008,6000)],1), (0,0,1), 0.0),      # 5 floor
    (np.stack([rng.uniform(0,6,6000), rng.uniform(0,6,6000),
               np.full(6000,0.15)+rng.normal(0,0.008,6000)],1),
     (0,0,1), -0.15),                        # 6 drift-split floor slab
]
P=np.vstack([q for q,_,_ in parts])
pid=np.concatenate([np.full(len(q),k,np.int64) for k,(q,_,_) in enumerate(parts)])
planes=[(np.array(n,float),float(d)) for _,n,d in parts]

# defaults (no join): nothing is close enough for the strict rule
_,_,n0 = ms.merge_coplanar(P,[p for p in planes],pid.copy())
print(f"strict only: {n0} merged (expected 0 -- this is the stacked-sheet bug)")
assert n0==0

pl,pd,nm = ms.merge_coplanar(P,[p for p in planes],pid.copy(),
                             join_ang=8.0, join_off=0.30, join_overlap=0.25)
groups={}
for k in range(7):
    tgt=pd[np.flatnonzero(pid==k)[0]]
    groups.setdefault(int(tgt),[]).append(k)
print(f"with join : {nm} merged -> {len(pl)} planes, groups {sorted(groups.values())}")
assert sorted(map(sorted,groups.values()))==[[0,1,2],[3],[4],[5,6]], groups
# the wall's three sheets became ONE plane sitting mid-wall
gwall=[g for g in groups if set(groups[g])=={0,1,2}][0]
n_,d_=pl[gwall]
x_at=-d_/n_[0]
print(f"merged wall sits at x={x_at:+.3f} (faces at 0.00/0.12/0.20 -> mid)")
assert 0.05 < x_at < 0.16
gfl=[g for g in groups if set(groups[g])=={5,6}][0]
zf=-pl[gfl][2] if False else None
nf,df=pl[gfl]; z_at=-df/nf[2]
assert 0.04 < z_at < 0.11, z_at
print(f"merged floor sits at z={z_at:+.3f} (slabs at 0.00/0.15 -> mid)")

# meshing the merged wall gives one continuous flat sheet
Q=P[pd==gwall]
V,F=ms.mesh_plane(Q, np.asarray(pl[gwall][0]), pl[gwall][1], cell=0.10)
assert np.abs(V @ np.asarray(pl[gwall][0]) + pl[gwall][1]).max() < 1e-9
present=lambda y,z: bool(((np.abs(V[:,1]-y)<0.12)&(np.abs(V[:,2]-z)<0.12)).any())
assert present(3.0,1.5) and present(0.5,0.5) and present(5.5,2.5)
print(f"merged wall meshes as ONE exactly-flat sheet ({len(F)} tris)")

# close_floor mechanism: inf fill closes the enclosed atrium, border stays open
n=400_000
x=rng.uniform(0,30,n); y=rng.uniform(0,20,n)
ring=~((x>9)&(x<21)&(y>6)&(y<14))
Qf=np.stack([x[ring],y[ring],np.zeros(ring.sum())],1)
V,_=ms.mesh_plane(Qf,np.array([0.,0.,1.]),0.0,cell=0.10,
                  max_fill_m2=float("inf"))
mid=bool(((np.abs(V[:,0]-15)<0.15)&(np.abs(V[:,1]-10)<0.15)).any())
out=bool(((np.abs(V[:,0]+2)<0.15)&(np.abs(V[:,1]-10)<0.15)).any())
print(f"close_floor: 96 m^2 atrium filled={mid} (want True), "
      f"outside-the-building filled={out} (want False)")
assert mid and not out
print("\nALL WALL-MERGE / CLOSE-FLOOR TESTS PASSED")
