"""Round objects stay smooth (escape the wall sweep, fail the box test);
box-like clusters become clean cuboids."""
import sys, types
import numpy as np
sys.path.insert(0, "/home/user/Others/lidar_mapping")
sys.modules.setdefault("open3d", types.ModuleType("open3d"))
for a in ("geometry","utility","io","core","t"):
    setattr(sys.modules["open3d"], a, types.SimpleNamespace())
import importlib.util
spec=importlib.util.spec_from_file_location("ms","/home/user/Others/lidar_mapping/02_pcd_to_mesh_sionna_v9.py")
ms=importlib.util.module_from_spec(spec); spec.loader.exec_module(ms)
rng=np.random.default_rng(13)

# ---- fit_cuboid: box yes, cylinder no, panel -> thin slab ------------------
def cabinet(n=5000, yaw=0.5):   # 0.8 x 0.5 x 1.2 box, 5 faces scanned
    pts=[]
    for _ in range(n):
        f=rng.integers(5)
        u=rng.uniform(-.4,.4); v=rng.uniform(-.25,.25); z=rng.uniform(0,1.2)
        p=[( .4 if f==0 else -.4 if f==1 else u),
           ( .25 if f==2 else -.25 if f==3 else v),
           (1.2 if f==4 else z)]
        if f>=4: p[0],p[1]=u,v
        pts.append(p)
    Q=np.array(pts)+rng.normal(0,0.006,(n,3))
    R=np.array([[np.cos(yaw),-np.sin(yaw),0],[np.sin(yaw),np.cos(yaw),0],[0,0,1]])
    return Q@R.T+np.array([4.,2.,0.])
def cylinder(n=5000, r=0.30):
    th=rng.uniform(0,2*np.pi,n)
    return np.stack([r*np.cos(th), r*np.sin(th), rng.uniform(0,1.5,n)],1) \
        + rng.normal(0,0.006,(n,3)) + np.array([1.,1.,0.])

box = ms.fit_cuboid(cabinet(), 0.04, 0.80)
assert box is not None, "cabinet must fit a cuboid"
V,F = box
assert V.shape==(8,3) and F.shape==(12,3)
# a yaw-rotated box: measure its EDGES from the corners, not the world AABB
eu=np.linalg.norm(V[1]-V[0]); ev=np.linalg.norm(V[3]-V[0])
ez=np.linalg.norm(V[4]-V[0])
L,Wd=max(eu,ev),min(eu,ev)
print(f"cabinet -> cuboid: 8 verts / 12 tris, edges {L:.2f} x {Wd:.2f} x "
      f"{ez:.2f} (true 0.80 x 0.50 x 1.20)")
assert abs(L-0.8)<0.05 and abs(Wd-0.5)<0.05 and abs(ez-1.2)<0.06
assert ms.fit_cuboid(cylinder(), 0.04, 0.80) is None, \
    "a 0.3 m cylinder must NOT be boxed (bulge ~0.09 m >> tol)"
print("cylinder  -> rejected (stays on the smooth Poisson path)")
panel = ms.fit_cuboid(np.stack([rng.uniform(0,1.0,3000),
                                rng.normal(0,0.005,3000),
                                rng.uniform(0,0.8,3000)],1), 0.04, 0.80)
assert panel is not None, "a flat panel is a thin slab, still a clean block"
print("flat panel -> thin slab cuboid")

# ---- wall sweep: pillar facets rejected, wall kept -------------------------
def scene():
    nW=9000
    wall=np.stack([np.full(nW,0.)+rng.normal(0,0.008,nW),
                   rng.uniform(0,6,nW), rng.uniform(0,3,nW)],1)
    nwall=np.tile([1.,0,0],(nW,1))
    cyl=cylinder(1500, r=0.30)+np.array([2.,2.,0.])   # ~lattice density
    th=np.arctan2(cyl[:,1]-3.0, cyl[:,0]-3.0)
    ncyl=np.stack([np.cos(th),np.sin(th),np.zeros(len(cyl))],1)
    P=np.vstack([wall,cyl]); N=np.vstack([nwall,ncyl])
    return P,N,nW

P,N,nW = scene()
pl,pid = ms.extract_planes_two_phase(P,N,0.06,dist=0.08,ang_deg=35.0,
                                     big_area=8.0,wall_area=0.5,
                                     wall_min_len=1.2)
wall_cap=(pid[:nW]>=0).mean(); cyl_cap=(pid[nW:]>=0).mean()
print(f"guard ON : wall captured {wall_cap:.0%}, "
      f"cylinder captured as planes {cyl_cap:.0%} (want ~0%)")
assert wall_cap>0.9 and cyl_cap<0.05
P,N,nW = scene()
pl0,pid0 = ms.extract_planes_two_phase(P,N,0.06,dist=0.08,ang_deg=35.0,
                                       big_area=8.0,wall_area=0.5,
                                       wall_min_len=0)
cyl0=(pid0[nW:]>=0).mean()
print(f"guard OFF: cylinder captured as planes {cyl0:.0%} "
      f"(the faceting bug this fixes)")
assert cyl0>0.25, "without the guard the pillar should get faceted"

# ---- cylinder primitive: slender pillar comes out ROUND --------------------
def pillar(r=0.14, n=4000):
    th=rng.uniform(0,2*np.pi,n)
    return np.stack([2.+r*np.cos(th), 5.+r*np.sin(th),
                     rng.uniform(0,2.6,n)],1)+rng.normal(0,0.006,(n,3))
Q=pillar()
cyl_fit=ms.fit_cylinder(Q, 0.035, 0.80)
assert cyl_fit is not None, "a clean pillar must fit the cylinder"
Vc,Fc=cyl_fit
r_est=np.linalg.norm(Vc[:48,:2]-[2.,5.],axis=1).mean()
print(f"pillar r=0.14 -> cylinder primitive: {len(Vc)} verts, "
      f"{len(Fc)} tris, r={r_est:.3f}")
assert abs(r_est-0.14)<0.02
# the same slender pillar PASSES the box test (bulge ~0.3*r ~ 0.04 = tol),
# which is why cylinders were coming out square: order matters
assert ms.fit_cuboid(Q, 0.04, 0.80) is not None, \
    "expected: slender pillars fool the box test -- cylinder must go first"
print("           ...and it fools fit_cuboid, so cylinder-first is required")
# a real box must never fit a cylinder (corners are far off any circle)
assert ms.fit_cylinder(cabinet(), 0.035, 0.80) is None
print("cabinet   -> cylinder rejected (corners kill the circle fit)")

# ---- sliver guards: the 'protruding planes' artefact -----------------------
t=rng.uniform(0,1,3000)
ghost=np.stack([t*2.0, t*1.2, t*1.8],1)+rng.normal(0,0.01,(3000,3))
assert ms.fit_cuboid(ghost, 0.04, 0.80) is None, \
    "a diagonal ghost line must not become a sliver box"
print("diagonal ghost line -> rejected (dims + face-coverage guards)")

# ---- floor fill: everything closed EXCEPT the largest hole -----------------
n=500_000
x=rng.uniform(0,30,n); y=rng.uniform(0,20,n)
atrium=(x>9)&(x<21)&(y>6)&(y<14)              # 96 m^2  -> stays open
big   =(x>1)&(x<7)&(y>1)&(y<6)                # 30 m^2  -> closed
small =(x>23)&(x<27)&(y>15)&(y<17)            # 8 m^2   -> closed
keep=~(atrium|big|small)
Qf=np.stack([x[keep],y[keep],np.zeros(keep.sum())],1)
V,_=ms.mesh_plane(Qf,np.array([0.,0.,1.]),0.0,cell=0.10,
                  max_fill_m2=20.0, fill_all_but_largest=True)
at=lambda px,py: bool(((np.abs(V[:,0]-px)<0.15)&(np.abs(V[:,1]-py)<0.15)).any())
print(f"all_but_center: 30 m^2 patch filled={at(4,3.5)}, "
      f"8 m^2 filled={at(25,16)}, atrium open={not at(15,10)}, "
      f"outside open={not at(-2,10)}")
assert at(4,3.5) and at(25,16) and not at(15,10) and not at(-2,10)

import re
src=open("/home/user/Others/lidar_mapping/02_pcd_to_mesh_sionna_v9.py").read()
assert re.search(r'floor_fill\s*=\s*"all_but_center"', src)
print('floor_fill defaults to "all_but_center"')
print("\nALL ROUND-OBJECT / CUBOID / FILL TESTS PASSED")
