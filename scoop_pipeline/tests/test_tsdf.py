#!/usr/bin/env python
"""Self-test for 01_build_map's TSDF fusion (fusion: "tsdf") against its
voxel union, on a synthetic 10 x 8 x 3 m room scanned 300 times, each scan
with 1 cm range noise and its own pose error (1.5 cm, 0.15 deg): the smear
a real map has. Both fuse the same points at 2 cm; map_quality measures
them. The union must keep the smear's full width, the TSDF must be thinner
and its surfaces where the true ones are. Needs numpy, scipy, open3d.

    python tests/test_tsdf.py
"""
import importlib.util
import os
import sys

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "map_stages"))
sys.path.insert(0, os.path.join(ROOT, "analysis", "mapping"))

import map_quality as mq                                               # noqa: E402

spec = importlib.util.spec_from_file_location("build_map", os.path.join(ROOT, "map_stages",
                                                                        "01_build_map.py"))
b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)


def room(n, rng):
    """n points on the inside of the room (walls x=0,10, y=0,8; floor z=0,
    ceiling z=3)."""
    face = rng.integers(0, 6, n)
    u, v = rng.random(n), rng.random(n)
    P = np.c_[10 * u, 8 * v, 3 * rng.random(n)]
    P[face == 0, 0] = 0
    P[face == 1, 0] = 10
    P[face == 2, 1] = 0
    P[face == 3, 1] = 8
    P[face == 4] = np.c_[10 * u, 8 * v, np.zeros(n)][face == 4]
    P[face == 5] = np.c_[10 * u, 8 * v, np.full(n, 3.0)][face == 5]
    return P


def main():
    rng = np.random.default_rng(0)
    union = b.VoxelAccumulator(0.02)
    tsdf = b.TsdfAccumulator(0.02, 0.08)
    for k in range(300):
        a = 2 * np.pi * k / 100
        o = np.array([5 + 2.5 * np.cos(a), 4 + 2 * np.sin(a), 0.6])
        P = room(20000, rng)
        d = P - o
        r = np.linalg.norm(d, axis=1, keepdims=True)
        P = P + d / r * rng.normal(0, 0.01, (len(P), 1))           # range noise
        R = Rotation.from_euler("z", rng.normal(0, 0.15), degrees=True).as_matrix()
        e = rng.normal(0, 0.015, 3)
        Pw = (P - o) @ R.T + o + e                                   # the scan's pose error
        union.add(Pw)
        tsdf.add(o + e, Pw)
    U = union.points()
    T = tsdf.surface(3)
    fails = []
    res = {}
    for name, M in (("union", U), ("tsdf", T)):
        st, _ = mq.patch_stats(M, n_samples=3000)
        wall = st[:, 1] > 60
        flat = st[:, 1] < 30
        res[name] = (np.median(st[wall, 0]) * 100, np.median(st[flat, 0]) * 100, len(M))
    # where the TSDF puts the x = 0 wall and the floor (away from corners)
    inner = (T[:, 1] > 1) & (T[:, 1] < 7) & (T[:, 2] > 0.5) & (T[:, 2] < 2.5)
    wall_x = T[inner & (T[:, 0] < 0.3), 0]
    floor = T[(T[:, 0] > 1) & (T[:, 0] < 9) & (T[:, 1] > 1) & (T[:, 1] < 7) & (T[:, 2] < 0.3), 2]
    print(f"  union  walls {res['union'][0]:.1f} cm, floors {res['union'][1]:.1f} cm ({res['union'][2]} points)")
    print(f"  tsdf   walls {res['tsdf'][0]:.1f} cm, floors {res['tsdf'][1]:.1f} cm ({res['tsdf'][2]} points)")
    print(f"  tsdf   wall x=0 at {np.mean(wall_x) * 1000:+.1f} mm, floor z=0 at {np.mean(floor) * 1000:+.1f} mm")
    if not res["tsdf"][0] < 0.6 * res["union"][0]:
        fails.append("TSDF walls not thinner than the union's")
    if not (abs(np.mean(wall_x)) < 0.005 and abs(np.mean(floor)) < 0.005):
        fails.append("TSDF surfaces off their true place")
    if len(T) < 0.2 * len(U):
        fails.append("TSDF lost most of the surface")
    for f in fails:
        print("  FAIL  " + f)
    print("tsdf ok" if not fails else "tsdf FAILED")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
