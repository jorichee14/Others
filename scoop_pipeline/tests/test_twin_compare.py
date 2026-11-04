#!/usr/bin/env python
"""Self-tests for analysis/twin_compare.py on a synthetic twin: a room
(floor, walls, ceiling, a counter along one wall) with three fixed objects.
The "map": what a LiDAR sees of it along a loop, with one object moved, one
gone and a sofa that is not in the twin, 0.3 % too big, 8 mm noise, in
another frame. The room's size must show the 0.3 %, the objects their state.
Needs numpy, scipy.

    python tests/test_twin_compare.py
"""
import os
import sys
import traceback

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "analysis"))
sys.path.insert(0, os.path.join(ROOT, "processing"))

import twin_compare as tc                                               # noqa: E402

FAILED = []
NAMES = ["floor", "wall", "ceiling", "builtin", "fridge", "table", "cabinet"]


def box(lo, hi, step=0.02, top=True, bottom=False):
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    out = []
    for ax in range(3):
        for side, v in ((0, lo[ax]), (1, hi[ax])):
            if ax == 2 and ((side == 1 and not top) or (side == 0 and not bottom)):
                continue
            u, w = [a for a in range(3) if a != ax]
            U, W = np.meshgrid(np.arange(lo[u], hi[u] + 1e-9, step), np.arange(lo[w], hi[w] + 1e-9, step))
            p = np.zeros((U.size, 3))
            p[:, ax], p[:, u], p[:, w] = v, U.ravel(), W.ravel()
            out.append(p)
    return np.vstack(out)



def plane(c, u, v, nu, nv, step=0.02):
    a, b = np.meshgrid(np.arange(0, nu, step), np.arange(0, nv, step))
    return np.asarray(c, float) + a.ravel()[:, None] * np.asarray(u, float) + b.ravel()[:, None] * np.asarray(v, float)


def twin():
    X, Y, Z = np.eye(3)
    parts = [("floor", plane([0, 0, 0], X, Y, 8, 5)), ("ceiling", plane([0, 0, 2.7], X, Y, 8, 5)),
             ("wall", np.vstack([plane([0, 0, 0], X, Z, 8, 2.7), plane([0, 5, 0], X, Z, 8, 2.7),
                                 plane([0, 0, 0], Y, Z, 5, 2.7), plane([8, 0, 0], Y, Z, 5, 2.7)])),
             ("wall", np.vstack([plane([-0.2, -0.2, 0], X, Z, 8.4, 2.7), plane([8.2, -0.2, 0], Y, Z, 5.4, 2.7)])),
             ("builtin", box([0.0, 4.4, 0], [3.0, 5.0, 0.9])),
             ("fridge", box([6.5, 4.3, 0], [7.2, 4.95, 1.8])),
             ("table", box([3.0, 1.5, 0.7], [4.6, 2.3, 0.75], bottom=True)),
             ("cabinet", box([7.5, 0.5, 0], [7.95, 1.4, 1.2]))]
    P = np.vstack([p for _, p in parts])
    L = np.concatenate([np.full(len(p), NAMES.index(n)) for n, p in parts]).astype(np.uint8)
    return P, L


def fake_map(P, L, rng):
    k_tab, k_cab = NAMES.index("table"), NAMES.index("cabinet")
    Q = P.copy()
    c = P[L == k_tab].mean(0)
    Q[L == k_tab] = (P[L == k_tab] - c) @ tc.rot_z(20).T + c + [0.5, 0.4, 0]
    Q = Q[L != k_cab]                                                   # gone
    sofa = box([1.0, 0.3, 0], [3.0, 1.2, 0.45])
    Q = np.vstack([Q, sofa])
    t = np.linspace(0, 2 * np.pi, 80)
    path = np.c_[4 + 2.3 * np.cos(t), 2.5 + 1.2 * np.sin(t), np.full(80, 1.0)]
    seen = tc.visible(Q, tc.poses_along(path, 0.5), 1.0, 15, 22.5, log=lambda *a: None)
    M = (Q[seen] + rng.normal(0, 0.008, (int(seen.sum()), 3))) * 1.003
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = tc.rot_z(-40), [2.0, -1.0, -0.7]
    return M @ T[:3, :3].T + T[:3, 3], (path * 1.003) @ T[:3, :3].T + T[:3, 3]


def test_room_and_objects():
    rng = np.random.default_rng(0)
    P, L = twin()
    M, path = fake_map(P, L, rng)
    r, *_ = tc.compare(M, P, L, NAMES, path, voxel=0.03, min_r=1.0, log=lambda *a: None)
    size = {d["size"]: d for d in r["room"]}
    for name, true_mm in (("length (x)", 24.0), ("width (y)", 15.0), ("height", 8.1)):
        assert abs(size[name]["diff_mm"] - true_mm) < 6.0, (name, size[name], true_mm)
    objs = {o["label"]: o for o in r["objects"]}
    assert objs["fridge"]["status"] == "in place", objs["fridge"]
    assert objs["table"]["status"] == "moved" and abs(objs["table"]["moved_cm"] - 64) < 8, objs["table"]
    assert abs(objs["table"]["turned_deg"] % 180 - 20) < 4, objs["table"]
    assert objs["cabinet"]["status"] == "not found", objs["cabinet"]
    for p in r["planes"]:
        assert abs(p["offset_mm"]) < 20, p


def check(name, fn, *args):
    try:
        fn(*args)
        print("  ok    %s" % name)
    except Exception:
        FAILED.append(name)
        print("  FAIL  %s" % name)
        traceback.print_exc()


def main():
    tests = [(k, f) for k, f in sorted(globals().items()) if k.startswith("test_")]
    for name, fn in tests:
        check(name, fn)
    print(f"\n{len(tests) - len(FAILED)}/{len(tests)} passed")
    sys.exit(1 if FAILED else 0)


if __name__ == "__main__":
    main()
