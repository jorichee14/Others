#!/usr/bin/env python
"""Self-tests for assets/split_scene.py: a synthetic room (floor, walls,
ceiling), two registry objects and one box the registry does not name. The
split: each object's surfaces to it, the room's planes to the shell, the
unnamed box to "other"; each part meshed, every mesh at decreasing levels of
detail. Needs numpy, scipy, open3d.

    python tests/test_split_scene.py
"""
import json
import os
import shutil
import sys
import tempfile
import traceback

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "assets"))

import split_scene as ss                                                # noqa: E402
from objects import write_ply                                           # noqa: E402

FAILED = []


def box_surface(c, size, step):
    c, s = np.asarray(c, float), np.asarray(size, float)
    pts = []
    for ax in range(3):
        for sign in (-0.5, 0.5):
            u, v = [a for a in range(3) if a != ax]
            U, V = np.meshgrid(np.arange(-s[u] / 2, s[u] / 2 + 1e-9, step),
                               np.arange(-s[v] / 2, s[v] / 2 + 1e-9, step))
            p = np.zeros((U.size, 3))
            p[:, ax] = sign * s[ax]
            p[:, u], p[:, v] = U.ravel(), V.ravel()
            pts.append(p + c)
    return np.vstack(pts)


def make(d):
    import open3d as o3d
    rng = np.random.default_rng(0)
    room = box_surface([0, 0, 1.25], [8, 6, 2.5], 0.03)
    tab = box_surface([-1.5, 1.0, 0.4], [1.2, 0.7, 0.8], 0.02)
    tab = tab[tab[:, 2] > 0.005]                                       # standing on the floor
    mon = box_surface([-1.5, 1.0, 1.0], [0.5, 0.1, 0.4], 0.01)
    other = box_surface([2.0, -1.5, 0.3], [0.6, 0.6, 0.6], 0.02)
    other = other[other[:, 2] > 0.005]
    allp = np.vstack([room, tab, mon, other]) + rng.normal(0, 0.004, (len(room) + len(tab) + len(mon) + len(other), 3))
    col = np.vstack([np.tile([0.6, 0.6, 0.6], (len(room), 1)), np.tile([0.2, 0.7, 0.2], (len(tab), 1)),
                     np.tile([0.1, 0.1, 0.8], (len(mon), 1)), np.tile([0.8, 0.3, 0.1], (len(other), 1))])
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(allp))
    pc.colors = o3d.utility.Vector3dVector(col)
    mp = os.path.join(d, "room.ply")
    o3d.io.write_point_cloud(mp, pc)
    reg = os.path.join(d, "registry")
    os.makedirs(os.path.join(reg, "points"))
    objs = []
    for oid, lab, P in (("table_00", "table", tab), ("monitor_00", "monitor", mon)):
        sub = P[rng.random(len(P)) < 0.5]                              # the registry's own, sparser
        write_ply(os.path.join(reg, "points", oid + ".ply"), sub, np.zeros((len(sub), 3), np.uint8))
        objs.append({"id": oid, "label": lab, "points": "points/%s.ply" % oid, "centroid": P.mean(0).tolist(),
                     "aabb_min": P.min(0).tolist(), "aabb_max": P.max(0).tolist()})
    objs[1]["on"] = "table_00"
    json.dump({"objects": objs}, open(os.path.join(reg, "objects.json"), "w"))
    tr = os.path.join(d, "traj.tum")
    t = np.linspace(0, 2 * np.pi, 60)
    np.savetxt(tr, np.c_[t, 2.5 * np.cos(t), 1.8 * np.sin(t), np.full(60, 1.0), np.zeros((60, 3)), np.ones(60)])
    return mp, reg, tr, {"room": len(room), "table": len(tab), "monitor": len(mon), "other": len(other)}


def test_split_and_levels(d):
    mp, reg, tr, n = make(d)
    out = os.path.join(d, "scene")
    ss.main([mp, reg, out, "--traj", tr, "--room", "0", "-4", "4", "-3", "3",
             "--shell-voxel", "0.10", "--other-voxel", "0.04", "--object-voxel", "0.02"])
    sc = json.load(open(os.path.join(out, "scene.json")))
    objs = {o["id"]: o for o in sc["objects"]}
    assert abs(objs["table_00"]["points"] - n["table"]) / n["table"] < 0.05, (objs["table_00"]["points"], n)
    assert abs(objs["monitor_00"]["points"] - n["monitor"]) / n["monitor"] < 0.05
    assert abs(sc["floor_z"]) < 0.03 and abs(sc["ceiling_z"] - 2.5) < 0.03, (sc["floor_z"], sc["ceiling_z"])
    assert abs(sc["parts"]["shell"]["points"] - n["room"]) / n["room"] < 0.05, (sc["parts"]["shell"], n)
    assert abs(sc["parts"]["other"]["points"] - n["other"]) / n["other"] < 0.15, (sc["parts"]["other"], n)
    for rec in [sc["parts"]["shell"], sc["parts"]["other"]] + sc["objects"]:
        faces = [l["faces"] for l in rec["levels"]]
        assert len(faces) == 3 and faces[0] > faces[1] > faces[2] > 0, faces
        for l in rec["levels"]:
            assert os.path.isfile(os.path.join(out, l["file"]))
    assert objs["monitor_00"]["on"] == "table_00"


def check(name, fn, *args):
    try:
        fn(*args)
        print("  ok    %s" % name)
    except Exception:
        FAILED.append(name)
        print("  FAIL  %s" % name)
        traceback.print_exc()


def main():
    d = tempfile.mkdtemp(prefix="split_scene_")
    tests = [(k, f) for k, f in sorted(globals().items()) if k.startswith("test_")]
    try:
        for name, fn in tests:
            args = []
            if fn.__code__.co_argcount == 1:
                args = [os.path.join(d, name)]
                os.makedirs(args[0])
            check(name, fn, *args)
    finally:
        shutil.rmtree(d, ignore_errors=True)
    print(f"\n{len(tests) - len(FAILED)}/{len(tests)} passed")
    sys.exit(1 if FAILED else 0)


if __name__ == "__main__":
    main()
