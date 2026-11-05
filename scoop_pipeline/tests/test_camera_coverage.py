#!/usr/bin/env python
"""Self-tests for analysis/camera_coverage.py: an 8 x 5 m room with a
partition stub and a table, four cameras high in the corners aimed at the
walls. Re-aimed, the floor seen by two or more of them must grow to most of
it; aims, occlusion and the floor plane checked on their own. Needs numpy,
scipy, open3d, opencv.

    python tests/test_camera_coverage.py
"""
import json
import os
import shutil
import sys
import tempfile
import traceback

import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "analysis"))

import camera_coverage as cc                                            # noqa: E402

FAILED = []


def box(lo, hi):
    import open3d as o3d
    m = o3d.geometry.TriangleMesh.create_box(*(np.asarray(hi, float) - np.asarray(lo, float)))
    m.translate(lo)
    return m


def room_mesh():
    import open3d as o3d
    m = o3d.geometry.TriangleMesh()
    for lo, hi in (([0, 0, -0.005], [8, 5, 0]), ([0, 0, 2.6], [8, 5, 2.605]), ([-0.1, 0, 0], [0, 5, 2.6]),
                   ([8, 0, 0], [8.1, 5, 2.6]), ([0, -0.1, 0], [8, 0, 2.6]), ([0, 5, 0], [8, 5.1, 2.6]),
                   ([4.0, 0, 0], [4.1, 2.5, 2.6]),                     # partition stub
                   ([1.5, 3.0, 0], [3.0, 4.0, 0.8])):                  # table
        m += box(lo, hi)
    return m


def room_points(m, n=400000):
    return np.asarray(m.sample_points_uniformly(n).points)


def look(o, target):
    d = np.asarray(target, float) - o
    p = np.degrees(np.arctan2(d[1], d[0]))
    t = np.degrees(np.arcsin(d[2] / np.linalg.norm(d)))
    return cc.aim_rotation(p, t)


def test_aim_rotation():
    R = cc.aim_rotation(90, -30)
    assert np.allclose(R.T @ R, np.eye(3), atol=1e-9) and np.linalg.det(R) > 0
    p, t = cc.pan_tilt(R)
    assert abs(p - 90) < 1e-6 and abs(t + 30) < 1e-6
    assert R[2, 0] == 0 or abs(R[2, 0]) < 1e-9                          # image x level: no roll


def test_floor_plane_tilted():
    rng = np.random.default_rng(0)
    xy = rng.uniform(0, 8, (20000, 2))
    z = -0.7 - 0.018 * xy[:, 0] + rng.normal(0, 0.01, len(xy))
    clutter = np.c_[rng.uniform(0, 8, (5000, 2)), rng.uniform(0, 2, 5000)]
    abc = cc.floor_plane(np.vstack([np.c_[xy, z], clutter]))
    assert abs(abc[0] + 0.7) < 0.01 and abs(abc[1] + 0.018) < 0.002 and abs(abc[2]) < 0.002, abc


def test_reaim(d):
    import open3d as o3d
    m = room_mesh()
    mp, mh = os.path.join(d, "room.ply"), os.path.join(d, "mesh.ply")
    o3d.io.write_point_cloud(mp, o3d.geometry.PointCloud(o3d.utility.Vector3dVector(room_points(m))))
    o3d.io.write_triangle_mesh(mh, m)
    from scipy.spatial.transform import Rotation as Rot
    tr = []
    for k, (o, wall) in enumerate(((0.2, 0.2), (7.8, 0.2), (7.8, 4.8), (0.2, 4.8))):
        p = np.array([o, wall, 2.4])
        R = look(p, [p[0] + (-1 if p[0] > 4 else 1) * 0.1, p[1], 2.4])  # nearly at its own wall
        tr.append({"parent": "map", "child": "cam_%d" % (k + 1), "translation": p.tolist(),
                   "rotation_xyzw": Rot.from_matrix(R).as_quat().tolist()})
    cy = os.path.join(d, "cameras.yaml")
    yaml.safe_dump({"transforms": tr}, open(cy, "w"))
    out = os.environ.get("COVERAGE_OUT") or os.path.join(d, "out")
    cc.main([cy, out, "--map", mp, "--mesh", mh, "--room", "0", "0", "8", "0", "5", "--cell", "0.25",
             "--pan-step", "10", "--tilt", "10,70,10", "--restarts", "5", "--exact", "30", "--exact-aims", "15"])
    r = json.load(open(os.path.join(out, "coverage.json")))
    assert r["before"]["share_2plus"] < 0.4, r["before"]
    assert r["after"]["share_2plus"] > 0.85, r["after"]
    assert r["after"]["share_1plus"] >= r["after"]["share_2plus"]
    assert r["objective"]["bound"] is None or r["objective"]["after"] <= r["objective"]["bound"] + 1e-6
    # the table's cells are not targets
    assert r["cells"] < (7.8 / 0.25) * (4.8 / 0.25) - 10
    for f in ("cameras_aimed.yaml", "coverage_before.png", "coverage_after.png"):
        assert os.path.isfile(os.path.join(out, f)), f


def check(name, fn, *args):
    try:
        fn(*args)
        print("  ok    %s" % name)
    except Exception:
        FAILED.append(name)
        print("  FAIL  %s" % name)
        traceback.print_exc()


def main():
    d = tempfile.mkdtemp(prefix="camera_coverage_")
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
