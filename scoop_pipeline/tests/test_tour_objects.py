#!/usr/bin/env python
"""Self-tests for assets/tour_objects.py: a room with a partition, a box in
front of it and one behind. Seen from the near side the first is boxed where
it projects, the hidden one not; the whole video run end to end. Needs numpy,
open3d, opencv.

    python tests/test_tour_objects.py
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
sys.path.insert(0, os.path.join(ROOT, "datasets"))

import tour_objects as to                                               # noqa: E402

FAILED = []


def box_mesh(lo, hi):
    import open3d as o3d
    m = o3d.geometry.TriangleMesh.create_box(*(np.asarray(hi) - np.asarray(lo)))
    m.translate(lo)
    return m


def scene_parts():
    import open3d as o3d
    m = o3d.geometry.TriangleMesh()
    for lo, hi in (([-0.1, -3, -0.1], [8, 3, 0]), ([-0.1, -3, 2.5], [8, 3, 2.6]),          # floor, ceiling
                   ([-0.1, -3.1, 0], [8, -3, 2.5]), ([-0.1, 3, 0], [8, 3.1, 2.5]),
                   ([-0.2, -3, 0], [-0.1, 3, 2.5]), ([8, -3, 0], [8.1, 3, 2.5]),
                   ([4.0, -3, 0], [4.1, 3, 2.5]),                                         # the partition
                   ([2.5, -0.4, 0], [3.1, 0.4, 0.9]), ([6.0, -0.4, 0], [6.6, 0.4, 0.9])):  # in front, behind
        m += box_mesh(lo, hi)
    return m


def box_points(lo, hi, step=0.03):
    g = [np.arange(a, b + 1e-9, step) for a, b in zip(lo, hi)]
    X, Y, Z = np.meshgrid(*g, indexing="ij")
    P = np.c_[X.ravel(), Y.ravel(), Z.ravel()]
    on = (np.isclose(P, lo, atol=step / 2) | np.isclose(P, hi, atol=step / 2)).any(1)
    return P[on]


def test_visible_and_hidden():
    import open3d as o3d
    sc = o3d.t.geometry.RaycastingScene()
    sc.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(scene_parts()))
    objs = [{"id": "table_00", "label": "table"}, {"id": "fridge_00", "label": "fridge"}]
    pts = [box_points([2.5, -0.4, 0], [3.1, 0.4, 0.9]), box_points([6.0, -0.4, 0], [6.6, 0.4, 0.9])]
    W, H = 640, 360
    f = (H / 2) / np.tan(np.radians(70) / 2)
    b = to.frame_boxes(objs, pts, sc, [0.5, 0, 1.6], [3.0, 0, 1.6], W, H, f)
    ids = [x["id"] for x in b]
    assert ids == ["table_00"], b
    x0, y0, x1, y1 = b[0]["box"]
    assert x0 < W / 2 < x1 and y1 > H / 2, b                         # ahead, below the eye
    assert abs(b[0]["z"] - 2.0) < 0.3, b


def test_video(d):
    import open3d as o3d
    import cv2
    from lod3dgs import video_writer
    mp = os.path.join(d, "mesh.ply")
    o3d.io.write_triangle_mesh(mp, scene_parts())
    reg = os.path.join(d, "reg")
    os.makedirs(os.path.join(reg, "points"))
    from objects import write_ply
    P = box_points([2.5, -0.4, 0], [3.1, 0.4, 0.9])
    write_ply(os.path.join(reg, "points", "table_00.ply"), P, np.zeros((len(P), 3), np.uint8))
    json.dump({"objects": [{"id": "table_00", "label": "table", "points": "points/table_00.ply"}]},
              open(os.path.join(reg, "objects.json"), "w"))
    vid = os.path.join(d, "fly.mp4")
    vw = video_writer(vid, 10, (320, 180))
    for _ in range(5):
        vw.write(np.full((180, 320, 3), 40, np.uint8))
    vw.release()
    E = [[0.5 + 0.1 * k, 0, 1.6] for k in range(5)]
    json.dump({"fps": 10, "eye": E, "look": [[3.0, 0, 1.6]] * 5, "mesh": mp},
              open(os.path.join(d, "fly_path.json"), "w"))
    to.main([vid, reg])
    cap = cv2.VideoCapture(os.path.join(d, "fly_objects.mp4"))
    n, drawn = 0, 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        n += 1
        drawn += int((np.abs(img.astype(int) - 40).sum(2) > 120).mean() > 0.01)
    assert n == 5 and drawn == 5, (n, drawn)


def check(name, fn, *args):
    try:
        fn(*args)
        print("  ok    %s" % name)
    except Exception:
        FAILED.append(name)
        print("  FAIL  %s" % name)
        traceback.print_exc()


def main():
    d = tempfile.mkdtemp(prefix="tour_objects_")
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
