#!/usr/bin/env python
"""Self-tests for assets/objects.py: a synthetic room (walls, floor, two tables
and a chair) rendered from a ring of camera poses into a Replica-format export
(colour, depth, traj.txt, splatam_data_config.yaml) with masks as the npz
segmenter reads them; the registry must hold three objects -- two tables kept
apart, the chair -- at the right places, with points, masks and views.
Needs numpy, opencv, pyyaml.

    python tests/test_asset_objects.py
"""
import json
import os
import shutil
import sys
import tempfile
import traceback

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "assets"))

import objects as ob                                                    # noqa: E402
import views as vw                                                      # noqa: E402

FAILED = []
W, H, FX, CX, CY = 320, 240, 260.0, 160.0, 120.0
K = np.array([[FX, 0, CX], [0, FX, CY], [0, 0, 1.0]])


def box_surface(c, size, step=0.01):
    """Points on the surface of an axis-aligned box."""
    c, s = np.asarray(c, float), np.asarray(size, float)
    pts = []
    for ax in range(3):
        for sign in (-0.5, 0.5):
            u, v = [a for a in range(3) if a != ax]
            gu = np.arange(-s[u] / 2, s[u] / 2 + 1e-9, step)
            gv = np.arange(-s[v] / 2, s[v] / 2 + 1e-9, step)
            U, V = np.meshgrid(gu, gv)
            p = np.zeros((U.size, 3))
            p[:, ax] = sign * s[ax]
            p[:, u], p[:, v] = U.ravel(), V.ravel()
            pts.append(p + c)
    return np.vstack(pts)


def scene():
    """(points, object id per point: 0 room, 1.. objects, labels, colours)."""
    room = box_surface([0, 0, 1.25], [8, 6, 2.5], 0.03)
    objs = [("table", [-1.8, 0.8, 0.375], [1.2, 0.7, 0.75], (40, 160, 40)),
            ("table", [1.8, -0.9, 0.375], [1.2, 0.7, 0.75], (40, 160, 40)),
            ("chair", [0.2, 1.6, 0.45], [0.5, 0.5, 0.9], (200, 40, 40))]
    P, I, C = [room], [np.zeros(len(room), int)], [np.tile((150, 150, 150), (len(room), 1))]
    for k, (_, c, s, col) in enumerate(objs, 1):
        p = box_surface(c, s, 0.008)
        P.append(p); I.append(np.full(len(p), k)); C.append(np.tile(col, (len(p), 1)))
    return np.vstack(P), np.concatenate(I), [o[0] for o in objs], np.vstack(C), objs


def look_at(eye, target):
    """camera-to-world, OpenCV axes (x right, y down, z forward)."""
    z = np.asarray(target, float) - eye
    z /= np.linalg.norm(z)
    x = np.cross(z, [0, 0, 1.0]); x /= np.linalg.norm(x)
    y = np.cross(z, x)
    T = np.eye(4)
    T[:3, :3] = np.c_[x, y, z]
    T[:3, 3] = eye
    return T


def render(P, I, C, T):
    """z-buffered splat: depth, id, colour images."""
    Tcw = np.linalg.inv(T)
    pc = P @ Tcw[:3, :3].T + Tcw[:3, 3]
    ok = pc[:, 2] > 0.1
    pc, ii, cc = pc[ok], I[ok], C[ok]
    u = np.round(FX * pc[:, 0] / pc[:, 2] + CX).astype(int)
    v = np.round(FX * pc[:, 1] / pc[:, 2] + CY).astype(int)
    depth = np.full((H, W), np.inf)
    ids = np.full((H, W), -1)
    col = np.zeros((H, W, 3), np.uint8)
    order = np.argsort(-pc[:, 2])                      # far first, near overwrites
    for du in (-1, 0, 1):                              # splat 3x3: close gaps
        for dv in (-1, 0, 1):
            uu, vv = u[order] + du, v[order] + dv
            m = (uu >= 0) & (uu < W) & (vv >= 0) & (vv < H)
            z = pc[order, 2][m]
            uu, vv = uu[m], vv[m]
            o = order[m]
            for a, b, zz, k in zip(vv, uu, z, o):
                if zz < depth[a, b]:
                    depth[a, b] = zz; ids[a, b] = ii[k]; col[a, b] = cc[k]
    depth[~np.isfinite(depth)] = 0
    return depth, ids, col


def make_export(d, n_frames=16, missed=None):
    """missed: {label: frames} the segmenter does not find it in."""
    missed = missed or {}
    P, I, labels, C, objs = scene()
    ex = os.path.join(d, "export")
    os.makedirs(os.path.join(ex, "results"))
    masks = os.path.join(d, "masks")
    os.makedirs(masks)
    traj = []
    for i in range(n_frames):
        a = 2 * np.pi * i / n_frames
        T = look_at(np.array([3.0 * np.cos(a), 2.2 * np.sin(a), 1.4]), [0, 0, 0.5])
        depth, ids, col = render(P, I, C, T)
        cv2.imwrite(os.path.join(ex, "results", "frame%06d.jpg" % i), col[:, :, ::-1])
        cv2.imwrite(os.path.join(ex, "results", "depth%06d.png" % i),
                    np.clip(depth * 6553.5, 0, 65535).astype(np.uint16))
        traj.append(" ".join("%.9f" % x for x in T.ravel()))
        ms, ls, ss = [], [], []
        for k, lab in enumerate(labels, 1):
            m = ids == k
            if m.sum() > 50 and i not in missed.get(lab, ()):
                # the segmenter's mask bleeds 2 px onto what is behind
                ms.append(cv2.dilate(m.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0)
                ls.append(lab); ss.append(0.9)
        np.savez(os.path.join(masks, "%06d.npz" % i), masks=np.array(ms, bool).reshape(-1, H, W),
                 labels=np.array(ls), scores=np.array(ss))
    open(os.path.join(ex, "traj.txt"), "w").write("\n".join(traj) + "\n")
    open(os.path.join(ex, "splatam_data_config.yaml"), "w").write(
        "dataset_name: \"replica\"\ncamera_params:\n  image_height: %d\n  image_width: %d\n"
        "  fx: %f\n  fy: %f\n  cx: %f\n  cy: %f\n  png_depth_scale: 6553.5\n  crop_edge: 0\n"
        % (H, W, FX, FX, CX, CY))
    return ex, masks, objs


def test_registry_of_a_synthetic_room(d):
    ex, masks, objs = make_export(d)
    out = os.path.join(d, "registry")
    ob.main([ex, out, "--prompts", "table,chair", "--segmenter", "npz:" + masks,
             "--every", "1", "--min-points", "100"])
    reg = json.load(open(os.path.join(out, "objects.json")))["objects"]
    labels = sorted(r["label"] for r in reg)
    assert labels == ["chair", "table", "table"], labels
    for lab, c, s, _ in objs:
        near = [r for r in reg if r["label"] == lab
                and np.linalg.norm(np.array(r["centroid"][:2]) - c[:2]) < 0.35]
        assert len(near) == 1, (lab, c, [r["centroid"] for r in reg])
        r = near[0]
        lo, hi = np.array(r["aabb_min"]), np.array(r["aabb_max"])
        # the 2 px bleed is eroded away: nothing of the floor or a wall in it
        assert (lo > np.array(c) - np.array(s) / 2 - 0.06).all(), (lab, lo)
        assert (hi < np.array(c) + np.array(s) / 2 + 0.06).all(), (lab, hi)
        assert r["n_views"] >= 4 and len(r["best_views"]) == 3
        assert os.path.isfile(os.path.join(out, r["points"]))
        for v in r["best_views"]:
            assert os.path.isfile(os.path.join(out, "masks", "%s_%06d.png" % (r["id"], v["frame"])))
    f = json.load(open(os.path.join(out, "frames.json")))
    assert f["prompts"] == ["table", "chair"] and len(f["frames"]) == 16


def test_all_views_include_the_missed_ones(d):
    """The segmenter misses the chair in 4 frames: views.py finds it there,
    marks those views unsegmented, ranks views for SAM 3D, writes masks."""
    ex, masks, objs = make_export(d, missed={"chair": (2, 5, 9, 13)})
    out = os.path.join(d, "registry")
    ob.main([ex, out, "--prompts", "table,chair", "--segmenter", "npz:" + masks,
             "--every", "1", "--min-points", "100"])
    vw.main([ex, out, "--min-pixels", "200"])
    reg = {r["id"]: r for r in json.load(open(os.path.join(out, "objects.json")))["objects"]}
    chair = reg["chair_00"]
    by = {v["frame"]: v for v in chair["views_all"]}
    for f in (2, 5, 9, 13):
        assert f in by and not by[f]["segmented"], (f, sorted(by))
    assert chair["segmented"] == 12 and chair["seen"] >= 16, (chair["segmented"], chair["seen"])
    for r in reg.values():
        assert 1 <= len(r["best_for_3d"]) <= 5
        ranks = [b["rank"] for b in r["best_for_3d"]]
        assert ranks == sorted(ranks, reverse=True)
        for b in r["best_for_3d"]:
            m = cv2.imread(os.path.join(out, b["mask"]), cv2.IMREAD_GRAYSCALE)
            assert m is not None and (m > 0).sum() >= 200
        for v in r["views_all"]:
            assert 0.1 <= v["visible"] <= 1.0 and 0.0 <= v["cut"] <= 1.0


def test_occluded_points_are_not_visible():
    """A point behind the measured surface at its pixel is hidden."""
    depth = np.full((H, W), 2.0, np.float32)
    T = np.eye(4)
    P = np.array([[0.0, 0.0, 2.0], [0.0, 0.0, 3.0], [0.1, 0.0, 2.02]])
    v = vw.view_of(P, depth, K, T, 0.05, 0.02)
    assert abs(v["visible"] - 2 / 3) < 1e-3, v


def test_prompts_from_tags():
    tags = [["chair", "table", "wall", "floor"], ["chair", "monitor"], ["chair", "table"],
            ["plant"]] + [["chair"]] * 6
    assert ob.prompts_from_tags(tags, 0.15) == ["chair", "table"]


def check(name, fn, *args):
    try:
        fn(*args)
        print("  ok    %s" % name)
    except Exception:
        FAILED.append(name)
        print("  FAIL  %s" % name)
        traceback.print_exc()


def main():
    d = tempfile.mkdtemp(prefix="asset_objects_")
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
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
