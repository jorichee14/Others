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
import consolidate as cs                                                # noqa: E402

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


def scene(extra=()):
    """(points, object id per point: 0 room, 1.. objects, labels, colours)."""
    room = box_surface([0, 0, 1.25], [8, 6, 2.5], 0.03)
    objs = [("table", [-1.8, 0.8, 0.375], [1.2, 0.7, 0.75], (40, 160, 40)),
            ("table", [1.8, -0.9, 0.375], [1.2, 0.7, 0.75], (40, 160, 40)),
            ("chair", [0.2, 1.6, 0.45], [0.5, 0.5, 0.9], (200, 40, 40))] + list(extra)
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


def make_export(d, n_frames=16, missed=None, extra=()):
    """missed: {label: frames} the segmenter does not find it in."""
    missed = missed or {}
    P, I, labels, C, objs = scene(extra)
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


def test_consolidate_merges_names_keeps_things_on_others_prunes_false_hits(d):
    """Tables called 'desk' in odd frames, a cup on a table, a false 'plant'
    in two frames: one object per table (both names kept), the cup on its
    table, the plant pruned."""
    cup = ("cup", [-1.8, 0.8, 0.80], [0.12, 0.12, 0.10], (230, 230, 40))
    ex, masks, objs = make_export(d, extra=[cup])
    for f in sorted(os.listdir(masks)):
        i = int(f[:6])
        z = np.load(os.path.join(masks, f))
        ms, ls, ss = list(z["masks"]), list(z["labels"]), list(z["scores"])
        if i % 2:
            ls = ["desk" if l == "table" else l for l in ls]
        if i in (7, 8):                                # one patch of wall, called a plant twice
            e = ob.Export(ex)
            dep, T = e.depth(i), e.poses[i]
            v, u = np.mgrid[0:H, 0:W]
            pc = np.dstack([(u - CX) * dep / FX, (v - CY) * dep / FX, dep]).reshape(-1, 3)
            pw = pc @ T[:3, :3].T + T[:3, 3]
            m = ((dep.reshape(-1) > 0) & (pw[:, 0] > 3.9) & (np.abs(pw[:, 1]) < 0.4)
                 & (pw[:, 2] > 1.0) & (pw[:, 2] < 1.6)).reshape(H, W)
            if m.sum() > 50:
                ms.append(m); ls.append("plant"); ss.append(0.6)
        np.savez(os.path.join(masks, f), masks=np.array(ms, bool).reshape(-1, H, W),
                 labels=np.array(ls), scores=np.array(ss))
    out = os.path.join(d, "registry")
    ob.main([ex, out, "--prompts", "table,desk,chair,cup,plant", "--segmenter", "npz:" + masks,
             "--every", "1", "--min-points", "20", "--erode", "1", "--max-depth", "8"])
    before = json.load(open(os.path.join(out, "objects.json")))["objects"]
    assert sorted(r["label"] for r in before).count("desk") == 2, [r["id"] for r in before]
    cs.main([out, "merge"])
    reg = json.load(open(os.path.join(out, "objects.json")))["objects"]
    tables = [r for r in reg if set(r["labels"]) == {"table", "desk"}]
    assert len(tables) == 2, [(r["id"], r["labels"]) for r in reg]
    cupr = [r for r in reg if r["label"] == "cup"]
    assert len(cupr) == 1 and cupr[0].get("on") in {t["id"] for t in tables}, cupr
    near = min(tables, key=lambda t: np.linalg.norm(np.array(t["centroid"][:2]) - [-1.8, 0.8]))
    assert cupr[0]["on"] == near["id"]
    for r in reg:
        assert os.path.isfile(os.path.join(out, r["points"])), r["points"]
    vw.main([ex, out, "--min-pixels", "50"])
    cs.main([out, "prune"])
    reg = json.load(open(os.path.join(out, "objects.json")))["objects"]
    assert "plant" not in {r["label"] for r in reg}, [r["id"] for r in reg]
    assert sum(1 for r in reg if r["label"] in ("table", "desk")) == 2
    for r in reg:
        assert os.path.isfile(os.path.join(out, r["points"]))
        for b in r["best_for_3d"]:
            assert os.path.isfile(os.path.join(out, b["mask"])), b["mask"]
    pr = json.load(open(os.path.join(out, "objects_pruned.json")))["objects"]
    assert [p["label"] for p in pr] == ["plant"], [p["id"] for p in pr]
    cs.main([out, "prune", "--min-views", "100"])        # again, stricter: added, not replaced
    pr = json.load(open(os.path.join(out, "objects_pruned.json")))
    assert pr["objects"][0]["label"] == "plant" and len(pr["objects"]) == 5 and len(pr["runs"]) == 2


def test_parts_absorb_a_misnamed_piece_keep_a_thing_on_it(d):
    """One table's end is also called 'cabinet' in 2 of the ~8 frames it is in
    sight from (a quarter of the table, from some angles only); a cup on that table is found whenever in
    sight. parts: the cabinet goes into the table (a vote there), the cup
    stays, on the table."""
    cup = ("cup", [-1.8, 0.8, 0.80], [0.12, 0.12, 0.10], (230, 230, 40))
    ex, masks, objs = make_export(d, extra=[cup])
    e = ob.Export(ex)
    v, u = np.mgrid[0:H, 0:W]
    for i in (8, 13):
        z = np.load(os.path.join(masks, "%06d.npz" % i))
        ms, ls, ss = list(z["masks"]), list(z["labels"]), list(z["scores"])
        dep, T = e.depth(i), e.poses[i]
        pc = np.dstack([(u - CX) * dep / FX, (v - CY) * dep / FX, dep]).reshape(-1, 3)
        pw = pc @ T[:3, :3].T + T[:3, 3]
        end = ((dep.reshape(-1) > 0) & (pw[:, 0] < -2.1) & (pw[:, 0] > -2.45)
               & (np.abs(pw[:, 1] - 0.8) < 0.4) & (pw[:, 2] < 0.8)).reshape(H, W)
        if end.sum() > 30:
            ms.append(end); ls.append("cabinet"); ss.append(0.7)
        np.savez(os.path.join(masks, "%06d.npz" % i), masks=np.array(ms, bool).reshape(-1, H, W),
                 labels=np.array(ls), scores=np.array(ss))
    out = os.path.join(d, "registry")
    ob.main([ex, out, "--prompts", "table,chair,cup,cabinet", "--segmenter", "npz:" + masks,
             "--every", "1", "--min-points", "20", "--erode", "2"])    # the test masks bleed 2 px
    reg = json.load(open(os.path.join(out, "objects.json")))["objects"]
    assert "cabinet" in {r["label"] for r in reg}, [r["id"] for r in reg]
    cs.main([out, "merge"])
    vw.main([ex, out, "--min-pixels", "50"])
    cs.main([out, "parts"])
    reg = json.load(open(os.path.join(out, "objects.json")))["objects"]
    labels = sorted(r["label"] for r in reg)
    assert labels == ["chair", "cup", "table", "table"], labels
    near = min((r for r in reg if r["label"] == "table"),
               key=lambda t: np.linalg.norm(np.array(t["centroid"][:2]) - [-1.8, 0.8]))
    assert "cabinet" in near["labels"] and [p["label"] for p in near.get("parts", [])] == ["cabinet"]
    assert near["agreement"] > 0.5 and not near["uncertain"]
    cupr = [r for r in reg if r["label"] == "cup"][0]
    assert cupr.get("on") == near["id"], cupr.get("on")
    ab = json.load(open(os.path.join(out, "objects_absorbed.json")))["absorbed"]
    assert [a["label"] for a in ab] == ["cabinet"] and ab[0]["into"] == near["id"]
    for r in reg:
        assert os.path.isfile(os.path.join(out, r["points"]))


def test_shell_objects_found_on_walls():
    """A 3 m strip on a wall is shell; a 0.8 m patch on it, a cabinet with its
    front 0.5 m out, a strip 3 m long but mid-room are not."""
    rng = np.random.default_rng(0)
    room = (0.0, 0.0, 8.0, 0.0, 6.0)
    strip = np.c_[rng.uniform(1, 4, 2000), rng.normal(0.02, 0.01, 2000), rng.uniform(0.5, 1.5, 2000)]
    patch = np.c_[rng.uniform(1, 1.8, 500), rng.normal(0.02, 0.01, 500), rng.uniform(0.5, 1.5, 500)]
    cab = np.c_[rng.uniform(1, 3, 1000), rng.normal(0.5, 0.01, 1000), rng.uniform(0, 0.8, 1000)]
    mid = np.c_[rng.uniform(1, 4, 1000), rng.normal(3.0, 0.01, 1000), rng.uniform(0, 1, 1000)]
    corner = np.r_[np.c_[rng.uniform(0, 2, 1000), np.full(1000, 0.03), rng.uniform(0, 2, 1000)],
                   np.c_[np.full(1000, 0.03), rng.uniform(0, 2, 1000), rng.uniform(0, 2, 1000)]]
    assert cs.on_walls(strip, room)[0] and cs.on_walls(corner, room)[0]
    assert not cs.on_walls(patch, room)[0]
    assert not cs.on_walls(cab, room)[0] and not cs.on_walls(mid, room)[0]
    turned = strip @ np.array([[np.cos(0.1), -np.sin(0.1), 0], [np.sin(0.1), np.cos(0.1), 0], [0, 0, 1]]).T
    assert cs.on_walls(turned, (np.degrees(0.1),) + room[1:])[0]


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
