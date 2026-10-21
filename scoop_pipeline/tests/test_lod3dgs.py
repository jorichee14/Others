#!/usr/bin/env python
"""Self-test for datasets/lod3dgs.py on a synthetic room.

A 6 x 4 x 3 m room with no ceiling: a dark floor and four walls of four
colours. Twelve cameras inside it at 1.5 m, turned all round and tilted a
little, 160 x 120, f 100, principal point off the centre (70.3, 64.6).
Their images and depth are rendered exactly (ray against plane) and written
as a replica export; the map is the five surfaces sampled every 2 cm.

It must: recentre the camera (the largest window centred on the principal
point) so that the written images, depth and poses agree with the room
rendered by the centred camera at the written pose; store depth the way
LOD-3DGS reads it (warped grey, alpha where there is depth); remove the
surface Poisson puts over the open top (no starting point above the walls,
every point near the map); give the points on each surface its colour.
Then `mesh` on the room with a ceiling, the map coloured: the mesh in the
map's colours (each vertex its nearest map point's, no normals so it is
shown unlit), named after the map, and its copy without the ceiling (auto:
the highest height peak) keeping the walls. Then `view` on an 8 x 6 m room
with noisy walls and stray points, the floor missing in four places: a 1 m2
hole (filled, at the floor's height and colour), a 5 m2 hole over --max-hole
(left), under a box (blocked: left) and a stair opening onto a floor 1 m
lower (a void: left), and a courtyard with no ceiling over it (left), a bench in it removed
(a loose piece open to the sky) while the box stays (under the ceiling);
the sensor height found from the floor under the
poses, the strays gone, the walls flat, nothing added outside the room. Then
the mirror ghosts of a polished floor and wall (under the floor, behind the
wall: no direct ray from the path reaches them) and a loose 5 m2 sheet removed
by both mesh and view, the room's own surfaces kept. And --ceil auto on a
5 m hall by a 3 m arcade: each ceiling off at its own height, the walls kept;
`sionna` on its mesh: floor, ceiling and walls apart, loaded and traced by
Sionna RT.

    python scoop_pipeline/tests/test_lod3dgs.py
"""
import os
import shutil
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "datasets"))

import cv2                                                          # noqa: E402
from scipy.spatial import cKDTree                                   # noqa: E402
from scipy.spatial.transform import Rotation                        # noqa: E402
import lod3dgs as L                                                 # noqa: E402

ROOM = np.array([6.0, 4.0, 3.0])
# (axis, value, BGR colour)
PLANES = [(2, 0.0, (40, 40, 40)), (0, 0.0, (30, 30, 220)), (0, 6.0, (30, 200, 30)),
          (1, 0.0, (220, 40, 30)), (1, 4.0, (30, 210, 220))]
W, H, F, CX, CY = 160, 120, 100.0, 70.3, 64.6
SCALE = 6553.5


def render(T_wc, W, H, fx, fy, cx, cy):
    """Exact BGR image and z-depth of the room (0 / black where nothing is hit)."""
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    d = np.stack([(u - cx) / fx, (v - cy) / fy, np.ones_like(u, float)], -1) @ T_wc[:3, :3].T
    o = T_wc[:3, 3]
    best = np.full((H, W), np.inf)
    img = np.zeros((H, W, 3), np.uint8)
    for a, c, col in PLANES:
        with np.errstate(divide="ignore", invalid="ignore"):
            s = (c - o[a]) / d[..., a]
        p = o + s[..., None] * d
        ok = (s > 0) & np.all((p >= -1e-9) & (p <= ROOM + 1e-9), -1) & (s < best)
        best[ok] = s[ok]
        img[ok] = col
    best[~np.isfinite(best)] = 0
    return img, best


def camera(x, y, yaw, pitch):
    f = np.array([np.cos(yaw) * np.cos(pitch), np.sin(yaw) * np.cos(pitch), np.sin(pitch)])
    r = np.cross(f, [0, 0, 1.0])
    r /= np.linalg.norm(r)
    T = np.eye(4)
    T[:3, :3] = np.stack([r, np.cross(f, r), f], 1)        # x right, y down, z forward
    T[:3, 3] = [x, y, 1.5]
    return T


def surfaces(step=0.02):
    g = lambda n: np.arange(step / 2, n, step)            # noqa: E731
    pts = []
    for a, c, _ in PLANES:
        o = [i for i in range(3) if i != a]
        A, B = np.meshgrid(g(ROOM[o[0]]), g(ROOM[o[1]]))
        p = np.zeros((A.size, 3))
        p[:, o[0]], p[:, o[1]], p[:, a] = A.ravel(), B.ravel(), c
        pts.append(p)
    return np.concatenate(pts)


def write_pcd(path, P):
    with open(path, "wb") as f:
        f.write(("VERSION 0.7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\n"
                 f"WIDTH {len(P)}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS {len(P)}\n"
                 "DATA binary\n").encode())
        P.astype("<f4").tofile(f)


def read_ply(path):
    with open(path, "rb") as f:
        head = b""
        while not head.endswith(b"end_header\n"):
            head += f.readline()
        n = int(head.split(b"element vertex ")[1].split()[0])
        dt = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("nx", "<f4"), ("ny", "<f4"),
                       ("nz", "<f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")])
        a = np.fromfile(f, dt, n)
    return np.stack([a["x"], a["y"], a["z"]], 1), np.stack([a["red"], a["green"], a["blue"]], 1)


def check_mesh(tmp):
    import open3d as o3d
    planes = PLANES + [(2, ROOM[2], (200, 120, 240))]
    pts, cols = [], []
    for a, c, col in planes:
        o = [i for i in range(3) if i != a]
        g = lambda n: np.arange(0.01, n, 0.02)            # noqa: E731
        A, B = np.meshgrid(g(ROOM[o[0]]), g(ROOM[o[1]]))
        p = np.zeros((A.size, 3))
        p[:, o[0]], p[:, o[1]], p[:, a] = A.ravel(), B.ravel(), c
        pts.append(p)
        cols.append(np.tile(np.array(col[::-1], float) / 255, (len(p), 1)))
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.concatenate(pts)))
    pc.colors = o3d.utility.Vector3dVector(np.concatenate(cols))
    mp = os.path.join(tmp, "map_final_mapping_20260101.pcd")
    o3d.io.write_point_cloud(mp, pc)
    tr = os.path.join(tmp, "traj.txt")
    np.savetxt(tr, [[i, 1 + 0.4 * i, 1.0 + 0.2 * i, 1.0, 0, 0, 0, 1] for i in range(10)])
    r = L.map_mesh(mp, tr, ceil="auto")
    assert os.path.basename(r["out"]) == "mesh_mapping_20260101.ply", r["out"]
    m = o3d.io.read_triangle_mesh(r["out"])
    V, C = np.asarray(m.vertices), np.asarray(m.vertex_colors)
    assert r["coloured"] and len(C) == len(V)
    assert not m.has_vertex_normals(), "normals written: viewers would light (grey) the colours"
    # every vertex in the colour of its nearest map point, not an average of several
    _, i = cKDTree(np.asarray(pc.points)).query(V)
    assert np.abs(C - np.asarray(pc.colors)[i]).max() < 1.5 / 255
    top = V[:, 2] > ROOM[2] - 0.02
    assert top.sum() > 1000, top.sum()
    assert np.abs(np.median(C[top], 0) * 255 - [200, 120, 240][::-1]).max() < 15
    assert abs(r["ceil_z"] - (ROOM[2] - 0.10)) < 0.06, r["ceil_z"]
    n = o3d.io.read_triangle_mesh(r["noceil"])
    W = np.asarray(n.vertices)
    assert W[:, 2].max() < r["ceil_z"] + 0.05, W[:, 2].max()
    walls = (np.abs(W[:, 0]) < 0.03) & (W[:, 2] > 2.5)
    assert walls.sum() > 50, walls.sum()
    return r


def check_view(tmp):
    import open3d as o3d
    rng = np.random.default_rng(1)
    g = lambda a, b: np.arange(a, b, 0.02)                     # noqa: E731
    X, Y = np.meshgrid(g(0, 8), g(0, 6), indexing="ij")
    F = np.c_[X.ravel(), Y.ravel(), np.zeros(X.size)]
    inside = lambda P, x0, x1, y0, y1: ((P[:, 0] > x0) & (P[:, 0] < x1)   # noqa: E731
                                         & (P[:, 1] > y0) & (P[:, 1] < y1))
    A, B, BOX, VOID = (1.5, 2.5, 3.5, 4.5), (4.5, 7.0, 3.6, 5.6), (1.0, 2.0, 0.8, 1.8), (3.5, 4.5, 0.8, 1.8)
    YARD = (5.5, 7.0, 0.8, 2.3)                                  # no ceiling over it: a courtyard
    for r in (A, B, BOX, VOID, YARD):
        F = F[~inside(F, *r)]
    floor_col = np.array([0.55, 0.45, 0.35])
    pts, cols = [F], [np.clip(floor_col + rng.normal(0, 0.03, (len(F), 3)), 0, 1)]
    for axis, val, col in ((0, 0.0, (0.8, 0.2, 0.2)), (0, 8.0, (0.2, 0.8, 0.2)),
                           (1, 0.0, (0.2, 0.2, 0.8)), (1, 6.0, (0.8, 0.8, 0.2))):
        a, b = np.meshgrid(g(0, 8 if axis else 6), g(0, 2.5), indexing="ij")
        W = np.zeros((a.size, 3))
        W[:, 1 - axis], W[:, 2], W[:, axis] = a.ravel(), b.ravel(), val + rng.normal(0, 0.02, a.size)
        pts.append(W)
        cols.append(np.tile(col, (len(W), 1)))
    bx, by = np.meshgrid(g(BOX[0], BOX[1]), g(BOX[2], BOX[3]), indexing="ij")
    pts.append(np.c_[bx.ravel(), by.ravel(), np.full(bx.size, 0.5)])        # the box's top
    cols.append(np.tile([0.3, 0.3, 0.3], (bx.size, 1)))
    vx, vy = np.meshgrid(g(VOID[0], VOID[1]), g(VOID[2], VOID[3]), indexing="ij")
    pts.append(np.c_[vx.ravel(), vy.ravel(), np.full(vx.size, -1.0)])      # the floor below
    cols.append(np.tile([0.5, 0.5, 0.5], (vx.size, 1)))
    cx, cy = np.meshgrid(g(0, 8), g(0, 6), indexing="ij")
    roof = np.c_[cx.ravel(), cy.ravel(), np.full(cx.size, 2.5)]
    roof = roof[~inside(roof, *YARD)]
    pts.append(roof)                                           # the ceiling, open over the yard
    cols.append(np.tile([0.9, 0.9, 0.9], (len(roof), 1)))
    ux, uy = np.meshgrid(g(6.0, 6.6), g(1.3, 1.7), indexing="ij")
    pts.append(np.c_[ux.ravel(), uy.ravel(), np.full(ux.size, 0.45)])      # a bench in the courtyard
    cols.append(np.tile([0.4, 0.3, 0.2], (ux.size, 1)))
    stray = np.c_[rng.uniform(1, 7, 300), rng.uniform(1, 5, 300), rng.uniform(0.8, 2.0, 300)]
    pts.append(stray)
    cols.append(np.tile([1.0, 1.0, 1.0], (300, 1)))
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.concatenate(pts)))
    pc.colors = o3d.utility.Vector3dVector(np.concatenate(cols))
    mp = os.path.join(tmp, "map_final_view_test.pcd")
    o3d.io.write_point_cloud(mp, pc)
    t = np.linspace(0, 2 * np.pi, 200)
    tr = os.path.join(tmp, "traj_view.txt")
    np.savetxt(tr, np.c_[t, 4 + 3 * np.cos(t), 3 + 2 * np.sin(t), np.full_like(t, 0.6),
                         np.zeros((len(t), 3)), np.ones(len(t))])
    r = L.map_view(mp, tr, max_hole=3.0)
    assert os.path.basename(r["out"]) == "view_view_test.pcd", r["out"]
    assert abs(r["sensor_height"] - 0.6) < 0.03, r["sensor_height"]
    assert r["open"] >= 1, r
    v = o3d.io.read_point_cloud(r["out"])
    P, C = np.asarray(v.points), np.asarray(v.colors)
    low = np.abs(P[:, 2]) < 0.04
    a_in = inside(P, 1.6, 2.4, 3.6, 4.4) & low
    assert a_in.sum() > 1200, ("hole A not filled", a_in.sum())
    assert np.abs(np.median(C[a_in], 0) - floor_col).max() < 0.05, np.median(C[a_in], 0)
    for name, rr in (("big hole", (4.6, 6.9, 3.7, 5.5)), ("under the box", (1.1, 1.9, 0.9, 1.7)),
                     ("stair opening", (3.6, 4.4, 0.9, 1.7)), ("courtyard", (5.6, 6.9, 0.9, 2.2))):
        n = (inside(P, *rr) & low).sum()
        assert n < 10, (name, "filled", n)
    assert ((P[:, 0] < -0.15) | (P[:, 0] > 8.15) | (P[:, 1] < -0.15) | (P[:, 1] > 6.15)).sum() == 0
    air = inside(P, 0.3, 7.7, 0.3, 5.7) & (P[:, 2] > 0.8) & (P[:, 2] < 2.3) & ~inside(P, 0.9, 2.1, 0.7, 1.9)
    assert air.sum() < 30, ("strays left", air.sum())
    bench = inside(P, 6.0, 6.6, 1.3, 1.7) & (np.abs(P[:, 2] - 0.45) < 0.05)
    assert bench.sum() == 0, ("a loose piece in the courtyard kept", bench.sum())
    box = inside(P, 1.0, 2.0, 0.8, 1.8) & (np.abs(P[:, 2] - 0.5) < 0.05)
    assert box.sum() > 1000, ("the box (furniture under the ceiling) removed", box.sum())
    wall = (np.abs(P[:, 0]) < 0.15) & (P[:, 2] > 0.3) & (P[:, 2] < 2.0) & (P[:, 1] > 0.5) & (P[:, 1] < 5.5)
    th = np.subtract(*np.percentile(P[wall, 0], [95, 5]))
    assert th < 0.02, ("wall not flat", th)
    return r, th


def check_ghosts(tmp):
    """A closed 6 x 4 x 3 room as a polished floor and wall would show it:
    a ghost of the ceiling 3 m under the floor and of a cabinet behind the
    x = 0 wall (no direct ray from the path reaches either), and a loose
    5 m2 sheet floating tilted in the room. mesh and view: all three gone, the
    room's own surfaces kept."""
    import open3d as o3d
    rng = np.random.default_rng(2)
    g = lambda a, b: np.arange(a, b, 0.02)                     # noqa: E731

    def sheet(axis, val, ra, rb):
        A, B = np.meshgrid(g(*ra), g(*rb), indexing="ij")
        o = [i for i in range(3) if i != axis]
        p = np.zeros((A.size, 3))
        p[:, o[0]], p[:, o[1]], p[:, axis] = A.ravel(), B.ravel(), val
        return p
    room = [sheet(2, 0.0, (0, 6), (0, 4)), sheet(2, 3.0, (0, 6), (0, 4)),
            sheet(0, 0.0, (0, 4), (0, 3)), sheet(0, 6.0, (0, 4), (0, 3)),
            sheet(1, 0.0, (0, 6), (0, 3)), sheet(1, 4.0, (0, 6), (0, 3))]
    ghost = [sheet(2, -3.0, (0, 6), (0, 4)),                                     # the ceiling, mirrored
             sheet(0, -1.0, (1, 2), (0.5, 1.5)), sheet(1, 1.0, (-1.0, -0.2), (0.5, 1.5)),
             sheet(1, 2.0, (-1.0, -0.2), (0.5, 1.5))]                             # a cabinet, behind x = 0
    pa = sheet(2, 0.0, (1.0, 3.5), (1.0, 3.0))
    pa[:, 2] = 1.6 + 0.2 * (pa[:, 0] - 1.0)
    panel = [pa]                                    # a loose 5 m2 sheet, tilted, floating (seen through glass)
    P = np.concatenate(room + ghost + panel)
    P += rng.normal(0, 0.003, P.shape)
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P))
    pc.colors = o3d.utility.Vector3dVector(rng.uniform(0.3, 0.7, P.shape))
    mp = os.path.join(tmp, "map_final_ghosts.pcd")
    o3d.io.write_point_cloud(mp, pc)
    tr = os.path.join(tmp, "traj_ghosts.txt")
    np.savetxt(tr, [[i, 0.8 + 0.4 * i, 1.0 + 0.15 * i, 1.2, 0, 0, 0, 1] for i in range(12)])

    def bad(V):
        return {"under the floor": int((V[:, 2] < -0.5).sum()), "behind the wall": int((V[:, 0] < -0.3).sum()),
                "panel": int(((V[:, 2] > 1.5) & (V[:, 2] < 2.2) & (V[:, 0] > 1.1) & (V[:, 0] < 3.4)
                              & (V[:, 1] > 1.1) & (V[:, 1] < 2.9)).sum())}
    floor = lambda V: int(((np.abs(V[:, 2]) < 0.03) & (V[:, 0] > 0.3) & (V[:, 0] < 5.7)   # noqa: E731
                           & (V[:, 1] > 0.3) & (V[:, 1] < 3.7)).sum())
    keep = L.map_mesh(mp, tr, os.path.join(tmp, "m_keep.ply"), ghosts=False, min_piece=0)
    clean = L.map_mesh(mp, tr, os.path.join(tmp, "m_clean.ply"))
    Vk = np.asarray(o3d.io.read_triangle_mesh(keep["out"]).vertices)
    Vc = np.asarray(o3d.io.read_triangle_mesh(clean["out"]).vertices)
    assert min(bad(Vk).values()) > 0, ("the scene has no ghost to remove", bad(Vk))
    assert max(bad(Vc).values()) == 0, ("mesh: left", bad(Vc))
    assert floor(Vc) > 0.97 * floor(Vk), ("mesh: floor lost", floor(Vc), floor(Vk))
    v = L.map_view(mp, tr, os.path.join(tmp, "v_clean.pcd"), snap=False, fill=False)
    Pv = np.asarray(o3d.io.read_point_cloud(v["out"]).points)
    assert max(bad(Pv).values()) == 0, ("view: left", bad(Pv))
    assert floor(Pv) > 0.97 * floor(P), ("view: floor lost", floor(Pv), floor(P))
    return clean, v


def check_body(tmp):
    """A closed 6 x 4 x 3 room the robot drives along y = 2 (LiDAR 0.8 m up),
    a sheet standing across its way (x = 3, wall to wall, 0.2-2.6 m: joined
    to both walls it crosses, as a sheet seen in glass) and a partition it
    drives past 0.35 m away: view and mesh clear the sheet whole, the walls,
    the floor and the partition stay."""
    import open3d as o3d
    rng = np.random.default_rng(7)
    g = lambda a, b: np.arange(a, b, 0.02)                     # noqa: E731

    def sheet(axis, val, ra, rb):
        A, B = np.meshgrid(g(*ra), g(*rb), indexing="ij")
        o = [i for i in range(3) if i != axis]
        p = np.zeros((A.size, 3))
        p[:, o[0]], p[:, o[1]], p[:, axis] = A.ravel(), B.ravel(), val
        return p
    room = [sheet(2, 0.0, (0, 6), (0, 4)), sheet(2, 3.0, (0, 6), (0, 4)),
            sheet(0, 0.0, (0, 4), (0, 3)), sheet(0, 6.0, (0, 4), (0, 3)),
            sheet(1, 0.0, (0, 6), (0, 3)), sheet(1, 4.0, (0, 6), (0, 3))]
    part = [sheet(1, 2.35, (4.0, 5.0), (0, 1.5))]                 # a partition, 0.35 m from the way
    across = [sheet(0, 3.0, (0, 4), (0.2, 2.6))]                  # a sheet across the way
    P = np.concatenate(room + part + across) + rng.normal(0, 0.003, (1, 3))
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P))
    pc.colors = o3d.utility.Vector3dVector(rng.uniform(0.3, 0.7, P.shape))
    mp = os.path.join(tmp, "map_final_body.pcd")
    o3d.io.write_point_cloud(mp, pc)
    tr = os.path.join(tmp, "traj_body.txt")
    x = np.arange(0.6, 5.4, 0.05)
    np.savetxt(tr, np.c_[np.arange(len(x)), x, np.full_like(x, 2.0), np.full_like(x, 0.8),
                         np.zeros((len(x), 3)), np.ones(len(x))])
    count = lambda V, x0, x1, y0, y1, z0, z1: int(((V[:, 0] > x0) & (V[:, 0] < x1) & (V[:, 1] > y0)  # noqa: E731
                                                   & (V[:, 1] < y1) & (V[:, 2] > z0) & (V[:, 2] < z1)).sum())
    v = L.map_view(mp, tr, os.path.join(tmp, "v_body.pcd"), snap=False, fill=False, body=0.25)
    Pv = np.asarray(o3d.io.read_point_cloud(v["out"]).points)
    m = L.map_mesh(mp, tr, os.path.join(tmp, "m_body.ply"), body=0.25)
    Vm = np.asarray(o3d.io.read_triangle_mesh(m["out"]).vertices)
    for name, V, ref in (("view", Pv, P), ("mesh", Vm, None)):
        assert count(V, 2.9, 3.1, 0.3, 3.7, 0.3, 2.5) == 0, (name, "the sheet across the way kept",
                                                              count(V, 2.9, 3.1, 0.3, 3.7, 0.3, 2.5))
        assert count(V, 4.1, 4.9, 2.3, 2.4, 0.1, 1.4) > 200, (name, "the partition removed")
        assert count(V, 0.5, 5.5, -0.05, 0.05, 0.3, 2.7) > 2000, (name, "a side wall removed")
        if ref is not None:
            fl = count(ref, 0.3, 5.7, 0.3, 3.7, -0.02, 0.02)
            assert count(V, 0.3, 5.7, 0.3, 3.7, -0.02, 0.02) > 0.97 * fl, (name, "the floor lost")
    return v["body_points"]


def check_sky(tmp):
    """A 10 x 8 m building round a 3 x 3 m courtyard (no ceiling over it),
    the robot driving round it, a slanted sheet in the courtyard joined to
    the courtyard's wall it reaches: view clears the courtyard's inside, the
    sheet with it, and keeps the courtyard's walls, the outer walls and the
    floor the robot drove on."""
    import open3d as o3d
    g = lambda a, b: np.arange(a, b, 0.02)                     # noqa: E731

    def sheet(axis, val, ra, rb):
        A, B = np.meshgrid(g(*ra), g(*rb), indexing="ij")
        o = [i for i in range(3) if i != axis]
        p = np.zeros((A.size, 3))
        p[:, o[0]], p[:, o[1]], p[:, axis] = A.ravel(), B.ravel(), val
        return p
    yard = lambda Q: (Q[:, 0] > 4) & (Q[:, 0] < 7) & (Q[:, 1] > 2) & (Q[:, 1] < 5)   # noqa: E731
    ceil = sheet(2, 3.0, (0, 10), (0, 8))
    ceil = ceil[~yard(ceil)]
    walls = [sheet(0, 0.0, (0, 8), (0, 3)), sheet(0, 10.0, (0, 8), (0, 3)), sheet(1, 0.0, (0, 10), (0, 3)),
             sheet(1, 8.0, (0, 10), (0, 3)), sheet(0, 4.0, (2, 5), (0, 3)), sheet(0, 7.0, (2, 5), (0, 3)),
             sheet(1, 2.0, (4, 7), (0, 3)), sheet(1, 5.0, (4, 7), (0, 3))]
    u, z = np.meshgrid(g(0, 1), g(0.5, 2.8), indexing="ij")
    xy = np.array([4.5, 2.5]) + u.ravel()[:, None] * np.array([2.49, 2.0])   # reaches the x = 7 wall
    slant = np.c_[xy, z.ravel()]
    floor = sheet(2, 0.0, (0, 10), (0, 8))
    P = np.concatenate([floor, ceil, *walls, slant])
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P))
    pc.colors = o3d.utility.Vector3dVector(np.full(P.shape, 0.5))
    mp = os.path.join(tmp, "map_final_sky.pcd")
    o3d.io.write_point_cloud(mp, pc)
    loop = np.concatenate([np.c_[np.arange(1.5, 8.5, 0.1), np.full(70, 1.0)],
                           np.c_[np.full(60, 8.5), np.arange(1.0, 7.0, 0.1)],
                           np.c_[np.arange(8.5, 1.5, -0.1), np.full(70, 7.0)],
                           np.c_[np.full(60, 1.5), np.arange(7.0, 1.0, -0.1)]])
    tr = os.path.join(tmp, "traj_sky.txt")
    np.savetxt(tr, np.c_[np.arange(len(loop)), loop, np.full(len(loop), 0.8),
                         np.zeros((len(loop), 3)), np.ones(len(loop))])
    v = L.map_view(mp, tr, os.path.join(tmp, "v_sky.pcd"), snap=False, fill=False, sky=True)
    V = np.asarray(o3d.io.read_point_cloud(v["out"]).points)
    count = lambda Q, x0, x1, y0, y1, z0, z1: int(((Q[:, 0] > x0) & (Q[:, 0] < x1) & (Q[:, 1] > y0)  # noqa: E731
                                                   & (Q[:, 1] < y1) & (Q[:, 2] > z0) & (Q[:, 2] < z1)).sum())
    inner = count(V, 4.8, 6.2, 2.8, 4.2, 0.4, 2.9)
    assert inner == 0, ("the courtyard's inside kept (the sheet)", inner)
    for name, box in (("the courtyard wall x = 7", (6.98, 7.02, 2.2, 4.8, 0.3, 2.7)),
                      ("the courtyard wall y = 2", (4.2, 6.8, 1.98, 2.02, 0.3, 2.7)),
                      ("an outer wall", (0.5, 9.5, -0.02, 0.02, 0.3, 2.7)),
                      ("the floor driven on", (0.5, 9.5, 0.3, 1.7, -0.02, 0.02))):
        assert count(V, *box) > 0.95 * count(P, *box), (name, count(V, *box), count(P, *box))
    return v["sky_points"]


def check_scans(tmp):
    """scan_clear: a closed 6 x 4 x 3 room scanned from a path (rays cast
    against the room alone, +-22 deg, the LiDAR turned 17 deg), the map
    holding a wall's reflection: a slanted sheet cutting through two walls.
    Of its points the scans saw (in view of 3 scans or more, away from the
    walls) 95 % go: the laser went through them; the room stays."""
    import open3d as o3d
    from scipy.spatial.transform import Rotation as Rot
    g = lambda a, b: np.arange(a, b, 0.02)                     # noqa: E731

    def sheet(axis, val, ra, rb):
        A, B = np.meshgrid(g(*ra), g(*rb), indexing="ij")
        o = [i for i in range(3) if i != axis]
        p = np.zeros((A.size, 3))
        p[:, o[0]], p[:, o[1]], p[:, axis] = A.ravel(), B.ravel(), val
        return p
    room = np.concatenate([sheet(2, 0.0, (0, 6), (0, 4)), sheet(2, 3.0, (0, 6), (0, 4)),
                           sheet(0, 0.0, (0, 4), (0, 3)), sheet(0, 6.0, (0, 4), (0, 3)),
                           sheet(1, 0.0, (0, 6), (0, 3)), sheet(1, 4.0, (0, 6), (0, 3))])
    u, z = np.meshgrid(g(0, 1), g(0.2, 2.6), indexing="ij")
    ghost = np.c_[np.array([2.0, 0.0]) + u.ravel()[:, None] * np.array([2.0, 4.0]), z.ravel()]
    P = np.concatenate([room, ghost])
    box = o3d.geometry.TriangleMesh.create_box(6, 4, 3)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(box))
    az, el = np.meshgrid(np.radians(np.arange(-180, 180, 0.5)), np.radians(np.arange(-22, 22, 1.5)))
    d = np.c_[(np.cos(el) * np.cos(az)).ravel(), (np.cos(el) * np.sin(az)).ravel(), np.sin(el).ravel()]
    R = Rot.from_euler("z", 17, degrees=True).as_matrix()
    scans = []
    for x in np.arange(0.6, 5.4, 0.2):
        t = np.array([x, 1.0, 0.8])
        hit = scene.cast_rays(o3d.core.Tensor(np.c_[np.tile(t, (len(d), 1)), d @ R.T].astype(np.float32)))
        r = hit["t_hit"].numpy()
        ok = np.isfinite(r)
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = R, t
        scans.append((T, d[ok] * r[ok, None]))
    gone, thr, at = L.scan_clear(P, scans, counts=True)
    g_room, g_ghost = gone[:len(room)], gone[len(room):]
    seen = (thr + at)[len(room):] >= 3
    away = np.min(np.abs(np.c_[ghost[:, 1], ghost[:, 1] - 4.0]), axis=1) > 0.3
    sel = seen & away
    assert sel.sum() > 500, ("too little of the sheet in view to test", int(sel.sum()))
    assert g_ghost[sel].mean() > 0.95, ("the reflection the laser went through kept", g_ghost[sel].mean())
    assert g_room.mean() < 0.005, ("the room cleared", g_room.mean())
    return g_ghost[sel].mean(), g_room.mean(), sel.mean()


def check_ceilings(tmp):
    """A 7 x 6 m hall 5 m high by a 5 x 6 m arcade 3 m high, the robot
    through both: --ceil auto takes off each one's own ceiling (the hall's
    at the highest peak, the arcade's under it) and keeps the walls below
    each, in view and mesh."""
    import open3d as o3d
    g = lambda a, b: np.arange(a, b, 0.02)                     # noqa: E731

    def sheet(axis, val, ra, rb):
        A, B = np.meshgrid(g(*ra), g(*rb), indexing="ij")
        o = [i for i in range(3) if i != axis]
        p = np.zeros((A.size, 3))
        p[:, o[0]], p[:, o[1]], p[:, axis] = A.ravel(), B.ravel(), val
        return p
    P = np.concatenate([sheet(2, 0.0, (0, 12), (0, 6)), sheet(2, 5.0, (0, 7), (0, 6)),
                        sheet(2, 3.0, (7, 12), (0, 6)), sheet(0, 0.0, (0, 6), (0, 5)),
                        sheet(0, 7.0, (0, 6), (3, 5)), sheet(0, 12.0, (0, 6), (0, 3)),
                        sheet(1, 0.0, (0, 7), (0, 5)), sheet(1, 6.0, (0, 7), (0, 5)),
                        sheet(1, 0.0, (7, 12), (0, 3)), sheet(1, 6.0, (7, 12), (0, 3))])
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P))
    pc.colors = o3d.utility.Vector3dVector(np.full(P.shape, 0.5))
    mp = os.path.join(tmp, "map_final_ceil.pcd")
    o3d.io.write_point_cloud(mp, pc)
    x = np.arange(1.0, 11.0, 0.1)
    tr = os.path.join(tmp, "traj_ceil.txt")
    np.savetxt(tr, np.c_[np.arange(len(x)), x, np.full(len(x), 3.0), np.full(len(x), 0.8),
                         np.zeros((len(x), 3)), np.ones(len(x))])
    count = lambda Q, x0, x1, z0, z1: int(((Q[:, 0] > x0) & (Q[:, 0] < x1) & (Q[:, 2] > z0)  # noqa: E731
                                           & (Q[:, 2] < z1)).sum())
    v = L.map_view(mp, tr, os.path.join(tmp, "v_ceil.pcd"), snap=False, fill=False, ceil="auto")
    V = np.asarray(o3d.io.read_point_cloud(v["out"]).points)
    m = L.map_mesh(mp, tr, os.path.join(tmp, "m_ceil.ply"), voxel=0.05, ceil="auto")
    N = np.asarray(o3d.io.read_triangle_mesh(m["noceil"]).vertices)
    for name, Q in (("view", V), ("mesh", N)):
        assert count(Q, 0.5, 6.5, 4.95, 5.1) == 0, (name, "the hall's ceiling kept")
        assert count(Q, 7.6, 11.9, 2.95, 3.1) == 0, (name, "the arcade's ceiling kept")
        assert count(Q, 0.5, 6.5, 3.5, 4.8) > 0, (name, "the hall's walls cut to the arcade's height")
        assert count(Q, 7.6, 11.9, 1.5, 2.8) > 0, (name, "the arcade's walls gone")
    return count(V, 7.6, 11.9, 2.95, 3.1), count(N, 7.6, 11.9, 2.95, 3.1)


def check_sionna(tmp):
    """sionna on the mesh of a 5 m hall by a 3 m arcade (points with 5 mm
    noise) holding a table, a cabinet, a round pillar and a lamp hanging from
    the hall's ceiling: the pillar with the walls, the lamp with the ceiling,
    the table and the cabinet objects, rebuilt flat (the table top within
    2 mm, at its height); the scene loaded by Sionna RT when installed (four shapes) and a
    floor bounce traced 2.05 ns after the direct path; walk starting 1.6 m
    over the floor at the first pose, facing along the path."""
    import open3d as o3d
    rng = np.random.default_rng(0)
    g = lambda a, b: np.arange(a, b, 0.02)                     # noqa: E731

    def sheet(axis, val, ra, rb):
        A, B = np.meshgrid(g(*ra), g(*rb), indexing="ij")
        o = [i for i in range(3) if i != axis]
        p = np.zeros((A.size, 3))
        p[:, o[0]], p[:, o[1]], p[:, axis] = A.ravel(), B.ravel(), val
        return p

    def box(x0, x1, y0, y1, z0, z1, top=True, bottom=False):
        f = [sheet(0, x0, (y0, y1), (z0, z1)), sheet(0, x1, (y0, y1), (z0, z1)),
             sheet(1, y0, (x0, x1), (z0, z1)), sheet(1, y1, (x0, x1), (z0, z1))]
        return np.concatenate(f + ([sheet(2, z1, (x0, x1), (y0, y1))] if top else [])
                              + ([sheet(2, z0, (x0, x1), (y0, y1))] if bottom else []))
    t_, z_ = np.meshgrid(np.arange(0, 2 * np.pi, 0.02 / 0.25), g(0, 5))
    pillar = np.c_[4.0 + 0.25 * np.cos(t_.ravel()), 1.5 + 0.25 * np.sin(t_.ravel()), z_.ravel()]
    table = np.concatenate([box(2, 3.2, 4, 4.8, 0.72, 0.75)]
                           + [box(x, x + 0.06, y, y + 0.06, 0, 0.72, top=False)
                              for x in (2.05, 3.09) for y in (4.05, 4.69)])
    cabinet = box(9.0, 9.6, 5.5, 5.98, 0, 1.2)
    lamp = np.concatenate([box(4.7, 5.3, 2.7, 3.3, 4.55, 4.7, top=False, bottom=True),
                           box(4.97, 5.03, 2.97, 3.03, 4.7, 5.0, top=False)])
    P = np.concatenate([sheet(2, 0.0, (0, 12), (0, 6)), sheet(2, 5.0, (0, 7), (0, 6)),
                        sheet(2, 3.0, (7, 12), (0, 6)), sheet(0, 0.0, (0, 6), (0, 5)),
                        sheet(0, 7.0, (0, 6), (3, 5)), sheet(0, 12.0, (0, 6), (0, 3)),
                        sheet(1, 0.0, (0, 7), (0, 5)), sheet(1, 6.0, (0, 7), (0, 5)),
                        sheet(1, 0.0, (7, 12), (0, 3)), sheet(1, 6.0, (7, 12), (0, 3)),
                        pillar, table, cabinet, lamp])
    P = P + rng.normal(0, 0.005, P.shape)
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P))
    pc.colors = o3d.utility.Vector3dVector(np.full(P.shape, 0.5))
    mp = os.path.join(tmp, "view_radio.ply")
    o3d.io.write_point_cloud(mp, pc)
    x = np.arange(1.0, 11.0, 0.1)
    tr = os.path.join(tmp, "traj_radio.txt")
    np.savetxt(tr, np.c_[np.arange(len(x)), x, np.full(len(x), 3.0), np.full(len(x), 0.8),
                         np.zeros((len(x), 3)), np.ones(len(x))])
    m = L.map_mesh(mp, tr, voxel=0.03, min_piece=0)
    assert os.path.basename(m["out"]) == "mesh_view_radio.ply", m["out"]
    r = L.sionna_mesh(m["out"], faces=20000, materials={"walls": ("brick", 0.1)})
    assert L.default_points(m["out"]) == mp
    parts = {}
    for n in ("floor", "ceiling", "walls", "objects"):
        q = o3d.io.read_triangle_mesh(os.path.join(os.path.dirname(r["scene"]), "meshes", n + ".ply"))
        assert len(q.triangles) and not q.has_vertex_colors(), n
        parts[n] = np.asarray(q.vertices)[np.asarray(q.triangles)].mean(1)
    inside = lambda Q, x0, x1, y0, y1, z0, z1: ((Q[:, 0] > x0) & (Q[:, 0] < x1) & (Q[:, 1] > y0)  # noqa: E731
                                               & (Q[:, 1] < y1) & (Q[:, 2] > z0) & (Q[:, 2] < z1)).sum()
    assert np.abs(parts["floor"][:, 2]).max() < 0.2, ("floor off the floor", parts["floor"][:, 2].max())
    for name, part, b_ in (("the pillar", "walls", (3.7, 4.3, 1.2, 1.8, 0.5, 4.5)),
                           ("the lamp", "ceiling", (4.6, 5.4, 2.6, 3.4, 4.5, 4.75)),
                           ("the table", "objects", (2.1, 3.1, 4.1, 4.7, 0.6, 0.8)),
                           ("the cabinet", "objects", (9.05, 9.55, 5.45, 5.95, 0.3, 1.25))):
        assert inside(parts[part], *b_) > 0, (name, "not with", part)
        for other in parts:
            if other != part:
                assert inside(parts[other], *b_) <= 0.05 * inside(parts[part], *b_), (name, "in", other)
    O = o3d.io.read_triangle_mesh(os.path.join(os.path.dirname(r["scene"]), "meshes", "objects.ply"))
    Vo = np.asarray(O.vertices)
    top = (np.abs(Vo[:, 2] - 0.75) < 0.03) & (Vo[:, 0] > 2.2) & (Vo[:, 0] < 3.0) & (Vo[:, 1] > 4.2) & \
        (Vo[:, 1] < 4.6)
    assert top.sum() > 50 and Vo[top, 2].std() < 0.002 and abs(np.median(Vo[top, 2]) - 0.75) < 0.005, \
        ("the table top not flat", Vo[top, 2].std(), np.median(Vo[top, 2]))
    one = o3d.io.read_triangle_mesh(r["view"])
    Vv, Cv = np.asarray(one.vertices), np.asarray(one.vertex_colors)
    grey = (np.abs(Cv[np.abs(Vv[:, 2]) < 0.02] - 0.8).max(1) < 0.01).mean()
    assert grey > 0.9, ("the floor not grey", grey)
    full = o3d.io.read_triangle_mesh(m["out"])
    assert len(one.triangles) > 0.8 * len(full.triangles), ("not at full resolution", len(one.triangles))
    assert os.path.exists(r["labels"])
    T = np.c_[x, np.full(len(x), 3.0), np.full(len(x), 0.8)]
    e, c = L.walk_start(Vv, T)
    assert np.allclose(e, [1.0, 3.0, 1.6], atol=0.05) and c[0] > e[0], ("walk start", e, c)
    xml = open(r["scene"]).read()
    assert 'value="brick"' in xml and 'value="wood"' in xml and 'value="concrete"' in xml, xml
    try:
        import sionna.rt as rt
    except ImportError:
        return r, None
    sc = rt.load_scene(r["scene"])
    assert set(sc.objects) == {"floor", "ceiling", "walls", "objects"}, set(sc.objects)
    sc.tx_array = rt.PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")
    sc.rx_array = rt.PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")
    sc.add(rt.Transmitter("tx", position=[5.5, 3.0, 1.5]))
    sc.add(rt.Receiver("rx", position=[10.5, 3.0, 1.5]))
    _, tau = rt.PathSolver()(sc, max_depth=1, refraction=False).cir(out_type="numpy")
    d = np.unique(np.round(tau[tau >= 0] * 1e9, 2))
    bounce = (np.hypot(5.0, 3.0) - 5.0) / 0.299792458
    assert d[0] == 0 and np.abs(d - bounce).min() < 0.05, ("no floor bounce", d[:6], bounce)
    return r, d[:4]


def main():
    tmp = tempfile.mkdtemp(prefix="lod3dgs_")
    try:
        rep = os.path.join(tmp, "replica_map_mapping_A_20260101")
        os.makedirs(os.path.join(rep, "results"))
        poses = [camera(x, y, yaw, pitch) for (x, y) in ((2.0, 1.5), (4.0, 2.5))
                 for yaw, pitch in zip(np.linspace(0, 2 * np.pi, 6, endpoint=False),
                                       (0.0, -0.2, 0.1, -0.3, 0.0, -0.15))]
        for i, T in enumerate(poses):
            img, D = render(T, W, H, F, F, CX, CY)
            cv2.imwrite(os.path.join(rep, "results", f"frame{i:06d}.jpg"), img,
                        [cv2.IMWRITE_JPEG_QUALITY, 100])
            cv2.imwrite(os.path.join(rep, "results", f"depth{i:06d}.png"),
                        np.rint(D * SCALE).astype(np.uint16))
        np.savetxt(os.path.join(rep, "traj.txt"), np.array([T.reshape(-1) for T in poses]))
        open(os.path.join(rep, "splatam_data_config.yaml"), "w").write(
            f"dataset_name: \"replica\"\ncamera_params:\n  image_height: {H}\n"
            f"  image_width: {W}\n  fx: {F}\n  fy: {F}\n  cx: {CX}\n  cy: {CY}\n"
            f"  png_depth_scale: {SCALE}\n  crop_edge: 0\n")
        M = surfaces()
        write_pcd(os.path.join(tmp, "map.pcd"), M)

        r = L.convert(rep, os.path.join(tmp, "map.pcd"))
        out = r["out"]
        assert os.path.basename(out) == "lod3dgs_map_mapping_A_20260101", out

        # camera: the largest window centred on (CX, CY)
        cam = open(os.path.join(out, "sparse/0/cameras.txt")).read().splitlines()[-1].split()
        Wc, Hc = int(cam[2]), int(cam[3])
        assert cam[1] == "PINHOLE" and (Wc, Hc) == (141, 109), cam
        assert abs(float(cam[6]) - (Wc - 1) / 2) < 1e-6 and abs(float(cam[7]) - (Hc - 1) / 2) < 1e-6

        # images, depth and poses agree with the room seen by the centred camera
        rows = [l.split() for l in open(os.path.join(out, "sparse/0/images.txt"))
                if l.strip() and not l.startswith("#")]
        assert len(rows) == len(poses)
        for row in rows:
            qw, qx, qy, qz, tx, ty, tz = map(float, row[1:8])
            T_cw = np.eye(4)
            T_cw[:3, :3] = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
            T_cw[:3, 3] = [tx, ty, tz]
            img, D = render(np.linalg.inv(T_cw), Wc, Hc, F, F, (Wc - 1) / 2, (Hc - 1) / 2)
            name = os.path.splitext(row[9])[0]
            got = cv2.imread(os.path.join(out, "images", row[9]))
            la = cv2.imread(os.path.join(out, "depths", name + ".png"), cv2.IMREAD_UNCHANGED)
            assert got.shape == (Hc, Wc, 3) and la.shape == (Hc, Wc, 4), (got.shape, la.shape)
            # away from colour edges (bilinear resampling blends across them)
            flat = cv2.erode((cv2.Laplacian(img.astype(np.float32), cv2.CV_32F) == 0)
                             .all(-1).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
            err = np.abs(got.astype(int) - img.astype(int)).max(-1)[flat]
            assert np.percentile(err, 99) < 12, (name, np.percentile(err, 99))
            assert (la[..., 0] == la[..., 1]).all() and (la[..., 0] == la[..., 2]).all()
            valid = la[..., 3] == 255
            assert np.mean(valid == (D > 0)) > 0.98, (name, np.mean(valid == (D > 0)))
            # 8-bit steps of 7.8 cm below 10 m (half of it), and the nearest pixel up
            # to half a pixel away (the depth's own slope across it)
            both = valid & (D > 0) & flat
            slope = np.max(np.abs(np.gradient(D)), 0)[both]
            derr = np.abs(L.unwarp_la(la[..., 0])[both] - D[both])
            assert np.mean(derr <= 0.04 + 0.75 * slope) > 0.99, (name, derr.max())

        # starting points: on the walls and floor, nothing over the open top
        P, rgb = read_ply(os.path.join(out, "sparse/0/points3D.ply"))
        assert P[:, 2].max() < ROOM[2] + 0.1, P[:, 2].max()
        dist, _ = cKDTree(M).query(P)
        assert dist.max() < 0.15 and np.percentile(dist, 95) < 0.03, (dist.max(),)
        assert r["faces_kept"] < r["faces"]
        nn = np.sort(cKDTree(P).query(P, k=2)[0][:, 1])
        assert 0.02 < np.median(nn) < 0.06, np.median(nn)
        assert r["coloured"] > 0.6, r["coloured"]
        for a, c, col in PLANES:
            o = [i for i in range(3) if i != a]
            on = ((np.abs(P[:, a] - c) < 0.01) & np.all(P[:, o] > 0.1, 1)
                  & np.all(P[:, o] < ROOM[o] - 0.1, 1) & (rgb != 128).any(1))
            assert on.sum() > 200, (a, c, on.sum())
            med = np.median(rgb[on].astype(int), 0)
            assert np.abs(med - np.array(col)[::-1]).max() < 15, (a, c, med, col)
        rv, th = check_view(tmp)
        print(f"ok view: sensor {rv['sensor_height']:.2f} m up, {rv['holes']} hole filled "
              f"({rv['filled_m2']:.1f} m2), {rv['skipped']} left, wall {th * 100:.1f} cm thick")
        gm, gv = check_ghosts(tmp)
        print(f"ok ghosts: mesh {gm['ghost_faces']} ghost and {gm['piece_faces']} loose faces out, "
              f"view {gv['ghost_points']} ghost and {gv['piece_points']} loose points out, the room kept")
        nb = check_body(tmp)
        print(f"ok body: a sheet across the robot's way cleared whole ({nb} points), "
              "walls, floor and a partition 0.35 m away kept")
        ns = check_sky(tmp)
        print(f"ok sky: a courtyard's inside cleared ({ns} points), a sheet joined to its wall with it; "
              "its walls, the outer walls and the floor kept")
        gs, gr, sv = check_scans(tmp)
        print(f"ok scans: {100 * gs:.0f} % of a reflection the scans saw through cleared "
              f"({100 * sv:.0f} % of it in their view), {100 * gr:.2f} % of the room")
        cv, cm = check_ceilings(tmp)
        print(f"ok ceilings: a 5 m hall's and a 3 m arcade's each cut at its own height ({cv} view points, "
              f"{cm} mesh vertices of the arcade's left), the walls under each kept")
        rs, ds = check_sionna(tmp)
        print(f"ok sionna: floor {rs['floor']}, ceiling {rs['ceiling']}, walls {rs['walls']}, "
              f"objects {rs['objects']} ({rs['n_objects']} of them); a pillar with the walls, a lamp with "
              "the ceiling, a table and a cabinet objects, the table top flat"
              + (f"; Sionna RT traced delays {ds} ns" if ds is not None
                 else " (Sionna RT not installed: not loaded)"))
        rm = check_mesh(tmp)
        print(f"ok mesh: {rm['faces_kept']} faces, ceiling cut at z = {rm['ceil_z']:+.2f}, "
              f"{rm['noceil_faces']} faces without it")
        print(f"ok: {r['frames']} frames, camera {Wc}x{Hc}, {r['points']} points "
              f"({100 * r['coloured']:.0f}% coloured), mesh {r['faces_kept']}/{r['faces']} faces")
    finally:
        shutil.rmtree(tmp)


if __name__ == "__main__":
    main()
