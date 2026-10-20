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
map's colours (each vertex the median of its 5 nearest map points', no normals so it is
shown unlit), named after the map, and its copy without the ceiling (auto:
the highest height peak) keeping the walls; `cut` making that copy from
the mesh already made; auto in a building of several floors taking the
ceiling over the robot (2.8 m), not shelf tops (1.6 m) or the roof (6.5 m). Then `view` on an 8 x 6 m room
with noisy walls and stray points, the floor missing in four places: a 1 m2
hole (filled, at the floor's height and colour), a 5 m2 hole over --max-hole
(left), under a box (blocked: left) and a stair opening onto a floor 1 m
lower (a void: left), and a courtyard with no ceiling over it (left), a bench in it removed
(a loose piece open to the sky) while the box stays (under the ceiling);
the sensor height found from the floor under the
poses, the strays gone, the walls flat, nothing added outside the room. Then
the mirror ghosts of a polished floor and wall (under the floor, behind the
wall: no direct ray from the path reaches them) and a loose 5 m2 sheet removed
by both mesh and view, the room's own surfaces kept. Then objects (a ball,
a rack of thin bars) meshed on their own, finer: their faces on them.

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
    # every vertex the median colour of its 5 nearest map points, not Poisson's average
    _, i = cKDTree(np.asarray(pc.points)).query(V, k=5)
    assert np.abs(C - np.median(np.asarray(pc.colors)[i], axis=1)).max() < 1.5 / 255
    top = V[:, 2] > ROOM[2] - 0.02
    assert top.sum() > 1000, top.sum()
    assert np.abs(np.median(C[top], 0) * 255 - [200, 120, 240][::-1]).max() < 15
    assert abs(r["ceil_z"] - (ROOM[2] - 0.10)) < 0.06, r["ceil_z"]
    n = o3d.io.read_triangle_mesh(r["noceil"])
    W = np.asarray(n.vertices)
    assert W[:, 2].max() < r["ceil_z"] + 0.05, W[:, 2].max()
    walls = (np.abs(W[:, 0]) < 0.03) & (W[:, 2] > 2.5)
    assert walls.sum() > 50, walls.sum()
    # cut: the copy without the ceiling from the mesh already made
    c = L.cut_file(r["out"], tr)
    Wc = np.asarray(o3d.io.read_triangle_mesh(c["out"]).vertices)
    assert abs(c["ceil_z"] - (ROOM[2] - 0.10)) < 0.06 and Wc[:, 2].max() < c["ceil_z"] + 0.05, c
    # auto in a building of several floors: the ceiling over the robot, not the roof
    rng = np.random.default_rng(5)
    z = np.concatenate([rng.normal(0, 0.01, 40000), rng.normal(1.6, 0.01, 20000),     # shelf tops
                        rng.normal(2.8, 0.01, 30000), rng.normal(6.5, 0.01, 60000),
                        rng.uniform(0, 6.5, 20000)])
    _, hc, hn = L.height_histogram(z)
    got = L.ceiling_height(hc, hn, above=0.0 + L.CEIL_MIN)
    assert abs(got - 2.8) < 0.05, got
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


def check_objects(tmp):
    """A 6 x 4 x 3 room with a ball (r 0.3 m) and a rack of five thin bars
    (r 2 cm, 1.2 m tall): meshed on their own (the default), the objects'
    faces lie on them (90 % of the vertices within 1.5 cm of the ball's
    surface; every bar meshed, its faces at its 2 cm radius, 90 % within 6 cm
    of its axis: the most a map of 2 cm points gives); with one Poisson for
    everything the ball gets the shells Poisson closes it with (75 %)."""
    import open3d as o3d
    rng = np.random.default_rng(4)
    g = lambda a, b: np.arange(a, b, 0.02)                     # noqa: E731
    pts = []
    for axis, val, ra, rb in ((2, 0.0, (0, 6), (0, 4)), (2, 3.0, (0, 6), (0, 4)),
                              (0, 0.0, (0, 4), (0, 3)), (0, 6.0, (0, 4), (0, 3)),
                              (1, 0.0, (0, 6), (0, 3)), (1, 4.0, (0, 6), (0, 3))):
        A, B = np.meshgrid(g(*ra), g(*rb), indexing="ij")
        o = [i for i in range(3) if i != axis]
        p = np.zeros((A.size, 3))
        p[:, o[0]], p[:, o[1]], p[:, axis] = A.ravel(), B.ravel(), val
        pts.append(p)
    t = np.linspace(0, 2 * np.pi, 60)                          # a loop round the room: both sides seen
    path = np.c_[3 + 2.4 * np.cos(t), 2 + 1.5 * np.sin(t), np.full_like(t, 1.2)]
    ball_c = np.array([2.0, 2.0, 0.8])
    u = rng.normal(size=(12000, 3))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    b = ball_c + 0.3 * u
    seen = (np.einsum("ij,kj->ik", u, path) - np.einsum("ij,ij->i", u, b)[:, None] > 0).any(1)
    pts.append(b[seen])                                        # what a sensor on the path can see of it
    bars = [np.array([3.2 + 0.3 * i, 1.5]) for i in range(5)]
    for b in bars:
        t, z = np.meshgrid(np.linspace(0, 2 * np.pi, 7, endpoint=False), g(0.0, 1.2))
        pts.append(np.c_[b[0] + 0.02 * np.cos(t.ravel()), b[1] + 0.02 * np.sin(t.ravel()), z.ravel()])
    P = np.concatenate(pts) + rng.normal(0, 0.003, (sum(len(p) for p in pts), 3))
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P))
    pc.colors = o3d.utility.Vector3dVector(rng.uniform(0.3, 0.7, P.shape))
    mp = os.path.join(tmp, "map_final_objects.pcd")
    o3d.io.write_point_cloud(mp, pc)
    tr = os.path.join(tmp, "traj_objects.txt")
    np.savetxt(tr, np.c_[t, path, np.zeros((len(t), 3)), np.ones(len(t))])

    def ball_fit(path):
        V = np.asarray(o3d.io.read_triangle_mesh(path).vertices)
        near = np.linalg.norm(V - ball_c, axis=1) < 0.5
        return float(np.mean(np.abs(np.linalg.norm(V[near] - ball_c, axis=1) - 0.3) < 0.015)), V
    r = L.map_mesh(mp, tr, os.path.join(tmp, "o_split.ply"))
    one = L.map_mesh(mp, tr, os.path.join(tmp, "o_one.ply"), objects=False)
    fs, V = ball_fit(r["out"])
    f1, _ = ball_fit(one["out"])
    assert r["object_faces"] > 0, r
    assert fs > 0.9 and fs > f1 + 0.1, ("objects' mesh off the ball", fs, "one Poisson", f1)
    for b in bars:
        on = (np.linalg.norm(V[:, :2] - b, axis=1) < 0.1) & (V[:, 2] > 0.2) & (V[:, 2] < 1.0)
        assert on.sum() > 30, ("bar not meshed", b, on.sum())
        w = np.linalg.norm(V[on, :2] - b, axis=1)
        assert abs(np.median(w) - 0.02) < 0.012 and np.percentile(w, 90) < 0.06, \
            ("bar not meshed on it", b, np.percentile(w, [50, 90]))
    return fs, f1


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
        print(f"ok ghosts: mesh {gm['ghost_faces']} ghost faces and {gm['piece_points']} loose points out, "
              f"view {gv['ghost_points']} ghost and {gv['piece_points']} loose points out, the room kept")
        fs, f1 = check_objects(tmp)
        print(f"ok objects: {100 * fs:.0f} % of the ball's vertices on it meshed on their own "
              f"({100 * f1:.0f} % with one Poisson), every bar meshed at its radius")
        rm = check_mesh(tmp)
        print(f"ok mesh: {rm['faces_kept']} faces, ceiling cut at z = {rm['ceil_z']:+.2f}, "
              f"{rm['noceil_faces']} faces without it")
        print(f"ok: {r['frames']} frames, camera {Wc}x{Hc}, {r['points']} points "
              f"({100 * r['coloured']:.0f}% coloured), mesh {r['faces_kept']}/{r['faces']} faces")
    finally:
        shutil.rmtree(tmp)


if __name__ == "__main__":
    main()
