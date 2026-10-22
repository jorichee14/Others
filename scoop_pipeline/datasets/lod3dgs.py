#!/usr/bin/env python3
"""A Replica export (mcap_convert.py --format replica) plus the pass's map as a
LOD-3DGS scene (LetsGo, github.com/zhaofuq/LOD-3DGS; Cui et al., ACM TOG 43(6),
2024): COLMAP layout with known poses, no structure from motion.

    python scoop_pipeline/datasets/lod3dgs.py scene <replica dataset> --map <map .pcd/.ply>
        [<out scene>] [--every 1] [--voxel 0.02] [--poisson-depth 0] [--trim 0.10]
        [--spacing 0.04] [--color-views 300] [--jpeg-quality 95]
    python scoop_pipeline/datasets/lod3dgs.py mesh <map .pcd/.ply> --traj <TUM of the map>
        [<out mesh .ply>] [--ceil auto|Z] [--ceil-margin 0.10] [--voxel 0.02]
        [--poisson-depth 0] [--trim 0.10] [--show]
    python scoop_pipeline/datasets/lod3dgs.py sionna <mesh .ply, with its ceiling> [<out folder>]
        [--traj <TUM, the mesh's frame>] [--faces 0] [--split-objects] [--material walls=brick:0.1]
    python scoop_pipeline/datasets/lod3dgs.py walk <mesh .ply> [--traj <TUM, the mesh's frame>]
        [--at X,Y] [--faces 2000000] [--eye 1.6]

`mesh` is the mesh step alone, to look at: the map's Poisson mesh, trimmed,
in the map's colours (map_final_<tag>.pcd is colorized), normals turned to
the nearest pose of --traj (the trajectory the map was built from). Written
next to the map as mesh_<tag>.ply (map_final_<tag>.pcd; else <map>_mesh.ply),
with the height histogram of the map printed (z up). --ceil also writes a
viewing copy without the ceiling, <mesh>_noceil.ply: every face above Z
removed, or with "auto" above the ceiling over each place (the lowest
horizontal surface 2.3 m or more over the floor there, so a low arcade by a
tall hall loses its own) and the highest histogram peak, less --ceil-margin. Only a viewing copy, as 06_cut's: a scene keeps the ceiling,
the cameras see it.

Layout (what LOD-3DGS's Octree loader reads; train.py -s <scene> --use_lod):

    <scene>/images/frame000000.jpg     colour, principal point moved to the centre
    <scene>/depths/frame000000.png     depth for the depth loss: 8-bit grey + alpha
                                       (stored RGBA, R = G = B), grey = warped depth
                                       d / 20 below 10 m, 1 - 5 / d beyond
                                       (utils/image_utils.warpped_depth), alpha 255
                                       where there is depth
    <scene>/sparse/0/cameras.txt       one PINHOLE camera
    <scene>/sparse/0/images.txt        world-to-camera per frame (qw qx qy qz tx ty tz)
    <scene>/sparse/0/points3D.ply      the starting points: sampled from the mesh
    <scene>/mesh.ply                   the trimmed mesh they were sampled from
    <scene>/report.txt

Default <scene>: next to the dataset, replica_<rest> -> lod3dgs_<rest>.

The starting points follow the paper (Sec. 3.3, 4.1, App. B, Tab. 3): the map
is meshed with Poisson reconstruction (normals turned towards the nearest
camera), every face farther than --trim from a map point is removed (Poisson
closes open sides and holes with surfaces nothing saw), and the mesh is
sampled evenly at --spacing (4 cm in the paper; mesh points gave 2.5-3.4 dB
PSNR over the raw LiDAR points at the same spacing). Each point gets the mean
colour of the --color-views frames that see it (its depth agrees with the
frame's depth image); points no frame sees are grey.

LOD-3DGS takes the principal point at the image centre (it ignores cx, cy),
so every image and depth image is resampled onto the largest window centred
on (cx, cy): pixel u of the output is u - (W' - 1) / 2 + cx of the input.

The depth images are the export's (--depth-source map: dense, from the same
map). The poses are the export's traj.txt, so the map must be in their frame:
the map the export rendered its depth from.
"""
import argparse
import glob
import logging
import os
import re
import sys

import cv2
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

log = logging.getLogger("lod3dgs")


class ConvertError(RuntimeError):
    pass


# --------------------------------------------------------------------------- #
# the Replica export
# --------------------------------------------------------------------------- #

def read_replica(path):
    """(camera dict, Nx4x4 camera-to-world, colour paths, depth paths)."""
    cfg = os.path.join(path, "splatam_data_config.yaml")
    if not os.path.exists(cfg):
        raise ConvertError(f"{path}: no splatam_data_config.yaml, not a replica export")
    text = open(cfg).read()
    cam = {}
    for k in ("image_height", "image_width", "fx", "fy", "cx", "cy", "png_depth_scale"):
        m = re.search(rf"^\s*{k}:\s*([-0-9.eE+]+)", text, re.M)
        if m is None:
            raise ConvertError(f"{cfg}: no {k}")
        cam[k] = float(m.group(1))
    T = np.loadtxt(os.path.join(path, "traj.txt"), ndmin=2).reshape(-1, 4, 4)
    rgb = sorted(glob.glob(os.path.join(path, "results", "frame*.jpg")))
    dep = sorted(glob.glob(os.path.join(path, "results", "depth*.png")))
    if not (len(rgb) == len(dep) == len(T)) or not len(T):
        raise ConvertError(f"{path}: {len(rgb)} colour, {len(dep)} depth images and "
                           f"{len(T)} poses")
    return cam, T, rgb, dep


def centred_camera(cam):
    """(W', H', x offset, y offset): the largest window centred on (cx, cy);
    output pixel u is input pixel u + x offset."""
    W, H, cx, cy = cam["image_width"], cam["image_height"], cam["cx"], cam["cy"]
    Wc = int(np.floor(min(2 * cx + 1, 2 * (W - 1 - cx) + 1) + 1e-6))
    Hc = int(np.floor(min(2 * cy + 1, 2 * (H - 1 - cy) + 1) + 1e-6))
    if Wc < 16 or Hc < 16:
        raise ConvertError(f"principal point ({cx}, {cy}) leaves no centred window")
    return Wc, Hc, cx - (Wc - 1) / 2.0, cy - (Hc - 1) / 2.0


def recentre(img, Wc, Hc, ox, oy, interp):
    M = np.array([[1, 0, ox], [0, 1, oy]], np.float64)
    return cv2.warpAffine(img, M, (Wc, Hc), flags=interp | cv2.WARP_INVERSE_MAP,
                          borderMode=cv2.BORDER_REPLICATE)


def depth_la(d):
    """Depth (metres, 0 = none) -> the BGRA image LOD-3DGS reads as grey + alpha."""
    w = np.where(d < 10.0, d / 20.0, 1.0 - 5.0 / np.maximum(d, 1e-6))
    g = np.clip(np.rint(w * 255), 0, 255).astype(np.uint8)
    a = np.where(d > 0, 255, 0).astype(np.uint8)
    g[d <= 0] = 0
    return np.dstack([g, g, g, a])


def unwarp_la(g):
    """The inverse of depth_la's grey channel (uint8) -> metres; for checks."""
    w = g.astype(np.float64) / 255.0
    return np.where(w < 0.5, 20.0 * w, 5.0 / np.maximum(1.0 - w, 1e-6))


# --------------------------------------------------------------------------- #
# mesh and starting points
# --------------------------------------------------------------------------- #

def _o3d():
    try:
        import open3d as o3d
    except ImportError as e:
        raise ConvertError(f"meshing needs open3d ({e}); pip install open3d") from e
    return o3d


def build_mesh(map_path, centres, voxel=0.02, depth=0, trim=0.10):
    """Poisson mesh of the map, faces farther than trim from a map point removed.
    centres: camera positions, the side each normal is turned to."""
    o3d = _o3d()
    if isinstance(map_path, str):
        pc = o3d.io.read_point_cloud(os.path.expanduser(map_path))
    else:
        pc, map_path = map_path, "the view's points"
    n0 = len(pc.points)
    if not n0:
        raise ConvertError(f"{map_path}: no points")
    if voxel > 0:
        pc = pc.voxel_down_sample(voxel)
    pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(
        radius=max(4 * voxel, 0.05), max_nn=30))
    P, N = np.asarray(pc.points), np.asarray(pc.normals)
    _, near = cKDTree(centres).query(P)
    flip = np.einsum("ij,ij->i", N, centres[near] - P) < 0
    N[flip] *= -1
    pc.normals = o3d.utility.Vector3dVector(N)
    if depth <= 0:          # octree cells about two voxels across
        ext = float(np.max(P.max(0) - P.min(0))) * 1.1
        depth = int(np.clip(np.ceil(np.log2(ext / max(2 * voxel, 1e-3))), 8, 13))
    log.info(f"  map {map_path}: {n0} points -> {len(P)} at {voxel} m; Poisson depth {depth}")
    mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pc, depth=depth)
    F, V = np.asarray(mesh.triangles), np.asarray(mesh.vertices)
    nf = len(F)
    dist, _ = cKDTree(P).query(V[F].mean(1))
    mesh.remove_triangles_by_mask(dist > trim)
    mesh.remove_unreferenced_vertices()
    mesh.remove_degenerate_triangles()
    log.info(f"  mesh: {nf} faces, {nf - len(mesh.triangles)} farther than {trim} m "
             f"from the map removed")
    if not len(mesh.triangles):
        raise ConvertError("no mesh face is near the map: is the map the right one?")
    return mesh, {"map_points": n0, "map_voxel_points": len(P), "poisson_depth": depth,
                  "faces": nf, "faces_kept": len(mesh.triangles)}


def sphere_dirs(n, rng):
    """n directions evenly over the sphere (Fibonacci), turned at random."""
    i = np.arange(n) + 0.5
    z = 1 - 2 * i / n
    r = np.sqrt(1 - z * z)
    a = np.pi * (1 + 5 ** 0.5) * i
    d = np.c_[r * np.cos(a), r * np.sin(a), z]
    return d @ Rotation.random(random_state=rng).as_matrix().T


def seen_from_path(mesh, T, poses=1000, rays=50000, reach=0.25, seed=0):
    """Per face of the mesh: seen directly from the path. Rays in every
    direction from `poses` positions of T, each stopped by the first face it
    meets; a face is seen when it, or a face within `reach` of it, was hit.
    A mirror ghost (the room the LiDAR saw in polished stone or glass, put
    behind it) is reached by no direct ray: the mirror stops them."""
    o3d = _o3d()
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    F, V = np.asarray(mesh.triangles), np.asarray(mesh.vertices)
    hit = np.zeros(len(F), bool)
    rng = np.random.default_rng(seed)
    O = T[np.linspace(0, len(T) - 1, min(poses, len(T))).astype(int)]
    for k in range(0, len(O), 20):
        orig = np.repeat(O[k:k + 20], rays, 0)
        dirs = np.concatenate([sphere_dirs(rays, rng) for _ in range(len(O[k:k + 20]))])
        ans = scene.cast_rays(o3d.core.Tensor(np.hstack([orig, dirs]).astype(np.float32)))
        ids = ans["primitive_ids"].numpy()
        hit[ids[ids != o3d.t.geometry.RaycastingScene.INVALID_ID]] = True
    cen = V[F].mean(1)
    if not hit.any():
        return hit
    d, _ = cKDTree(cen[hit]).query(cen, distance_upper_bound=reach, workers=-1)
    return np.isfinite(d)


def remove_ghosts(mesh, T, **kw):
    """(the mesh without the faces no direct ray from the path reaches, faces removed)."""
    seen = seen_from_path(mesh, T, **kw)
    out = type(mesh)(mesh)
    out.remove_triangles_by_mask(~seen)
    out.remove_unreferenced_vertices()
    return out, int((~seen).sum())


def ghost_points(P, T, voxel=0.06, near=0.12, **kw):
    """Per point: on a mirror ghost. A coarse mesh of the points is tested
    with seen_from_path; a point whose nearest face (within `near`) no direct
    ray reaches is a ghost; a point with no face near is kept."""
    o3d = _o3d()
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P))
    m, _ = build_mesh(pc, T, voxel=voxel, trim=2 * voxel)
    seen = seen_from_path(m, T, **kw)
    cen = np.asarray(m.vertices)[np.asarray(m.triangles)].mean(1)
    d, i = cKDTree(cen).query(P, distance_upper_bound=near, workers=-1)
    ok = np.isfinite(d)
    ghost = np.zeros(len(P), bool)
    ghost[ok] = ~seen[i[ok]]
    return ghost


def loose(ids, X, area, T, h, min_area=50.0, roof_h=2.0, float_h=1.2, cell=0.4):
    """Per piece (ids: each element's piece, X: its position, area: per piece
    m2): to remove. The pieces of min_area or more are the building; any
    other is removed when it is open to the sky (under half of it has the
    building's ceiling over it: points roof_h or more above the floor in its
    0.4 m columns; a courtyard, a facade seen through glass) or when it
    floats (its bottom float_h or more above the floor: a lamp, a sign, a
    reflection, a sheet seen through glass). One under the ceiling and down
    at the floor is furniture: kept. The floor under each element is the
    nearest pose's height less the sensor height h."""
    small = area < min_area
    if not small.any():
        return small
    _, near = cKDTree(T[:, :2]).query(X[:, :2], workers=-1)
    up = X[:, 2] - (T[near, 2] - h)                                 # height above the floor
    lo = X[:, :2].min(0)
    ij = np.floor((X[:, :2] - lo) / cell).astype(np.int64)
    ny = int(ij[:, 1].max()) + 1
    col = ij[:, 0] * ny + ij[:, 1]
    big = ~small[ids]
    roofed = np.zeros(int(col.max()) + 1, bool)
    roofed[col[big & (up >= roof_h)]] = True
    n = len(area)
    cnt = np.bincount(ids, minlength=n)
    sky = np.bincount(ids, (~roofed[col]).astype(float), n) / np.maximum(cnt, 1) > 0.5
    bottom = np.full(n, np.inf)
    np.minimum.at(bottom, ids, up)
    return small & (sky | (bottom >= float_h))


def drop_pieces(mesh, T, h, min_area=50.0):
    """(the mesh without its loose pieces (see loose()), faces removed)."""
    tri, _, area = mesh.cluster_connected_triangles()
    tri, area = np.asarray(tri), np.asarray(area)
    cen = np.asarray(mesh.vertices)[np.asarray(mesh.triangles)].mean(1)
    gone = loose(tri, cen, area, T, h, min_area)[tri]
    out = type(mesh)(mesh)
    out.remove_triangles_by_mask(gone)
    out.remove_unreferenced_vertices()
    return out, int(gone.sum())


def piece_points(P, T, h, min_area=50.0, cell=0.1):
    """Per point: in a loose piece (see loose()); a piece is the points of
    10 cm cells touching, its area their count x cell^2."""
    from scipy import ndimage
    lo = P.min(0)
    while np.prod(np.ceil((P.max(0) - lo) / cell) + 1) > 4e8:
        cell *= 1.5
    ijk = np.floor((P - lo) / cell).astype(np.int64)
    occ = np.zeros(tuple(ijk.max(0) + 1), bool)
    occ[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = True
    lab, n = ndimage.label(occ, structure=np.ones((3, 3, 3), bool))
    area = np.bincount(lab.ravel(), minlength=n + 1) * cell * cell
    ids = lab[ijk[:, 0], ijk[:, 1], ijk[:, 2]]
    return loose(ids, P, area, T, h, min_area)[ids]


def read_tum_positions(path, stride=10):
    """Positions (Nx3) of a TUM trajectory (t x y z qx qy qz qw), every stride-th."""
    a = np.loadtxt(os.path.expanduser(path), comments="#", ndmin=2)
    if a.shape[1] < 4 or not len(a):
        raise ConvertError(f"{path}: not a TUM trajectory")
    return a[::max(1, int(stride)), 1:4]


def height_histogram(z, step=0.05, width=50):
    """(text, bin centres, counts) of the heights; the text one line a bin, top first."""
    edges = np.arange(np.floor(z.min() / step) * step, z.max() + step, step)
    counts, edges = np.histogram(z, bins=edges)
    centres = 0.5 * (edges[:-1] + edges[1:])
    peak = max(int(counts.max()), 1)
    lines = [f"  z={c:+7.2f} | {'#' * int(round(width * n / peak)):<{width}} {n:>10,d}"
             for c, n in zip(centres[::-1], counts[::-1]) if n]
    return "\n".join(lines), centres, counts


def ceiling_height(centres, counts):
    """The highest strong peak of the histogram: the largest bin of the top 40 %
    of the height range."""
    top = centres >= centres[0] + 0.6 * (centres[-1] - centres[0])
    return float(centres[top][np.argmax(counts[top])])


def cut_above(mesh, z):
    """A copy of the mesh without the faces whose centre is above z: a
    height, or a function of the centres giving one each."""
    out = type(mesh)(mesh)
    F, V = np.asarray(out.triangles), np.asarray(out.vertices)
    c = V[F].mean(1)
    out.remove_triangles_by_mask(c[:, 2] > (z(c) if callable(z) else z))
    out.remove_unreferenced_vertices()
    return out


def local_ceiling(P, T, h, cell=0.5, roof_h=2.3, flat=0.9, layer=0.05, share=0.3, edge=1):
    """The ceiling over each place, as a function of points giving the height
    of the ceiling over each (inf where none: a courtyard). Per column (cell
    m, in plan): the lowest layer (layer m thick) of horizontal surfaces
    (|n_z| over flat) roof_h or more above the floor holding share of the
    column's densest one; then the lowest of the columns within `edge`, so
    the wall a ceiling meets is cut with it. A building of several heights
    (a low arcade by a tall hall) has each cut at its own."""
    from scipy import ndimage
    o3d = _o3d()
    lo = P[:, :2].min(0)
    nx, ny = np.floor((P[:, :2].max(0) - lo) / cell).astype(np.int64) + 1
    grid = np.full((nx, ny), np.inf)
    _, near = cKDTree(T[:, :2]).query(P[:, :2], workers=-1)
    hi = np.flatnonzero(P[:, 2] - (T[near, 2] - h) >= roof_h)
    if len(hi):
        q = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P[hi]))
        q.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=20))
        H = P[hi[np.abs(np.asarray(q.normals)[:, 2]) > flat]]
        if len(H):
            ij = np.floor((H[:, :2] - lo) / cell).astype(np.int64)
            col = ij[:, 0] * ny + ij[:, 1]
            zb = np.floor(H[:, 2] / layer).astype(np.int64)
            pairs, n = np.unique(np.c_[col, zb], axis=0, return_counts=True)   # by column, then up
            first = np.r_[0, np.flatnonzero(np.diff(pairs[:, 0])) + 1]
            top = np.maximum.reduceat(n, first)
            ok = n >= np.maximum(3, share * np.repeat(top, np.diff(np.r_[first, len(n)])))
            c, i = np.unique(pairs[ok, 0], return_index=True)
            grid.ravel()[c] = pairs[ok][i, 1] * layer
    if edge:
        grid = ndimage.minimum_filter(grid, size=2 * edge + 1, mode="constant", cval=np.inf)

    def at(X):
        k = np.floor((np.asarray(X)[:, :2] - lo) / cell).astype(np.int64)
        inside = (k[:, 0] >= 0) & (k[:, 0] < nx) & (k[:, 1] >= 0) & (k[:, 1] < ny)
        z = np.full(len(k), np.inf)
        z[inside] = grid[k[inside, 0], k[inside, 1]]
        return z
    return at


def default_mesh_out(map_path):
    d, name = os.path.split(os.path.abspath(map_path))
    stem = os.path.splitext(name)[0]
    if stem.startswith("map_final_"):
        return os.path.join(d, "mesh_" + stem[len("map_final_"):] + ".ply")
    if stem.startswith("view_"):
        return os.path.join(d, "mesh_" + stem + ".ply")
    return os.path.join(d, stem + "_mesh.ply")


def crisp_colours(mesh, map_path, subdivide=1):
    """The mesh split `subdivide` times (each face into four) and every vertex
    given the colour of the map's nearest point, at full resolution: Poisson
    averages the thinned map's colours, which blurs and greys them."""
    o3d = _o3d()
    if subdivide > 0:
        mesh = mesh.subdivide_midpoint(number_of_iterations=int(subdivide))
    src = o3d.io.read_point_cloud(os.path.expanduser(map_path))
    if not src.has_colors():
        return mesh, False
    _, idx = cKDTree(np.asarray(src.points)).query(np.asarray(mesh.vertices), workers=-1)
    mesh.vertex_colors = o3d.utility.Vector3dVector(np.asarray(src.colors)[idx])
    return mesh, True


def map_mesh(map_path, traj, out=None, *, voxel=0.02, poisson_depth=0, trim=0.10,
         ceil=None, ceil_margin=0.10, show=False, subdivide=1, normals=False, ghosts=True,
         min_piece=50.0, body=0.0):
    """The mesh step alone: mesh of the map (+ a copy without the ceiling),
    each vertex in the colour of the map's nearest point; written without
    normals unless asked, so a viewer shows the colours unlit."""
    o3d = _o3d()
    out = out or default_mesh_out(map_path)
    centres = read_tum_positions(traj)
    src, nbody_pts = map_path, 0
    if body > 0:
        pc = o3d.io.read_point_cloud(os.path.expanduser(map_path))
        P0 = np.asarray(pc.points)
        T0 = read_tum_positions(traj, stride=1)
        g = body_clear(P0, T0, sensor_height(P0, T0)[0], body)
        src, nbody_pts = pc.select_by_index(np.flatnonzero(~g)), int(g.sum())
    m, mstat = build_mesh(src, centres, voxel, poisson_depth, trim)
    nghost = 0
    if ghosts:
        m, nghost = remove_ghosts(m, read_tum_positions(traj, stride=1))
    Tf = read_tum_positions(traj, stride=1)
    hs, _ = sensor_height(np.asarray(m.vertices), Tf)
    nbody = 0
    if body > 0:
        cen = np.asarray(m.vertices)[np.asarray(m.triangles)].mean(1)
        gone = in_body(cen, Tf, hs, body)
        m.remove_triangles_by_mask(gone)
        m.remove_unreferenced_vertices()
        nbody = int(gone.sum())
    npiece = 0
    if min_piece > 0:
        m, npiece = drop_pieces(m, Tf, hs, min_piece)
    m, _ = crisp_colours(m, map_path, subdivide)
    mstat["faces_kept"] = len(m.triangles)
    if normals:
        m.compute_vertex_normals()      # from the split faces (the midpoints' would not be unit)
    else:
        m.vertex_normals = o3d.utility.Vector3dVector(np.zeros((0, 3)))
        m.triangle_normals = o3d.utility.Vector3dVector(np.zeros((0, 3)))
    o3d.io.write_triangle_mesh(out, m)
    V = np.asarray(m.vertices)
    text, hc, hn = height_histogram(V[:, 2])
    log.info("height of the mesh (5 cm bins, top first):\n" + text)
    res = {"out": out, "vertices": len(V), "coloured": m.has_vertex_colors(), **mstat}
    report = (f"mesh {out}: {mstat['faces_kept']} faces ({mstat['faces']} from Poisson before the "
              f"{trim} m trim{f', split {subdivide}x' if subdivide else ''}), {len(V)} vertices, "
              f"{'each in its nearest map point colour' if m.has_vertex_colors() else 'no colour (map has none)'}"
              f"{'' if normals else ', no normals (shown unlit)'}"
              + (f"\n  mirror ghosts (faces no direct ray from the path reaches) removed: {nghost} faces"
                 if ghosts else "")
              + (f"\n  what the robot drove through ({body} m round its path, floor to over the LiDAR; "
                 f"a flat surface it crossed, whole) removed: {nbody_pts} points, then {nbody} faces"
                 if body > 0 else "")
              + (f"\n  loose pieces (not the building) open to the sky or floating removed: "
                 f"{npiece} faces" if min_piece > 0 else ""))
    res.update(ghost_faces=nghost, piece_faces=npiece, body_faces=nbody)
    shown = m
    if ceil is not None:
        if str(ceil) == "auto":
            peak = ceiling_height(hc, hn)
            z = peak - ceil_margin
            here = local_ceiling(np.asarray(m.vertices), Tf, hs)
            how = (f"or below each place's own ceiling, less {ceil_margin} m; "
                   f"the highest, peak {peak:+.2f} less {ceil_margin} m")
            nc = cut_above(m, lambda c: np.minimum(z, here(c) - ceil_margin))
        else:
            z = float(ceil)
            how = "given"
            nc = cut_above(m, z)
        if min_piece > 0:
            nc, _ = drop_pieces(nc, Tf, hs, min_piece)      # what hung from the ceiling now floats
        p = os.path.splitext(out)[0] + "_noceil.ply"
        o3d.io.write_triangle_mesh(p, nc)
        report += (f"\nno ceiling {p}: faces above z = {z:+.2f} ({how}) removed, "
                   f"{len(nc.triangles)} kept")
        res.update(noceil=p, ceil_z=z, noceil_faces=len(nc.triangles))
        shown = nc
    log.info(report)
    if show:
        shown.compute_vertex_normals()
        o3d.visualization.draw_geometries([shown], window_name=os.path.basename(out))
    return res


def default_view_out(map_path):
    d, name = os.path.split(os.path.abspath(map_path))
    stem = os.path.splitext(name)[0]
    if stem.startswith("map_final_"):
        return os.path.join(d, "view_" + stem[len("map_final_"):] + ".pcd")
    return os.path.join(d, stem + "_view.pcd")


def snap_planes(P, cell=0.4, sig_max=0.04, thin=0.35, min_pts=30):
    """(P with the points of every flat cell moved onto the cell's plane, the
    number moved). A cell of edge `cell` is flat when its points' smallest
    spread is under sig_max and under thin x the next: a wall or a floor, not
    a corner, an edge or an object."""
    lo = P.min(0)
    k = np.floor((P - lo) / cell).astype(np.int64)
    dims = k.max(0) + 1
    _, inv, cnt = np.unique((k[:, 0] * dims[1] + k[:, 1]) * dims[2] + k[:, 2],
                            return_inverse=True, return_counts=True)
    inv = inv.ravel()
    m_ = len(cnt)
    X = P - lo
    mu = np.stack([np.bincount(inv, X[:, i], m_) for i in range(3)], 1) / cnt[:, None]
    C = np.empty((m_, 3, 3))
    for i in range(3):
        for j in range(i, 3):
            C[:, i, j] = C[:, j, i] = (np.bincount(inv, X[:, i] * X[:, j], m_) / cnt
                                       - mu[:, i] * mu[:, j])
    w, v = np.linalg.eigh(C)
    sig = np.sqrt(np.clip(w, 0, None))
    flat = (cnt >= min_pts) & (sig[:, 0] < sig_max) & (sig[:, 0] < thin * sig[:, 1])
    on = flat[inv]
    n = v[inv[on], :, 0]
    d = np.einsum("ij,ij->i", X[on] - mu[inv[on]], n)
    Q = P.copy()
    Q[on] -= d[:, None] * n
    return Q, int(on.sum())


def in_body(X, T, h, radius=0.25, above=0.2, floor_gap=0.05):
    """Per point of X: inside the space the robot's body took along its path
    (within radius, in plan, of a pose of T, the LiDAR's track; from
    floor_gap over the floor, the pose's height less the sensor height h, to
    `above` over the LiDAR). The robot drove there: nothing solid is, so
    what the map has there (a sheet seen in glass that cuts across the way,
    joined to the walls it crosses) is not real."""
    out = np.zeros(len(X), bool)
    if radius <= 0 or not len(X):
        return out
    tree = cKDTree(T[:, :2])
    for k in range(0, len(X), 2_000_000):
        x = X[k:k + 2_000_000]
        d, i = tree.query(x[:, :2], k=8, distance_upper_bound=radius, workers=-1)
        ok = np.isfinite(d)
        i = np.where(ok, i, 0)
        z = x[:, 2][:, None]
        inside = ok & (z > T[i, 2] - h + floor_gap) & (z < T[i, 2] + above)
        out[k:k + 2_000_000] = inside.any(1)
    return out


def body_clear(P, T, h, radius=0.25, cell=0.4, sig_max=0.04, thin=0.35, min_pts=30,
               angle=15.0, gap=0.06):
    """Per point: to clear, as not real, because the robot drove through it:
    the points in its body's space (in_body), and every flat surface the
    robot crossed, whole (a sheet seen in glass that cuts across the way and
    joins the walls it crosses). A surface: flat 0.4 m cells (as snap_planes
    finds them) joined while their normals agree to `angle` and their planes
    to `gap`; crossed: it holds points in the body's space and the poses
    there lie on both sides of it. A wall the robot drove along has them on
    one side only: kept."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    hit = in_body(P, T, h, radius)
    if not hit.any():
        return hit
    lo = P.min(0)
    k = np.floor((P - lo) / cell).astype(np.int64)
    dims = k.max(0) + 3
    k += 1
    key = (k[:, 0] * dims[1] + k[:, 1]) * dims[2] + k[:, 2]
    K, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    inv = inv.ravel()
    m_ = len(K)
    X = P - lo
    mu = np.stack([np.bincount(inv, X[:, i], m_) for i in range(3)], 1) / cnt[:, None]
    C = np.empty((m_, 3, 3))
    for i in range(3):
        for j in range(i, 3):
            C[:, i, j] = C[:, j, i] = (np.bincount(inv, X[:, i] * X[:, j], m_) / cnt
                                       - mu[:, i] * mu[:, j])
    w, v = np.linalg.eigh(C)
    sig = np.sqrt(np.clip(w, 0, None))
    flat = (cnt >= min_pts) & (sig[:, 0] < sig_max) & (sig[:, 0] < thin * sig[:, 1])
    n = v[:, :, 0]
    kc = np.stack([K // (dims[1] * dims[2]), (K // dims[2]) % dims[1], K % dims[2]], 1)
    rows, cols = [], []
    for o in [(a, b, c) for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1)
              if (a, b, c) > (0, 0, 0)]:
        nb = ((kc[:, 0] + o[0]) * dims[1] + kc[:, 1] + o[1]) * dims[2] + kc[:, 2] + o[2]
        j = np.minimum(np.searchsorted(K, nb), m_ - 1)
        ok = (K[j] == nb) & flat & flat[j]
        i = np.flatnonzero(ok)
        j = j[ok]
        same = (np.abs(np.einsum("ij,ij->i", n[i], n[j])) > np.cos(np.radians(angle))) & \
            (np.abs(np.einsum("ij,ij->i", mu[j] - mu[i], n[i])) < gap)
        rows.append(i[same])
        cols.append(j[same])
    r_, c_ = np.concatenate(rows), np.concatenate(cols)
    _, seg = connected_components(coo_matrix((np.ones(len(r_)), (r_, c_)), shape=(m_, m_)), directed=False)
    out = hit.copy()
    hit_cells = np.unique(inv[hit])
    ptree = cKDTree(T[:, :2])
    for sgi in np.unique(seg[hit_cells[flat[hit_cells]]]):
        cells = np.flatnonzero((seg == sgi) & flat)
        hc = np.intersect1d(cells, hit_cells)
        c0 = hc[0]
        near = np.unique(np.concatenate(ptree.query_ball_point(mu[hc, :2] + lo[:2], radius + cell)))
        if not len(near):
            continue
        sd = (T[near] - (mu[c0] + lo)) @ n[c0]
        if sd.min() < -0.05 and sd.max() > 0.05:          # the robot went through it
            out |= np.isin(inv, cells)
            # its edges, in the cells it shares with a wall or the floor: the
            # points on its plane that face as it does (the wall's face across)
            nbk = np.unique((K[cells][:, None] + np.array(
                [(a * dims[1] + b) * dims[2] + c for a in (-1, 0, 1) for b in (-1, 0, 1)
                 for c in (-1, 0, 1)])[None]).ravel())
            j = np.searchsorted(K, nbk)
            j = j[(j < m_) & (K[np.minimum(j, m_ - 1)] == nbk)]
            j = j[~np.isin(j, cells)]
            cand = np.flatnonzero(np.isin(inv, j) & ~out)
            if len(cand) >= 3:
                o3d = _o3d()
                q = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P[cand]))
                q.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.06, max_nn=20))
                qn = np.asarray(q.normals)
                ns = n[cells]
                ns = ns * np.sign(ns @ ns[0])[:, None]
                nm = ns.mean(0) / np.linalg.norm(ns.mean(0))
                pm = (mu[cells] + lo).mean(0)
                on = (np.abs(qn @ nm) > np.cos(np.radians(30))) & (np.abs((P[cand] - pm) @ nm) < 0.04)
                out[cand[on]] = True
    return out


def sky_points(P, T, h, cell=0.4, roof_h=2.0, flat=0.9, keep_path=0.6, edge=1):
    """Per point: over open sky, a courtyard: in a column (cell m, in plan)
    with no ceiling of the building over it (horizontal surfaces roof_h or
    more above the floor: |n_z| over flat, so a slanted sheet is no ceiling)
    within `edge` columns, and more than keep_path from where the robot
    drove. All of it, whatever it touches: a sheet in the courtyard joined to
    the wall it crosses goes too; that wall stays, the ceiling reaching it."""
    from scipy import ndimage
    o3d = _o3d()
    _, near = cKDTree(T[:, :2]).query(P[:, :2], workers=-1)
    up = P[:, 2] - (T[near, 2] - h)
    lo = P[:, :2].min(0)
    ij = np.floor((P[:, :2] - lo) / cell).astype(np.int64)
    nx, ny = ij.max(0) + 1
    hi = np.flatnonzero(up >= roof_h)
    roof = np.zeros((nx, ny), bool)
    if len(hi):
        q = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P[hi]))
        q.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=20))
        hor = np.abs(np.asarray(q.normals)[:, 2]) > flat
        roof[ij[hi[hor], 0], ij[hi[hor], 1]] = True
    if edge:
        roof = ndimage.binary_dilation(roof, np.ones((2 * edge + 1, 2 * edge + 1), bool))
    gx, gy = np.meshgrid(lo[0] + (np.arange(nx) + 0.5) * cell, lo[1] + (np.arange(ny) + 0.5) * cell,
                         indexing="ij")
    d, _ = cKDTree(T[:, :2]).query(np.c_[gx.ravel(), gy.ravel()], workers=-1)
    sky = ~roof & (d.reshape(nx, ny) > keep_path)
    return sky[ij[:, 0], ij[:, 1]]


def scan_clear(P, scans, voxel=0.1, max_range=20.0, res=0.5, margin=0.25, rel=0.02,
               min_through=3, ratio=2.0, counts=False):
    """Per point of the map P: empty space by the LiDAR's own scans. scans:
    (T_world_lidar 4x4, the scan's points in the LiDAR frame) each. Every
    scan as a range image (res degrees a pixel, the nearest return in each);
    each 10 cm cell of the map within max_range and the scan's field of view
    projected into it: seen through when the scan's return in its direction
    lies beyond it (by margin + rel x range), seen at when the return is on
    it. A cell seen through min_through times or more and over ratio x as
    often as seen at is empty space: the laser went through where the map
    has something (a wall's reflection in glass, which is there only seen in
    the glass). A direction with no return (glass, sky) says nothing; a cell
    no scan saw is kept. counts: also each point's (through, at)."""
    lo = P.min(0)
    k = np.floor((P - lo) / voxel).astype(np.int64)
    dims = k.max(0) + 1
    _, inv, cnt = np.unique((k[:, 0] * dims[1] + k[:, 1]) * dims[2] + k[:, 2],
                            return_inverse=True, return_counts=True)
    inv = inv.ravel()
    Q = np.stack([np.bincount(inv, P[:, i]) for i in range(3)], 1) / cnt[:, None]
    tree = cKDTree(Q)
    through = np.zeros(len(Q), np.int32)
    at = np.zeros(len(Q), np.int32)
    rr = np.radians(res)
    na = int(round(2 * np.pi / rr))
    n_scans = 0
    for T, pl in scans:
        r = np.linalg.norm(pl, axis=1)
        ok = r > 0.3
        pl, r = pl[ok], r[ok]
        if len(r) < 100:
            continue
        el = np.arcsin(np.clip(pl[:, 2] / r, -1, 1))
        e0, e1 = el.min(), el.max()
        ne = int((e1 - e0) / rr) + 1
        img = np.full((ne, na), np.inf)
        ia = (np.floor((np.arctan2(pl[:, 1], pl[:, 0]) + np.pi) / rr).astype(np.int64)) % na
        np.minimum.at(img, (np.minimum(((el - e0) / rr).astype(np.int64), ne - 1), ia), r)
        R, t = T[:3, :3], T[:3, 3]
        cand = np.asarray(tree.query_ball_point(t, max_range), np.int64)
        if not len(cand):
            continue
        q = (Q[cand] - t) @ R                                # into the LiDAR frame
        rq = np.linalg.norm(q, axis=1)
        eq = np.arcsin(np.clip(q[:, 2] / np.maximum(rq, 1e-9), -1, 1))
        inview = (eq >= e0) & (eq <= e1) & (rq > 0.5)
        cand, q, rq, eq = cand[inview], q[inview], rq[inview], eq[inview]
        aq = (np.floor((np.arctan2(q[:, 1], q[:, 0]) + np.pi) / rr).astype(np.int64)) % na
        rs = img[np.minimum(((eq - e0) / rr).astype(np.int64), ne - 1), aq]
        tol = margin + rel * rq
        through[cand[np.isfinite(rs) & (rs > rq + tol)]] += 1
        at[cand[np.abs(rs - rq) < tol]] += 1
        n_scans += 1
    empty = (through >= min_through) & (through > ratio * at)
    log.info(f"  scans: {n_scans} used; {int(empty.sum())} of {len(Q)} map cells ({voxel * 100:g} cm) seen "
             f"through, empty space")
    return (empty[inv], through[inv], at[inv]) if counts else empty[inv]


def bag_scans(bag, topic, traj, every=5):
    """(T_world_lidar, points in the LiDAR frame) of every `every`-th scan of
    the bag's points topic with a pose of traj (TUM, the map's frame) within
    50 ms, each deskewed to its pose: pipeline_common.iter_scans."""
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "map_stages"))
    from pipeline_common import iter_scans, load_traj
    tr_t, tr_T = load_traj(os.path.expanduser(traj))
    for n, (j, p) in enumerate(iter_scans(os.path.expanduser(bag), topic, tr_t, 0.05, 0.5, 60.0,
                                          tr_T=tr_T, deskew=True, frame="pose", verbose=False)):
        if n % max(1, int(every)) == 0:
            yield tr_T[j], p


def sensor_height(P, T, cell=0.5, samples=300):
    """Height of the sensor above the floor: per sampled pose, the densest
    2 cm layer of the points within cell/2 under it, 0.1 to 3 m below."""
    lo = P[:, :2].min(0)
    k = np.floor((P[:, :2] - lo) / cell).astype(np.int64)
    ny = int(k[:, 1].max()) + 1
    key = k[:, 0] * ny + k[:, 1]
    order = np.argsort(key, kind="stable")
    ks = key[order]
    hs = []
    for t in T[np.linspace(0, len(T) - 1, min(samples, len(T))).astype(int)]:
        c = np.floor((t[:2] - lo) / cell).astype(np.int64)
        a, b = np.searchsorted(ks, [c[0] * ny + c[1], c[0] * ny + c[1] + 1])
        z = P[order[a:b], 2]
        z = z[(z < t[2] - 0.1) & (z > t[2] - 3.0)]
        if len(z) < 20:
            continue
        cnt, e = np.histogram(z, bins=np.arange(z.min(), z.max() + 0.04, 0.02))
        hs.append(t[2] - 0.5 * (e[np.argmax(cnt)] + e[np.argmax(cnt) + 1]))
    if not hs:
        raise ConvertError("no floor under the trajectory: is --traj the map's trajectory?")
    return float(np.median(hs)), len(hs)


def fill_floor(P, C, T, h, cell=0.04, band=0.12, close=0.2, max_hole=20.0, spacing=0.02,
               roof=0.5):
    """Floor points for the holes in the floor: (points, colours, stats).

    The floor's height under every cell is the nearest pose's less the sensor
    height h; a cell is floor when it holds points within band of it, blocked
    when it holds points 0.15-0.6 m above it (walls, furniture) and a void
    when points lie more than 0.3 m below it (stairs, an atrium). Filled: the
    cells enclosed by floor and blocked cells, and gaps up to 2 x close in the
    floor, never blocked or void, each hole up to max_hole m2 and with a
    roof over at least `roof` of it when larger than 1 m2 (points 1.8 m or
    more above the floor, in the cell or the next: a ceiling; a courtyard is open to
    the sky; roof=0 fills it too); at the
    height and in the colours of the floor around them (8 nearest floor
    cells)."""
    from scipy import ndimage
    lo = P[:, :2].min(0) - 2 * cell
    nx, ny = (np.floor((P[:, :2].max(0) + 2 * cell - lo) / cell).astype(int) + 1)
    ij = np.floor((P[:, :2] - lo) / cell).astype(np.int64)
    lin = ij[:, 0] * ny + ij[:, 1]
    gx, gy = np.meshgrid(lo[0] + (np.arange(nx) + 0.5) * cell, lo[1] + (np.arange(ny) + 0.5) * cell,
                         indexing="ij")
    _, near = cKDTree(T[:, :2]).query(np.c_[gx.ravel(), gy.ravel()])
    level = T[near, 2] - h                                   # floor height per cell
    dz = P[:, 2] - level[lin]
    fl = np.abs(dz) < band
    N = nx * ny
    nfl = np.bincount(lin[fl], minlength=N)
    zf = np.bincount(lin[fl], P[fl, 2], N) / np.maximum(nfl, 1)
    cf = np.stack([np.bincount(lin[fl], C[fl, i], N) for i in range(3)], 1) / np.maximum(nfl, 1)[:, None]
    floor = (nfl >= 3).reshape(nx, ny)          # 3 of the 4 a 2 cm map gives: a half-covered edge cell is filled
    blocked = (np.bincount(lin[(dz > 0.15) & (dz < 0.6)], minlength=N) >= 2).reshape(nx, ny)
    void = (np.bincount(lin[dz < -0.3], minlength=N) >= 2).reshape(nx, ny)
    r = max(1, int(round(close / cell)))
    disk = np.hypot(*np.mgrid[-r:r + 1, -r:r + 1]) <= r
    covered = ndimage.binary_dilation((np.bincount(lin[dz > 1.8], minlength=N) >= 1).reshape(nx, ny),
                                      np.ones((3, 3), bool))   # a ceiling's own speckle is no sky
    cand = (ndimage.binary_fill_holes(floor | blocked) | ndimage.binary_closing(floor, disk))
    cand &= ~floor & ~blocked & ~void
    lab, nlab = ndimage.label(cand)
    ncell = np.bincount(lab.ravel(), minlength=nlab + 1)
    area = ncell * cell * cell
    roofed = np.bincount(lab.ravel(), covered.ravel().astype(float), nlab + 1) / np.maximum(ncell, 1)
    big, sky = area > max_hole, (roofed < roof) & (area > 1.0)     # gaps up to 1 m2 always filled
    big[0] = sky[0] = False
    keep = ~big & ~sky
    keep[0] = False
    holes = keep[lab]
    stats = {"floor_m2": float(floor.sum() * cell * cell), "holes": int(keep.sum()),
             "filled_m2": float(holes.sum() * cell * cell),
             "skipped": int(big[1:].sum()), "skipped_m2": float(area[big].sum()),
             "open": int((sky & ~big)[1:].sum()), "open_m2": float(area[sky & ~big].sum())}
    if not holes.any():
        return np.zeros((0, 3)), np.zeros((0, 3)), stats
    # points every `spacing` in the hole cells; height and colour of the floor around
    sub = max(1, int(round(cell / spacing)))
    off = (np.arange(sub) + 0.5) / sub * cell
    hi, hj = np.nonzero(holes)
    ox, oy = np.meshgrid(off, off, indexing="ij")
    xy = np.c_[(lo[0] + np.repeat(hi, sub * sub) * cell + np.tile(ox.ravel(), len(hi))),
               (lo[1] + np.repeat(hj, sub * sub) * cell + np.tile(oy.ravel(), len(hi)))]
    fi = np.flatnonzero(floor.ravel())
    fxy = np.c_[gx.ravel()[fi], gy.ravel()[fi]]
    d, k = cKDTree(fxy).query(xy, k=min(8, len(fi)))
    w = 1.0 / np.maximum(d, 1e-3)
    w /= w.sum(1, keepdims=True)
    z = (w * zf[fi][k]).sum(1)
    col = np.einsum("nk,nkc->nc", w, cf[fi][k])
    stats["points"] = len(xy)
    return np.c_[xy, z], col, stats


def map_view(map_path, traj, out=None, *, outliers=True, snap=True, fill=True,
             ceil=None, ceil_margin=0.10, max_hole=20.0, roof=0.5, ghosts=True, min_piece=50.0,
             body=0.0, sky=False, bag=None, topic=None, scan_every=5):
    """A viewing copy of the map: stray points out, walls and floors flat,
    the floor's holes filled; points, so the colours stay the map's own."""
    o3d = _o3d()
    out = out or default_view_out(map_path)
    pc = o3d.io.read_point_cloud(os.path.expanduser(map_path))
    if not len(pc.points):
        raise ConvertError(f"{map_path}: no points")
    n0 = len(pc.points)
    report = [f"map {map_path}: {n0} points"]
    if outliers:
        pc, _ = pc.remove_radius_outlier(nb_points=8, radius=0.08)
        report.append(f"  stray points (fewer than 8 others within 8 cm) removed: {n0 - len(pc.points)}")
    P = np.asarray(pc.points).copy()
    C = np.asarray(pc.colors).copy() if pc.has_colors() else np.full((len(P), 3), 0.6)
    res = {"out": out, "map_points": n0}
    T1 = read_tum_positions(traj, stride=1)
    h, nh = sensor_height(P, T1)
    if ghosts:
        g = ghost_points(P, T1)
        P, C = P[~g], C[~g]
        report.append(f"  mirror ghosts (behind polished stone or glass: no direct ray from the path "
                      f"reaches them) removed: {int(g.sum())} points")
        res["ghost_points"] = int(g.sum())
    if bag:
        # the LiDAR's own scans: where the laser went through, nothing is
        g = scan_clear(P, bag_scans(bag, topic, traj, scan_every))
        P, C = P[~g], C[~g]
        report.append(f"  seen through by the LiDAR's own scans (every {scan_every}th of the bag), empty "
                      f"space (a wall's reflection in glass): {int(g.sum())} points removed")
        res["scan_points"] = int(g.sum())
    if sky:
        g = sky_points(P, T1, h)
        P, C = P[~g], C[~g]
        report.append(f"  over open sky (a courtyard: no ceiling of the building over it, away from the "
                      f"robot's way), whatever it touches, removed: {int(g.sum())} points")
        res["sky_points"] = int(g.sum())
    if body > 0:
        g = body_clear(P, T1, h, body)
        P, C = P[~g], C[~g]
        report.append(f"  what the robot drove through ({body} m round its path, floor to over the LiDAR; "
                      f"a flat surface it crossed, whole) removed: {int(g.sum())} points")
        res["body_points"] = int(g.sum())
    if min_piece > 0:
        g = piece_points(P, T1, h, min_piece)
        P, C = P[~g], C[~g]
        report.append(f"  loose pieces (not the building) open to the sky or floating removed: "
                      f"{int(g.sum())} points")
        res["piece_points"] = int(g.sum())
    if snap:
        P, moved = snap_planes(P)
        report.append(f"  points put onto their wall or floor plane: {moved} ({100 * moved / len(P):.0f} %)")
    if fill:
        Fp, Fc, st = fill_floor(P, C, T1[::5], h, max_hole=max_hole, roof=roof)
        report.append(f"  sensor {h:.2f} m above the floor ({nh} poses); floor seen {st['floor_m2']:.0f} m2; "
                      f"{st['holes']} holes filled, {st['filled_m2']:.1f} m2, {len(Fp)} points"
                      + (f"; {st['skipped']} larger than {max_hole:g} m2 left open "
                         f"({st['skipped_m2']:.0f} m2)" if st["skipped"] else "")
                      + (f"; {st['open']} open to the sky (a courtyard: no ceiling over "
                         f"{100 * (1 - roof):.0f} % of it) left open ({st['open_m2']:.0f} m2)"
                         if st["open"] else ""))
        P, C = np.vstack([P, Fp]), np.vstack([C, Fc])
        res.update(sensor_height=h, **st)
    if ceil is not None:
        if str(ceil) == "auto":
            _, hc, hn = height_histogram(P[:, 2])
            z = ceiling_height(hc, hn) - ceil_margin
            keep = P[:, 2] <= np.minimum(z, local_ceiling(P, T1, h)(P) - ceil_margin)
            how = f"or below each place's own ceiling, less {ceil_margin} m"
        else:
            z = float(ceil)
            keep = P[:, 2] <= z
            how = "given"
        report.append(f"  above z = {z:+.2f} ({how}) removed (the ceiling): {int((~keep).sum())} points")
        P, C = P[keep], C[keep]
        if min_piece > 0:
            g = piece_points(P, T1, h, min_piece)       # what hung from the ceiling now floats
            P, C = P[~g], C[~g]
            report.append(f"  then loose (hung from the ceiling): {int(g.sum())} points")
        res["ceil_z"] = z
    v = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P))
    v.colors = o3d.utility.Vector3dVector(np.clip(C, 0, 1))
    o3d.io.write_point_cloud(out, v)
    report.append(f"view {out}: {len(P)} points (a viewing copy: the reference map is unchanged)")
    log.info("\n".join(report))
    res["points"] = len(P)
    return res


def anchor_file(path, frame, out=None):
    """A cloud or mesh made in GLIM's world moved into the anchored map frame
    with 05's T_N_world (frames/anchor_frame.json), as the anchored map and
    the anchored track are: <name>_anchored. Lets view and mesh run on
    map_final and the BA trajectory and their results be anchored after."""
    import json
    o3d = _o3d()
    path = os.path.expanduser(path)
    stem, ext = os.path.splitext(path)
    out = out or stem + "_anchored" + ext
    af = json.load(open(os.path.expanduser(frame)))
    if af.get("input_camera_anchored"):
        raise ConvertError(f"{frame}: the map was anchored at the camera start; T_N_world is not the cloud's move")
    T = np.array(af["T_N_world"], float)
    m = o3d.io.read_triangle_mesh(path)
    if len(m.triangles):
        m.transform(T)
        o3d.io.write_triangle_mesh(out, m)
        n = f"{len(m.triangles)} faces"
    else:
        pc = o3d.io.read_point_cloud(path)
        if not len(pc.points):
            raise ConvertError(f"{path}: no faces or points")
        pc.transform(T)
        o3d.io.write_point_cloud(out, pc)
        n = f"{len(pc.points)} points"
    log.info(f"{out}: {n} moved into the anchored frame (T_N_world of {frame})")
    return out


SIONNA_MATERIALS = {"floor": ("concrete", 0.2), "ceiling": ("concrete", 0.2), "walls": ("concrete", 0.2),
                    "objects": ("wood", 0.05), "glass": ("glass", 0.01)}
PART_COLOURS = {"floor": (0.6, 0.6, 0.6), "walls": (0.85, 0.78, 0.62), "ceiling": (0.95, 0.95, 0.95),
                "objects": (0.95, 0.5, 0.1), "glass": (0.55, 0.8, 0.95)}
PARTS = ("floor", "walls", "ceiling", "objects", "glass")


def floor_grid(V, cell=1.0):
    """(height function, lo, shape): the floor under each cell (m, in plan)
    of the vertices V, their lowest 1 % there, a cell with none the lowest of
    its neighbours'."""
    from scipy import ndimage
    lo = V[:, :2].min(0)
    nx, ny = np.floor((V[:, :2].max(0) - lo) / cell).astype(np.int64) + 1
    lin = np.floor((V[:, :2] - lo) / cell).astype(np.int64) @ np.array([ny, 1])
    o = np.lexsort((V[:, 2], lin))
    starts = np.r_[0, np.flatnonzero(np.diff(lin[o])) + 1]
    cnt = np.diff(np.r_[starts, len(o)])
    g = np.full(nx * ny, np.inf)
    g[lin[o][starts]] = V[o[starts + cnt // 100], 2]
    g = ndimage.minimum_filter(g.reshape(nx, ny), size=3, mode="nearest")

    def at(X):
        k = np.clip(np.floor((np.asarray(X)[:, :2] - lo) / cell).astype(np.int64), 0, [nx - 1, ny - 1])
        return g[k[:, 0], k[:, 1]]
    return at


def plane_faces(C, N, cell=0.4, sig_max=0.03, thin=0.35, min_pts=20, angle=15.0, gap=0.06,
                near=0.04, facing=30.0, corners=None, area=None, min_area=0.04):
    """Per face (centres C, unit normals N): the flat surface it lies on, or
    -1. In each 0.4 m cell, the faces of each way they face (8 ways round
    and up and down, so the two sides of a thin wall in one cell are two
    flat surfaces, not one thick one), flat as snap_planes finds them;
    joined, in a cell and the next, while they face the same way to `angle`
    and planes to `gap`; a face is on its group's surface when within
    `near` of its plane and facing it to `facing`. corners (the faces'
    corners, F x 3 x 3) and area: each face counted by its centre and
    corners, weighted by its area, and a group flat from min_area m2 (not
    min_pts faces): a flat wall drawn in a few large faces is still seen
    flat. Also each surface's |n_z|."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    NB = 10
    way = np.where(N[:, 2] > 0.7, 8, np.where(N[:, 2] < -0.7, 9,
                   (np.floor((np.arctan2(N[:, 1], N[:, 0]) + np.pi) / (np.pi / 4)).astype(np.int64)) % 8))
    lo = C.min(0)
    k = np.floor((C - lo) / cell).astype(np.int64) + 1
    dims = k.max(0) + 2
    key = ((k[:, 0] * dims[1] + k[:, 1]) * dims[2] + k[:, 2]) * NB + way
    K, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    inv = inv.ravel()
    m_ = len(K)
    X = C - lo
    if corners is not None and area is not None:              # centre and corners, by area
        Pk = np.concatenate([X] + [corners[:, i] - lo for i in range(3)])
        ik = np.tile(inv, 4)
        wk = np.tile(area / 4.0, 4)
    else:
        Pk, ik, wk = X, inv, np.ones(len(X))
    sw = np.maximum(np.bincount(ik, wk, m_), 1e-12)
    mu = np.stack([np.bincount(ik, wk * Pk[:, i], m_) for i in range(3)], 1) / sw[:, None]
    S = np.empty((m_, 3, 3))
    for i in range(3):
        for j in range(i, 3):
            S[:, i, j] = S[:, j, i] = np.bincount(ik, wk * Pk[:, i] * Pk[:, j], m_) / sw - mu[:, i] * mu[:, j]
    w, v = np.linalg.eigh(S)
    sig = np.sqrt(np.clip(w, 0, None))
    enough = (np.bincount(inv, area, m_) >= min_area) if area is not None else (cnt >= min_pts)
    flat = enough & (sig[:, 0] < sig_max) & (sig[:, 0] < thin * sig[:, 1])
    n = v[:, :, 0]
    nm = np.stack([np.bincount(inv, N[:, i], m_) for i in range(3)], 1)      # the way its faces face
    n[np.einsum("ij,ij->i", n, nm) < 0] *= -1
    Ks = K // NB
    kc = np.stack([Ks // (dims[1] * dims[2]), (Ks // dims[2]) % dims[1], Ks % dims[2]], 1)
    rows, cols = [np.arange(m_)], [np.arange(m_)]
    cos_a = np.cos(np.radians(angle))
    for o in [(a, b, c) for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1) if (a, b, c) >= (0, 0, 0)]:
        sp = ((kc[:, 0] + o[0]) * dims[1] + kc[:, 1] + o[1]) * dims[2] + kc[:, 2] + o[2]
        for wb in range(NB):
            nb = sp * NB + wb
            j = np.minimum(np.searchsorted(K, nb), m_ - 1)
            ok = (K[j] == nb) & flat & flat[j] & (j != np.arange(m_))
            i, j = np.flatnonzero(ok), j[ok]
            same = (np.einsum("ij,ij->i", n[i], n[j]) > cos_a) & \
                (np.abs(np.einsum("ij,ij->i", mu[j] - mu[i], n[i])) < gap)        # facing the same way
            rows.append(i[same])
            cols.append(j[same])
    r_, c_ = np.concatenate(rows), np.concatenate(cols)
    _, seg = connected_components(coo_matrix((np.ones(len(r_)), (r_, c_)), shape=(m_, m_)), directed=False)
    nc = n[inv]
    on = flat[inv] & (np.abs(np.einsum("ij,ij->i", X - mu[inv], nc)) < near) & \
        (np.abs(np.einsum("ij,ij->i", N, nc)) > np.cos(np.radians(facing)))
    sid = np.where(on, seg[inv], -1)
    nz = np.zeros(seg.max() + 1)
    nz[seg[flat]] = np.abs(n[flat, 2])
    return sid, nz


def hanging_faces(m, T, ceil_margin=0.10, min_piece=50.0, reach=0.3):
    """Per face of the mesh m (with its ceiling): what `mesh --ceil auto`
    drops from its _noceil copy below the cut. The ceiling cut as there
    (each place's own ceiling and the highest peak, less ceil_margin), then
    the pieces left loose (loose(): floating, or open to the sky) that reach
    up to the cut (within reach m): what hung from the ceiling only, such as
    a reflection joined to a mirrored ceiling; gone from the copy, so here
    too; only one that was joined to the building before the cut (hung from
    its ceiling: a rack whose top the cut took is joined to nothing). A
    pillar (from the floor, 1 m across at most) stays. A loose piece well under the cut (a table, a cabinet: open to the
    sky only because the ceiling is cut away) stays. T: the trajectory in
    the mesh's frame."""
    V, F = np.asarray(m.vertices), np.asarray(m.triangles)
    hs, _ = sensor_height(V, T)
    _, hc, hn = height_histogram(V[:, 2])
    z = ceiling_height(hc, hn) - ceil_margin
    c = V[F].mean(1)
    cut = np.minimum(z, local_ceiling(V, T, hs)(c) - ceil_margin)
    above = c[:, 2] > cut
    kept = np.flatnonzero(~above)
    sub = type(m)(m)
    sub.remove_triangles_by_mask(above)                  # keeps the others' order
    tri, _, area = sub.cluster_connected_triangles()
    tri, area = np.asarray(tri), np.asarray(area)
    gap = np.full(len(area), np.inf)                     # each piece's nearest to the cut
    np.minimum.at(gap, tri, cut[kept] - c[kept, 2])
    # a pillar stands on the floor, its footprint small: not hanging
    _, near = cKDTree(T[:, :2]).query(c[kept, :2], workers=-1)
    up = c[kept, 2] - (T[near, 2] - hs)
    low = np.full(len(area), np.inf)
    np.minimum.at(low, tri, up)
    lo2, hi2 = np.full((len(area), 2), np.inf), np.full((len(area), 2), -np.inf)
    np.minimum.at(lo2, tri, c[kept, :2])
    np.maximum.at(hi2, tri, c[kept, :2])
    pillar = (low <= 0.3) & ((hi2 - lo2).max(1) <= 1.0)
    # it hung from the ceiling: before the cut, joined to the building (a rack
    # whose top the cut took is joined to nothing)
    tri0, _, area0 = m.cluster_connected_triangles()
    tri0, area0 = np.asarray(tri0), np.asarray(area0)
    joined = np.zeros(len(area), bool)
    joined[np.unique(tri[area0[tri0[kept]] >= min_piece])] = True
    drop = loose(tri, c[kept], area, T, hs, min_piece) & (gap <= reach) & ~pillar & joined
    gone = np.zeros(len(F), bool)
    gone[kept[drop[tri]]] = True
    return gone


def label_faces(m, flat=0.9, band=0.12, roof_h=2.0, wall_area=0.5, wall_high=2.2, vertical=0.3,
                on_wall=0.05, facing_deg=30.0, reach=0.2, min_obj=0.05, seam=0.08, ground=0.3, against=0.1,
                pillar_r=(0.1, 1.0), pillar_fit=0.04, attach=0.1, attach_depth=0.2, attach_upright=0.9,
                mounted=0.3, below=0.25, panel_near=0.5, panel_long=0.5, panel_side=0.8,
                arch_top=1.9, arch_low=1.7, ledge_near=0.3, top_near=0.3, float_low=0.3, hug=0.6,
                high_cut=2.0, high_near=0.5, split_on=0.1, with_wall=0.25,
                stair_rise=(0.25, 1.5), stair_side=0.15, knee_w=1.5, knee_thin=0.1, knee_end=0.35,
                T=None, why=(), planes=None, why_objects=0):
    """Per face of the mesh m: "floor", "ceiling", "walls" or "objects", and
    each object face its object's number (-1 for the rest).

    The floor: with T (the trajectory, the mesh's frame) the nearest pose's
    height less the sensor height, as fill_floor takes it, so whatever lies
    under the floor (a reflection in polished stone, a step down) cannot
    lower it; without, the lowest of each cell round (floor_grid). Floor:
    horizontal (|n_z| over flat) within band of it (with T: either side);
    With T, what is more than `below` m under the floor is "gone": the room
    mirrored in a polished floor (Poisson joins the reflection of a wall to
    the wall, a deep unflat piece no rule takes); left out as a reflection.
    ceiling: horizontal roof_h or more above it and within 0.3 m of the
    highest such in its 0.5 m column and the next (a tall rack's or
    cupboard's top under it is not).

    Walls: flat sheets standing on the floor. A flat surface (plane_faces)
    near vertical (|n_z| under `vertical`) of wall_area m2 or more reaching
    wall_high or more over the floor; every face on its plane (within
    on_wall, facing it to facing_deg, inside its extent along it, reach m
    more, from its top down to the floor: the wall goes on behind the
    furniture in front of it) is wall when some of them come down to the
    floor (`ground` m) or reach up to the ceiling over them (0.3 m: a wall
    over glass the map has not, a bulkhead). What is left of a wall's own shape (a pillar's
    rounded edges, a wall's end), facing sideways from the floor to
    wall_high, 70 % of it within `against` of wall faces, goes with them. A
    rack, chairs stacked: no flat sheet, and not against a wall all over:
    objects. A round pillar's arcs, facing sideways from the floor to
    wall_high and round in plan (a circle of radius pillar_r, within
    pillar_fit m): walls (a rack's poles stand on a rectangle).

    The rest facing sideways within seam of a floor, wall or ceiling face
    (the rounded strips where they meet) goes with it; one facing up (a
    table top by a wall) does not. Everything else, in connected pieces, is
    objects, but: stairs and ramps (all their faces, treads, risers, the
    slope under them, the rail, climbing along the piece's length,
    stair_rise m per m, over 0.8 m; a shelf's stack in place does not) and
    all that is against a flight (within stair_side: its sides, a landing,
    the floor above it, and what is against those) are walls; so is a low wall between pillars (under windows whose glass is
    not in the map: a piece from the floor (0.3 m), facing sideways mostly,
    knee_w m long or more, thin across (under knee_thin m, 1 sigma), both
    ends within knee_end of a wall face; a row of cabinets is deeper and
    has a free end); and so is all that is attached to the walls (touching a
    wall face within `attach` m and nowhere farther than attach_depth from
    them, or facing sideways attach_upright of it, or off the floor (its
    bottom over `mounted` m: on the wall), or reaching wall_high, 30 % of
    it above 1.8 m, facing sideways 60 % (a wall cluttered; not furniture
    with a sliver of wall joined), however far: a door's or
    window's frame, a sill, a panel, a cabinet or a sign on a wall, a
    pillar's curve, a piece of wall, and what is attached to those); and
    so is a sheet near a wall (thin across, under knee_thin m, panel_long m
    long or more, facing sideways panel_side, within panel_near m of a
    wall face: a door, set in its frame or ajar, a strip or sliver of wall),
    and all that floats (off the floor, over float_low m) within ledge_near
    m of a wall or of what is wall by these rules (a ledge, a lintel, a
    door's part, a wall's top), and a wall's top (a thick wall, a parapet: facing up, wall
    within top_near m along both its long edges; a table by a wall has it
    on one side), and what touches a wall, reaches wall_high and is nowhere
    farther than hug m from the walls (a wall cluttered); of what is left
    objects, the faces within split_on m of a wall plane (the wall's own,
    fused with the furniture against it; a door set in its frame) and those
    over high_cut m within high_near m of a wall are wall (the furniture
    under them stays objects), and of the pieces that leaves, one floating
    (off the floor) within ledge_near of a wall, or with 90 % of it within
    with_wall of one (a frame, a strip; a chair reaches out more), is wall
    too. Distances to the walls count the wall's plane where the map has
    no wall (hidden by what stands against it);
    and so is a doorway's frame whose wall is not in the map (its jambs and
    the ledge over them: on the floor at both ends, reaching arch_top m,
    nothing under arch_low m in the middle; a table's top is lower, a
    rack's shelves fill its middle): objects are only what
    stands out or apart (a chair or a desk pushed to a wall reaches out
    0.4 m and more); a piece reaching no higher than band (the floor's
    bumps) is floor and one under min_obj m2 (a scrap at a seam) the part
    of the nearest labelled face. why: plan points (x, y): the pieces left
    there and what each rule saw of them, logged; why_objects: the largest
    that many objects and what each rule saw, logged. planes: a list given gets
    each wall's plane (centre, normal, along, extent, top, floor)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    V, F = np.asarray(m.vertices), np.asarray(m.triangles)
    m.compute_triangle_normals()
    N = np.asarray(m.triangle_normals)
    C = V[F].mean(1)
    A = 0.5 * np.linalg.norm(np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]), axis=1)
    if T is not None:
        h, _ = sensor_height(V, T)
        _, near = cKDTree(T[:, :2]).query(C[:, :2], workers=-1)
        dz = C[:, 2] - (T[near, 2] - h)
        on_floor = np.abs(dz) <= band
    else:
        dz = C[:, 2] - floor_grid(V)(C)
        on_floor = dz <= band
    hor = np.abs(N[:, 2]) > flat
    part = np.full(len(F), "", dtype=object)
    part[hor & on_floor] = "floor"
    if T is not None and below is not None:
        part[dz < -below] = "gone"         # under the floor: the room mirrored in the polished floor
    # ceiling: high and flat, and the highest such overhead (a rack's top under it is not)
    hi_ = np.flatnonzero(hor & (dz >= roof_h))
    top = None
    if len(hi_):
        from scipy import ndimage
        lo_ = C[:, :2].min(0)
        ij = np.floor((C[hi_, :2] - lo_) / 0.5).astype(np.int64)
        nx_, ny_ = ij.max(0) + 1
        top = np.full((nx_, ny_), -np.inf)
        np.maximum.at(top, (ij[:, 0], ij[:, 1]), C[hi_, 2])
        top = ndimage.maximum_filter(top, size=3, mode="nearest")
        part[hi_[C[hi_, 2] >= top[ij[:, 0], ij[:, 1]] - 0.3]] = "ceiling"

    def ceil_at(X):
        """The ceiling's height over plan points (inf where none)."""
        z_ = np.full(len(X), np.inf)
        if top is None:
            return z_
        k_ = np.floor((X[:, :2] - lo_) / 0.5).astype(np.int64)
        ok_ = (k_ >= 0).all(1) & (k_[:, 0] < top.shape[0]) & (k_[:, 1] < top.shape[1])
        z_[ok_] = top[k_[ok_, 0], k_[ok_, 1]]
        z_[~np.isfinite(z_)] = np.inf
        return z_
    # the wall planes
    sid, snz = plane_faces(C, N, corners=V[F], area=A)
    s_ok = sid >= 0
    ns = len(snz)
    s_area = np.bincount(sid[s_ok], A[s_ok], ns)
    s_hi = np.full(ns, -np.inf)
    np.maximum.at(s_hi, sid[s_ok], dz[s_ok])
    s_lo = np.full(ns, np.inf)
    np.minimum.at(s_lo, sid[s_ok], dz[s_ok])
    wall_s = np.flatnonzero((snz < vertical) & (s_area >= wall_area) & (s_hi >= wall_high))
    cosf = np.cos(np.radians(facing_deg))

    def plane(k):
        """A flat surface's plane: (centre, unit normal in plan, along it, its members)."""
        mem = np.flatnonzero(sid == k)
        w = A[mem] / A[mem].sum()
        mu = w @ C[mem]
        n = np.linalg.svd((C[mem] - mu) * np.sqrt(w)[:, None], full_matrices=False)[2][2]
        n[2] = 0.0
        n /= np.linalg.norm(n) or 1.0
        return mu, n, np.array([-n[1], n[0], 0.0]), mem

    def claim(k, cand, ctree):
        """The faces of cand on surface k's plane: within on_wall, facing it,
        inside its extent along it (reach m more), up to its top, down to the floor."""
        mu, n, t, mem = plane(k)
        u = (C[mem] - mu) @ t
        u0, u1 = u.min() - reach, u.max() + reach
        z1 = C[mem, 2].max() + reach
        mid = mu + t * (u0 + u1) / 2
        idx = cand[np.asarray(ctree.query_ball_point(mid[:2], (u1 - u0) / 2 + on_wall + 0.01), np.int64)]
        if not len(idx):
            return idx
        uu = (C[idx] - mu) @ t
        ok = (np.abs((C[idx] - mu) @ n) < on_wall) & (np.abs(N[idx] @ n) > cosf) & (uu >= u0) & \
            (uu <= u1) & (C[idx, 2] <= z1)
        return idx[ok]

    pls = planes if planes is not None else []
    cand = np.flatnonzero(part == "")
    if len(wall_s) and len(cand):
        ctree = cKDTree(C[cand, :2])
        for k in wall_s:
            got = claim(k, cand, ctree)
            # it stands on the floor somewhere, or hangs from the ceiling (over glass, a bulkhead)
            if len(got) and ((dz[got] <= ground).any() or (C[got, 2] >= ceil_at(C[got]) - 0.3).any()):
                part[got] = "walls"
                if True:
                    mu, n, t, _ = plane(k)
                    u = (C[got] - mu) @ t
                    pls.append({"mu": mu, "n": n, "t": t, "u0": float(u.min()), "u1": float(u.max()),
                                   "top": float(C[got, 2].max()),
                                   "floor": float(np.median(C[got, 2] - dz[got]))})
    # the seams: what is left within `seam` of a floor, wall or ceiling face goes with it
    rest = np.flatnonzero((part == "") & ~hor)
    lab = np.flatnonzero((part == "floor") | (part == "walls") | (part == "ceiling"))
    if len(rest) and len(lab):
        d, i = cKDTree(C[lab]).query(C[rest], workers=-1, distance_upper_bound=seam)
        hit = np.isfinite(d)
        part[rest[hit]] = part[lab[i[hit]]]
    def pieces(idx):
        """The faces idx in connected pieces: each one's piece number."""
        Fr = F[idx]
        e = np.r_[Fr[:, 0], Fr[:, 1]], np.r_[Fr[:, 1], Fr[:, 2]]
        _, vc = connected_components(coo_matrix((np.ones(len(e[0])), e), shape=(len(V), len(V))),
                                     directed=False)
        _, cm = np.unique(vc[Fr[:, 0]], return_inverse=True)
        return cm.ravel()

    def plane_dist(idx, margin=0.3, far=2.0):
        """Each face's (idx) distance to the nearest wall plane, inside its
        extent (margin m round it, along and up; inf if none within far m):
        the wall itself where the map has none (hidden by what stands there)."""
        d = np.full(len(idx), np.inf)
        if not pls or not len(idx):
            return d
        tree2 = cKDTree(C[idx, :2])
        for pl in pls:
            mu, n, t = pl["mu"], pl["n"], pl["t"]
            mid = mu[:2] + t[:2] * (pl["u0"] + pl["u1"]) / 2
            j = np.asarray(tree2.query_ball_point(mid, np.hypot((pl["u1"] - pl["u0"]) / 2 + margin, far)), np.int64)
            if not len(j):
                continue
            q = C[idx[j]] - mu
            u, dn = q @ t, np.abs(q @ n)
            ok = ((u > pl["u0"] - margin) & (u < pl["u1"] + margin) & (dn < far)
                  & (C[idx[j], 2] > pl["floor"] - margin) & (C[idx[j], 2] < pl["top"] + margin))
            d[j[ok]] = np.minimum(d[j[ok]], dn[ok])
        return d

    # the rest, in connected pieces
    rest = np.flatnonzero(part == "")
    obj = np.full(len(F), -1)
    if len(rest):
        comp = pieces(rest)
        dpl = plane_dist(rest)                       # to the wall's plane: there even where the map has no wall
        nc = comp.max() + 1
        c_area = np.bincount(comp, A[rest], nc)
        c_hi = np.full(nc, -np.inf)
        np.maximum.at(c_hi, comp, dz[rest])
        c_lo = np.full(nc, np.inf)
        np.minimum.at(c_lo, comp, dz[rest])
        side = np.bincount(comp, A[rest] * (np.abs(N[rest, 2]) < vertical), nc) / np.maximum(c_area, 1e-12)
        high = np.bincount(comp, A[rest] * (dz[rest] >= 1.8), nc) / np.maximum(c_area, 1e-12)   # above furniture
        kind = np.full(nc, "objects", dtype=object)
        # what is left of a wall's own shape (a pillar's rounded edges, a wall end):
        # tall, facing sideways, from the floor, nearly all of it against wall faces
        wf0 = np.flatnonzero(part == "walls")
        if len(wf0):
            dw, _ = cKDTree(C[wf0]).query(C[rest], workers=-1, distance_upper_bound=against)
            frac = np.bincount(comp, A[rest] * np.isfinite(dw), nc) / np.maximum(c_area, 1e-12)
            kind[(side >= 0.75) & (c_lo <= 0.5) & (c_hi >= wall_high) & (frac >= 0.7)] = "walls"
        # a round pillar's arcs: tall, facing sideways, from the floor, round in plan
        order_ = np.argsort(comp, kind="stable")
        cuts_ = np.searchsorted(comp[order_], np.arange(nc + 1))
        for k_ in np.flatnonzero((kind == "objects") & (side >= 0.85) & (c_lo <= 0.5) & (c_hi >= wall_high)):
            xy = C[rest[order_[cuts_[k_]:cuts_[k_ + 1]]], :2]
            if len(xy) < 30:
                continue
            M_ = np.c_[2 * xy, np.ones(len(xy))]                         # (x-a)^2+(y-b)^2=r^2, linear in a, b, c
            (ca, cb, cc), *_ = np.linalg.lstsq(M_, (xy ** 2).sum(1), rcond=None)
            r_ = np.sqrt(max(cc + ca ** 2 + cb ** 2, 0.0))
            res = np.hypot(xy[:, 0] - ca, xy[:, 1] - cb) - r_
            if pillar_r[0] <= r_ <= pillar_r[1] and res.std() < pillar_fit:
                kind[k_] = "walls"
        # stairs and ramps: all their faces climb along the flight (treads, risers,
        # the slope under them, the rail); furniture and a shelf's stack do not
        order = np.argsort(comp, kind="stable")
        cuts = np.searchsorted(comp[order], np.arange(nc + 1))
        slope, fit = np.full(nc, np.nan), np.full(nc, np.nan)
        flight = np.zeros(nc, bool)
        for k_ in np.flatnonzero((c_hi - c_lo >= 0.8) & (c_area >= 0.5)):
            sel = rest[order[cuts[k_]:cuts[k_ + 1]]]
            if len(sel) < 30:
                continue
            zz, w = C[sel, 2], A[sel]
            xy = C[sel, :2] - (w @ C[sel, :2]) / w.sum()
            u = xy @ np.linalg.svd(xy * np.sqrt(w)[:, None], full_matrices=False)[2][0]
            a_, b_ = np.polyfit(u, zz, 1, w=np.sqrt(w))
            r2 = 1 - np.average((zz - (a_ * u + b_)) ** 2, weights=w) / max(np.average(
                (zz - np.average(zz, weights=w)) ** 2, weights=w), 1e-12)
            slope[k_], fit[k_] = abs(a_), r2
            if stair_rise[0] <= abs(a_) <= stair_rise[1] and r2 >= 0.6:
                kind[k_] = "walls"
                flight[k_] = True
        # all of the stairs: what is against a flight (its sides, the landing, the floor
        # above it), and what is against that, goes with it
        if flight.any():
            stair = flight.copy()
            for _ in range(2):
                fl = stair[comp]
                d_, _ = cKDTree(C[rest[fl]]).query(C[rest[~fl]], workers=-1, distance_upper_bound=stair_side)
                near = np.zeros(nc, bool)
                near[np.unique(comp[~fl][np.isfinite(d_)])] = True
                new = near & ~stair & (kind == "objects")
                if not new.any():
                    break
                kind[new] = "walls"
                stair |= new
        # low walls between pillars: long, thin across, both ends at a wall
        touched = np.zeros(nc, bool)                          # pieces touching a wall, for the high cut
        wf = np.flatnonzero(part == "walls")
        if len(wf):
            wtree = cKDTree(C[wf])
            Cr, wr = C[rest], A[rest]
            sw = np.maximum(c_area, 1e-12)
            mx, my = np.bincount(comp, wr * Cr[:, 0], nc) / sw, np.bincount(comp, wr * Cr[:, 1], nc) / sw
            mz = np.bincount(comp, wr * Cr[:, 2], nc) / sw
            sxx = np.bincount(comp, wr * (Cr[:, 0] - mx[comp]) ** 2, nc) / sw
            syy = np.bincount(comp, wr * (Cr[:, 1] - my[comp]) ** 2, nc) / sw
            sxy = np.bincount(comp, wr * (Cr[:, 0] - mx[comp]) * (Cr[:, 1] - my[comp]), nc) / sw
            th = 0.5 * np.arctan2(2 * sxy, sxx - syy)
            ax_ = np.c_[np.cos(th), np.sin(th)]
            across = np.sqrt(np.clip((sxx + syy) / 2 - np.sqrt(((sxx - syy) / 2) ** 2 + sxy ** 2), 0, None))
            pr = (Cr[:, 0] - mx[comp]) * ax_[comp, 0] + (Cr[:, 1] - my[comp]) * ax_[comp, 1]
            p0, p1 = np.full(nc, np.inf), np.full(nc, -np.inf)
            np.minimum.at(p0, comp, pr)
            np.maximum.at(p1, comp, pr)
            for k_ in np.flatnonzero((kind == "objects") & (c_lo <= 0.3) & (c_hi < wall_high) & (side >= 0.6)
                                     & (p1 - p0 >= knee_w) & (across < knee_thin)):
                ends = [np.r_[mx[k_] + ax_[k_, 0] * e, my[k_] + ax_[k_, 1] * e, mz[k_]] for e in (p0[k_], p1[k_])]
                if all(len(wtree.query_ball_point(e_, knee_end)) for e_ in ends):
                    kind[k_] = "walls"
            # a doorway's frame (jambs and the ledge over them, the wall round it not in the
            # map): on the floor at both ends, nothing low in the middle
            dzr = dz[rest]
            for k_ in np.flatnonzero((kind == "objects") & (c_lo <= 0.3) & (c_hi >= arch_top) & (p1 - p0 >= 0.6)):
                sel = order[cuts[k_]:cuts[k_ + 1]]
                u, zz = pr[sel] - p0[k_], dzr[sel]
                L_ = p1[k_] - p0[k_]
                mid = (u > 0.3 * L_) & (u < 0.7 * L_)
                if (mid.any() and zz[mid].min() >= arch_low and (zz[u < 0.2 * L_] <= 0.3).any()
                        and (zz[u > 0.8 * L_] <= 0.3).any()):
                    kind[k_] = "walls"
            # attached to the walls (a door's or window's frame, a sill, a panel on a wall,
            # what is left of a pillar): touching a wall face and nowhere farther from the
            # walls than attach_depth is wall; objects are what stands out or apart (a chair
            # or a desk pushed to a wall reaches out 0.4 m and more)
            for _ in range(6):
                wall_now = np.r_[wf, rest[kind[comp] == "walls"]]
                if not len(wall_now):
                    break
                sel_f = (kind == "objects")[comp]
                if not sel_f.any():
                    break
                d_, _ = cKDTree(C[wall_now]).query(C[rest[sel_f]], workers=-1,
                                                   distance_upper_bound=max(attach_depth, panel_near, ledge_near, top_near, hug))
                d_ = np.minimum(d_, dpl[sel_f])      # or to the wall's plane, where the wall is hidden
                cs = comp[sel_f]
                touch, out, by = np.zeros(nc, bool), np.zeros(nc, bool), np.zeros(nc, bool)
                touch[np.unique(cs[d_ <= attach])] = True
                touched |= touch
                by[np.unique(cs[d_ <= panel_near])] = True
                led, far_ = np.zeros(nc, bool), np.zeros(nc, bool)
                far_[np.unique(cs[~(d_ <= hug)])] = True
                led[np.unique(cs[d_ <= ledge_near])] = True
                # a wall's top (a thick wall, a parapet): flat, wall along both its long edges
                top_ = np.zeros(nc, bool)
                qa = Cr[sel_f, 0] * -ax_[cs, 1] + Cr[sel_f, 1] * ax_[cs, 0]       # across the piece
                for k_ in np.flatnonzero((kind == "objects") & (side < 0.4) & (p1 - p0 >= 0.5)):
                    w_ = cs == k_
                    if w_.sum() < 20:
                        continue
                    q_, dd = qa[w_], d_[w_]
                    lo_, hi_ = np.quantile(q_, [0.1, 0.9])
                    if (hi_ - lo_ > 0.05 and (dd[q_ <= lo_] <= top_near).mean() >= 0.5
                            and (dd[q_ >= hi_] <= top_near).mean() >= 0.5):
                        top_[k_] = True
                d_ = np.where(d_ <= attach_depth, d_, np.inf)
                out[np.unique(cs[~np.isfinite(d_)])] = True        # some of it farther than attach_depth
                # a pillar's curve, a wall's piece; and off the floor, mounted on the wall
                new = touch & (~out | (side >= attach_upright) | (c_lo > mounted)
                               | ((c_hi >= wall_high) & (side >= 0.6) & (high >= 0.3)))  # a wall cluttered
                new |= by & (across < knee_thin) & (p1 - p0 >= panel_long) & (side >= panel_side)  # a door, a strip
                new |= led & (c_lo >= float_low)       # floating by a wall: a ledge, a lintel, a door's part
                new |= top_
                new |= touch & (c_hi >= wall_high) & ~far_     # hugging a wall up high: wall, cluttered
                if not new.any():
                    break
                kind[new] = "walls"
            if why_objects:                             # the largest objects and what each rule saw
                big_o = np.flatnonzero(kind == "objects")
                big_o = big_o[np.argsort(-c_area[big_o])][:int(why_objects)]
                wt = cKDTree(C[np.r_[wf, rest[kind[comp] == "walls"]]])
                lines = [f"the {len(big_o)} largest objects: x y z (centre), m2, height over the floor, % facing "
                         "sideways, long x across (1 sigma), nearest / farthest from a wall, ends from a wall"]
                for k_ in big_o:
                    sel = rest[comp == k_]
                    dd, _ = wt.query(C[sel], workers=-1, distance_upper_bound=5.0)
                    ends = [np.r_[mx[k_] + ax_[k_, 0] * e, my[k_] + ax_[k_, 1] * e, mz[k_]]
                            for e in (p0[k_], p1[k_])]
                    de = [wtree.query(e_)[0] for e_ in ends]
                    lines.append(f"  {mx[k_]:8.2f} {my[k_]:8.2f} {mz[k_]:6.2f}  {c_area[k_]:6.2f} m2  "
                                 f"{c_lo[k_]:5.2f}-{c_hi[k_]:5.2f} m  {100 * side[k_]:3.0f} %  "
                                 f"{p1[k_] - p0[k_]:5.2f} x {100 * across[k_]:5.1f} cm  "
                                 f"{dd.min():5.2f} / {np.nanmax(np.where(np.isfinite(dd), dd, 5.0)):5.2f} m  "
                                 f"ends {de[0]:.2f} / {de[1]:.2f} m")
                log.info("\n".join(lines))
            for q in why:                               # what each rule saw of the pieces there
                d2 = np.hypot(Cr[:, 0] - q[0], Cr[:, 1] - q[1])
                ks = np.unique(comp[d2 < 1.0])
                ks = ks[np.argsort(-c_area[ks])][:4]
                lines = [f"why ({q[0]:.2f}, {q[1]:.2f}): {len(ks)} pieces left within 1 m"
                         + ("" if len(ks) else " (all there already floor, wall or ceiling)")]
                for k_ in ks:
                    ends = [np.r_[mx[k_] + ax_[k_, 0] * e, my[k_] + ax_[k_, 1] * e, mz[k_]]
                            for e in (p0[k_], p1[k_])]
                    de = [wtree.query(e_)[0] for e_ in ends]
                    lines.append(
                        f"  piece {k_}: {c_area[k_]:.2f} m2 -> {kind[k_]}; height {c_lo[k_]:.2f}-{c_hi[k_]:.2f} m "
                        f"over the floor; sideways {100 * side[k_]:.0f} %; long {p1[k_] - p0[k_]:.2f} m, across "
                        f"{100 * across[k_]:.1f} cm (1 sigma); ends {de[0]:.2f} / {de[1]:.2f} m from a wall; "
                        + (f"climbs {slope[k_]:.2f} m per m, fit {fit[k_]:.2f}" if np.isfinite(slope[k_])
                           else "no climb tested (under 0.8 m of height or 0.5 m2)"))
                log.info("\n".join(lines))
        kind[c_hi <= band] = "floor"
        part[rest] = kind[comp]
        # on a wall's plane, whatever it is joined to (a wall's faces fused with the
        # furniture against it, a door set in its frame): within split_on of it, wall
        part[rest[(part[rest] == "objects") & (dpl <= split_on)]] = "walls"
        # what is joined to furniture against a wall but up high by it (a piece touching a
        # wall: its faces over high_cut m within high_near m of a wall face): wall; the
        # furniture under it stays objects (a lamp hanging near a pillar touches nothing)
        wl_ = np.flatnonzero(part == "walls")                  # the walls before the high cut
        hi_ = rest[(part[rest] == "objects") & (dz[rest] >= high_cut) & touched[comp]]
        if len(wl_) and len(hi_):
            d_, _ = cKDTree(C[wl_]).query(C[hi_], workers=-1, distance_upper_bound=high_near)
            part[hi_[np.isfinite(d_)]] = "walls"
        # what is left, in pieces again (the split leaves fragments): one floating (off the
        # floor) within ledge_near of a wall, or with 90 % of it within with_wall of one (a
        # door's part, a frame, a strip; a chair reaches out more), is wall: near the walls
        # as they were before the high cut (a rack's top taken by it does not pull the rack
        # after it); a scrap under min_obj m2 goes with the nearest labelled face
        ob = np.flatnonzero(part == "objects")
        if len(ob) and len(wl_):
            cm = pieces(ob)
            nk = cm.max() + 1
            k_lo = np.full(nk, np.inf)
            np.minimum.at(k_lo, cm, dz[ob])
            d_, _ = cKDTree(C[wl_]).query(C[ob], workers=-1, distance_upper_bound=ledge_near)
            d_ = np.minimum(d_, plane_dist(ob))
            near_ = np.zeros(nk, bool)
            near_[np.unique(cm[d_ <= ledge_near])] = True
            o_ = np.lexsort((d_, cm))
            cs_, ds_ = cm[o_], d_[o_]
            st = np.r_[0, np.flatnonzero(np.diff(cs_)) + 1]
            cnt = np.diff(np.r_[st, len(cs_)])
            far90 = np.full(nk, np.inf)
            far90[cs_[st]] = ds_[st + np.minimum((0.9 * cnt).astype(np.int64), cnt - 1)]
            new = (near_ & (k_lo >= float_low)) | (far90 <= with_wall)
            part[ob[new[cm]]] = "walls"
        if len(ob):
            if not len(wl_):
                cm = pieces(ob)
                nk = cm.max() + 1
            sel = part[ob] == "objects"
            k_area = np.bincount(cm, A[ob] * sel, nk)
            sf = ob[sel & (k_area[cm] < min_obj)]           # scraps
            part[sf] = ""
            sel = part[ob] == "objects"
            _, num = np.unique(cm[sel], return_inverse=True)
            obj[ob[sel]] = num.ravel()
            lab = np.flatnonzero(part != "")
            if len(sf) and len(lab):
                _, i = cKDTree(C[lab]).query(C[sf], workers=-1)
                part[sf], obj[sf] = part[lab[i]], obj[lab[i]]
            elif len(sf):
                part[sf] = "objects"
    return part, obj


def flatten_floor(M, floor, cell=1.0, level=0.95, most=0.4):
    """The vertices of the floor faces (mask) of mesh M put on the floor's
    own level: per cell (m, in plan), the median height of its flat floor
    faces (|n_z| over level), between cells the four nearest blended
    (inverse distance: no steps); a vertex is moved only up to `most` m (a
    floor at another level, a landing, is not a bump). The bumps
    under furniture (legs, feet melted into the floor by Poisson) go flat.
    The number of vertices moved and the most any moved (m)."""
    o3d = _o3d()
    M.compute_triangle_normals()
    V, F = np.asarray(M.vertices).copy(), np.asarray(M.triangles)
    C = V[F].mean(1)
    ref = floor & (np.abs(np.asarray(M.triangle_normals)[:, 2]) > level)
    if not ref.any():
        return 0, 0.0
    lo = C[ref, :2].min(0)
    ij = np.floor((C[ref, :2] - lo) / cell).astype(np.int64)
    ny = int(ij[:, 1].max()) + 2
    key = ij[:, 0] * ny + ij[:, 1]
    o = np.lexsort((C[ref, 2], key))
    ks = key[o]
    starts = np.r_[0, np.flatnonzero(np.diff(ks)) + 1]
    cnt = np.diff(np.r_[starts, len(ks)])
    cz = C[ref, 2][o][starts + cnt // 2]                  # each cell's median
    cxy = lo + (np.c_[ks[starts] // ny, ks[starts] % ny] + 0.5) * cell
    vi = np.unique(F[floor])
    kk = min(4, len(cz))
    d, near = cKDTree(cxy).query(V[vi, :2], k=kk, workers=-1)
    d, near = d.reshape(len(vi), kk), near.reshape(len(vi), kk)
    w = 1.0 / np.maximum(d, 1e-3)
    z = (w * cz[near]).sum(1) / w.sum(1)
    moved = np.abs(V[vi, 2] - z)
    ok = moved <= most                       # bumps only: a floor at another level is not the bump's
    V[vi[ok], 2] = z[ok]
    moved = moved[ok]
    M.vertices = o3d.utility.Vector3dVector(V)
    return int((moved > 1e-4).sum()), float(moved.max()) if len(moved) else 0.0


def segment_colours(M, part, obj, planes, weight=1.0, snap=0.06, margin=0.1, glass=(0.72, 0.82, 0.88)):
    """The mesh's colours made one per segment, so it looks clean: each
    wall plane the median colour of its vertices (within snap m of it:
    the median leaves the glints out), what stands off the planes
    (pillars, frames, stairs) the median of its own connected piece, the
    ceiling one colour, the floor one, each object one, glass the tint
    `glass`; blended with the colours as they are by `weight` (1: flat).
    The new vertex colours (n x 3)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    V, F = np.asarray(M.vertices), np.asarray(M.triangles)
    C = np.asarray(M.vertex_colors) if M.has_vertex_colors() else np.full((len(V), 3), 0.6)
    Cn = C.copy()

    def paint(vi, colour):
        Cn[vi] = colour

    def flat(vi):
        if len(vi):
            paint(vi, np.median(C[vi], axis=0))
    for name in ("floor", "ceiling"):
        flat(np.unique(F[part == name]))
    for k in np.unique(obj[part == "objects"]):
        flat(np.unique(F[(part == "objects") & (obj == k)]))
    wf = np.flatnonzero(part == "walls")
    if len(wf):
        vi = np.unique(F[wf])
        best, which = np.full(len(vi), np.inf), np.full(len(vi), -1)
        for j, pl in enumerate(planes or []):
            mu, n, t = pl["mu"], pl["n"], pl["t"]
            q = V[vi] - mu
            d, u = np.abs(q @ n), q @ t
            ok = ((d < snap) & (d < best) & (u > pl["u0"] - margin) & (u < pl["u1"] + margin)
                  & (V[vi, 2] > pl["floor"] - margin) & (V[vi, 2] < pl["top"] + margin))
            best[ok], which[ok] = d[ok], j
        for j in np.unique(which[which >= 0]):
            flat(vi[which == j])
        off = vi[which < 0]                      # off the planes: each connected piece of those faces
        if len(off):
            isoff = np.zeros(len(V), bool)
            isoff[off] = True
            Fo = F[wf][isoff[F[wf]].all(1)]
            if len(Fo):
                e = np.r_[Fo[:, 0], Fo[:, 1]], np.r_[Fo[:, 1], Fo[:, 2]]
                _, vc = connected_components(coo_matrix((np.ones(len(e[0])), e), shape=(len(V), len(V))),
                                             directed=False)
                vo = np.unique(Fo)
                for k in np.unique(vc[vo]):
                    flat(vo[vc[vo] == k])
    paint(np.unique(F[part == "glass"]), np.asarray(glass, float))     # last: a pane's rim is the wall's too
    return weight * Cn + (1 - weight) * C


def flatten_walls(M, part, planes, snap=0.06, margin=0.1):
    """The vertices of the wall faces (part == "walls") within snap m of a
    wall plane (inside its extent, margin m round it) put on it, each on its
    nearest plane: the wall smooth (the meshing's ripples gone); what stands
    out of it farther (a ledge, a frame, a pillar) is kept in shape but
    smoothed (Taubin, its ripples gone too). The number of vertices put on
    a plane, the most any moved (m), the number smoothed."""
    o3d = _o3d()
    V, F = np.asarray(M.vertices).copy(), np.asarray(M.triangles)
    vi = np.unique(F[part == "walls"]) if (part == "walls").any() else np.zeros(0, np.int64)
    if not planes or not len(vi):
        return 0, 0.0
    best, shift = np.full(len(vi), np.inf), np.zeros((len(vi), 3))
    for pl in planes:
        mu, n, t = pl["mu"], pl["n"], pl["t"]
        q = V[vi] - mu
        d, u = q @ n, q @ t
        ok = ((np.abs(d) < snap) & (np.abs(d) < best) & (u > pl["u0"] - margin) & (u < pl["u1"] + margin)
              & (V[vi, 2] > pl["floor"] - margin) & (V[vi, 2] < pl["top"] + margin))
        best[ok] = np.abs(d[ok])
        shift[ok] = -d[ok, None] * n
    on = np.isfinite(best)
    V[vi[on]] += shift[on]
    off = vi[~on]
    other = np.unique(F[part != "walls"]) if (part != "walls").any() else np.zeros(0, np.int64)
    off = off[~np.isin(off, other)]                  # not a vertex the floor or the ceiling shares (a seam)
    if len(off):
        M.vertices = o3d.utility.Vector3dVector(V)
        Vs = np.asarray(M.filter_smooth_taubin(number_of_iterations=10).vertices)
        V[off] = Vs[off]
    M.vertices = o3d.utility.Vector3dVector(V)
    return int(on.sum()), float(best[on].max()) if on.any() else 0.0, int(len(off))


def close_holes(M, part, obj, radius=0.8, pane=0.2):
    """The mesh's holes up to `radius` m across closed (Open3D's fill_holes:
    their rims triangulated; a doorway is larger, left open): the glass
    panes the lidar saw through, the gaps the meshing left. The new faces
    are added to M and take the part (and object) of the nearest labelled
    face (next to faces left out, "gone": left out too); a hole of pane m2
    or more in a wall is glass. (part, obj) grown,
    and the number of faces added."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    o3d = _o3d()
    F0 = np.asarray(M.triangles)
    tm = o3d.t.geometry.TriangleMesh(o3d.core.Tensor(np.asarray(M.vertices)), o3d.core.Tensor(F0))
    F1 = tm.fill_holes(hole_size=radius).triangle.indices.numpy()
    if len(F1) <= len(F0) or not np.array_equal(F1[:len(F0)], F0):
        return part, obj, 0
    Fn = F1[len(F0):]
    V = np.asarray(M.vertices)
    lab = np.flatnonzero(part != "")                 # (a hole next to faces left out is left out too)
    _, i = cKDTree(V[F0[lab]].mean(1)).query(V[Fn].mean(1), workers=-1)
    pn, on = part[lab[i]].copy(), obj[lab[i]].copy()
    # each hole's faces (joined), its area: a pane of glass in a wall
    e = np.r_[Fn[:, 0], Fn[:, 1]], np.r_[Fn[:, 1], Fn[:, 2]]
    _, vc = connected_components(coo_matrix((np.ones(len(e[0])), e), shape=(len(V), len(V))), directed=False)
    _, hole = np.unique(vc[Fn[:, 0]], return_inverse=True)
    hole = hole.ravel()
    An = 0.5 * np.linalg.norm(np.cross(V[Fn[:, 1]] - V[Fn[:, 0]], V[Fn[:, 2]] - V[Fn[:, 0]]), axis=1)
    big = np.bincount(hole, An)[hole] >= pane
    pn[(pn == "walls") & big] = "glass"
    M.triangles = o3d.utility.Vector3iVector(F1)
    return np.r_[part, pn], np.r_[obj, on], len(Fn)


def fill_floor_part(M, part, cell=0.2, max_hole=100.0):
    """The floor's holes (where the polished floor's reflection was cut away,
    or furniture hid it): the cells (cell m, in plan) with no floor face
    that the floor, and the walls and furniture standing on it (faces
    within 0.5 m over it), close all round, up to max_hole m2 (a courtyard
    is larger), filled with flat patches at the floor's level there (the
    nearest floor faces'), in the floor's colour. (vertices, faces,
    colours) of the patches, to add."""
    from scipy import ndimage
    V, F = np.asarray(M.vertices), np.asarray(M.triangles)
    fl = np.flatnonzero(part == "floor")
    if not len(fl):
        return np.zeros((0, 3)), np.zeros((0, 3), np.int64), np.zeros((0, 3))
    C = V[F[fl]].mean(1)
    lo = C[:, :2].min(0) - cell
    ij = np.floor((C[:, :2] - lo) / cell).astype(np.int64)
    nx, ny = ij.max(0) + 2
    have = np.zeros((nx, ny), bool)
    have[ij[:, 0], ij[:, 1]] = True
    on = np.flatnonzero((part == "walls") | (part == "objects"))      # standing on the floor: close a hole too
    if len(on):
        Co = V[F[on]].mean(1)
        _, jn = cKDTree(C[:, :2]).query(Co[:, :2], workers=-1)
        low = Co[:, 2] - C[jn, 2] < 0.5
        ijo = np.floor((Co[low, :2] - lo) / cell).astype(np.int64)
        ok = (ijo >= 0).all(1) & (ijo[:, 0] < nx) & (ijo[:, 1] < ny)
        closing = have.copy()
        closing[ijo[ok, 0], ijo[ok, 1]] = True
    else:
        closing = have
    holes = ndimage.binary_fill_holes(closing) & ~have
    lab, nl = ndimage.label(holes)
    if not nl:
        return np.zeros((0, 3)), np.zeros((0, 3), np.int64), np.zeros((0, 3))
    small = np.bincount(lab.ravel(), minlength=nl + 1) * cell * cell <= max_hole
    small[0] = False
    iu, iz = np.nonzero(small[lab])
    if not len(iu):
        return np.zeros((0, 3)), np.zeros((0, 3), np.int64), np.zeros((0, 3))
    corners = np.array([(0, 0), (1, 0), (1, 1), (0, 1)], float)
    P = (lo + (np.c_[iu, iz][:, None, :] + corners) * cell).reshape(-1, 2)
    _, jn = cKDTree(C[:, :2]).query(P, workers=-1)
    P = np.c_[P, C[jn, 2]]
    q = 4 * np.arange(len(iu))[:, None]
    Fs = np.vstack([q + np.array([0, 1, 2]), q + np.array([0, 2, 3])])
    col = np.asarray(M.vertex_colors) if M.has_vertex_colors() else None
    Cs = col[F[fl[jn], 0]] if col is not None else np.full((len(P), 3), 0.6)
    return P, Fs, Cs


def fill_walls(M, part, planes, cell=0.1, on=0.03, depth=0.4, max_hole=1.0, door_w=2.5):
    """The wall hidden behind furniture, which the mesh does not have: on
    each wall plane, the cells (cell m) from the floor to its top where no
    wall face is (within depth m of it: on it, or set back like a door or a
    panel) but something stands against it (within depth m, either side:
    furniture pushed to the wall), filled
    with a flat patch on the plane, in the wall's own colour. Doors,
    windows and openings, larger and nothing against them: left open; and
    the wall's own holes, closed all round by it (the floor closing them
    below, its top above), up to max_hole m2: the wall continuous. A larger
    one that is not a doorway (off the floor: a window; or wider than
    door_w m: a glass wall) is glass, the lidar saw through it: filled the
    same, apart. ((vertices, faces, colours) of the wall patches, the same
    of the glass), to add."""
    V, F = np.asarray(M.vertices), np.asarray(M.triangles)
    C = V[F].mean(1)
    col = np.asarray(M.vertex_colors) if M.has_vertex_colors() else None
    wl, ob = np.flatnonzero(part == "walls"), np.flatnonzero(part == "objects")
    empty = (np.zeros((0, 3)), np.zeros((0, 3), np.int64), np.zeros((0, 3)))
    if not planes or not len(wl):
        return empty, empty
    if not len(ob):
        ob = wl[:1]
    wtree, otree = cKDTree(C[wl, :2]), cKDTree(C[ob, :2])
    out = {"walls": [[], [], [], 0], "glass": [[], [], [], 0]}       # vertices, faces, colours, n vertices
    for pl in planes:
        mu, n, t = pl["mu"], pl["n"], pl["t"]
        u0, u1, z0, z1 = pl["u0"], pl["u1"], pl["floor"], pl["top"]
        nu, nz = int(np.ceil((u1 - u0) / cell)), int(np.ceil((z1 - z0) / cell))
        if nu < 1 or nz < 1:
            continue
        mid, rad = mu[:2] + t[:2] * (u0 + u1) / 2, (u1 - u0) / 2 + depth

        def cells(idx):
            u = (C[idx] - mu) @ t
            iu = np.floor((u - u0) / cell).astype(np.int64)
            iz = np.floor((C[idx, 2] - z0) / cell).astype(np.int64)
            ok = (iu >= 0) & (iu < nu) & (iz >= 0) & (iz < nz)
            g = np.zeros((nu, nz), bool)
            g[iu[ok], iz[ok]] = True
            return g
        wi = wl[np.asarray(wtree.query_ball_point(mid, rad), np.int64)]
        wi = wi[np.abs((C[wi] - mu) @ n) < depth]      # on the plane, or set back from it (a door, a panel):
        oi = ob[np.asarray(otree.query_ball_point(mid, rad), np.int64)]            # the wall is there
        oi = oi[np.abs((C[oi] - mu) @ n) < depth]
        from scipy import ndimage
        have = ndimage.binary_dilation(cells(wi), np.ones((3, 3), bool)) if len(wi) else np.zeros((nu, nz), bool)
        front = ndimage.binary_dilation(cells(oi), np.ones((3, 3), bool)) if len(oi) else np.zeros((nu, nz), bool)
        # the wall's own holes: closed all round by it (the floor closing it below, its top
        # above): up to max_hole m2 wall; larger and not a doorway (off the floor, or wider
        # than a door) glass
        ones = np.ones((nu, 1), bool)
        closed = ndimage.binary_fill_holes(np.c_[ones, have, ones])[:, 1:-1] & ~have
        lab_, nlab_ = ndimage.label(closed)
        glass = np.zeros((nu, nz), bool)
        if nlab_:
            area = np.bincount(lab_.ravel(), minlength=nlab_ + 1) * cell * cell
            hu, hz = np.nonzero(closed)
            hl = lab_[hu, hz]
            z_lo, u_lo, u_hi = np.full(nlab_ + 1, nz), np.full(nlab_ + 1, nu), np.full(nlab_ + 1, -1)
            np.minimum.at(z_lo, hl, hz)
            np.minimum.at(u_lo, hl, hu)
            np.maximum.at(u_hi, hl, hu)
            small = area <= max_hole
            pane = ~small & ((z_lo > 0) | ((u_hi - u_lo + 1) * cell > door_w))
            small[0] = pane[0] = False
            glass = pane[lab_]
            closed = small[lab_]
        for name, g in (("walls", (front | closed) & ~have & ~glass), ("glass", glass)):
            iu, iz = np.nonzero(g)
            if not len(iu):
                continue
            Vs, Fs, Cs, nv = out[name]
            a_u, a_z = u0 + iu * cell, z0 + iz * cell
            corners = [(0, 0), (1, 0), (1, 1), (0, 1)]
            P = np.stack([np.c_[mu[0] + t[0] * (a_u + du * cell), mu[1] + t[1] * (a_u + du * cell), a_z + dz_ * cell]
                          for du, dz_ in corners], 1).reshape(-1, 3)
            q = nv + 4 * np.arange(len(iu))[:, None]
            Fs.append(np.vstack([q + np.array([0, 1, 2]), q + np.array([0, 2, 3])]))
            Vs.append(P)
            if col is not None and len(wi):              # each corner the nearest wall vertex's colour: the
                vw = np.unique(F[wi].ravel())              # wall's own mottling carried on, not a flat block
                _, jn = cKDTree(V[vw]).query(P, workers=-1)
                Cs.append(col[vw[jn]])
            else:
                Cs.append(np.full((len(P), 3), 0.6))
            out[name][3] += len(P)
    empty = (np.zeros((0, 3)), np.zeros((0, 3), np.int64), np.zeros((0, 3)))
    return tuple((np.vstack(o[0]), np.vstack(o[1]), np.vstack(o[2])) if o[0] else empty
                 for o in (out["walls"], out["glass"]))


def sionna_mesh(mesh_path, out=None, faces=0, materials=None, label_faces_n=4000000, split_objects=False,
                object_faces=0,
                floor_colour=(0.8, 0.8, 0.8), traj=None, ceil_margin=0.10, min_piece=50.0, normals=False,
                why=(), flat_floor=True, fill=True, why_objects=0, flat_walls=True, colours=True,
                flat_colours=0.0):
    """The mesh (the one with its ceiling: the radio waves bounce off it) as
    a Sionna RT scene, its own faces split into floor, walls, ceiling and
    objects: <out>/scene.xml (what Sionna loads: sionna.rt.load_scene) with
    <out>/meshes/{floor,walls,ceiling,objects}.ply, each an ITU radio
    material (materials: {part: (ITU type, thickness m)} over
    SIONNA_MATERIALS; split_objects: each object its own
    meshes/objects/obj_NNN.ply), each in the mesh's colours (the floor
    floor_colour; Sionna shows its materials' own; normals: with vertex
    normals, so a viewer shades them as surfaces). Every face is in one
    part or left out (written to <out>_left_out.ply), checked, and the
    counts written to <out>/parts.json. To
    look at: <out>.ply, all of it in its own colours with the floor
    floor_colour, and <out>_labels.ply in the parts' colours (PART_COLOURS).

    Nothing is rebuilt: faces > 0 simplifies the floor, walls and ceiling
    together to about that many (quadric: flat stays flat, in few faces) and
    object_faces > 0 the objects on their own to that many, so the objects
    keep finer detail (0: as meshed); else it is taken as it is. The labels
    (label_faces) are found on an even copy of about label_faces_n faces
    (vertex clustering: faces of one size, so a flat wall keeps enough of
    them to be seen flat) and each face takes its nearest one's. traj (the mesh's frame): the floor level from it (so a
    reflection under the floor cannot pull it down), and what the _noceil copy drops
    as hanging from the ceiling (hanging_faces: a reflection joined to a
    mirrored ceiling) is left out. flat_floor: the floor flattened onto its
    level (flatten_floor); flat_walls: the walls smoothed onto their planes
    (flatten_walls); fill: the walls filled behind furniture and their small
    holes (fill_walls); colours False: the parts and <out>.ply written
    without colours (Sionna uses none: smaller files); flat_colours > 0:
    the colours made one per segment (segment_colours), blended in by that
    much (1: flat), so it looks clean."""
    o3d = _o3d()
    mesh_path = os.path.expanduser(mesh_path)
    base = os.path.splitext(out)[0] if out else os.path.splitext(mesh_path)[0] + "_sionna"
    mats = {**SIONNA_MATERIALS, **(materials or {})}
    M = o3d.io.read_triangle_mesh(mesh_path)
    if not len(M.triangles):
        raise ConvertError(f"{mesh_path}: no faces (give it the mesh, not the view)")
    n0 = len(M.triangles)
    M.remove_duplicated_vertices()
    m = M
    if label_faces_n and len(M.triangles) > label_faces_n:
        # an even copy (vertex clustering): quadric decimation leaves a flat wall a few
        # large faces, too few in a 0.4 m cell for plane_faces to see it flat
        vox = float(np.sqrt(2.0 * M.get_surface_area() / label_faces_n))
        log.info(f"labelling on an even copy of about {label_faces_n} faces ({100 * vox:.0f} cm) ...")
        m = M.simplify_vertex_clustering(voxel_size=vox,
                                         contraction=_o3d().geometry.SimplificationContraction.Average)
        m.remove_degenerate_triangles()
        m.remove_duplicated_triangles()
        m.remove_unreferenced_vertices()
    T = read_tum_positions(traj, stride=1) if traj else None
    planes = []
    part, obj = label_faces(m, T=T, why=why, planes=planes, why_objects=why_objects)
    V, F = np.asarray(m.vertices), np.asarray(m.triangles)
    C = V[F].mean(1)
    A = 0.5 * np.linalg.norm(np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]), axis=1)
    report = [f"sionna {mesh_path}: {n0} faces"]
    if traj and (part == "gone").any():
        report.append(f"  under the floor (the room mirrored in the polished floor): "
                      f"{A[part == 'gone'].sum():.1f} m2 left out")
    if traj:
        hang = hanging_faces(m, T, ceil_margin, min_piece)
        part[hang] = "gone"
        obj[hang] = -1
        report.append(f"  hanging from the ceiling only (as _noceil drops them: a reflection joined to a "
                      f"mirrored ceiling): {A[hang].sum():.1f} m2 left out")
    # every face of the mesh: its nearest labelled face's part (and object)
    if m is M:
        fpart, fobj = part, obj
    else:
        Vf, Ff = np.asarray(M.vertices), np.asarray(M.triangles)
        tree = cKDTree(C)
        fpart, fobj = np.empty(len(Ff), dtype=object), np.empty(len(Ff), np.int64)
        for a0 in range(0, len(Ff), 2000000):
            _, i = tree.query(Vf[Ff[a0:a0 + 2000000]].mean(1), workers=-1)
            fpart[a0:a0 + 2000000], fobj[a0:a0 + 2000000] = part[i], obj[i]
        del tree
    # the holes up to 0.8 m across closed: the glass panes the lidar saw through (glass),
    # the gaps the meshing left (the part round them); doorways, larger, stay open
    n_fill = 0
    if fill:
        fpart, fobj, nh = close_holes(M, fpart, fobj)
        n_fill += nh
        report.append(f"  holes up to 0.8 m across closed: {nh} faces added "
                      f"({int((fpart[len(fpart) - nh:] == 'glass').sum()) if nh else 0} of them glass panes)")
    # simplified apart: floor, walls and ceiling together to `faces` (their seams
    # stay shut), the objects on their own to `object_faces`, finer
    if (faces and (fpart != "objects").sum() > faces) or (object_faces and (fpart == "objects").sum() > object_faces):
        def piece(mask, n, keep_ids):
            q = type(M)(M)
            q.remove_triangles_by_mask(~mask)
            q.remove_unreferenced_vertices()
            n_in = len(q.triangles)
            if n and n_in > n:
                q = q.simplify_quadric_decimation(target_number_of_triangles=int(n))
                q.remove_degenerate_triangles()
                q.remove_unreferenced_vertices()
            lab = np.flatnonzero(keep_ids)                       # its faces' parts again, from the labelled
            Vq, Fq = np.asarray(q.vertices), np.asarray(q.triangles)
            _, i = cKDTree(C[lab]).query(Vq[Fq].mean(1), workers=-1) if len(Fq) else (None, np.zeros(0, int))
            return q, part[lab][i], obj[lab][i], n_in
        structure = (fpart == "floor") | (fpart == "walls") | (fpart == "ceiling")
        qs, ps, os_, ns = piece(structure, faces, (part == "floor") | (part == "walls") | (part == "ceiling"))
        qo, po, oo, no = piece(fpart == "objects", object_faces, part == "objects")
        qg = type(M)(M)
        qg.remove_triangles_by_mask(fpart != "gone")
        qg.remove_unreferenced_vertices()
        pg = np.full(len(qg.triangles), "gone", dtype=object)
        M = qs + qo + qg
        fpart = np.r_[ps, po, pg]
        fobj = np.r_[os_, oo, np.full(len(pg), -1)]
        report.append(f"  floor, walls and ceiling simplified {ns} -> {len(qs.triangles)} faces, objects "
                      + (f"{no} -> {len(qo.triangles)}" if object_faces and no > object_faces else f"{no} as meshed")
                      + " (finer)")
    if flat_floor:
        nmv, dmax = flatten_floor(M, fpart == "floor")
        report.append(f"  floor flattened onto its own level: {nmv} vertices moved, {100 * dmax:.1f} cm at most "
                      "(the bumps under furniture)")
    if flat_floor:
        nmv, dmax = flatten_floor(M, fpart == "ceiling", most=0.15)
        report.append(f"  ceiling flattened onto its own levels: {nmv} vertices moved, {100 * dmax:.1f} cm at most")
    if flat_walls:
        nmv, dmax, nsm = flatten_walls(M, fpart, planes)
        report.append(f"  walls smoothed onto their planes: {nmv} vertices moved, {100 * dmax:.1f} cm at most; "
                      f"off them (pillars, stairs, frames) {nsm} vertices smoothed")
    def add(Vn, Fn, Cn, name):
        nonlocal fpart, fobj, n_fill
        nv0 = len(M.vertices)
        col0 = np.asarray(M.vertex_colors).copy() if M.has_vertex_colors() else None   # before the count changes
        M.vertices = o3d.utility.Vector3dVector(np.vstack([np.asarray(M.vertices), Vn]))
        if col0 is not None:
            M.vertex_colors = o3d.utility.Vector3dVector(np.vstack([col0, Cn]))
        M.triangles = o3d.utility.Vector3iVector(np.vstack([np.asarray(M.triangles), Fn + nv0]))
        fpart = np.r_[fpart, np.full(len(Fn), name, dtype=object)]
        fobj = np.r_[fobj, np.full(len(Fn), -1)]
        n_fill += len(Fn)
    if fill:
        Vn, Fn, Cn = fill_floor_part(M, fpart)
        if len(Fn):
            add(Vn, Fn, Cn, "floor")
            report.append(f"  floor filled where it was missing (its reflection cut away, furniture over it): "
                          f"{len(Fn) // 2 * 0.04:.1f} m2, {len(Fn)} faces added")
        (Vn, Fn, Cn), (Vg, Fg, Cg) = fill_walls(M, fpart, planes)
        if len(Fn):
            add(Vn, Fn, Cn, "walls")
            report.append(f"  walls filled behind furniture (the mesh has none there) and their small holes: "
                          f"{len(Fn) // 2 * 0.01:.1f} m2, {len(Fn)} faces added")
        if len(Fg):
            add(Vg, Fg, Cg, "glass")
            report.append(f"  glass: the walls' larger openings that are not doorways (windows, glass walls the "
                          f"lidar saw through) filled apart: {len(Fg) // 2 * 0.01:.1f} m2, {len(Fg)} faces added")
    if flat_colours and colours:
        M.vertex_colors = o3d.utility.Vector3dVector(segment_colours(M, fpart, fobj, planes, weight=flat_colours))
        report.append(f"  colours one per segment (each wall plane, pillar, object; the ceiling; glass tinted), "
                      f"blended in {100 * flat_colours:.0f} %")
    Vf, Ff = np.asarray(M.vertices), np.asarray(M.triangles)
    Af = 0.5 * np.linalg.norm(np.cross(Vf[Ff[:, 1]] - Vf[Ff[:, 0]], Vf[Ff[:, 2]] - Vf[Ff[:, 0]]), axis=1)
    for p_ in PARTS:
        sel = fpart == p_
        if p_ == "glass" and not sel.any():
            continue
        report.append(f"  {p_}: {Af[sel].sum():.0f} m2, {int(sel.sum())} faces"
                      + (f", {len(np.unique(fobj[sel]))} objects" if p_ == "objects" and sel.any() else ""))

    def bare(q, colours=False):
        q = type(q)(q)
        for attr in ("vertex_normals", "triangle_normals") + (() if colours else ("vertex_colors",)):
            setattr(q, attr, o3d.utility.Vector3dVector(np.zeros((0, 3))))
        return q

    def sub_mesh(mask):
        q = type(M)(M)
        q.remove_triangles_by_mask(~mask)
        q.remove_unreferenced_vertices()
        return q

    sdir = base
    os.makedirs(os.path.join(sdir, "meshes"), exist_ok=True)
    xml_mats, shapes, used = {}, [], {}

    def material(name):
        typ, th = mats[name]
        mid = f"{name}_{typ}"          # one per part: Sionna merges shapes sharing a material
        if mid not in xml_mats:
            xml_mats[mid] = [f'    <bsdf type="itu-radio-material" id="mat-{mid}">',
                             f'        <string name="type" value="{typ}"/>',
                             f'        <float name="thickness" value="{float(th):g}"/>',
                             "    </bsdf>"]
        return mid

    def shape(sid, rel, mid):
        shapes.extend([f'    <shape type="ply" id="mesh-{sid}">',
                       f'        <string name="filename" value="{rel}"/>',
                       '        <boolean name="face_normals" value="true"/>',
                       f'        <ref id="mat-{mid}" name="bsdf"/>',
                       "    </shape>"])

    def coloured(q, name):
        """A part in the mesh's colours, the floor floor_colour (colours False: none)."""
        q = bare(q, colours=colours)
        if colours and not q.has_vertex_colors():
            q.paint_uniform_color((0.6, 0.6, 0.6))
        if colours and name == "floor" and floor_colour is not None:
            q.paint_uniform_color(floor_colour)
        if normals:                    # a viewer then shades it as a surface (its colours greyed)
            q.compute_vertex_normals()
        return q

    lab = o3d.geometry.TriangleMesh()
    for name in PARTS:
        sel = fpart == name
        if not sel.any():
            continue
        used[name] = int(sel.sum())
        mid = material(name)
        q = sub_mesh(sel)
        ql = bare(q)
        ql.paint_uniform_color(PART_COLOURS[name])
        lab += ql
        if name == "objects" and split_objects:
            os.makedirs(os.path.join(sdir, "meshes", "objects"), exist_ok=True)
            for k, oid in enumerate(np.unique(fobj[sel])):
                rel = f"meshes/objects/obj_{k:03d}.ply"
                o3d.io.write_triangle_mesh(os.path.join(sdir, rel), coloured(sub_mesh(sel & (fobj == oid)), name))
                shape(f"obj_{k:03d}", rel, mid)
        else:
            o3d.io.write_triangle_mesh(os.path.join(sdir, "meshes", name + ".ply"), coloured(q, name))
            shape(name, f"meshes/{name}.ply", mid)
    sx = os.path.join(sdir, "scene.xml")
    open(sx, "w").write("\n".join(['<scene version="2.1.0">', *sum(xml_mats.values(), []), *shapes,
                                   "</scene>"]) + "\n")
    view = bare(sub_mesh(fpart != "gone"), colours=colours)
    if colours and not view.has_vertex_colors():
        view.paint_uniform_color((0.6, 0.6, 0.6))
    if colours and floor_colour is not None:
        Cv = np.asarray(view.vertex_colors).copy()
        Cv[np.unique(np.asarray(view.triangles)[fpart[fpart != "gone"] == "floor"])] = floor_colour
        view.vertex_colors = o3d.utility.Vector3dVector(Cv)
    vp, lp = base + ".ply", base + "_labels.ply"
    o3d.io.write_triangle_mesh(vp, view)
    o3d.io.write_triangle_mesh(lp, lab)
    # every face accounted: in a part, or left out (written apart to look at)
    n_out = int((fpart == "gone").sum())
    if sum(used.values()) + n_out != len(Ff):
        raise ConvertError(f"faces not accounted: {len(Ff)} in, {sum(used.values())} in the parts, {n_out} left out")
    op = None
    if n_out:
        op = base + "_left_out.ply"
        o3d.io.write_triangle_mesh(op, bare(sub_mesh(fpart == "gone"), colours=True))
    import json
    json.dump({"mesh": mesh_path, "faces_in": int(n0), "faces": int(len(Ff)),
               "parts": {n: {"faces": used[n], "m2": round(float(Af[fpart == n].sum()), 2),
                             "itu": mats[n][0], "thickness_m": float(mats[n][1])} for n in used},
               "left_out": {"faces": n_out, "m2": round(float(Af[fpart == "gone"].sum()), 2), "file": op},
               "filled": {"faces": n_fill, "m2": round(float(Af[len(Af) - n_fill:].sum()), 2) if n_fill else 0.0}},
              open(os.path.join(sdir, "parts.json"), "w"), indent=1)
    report += [f"every face accounted: {len(Ff)} = " + " + ".join(f"{used[n]} {n}" for n in used)
               + f" + {n_out} left out" + (f" ({op})" if op else "")
               + (f"  [{n_fill} faces added: holes closed, the floor where it was missing, the walls behind "
                  "furniture]" if n_fill else "")
               + (f"  [the mesh's {n0} faces simplified to {len(Ff) - n_fill} first]"
                  if len(Ff) - n_fill != n0 else "")]
    report += [f"sionna scene {sx} (loads meshes/ beside it; keep the folder whole; parts.json: the counts):"]
    report += [f"  {n}: {used[n]} faces, ITU {mats[n][0]} {float(mats[n][1]):g} m" for n in used]
    report += [f"  load: sionna.rt.load_scene(\"{sx}\"" + (", merge_shapes=False" if split_objects else "")
               + ")",
               f"to look at: {vp} (its own colours, the floor grey), {lp} (floor grey, walls beige, "
               "ceiling white, objects orange)"]
    log.info("\n".join(report))
    return {"out": sx, "scene": sx, "view": vp, "labels": lp, "faces": sum(used.values()), **used,
            "left_out": n_out, "left_out_file": op, "filled": n_fill,
            "n_objects": int(len(np.unique(fobj[fpart == "objects"])))}


def walk_start(V, T=None, at=None, eye=1.6, ahead=2.0):
    """(eye, look-at) to start walking a mesh of vertices V: at (x, y), else
    the first pose of T (the trajectory, the mesh's frame) looking along the
    path, else the middle of the mesh looking along +x; eye m over the floor
    there (the lowest 2 % of the vertices within 0.5 m in plan)."""
    if at is not None:
        xy, d = np.asarray(at, float)[:2], np.array([1.0, 0.0])
    elif T is not None and len(T):
        xy = T[0, :2]
        far = np.flatnonzero(np.linalg.norm(T[:, :2] - xy, axis=1) > ahead)
        d = (T[far[0], :2] - xy) if len(far) else np.array([1.0, 0.0])
    else:
        xy, d = 0.5 * (V[:, :2].min(0) + V[:, :2].max(0)), np.array([1.0, 0.0])
    near = np.linalg.norm(V[:, :2] - xy, axis=1) < 0.5
    floor = np.percentile(V[near, 2], 2) if near.any() else V[:, 2].min()
    e = np.r_[xy, floor + eye]
    return e, e + np.r_[d / np.linalg.norm(d), 0.0]


def smooth_colours(F, C, passes=1):
    """The mesh's colours (n x 3) blended with their neighbours' over the
    faces F (half each), `passes` times: the stipple of one colour a
    vertex softened."""
    C = np.asarray(C, float)
    n = len(C)
    for _ in range(passes):
        acc, deg = np.zeros((n, 3)), np.zeros(n)
        for a, b in ((0, 1), (1, 0), (1, 2), (2, 1), (0, 2), (2, 0)):
            deg += np.bincount(F[:, a], minlength=n)
            for ch in range(3):
                acc[:, ch] += np.bincount(F[:, a], weights=C[F[:, b], ch], minlength=n)
        has = deg > 0
        C = C.copy()
        C[has] = 0.5 * C[has] + 0.5 * acc[has] / deg[has, None]
    return C


def despeckle_colours(F, C, thresh=0.12, passes=2):
    """The mesh's colours (n x 3) with the speckle gone: a vertex much
    brighter or darker (thresh, in luminance) than the mean of its
    neighbours over the faces F (a glint, a shadow dot the camera's
    colouring left) takes that mean; `passes` times. (colours, vertices
    changed in the last pass)."""
    C = np.asarray(C, float).copy()
    n = len(C)
    w = np.array([0.299, 0.587, 0.114])
    changed = 0
    for _ in range(passes):
        acc, deg = np.zeros((n, 3)), np.zeros(n)
        for a, b in ((0, 1), (1, 0), (1, 2), (2, 1), (0, 2), (2, 0)):
            deg += np.bincount(F[:, a], minlength=n)
            for ch in range(3):
                acc[:, ch] += np.bincount(F[:, a], weights=C[F[:, b], ch], minlength=n)
        mean = acc / np.maximum(deg, 1)[:, None]
        out = (np.abs(C @ w - mean @ w) > thresh) & (deg > 0)
        C[out] = mean[out]
        changed = int(out.sum())
    return C, changed


def enhance_colours(C, low=1.0, high=99.0, sat=1.15, gamma=1.0, wb=True):
    """The mesh's colours (n x 3, 0-1: the camera's, flat and greyish) made
    vivid: the cast gone (wb: each channel scaled to the mean of the three,
    within a quarter: the green of fluorescent light, the blue of daylight;
    a white room white), then stretched so the low-th and high-th
    percentiles (the lowest and highest channel's) become 0 and 1 (the
    haze and the shadows' grey gone), the saturation times `sat` (about the
    grey of each colour) and a gamma (under 1: the shadows lifted). The
    same n x 3, clipped to 0-1."""
    C = np.asarray(C, float)
    if not len(C):
        return C
    if wb:
        mean = C.mean(0)
        C = C * np.clip(mean.mean() / np.maximum(mean, 1e-3), 0.8, 1.25)
    lo, hi = np.percentile(C, low, axis=0).min(), np.percentile(C, high, axis=0).max()
    C = (C - lo) / max(hi - lo, 1e-3)
    grey = C @ [0.299, 0.587, 0.114]
    C = grey[:, None] + sat * (C - grey[:, None])
    return np.clip(np.clip(C, 0, 1) ** gamma, 0, 1)


def mesh_blocks(m, cell=4.0, one_side=False):
    """The mesh in blocks of `cell` m in plan (each face with its first
    vertex's block), its faces doubled so a wall is seen from both sides
    (one_side: as stored): (the blocks' meshes, their centres in plan,
    unlit: they carry the mesh's colours). A renderer then draws only the
    blocks near the eye and in view."""
    o3d = _o3d()
    F, V = np.asarray(m.triangles), np.asarray(m.vertices)
    Cc = np.asarray(m.vertex_colors) if m.has_vertex_colors() else None
    if not one_side:
        F = np.vstack([F, F[:, ::-1]])
    lo = V[:, :2].min(0)
    ij = np.floor((V[F[:, 0], :2] - lo) / cell).astype(np.int64)
    ny = ij[:, 1].max() + 1
    key = ij[:, 0] * ny + ij[:, 1]
    order = np.argsort(key, kind="stable")
    ks, starts = np.unique(key[order], return_index=True)
    ends = np.r_[starts[1:], len(order)]
    blocks, centres = [], []
    for k, a_, b_ in zip(ks, starts, ends):
        vu, inv = np.unique(F[order[a_:b_]], return_inverse=True)
        q = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(V[vu]), o3d.utility.Vector3iVector(inv.reshape(-1, 3)))
        if Cc is not None:
            q.vertex_colors = o3d.utility.Vector3dVector(Cc[vu])
        else:
            q.compute_vertex_normals()
        blocks.append(q)
        centres.append(lo + (np.array([k // ny, k % ny]) + 0.5) * cell)
    return blocks, np.array(centres), Cc is not None


def loop_path(m, inset=1.5, cell=0.2, floor_tol=0.25, smooth=1.0, eye=1.6, fill_hole=10.0):
    """A closed path once round the area, in plan: the floor (the faces
    facing up at the floor's level, the lowest band holding a fifth of the
    most any band holds; stray faces under it do not count)
    as a grid of cell m, its small gaps closed, its holes up to fill_hole
    m2 filled (furniture's footprints; a courtyard, larger, is not floor),
    less every cell holding a face at walking height (0.3 to eye + 0.3 m over the floor: a wall,
    however thin, a pillar, furniture), shrunk inset m from those, the
    largest piece's outline (a room behind a door narrower than 2 inset
    falls off: the main area only), smoothed over `smooth` m. (the path,
    n x 2 about cell m apart; the floor level; free, the shrunk grid; its
    origin, with cell: a point (x, y) is on it at free[(x, y) - lo) / cell])."""
    from scipy import ndimage
    V, F = np.asarray(m.vertices), np.asarray(m.triangles)
    C = V[F].mean(1)
    N = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    A = np.linalg.norm(N, axis=1) / 2
    up = np.abs(N[:, 2]) > 0.9 * np.maximum(2 * A, 1e-12)
    zs = C[up, 2]
    hist, edges = np.histogram(zs, bins=np.arange(zs.min(), zs.max() + 0.05, 0.05), weights=A[up])
    z0 = edges[np.flatnonzero(hist >= 0.2 * hist.max())[0]] + 0.025     # the lowest band with much of it
    fl = up & (np.abs(C[:, 2] - z0) < floor_tol)
    o3d = _o3d()
    q = o3d.geometry.TriangleMesh(m.vertices, o3d.utility.Vector3iVector(F[fl]))   # the floor's faces, sampled
    Pf = np.asarray(q.sample_points_uniformly(max(1000, int(8 * A[fl].sum() / cell ** 2))).points)   # (large ones
    Pf = np.vstack([Pf, C[fl]])                                                                       # too)
    lo = Pf[:, :2].min(0) - 2 * cell
    ij = np.floor((Pf[:, :2] - lo) / cell).astype(np.int64)
    nx, ny = ij.max(0) + 3
    g = np.zeros((nx, ny), bool)
    g[ij[:, 0], ij[:, 1]] = True
    g = ndimage.binary_closing(g, np.ones((3, 3), bool))
    holes = ndimage.binary_fill_holes(g) & ~g
    lab_h, nh = ndimage.label(holes)
    if nh:
        small = np.bincount(lab_h.ravel()) * cell * cell <= fill_hole
        small[0] = False
        g |= small[lab_h]
    blk = (C[:, 2] > z0 + 0.3) & (C[:, 2] < z0 + eye + 0.3)              # at walking height: in the way
    ijb = np.floor((C[blk, :2] - lo) / cell).astype(np.int64)
    ok = (ijb >= 0).all(1) & (ijb[:, 0] < nx) & (ijb[:, 1] < ny)
    g[ijb[ok, 0], ijb[ok, 1]] = False
    g0 = g
    while True:                                      # shrunk inset m; a cramped place: less, told
        r = max(1, int(round(inset / cell)))
        yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
        g = ndimage.binary_erosion(g0, (xx ** 2 + yy ** 2) <= r * r)
        lab, nl = ndimage.label(g)
        if nl and np.bincount(lab.ravel())[1:].max() * cell * cell >= 4.0:
            break
        if inset <= 0.4:
            raise ConvertError("no free floor to fly round (a wall, a pillar or furniture within 0.4 m everywhere)")
        inset = round(inset / 2, 2)
        log.warning(f"no free floor that far from the walls and furniture: --inset {inset} m tried instead")
    big = np.argmax(np.bincount(lab.ravel())[1:]) + 1
    cs, _ = cv2.findContours((lab == big).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    pts = max(cs, key=len).reshape(-1, 2)[:, ::-1].astype(float)     # (row, col) = (i, j)
    P = lo + (pts + 0.5) * cell
    k = max(1, int(round(smooth / cell)))
    Pp = np.vstack([P[-k:], P, P[:k]])
    ker = np.ones(2 * k + 1) / (2 * k + 1)
    P = np.c_[np.convolve(Pp[:, 0], ker, "valid"), np.convolve(Pp[:, 1], ker, "valid")]
    return P, z0, lab == big, lo


class _FFmpegWriter:
    """Frames (BGR, h x w x 3) to an H.264 .mp4 through ffmpeg (yuv420p:
    every player and browser plays it; OpenCV's own mp4v many show black)."""

    def __init__(self, out, fps, size):
        import subprocess
        self.p = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
                                   "-s", f"{size[0]}x{size[1]}", "-r", str(fps), "-i", "-", "-an", "-c:v", "libx264",
                                   "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                                   out], stdin=subprocess.PIPE)

    def write(self, bgr):
        self.p.stdin.write(np.ascontiguousarray(bgr).tobytes())

    def release(self):
        self.p.stdin.close()
        self.p.wait()


def video_writer(out, fps, size):
    """A writer (write(bgr), release()) for the .mp4 out: ffmpeg (H.264) when
    it is installed, else OpenCV's mp4v."""
    import shutil
    if shutil.which("ffmpeg"):
        return _FFmpegWriter(out, fps, size)
    log.warning("no ffmpeg: the video in OpenCV's mp4v, which some players show black (apt install ffmpeg)")
    vw = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (size[0], size[1]))
    if not vw.isOpened():
        raise ConvertError(f"cannot write {out}")
    return vw


def fly_video(mesh_path, out=None, path=None, inset=1.5, eye=1.6, speed=1.0, fps=30, size=(1280, 720), fov=70.0,
              ahead=2.0, faces=0, vivid=True, sat=1.15, seconds=0, hidden=False, start=(0.0, 0.0), spin=8.0,
              wb=True, smooth=1):
    """A video (out, .mp4; default <mesh>_fly.mp4) flying once round the
    area inside the mesh at eye m over the floor, speed m/s, looking ahead
    m along the way: the path loop_path's (inset m from the walls), or
    `path`, a list of (x, y) waypoints joined in a loop; from `start` (x,
    y; the map's origin), turning once round there over `spin` s, then
    straight to the loop's nearest point (start None, or not on the free
    floor: from that point). Rendered in Open3D's OpenGL window (the one
    whose frames read back on every driver; hidden: not shown), the flight
    seen as it is recorded, the mesh unlit in its colours made vivid, seen
    from both sides, all of it however far. seconds > 0: that long at most. The
    path (n x 2) and the frames written."""
    o3d = _o3d()
    mesh_path = os.path.expanduser(mesh_path)
    out = os.path.expanduser(out) if out else os.path.splitext(mesh_path)[0] + "_fly.mp4"
    m = o3d.io.read_triangle_mesh(mesh_path)
    if not len(m.triangles):
        raise ConvertError(f"{mesh_path}: no faces (give it a mesh, not a view)")
    if faces and len(m.triangles) > faces:
        log.info(f"simplifying {len(m.triangles)} faces to {faces} ...")
        m.remove_duplicated_vertices()
        m = m.simplify_quadric_decimation(target_number_of_triangles=int(faces))
    if vivid and m.has_vertex_colors():
        Cd, nsp = despeckle_colours(np.asarray(m.triangles), np.asarray(m.vertex_colors))
        if smooth:
            Cd = smooth_colours(np.asarray(m.triangles), Cd, smooth)
        log.info(f"colours: {nsp} speckles smoothed over, blended {smooth} x, then made vivid"
                 + (", white balanced" if wb else ""))
        m.vertex_colors = o3d.utility.Vector3dVector(enhance_colours(Cd, sat=sat, wb=wb))
    P, z0, free, lo_ = loop_path(m, inset, eye=eye)
    if path is not None and len(path):
        P = np.asarray(path, float)[:, :2]
        P = np.vstack([P, P[:1]])
        seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
        P = np.vstack([np.linspace(P[i], P[i + 1], max(2, int(np.ceil(seg[i] / 0.2))), endpoint=False)
                       for i in range(len(seg))])
    # from the start to the loop's nearest point, round it and that far again (to look ahead)
    st_ = None
    if start is not None:
        st_ = np.asarray(start, float)[:2]
        ijs = np.floor((st_ - lo_) / 0.2).astype(np.int64)
        if not ((ijs >= 0).all() and ijs[0] < free.shape[0] and ijs[1] < free.shape[1] and free[ijs[0], ijs[1]]):
            log.warning(f"the start ({st_[0]:.1f}, {st_[1]:.1f}) is not on the free floor ({inset} m from "
                        "anything): starting on the loop")
            st_ = None
    k0 = int(np.argmin(np.linalg.norm(P - (st_ if st_ is not None else P[0]), axis=1)))
    P = np.roll(P, -k0, axis=0)
    way = np.vstack([P, P[:1]])
    if st_ is not None and np.linalg.norm(P[0] - st_) > 0.05:
        way = np.vstack([np.linspace(st_, P[0], max(2, int(np.ceil(np.linalg.norm(P[0] - st_) / 0.2))),
                                     endpoint=False), way])
    seg = np.linalg.norm(np.diff(way, axis=0), axis=1)
    sdist = np.r_[0, np.cumsum(seg)]
    total = sdist[-1]
    lseg = np.linalg.norm(np.diff(np.vstack([P, P[:1]]), axis=0), axis=1)
    lsd, ltot = np.r_[0, np.cumsum(lseg)], lseg.sum()

    def along(d):
        """The way's point d m along it (past its end: on round the loop)."""
        d = np.atleast_1d(d)
        on = d <= total
        q = np.empty((len(d), 2))
        q[on] = np.c_[np.interp(d[on], sdist, way[:, 0]), np.interp(d[on], sdist, way[:, 1])]
        dl = (d[~on] - total) % ltot
        q[~on] = np.c_[np.interp(dl, lsd, np.r_[P[:, 0], P[0, 0]]), np.interp(dl, lsd, np.r_[P[:, 1], P[0, 1]])]
        return q
    nmove = int(total / speed * fps)
    at_ = np.arange(nmove) * speed / fps
    E, Lk = along(at_), along(at_ + ahead)
    nspin = int(spin * fps) if (st_ is not None and spin > 0) else 0
    if nspin:
        yaw0 = np.arctan2(*(along(np.array([ahead]))[0] - st_)[::-1])    # ending facing the way to go
        ang = yaw0 + 2 * np.pi * np.arange(nspin) / nspin
        E = np.vstack([np.tile(st_, (nspin, 1)), E])
        Lk = np.vstack([st_ + ahead * np.c_[np.cos(ang), np.sin(ang)], Lk])
    nfr = len(E)
    if seconds:
        nfr = min(nfr, int(seconds * fps))
    log.info(f"fly {mesh_path}: " + (f"a turn round at ({st_[0]:.1f}, {st_[1]:.1f}) over {spin:g} s, " if nspin else "")
             + f"{total:.0f} m of way (a loop of {ltot:.0f} m), {nfr} frames at {fps} fps ({nfr / fps:.0f} s), "
             f"{len(m.triangles)} faces -> {out}")
    F_ = np.asarray(m.triangles)                   # every face both ways round: seen from both sides (the
    m.triangles = o3d.utility.Vector3iVector(np.vstack([F_, F_[:, ::-1]]))   # viewer's own back-face switch
    vis = o3d.visualization.Visualizer()           # is not honoured by every driver)
    if not vis.create_window(f"fly {os.path.basename(mesh_path)}: recording", size[0], size[1], visible=not hidden):
        raise ConvertError("no window: fly needs a display (the OpenGL one; ssh -X, or a desktop)")
    vis.add_geometry(m)
    ro = vis.get_render_option()
    ro.light_on = False                            # the colours as stored
    ro.mesh_show_back_face = True
    ro.background_color = np.array([0.12, 0.12, 0.12])
    vc = vis.get_view_control()
    vc.set_constant_z_near(0.05)
    vc.set_constant_z_far(1000.0)                  # all of it, however far
    cam = vc.convert_to_pinhole_camera_parameters()
    W, H = cam.intrinsic.width, cam.intrinsic.height          # the window as made (a scaled desktop: larger)
    if (W, H) != tuple(size):
        log.info(f"  the window came out {W}x{H}: the video that size")
    f_ = (H / 2) / np.tan(np.radians(fov) / 2)
    cam.intrinsic.set_intrinsics(W, H, f_, f_, W / 2 - 0.5, H / 2 - 0.5)
    vw = video_writer(out, fps, (W, H))
    first = os.path.splitext(out)[0] + "_frame0.png"
    up = np.array([0.0, 0.0, 1.0])
    for i in range(nfr):
        e_, l_ = np.r_[E[i], z0 + eye], np.r_[Lk[i], z0 + eye]
        fwd = l_ - e_
        fwd /= np.linalg.norm(fwd) or 1.0
        right = np.cross(fwd, up)
        right /= np.linalg.norm(right) or 1.0
        down = np.cross(fwd, right)
        R = np.vstack([right, down, fwd])              # camera x right, y down, z forward
        ext = np.eye(4)
        ext[:3, :3], ext[:3, 3] = R, -R @ e_
        cam.extrinsic = ext
        vc.convert_from_pinhole_camera_parameters(cam, allow_arbitrary=True)
        vis.poll_events()
        vis.update_renderer()
        img = (np.clip(np.asarray(vis.capture_screen_float_buffer(do_render=True)), 0, 1) * 255).astype(np.uint8)
        bgr = cv2.cvtColor(np.ascontiguousarray(img), cv2.COLOR_RGB2BGR)
        if i == 0:
            cv2.imwrite(first, bgr)
        vw.write(bgr)
        if i % (10 * fps) == 0:
            log.info(f"  {i / fps:.0f} s of {nfr / fps:.0f}")
    vw.release()
    vis.destroy_window()
    log.info(f"{out}: {nfr} frames; its first frame {first}")
    return P, nfr


def walk(mesh_path, traj=None, at=None, faces=0, eye=1.6, fov=70.0, shot=None, vivid=True, sat=1.15,
         one_side=False, step=0.1, turn=4.0, near=15.0, cell=4.0, wb=True, smooth=1):
    """Walk inside a mesh (with its ceiling) in a window, started eye m over
    the floor at walk_start's place, the mesh unlit in its colours (vivid:
    enhance_colours, saturation sat) and seen from both sides (one_side: as
    stored, half the triangles); simplified to `faces` first when larger
    (quadric; 0: all of it). The mesh is cut into blocks of `cell` m in
    plan and only the blocks within `near` m of the eye are drawn (and, of
    those, the renderer draws only the ones in view), so all of a 40 M face
    mesh walks smoothly. Each key event moves step m or turns `turn`
    degrees, whatever the frame rate (Open3D's own fly mode moves per
    frame: stuck when a big mesh draws slowly); the mouse drag looks round.
    shot: write the start view to this PNG and close."""
    o3d = _o3d()
    from open3d.visualization import gui, rendering
    m = o3d.io.read_triangle_mesh(os.path.expanduser(mesh_path))
    if not len(m.triangles):
        raise ConvertError(f"{mesh_path}: no faces (give it a mesh, not a view)")
    n0 = len(m.triangles)
    if faces and n0 > faces:
        log.info(f"simplifying {n0} faces to {faces} ...")
        m.remove_duplicated_vertices()
        m = m.simplify_quadric_decimation(target_number_of_triangles=int(faces))
    if vivid and m.has_vertex_colors():
        Cd, nsp = despeckle_colours(np.asarray(m.triangles), np.asarray(m.vertex_colors))
        if smooth:
            Cd = smooth_colours(np.asarray(m.triangles), Cd, smooth)
        log.info(f"colours: {nsp} speckles smoothed over, blended {smooth} x, then made vivid"
                 + (", white balanced" if wb else ""))
        m.vertex_colors = o3d.utility.Vector3dVector(enhance_colours(Cd, sat=sat, wb=wb))
    V = np.asarray(m.vertices)
    e, c = walk_start(V, read_tum_positions(traj, stride=1) if traj else None, at, eye)
    blocks, centres, unlit = mesh_blocks(m, cell, one_side)
    log.info(f"walk {mesh_path}: {len(m.triangles)} faces in {len(blocks)} blocks of {cell:g} m, "
             f"those within {near:g} m drawn; from ({e[0]:.1f}, {e[1]:.1f}), eye {eye} m over the floor\n"
             f"  turn: the arrow keys (left/right turn, up/down tilt, {turn:g} deg each), or drag with the mouse\n"
             f"  move: W forward, S back, A left, D right, Q up, Z down ({step:g} m each; hold a key to keep "
             "going), R back to the start\n  Escape or close the window: out")
    app = gui.Application.instance
    app.initialize()
    win = app.create_window(f"walk {os.path.basename(mesh_path)}", 1600, 900)
    sw = gui.SceneWidget()
    sw.scene = rendering.Open3DScene(win.renderer)
    mat = rendering.MaterialRecord()
    mat.shader = "defaultUnlit" if unlit else "defaultLit"
    for k, q in enumerate(blocks):
        sw.scene.add_geometry(f"b{k}", q, mat)
    shown = np.ones(len(blocks), bool)
    sw.scene.show_skybox(False)
    sw.scene.view.set_post_processing(False)     # no tone mapping: the colours as stored, not washed out
    sw.scene.set_background(np.array([0.12, 0.12, 0.12, 1.0], np.float32))   # small holes dark, not white
    sw.setup_camera(fov, sw.scene.bounding_box, c.astype(np.float32))
    cam = sw.scene.camera
    up = np.array([0.0, 0.0, 1.0], np.float32)

    def place(eye_, look_):
        cam.look_at(look_.astype(np.float32), eye_.astype(np.float32), up)
        cam.set_projection(fov, win.size.width / max(win.size.height, 1), 0.05, near + 2 * cell,
                           rendering.Camera.FovType.Vertical)
        want = np.linalg.norm(centres - eye_[:2], axis=1) <= near + cell * 0.71
        for k in np.flatnonzero(want != shown):
            sw.scene.show_geometry(f"b{k}", bool(want[k]))
        shown[:] = want
        sw.force_redraw()
    place(e, c)
    sw.set_view_controls(gui.SceneWidget.Controls.FLY)

    def on_key(ev):
        if ev.type != gui.KeyEvent.Type.DOWN:
            return gui.Widget.EventCallbackResult.IGNORED
        M_ = np.asarray(cam.get_model_matrix())
        eye_, fwd, right = M_[:3, 3].copy(), -M_[:3, 2], M_[:3, 0]
        flat = np.r_[fwd[:2], 0.0]
        flat /= np.linalg.norm(flat) or 1.0
        K = gui.KeyName
        if ev.key == K.ESCAPE:
            win.close()
            return gui.Widget.EventCallbackResult.CONSUMED
        if ev.key == K.R:
            place(e, c)
            return gui.Widget.EventCallbackResult.CONSUMED
        move = {K.W: flat, K.S: -flat, K.A: -right, K.D: right, K.Q: up, K.Z: -up}
        if ev.key in move:
            eye_ = eye_ + step * np.asarray(move[ev.key], float)
            place(eye_, eye_ + fwd)
            return gui.Widget.EventCallbackResult.CONSUMED
        if ev.key in (K.LEFT, K.RIGHT, K.UP, K.DOWN):
            yaw = np.arctan2(fwd[1], fwd[0]) + np.radians(turn) * (1 if ev.key == K.LEFT else -1 if ev.key == K.RIGHT else 0)
            pitch = np.arcsin(np.clip(fwd[2], -1, 1)) + np.radians(turn) * (1 if ev.key == K.UP else -1 if ev.key == K.DOWN else 0)
            pitch = np.clip(pitch, np.radians(-85), np.radians(85))
            fwd = np.array([np.cos(pitch) * np.cos(yaw), np.cos(pitch) * np.sin(yaw), np.sin(pitch)])
            place(eye_, eye_ + fwd)
            return gui.Widget.EventCallbackResult.CONSUMED
        return gui.Widget.EventCallbackResult.IGNORED
    sw.set_on_key(on_key)
    win.add_child(sw)
    if shot:
        import threading
        import time

        def snap():
            time.sleep(3)

            def save():
                img = app.render_to_image(sw.scene, win.size.width, win.size.height)
                o3d.io.write_image(os.path.expanduser(shot), img)
            app.post_to_main_thread(win, save)
            time.sleep(2)
            app.post_to_main_thread(win, app.quit)
        threading.Thread(target=snap, daemon=True).start()
    app.run()


def sample_mesh(mesh, spacing=0.04):
    """Points evenly over the mesh, about `spacing` apart."""
    area = mesh.get_surface_area()
    n = max(1000, int(np.ceil(2 * area / spacing ** 2)))
    pc = mesh.sample_points_uniformly(n).voxel_down_sample(spacing)
    return np.asarray(pc.points), area


def colour_points(P, frames, Kc, Wc, Hc, tol=0.05, rel=0.02):
    """Mean colour of each point over the frames that see it: (rgb uint8, seen).
    frames: (camera-to-world, BGR image, depth metres), all in the centred camera."""
    fx, fy, cx, cy = Kc
    acc = np.zeros((len(P), 3))
    cnt = np.zeros(len(P))
    for T, bgr, D in frames:
        R, t = T[:3, :3], T[:3, 3]
        Pc = (P - t) @ R
        z = Pc[:, 2]
        i = np.flatnonzero(z > 0.05)
        u = np.rint(fx * Pc[i, 0] / z[i] + cx).astype(np.int64)
        v = np.rint(fy * Pc[i, 1] / z[i] + cy).astype(np.int64)
        m = (u >= 0) & (u < Wc) & (v >= 0) & (v < Hc)
        i, u, v = i[m], u[m], v[m]
        d = D[v, u]
        m = (d > 0) & (np.abs(z[i] - d) < np.maximum(tol, rel * d))
        i, u, v = i[m], u[m], v[m]
        acc[i] += bgr[v, u, ::-1]
        cnt[i] += 1
    seen = cnt > 0
    rgb = np.full((len(P), 3), 128, np.uint8)
    rgb[seen] = np.clip(np.rint(acc[seen] / cnt[seen, None]), 0, 255).astype(np.uint8)
    return rgb, seen


def lod_levels(P, rgb, tau=0.04, min_points=10000, most=16):
    """The paper's multi-resolution start (Sec. 4.2): the finest level every
    point (the mesh sampled every tau), each coarser one a point per cell twice
    the last one's size (the point nearest the cell's centre, roughly Poisson
    disk), until a level has fewer than min_points (the paper: 4 cm, 10,000).
    [(points, rgb)], coarsest first: level_0 as LOD-3DGS numbers them, each one
    every point at its resolution, as its octree loader gives them."""
    levels = [(P, rgb)]
    s = float(tau)
    while len(levels[-1][0]) >= min_points and len(levels) < most:
        s *= 2
        Q, c = levels[-1]
        k = np.floor(Q / s).astype(np.int64) + (1 << 20)
        key = (k[:, 0] << 42) | (k[:, 1] << 21) | k[:, 2]
        d = np.linalg.norm(Q - (np.floor(Q / s) + 0.5) * s, axis=1)
        o = np.lexsort((d, key))
        first = o[np.r_[True, key[o][1:] != key[o][:-1]]]
        levels.append((Q[first], c[first]))
    return levels[::-1]


def write_ply(path, P, rgb):
    """x y z nx ny nz red green blue, binary, as LOD-3DGS's storePly writes it."""
    dt = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("nx", "<f4"), ("ny", "<f4"),
                   ("nz", "<f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    a = np.zeros(len(P), dt)
    a["x"], a["y"], a["z"] = P.T
    a["red"], a["green"], a["blue"] = rgb.T
    with open(path, "wb") as f:
        f.write(("ply\nformat binary_little_endian 1.0\n"
                 f"element vertex {len(P)}\n"
                 + "".join(f"property {'float' if dt[k].kind == 'f' else 'uchar'} {k}\n"
                           for k in dt.names)
                 + "end_header\n").encode("ascii"))
        a.tofile(f)


# --------------------------------------------------------------------------- #
# the scene
# --------------------------------------------------------------------------- #

def default_out(dataset):
    d = os.path.abspath(dataset.rstrip("/"))
    name = os.path.basename(d)
    name = "lod3dgs_" + (name[len("replica_"):] if name.startswith("replica_") else name)
    return os.path.join(os.path.dirname(d), name)


def convert(dataset, map_path, out=None, *, every=1, voxel=0.02, poisson_depth=0,
            trim=0.10, spacing=0.04, color_views=300, jpeg_quality=95, mesh_path=None,
            lod_min=10000):
    out = out or default_out(dataset)
    cam, T_all, rgb_all, dep_all = read_replica(dataset)
    keep = np.arange(0, len(T_all), max(1, int(every)))
    Wc, Hc, ox, oy = centred_camera(cam)
    fx, fy = cam["fx"], cam["fy"]
    Kc = (fx, fy, (Wc - 1) / 2.0, (Hc - 1) / 2.0)
    scale = cam["png_depth_scale"]
    log.info(f"{dataset}: {len(T_all)} frames, {len(keep)} kept (every {every}); "
             f"{int(cam['image_width'])}x{int(cam['image_height'])} -> {Wc}x{Hc} centred "
             f"on ({cam['cx']:.2f}, {cam['cy']:.2f})")
    for sub in ("images", "depths", "sparse/0"):
        os.makedirs(os.path.join(out, sub), exist_ok=True)

    cview = set(keep[np.unique(np.linspace(0, len(keep) - 1,
                                           min(len(keep), max(1, color_views))).astype(int))])
    q = [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)]
    lines, cframes, valid = [], [], []
    for n, i in enumerate(keep):
        name = f"frame{n:06d}"
        bgr = cv2.imread(rgb_all[i], cv2.IMREAD_COLOR)
        d16 = cv2.imread(dep_all[i], cv2.IMREAD_UNCHANGED)
        if bgr is None or d16 is None:
            raise ConvertError(f"cannot read {rgb_all[i]} / {dep_all[i]}")
        if d16.shape[:2] != bgr.shape[:2]:
            d16 = cv2.resize(d16, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
        bgr = recentre(bgr, Wc, Hc, ox, oy, cv2.INTER_LINEAR)
        D = recentre(d16, Wc, Hc, ox, oy, cv2.INTER_NEAREST).astype(np.float32) / scale
        cv2.imwrite(os.path.join(out, "images", name + ".jpg"), bgr, q)
        cv2.imwrite(os.path.join(out, "depths", name + ".png"), depth_la(D))
        valid.append(float((D > 0).mean()))
        T_cw = np.linalg.inv(T_all[i])
        qx, qy, qz, qw = Rotation.from_matrix(T_cw[:3, :3]).as_quat()
        tx, ty, tz = T_cw[:3, 3]
        lines.append(f"{n + 1} {qw:.12f} {qx:.12f} {qy:.12f} {qz:.12f} "
                     f"{tx:.9f} {ty:.9f} {tz:.9f} 1 {name}.jpg\n\n")
        if i in cview:
            cframes.append((T_all[i], bgr, D))
        if (n + 1) % 500 == 0:
            log.info(f"  {n + 1} frames")

    with open(os.path.join(out, "sparse/0/cameras.txt"), "w") as f:
        f.write("# CAMERA_ID MODEL WIDTH HEIGHT fx fy cx cy (LOD-3DGS takes cx, cy as "
                "the image centre)\n"
                f"1 PINHOLE {Wc} {Hc} {fx:.6f} {fy:.6f} {Kc[2]:.6f} {Kc[3]:.6f}\n")
    with open(os.path.join(out, "sparse/0/images.txt"), "w") as f:
        f.write("# IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME (world to camera), "
                "then an empty line\n" + "".join(lines))

    if mesh_path:
        log.info(f"mesh {mesh_path} (given)")
        mesh = _o3d().io.read_triangle_mesh(os.path.expanduser(mesh_path))
        if not len(mesh.triangles):
            raise ConvertError(f"{mesh_path}: no faces")
        mstat = {"faces": len(mesh.triangles), "faces_kept": len(mesh.triangles)}
    else:
        log.info("mesh")
        centres = T_all[keep, :3, 3]
        mesh, mstat = build_mesh(map_path, centres, voxel, poisson_depth, trim)
        _o3d().io.write_triangle_mesh(os.path.join(out, "mesh.ply"), mesh)
    P, area = sample_mesh(mesh, spacing)
    log.info(f"  {len(P)} points at {spacing} m over {area:.0f} m2; colour from "
             f"{len(cframes)} frames")
    rgb, seen = colour_points(P, cframes, Kc, Wc, Hc)
    write_ply(os.path.join(out, "sparse/0/points3D.ply"), P, rgb)
    levels = lod_levels(P, rgb, spacing, lod_min)
    os.makedirs(os.path.join(out, "lod"), exist_ok=True)
    for f in glob.glob(os.path.join(out, "lod", "level_*.ply")):
        os.remove(f)
    for k, (Q, c) in enumerate(levels):
        write_ply(os.path.join(out, "lod", f"level_{k}.ply"), Q, c)
    sizes = [len(Q) for Q, _ in levels]
    log.info(f"  {len(levels)} LOD levels, coarsest first: {sizes}")

    report = (
        f"dataset             : {os.path.abspath(dataset)}\n"
        f"map                 : {os.path.abspath(map_path)}\n"
        f"frames              : {len(keep)} of {len(T_all)} (every {every})\n"
        f"camera              : PINHOLE {Wc}x{Hc} fx={fx:.3f} fy={fy:.3f}, centred "
        f"(input {int(cam['image_width'])}x{int(cam['image_height'])}, "
        f"cx={cam['cx']:.3f} cy={cam['cy']:.3f})\n"
        f"depth valid px      : mean {100 * np.mean(valid):.1f}%, "
        f"min {100 * np.min(valid):.1f}%\n"
        + (f"mesh                : {os.path.abspath(mesh_path)}, {mstat['faces']} faces\n"
           if mesh_path else
           f"map points          : {mstat['map_points']} -> {mstat['map_voxel_points']} "
           f"at {voxel} m\n"
           f"Poisson depth       : {mstat['poisson_depth']}\n"
           f"mesh faces          : {mstat['faces']}, {mstat['faces_kept']} kept "
           f"(trim {trim} m)\n") +
        f"mesh area           : {area:.1f} m2\n"
        f"starting points     : {len(P)} at {spacing} m, {100 * seen.mean():.1f}% "
        f"coloured from {len(cframes)} frames\n"
        f"LOD levels          : {len(levels)} in lod/, coarsest first {sizes} "
        f"(cells {spacing} m doubling, until under {lod_min} points)\n"
        f"train               : python train.py -s {os.path.abspath(out)} --use_lod "
        f"--depths depths --data_device cpu -r 1\n")
    with open(os.path.join(out, "report.txt"), "w") as f:
        f.write(report)
    log.info("\n" + report)
    return {"out": out, "frames": len(keep), "camera": (Wc, Hc) + Kc, "points": len(P),
            "coloured": float(seen.mean()), "levels": sizes, **mstat}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def meshing(p):
        p.add_argument("--voxel", type=float, default=0.02, help="map voxel before meshing (m)")
        p.add_argument("--poisson-depth", type=int, default=0,
                       help="Poisson octree depth (0: cells about two voxels across)")
        p.add_argument("--trim", type=float, default=0.10,
                       help="remove mesh faces farther than this from a map point (m)")

    sc = sub.add_parser("scene", help="a replica export + the map -> a LOD-3DGS scene")
    sc.add_argument("dataset", help="a replica export (mcap_convert.py --format replica)")
    sc.add_argument("out", nargs="?", default=None, help="scene folder (default: next to it)")
    sc.add_argument("--map", required=True, help="the map the export's depth came from")
    sc.add_argument("--every", type=int, default=1, help="keep every Nth frame")
    meshing(sc)
    sc.add_argument("--spacing", type=float, default=0.04, help="starting point spacing (m)")
    sc.add_argument("--color-views", type=int, default=300,
                    help="frames the point colours are taken from (evenly spread)")
    sc.add_argument("--jpeg-quality", type=int, default=95)
    sc.add_argument("--mesh", default=None,
                    help="sample this mesh (e.g. `mesh`'s mesh_<tag>.ply) instead of meshing the map")
    sc.add_argument("--lod-min", type=int, default=10000,
                    help="LOD levels: halve until a level has fewer points (the paper's 10,000)")

    me = sub.add_parser("mesh", help="the mesh of a map alone, to look at")
    me.add_argument("map", help="the map (.pcd/.ply), e.g. mapping/map_final_<tag>.pcd")
    me.add_argument("out", nargs="?", default=None, help="mesh .ply (default: next to the map)")
    me.add_argument("--traj", required=True,
                    help="TUM trajectory the map was built from (normals turn to it)")
    meshing(me)
    me.add_argument("--ceil", default=None,
                    help="also a copy without the ceiling: faces above Z, or 'auto'")
    me.add_argument("--ceil-margin", type=float, default=0.10,
                    help="[auto] cut this far below the ceiling peak (m)")
    me.add_argument("--show", action="store_true", help="open it in an Open3D window")
    me.add_argument("--subdivide", type=int, default=1,
                    help="split every face into four this many times before colouring (finer colour)")
    me.add_argument("--min-piece", type=float, default=50.0,
                    help="pieces not joined to one of this size or more (the building, m2) are removed "
                         "when open to the sky or floating (0 keeps them)")
    me.add_argument("--body", type=float, default=0.0,
                    help="also clear the space the robot drove through: this far (m) round its path, "
                         "from the floor to over the LiDAR, e.g. 0.25 (0: not)")
    me.add_argument("--keep-ghosts", action="store_true",
                    help="keep the mirror ghosts (faces no direct ray from the path reaches)")
    me.add_argument("--normals", action="store_true",
                    help="write vertex normals (viewers then light the mesh, greying its colours)")
    vw = sub.add_parser("view", help="a viewing copy of the map: walls flat, floor holes filled")
    vw.add_argument("map", help="the map (.pcd/.ply), e.g. mapping/map_final_<tag>.pcd")
    vw.add_argument("out", nargs="?", default=None, help="view .pcd (default: next to the map)")
    vw.add_argument("--traj", required=True, help="TUM trajectory the map was built from (the floor)")
    vw.add_argument("--no-outliers", action="store_true", help="keep the stray points")
    vw.add_argument("--no-snap", action="store_true", help="leave walls and floors as mapped")
    vw.add_argument("--no-fill", action="store_true", help="leave the floor's holes")
    vw.add_argument("--min-piece", type=float, default=50.0,
                    help="pieces not joined to one of this size or more (the building, m2) are removed "
                         "when open to the sky or floating (0 keeps them)")
    vw.add_argument("--bag", default=None,
                    help="the bag the map was made from: clear what its scans saw through (reflections)")
    vw.add_argument("--topic", default="/mobile_2/ouster/points", help="--bag: the points topic")
    vw.add_argument("--scan-every", type=int, default=5, help="--bag: use every Nth scan")
    vw.add_argument("--sky", action="store_true",
                    help="also clear what is over open sky (a courtyard), whatever it touches")
    vw.add_argument("--body", type=float, default=0.0,
                    help="also clear the space the robot drove through: this far (m) round its path, "
                         "from the floor to over the LiDAR, e.g. 0.25 (0: not)")
    vw.add_argument("--keep-ghosts", action="store_true",
                    help="keep the mirror ghosts (points no direct ray from the path reaches)")
    vw.add_argument("--max-hole", type=float, default=20.0, help="largest hole filled (m2)")
    vw.add_argument("--roof", type=float, default=0.5,
                    help="share of a hole that must have a ceiling over it (0: courtyards too)")
    vw.add_argument("--ceil", default=None, help="remove points above Z, or 'auto'")
    vw.add_argument("--ceil-margin", type=float, default=0.10)
    an = sub.add_parser("anchor", help="a cloud or mesh made in GLIM's world into the anchored frame (05)")
    an.add_argument("file", help="a cloud or mesh made from map_final with the BA trajectory")
    an.add_argument("out", nargs="?", default=None, help="default: <file>_anchored")
    an.add_argument("--frame", required=True, help="05's frames/anchor_frame.json")
    sn = sub.add_parser("sionna", help="a mesh as a Sionna RT scene: its faces split into floor, walls, "
                                       "ceiling, objects")
    sn.add_argument("file", help="the mesh with its ceiling, e.g. mesh_view_<tag>_anchored.ply (not _noceil)")
    sn.add_argument("out", nargs="?", default=None,
                    help="the scene folder (default: <file>_sionna; also <out>.ply and <out>_labels.ply)")
    sn.add_argument("--traj", default=None,
                    help="the trajectory in the mesh's frame: leave out what hangs from the ceiling only, as "
                         "_noceil does (a reflection); anchored mesh: odometry/reference_<tag>/traj_mobile_2_lidar.tum")
    sn.add_argument("--faces", type=int, default=0,
                    help="floor, walls and ceiling simplified together to about this many faces (0: as meshed)")
    sn.add_argument("--object-faces", type=int, default=0,
                    help="the objects simplified on their own to about this many faces, finer than the "
                         "rest (0: as meshed)")
    sn.add_argument("--split-objects", action="store_true",
                    help="each object its own meshes/objects/obj_NNN.ply (a material each, by editing scene.xml)")
    sn.add_argument("--why", action="append", default=[], metavar="X,Y",
                    help="log what each rule saw of the pieces left within 1 m of this plan point (the mesh's "
                         "frame; pick it in CloudCompare); repeat for more")
    sn.add_argument("--why-objects", type=int, default=0, metavar="N",
                    help="log the N largest objects and what each rule saw of them (where, size, height, "
                         "facing, distance to the walls)")
    sn.add_argument("--no-fill-walls", action="store_true",
                    help="leave the walls as meshed (default: the wall hidden behind furniture, which the mesh "
                         "has not, filled on its plane; doors and windows left open)")
    sn.add_argument("--keep-walls", action="store_true",
                    help="leave the walls as meshed (default: their faces within 6 cm of a wall plane put on it, "
                         "smooth)")
    sn.add_argument("--keep-floor", action="store_true",
                    help="leave the floor as meshed (default: its faces flattened onto the floor's level, the "
                         "bumps under furniture gone)")
    sn.add_argument("--no-colours", action="store_true",
                    help="write the parts and <out>.ply without colours (Sionna uses none: smaller files)")
    sn.add_argument("--normals", action="store_true",
                    help="write the parts with vertex normals: a viewer shades them as surfaces (colours greyed)")
    sn.add_argument("--flat-colours", type=float, nargs="?", const=1.0, default=0.0, metavar="W",
                    help="one colour per segment (each wall plane the median of its own, each pillar and object "
                         "its own, the ceiling one, glass tinted), blended in W (default 1: flat; 0.7 keeps a "
                         "trace of signs and posters)")
    sn.add_argument("--floor-grey", type=float, default=0.8,
                    help="the floor of <out>.ply in this grey (0 black to 1 white; negative: its own colours)")
    sn.add_argument("--material", action="append", default=[], metavar="PART=TYPE[:THICKNESS]",
                    help="floor, ceiling, walls or objects as an ITU material (concrete, brick, plasterboard, "
                         "wood, glass, ceiling_board, marble, floorboard, metal, ...), thickness m; "
                         "default concrete:0.2, objects wood:0.05; repeat for each part")
    fv = sub.add_parser("fly", help="a video flying once round the area inside a mesh")
    fv.add_argument("file", help="a mesh, e.g. mesh_view_<tag>_anchored.ply or its _sionna.ply")
    fv.add_argument("out", nargs="?", default=None, help="the video (.mp4; default <file>_fly.mp4)")
    fv.add_argument("--path", default=None, metavar="X,Y;X,Y;...",
                    help="waypoints (m) joined in a loop (default: once round the area, --inset from the walls)")
    fv.add_argument("--inset", type=float, default=1.5, help="the loop this far from the walls and furniture (m)")
    fv.add_argument("--start", default="0,0", metavar="X,Y",
                    help="start here (default the map's origin; 'loop': on the loop), turn round, then to the loop")
    fv.add_argument("--spin", type=float, default=8.0, help="the turn round at the start takes this long (s; 0: none)")
    fv.add_argument("--speed", type=float, default=1.0, help="m/s")
    fv.add_argument("--fps", type=int, default=30)
    fv.add_argument("--size", default="1280x720", help="WxH")
    fv.add_argument("--seconds", type=float, default=0, help="at most this long (0: the whole loop)")
    fv.add_argument("--ahead", type=float, default=2.0, help="look this far ahead along the way (m)")
    fv.add_argument("--eye", type=float, default=1.6, help="eye height over the floor (m)")
    fv.add_argument("--faces", type=int, default=0, help="simplify to this many faces first (0: all of it)")
    fv.add_argument("--plain", action="store_true", help="the colours as stored")
    fv.add_argument("--sat", type=float, default=1.15, help="the saturation boost when made vivid")
    fv.add_argument("--cast", action="store_true", help="keep the colours' cast (default: white balanced)")
    fv.add_argument("--smooth", type=int, default=1, help="blend each colour with its neighbours this many times")
    fv.add_argument("--hidden", action="store_true", help="the window not shown")
    wk = sub.add_parser("walk", help="walk inside a mesh (with its ceiling) in a window")
    wk.add_argument("file", help="a mesh, e.g. mesh_view_<tag>_anchored.ply or its _sionna.ply")
    wk.add_argument("--traj", default=None,
                    help="a trajectory in the mesh's frame: start at its first pose (anchored mesh: "
                         "odometry/reference_<tag>/traj_mobile_2_lidar.tum)")
    wk.add_argument("--at", default=None, metavar="X,Y", help="start here instead")
    wk.add_argument("--faces", type=int, default=0,
                    help="simplify to this many faces first (default 0: all of it; only what is near is drawn)")
    wk.add_argument("--near", type=float, default=15.0, help="draw the mesh within this many metres of the eye")
    wk.add_argument("--cell", type=float, default=4.0, help="the blocks the mesh is cut into (m, in plan)")
    wk.add_argument("--plain", action="store_true",
                    help="the colours as stored (default: made vivid, the camera's haze and grey gone)")
    wk.add_argument("--sat", type=float, default=1.15, help="the saturation boost when made vivid")
    wk.add_argument("--cast", action="store_true", help="keep the colours' cast (default: white balanced)")
    wk.add_argument("--smooth", type=int, default=1, help="blend each colour with its neighbours this many times")
    wk.add_argument("--step", type=float, default=0.1, help="metres per key event (W/S/A/D/Q/Z)")
    wk.add_argument("--turn", type=float, default=4.0, help="degrees per arrow key event")
    wk.add_argument("--one-side", action="store_true",
                    help="the faces one-sided as stored (default: doubled, so a wall is seen from both sides; "
                         "halves the load for a big mesh)")
    wk.add_argument("--eye", type=float, default=1.6, help="eye height over the floor (m)")
    wk.add_argument("--shot", default=None, help="write the start view to this PNG and close")
    a = ap.parse_args()
    for k in ("map", "file", "traj", "frame", "dataset"):
        p = getattr(a, k, None)
        if p and not os.path.exists(os.path.expanduser(p)):
            ap.error(f"no file {p}" + ("  (make it with `view` first)" if "view_" in os.path.basename(p) else ""))
    if getattr(a, "bag", None) and not os.path.exists(os.path.expanduser(a.bag)):
        ap.error(f"no bag {a.bag}  (find it: find <data> -name metadata.yaml -path '*<date>*')")
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        if a.cmd == "anchor":
            anchor_file(a.file, a.frame, a.out)
        elif a.cmd == "fly":
            wh = [int(v) for v in a.size.lower().split("x")]
            fly_video(a.file, a.out, [[float(v) for v in p_.split(",")] for p_ in a.path.split(";")] if a.path else None,
                      inset=a.inset, eye=a.eye, speed=a.speed, fps=a.fps, size=(wh[0], wh[1]), ahead=a.ahead,
                      faces=a.faces, vivid=not a.plain, seconds=a.seconds, hidden=a.hidden, spin=a.spin,
                      sat=a.sat, wb=not a.cast, smooth=a.smooth,
                      start=None if a.start.lower() == "loop" else [float(v) for v in a.start.split(",")])
        elif a.cmd == "walk":
            walk(a.file, a.traj, [float(v) for v in a.at.split(",")] if a.at else None,
                 faces=a.faces, eye=a.eye, shot=a.shot, vivid=not a.plain, sat=a.sat, one_side=a.one_side,
                 step=a.step, turn=a.turn, near=a.near, cell=a.cell, wb=not a.cast, smooth=a.smooth)
        elif a.cmd == "sionna":
            mats = {}
            for spec in a.material:
                part, _, v = spec.partition("=")
                typ, _, th = v.partition(":")
                if part not in SIONNA_MATERIALS or not typ:
                    ap.error(f"--material {spec}: PART=TYPE[:THICKNESS], PART one of "
                             + ", ".join(SIONNA_MATERIALS))
                mats[part] = (typ, float(th) if th else SIONNA_MATERIALS[part][1])
            sionna_mesh(a.file, a.out, faces=a.faces, object_faces=a.object_faces, materials=mats,
                        split_objects=a.split_objects,
                        traj=a.traj, normals=a.normals, flat_floor=not a.keep_floor, fill=not a.no_fill_walls,
                        colours=not a.no_colours, flat_colours=a.flat_colours,
                        flat_walls=not a.keep_walls,
                        why=[tuple(float(v) for v in w.split(",")[:2]) for w in a.why],
                        why_objects=a.why_objects,
                        floor_colour=None if a.floor_grey < 0 else (a.floor_grey,) * 3)
        elif a.cmd == "scene":
            convert(a.dataset, a.map, a.out, every=a.every, voxel=a.voxel,
                    poisson_depth=a.poisson_depth, trim=a.trim, spacing=a.spacing,
                    color_views=a.color_views, jpeg_quality=a.jpeg_quality,
                    mesh_path=a.mesh, lod_min=a.lod_min)
        elif a.cmd == "view":
            map_view(a.map, a.traj, a.out, outliers=not a.no_outliers, snap=not a.no_snap,
                     fill=not a.no_fill, ceil=a.ceil, ceil_margin=a.ceil_margin,
                     max_hole=a.max_hole, roof=a.roof, ghosts=not a.keep_ghosts,
                     min_piece=a.min_piece, body=a.body, sky=a.sky, bag=a.bag,
                     topic=a.topic, scan_every=a.scan_every)
        else:
            map_mesh(a.map, a.traj, a.out, voxel=a.voxel, poisson_depth=a.poisson_depth,
                     trim=a.trim, ceil=a.ceil, ceil_margin=a.ceil_margin, show=a.show,
                     subdivide=a.subdivide, normals=a.normals, ghosts=not a.keep_ghosts,
                     min_piece=a.min_piece, body=a.body)
    except ConvertError as e:
        sys.exit(str(e))


if __name__ == "__main__":
    main()
