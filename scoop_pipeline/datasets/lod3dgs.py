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

`mesh` is the mesh step alone, to look at: the map's Poisson mesh, trimmed,
in the map's colours (map_final_<tag>.pcd is colorized), normals turned to
the nearest pose of --traj (the trajectory the map was built from). Written
next to the map as mesh_<tag>.ply (map_final_<tag>.pcd; else <map>_mesh.ply),
with the height histogram of the map printed (z up). --ceil also writes a
viewing copy without the ceiling, <mesh>_noceil.ply: every face above Z
removed, or with "auto" above the highest histogram peak (the ceiling) less
--ceil-margin. Only a viewing copy, as 06_cut's: a scene keeps the ceiling,
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


CEIL_MIN = 2.1          # a ceiling is this far above the floor at least (m)


def ceiling_faces(mesh, T, h, flat=0.8):
    """Per face of the mesh: a ceiling, a horizontal face (|n_z| > flat)
    CEIL_MIN or more above the floor under it (the nearest pose's height less
    the sensor height h). Walls keep their full height whatever the ceiling's
    height is there, over a low ceiling and a high one alike."""
    F, V = np.asarray(mesh.triangles), np.asarray(mesh.vertices)
    if not len(F):
        return np.zeros(0, bool)
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    n = np.cross(b - a, c - a)
    nz = np.abs(n[:, 2]) / np.maximum(np.linalg.norm(n, axis=1), 1e-12)
    cen = (a + b + c) / 3
    _, near = cKDTree(T[:, :2]).query(cen[:, :2], workers=-1)
    up = cen[:, 2] - (T[near, 2] - h)
    return (nz > flat) & (up >= CEIL_MIN)


def ceiling_points(P, T, h, flat=0.8, r=0.1):
    """Per point: on a ceiling (see ceiling_faces), its normal from the
    points within r."""
    o3d = _o3d()
    _, near = cKDTree(T[:, :2]).query(P[:, :2], workers=-1)
    up = P[:, 2] - (T[near, 2] - h)
    hi = np.flatnonzero(up >= CEIL_MIN)
    out = np.zeros(len(P), bool)
    if len(hi):
        pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P[hi]))
        pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=r, max_nn=30))
        out[hi] = np.abs(np.asarray(pc.normals)[:, 2]) > flat
    return out


def remove_ceiling_mesh(mesh, T, h, ceil, margin):
    """(the mesh without its ceiling, how, faces removed): ceil "auto" the
    ceiling surfaces (ceiling_faces), a number every face above it."""
    if str(ceil) == "auto":
        gone = ceiling_faces(mesh, T, h)
        how = f"the ceilings: horizontal faces {CEIL_MIN} m or more above the floor"
    else:
        F, V = np.asarray(mesh.triangles), np.asarray(mesh.vertices)
        gone = V[F].mean(1)[:, 2] > float(ceil)
        how = f"every face above z = {float(ceil):+.2f}"
    out = type(mesh)(mesh)
    out.remove_triangles_by_mask(gone)
    out.remove_unreferenced_vertices()
    return out, how, int(gone.sum())


def cut_file(path, traj, ceil="auto", margin=0.10, out=None):
    """A copy of a mesh or a point cloud without what is above the ceiling
    (ceil_z): <name>_noceil, the mesh not made again."""
    o3d = _o3d()
    path = os.path.expanduser(path)
    stem, ext = os.path.splitext(path)
    out = out or stem + "_noceil" + ext
    T = read_tum_positions(traj, stride=1)
    m = o3d.io.read_triangle_mesh(path)
    if len(m.triangles):
        h = sensor_height(np.asarray(m.vertices), T)[0]
        c, how, _ = remove_ceiling_mesh(m, T, h, ceil, margin)
        o3d.io.write_triangle_mesh(out, c)
        n, kept = len(m.triangles), len(c.triangles)
    else:
        pc = o3d.io.read_point_cloud(path)
        if not len(pc.points):
            raise ConvertError(f"{path}: no faces or points")
        P = np.asarray(pc.points)
        h = sensor_height(P, T)[0]
        if str(ceil) == "auto":
            gone = ceiling_points(P, T, h)
            how = f"the ceilings: horizontal surfaces {CEIL_MIN} m or more above the floor"
        else:
            gone = P[:, 2] > float(ceil)
            how = f"every point above z = {float(ceil):+.2f}"
        c = pc.select_by_index(np.flatnonzero(~gone))
        o3d.io.write_point_cloud(out, c)
        n, kept = len(P), len(c.points)
    log.info(f"{out}: {how} removed, {kept} of {n} kept")
    return {"out": out, "kept": kept, "removed": n - kept}


def default_mesh_out(map_path):
    d, name = os.path.split(os.path.abspath(map_path))
    stem = os.path.splitext(name)[0]
    if stem.startswith("map_final_"):
        return os.path.join(d, "mesh_" + stem[len("map_final_"):] + ".ply")
    if stem.startswith("view_"):
        return os.path.join(d, "mesh_" + stem + ".ply")
    return os.path.join(d, stem + "_mesh.ply")


def mesh_objects(pc, centres, cell=0.012, trim=0.03, tile=4.0, min_pts=200, smooth=10):
    """The mesh of the points that are not walls or floors (objects), on its
    own and finer: Poisson tile by tile (tile m, with 0.2 m of the next tiles
    so nothing is cut at an edge), octree cells about `cell` across, the
    tenth of vertices least backed by points (Poisson's density) and every
    face farther than `trim` from a point removed (no bubbles), each tile
    keeping the faces whose centre is in it; then `smooth` passes of Taubin
    smoothing (evens a ragged surface out without shrinking it)."""
    o3d = _o3d()
    P = np.asarray(pc.points)
    out = o3d.geometry.TriangleMesh()
    if len(P) < min_pts:
        return out
    lo = P[:, :2].min(0)
    ij = np.floor((P[:, :2] - lo) / tile).astype(np.int64)
    ctree = cKDTree(centres)
    for key in np.unique(ij, axis=0):
        x0 = lo + key * tile
        sel = np.flatnonzero(np.all((P[:, :2] >= x0 - 0.2) & (P[:, :2] < x0 + tile + 0.2), 1))
        if len(sel) < min_pts:
            continue
        sub = pc.select_by_index(sel)
        sub.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.06, max_nn=30))
        Q, N = np.asarray(sub.points), np.asarray(sub.normals)
        _, near = ctree.query(Q)
        flip = np.einsum("ij,ij->i", N, centres[near] - Q) < 0
        N[flip] *= -1
        sub.normals = o3d.utility.Vector3dVector(N)
        ext = float(np.max(Q.max(0) - Q.min(0)))
        depth = int(np.clip(np.ceil(np.log2(max(ext, cell) / cell)), 6, 11))
        m, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(sub, depth=depth)
        dens = np.asarray(dens)
        if len(dens):
            m.remove_vertices_by_mask(dens < np.quantile(dens, 0.1))
        F, V = np.asarray(m.triangles), np.asarray(m.vertices)
        if not len(F):
            continue
        c = V[F].mean(1)
        d, _ = cKDTree(Q).query(c)
        inside = np.all((c[:, :2] >= x0) & (c[:, :2] < x0 + tile), 1)
        m.remove_triangles_by_mask((d > trim) | ~inside)
        m.remove_unreferenced_vertices()
        if smooth and len(m.triangles):
            m = m.filter_smooth_taubin(number_of_iterations=int(smooth))
        out += m
    return out


def crisp_colours(mesh, map_path, subdivide=1, k=5):
    """The mesh split `subdivide` times (each face into four) and every vertex
    given the median colour of the map's k nearest points, at full
    resolution: Poisson averages the thinned map's colours, which blurs and
    greys them; the median of a few keeps an edge and drops a point coloured
    wrong (a wall's colour put on a chair's edge), where the nearest alone
    (k=1) copies it."""
    o3d = _o3d()
    if subdivide > 0:
        mesh = mesh.subdivide_midpoint(number_of_iterations=int(subdivide))
    src = o3d.io.read_point_cloud(os.path.expanduser(map_path))
    if not src.has_colors():
        return mesh, False
    k = max(1, int(k))
    _, idx = cKDTree(np.asarray(src.points)).query(np.asarray(mesh.vertices), k=k, workers=-1)
    C = np.asarray(src.colors)
    col = C[idx] if k == 1 else np.median(C[idx], axis=1)
    mesh.vertex_colors = o3d.utility.Vector3dVector(col)
    return mesh, True


def map_mesh(map_path, traj, out=None, *, voxel=0.02, poisson_depth=0, trim=0.10,
         ceil=None, ceil_margin=0.10, show=False, subdivide=1, normals=False, ghosts=True,
         min_piece=50.0, objects=True, object_cell=0.012, object_trim=0.03, flat_sigma=0.015,
         colour_k=5, object_smooth=10, faces=0):
    """The mesh step alone: mesh of the map (+ a copy without the ceiling),
    each vertex in the colour of the map's nearest point; written without
    normals unless asked, so a viewer shows the colours unlit."""
    o3d = _o3d()
    out = out or default_mesh_out(map_path)
    centres = read_tum_positions(traj)
    Tf = read_tum_positions(traj, stride=1)
    pc = o3d.io.read_point_cloud(os.path.expanduser(map_path))
    if not len(pc.points):
        raise ConvertError(f"{map_path}: no points")
    nloose, hs = 0, None
    if min_piece > 0:
        # loose pieces out of the points first, so neither mesh gets them; a
        # piece is what touches (10 cm), so a thing on a wall stays with it
        P0 = np.asarray(pc.points)
        hs, _ = sensor_height(P0, Tf)
        gone = piece_points(P0, Tf, hs, min_piece)
        pc = pc.select_by_index(np.flatnonzero(~gone))
        nloose = int(gone.sum())
    src = pc
    nobj = 0
    if objects:
        # walls, floors, ceilings (0.4 m cells flat to flat_sigma and at least
        # half filled: the view copy's walls are; a curved object or a rack's
        # bars are not) meshed as the building;
        # the rest (objects) on their own, finer
        inv, flat, _, _ = flat_cells(np.asarray(pc.points), sig_max=flat_sigma, cover=0.5)
        fl = flat[inv]
        src = pc.select_by_index(np.flatnonzero(fl))
        obj = pc.select_by_index(np.flatnonzero(~fl))
        log.info(f"  building (flat cells) {int(fl.sum())} points, objects {int((~fl).sum())}")
    m, mstat = build_mesh(src, centres, voxel, poisson_depth, trim)
    npiece = 0
    if min_piece > 0:
        m, npiece = drop_pieces(m, Tf, hs, min_piece)          # Poisson's own loose bits
    if subdivide > 0:
        m = m.subdivide_midpoint(number_of_iterations=int(subdivide))
    if objects:
        mo = mesh_objects(obj, centres, object_cell, object_trim, smooth=object_smooth)
        nobj = len(mo.triangles)
        m += mo
    nghost = 0
    if ghosts:
        m, nghost = remove_ghosts(m, Tf)
    nfull = len(m.triangles)
    if faces and nfull > faces:
        # few large faces (a radio scene: flat walls in a handful of
        # triangles, objects coarse), quadric error so planes stay planes
        m = m.simplify_quadric_decimation(target_number_of_triangles=int(faces))
        m.remove_degenerate_triangles()
        m.remove_unreferenced_vertices()
    m, _ = crisp_colours(m, map_path, 0, colour_k)
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
              f"{trim} m trim{f', split {subdivide}x' if subdivide else ''})"
              f"{f' + {nobj} of objects ({object_cell * 100:g} cm, {object_trim * 100:g} cm trim)' if objects else ''}"
              f", {len(V)} vertices, "
              f"{f'each the median colour of its {colour_k} nearest map points' if m.has_vertex_colors() else 'no colour (map has none)'}"
              f"{'' if normals else ', no normals (shown unlit)'}"
              + (f"\n  simplified from {nfull} to {len(m.triangles)} faces" if faces and nfull > faces else "")
              + (f"\n  mirror ghosts (faces no direct ray from the path reaches) removed: {nghost} faces"
                 if ghosts else "")
              + (f"\n  loose pieces (not the building) open to the sky or floating removed: "
                 f"{nloose} points, then {npiece} faces" if min_piece > 0 else ""))
    res.update(ghost_faces=nghost, piece_faces=npiece, piece_points=nloose, object_faces=nobj)
    shown = m
    if ceil is not None:
        if hs is None:
            hs, _ = sensor_height(V, Tf)
        nc, how, ngone = remove_ceiling_mesh(m, Tf, hs, ceil, ceil_margin)
        if min_piece > 0 and not objects:
            nc, _ = drop_pieces(nc, Tf, hs, min_piece)      # what hung from the ceiling now floats
        p = os.path.splitext(out)[0] + "_noceil.ply"
        o3d.io.write_triangle_mesh(p, nc)
        report += f"\nno ceiling {p}: {how}, {ngone} faces, removed; {len(nc.triangles)} kept"
        res.update(noceil=p, noceil_faces=len(nc.triangles), ceiling_faces=ngone)
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


def flat_cells(P, cell=0.4, sig_max=0.04, thin=0.35, min_pts=30, cover=0.0, sub=0.04):
    """(each point's cell, per cell: flat, centre, normal). A cell of edge
    `cell` is flat when its points' smallest spread is under sig_max and under
    thin x the next: a wall or a floor, not a corner, an edge or an object;
    with cover > 0 also when its points fill at least that share of the
    cell's plane in `sub` squares (two bars side by side span a plane but do
    not fill it)."""
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
    if cover > 0:
        rel = X - mu[inv]
        n_side = int(np.ceil(cell * np.sqrt(3) / sub))
        a = np.floor(np.einsum("ij,ij->i", rel, v[inv, :, 1]) / sub).astype(np.int64) + n_side
        b = np.floor(np.einsum("ij,ij->i", rel, v[inv, :, 2]) / sub).astype(np.int64) + n_side
        side = 2 * n_side + 1
        filled = np.bincount(np.unique(inv * side * side + a * side + b) // (side * side), minlength=m_)
        flat &= filled >= cover * (cell / sub) ** 2
    return inv, flat, mu + lo, v[:, :, 0]


def snap_planes(P, **kw):
    """(P with the points of every flat cell (flat_cells) moved onto the
    cell's plane, the number moved)."""
    inv, flat, mu, n = flat_cells(P, **kw)
    on = flat[inv]
    nn = n[inv[on]]
    d = np.einsum("ij,ij->i", P[on] - mu[inv[on]], nn)
    Q = P.copy()
    Q[on] -= d[:, None] * nn
    return Q, int(on.sum())


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
             ceil=None, ceil_margin=0.10, max_hole=20.0, roof=0.5, ghosts=True, min_piece=50.0):
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
        keep = ~(ceiling_points(P, T1, h) if str(ceil) == "auto" else P[:, 2] > float(ceil))
        report.append(f"  the ceiling removed ({'horizontal surfaces 2.1 m or more above the floor' if str(ceil) == 'auto' else f'above z = {float(ceil):+.2f}'}): "
                      f"{int((~keep).sum())} points")
        P, C = P[keep], C[keep]
        if min_piece > 0:
            g = piece_points(P, T1, h, min_piece)       # what hung from the ceiling now floats
            P, C = P[~g], C[~g]
            report.append(f"  then loose (hung from the ceiling): {int(g.sum())} points")
    v = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P))
    v.colors = o3d.utility.Vector3dVector(np.clip(C, 0, 1))
    o3d.io.write_point_cloud(out, v)
    report.append(f"view {out}: {len(P)} points (a viewing copy: the reference map is unchanged)")
    log.info("\n".join(report))
    res["points"] = len(P)
    return res


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
            trim=0.10, spacing=0.04, color_views=300, jpeg_quality=95):
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

    log.info("mesh")
    centres = T_all[keep, :3, 3]
    mesh, mstat = build_mesh(map_path, centres, voxel, poisson_depth, trim)
    _o3d().io.write_triangle_mesh(os.path.join(out, "mesh.ply"), mesh)
    P, area = sample_mesh(mesh, spacing)
    log.info(f"  {len(P)} points at {spacing} m over {area:.0f} m2; colour from "
             f"{len(cframes)} frames")
    rgb, seen = colour_points(P, cframes, Kc, Wc, Hc)
    write_ply(os.path.join(out, "sparse/0/points3D.ply"), P, rgb)

    report = (
        f"dataset             : {os.path.abspath(dataset)}\n"
        f"map                 : {os.path.abspath(map_path)}\n"
        f"frames              : {len(keep)} of {len(T_all)} (every {every})\n"
        f"camera              : PINHOLE {Wc}x{Hc} fx={fx:.3f} fy={fy:.3f}, centred "
        f"(input {int(cam['image_width'])}x{int(cam['image_height'])}, "
        f"cx={cam['cx']:.3f} cy={cam['cy']:.3f})\n"
        f"depth valid px      : mean {100 * np.mean(valid):.1f}%, "
        f"min {100 * np.min(valid):.1f}%\n"
        f"map points          : {mstat['map_points']} -> {mstat['map_voxel_points']} "
        f"at {voxel} m\n"
        f"Poisson depth       : {mstat['poisson_depth']}\n"
        f"mesh faces          : {mstat['faces']}, {mstat['faces_kept']} kept "
        f"(trim {trim} m)\n"
        f"mesh area           : {area:.1f} m2\n"
        f"starting points     : {len(P)} at {spacing} m, {100 * seen.mean():.1f}% "
        f"coloured from {len(cframes)} frames\n"
        f"train               : python train.py -s {os.path.abspath(out)} --use_lod "
        f"--depths depths --data_device cpu -r 1\n")
    with open(os.path.join(out, "report.txt"), "w") as f:
        f.write(report)
    log.info("\n" + report)
    return {"out": out, "frames": len(keep), "camera": (Wc, Hc) + Kc, "points": len(P),
            "coloured": float(seen.mean()), **mstat}


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

    me = sub.add_parser("mesh", help="the mesh of a map alone, to look at")
    me.add_argument("map", help="the map (.pcd/.ply), e.g. mapping/map_final_<tag>.pcd")
    me.add_argument("out", nargs="?", default=None, help="mesh .ply (default: next to the map)")
    me.add_argument("--traj", required=True,
                    help="TUM trajectory the map was built from (normals turn to it)")
    meshing(me)
    me.add_argument("--ceil", default=None,
                    help="also a copy without the ceiling: 'auto' (the ceiling surfaces: horizontal "
                         "faces 2.1 m or more above the floor; walls keep their height), or every face "
                         "above this height (m)")
    me.add_argument("--ceil-margin", type=float, default=0.10,
                    help="[auto] cut this far below the ceiling peak (m)")
    me.add_argument("--show", action="store_true", help="open it in an Open3D window")
    me.add_argument("--subdivide", type=int, default=1,
                    help="split every face into four this many times before colouring (finer colour)")
    me.add_argument("--min-piece", type=float, default=50.0,
                    help="pieces not joined to one of this size or more (the building, m2) are removed "
                         "when open to the sky or floating (0 keeps them)")
    me.add_argument("--keep-ghosts", action="store_true",
                    help="keep the mirror ghosts (faces no direct ray from the path reaches)")
    me.add_argument("--no-objects", action="store_true",
                    help="one Poisson for everything (objects not meshed on their own)")
    me.add_argument("--object-cell", type=float, default=0.012, help="objects' octree cell (m)")
    me.add_argument("--object-trim", type=float, default=0.03,
                    help="objects' faces farther than this from a point removed (m)")
    me.add_argument("--object-smooth", type=int, default=10,
                    help="passes of Taubin smoothing of the objects' mesh (0: none)")
    me.add_argument("--radio", action="store_true",
                    help="a coarse mesh for radio ray tracing (Sionna RT): 5 cm, gaps up to 0.5 m "
                         "closed, objects coarse, simplified to --faces, ceiling kept; <mesh>_radio.ply")
    me.add_argument("--faces", type=int, default=0,
                    help="simplify to about this many faces (quadric; 0: not; --radio: 300000)")
    me.add_argument("--colour-k", type=int, default=5,
                    help="each vertex the median colour of this many nearest map points (1: the nearest)")
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
    vw.add_argument("--keep-ghosts", action="store_true",
                    help="keep the mirror ghosts (points no direct ray from the path reaches)")
    vw.add_argument("--max-hole", type=float, default=20.0, help="largest hole filled (m2)")
    vw.add_argument("--roof", type=float, default=0.5,
                    help="share of a hole that must have a ceiling over it (0: courtyards too)")
    vw.add_argument("--ceil", default=None,
                    help="remove the ceiling: 'auto' (horizontal surfaces 2.1 m or more above the floor), "
                         "or every point above this height (m)")
    vw.add_argument("--ceil-margin", type=float, default=0.10)
    ct = sub.add_parser("cut", help="a copy of a mesh or point cloud without the ceiling")
    ct.add_argument("file", help="a mesh or a point cloud (.ply/.pcd)")
    ct.add_argument("out", nargs="?", default=None, help="default: <file>_noceil")
    ct.add_argument("--traj", required=True, help="TUM trajectory in the same frame (the floor driven on)")
    ct.add_argument("--ceil", default="auto",
                    help="'auto': the ceiling surfaces (horizontal, 2.1 m or more above the floor); "
                         "or a height (m): everything above it")
    ct.add_argument("--ceil-margin", type=float, default=0.10)
    a = ap.parse_args()
    if getattr(a, "ceil", None) not in (None, "auto"):
        try:
            float(a.ceil)
        except ValueError:
            ap.error(f"--ceil {a.ceil}: a height in m (e.g. 2.6), or auto")
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        if a.cmd == "scene":
            convert(a.dataset, a.map, a.out, every=a.every, voxel=a.voxel,
                    poisson_depth=a.poisson_depth, trim=a.trim, spacing=a.spacing,
                    color_views=a.color_views, jpeg_quality=a.jpeg_quality)
        elif a.cmd == "cut":
            cut_file(a.file, a.traj, a.ceil, a.ceil_margin, a.out)
        elif a.cmd == "view":
            map_view(a.map, a.traj, a.out, outliers=not a.no_outliers, snap=not a.no_snap,
                     fill=not a.no_fill, ceil=a.ceil, ceil_margin=a.ceil_margin,
                     max_hole=a.max_hole, roof=a.roof, ghosts=not a.keep_ghosts,
                     min_piece=a.min_piece)
        else:
            if a.radio:
                a.voxel, a.trim, a.subdivide = 0.05, 0.25, 0
                a.object_cell, a.object_trim = 0.05, 0.06
                a.faces = a.faces or 300000
                a.out = a.out or os.path.splitext(default_mesh_out(a.map))[0] + "_radio.ply"
            map_mesh(a.map, a.traj, a.out, voxel=a.voxel, poisson_depth=a.poisson_depth,
                     trim=a.trim, ceil=a.ceil, ceil_margin=a.ceil_margin, show=a.show,
                     subdivide=a.subdivide, normals=a.normals, ghosts=not a.keep_ghosts,
                     min_piece=a.min_piece, objects=not a.no_objects, object_cell=a.object_cell,
                     object_trim=a.object_trim, colour_k=a.colour_k, object_smooth=a.object_smooth,
                     faces=a.faces)
    except ConvertError as e:
        sys.exit(str(e))


if __name__ == "__main__":
    main()
