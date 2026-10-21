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
    python scoop_pipeline/datasets/lod3dgs.py sionna <mesh .ply, with its ceiling> [<out .ply>]
        [--faces 500000] [--floor-grey 0.8] [--scene] [--material walls=brick:0.1]

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


SIONNA_MATERIALS = {"floor": ("concrete", 0.2), "ceiling": ("concrete", 0.2), "walls": ("concrete", 0.2)}


def sionna_mesh(mesh_path, out=None, faces=500000, scene=False, materials=None, flat=0.9, cell=1.0,
                band=0.2, roof_h=2.0, floor_colour=(0.8, 0.8, 0.8)):
    """A mesh for Sionna RT of a mesh made by `mesh` (the one with its
    ceiling: the radio waves bounce off it): <mesh>_sionna.ply, one file, the
    mesh simplified to about `faces` (quadric error: a flat wall stays flat in
    few faces), in its vertex colours but the floor floor_colour (None: its
    own), no normals. The floor: of the horizontal faces (|n_z| over flat)
    in each cell (m, in plan), those within band of the full mesh's lowest
    there. scene: also a Sionna scene
    of it, <mesh>_sionna/scene.xml with meshes/{floor,ceiling,walls}.ply, each
    an ITU radio material: the floor; the ceiling, horizontal faces roof_h or
    more above it; walls, the rest (a table top, furniture). materials: {part: (ITU type,
    thickness m)} over SIONNA_MATERIALS. Metres, the mesh's frame."""
    o3d = _o3d()
    mesh_path = os.path.expanduser(mesh_path)
    out = out or os.path.splitext(mesh_path)[0] + "_sionna.ply"
    mats = {**SIONNA_MATERIALS, **(materials or {})}
    m = o3d.io.read_triangle_mesh(mesh_path)
    if not len(m.triangles):
        raise ConvertError(f"{mesh_path}: no faces (give it the mesh, not the view)")
    n0 = len(m.triangles)
    m.remove_duplicated_vertices()
    # the floor's height per cell from the full mesh (simplified, a floor is a
    # few large faces, most cells holding none's centre): its lowest 1 %
    from scipy import ndimage
    V0 = np.asarray(m.vertices)
    lo = V0[:, :2].min(0)
    nx, ny = np.floor((V0[:, :2].max(0) - lo) / cell).astype(np.int64) + 1
    lin = np.floor((V0[:, :2] - lo) / cell).astype(np.int64) @ np.array([ny, 1])
    o = np.lexsort((V0[:, 2], lin))
    starts = np.r_[0, np.flatnonzero(np.diff(lin[o])) + 1]
    cnt = np.diff(np.r_[starts, len(o)])
    floor_z = np.full(nx * ny, np.inf)
    floor_z[lin[o][starts]] = V0[o[starts + cnt // 100], 2]
    floor_z = ndimage.minimum_filter(floor_z.reshape(nx, ny), size=3, mode="nearest")
    del V0, lin, o
    if faces and n0 > faces:
        m = m.simplify_quadric_decimation(target_number_of_triangles=int(faces))
    m.remove_degenerate_triangles()
    m.remove_unreferenced_vertices()
    m.compute_triangle_normals()
    V, F = np.asarray(m.vertices), np.asarray(m.triangles)
    c = V[F].mean(1)
    hor = np.abs(np.asarray(m.triangle_normals)[:, 2]) > flat
    k = np.clip(np.floor((c[:, :2] - lo) / cell).astype(np.int64), 0, [nx - 1, ny - 1])
    low = floor_z[k[:, 0], k[:, 1]]
    part = np.full(len(F), "walls", dtype=object)
    part[hor & (c[:, 2] <= low + band)] = "floor"
    part[hor & (c[:, 2] >= low + roof_h)] = "ceiling"
    if floor_colour is not None and m.has_vertex_colors():
        C = np.asarray(m.vertex_colors).copy()
        C[np.unique(F[part == "floor"])] = floor_colour
        m.vertex_colors = o3d.utility.Vector3dVector(C)
    w = type(m)(m)
    for attr in ("vertex_normals", "triangle_normals"):         # shown unlit, in its colours
        setattr(w, attr, o3d.utility.Vector3dVector(np.zeros((0, 3))))
    o3d.io.write_triangle_mesh(out, w)
    res = {"out": out, "faces": len(F), "floor_faces": int((part == "floor").sum())}
    msg = (f"sionna mesh {out}: {len(F)} faces (from {n0})"
           + (", in the mesh's colours" if m.has_vertex_colors() else "")
           + (f", the floor ({res['floor_faces']} faces) grey {floor_colour[0]:g}"
              if floor_colour is not None and m.has_vertex_colors() else ""))
    if not scene:
        log.info(msg)
        return res
    sdir = os.path.splitext(out)[0]
    os.makedirs(os.path.join(sdir, "meshes"), exist_ok=True)
    xml_mats, shapes, used = {}, [], {}
    for name in ("floor", "ceiling", "walls"):
        sel = part == name
        if not sel.any():
            continue
        q = type(m)(m)
        q.remove_triangles_by_mask(~sel)
        q.remove_unreferenced_vertices()
        for attr in ("vertex_normals", "triangle_normals", "vertex_colors"):   # the material's colour shown
            setattr(q, attr, o3d.utility.Vector3dVector(np.zeros((0, 3))))
        o3d.io.write_triangle_mesh(os.path.join(sdir, "meshes", name + ".ply"), q)
        used[name] = int(sel.sum())
        typ, th = mats[name]
        mid = f"{typ}_{round(float(th) * 100):d}cm"          # one material, parts sharing it
        if mid not in xml_mats:
            xml_mats[mid] = [f'    <bsdf type="itu-radio-material" id="mat-{mid}">',
                             f'        <string name="type" value="{typ}"/>',
                             f'        <float name="thickness" value="{float(th):g}"/>',
                             "    </bsdf>"]
        shapes += [f'    <shape type="ply" id="mesh-{name}">',
                   f'        <string name="filename" value="meshes/{name}.ply"/>',
                   '        <boolean name="face_normals" value="true"/>',
                   f'        <ref id="mat-{mid}" name="bsdf"/>',
                   "    </shape>"]
    xml = ['<scene version="2.1.0">', *sum(xml_mats.values(), []), *shapes, "</scene>"]
    sx = os.path.join(sdir, "scene.xml")
    open(sx, "w").write("\n".join(xml) + "\n")
    log.info(msg + f"\nsionna scene {sx}:"
             + "".join(f"\n  {n}: {used[n]} faces, ITU {mats[n][0]} {float(mats[n][1]):g} m" for n in used)
             + "\n  load: sionna.rt.load_scene(\"" + sx + "\")")
    return {**res, "scene": sx, **used}


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
    sn = sub.add_parser("sionna", help="a mesh for Sionna RT (simplified) of a mesh made by `mesh`")
    sn.add_argument("file", help="the mesh with its ceiling, e.g. mesh_view_<tag>_anchored.ply (not _noceil)")
    sn.add_argument("out", nargs="?", default=None, help="the mesh (default: <file>_sionna.ply)")
    sn.add_argument("--faces", type=int, default=500000, help="simplify to about this many faces (0: not)")
    sn.add_argument("--floor-grey", type=float, default=0.8,
                    help="the floor in this grey (0 black to 1 white; negative: its own colours)")
    sn.add_argument("--scene", action="store_true",
                    help="also a Sionna scene of it: <out>/scene.xml, floor, ceiling and walls apart, "
                         "each an ITU material (--material)")
    sn.add_argument("--material", action="append", default=[], metavar="PART=TYPE[:THICKNESS]",
                    help="floor, ceiling or walls as an ITU material (concrete, brick, plasterboard, wood, "
                         "glass, ceiling_board, marble, floorboard, metal, ...), thickness m; "
                         "default concrete:0.2 each; repeat for each part")
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
        elif a.cmd == "sionna":
            mats = {}
            for spec in a.material:
                part, _, v = spec.partition("=")
                typ, _, th = v.partition(":")
                if part not in SIONNA_MATERIALS or not typ:
                    ap.error(f"--material {spec}: PART=TYPE[:THICKNESS], PART one of floor, ceiling, walls")
                mats[part] = (typ, float(th) if th else SIONNA_MATERIALS[part][1])
            sionna_mesh(a.file, a.out, faces=a.faces, scene=a.scene, materials=mats,
                        floor_colour=None if a.floor_grey < 0 else (a.floor_grey,) * 3)
        elif a.cmd == "scene":
            convert(a.dataset, a.map, a.out, every=a.every, voxel=a.voxel,
                    poisson_depth=a.poisson_depth, trim=a.trim, spacing=a.spacing,
                    color_views=a.color_views, jpeg_quality=a.jpeg_quality)
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
