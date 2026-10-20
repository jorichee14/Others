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
    pc = o3d.io.read_point_cloud(os.path.expanduser(map_path))
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
    """A copy of the mesh without the faces whose centre is above z."""
    out = type(mesh)(mesh)
    F, V = np.asarray(out.triangles), np.asarray(out.vertices)
    out.remove_triangles_by_mask(V[F].mean(1)[:, 2] > z)
    out.remove_unreferenced_vertices()
    return out


def default_mesh_out(map_path):
    d, name = os.path.split(os.path.abspath(map_path))
    stem = os.path.splitext(name)[0]
    if stem.startswith("map_final_"):
        return os.path.join(d, "mesh_" + stem[len("map_final_"):] + ".ply")
    return os.path.join(d, stem + "_mesh.ply")


def map_mesh(map_path, traj, out=None, *, voxel=0.02, poisson_depth=0, trim=0.10,
         ceil=None, ceil_margin=0.10, show=False):
    """The mesh step alone: mesh of the map (+ a copy without the ceiling)."""
    o3d = _o3d()
    out = out or default_mesh_out(map_path)
    centres = read_tum_positions(traj)
    m, mstat = build_mesh(map_path, centres, voxel, poisson_depth, trim)
    o3d.io.write_triangle_mesh(out, m)
    V = np.asarray(m.vertices)
    text, hc, hn = height_histogram(V[:, 2])
    log.info("height of the mesh (5 cm bins, top first):\n" + text)
    res = {"out": out, "vertices": len(V), "coloured": m.has_vertex_colors(), **mstat}
    report = (f"mesh {out}: {mstat['faces_kept']} faces ({mstat['faces']} before the "
              f"{trim} m trim), {len(V)} vertices, "
              f"{'coloured from the map' if m.has_vertex_colors() else 'no colour (map has none)'}")
    shown = m
    if ceil is not None:
        if str(ceil) == "auto":
            peak = ceiling_height(hc, hn)
            z = peak - ceil_margin
            how = f"ceiling peak {peak:+.2f} less {ceil_margin} m"
        else:
            z = float(ceil)
            how = "given"
        nc = cut_above(m, z)
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
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        if a.cmd == "scene":
            convert(a.dataset, a.map, a.out, every=a.every, voxel=a.voxel,
                    poisson_depth=a.poisson_depth, trim=a.trim, spacing=a.spacing,
                    color_views=a.color_views, jpeg_quality=a.jpeg_quality)
        else:
            map_mesh(a.map, a.traj, a.out, voxel=a.voxel, poisson_depth=a.poisson_depth,
                     trim=a.trim, ceil=a.ceil, ceil_margin=a.ceil_margin, show=a.show)
    except ConvertError as e:
        sys.exit(str(e))


if __name__ == "__main__":
    main()
