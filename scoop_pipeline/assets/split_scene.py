#!/usr/bin/env python3
"""A room's map split into its structure and its objects, each meshed at its
own resolution, every mesh at several levels of detail.

    python3 scoop_pipeline/assets/split_scene.py <room map .ply/.pcd> <registry> <out dir>
        --traj <TUM of the camera / LiDAR, the map's frame>
        --room YAW XMIN XMAX YMIN YMAX
        [--assign 0.03] [--shell-tol 0.06]
        [--shell-voxel 0.08] [--other-voxel 0.04] [--object-voxel 0.02]
        [--lod 1,0.3,0.1] [--no-mesh]

The map is the room at full resolution, coloured, in the registry's frame
(ED305: ed305_room_ceil.ply, the anchored map cropped to the room with its
ceiling); the registry is assets/ after consolidate (objects.json and each
object's measured points). --room is crop_room.py's --yaw and --bounds.

1. Objects: every map point within --assign of a registry object's points is
   that object's (the nearest one's), so each object takes its surfaces from
   the full map, colours included, not from the registry's own points.
2. Shell: of the rest, the points within --shell-tol of the floor, the
   ceiling (the lowest / highest strong height peaks) or one of the four
   walls (--room) -- the room's planes.
3. Other: what is left -- things the registry does not name, fixtures,
   clutter. Kept apart so nothing of the map is lost.
4. Each part meshed alone (datasets/lod3dgs.build_mesh: Poisson with cells
   about one voxel across, normals turned to the nearest --traj pose, faces
   farther than 3 voxels from a point trimmed): the shell coarse (--shell-voxel), the other part in
   between, every object fine (--object-voxel): a big flat wall needs few
   faces, a monitor on a desk many.
5. Levels of detail: every mesh simplified to each --lod share of its faces
   (quadric decimation): <name>_lod0.ply (as meshed), _lod1, _lod2, ...

<out dir>:
  points/shell.ply, points/other.ply, points/objects/<id>.ply
  meshes/shell_lod*.ply, meshes/other_lod*.ply, meshes/objects/<id>_lod*.ply
  scene.json   per part: points, voxel, faces per level, files; per object
               its label, centroid, box and "on" (from the registry)
Needs open3d, scipy.
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "datasets"))
sys.path.insert(0, os.path.join(ROOT, "analysis"))

from consolidate import read_ply                                        # noqa: E402


def height_peaks(z, bin_m=0.02, frac=0.3):
    """(lowest, highest) height peak holding at least frac of the largest."""
    h, e = np.histogram(z, np.arange(z.min(), z.max() + bin_m, bin_m))
    hs = np.convolve(h, np.ones(3) / 3, mode="same")
    c = 0.5 * (e[:-1] + e[1:])
    pk = [i for i in range(1, len(hs) - 1) if hs[i] >= hs[i - 1] and hs[i] > hs[i + 1]]
    top = max(hs[i] for i in pk)
    strong = [i for i in pk if hs[i] >= frac * top]
    return float(c[min(strong)]), float(c[max(strong)])


def split(xyz, objs_xyz, room, assign=0.03, shell_tol=0.06, log=print):
    """-> (object index per point, -1 none; shell mask; floor z, ceiling z)."""
    from scipy.spatial import cKDTree
    owner = np.full(len(xyz), -1, np.int64)
    if objs_xyz:
        P = np.vstack(objs_xyz)
        lab = np.concatenate([np.full(len(p), k) for k, p in enumerate(objs_xyz)])
        d, j = cKDTree(P).query(xyz, distance_upper_bound=assign, workers=-1)
        hit = np.isfinite(d)
        owner[hit] = lab[j[hit]]
    rest = owner < 0
    floor, ceil = height_peaks(xyz[rest, 2])
    yaw, x0, x1, y0, y1 = room
    a = np.radians(-yaw)
    c, s = np.cos(a), np.sin(a)
    x = c * xyz[:, 0] - s * xyz[:, 1]
    y = s * xyz[:, 0] + c * xyz[:, 1]
    plane = np.minimum.reduce([np.abs(xyz[:, 2] - floor), np.abs(xyz[:, 2] - ceil),
                               np.abs(x - x0), np.abs(x - x1), np.abs(y - y0), np.abs(y - y1)])
    shell = rest & (plane <= shell_tol)
    log("  floor z %.3f, ceiling z %.3f; %d points: %d in objects, %d shell, %d other"
        % (floor, ceil, len(xyz), int((~rest).sum()), int(shell.sum()), int((rest & ~shell).sum())))
    return owner, shell, floor, ceil


def mesh_levels(pc, centres, voxel, lods, out_stem, log=print):
    """Poisson at voxel, then each lod share of its faces -> [(file, faces)]."""
    import open3d as o3d
    import lod3dgs
    # Poisson cells about one voxel across: build_mesh's own choice never goes
    # below depth 8 (256 cells), which leaves a 10 m shell at 4 cm whatever its voxel
    P = np.asarray(pc.points)
    ext = float(np.max(P.max(0) - P.min(0))) * 1.1
    depth = int(np.clip(np.ceil(np.log2(max(ext, voxel) / voxel)), 5, 12))
    try:
        mesh, info = lod3dgs.build_mesh(pc, centres, voxel=voxel, depth=depth,
                                        trim=max(3 * voxel, 0.05))
    except lod3dgs.ConvertError as e:
        log("    %s: no mesh (%s)" % (os.path.basename(out_stem), e))
        return []
    mesh.compute_vertex_normals()
    n0 = len(mesh.triangles)
    out = []
    for k, share in enumerate(lods):
        m = mesh if share >= 1.0 else mesh.simplify_quadric_decimation(max(4, int(n0 * share)))
        f = "%s_lod%d.ply" % (out_stem, k)
        o3d.io.write_triangle_mesh(f, m)
        out.append((f, len(m.triangles)))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__.split("\n\n", 1)[1])
    ap.add_argument("map")
    ap.add_argument("registry")
    ap.add_argument("out")
    ap.add_argument("--traj", required=True)
    ap.add_argument("--room", type=float, nargs=5, required=True,
                    metavar=("YAW", "XMIN", "XMAX", "YMIN", "YMAX"))
    ap.add_argument("--assign", type=float, default=0.03)
    ap.add_argument("--shell-tol", type=float, default=0.06)
    ap.add_argument("--shell-voxel", type=float, default=0.08)
    ap.add_argument("--other-voxel", type=float, default=0.04)
    ap.add_argument("--object-voxel", type=float, default=0.02)
    ap.add_argument("--lod", default="1,0.3,0.1")
    ap.add_argument("--no-mesh", action="store_true")
    a = ap.parse_args(argv)
    import open3d as o3d
    import lod3dgs
    out = os.path.expanduser(a.out)
    for d in ("points/objects", "meshes/objects"):
        os.makedirs(os.path.join(out, d), exist_ok=True)
    pc = o3d.io.read_point_cloud(os.path.expanduser(a.map))
    xyz = np.asarray(pc.points)
    if not len(xyz):
        raise SystemExit("no points in %s" % a.map)
    reg = os.path.expanduser(a.registry)
    objs = json.load(open(os.path.join(reg, "objects.json")))["objects"]
    print("%s: %d points; %s: %d objects" % (a.map, len(xyz), reg, len(objs)))
    objs_xyz = [read_ply(os.path.join(reg, o["points"]))[0] for o in objs]
    owner, shell, floor, ceil = split(xyz, objs_xyz, tuple(a.room), a.assign, a.shell_tol)
    centres = lod3dgs.read_tum_positions(a.traj, stride=10)
    lods = [float(v) for v in a.lod.split(",")]
    scene = {"map": os.path.abspath(os.path.expanduser(a.map)), "registry": reg,
             "room": a.room, "floor_z": floor, "ceiling_z": ceil, "lod_shares": lods,
             "parts": {}, "objects": []}

    def part(name, mask, voxel, sub=""):
        sel = pc.select_by_index(np.flatnonzero(mask).tolist())
        pf = os.path.join(out, "points", sub, name + ".ply")
        o3d.io.write_point_cloud(pf, sel)
        rec = {"points": int(mask.sum()), "points_file": os.path.relpath(pf, out), "voxel": voxel}
        if not a.no_mesh and mask.sum() >= 50:
            lv = mesh_levels(sel, centres, voxel, lods, os.path.join(out, "meshes", sub, name))
            rec["levels"] = [{"file": os.path.relpath(f, out), "faces": n} for f, n in lv]
        return rec

    print("meshing: shell at %.2f m, other at %.2f m, objects at %.2f m; levels %s"
          % (a.shell_voxel, a.other_voxel, a.object_voxel, a.lod))
    scene["parts"]["shell"] = part("shell", shell, a.shell_voxel)
    scene["parts"]["other"] = part("other", (owner < 0) & ~shell, a.other_voxel)
    for k, o in enumerate(objs):
        m = owner == k
        rec = part(o["id"], m, a.object_voxel, "objects")
        rec.update({"id": o["id"], "label": o["label"], "agreement": o.get("agreement"),
                    "on": o.get("on"), "centroid": o["centroid"],
                    "aabb_min": o["aabb_min"], "aabb_max": o["aabb_max"]})
        scene["objects"].append(rec)
        if (k + 1) % 20 == 0:
            print("    %d / %d objects" % (k + 1, len(objs)))
    json.dump(scene, open(os.path.join(out, "scene.json"), "w"), indent=1)
    for name in ("shell", "other"):
        p = scene["parts"][name]
        print("  %-6s %9d points, %s" % (name, p["points"], ", ".join(
            "lod%d %d faces" % (k, l["faces"]) for k, l in enumerate(p.get("levels", [])))))
    fo = [sum(l["faces"] for l in o.get("levels", [])[:1]) for o in scene["objects"]]
    print("  objects: %d, %d points, %d faces at lod0" % (
        len(objs), sum(o["points"] for o in scene["objects"]), sum(fo)))
    print("-> %s (points/, meshes/, scene.json)" % out)


if __name__ == "__main__":
    sys.exit(main())
