#!/usr/bin/env python3
"""Convert a point cloud between formats (pcd, ply, xyz, xyzrgb, pts), and
optionally thin it, e.g. a map for CloudCompare or a viewer.

    python processing/convert_cloud.py <in> [<out>] [--voxel 0.05] [--ascii]

    e.g. python processing/convert_cloud.py \\
             <data>/processed/<date>/<pass>/mapping/map_final_<pass>_<date>.pcd
         -> map_final_<pass>_<date>.ply next to it

<out>     the output file; its extension picks the format (default: <in> as .ply)
--voxel   thin to one point per voxel of this size (m) first; the default
          output name then ends in _<voxel in cm>cm (map_final_..._5cm.ply)
--ascii   text instead of binary (much larger; only for tools that need it)

Colours and normals are kept. Refuses to overwrite an existing file.
"""
import argparse
import os
import sys

FORMATS = (".pcd", ".ply", ".xyz", ".xyzrgb", ".xyzn", ".pts")


def default_out(src, voxel):
    stem = os.path.splitext(src)[0]
    if voxel:
        stem += "_%gcm" % round(voxel * 100, 2)
    return stem + ".ply"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src")
    ap.add_argument("out", nargs="?")
    ap.add_argument("--voxel", type=float, default=0.0, help="thin to this voxel size (m)")
    ap.add_argument("--ascii", action="store_true", help="write text instead of binary")
    a = ap.parse_args()

    src = os.path.expanduser(a.src)
    out = os.path.expanduser(a.out) if a.out else default_out(src, a.voxel)
    for p in (src, out):
        if os.path.splitext(p)[1].lower() not in FORMATS:
            sys.exit(f"{p}: unknown format; one of {', '.join(FORMATS)}")
    if not os.path.isfile(src):
        sys.exit(f"no file {src}")
    if os.path.exists(out):
        sys.exit(f"{out} exists; remove it or name another output")

    import open3d as o3d
    pcd = o3d.io.read_point_cloud(src)
    n = len(pcd.points)
    if n == 0:
        sys.exit(f"{src}: no points read")
    print(f"read  {src}: {n} points"
          f"{', colours' if pcd.has_colors() else ''}{', normals' if pcd.has_normals() else ''}")
    if a.voxel > 0:
        pcd = pcd.voxel_down_sample(a.voxel)
        print(f"thinned to {a.voxel * 100:g} cm: {len(pcd.points)} points "
              f"({100 * len(pcd.points) / n:.0f} %)")
    if not o3d.io.write_point_cloud(out, pcd, write_ascii=a.ascii):
        sys.exit(f"could not write {out}")
    print(f"wrote {out}: {len(pcd.points)} points, {os.path.getsize(out) / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
