#!/usr/bin/env python3
"""The scanned building as a USD stage for Isaac Sim: the pipeline's mesh
(mesh_view_<tag>.ply, recoloured or not) decimated to a size the renderer
likes, its vertex colours kept, z up, metres, in the map frame.

    python3 isaac_coop_sim/export_usd.py <mesh.ply> <out.usd> [--faces 3000000]
        [--frame frames/anchor_frame.json] [--no-colours]

--frame moves a mesh made in the mapping pass's GLIM world into the anchored
map frame (05's T_N_world), the frame the surveys' and coop runs'
trajectories and boards are in, so a scenario built from those drives
through the right doors. Written next to the USD: <out>.json with the
face count, the bounding box and the floor height (the most populated
height in the lower part of the mesh), which the scenario builder reads.
Open3D decimates; usd-core (pip) writes the stage: one Mesh prim
/World/building with primvars:displayColor per vertex, a physics
collision (meshes are static), the default prim set, metres per unit 1,
up axis Z.
"""
import argparse
import json
import os
import sys

import numpy as np


def floor_height(z, lower=0.4):
    c, e = np.histogram(z, bins=200)
    lo = max(int(len(c) * lower), 1)
    return float(0.5 * (e[:-1] + e[1:])[int(np.argmax(c[:lo]))])


def load_mesh(path, faces=0, frame=None, colours=True, log=print):
    """(V, F, C or None, info) of the mesh, decimated to about `faces`,
    moved into the anchored frame with `frame`."""
    import open3d as o3d
    m = o3d.io.read_triangle_mesh(os.path.expanduser(path))
    n0 = len(m.triangles)
    if not n0:
        raise SystemExit(f"{path}: no faces")
    log(f"{path}: {n0} faces, {len(m.vertices)} vertices, colours {'yes' if m.has_vertex_colors() else 'no'}")
    if faces and n0 > faces:
        m = m.simplify_quadric_decimation(int(faces))
        m.remove_degenerate_triangles()
        m.remove_unreferenced_vertices()
        log(f"  decimated to {len(m.triangles)} faces")
    V = np.asarray(m.vertices, dtype=np.float64)
    if frame:
        T = np.array(json.load(open(os.path.expanduser(frame)))["T_N_world"], float)
        V = V @ T[:3, :3].T + T[:3, 3]
        log(f"  moved into the anchored frame of {frame}")
    F = np.asarray(m.triangles, dtype=np.int64)
    C = np.asarray(m.vertex_colors, dtype=np.float32) if (colours and m.has_vertex_colors()) else None
    info = {"faces": int(len(F)), "vertices": int(len(V)),
            "bbox_min": V.min(0).round(3).tolist(), "bbox_max": V.max(0).round(3).tolist(),
            "floor_z": round(floor_height(V[:, 2]), 3), "frame": frame, "source": os.path.abspath(path)}
    return V, F, C, info


def write_usd(out, V, F, C=None, prim="/World/building", log=print):
    from pxr import Usd, UsdGeom, UsdPhysics, Vt, Gf, Sdf
    stage = Usd.Stage.CreateNew(os.path.expanduser(out))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    mesh = UsdGeom.Mesh.Define(stage, prim)
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(V.astype(np.float32)))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(F), 3, np.int32)))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(F.astype(np.int32).ravel()))
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreateDoubleSidedAttr(True)
    ext = UsdGeom.Mesh.ComputeExtentFromPlugins(mesh, Usd.TimeCode.Default())
    if ext:
        mesh.CreateExtentAttr(ext)
    if C is not None:
        pv = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("displayColor", Sdf.ValueTypeNames.Color3fArray,
                                                     UsdGeom.Tokens.vertex)
        pv.Set(Vt.Vec3fArray.FromNumpy(np.clip(C, 0, 1).astype(np.float32)))
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    mesh.GetPrim().CreateAttribute("physics:approximation", Sdf.ValueTypeNames.Token).Set("none")
    stage.GetRootLayer().Save()
    log(f"wrote {out}: {len(F)} faces at {prim}" + (", vertex colours" if C is not None else ""))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mesh")
    ap.add_argument("out")
    ap.add_argument("--faces", type=int, default=3_000_000, help="decimate to about this many (0: as is)")
    ap.add_argument("--frame", default=None, help="anchor_frame.json: into the anchored map frame")
    ap.add_argument("--no-colours", action="store_true")
    a = ap.parse_args()
    V, F, C, info = load_mesh(a.mesh, a.faces, a.frame, not a.no_colours)
    write_usd(a.out, V, F, C)
    jp = os.path.splitext(os.path.expanduser(a.out))[0] + ".json"
    json.dump(info, open(jp, "w"), indent=1)
    print(f"wrote {jp}: floor z {info['floor_z']} m, bbox {info['bbox_min']} .. {info['bbox_max']}")


if __name__ == "__main__":
    main()
