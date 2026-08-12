#!/usr/bin/env python3
"""
04_prep_sionna.py -- clean the meshed map and emit a loadable Sionna RT scene.

    python3 04_prep_sionna.py mesh_sionna.ply [out_dir]

It picks up mesh_sionna_walls/_floor/_ceiling/_objects.ply written by
02_pcd_to_mesh_sionna_v9.py when they exist, and falls back to the single
combined mesh otherwise. Output is a Mitsuba scene directory that
`sionna.rt.load_scene()` opens directly:

    out_dir/scene.xml
    out_dir/meshes/{walls,floor,ceiling,objects}.ply

WHAT SIONNA ACTUALLY WANTS -- AND WHAT IT DOES NOT
--------------------------------------------------
NOT watertight. This is the most common wrong turn. A RadioMaterial already
models a slab through its `thickness` property, so ONE zero-thickness surface
IS the wall. Wrapping that wall in a closed shell gives it two parallel faces,
every ray crosses two interfaces, and the material is applied twice --
reflection and transmission both come out wrong. Doorways and windows are the
same story: an opening that leaks RF is physically correct, and sealing it
turns a propagation path into a reflector.

What it does want:
  * per-object materials. Sionna binds a mesh to a radio material through the
    BSDF id `mat-<name>`, so the split into walls / floor / ceiling / objects
    is what buys per-surface ITU properties.
  * no coincident duplicate faces. Two triangles in the same place make rays
    hit the same interface twice; the checks below look for this explicitly,
    because it is invisible in a viewer and silently corrupts results.
  * a triangle budget. Tracing cost scales with geometry; this reports it and
    warns past the point where interactive work gets unpleasant.
  * metres. Already true throughout this pipeline.

ITU material names are Sionna's own (itu_concrete, itu_brick, itu_plasterboard,
itu_wood, itu_glass, itu_ceiling_board, itu_floorboard, itu_metal, ...). Edit
MATERIALS below to match what the building is actually made of -- the defaults
are a plausible indoor guess, not a measurement.
"""

import os
import sys
import time

import numpy as np
import open3d as o3d

# part name -> (Sionna ITU material, viewer colour). The colour is cosmetic;
# only the material name affects propagation.
MATERIALS = {
    "walls":   ("itu_concrete",      (0.82, 0.82, 0.86)),
    "floor":   ("itu_floorboard",    (0.55, 0.51, 0.46)),
    "ceiling": ("itu_ceiling_board", (0.90, 0.90, 0.92)),
    "objects": ("itu_wood",          (0.90, 0.50, 0.18)),
    "mesh":    ("itu_concrete",      (0.75, 0.75, 0.78)),
}

MIN_COMPONENT_AREA = 0.05     # m^2; smaller shells are scan debris that cost
                              # tracing time and contribute nothing
MERGE_TOL = 1e-4              # m; weld vertices closer than this
WARN_TRIS = 500_000           # past this, ray tracing gets slow to iterate on


def clean(mesh, name):
    """Remove what degrades a ray-traced result, and report what was found."""
    n0 = len(mesh.triangles)
    mesh.merge_close_vertices(MERGE_TOL)
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()   # coincident faces = double interfaces
    mesh.remove_degenerate_triangles()   # zero-area = NaN normals in tracing
    mesh.remove_unreferenced_vertices()
    n1 = len(mesh.triangles)

    dropped = 0
    if MIN_COMPONENT_AREA > 0 and n1:
        idx, _, areas = mesh.cluster_connected_triangles()
        idx = np.asarray(idx)
        areas = np.asarray(areas)
        crumbs = areas[idx] < MIN_COMPONENT_AREA
        dropped = int((areas < MIN_COMPONENT_AREA).sum())
        if crumbs.any():
            mesh.remove_triangles_by_mask(crumbs)
            mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()
    print(f"  {name:<9} {len(mesh.triangles):>9,} tris "
          f"({mesh.get_surface_area():>9,.0f} m^2)   "
          f"removed {n0 - n1:,} duplicate/degenerate, {dropped} crumb shells")
    return mesh


def audit(mesh, name):
    """Flag the failure modes that are invisible in a viewer but wrong in RT."""
    warns = []
    if mesh.is_self_intersecting():
        warns.append("self-intersecting (rays may hit one surface twice)")
    if mesh.is_watertight():
        # for a scanned building this almost always means a closed shell was
        # built around single-sided structure -- the double-wall problem
        warns.append("WATERTIGHT: if this is a wall shell rather than a solid "
                     "object, the material will be applied twice")
    for w in warns:
        print(f"    warning [{name}] {w}")
    return warns


def write_scene(parts, out_dir):
    """Mitsuba XML in the layout sionna.rt.load_scene expects.

    Sionna resolves a radio material from the BSDF id `mat-<material>`, so the
    id strings below are the actual contract -- the diffuse rgb is only what a
    preview renderer shows."""
    mesh_dir = os.path.join(out_dir, "meshes")
    os.makedirs(mesh_dir, exist_ok=True)
    lines = ['<scene version="2.1.0">',
             '    <default name="spp" value="4096"/>',
             '    <integrator type="path"/>']
    used = {}
    for name, mesh in parts.items():
        mat, rgb = MATERIALS.get(name, MATERIALS["mesh"])
        used[mat] = rgb
    for mat, rgb in used.items():
        lines += [f'    <bsdf type="diffuse" id="mat-{mat}">',
                  f'        <rgb value="{rgb[0]:.4f} {rgb[1]:.4f} '
                  f'{rgb[2]:.4f}" name="reflectance"/>',
                  '    </bsdf>']
    for name, mesh in parts.items():
        mat, _ = MATERIALS.get(name, MATERIALS["mesh"])
        rel = f"meshes/{name}.ply"
        o3d.io.write_triangle_mesh(os.path.join(out_dir, rel), mesh)
        lines += [f'    <shape type="ply" id="mesh-{name}">',
                  f'        <string name="filename" value="{rel}"/>',
                  f'        <ref id="mat-{mat}" name="bsdf"/>',
                  # face_normals: use the geometric normal, not an interpolated
                  # one. Interpolated normals smooth a faceted scan and send
                  # specular reflections in directions the geometry does not
                  # support.
                  '        <boolean name="face_normals" value="true"/>',
                  '    </shape>']
    lines.append('</scene>')
    path = os.path.join(out_dir, "scene.xml")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return path


def main():
    infile = sys.argv[1] if len(sys.argv) > 1 else "mesh_sionna.ply"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "sionna_scene"
    t0 = time.time()
    stem = infile.rsplit(".", 1)[0]

    parts = {}
    for name in ("walls", "floor", "ceiling", "objects"):
        p = f"{stem}_{name}.ply"
        if os.path.exists(p):
            m = o3d.io.read_triangle_mesh(p)
            if len(m.triangles):
                parts[name] = m
    if not parts:
        if not os.path.exists(infile):
            sys.exit(f"ERROR: {infile} not found")
        m = o3d.io.read_triangle_mesh(infile)
        if not len(m.triangles):
            sys.exit(f"ERROR: {infile} has no triangles")
        parts["mesh"] = m
        print("No split parts found -- using the combined mesh with a single "
              "material.\nRun 02_pcd_to_mesh_sionna_v9.py with split=True for "
              "per-surface ITU materials.")

    print(f"\nCleaning {len(parts)} part(s):")
    total = 0
    all_warns = []
    for name in list(parts):
        parts[name] = clean(parts[name], name)
        if len(parts[name].triangles) == 0:
            del parts[name]
            continue
        all_warns += audit(parts[name], name)
        total += len(parts[name].triangles)

    os.makedirs(out_dir, exist_ok=True)
    path = write_scene(parts, out_dir)

    print(f"\nWrote {path}")
    for name in parts:
        mat, _ = MATERIALS.get(name, MATERIALS["mesh"])
        print(f"  meshes/{name}.ply -> {mat}")
    print(f"\ntotal {total:,} triangles")
    if total > WARN_TRIS:
        print(f"  above {WARN_TRIS:,}: tracing will be slow to iterate on.\n"
              f"  Raise plane_tri_size (structure) or obj_target_tris\n"
              f"  (objects) in 02_pcd_to_mesh_sionna_v9.py.")
    if not all_warns:
        print("  no geometry warnings")

    print(f"""
Load it:

    from sionna.rt import load_scene, PlanarArray, Transmitter, Receiver
    scene = load_scene("{path}")
    print(scene.objects.keys())          # mesh-walls, mesh-floor, ...
    scene.get("mesh-walls").radio_material = "itu_brick"   # to override

Materials are a plausible indoor guess, not a measurement -- edit MATERIALS
in this script to match the actual construction before trusting the results.
Done in {time.time() - t0:.1f} s""")


if __name__ == "__main__":
    main()
