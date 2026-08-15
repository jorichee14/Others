#!/usr/bin/env python3

"""

Generate the procedural placeholder asset library for stage [7] synthesize.



Stage [7] does not paste an asset into the map verbatim -- it FITS the asset to

the measured cluster (yaw + anisotropic scale + ICP) and repaints it from the

observed colours. So a placeholder does not need to be a beautiful model; it

needs to be metrically honest and consistently framed, because those two

properties are what the fitter reads:



  * METRIC        real-world metres, so the initial scale is ~1 and the clamp

                  band on the anisotropic fit means something. A chair modelled

                  0.9 m tall lets a 0.85 m measured chair land at 0.94x, well

                  inside the band; a unit-cube chair would saturate the clamp

                  and the fit would carry no information.

  * FRAMED        +z up, +y front, origin at the footprint centre with the base

                  at z = 0. The floor snap is then a pure translate (set z to

                  the classified floor plane), and the yaw search starts from a

                  known front instead of an arbitrary one.



Everything here is deterministic: same script, same bytes. Real models drop

into the same manifest slots later without touching pipeline code -- see

README.md for the contract a hand-made asset has to honour.



    python3 make_assets.py [--out DIR] [--verify]

"""



import argparse

import json

import os

import numpy as np

import open3d as o3d





# =============================================================================

# CONVENTIONS  (the manifest states these; every builder below obeys them)

# =============================================================================

UP = "+z"          # gravity axis, matching the GLIM world frame

FRONT = "+y"       # the face a viewer normally sees: chair seat, TV screen

ANCHOR = "base"    # origin at footprint centre, min(z) == 0



# Palette. Placeholder colours only -- synthesize transfers the measured

# colours over these wherever the LiDAR/camera actually saw the surface. They

# still matter for the UNSEEN back faces, which keep whatever is baked here

# tinted toward the instance's median colour, so plausible beats arbitrary.

C_WOOD = (0.55, 0.40, 0.27)

C_WOOD_D = (0.40, 0.28, 0.19)

C_FABRIC = (0.42, 0.45, 0.52)

C_FABRIC_D = (0.33, 0.36, 0.43)

C_METAL = (0.62, 0.63, 0.66)

C_DARK = (0.13, 0.13, 0.15)

C_SCREEN = (0.06, 0.07, 0.09)

C_WHITE = (0.88, 0.88, 0.86)

C_TERRA = (0.66, 0.36, 0.24)

C_SOIL = (0.24, 0.18, 0.13)

C_LEAF = (0.24, 0.45, 0.22)

C_LEAF_L = (0.33, 0.56, 0.28)

C_CERAMIC = (0.72, 0.74, 0.78)





# =============================================================================

# PRIMITIVES

# =============================================================================

def _paint(mesh, color):

    mesh.paint_uniform_color(color)

    mesh.compute_vertex_normals()

    return mesh





def box(size, center, color):

    """Axis-aligned box of extent `size` centred at `center`."""

    w, d, h = size

    m = o3d.geometry.TriangleMesh.create_box(w, d, h)

    m.translate(np.asarray(center, float) - np.array([w, d, h]) * 0.5)

    return _paint(m, color)





def cyl(radius, height, center, color, axis="z", res=24):

    """Cylinder centred at `center`, aligned to `axis`."""

    m = o3d.geometry.TriangleMesh.create_cylinder(radius, height, resolution=res)

    if axis == "x":

        m.rotate(o3d.geometry.get_rotation_matrix_from_xyz((0, np.pi / 2, 0)),

                 center=(0, 0, 0))

    elif axis == "y":

        m.rotate(o3d.geometry.get_rotation_matrix_from_xyz((np.pi / 2, 0, 0)),

                 center=(0, 0, 0))

    m.translate(np.asarray(center, float))

    return _paint(m, color)





def sphere(radius, center, color, res=14):

    m = o3d.geometry.TriangleMesh.create_sphere(radius, resolution=res)

    m.translate(np.asarray(center, float))

    return _paint(m, color)





def revolve(profile, color, seg=40, cap=True):

    """Solid of revolution about +z from a bottom-to-top (radius, z) profile.



    Open3D has no cone-frustum or lathe primitive, and pots, vases and tapered

    legs are all frusta. Radius 0 at an end becomes a pole vertex rather than a

    ring of coincident vertices, which would leave degenerate triangles that

    corrupt the normals -- and normals are what the asset's shading, and the

    surface-normal test in the fitter, both read.

    """

    prof = np.asarray(profile, float)

    ang = np.linspace(0.0, 2.0 * np.pi, seg, endpoint=False)

    ca, sa = np.cos(ang), np.sin(ang)



    verts, rings = [], []

    for r, z in prof:

        if r <= 1e-9:

            rings.append(("pole", len(verts)))

            verts.append([0.0, 0.0, z])

        else:

            rings.append(("ring", len(verts)))

            verts.extend(np.stack([r * ca, r * sa, np.full(seg, z)], 1))



    tris = []

    for i in range(len(prof) - 1):

        (k0, i0), (k1, i1) = rings[i], rings[i + 1]

        for j in range(seg):

            jn = (j + 1) % seg

            if k0 == "ring" and k1 == "ring":

                # winding chosen so the face normal is +r (outward): with rings

                # running counter-clockwise in xy and the profile running

                # bottom-to-top, (lower_j, lower_j+1, upper_j+1) gives t_hat x z

                tris.append([i0 + j, i0 + jn, i1 + jn])

                tris.append([i0 + j, i1 + jn, i1 + j])

            elif k0 == "pole":

                tris.append([i0, i1 + jn, i1 + j])          # bottom pole: -z

            else:

                tris.append([i1, i0 + j, i0 + jn])          # top pole: +z



    if cap:

        for end, (k, base) in ((0, rings[0]), (-1, rings[-1])):

            if k != "ring":

                continue

            z = prof[end][1]

            c = len(verts)

            verts.append([0.0, 0.0, z])

            for j in range(seg):

                jn = (j + 1) % seg

                tris.append([c, base + jn, base + j] if end == 0

                            else [c, base + j, base + jn])



    m = o3d.geometry.TriangleMesh(

        o3d.utility.Vector3dVector(np.asarray(verts, float)),

        o3d.utility.Vector3iVector(np.asarray(tris, np.int32)))

    return _paint(m, color)





def combine(parts):

    out = o3d.geometry.TriangleMesh()

    for p in parts:

        out += p

    out.compute_vertex_normals()

    return out





def frame(mesh):

    """Move a mesh into the asset frame: base at z=0, footprint centred on xy.



    Applied to every asset as the last step, so a builder can lay parts out in

    whatever local coordinates are convenient and still emit a conforming asset.

    """

    lo = mesh.get_min_bound()

    hi = mesh.get_max_bound()

    mesh.translate((-(lo[0] + hi[0]) * 0.5, -(lo[1] + hi[1]) * 0.5, -lo[2]))

    return mesh





# =============================================================================

# ASSETS

# =============================================================================

# Each builder returns a mesh in loose local coordinates; frame() fixes it up.

# Dimensions are ordinary furniture sizes, not idealised ones -- the fitter's

# job is easier the closer the placeholder starts to a real object.



def chair():

    """Four-leg dining chair. Back at -y so a sitter faces +y (the front)."""

    p = []

    for sx in (-1, 1):

        for sy in (-1, 1):

            p.append(cyl(0.018, 0.44, (sx * 0.185, sy * 0.205, 0.22), C_WOOD_D))

    p.append(box((0.44, 0.46, 0.045), (0, 0, 0.4625), C_FABRIC))

    # open back: two stiles, a top rail and two slats. A SOLID back panel would

    # be simpler, but the slats would then sit inside it -- coincident faces

    # that z-fight, and interior geometry that uniform sampling turns into

    # points buried in the asset where no surface exists.

    for sx in (-1, 1):

        p.append(box((0.045, 0.045, 0.400),

                     (sx * 0.1975, -0.2075, 0.685), C_WOOD))

    p.append(box((0.44, 0.045, 0.075), (0, -0.2075, 0.8475), C_WOOD))

    for z in (0.600, 0.720):

        p.append(box((0.35, 0.028, 0.070), (0, -0.2075, z), C_WOOD))

    return combine(p)





def dining_table():

    """Rectangular 4-seat table."""

    p = [box((1.40, 0.80, 0.040), (0, 0, 0.730), C_WOOD)]

    for sx in (-1, 1):

        for sy in (-1, 1):

            p.append(box((0.06, 0.06, 0.710),

                         (sx * 0.635, sy * 0.335, 0.355), C_WOOD_D))

    p.append(box((1.30, 0.035, 0.070), (0, -0.345, 0.675), C_WOOD_D))

    p.append(box((1.30, 0.035, 0.070), (0, 0.345, 0.675), C_WOOD_D))

    return combine(p)





def couch():

    """Three-seat sofa, back at -y."""

    p = [box((1.90, 0.85, 0.25), (0, 0, 0.225), C_FABRIC_D)]

    for sx in (-1, 1):

        for sy in (-1, 1):

            p.append(cyl(0.025, 0.10, (sx * 0.86, sy * 0.35, 0.05), C_DARK))

    p.append(box((1.90, 0.20, 0.45), (0, -0.325, 0.575), C_FABRIC))

    for sx in (-1, 1):

        p.append(box((0.18, 0.85, 0.28), (sx * 0.86, 0, 0.490), C_FABRIC))

    for x in (-0.55, 0.0, 0.55):

        p.append(box((0.52, 0.62, 0.13), (x, 0.08, 0.415), C_FABRIC))

    return combine(p)





def potted_plant():

    """Tapered pot, soil, stem, foliage cluster."""

    rng = np.random.default_rng(7)          # fixed: same bytes every run

    p = [revolve([(0.115, 0.00), (0.125, 0.015), (0.165, 0.265),

                  (0.180, 0.285), (0.180, 0.300)], C_TERRA),

         revolve([(0.170, 0.282), (0.170, 0.292), (0.0, 0.292)], C_SOIL),

         cyl(0.013, 0.24, (0, 0, 0.40), C_WOOD_D)]

    for i in range(9):

        d = rng.normal(0.0, 0.085, 3)

        d[2] = abs(d[2]) * 0.9

        c = np.array([0.0, 0.0, 0.585]) + d

        p.append(sphere(rng.uniform(0.070, 0.108), c,

                        C_LEAF if i % 2 else C_LEAF_L))

    return combine(p)





def tv():

    """Flat panel on a pedestal stand, screen facing +y.



    The wall-mounted case is not a separate asset: stage [7] derives the back

    plane from this bbox, and a TV judged wall-mounted is removed anyway.

    """

    p = [box((0.42, 0.24, 0.022), (0, 0, 0.011), C_DARK),

         box((0.11, 0.07, 0.105), (0, 0, 0.074), C_METAL),

         box((1.10, 0.048, 0.625), (0, 0, 0.438), C_DARK),

         box((1.06, 0.010, 0.585), (0, 0.029, 0.438), C_SCREEN)]

    return combine(p)





def bed():

    """Double bed, headboard at -y."""

    p = [box((1.55, 2.05, 0.26), (0, 0, 0.180), C_WOOD_D),

         box((1.50, 2.00, 0.24), (0, 0, 0.430), C_WHITE),

         box((1.58, 0.05, 0.55), (0, -1.030, 0.400), C_WOOD)]

    for sx in (-1, 1):

        for sy in (-1, 1):

            p.append(box((0.07, 0.07, 0.05),

                         (sx * 0.72, sy * 0.96, 0.025), C_WOOD_D))

    for sx in (-1, 1):

        p.append(box((0.62, 0.38, 0.11), (sx * 0.38, -0.78, 0.605), C_WHITE))

    return combine(p)





def refrigerator():

    """Two-door upright, doors and handles on +y."""

    p = [box((0.70, 0.68, 1.80), (0, 0, 0.90), C_METAL),

         box((0.70, 0.030, 1.16), (0, 0.355, 1.205), C_WHITE),

         box((0.70, 0.030, 0.60), (0, 0.355, 0.315), C_WHITE)]

    for z in (1.02, 0.50):

        p.append(cyl(0.016, 0.34, (0.26, 0.378, z), C_DARK, axis="z"))

    return combine(p)





def vase():

    """Lathed ceramic vase -- yaw-symmetric, so the fitter skips the yaw search."""

    return revolve([(0.058, 0.000), (0.070, 0.020), (0.105, 0.120),

                    (0.110, 0.170), (0.088, 0.245), (0.062, 0.300),

                    (0.072, 0.340), (0.062, 0.350)], C_CERAMIC, seg=48)





def laptop():

    """Open laptop, hinge at -y, screen tilted back ~15 degrees."""

    base = box((0.33, 0.235, 0.016), (0, 0, 0.008), C_METAL)

    kb = box((0.30, 0.155, 0.003), (0, 0.030, 0.017), C_DARK)

    scr = box((0.33, 0.011, 0.215), (0, 0, 0.1075), C_METAL)

    face = box((0.305, 0.004, 0.190), (0, 0.008, 0.1075), C_SCREEN)

    lid = combine([scr, face])

    lid.rotate(o3d.geometry.get_rotation_matrix_from_xyz((np.deg2rad(15), 0, 0)),

               center=(0, 0, 0))

    lid.translate((0, -0.112, 0.016))

    return combine([base, kb, lid])





# class name -> (builder, support surface, yaw symmetry order)

#

#   support        where stage [7] snaps it: "floor" sits on the classified

#                  floor plane, "surface" on a table/shelf top when one is under

#                  the instance and the floor otherwise.

#   yaw_symmetry   rotations of 360/n that leave the shape unchanged. 1 = none

#                  (full yaw search plus the 180-degree view disambiguation),

#                  2 = front/back alike, 0 = continuous (skip the search).

#                  It is a fitter cost knob AND a correctness one: searching

#                  yaw on a round vase fits noise.

ASSETS = {

    "chair":        (chair,        "floor",   1),

    "dining table": (dining_table, "floor",   2),

    "couch":        (couch,        "floor",   1),

    "potted plant": (potted_plant, "any",     0),

    "tv":           (tv,           "surface", 1),

    "bed":          (bed,          "floor",   1),

    "refrigerator": (refrigerator, "floor",   1),

    "vase":         (vase,         "surface", 0),

    "laptop":       (laptop,       "surface", 1),

}





def slug(name):

    return name.replace(" ", "_")





def build(out_dir):

    mesh_root = os.path.join(out_dir, "meshes")

    entries = {}

    for cls, (fn, support, sym) in ASSETS.items():

        m = frame(fn())

        m.compute_vertex_normals()

        d = os.path.join(mesh_root, slug(cls))

        os.makedirs(d, exist_ok=True)

        rel = os.path.join("meshes", slug(cls), f"{slug(cls)}_01.ply")

        o3d.io.write_triangle_mesh(os.path.join(out_dir, rel), m,

                                   write_ascii=False, compressed=True)

        size = (m.get_max_bound() - m.get_min_bound()).round(4).tolist()

        entries[cls] = {

            "support": support,

            "yaw_symmetry": sym,

            "variants": [{

                "path": rel.replace(os.sep, "/"),

                "size": size,

                "vertices": len(m.vertices),

                "triangles": len(m.triangles),

                "source": "procedural",

            }],

        }

        print(f"  {cls:<14} {size[0]:5.2f} x {size[1]:5.2f} x {size[2]:5.2f} m"

              f"   {len(m.vertices):6d} v  {len(m.triangles):6d} f")



    manifest = {

        "version": 1,

        "generated_by": "make_assets.py",

        "conventions": {

            "units": "m",

            "up": UP,

            "front": FRONT,

            "anchor": ANCHOR,

            "note": "origin at footprint centre, min(z) == 0, colours are "

                    "placeholders that synthesize repaints from measurement",

        },

        "assets": entries,

    }

    path = os.path.join(out_dir, "manifest.json")

    with open(path, "w") as f:

        json.dump(manifest, f, indent=2)

        f.write("\n")

    print(f"  -> {path}  ({len(entries)} classes)")

    return manifest





def verify(out_dir, manifest):

    """Re-load every asset and assert the contract stage [7] relies on.



    Cheap, and it catches the failures that would otherwise surface as a chair

    floating 40 cm above the floor after a two-hour pipeline run: a builder that

    forgot frame(), a mesh written without vertex colours, a manifest size that

    drifted from the geometry.

    """

    print("verify:")

    bad = 0

    for cls, e in manifest["assets"].items():

        for v in e["variants"]:

            p = os.path.join(out_dir, v["path"])

            m = o3d.io.read_triangle_mesh(p)

            lo, hi = m.get_min_bound(), m.get_max_bound()

            errs = []

            if not m.has_vertex_colors():

                errs.append("no vertex colours")

            if abs(lo[2]) > 1e-6:

                errs.append(f"base not at z=0 (min z {lo[2]:+.4f})")

            for ax, nm in ((0, "x"), (1, "y")):

                if abs(lo[ax] + hi[ax]) > 1e-6:

                    errs.append(f"{nm} not centred ({(lo[ax] + hi[ax]) / 2:+.4f})")

            if not np.allclose(hi - lo, v["size"], atol=1e-3):

                errs.append(f"size drift: mesh {(hi - lo).round(4)} "

                            f"vs manifest {v['size']}")

            if hi[2] <= 0.01:

                errs.append("degenerate height")

            # the path stage [7] actually uses: mesh -> points at map density

            pc = m.sample_points_uniformly(2000, use_triangle_normal=False)

            if len(pc.points) != 2000 or not pc.has_colors():

                errs.append("uniform sampling did not carry colours")

            if errs:

                bad += 1

                print(f"  FAIL {cls}: " + "; ".join(errs))

            else:

                print(f"  ok   {cls:<14} bbox {(hi - lo).round(3)}  "

                      f"base z {lo[2]:.1e}  sampled {len(pc.points)} pts")

    if bad:

        raise SystemExit(f"{bad} asset(s) failed verification")

    print("  all assets conform")





def main():

    ap = argparse.ArgumentParser(description=__doc__)

    ap.add_argument("--out", default=os.path.dirname(os.path.abspath(__file__)),

                    help="asset root (default: this script's directory)")

    ap.add_argument("--verify", action="store_true",

                    help="re-load and check the written assets")

    a = ap.parse_args()

    print(f"building placeholder assets -> {a.out}")

    man = build(a.out)

    if a.verify:

        verify(a.out, man)





if __name__ == "__main__":

    main()
