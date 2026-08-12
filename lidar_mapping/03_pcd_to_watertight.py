#!/usr/bin/env python3
"""
03_pcd_to_watertight.py -- point cloud -> genuinely WATERTIGHT mesh.

    python3 03_pcd_to_watertight.py input.pcd [output.ply]

WHAT "WATERTIGHT" COSTS YOU
---------------------------
A building scanned from the inside is not a closed surface. Doorways lead
somewhere the scanner never went, windows open onto unscanned space, ceilings
and courtyards are missing entirely. Closing that is not a repair -- it is an
INVENTION, and the mesh will contain surface that no sensor ever measured.

So be sure you need it. Sionna RT does NOT: it ray-traces open geometry
happily, and 02_pcd_to_mesh_sionna_v9.py deliberately keeps openings because
a doorway that leaks RF is physically correct. Watertight is for volume
computation, boolean/CSG operations, 3D printing, and CFD or FEM meshing --
jobs where "inside" has to be well defined.

HOW IT GETS THERE
-----------------
Poisson reconstruction is the engine, because its output is a closed manifold
BY CONSTRUCTION -- it solves for an indicator function and extracts one
isosurface, so there is nothing to seal. Every hole you see in a Poisson mesh
was made afterwards by cropping. That gives two honest modes:

  trim_density = 0   faithful to Poisson: closed already, minimal repair, but
                     it bridges scan gaps with invented bulges
  trim_density > 0   cut the least-supported surface away, then explicitly
                     re-close: tighter to the data, but the repair stage has
                     to reconstruct what was cut

Repair, in the order that matters:
  1. drop degenerate + duplicated triangles (they defeat every later test)
  2. keep the largest connected component (islands can never be part of one
     closed surface, and hole-fillers will happily bridge them into knots)
  3. remove non-manifold edges, then fill the remaining boundary loops
  4. verify, and say plainly whether it worked

Hole filling uses pymeshfix when installed -- it is the most reliable
watertight repair available -- and falls back to Open3D's tensor
`fill_holes` otherwise.

VERIFICATION
------------
The script does not claim success on the strength of having run. It checks
is_watertight, edge- and vertex-manifoldness, orientability, self-
intersection and the Euler characteristic, then prints the enclosed volume,
which is only meaningful for a genuinely closed surface. If it cannot close
the mesh it says so and tells you which knob to move, rather than writing a
file that merely looks finished.
"""

import sys
import time

import numpy as np
import open3d as o3d


def stage(t0, msg):
    print(f"{msg}  [{time.time() - t0:.0f} s]")


def report(mesh, label):
    """The full manifold picture. Watertight alone is not enough: a mesh can
    be closed and still be non-manifold or self-intersecting, which breaks
    booleans and slicers even though every edge has two faces."""
    wt = mesh.is_watertight()
    em = mesh.is_edge_manifold()
    vm = mesh.is_vertex_manifold()
    orient = mesh.is_orientable()
    V = len(mesh.vertices)
    F = len(mesh.triangles)
    E = int(np.unique(np.sort(np.asarray(mesh.triangles)[:, [0, 1, 1, 2, 2, 0]]
                              .reshape(-1, 2), axis=1), axis=0).shape[0])
    euler = V - E + F
    print(f"  {label}:")
    print(f"    vertices {V:,}  triangles {F:,}  edges {E:,}")
    print(f"    watertight {wt} | edge-manifold {em} | vertex-manifold {vm} "
          f"| orientable {orient}")
    # chi = 2 for a sphere-like closed surface; 2-2g for genus g. A doorway
    # that got sealed into a tunnel shows up here as a lower chi, which is a
    # useful sanity signal even when watertight says True.
    print(f"    Euler characteristic {euler} "
          f"(2 = sphere-like, 2-2g for genus g)")
    if wt:
        try:
            print(f"    enclosed volume {mesh.get_volume():.2f} m^3")
        except Exception:
            print("    enclosed volume unavailable (inconsistent orientation)")
    return wt


def largest_component(mesh):
    """Keep only the biggest connected piece.

    Islands cannot be part of a single closed surface, and hole-fillers
    bridge them into non-manifold knots if left in place."""
    idx, counts, _ = mesh.cluster_connected_triangles()
    idx = np.asarray(idx)
    counts = np.asarray(counts)
    if counts.size <= 1:
        return mesh, 0
    keep = int(counts.argmax())
    mesh.remove_triangles_by_mask(idx != keep)
    mesh.remove_unreferenced_vertices()
    return mesh, int(counts.size - 1)


def fill_holes(mesh, hole_size, use_pymeshfix=True):
    """Close remaining boundary loops. Returns (mesh, method_used)."""
    if use_pymeshfix:
        try:
            from pymeshfix import MeshFix
            mf = MeshFix(np.asarray(mesh.vertices),
                         np.asarray(mesh.triangles).astype(np.int32))
            # joincomp bridges separate shells; remove_smallest keeps the
            # result single-shell, which is what watertight requires
            mf.repair(verbose=False, joincomp=True,
                      remove_smallest_components=True)
            out = o3d.geometry.TriangleMesh(
                o3d.utility.Vector3dVector(np.asarray(mf.v)),
                o3d.utility.Vector3iVector(np.asarray(mf.f)))
            out.compute_vertex_normals()
            return out, "pymeshfix"
        except ImportError:
            pass
        except Exception as e:
            print(f"    pymeshfix failed ({type(e).__name__}), trying Open3D")
    try:
        t = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
        t = t.fill_holes(hole_size=float(hole_size))
        out = t.to_legacy()
        out.compute_vertex_normals()
        return out, "open3d.fill_holes"
    except Exception as e:
        print(f"    fill_holes unavailable ({type(e).__name__})")
        return mesh, "none"


def main():
    # ------------------------------------------------------------------ #
    #  TUNABLES
    # ------------------------------------------------------------------ #
    voxel         = 0.03   # working resolution (m). Raise for speed/memory
    sor_nb, sor_std = 20, 2.0
    normal_knn    = 30     # neighbours for normal estimation
    orient_knn    = 50     # MST neighbours for consistent orientation. This
                           #   is the slow step; Poisson is worthless without
                           #   it because it needs INWARD/OUTWARD to be
                           #   globally consistent
    poisson_depth = 10     # 10 ~ 4 cm at building scale, 11 ~ 2 cm (4x cost)
    poisson_scale = 1.1    # cube expansion; >1 keeps the closing surface off
                           #   the data
    trim_density  = 0.0    # 0 = keep Poisson's closed output as-is (SAFEST
                           #   for watertightness). 0.01-0.05 crops the
                           #   least-supported surface -- tighter to the data,
                           #   but then holes must be re-closed
    hole_size     = 1e6    # max boundary loop to fill (m^2-ish, generous)
    use_pymeshfix = True   # pip install pymeshfix -- best repair available
    target_tris   = 0      # 0 = no decimation. Decimation can reintroduce
                           #   non-manifold edges, so it is verified after
    smooth_iters  = 0      # Taubin iterations (volume-preserving); 0 = off
    keep_colors   = True   # sample vertex colours from the cloud
    # ------------------------------------------------------------------ #

    infile = sys.argv[1] if len(sys.argv) > 1 else "map.pcd"
    outfile = sys.argv[2] if len(sys.argv) > 2 else "watertight.ply"
    t0 = time.time()

    pcd = o3d.io.read_point_cloud(infile)
    if len(pcd.points) == 0:
        sys.exit(f"ERROR: no points read from {infile}")
    print(f"Loaded {len(pcd.points):,} points from {infile}")

    if voxel > 0:
        pcd = pcd.voxel_down_sample(voxel)
    pcd, _ = pcd.remove_statistical_outlier(sor_nb, sor_std)
    stage(t0, f"[1] cloud: {len(pcd.points):,} points at {voxel} m")
    src = np.asarray(pcd.points)
    src_col = np.asarray(pcd.colors) if pcd.has_colors() else None

    print("[2] normals + consistent orientation (the slow step)...")
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(
        radius=3 * max(voxel, 1e-3), max_nn=normal_knn))
    pcd.orient_normals_consistent_tangent_plane(orient_knn)
    stage(t0, "    oriented")

    print(f"[3] Poisson (depth={poisson_depth}) -- closed by construction...")
    mesh, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=poisson_depth, scale=poisson_scale, linear_fit=True)
    dens = np.asarray(dens)
    stage(t0, f"    {len(mesh.vertices):,} verts, {len(mesh.triangles):,} tris")
    report(mesh, "raw Poisson")

    if trim_density > 0:
        thr = np.quantile(dens, trim_density)
        mesh.remove_vertices_by_mask(dens < thr)
        mesh.remove_unreferenced_vertices()
        stage(t0, f"[4] trimmed below density quantile {trim_density} "
                  f"-> {len(mesh.vertices):,} verts (holes now expected)")

    print("[5] cleanup...")
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_unreferenced_vertices()
    mesh, dropped = largest_component(mesh)
    if dropped:
        print(f"    dropped {dropped} disconnected component(s)")
    mesh.remove_non_manifold_edges()
    mesh.remove_unreferenced_vertices()
    stage(t0, f"    {len(mesh.vertices):,} verts, {len(mesh.triangles):,} tris")

    if not mesh.is_watertight():
        print("[6] not closed yet -> filling holes...")
        mesh, how = fill_holes(mesh, hole_size, use_pymeshfix)
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_unreferenced_vertices()
        mesh, _ = largest_component(mesh)
        stage(t0, f"    repaired via {how}")
    else:
        print("[6] already closed, no hole filling needed")

    if smooth_iters > 0:
        mesh = mesh.filter_smooth_taubin(number_of_iterations=smooth_iters)
        stage(t0, f"[7] Taubin x{smooth_iters}")

    if target_tris > 0 and len(mesh.triangles) > target_tris:
        before = mesh.is_watertight()
        mesh = mesh.simplify_quadric_decimation(target_tris)
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_unreferenced_vertices()
        stage(t0, f"[8] decimated to {len(mesh.triangles):,} tris")
        if before and not mesh.is_watertight():
            # decimation collapses edges and can punch through thin sheets
            print("    WARNING decimation broke watertightness -> re-filling")
            mesh, how = fill_holes(mesh, hole_size, use_pymeshfix)
            mesh, _ = largest_component(mesh)

    mesh.compute_vertex_normals()
    mesh.orient_triangles()

    if keep_colors and src_col is not None and len(mesh.vertices):
        from scipy.spatial import cKDTree
        _, nn = cKDTree(src).query(np.asarray(mesh.vertices), workers=-1)
        mesh.vertex_colors = o3d.utility.Vector3dVector(src_col[nn])
        stage(t0, "[9] vertex colours sampled from the cloud")

    print()
    ok = report(mesh, "FINAL")
    if mesh.is_self_intersecting():
        print("    NOTE self-intersecting: closed, but booleans/slicers may "
              "still object.\n         pip install pymeshfix for a cleaner "
              "repair")

    o3d.io.write_triangle_mesh(outfile, mesh)
    print(f"\nWrote {outfile}")
    if ok:
        print("WATERTIGHT: every edge borders exactly two triangles and the "
              "volume is well defined.")
    else:
        print("NOT WATERTIGHT. In order of effect:\n"
              "  1. set trim_density = 0 (cropping is what opens the mesh)\n"
              "  2. pip install pymeshfix (far better repair than fill_holes)\n"
              "  3. raise poisson_depth if detail was lost, lower it if the\n"
              "     surface fragmented\n"
              "  4. check the input is ONE connected scan -- a cloud in two\n"
              "     separated pieces cannot become one closed surface")
    print(f"Done in {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
