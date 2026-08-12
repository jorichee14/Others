#!/usr/bin/env python3
"""
02_pcd_to_mesh_sionna_v9.py

PLANE-FIRST faithful mesh for Sionna RT.

Why v8 looked awful, and what v9 changes
----------------------------------------
v8 ran Poisson over the whole cloud first and then tried to flatten the
resulting mesh with RANSAC. That inherits every Poisson pathology: wavy
wall sheets wherever planarization missed, blobby closures across scan
gaps, and a decimate->re-project cycle that leaves kinks at plane joints.
On a cloud that still contains dynamic ghosts (see stage 01's free-space
carving) Poisson is hopeless -- it wraps every ghost wisp in a surface.

v9 never lets Poisson touch the structure:

  STRUCTURE (exact planes, meshed directly)
    Planes are RANSAC-fitted DIRECTLY to the denoised point cloud
    (two-phase: large planes of any orientation, then a vertical-only
    sweep for short wall segments; coplanar fragments merged into one
    plane). Each plane is then meshed in 2D: inlier points are rasterized
    onto the plane at grid_cell resolution, the occupancy image is
    morphologically closed (seals scan gaps), small floating islands are
    dropped, a 1-cell dilation seals wall/wall and wall/floor joints, and
    the occupied cells are grid-triangulated. The result is EXACTLY flat
    by construction, has real holes (doorways/windows survive), accurate
    extent (grows only where points are), and a low, uniform triangle
    count -- ideal for ray tracing.

  OBJECTS (fine detail, Poisson only here)
    Fine-cloud points farther than obj_keep_dist from the structure mesh
    are object points (everything closer is wall/floor skin = scan noise,
    dropped). Objects are DBSCAN-clustered (crumbs dropped BEFORE
    meshing), Poisson-reconstructed at obj_depth with a tight crop,
    Taubin-smoothed, then film/crumb components are removed.

  NOISE
    Fine cloud: statistical + radius outlier removal (radius OR kills the
    low-density ghost wisps SOR misses), then iterated MLS projection
    (collapses multi-scan wall thickening onto the true surface). All
    structure, objects and colour derive from the denoised cloud.

  COLOUR
    All vertices sample colour from the fine cloud (k-neighbour weighted).
    Cosmetic only -- Sionna materials come from the split files.

Output: <out>.ply plus <out>_walls.ply / <out>_floor.ply /
<out>_ceiling.ply / <out>_objects.ply for per-part ITU materials in
Sionna (e.g. itu_concrete / itu_floorboard / itu_ceiling_board /
itu_metal). Units stay meters. Not watertight by design (openings
preserved) -- Sionna does not require watertightness.

Usage
-----
    python3 02_pcd_to_mesh_sionna_v9.py static_or_final.pcd [mesh_sionna.ply]

Feed it the CLEANED cloud from stage 01 (static.pcd / denoised.pcd /
map_final.pcd) -- garbage in, garbage out: no mesher can fix ghosts.
"""

import sys
import time

import numpy as np
import open3d as o3d

try:
    from scipy.spatial import cKDTree
    from scipy import ndimage
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False


# ---------------------------------------------------------------------- #
#  denoise + colour helpers
# ---------------------------------------------------------------------- #

def mls_denoise(P, k=16, iters=2, batch=1_500_000, max_step=0.15,
                verbose=None):
    """Iterated MLS projection denoise: each point moves onto the local
    least-squares plane of its k nearest neighbours (clamped step along the
    local normal). Collapses SLAM noise and multi-scan wall thickening onto
    the true surface before ANY meshing sees the points."""
    P = P.astype(np.float32, copy=True)
    for it in range(iters):
        tree = cKDTree(P)
        out = np.empty_like(P)
        moved = 0.0
        for s in range(0, len(P), batch):
            q = P[s:s + batch]
            _, idx = tree.query(q, k=k, workers=-1)
            nb = P[idx]
            c = nb.mean(axis=1)
            d = nb - c[:, None, :]
            cov = np.einsum('bki,bkj->bij', d, d)
            _, vec = np.linalg.eigh(cov)
            n = vec[..., 0]
            off = np.einsum('bi,bi->b', q - c, n)
            off = np.clip(off, -max_step, max_step)
            out[s:s + batch] = q - off[:, None] * n
            moved += float(np.abs(off).sum())
        P = out
        if verbose:
            verbose(f"    MLS iter {it + 1}/{iters}: mean move "
                    f"{1000 * moved / len(P):.1f} mm")
    return P


def smooth_point_colours(P, C, k=6, batch=1_500_000):
    """Average colours over k nearest neighbours (self excluded): removes
    RGB speckle while preserving gradients."""
    tree = cKDTree(P)
    out = np.empty_like(C)
    for s in range(0, len(P), batch):
        d, idx = tree.query(P[s:s + batch], k=k + 1, workers=-1)
        d, idx = d[:, 1:], idx[:, 1:]
        w = 1.0 / np.maximum(d, 1e-3)
        w /= w.sum(axis=1, keepdims=True)
        nbc = (C[idx] * w[..., None]).sum(axis=1)
        out[s:s + batch] = 0.4 * C[s:s + batch] + 0.6 * nbc
    return np.clip(out, 0.0, 1.0)


def sample_colours(verts, col_pts, col_rgb, k=4):
    """Inverse-distance-weighted colour from the k nearest fine-cloud points."""
    d, idx = cKDTree(col_pts).query(verts, k=k, workers=-1)
    if k == 1:
        return col_rgb[idx]
    w = 1.0 / np.maximum(d, 1e-6)
    w /= w.sum(axis=1, keepdims=True)
    return (col_rgb[idx] * w[..., None]).sum(axis=1)


# ---------------------------------------------------------------------- #
#  plane extraction on POINTS (unoriented normals -> no MST needed here)
# ---------------------------------------------------------------------- #

def _refit_plane(V, ref_n):
    c = V.mean(axis=0)
    n = np.linalg.svd(V - c, full_matrices=False)[2][-1]
    if np.dot(n, ref_n) < 0:
        n = -n
    return n, float(-np.dot(n, c))


def extract_planes(P, N, dist, ang_deg, min_inliers, max_planes,
                   iters=200, score_cap=40000, seed=0, nz_max=None):
    """Sequential point-normal RANSAC on the point cloud. Uses |N.n| so
    UNORIENTED normals suffice (no slow MST orientation for structure).
    Opposite faces of a real wall are separated by its thickness, which is
    > dist, so they still resolve to distinct planes.
    nz_max: if set, only near-vertical seeds are proposed (wall-only sweep)."""
    rng = np.random.default_rng(seed)
    cos_a = np.cos(np.deg2rad(ang_deg))
    pid = np.full(len(P), -1, dtype=np.int64)
    avail = np.ones(len(P), dtype=bool)
    planes = []
    for _ in range(max_planes):
        idx = np.flatnonzero(avail)
        if len(idx) < min_inliers:
            break
        Pr, Nr = P[idx], N[idx]
        cand = np.arange(len(idx))
        if nz_max is not None:
            cand = np.flatnonzero(np.abs(Nr[:, 2]) < nz_max)
            if len(cand) < min_inliers:
                break
        sub = (rng.choice(len(idx), score_cap, replace=False)
               if len(idx) > score_cap else np.arange(len(idx)))
        Ps, Ns = Pr[sub], Nr[sub]
        best_score, best = -1, None
        for _ in range(iters):
            i = int(cand[rng.integers(len(cand))])
            n = Nr[i]
            d0 = -float(np.dot(n, Pr[i]))
            sc = int(((np.abs(Ps @ n + d0) < dist)
                      & (np.abs(Ns @ n) > cos_a)).sum())
            if sc > best_score:
                best_score, best = sc, (n, d0)
        if best is None:
            break
        n, d0 = best
        m = (np.abs(Pr @ n + d0) < dist) & (np.abs(Nr @ n) > cos_a)
        if m.sum() < min_inliers:
            break
        n, d0 = _refit_plane(Pr[m], n)
        m = (np.abs(Pr @ n + d0) < dist) & (np.abs(Nr @ n) > cos_a)
        if m.sum() < min_inliers:
            break
        g = idx[np.flatnonzero(m)]
        pid[g] = len(planes)
        planes.append((n, d0))
        avail[g] = False
    return planes, pid


def extract_planes_two_phase(P, N, struct_voxel, dist, ang_deg,
                             big_area=8.0, wall_area=1.0, wall_nz=0.35,
                             max_planes=200):
    """Phase 1: large planes of any orientation. Phase 2: VERTICAL-ONLY
    planes at a much lower area threshold, so short wall segments (doorway-
    and corner-fragmented walls) still become flat structure while
    furniture-sized verticals stay detailed objects. Area thresholds (m^2)
    are converted to point counts via the struct-cloud lattice density."""
    density = 0.5 / (struct_voxel ** 2)        # ~pts per m^2 of surface
    mi1 = max(500, int(big_area * density))
    mi2 = max(150, int(wall_area * density))
    planes, pid = extract_planes(
        P, N, dist=dist, ang_deg=ang_deg, min_inliers=mi1,
        max_planes=60)
    rem = np.flatnonzero(pid < 0)
    if len(rem) > mi2 and len(planes) < max_planes:
        planes2, pid2 = extract_planes(
            P[rem], N[rem], dist=dist, ang_deg=ang_deg, min_inliers=mi2,
            max_planes=max_planes - len(planes), iters=150,
            score_cap=30000, seed=1, nz_max=wall_nz)
        hit = pid2 >= 0
        pid[rem[hit]] = pid2[hit] + len(planes)
        planes = planes + planes2
    return planes, pid


def merge_coplanar(P, planes, pid, ang_deg=5.0, off_tol=0.08):
    """Unify planes that are fragments of one physical surface (normals
    agree within ang_deg -- sign-agnostic -- AND centroids mutually within
    off_tol of each other's plane). One wall fragmented into patches becomes
    ONE plane, so its grid mesh has no kinks. Opposite faces of a wall are
    a wall-thickness apart and fail the offset test."""
    K = len(planes)
    if K < 2:
        return planes, pid, 0
    ns_ = np.array([p[0] for p in planes])
    ds_ = np.array([p[1] for p in planes])
    cents = np.zeros((K, 3))
    for k in range(K):
        m = pid == k
        if m.any():
            cents[k] = P[m].mean(axis=0)
    cos_t = np.cos(np.deg2rad(ang_deg))
    parent = list(range(K))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i in range(K):
        for j in range(i + 1, K):
            if abs(float(ns_[i] @ ns_[j])) < cos_t:
                continue
            if abs(float(cents[j] @ ns_[i] + ds_[i])) > off_tol:
                continue
            if abs(float(cents[i] @ ns_[j] + ds_[j])) > off_tol:
                continue
            parent[find(i)] = find(j)

    groups = {}
    for k in range(K):
        groups.setdefault(find(k), []).append(k)
    new_planes = []
    remap = np.full(K, -1, dtype=np.int64)
    for members in groups.values():
        gid = len(new_planes)
        for m_ in members:
            remap[m_] = gid
        sel = np.isin(pid, members)
        if sel.any():
            n, d = _refit_plane(P[sel], ns_[members[0]])
            new_planes.append((n, d))
        else:
            new_planes.append(planes[members[0]])
    pid_out = np.where(pid >= 0, remap[np.maximum(pid, 0)], -1)
    return new_planes, pid_out, K - len(new_planes)


# ---------------------------------------------------------------------- #
#  per-plane 2D grid meshing (exactly flat, holes preserved)
# ---------------------------------------------------------------------- #

def plane_basis(n):
    a = (np.array([0.0, 0.0, 1.0]) if abs(n[2]) < 0.9
         else np.array([1.0, 0.0, 0.0]))
    eu = np.cross(n, a)
    eu /= np.linalg.norm(eu)
    ev = np.cross(n, eu)
    return eu, ev


def mesh_plane(Q, n, d, cell=0.10, close_cells=3, min_region_m2=0.5,
               seal_dilate=1, max_fill_m2=2.0):
    """Grid-triangulate ONE plane from its inlier points.

    Rasterize the inliers on the plane at `cell` resolution, binary-close
    (seals scan gaps up to ~close_cells*cell wide; doorways and windows are
    larger and stay open), drop occupancy islands under min_region_m2, fill
    ENCLOSED holes up to max_fill_m2, dilate seal_dilate cells (closes the
    corner gap where two planes meet and the wall-base gap at the floor),
    then emit two triangles per occupied cell with shared corner vertices.
    Returns (V, F) or None.

    Two different hole mechanisms, because scan holes come in two shapes.
    Closing is width-based and handles hairline gaps between passes. It
    cannot reach an occlusion shadow -- the patch of floor behind a pillar or
    under a desk is often metres long and half a metre wide, far wider than
    any close_cells you could set without also sealing doorways. That is what
    max_fill_m2 is for: it fills background regions ENCLOSED by surface, up
    to an area cap, so a 2 m^2 shadow is floored while a courtyard or atrium
    of hundreds of m^2 is left open. Filling by area rather than by width is
    what lets those two cases be separated at all.

    A background region touching the grid border is outside the surface, not
    a hole in it, and is never filled -- so the plane never grows past its
    measured extent."""
    o = Q.mean(axis=0)
    o = o - (o @ n + d) * n                    # origin exactly on the plane
    eu, ev = plane_basis(n)
    s = (Q - o) @ eu
    t = (Q - o) @ ev
    smin = float(s.min())
    tmin = float(t.min())
    gi = np.floor((s - smin) / cell).astype(np.int64)
    gj = np.floor((t - tmin) / cell).astype(np.int64)
    pad = max(close_cells, seal_dilate, 1)
    W = int(gi.max()) + 1 + 2 * pad
    H = int(gj.max()) + 1 + 2 * pad
    smin -= pad * cell
    tmin -= pad * cell
    occ = np.zeros((W, H), dtype=bool)
    occ[gi + pad, gj + pad] = True
    if close_cells > 1:
        occ = ndimage.binary_closing(
            occ, structure=np.ones((close_cells, close_cells)))
    if min_region_m2 > 0:
        lab, nl = ndimage.label(occ)
        if nl > 1:
            sizes = ndimage.sum(occ, lab, np.arange(1, nl + 1))
            keep = np.flatnonzero(sizes * cell * cell >= min_region_m2) + 1
            occ = np.isin(lab, keep)
    if max_fill_m2 > 0:
        holes, nh = ndimage.label(~occ)
        if nh:
            # the pad ring guarantees the exterior touches the border, so any
            # component that does is outside rather than an interior hole
            border = np.unique(np.concatenate(
                [holes[0, :], holes[-1, :], holes[:, 0], holes[:, -1]]))
            sizes = ndimage.sum(~occ, holes, np.arange(1, nh + 1))
            fill = np.flatnonzero(sizes * cell * cell <= max_fill_m2) + 1
            fill = np.setdiff1d(fill, border)
            if fill.size:
                occ |= np.isin(holes, fill)
    if seal_dilate > 0:
        occ = ndimage.binary_dilation(occ, iterations=seal_dilate)
    if not occ.any():
        return None
    ci, cj = np.nonzero(occ)
    mark = np.zeros((W + 1, H + 1), dtype=bool)
    mark[ci, cj] = True
    mark[ci + 1, cj] = True
    mark[ci, cj + 1] = True
    mark[ci + 1, cj + 1] = True
    corner_id = np.full((W + 1, H + 1), -1, dtype=np.int64)
    ki, kj = np.nonzero(mark)
    corner_id[ki, kj] = np.arange(len(ki))
    V = (o[None, :]
         + np.outer(smin + ki * cell, eu)
         + np.outer(tmin + kj * cell, ev))
    a = corner_id[ci, cj]
    b = corner_id[ci + 1, cj]
    c2 = corner_id[ci + 1, cj + 1]
    d2 = corner_id[ci, cj + 1]
    F = np.concatenate([np.stack([a, b, c2], axis=1),
                        np.stack([a, c2, d2], axis=1)])
    return V, F


# ---------------------------------------------------------------------- #

def main():
    # ------------------------------------------------------------------ #
    #  TUNABLES
    # ------------------------------------------------------------------ #
    fine_voxel   = 0.03   # detail resolution: objects + colour source (m)
    sor_nb, sor_std = 20, 2.0
    ror_radius   = 0.12   # radius outlier removal: kills low-density ghost
    ror_min_pts  = 5      #   wisps that SOR misses (residual dynamics/noise)
    mls_k        = 24     # MLS denoise neighbourhood (12-24)
    mls_iters    = 2      # 2 = strong, 3 = maximum (may round crisp edges)
    colour_smooth_k = 6   # RGB speckle smoothing neighbourhood

    struct_voxel = 0.06   # plane-fitting resolution (m)
    plane_dist   = 0.08   # RANSAC inlier distance: > residual wall waviness
                          #   after MLS, < half the thinnest real wall
    plane_ang    = 35.0   # max normal deviation from the plane normal (deg)
    big_area     = 8.0    # phase 1: min plane area (m^2), any orientation
    wall_area    = 1.0    # phase 2: min VERTICAL plane area (m^2); lower ->
                          #   flatten shorter wall bits, but below ~1.0
                          #   furniture faces start becoming "walls"
    max_planes   = 200
    merge_ang    = 5.0    # merge coplanar fragments: normal tolerance (deg)
    merge_off    = 0.08   #   ...and mutual offset tolerance (m)

    grid_cell    = 0.10   # structure mesh resolution (m). Vertex colours
                          #   carry photo detail at this pitch; 0.10 is
                          #   plenty for RF, raise to 0.2 for a lighter mesh
    close_cells  = 3      # seal scan gaps up to ~close_cells*grid_cell wide
                          #   (0.3 m default); doorways/windows stay open
    min_region_m2 = 0.5   # drop floating occupancy islands on a plane
    max_fill_m2  = 2.0    # fill holes ENCLOSED by a surface up to this area
                          #   (m^2). Occlusion shadows -- floor behind a
                          #   pillar, under a desk -- are metres long and far
                          #   too wide for close_cells to reach. Raise to
                          #   floor bigger unscanned patches; a courtyard or
                          #   atrium is hundreds of m^2 and stays open either
                          #   way. 0 disables.
    seal_dilate  = 1      # grow each plane 1 cell to close corner/base seams

    obj_keep_dist = 0.15  # fine points closer than this to the structure
                          #   mesh are wall/floor skin (noise) -> dropped;
                          #   farther = real object. Raise to shave more off
                          #   the walls, lower to keep shallow detail
    obj_min_pts  = 300    # DBSCAN clusters smaller than this are crumbs
                          #   and never reach the mesher
    obj_eps_mult = 3.0    # DBSCAN eps = obj_eps_mult * fine_voxel
    obj_depth    = 10     # object Poisson depth (10 ~ 4 cm here; 11 heavier)
    obj_crop_mult = 2.5   # crop Poisson invention beyond this * fine_voxel
    obj_smooth   = 15     # Taubin iterations on the object mesh
    obj_min_area = 0.05   # drop object mesh components under this (m^2)
    obj_skin_dist = 0.25  # drop object components hugging the structure
                          #   (median vertex distance below this, nothing
                          #   protruding past 1.5x) -- residual wall films

    colour_k     = 4      # colour = weighted average of k nearest points
    split        = True   # write per-class files for Sionna materials
    class_rgb    = {'walls':   (0.82, 0.82, 0.86),
                    'floor':   (0.55, 0.51, 0.46),
                    'ceiling': (0.90, 0.90, 0.92),
                    'objects': (0.90, 0.50, 0.18)}
    # ------------------------------------------------------------------ #

    if not HAVE_SCIPY:
        sys.exit("ERROR: scipy required")

    infile = sys.argv[1] if len(sys.argv) > 1 else "map.pcd"
    outfile = sys.argv[2] if len(sys.argv) > 2 else "mesh_sionna.ply"
    t0 = time.time()

    def stage(msg):
        print(f"{msg}  [{time.time() - t0:.0f} s]")

    # --- load + denoise ------------------------------------------------ #
    raw = o3d.io.read_point_cloud(infile)
    if len(raw.points) == 0:
        sys.exit(f"ERROR: no points read from {infile}")
    print(f"Loaded {len(raw.points):,} points from {infile}")

    fine = raw.voxel_down_sample(voxel_size=fine_voxel)
    del raw
    fine, _ = fine.remove_statistical_outlier(nb_neighbors=sor_nb,
                                              std_ratio=sor_std)
    n_before = len(fine.points)
    fine, _ = fine.remove_radius_outlier(nb_points=ror_min_pts,
                                         radius=ror_radius)
    stage(f"fine cloud ({fine_voxel} m): {len(fine.points):,} points "
          f"(radius filter dropped {n_before - len(fine.points):,} "
          f"ghost-wisp points)")

    Pf = np.asarray(fine.points).astype(np.float32)
    Cf = (np.asarray(fine.colors).astype(np.float32)
          if fine.has_colors() else None)

    print("Denoising (iterated MLS projection)...")
    Pf = mls_denoise(Pf, k=mls_k, iters=mls_iters, verbose=stage)
    if Cf is not None:
        Cf = smooth_point_colours(Pf, Cf, k=colour_smooth_k)
    fine = o3d.geometry.PointCloud(
        o3d.utility.Vector3dVector(Pf.astype(np.float64)))
    stage("denoise done (structure + objects + colour all derive from "
          "the denoised cloud)")

    struct = fine.voxel_down_sample(voxel_size=struct_voxel)
    P = np.asarray(struct.points)
    struct.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(
        radius=3 * struct_voxel, max_nn=30))
    N = np.asarray(struct.normals)             # unoriented is fine here
    stage(f"structure cloud ({struct_voxel} m): {len(P):,} points, "
          f"normals estimated (no MST needed)")

    # =================================================================== #
    #  STAGE A - structure: planes fitted to POINTS, grid-meshed flat
    # =================================================================== #
    print("\n[A] Extracting planes (two-phase RANSAC on points)...")
    planes, pid = extract_planes_two_phase(
        P, N, struct_voxel, dist=plane_dist, ang_deg=plane_ang,
        big_area=big_area, wall_area=wall_area, max_planes=max_planes)
    planes, pid, n_merged = merge_coplanar(
        P, planes, pid, ang_deg=merge_ang, off_tol=merge_off)
    n_wall = sum(1 for n_, _ in planes if abs(n_[2]) < 0.35)
    n_horz = sum(1 for n_, _ in planes if abs(n_[2]) > 0.75)
    stage(f"    {len(planes)} planes after merging {n_merged} coplanar "
          f"fragments ({n_wall} vertical, {n_horz} horizontal), "
          f"{100.0 * (pid >= 0).mean():.0f}% of struct points on planes")
    if n_wall == 0:
        print("    WARNING: no vertical planes found - raise plane_dist")

    # classify each plane: wall / floor / ceiling (slanted -> walls bucket)
    z_lo = float(np.percentile(P[:, 2], 2.0))
    z_hi = float(np.percentile(P[:, 2], 98.0))
    z_mid = 0.5 * (z_lo + z_hi)
    plane_class = []
    for k, (n_, d_) in enumerate(planes):
        if abs(n_[2]) > 0.75:
            zk = float(np.median(P[pid == k][:, 2])) if (pid == k).any() \
                else -d_ * n_[2]
            plane_class.append('floor' if zk < z_mid else 'ceiling')
        else:
            plane_class.append('walls')

    print("[A] Grid-meshing each plane (exactly flat by construction)...")
    Vs_all, Fs_all, tri_class = [], [], []
    voff = 0
    for k, (n_, d_) in enumerate(planes):
        Q = P[pid == k]
        if len(Q) == 0:
            continue
        out = mesh_plane(Q, n_, d_, cell=grid_cell, close_cells=close_cells,
                         min_region_m2=min_region_m2,
                         seal_dilate=seal_dilate, max_fill_m2=max_fill_m2)
        if out is None:
            continue
        Vk, Fk = out
        Vs_all.append(Vk)
        Fs_all.append(Fk + voff)
        tri_class.append(np.full(len(Fk), plane_class[k], dtype=object))
        voff += len(Vk)
    if not Vs_all:
        sys.exit("ERROR: no planes could be meshed - check plane_dist / "
                 "input cloud")
    struct_mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(np.vstack(Vs_all)),
        o3d.utility.Vector3iVector(np.vstack(Fs_all)))
    struct_tri_class = np.concatenate(tri_class)
    area = struct_mesh.get_surface_area()
    stage(f"[A] flat structure: {len(struct_mesh.vertices):,} verts, "
          f"{len(struct_mesh.triangles):,} tris, {area:,.0f} m^2")

    # =================================================================== #
    #  STAGE B - objects: fine points off the structure -> detailed mesh
    # =================================================================== #
    print("\n[B] Selecting object points from the fine cloud...")
    n_samp = max(20000, int(area / (0.05 ** 2)))
    samp = struct_mesh.sample_points_uniformly(number_of_points=n_samp)
    struct_tree = cKDTree(np.asarray(samp.points))
    d, _ = struct_tree.query(Pf, k=1, workers=-1)
    obj_sel = d > obj_keep_dist
    stage(f"    {int(obj_sel.sum()):,} object points "
          f"({100.0 * obj_sel.mean():.0f}% of fine cloud); the rest is "
          f"structure skin within {obj_keep_dist} m of the flat mesh")
    if obj_sel.mean() > 0.30:
        print("    NOTE: object fraction above 30% usually means wall area "
              "is still leaking into\n    the object stage -> lower "
              "wall_area to planarize shorter wall bits, or check that\n"
              "    stage 01 dynamic removal (with carving) ran on this cloud")

    obj_mesh = None
    if obj_sel.sum() > 5000:
        obj = o3d.geometry.PointCloud(
            o3d.utility.Vector3dVector(Pf[obj_sel].astype(np.float64)))
        # crumbs never reach the mesher: cluster first, keep real objects
        labels = np.asarray(obj.cluster_dbscan(
            eps=obj_eps_mult * fine_voxel, min_points=10))
        keep_lab = [l for l in range(labels.max() + 1)
                    if (labels == l).sum() >= obj_min_pts]
        keep_m = np.isin(labels, keep_lab)
        obj = obj.select_by_index(np.flatnonzero(keep_m))
        stage(f"[B] {len(keep_lab)} object clusters kept "
              f"({int(keep_m.sum()):,} pts); "
              f"{int((~keep_m).sum()):,} crumb points dropped pre-mesh")

        if len(obj.points) > 5000:
            obj.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(
                radius=3 * fine_voxel, max_nn=30))
            obj.orient_normals_consistent_tangent_plane(30)
            stage("[B] object normals oriented (MST on objects only)")

            print(f"[B] Object Poisson (depth={obj_depth})...")
            obj_mesh, _ = \
                o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                    obj, depth=obj_depth, linear_fit=True)
            dv, _ = cKDTree(np.asarray(obj.points)).query(
                np.asarray(obj_mesh.vertices), k=1, workers=-1)
            obj_mesh.remove_vertices_by_mask(dv > obj_crop_mult * fine_voxel)
            obj_mesh.remove_unreferenced_vertices()
            if obj_smooth > 0:
                obj_mesh = obj_mesh.filter_smooth_taubin(
                    number_of_iterations=obj_smooth)
            obj_mesh.remove_duplicated_vertices()
            obj_mesh.remove_duplicated_triangles()
            obj_mesh.remove_unreferenced_vertices()
            if obj_min_area > 0 and len(obj_mesh.triangles):
                cl, _, careas = obj_mesh.cluster_connected_triangles()
                cl = np.asarray(cl)
                careas = np.asarray(careas)
                obj_mesh.remove_triangles_by_mask(careas[cl] < obj_min_area)
                obj_mesh.remove_unreferenced_vertices()
            if obj_skin_dist > 0 and len(obj_mesh.triangles):
                cl2, _, _ = obj_mesh.cluster_connected_triangles()
                cl2 = np.asarray(cl2)
                dv, _ = struct_tree.query(np.asarray(obj_mesh.vertices),
                                          k=1, workers=-1)
                tris_o = np.asarray(obj_mesh.triangles)
                ncomp = int(cl2.max()) + 1 if len(cl2) else 0
                skin = np.zeros(ncomp, dtype=bool)
                for c in range(ncomp):
                    vs = np.unique(tris_o[cl2 == c])
                    if len(vs) and (np.median(dv[vs]) < obj_skin_dist
                                    and np.percentile(dv[vs], 90)
                                    < 1.5 * obj_skin_dist):
                        skin[c] = True
                if skin.any():
                    obj_mesh.remove_triangles_by_mask(skin[cl2])
                    obj_mesh.remove_unreferenced_vertices()
                    stage(f"[B] removed {int(skin.sum()):,} wall-skin films")
            if len(obj_mesh.triangles) == 0:
                obj_mesh = None
            else:
                stage(f"[B] object mesh: {len(obj_mesh.vertices):,} verts, "
                      f"{len(obj_mesh.triangles):,} tris")
        else:
            print("[B] too few clustered object points - skipping")
    else:
        print("[B] too few object points - skipping object stage")

    # =================================================================== #
    #  combine + colour + write
    # =================================================================== #
    combined = o3d.geometry.TriangleMesh(struct_mesh)
    if obj_mesh is not None:
        combined += obj_mesh
    combined.compute_vertex_normals()

    if Cf is not None:
        print(f"\nColour: {colour_k}-neighbour weighted sampling from the "
              f"{fine_voxel} m cloud...")
        cols = sample_colours(np.asarray(combined.vertices), Pf, Cf,
                              colour_k)
        combined.vertex_colors = o3d.utility.Vector3dVector(
            np.clip(cols, 0.0, 1.0))
        stage("    colour done")
    else:
        n_struct_tri = len(struct_mesh.triangles)
        vc = np.tile(class_rgb['objects'],
                     (len(combined.vertices), 1))
        combined.vertex_colors = o3d.utility.Vector3dVector(vc)
        print("Source cloud has no colour -> class colours applied.")

    o3d.io.write_triangle_mesh(outfile, combined)
    print(f"\nWrote {outfile}")
    print(f"  vertices : {len(combined.vertices):,}")
    print(f"  triangles: {len(combined.triangles):,}")

    if split:
        stem = outfile.rsplit('.', 1)[0]
        tris = np.asarray(struct_mesh.triangles)
        parts = {}
        for name in ('walls', 'floor', 'ceiling'):
            sel = np.flatnonzero(struct_tri_class == name)
            if len(sel) == 0:
                continue
            part = o3d.geometry.TriangleMesh(struct_mesh)
            part.triangles = o3d.utility.Vector3iVector(tris[sel])
            part.remove_unreferenced_vertices()
            parts[name] = part
        if obj_mesh is not None and len(obj_mesh.triangles):
            parts['objects'] = obj_mesh
        for name, part in parts.items():
            part.compute_vertex_normals()
            pv = np.asarray(part.vertices)
            if Cf is not None:
                pc = sample_colours(pv, Pf, Cf, colour_k)
            else:
                pc = np.tile(class_rgb[name], (len(pv), 1))
            part.vertex_colors = o3d.utility.Vector3dVector(
                np.clip(pc, 0.0, 1.0))
            path = f"{stem}_{name}.ply"
            o3d.io.write_triangle_mesh(path, part)
            print(f"  {path}: {len(part.triangles):,} tris")

    print(f"\nDone in {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
