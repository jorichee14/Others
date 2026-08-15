#!/usr/bin/env python3
"""
Asset fitting and surface repair for stage [7] synthesize.

The asset is a SHAPE PRIOR, not the output. Stage [7] keeps the measured points
and adds an asset that has been bent toward them:

    fit      yaw from the instance's horizontal PCA, anisotropic scale against
             the asset's own bbox (clamped), then ICP with the rotation
             projected back onto the gravity axis so it stays a yaw
    snap     base translated onto the classified support plane
    repaint  every asset point takes the median colour of the measured points
             around it; unobserved faces keep the baked colour, tinted toward
             the instance median
    gate     if the fit does not actually explain the measurement, NO asset is
             emitted and the raw points stand alone

That last step is the important one. A bad fit must degrade to "unchanged",
never to "two overlapping chairs" -- which is the failure mode of keeping the
measured points and adding an asset regardless.

Removal is the other half: deleting an object leaves a hole in the surface it
was attached to, because the LiDAR never saw behind it. fill_plane_hole()
re-samples the classified plane across exactly the missing footprint, at the
local density, coloured from the surviving ring.
"""

import json
import os
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

# Interface level, checked by 01_build_map.py at import. These files are
# versioned TOGETHER: the stage script calls into them by keyword, so a
# hand-copied mismatch surfaces as a TypeError deep inside a run rather
# than as a message you can act on. Bump this whenever a signature the
# stage script uses changes.
API = 3


def rotz(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def yaw_of(R):
    return float(np.arctan2(R[1, 0], R[0, 0]))


def _inv_rigid(T):
    Ti = np.eye(4)
    Ti[:3, :3] = T[:3, :3].T
    Ti[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return Ti


# =============================================================================
# LIBRARY
# =============================================================================
class AssetLibrary:
    """manifest.json -> meshes, sampled to points on demand and cached."""

    def __init__(self, root):
        self.root = root
        path = os.path.join(root, "manifest.json")
        if not os.path.exists(path):
            raise SystemExit(f"no asset manifest at {path} -- run "
                             f"assets/make_assets.py to generate the "
                             f"placeholder library")
        self.man = json.load(open(path))
        self.conv = self.man.get("conventions", {})
        self.assets = self.man.get("assets", {})
        self._cache = {}
        up = self.conv.get("up", "+z")
        if up != "+z":
            raise SystemExit(f"asset manifest declares up={up}; this pipeline "
                             f"assumes +z (GLIM's gravity axis)")

    def has(self, cls_name):
        return cls_name in self.assets

    def describe(self):
        return (f"{len(self.assets)} classes: "
                + ", ".join(sorted(self.assets)))

    def pick(self, cls_name, target_size=None):
        """Best variant for a measured bbox, by SHAPE not size.

        Size is what the fit is about to solve for, so matching on it would be
        circular -- and would prefer a variant merely because the instance was
        clipped. Aspect ratio is scale-free and is the thing scaling cannot
        fix: an armchair and a three-seat sofa are both "couch", and no
        clamped anisotropic scale turns one into the other.
        """
        e = self.assets.get(cls_name)
        if not e:
            return None
        vs = e["variants"]
        if len(vs) > 1 and target_size is not None:
            t = np.asarray(target_size, float)
            t = t / max(t.max(), 1e-9)
            def err(v):
                a = np.asarray(v["size"], float)
                return float(np.linalg.norm(a / max(a.max(), 1e-9) - t))
            vs = sorted(vs, key=err)
        v = vs[0]
        return {"variant": v, "support": e.get("support", "floor"),
                "yaw_symmetry": int(e.get("yaw_symmetry", 1)),
                "size": np.asarray(v["size"], float),
                "path": os.path.join(self.root, v["path"])}

    def points(self, meta, density):
        """Asset sampled to (points, colours) at `density` points per m^2."""
        key = (meta["path"], round(float(density), 3))
        if key in self._cache:
            return self._cache[key]
        mesh = o3d.io.read_triangle_mesh(meta["path"])
        if len(mesh.triangles) == 0:
            raise SystemExit(f"asset {meta['path']} has no faces")
        mesh.compute_vertex_normals()
        area = float(mesh.get_surface_area())
        n = int(np.clip(area * float(density), 500, 400_000))
        pc = mesh.sample_points_uniformly(n, use_triangle_normal=False)
        p = np.asarray(pc.points)
        c = (np.asarray(pc.colors) if pc.has_colors()
             else np.full((len(p), 3), 0.5))
        self._cache[key] = (p, c)
        return p, c


# =============================================================================
# FIT
# =============================================================================
def principal_yaw(pts):
    """Dominant horizontal direction of a point set.

    Objects stand upright, so yaw is the only rotational freedom worth
    estimating -- and estimating roll/pitch from a partial, one-sided scan
    reliably produces a tilted chair.
    """
    xy = pts[:, :2] - pts[:, :2].mean(0)
    if len(xy) < 3:
        return 0.0
    w, V = np.linalg.eigh(xy.T @ xy / len(xy))
    v = V[:, int(np.argmax(w))]
    return float(np.arctan2(v[1], v[0]))


def robust_extent(pts, lo=2.0, hi=98.0):
    """Percentile bbox: min/max would hand the scale to a single stray point."""
    a = np.percentile(pts, lo, axis=0)
    b = np.percentile(pts, hi, axis=0)
    return np.maximum(b - a, 1e-3)


def _place(asset_pts, yaw, scale, centre_xy, base_z):
    p = (asset_pts * scale) @ rotz(yaw).T
    p[:, 0] += centre_xy[0]
    p[:, 1] += centre_xy[1]
    p[:, 2] += base_z
    return p


def fit_asset(asset_pts, asset_size, inst_pts, yaw_symmetry, base_z,
              scale_range=(0.75, 1.35), icp=True, icp_dist=0.10,
              tol=0.08):
    """Fit an asset to a measured instance.

    Returns a dict with the placement and its quality, or None if the fit does
    not explain the measurement well enough to be worth emitting.

    The quality metric is COVERAGE OF THE MEASUREMENT -- what fraction of the
    scanned points have asset surface near them -- not ICP's own fitness. ICP
    scores the source, and the source is an asset whose back and underside were
    never scanned, so its fitness is capped well below 1 by construction and
    says nothing about whether the fit is right.
    """
    inst_pts = np.asarray(inst_pts, float)
    if len(inst_pts) < 10:
        return None
    centre_xy = np.median(inst_pts[:, :2], axis=0)
    y0 = principal_yaw(inst_pts)

    if yaw_symmetry == 0:
        cands = [0.0]                       # continuous: any yaw is the same
    elif yaw_symmetry >= 2:
        cands = [y0, y0 + np.pi / 2]
    else:
        # PCA gives an unsigned axis, and either asset axis may align to it
        cands = [y0, y0 + np.pi / 2, y0 + np.pi, y0 + 3 * np.pi / 2]

    best = None
    for yaw in cands:
        q = (inst_pts - np.append(centre_xy, base_z)) @ rotz(yaw)
        ext = robust_extent(q)
        s = np.clip(ext / np.maximum(asset_size, 1e-6),
                    scale_range[0], scale_range[1])
        p = _place(asset_pts, yaw, s, centre_xy, base_z)
        # score MEASUREMENT -> ASSET, never the reverse. A scan is one-sided:
        # the back of a chair against a wall was never measured, so asset->
        # measurement charges every candidate for surface the sensor could not
        # have seen, and that penalty swamps the orientation signal it is
        # supposed to be measuring. The right question is "does this pose
        # explain what I DID see", which is also the gate applied at the end.
        d, _ = cKDTree(p).query(inst_pts, k=1)
        score = float(np.median(d))
        if best is None or score < best["score"]:
            best = {"yaw": float(yaw), "scale": s, "pts": p, "score": score}

    if best is None:
        return None
    yaw, scale = best["yaw"], best["scale"]
    placed = best["pts"]

    if icp and len(inst_pts) >= 50:
        # MEASUREMENT is the source, asset the target -- the opposite of the
        # obvious arrangement, and for the same reason the candidate score is
        # one-way. ICP pairs every SOURCE point with its nearest target, so an
        # asset source drags its never-observed back and underside toward
        # whatever happens to be closest and skews the whole alignment. With
        # the measurement as source, every correspondence is a real observation
        # pulling toward the asset, and unseen asset surface exerts no pull.
        src = o3d.geometry.PointCloud()
        src.points = o3d.utility.Vector3dVector(inst_pts)
        tgt = o3d.geometry.PointCloud()
        tgt.points = o3d.utility.Vector3dVector(placed)
        reg = o3d.pipelines.registration.registration_icp(
            src, tgt, icp_dist, np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=40))
        # ...which means the result maps measurement onto asset; the asset has
        # to move by its INVERSE
        T = _inv_rigid(np.asarray(reg.transformation))
        # Project onto yaw + translation. Unconstrained ICP on a one-sided scan
        # happily tips a chair 8 degrees to close the gap on the side it can
        # see, and a tilted chair is a worse answer than a slightly misaligned
        # upright one.
        #
        # The projection has to be applied as a TRANSFORM, not as a yaw and a
        # translation read off separately and re-applied around the asset's own
        # centre. ICP's rotation is about the WORLD origin, so for an object
        # standing 2.5 m away the rotation carries most of the displacement and
        # T[:3, 3] alone is meaningless -- that mistake puts the asset metres
        # from the points it was fitted to, while still reporting a correct yaw.
        dyaw = yaw_of(T[:3, :3])
        Rz = rotz(dyaw)
        shift = T[:3, 3]
        placed = placed @ Rz.T + shift
        centre = Rz @ np.append(centre_xy, base_z) + shift
        yaw = yaw + dyaw
        # re-seat on the support: ICP is free to slide the asset off the floor,
        # and the support plane is a harder constraint than the correspondence
        placed[:, 2] += base_z - placed[:, 2].min()
        centre_xy = centre[:2]

    d_meas, _ = cKDTree(placed).query(inst_pts, k=1)
    coverage = float((d_meas < tol).mean())
    rmse = float(np.sqrt(np.mean(np.minimum(d_meas, 1.0) ** 2)))
    return {"pts": placed, "yaw": float(yaw), "scale": scale,
            "centre": np.append(centre_xy, base_z),
            "coverage": coverage, "rmse": rmse}


def pose_matrix(yaw, scale, centre):
    """4x4 the fit implies, for scene.json (stage 03 wants meshes, not points)."""
    T = np.eye(4)
    T[:3, :3] = rotz(yaw) @ np.diag(scale)
    T[:3, 3] = centre
    return T


# =============================================================================
# REPAINT
# =============================================================================
def repaint(asset_pts, asset_col, inst_pts, inst_col, radius=0.08, k=6,
            tint=0.65):
    """Take the measured colours wherever there are any.

    An asset point with measured neighbours becomes their median -- a beige
    chair comes out beige rather than placeholder blue. A point with none (the
    back of a couch against a wall, the underside of a table) keeps its baked
    colour blended toward the instance's median, so unobserved faces stay
    plausible and consistent with the observed ones instead of reverting to a
    palette colour that belongs to no real object in the room.
    """
    out = np.array(asset_col, float, copy=True)
    if len(inst_pts) == 0:
        return out
    med = np.median(inst_col, axis=0) if len(inst_col) else np.full(3, 0.5)
    kk = min(int(k), len(inst_pts))
    d, i = cKDTree(inst_pts).query(asset_pts, k=kk,
                                   distance_upper_bound=float(radius))
    if kk == 1:
        d = d[:, None]; i = i[:, None]
    ok = np.isfinite(d)
    n_ok = ok.sum(1)
    seen = n_ok > 0
    if seen.any():
        safe = np.where(ok, i, 0)
        cols = inst_col[safe]                       # (N, k, 3)
        cols[~ok] = np.nan
        out[seen] = np.nanmedian(cols[seen], axis=1)
    if (~seen).any():
        out[~seen] = (1.0 - tint) * out[~seen] + tint * med
    return np.clip(out, 0.0, 1.0)


# =============================================================================
# SURFACE REPAIR
# =============================================================================
def plane_basis(n):
    n = np.asarray(n, float) / np.linalg.norm(n)
    a = np.array([0.0, 0.0, 1.0]) if abs(n[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(n, a); u /= np.linalg.norm(u)
    return u, np.cross(n, u)


def fill_plane_hole(plane, removed_pts, surface_pts, surface_col,
                    spacing=0.02, margin=0.03):
    """Re-sample a plane across the footprint an object used to occupy.

    Deleting a wall-mounted TV leaves a TV-shaped void, because the LiDAR never
    measured the wall behind it. The void is filled by rasterising the plane
    over exactly the missing region: cells near the removed footprint, and NOT
    near any surviving surface point. That second test is what keeps the patch
    to the actual hole -- without it the fill spills across the bbox and paves
    over wall the sensor did see, at a synthetic density that reads as a seam.

    Colours come from the surrounding surviving ring, so the patch matches the
    wall it is set into rather than the object that was removed.
    """
    if len(removed_pts) == 0:
        return np.empty((0, 3)), np.empty((0, 3))
    n = np.asarray(plane["n"], float)
    d = float(plane["d"])
    u, v = plane_basis(n)
    origin = -d * n

    def to2d(p):
        q = p - origin
        return np.stack([q @ u, q @ v], 1)

    r2 = to2d(removed_pts)
    lo = r2.min(0) - margin
    hi = r2.max(0) + margin
    if np.any(hi - lo > 20.0):                      # sanity: not a whole wall
        return np.empty((0, 3)), np.empty((0, 3))
    gx = np.arange(lo[0], hi[0] + spacing, spacing)
    gy = np.arange(lo[1], hi[1] + spacing, spacing)
    if gx.size * gy.size > 4_000_000:
        return np.empty((0, 3)), np.empty((0, 3))
    G = np.stack(np.meshgrid(gx, gy, indexing="ij"), -1).reshape(-1, 2)

    # inside the hole footprint...
    near_hole, _ = cKDTree(r2).query(G, k=1)
    keep = near_hole < spacing * 1.5
    if not keep.any():
        return np.empty((0, 3)), np.empty((0, 3))
    G = G[keep]

    # ...and not where the surface is already measured
    s3 = surface_pts
    if len(s3):
        s2 = to2d(s3)
        box = ((s2[:, 0] > lo[0] - 0.5) & (s2[:, 0] < hi[0] + 0.5)
               & (s2[:, 1] > lo[1] - 0.5) & (s2[:, 1] < hi[1] + 0.5))
        ring2 = s2[box]
        ringc = surface_col[box] if surface_col is not None else None
        if len(ring2):
            dn, _ = cKDTree(ring2).query(G, k=1)
            G = G[dn > spacing * 1.5]
    else:
        ring2 = np.empty((0, 2)); ringc = None
    if len(G) == 0:
        return np.empty((0, 3)), np.empty((0, 3))

    pts = origin + G[:, 0:1] * u + G[:, 1:2] * v
    if ringc is not None and len(ring2) >= 3:
        kk = min(8, len(ring2))
        _, idx = cKDTree(ring2).query(G, k=kk)
        if kk == 1:
            idx = idx[:, None]
        cols = np.median(ringc[idx], axis=1)
    else:
        cols = np.full((len(pts), 3), 0.6)
    return pts, np.clip(cols, 0.0, 1.0)
