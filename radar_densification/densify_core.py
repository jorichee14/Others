#!/usr/bin/env python3
"""
Core math for radar point-cloud densification (no ROS imports).

Implements the point-cloud-domain analog of DREAM-PCD's signal-processing
stages (arXiv:2309.15374), for radars that only output on-chip detections
(x, y, z, snr, doppler) instead of raw ADC data:

  1. Ego-velocity / ego-twist estimation from Doppler (RANSAC over the
     static returns) — "remove ego speed".
  2. Static / dynamic separation by Doppler residual against the fitted
     ego motion.
  3. Non-coherent accumulation: static points from many frames, transformed
     into a fixed world frame, accumulated on a voxel evidence grid.
     Per-voxel evidence (distinct frames seen, hits, SNR) replaces the
     coherent power integration of the raw-data pipeline; thresholding the
     evidence is the densify-while-denoise step.
  4. Aperture substitute: each detection carries an anisotropic covariance
     (sharp in range, wide in cross-range, growing with range). Points are
     fused per voxel in information form, so observations of the same
     surface from different viewpoints / different radars intersect their
     error ellipsoids — cross-range error shrinks the way a larger aperture
     would. Coherent SAA itself is impossible without phase.

All functions are pure numpy so they can be unit-tested and reused by both
the live ROS 2 node and the offline rosbag tool.
"""

import struct

import numpy as np

# --------------------------------------------------------------------------
# small SO(3) helpers
# --------------------------------------------------------------------------

def quat_to_R(q_xyzw):
    """Quaternion [x,y,z,w] -> 3x3 rotation matrix."""
    x, y, z, w = q_xyzw
    n = x * x + y * y + z * z + w * w
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    return np.array([
        [1 - s * (y * y + z * z), s * (x * y - w * z), s * (x * z + w * y)],
        [s * (x * y + w * z), 1 - s * (x * x + z * z), s * (y * z - w * x)],
        [s * (x * z - w * y), s * (y * z + w * x), 1 - s * (x * x + y * y)],
    ])


def skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]], float)


def so3_exp(w):
    """Rodrigues: rotation vector -> rotation matrix."""
    th = np.linalg.norm(w)
    if th < 1e-12:
        return np.eye(3) + skew(w)
    a = w / th
    K = skew(a)
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)


# --------------------------------------------------------------------------
# 1+2. ego motion from Doppler, static/dynamic split
# --------------------------------------------------------------------------

def _ransac_linear(A, b, n_min, thresh, iters, rng):
    """RANSAC for A x = b. Returns (x, inlier_mask) or (None, None)."""
    n = A.shape[0]
    if n < n_min:
        return None, None
    best_x, best_in, best_cnt = None, None, -1
    for _ in range(iters):
        idx = rng.choice(n, size=n_min, replace=False)
        As, bs = A[idx], b[idx]
        # reject degenerate samples
        if np.linalg.matrix_rank(As) < A.shape[1]:
            continue
        x, *_ = np.linalg.lstsq(As, bs, rcond=None)
        r = np.abs(A @ x - b)
        mask = r < thresh
        cnt = int(mask.sum())
        if cnt > best_cnt:
            best_cnt, best_x, best_in = cnt, x, mask
    if best_x is None:
        return None, None
    # refine on the consensus set (two rounds)
    for _ in range(2):
        if best_in.sum() < A.shape[1]:
            break
        x, *_ = np.linalg.lstsq(A[best_in], b[best_in], rcond=None)
        best_in = np.abs(A @ x - b) < thresh
        best_x = x
    return best_x, best_in


def estimate_sensor_velocity(pts, doppler, thresh=0.15, iters=60, seed=0,
                             static_frac=0.6, static_doppler=0.07):
    """
    Single-radar 3-DOF ego velocity from one frame (Kellner-style).

    For a static world point with unit direction u (radar frame) and sensor
    velocity v (radar frame): measured doppler d = s * (u . v), where the
    sign convention s is absorbed into v (classification is s-invariant;
    only the direction of the returned v depends on it — see doppler_sign
    at the caller).

    Returns (v, static_mask). v is None when the frame cannot support a fit
    (too few points); the mask then falls back to |doppler| gating.
    """
    pts = np.asarray(pts, float)
    doppler = np.asarray(doppler, float)
    r = np.linalg.norm(pts, axis=1)
    ok = r > 1e-6
    u = np.zeros_like(pts)
    u[ok] = pts[ok] / r[ok, None]

    # sensor-not-moving shortcut: most returns already ~0 doppler
    if (np.abs(doppler) < static_doppler).mean() >= static_frac:
        return np.zeros(3), np.abs(doppler) < thresh

    rng = np.random.default_rng(seed)
    v, inl = _ransac_linear(u[ok], doppler[ok], 3, thresh, iters, rng)
    if v is None:
        return None, np.abs(doppler) < thresh
    mask = np.zeros(len(pts), bool)
    mask[np.where(ok)[0]] = inl
    return v, mask


def estimate_base_twist(per_radar, thresh=0.15, iters=120, seed=0,
                        min_cond=1e-3):
    """
    Joint 6-DOF ego twist (v, w) of the BASE frame from ALL radars at once —
    the multi-radar "sparse aperture": each radar k, mounted at
    (R_k, t_k) = T_base_radar, sees

        d_i = s * u_i^T R_k^T ( v + w x t_k )
            = s * u_i^T R_k^T [ I  -[t_k]x ] [v; w]

    With three radars looking in different directions the lever arms make
    the yaw/pitch/roll rates observable — a single radar cannot give w.

    per_radar: list of dicts {pts (N,3 radar frame), doppler (N,), R, t}.
    Returns (v, w, masks) where masks is the per-radar static masks;
    (None, None, fallback_masks) when the joint solve is unusable — caller
    should fall back to per-radar estimate_sensor_velocity.
    """
    rows, rhs, sizes = [], [], []
    for item in per_radar:
        pts = np.asarray(item["pts"], float)
        d = np.asarray(item["doppler"], float)
        r = np.linalg.norm(pts, axis=1)
        ok = r > 1e-6
        u = np.zeros_like(pts)
        u[ok] = pts[ok] / r[ok, None]
        proj = u @ item["R"].T                      # u^T R^T, row-wise
        A = np.hstack([proj, -proj @ skew(item["t"])])
        A[~ok] = 0.0
        rows.append(A)
        rhs.append(d)
        sizes.append(len(d))
    A = np.vstack(rows) if rows else np.zeros((0, 6))
    b = np.concatenate(rhs) if rhs else np.zeros(0)

    if len(b) < 12:
        return None, None, None
    if (np.abs(b) < 0.07).mean() >= 0.6:            # rig is not moving
        masks, i0 = [], 0
        for n in sizes:
            masks.append(np.abs(b[i0:i0 + n]) < thresh)
            i0 += n
        return np.zeros(3), np.zeros(3), masks

    rng = np.random.default_rng(seed)
    x, inl = _ransac_linear(A, b, 6, thresh, iters, rng)
    if x is None:
        return None, None, None
    sv = np.linalg.svd(A[inl], compute_uv=False)
    if sv[-1] / sv[0] < min_cond:                   # w not observable
        return None, None, None
    masks, i0 = [], 0
    for n in sizes:
        masks.append(inl[i0:i0 + n])
        i0 += n
    return x[:3], x[3:], masks


# --------------------------------------------------------------------------
# 4. anisotropic per-detection covariance (the radar noise model)
# --------------------------------------------------------------------------

def detection_information(pts_radar, sigma_r=0.05, sigma_az_deg=3.0,
                          sigma_el_deg=8.0, snr=None, snr_ref=200.0,
                          snr_cap=4.0):
    """
    Per-point 3x3 information matrix (inverse covariance) in the RADAR frame.

    Noise is expressed in the detection's own (radial, azimuth-tangent,
    elevation-tangent) basis: sigma_r along the ray, r*sigma_az and
    r*sigma_el across it — the same model the calibration solver uses.
    Optionally scaled by SNR (stronger returns are trusted more, capped).

    Returns (N,3,3) array.
    """
    pts = np.asarray(pts_radar, float)
    n = len(pts)
    r = np.linalg.norm(pts, axis=1)
    r = np.maximum(r, 0.05)
    ur = pts / r[:, None]

    # azimuth tangent: horizontal, perpendicular to the ray
    up = np.array([0.0, 0.0, 1.0])
    ua = np.cross(up, ur)
    na = np.linalg.norm(ua, axis=1)
    bad = na < 1e-6                                 # ray ~ vertical
    ua[bad] = np.array([1.0, 0.0, 0.0])
    na[bad] = 1.0
    ua = ua / na[:, None]
    ue = np.cross(ur, ua)                           # elevation tangent

    s_az = np.radians(sigma_az_deg)
    s_el = np.radians(sigma_el_deg)
    var = np.stack([np.full(n, sigma_r ** 2),
                    (r * s_az) ** 2,
                    (r * s_el) ** 2], axis=1)       # (N,3)
    w = np.ones(n)
    if snr is not None:
        snr = np.asarray(snr, float)
        w = np.clip(snr / max(snr_ref, 1e-6), 1.0 / snr_cap, snr_cap)

    B = np.stack([ur, ua, ue], axis=2)              # (N,3,3) columns = basis
    inv_var = w[:, None] / var                      # (N,3)
    return np.einsum('nij,nj,nkj->nik', B, inv_var, B)


def rotate_information(Lam, R):
    """Rotate (N,3,3) information matrices by a single 3x3 R (world = R x)."""
    return np.einsum('ij,njk,lk->nil', R, Lam, R)


# --------------------------------------------------------------------------
# 3. the voxel evidence map (non-coherent accumulation)
# --------------------------------------------------------------------------

class VoxelEvidenceMap:
    """
    Global voxel grid accumulating evidence for static structure.

    Per voxel:
      Lam, eta   information-form position fusion: p_hat = Lam^-1 eta.
                 The fused point is NOT snapped to the voxel centre — the
                 voxel only buckets correspondences; the estimate keeps
                 sub-voxel accuracy and sharpens as viewpoints diversify.
      hits       number of detections
      frames     number of DISTINCT frames that hit the voxel (the
                 non-coherent integration count; thresholding this is what
                 kills flicker noise and multipath ghosts)
      snr_sum    accumulated SNR (mean SNR is exported as intensity)
    """

    def __init__(self, voxel=0.10):
        self.voxel = float(voxel)
        self.cells = {}

    def add(self, pts_world, Lam_world, snr, frame_id):
        pts = np.asarray(pts_world, float)
        if len(pts) == 0:
            return
        if snr is None:
            snr = np.zeros(len(pts))
        keys = np.floor(pts / self.voxel).astype(np.int64)
        eta = np.einsum('nij,nj->ni', Lam_world, pts)
        for i in range(len(pts)):
            k = (int(keys[i, 0]), int(keys[i, 1]), int(keys[i, 2]))
            c = self.cells.get(k)
            if c is None:
                c = {"Lam": np.zeros((3, 3)), "eta": np.zeros(3),
                     "hits": 0, "frames": 0, "last_frame": -1,
                     "snr_sum": 0.0}
                self.cells[k] = c
            c["Lam"] += Lam_world[i]
            c["eta"] += eta[i]
            c["hits"] += 1
            c["snr_sum"] += float(snr[i])
            if frame_id != c["last_frame"]:
                c["frames"] += 1
                c["last_frame"] = frame_id

    def extract(self, min_frames=3, min_hits=0):
        """
        Densified static cloud: fused position + attributes per voxel that
        passes the evidence threshold. Returns (pts (M,3), attrs dict).
        """
        pts, frames, hits, snr = [], [], [], []
        for c in self.cells.values():
            if c["frames"] < min_frames or c["hits"] < max(min_hits, 1):
                continue
            try:
                p = np.linalg.solve(c["Lam"], c["eta"])
            except np.linalg.LinAlgError:
                continue
            pts.append(p)
            frames.append(c["frames"])
            hits.append(c["hits"])
            snr.append(c["snr_sum"] / c["hits"])
        if not pts:
            return np.zeros((0, 3)), {"frames": np.zeros(0), "hits": np.zeros(0),
                                      "snr": np.zeros(0)}
        return (np.array(pts),
                {"frames": np.array(frames), "hits": np.array(hits),
                 "snr": np.array(snr)})

    def __len__(self):
        return len(self.cells)


def radius_outlier_filter(pts, attrs, voxel, min_neighbors=2):
    """
    Final cleanup: drop voxels with < min_neighbors occupied 26-neighbours.
    Cheap stand-in for the learned denoiser of the raw-data pipeline —
    isolated survivors of the evidence threshold are almost always
    multipath ghosts, not structure. min_neighbors=0 disables.
    """
    if min_neighbors <= 0 or len(pts) == 0:
        return pts, attrs
    occ = set(map(tuple, np.floor(pts / voxel).astype(np.int64)))
    keep = []
    for i, p in enumerate(np.floor(pts / voxel).astype(np.int64)):
        n = 0
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    if dx == dy == dz == 0:
                        continue
                    if (p[0] + dx, p[1] + dy, p[2] + dz) in occ:
                        n += 1
                        if n >= min_neighbors:
                            break
                if n >= min_neighbors:
                    break
            if n >= min_neighbors:
                break
        keep.append(n >= min_neighbors)
    keep = np.array(keep)
    return pts[keep], {k: v[keep] for k, v in attrs.items()}


# --------------------------------------------------------------------------
# output
# --------------------------------------------------------------------------

def save_ply(path, pts, attrs):
    """Binary little-endian PLY with x,y,z,intensity(snr),frames,hits."""
    n = len(pts)
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property float intensity\n"
        "property int frames\nproperty int hits\n"
        "end_header\n"
    )
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        for i in range(n):
            f.write(struct.pack(
                "<ffffii",
                float(pts[i, 0]), float(pts[i, 1]), float(pts[i, 2]),
                float(attrs["snr"][i]),
                int(attrs["frames"][i]), int(attrs["hits"][i])))
