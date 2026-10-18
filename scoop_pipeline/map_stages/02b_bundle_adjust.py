#!/usr/bin/env python3
"""
STAGE 02b - plane bundle adjustment: all poses at once, so that every plane
of the map is as thin as the scans allow.

  bag + traj_lidar_refined.txt (02)  ->  keyframes  ->  planes  ->  BA
  ->  traj_lidar_ba.txt

Stage 02 registers each scan alone to a map built from the poses being
fixed, so it can only centre a scan in that map's blur. Bundle adjustment
(BALM, Liu & Zhang 2021; HBA, Liu et al. 2023) moves all poses together:

  min over poses T   sum_f  lambda_min( Cov_f(T) )  +  odometry  +  anchor

Cov_f is the covariance of the points of plane f from every keyframe that
saw it, each through its keyframe's pose: lambda_min is the plane's squared
thickness summed over its points. The plane's own parameters are solved in
closed form (its normal is the eigenvector of lambda_min), which leaves the
poses as the only unknowns.

  1. keyframes: the scans of each kf_period s, deskewed and placed in the
     frame of the keyframe's middle pose with the input trajectory, thinned
     to kf_voxel. Within a keyframe the input's relative poses are kept.
  2. planes: the keyframes' points in the world, in voxel cubes; a cube is
     a plane when its points are flat (thinner than plane_sigma_max, and
     much thinner than wide) and at least min_kf keyframes saw it. Each
     keyframe's points in a cube are kept as one point cluster (N, sum x,
     sum x x^T), which is all the cost needs (BALM2's point clusters).
  3. Levenberg-Marquardt on the keyframe poses: Gauss-Newton with the planes
     eliminated (Schur complement), the planes re-solved in closed form after
     every step. Pairs far off their plane are down-weighted (Huber). The
     relative motion between keyframes is kept as a soft factor (odometry,
     from the input), so directions no plane sees (a corridor's length)
     stay where the input put them; each keyframe's roll and pitch are held
     near the input's (gravity: small floor cubes cannot see a slow vertical
     bend); the first keyframe is held fixed.
  4. steps 2-3 again (outer rounds), the planes found anew from the
     adjusted poses.
  5. every pose of the input moved by its keyframes' correction, interpolated
     in time between keyframes -> traj_lidar_ba.txt; its loop closure on the
     boards against GLIM's (loop_check).
  6. timing: whether the input's pose time is its scans' (timing_check), so
     every mapping pass re-checks glim.pose_time_offset; in the printout
     ([4]) and in <output>_report.json.

The plane thickness printed (5-95 % equivalent, 3.29 sigma) is the
quantity analysis/mapping/map_quality.py measures, on the BA's planes.

  python3 02b_bundle_adjust.py [pipeline_config.json] [--rounds N]
                               [--input TRAJ] [--output NAME]

Which input: GLIM's own trajectory (traj_lidar.txt) keeps GLIM's IMU-smooth
motion inside each keyframe, and its roll / pitch for the gravity factor;
02's (the default) carries 02's per-scan registration noise into every
keyframe, as the BA keeps the input's motion within a keyframe.

Config block (all optional, under "02b_bundle_adjust"):
  "input":            traj_lidar_refined.txt (02), else the seed trajectory
  "output":           "traj_lidar_ba.txt"
  "kf_period":        1.0     # s of scans per keyframe
  "kf_voxel":         0.05    # keyframe cloud thinning (m)
  "max_range":        12.0    # points further than this are not used (m)
  "voxel":            0.5     # plane cube size (m)
  "plane_sigma_max":  0.06    # a cube thicker than this (sigma, m) is not a plane
  "thinness":         0.3     # ... nor one whose sigma is above this x its width sigma
  "min_kf":           3       # keyframes that must see a plane
  "min_pts":          10      # points of one keyframe in a plane
  "max_planes":       12000
  "point_sigma":      0.02    # m, a point's noise along the normal
  "n_eff":            30      # a keyframe counts at most this many points per plane
                              # (its points share one pose error)
  "huber":            0.05    # m, rms of a keyframe on its plane above which it is down-weighted
  "odom_sigma_t":     0.03    # m, per keyframe step
  "odom_sigma_r_deg": 0.3     # deg, per keyframe step
  "gravity_sigma_deg": 0.1    # deg, a keyframe's roll / pitch from the input's (0 = off)
  "rounds":           3       # outer rounds (planes found anew)
  "iters":            10      # LM iterations per round
  "cache":            true    # keep the keyframes (<output>_keyframes.npz) for re-runs
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import scipy.sparse as sp
from scipy.linalg import cho_factor, cho_solve
from scipy.spatial.transform import Rotation

from pipeline_common import load_pipeline, load_traj, traj_quats, iter_scans

CHI = 3.29                       # 5-95 % spread of a normal = 3.29 sigma


def write_traj(path, times, T):
    q = traj_quats(T)
    with open(path, "w") as fh:
        for t, M, qq in zip(times, T, q):
            fh.write("%.9f %.6f %.6f %.6f %.9f %.9f %.9f %.9f\n"
                     % (t, *M[:3, 3], *qq))


def skew(v):
    """(M,3) -> (M,3,3) cross-product matrices."""
    z = np.zeros(len(v))
    return np.stack([z, -v[:, 2], v[:, 1], v[:, 2], z, -v[:, 0],
                     -v[:, 1], v[:, 0], z], 1).reshape(-1, 3, 3)


def voxel_thin(p, v):
    """Centroid per voxel of size v."""
    k = np.floor(p / v).astype(np.int64)
    _, inv, cnt = np.unique(k, axis=0, return_inverse=True, return_counts=True)
    inv = inv.ravel()
    out = np.zeros((len(cnt), 3))
    for i in range(3):
        out[:, i] = np.bincount(inv, p[:, i], len(cnt))
    return (out / cnt[:, None]).astype(np.float32)


# ---------------------------------------------------------------- keyframes
def keyframe_plan(times, period):
    """Keyframe of every pose, and each keyframe's pose index (the middle
    pose of its period)."""
    w = np.floor((times - times[0]) / period).astype(np.int64)
    uniq, kf_of = np.unique(w, return_inverse=True)
    anchor = np.array([int(np.flatnonzero(kf_of == k)[(kf_of == k).sum() // 2])
                       for k in range(len(uniq))])
    return kf_of, anchor


def build_keyframes(P, c, s, times, T_in, kf_of, anchor):
    """Each keyframe's points, in its anchor pose's frame, thinned."""
    bag = P.dataset["bag"]
    topic = c["points_topic"]
    hi = min(float(s["lidar_max"]), float(c["max_range"]))
    K = len(anchor)
    clouds = [np.zeros((0, 3), np.float32) for _ in range(K)]
    buf = {}

    def flush(k):
        pts = np.vstack(buf.pop(k))
        R, t = T_in[anchor[k], :3, :3], T_in[anchor[k], :3, 3]
        clouds[k] = voxel_thin((pts - t) @ R, float(c["kf_voxel"]))

    n = 0
    t0 = time.time()
    for j, pw in iter_scans(bag, topic, times, s["time_tol"], s["lidar_min"], hi, T_in,
                            c["deskew"], c["deskew_bins"], frame="world"):
        k = int(kf_of[j])
        buf.setdefault(k, []).append(pw)
        for kk in [kk for kk in buf if kk < k - 1]:
            flush(kk)
        n += 1
        if n % 1000 == 0:
            print(f"    {n} scans -> {k + 1}/{K} keyframes ({time.time() - t0:.0f} s)",
                  flush=True)
    for kk in list(buf):
        flush(kk)
    if n < 0.5 * len(times):
        raise SystemExit(f"only {n} of {len(times)} scans have a trajectory pose within "
                         f"01_build_map.time_tol ({s['time_tol']} s) of their stamp: are the "
                         "trajectory's stamps the scans' (not moved in time)?")
    print(f"    {n} scans in {K} keyframes, median {np.median([len(x) for x in clouds]):.0f} "
          f"points each at {c['kf_voxel'] * 100:g} cm ({time.time() - t0:.0f} s)")
    return clouds


# ------------------------------------------------------------------- planes
MOM = 10    # N, sx, sy, sz, xx, xy, xz, yy, yz, zz


def moments(p, inv, n):
    x, y, z = p[:, 0], p[:, 1], p[:, 2]
    cols = (np.ones(len(p)), x, y, z, x * x, x * y, x * z, y * y, y * z, z * z)
    return np.stack([np.bincount(inv, w, n) for w in cols], 1)


def cluster_matrix(m):
    """(M,10) moments -> (M,4,4) homogeneous clusters sum [x;1][x;1]^T."""
    C = np.empty((len(m), 4, 4))
    C[:, 0, 0], C[:, 0, 1], C[:, 0, 2] = m[:, 4], m[:, 5], m[:, 6]
    C[:, 1, 1], C[:, 1, 2], C[:, 2, 2] = m[:, 7], m[:, 8], m[:, 9]
    C[:, 1, 0], C[:, 2, 0], C[:, 2, 1] = m[:, 5], m[:, 6], m[:, 8]
    C[:, :3, 3] = m[:, 1:4]
    C[:, 3, :3] = m[:, 1:4]
    C[:, 3, 3] = m[:, 0]
    return C


def to_world(C, T):
    """Clusters in keyframe frames -> world: T C T^T."""
    return np.einsum("mij,mjk,mlk->mil", T, C, T)


def plane_fit(Cw):
    """(F,4,4) summed world clusters -> normal (F,3), offset d (F,), the
    eigenvalues of the centred scatter (F,3), ascending."""
    N = Cw[:, 3, 3]
    s = Cw[:, :3, 3]
    S = Cw[:, :3, :3] - np.einsum("fi,fj->fij", s, s) / N[:, None, None]
    lam, vec = np.linalg.eigh(S)
    n = vec[:, :, 0]
    d = np.einsum("fi,fi->f", n, s) / N
    return n, d, lam


def find_planes(clouds, T, c, rng):
    """Pairs (keyframe, plane) with their clusters, for the current poses."""
    v = float(c["voxel"])
    kf, keys, mom = [], [], []
    for k, p in enumerate(clouds):
        if not len(p):
            continue
        pk = p.astype(np.float64)
        pw = pk @ T[k, :3, :3].T + T[k, :3, 3]
        q = np.floor(pw / v).astype(np.int64) + (1 << 20)
        key = (q[:, 0] << 42) | (q[:, 1] << 21) | q[:, 2]
        uk, inv = np.unique(key, return_inverse=True)
        m = moments(pk, inv.ravel(), len(uk))
        ok = m[:, 0] >= int(c["min_pts"])
        kf.append(np.full(ok.sum(), k))
        keys.append(uk[ok])
        mom.append(m[ok])
    kf, keys, mom = np.concatenate(kf), np.concatenate(keys), np.concatenate(mom)
    o = np.argsort(keys, kind="stable")
    kf, keys, mom = kf[o], keys[o], mom[o]
    C = cluster_matrix(mom)
    Cw = to_world(C, T[kf])
    starts = np.flatnonzero(np.r_[True, keys[1:] != keys[:-1]])
    count = np.diff(np.r_[starts, len(keys)])
    A = np.add.reduceat(Cw, starts, axis=0)
    n, d, lam = plane_fit(A)
    N = A[:, 3, 3]
    sig = np.sqrt(np.maximum(lam / N[:, None], 0))
    good = ((count >= int(c["min_kf"])) & (sig[:, 0] < float(c["plane_sigma_max"]))
            & (sig[:, 0] < float(c["thinness"]) * sig[:, 1]) & (sig[:, 1] > 0.1 * v))
    cand = np.flatnonzero(good)
    if len(cand) > int(c["max_planes"]):
        cand = np.sort(rng.choice(cand, int(c["max_planes"]), replace=False))
    sel = np.zeros(len(starts), bool)
    sel[cand] = True
    plane_of_group = np.full(len(starts), -1)
    plane_of_group[cand] = np.arange(len(cand))
    grp = np.repeat(np.arange(len(starts)), count)
    keep = sel[grp]
    return dict(kf=kf[keep], plane=plane_of_group[grp[keep]], C=C[keep],
                n_planes=len(cand), n_cubes=len(starts),
                normal_z=np.abs(n[cand][:, 2]))


# ------------------------------------------------------------------- solver
def pose_mats(R, t):
    T = np.tile(np.eye(4), (len(R), 1, 1))
    T[:, :3, :3], T[:, :3, 3] = R, t
    return T


def planes_for(pr, R, t):
    """Closed-form planes for the current poses, weighted clusters."""
    Cw = to_world(pr["Cs"], pose_mats(R, t)[pr["kf"]])
    A = np.zeros((pr["n_planes"], 4, 4))
    np.add.at(A, pr["plane"], Cw)
    n, d, lam = plane_fit(A)
    return n, d, lam[:, 0]


def odom_terms(R, t, Rm, tm, W, want_jac):
    """Relative-pose factors between consecutive keyframes: residuals
    (K-1,6) [rotation, translation], cost, and their Jacobians."""
    Rrel = np.einsum("kji,kjl->kil", R[:-1], R[1:])
    trel = np.einsum("kji,kj->ki", R[:-1], t[1:] - t[:-1])
    eR = Rotation.from_matrix(np.einsum("kji,kjl->kil", Rm, Rrel)).as_rotvec()
    et = trel - tm
    e = np.hstack([eR, et])
    cost = float(np.einsum("ki,i,ki->", e, W, e))
    if not want_jac:
        return cost
    K1 = len(e)
    Ja = np.zeros((K1, 6, 6))
    Jb = np.zeros((K1, 6, 6))
    Ja[:, :3, :3] = -np.transpose(Rrel, (0, 2, 1))
    Jb[:, :3, :3] = np.eye(3)
    Ja[:, 3:, :3] = skew(trel)
    Ja[:, 3:, 3:] = -np.transpose(R[:-1], (0, 2, 1))
    Jb[:, 3:, 3:] = np.transpose(R[:-1], (0, 2, 1))
    return cost, e, Ja, Jb


def total_cost(pr, R, t, R0, t0, odo):
    _, _, lam = planes_for(pr, R, t)
    c_pl = float(lam.sum())
    c_od = odom_terms(R, t, odo["Rm"], odo["tm"], odo["W"], False) if len(R) > 1 else 0.0
    e0 = np.r_[Rotation.from_matrix(R0[0].T @ R[0]).as_rotvec(), t[0] - t0[0]]
    eg = R[:, 2, :] - odo["g0"]
    return c_pl + c_od + odo["wg"] * float((eg * eg).sum()) + 1e8 * float(e0 @ e0), c_pl


def gn_step(pr, R, t, R0, t0, odo, mu):
    """One damped Gauss-Newton step on the poses, planes eliminated."""
    K, F = len(R), pr["n_planes"]
    n, d, _ = planes_for(pr, R, t)
    kf, pl, C = pr["kf"], pr["plane"], pr["Cs"]
    Rk, tk, nk, dk = R[kf], t[kf], n[pl], d[pl]
    # a basis of each plane's normal's tangent space
    a = np.where(np.abs(n[:, :1]) < 0.9, np.array([[1.0, 0, 0]]), np.array([[0, 1.0, 0]]))
    b1 = np.cross(n, a)
    b1 /= np.linalg.norm(b1, axis=1, keepdims=True)
    b2 = np.cross(n, b1)
    B = np.stack([b1, b2], 2)[pl]                                     # (M,3,2)
    m = np.einsum("mji,mj->mi", Rk, nk)                               # R^T n
    pi = np.hstack([m, (np.einsum("mi,mi->m", nk, tk) - dk)[:, None]])
    M = len(kf)
    Jp = np.zeros((M, 4, 6))
    Jp[:, :3, :3] = skew(m)
    Jp[:, 3, 3:] = nk
    Jf = np.zeros((M, 4, 3))
    Jf[:, :3, :2] = np.einsum("mji,mjk->mik", Rk, B)
    Jf[:, 3, :2] = np.einsum("mi,mik->mk", tk, B)
    Jf[:, 3, 2] = -1.0
    CJp, CJf = C @ Jp, C @ Jf
    Hpp = np.einsum("mai,maj->mij", Jp, CJp)
    Hpf = np.einsum("mai,maj->mij", Jp, CJf)
    Hff_p = np.einsum("mai,maj->mij", Jf, CJf)
    Cpi = np.einsum("mab,mb->ma", C, pi)
    gp = np.einsum("mai,ma->mi", Jp, Cpi)
    gf_p = np.einsum("mai,ma->mi", Jf, Cpi)
    Hff = np.zeros((F, 3, 3))
    np.add.at(Hff, pl, Hff_p)
    gf = np.zeros((F, 3))
    np.add.at(gf, pl, gf_p)
    lam, U = np.linalg.eigh(Hff + 1e-9 * np.eye(3))
    Hih = np.einsum("fij,fj,fkj->fik", U, 1.0 / np.sqrt(np.maximum(lam, 1e-12)), U)  # H^-1/2
    Gb = np.einsum("mij,mjk->mik", Hpf, Hih[pl])                      # (M,6,3)
    rows = (6 * kf[:, None, None] + np.arange(6)[None, :, None]) + 0 * np.arange(3)
    cols = (3 * pl[:, None, None] + np.arange(3)[None, None, :]) + 0 * rows
    G = sp.csr_matrix((Gb.ravel(), (rows.ravel(), cols.ravel())), shape=(6 * K, 3 * F))
    S = -(G @ G.T).toarray()
    b = np.zeros(6 * K)
    np.add.at(b, (6 * kf[:, None] + np.arange(6)).ravel(), gp.ravel())
    b -= G @ np.einsum("fij,fj->fi", Hih, gf).ravel()
    blk = np.zeros((K, 6, 6))
    np.add.at(blk, kf, Hpp)
    idx = 6 * np.arange(K)[:, None] + np.arange(6)
    S[idx[:, :, None], idx[:, None, :]] += blk
    if K > 1:
        _, e, Ja, Jb = odom_terms(R, t, odo["Rm"], odo["tm"], odo["W"], True)
        W = odo["W"]
        for (A_, B_), (ia, ib) in (((Ja, Ja), (0, 0)), ((Ja, Jb), (0, 1)),
                                   ((Jb, Ja), (1, 0)), ((Jb, Jb), (1, 1))):
            blkab = np.einsum("kai,a,kaj->kij", A_, W, B_)
            ra, rb = idx[:-1] if ia == 0 else idx[1:], idx[:-1] if ib == 0 else idx[1:]
            S[ra[:, :, None], rb[:, None, :]] += blkab
        np.add.at(b, idx[:-1].ravel(), np.einsum("kai,a,ka->ki", Ja, W, e).ravel())
        np.add.at(b, idx[1:].ravel(), np.einsum("kai,a,ka->ki", Jb, W, e).ravel())
    if odo["wg"] > 0:
        # gravity: e = R^T z - R0^T z, de/dtheta = [R^T z]x
        gb = R[:, 2, :]
        Jg = skew(gb)
        eg = gb - odo["g0"]
        rot = idx[:, :3]
        S[rot[:, :, None], rot[:, None, :]] += odo["wg"] * np.einsum("kai,kaj->kij", Jg, Jg)
        np.add.at(b, rot.ravel(), (odo["wg"] * np.einsum("kai,ka->ki", Jg, eg)).ravel())
    e0 = np.r_[Rotation.from_matrix(R0[0].T @ R[0]).as_rotvec(), t[0] - t0[0]]
    S[:6, :6] += 1e8 * np.eye(6)
    b[:6] += 1e8 * e0
    S += mu * np.diag(np.maximum(np.diag(S), 1e-6))
    dx = -cho_solve(cho_factor(S, lower=True, check_finite=False), b, check_finite=False)
    dx = dx.reshape(K, 6)
    Rn = R @ Rotation.from_rotvec(dx[:, :3]).as_matrix()
    tn = t + dx[:, 3:]
    return Rn, tn, dx


def robust_scale(pr, R, t, c):
    """Per-pair weights: point noise, the n_eff cap, Huber on the pair's rms."""
    n, d, _ = planes_for(dict(pr, Cs=pr["C"]), R, t)
    kf, pl, C = pr["kf"], pr["plane"], pr["C"]
    m = np.einsum("mji,mj->mi", R[kf], n[pl])
    pi = np.hstack([m, (np.einsum("mi,mi->m", n[pl], t[kf]) - d[pl])[:, None]])
    N = C[:, 3, 3]
    rms = np.sqrt(np.maximum(np.einsum("ma,mab,mb->m", pi, C, pi), 0) / N)
    h = float(c["huber"])
    w = np.where(rms <= h, 1.0, h / np.maximum(rms, 1e-12))
    s = w * np.minimum(N, float(c["n_eff"])) / N / float(c["point_sigma"]) ** 2
    return C * s[:, None, None], rms, w


def thickness_report(pr, R, t):
    """Plane thickness (cm, 3.29 sigma) of walls / floors, unweighted."""
    _, _, lam = planes_for(dict(pr, Cs=pr["C"]), R, t)
    N = np.zeros(pr["n_planes"])
    np.add.at(N, pr["plane"], pr["C"][:, 3, 3])
    th = CHI * np.sqrt(np.maximum(lam, 0) / N) * 100
    wall = pr["normal_z"] < np.cos(np.radians(60))
    flat = pr["normal_z"] > np.cos(np.radians(30))
    med = lambda v: float(np.median(v)) if len(v) else float("nan")          # noqa: E731
    return med(th), med(th[wall]), med(th[flat])


# ------------------------------------------------------------- corrections
def apply_corrections(times, T_in, anchor, R_new, t_new):
    """Every input pose moved by its keyframes' correction, interpolated in
    time: T' = dT(t) T, dT_k = T'_k T_k^-1 (world frame)."""
    Ta = T_in[anchor]
    dR = R_new @ np.transpose(Ta[:, :3, :3], (0, 2, 1))
    dt = t_new - np.einsum("kij,kj->ki", dR, Ta[:, :3, 3])
    ta = times[anchor]
    if len(anchor) == 1:
        Rc = np.repeat(dR, len(times), 0)
        tc = np.repeat(dt, len(times), 0)
    else:
        from scipy.spatial.transform import Slerp
        tq = np.clip(times, ta[0], ta[-1])
        Rc = Slerp(ta, Rotation.from_matrix(dR))(tq).as_matrix()
        tc = np.stack([np.interp(tq, ta, dt[:, i]) for i in range(3)], 1)
    T = T_in.copy()
    T[:, :3, :3] = Rc @ T_in[:, :3, :3]
    T[:, :3, 3] = np.einsum("nij,nj->ni", Rc, T_in[:, :3, 3]) + tc
    return T


# --------------------------------------------------------------------- main
TIMING_MS, TIMING_R = 10.0, 0.5    # flag the input's pose time beyond these


def timing_check(inp, times, T_in, T_out):
    """Is the input's pose time on its scans'? The adjusted poses are fixed
    by the scans, so a pose stamped dt off in the input shows as a heading
    error against them that follows the turning, e = w dt
    (compare_trajs.heading_vs_turning). Flagged when |dt| >= 10 ms and
    r >= 0.5 (20261015: GLIM as output 45.8 ms, r 0.92; with the 46 ms
    export correction 3.9 ms, r 0.23). The correction the input was exported
    with is read from glim.yaml beside it; the right one is it + dt."""
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "..", "analysis", "odom"))
    from compare_trajs import heading_vs_turning
    h = heading_vs_turning(times, T_in, times, T_out)
    applied = 0.0
    meta = os.path.join(os.path.dirname(inp), "glim.yaml")
    if os.path.isfile(meta):
        for ln in open(meta):
            if ln.startswith("pose_time_offset_s:"):
                applied = float(ln.split(":", 1)[1] or 0)
    dt, r = h.get("dt_ms", 0.0), h.get("r_timing", 0.0)
    h.update(applied_offset_ms=applied * 1000,
             on_time=not (abs(dt) >= TIMING_MS and r >= TIMING_R))
    print(f"\n[4] timing: the input's heading error against the result follows the turning "
          f"as {dt:+.1f} ms (r {r:.2f}); input exported with pose_time_offset "
          f"{applied * 1000:.0f} ms")
    if h["on_time"]:
        print("    -> on time: the input's poses are its scans' (keep the offset)")
    else:
        print(f"    -> OFF by {dt:+.0f} ms: set glim.pose_time_offset to "
              f"{applied + dt / 1000:.3f} in configs/recording.yaml for this machine, "
              "re-export GLIM (processing/glim_check.py) and run 02b again")
    return h


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", nargs="?", default="pipeline_config.json")
    ap.add_argument("--rounds", type=int, default=None)
    ap.add_argument("--input", help="trajectory to adjust (a bare name is in odometry/<machine>/)")
    ap.add_argument("--output", help="output name (default traj_lidar_ba.txt)")
    a = ap.parse_args()

    P = load_pipeline(a.config)
    s = P.cfg["01_build_map"]
    c = dict(input="", output="traj_lidar_ba.txt", kf_period=1.0, kf_voxel=0.05,
             max_range=12.0, voxel=0.5, plane_sigma_max=0.06, thinness=0.3,
             min_kf=3, min_pts=10, max_planes=12000, point_sigma=0.02, n_eff=30,
             huber=0.05, odom_sigma_t=0.03, odom_sigma_r_deg=0.3, gravity_sigma_deg=0.1,
             rounds=3, iters=10,
             cache=True, points_topic="",
             deskew=bool(s.get("deskew", True)), deskew_bins=max(2, int(s.get("deskew_bins", 100))))
    c.update(P.cfg.get("02b_bundle_adjust", {}))
    if a.rounds is not None:
        c["rounds"] = a.rounds
    if a.input:
        c["input"] = a.input
    if a.output:
        c["output"] = a.output
    c["points_topic"] = (c["points_topic"] or P.dataset.get("points_topic")
                         or "/ouster/points")
    print(P.describe())

    inp = P.outp(c["input"]) if c["input"] else P.outp("traj_lidar_refined.txt")
    if not os.path.exists(inp):
        inp = P.seed_traj()
    out = P.outp(c["output"])
    if os.path.abspath(inp) == os.path.abspath(out):
        raise SystemExit(f"input and output are the same file ({out}); set 02b_bundle_adjust.input")
    times, T_in = load_traj(inp)
    kf_of, anchor = keyframe_plan(times, float(c["kf_period"]))
    K = len(anchor)
    print(f"input {inp}: {len(times)} poses -> {K} keyframes of {c['kf_period']:g} s")

    # 1. keyframes (cached: reading the bag is the slow part)
    cache = os.path.splitext(out)[0] + "_keyframes.npz"
    key = dict(input=os.path.abspath(inp), mtime=os.path.getmtime(inp),
               bag=os.path.abspath(os.path.expanduser(P.dataset["bag"])),
               **{k: c[k] for k in ("kf_period", "kf_voxel", "max_range", "deskew",
                                    "deskew_bins", "points_topic")})
    clouds = None
    if c["cache"] and os.path.exists(cache):
        z = np.load(cache, allow_pickle=False)
        if json.loads(str(z["key"])) == json.loads(json.dumps(key)):
            sizes = z["sizes"]
            clouds = np.split(z["points"], np.cumsum(sizes)[:-1])
            print(f"[1] keyframes from {cache}")
    if clouds is None:
        print("[1] keyframes: reading the scans")
        clouds = build_keyframes(P, c, s, times, T_in, kf_of, anchor)
        if c["cache"]:
            np.savez(cache, points=np.vstack(clouds), sizes=np.array([len(x) for x in clouds]),
                     key=json.dumps(key))
            print(f"    kept in {cache} ({os.path.getsize(cache) / 1e6:.0f} MB; "
                  f"delete it to read the bag again)")

    R0 = T_in[anchor, :3, :3].copy()
    t0 = T_in[anchor, :3, 3].copy()
    R, t = R0.copy(), t0.copy()
    odo = dict(Rm=np.einsum("kji,kjl->kil", R0[:-1], R0[1:]),
               tm=np.einsum("kji,kj->ki", R0[:-1], t0[1:] - t0[:-1]),
               W=np.r_[[1 / np.radians(float(c["odom_sigma_r_deg"])) ** 2] * 3,
                       [1 / float(c["odom_sigma_t"]) ** 2] * 3],
               g0=R0[:, 2, :].copy(),           # R0^T z: gravity in each keyframe
               wg=(1 / np.radians(float(c["gravity_sigma_deg"])) ** 2
                   if float(c["gravity_sigma_deg"]) > 0 else 0.0))
    rng = np.random.default_rng(0)
    history = []
    for rnd in range(1, int(c["rounds"]) + 1):
        print(f"\n=== round {rnd}/{c['rounds']} ===")
        pr = find_planes(clouds, pose_mats(R, t), c, rng)
        if pr["n_planes"] < 10:
            raise SystemExit(f"only {pr['n_planes']} planes found in {pr['n_cubes']} cubes; "
                             "check voxel / plane_sigma_max")
        pr["Cs"], rms, w = robust_scale(pr, R, t, c)
        per_kf = np.bincount(pr["kf"], minlength=K)
        print(f"[2] {pr['n_planes']} planes of {pr['n_cubes']} cubes "
              f"({(pr['normal_z'] < 0.5).sum()} walls, {(pr['normal_z'] > 0.87).sum()} floors/ceilings), "
              f"{len(pr['kf'])} keyframe-plane pairs, {(per_kf == 0).sum()} keyframes without a plane; "
              f"{(w < 1).mean() * 100:.0f} % pairs down-weighted")
        th0 = thickness_report(pr, R, t)
        E, _ = total_cost(pr, R, t, R0, t0, odo)
        mu = 1e-4
        tic = time.time()
        for it in range(int(c["iters"])):
            for _ in range(8):
                Rn, tn, dx = gn_step(pr, R, t, R0, t0, odo, mu)
                En, _ = total_cost(pr, Rn, tn, R0, t0, odo)
                if En < E:
                    break
                mu *= 10
            else:
                print(f"    iteration {it + 1}: no step lowers the cost; stopping")
                break
            rel = (E - En) / max(E, 1e-12)
            R, t, E = Rn, tn, En
            mu = max(mu / 10, 1e-8)
            step = np.linalg.norm(dx[:, 3:], axis=1)
            print(f"    iteration {it + 1}: cost -{rel * 100:.2f} %, step max "
                  f"{step.max() * 100:.2f} cm / {np.degrees(np.linalg.norm(dx[:, :3], axis=1)).max():.3f} deg "
                  f"({time.time() - tic:.0f} s)", flush=True)
            if rel < 1e-4:
                break
        th1 = thickness_report(pr, R, t)
        moved = np.linalg.norm(t - t0, axis=1) * 100
        tilt = np.degrees(np.arccos(np.clip(np.einsum("ki,ki->k", R[:, 2, :], odo["g0"]), -1, 1)))
        print(f"[3] plane thickness (3.29 sigma) all / walls / floors: "
              f"{th0[0]:.1f} / {th0[1]:.1f} / {th0[2]:.1f} cm  ->  "
              f"{th1[0]:.1f} / {th1[1]:.1f} / {th1[2]:.1f} cm")
        print(f"    keyframes moved from the input: median {np.median(moved):.1f} cm, "
              f"p95 {np.percentile(moved, 95):.1f} cm, max {moved.max():.1f} cm; vertical "
              f"median {np.median(np.abs(t - t0)[:, 2]) * 100:.1f} cm, p95 "
              f"{np.percentile(np.abs(t - t0)[:, 2], 95) * 100:.1f} cm; roll/pitch change "
              f"median {np.median(tilt):.3f} deg, max {tilt.max():.3f} deg")
        history.append(dict(round=rnd, planes=pr["n_planes"], thickness_before_cm=th0,
                            thickness_after_cm=th1, moved_median_cm=float(np.median(moved)),
                            moved_p95_cm=float(np.percentile(moved, 95))))

    T_out = apply_corrections(times, T_in, anchor, R, t)
    write_traj(out, times, T_out)
    try:
        timing = timing_check(inp, times, T_in, T_out)
    except Exception as e:                     # the poses stand either way
        timing = {"error": f"{type(e).__name__}: {e}"}
        print(f"\n(timing not checked: {timing['error']})")
    with open(os.path.splitext(out)[0] + "_report.json", "w") as fh:
        json.dump(dict(input=inp, output=out, keyframes=K, settings={k: c[k] for k in c},
                       rounds=history, timing=timing), fh, indent=1)
    d = np.linalg.norm(T_out[:, :3, 3] - T_in[:, :3, 3], axis=1) * 100
    print(f"\nwrote {out}: {len(times)} poses, moved median {np.median(d):.1f} cm, "
          f"p95 {np.percentile(d, 95):.1f} cm, max {d.max():.1f} cm")
    try:
        import loop_check
        loop_check.check_trajectory(a.config, out, label="BA")
    except Exception as e:                     # the poses stand either way
        print(f"\n(loop closure not measured: {type(e).__name__}: {e})")
    print(f"\nnext: set dataset.traj to \"{c['output']}\", delete "
          f"{', '.join(P.pcd(b) for b in ('merged', 'static', 'denoised', 'colored'))} "
          f"in {P.folder_for('x.pcd')}, re-run 01_build_map.py, then "
          f"analysis/mapping/map_quality.py and analysis/mapping/wall_build.py")


if __name__ == "__main__":
    sys.exit(main())
