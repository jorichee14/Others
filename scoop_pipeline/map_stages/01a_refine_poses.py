#!/usr/bin/env python3
"""
STAGE 01a - re-derive the trajectory by registering every scan to the MAP.

  bag (/ouster/points) + seed traj  ->  reference map  ->  per-scan ICP
  ->  traj_lidar_refined.txt

This is the construction used by the survey-prior datasets (Newer College,
Oxford Spires, ConSLAM): the reference map is frozen, and each LiDAR scan is
localized IN it independently, so the poses inherit the map's consistency
instead of the odometry's drift. The only difference here is that the prior
map is self-built (from the seed trajectory) rather than survey-grade.

Why this can fix what pose-graph tuning and HBA could not: a pose graph can
only remove error where two passes actually observe the same place, and plane
bundle adjustment (HBA) only optimizes voxels that pass its planarity test --
a 12 cm smear in a 1 m voxel is rejected before it is ever considered. ICP
against the accumulated map has neither restriction: every scan is pulled onto
the mean surface of ALL scans that saw it, so within-pass smear averages out.

Each scan is initialized from its SEED pose, never from the previous scan's
refined pose, so errors cannot accumulate along the trajectory. A registration
that wants to move a scan further than max_shift / max_rot is rejected and the
seed pose is kept -- one bad ICP basin can never drag the trajectory with it.

Outputs go to <processed>/odometry/<machine>/ and the round-1 map is read from
<processed>/mapping/ (data/processed/<date>/<pass>/, pipeline_common.ROUTES).

Multiple rounds are done in-process: round 1 registers against the map implied
by the seed poses (or, better, the pipeline's denoised.pcd, which has already
had dynamic objects and outliers removed), and each later round rebuilds the
reference map from the poses the previous round produced. The per-round
correction statistics printed at the end are the convergence evidence: when
round N moves the poses by millimetres, the trajectory and the map agree.

  python3 01a_refine_poses.py [pipeline_config.json] [--rounds N] [--restart]

Rounds already written (traj_lidar_refined_r1.txt, _r2, ...) are not redone:
--rounds 3 after a 2-round run does round 3 only, starting from _r2's poses
(and the convergence table includes the earlier rounds, kept in
traj_lidar_refined_rounds.json). --restart starts again from round 1.

Config block (all optional, under "01a_refine"):
  "map":            denoised_<tag>.pcd, what 01 wrote  # reference for round 1; "" = build it
  "target_voxel":   0.05    # reference map resolution (m)
  "scan_voxel":     0.10    # per-scan downsample before ICP (m)
  "max_corr":       [0.4, 0.2, 0.1]   # coarse-to-fine correspondence gates
  "iters_per_gate": 5
  "huber":          0.05    # robust kernel scale (m)
  "min_corr":       200     # fewer correspondences -> keep the seed pose
  "max_shift":      0.50    # reject a correction larger than this (m)
  "max_rot_deg":    5.0
  "output":         "traj_lidar_refined.txt"
  "deskew":         01_build_map.deskew   # per-point motion compensation, through
  "deskew_bins":    01_build_map.deskew_bins  # the trajectory of the current round
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore

from pipeline_common import load_pipeline, pc2_xyzt, traj_quats, deskew_to_pose

TS = get_typestore(Stores.ROS2_HUMBLE)

def load_traj(path):
    """TUM file -> (times, (N,4,4) poses). Same format the pipeline writes."""
    d = np.loadtxt(path)
    t = d[:, 0]
    T = np.tile(np.eye(4), (len(d), 1, 1))
    T[:, :3, 3] = d[:, 1:4]
    x, y, z, w = d[:, 4], d[:, 5], d[:, 6], d[:, 7]
    T[:, 0, 0] = 1 - 2 * (y * y + z * z)
    T[:, 0, 1] = 2 * (x * y - z * w)
    T[:, 0, 2] = 2 * (x * z + y * w)
    T[:, 1, 0] = 2 * (x * y + z * w)
    T[:, 1, 1] = 1 - 2 * (x * x + z * z)
    T[:, 1, 2] = 2 * (y * z - x * w)
    T[:, 2, 0] = 2 * (x * z - y * w)
    T[:, 2, 1] = 2 * (y * z + x * w)
    T[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return t, T


def write_traj(path, times, T):
    """(N,4,4) -> TUM 't x y z qx qy qz qw', timestamps preserved verbatim."""
    R = T[:, :3, :3]
    tr = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    q = np.zeros((len(T), 4))
    b0 = tr > 0
    for i in np.flatnonzero(b0):
        s = np.sqrt(tr[i] + 1.0) * 2
        q[i] = [(R[i, 2, 1] - R[i, 1, 2]) / s, (R[i, 0, 2] - R[i, 2, 0]) / s,
                (R[i, 1, 0] - R[i, 0, 1]) / s, 0.25 * s]
    for i in np.flatnonzero(~b0):
        d = np.array([R[i, 0, 0], R[i, 1, 1], R[i, 2, 2]])
        k = int(np.argmax(d))
        j, l = (k + 1) % 3, (k + 2) % 3
        s = np.sqrt(1.0 + R[i, k, k] - R[i, j, j] - R[i, l, l]) * 2
        qq = np.zeros(4)
        qq[k] = 0.25 * s
        qq[j] = (R[i, j, k] + R[i, k, j]) / s
        qq[l] = (R[i, l, k] + R[i, k, l]) / s
        qq[3] = (R[i, l, j] - R[i, j, l]) / s
        q[i] = qq
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    out = np.column_stack([times, T[:, :3, 3], q])
    np.savetxt(path, out, fmt="%.9f %.6f %.6f %.6f %.9f %.9f %.9f %.9f")


def nearest_idx(tr_t, t):
    i = int(np.searchsorted(tr_t, t))
    if i <= 0:
        return 0
    if i >= len(tr_t):
        return len(tr_t) - 1
    return i if (tr_t[i] - t) < (t - tr_t[i - 1]) else i - 1


def iter_scans(bag, topic, tr_t, tol, lo, hi, tr_T=None, deskew=False, nb=100):
    """Yield (pose_index, points) for every associated scan, in the sensor
    frame of that pose.

    deskew (with tr_T, the trajectory of this round): each point is moved
    from the sensor frame at its own time (cloud stamp + its 't') into the
    frame of pose j through the trajectory, T_j^-1 T(t + dt) p -- the same
    placement as 01_build_map's deskew -- so a rigid registration of the
    scan is not fitting a sweep smeared by the motion during it."""
    tr_q = traj_quats(tr_T) if deskew and tr_T is not None else None
    told = [False]
    with AnyReader([Path(bag)], default_typestore=TS) as r:
        conns = [c for c in r.connections
                 if c.topic == topic and "PointCloud2" in c.msgtype]
        if not conns:
            have = sorted({c.topic for c in r.connections})
            raise SystemExit(f"topic {topic!r} not in bag; available: {have}")
        for conn, _, raw in r.messages(connections=conns):
            msg = r.deserialize(raw, conn.msgtype)
            t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            j = nearest_idx(tr_t, t)
            if abs(tr_t[j] - t) > tol:
                continue
            p, dt = pc2_xyzt(msg)
            p = p.astype(np.float64)
            keep = np.isfinite(p).all(1)
            d = np.linalg.norm(p, axis=1)
            keep &= (d > lo) & (d < hi)
            if keep.sum() < 100:
                continue
            p = p[keep]
            if tr_q is not None and dt is not None:
                p = deskew_to_pose(p, dt[keep], t, j, tr_t, tr_T, tr_q, nb)
                if not told[0]:
                    told[0] = True
                    print(f"    [deskew] ON, {nb} bins (points into the frame of "
                          f"their scan's pose, through this round's trajectory)")
            yield j, p


def detect_points_topic(bag, override="", n_poses=0):
    """Pick the LiDAR cloud topic.

    Message count alone is a bad discriminator -- a radar or an RGB-D camera
    can easily publish more clouds than the LiDAR. The trajectory was produced
    from the LiDAR, so the topic whose message count is closest to the number
    of poses is the right one; sensors that are never the mapping LiDAR are
    demoted outright."""
    if override:
        return override
    demote = ("radar", "zed", "depth", "rgbd", "realsense", "camera")
    with AnyReader([Path(bag)], default_typestore=TS) as r:
        cands = [(c.topic, c.msgcount) for c in r.connections
                 if "PointCloud2" in c.msgtype]
    if not cands:
        raise SystemExit("no PointCloud2 topic in the bag")
    if len(cands) == 1:
        return cands[0][0]

    def score(tc):
        topic, count = tc
        bad = any(k in topic.lower() for k in demote)
        near = abs(count - n_poses) / max(n_poses, 1) if n_poses else 0.0
        return (bad, near, -count)

    ranked = sorted(cands, key=score)
    topic = ranked[0][0]
    print(f"    PointCloud2 topics in bag: "
          f"{[f'{t} ({n})' for t, n in cands]}")
    print(f"    -> using {topic!r} ({dict(cands)[topic]} msgs vs "
          f"{n_poses} poses); set 01a_refine.points_topic to override")
    return topic


class VoxelAvg:
    """Incremental voxel accumulator keeping the CENTROID of each voxel.

    Voxel centres would quantize the reference map onto the grid -- +/-2.5 cm
    of injected error at a 5 cm voxel, which is the same order as the pose
    corrections being solved for, so the ICP would chase the grid instead of
    the surface. Centroids cost one float64 sum plus a count per voxel and
    keep the map at the sensor's own accuracy."""
    OFF, BITS = 1 << 19, 20

    def __init__(self, voxel):
        self.inv = 1.0 / float(voxel)
        self.keys = None
        self.sums = None
        self.cnts = None
        self._kb, self._pb, self._n = [], [], 0

    def add(self, pts):
        v = np.floor(pts * self.inv).astype(np.int64) + self.OFF
        self._kb.append((v[:, 0] << (2 * self.BITS))
                        | (v[:, 1] << self.BITS) | v[:, 2])
        self._pb.append(pts)
        self._n += len(pts)
        if self._n > 20_000_000:
            self.flush()

    def flush(self):
        if not self._kb:
            return
        k = np.concatenate(self._kb)
        p = np.concatenate(self._pb)
        self._kb, self._pb, self._n = [], [], 0
        c = np.ones(len(k), np.int64)
        if self.keys is not None:
            k = np.concatenate([self.keys, k])
            p = np.concatenate([self.sums, p])
            c = np.concatenate([self.cnts, c])
        order = np.argsort(k, kind="stable")
        k, p, c = k[order], p[order], c[order]
        first = np.concatenate(([True], k[1:] != k[:-1]))
        start = np.flatnonzero(first)
        cs = np.cumsum(p, axis=0)
        cc = np.cumsum(c)
        end = np.concatenate((start[1:], [len(k)])) - 1
        tot_p, tot_c = cs[end], cc[end]
        self.keys = k[start]
        self.sums = np.vstack((tot_p[:1], np.diff(tot_p, axis=0)))
        self.cnts = np.concatenate((tot_c[:1], np.diff(tot_c)))

    def points(self):
        self.flush()
        return self.sums / self.cnts[:, None]


class Reference:
    """Frozen reference map, in two forms.

    * a KD-tree over the points, for the fine nearest-neighbour polish;
    * a table of local PLANES -- one centroid + normal per plane_voxel cell --
      for the coarse stage.

    The plane table is what makes a smeared map correctable. Nearest-neighbour
    point-to-plane cannot see a scan that sits 3 cm off the mean surface of a
    10 cm-thick smear: every scan point still finds a map point a millimetre
    away inside the slab, so the residual is ~0 wherever in the smear the scan
    happens to lie, and ICP declares victory. Fitting one plane over a cell
    WIDER than the smear restores the gradient -- the residual is measured
    against the slab's mean surface, so each scan is pulled to where the bulk
    of the evidence says the wall is, and the smear collapses over rounds.

    Cells are WEIGHTED by planarity rather than gated on it. A hard threshold
    is self-defeating here: measured on this data a 0.35 gate admitted only
    half the wall cells, and the half it admitted were 7.6 cm thick while the
    half it rejected were 15.3 cm -- i.e. it optimized the surfaces that were
    already good and skipped the ones carrying the error, which is exactly why
    plane bundle adjustment (HBA) was a no-op here. A soft weight lets a
    thick-but-flat wall contribute fully and lets a corner fade out."""

    def __init__(self, pts, knn=20, plane_voxel=0.4, planarity=1.0):
        pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
        print(f"    estimating normals over {len(pts)} points ...", flush=True)
        pc.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(knn=knn))
        self.pts = np.asarray(pc.points)
        self.nrm = np.asarray(pc.normals)
        del pc
        print("    building KD-tree ...", flush=True)
        self.tree = cKDTree(self.pts)
        self._build_planes(plane_voxel, planarity)

    def plane_of(self, pts):
        """Rows of the plane table for the cell each point falls in (-1 = none)."""
        v = np.floor(pts / self.pvoxel).astype(np.int64) + (1 << 19)
        k = (v[:, 0] << 40) | (v[:, 1] << 20) | v[:, 2]
        pos = np.searchsorted(self.ksorted, k)
        pos = np.clip(pos, 0, len(self.ksorted) - 1)
        hit = self.ksorted[pos] == k
        row = np.where(hit, self.korder[pos], -1)
        return row

    def _build_planes(self, voxel, planarity):
        v = np.floor(self.pts / voxel).astype(np.int64) + (1 << 19)
        k = (v[:, 0] << 40) | (v[:, 1] << 20) | v[:, 2]
        order = np.argsort(k, kind="stable")
        ks, ps = k[order], self.pts[order]
        first = np.concatenate(([True], ks[1:] != ks[:-1]))
        start = np.flatnonzero(first)
        end = np.concatenate((start[1:], [len(ks)]))
        cen, nrm, keys, wts = [], [], [], []
        for a, b in zip(start, end):
            if b - a < 30:
                continue
            P = ps[a:b]
            c = P.mean(0)
            Q = P - c
            w, V = np.linalg.eigh(Q.T @ Q / (b - a))
            if w[1] <= 0:
                continue
            ratio = np.sqrt(max(w[0], 0.0)) / np.sqrt(w[1])   # w[0] can be -1e-18 on a flat cell
            cw = 1.0 - min(ratio / planarity, 1.0)
            if cw <= 0.02:                    # a corner or clutter, no plane
                continue
            cen.append(c)
            nrm.append(V[:, 0])
            keys.append(ks[a])
            wts.append(cw)
        self.pcen = np.array(cen)
        self.pnrm = np.array(nrm)
        self.pw = np.array(wts)
        self.pvoxel = voxel
        # key -> row, so a scan point is matched to the plane of the cell it
        # actually falls in. Nearest-centroid matching would let floor points
        # be dragged onto a wall plane a few decimetres away.
        self.pkeys = np.array(keys, dtype=np.int64)
        self.korder = np.argsort(self.pkeys)
        self.ksorted = self.pkeys[self.korder]
        print(f"    plane table: {len(cen)} cells @ {voxel} m "
              f"(median planarity weight {np.median(wts):.2f})")


def _solve(pw, c, n, r, huber, prior=None, beta=0.0, cw=None):
    """One linearized point-to-plane step about the sensor position c.

    `prior` anchors the pose to its seed in directions the geometry does not
    constrain. A corridor is the canonical case: parallel walls plus floor
    leave translation ALONG the corridor unobservable, so an unregularized
    solve slides the scan freely down the hallway -- the map looks thinner
    while the trajectory silently goes wrong. Damping proportional to the
    data strength in each block (Tikhonov, beta ~ 0.05) is felt only where
    the evidence is weak, so observable directions are untouched."""
    w = np.minimum(1.0, huber / np.maximum(np.abs(r), 1e-9))
    if cw is not None:
        w = w * cw
    A = np.hstack([np.cross(pw - c, n), n])
    Aw = A * w[:, None]
    H = Aw.T @ A
    g = -Aw.T @ r
    if prior is not None and beta > 0:
        lam_r = beta * np.trace(H[:3, :3]) / 3.0
        lam_t = beta * np.trace(H[3:, 3:]) / 3.0
        lam = np.array([lam_r] * 3 + [lam_t] * 3)
        H = H + np.diag(lam)
        g = g - lam * prior
    try:
        x = np.linalg.solve(H + 1e-9 * np.eye(6), g)
    except np.linalg.LinAlgError:
        return None
    return x[:3], x[3:]


def _apply(T, w_, t_, c):
    th = np.linalg.norm(w_)
    if th < 1e-12:
        dR = np.eye(3)
    else:
        k = w_ / th
        K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
        dR = np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)
    T[:3, :3] = dR @ T[:3, :3]
    T[:3, 3] = c + t_
    return th, np.linalg.norm(t_)


def register(ref, pts_local, T0, cfg):
    """Two-stage ICP of one scan against the reference map.

    Stage 1 (plane): residuals against local plane fits spanning a cell wider
    than the smear -- this is the stage that can actually see, and remove, a
    scan's offset within a thick surface.
    Stage 2 (nearest-neighbour): the original fine polish, which recovers the
    detail a 0.4 m plane fit averages away (corners, door reveals).

    Rotation is parameterized about the sensor position, not the world origin:
    at 30 m from origin a world-frame lever arm makes the 6x6 system badly
    conditioned and the correction leaks into translation.

    Returns (T, n_corr, rms, rms_plane_before, rms_plane_after)."""
    T = T0.copy()
    huber = float(cfg["huber"])
    r0 = r1 = np.nan

    if len(ref.pcen) and int(cfg.get("plane_iters", 8)) > 0:
        ph = float(cfg.get("plane_huber", 0.10))
        for it in range(int(cfg["plane_iters"])):
            c = T[:3, 3]
            p = pts_local @ T[:3, :3].T + c
            i = ref.plane_of(p)
            ok = i >= 0
            if ok.sum() < int(cfg["min_corr"]):
                break
            pw = p[ok]
            n = ref.pnrm[i[ok]]
            r = np.einsum("ij,ij->i", pw - ref.pcen[i[ok]], n)
            if it == 0:
                r0 = float(np.sqrt(np.mean(r ** 2)))
            drot = T[:3, :3] @ T0[:3, :3].T
            ang = np.arccos(np.clip((np.trace(drot) - 1) / 2, -1, 1))
            axis = np.zeros(3)
            if ang > 1e-9:
                axis = np.array([drot[2, 1] - drot[1, 2],
                                 drot[0, 2] - drot[2, 0],
                                 drot[1, 0] - drot[0, 1]]) / (2 * np.sin(ang)) * ang
            prior = np.concatenate([axis, T[:3, 3] - T0[:3, 3]])
            sol = _solve(pw, c, n, r, ph, prior,
                         float(cfg.get("prior_beta", 0.05)),
                         cw=ref.pw[i[ok]])
            if sol is None:
                break
            th, tn = _apply(T, sol[0], sol[1], c)
            r1 = float(np.sqrt(np.mean(r ** 2)))
            if th < 1e-5 and tn < 1e-4:
                break

    for gate in cfg["max_corr"]:
        for _ in range(int(cfg["iters_per_gate"])):
            c = T[:3, 3]
            p = pts_local @ T[:3, :3].T + c
            dist, idx = ref.tree.query(p, workers=-1,
                                       distance_upper_bound=float(gate))
            ok = np.isfinite(dist)
            if ok.sum() < int(cfg["min_corr"]):
                return T, int(ok.sum()), np.nan, r0, r1
            pw = p[ok]
            n = ref.nrm[idx[ok]]
            r = np.einsum("ij,ij->i", pw - ref.pts[idx[ok]], n)
            sol = _solve(pw, c, n, r, huber)
            if sol is None:
                break
            th, tn = _apply(T, sol[0], sol[1], c)
            if th < 1e-5 and tn < 1e-4:
                break
    w = np.minimum(1.0, huber / np.maximum(np.abs(r), 1e-9))
    return T, int(ok.sum()), float(np.sqrt(np.mean((r * w) ** 2))), r0, r1


def build_reference(bag, topic, times, T, cfg, s):
    """Accumulate the reference map from a set of poses."""
    acc = VoxelAvg(cfg["target_voxel"])
    n = 0
    for j, p in iter_scans(bag, topic, times, s["time_tol"],
                           s["lidar_min"], s["lidar_max"], T, cfg["deskew"],
                           cfg["deskew_bins"]):
        acc.add(p @ T[j][:3, :3].T + T[j][:3, 3])
        n += 1
        if n % 1000 == 0:
            print(f"    {n} scans accumulated", flush=True)
    pts = acc.points()
    print(f"    reference map: {n} scans -> {len(pts)} voxel centroids "
          f"@ {cfg['target_voxel']} m")
    return pts


def resume_point(P, base, times):
    """(k, poses) of the last round written so far -- traj_lidar_refined_r1,
    _r2, ... in a row -- or None. Its stamps must be the seed's: a file
    from another seed trajectory is not continued."""
    k, path = 0, None
    while os.path.exists(P.outp(f"{base[0]}_r{k + 1}{base[1]}")):
        k += 1
        path = P.outp(f"{base[0]}_r{k}{base[1]}")
    if not k:
        return None
    t, T = load_traj(path)
    if len(t) != len(times) or not np.allclose(t, times, atol=1e-6):
        raise SystemExit(f"{path} has other stamps than the seed trajectory "
                         f"({len(t)} vs {len(times)} poses): it was refined from "
                         f"another seed. --restart starts again from this seed")
    return k, T


def rounds_from_files(P, base, T_seed, k, known):
    """Correction statistics of rounds 1..k rebuilt from the trajectories:
    how far each round moved the poses it refined (poses it left alone,
    rejected or without a scan, are equal in both files and not counted).
    The residual is not in the files: nan."""
    have = {int(r[0]): tuple(r) for r in known}
    prev, out = T_seed, []
    for r in range(1, k + 1):
        _, T = load_traj(P.outp(f"{base[0]}_r{r}{base[1]}"))
        if r in have:
            out.append(have[r])
        else:
            d = np.linalg.norm(T[:, :3, 3] - prev[:, :3, 3], axis=1)
            d = d[d > 2e-6] * 100            # files keep 1e-6 m
            out.append((r, float(np.median(d)) if d.size else 0.0,
                        float(np.percentile(d, 95)) if d.size else 0.0, float("nan")))
        prev = T
    return out


def load_rounds(path):
    """Per-round statistics of earlier runs (for the convergence table)."""
    try:
        with open(path) as f:
            return json.load(f)["rounds"]
    except (OSError, ValueError, KeyError):
        return []


def save_rounds(path, history, seed):
    with open(path, "w") as f:
        json.dump({"seed": seed, "columns": ["round", "median_corr_cm", "p95_corr_cm",
                                             "median_residual_cm"],
                   "rounds": [list(r) for r in history]}, f, indent=1)


def print_convergence(history):
    print("\n=== convergence (report this) ===")
    print(f"{'round':>6} | {'median corr':>12} | {'p95 corr':>10} | "
          f"{'median residual':>16}")
    for r, m, p, res in history:
        rs = f"{res:13.2f} cm" if np.isfinite(res) else f"{'n/a':>16}"
        print(f"{r:6d} | {m:9.2f} cm | {p:7.2f} cm | {rs}")
    print("\nA round that moves poses by a few millimetres means the "
          "trajectory and the map are self-consistent.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", nargs="?", default="pipeline_config.json")
    ap.add_argument("--rounds", type=int, default=None,
                    help="rounds in total; rounds already written are not redone")
    ap.add_argument("--restart", action="store_true",
                    help="start again from round 1 instead of continuing")
    args = ap.parse_args()

    P = load_pipeline(args.config)
    cfg_all = P.cfg
    ds = P.dataset
    s = cfg_all["01_build_map"]
    c = dict(map=P.pcd("denoised"), target_voxel=0.05, scan_voxel=0.10,
             max_corr=[0.4, 0.2, 0.1], iters_per_gate=5, huber=0.05,
             min_corr=200, max_shift=0.50, max_rot_deg=5.0, rounds=2,
             plane_voxel=0.4, plane_iters=8, prior_beta=0.05,
             plane_huber=0.10, planarity=1.0,
             points_topic="", output="traj_lidar_refined.txt")
    # deskew as 01_build_map does, so the map and the scans registered to it
    # place points the same way (01a_refine.deskew overrides)
    c.update(deskew=bool(s.get("deskew", True)), deskew_bins=max(2, int(s.get("deskew_bins", 100))))
    c.update(cfg_all.get("01a_refine", {}))
    if args.rounds is not None:
        c["rounds"] = args.rounds
    bag = ds["bag"]
    print(P.describe())

    seed = P.outp(ds["traj"])
    times, T_seed = load_traj(seed)
    print(f"seed trajectory: {seed}  ({len(times)} poses)")
    topic = detect_points_topic(bag, c["points_topic"], len(times))
    print(f"points topic: {topic}")

    T_cur = T_seed.copy()
    history = []
    first, base = 1, os.path.splitext(c["output"])
    log_path = P.outp(f"{base[0]}_rounds.json")
    if not args.restart:
        done = resume_point(P, base, times)
        if done:
            k, T_cur = done
            history = [tuple(r) for r in load_rounds(log_path) if r[0] <= k]
            if len(history) < k:     # rounds from before the statistics were kept
                history = rounds_from_files(P, base, T_seed, k, history)
            if k >= int(c["rounds"]):
                print(f"\n{k} rounds already done ({P.outp(f'{base[0]}_r{k}{base[1]}')}); "
                      f"--rounds {k + 1} continues, --restart starts again")
                print_convergence(history)
                return
            first = k + 1
            print(f"\ncontinuing after round {k}: poses from "
                  f"{P.outp(f'{base[0]}_r{k}{base[1]}')}")
    for rnd in range(first, int(c["rounds"]) + 1):
        print(f"\n=== round {rnd}/{c['rounds']} ===")
        map_path = P.outp(c["map"]) if c["map"] else ""
        if rnd == 1 and map_path and os.path.exists(map_path):
            print(f"[ref] loading {map_path}")
            pc = o3d.io.read_point_cloud(map_path)
            print(f"    {len(pc.points)} pts -> downsampling to "
                  f"{c['target_voxel']} m")
            pts = np.asarray(pc.voxel_down_sample(c["target_voxel"]).points)
            del pc
        else:
            print("[ref] accumulating reference map from current poses")
            pts = build_reference(bag, topic, times, T_cur, c, s)
        ref = Reference(pts, plane_voxel=float(c["plane_voxel"]),
                        planarity=float(c["planarity"]))
        del pts

        print("[icp] registering scans")
        t0 = time.time()
        T_new = T_cur.copy()
        shifts, rots, rms_all, rms_w, rms_f = [], [], [], [], []
        n_done = n_rej = n_thin = 0
        max_rot = np.deg2rad(float(c["max_rot_deg"]))
        for j, p in iter_scans(bag, topic, times, s["time_tol"],
                               s["lidar_min"], s["lidar_max"], T_cur, c["deskew"],
                               c["deskew_bins"]):
            pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(p))
            p_ds = np.asarray(pc.voxel_down_sample(c["scan_voxel"]).points)
            del pc
            # seed, never the previous scan's result: no error accumulation
            T_j, n_corr, rms, rw, rf = register(ref, p_ds, T_cur[j].copy(), c)
            if n_corr < int(c["min_corr"]):
                n_thin += 1
                continue
            d = np.linalg.norm(T_j[:3, 3] - T_cur[j][:3, 3])
            dR = T_j[:3, :3] @ T_cur[j][:3, :3].T
            ang = np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1))
            if d > float(c["max_shift"]) or ang > max_rot:
                n_rej += 1
                continue
            T_new[j] = T_j
            shifts.append(d)
            rots.append(np.degrees(ang))
            rms_all.append(rms)
            rms_w.append(rw)
            rms_f.append(rf)
            n_done += 1
            if n_done % 500 == 0:
                print(f"    {n_done} scans, median shift "
                      f"{np.median(shifts) * 100:.2f} cm "
                      f"({time.time() - t0:.0f}s)", flush=True)

        if not shifts:
            raise SystemExit(
                f"round {rnd}: no scan registered ({n_thin} had too few "
                f"correspondences, {n_rej} were rejected).\n"
                f"  - wrong points topic? currently {topic!r}\n"
                f"  - seed poses not aligned with the reference map?\n"
                f"  - try a larger first entry in 01a_refine.max_corr")
        sh = np.array(shifts) * 100
        ro = np.array(rots)
        rr = np.array([r for r in rms_all if np.isfinite(r)]) * 100
        print(f"\n[round {rnd}] {n_done} refined, {n_rej} rejected "
              f"(over max_shift/max_rot), {n_thin} too few correspondences")
        print(f"    correction  median {np.median(sh):.2f} cm   "
              f"p95 {np.percentile(sh, 95):.2f} cm   max {sh.max():.2f} cm")
        print(f"    rotation    median {np.median(ro):.3f} deg  "
              f"p95 {np.percentile(ro, 95):.3f} deg")
        print(f"    residual    median {np.median(rr):.2f} cm  "
              f"(point-to-plane, robust RMS)")
        rwv = np.array([v for v in rms_w if np.isfinite(v)]) * 100
        rfv = np.array([v for v in rms_f if np.isfinite(v)]) * 100
        if rwv.size and rfv.size:
            print(f"    plane resid median {np.median(rwv):.2f} cm before -> "
                  f"{np.median(rfv):.2f} cm after (offset from the local "
                  f"plane fit; this is what the smear is made of)")
        history.append((rnd, float(np.median(sh)), float(np.percentile(sh, 95)),
                        float(np.median(rr))))
        T_cur = T_new
        del ref

        out_r = P.outp(f"{base[0]}_r{rnd}{base[1]}")
        write_traj(out_r, times, T_cur)
        out = P.outp(c["output"])
        write_traj(out, times, T_cur)
        save_rounds(log_path, history, seed)
        print(f"    wrote {out_r} (and {out})")

    print_convergence(history)
    print(f"\nnext: set dataset.traj to \"{c['output']}\" (found in "
          f"{P.folder_for(c['output'])}), delete "
          f"{', '.join(P.pcd(b) for b in ('merged', 'static', 'denoised', 'colored'))} "
          f"in {P.folder_for('x.pcd')}, re-run 01_build_map.py, "
          f"then analysis/mapping/map_quality.py {P.outp(P.pcd('denoised'))}")


if __name__ == "__main__":
    main()

