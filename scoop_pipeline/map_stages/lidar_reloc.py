#!/usr/bin/env python3
"""
LiDAR relocalization of a run's start in a reference pass's anchored map,
with the checks that say whether to trust it. Called by 06_init_from_boards.py
for a run (dataset.reference_pass); 08 then seeds from it.

GLIM starts every run in its own world, so the run and the reference map
differ by one rigid transform, T_map_glim. It is found from the LiDAR alone:

  1. window: the scans of the first `window_s` seconds of the GLIM trajectory,
     deskewed and placed through their GLIM poses (locally exact), as one
     cloud in the GLIM world;
  2. global: both frames are gravity-aligned (GLIM's; 03 keeps z), so the
     search is over yaw and x/y: the wall band (0.2-2.5 m above each cloud's
     floor) of the window, turned through every yaw_step_deg, is correlated
     (FFT) with the map's wall band on a grid_m grid; z from the two floors;
  3. the best `candidates` peaks (1 m / 10 deg apart) each polished by
     point-to-plane ICP, coarse to fine; one that ends tilted more than
     max_tilt_deg is dropped; the most window points on the map (within
     fit_m) wins.

Then it is CHECKED, and refused if a check fails:

  fit         share of window points within fit_m of the map's surface
              (along its normal), and their rms;
  unique      a second fit, metres or degrees away, must leave clearly more of
              the window off the map (a symmetric building fits two ways);
  repeat      a second window, check_after_s later, relocalized on its own,
              must agree with the first carried forward by GLIM;
  board       (06, after this) the ZED's anchor-board dwell -- not used in
              the fit -- must agree with the LiDAR start pose;
  look        mapping/reloc_check_<tag>.ply: the window (red) on the map
              (grey) around the start, for a viewer.

Config: "06_init": {"relocalize": {...}}, all optional:
  "enabled":       false (the start comes from the board dwell unless this is on)
  "anchor_cam":    "zed"        # the camera entry 08 anchors on
  "run_traj":      <work>/<date>/<pass>/<machine>/glim/traj_lidar.txt
  "map":           the reference pass's map_final_<tag>_anchored.pcd
  "window_s":      5.0,  "check_after_s": 60.0
  "voxel":         0.10, "grid_m": 0.20, "yaw_step_deg": 2.0, "candidates": 6
  "max_tilt_deg":  3.0,  "fit_m": 0.05, "min_fitness": 0.5
  "rival_ratio":   2.0, "rival_gap": 0.05   # a distinct fit must leave >= 2x and
                                            # >= 5 points more of the window off the map
  "check_max_cm":  10.0, "check_max_deg": 0.5
"""
import importlib
import os
import sys

import numpy as np

from pipeline_common import traj_quats, interp_poses

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DEFAULTS = dict(window_s=5.0, check_after_s=60.0, voxel=0.10, grid_m=0.20,
                yaw_step_deg=2.0, candidates=6, max_tilt_deg=3.0, fit_m=0.05, min_fitness=0.5,
                rival_ratio=2.0, rival_gap=0.05, check_max_cm=10.0, check_max_deg=0.5,
                anchor_cam="zed")


def _o3d():
    import open3d as o3d
    return o3d, o3d.pipelines.registration


def tilt_deg(R):
    return float(np.degrees(np.arccos(np.clip(R[2, 2], -1.0, 1.0))))


def yaw_deg(R):
    return float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))


def ang_deg(R):
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1.0, 1.0))))


def pose_at(tr_t, tr_T, tr_q, t):
    R, p = interp_poses(tr_t, tr_T, tr_q, np.array([t], float))
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R[0], p[0]
    return T


def _cloud(pts, voxel, normals=True):
    o3d, _ = _o3d()
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.asarray(pts, float)))
    pc = pc.voxel_down_sample(voxel)
    if normals:
        pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 2.5,
                                                                 max_nn=30))
    return pc


def _icp(src, tgt, T0, voxel):
    _, REG = _o3d()
    T = T0
    for d in (10 * voxel, 5 * voxel, 2.5 * voxel, voxel):
        T = REG.registration_icp(src, tgt, d, T, REG.TransformationEstimationPointToPlane(),
                                 REG.ICPConvergenceCriteria(max_iteration=40)).transformation
    return T


def window_cloud(bag, topic, tr_t, tr_T, s01, t_from, t_to):
    """Deskewed scans with GLIM pose times in [t_from, t_to], in the GLIM world."""
    a = importlib.import_module("01a_refine_poses")
    pts, n = [], 0
    for j, p in a.iter_scans(bag, topic, tr_t, s01["time_tol"], s01["lidar_min"],
                             s01["lidar_max"], tr_T, bool(s01.get("deskew", True)),
                             max(2, int(s01.get("deskew_bins", 100)))):
        if tr_t[j] < t_from:
            continue
        if tr_t[j] > t_to:
            break
        pts.append(p @ tr_T[j][:3, :3].T + tr_T[j][:3, 3])
        n += 1
    if not pts:
        raise SystemExit("relocalize: no scans between %.1f and %.1f s of the run "
                         "trajectory" % (t_from - tr_t[0], t_to - tr_t[0]))
    return np.concatenate(pts), n


def _floor(z):
    return float(np.percentile(z, 1.0))


def wall_band(pts, lo=0.2, hi=2.5):
    """Points between lo and hi above the cloud's own floor: walls and
    furniture, without the floor and the ceiling that carry no x/y."""
    f = _floor(pts[:, 2])
    return pts[(pts[:, 2] > f + lo) & (pts[:, 2] < f + hi)], f


class MapGrid:
    """The map's wall band as an occupancy grid, dilated one cell, ready for
    FFT correlation against a window."""

    def __init__(self, pts, g):
        from scipy import ndimage
        band, self.floor = wall_band(pts)
        self.g = g
        self.x0 = band[:, :2].min(0) - 2.0
        ij = np.floor((band[:, :2] - self.x0) / g).astype(int)
        n = ij.max(0) + 3
        M = np.zeros(n, np.float32)
        M[ij[:, 0], ij[:, 1]] = 1.0
        self.M = ndimage.maximum_filter(M, size=3)


def search_2d(win_pts, grid, c):
    """Every yaw (yaw_step_deg) x every x/y offset: the share of the window's
    wall cells that land on the map's walls. -> peaks [(score, T0)], best
    first, at least 1 m / 10 deg apart. T0 has z from the two floors."""
    from scipy.signal import fftconvolve
    g = grid.g
    band, fw = wall_band(win_pts)
    cw = band[:, :2].mean(0)
    rel0 = band[:, :2] - cw
    peaks = []
    for yaw in np.arange(0.0, 360.0, float(c["yaw_step_deg"])):
        cy, sy = np.cos(np.radians(yaw)), np.sin(np.radians(yaw))
        rel = rel0 @ np.array([[cy, sy], [-sy, cy]])          # R(yaw) applied
        w0 = rel.min(0)
        ij = np.floor((rel - w0) / g).astype(int)
        W = np.zeros(ij.max(0) + 1, np.float32)
        W[ij[:, 0], ij[:, 1]] = 1.0
        nW = float(W.sum())
        C = fftconvolve(grid.M, W[::-1, ::-1], mode="full") / nW
        for _ in range(3):                                     # 3 peaks per yaw
            k = np.unravel_index(int(np.argmax(C)), C.shape)
            sc = float(C[k])
            if sc <= 0:
                break
            shift = np.array(k) - (np.array(W.shape) - 1)
            T0 = np.eye(4)
            T0[:2, :2] = [[cy, -sy], [sy, cy]]
            T0[:2, 3] = -T0[:2, :2] @ cw + grid.x0 - w0 + shift * g
            T0[2, 3] = grid.floor - fw
            peaks.append((sc, T0))
            r = int(round(1.0 / g))
            C[max(k[0] - r, 0):k[0] + r + 1, max(k[1] - r, 0):k[1] + r + 1] = 0
    peaks.sort(key=lambda x: -x[0])
    out = []
    for sc, T0 in peaks:
        if all(distinct(T0, T1) for _, T1 in out):
            out.append((sc, T0))
        if len(out) >= int(c["candidates"]):
            break
    return out


def register(src_pts, tgt, grid, c, label, score):
    """2-D wall correlation over every heading (both frames are gravity-
    aligned), then point-to-plane ICP from the best few, scored by the share
    of window points on the map. -> (candidates [(T, fitness, rmse)], best
    first; src cloud)."""
    _, REG = _o3d()
    v = float(c["voxel"])
    src = _cloud(src_pts, v)
    fine = np.asarray(_cloud(src_pts, v / 2, normals=False).points)
    peaks = search_2d(src_pts, grid, c)
    print("    %s: %d headings x every offset -> best wall overlap %s"
          % (label, int(round(360 / float(c["yaw_step_deg"]))),
             ", ".join("%.2f" % sc for sc, _ in peaks)))
    cands = []
    for k, (sc, T0) in enumerate(peaks):
        T = _icp(src, tgt, T0, v)
        fit, rmse = score(fine, T, float(c["fit_m"]))
        print("    %s candidate %d (overlap %.2f): on map %5.1f %%  rmse %.1f cm  t=%s  "
              "yaw %.1f deg  tilt %.2f deg"
              % (label, k, sc, fit * 100, rmse * 100,
                 np.round(T[:3, 3], 2).tolist(), yaw_deg(T[:3, :3]), tilt_deg(T[:3, :3])))
        if tilt_deg(T[:3, :3]) <= float(c["max_tilt_deg"]):
            cands.append((T, fit, rmse))
        else:
            print("      tilt over %.1f deg -> dropped" % float(c["max_tilt_deg"]))
    cands.sort(key=lambda x: -x[1])
    return cands, src


class Scorer:
    """How well a window sits on the map: each window point (at voxel/2)
    against the nearest map point's PLANE -- the distance along its normal,
    so the voxel spacing of either cloud does not count as misfit."""

    def __init__(self, tgt):
        from scipy.spatial import cKDTree
        self.q = np.asarray(tgt.points)
        self.n = np.asarray(tgt.normals)
        self.tree = cKDTree(self.q)

    def __call__(self, pts, T, fit_m, reach=0.3):
        p = pts @ T[:3, :3].T + T[:3, 3]
        d, i = self.tree.query(p, distance_upper_bound=reach, workers=-1)
        ok = np.isfinite(d)
        r = np.full(len(p), np.inf)
        r[ok] = np.abs(np.einsum("ij,ij->i", p[ok] - self.q[i[ok]], self.n[i[ok]]))
        good = r < fit_m
        rmse = float(np.sqrt(np.mean(r[good] ** 2))) if good.any() else float("nan")
        return float(good.mean()), rmse


def distinct(Ta, Tb):
    return (np.linalg.norm(Ta[:3, 3] - Tb[:3, 3]) > 1.0
            or ang_deg(Ta[:3, :3].T @ Tb[:3, :3]) > 10.0)


def relocalize(P, T_lidar_cam=None):
    """-> dict: t0, T_map_glim, T_map_lidar0 (and T_map_cam0 with T_lidar_cam),
    the run trajectory, and every check's numbers. SystemExit when a check fails."""
    o3d, REG = _o3d()
    a = importlib.import_module("01a_refine_poses")
    c = dict(DEFAULTS)
    c.update((P.cfg.get("06_init") or {}).get("relocalize") or {})
    s01 = P.cfg["01_build_map"]
    bag = os.path.expanduser(P.dataset["bag"])
    run_traj = os.path.expanduser(c.get("run_traj") or os.path.join(
        os.path.dirname(os.path.normpath(bag)), P.machine, "glim", "traj_lidar.txt"))
    ref_map = os.path.expanduser(c.get("map") or (P.ref_file("anchored_map") if P.reference
                                                  else P.outp("map_final_{tag}_anchored.pcd")))
    for f, what in ((run_traj, "run trajectory (06_init.relocalize.run_traj)"),
                    (ref_map, "anchored reference map (06_init.relocalize.map)")):
        if not os.path.exists(f):
            raise SystemExit("relocalize: %s not found: %s" % (what, f))
    tr_t, tr_T = a.load_traj(run_traj)
    tr_q = traj_quats(tr_T)
    topic = a.detect_points_topic(bag, P.cfg.get("01a_refine", {}).get("points_topic", ""),
                                  len(tr_t))
    print("\n=== LiDAR relocalization in the reference map ===")
    print("  run trajectory: %s (%d poses, %.1f s)" % (run_traj, len(tr_t), tr_t[-1] - tr_t[0]))
    print("  reference map:  %s" % ref_map)

    ref = o3d.io.read_point_cloud(ref_map)
    print("  %d map points" % len(ref.points))
    v = float(c["voxel"])
    tgt = ref.voxel_down_sample(v)
    tgt.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=v * 2.5, max_nn=30))
    grid = MapGrid(np.asarray(tgt.points), float(c["grid_m"]))
    score = Scorer(tgt)
    print("  map wall grid: %d x %d cells of %.2f m" % (grid.M.shape + (float(c["grid_m"]),)))
    ref_pts = np.asarray(tgt.points)
    del ref

    # ---- 1-3: the start window
    t0 = tr_t[0]
    w0, n0 = window_cloud(bag, topic, tr_t, tr_T, s01, t0, t0 + float(c["window_s"]))
    print("  start window: %d scans over %.1f s -> %d points" % (n0, float(c["window_s"]), len(w0)))
    cands, src0 = register(w0, tgt, grid, c, "start", score)
    if not cands:
        raise SystemExit("relocalize: every candidate tilted the run more than %.1f deg "
                         "after ICP -- wrong map, or the start sees too little of it; "
                         "refused" % float(c["max_tilt_deg"]))
    T, fit, rmse = cands[0]
    checks = {"fitness": fit, "rmse_cm": rmse * 100, "fit_m": float(c["fit_m"])}
    print("\n  [fit]    %.1f %% of the window within %.0f cm of the map, rmse %.2f cm  (min %.0f %%)"
          % (fit * 100, float(c["fit_m"]) * 100, rmse * 100, float(c["min_fitness"]) * 100))
    if fit < float(c["min_fitness"]):
        raise SystemExit("relocalize: only %.1f %% of the start window lies on the map "
                         "-- refused" % (fit * 100))

    rival = next(((T2, f2) for T2, f2, _ in cands[1:] if distinct(T, T2)), None)
    if rival is not None:
        dt = np.linalg.norm(rival[0][:3, 3] - T[:3, 3])
        da = ang_deg(rival[0][:3, :3].T @ T[:3, :3])
        # what tells two fits apart is what each leaves OFF the map: walls,
        # floor and ceiling fit a symmetric room either way, the things that
        # break the symmetry fit only one way
        off, roff = 1.0 - fit, 1.0 - rival[1]
        need = max(float(c["rival_ratio"]) * off, off + float(c["rival_gap"]))
        print("  [unique] next distinct fit, %.1f m / %.0f deg away, leaves %.1f %% of the "
              "window off the map vs %.1f %% for this one (must be >= %.1f %%)"
              % (dt, da, roff * 100, off * 100, need * 100))
        checks.update(rival_off=roff, best_off=off, rival_m=dt, rival_deg=da)
        if roff < need:
            raise SystemExit("relocalize: a different fit sits on the map almost as well -- "
                             "the start is ambiguous (symmetric place); refused")
    else:
        print("  [unique] every other candidate converged to this same fit")

    T_map_glim = T
    T_map_lidar0 = T_map_glim @ tr_T[0]

    # ---- repeat: a second, independent window
    t1 = t0 + float(c["check_after_s"])
    if tr_t[-1] > t1 + float(c["window_s"]):
        w1, n1 = window_cloud(bag, topic, tr_t, tr_T, s01, t1, t1 + float(c["window_s"]))
        print("\n  repeat window at +%.0f s: %d scans -> %d points" % (t1 - t0, n1, len(w1)))
        c1, _ = register(w1, tgt, grid, c, "repeat", score)
        if not c1:
            raise SystemExit("relocalize: the repeat window found no fit -- refused")
        T1 = c1[0][0]
        Tg = pose_at(tr_t, tr_T, tr_q, t1)
        A, B = T_map_glim @ Tg, T1 @ Tg
        dcm = np.linalg.norm(A[:3, 3] - B[:3, 3]) * 100
        ddeg = ang_deg(A[:3, :3].T @ B[:3, :3])
        print("  [repeat] start fit carried %.0f s by GLIM vs the window's own fit: "
              "%.1f cm / %.3f deg  (max %.0f cm / %.2f deg)"
              % (t1 - t0, dcm, ddeg, float(c["check_max_cm"]), float(c["check_max_deg"])))
        checks.update(repeat_after_s=t1 - t0, repeat_cm=dcm, repeat_deg=ddeg,
                      repeat_fitness=c1[0][1])
        if dcm > float(c["check_max_cm"]) or ddeg > float(c["check_max_deg"]):
            raise SystemExit("relocalize: the two windows disagree -- refused")
    else:
        print("\n  [repeat] run shorter than check_after_s + window_s: skipped")

    # ---- look: the window on the map around the start
    near = np.linalg.norm(ref_pts - T_map_lidar0[:3, 3], axis=1) < 25.0
    win = np.asarray(src0.transform(T_map_glim).points)
    look = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(
        np.vstack([ref_pts[near], win])))
    look.colors = o3d.utility.Vector3dVector(np.vstack([
        np.full((int(near.sum()), 3), 0.6), np.tile([1.0, 0.0, 0.0], (len(win), 1))]))
    look_p = P.outp("reloc_check_{tag}.ply")
    o3d.io.write_point_cloud(look_p, look)
    print("  [look]   %s (red: the start window, grey: the map within 25 m)" % look_p)

    print("\n  map <- GLIM world: t=%s m  yaw %.2f deg  tilt %.3f deg"
          % (np.round(T_map_glim[:3, 3], 4).tolist(), yaw_deg(T_map_glim[:3, :3]),
             tilt_deg(T_map_glim[:3, :3])))
    print("  LiDAR at the start (t=%.3f) in map: xyz=%s yaw %.2f deg"
          % (t0, np.round(T_map_lidar0[:3, 3], 4).tolist(), yaw_deg(T_map_lidar0[:3, :3])))
    out = {"t0": float(t0), "T_map_glim": T_map_glim, "T_map_lidar0": T_map_lidar0,
           "run_traj": run_traj, "ref_map": ref_map, "checks": checks, "look": look_p,
           "anchor_cam": c["anchor_cam"], "tr": (tr_t, tr_T, tr_q)}
    if T_lidar_cam is not None:
        out["T_map_cam0"] = T_map_lidar0 @ np.asarray(T_lidar_cam, float)
    return out


def board_check(r, hits, T_lidar_cam):
    """The board dwell (06's camera-mode hits) against the LiDAR start pose at
    the same times: -> (cm, deg, n). Neither enters the other."""
    tr_t, tr_T, tr_q = r["tr"]
    d_t, d_r = [], []
    for h in hits:
        if not (tr_t[0] <= h["t"] <= tr_t[-1]):
            continue
        T_l = r["T_map_glim"] @ pose_at(tr_t, tr_T, tr_q, h["t"]) @ np.asarray(T_lidar_cam, float)
        d_t.append(np.linalg.norm(T_l[:3, 3] - h["T_map_cam"][:3, 3]) * 100)
        d_r.append(ang_deg(T_l[:3, :3].T @ h["T_map_cam"][:3, :3]))
    if not d_t:
        return None
    return float(np.median(d_t)), float(np.median(d_r)), len(d_t)
