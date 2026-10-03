#!/usr/bin/env python3
"""
STAGE 04 - a run's LiDAR trajectory in map: every scan registered to the
reference pass's FROZEN anchored map.

  seed      the run's GLIM trajectory (its own world), placed in map by 03's
            session anchor: T_map_glim = T_map_cam(t_a) @ inv(T_glim_lidar(t_a)
            @ T_lidar_cam), t_a = the anchor's time. Nothing else -- no camera
            odometry.
  per scan  each scan, deskewed through the current poses, starts from ITS OWN
            seed pose (never the previous scan's result, so errors cannot
            accumulate) and is registered to the map by 02's ICP: planes
            fitted over plane_voxel cells first, then nearest-neighbour
            point-to-plane, both anchored to the seed (prior_beta) where the
            map leaves a direction open, so a corridor scan cannot slide
            along it on noise. A correction larger than max_shift / max_rot_deg,
            or a scan with fewer than min_corr correspondences, keeps its seed.
  rounds    round 2 on deskews with the previous round's poses; the map never
            changes. The per-round corrections are the convergence evidence.

Outputs (odometry/reference_<tag>/):
  traj_<name>.tum         T_map_lidar per scan (03, 04, 09 read it)
  traj_<name>_in_cam.tum  the camera optical frame (T_lidar_camera), for 10
  quality_<name>.csv      per scan: t, status (ok / seed_kept_far / seed_kept_thin
                          / no_scan), correspondences, residual, correction, the
                          direction the map pins least (weak_x/y/z, in map), its
                          share of the constraint (box room ~0.33, corridor ~0), the
                          seed's share of the result along it, seed_held (>= 0.5)
  seed_held_<name>.csv    runs of seed-held scans: when, where, how long, the open
                          direction, and how far the GLIM seed is off along it at
                          the run's edges (the map's correction of the constrained
                          scans either side): inside, the error runs between them
  summary_<name>.json     the numbers below

Config "04_reference" (all optional):
  "name":        "<machine>_lidar"
  "points_topic": the LiDAR cloud topic (default: detected)
  "run_traj":    <work>/<date>/<pass>/<machine>/glim/traj_lidar.txt
  "anchor_cam":  "zed"       # the session_anchor.json camera to place the seed by
  "ref_map":     the reference pass's map_final_<tag>_anchored.pcd
  "rounds": 2, "target_voxel": 0.05, "scan_voxel": 0.10, "plane_voxel": 0.4,
  "max_corr": [0.4, 0.2, 0.1], "iters_per_gate": 5, "huber": 0.05,
  "plane_iters": 8, "plane_huber": 0.10, "prior_beta": 0.05, "planarity": 1.0,
  "min_corr": 200, "max_shift": 0.5, "max_rot_deg": 5.0,
  "seed_held_at": 0.5, "segment_gap_s": 1.0,
  "deskew": 01_build_map.deskew, "deskew_bins": 01_build_map.deskew_bins

  python3 04_reference_traj.py pipeline_config_<run>.json
"""
import importlib
import json
import os
import sys
import time

import numpy as np
import open3d as o3d

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline_common import load_pipeline, make_T, traj_quats, interp_poses   # noqa: E402

A = importlib.import_module("02_refine_poses")

DEFAULTS = dict(rounds=2, target_voxel=0.05, scan_voxel=0.10, plane_voxel=0.4,
                max_corr=[0.4, 0.2, 0.1], iters_per_gate=5, huber=0.05, plane_iters=8,
                plane_huber=0.10, prior_beta=0.05, planarity=1.0, min_corr=200,
                max_shift=0.5, max_rot_deg=5.0, anchor_cam="zed", points_topic="",
                seed_held_at=0.5, segment_gap_s=1.0)


def pose_at(tr_t, tr_T, t):
    R, p = interp_poses(tr_t, tr_T, traj_quats(tr_T), np.array([t], float))
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R[0], p[0]
    return T


def seed_from_anchor(sa_path, cam, tr_t, tr_T, T_lc):
    """map <- GLIM world from 03's session anchor of camera `cam`."""
    sa = json.load(open(sa_path))
    cams = sa.get("cameras", {})
    if cam not in cams or "map_to_cam" not in cams[cam]:
        raise SystemExit("%s has no camera '%s' (have %s): run 03 first"
                         % (sa_path, cam, sorted(cams)))
    rec = cams[cam]
    T_map_cam = make_T(rec["map_to_cam"]["xyz"], rec["map_to_cam"]["qxyzw"])
    t_a = float(rec.get("dwell_t_end") or tr_t[0])
    if not (tr_t[0] <= t_a <= tr_t[-1]):
        raise SystemExit("session anchor time %.3f is outside the run trajectory "
                         "[%.3f, %.3f]" % (t_a, tr_t[0], tr_t[-1]))
    T = T_map_cam @ np.linalg.inv(pose_at(tr_t, tr_T, t_a) @ T_lc)
    return T, t_a, rec.get("mode", "?")


def register_round(ref, bag, topic, times, T_cur, c, s01):
    """One pass over the scans -> (new poses, per-scan rows, stats)."""
    T_new = T_cur.copy()
    rows = {}
    max_rot = np.deg2rad(float(c["max_rot_deg"]))
    t0 = time.time()
    n = 0
    for j, p in A.iter_scans(bag, topic, times, s01["time_tol"], s01["lidar_min"],
                             s01["lidar_max"], T_cur, c["deskew"], c["deskew_bins"]):
        pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(p))
        p_ds = np.asarray(pc.voxel_down_sample(c["scan_voxel"]).points)
        T_j, n_corr, rms, _, _ = A.register(ref, p_ds, T_cur[j].copy(), c)
        T_fit = T_j if np.isfinite(T_j).all() else T_cur[j]
        v, share, _ = A.constraint(ref, p_ds, T_fit, float(c["max_corr"][-1]))
        d = float(np.linalg.norm(T_j[:3, 3] - T_cur[j][:3, 3]))
        dR = T_j[:3, :3] @ T_cur[j][:3, :3].T
        ang = float(np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1)))
        if not np.isfinite(T_j).all():
            status, d, ang = "seed_kept_thin", float("nan"), float("nan")
        elif n_corr < int(c["min_corr"]):
            status = "seed_kept_thin"
        elif d > float(c["max_shift"]) or ang > max_rot:
            status = "seed_kept_far"
        else:
            status = "ok"
            T_new[j] = T_j
        # the seed's share of the result along the weakest direction: the
        # prior adds beta/3 of the data's total weight in every direction,
        # the map adds `share` of it along v
        k = float(c["prior_beta"]) / 3.0
        seed_share = k / (k + share) if np.isfinite(share) else 1.0
        rows[j] = (status, n_corr, rms, d, np.degrees(ang), v, share, seed_share)
        n += 1
        if n % 1000 == 0:
            ok = sum(1 for r in rows.values() if r[0] == "ok")
            print("    %d scans (%d registered) %.0f s" % (n, ok, time.time() - t0), flush=True)
    return T_new, rows


def held_segments(times, T_fin, T_seed, rows, held_at, gap_s):
    """Runs of seed-held scans (gaps up to gap_s s bridged). For each: where,
    how long, its open direction, and how far the GLIM seed is off along it at
    the run's two edges -- the map's correction (final - seed) along that
    direction at the constrained scans just before and just after. Inside the
    run the seed holds that direction, so its error there goes from the one
    edge value to the other."""
    held = sorted(j for j, r in rows.items() if r[7] >= held_at)
    good = np.array(sorted(j for j, r in rows.items() if r[7] < held_at and r[0] == "ok"))
    corr = T_fin[:, :3, 3] - T_seed[:, :3, 3]
    runs, cur = [], []
    for j in held:
        if cur and times[j] - times[cur[-1]] > gap_s:
            runs.append(cur)
            cur = []
        cur.append(j)
    if cur:
        runs.append(cur)
    out = []
    for r in runs:
        V = np.array([rows[j][5] for j in r if np.isfinite(rows[j][5]).all()])
        if len(V):
            V = V * np.sign(V @ V[0])[:, None]          # one sign for +-v
            w = V.mean(0)
            w = w / max(np.linalg.norm(w), 1e-12)
        else:
            w = np.full(3, np.nan)
        before = good[good < r[0]]
        after = good[good > r[-1]]
        ok_w = np.isfinite(w).all()
        off0 = float(corr[before[-1]] @ w) * 100 if len(before) and ok_w else float("nan")
        off1 = float(corr[after[0]] @ w) * 100 if len(after) and ok_w else float("nan")
        P = T_fin[r, :3, 3]
        out.append({"t0": float(times[r[0]]), "t1": float(times[r[-1]]), "n": len(r),
                    "path_m": float(np.sum(np.linalg.norm(np.diff(P, axis=0), axis=1))),
                    "xyz": P.mean(0), "weak": w,
                    "seed_share": float(np.median([rows[j][7] for j in r])),
                    "off_before_cm": off0, "off_after_cm": off1,
                    "off_cm": float(np.nanmax(np.abs([off0, off1])))
                    if np.isfinite([off0, off1]).any() else float("nan")})
    return out


def main():
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "pipeline_config.json"
    P = load_pipeline(cfg_path)
    s01 = P.cfg["01_build_map"]
    c = dict(DEFAULTS)
    c.update(deskew=bool(s01.get("deskew", True)),
             deskew_bins=max(2, int(s01.get("deskew_bins", 100))))
    c.update(P.cfg.get("04_reference") or {})
    # both ICP stages anchored to the seed where the map leaves a direction open
    c.setdefault("nn_prior_beta", c["prior_beta"])
    name = c.get("name") or "%s_lidar" % P.machine
    bag = os.path.expanduser(c.get("bag") or P.dataset["bag"])
    run_traj = os.path.expanduser(c.get("run_traj") or os.path.join(
        os.path.dirname(os.path.normpath(bag)), P.machine, "glim", "traj_lidar.txt"))
    ref_map = (P.outp(c["ref_map"]) if c.get("ref_map") else
               P.ref_file("anchored_map") if P.reference else
               P.outp("map_final_{tag}_anchored.pcd"))
    sa_path = P.outp(c.get("session_anchor") or "session_anchor.json")
    outd = P.reference_dir()
    os.makedirs(outd, exist_ok=True)
    print(P.describe())
    for k, v in (("bag", bag), ("run trajectory", run_traj), ("session anchor", sa_path),
                 ("reference map", ref_map), ("out_dir", outd)):
        print("%-15s %s" % (k + ":", v))
    for f in (run_traj, sa_path, ref_map):
        if not os.path.exists(f):
            raise SystemExit("not found: %s" % f)

    T_lc = np.asarray(P.sensor.T_lidar_camera, float)
    times, T_glim = A.load_traj(run_traj)
    T_map_glim, t_a, mode = seed_from_anchor(sa_path, c["anchor_cam"], times, T_glim, T_lc)
    T_seed = np.einsum("ij,njk->nik", T_map_glim, T_glim)
    print("\nseed: %d GLIM poses (%.1f s) placed in map by 03's '%s' anchor (%s) at t=%.3f"
          % (len(times), times[-1] - times[0], c["anchor_cam"], mode, t_a))
    topic = A.detect_points_topic(bag, c["points_topic"], len(times))

    print("\n[ref] %s" % ref_map)
    pc = o3d.io.read_point_cloud(ref_map)
    print("    %d pts -> %.2f m voxels" % (len(pc.points), float(c["target_voxel"])))
    pts = np.asarray(pc.voxel_down_sample(float(c["target_voxel"])).points)
    del pc
    ref = A.Reference(pts, plane_voxel=float(c["plane_voxel"]), planarity=float(c["planarity"]))
    del pts

    T_cur, history, rows = T_seed.copy(), [], {}
    for rnd in range(1, int(c["rounds"]) + 1):
        print("\n=== round %d/%d: every scan to the frozen map ===" % (rnd, int(c["rounds"])))
        T_new, rows = register_round(ref, bag, topic, times, T_cur, c, s01)
        moved = [i for i, r in rows.items() if r[0] == "ok"]
        d = np.linalg.norm(T_new[moved, :3, 3] - T_cur[moved, :3, 3], axis=1) * 100
        res = np.array([rows[i][2] for i in moved if np.isfinite(rows[i][2])]) * 100
        counts = {k: sum(1 for r in rows.values() if r[0] == k)
                  for k in ("ok", "seed_kept_far", "seed_kept_thin")}
        print("  %d scans: %d registered, %d kept their seed (correction over max_shift/"
              "max_rot), %d too few correspondences"
              % (len(rows), counts["ok"], counts["seed_kept_far"], counts["seed_kept_thin"]))
        print("  correction this round: median %.2f cm, p95 %.2f cm | residual median %.2f cm"
              % (np.median(d), np.percentile(d, 95), np.median(res)))
        history.append({"round": rnd, "registered": counts["ok"], "scans": len(rows),
                        "median_corr_cm": float(np.median(d)),
                        "p95_corr_cm": float(np.percentile(d, 95)),
                        "median_residual_cm": float(np.median(res))})
        T_cur = T_new

    # ---- outputs
    p_lidar = os.path.join(outd, "traj_%s.tum" % name)
    p_cam = os.path.join(outd, "traj_%s_in_cam.tum" % name)
    A.write_traj(p_lidar, times, T_cur)
    A.write_traj(p_cam, times, np.einsum("nij,jk->nik", T_cur, T_lc))
    held_at = float(c["seed_held_at"])
    nan3 = np.full(3, np.nan)
    with open(os.path.join(outd, "quality_%s.csv" % name), "w") as f:
        f.write("t,status,n_corr,residual_cm,last_correction_cm,last_correction_deg,"
                "weak_x,weak_y,weak_z,weak_share,seed_share,seed_held\n")
        for i, t in enumerate(times):
            st, nc, rms, dd, da, v, sh, ss = rows.get(
                i, ("no_scan", 0, np.nan, np.nan, np.nan, nan3, np.nan, 1.0))
            f.write("%.9f,%s,%d,%.3f,%.3f,%.4f,%.4f,%.4f,%.4f,%.4f,%.3f,%d\n"
                    % (t, st, nc, rms * 100, dd * 100, da, v[0], v[1], v[2], sh, ss,
                       int(ss >= held_at)))
    segs = held_segments(times, T_cur, T_seed, rows, held_at, float(c["segment_gap_s"]))
    with open(os.path.join(outd, "seed_held_%s.csv" % name), "w") as f:
        f.write("t_start,t_end,duration_s,scans,path_m,x,y,z,weak_x,weak_y,weak_z,"
                "median_seed_share,seed_off_before_cm,seed_off_after_cm,seed_off_max_cm\n")
        for g in segs:
            f.write("%.3f,%.3f,%.2f,%d,%.2f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.2f,%.2f,%.2f\n"
                    % (g["t0"], g["t1"], g["t1"] - g["t0"], g["n"], g["path_m"], *g["xyz"],
                       *g["weak"], g["seed_share"], g["off_before_cm"], g["off_after_cm"],
                       g["off_cm"]))
    total = np.linalg.norm(T_cur[:, :3, 3] - T_seed[:, :3, 3], axis=1) * 100
    n_held = sum(1 for r in rows.values() if r[7] >= held_at)
    summary_held = {"seed_held_at": held_at, "scans": n_held,
                    "fraction": n_held / max(len(rows), 1),
                    "segments": len(segs),
                    "seconds": float(sum(g["t1"] - g["t0"] for g in segs)),
                    "max_seed_off_cm": float(np.nanmax(
                        [g["off_cm"] for g in segs])) if segs else 0.0}
    summary = {"name": name, "bag": bag, "run_traj": run_traj, "ref_map": ref_map,
               "seed_held": summary_held,
               "session_anchor": sa_path, "anchor_cam": c["anchor_cam"], "anchor_mode": mode,
               "t_anchor": t_a, "T_map_glim_seed": T_map_glim.tolist(), "rounds": history,
               "registered_last_round": history[-1]["registered"], "poses": len(times),
               "seed_to_final_cm": {"median": float(np.median(total)),
                                    "p95": float(np.percentile(total, 95)),
                                    "max": float(total.max())}}
    json.dump(summary, open(os.path.join(outd, "summary_%s.json" % name), "w"), indent=2)

    print("\n=== %s ===" % name)
    print("round | registered | median corr | p95 corr | median residual")
    for h in history:
        print("%5d | %5d/%-5d | %8.2f cm | %5.2f cm | %8.2f cm"
              % (h["round"], h["registered"], h["scans"], h["median_corr_cm"],
                 h["p95_corr_cm"], h["median_residual_cm"]))
    print("GLIM seed -> final: median %.1f cm, p95 %.1f cm, max %.1f cm (how far the map "
          "moved the run's own odometry)" % (np.median(total), np.percentile(total, 95),
                                              total.max()))
    print("\nscans where the map leaves a direction open and the GLIM seed holds it "
          "(seed share >= %.0f %%): %d of %d (%.1f %%), in %d segment(s), %.1f s in all"
          % (held_at * 100, n_held, len(rows), 100 * summary_held["fraction"], len(segs),
             summary_held["seconds"]))
    long_ = sorted(segs, key=lambda g: -(g["t1"] - g["t0"]))[:10]
    if long_:
        print("  longest (seed off = how far GLIM is off along the open direction at the "
              "segment's edges; inside, the error runs from one to the other):")
        print("    %8s %7s %6s %7s  %-22s %-20s %s"
              % ("start s", "dur s", "scans", "path m", "where (map xyz)", "open direction",
                 "seed off before / after"))
        fmt = lambda v: "%.1f" % v if np.isfinite(v) else "n/a"           # noqa: E731
        for g in long_:
            print("    %8.1f %7.1f %6d %7.1f  %-22s %-20s %s / %s cm"
                  % (g["t0"] - times[0], g["t1"] - g["t0"], g["n"], g["path_m"],
                     str(np.round(g["xyz"], 1).tolist()), str(np.round(g["weak"], 2).tolist()),
                     fmt(g["off_before_cm"]), fmt(g["off_after_cm"])))
    print("wrote %s\n      %s\n      quality_%s.csv, seed_held_%s.csv, summary_%s.json"
          % (p_lidar, p_cam, name, name, name))


if __name__ == "__main__":
    main()
