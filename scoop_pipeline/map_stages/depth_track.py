#!/usr/bin/env python3
"""
A depth camera's trajectory in map, for a robot without a LiDAR (mobile_2's
RealSense): its own odometry (processing/vslam.py) as the seed, every depth
frame registered to the reference pass's frozen map with stage 04's ICP.
Stage 04 runs it for each entry of "04_reference" "tracks":

  "tracks": [{"name": "mobile_2_depth",
              "depth_topic": "/mobile_2/depth/image_rect_raw",
              "camera_info_topic": "/mobile_2/depth/camera_info",
              "depth_frame": "camera_depth_optical_frame",
              "run_traj": <work>/<date>/<pass>/<machine>/vslam/traj_vslam.txt,
              "body_frame": "camera_link",          # the frame run_traj is of
              "cam_frame": "camera_color_optical_frame",
              "anchor_cam": "realsense",            # 03's camera that places the start
              "depth_scale": 0.001, "range": [0.3, 4.0], "pixel_step": 4,
              "frame_stride": 2, "scan_voxel": 0.05, "max_corr": [0.2, 0.1, 0.05],
              "min_corr": 300, "max_shift": 0.5, "max_rot_deg": 5.0, "rounds": 2}]

  seed      the odometry (its own world), placed in map by 03's session anchor
            of `anchor_cam` (the board dwell at the start), as 04 places GLIM
  pieces    the odometry is cut where it breaks: no odometry for over gap_s
            (tracking lost) or a step over jump_m / jump_deg between two poses
            (a reset). Within a piece it is locally right; between pieces the
            relation is unknown. A depth frame inside a break gets no pose.
  per frame each depth frame (every frame_stride-th), back-projected every
            pixel_step-th pixel within range, in the body frame, starts from its
            seed pose carried by the last registered frame's correction (the
            odometry drifts; the map does not) and is registered to the map.
            A correction over max_shift / max_rot_deg, or fewer than min_corr
            correspondences, keeps the seed.
  re-find   a new piece (or stall_frames rejected frames in a row) is found in
            the map again: window_s of its frames (2x, 4x when another pose fits
            nearly as well) stacked through the piece's odometry, searched around the last registered pose (radius and yaw
            growing with the time since: speed_max, yaw_rate_max) by the share
            of points on the map, the best candidates polished by ICP; taken
            when the share within fit_m is at least min_fit and every distinct
            pose (0.3 m / 10 deg away) leaves rival_ratio x as many points off
            the map, and rival_gap more (as lidar_reloc). Not found: the frames
            wait for the next try.
  rounds    round 2 starts from round 1's poses; the map never changes.

Outputs (odometry/reference_<tag>/), at the depth frames' stamps:
  traj_<name>.tum         T_map_body          (pass_tf: <origin> ~~ body_frame)
  traj_<name>_in_cam.tum  T_map_cam           (10: /<machine>/global_pose)
  quality_<name>.csv      per frame: status, correspondences, residual, correction
  summary_<name>.json
"""
import json
import os
import sys
import time

import numpy as np
import open3d as o3d

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

DEFAULTS = dict(depth_topic="/mobile_2/depth/image_rect_raw",
                camera_info_topic="/mobile_2/depth/camera_info",
                depth_frame="camera_depth_optical_frame", body_frame="camera_link",
                cam_frame="camera_color_optical_frame", anchor_cam="realsense",
                machine="mobile_2", depth_scale=0.001, range=[0.3, 4.0], pixel_step=4,
                frame_stride=2, scan_voxel=0.05, max_corr=[0.2, 0.1, 0.05], min_corr=300,
                max_shift=0.5, max_rot_deg=5.0, rounds=2, time_tol=0.05, carry=True,
                gap_s=0.3, jump_m=0.15, jump_deg=10.0, stall_frames=15, window_s=2.0,
                speed_max=0.8, yaw_rate_max=60.0, search_step=0.2, yaw_step_deg=6.0,
                search_voxel=0.10, fit_m=0.05, min_fit=0.6, rival_ratio=2.0, rival_gap=0.05,
                candidates=8)


def rel_pose(edges, a, b):
    """T_a_b through the static tree."""
    from scoop import tftree
    ra, rb = tftree._root(a, edges), tftree._root(b, edges)
    if ra != rb:
        raise SystemExit("%s and %s are not in one tree (roots %s, %s)" % (a, b, ra, rb))
    return np.linalg.inv(tftree._static_pose(ra, a, edges)) @ tftree._static_pose(ra, b, edges)


def camera_k(reader, topic):
    for _, _, payload, decode in reader.iter_raw([topic]):
        m = decode(payload)
        k = np.asarray(m.k if hasattr(m, "k") else m.K, float).reshape(3, 3)
        return k, int(m.width), int(m.height)
    raise SystemExit("no messages on %s" % topic)


def backproject(img, K, scale, lo, hi, step):
    """Depth image -> points (N,3) in its optical frame, every step-th pixel."""
    d = img[::step, ::step].astype(np.float64) * scale
    v, u = np.mgrid[0:img.shape[0]:step, 0:img.shape[1]:step]
    ok = np.isfinite(d) & (d > lo) & (d < hi)
    z = d[ok]
    x = (u[ok] - K[0, 2]) / K[0, 0] * z
    y = (v[ok] - K[1, 2]) / K[1, 1] * z
    return np.column_stack([x, y, z])


def read_frames(bag, c, T_body_depth, t_lo, t_hi):
    """Every frame_stride-th depth frame within [t_lo, t_hi]: (times, [points
    in the body frame, voxel-downsampled])."""
    from scoop import bag as sbag
    r = sbag.open_bag(bag)
    K, _, _ = camera_k(r, c["camera_info_topic"])
    lo, hi = c["range"]
    times, scans = [], []
    n = 0
    t0 = time.time()
    for fr in sbag.iter_images(r, c["depth_topic"], mode="raw"):
        n += 1
        if (n - 1) % int(c["frame_stride"]) or not (t_lo <= fr.stamp <= t_hi):
            continue
        p = backproject(fr.image, K, float(c["depth_scale"]), lo, hi, int(c["pixel_step"]))
        if len(p) < int(c["min_corr"]):
            continue
        p = p @ T_body_depth[:3, :3].T + T_body_depth[:3, 3]
        pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(p))
        scans.append(np.asarray(pc.voxel_down_sample(float(c["scan_voxel"])).points))
        times.append(fr.stamp)
        if len(times) % 1000 == 0:
            print("    %d frames read (%.0f s)" % (len(times), time.time() - t0), flush=True)
    o = np.argsort(times)
    return np.asarray(times)[o], [scans[i] for i in o]


def register_frames(A, ref, scans, T_cur, c, carry):
    """One pass: each frame from its seed (carried by the last registered
    frame's correction when `carry`) -> (poses, rows)."""
    T_new = T_cur.copy()
    rows = []
    C = np.eye(4)
    max_rot = np.deg2rad(float(c["max_rot_deg"]))
    for j, p in enumerate(scans):
        T0 = C @ T_cur[j] if carry else T_cur[j]
        T_j, n_corr, rms, _, _ = A.register(ref, p, T0.copy(), c)
        ok_T = np.isfinite(T_j).all()
        d = float(np.linalg.norm(T_j[:3, 3] - T0[:3, 3])) if ok_T else float("nan")
        dR = T_j[:3, :3] @ T0[:3, :3].T if ok_T else np.eye(3)
        ang = float(np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1))) if ok_T else float("nan")
        if not ok_T or n_corr < int(c["min_corr"]):
            status = "seed_kept_thin"
        elif d > float(c["max_shift"]) or ang > max_rot:
            status = "seed_kept_far"
        else:
            status = "ok"
        if status == "ok":
            T_new[j] = T_j
            if carry:
                C = T_j @ np.linalg.inv(T_cur[j])
        else:
            T_new[j] = T0
        rows.append((status, n_corr, rms, d, np.degrees(ang)))
    return T_new, rows


def pieces(o_t, o_T, c):
    """Piece index per odometry pose: a new piece after a gap or a jump."""
    dt = np.diff(o_t)
    step = np.linalg.norm(np.diff(o_T[:, :3, 3], axis=0), axis=1)
    R = np.einsum("nji,njk->nik", o_T[:-1, :3, :3], o_T[1:, :3, :3])
    ang = np.degrees(np.arccos(np.clip((np.trace(R, axis1=1, axis2=2) - 1) / 2, -1, 1)))
    brk = (dt > float(c["gap_s"])) | (step > float(c["jump_m"])) | (ang > float(c["jump_deg"]))
    return np.concatenate([[0], np.cumsum(brk)])


def rotz(deg):
    a = np.radians(deg)
    return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])


def refind(A, ref, pts, T_guess, radius, yaw, c):
    """`pts` (body frame at the pose sought) placed in map around T_guess:
    -> (T or None, fit, rival fit, n candidates)."""
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    P = np.asarray(pc.voxel_down_sample(float(c["search_voxel"])).points)
    Pf = np.asarray(pc.voxel_down_sample(float(c["scan_voxel"])).points)
    st = float(c["search_step"])
    g = np.arange(-radius, radius + 1e-9, st)
    gx, gy = np.meshgrid(g, g)
    off = np.column_stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)])
    off = off[np.linalg.norm(off, axis=1) <= radius + 1e-9]
    yaws = np.arange(-yaw, yaw + 1e-9, float(c["yaw_step_deg"])) if yaw < 180 else \
        np.arange(-180.0, 180.0, float(c["yaw_step_deg"]))
    c0 = T_guess[:3, 3]
    gate = 2 * float(c["search_voxel"])
    scored = []
    for dy in yaws:
        R = rotz(dy) @ T_guess[:3, :3]
        Q = P @ R.T + c0
        X = (Q[None, :, :] + off[:, None, :]).reshape(-1, 3)
        d, _ = ref.tree.query(X, workers=-1, distance_upper_bound=gate)
        share = np.isfinite(d).reshape(len(off), len(P)).mean(1)
        for k in np.argsort(-share)[:3]:
            scored.append((share[k], dy, off[k]))
    scored.sort(key=lambda x: -x[0])
    picks = []
    for sc, dy, o in scored:                         # distinct: 0.5 m / 15 deg apart
        if all(np.linalg.norm(o - q[2]) > 0.5 or abs(dy - q[1]) > 15 for q in picks):
            picks.append((sc, dy, o))
        if len(picks) >= int(c["candidates"]):
            break
    cr = dict(c, max_corr=[0.3, 0.15, 0.05], prior_beta=0.0, nn_prior_beta=0.0)
    res = []
    for _, dy, o in picks:
        T0 = np.eye(4)
        T0[:3, :3], T0[:3, 3] = rotz(dy) @ T_guess[:3, :3], c0 + o
        T, n, rms, _, _ = A.register(ref, Pf, T0, cr)
        if not np.isfinite(T).all():
            continue
        d, _ = ref.tree.query(Pf @ T[:3, :3].T + T[:3, 3], workers=-1,
                              distance_upper_bound=float(c["fit_m"]))
        res.append((float(np.isfinite(d).mean()), T))
    if not res:
        return None, 0.0, 0.0, len(picks)
    res.sort(key=lambda x: -x[0])
    fit, T = res[0]
    rival = 0.0
    for f2, T2 in res[1:]:
        dR = T2[:3, :3] @ T[:3, :3].T
        a = np.degrees(np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1)))
        if np.linalg.norm(T2[:3, 3] - T[:3, 3]) > 0.3 or a > 10:
            rival = max(rival, f2)
    off, off_r = 1.0 - fit, 1.0 - rival              # as lidar_reloc: a distinct pose must
    ok = fit >= float(c["min_fit"]) and off_r >= float(c["rival_ratio"]) * off \
        and off_r - off >= float(c["rival_gap"])     # leave clearly more of the window off
    return (T if ok else None), fit, rival, len(picks)


def first_round(A, ref, scans, times, T_odom, piece, C0, piece0, c, log=print):
    """Round 1 over the pieces: carry within a piece, re-find between them.
    -> (poses (nan where unplaced), rows, re-find log)."""
    n = len(scans)
    T_out = np.full((n, 4, 4), np.nan)
    rows = [("unplaced", 0, np.nan, np.nan, np.nan)] * n
    max_rot = np.deg2rad(float(c["max_rot_deg"]))
    C, cur = (C0, piece0) if C0 is not None else (None, None)
    last_j, bad, events = None, 0, []
    win = float(c["window_s"])
    j = 0
    while j < n:
        if C is not None and piece[j] != cur:
            C = None                                 # a new piece: its place is unknown
        if C is None:
            if last_j is None and piece[j] != piece0:
                j += 1                               # before the anchored piece: nothing to go by
                continue
            if last_j is None:
                C, cur = C0, piece0
                continue
            # window: this piece's frames within window_s, in the body frame at
            # j; when another pose fits nearly as well, a longer one (2x, 4x)
            inv0 = np.linalg.inv(T_odom[j])
            since = times[j] - times[last_j]
            radius = min(4.0, 0.3 + float(c["speed_max"]) * since)
            yaw = min(180.0, 20.0 + float(c["yaw_rate_max"]) * since)
            k_prev = None
            for wl in (win, 2 * win, 4 * win):
                k, stack = j, []
                while k < n and piece[k] == piece[j] and times[k] - times[j] <= wl:
                    M = inv0 @ T_odom[k]
                    stack.append(scans[k] @ M[:3, :3].T + M[:3, 3])
                    k += 1
                if k == k_prev:
                    break                            # the piece ends: no longer window
                k_prev = k
                T, fit, rival, nc = refind(A, ref, np.vstack(stack), T_out[last_j], radius, yaw, c)
                log("    re-find at %.1f s (%.1f s since the last pose, +-%.1f m / +-%.0f deg, "
                    "%.0f s window): fit %.2f, rival %.2f -> %s"
                    % (times[j] - times[0], since, radius, yaw, times[k - 1] - times[j], fit,
                       rival, "placed" if T is not None else "not found"))
                if T is not None or fit < float(c["min_fit"]):
                    break                            # found, or nothing fits: no use looking longer
            events.append({"t": float(times[j]), "since_s": float(since), "radius_m": radius,
                           "yaw_deg": yaw, "window_s": float(times[k - 1] - times[j]),
                           "fit": fit, "rival": rival, "found": T is not None})
            if T is None:
                j = k                                # try again after this window
                continue
            C, cur, bad = T @ inv0, piece[j], 0
        T0 = C @ T_odom[j]
        T_j, n_corr, rms, _, _ = A.register(ref, scans[j], T0.copy(), c)
        ok_T = np.isfinite(T_j).all()
        d = float(np.linalg.norm(T_j[:3, 3] - T0[:3, 3])) if ok_T else float("nan")
        dR = T_j[:3, :3] @ T0[:3, :3].T if ok_T else np.eye(3)
        ang = float(np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1))) if ok_T else float("nan")
        if not ok_T or n_corr < int(c["min_corr"]):
            status = "seed_kept_thin"
        elif d > float(c["max_shift"]) or ang > max_rot:
            status = "seed_kept_far"
        else:
            status = "ok"
        if status == "ok":
            T_out[j] = T_j
            if c["carry"]:
                C = T_j @ np.linalg.inv(T_odom[j])
            last_j, bad = j, 0
        else:
            T_out[j] = T0
            bad += 1
            if bad >= int(c["stall_frames"]) and last_j is not None:
                C, bad = None, 0                     # the odometry slipped: find it again
                T_out[j - int(c["stall_frames"]) + 1:j + 1] = np.nan
                rows[j - int(c["stall_frames"]) + 1:j + 1] = \
                    [("unplaced", 0, np.nan, np.nan, np.nan)] * int(c["stall_frames"])
                j += 1
                continue
        rows[j] = (status, n_corr, rms, d, np.degrees(ang))
        j += 1
    return T_out, rows, events


def run(P, tr, base, ref, outd):
    """One depth track (stage 04's "tracks" entry `tr`; `base` = 04's settings
    for what the entry leaves out, `ref` = 04's frozen map)."""
    import importlib
    from scoop import tftree
    A = importlib.import_module("02_refine_poses")
    R04 = importlib.import_module("04_reference_traj")
    from pipeline_common import traj_quats, interp_poses
    c = dict(base)
    c.update(DEFAULTS)
    c.update({k: v for k, v in tr.items()})
    c.setdefault("nn_prior_beta", c["prior_beta"])
    name = c["name"]
    bag = os.path.expanduser(c.get("bag") or P.dataset["bag"])
    if c.get("run_traj"):
        run_traj = os.path.expanduser(c["run_traj"])
    else:
        sys.path.insert(0, os.path.join(os.path.dirname(HERE), "processing"))
        import vslam
        run_traj = vslam.traj_path(P, c["machine"])
    sa_path = P.outp(c.get("session_anchor") or "session_anchor.json")
    print("\n##### track %s: %s's %s, seeded by %s #####" % (name, c["machine"],
                                                           c["depth_topic"], run_traj))
    for f in (run_traj, sa_path):
        if not os.path.exists(f):
            raise SystemExit("not found: %s" % f)
    edges = tftree.tf_edges([bag], seconds=1.0)
    T_bd = rel_pose(edges, c["body_frame"], c["depth_frame"])
    T_bc = rel_pose(edges, c["body_frame"], c["cam_frame"])
    o_t, o_T = A.load_traj(run_traj)
    T_map_odom, t_a, mode = R04.seed_from_anchor(sa_path, c["anchor_cam"], o_t, o_T, T_bc)
    print("seed: %d odometry poses (%.1f s) placed in map by 03's '%s' anchor (%s) at t=%.3f"
          % (len(o_t), o_t[-1] - o_t[0], c["anchor_cam"], mode, t_a))

    print("reading %s (every %d-th frame, every %d-th pixel, %.1f-%.1f m) ..."
          % (c["depth_topic"], int(c["frame_stride"]), int(c["pixel_step"]), *c["range"]))
    times, scans = read_frames(bag, c, T_bd, o_t[0], o_t[-1])
    if not len(times):
        raise SystemExit("no depth frames within the odometry's time span")
    print("    %d frames, median %d points each" % (len(times), int(np.median([len(s) for s in scans]))))
    # the odometry's pieces; a frame inside a break has no seed
    pc_o = pieces(o_t, o_T, c)
    i = np.clip(np.searchsorted(o_t, times), 1, len(o_t) - 1)
    keep = (pc_o[i - 1] == pc_o[i]) & (times >= o_t[0]) & (times <= o_t[-1])
    times, scans, piece = times[keep], [scans[k] for k in np.flatnonzero(keep)], pc_o[i - 1][keep]
    n_br = int(pc_o[-1])
    print("    odometry in %d piece(s) (%d break(s): gaps over %.1f s or jumps over %.2f m / "
          "%.0f deg); %d frames inside breaks left out"
          % (n_br + 1, n_br, float(c["gap_s"]), float(c["jump_m"]), float(c["jump_deg"]),
             int((~keep).sum())))
    Rq, pq = interp_poses(o_t, o_T, traj_quats(o_T), times)
    T_odom = np.tile(np.eye(4), (len(times), 1, 1))
    T_odom[:, :3, :3], T_odom[:, :3, 3] = Rq, pq
    T_seed = np.einsum("ij,njk->nik", T_map_odom, T_odom)
    ia = int(np.clip(np.searchsorted(o_t, t_a), 0, len(o_t) - 1))
    piece0 = int(pc_o[ia])

    history, rows, events, T_cur = [], [], [], None
    for rnd in range(1, int(c["rounds"]) + 1):
        print("\n=== %s round %d/%d: every depth frame to the frozen map ===" % (name, rnd, int(c["rounds"])))
        t0 = time.time()
        if rnd == 1:
            T_cur, rows, events = first_round(A, ref, scans, times, T_odom, piece, T_map_odom,
                                              piece0, c)
        else:
            idx = np.flatnonzero(np.isfinite(T_cur).all(axis=(1, 2)))
            T_p, r_p = register_frames(A, ref, [scans[k] for k in idx], T_cur[idx], c, False)
            T_cur = T_cur.copy()
            T_cur[idx] = T_p
            rows = list(rows)
            for k, r in zip(idx, r_p):
                rows[k] = r
        ok = [i for i, r in enumerate(rows) if r[0] == "ok"]
        d = np.array([rows[i][3] for i in ok]) * 100 if ok else np.zeros(1)
        res = np.array([rows[i][2] for i in ok if np.isfinite(rows[i][2])]) * 100
        counts = {k: sum(1 for r in rows if r[0] == k)
                  for k in ("ok", "seed_kept_far", "seed_kept_thin", "unplaced")}
        print("  %d frames: %d registered, %d kept their seed (correction over max_shift/max_rot), "
              "%d too few correspondences, %d not placed  (%.0f s)"
              % (len(rows), counts["ok"], counts["seed_kept_far"], counts["seed_kept_thin"],
                 counts["unplaced"], time.time() - t0))
        print("  correction this round: median %.2f cm, p95 %.2f cm | residual median %.2f cm"
              % (np.median(d), np.percentile(d, 95), np.median(res) if len(res) else float("nan")))
        history.append({"round": rnd, "registered": counts["ok"], "frames": len(rows),
                        "not_placed": counts["unplaced"],
                        "median_corr_cm": float(np.median(d)),
                        "p95_corr_cm": float(np.percentile(d, 95)),
                        "median_residual_cm": float(np.median(res)) if len(res) else None})

    placed = np.isfinite(T_cur).all(axis=(1, 2))
    p_body = os.path.join(outd, "traj_%s.tum" % name)
    p_cam = os.path.join(outd, "traj_%s_in_cam.tum" % name)
    A.write_traj(p_body, times[placed], T_cur[placed])
    A.write_traj(p_cam, times[placed], np.einsum("nij,jk->nik", T_cur[placed], T_bc))
    with open(os.path.join(outd, "quality_%s.csv" % name), "w") as f:
        f.write("t,piece,status,n_corr,residual_cm,correction_cm,correction_deg\n")
        for t, pc_, (st, nc, rms, dd, da) in zip(times, piece, rows):
            f.write("%.9f,%d,%s,%d,%.3f,%.3f,%.4f\n" % (t, pc_, st, nc, rms * 100, dd * 100, da))
    a0 = placed & (piece == piece0)                 # the anchored piece: the seed means something
    drift = np.linalg.norm(T_cur[a0, :3, 3] - T_seed[a0, :3, 3], axis=1) * 100 \
        if a0.any() else np.zeros(1)
    gaps = np.diff(times[placed]) if placed.sum() > 1 else np.zeros(0)
    summary = {"name": name, "machine": c["machine"], "bag": bag, "run_traj": run_traj,
               "body_frame": c["body_frame"], "cam_frame": c["cam_frame"],
               "session_anchor": sa_path, "anchor_cam": c["anchor_cam"], "anchor_mode": mode,
               "t_anchor": t_a, "T_map_odom_seed": T_map_odom.tolist(), "rounds": history,
               "registered_last_round": history[-1]["registered"], "frames": len(times),
               "poses": int(placed.sum()), "odometry_pieces": n_br + 1, "refind": events,
               "longest_unplaced_s": float(gaps.max()) if len(gaps) else None,
               "anchored_piece_seed_to_final_cm": {"median": float(np.median(drift)),
                                                   "p95": float(np.percentile(drift, 95)),
                                                   "max": float(drift.max()),
                                                   "at_end": float(drift[-1])}}
    json.dump(summary, open(os.path.join(outd, "summary_%s.json" % name), "w"), indent=2)
    print("\n=== %s ===" % name)
    print("round | registered | median corr | p95 corr | median residual")
    for h in history:
        print("%5d | %5d/%-5d | %8.2f cm | %5.2f cm | %8s cm"
              % (h["round"], h["registered"], h["frames"], h["median_corr_cm"], h["p95_corr_cm"],
                 "%.2f" % h["median_residual_cm"] if h["median_residual_cm"] is not None else "n/a"))
    found = sum(e["found"] for e in events)
    print("odometry pieces: %d; re-finds: %d placed, %d not found; %d of %d frames have a pose "
          "(longest stretch without: %.1f s)"
          % (n_br + 1, found, len(events) - found, int(placed.sum()), len(times),
             summary["longest_unplaced_s"] or 0.0))
    print("anchored piece, odometry seed -> final: median %.1f cm, p95 %.1f cm, max %.1f cm, "
          "%.1f cm at its end (how far the odometry drifted from the map)"
          % (np.median(drift), np.percentile(drift, 95), drift.max(), drift[-1]))
    print("wrote %s\n      %s\n      quality_%s.csv, summary_%s.json" % (p_body, p_cam, name, name))
    return p_body, p_cam
