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
  per frame each depth frame (every frame_stride-th), back-projected every
            pixel_step-th pixel within range, in the body frame, starts from its
            seed pose carried by the last registered frame's correction (the
            odometry drifts; the map does not) and is registered to the map.
            A correction over max_shift / max_rot_deg, or fewer than min_corr
            correspondences, keeps the seed.
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
                max_shift=0.5, max_rot_deg=5.0, rounds=2, time_tol=0.05, carry=True)


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
    Rq, pq = interp_poses(o_t, o_T, traj_quats(o_T), times)
    T_odom = np.tile(np.eye(4), (len(times), 1, 1))
    T_odom[:, :3, :3], T_odom[:, :3, 3] = Rq, pq
    T_seed = np.einsum("ij,njk->nik", T_map_odom, T_odom)

    T_cur, history, rows = T_seed.copy(), [], []
    for rnd in range(1, int(c["rounds"]) + 1):
        print("\n=== %s round %d/%d: every depth frame to the frozen map ===" % (name, rnd, int(c["rounds"])))
        t0 = time.time()
        T_new, rows = register_frames(A, ref, scans, T_cur, c, bool(c["carry"]))
        ok = [i for i, r in enumerate(rows) if r[0] == "ok"]
        d = np.linalg.norm(T_new[ok, :3, 3] - T_cur[ok, :3, 3], axis=1) * 100 if ok else np.zeros(1)
        res = np.array([rows[i][2] for i in ok if np.isfinite(rows[i][2])]) * 100
        counts = {k: sum(1 for r in rows if r[0] == k) for k in ("ok", "seed_kept_far", "seed_kept_thin")}
        print("  %d frames: %d registered, %d kept their seed (correction over max_shift/max_rot), "
              "%d too few correspondences  (%.0f s)"
              % (len(rows), counts["ok"], counts["seed_kept_far"], counts["seed_kept_thin"],
                 time.time() - t0))
        print("  correction this round: median %.2f cm, p95 %.2f cm | residual median %.2f cm"
              % (np.median(d), np.percentile(d, 95), np.median(res) if len(res) else float("nan")))
        history.append({"round": rnd, "registered": counts["ok"], "frames": len(rows),
                        "median_corr_cm": float(np.median(d)),
                        "p95_corr_cm": float(np.percentile(d, 95)),
                        "median_residual_cm": float(np.median(res)) if len(res) else None})
        T_cur = T_new

    p_body = os.path.join(outd, "traj_%s.tum" % name)
    p_cam = os.path.join(outd, "traj_%s_in_cam.tum" % name)
    A.write_traj(p_body, times, T_cur)
    A.write_traj(p_cam, times, np.einsum("nij,jk->nik", T_cur, T_bc))
    with open(os.path.join(outd, "quality_%s.csv" % name), "w") as f:
        f.write("t,status,n_corr,residual_cm,correction_cm,correction_deg\n")
        for t, (st, nc, rms, dd, da) in zip(times, rows):
            f.write("%.9f,%s,%d,%.3f,%.3f,%.4f\n" % (t, st, nc, rms * 100, dd * 100, da))
    drift = np.linalg.norm(T_cur[:, :3, 3] - T_seed[:, :3, 3], axis=1) * 100
    summary = {"name": name, "machine": c["machine"], "bag": bag, "run_traj": run_traj,
               "ref_map": getattr(ref, "path", None), "body_frame": c["body_frame"],
               "cam_frame": c["cam_frame"], "session_anchor": sa_path,
               "anchor_cam": c["anchor_cam"], "anchor_mode": mode, "t_anchor": t_a,
               "T_map_odom_seed": T_map_odom.tolist(), "rounds": history,
               "registered_last_round": history[-1]["registered"], "poses": len(times),
               "seed_to_final_cm": {"median": float(np.median(drift)),
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
    print("odometry seed -> final: median %.1f cm, p95 %.1f cm, max %.1f cm, %.1f cm at the end "
          "(how far the odometry drifted from the map)"
          % (np.median(drift), np.percentile(drift, 95), drift.max(), drift[-1]))
    print("wrote %s\n      %s\n      quality_%s.csv, summary_%s.json" % (p_body, p_cam, name, name))
    return p_body, p_cam
