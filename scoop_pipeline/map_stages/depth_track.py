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
            relation is unknown.
  per frame each depth frame (every frame_stride-th), back-projected every
            pixel_step-th pixel within range, in the body frame, is registered
            to the map from
              odometry  where it runs unbroken since the last placed frame: its
                        pose carried by that frame's correction (the odometry
                        drifts; the map does not)
              tracking  else (a break, or a frame inside one): the last placed
                        pose moved on at the same velocity (at most predict_s
                        ahead) -- the map alone carries the robot through the
                        odometry's breaks, frame to frame
            Taken when the correction is within max_shift / max_rot_deg, there
            are min_corr correspondences and at least track_min_fit of the
            points have one. A rejected frame keeps the odometry's pose (or
            none, when tracking).
  re-find   after bridge_s without a placed frame the robot is searched for:
            window_s of frames (2x, 4x when another pose fits nearly as well;
            stacked through the odometry where it runs), around the predicted
            pose (radius and yaw growing with the time since: speed_max,
            yaw_rate_max) by the share
            of points on the map, the best candidates polished by ICP; taken
            when the share within fit_m is at least min_fit and every distinct
            pose (0.3 m / 10 deg away) leaves rival_ratio x as many points off
            the map, and rival_gap more (as lidar_reloc). When several places
            fit (within hyp_tol of the best), each is tracked on for hyp_s
            (hypotheses): the one placing at least hyp_min_frames frames, at
            least hyp_min_ok of those it tried, with every other one scoring at
            most hyp_ratio of it, is taken -- a wrong place stops fitting as the
            robot moves on; while two keep fitting alike (a corridor), for 2x,
            4x hyp_s. Not found: the frames wait for the next try.
  boards    every board sighting of the colour camera (board_image_topic, every
            board_stride-th image, within board_max_range and board_max_reproj)
            is a fix: the body in map from the board's pose (the reference
            pass's, dataset.boards_from's). A frame within board_tol_s of one
            is placed from it when the track is lost, or when the track is
            more than board_reset_m from it; a design with several copies
            (anchor / anchor_b) gives one candidate per copy, the depth frame
            picks. Track - board is reported (the residual).
  lidar     mobile_1's LiDAR sees mobile_2 (map_stages/lidar_sightings.py:
            robot-sized clusters off the map in mobile_2's height band, around
            its floor-to-camera height): when lost, the search is centred on a
            candidate within reach (speed_max x the time since the last pose;
            lidar_search_m around it, taken when unambiguous there); when tied
            candidate places are tracked on, the one whose path runs through
            candidates (lidar_near_m) wins -- lidar_min_support poses, every
            other one at most half as many. Never on its own: a candidate only
            chooses among places the depth frames fit.
  back      from every frame placed again after a loss, the track runs
            backwards in time over the frames before it, until it meets a
            placed frame or is lost.
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
                gap_s=0.3, jump_m=0.15, jump_deg=10.0, window_s=2.0,
                speed_max=0.8, yaw_rate_max=60.0, search_step=0.2, yaw_step_deg=6.0,
                search_voxel=0.10, fit_m=0.05, min_fit=0.6, rival_ratio=2.0, rival_gap=0.05,
                candidates=8, bridge_s=2.0, predict_s=0.5, track_min_fit=0.5,
                hyp_tol=0.1, hyp_max=6, hyp_s=8.0, hyp_min_frames=20, hyp_min_ok=0.7,
                hyp_ratio=0.8, board_fixes=True, board_image_topic="/mobile_2/color/image_raw",
                board_info_topic="/mobile_2/color/camera_info", board_rectified=False,
                board_stride=2, board_max_range=2.5, board_max_reproj=1.0, board_tol_s=0.1,
                board_reset_m=0.3, lidar_sightings=True, lidar={}, lidar_near_m=1.0,
                lidar_tol_s=0.15, lidar_search_m=0.8, lidar_min_support=5)


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
        return None, 0.0, 0.0, []
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
    return (T if ok else None), fit, rival, res


def predict(T_last, T_prev, dt0, dt, c):
    """The last placed pose moved on by the motion between the two last placed
    frames (dt0 apart), dt ahead (constant velocity, at most predict_s)."""
    from scipy.spatial.transform import Rotation
    if T_prev is None or dt0 > 1.0 or dt0 <= 0:
        return T_last.copy()
    s = min(dt, float(c["predict_s"])) / dt0
    D = np.linalg.inv(T_prev) @ T_last
    M = np.eye(4)
    M[:3, :3] = Rotation.from_rotvec(Rotation.from_matrix(D[:3, :3]).as_rotvec() * s).as_matrix()
    M[:3, 3] = D[:3, 3] * s
    return T_last @ M


def step(A, ref, scan, T0, c):
    """One frame to the map from T0 -> (T, status, row)."""
    T_j, n_corr, rms, _, _ = A.register(ref, scan, T0.copy(), c)
    ok_T = np.isfinite(T_j).all()
    d = float(np.linalg.norm(T_j[:3, 3] - T0[:3, 3])) if ok_T else float("nan")
    dR = T_j[:3, :3] @ T0[:3, :3].T if ok_T else np.eye(3)
    ang = float(np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1))) if ok_T else float("nan")
    share = n_corr / max(len(scan), 1)
    if not ok_T or n_corr < int(c["min_corr"]) or share < float(c["track_min_fit"]):
        status = "seed_kept_thin"
    elif d > float(c["max_shift"]) or ang > np.deg2rad(float(c["max_rot_deg"])):
        status = "seed_kept_far"
    else:
        status = "ok"
    return T_j, status, (status, n_corr, rms, d, np.degrees(ang)), share


class Tracker:
    """Where the track stands: the two last placed frames, the odometry's
    correction while its piece runs."""
    def __init__(self):
        self.last = self.prev = None                 # (index, pose)
        self.C = self.cur = None

    def seed(self, j, times, T_odom, piece, c):
        """(T0, source) for frame j, or (None, None) when lost."""
        since = times[j] - times[self.last[0]]
        if piece[j] >= 0 and piece[j] == self.cur and self.C is not None:
            return self.C @ T_odom[j], "odom"
        if since <= float(c["bridge_s"]):
            Tp = self.prev[1] if self.prev else None
            dt0 = times[self.last[0]] - times[self.prev[0]] if self.prev else 0.0
            return predict(self.last[1], Tp, dt0, since, c), "track"
        return None, None

    def placed(self, j, T, T_odom, piece):
        self.prev, self.last = self.last, (j, T)
        if piece[j] >= 0:
            self.C, self.cur = T @ np.linalg.inv(T_odom[j]), piece[j]
        else:
            self.C = self.cur = None


def track_ahead(A, ref, scans, times, T_odom, piece, j, T_first, t_end, c):
    """A hypothesis: frame j at T_first, tracked on until t_end or until lost.
    -> (tracker, {k: (pose, row)}, frames tried, frames placed, summed share)."""
    tk = Tracker()
    out = {}
    T_j, status, row, share = step(A, ref, scans[j], T_first, c)
    if status != "ok":
        return tk, out, 1, 0, 0.0
    tk.placed(j, T_j, T_odom, piece)
    out[j] = (T_j, row)
    tried, placed, score = 1, 1, share
    k = j + 1
    while k < len(scans) and times[k] <= t_end:
        T0, src = tk.seed(k, times, T_odom, piece, c)
        if T0 is None:
            break                                    # lost: this hypothesis ends here
        T_k, status, row, share = step(A, ref, scans[k], T0, c)
        tried += 1
        if status == "ok":
            tk.placed(k, T_k, T_odom, piece)
            out[k] = (T_k, row)
            placed += 1
            score += share
        k += 1
    return tk, out, tried, placed, score


def board_fixes(P, c, bag, T_bc, cache, log=print):
    """Every board sighting of the colour camera -> [(t, board, T_map_body,
    range, reproj)], one entry per copy of the design seen. Cached in `cache`."""
    import pipeline_boards as PB
    key = "%s|%s|%d|%.2f|%.2f" % (bag, c["board_image_topic"], int(c["board_stride"]),
                                  float(c["board_max_range"]), float(c["board_max_reproj"]))
    if os.path.exists(cache):
        d = json.load(open(cache))
        if d.get("key") == key:
            return [(f["t"], f["board"], np.array(f["T"]), f["range"], f["reproj"])
                    for f in d["fixes"]]
    af = json.load(open(P.anchor_frame()))
    recs = dict(af.get("boards") or {})
    for n, r in P.extra_boards().items():
        recs.setdefault(n, r)
    axes, origin = af.get("board_axes", "opencv"), af.get("board_origin", "corner")
    reg = P.cfg.get("boards", {})
    by_design = {}
    for n, r in recs.items():
        by_design.setdefault(r.get("design", n), []).append((n, PB.T_from_record(r)))
    designs = {d: PB.Board(d, reg[d]) for d in by_design if d in reg}
    infos, _, _ = PB.read_tf_and_info(bag, [c["board_info_topic"]], want_tf=False, verbose=False)
    K, D, _ = PB.pick_intrinsics(infos, c["board_info_topic"], bool(c["board_rectified"]))
    T_cb = np.linalg.inv(T_bc)
    out, n_img, t0 = [], 0, time.time()
    log("board sightings on %s (every %d-th image; boards %s) ..."
        % (c["board_image_topic"], int(c["board_stride"]),
           ", ".join("%s: %s" % (d, "/".join(n for n, _ in v)) for d, v in by_design.items())))
    for st, gray in PB.iter_images(bag, c["board_image_topic"], stride=int(c["board_stride"])):
        n_img += 1
        for dn, B in designs.items():
            det = B.detect(gray, K, D)
            if det is None or det.reproj > float(c["board_max_reproj"]):
                continue
            T_cam_board = det.T @ B.frame_fix(axes, origin)
            rng = float(np.linalg.norm(T_cam_board[:3, 3]))
            if rng > float(c["board_max_range"]):
                continue
            T_board_cam = np.linalg.inv(T_cam_board)
            for name, T_map_board in by_design[dn]:
                out.append((st, name, T_map_board @ T_board_cam @ T_cb, rng, float(det.reproj)))
        if n_img % 2000 == 0:
            log("    %d images, %d sightings (%.0f s)" % (n_img, len(out), time.time() - t0))
    json.dump({"key": key, "fixes": [{"t": t, "board": b, "T": T.tolist(), "range": r,
                                      "reproj": e} for t, b, T, r, e in out]},
              open(cache, "w"))
    return out


def fix_lookup(fixes, times, tol):
    """depth frame index -> [T_map_body candidates] from the sightings within tol."""
    out = {}
    if not fixes:
        return out
    ft = np.array([f[0] for f in fixes])
    for i, t in enumerate(times):
        k = np.flatnonzero(np.abs(ft - t) <= tol)
        if len(k):
            # the nearest sighting of each board
            best = {}
            for kk in k:
                b = fixes[kk][1]
                if b not in best or abs(ft[kk] - t) < abs(ft[best[b]] - t):
                    best[b] = kk
            out[i] = [(fixes[kk][1], fixes[kk][2]) for kk in best.values()]
    return out


def from_board(A, ref, scan, cands, c):
    """The best-fitting of the board's candidate poses for this frame, polished
    on the map -> (T, row, board) or None."""
    best = None
    for b, Tb in cands:
        T, status, row, share = step(A, ref, scan, Tb, c)
        if status == "ok" and row[3] <= float(c["board_reset_m"]) and \
                (best is None or share > best[3]):
            best = (T, row, b, share)
    return best[:3] if best else None


def first_round(A, ref, scans, times, T_odom, piece, C0, piece0, c, log=print, fixes=None,
                sight=None):
    """Round 1, frame by frame. A frame starts from the odometry (carried by
    the last placed frame's correction) where the odometry runs unbroken
    since that frame; else from the last placed pose moved on at the same
    velocity (tracked on the map through the odometry's breaks); after
    bridge_s without a placed frame, it is searched for (refind), and when
    several places fit, each is tracked on for hyp_s and the one that keeps
    fitting is taken. -> (poses (nan where unplaced), rows, re-find log)."""
    n = len(scans)
    T_out = np.full((n, 4, 4), np.nan)
    rows = [("unplaced", 0, np.nan, np.nan, np.nan)] * n
    tk = Tracker()
    events = []
    win = float(c["window_s"])
    j = 0
    while j < n:
        if tk.last is None:                          # the start: the anchored piece
            if piece[j] != piece0:
                j += 1
                continue
            T0, src = C0 @ T_odom[j], "odom"
        else:
            T0, src = tk.seed(j, times, T_odom, piece, c)
        cands = (fixes or {}).get(j)
        if cands and tk.last is not None and (T0 is None or min(
                np.linalg.norm(T0[:3, 3] - Tb[:3, 3]) for _, Tb in cands) > float(c["board_reset_m"])):
            hit = from_board(A, ref, scans[j], cands, c)
            if hit is not None:                      # a board says where: placed (again) from it
                T_b, row, b = hit
                if T0 is None:
                    events.append({"t": float(times[j]), "since_s": float(times[j] - times[tk.last[0]]),
                                   "found": True, "board": b})
                    log("    placed again at %.1f s from board '%s' (%.1f s since the last pose)"
                        % (times[j] - times[0], b, times[j] - times[tk.last[0]]))
                else:
                    log("    reset at %.1f s by board '%s': the track was %.2f m off it"
                        % (times[j] - times[0], b, np.linalg.norm(T0[:3, 3] - T_b[:3, 3])))
                    events.append({"t": float(times[j]), "reset_m": float(np.linalg.norm(
                        T0[:3, 3] - T_b[:3, 3])), "found": True, "board": b})
                T_out[j], rows[j] = T_b, row
                tk.placed(j, T_b, T_odom, piece)
                j += 1
                continue
        if T0 is None:                               # lost for bridge_s: search
            since = times[j] - times[tk.last[0]]
            Tp = tk.prev[1] if tk.prev else None
            dt0 = times[tk.last[0]] - times[tk.prev[0]] if tk.prev else 0.0
            centre = predict(tk.last[1], Tp, dt0, since, c)
            radius = min(4.0, 0.3 + float(c["speed_max"]) * since)
            yaw = min(180.0, 20.0 + float(c["yaw_rate_max"]) * since)
            k, T, fit, rival, cands, how = j + 1, None, 0.0, 0.0, [], ""
            k_prev = None
            if sight is not None:                    # mobile_1's LiDAR sees a robot within reach
                cc = sight.at(times[j], 0.3)
                if len(cc):
                    dd = np.linalg.norm(cc - centre[:2, 3], axis=1)
                    reach = 0.3 + float(c["speed_max"]) * since     # no cap: candidates are few
                    for q in np.argsort(dd)[:5]:
                        if dd[q] > reach:
                            break
                        cen = centre.copy()
                        cen[:2, 3] = cc[q]
                        Tq, fq, rq, cq = refind(A, ref, scans[j], cen, float(c["lidar_search_m"]),
                                                180.0, c)       # unambiguous there, or not taken
                        if Tq is not None and np.linalg.norm(Tq[:2, 3] - cc[q]) <= \
                                float(c["lidar_search_m"]):
                            events.append({"t": float(times[j]), "since_s": float(since),
                                           "found": True, "lidar": cc[q].tolist(), "fit": cq[0][0]})
                            log("    placed again at %.1f s at a robot mobile_1's LiDAR sees "
                                "(%.1f, %.1f), %.1f s since the last pose; fit %.2f"
                                % (times[j] - times[0], cc[q][0], cc[q][1], since, cq[0][0]))
                            T = Tq
                            break
            for wl in (win, 2 * win, 4 * win) if T is None else ():
                k, stack = j, []
                inv0 = np.linalg.inv(T_odom[j])
                while k < n and times[k] - times[j] <= wl and \
                        (k == j or (piece[j] >= 0 and piece[k] == piece[j])):
                    M = inv0 @ T_odom[k]
                    stack.append(scans[k] @ M[:3, :3].T + M[:3, 3])
                    k += 1
                if k == k_prev:
                    break
                k_prev = k
                T, fit, rival, cands = refind(A, ref, np.vstack(stack), centre, radius, yaw, c)
                if T is not None or fit < float(c["min_fit"]):
                    break
            ev = {"t": float(times[j]), "since_s": float(since), "radius_m": radius,
                  "yaw_deg": yaw, "fit": fit, "rival": rival, "found": False}
            if T is not None and events and events[-1].get("lidar") and events[-1]["t"] == float(times[j]):
                T0, src = T, "refind"                # placed from the LiDAR candidate (logged)
            elif T is not None:
                T0, src, how = T, "refind", "placed"
            else:
                # several places fit: track each on; the one that keeps fitting wins
                hyps = [Tc for f, Tc in cands
                        if f >= float(c["min_fit"]) and f >= fit - float(c["hyp_tol"])]
                hyps = hyps[:int(c["hyp_max"])]
                # tracked on for hyp_s; while two keep fitting alike (a corridor:
                # shifted along it, the view is the same), for 2x, 4x as long
                for hs in (1, 2, 4):
                    res = [track_ahead(A, ref, scans, times, T_odom, piece, j, Th,
                                       times[j] + hs * float(c["hyp_s"]), c) for Th in hyps]
                    res.sort(key=lambda r: -r[4])
                    uniq = []                        # tracked onto the same place: one
                    for r in res:
                        if r[0].last is None:
                            continue
                        Tl = r[0].last[1]
                        if all(np.linalg.norm(Tl[:3, 3] - u[0].last[1][:3, 3]) > 0.3 or
                               np.degrees(np.arccos(np.clip((np.trace(
                                   Tl[:3, :3] @ u[0].last[1][:3, :3].T) - 1) / 2, -1, 1))) > 10
                               for u in uniq):
                            uniq.append(r)
                    res = uniq
                    if len(res) < 2 or res[1][4] <= float(c["hyp_ratio"]) * res[0][4]:
                        break                        # one left, or one clearly ahead
                    if times[res[0][0].last[0]] < times[j] + hs * float(c["hyp_s"]) - 1.0:
                        break                        # they did not get that far: no use going on
                    hyps = [r[1][j][0] for r in res if j in r[1]]
                if len(res) > 1 and sight is not None and res[1][4] > float(c["hyp_ratio"]) * res[0][4]:
                    # tied on the depth frames: the one whose path runs through
                    # robots mobile_1's LiDAR sees, if one does clearly
                    sup = []
                    for r in res:
                        ks = sorted(r[1])[::3]
                        sup.append(sight.support(times[ks], np.array([r[1][kk][0][:3, 3] for kk in ks]),
                                                 float(c["lidar_near_m"]), float(c["lidar_tol_s"])))
                    o = np.argsort(sup)[::-1]
                    if sup[o[0]] >= int(c["lidar_min_support"]) and \
                            sup[o[1]] <= 0.5 * sup[o[0]] and res[o[0]][3] / max(res[o[0]][2], 1) >= \
                            float(c["hyp_min_ok"]):
                        log("    tie at %.1f s broken by mobile_1's LiDAR: %d poses on a robot it "
                            "sees, the next %d" % (times[j] - times[0], sup[o[0]], sup[o[1]]))
                        res = [res[o[0]]]
                if res:
                    b0 = res[0]
                    s2 = res[1][4] if len(res) > 1 else 0.0
                    ok_frac = b0[3] / max(b0[2], 1)
                    span = (max(b0[1]) - j) if b0[1] else 0
                    if b0[3] >= int(c["hyp_min_frames"]) and ok_frac >= float(c["hyp_min_ok"]) \
                            and s2 <= float(c["hyp_ratio"]) * b0[4]:
                        for kk, (Tk, row) in b0[1].items():
                            T_out[kk], rows[kk] = Tk, row
                        tk = b0[0]
                        how = "placed after tracking %d hypotheses: %d of %d frames fit (%.0f s), " \
                              "next best %.0f%% of its score" % (
                                  len(res), b0[3], b0[2], times[j + span] - times[j],
                                  100 * s2 / max(b0[4], 1e-9))
                        ev.update(found=True, hypotheses=len(res), placed=b0[3])
                        events.append(ev)
                        log("    re-find at %.1f s (%.1f s since the last pose): %s"
                            % (times[j] - times[0], since, how))
                        j = tk.last[0] + 1
                        continue
                    how = "not found (%d hypotheses: best %d of %d frames fit, next best %.0f%% " \
                          "of its score)" % (len(res), b0[3], b0[2], 100 * s2 / max(b0[4], 1e-9))
                else:
                    how = "not found"
            if not (src == "refind" and not how):
                ev["found"] = T is not None
                events.append(ev)
                log("    re-find at %.1f s (%.1f s since the last pose, +-%.1f m / +-%.0f deg): "
                    "fit %.2f, rival %.2f -> %s" % (times[j] - times[0], since, radius, yaw, fit,
                                                    rival, how))
            if T is None:
                j = max(k, j + 1)
                continue
        T_j, status, row, _ = step(A, ref, scans[j], T0, c)
        if status == "ok":
            T_out[j] = T_j
            tk.placed(j, T_j, T_odom, piece)
            rows[j] = row
        elif src == "odom":                          # the odometry's pose, as a stopgap
            T_out[j] = T0
            rows[j] = row
        j += 1
    filled = fill_back(A, ref, scans, times, T_odom, piece, T_out, rows, c)
    if filled:
        log("    tracked back from where the robot was found again: %d more frames placed" % filled)
    return T_out, rows, events


def fill_back(A, ref, scans, times, T_odom, piece, T_out, rows, c):
    """From every placed frame whose predecessor is not, the track run
    backwards in time (the same tracking, time reversed) until it meets a
    placed frame or is lost. -> how many frames it placed."""
    n = len(scans)
    nt = -times
    placed = np.isfinite(T_out).all(axis=(1, 2))
    count = 0
    for j in range(n - 1, 0, -1):
        if not placed[j] or placed[j - 1]:
            continue
        tk = Tracker()
        tk.last = (j, T_out[j])
        if j + 1 < n and placed[j + 1]:
            tk.prev = (j + 1, T_out[j + 1])
        if piece[j] >= 0:
            tk.C, tk.cur = T_out[j] @ np.linalg.inv(T_odom[j]), piece[j]
        k = j - 1
        while k >= 0 and not placed[k]:
            T0, src = tk.seed(k, nt, T_odom, piece, c)
            if T0 is None:
                break
            T_k, status, row, _ = step(A, ref, scans[k], T0, c)
            if status == "ok":
                T_out[k], rows[k], placed[k] = T_k, row, True
                tk.placed(k, T_k, T_odom, piece)
                count += 1
            k -= 1
    return count


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
    # the odometry's pieces; a frame inside a break has none (piece -1): it
    # is tracked on the map from the frames before it
    pc_o = pieces(o_t, o_T, c)
    i = np.clip(np.searchsorted(o_t, times), 1, len(o_t) - 1)
    inside = (pc_o[i - 1] == pc_o[i]) & (times >= o_t[0]) & (times <= o_t[-1])
    piece = np.where(inside, pc_o[i - 1], -1)
    n_br = int(pc_o[-1])
    print("    odometry in %d piece(s) (%d break(s): gaps over %.1f s or jumps over %.2f m / "
          "%.0f deg); %d frames inside breaks, tracked on the map alone"
          % (n_br + 1, n_br, float(c["gap_s"]), float(c["jump_m"]), float(c["jump_deg"]),
             int((~inside).sum())))
    Rq, pq = interp_poses(o_t, o_T, traj_quats(o_T), times)
    T_odom = np.tile(np.eye(4), (len(times), 1, 1))
    T_odom[:, :3, :3], T_odom[:, :3, 3] = Rq, pq
    T_odom[piece < 0] = np.eye(4)
    T_seed = np.einsum("ij,njk->nik", T_map_odom, T_odom)
    ia = int(np.clip(np.searchsorted(o_t, t_a), 0, len(o_t) - 1))
    piece0 = int(pc_o[ia])

    fixes, fix_at = [], {}
    if c["board_fixes"]:
        fixes = board_fixes(P, c, bag, T_bc, os.path.join(outd, "board_fixes_%s.json" % name))
        fix_at = fix_lookup(fixes, times, float(c["board_tol_s"]))
        per = {}
        for f in fixes:
            per.setdefault(f[1], []).append(f[0])
        print("    %d board sightings (%s), %d depth frames with one"
              % (len(fixes), ", ".join("%s: %d, %.0f-%.0f s" % (b, len(v), min(v) - times[0],
                                                               max(v) - times[0])
                                       for b, v in sorted(per.items())), len(fix_at)))

    sight = None
    if c["lidar_sightings"]:
        import lidar_sightings as LS
        ia0 = int(np.flatnonzero(piece == piece0)[0]) if (piece == piece0).any() else 0
        z_cam = float(T_seed[ia0][2, 3])
        # the floor under mobile_2's start, from the map: its body is from just
        # above it to just above its camera
        near = np.linalg.norm(ref.pts[:, :2] - T_seed[ia0][:2, 3], axis=1) < 1.0
        zz = ref.pts[near & (ref.pts[:, 2] < z_cam), 2]
        floor = float(np.percentile(zz, 2)) if len(zz) > 50 else z_cam - 0.4
        print("    floor under mobile_2's start %.2f m, its camera %.2f m" % (floor, z_cam))
        lc = dict(LS.DEFAULTS, z_lo=floor + 0.08, z_hi=z_cam + 0.15)
        lc.update(c.get("lidar") or {})
        scans_l = LS.detect(P, ref, lc, os.path.join(outd, "lidar_sightings_%s.json" % name))
        sight = LS.Sightings(scans_l)
        nc = sum(len(x[1]) for x in scans_l)
        print("    mobile_1's LiDAR: %d scans, %d robot-sized clusters off the map at %.2f-%.2f m"
              % (len(scans_l), nc, lc["z_lo"], lc["z_hi"]))

    history, rows, events, T_cur = [], [], [], None
    for rnd in range(1, int(c["rounds"]) + 1):
        print("\n=== %s round %d/%d: every depth frame to the frozen map ===" % (name, rnd, int(c["rounds"])))
        t0 = time.time()
        if rnd == 1:
            T_cur, rows, events = first_round(A, ref, scans, times, T_odom, piece, T_map_odom,
                                              piece0, c, fixes=fix_at, sight=sight)
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
    # track - board, where a board was seen (the nearest copy of the design)
    resid = []
    for jj, cands in fix_at.items():
        if placed[jj]:
            resid.append(min(np.linalg.norm(T_cur[jj][:3, 3] - Tb[:3, 3]) for _, Tb in cands) * 100)
    if resid:
        print("  track - board sightings: median %.1f cm, p95 %.1f cm, max %.1f cm over %d frames"
              % (np.median(resid), np.percentile(resid, 95), max(resid), len(resid)))
    lid, seen, looked = [], 0, 0
    if sight is not None:
        # where mobile_2 is placed and mobile_1 1.5-15 m from it: how often does
        # mobile_1's LiDAR show a candidate within 1 m, and how far off
        m1 = os.path.join(P.reference_dir(), "traj_%s.tum" % ((c.get("lidar") or {}).get(
            "lidar_track", "mobile_1_lidar")))
        t1, T1 = A.load_traj(m1) if os.path.exists(m1) else (np.zeros(0), None)
        for jj in np.flatnonzero(placed)[::3]:
            if len(t1):
                i1 = int(np.argmin(np.abs(t1 - times[jj])))
                dist = float(np.linalg.norm(T1[i1][:2, 3] - T_cur[jj][:2, 3]))
                if abs(t1[i1] - times[jj]) > 0.2 or not 1.5 <= dist <= 15.0:
                    continue
            looked += 1
            cc = sight.at(times[jj], float(c["lidar_tol_s"]))
            if len(cc):
                dmin = float(np.min(np.linalg.norm(cc - T_cur[jj][:2, 3], axis=1)))
                if dmin <= 1.0:
                    seen += 1
                    lid.append(dmin * 100)
        if looked:
            print("  mobile_1's LiDAR, where mobile_2 is placed 1.5-15 m from it: a candidate within "
                  "1 m in %d of %d frames (%.0f %%)%s" % (
                      seen, looked, 100.0 * seen / looked,
                      ", %.0f cm off (median), %.0f cm (p95)" % (np.median(lid), np.percentile(lid, 95))
                      if lid else ""))
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
               "board_sightings": len(fixes),
               "track_minus_board_cm": {"frames": len(resid),
                                        "median": float(np.median(resid)) if resid else None,
                                        "p95": float(np.percentile(resid, 95)) if resid else None,
                                        "max": float(max(resid)) if resid else None},
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
