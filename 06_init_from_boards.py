#!/usr/bin/env python3
"""
STAGE 06 - localise every SLAM origin in a NEW bag into `map`, from the board
each sensor can see.

`map -> board` is FIXED (the boards do not move). `<slam origin> -> board` is
run-specific, because a SLAM origin lands somewhere new every recording. So for
each sensor, from THIS bag:

    T_origin_board = T_origin_cam(t) @ T_cam_board(t)     (averaged; board is static)
    T_map_origin   = T_map_board @ inv(T_origin_board)    (T_map_board fixed, from 03)

The ZED anchors off the small 4x4 anchor board. The RealSense anchors off the
big 5x5 board -- which is why stage 03 had to place that board in map first.

CAMERA-MODE SENSORS MEASURE THE OPENING DWELL ONLY (dwell_only, default true)
-----------------------------------------------------------------------------
The session protocol parks each robot in front of its board BEFORE it moves;
the pose this stage exists to measure is that starting pose. So camera-mode
sightings are walked in time order and the dwell ends at the first sighting
whose board-relative position departs the initial cluster by more than
static_tol_mm -- that departure IS the robot starting to move -- and every
later sighting (drive-bys, re-sightings from elsewhere, the OTHER instance of
a shared design, PnP outliers) is ignored by construction. Within the dwell,
cluster_link drops outlier frames that do not agree with the majority. Set
"dwell_only": false on a sensor to keep every sighting and fall back to the
old moving/snapshot machinery.

BOARD FRAME CONVENTION  (read this before touching anything)
------------------------------------------------------------
solvePnP returns the board in OpenCV's native frame: origin at the TOP-left
corner (since OpenCV 4.6), x along the columns, y DOWN, z INTO the board.
Stage 03 may report its board poses in a different frame -- typically
board_axes="ros" (x = outward normal, y = left, z = up) with
board_origin="center". Both are rigid corrections applied to T_cam_board.

This stage reads `board_axes` / `board_origin` straight out of anchor_frame.json
and applies the SAME correction to its own detections. If it did not, every
T_map_origin here would be wrong by that fixed rotation and offset -- and wrong
in a way that still looks like a believable extrinsic, which is the worst kind
of wrong. Do not hardcode the convention in this file; let 03's export drive it.

TF SINGLE-PARENT RULE
---------------------
`map` may have exactly ONE parent. You cannot publish both `map_zed -> map` and
`map_realsense -> map`; tf2 will reject the second. So exactly one sensor is the
"bridge_owner" and gets emitted with map as a CHILD; every other origin is
attached as a CHILD of map:

    map_zed --> map --> map_realsense --> odom_realsense --> camera_link --> ...
                 +----> board, board_rs (fixed)

Both origins keep their own live children, so nothing conflicts.

isaac_ros_visual_slam NOTE
--------------------------
Out of the box it publishes frames literally named `map` and `odom`, which
collide head-on with the LiDAR `map`. Remap them before recording or replaying:

    map_frame:=map_realsense   odom_frame:=odom_realsense
    base_frame:=<your base>    publish_map_to_odom_tf:=true

Then `origin_frame` for the RealSense sensor below is `map_realsense`. If you
would rather anchor the pure-VO chain, set it to `odom_realsense` instead --
same maths, one less loop-closure jump.

NO-TF / POSE-TOPIC SENSORS
---------------------------
If a run never recorded TF for a sensor at all -- e.g. isaac_ros_visual_slam
was recorded via its /tracking/odometry topic only, with no /tf and no
/tf_static for that rig -- there is no `origin -> cam` edge to look up. Set:

    "pose_topic":        "/visual_slam/tracking/odometry"
    "pose_child_frame":  "camera_link"            (the odometry's child_frame_id)
    "cam_extrinsic_xyzquat": [x,y,z,qx,qy,qz,qw]  (FIXED cam_link -> cam_optical)

and `origin_frame` becomes whatever frame_id the odometry itself is published
in. T_origin_cam(t) is then built as T_origin_child(t) @ T_child_cam instead of
a TF chain lookup. Everything downstream is identical either way.

  python3 06_init_from_boards.py [pipeline_config.json]
"""
import os
import sys
import json
import numpy as np

from pipeline_common import load_pipeline, qR, R_to_q, make_T, ang_deg
from pipeline_boards import (Board, read_bag, read_pose_topics, pick_intrinsics,
                             avg_T, spread, fmt_spread, stp, T_record,
                             T_from_record, cluster_link)


def get_stage(P, *names):
    for n in names:
        try:
            return n, P.stage(n)
        except Exception:
            if n in P.cfg:
                return n, P.cfg[n]
    raise SystemExit("no stage config found; expected one of %s" % (names,))


def boards_in_map(af, s):
    """{name: (T_map_board, record)}. Prefers stage 03's export; config can
    override. Names are stage-03 INSTANCE names; record["design"] is the
    registry entry to build the detector from."""
    out = {}
    for name, rec in (af.get("boards") or {}).items():
        out[name] = (T_from_record(rec), rec)
    for name, rec in (s.get("boards_in_map") or {}).items():
        out[name] = (T_from_record(rec), dict(rec, method="config_override"))
    if "anchor" not in out:
        q = af.get("map_to_board_qxyzw") or s.get("map_to_board_qxyzw")
        if q is not None:
            T = np.eye(4); T[:3, :3] = qR(q)
            out["anchor"] = (T, {"frame": "board", "method": "legacy_qxyzw"})
    return out


def rpy_deg(R):
    """ZYX yaw-pitch-roll in degrees, for human-readable output only."""
    sy = -R[2, 0]
    sy = max(-1.0, min(1.0, sy))
    pitch = np.arcsin(sy)
    if abs(sy) < 0.9999:
        roll = np.arctan2(R[2, 1], R[2, 2])
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        roll = np.arctan2(-R[1, 2], R[1, 1])
        yaw = 0.0
    return np.degrees([roll, pitch, yaw])


def localise_camera_only(sensor, board, T_map_board, T_fix, imgs, K, D,
                         tol_m=0.05, dwell_only=True, trim_s=0.5):
    """No SLAM origin in this bag -- put the CAMERA itself in map.

    A board sighting gives T_cam_board outright, so
        T_map_cam(t) = T_map_board @ inv(T_cam_board(t))
    needs no odometry, no TF chain and no clock sync: one image is one pose.

    With dwell_only (the default) only the OPENING static dwell is kept: the
    dwell ends at the first sighting whose board-relative position departs the
    initial cluster by more than tol_m (= the robot started moving), and
    cluster_link at tol_m drops in-dwell outliers that disagree with the
    majority. Note what agreement CANNOT do: T_cam_board is purely relative,
    so a stare at the wrong INSTANCE of a shared design agrees with itself
    perfectly -- the config's "board" field, not the data, decides which
    physical board this dwell is attributed to.

    Returns (hits, n_seen_total, departure); departure is (t, displacement_m)
    of the first post-dwell sighting, or None."""
    out = []
    seen = 0
    for st, gray in imgs:
        d = board.detect(gray, K, D)
        if d is None:
            continue
        seen += 1
        T_cam_board = d.T @ T_fix
        T_board_cam = np.linalg.inv(T_cam_board)
        out.append({"t": st, "T_board_cam": T_board_cam,
                    "T_map_cam": T_map_board @ T_board_cam,
                    "reproj": d.reproj, "n": d.n,
                    "range": float(np.linalg.norm(T_cam_board[:3, 3]))})
    dep = None
    if dwell_only and out:
        P = np.array([h["T_board_cam"][:3, 3] for h in out])
        ref = np.median(P[:min(5, len(P))], axis=0)
        end = len(out)
        for i in range(len(P)):
            disp = float(np.linalg.norm(P[i] - ref))
            if disp > tol_m:
                dep = (out[i]["t"], disp)
                end = i
                break
        out = out[:end]
        if dep is not None and len(out) > 5:
            # A slow start creeps for up to tol_m before the departure triggers,
            # and in a SHORT dwell that tail carries real weight. Drop the last
            # trim_s before the departure; keep at least 5 sightings.
            keep = [h for h in out if h["t"] <= dep[0] - trim_s]
            if len(keep) >= 5 and len(keep) < len(out):
                print("  trimmed %d sighting(s) in the %.1f s before the "
                      "departure (slow-start creep guard)"
                      % (len(out) - len(keep), trim_s))
                out = keep
        if len(out) > 1:
            lab = cluster_link(np.array([h["T_board_cam"][:3, 3] for h in out]),
                               tol_m)
            counts = np.bincount(lab)
            keep = int(np.argmax(counts))
            if counts[keep] < len(out):
                print("  agree filter inside the dwell: dropped %d outlier "
                      "sighting(s)" % (len(out) - counts[keep]))
                out = [h for h, l in zip(out, lab) if l == keep]
    return out, seen, dep


def localise(sensor, board, T_map_board, T_fix, imgs, K, D, tree, s,
             pose_streams=None):
    """T_map_origin for one sensor, or None."""
    origin = sensor["origin_frame"]
    cam = sensor["cam_frame"]
    target = int(sensor.get("detect_target", s.get("detect_target", 30)))
    guard = bool(sensor.get("guard_identity", False))
    min_norm = float(sensor.get("min_origin_pose_norm", 1e-3))
    max_std = float(sensor.get("max_std_mm", s.get("max_std_mm", 30.0)))

    pose_topic = sensor.get("pose_topic")
    ps = T_child_cam = pchild = pose_gap = None
    if pose_topic:
        pchild = sensor.get("pose_child_frame")
        if not pchild:
            raise SystemExit("sensor '%s' sets pose_topic but no pose_child_frame "
                             "(the odometry's child_frame_id)" % sensor["name"])
        ext = sensor.get("cam_extrinsic_xyzquat")
        if not ext or len(ext) != 7:
            raise SystemExit("sensor '%s' sets pose_topic but no (or malformed) "
                             "cam_extrinsic_xyzquat: need the fixed "
                             "[x,y,z,qx,qy,qz,qw] %s -> %s transform"
                             % (sensor["name"], pchild, cam))
        T_child_cam = make_T(ext[0:3], ext[3:7])
        ps = (pose_streams or {}).get(pose_topic)
        if ps is None or len(ps) == 0:
            raise SystemExit("sensor '%s': no messages found on pose_topic '%s'"
                             % (sensor["name"], pose_topic))
        pose_gap = float(sensor.get("pose_max_gap", s.get("tf_max_gap", 0.2)))
        print("  pose source: %s (child=%s) + fixed extrinsic -> %s  (%d samples)"
              % (pose_topic, pchild, cam, len(ps)))

    # Scan the WHOLE stream for this topic -- don't stop at the first `target`
    # hits. Camera topics in a merged bag are not guaranteed to start recording
    # at the same wall-clock moment, so "first N frames" is first-N-of-THIS-topic,
    # not first-N-of-the-session.
    all_ests = []; all_stamps = []; all_cam = []
    seen = 0; no_tf = 0; blank = 0
    for st, gray in imgs:
        d = board.detect(gray, K, D)
        if d is None:
            continue
        seen += 1
        # SAME frame convention stage 03 exported in -- see the module docstring
        T_cam_board = d.T @ T_fix
        if pose_topic:
            T_origin_child = ps.lookup(st, pose_gap)
            T_org_cam = None if T_origin_child is None else T_origin_child @ T_child_cam
        else:
            T_org_cam = tree.lookup(origin, cam, st)
        if T_org_cam is None:
            no_tf += 1
            continue
        # Optional blank guard. Correct for a SLAM map frame that publishes an
        # identity pose before it converges. WRONG for an odometry origin, where
        # identity at t=0 is the definition -- leave guard_identity false there
        # and let the scatter test below catch bad frames instead.
        if guard and (np.linalg.norm(T_org_cam[:3, 3]) < min_norm
                      and np.allclose(T_org_cam[:3, :3], np.eye(3), atol=1e-6)):
            blank += 1
            continue
        all_ests.append(T_org_cam @ T_cam_board)   # board expressed in the origin
        all_stamps.append(st)
        all_cam.append(T_cam_board)

    if imgs:
        t_first_img, t_last_img = imgs[0][0], imgs[-1][0]
        span = max(t_last_img - t_first_img, 1e-9)
        if all_stamps:
            dt = all_stamps[0] - t_first_img
            print("  first valid sighting: t=%.3f  (%.3f s / %.1f%% into this "
                  "camera's own %.1f s stream)"
                  % (all_stamps[0], dt, 100.0 * dt / span, span))
        else:
            print("  no valid sighting anywhere in this camera's %.1f s stream "
                  "(%d frames scanned)" % (span, len(imgs)))

    # Cap to `target`, but SPREAD across every sighting rather than keeping the
    # first N -- consecutive frames from one dwell are correlated and understate
    # the true scatter.
    if len(all_ests) > target:
        idx = sorted(set(np.linspace(0, len(all_ests) - 1, target).round().astype(int)))
        ests = [all_ests[i] for i in idx]
        cams = [all_cam[i] for i in idx]
    else:
        ests, cams = all_ests, all_cam

    print("  board sightings %d | used %d (spread across full stream) | "
          "no TF %s<-%s %d | blank %d"
          % (seen, len(ests), origin, cam, no_tf, blank))
    if not ests:
        print("  ! could not localise '%s'." % sensor["name"])
        if seen and no_tf:
            if pose_topic:
                print("    the board WAS seen but %s had no sample within %.3f s "
                      "of those stamps. Check pose_topic coverage and "
                      "pose_max_gap / tf_max_gap." % (pose_topic, pose_gap))
            else:
                print("    the board WAS seen but %s <- %s never resolved at those "
                      "stamps. Check the frame names against the bag's TF and that "
                      "the SLAM node was publishing then." % (origin, cam))
        elif not seen:
            print("    the board was never detected on %s." % sensor["image_topic"])
        return None

    T_origin_board, n_used = avg_T(ests)
    sp = spread(ests, T_origin_board)
    rng = [float(np.linalg.norm(T[:3, 3])) for T in cams]
    print("  board in %s: %s  (%d/%d after outlier reject; viewing range "
          "%.2f..%.2f m)" % (origin, fmt_spread(sp), n_used, len(ests),
                             min(rng), max(rng)))
    if sp[0] > max_std:
        print("  ! scatter %.1f mm exceeds max_std_mm=%.1f -> rejected. Some "
              "frames almost certainly used a stale or pre-convergence pose."
              % (sp[0], max_std))
        return None
    # the DIRECT, board-relative measurement -- no map composition involved
    T_board_origin = np.linalg.inv(T_origin_board)
    return T_map_board @ T_board_origin, sp, len(ests), T_board_origin


def main():
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "pipeline_config.json"
    P = load_pipeline(cfg_path)
    key, s = get_stage(P, "06_init", "06_init_zed")
    map_frame = s.get("map_frame", "map")
    board_cfgs = P.cfg.get("boards", {})

    try:
        af = json.load(open(s["anchor_frame"]))
    except (FileNotFoundError, OSError):
        af = {}
    bmap = boards_in_map(af, s)
    if not bmap:
        raise SystemExit("no board poses in map. Re-run 03_anchor.py so "
                         "anchor_frame.json carries a 'boards' block, or set "
                         "%s.boards_in_map in the config." % key)

    # frame convention -- must match whatever 03 exported, see module docstring
    axes = af.get("board_axes")
    borig = af.get("board_origin")
    if axes is None or borig is None:
        axes = axes or "opencv"
        borig = borig or "corner"
        print("! anchor_frame.json predates the board_axes/board_origin export; "
              "assuming '%s'/'%s'. If stage 03 was run with a different "
              "convention every pose below will be wrong by a fixed rotation "
              "and offset. Re-run 03 to be sure." % (axes, borig))
    print("board frame convention: axes=%s origin=%s (from anchor_frame.json)"
          % (axes, borig))

    print("\nboards in map:")
    for n, (T, rec) in sorted(bmap.items()):
        lc = rec.get("loop_closure") or {}
        note = ""
        if rec.get("drift_warning"):
            note += "  ! sections disagree %.1f mm" % rec.get("drift_spanned_mm", 0)
        if lc.get("significant"):
            note += "  ! loop closure %.1f mm" % lc.get("mm", 0)
        print("  %-12s frame=%-14s xyz=%s  (%s, %s views, std %s mm)%s"
              % (n, rec.get("frame", "?"), np.round(T[:3, 3], 4).tolist(),
                 rec.get("method", "?"), rec.get("n_views", "?"),
                 rec.get("std_mm", "?"), note))

    # A sensor that omits its topics inherits them from P.sensor (the ZED block
    # in calibration.json), so the ZED entry stays a two-liner.
    S = P.sensor
    sensors = s["sensors"]
    for x in sensors:
        if not x.get("image_topic"):
            x["image_topic"] = S.image_topic
            x.setdefault("camera_info_topic", S.camera_info_topic)
            x.setdefault("K", np.asarray(S.K).tolist())
            x.setdefault("dist", np.asarray(S.dist).tolist())
    itopics = [x["image_topic"] for x in sensors]
    ctopics = [x.get("camera_info_topic") for x in sensors if x.get("camera_info_topic")]
    ptopics = sorted(set(x["pose_topic"] for x in sensors if x.get("pose_topic")))

    # Does anything here actually need the bag's TF?
    #
    # The camera path never does: T_map_cam = T_map_board @ inv(T_cam_board),
    # where T_map_board comes from anchor_frame.json and T_cam_board from PnP on
    # this bag's own images. No TF chain, no odometry, no clock sync. TF is only
    # required to resolve origin_frame -> cam_frame for a SLAM-origin sensor,
    # and to auto-detect whether such an origin exists at all. Skipping it saves
    # a full pass over the bag.
    need_tf = any(sn.get("source", "auto") != "camera" and sn.get("origin_frame")
                  for sn in sensors)
    print("\nreading %s" % s["bag"])
    if not need_tf:
        print("  (skipping /tf: every sensor here is board-only, which needs no "
              "TF -- just anchor_frame.json and the images)")
    imgs, infos, tree, available = read_bag(
        s["bag"], image_topics=itopics, info_topics=ctopics, want_tf=need_tf,
        stride=s.get("img_stride", 1), max_images=s.get("max_images", 0),
        tf_max_gap=float(s.get("tf_max_gap", 0.2)))
    tf_frames = set(tree.frames()) if tree is not None else set()
    if need_tf:
        print("TF frames: %s" % (sorted(tf_frames),))

    starts = {t: imgs[t][0][0] for t in itopics if imgs.get(t)}
    if len(starts) > 1:
        t0 = min(starts.values())
        print("camera stream starts (relative to the earliest):")
        for t, st0 in sorted(starts.items(), key=lambda kv: kv[1]):
            print("  %-45s +%.3f s" % (t, st0 - t0))

    pose_streams = {}
    if ptopics:
        print("reading pose topics: %s" % ptopics)
        pose_streams = read_pose_topics(s["bag"], ptopics)
        for t in ptopics:
            n = len(pose_streams.get(t, []))
            print("  %s: %d samples%s"
                  % (t, n, "" if n else "  ! EMPTY -- topic not in this bag?"))

    results = {}
    cam_results = {}
    cam_out = {}
    lines = []
    for sensor in sensors:
        name = sensor["name"]
        bname = sensor["board"]
        if not sensor.get("cam_frame"):
            print("\n== %s ==  ! no cam_frame set -> skipped" % name)
            continue
        print("\n== %s ==  origin=%s  cam=%s  board=%s"
              % (name, sensor.get("origin_frame") or "(none: camera from board)",
                 sensor["cam_frame"], bname))
        if bname not in bmap:
            print("  ! board '%s' has no pose in map -> skipped (have: %s)"
                  % (bname, ", ".join(sorted(bmap))))
            continue
        T_map_board, brec = bmap[bname]
        # stage 03 keys boards by INSTANCE; the detector is built from the DESIGN
        design = brec.get("design", bname)
        if design not in board_cfgs:
            print("  ! design '%s' (for board '%s') not in the top-level 'boards' "
                  "registry -> skipped" % (design, bname))
            continue
        board = Board(design, board_cfgs[design])
        T_fix = board.frame_fix(axes, borig)
        print("  %s" % board.describe())
        frames = imgs.get(sensor["image_topic"], [])
        if not frames:
            print("  ! no images on %s\n    topics: %s"
                  % (sensor["image_topic"], "\n    ".join(available)))
            continue
        K, D, from_bag = pick_intrinsics(infos, sensor.get("camera_info_topic"),
                                         sensor.get("rectified", False),
                                         sensor.get("K"), sensor.get("dist"))
        print("  intrinsics: %s | fx=%.2f cx=%.2f"
              % ("bag CameraInfo" if from_bag else "config", K[0, 0], K[0, 2]))
        # Is there anything in this bag to localise an ORIGIN against?
        origin = sensor.get("origin_frame")
        has_pose = bool(sensor.get("pose_topic")) and \
            len(pose_streams.get(sensor.get("pose_topic"), [])) > 0
        has_tf = bool(origin) and origin in tf_frames
        mode = sensor.get("source", "auto")
        if mode == "auto":
            mode = "origin" if (has_pose or has_tf) else "camera"
        if mode == "camera":
            if origin and need_tf:
                print("  ! no SLAM origin available for '%s': frame '%s' is not in "
                      "this bag's TF%s. Falling back to CAMERA-in-map, measured "
                      "straight from the board -- no odometry needed."
                      % (name, origin,
                         " and its pose_topic carried no samples"
                         if sensor.get("pose_topic") else ""))
            dwell_only = bool(sensor.get("dwell_only", s.get("dwell_only", True)))
            tol_m = float(sensor.get("static_tol_mm",
                                     s.get("static_tol_mm", 50.0))) * 1e-3
            hits, seen, dep = localise_camera_only(
                sensor, board, T_map_board, T_fix, frames, K, D, tol_m,
                dwell_only, float(sensor.get("dwell_trim_s",
                                             s.get("dwell_trim_s", 0.5))))
            print("  board sightings %d%s"
                  % (seen, " | %d in the opening dwell" % len(hits)
                     if dwell_only else ""))
            if dep is not None:
                print("  dwell ends at t=%.3f (+%.2f s into the stream): board "
                      "position departed by %.0f mm -> the robot started moving; "
                      "all later sightings ignored"
                      % (dep[0], dep[0] - frames[0][0], dep[1] * 1000))
            if not hits:
                print("  ! the board was never detected on %s%s"
                      % (sensor["image_topic"],
                         " (or every sighting fell outside the opening dwell -- "
                         "was the robot already moving at record start?)"
                         if seen else ""))
                continue
            cam_results[name] = {"sensor": sensor, "board": bname, "hits": hits,
                                 "t_img0": frames[0][0], "t_img1": frames[-1][0],
                                 "n_frames": len(frames),
                                 "dwell_only": dwell_only,
                                 "departure_t": None if dep is None else dep[0]}
            continue
        r = localise(sensor, board, T_map_board, T_fix, frames, K, D, tree, s,
                     pose_streams)
        if r is None:
            continue
        T_map_origin, sp, n, T_board_origin = r
        results[name] = {"sensor": sensor, "T": T_map_origin, "spread": sp, "n": n,
                         "board": bname, "T_board_origin": T_board_origin}
        print("  %s -> %s: xyz=%s qxyzw=%s"
              % (map_frame, sensor["origin_frame"],
                 T_map_origin[:3, 3].round(6).tolist(),
                 R_to_q(T_map_origin[:3, :3]).round(6).tolist()))

    if not results and not cam_results:
        raise SystemExit("\nno sensor could be localised into %s." % map_frame)

    out = {"map_frame": map_frame, "bag": str(s["bag"]),
           "board_axes": axes, "board_origin": borig,
           "boards": {}, "bridges": {}, "cameras": cam_out}

    # ---- board anchors first: everything else hangs off these ---- #
    for n, (T, rec) in sorted(bmap.items()):
        frame = rec.get("frame", "board_%s" % n)
        out["boards"][n] = T_record(map_frame, frame, T)
        lines.append(("[%s -> %s]  (fixed board anchor)" % (map_frame, frame),
                      stp(map_frame, frame, T), True))

    def child_frame_for(sensor, name):
        """Frame name to publish this camera's pose INTO.

        Never the driver's own optical frame: tf2 permits exactly one parent per
        frame, and camera_color_optical_frame already has one
        (camera_link -> camera_color_frame -> ...). Publishing map -> that frame
        makes a second parent and tf2 drops the subtree. A fresh name has no
        parent, so map -> <name> is always safe to publish alongside the live
        driver tree. Override per sensor with "tf_child_frame"."""
        cf = sensor.get("tf_child_frame") or ("%s_pose" % name)
        if cf in tf_frames:
            print("  ! tf_child_frame '%s' ALREADY EXISTS in this bag's TF; "
                  "publishing map -> %s would give it two parents and tf2 will "
                  "reject it. Pick an unused name." % (cf, cf))
        return cf

    def emit_cam(sensor, name, cf, bframe, board, T_map_cam, T_board_cam, hdr):
        """Record and publish one camera pose.

        BOTH parents are printed and both go into the JSON, because
        T_map_cam = T_map_board @ T_board_cam -- they are the same pose, just
        expressed against different parents. Only ONE is emitted as a runnable
        static_transform_publisher: tf2 permits a single parent per frame, and
        publishing both would give '%s' two parents and get the subtree dropped.

        attach_to="map"   (default) pins the camera straight to map.
        attach_to="board" chains through the board anchor that is already
                          published above. Identical numbers, but if stage 03 is
                          ever re-run and the board pose shifts, a board-attached
                          camera follows automatically while a map-attached one
                          silently goes stale.

        Every sensor in the config goes through here, so adding a new one needs
        no code change -- it gets its own <name>_pose frame by default.
        """ % cf
        attach = sensor.get("attach_to", s.get("attach_to", "map"))
        print("     %s -> %s : xyz=%s qxyzw=%s"
              % (map_frame, cf, T_map_cam[:3, 3].round(6).tolist(),
                 R_to_q(T_map_cam[:3, :3]).round(6).tolist()))
        print("     %s -> %s : xyz=%s qxyzw=%s   (direct measurement)"
              % (bframe, cf, T_board_cam[:3, 3].round(6).tolist(),
                 R_to_q(T_board_cam[:3, :3]).round(6).tolist()))
        both = [("map",
                 "[%s -> %s]  %s" % (map_frame, cf, hdr),
                 stp(map_frame, cf, T_map_cam)),
                ("board",
                 "[%s -> %s]  %s  (same pose; reaches %s through the board "
                 "anchor above, and follows it if stage 03 is re-run)"
                 % (bframe, cf, hdr, map_frame),
                 stp(bframe, cf, T_board_cam))]
        for mode, h, body in both:
            active = (mode == attach)
            if not active:
                h += "\n  ALTERNATIVE PARENT -- do NOT publish this alongside " \
                     "the one above: tf2 allows '%s' exactly one parent and " \
                     "would drop the subtree. Commented out in the script." % cf
            lines.append((h, body, active))
        return attach

    # ------- camera-only sensors: pose per sighting, no origin involved ------- #
    for name, cr in cam_results.items():
        hits = cr["hits"]
        sensor = cr["sensor"]
        bframe = bmap[cr["board"]][1].get("frame", cr["board"])
        cf = child_frame_for(sensor, name)
        pos = np.array([h["T_map_cam"][:3, 3] for h in hits])
        span_mm = float(np.linalg.norm(pos.max(0) - pos.min(0)) * 1000)
        static_tol = float(sensor.get("static_tol_mm", s.get("static_tol_mm", 50.0)))
        print("\n=== %s: camera in map, straight from board '%s' ==="
              % (name, cr["board"]))
        print("  %d sightings over %.1f s | viewing range %.2f..%.2f m | "
              "mean reproj %.3f px"
              % (len(hits), hits[-1]["t"] - hits[0]["t"],
                 min(h["range"] for h in hits), max(h["range"] for h in hits),
                 float(np.mean([h["reproj"] for h in hits]))))
        print("  camera position bounding box across all sightings: %.1f mm"
              % span_mm)
        # How late did the board appear? Decides whether the FIRST sighting is
        # anywhere near the start of the recording, or just where the board
        # happened to come into view.
        stream = max(cr["t_img1"] - cr["t_img0"], 1e-9)
        lag = hits[0]["t"] - cr["t_img0"]
        print("  first sighting is %.1f s (%.1f%%) into this camera's %.1f s "
              "stream; %d of %d frames saw the board"
              % (lag, 100.0 * lag / stream, stream, len(hits), cr["n_frames"]))
        if lag > 0.2 * stream:
            print("    ! the board appears late, so the 'first' sighting is NOT "
                  "the start of this recording. moving_pose=\"first\" will pin "
                  "rs_pose wherever the camera happened to be then.")

        treat_static = span_mm <= static_tol or cr["dwell_only"]
        if cr["dwell_only"] and span_mm > static_tol:
            print("  span %.0f mm > static_tol_mm %.0f, but these sightings ARE "
                  "the opening dwell: each is within %.0f mm of the initial "
                  "position and departure/outliers are already removed, so the "
                  "span is bounded at 2x tol by construction. Peak-to-peak over "
                  "%d samples is a ~5-sigma statistic; this is PnP noise at "
                  "range, not motion -> averaging as STATIC."
                  % (span_mm, static_tol, static_tol, len(hits)))
        if treat_static:
            T, nu = avg_T([h["T_map_cam"] for h in hits],
                          weights=[h["n"] / max(h["reproj"], 0.05) ** 2 for h in hits])
            sp = spread([h["T_map_cam"] for h in hits], T)
            Tb, _ = avg_T([h["T_board_cam"] for h in hits])
            print("  -> STATIC (within %.0f mm). Averaged pose, %d/%d after "
                  "outlier reject, %s" % (static_tol, nu, len(hits), fmt_spread(sp)))
            print("     in board frame '%s' : xyz=[%.4f, %.4f, %.4f]  dist %.3f m"
                  % (bframe, Tb[0, 3], Tb[1, 3], Tb[2, 3],
                     float(np.linalg.norm(Tb[:3, 3]))))
            attach = emit_cam(sensor, name, cf, bframe, cr["board"], T, Tb,
                              "(STATIC camera, fixed by board '%s': %d views, "
                              "std %.1f mm; '%s' is a new frame coincident with "
                              "%s)" % (cr["board"], len(hits), sp[0], cf,
                                       sensor["cam_frame"]))
            cam_out[name] = {"mode": "static", "board": cr["board"],
                             "cam_frame": sensor["cam_frame"], "tf_child_frame": cf,
                             "attach_to": attach,
                             "dwell_only": cr["dwell_only"],
                             "departure_t": cr["departure_t"],
                             "n_views": len(hits), "std_mm": round(sp[0], 2),
                             "map_to_cam": T_record(map_frame, cf, T),
                             "board_to_cam": T_record(bframe, cf, Tb)}
        else:
            # first    earliest sighting -- the init/seed pose, if the board was
            #           visible from the start (check the lag warning above)
            # last     latest sighting
            # middle   median by index
            # best     the single most trustworthy fix: lowest reprojection error
            #          normalised by corner count. Accurate, but it can come from
            #          anywhere along the path, so it is NOT representative of
            #          where the camera spent its time
            # closest  nearest to the board -- usually the best-conditioned PnP
            pick = sensor.get("moving_pose", s.get("moving_pose", "first"))
            if pick == "last":
                h = hits[-1]
            elif pick == "middle":
                h = hits[len(hits) // 2]
            elif pick == "best":
                h = min(hits, key=lambda x: x["reproj"] / max(x["n"], 1) ** 0.5)
            elif pick == "closest":
                h = min(hits, key=lambda x: x["range"])
            else:
                pick = "first"
                h = hits[0]
            T = h["T_map_cam"]
            Tb = h["T_board_cam"]
            f = hits[0]["T_board_cam"][:3, 3]
            l = hits[-1]["T_board_cam"][:3, 3]
            print("  -> MOVING (%.0f mm > static_tol_mm %.0f). No single pose "
                  "describes this camera; emitting the '%s' sighting as a "
                  "SNAPSHOT and writing the full trajectory."
                  % (span_mm, static_tol, pick))
            print("     first sighting in '%s': [%.4f, %.4f, %.4f] (dist %.3f m)"
                  % (bframe, f[0], f[1], f[2], float(np.linalg.norm(f))))
            print("     last  sighting in '%s': [%.4f, %.4f, %.4f] (dist %.3f m)"
                  % (bframe, l[0], l[1], l[2], float(np.linalg.norm(l))))
            centroid = pos.mean(0)
            off = float(np.linalg.norm(h["T_map_cam"][:3, 3] - centroid) * 1000)
            print("     snapshot (%s, t=%.3f, %.1f s in, range %.2f m, reproj "
                  "%.3f px) in '%s': [%.4f, %.4f, %.4f]"
                  % (pick, h["t"], h["t"] - cr["t_img0"], h["range"], h["reproj"],
                     bframe, Tb[0, 3], Tb[1, 3], Tb[2, 3]))
            print("     it sits %.0f mm from the centroid of the whole path "
                  "(span %.0f mm) -- other options: first/last/middle/best/closest"
                  % (off, span_mm))
            attach = emit_cam(sensor, name, cf, bframe, cr["board"], T, Tb,
                              "(SNAPSHOT ONLY: camera moved %.0f mm; this is the "
                              "'%s' sighting at t=%.3f, not a fixed extrinsic -- "
                              "use the .tum trajectory for the real motion)"
                              % (span_mm, pick, h["t"]))
            cam_out[name] = {"mode": "moving", "board": cr["board"],
                             "cam_frame": sensor["cam_frame"], "tf_child_frame": cf,
                             "attach_to": attach,
                             "dwell_only": cr["dwell_only"],
                             "departure_t": cr["departure_t"],
                             "n_views": len(hits), "span_mm": round(span_mm, 1),
                             "snapshot": pick, "snapshot_stamp": round(h["t"], 6),
                             "snapshot_range_m": round(h["range"], 4),
                             "snapshot_reproj_px": round(h["reproj"], 4),
                             "snapshot_offset_from_centroid_mm": round(off, 1),
                             "first_sighting_lag_s": round(lag, 3),
                             "map_to_cam": T_record(map_frame, cf, T),
                             "board_to_cam": T_record(bframe, cf, Tb),
                             "first": T_record(bframe, cf, hits[0]["T_board_cam"]),
                             "last": T_record(bframe, cf, hits[-1]["T_board_cam"])}

        tum = os.path.join(os.path.dirname(s["output"]) or ".",
                           "%s_cam_in_map.tum" % name)
        with open(tum, "w") as fh:
            for hh in hits:
                Th = hh["T_map_cam"]; q = R_to_q(Th[:3, :3])
                fh.write("%.9f %.6f %.6f %.6f %.9f %.9f %.9f %.9f\n"
                         % (hh["t"], Th[0, 3], Th[1, 3], Th[2, 3],
                            q[0], q[1], q[2], q[3]))
        cam_out[name]["trajectory_tum"] = tum
        print("  wrote %s (%d poses)" % (tum, len(hits)))

    # --------- origin-mode sensors, relative to their anchor board --------- #
    if results:
        print("\n=== this bag relative to its anchor board (direct measurement) ===")
        for name, r in results.items():
            bframe = bmap[r["board"]][1].get("frame", r["board"])
            T = r["T_board_origin"]
            tt = T[:3, 3]
            rpy = rpy_deg(T[:3, :3])
            print("  %s: origin '%s' in board frame '%s' (board '%s')"
                  % (name, r["sensor"]["origin_frame"], bframe, r["board"]))
            print("      xyz = [%.4f, %.4f, %.4f] m   dist %.3f m"
                  % (tt[0], tt[1], tt[2], float(np.linalg.norm(tt))))
            print("      rpy = [%.2f, %.2f, %.2f] deg  |  %d views, std %.1f mm"
                  % (rpy[0], rpy[1], rpy[2], r["n"], r["spread"][0]))
            tb = np.linalg.inv(T)[:3, 3]
            print("      inverse: board at [%.4f, %.4f, %.4f] m in '%s'"
                  % (tb[0], tb[1], tb[2], r["sensor"]["origin_frame"]))

        names = list(results)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = results[names[i]], results[names[j]]
                if a["board"] == b["board"]:
                    continue
                expect = np.linalg.inv(bmap[a["board"]][0]) @ bmap[b["board"]][0]
                got = (np.linalg.inv(a["T_board_origin"]) @ np.linalg.inv(a["T"])
                       @ b["T"] @ b["T_board_origin"])
                mm = float(np.linalg.norm(expect[:3, 3] - got[:3, 3]) * 1000)
                print("  cross-check %s<->%s board geometry: %.1f mm / %.3f deg vs "
                      "stage 03%s" % (a["board"], b["board"], mm,
                                      ang_deg(expect[:3, :3], got[:3, :3]),
                                      "" if mm < 50 else "   ! large -- one of the "
                                      "two board poses is suspect"))

        owner = s.get("bridge_owner")
        if owner not in results:
            if owner:
                print("\n! bridge_owner '%s' was not localised; falling back." % owner)
            owner = next(iter(results))
        out["bridge_owner"] = owner
        print("\nbridge owner: %s (its origin is the PARENT of %s; all other "
              "origins are attached as children)" % (owner, map_frame))

        for name, r in results.items():
            origin = r["sensor"]["origin_frame"]
            T = r["T"]
            if name == owner:
                Ti = np.linalg.inv(T)
                out["bridges"][name] = dict(T_record(origin, map_frame, Ti),
                                            direction="origin_parent",
                                            board=r["board"], n_views=r["n"],
                                            std_mm=round(r["spread"][0], 2))
                lines.append(("[%s -> %s]  (live-safe: %s attached as CHILD)"
                              % (origin, map_frame, map_frame),
                              stp(origin, map_frame, Ti), True))
            else:
                board_frame = bmap[r["board"]][1].get("frame", r["board"])
                T_bo = r["T_board_origin"]
                out["bridges"][name] = dict(T_record(board_frame, origin, T_bo),
                                            direction="board_parent",
                                            board=r["board"],
                                            board_frame=board_frame,
                                            n_views=r["n"],
                                            std_mm=round(r["spread"][0], 2))
                lines.append(("[%s -> %s]  (measured directly against board '%s'; "
                              "%s is already a child of %s via the fixed board "
                              "anchor above, so this chains correctly through it)"
                              % (board_frame, origin, r["board"], board_frame,
                                 map_frame),
                              stp(board_frame, origin, T_bo), True))
            out["bridges"][name]["reference_map_to_origin"] = \
                T_record(map_frame, origin, T)
            out["bridges"][name]["board_to_origin"] = T_record(
                bmap[r["board"]][1].get("frame", r["board"]), origin,
                r["T_board_origin"])

    with open(s["output"], "w") as f:
        json.dump(out, f, indent=2)

    print("\n=== static transforms for THIS bag ===")
    for hdr, body, active in lines:
        if active:
            print("\n%s" % hdr)
            print(body)
        else:
            print("\n# %s" % hdr.replace("\n", "\n# "))
            print("\n".join("# " + x for x in body.split("\n")))

    script = s.get("script_out")
    if script:
        with open(script, "w") as f:
            f.write("#!/usr/bin/env bash\n# generated by 06_init_from_boards.py\n"
                    "# bag: %s\nset -e\n\n" % s["bag"])
            for hdr, body, active in lines:
                body = body.replace("  ros2", "ros2")
                if not active:
                    body = "\n".join("# " + x for x in body.split("\n"))
                    f.write("# %s\n%s\n\n" % (hdr.replace("\n", "\n# "), body))
                    continue
                f.write("# %s\n%s &\n\n" % (hdr.replace("\n", "\n# "), body))
            f.write("wait\n")
        print("\nwrote %s (chmod +x to run)" % script)
    print("wrote %s" % s["output"])


if __name__ == "__main__":
    main()
