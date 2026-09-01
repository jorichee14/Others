#!/usr/bin/env python3
"""
STAGE 07 - session-start poses: where does each mobile robot BEGIN this session,
measured straight off the board it is staring at?

Protocol this encodes: every session opens with each robot parked in front of
its own board -- mobile_1's ZED staring at the 4x4 'anchor', mobile_2's
RealSense staring at 'rs_anchor' -- and only then does the run start. This
script does exactly one job for that moment and nothing else:

    T_cam_board(t)   from PnP on each sighting during the opening dwell
    T_board_cam      averaged over the dwell only (the robot is static there,
                     so every sighting is the same pose plus pixel noise)
    T_map_cam        = T_map_board @ T_board_cam   (board pose fixed, from 03)

No odometry, no TF chain, no clock sync between robots: one image is one pose.

THE DWELL, NOT THE RUN
  Only the opening static dwell is used. Sightings are taken in time order and
  the dwell ends at the first sighting whose board-relative position departs
  from the dwell cluster by more than static_tol_mm -- that departure IS the
  robot starting to move, and it is reported. Everything after it (drive-bys,
  re-sightings from elsewhere, PnP outliers) is ignored by construction.
  Within the dwell, cluster_link at static_tol_mm additionally drops any early
  outlier frames that do not agree with the majority.

BOARD FRAME CONVENTION
  board_axes / board_origin are read from anchor_frame.json and the SAME
  frame_fix is applied to this script's detections, exactly as stage 06 does.
  If 03 exported in a different convention every pose here would be wrong by a
  fixed rotation that still looks like a believable extrinsic.

INSTANCE AMBIGUITY (read this once)
  'anchor' and 'anchor_b' are the same physical printout; a sighting cannot
  tell them apart and neither can any agreement test -- T_cam_board is purely
  relative. The 'board' field in the config must name the instance the robot
  is ACTUALLY parked in front of. The script warns when the configured board's
  design has multiple surveyed instances, but it cannot catch a wrong choice.

CONFIG
  Reads stage "07_session_init" from pipeline_config.json; if absent, falls
  back to "06_init" (bag, anchor_frame, sensors -- extra 06 keys are ignored).

  "07_session_init": {
    "bag": ".../mirc_dataset_coop2_20260828_merged",
    "anchor_frame": "map_stages_20260828_outputs/anchor_frame.json",
    "output": "session_start_poses.json",
    "script_out": "publish_session_tfs.sh",
    "map_frame": "map",
    "img_stride": 1, "max_images": 0,
    "static_tol_mm": 50.0, "min_views": 10, "max_std_mm": 30.0,
    "detect_target": 200,
    "sensors": [
      { "name": "realsense", "board": "rs_anchor",
        "cam_frame": "camera_color_optical_frame", "tf_child_frame": "rs_pose",
        "image_topic": "/mobile_2/color/image_raw",
        "camera_info_topic": "/mobile_2/color/camera_info", "rectified": false },
      { "name": "zed", "board": "anchor",
        "cam_frame": "zed_left_camera_optical_frame", "tf_child_frame": "zed_pose",
        "image_topic": "/mobile_1/zed/left/image_rect_color",
        "camera_info_topic": "/mobile_1/zed/left/camera_info", "rectified": true }
    ]
  }

  python3 07_session_init.py [pipeline_config.json]
"""
import os
import sys
import json
import numpy as np

from pipeline_common import load_pipeline, R_to_q
from pipeline_boards import (Board, read_bag, pick_intrinsics, avg_T, spread,
                             fmt_spread, stp, T_record, T_from_record,
                             cluster_link)


def get_stage(P, *names):
    for n in names:
        try:
            return n, P.stage(n)
        except Exception:
            if n in P.cfg:
                return n, P.cfg[n]
    raise SystemExit("no stage config found; expected one of %s" % (names,))


def dwell_hits(frames, board, T_fix, K, D, tol_m, target):
    """Sightings of the OPENING static dwell only.

    Walks the camera stream in time order. The dwell reference is the median
    board-relative position of the first few sightings; the dwell ends at the
    first sighting further than tol_m from it (= the robot moved). Within the
    dwell, cluster_link drops early outliers that disagree with the majority.

    Returns (hits, n_seen_total, departure) where departure is
    (t, displacement_m) of the first post-dwell sighting, or None."""
    raw = []
    for st, gray in frames:
        d = board.detect(gray, K, D)
        if d is None:
            continue
        T_cam_board = d.T @ T_fix
        T_board_cam = np.linalg.inv(T_cam_board)
        raw.append({"t": st, "T_board_cam": T_board_cam,
                    "T_cam_board": T_cam_board,
                    "reproj": d.reproj, "n": d.n,
                    "range": float(np.linalg.norm(T_cam_board[:3, 3]))})
    if not raw:
        return [], 0, None

    P = np.array([h["T_board_cam"][:3, 3] for h in raw])
    ref = np.median(P[:min(5, len(P))], axis=0)
    dep = None
    end = len(raw)
    for i in range(len(raw)):
        disp = float(np.linalg.norm(P[i] - ref))
        if disp > tol_m:
            dep = (raw[i]["t"], disp)
            end = i
            break
    hits = raw[:end]
    if len(hits) > 1:
        lab = cluster_link(np.array([h["T_board_cam"][:3, 3] for h in hits]),
                           tol_m)
        keep = int(np.argmax(np.bincount(lab)))
        dropped = sum(1 for l in lab if l != keep)
        hits = [h for h, l in zip(hits, lab) if l == keep]
        if dropped:
            print("  agree filter inside the dwell: dropped %d outlier "
                  "sighting(s)" % dropped)
    if len(hits) > target:
        # same viewpoint throughout -- more frames average pixel noise, not
        # bias; cap the count and spread the picks across the dwell
        idx = sorted(set(np.linspace(0, len(hits) - 1,
                                     target).round().astype(int)))
        hits = [hits[i] for i in idx]
    return hits, len(raw), dep


def main():
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "pipeline_config.json"
    P = load_pipeline(cfg_path)
    key, s = get_stage(P, "07_session_init", "06_init")
    map_frame = s.get("map_frame", "map")
    board_cfgs = P.cfg.get("boards", {})
    tol_m = float(s.get("static_tol_mm", 50.0)) * 1e-3
    min_views = int(s.get("min_views", 10))
    max_std = float(s.get("max_std_mm", 30.0))
    target = int(s.get("detect_target", 200))

    af = json.load(open(s["anchor_frame"]))
    axes = af.get("board_axes", "opencv")
    borig = af.get("board_origin", "corner")
    print("board frame convention: axes=%s origin=%s (from %s)"
          % (axes, borig, s["anchor_frame"]))

    bmap = {}
    for name, rec in (af.get("boards") or {}).items():
        bmap[name] = (T_from_record(rec), rec)
    if not bmap:
        raise SystemExit("no 'boards' block in %s -- re-run stage 03"
                         % s["anchor_frame"])
    # design -> surveyed instance names, for the ambiguity warning
    by_design = {}
    for n, (_, rec) in bmap.items():
        by_design.setdefault(rec.get("design", n), []).append(n)

    print("boards in map:")
    for n, (T, rec) in sorted(bmap.items()):
        print("  %-12s frame=%-14s xyz=%s  (std %s mm)"
              % (n, rec.get("frame", "?"), np.round(T[:3, 3], 4).tolist(),
                 rec.get("std_mm", "?")))

    sensors = s["sensors"]
    itopics = [x["image_topic"] for x in sensors]
    ctopics = [x.get("camera_info_topic") for x in sensors
               if x.get("camera_info_topic")]
    print("\nreading %s  (images only -- this stage never needs TF or odometry)"
          % s["bag"])
    imgs, infos, _, available = read_bag(
        s["bag"], image_topics=itopics, info_topics=ctopics, want_tf=False,
        stride=s.get("img_stride", 1), max_images=s.get("max_images", 0))

    out = {"map_frame": map_frame, "bag": str(s["bag"]),
           "board_axes": axes, "board_origin": borig, "sensors": {}}
    lines = []

    for sensor in sensors:
        name = sensor["name"]
        bname = sensor["board"]
        print("\n== %s ==  board=%s  cam=%s" % (name, bname, sensor["cam_frame"]))
        if bname not in bmap:
            print("  ! board '%s' not in anchor_frame.json (have: %s) -> skipped"
                  % (bname, ", ".join(sorted(bmap))))
            continue
        T_map_board, brec = bmap[bname]
        design = brec.get("design", bname)
        if design not in board_cfgs:
            print("  ! design '%s' not in the 'boards' registry -> skipped" % design)
            continue
        insts = by_design.get(design, [bname])
        if len(insts) > 1:
            print("  ! design '%s' has instances %s: this measurement is "
                  "attributed to '%s'. If the robot is actually parked at the "
                  "other instance, every pose below is wrong by the transform "
                  "between them -- the config, not the data, decides this."
                  % (design, insts, bname))
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

        hits, seen, dep = dwell_hits(frames, board, T_fix, K, D, tol_m, target)
        t0s = frames[0][0]
        print("  sightings: %d total on this topic, %d in the opening dwell"
              % (seen, len(hits)))
        if dep is not None:
            print("  dwell ends at t=%.2f s (+%.2f s into the stream): board "
                  "position departed by %.0f mm -> the robot started moving; "
                  "everything after this is ignored"
                  % (dep[0], dep[0] - t0s, dep[1] * 1000))
        elif hits:
            print("  no departure detected -- the robot never left the dwell "
                  "cluster within the scanned frames")
        if not hits:
            print("  ! board never sighted -> no pose for '%s'" % name)
            continue
        lag = hits[0]["t"] - t0s
        if lag > 2.0:
            print("  ! first sighting is %.1f s into the stream -- the dwell "
                  "convention expects the robot to already be staring at the "
                  "board when recording starts. Check the protocol." % lag)
        if len(hits) < min_views:
            print("  ! only %d agreeing sightings (< min_views=%d) -> rejected"
                  % (len(hits), min_views))
            continue

        w = [h["n"] / max(h["reproj"], 0.05) ** 2 for h in hits]
        T_board_cam, n_used = avg_T([h["T_board_cam"] for h in hits], weights=w)
        sp = spread([h["T_board_cam"] for h in hits], T_board_cam)
        rng = [h["range"] for h in hits]
        print("  dwell pose: %d/%d used, %s | range %.2f..%.2f m | "
              "mean reproj %.3f px | span t=%.2f..%.2f s"
              % (n_used, len(hits), fmt_spread(sp), min(rng), max(rng),
                 float(np.mean([h["reproj"] for h in hits])),
                 hits[0]["t"] - t0s, hits[-1]["t"] - t0s))
        if sp[0] > max_std:
            print("  ! scatter %.1f mm exceeds max_std_mm=%.1f -> rejected. "
                  "The 'dwell' was not actually static." % (sp[0], max_std))
            continue

        T_map_cam = T_map_board @ T_board_cam
        bframe = brec.get("frame", "board_%s" % bname)
        cf = sensor.get("tf_child_frame") or ("%s_pose" % name)
        tt = T_board_cam[:3, 3]
        print("  %s -> %s : xyz=[%.4f, %.4f, %.4f]  dist %.3f m  (direct, "
              "board-relative)"
              % (bframe, cf, tt[0], tt[1], tt[2], float(np.linalg.norm(tt))))
        print("  %s -> %s : xyz=%s qxyzw=%s"
              % (map_frame, cf, T_map_cam[:3, 3].round(6).tolist(),
                 R_to_q(T_map_cam[:3, :3]).round(6).tolist()))

        out["sensors"][name] = {
            "board": bname, "board_frame": bframe,
            "cam_frame": sensor["cam_frame"], "tf_child_frame": cf,
            "n_views": len(hits), "n_used": n_used,
            "std_mm": round(sp[0], 2), "max_mm": round(sp[1], 2),
            "max_deg": round(sp[2], 4),
            "mean_reproj_px": round(float(np.mean([h["reproj"] for h in hits])), 4),
            "dwell_t_start": round(hits[0]["t"], 6),
            "dwell_t_end": round(hits[-1]["t"], 6),
            "departure_t": None if dep is None else round(dep[0], 6),
            "board_to_cam": T_record(bframe, cf, T_board_cam),
            "map_to_cam": T_record(map_frame, cf, T_map_cam)}
        lines.append(("[%s -> %s]  (session-start pose of '%s', %d views, "
                      "std %.1f mm)" % (map_frame, cf, name, len(hits), sp[0]),
                      stp(map_frame, cf, T_map_cam)))

    if not out["sensors"]:
        raise SystemExit("\nno sensor produced a session-start pose.")

    with open(s.get("output", "session_start_poses.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("\nwrote %s" % s.get("output", "session_start_poses.json"))

    print("\n=== session-start static transforms ===")
    for hdr, body in lines:
        print("\n%s" % hdr)
        print(body)
    script = s.get("script_out")
    if script:
        with open(script, "w") as f:
            f.write("#!/usr/bin/env bash\n# generated by 07_session_init.py\n"
                    "# bag: %s\nset -e\n\n" % s["bag"])
            for hdr, body in lines:
                f.write("# %s\n%s &\n\n" % (hdr, body.replace("  ros2", "ros2")))
            f.write("wait\n")
        print("\nwrote %s (chmod +x to run)" % script)


if __name__ == "__main__":
    main()
