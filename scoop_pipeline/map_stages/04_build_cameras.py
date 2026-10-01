#!/usr/bin/env python3
"""
STAGE 04 - place each camera in the board-origin `map` frame.

Each camera entry now declares a "source":

  "source": "extrinsic_yaml"   (default, unchanged behaviour)
      T_world_child = T_world_lidar(stamp) @ T_lidar_camera @ T_zedleft_child
      T_map_child   = T_N_world @ T_world_child
      Needs the mapping bag's trajectory and the ZED->child extrinsic YAML saved
      during the mapping session, kept in data/raw/<date>/<pass>/infra_cameras/
      <camera>/ next to its recordings (a bare "extrinsic_yaml" name is looked up
      there, then in infra_cameras/). Used for the arducam.

  "source": "board"            (NEW)
      T_map_child = T_map_board @ inv(T_cam_board)
      The camera PnPs a board whose pose in map stage 03 already exported, and
      that is the whole computation: no trajectory, no time tolerance, no
      T_lidar_camera, no ZED->child extrinsic. Because it needs nothing from the
      mapping run, the images may come from ANY bag -- which is the point for
      the RealSense, whose stream is absent from the mapping bag. Point the
      entry's "bag" at the run where the RealSense sees the big 5x5 board.

      solvePnP returns the board in the camera OPTICAL frame (z forward), so
      what comes out is directly map -> camera_color_optical_frame. Do not route
      it through the camera link frame.

      A moving camera must NOT have its pose averaged, so 04 computes the pose
      per frame, measures the scatter, and only averages if the camera was
      actually static during the sighting; otherwise it takes the single
      lowest-reprojection frame and says so.

  python3 04_build_cameras.py [pipeline_config.json]
"""
import os
import sys
import json
import numpy as np

from pipeline_common import (load_pipeline, load_traj, pose_at, qR,
                             parse_extrinsic_yaml, R_to_q, ang_deg, dump_cameras_yaml)
from pipeline_boards import (Board, read_bag, pick_intrinsics, avg_T, spread,
                             fmt_spread, T_from_record)


def xyzquat_to_T(v):
    x, y, z = map(float, v[:3])
    T = np.eye(4)
    T[:3, :3] = qR([float(c) for c in v[3:7]])
    T[:3, 3] = [x, y, z]
    return T


def load_T_map_mapzed(af, s):
    """Pose of map_zed in map. Only needed to co-frame the ChArUco reference
    pose carried by an extrinsic YAML before differencing it. Returns None if
    unavailable -- board-sourced cameras never need it."""
    if "T_map_mapzed" in af:
        return np.array(af["T_map_mapzed"], float)
    if "T_mapzed_map" in af:
        return np.linalg.inv(np.array(af["T_mapzed_map"], float))
    if "map_to_mapzed_xyzquat" in s:
        return xyzquat_to_T(s["map_to_mapzed_xyzquat"])
    if "mapzed_to_map_xyzquat" in s:
        return np.linalg.inv(xyzquat_to_T(s["mapzed_to_map_xyzquat"]))
    return None


def boards_in_map(af, cfg):
    """{name: T_map_board} from anchor_frame.json, with a legacy fallback."""
    out = {}
    for name, rec in (af.get("boards") or {}).items():
        out[name] = (T_from_record(rec), rec)
    if not out and af.get("map_to_board_qxyzw"):
        T = np.eye(4)
        T[:3, :3] = qR(af["map_to_board_qxyzw"])
        out["anchor"] = (T, {"frame": "board", "method": "origin"})
    return out


# --------------------------------------------------------------------------- #
def camera_from_yaml(cam, ctx):
    name = cam["name"]; child = cam["child_frame"]; ypath = cam["extrinsic_yaml"]
    print("  extrinsic: %s" % ypath)
    if not os.path.exists(ypath):
        print("  ! not found -> skipped. Put the yaml saved during the mapping session "
              "there (data/raw/<date>/<pass>/infra_cameras/<camera>/)")
        return None
    info = parse_extrinsic_yaml(ypath)
    ext, stamp, pim_zed = info["ext"], info["stamp"], info["pose_in_map_zed"]
    if ext is None or stamp is None:
        print("  ! missing translation/quaternion_xyzw or stamp -> skipped")
        return None
    T_ml_lidar, gap = pose_at(ctx["tr_t"], ctx["tr_T"], stamp)
    if gap > ctx["tol"]:
        print("  ! stamp %.3f outside traj (gap %.0f ms) -> skipped" % (stamp, gap * 1000))
        return None

    T_world_child = T_ml_lidar @ ctx["T_lidar_cam"] @ ext
    T_map_child = ctx["T_N_world"] @ T_world_child
    t = T_map_child[:3, 3]; q = R_to_q(T_map_child[:3, :3])
    rec = {"name": name, "child_frame": child, "parent_frame": ctx["map_frame"],
           "source": "extrinsic_yaml", "stamp": stamp, "traj_gap_s": gap,
           "translation": t.tolist(), "quaternion_xyzw": q.tolist()}
    print("  map->%s: t=%s (traj gap %.0f ms)" % (child, t.round(4).tolist(), gap * 1000))

    if pim_zed is not None:
        T_map_mapzed = ctx["T_map_mapzed"]
        if T_map_mapzed is None:
            print("  ! ChArUco reference present but no map<->map_zed transform "
                  "known -> delta not computed (set map_to_mapzed_xyzquat, or "
                  "re-run 03 so anchor_frame.json carries T_map_mapzed).")
        else:
            # The reference lives in map_zed; map and map_zed differ by a large
            # yaw, so it must be lifted into map BEFORE differencing.
            pim_map = T_map_mapzed @ pim_zed
            dt = t - pim_map[:3, 3]
            rec["zed_pose_in_map_translation"] = pim_map[:3, 3].tolist()
            rec["raw_delta_cm"] = float(np.linalg.norm(dt) * 100)
            rec["raw_delta_deg"] = ang_deg(T_map_child[:3, :3], pim_map[:3, :3])
            print("  LiDAR-vs-ChArUco delta (co-framed in map): %.1f cm / %.2f deg"
                  % (rec["raw_delta_cm"], rec["raw_delta_deg"]))
    return rec


def camera_from_board(cam, ctx):
    name = cam["name"]; child = cam["child_frame"]
    bname = cam["board"]
    if bname not in ctx["boards"]:
        print("  ! board '%s' has no pose in map (stage 03 did not export it) "
              "-> skipped" % bname)
        return None
    T_map_board, brec = ctx["boards"][bname]
    B = Board(bname, ctx["board_cfgs"][bname])
    print("  %s" % B.describe())
    print("  board pose in map: %s (%s, %d views, std %.1f mm)"
          % (np.round(T_map_board[:3, 3], 4).tolist(), brec.get("method", "?"),
             brec.get("n_views", -1), brec.get("std_mm", float("nan"))))

    bag = cam.get("bag") or ctx["default_bag"]
    itopic = cam["image_topic"]; ctopic = cam.get("camera_info_topic")
    imgs, infos, _tree, available = read_bag(
        bag, image_topics=[itopic], info_topics=[ctopic] if ctopic else [],
        want_tf=False, stride=cam.get("img_stride", 1),
        max_images=cam.get("max_images", 0))
    frames = imgs.get(itopic, [])
    if not frames:
        print("  ! no images on %s in %s\n    topics: %s"
              % (itopic, bag, "\n    ".join(available)))
        return None
    K, D, from_bag = pick_intrinsics(infos, ctopic, cam.get("rectified", False),
                                     cam.get("K"), cam.get("dist"))
    print("  intrinsics: %s | fx=%.2f cx=%.2f"
          % ("bag CameraInfo" if from_bag else "config", K[0, 0], K[0, 2]))

    target = int(cam.get("detect_target", 20))
    poses = []   # (T_map_cam, reproj, stamp, n_corners)
    for st, gray in frames:
        if len(poses) >= target:
            break
        d = B.detect(gray, K, D)
        if d is None:
            continue
        poses.append((T_map_board @ np.linalg.inv(d.T), d.reproj, st, d.n))
    if not poses:
        print("  ! board '%s' never detected in %d frames of %s" % (bname, len(frames), itopic))
        return None

    Ts = [p[0] for p in poses]
    T_avg, _ = avg_T(Ts)          # (pose, number of poses kept)
    sp = spread(Ts, T_avg)
    static_mm = float(cam.get("static_tol_mm", 20.0))
    print("  detections %d | %s" % (len(poses), fmt_spread(sp)))

    if sp[1] <= static_mm:
        T_map_child = T_avg
        method = "board_avg"
        stamp = float(np.mean([p[2] for p in poses]))
        print("  camera was static over the sighting -> averaged %d poses" % len(poses))
    else:
        best = min(poses, key=lambda p: p[1])
        T_map_child = best[0]
        method = "board_single"
        stamp = best[2]
        print("  camera MOVED during the sighting (max %.1f mm > %.1f mm) -> using "
              "the single best frame (reproj %.2f px, %d corners, stamp %.3f)"
              % (sp[1], static_mm, best[1], best[3], best[2]))

    t = T_map_child[:3, 3]; q = R_to_q(T_map_child[:3, :3])
    print("  map->%s: t=%s q=%s" % (child, t.round(4).tolist(), q.round(5).tolist()))
    return {"name": name, "child_frame": child, "parent_frame": ctx["map_frame"],
            "source": method, "board": bname, "bag": str(bag), "stamp": stamp,
            "n_detections": len(poses),
            "spread_std_mm": round(sp[0], 2), "spread_max_mm": round(sp[1], 2),
            "spread_max_deg": round(sp[2], 4),
            "mean_reproj_px": round(float(np.mean([p[1] for p in poses])), 3),
            "translation": t.tolist(), "quaternion_xyzw": q.tolist()}


# --------------------------------------------------------------------------- #
def main():
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "pipeline_config.json"
    P = load_pipeline(cfg_path)
    S = P.sensor
    s = P.stage("04_build_cameras")

    s["anchor_frame"] = P.anchor_frame(os.path.basename(s["anchor_frame"]))
    af = json.load(open(s["anchor_frame"]))
    T_N_world = np.array(af["T_N_world"], float)
    map_frame = af.get("map_frame", s.get("map_frame", "map"))
    if P.reference:
        # a run (dataset.reference_pass): its poses are stage 08's LiDAR-ICP
        # track, registered to the reference pass's ANCHORED map -- already in
        # map, so no GLIM-world -> map step
        T_N_world = np.eye(4)

    ctx = {"map_frame": map_frame,
           "T_N_world": T_N_world,
           "tol": float(s.get("time_tol", 0.10)),
           "boards": boards_in_map(af, P.cfg),
           "board_cfgs": P.cfg.get("boards", {}),
           "default_bag": P.dataset["bag"],
           # map_zed is the ZED's own map of ONE session: the reference pass's
           # map -> map_zed does not hold for a run, so no ChArUco delta there
           "T_map_mapzed": None if P.reference else load_T_map_mapzed(af, s),
           "T_lidar_cam": S.T_lidar_camera,
           "tr_t": None, "tr_T": None}

    print("anchor_frame: %s  (map_frame='%s')" % (s["anchor_frame"], map_frame))
    print("boards in map: %s" % ", ".join(sorted(ctx["boards"])) or "(none)")

    cameras = [c for c in s["cameras"] if c.get("enabled", True)]
    needs_traj = any(c.get("source", "extrinsic_yaml") == "extrinsic_yaml"
                     for c in cameras)
    if needs_traj:
        traj = P.lidar_track_traj() if P.reference else P.outp(P.dataset["traj"])
        ctx["tr_t"], ctx["tr_T"] = load_traj(traj)
        print("traj: %s, %d poses [%.3f, %.3f]%s"
              % (traj, len(ctx["tr_t"]), ctx["tr_t"][0], ctx["tr_t"][-1],
                 " (stage 08, in map)" if P.reference else ""))

    out = []
    for cam in cameras:
        src = cam.get("source", "extrinsic_yaml")
        print("\n== %s (%s) ==  source: %s" % (cam["name"], cam["child_frame"], src))
        if src == "extrinsic_yaml":
            rec = camera_from_yaml(cam, ctx)
        elif src == "board":
            rec = camera_from_board(cam, ctx)
        else:
            raise SystemExit("camera '%s': unknown source '%s' "
                             "(use 'extrinsic_yaml' or 'board')" % (cam["name"], src))
        if rec is not None:
            out.append(rec)

    if not out:
        raise SystemExit("no cameras produced (check stamps vs traj / bags / topics).")
    dump_cameras_yaml(s["output"], map_frame, S.T_lidar_camera_7, out)
    print("\nwrote %d cameras -> %s" % (len(out), s["output"]))


if __name__ == "__main__":
    main()
