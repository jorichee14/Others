#!/usr/bin/env python3
"""One pass from raw to processed, every step in order.

    python scoop_pipeline/run_pass.py <raw pass folder> [--status] [--dry]
        [--from STEP] [--until STEP] [--only STEP] [--config FILE]

<raw pass folder> is data/raw/<date>/<pass>, e.g.
    ~/workspaces/isaac_ros-dev/data/raw/20260924/survey_1

Steps (a step whose output is there already is skipped):

  raw -> work
    process       processing/process_recording.py on every machine folder with an
                  Ouster packets bag or an SVO2: decoded, retimed, GLIM, ZED bags
    merge         processing/merge_session.py: check against record.yaml, merge
    vslam         processing/vslam.py: a robot's odometry by cuVSLAM from its stereo +
                  IMU in the merged bag (only with a "vslam" block: mobile_2 in coop)
  work -> processed (map_stages/, the pipeline config of this pass)
    mapping pass (no dataset.reference_pass):
    02_ba         02_bundle_adjust.py        GLIM's poses (time-corrected on export)
                                             bundle-adjusted on the planes: the reference
                                             trajectory (traj_lidar_ba.txt)
    time_check    check.py time              GLIM's pose time against it (on time / OFF)
    01_seed       01_build_map.py --seed     map from the GLIM poses (02b's reference) *
    02b_s2m       02b_scan_to_map.py         every scan registered to that map: the
                                             comparison (traj_lidar_refined.txt) *
    01_map        01_build_map.py            the map from dataset.traj (02's output),
                                             fused as 01_build_map.fusion says (TSDF)
    01_union      01_build_map.py --union    the same poses, union of points: the
                                             comparison map (only beside a TSDF) *
    report        check.py report            GLIM, + correction, (+ scan-to-map,) + BA
                                             against the reference; the map (and the union)
    * only with "comparisons": true in the config: the paper's comparisons, made
      once (20261015), not on every mapping pass
    a run in a mapping pass's map (dataset.reference_pass):
    03_init       03_init_from_boards.py     start pose from the anchor board
    04_reference  04_reference_traj.py       every scan registered to the anchored map
                                             (and 04's "tracks": mobile_2's depth frames)
    time_check    check.py time              GLIM's pose time against 04's track
    both:
    05_anchor     05_anchor.py               boards; a mapping pass: the anchored map
    06_cut        06_cut.py                  viewing copy, only when 06_cut has floor/ceil
    07_cameras    07_build_cameras.py        infra / parked cameras in map
    08_tfs        08_emit_tfs.py             frames/publish_tfs_<tag>.sh
    09_cloud      09_build_coop_cloud.py     a run's cloud in the map (runs only)
    10_poses      10_publish_poses.py        bags/<tag>_best_poses: poses + complete TF
    finalize      the merged bag again, with 10's /tf, /tf_static and poses in it,
                  as <merged bag>_final in its place (the stages read it for the
                  bag from then on)
                  (built next to it, then put in its place)
    comms         processing/comms_tables.py: NTP, Wi-Fi, ping, iperf, CSI tables with
                  each machine's position (processed/<date>/<pass>/comms/)
    csi_clean     processing/csi_clean.py: per link, QoS data frames without pilots,
                  H and H scaled to RSSI, both ends' poses (comms/csi/*_clean.npz)

--status  what is done, and stop      --dry     print the commands, run nothing
--from S  run S and every step after it, done or not
--until S stop after S                --only S  run S alone, done or not
A map stage whose block in the pipeline config has "enabled": false is not run
(e.g. a coop run: "05_anchor": {"enabled": false}, the boards as the mapping
pass placed them); "comms": {"enabled": false} leaves out comms and csi_clean
(a pass that recorded no comms topics, e.g. mapping_20261015).

--config  the pipeline config (default: the one in map_stages/ whose dataset is
          this pass, e.g. pipeline_config_survey_1.json; a run needs its own,
          "extends": "pipeline_config.json" with dataset.reference_pass)

--new-config  write the config of a new pass (map_stages/pipeline_config_<tag>.json)
          from its raw folder, show it, and stop: check it, then run the pass.
          A mapping pass: the robot with an Ouster (mobile_1 if both), the same
          settings as 20261015 (BA on GLIM, 2 cm TSDF). A survey: mobile_1 drives,
          in the same day's mapping pass's map; a parked mobile_2 placed by its
          LiDAR (or, without one, its board view). The bag: the robot's own when it
          is alone and has no ZED, else the merged one. --robot, --reference
          override what it picks. A coop run: both robots drive; mobile_2 with an
          Ouster bag as coop/site_1 (GLIM on it, every scan to the map), without as
          coop_2 (cuVSLAM + its depth against the map); the boards and the infra
          camera from the day's survey passes (boards_from / cameras_from). Every bag's
          time span is listed, one that does not overlap the run flagged (a bag
          from another run in the folder).

Datasets for other tools (datasets/mcap_convert.py) are made afterwards from the
merged bag and the processed outputs.
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
STAGES = os.path.join(ROOT, "map_stages")
sys.path.insert(0, ROOT)
sys.path.insert(0, STAGES)

MAPPING = ["02_ba", "time_check", "01_seed", "02b_s2m", "01_map", "01_union", "report"]
RUN = ["03_init", "04_reference", "time_check"]
# each map stage's block in the pipeline config
BLOCK = {"02_ba": "02_bundle_adjust", "01_seed": "02b_scan_to_map",
         "02b_s2m": "02b_scan_to_map", "01_map": "01_build_map", "01_union": "01_build_map",
         "03_init": "03_init", "04_reference": "04_reference", "05_anchor": "05_anchor",
         "06_cut": "06_cut", "07_cameras": "07_build_cameras", "08_tfs": "08_emit_tfs",
         "09_cloud": "09_build", "10_poses": "10_publish"}
ORDER = (["process", "merge", "vslam"] + MAPPING + [s for s in RUN if s not in MAPPING]
         + ["05_anchor", "06_cut", "07_cameras", "08_tfs", "09_cloud", "10_poses", "finalize",
            "comms", "csi_clean"])


def py(script, *args):
    return [sys.executable, script, *args]


def machines(raw):
    """Machine folders with something process_recording.py works on."""
    from scoop import recording
    from scoop.bag import BagError
    out = []
    if not os.path.isdir(raw):
        return out
    settings = recording.load_settings(None)
    for d in sorted(p for p in os.listdir(raw) if os.path.isdir(os.path.join(raw, p))):
        s = recording.settings_for(settings, d)                 # recording.yaml machines:
        try:
            recording.find_recording(os.path.join(raw, d), None,
                                     s["ouster"].get("packets_ns", "/ouster"),
                                     recording.zed_bags(s["zed"]))
        except (BagError, OSError, ValueError):
            continue
        out.append(os.path.join(raw, d))
    return out


def merged_bag(raw):
    """Where merge_session.py writes this pass's merged bag; None without bags."""
    from scoop import recording, session
    from scoop.bag import BagError
    try:
        return _merged_bag(raw, recording, session)
    except (BagError, OSError, ValueError):
        return None


def _merged_bag(raw, recording, session):
    plan = session.RecordPlan(None)
    settings = recording.load_settings(None)
    rawp = recording.Path(raw).expanduser().resolve()
    mode = plan.mode_of(rawp.name, rawp.parent.name,         # survey/site_1: "survey"
                        *(p.parent.name for p in rawp.rglob("metadata.yaml")))
    units = session.find_units(rawp, plan, settings, None, mode)
    return os.path.join(str(recording.work_dir_for(raw, None)),
                        "%s_merged" % session.session_name(units, plan, rawp))


def has_map_tf(bag):
    from scoop import tftree
    edges = tftree.tf_edges([bag], seconds=1.0)
    return any(e.parent == "map" for e in edges.values())


RS_BOARD = "board_4"        # the board mobile_2's RealSense sights (7x7, since 20261015)


def bag_spans(raw):
    """[(bag folder relative to raw, start s, end s)] of every rosbag2 folder
    below raw, from its metadata.yaml."""
    import yaml
    from pathlib import Path
    out = []
    for m in sorted(Path(raw).rglob("metadata.yaml")):
        info = (yaml.safe_load(open(m)) or {}).get("rosbag2_bagfile_information") or {}
        t0 = (info.get("starting_time") or {}).get("nanoseconds_since_epoch")
        d = (info.get("duration") or {}).get("nanoseconds")
        if t0 is not None and d is not None:
            out.append((str(m.parent.relative_to(raw)), t0 * 1e-9, (t0 + d) * 1e-9))
    return out


def check_spans(spans, log=print, least=0.5):
    """The bags' time spans side by side, against the longest one (the
    driving robot's): a bag that overlaps it by less than `least` of its own
    length is from another run (a sniffer bag copied into the wrong folder).
    -> the names of those."""
    import datetime
    if not spans:
        return []
    S, E = max(spans, key=lambda x: x[2] - x[1])[1:]
    log("the bags' time spans (UTC), against the longest:")
    bad = []
    for name, s, e in spans:
        ov = max(0.0, min(e, E) - max(s, S)) / max(e - s, 1e-9)
        flag = ""
        if ov < least:
            flag = "   ! overlaps the run only %.0f %%: from another run?" % (100 * ov)
            bad.append(name)
        log("  %-60s %s  %6.0f s%s" % (name, datetime.datetime.utcfromtimestamp(s).strftime("%Y-%m-%d %H:%M:%S"),
                                        e - s, flag))
    return bad


def new_config(raw, robot=None, reference=None, log=print, stages=STAGES):
    """--new-config: the pipeline config of the pass in ``raw``, written to
    map_stages/pipeline_config_<tag>.json (see the usage). -> its path."""
    from pathlib import Path
    from scoop import layout, recording, session
    import pipeline_common as pc
    pp = layout.pass_parts(raw)
    if not pp:
        raise SystemExit("%s is not below data/raw/<date>/<pass>/" % raw)
    _, date, pas = pp
    tag = layout.tag(date, pas)
    path = os.path.join(stages, "pipeline_config_%s.json" % tag)
    if os.path.exists(path):
        raise SystemExit("%s exists already: edit it, or remove it to write it again" % path)
    rawp = Path(raw).resolve()
    plan = session.RecordPlan(None)
    mode = plan.mode_of(rawp.name, rawp.parent.name,
                        *(p.parent.name for p in rawp.rglob("metadata.yaml")))
    if mode not in ("mapping", "survey", "coop"):
        raise SystemExit("mode %s (from the folder and bag names): --new-config writes mapping, "
                         "survey and coop passes" % (mode or "unknown"))
    settings = recording.load_settings(None)
    recs = {}
    for m in machines(raw):
        s = recording.settings_for(settings, os.path.basename(m))
        recs[os.path.basename(m)] = recording.find_recording(
            m, None, s["ouster"].get("packets_ns", "/ouster"), recording.zed_bags(s["zed"]))
    with_lidar = sorted(n for n, r in recs.items() if r.bag is not None)
    if robot is None:
        if mode == "survey" or len(with_lidar) != 1:
            robot = "mobile_1" if "mobile_1" in with_lidar else None
        else:
            robot = with_lidar[0]
    if robot not in with_lidar:
        raise SystemExit("which robot %s? with an Ouster bag: %s (--robot)"
                         % ("maps" if mode == "mapping" else "drives", with_lidar or "none"))
    folders = {p.relative_to(rawp).parts[0] for p in rawp.rglob("metadata.yaml")
               if len(p.relative_to(rawp).parts) > 2}
    if folders == {robot} and recs[robot].svo is None:
        bag = str(recs[robot].step("retimed"))            # alone, no ZED: its own bag
    else:
        bag = merged_bag(raw)
    home = os.path.expanduser("~")
    bag = "~" + bag[len(home):] if bag.startswith(home + os.sep) else bag
    boards = ["board_1", "board_2", "board_3", "board_4"]
    cfg = {"extends": "pipeline_config.json"}
    if mode == "mapping":
        cfg["_note"] = ("%s: mapping pass by %s, written by run_pass.py --new-config; as "
                        "20261015: 02 BA on GLIM (traj_lidar_ba_glim.txt, the reference) -> "
                        "01 TSDF at 2 cm -> report -> 05 -> 10 -> finalize" % (tag, robot))
        cfg["dataset"] = {"bag": bag, "traj": "traj_lidar_ba_glim.txt", "machine": robot}
        cfg["01_build_map"] = {"scan_voxel": 0.02, "final_voxel": 0.02, "fusion": "tsdf",
                               "tsdf": {"trunc": 0.08, "min_weight": 3}}
        cfg["02_bundle_adjust"] = {"input": "traj_lidar.txt", "output": "traj_lidar_ba_glim.txt"}
        cfg["05_anchor"] = {"boards_used": boards}
        cfg["07_build_cameras"] = {"enabled": False}
        cfg["08_emit_tfs"] = {"enabled": False}
        cfg["comms"] = {"enabled": False, "_note": "mapping mode: no wifi / ntp topics"}
    else:
        maps, surveys = [], []                       # the date's passes with a config here
        for c in sorted(glob.glob(os.path.join(stages, "pipeline_config*.json"))):
            d = pc.load_config(c).get("dataset") or {}
            rp = layout.pass_parts(d.get("bag") or "", below=1)
            if rp and rp[1] == date:
                name_ = "/".join(rp[2])
                if not d.get("reference_pass"):
                    if name_ not in maps:                 # the base config names the same pass
                        maps.append(name_)
                elif plan.mode_of(*rp[2]) == "survey" and name_ not in surveys:
                    surveys.append(name_)                 # a survey's 05 measured the boards
        if reference is None:
            if len(maps) != 1:
                raise SystemExit("the map to localize in: %s mapping passes on %s in map_stages/ "
                                 "(%s); name it with --reference <pass> or <date>/<pass>"
                                 % (len(maps), date, ", ".join(sorted(maps)) or "none"))
            reference = maps[0]
    if mode == "coop":
        lidar2 = "mobile_2" in with_lidar
        cfg["_note"] = ("%s: coop run, written by run_pass.py --new-config, in %s's map: both robots "
                        "drive. mobile_1: 03 (its ZED at its start board, through GLIM) -> 04 (every scan "
                        "to the anchored map). mobile_2: %s. No 05 (the boards as the day's survey "
                        "measured them: dataset.boards_from), no 07/08 (the infra camera where that "
                        "survey's 07 put it: cameras_from). Change \"board\" (03's realsense sensor and the "
                        "04 track) to the board mobile_2 starts in front of if not %s (board_scan.py "
                        "--image-topic /mobile_2/color/image_raw --camera-info /mobile_2/color/camera_info "
                        "shows which)."
                        % (tag, reference,
                           ("as coop/site_1 (20261015): GLIM on its own Ouster, its start from 03's "
                            "realsense (its RealSense at the board in the opening dwell, through its "
                            "LiDAR -> camera calibration) -> 04's track mobile_2_lidar, every scan to "
                            "the same map" if lidar2 else
                            "as coop_2 (20260924): vslam (cuVSLAM offline on its stereo + IMU) -> 03 (its "
                            "colour camera at the board in the opening dwell) -> 04's track mobile_2_depth "
                            "(method pgo: one pose graph over every depth frame, the start anchor, the "
                            "boards and the anchored map)"), RS_BOARD))
        cfg["dataset"] = {"bag": bag, "traj": None, "machine": robot, "reference_pass": reference,
                          "boards_from": surveys, "cameras_from": surveys}
        if not surveys:
            cfg["dataset"]["_boards_from"] = ("no survey of %s has a config in map_stages/: name the pass "
                                              "whose 05 measured the boards" % date)
        cfg["03_init"] = {"sensors": [{"name": "zed", "source": "lidar_odom"},
                                      {"name": "realsense", "source": "camera", "board": RS_BOARD,
                                       "dwell_only": True}]}
        if lidar2:
            track = {"name": "mobile_2_lidar", "kind": "lidar", "machine": "mobile_2",
                     "anchor_cam": "realsense", "body_frame": "os_lidar_mobile2",
                     "origin_frame": "map_realsense", "board": RS_BOARD}
        else:
            cfg["vslam"] = {"machine": "mobile_2"}
            track = {"name": "mobile_2_depth", "machine": "mobile_2", "body_frame": "camera_link",
                     "origin_frame": "map_realsense", "board": RS_BOARD, "anchor_cam": "realsense",
                     "method": "pgo"}
        cfg["04_reference"] = {"tracks": [track]}
        cfg["05_anchor"] = {"enabled": False}
        cfg["07_build_cameras"] = {"enabled": False}
        cfg["08_emit_tfs"] = {"enabled": False}
        cfg["10_publish"] = {"robots": [{"name": "mobile_2", "traj": "traj_%s_in_cam.tum" % track["name"],
                                         "optical_frame": "camera_color_optical_frame"}]}
    elif mode == "survey":
        cfg["_note"] = ("%s: survey, written by run_pass.py --new-config: %s drives, in %s's "
                        "map; 03 -> 04 -> 05 (the boards this run saw) -> 07 (parked and infra "
                        "cameras) -> 08 -> 10 -> finalize -> comms" % (tag, robot, reference))
        cfg["dataset"] = {"bag": bag, "traj": None, "machine": robot,
                          "reference_pass": reference}
        cfg["03_init"] = {"sensors": [{"name": "zed", "source": "lidar_odom"},
                                      {"name": "realsense", "enabled": False}]}
        cfg["05_anchor"] = {"boards_used": boards}
        if "mobile_2" in folders:
            cam = {"name": "realsense", "board": RS_BOARD, "bag": None,
                   "image_topic": "/mobile_2/color/image_raw",
                   "camera_info_topic": "/mobile_2/color/camera_info",
                   "rectified": False, "detect_target": 200}
            if "mobile_2" in with_lidar:                   # parked, with its Ouster
                cam = dict(cam, source="lidar", points_topic="/mobile_2/ouster/points")
            cfg["07_build_cameras"] = {"cameras": [cam]}
    with open(path, "w") as fh:
        json.dump(cfg, fh, indent=1)
        fh.write("\n")
    P = pc.load_pipeline(path)
    log("wrote %s:\n" % path)
    log(open(path).read())
    log("%s pass, %s; its sensors (from the base config's robots block):" % (mode, P.machine))
    if mode == "coop":
        log("  boards and the infra camera from: %s" % (", ".join(cfg["dataset"]["boards_from"]) or "NONE"))
        log("  mobile_2's start board: %s (the 04 track; change it if it starts elsewhere)" % RS_BOARD)
    for k in ("points_topic", "image_topic", "camera_info_topic", "camera_frame",
              "camera_map_frame", "rectified"):
        log("  %-18s %s" % (k, P.sensor_setting(k)))
    log("")
    bad = check_spans(bag_spans(raw), log)
    if bad:
        log("fix the bag(s) above before the merge: %s" % ", ".join(bad))
    log("\ncheck it, then: python3 scoop_pipeline/run_pass.py %s --status" % raw)
    return path


def plan_steps(raw, cfg):
    """[(step, applies, done, what, command or callable)]"""
    import pass_tf
    steps = []
    from pipeline_common import final_bag
    mb = merged_bag(raw)
    mb = final_bag(mb) if mb else mb                    # finalize renames it <merged>_final
    P = pass_tf.load_pipeline(cfg) if cfg else None
    db = os.path.normpath(P.dataset["bag"]) if P else None           # <bag>_final once finalized
    # one robot alone: the config reads its own processed bag in its machine
    # folder (work/<date>/<pass>/<machine>/), and there is nothing to merge
    own = bool(P and os.path.basename(os.path.dirname(db)) == P.machine)
    # once merged, the processed bags may be gone: processing is not redone for them
    have = os.path.isdir(db) if own else (mb is not None and os.path.isdir(mb))
    steps.append(("process", mb is not None or own, have, "%d machine folder(s)" % len(machines(raw))
                  if os.path.isdir(raw) else "no %s" % raw,
                  [py(os.path.join(ROOT, "processing", "process_recording.py"), m)
                   for m in machines(raw)]))
    steps.append(("merge", mb is not None and not own, have,
                  mb or "no bags below %s" % raw,
                  [py(os.path.join(ROOT, "processing", "merge_session.py"), raw)]))
    if not cfg:
        if mb:
            steps.append(comms_step(mb, None))
            steps.append(csi_clean_step(mb, None))
        return steps, None
    run = bool(P.reference)
    st = lambda name: os.path.join(STAGES, name)                     # noqa: E731
    tag = P.tag
    blk = lambda k: P.cfg.get(k) or {}                               # noqa: E731
    stg = lambda k, f, d: P.stage(k).get(f) or P.outp(d) if k in P.cfg else P.outp(d)  # noqa: E731
    sys.path.insert(0, os.path.join(ROOT, "analysis"))
    import trajs
    fusion = blk("01_build_map").get("fusion", "union")
    out = {
        "02_ba": trajs.reference_traj(P) if not run else None,
        "time_check": trajs.odometry_file(P, "pose_time_check.json"),
        "01_seed": P.outp("denoised_seed_%s.pcd" % tag),
        "02b_s2m": P.outp(blk("02b_scan_to_map").get("output", "traj_lidar_refined.txt")),
        "01_map": P.outp(blk("01_build_map").get("output") or P.pcd("map_final")),
        "01_union": P.outp("denoised_union_%s.pcd" % tag if tag else "denoised_union.pcd"),
        "report": os.path.join(P.out_dir, "mapping_report_%s.md" % tag if tag
                               else "mapping_report.md"),
        "03_init": stg("03_init", "output", "session_anchor.json"),
        "04_reference": P.lidar_track_traj(),
        "05_anchor": P.outp("boards_{tag}.json") if run
        else stg("05_anchor", "frame_out", "anchor_frame.json"),
        "06_cut": stg("06_cut", "output", "map_final_{tag}_anchored_noceil.pcd"),
        "07_cameras": stg("07_build_cameras", "output", "cameras_in_map.yaml"),
        "08_tfs": P.outp("publish_tfs_{tag}.sh"),
        "09_cloud": P.outp("cloud_in_ref_{tag}.pcd"),
        "10_poses": P.outp(blk("10_publish").get("output_bag") or "{tag}_best_poses"),
    }
    cut = blk("06_cut")
    applies = {s: not run for s in MAPPING}
    applies.update({s: run for s in RUN if s not in MAPPING})
    applies["time_check"] = True
    compare = bool(P.cfg.get("comparisons", False))       # the paper's, not every pass's
    for s_ in ("01_seed", "02b_s2m"):
        applies[s_] = not run and compare
    applies["01_union"] = not run and compare and fusion == "tsdf"
    applies.update({"05_anchor": True, "06_cut": not run and (cut.get("floor") is not None
                                                             or cut.get("ceil") is not None),
                    "07_cameras": True, "08_tfs": True, "09_cloud": run, "10_poses": True})
    for s_, k in BLOCK.items():                      # "enabled": false in its block: not run
        if blk(k).get("enabled", True) is False:
            applies[s_] = False
    cmds = {
        "02_ba": [py(st("02_bundle_adjust.py"), cfg)],
        "time_check": [py(os.path.join(ROOT, "check.py"), "time", cfg)],
        "01_seed": [py(st("01_build_map.py"), cfg, "--seed")],
        "02b_s2m": [py(st("02b_scan_to_map.py"), cfg)],
        "01_map": [py(st("01_build_map.py"), cfg)],
        "01_union": [py(st("01_build_map.py"), cfg, "--union")],
        "report": [py(os.path.join(ROOT, "check.py"), "report", cfg)],
        "03_init": [py(st("03_init_from_boards.py"), cfg)],
        "04_reference": [py(st("04_reference_traj.py"), cfg)],
        "05_anchor": [py(st("05_anchor.py"), cfg)],
        "06_cut": [py(st("06_cut.py"), cfg)],
        "07_cameras": [py(st("07_build_cameras.py"), cfg)],
        "08_tfs": [py(st("08_emit_tfs.py"), cfg) + [">", out["08_tfs"]]],
        "09_cloud": [py(st("09_build_coop_cloud.py"), cfg)],
        "10_poses": [py(st("10_publish_poses.py"), cfg)],
    }
    done = {k: bool(v) and os.path.exists(v) for k, v in out.items()}
    done["01_seed"] = done["01_seed"] or done["02b_s2m"]         # only 02b reads it
    tracks = [t for t in blk("04_reference").get("tracks") or [] if t.get("enabled", True)]
    done["04_reference"] = done["04_reference"] and all(
        os.path.exists(os.path.join(P.reference_dir(), "traj_%s.tum" % t["name"])) for t in tracks)
    if "vslam" in P.cfg and P.cfg["vslam"].get("enabled", True):
        sys.path.insert(0, os.path.join(ROOT, "processing"))
        import vslam
        vt = vslam.traj_path(P)
        steps.append(("vslam", True, os.path.exists(vt), vt,
                      [py(os.path.join(ROOT, "processing", "vslam.py"), cfg)]))
    # time_check follows 02 in a mapping pass, 04 in a run
    seq = ([s for s in MAPPING if s != "time_check"] + RUN if run
           else MAPPING + [s for s in RUN if s != "time_check"])
    for s in seq + ["05_anchor", "06_cut", "07_cameras", "08_tfs", "09_cloud", "10_poses"]:
        steps.append((s, applies[s], done[s], out[s], cmds[s]))
    if mb and not own and db != os.path.normpath(mb):
        print("note: dataset.bag of %s is %s; the merge writes %s" % (cfg, db, mb))
    steps.append(("finalize", True, lambda: finalized(db, out["10_poses"]), db,
                  lambda dry: finalize(db, out["10_poses"], dry)))
    on = (P.cfg.get("comms") or {}).get("enabled", True) is not False   # a pass with no comms topics
    for st_ in (comms_step(db, cfg), csi_clean_step(db, cfg)):
        steps.append(st_[:1] + (st_[1] and on,) + st_[2:])
    return steps, P


def csi_clean_step(bag, cfg):
    """CSI for use: data frames, no pilots, H scaled to RSSI, poses."""
    from pipeline_common import processed_dir_for
    proc = processed_dir_for(bag)
    done = os.path.join(proc, "comms", "csi", "csi_clean_summary.json")
    cmd = py(os.path.join(ROOT, "processing", "csi_clean.py"), proc)
    return ("csi_clean", True, os.path.exists(done), os.path.dirname(done),
            [cmd + (["--config", cfg] if cfg else [])])


def comms_step(bag, cfg):
    """NTP, Wi-Fi, ping, iperf, CSI tables (processed/<date>/<pass>/comms/).
    The bag is looked up when the step runs: finalize, just before it, renames
    the merged bag <bag>_final."""
    from pipeline_common import processed_dir_for, final_bag
    summ = os.path.join(processed_dir_for(bag), "comms", "summary.json")
    cmd = lambda: py(os.path.join(ROOT, "processing", "comms_tables.py"), final_bag(bag))  # noqa: E731
    return ("comms", True, os.path.exists(summ), os.path.dirname(summ),
            lambda dry: run_step("comms", [cmd() + (["--config", cfg] if cfg else [])], dry))


def final_name(merged):
    """<bag>_final: the name finalize gives the bag it makes."""
    from pipeline_common import FINAL
    m = os.path.normpath(merged)
    return m if m.endswith(FINAL) else m + FINAL


def has_poses(merged, poses):
    """The bag is newer than 10's bag and has the tree from map."""
    meta = lambda b: os.path.join(b, "metadata.yaml")                # noqa: E731
    return (os.path.exists(meta(merged)) and os.path.exists(meta(poses))
            and os.path.getmtime(meta(merged)) > os.path.getmtime(meta(poses))
            and has_map_tf(merged))


def finalized(merged, poses):
    """<bag>_final is there, with 10's TF and poses."""
    return has_poses(final_name(merged), poses)


def finalize(merged, poses, dry):
    """merged + 10's bag (its /tf, /tf_static and pose topics) -> <merged>_final,
    which replaces merged (one copy on disk; the stages read <bag>_final when
    dataset.bag is gone). A bag finalized in place before has them already:
    it is only renamed."""
    from scoop import bag, merge
    final = final_name(merged)
    if not os.path.isdir(poses):
        if dry:
            print("  %s + %s (once 10_poses wrote it) -> %s" % (merged, poses, final))
            return
        raise SystemExit("finalize: no %s (run 10_poses)" % poses)
    if merged != final and has_poses(merged, poses) and not os.path.exists(final):
        print("  %s has 10's TF and poses already -> renamed %s" % (merged, final))
        if not dry:
            os.rename(merged, final)
        return
    tp = sorted(bag.open_bag(poses).topics())
    new = final + "_tf"
    print("  %s + %s (%s) -> %s, then %s" % (merged, poses, ", ".join(tp), new, final))
    if dry:
        return
    if os.path.exists(new):
        raise SystemExit("%s exists: remove it (a finalize that stopped half way)" % new)
    size = merge.bag_bytes(merged)
    free = shutil.disk_usage(os.path.dirname(merged)).free
    if free < size + merge.MIN_FREE_GB * 1e9:
        raise SystemExit("finalize needs %.0f GB free next to the merged bag, %.0f GB are"
                         % (size / 1e9 + merge.MIN_FREE_GB, free / 1e9))
    merge.merge_bags([merged, poses], new, source={t: 1 for t in tp})
    shutil.rmtree(merged)
    os.rename(new, final)
    rename_bag_file(final, os.path.basename(new))
    print("  %s: the dataset with the complete TF and the poses (%s is gone; the stages "
          "read %s for it)" % (final, os.path.basename(merged), os.path.basename(final)))


def rename_bag_file(bag_dir, old_stem):
    """A bag folder renamed after it was written still holds <old_stem>_0.mcap
    and a metadata.yaml naming it: rename the file to <folder>_0.mcap and
    repoint metadata.yaml, the convention every other bag here follows."""
    import yaml
    old = f"{old_stem}_0.mcap"
    new = f"{os.path.basename(bag_dir)}_0.mcap"
    mp = os.path.join(bag_dir, "metadata.yaml")
    if old == new or not os.path.exists(os.path.join(bag_dir, old)) or not os.path.exists(mp):
        return
    os.rename(os.path.join(bag_dir, old), os.path.join(bag_dir, new))
    m = yaml.safe_load(open(mp))
    info = m.get("rosbag2_bagfile_information", {})
    info["relative_file_paths"] = [new if x == old else x for x in info.get("relative_file_paths", [])]
    for f in info.get("files", []):
        if f.get("path") == old:
            f["path"] = new
    yaml.safe_dump(m, open(mp, "w"), sort_keys=False)


def run_step(name, cmds, dry):
    if callable(cmds):
        return cmds(dry)
    for c in cmds:
        redirect = None
        if ">" in c:
            i = c.index(">")
            c, redirect = c[:i], c[i + 1]
        print("  $ " + " ".join(c) + (" > " + redirect if redirect else ""))
        if dry:
            continue
        if redirect:
            with open(redirect, "w") as fh:
                r = subprocess.run(c, stdout=fh)
        else:
            r = subprocess.run(c)
        if r.returncode:
            raise SystemExit("%s failed (exit %d); fix it, then run again (done steps are "
                             "skipped)" % (name, r.returncode))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("raw_dir")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--from", dest="start", choices=ORDER)
    ap.add_argument("--until", choices=ORDER)
    ap.add_argument("--only", choices=ORDER)
    ap.add_argument("--config", default=None)
    ap.add_argument("--new-config", action="store_true")
    ap.add_argument("--robot", default=None)
    ap.add_argument("--reference", default=None)
    a = ap.parse_args()
    import pass_tf
    raw = os.path.abspath(os.path.expanduser(a.raw_dir))
    if a.new_config:
        new_config(raw, a.robot, a.reference)
        return
    cfg = a.config or pass_tf.find_config(raw)
    steps, P = plan_steps(raw, cfg)
    print("pass: %s\nconfig: %s" % (raw, cfg or "none in map_stages/ for this pass"))
    if P is not None:
        print("kind: %s" % ("run in %s's map" % P.reference if P.reference else "mapping pass"))

    if a.status:
        for name, applies, done, what, _ in steps:
            done = done() if callable(done) else done
            state = "-" if not applies else {True: "done", False: "todo", None: "runs"}[done]
            print("  %-13s %-5s %s" % (name, state, what))
        if not cfg:
            print("no pipeline config for this pass: the map stages need one "
                  "(--new-config writes it)")
        return

    names = [s[0] for s in steps]
    lo = names.index(a.only or a.start) if (a.only or a.start) else 0
    hi = names.index(a.only or a.until) if (a.only or a.until) else len(names) - 1
    forced = bool(a.only or a.start)
    for name, applies, done, what, cmds in steps[lo:hi + 1]:
        if not applies:
            continue
        if callable(done):
            done = done()
        if done and not forced:
            print("[skip] %s: %s" % (name, what))
            continue
        print("\n== %s ==  %s" % (name, what))
        run_step(name, cmds, a.dry)
    if P is None:
        print("\nno pipeline config for this pass: write it with\n  python3 scoop_pipeline/"
              "run_pass.py %s --new-config\ncheck it, and run again" % raw)


if __name__ == "__main__":
    main()
