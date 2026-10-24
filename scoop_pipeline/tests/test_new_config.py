#!/usr/bin/env python
"""Self-test for run_pass.py --new-config: the config of a new pass, from its
raw folder (tiny bags that only carry the topics the choice is made from).

  mapping, mobile_1 with its ZED   robot mobile_1, the merged bag (the ZED's
                                   images join the LiDAR there), 20261015's settings
  mapping, mobile_2 alone          robot mobile_2, its own retimed bag; its
                                   sensors from the base config's robots block
  survey at a site                 mobile_1 drives, in that day's mapping pass's
                                   map (found), the parked mobile_2 by its LiDAR
  a config there already, a coop   refused

    python tests/test_new_config.py
"""
import json
import os
import shutil
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STAGES = os.path.join(ROOT, "map_stages")
sys.path[:0] = [ROOT, STAGES]

import run_pass                                                         # noqa: E402
import pipeline_common as pc                                            # noqa: E402
from scoop import rosmsg                                                # noqa: E402
from scoop.bagwrite import BagWriter                                    # noqa: E402

S = 10 ** 9


def bag(folder, name, topics, t0=1_790_000_000 * S, span=60 * S):
    w = BagWriter(os.path.join(folder, name))
    msg = rosmsg.msg_type("std_msgs/msg/String")(data="x")
    for i, t in enumerate(topics):
        w.write(t, msg, t0 + i)
        w.write(t, msg, t0 + span + i)
    w.close()


def robot(raw, machine, ns=None, svo=False, extra=()):
    d = os.path.join(raw, machine)
    os.makedirs(d)
    tag = "_".join(os.path.relpath(raw, os.path.dirname(os.path.dirname(raw))).split(os.sep))
    topics = ["%s/lidar_packets" % ns, "%s/metadata" % ns] if ns else []
    bag(d, "mirc_dataset_%s_%s" % (tag, machine), topics + list(extra) or ["/x"])
    if svo:
        open(os.path.join(d, "mirc_dataset_%s_%s.svo2" % (tag, machine)), "w").close()


def main():
    tmp = tempfile.mkdtemp(prefix="newcfg_")
    raw = os.path.join(tmp, "data", "raw")
    stages = os.path.join(tmp, "stages")
    os.makedirs(stages)
    shutil.copy(os.path.join(STAGES, "pipeline_config.json"), stages)
    # the base config names a pass of the day too (as map_stages/pipeline_config.json does): not a second map
    base = json.load(open(os.path.join(stages, "pipeline_config.json")))
    base.setdefault("dataset", {})["bag"] = "~/data/work/20261101/mapping/x_merged"
    json.dump(base, open(os.path.join(stages, "pipeline_config.json"), "w"), indent=1)
    fails = []

    def eq(got, want, what):
        if got != want:
            fails.append("%s: %r, want %r" % (what, got, want))

    def make(r, **kw):
        p = run_pass.new_config(r, log=lambda *_: None, stages=stages, **kw)
        return p, json.load(open(p)), pc.load_pipeline(p)

    try:
        # mapping by mobile_1, with its ZED
        r1 = os.path.join(raw, "20261101", "mapping")
        robot(r1, "mobile_1", "/ouster", svo=True)
        p, c, P = make(r1)
        eq(os.path.basename(p), "pipeline_config_mapping_20261101.json", "mapping 1 file")
        eq(c["dataset"]["machine"], "mobile_1", "mapping 1 robot")
        eq(c["dataset"]["bag"].endswith("work/20261101/mapping/mirc_dataset_mapping_20261101_merged"),
           True, "mapping 1 bag: " + c["dataset"]["bag"])
        eq((c["dataset"]["traj"], c["01_build_map"]["fusion"], c["02_bundle_adjust"]["output"]),
           ("traj_lidar_ba_glim.txt", "tsdf", "traj_lidar_ba_glim.txt"), "mapping 1 settings")
        eq((P.points_topic, P.camera_frame, P.rectified),
           ("/mobile_1/ouster/points", "zed_left_camera_optical_frame", True), "mapping 1 sensors")
        eq(pc.load_pipeline(p).tag, "mapping_20261101", "mapping 1 tag")

        # mapping by mobile_2 alone: its own bag
        r2 = os.path.join(raw, "20261102", "mapping")
        robot(r2, "mobile_2", "/mobile_2/ouster")
        p, c, P = make(r2)
        eq(c["dataset"]["machine"], "mobile_2", "mapping 2 robot")
        eq(c["dataset"]["bag"].endswith("work/20261102/mapping/mobile_2/"
                                        "mirc_dataset_20261102_mapping_mobile_2_retimed"),
           True, "mapping 2 bag: " + c["dataset"]["bag"])
        eq((P.points_topic, P.image_topic, P.camera_frame, P.camera_map_frame, P.rectified),
           ("/mobile_2/ouster/points", "/mobile_2/color/image_raw", "camera_color_optical_frame",
            "map_realsense", False), "mapping 2 sensors")

        # survey at a site on the day of mapping 1
        r3 = os.path.join(raw, "20261101", "survey", "site_2")
        robot(r3, "mobile_1", "/ouster", svo=True)
        robot(r3, "mobile_2", "/mobile_2/ouster")
        robot(r3, "infra_1", extra=["/infra_1/wifi/status"])
        p, c, P = make(r3)
        eq(os.path.basename(p), "pipeline_config_survey_site_2_20261101.json", "survey file")
        eq((c["dataset"]["machine"], c["dataset"]["reference_pass"], c["dataset"]["traj"]),
           ("mobile_1", "mapping", None), "survey dataset")
        eq(c["dataset"]["bag"].endswith("work/20261101/survey/site_2/"
                                        "mirc_dataset_survey_site_2_20261101_merged"),
           True, "survey bag: " + c["dataset"]["bag"])
        cam = c["07_build_cameras"]["cameras"][0]
        eq((cam["source"], cam["board"], cam["points_topic"]),
           ("lidar", "board_4", "/mobile_2/ouster/points"), "survey parked mobile_2")
        eq(c["05_anchor"]["boards_used"], ["board_1", "board_2", "board_3", "board_4"], "boards")
        eq("boards_moved" in c["dataset"], False, "boards_moved is not assumed")
        eq((c["03_init"].get("relocalize") or {}).get("enabled"), None,
           "no relocalize without boards_moved")
        # the boards put up again for this pass: what it measures wins
        os.remove(p)
        p, c, P = make(r3, boards_moved=True)
        eq(c["dataset"].get("boards_moved"), True, "boards_moved")
        eq(P.boards_moved, True, "boards_moved read back")
        # a moved board cannot place the run: 03 relocalizes on the map instead
        eq(c["03_init"]["relocalize"]["enabled"], True, "boards_moved -> relocalize")
        eq(P.ref_file("anchor").endswith("processed/20261101/mapping/frames/anchor_frame.json"),
           True, "survey reference")

        # refused: written already; a coop run
        for r, what in ((r1, "exists"), ):
            try:
                run_pass.new_config(r, log=lambda *_: None, stages=stages)
                fails.append("%s: not refused" % what)
            except SystemExit as e:
                eq("exists already" in str(e), True, what)
        # a coop run on the same day: in mapping 1's map, the boards and the infra camera
        # from the survey at site 2, mobile_2 by VSLAM and its depth track
        r4 = os.path.join(raw, "20261101", "coop_1")
        robot(r4, "mobile_1", "/ouster", svo=True)
        robot(r4, "mobile_2", "/mobile_2/ouster")
        robot(r4, "infra_1", extra=["/infra_1/wifi/status"])
        sn = os.path.join(r4, "mobile_1", "sniffer")                 # a sniffer bag from the day before
        os.makedirs(sn)
        bag(sn, "mirc_dataset_coop_1_20261101_mobile_1_sniffer", ["/mobile_1/sniffer/infra_1/csi"],
            t0=1_790_000_000 * S - 86400 * S)
        lines = []
        p = run_pass.new_config(r4, log=lines.append, stages=stages)
        c, P = json.load(open(p)), pc.load_pipeline(p)
        eq(os.path.basename(p), "pipeline_config_coop_1_20261101.json", "coop file")
        eq(any("another run" in ln and "sniffer" in ln for ln in lines), True,
           "the sniffer bag from another run not flagged: %s" % [ln for ln in lines if "overlap" in ln])
        eq(sum("another run" in ln for ln in lines), 1, "only the sniffer bag flagged")
        eq((c["dataset"]["machine"], c["dataset"]["reference_pass"], c["dataset"]["boards_from"],
            c["dataset"]["cameras_from"]), ("mobile_1", "mapping", ["survey/site_2"], ["survey/site_2"]),
           "coop dataset")
        eq(c["dataset"]["bag"].endswith("work/20261101/coop_1/mirc_dataset_coop_1_20261101_merged"),
           True, "coop bag: " + c["dataset"]["bag"])
        tr = c["04_reference"]["tracks"][0]                         # mobile_2 has an Ouster: its LiDAR track
        eq((tr["name"], tr["kind"], tr["machine"], tr["body_frame"], "vslam" in c),
           ("mobile_2_lidar", "lidar", "mobile_2", "os_lidar_mobile2", False), "coop track")
        # neither robot's start board is named: 03 looks for every board in the map
        eq("board" in tr, False, "coop track names no board")
        eq("board" in c["03_init"]["sensors"][1], False, "coop names no start board")
        eq("boards_moved" in c["dataset"], False, "coop boards_moved is not assumed")
        eq((c["03_init"].get("relocalize") or {}).get("enabled"), None,
           "no relocalize without boards_moved")
        eq((c["05_anchor"]["enabled"], c["07_build_cameras"]["enabled"], c["08_emit_tfs"]["enabled"]),
           (False, False, False), "coop stages off")
        eq(c["10_publish"]["robots"][0]["traj"], "traj_mobile_2_lidar_in_cam.tum", "coop 10_publish")

        # the boards were put up again for this site: the survey's poses replace
        # the reference's, and the run is placed on the map, not on a board
        os.remove(p)
        p = run_pass.new_config(r4, log=lines.append, stages=stages, boards_moved=True)
        c = json.load(open(p))
        eq(c["dataset"].get("boards_moved"), True, "coop boards_moved")
        eq(c["03_init"]["relocalize"]["enabled"], True, "coop boards_moved -> relocalize")
        eq(c["dataset"]["boards_from"], ["survey/site_2"], "coop boards_from still the survey's")

        # a coop run takes its OWN site's survey: the boards stand somewhere
        # else at each site, so another site's poses would be tens of m out
        eq(run_pass.same_site(["survey/site_1", "survey/site_2"], ("coop", "site_2")),
           ["survey/site_2"], "same site's survey")
        eq(run_pass.same_site(["survey/site_1", "survey/site_2"], ("coop_1",)),
           ["survey/site_1", "survey/site_2"], "no site in the pass: every survey")
        eq(run_pass.same_site(["survey/site_1"], ("coop", "site_3")),
           ["survey/site_1"], "no survey of this site: every survey")

        # a stray bag from another run, disjoint and LONGER than any of the
        # run's own: the odd one is it, not the three that overlap each other
        runbags = [("sniffer_from_another_run", 0.0, 304.0),
                   ("mobile_1", 2400.0, 2647.0), ("mobile_2", 2409.0, 2640.0),
                   ("infra_1", 2416.0, 2642.0)]
        eq(run_pass.check_spans(runbags, log=lambda _l: None),
           ["sniffer_from_another_run"], "the stray bag, not the three that overlap")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    for f in fails:
        print("  FAIL  " + f)
    print("new config " + ("FAILED" if fails else "ok"))
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
