#!/usr/bin/env python
"""Self-test for make_config's rtabmap_map and rtabmap_loc, no ROS: a
synthetic mobile_2 bag (the D455's left infrared image, its depth, camera
info, /tf_static from mobile_2_base_link down to the cameras, and
/mobile_2/global_pose).

rtabmap_map must map the colour image + the depth reprojected into it (or,
with --rtab-image infra, the left infrared image + its depth) on the pose topic,
relayed as TF twin_map -> mobile_2_base_link (RTAB-Map's odom frame), with
the database at /work/rtabmap.db, /tf_static and the pose topic played, a
long stop for the database, and refuse a bag without the pose topic.
rtabmap_loc must refuse without a database, copy it into the run, run
rgbd_odometry on the infrared image + depth next to RTAB-Map (on the colour
image) in localization mode (no incremental memory,
every node in working memory) and answer on /rtabmap/localization_pose, in
map, of mobile_2_base_link.

    python scoop_pipeline/tests/test_rtabmap_map.py
"""
import ast
import json
import os
import shutil
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "slam", "odometry"))

import test_odometry as TO                                          # noqa: E402
import make_config as MK                                            # noqa: E402
from scoop import rosmsg                                            # noqa: E402
from scoop.bagwrite import BagWriter                                # noqa: E402

S = TO.S


def write_bag(path, with_pose=True):
    T0 = TO.T0
    tfs = [("mobile_2_base_link", "camera_link", [0.1, 0.0, 0.3], [0, 0, 0, 1]),
           ("camera_link", "camera_color_optical_frame", [0.0, 0.015, 0.0], list(TO.Q_OPT))] + TO.CAM_TF
    recs = [(T0, "/tf_static", rosmsg.tf_message(tfs, T0))]
    for k in range(int(TO.DUR * 30)):
        t = T0 + int(k * S / 30)
        recs += [(t, "/mobile_2/infra1/image_rect_raw", TO.image(t, "camera_infra1_optical_frame")),
                 (t, "/mobile_2/infra1/camera_info", TO.info(t, "camera_infra1_optical_frame")),
                 (t, "/mobile_2/depth/image_rect_raw", TO.image(t, "camera_depth_optical_frame", "16UC1")),
                 (t, "/mobile_2/depth/camera_info", TO.info(t, "camera_depth_optical_frame"))]
        if k % 2 == 0:                                     # the colour camera at 15 Hz
            recs += [(t, "/mobile_2/color/image_raw", TO.image(t, "camera_color_optical_frame", "rgb8")),
                     (t, "/mobile_2/color/camera_info", TO.info(t, "camera_color_optical_frame", w=640, h=480))]
    if with_pose:
        for k in range(int(TO.DUR * 10)):
            t = T0 + k * S // 10
            recs.append((t, "/mobile_2/global_pose", rosmsg.pose_stamped(t, "map", [0.1 * k, 0, 0], [0, 0, 0, 1])))
    w = BagWriter(path)
    for t, topic, msg in sorted(recs, key=lambda r: r[0]):
        w.write(topic, msg, t)
    w.close()


def launch_vars(run):
    tree = ast.parse(open(os.path.join(run, "config", "rtabmap.launch.py")).read())
    return {n.targets[0].id: ast.literal_eval(n.value) for n in tree.body
            if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)}


def main():
    tmp = tempfile.mkdtemp()
    f = []
    try:
        bag = os.path.join(tmp, "bag")
        write_bag(bag)
        run = lambda name: os.path.join(tmp, "runs", name)             # noqa: E731
        r = MK.make("rtabmap_map", bag, "mobile_2", run("map"))
        v = launch_vars(run("map"))
        rt = v["RTABMAP"]
        if v["RELAY"] != ["/mobile_2/global_pose", "mobile_2_base_link", "twin_map"] \
                or rt["odom_frame_id"] != "twin_map" or rt["frame_id"] != "mobile_2_base_link" \
                or rt["database_path"] != "/work/rtabmap.db" or rt["Mem/IncrementalMemory"] != "true":
            f.append("rtabmap_map: relay %s, params %s" % (v["RELAY"], rt))
        # the colour image, the depth reprojected into it
        if dict(v["IMAGES"]) != {"rgb/image": "/mobile_2/color/image_raw", "depth/image": MK.REGISTERED,
                                 "rgb/camera_info": "/mobile_2/color/camera_info"} \
                or dict(v["REGISTER"]).get("depth_registered/image_rect") != MK.REGISTERED \
                or dict(v["REGISTER"]).get("rgb/camera_info") != "/mobile_2/color/camera_info" \
                or r["key_topic"] != "/mobile_2/color/image_raw" or abs(r["key_rate"]["hz"] - 15) > 0.5 \
                or "/mobile_2/depth/camera_info" not in r["play_topics"]:
            f.append("rtabmap_map colour: images %s, register %s, key %s %s, play %s"
                     % (v["IMAGES"], v["REGISTER"], r["key_topic"], r["key_rate"], r["play_topics"]))
        # --rtab-image infra: the left infrared image and its own depth, nothing reprojected
        MK.make("rtabmap_map", bag, "mobile_2", run("map_infra"), rtab_image="infra")
        vi = launch_vars(run("map_infra"))
        if dict(vi["IMAGES"]) != {"rgb/image": "/mobile_2/infra1/image_rect_raw",
                                  "depth/image": "/mobile_2/depth/image_rect_raw",
                                  "rgb/camera_info": "/mobile_2/infra1/camera_info"} or vi["REGISTER"]:
            f.append("rtabmap_map infra: images %s, register %s" % (vi["IMAGES"], vi["REGISTER"]))
        if "/tf_static" not in r["play_topics"] or "/mobile_2/global_pose" not in r["play_topics"] \
                or r["stop_s"] < 60 or r["world"] != "map":
            f.append("rtabmap_map: play %s, stop %s, world %s" % (r["play_topics"], r["stop_s"], r["world"]))
        if not os.path.exists(os.path.join(run("map"), "config", "gt_relay.py")):
            f.append("rtabmap_map: no gt_relay.py")
        compile(open(os.path.join(run("map"), "config", "gt_relay.py")).read(), "gt_relay.py", "exec")
        compile(open(os.path.join(run("map"), "config", "rtabmap.launch.py")).read(), "launch", "exec")

        nopose = os.path.join(tmp, "bag_nopose")
        write_bag(nopose, with_pose=False)
        try:
            MK.make("rtabmap_map", nopose, "mobile_2", run("map_nopose"))
            f.append("rtabmap_map without a pose topic: not refused")
        except (MK.Refused, MK.mc.ConvertError):
            pass
        try:
            MK.make("rtabmap_loc", bag, "mobile_2", run("loc_nodb"))
            f.append("rtabmap_loc without a database: not refused")
        except MK.Refused:
            pass

        db = os.path.join(tmp, "rtabmap.db")
        open(db, "wb").write(b"SQLite format 3\0fake")
        r = MK.make("rtabmap_loc", bag, "mobile_2", run("loc"), db=db)
        v = launch_vars(run("loc"))
        rt, od = v["RTABMAP"], v["ODOM"]
        if open(os.path.join(run("loc"), "rtabmap.db"), "rb").read() != open(db, "rb").read():
            f.append("rtabmap_loc: the database is not copied into the run")
        if rt["Mem/IncrementalMemory"] != "false" or rt["Mem/InitWMWithAllNodes"] != "true" \
                or rt["odom_frame_id"] != "odom_vis" or od["odom_frame_id"] != "odom_vis" or not od["publish_tf"] \
                or v["RELAY"] or v["ODOM"]["frame_id"] != "mobile_2_base_link" \
                or dict(v["ODOM_IMAGES"]).get("rgb/image") != "/mobile_2/infra1/image_rect_raw" \
                or dict(v["IMAGES"]).get("rgb/image") != "/mobile_2/color/image_raw":
            f.append("rtabmap_loc: rtabmap %s, odometry %s, relay %s" % (rt, od, v["RELAY"]))
        if (r["out_topic"], r["tum_frame"], r["world"]) != \
                ("/rtabmap/localization_pose", "mobile_2_base_link", "map"):
            f.append("rtabmap_loc: out %s %s %s" % (r["out_topic"], r["tum_frame"], r["world"]))
        rj = json.load(open(os.path.join(run("loc"), "run.json")))
        if rj["stop_s"] != 60:
            f.append("rtabmap_loc: run.json stop_s %s" % rj["stop_s"])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    if f:
        print("FAIL\n  " + "\n  ".join(f))
        sys.exit(1)
    print("rtabmap_map ok")


if __name__ == "__main__":
    main()
