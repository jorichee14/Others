#!/usr/bin/env python
"""Self-tests for a pass mapped by mobile_2 alone (map_stages/
pipeline_config_mapping_20261015.json): the config's mobile_2 settings, GLIM's
trajectory found next to a robot's own bag, and stage 01's undistortion of a
raw (not rectified) camera image. Needs numpy and opencv.

    python tests/test_mobile2_mapping.py
"""
import importlib.util
import os
import shutil
import sys
import tempfile
import traceback

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STAGES = os.path.join(ROOT, "map_stages")
sys.path.insert(0, ROOT)
sys.path.insert(0, STAGES)

import pipeline_common as pc                                           # noqa: E402

FAILED = []
CFG = os.path.join(STAGES, "pipeline_config_mapping_20261015.json")


def test_config_is_mobile_2_s_own_map():
    c = pc.load_config(CFG)
    d = c["dataset"]
    assert d["machine"] == "mobile_2" and not d.get("reference_pass")
    assert d["bag"].endswith("work/20261015/mapping/mobile_2/"
                             "mirc_dataset_mapping_2_run_20261015_mobile_2_retimed")
    assert (d["points_topic"], d["image_topic"], d["camera_info_topic"]) == \
        ("/mobile_2/ouster/points", "/mobile_2/color/image_raw", "/mobile_2/color/camera_info")
    assert (d["camera_frame"], d["camera_map_frame"], d["rectified"]) == \
        ("camera_color_optical_frame", "map_realsense", False)
    assert c["01_build_map"]["colorize"]["enable"]                     # inherited
    assert c["07_build_cameras"]["enabled"] is False and c["08_emit_tfs"]["enabled"] is False
    # every stage reads the robot's sensors from dataset alone
    P = pc.Pipeline(c, STAGES)
    assert (P.points_topic, P.camera_frame, P.camera_map_frame, P.rectified) == \
        ("/mobile_2/ouster/points", "camera_color_optical_frame", "map_realsense", False)
    assert P.lidar_track_traj().endswith("traj_mobile_2_lidar.tum")
    robots = {r["name"]: r for r in P.robots()}
    assert list(robots) == ["mobile_2"]                            # mobile_1 is not in this pass
    assert robots["mobile_2"]["traj"] == "traj_mobile_2_lidar_in_cam.tum"
    assert robots["mobile_2"]["optical_frame"] == "camera_color_optical_frame"
    assert robots["mobile_2"]["local_frame"] == "mobile_2_base_link"
    # the pass it belongs to: raw/20261015/mapping
    import pass_tf
    raw = os.path.expanduser("~/workspaces/isaac_ros-dev/data/raw/20261015/mapping")
    assert pass_tf.find_config(raw) == CFG, pass_tf.find_config(raw)


def test_sensors_only_in_dataset(d):
    """A robot's topic or camera in a stage block is refused, named."""
    import json
    cfg = os.path.join(d, "pipeline_config_x.json")
    json.dump({"extends": os.path.join(STAGES, "pipeline_config.json"),
               "05_anchor": {"cam_frame": "camera_color_optical_frame"}}, open(cfg, "w"))
    try:
        pc.load_config(cfg)
    except SystemExit as e:
        assert "05_anchor.cam_frame -> dataset.camera_frame" in str(e), e
    else:
        raise AssertionError("05_anchor.cam_frame was accepted")
    for c in ("pipeline_config.json", "pipeline_config_survey_1.json",
              "pipeline_config_coop_2.json", "pipeline_config_mapping_20260924.json"):
        P = pc.Pipeline(pc.load_config(os.path.join(STAGES, c)), STAGES)
        assert (P.points_topic, P.camera_frame, P.camera_map_frame, P.rectified) == \
            ("/mobile_1/ouster/points", "zed_left_camera_optical_frame", "map_zed", True), c
        own = P.robots()[0]
        assert (own["name"], own["traj"], own["optical_frame"]) == \
            ("mobile_1", "traj_mobile_1_lidar_in_cam.tum", "zed_left_camera_optical_frame"), c


def test_glim_traj_next_to_a_robot_s_own_bag(d):
    data = os.path.join(d, "data")
    for bag_dir, want in (
            (os.path.join(data, "work", "20261015", "mapping", "mobile_2", "x_retimed"),
             os.path.join(data, "work", "20261015", "mapping", "mobile_2", "glim",
                          "traj_lidar.txt")),
            (os.path.join(data, "work", "20260924", "mapping_A", "y_merged"),
             os.path.join(data, "work", "20260924", "mapping_A", "mobile_2", "glim",
                          "traj_lidar.txt"))):
        P = pc.Pipeline({"dataset": {"bag": bag_dir, "machine": "mobile_2", "traj": None}}, d)
        assert P.glim_traj() == want, (P.glim_traj(), want)


def test_raw_image_undistorted_onto_the_projection_s_pinhole():
    import cv2
    spec = importlib.util.spec_from_file_location("build_map", os.path.join(STAGES, "01_build_map.py"))
    try:
        bm = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(bm)
    except ImportError as e:                                  # no open3d here
        print(f"        (skipped: {e})")
        return

    class S:
        fx, fy, cx, cy = 427.703, 427.108, 423.132, 239.735
        dist = np.array([-0.0568, 0.0647, -0.00047, 0.00059, -0.0207]) * 4   # exaggerated
    W, H = 848, 480
    K = np.array([[S.fx, 0, S.cx], [0, S.fy, S.cy], [0, 0, 1]])
    X = np.array([[1.2, -0.7, 2.0]])                          # a point near a corner
    u0 = S.fx * X[0, 0] / X[0, 2] + S.cx                     # where 01 projects it
    v0 = S.fy * X[0, 1] / X[0, 2] + S.cy
    ud, _ = cv2.projectPoints(X, np.zeros(3), np.zeros(3), K, S.dist)   # where the raw image has it
    ud = ud.ravel()
    img = np.zeros((H, W, 3), np.uint8)
    cv2.circle(img, (int(round(ud[0])), int(round(ud[1]))), 3, (255, 255, 255), -1)
    assert abs(ud[0] - u0) > 5 or abs(ud[1] - v0) > 5         # the distortion matters here
    out = bm.undistorter(S, W, H, rectified=False)(img)
    ys, xs = np.nonzero(out[:, :, 0] > 100)
    assert abs(xs.mean() - u0) < 1.5 and abs(ys.mean() - v0) < 1.5, (xs.mean(), ys.mean(), u0, v0)
    same = bm.undistorter(S, W, H, rectified=True)(img)
    assert same is img                                        # rectified: untouched


def check(name, fn, *args):
    try:
        fn(*args)
        print("  ok    %s" % name)
    except Exception:
        FAILED.append(name)
        print("  FAIL  %s" % name)
        traceback.print_exc()


def main():
    d = tempfile.mkdtemp(prefix="mobile2_map_")
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    try:
        for name, fn in tests:
            args = []
            if fn.__code__.co_argcount == 1:
                args = [os.path.join(d, name)]
                os.makedirs(args[0])
            check(name, fn, *args)
    finally:
        shutil.rmtree(d, ignore_errors=True)
    print(f"\n{len(tests) - len(FAILED)}/{len(tests)} passed")
    sys.exit(1 if FAILED else 0)


if __name__ == "__main__":
    main()
