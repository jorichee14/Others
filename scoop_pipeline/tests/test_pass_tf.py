#!/usr/bin/env python
"""Self-test for map_stages/pass_tf.py through processing/merge_session.py:
the complete TF tree of a pass from the pipeline's outputs.

The bags carry the tree a recording has: the ZED's tracking (map_zed ~~
odom_zed ~~ zed_camera_link, set 1 m / 20 deg off on purpose) above its
static camera chain, os_sensor hung below the ZED optical frame by
static_tf.yaml, the arducam's ChArUco pose under map_zed (infra_1), and the
RealSense's own tree from camera_link (mobile_2).

  run (survey_1, in mapping_A's map): map -> map_zed (mobile_1's origin)
      from boards_<tag>.json, map_zed ~~ mobile_1 from 08's in-camera
      trajectory, the boards of the reference and of the run, arducam from 04,
      map -> map_realsense (mobile_2's origin) -> camera_link from 04;
  mapping pass (mapping_A): mobile_1 from the refined LiDAR trajectory
      through T_N_world, in the points' frame;
  09 on a bag merged before the map stages: the same tree in its bag, with
      the calibration the bag lacks (infra1_link) from static_tf.yaml.

Every lookup in the merged bag must give the pipeline's poses, and every
frame one parent.

    python scoop_pipeline/tests/test_pass_tf.py
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "map_stages"))

from scoop import rosmsg, tftree                                    # noqa: E402
from scoop.bagwrite import BagWriter                                # noqa: E402
from pipeline_common import dump_cameras_yaml, R_to_q               # noqa: E402
from pipeline_boards import read_tf_and_info                        # noqa: E402
import pass_tf                                                      # noqa: E402

T0 = 1_790_000_000.0
S = 1_000_000_000


def rotz(deg, t=(0, 0, 0), axis=(0, 0, 1)):
    from scipy.spatial.transform import Rotation
    T = np.eye(4)
    T[:3, :3] = Rotation.from_rotvec(np.radians(deg) * np.asarray(axis, float)).as_matrix()
    T[:3, 3] = t
    return T


def tq(T):
    return T[:3, 3].tolist(), R_to_q(T[:3, :3]).tolist()


def rec(T, frame):
    t, q = tq(T)
    return {"frame": frame, "xyz": t, "qxyzw": q}


def true_cam(t):
    """mobile_1's ZED optical frame in map at t s."""
    return rotz(30 * t / 10, (0.5 + 0.4 * t, 0.2 * t, 0.3)) @ rotz(-90, axis=(1, 0, 0))


CHAIN = [("zed_camera_link", "zed_camera_center", rotz(0, (0, 0, 0.015))),
         ("zed_camera_center", "zed_left_camera_frame", rotz(0, (0, 0.06, 0))),
         ("zed_left_camera_frame", "zed_left_camera_optical_frame",
          rotz(-90) @ rotz(-90, axis=(1, 0, 0)))]
RS_CHAIN = [("camera_link", "camera_color_frame", rotz(0, (0, 0.015, 0))),
            ("camera_color_frame", "camera_color_optical_frame",
             rotz(-90) @ rotz(-90, axis=(1, 0, 0)))]
T_MAP_MAPZED = rotz(12.0, (0.6, -0.1, 0.2))
T_ARDU = rotz(150, (-2.8, 15.0, 1.7)) @ rotz(-100, axis=(1, 0, 0))
T_RS = rotz(-40, (-24.4, 1.4, 0.3)) @ rotz(-95, axis=(1, 0, 0))
T_N_WORLD = rotz(137, (6.3, -2.1, 0.15))
T_RADAR = rotz(30, (0.017773, -0.208023, -0.155788), axis=(1, 1, 0))   # arducam -> infra1_link


def tf_static(w, items, t_ns):
    w.write("/tf_static", rosmsg.tf_message([(p, c, *tq(T)) for p, c, T in items], t_ns), t_ns)


def write_bags(raw):
    """mobile_1 (ZED tracking + camera chain + points), infra_1 (arducam under
    map_zed), mobile_2 (RealSense tree)."""
    stamps = np.arange(0.0, 10.0, 0.1)
    m1 = os.path.join(raw, "mobile_1", "m1")
    os.makedirs(os.path.dirname(m1))
    with BagWriter(m1) as w:
        tf_static(w, CHAIN, int(T0 * S))
        pts = np.zeros(4, dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4")])
        wrong = rotz(20, (1.0, 0, 0))                           # the ZED's tracking, off
        for t in stamps:
            ns = int((T0 + t) * S)
            T_mz_link = np.linalg.inv(T_MAP_MAPZED) @ wrong @ true_cam(t) @ np.linalg.inv(
                CHAIN[0][2] @ CHAIN[1][2] @ CHAIN[2][2])
            w.write("/tf", rosmsg.tf_message([("map_zed", "odom_zed", [0, 0, 0], [0, 0, 0, 1]),
                                              ("odom_zed", "zed_camera_link", *tq(T_mz_link))],
                                             ns), ns)
            w.write("/mobile_1/ouster/points", rosmsg.pointcloud2(pts, ns, "os_sensor"), ns)
    i1 = os.path.join(raw, "infra_1", "i1")
    os.makedirs(os.path.dirname(i1))
    with BagWriter(i1) as w:
        # infra_1 starts recording a minute before mobile_1's ZED publishes /tf
        w.write("/infra_1/diagnostics", rosmsg.string("up"), int((T0 - 60) * S))
        tf_static(w, [("map_zed", "arducam_optical_frame", rotz(5, (9, 9, 9)))], int(T0 * S))
    m2 = os.path.join(raw, "mobile_2", "m2")
    os.makedirs(os.path.dirname(m2))
    with BagWriter(m2) as w:
        tf_static(w, RS_CHAIN, int(T0 * S))
    return [m1, i1, m2], stamps


def outputs(data):
    """The pipeline outputs of mapping_A (reference) and survey_1 (run)."""
    ref = os.path.join(data, "processed", "20260101", "mapping_A")
    run = os.path.join(data, "processed", "20260101", "survey_1")
    for d in (ref, run):
        os.makedirs(os.path.join(d, "frames"))
    T_cam_sensor = rotz(90, (0.07, 0, -0.09)) @ rotz(90, axis=(0, 1, 0))
    json.dump({"T_N_world": T_N_WORLD.tolist(), "T_map_mapzed": rotz(77).tolist(),
               "boards": {"anchor": rec(np.eye(4), "board"),
                          "rs_anchor": rec(rotz(10, (-24.4, 2.2, 0.1)), "board_rs")}},
              open(os.path.join(ref, "frames", "anchor_frame.json"), "w"))
    json.dump({"T_map_mapzed": T_MAP_MAPZED.tolist(),
               "boards": {"anchor": rec(rotz(0.2, (0.004, 0, 0)), "board"),
                          "anchor_b": rec(rotz(45, (-7.2, 12.6, 0.03)), "board_b")}},
              open(os.path.join(run, "frames", "boards_survey_1_20260101.json"), "w"))
    od = os.path.join(run, "odometry", "reference_survey_1_20260101")
    os.makedirs(od)
    with open(os.path.join(od, "traj_mobile_1_lidar_in_cam.tum"), "w") as f:
        for t in np.arange(0.0, 10.0, 0.1):
            p, q = tq(true_cam(t))
            f.write("%.9f %s %s\n" % (T0 + t, " ".join(map(str, p)), " ".join(map(str, q))))
    cams = []
    for name, child, T in (("arducam", "arducam_optical_frame", T_ARDU),
                           ("realsense", "camera_color_optical_frame", T_RS)):
        p, q = tq(T)
        cams.append({"name": name, "child_frame": child, "source": "x", "stamp": T0,
                     "translation": p, "quaternion_xyzw": q})
    dump_cameras_yaml(os.path.join(run, "frames", "cameras_in_map.yaml"), "map",
                      [0] * 7, cams)
    # mapping_A's refined trajectory: os_sensor in GLIM's world
    om = os.path.join(ref, "odometry", "mobile_1")
    os.makedirs(om)
    with open(os.path.join(om, "traj_lidar_refined.txt"), "w") as f:
        for t in np.arange(0.0, 10.0, 0.1):
            p, q = tq(np.linalg.inv(T_N_WORLD) @ true_cam(t) @ T_cam_sensor)
            f.write("%.9f %s %s\n" % (T0 + t, " ".join(map(str, p)), " ".join(map(str, q))))
    return T_cam_sensor


def configs(stages, data):
    base = {"03_anchor": {}, "04_build_cameras": {
        "output": "cameras_in_map.yaml",
        "cameras": [{"name": "realsense", "origin_frame": "map_realsense"}]},
            "08_reference": {"name": "mobile_1_lidar"},
            "09_publish": {"robots": [{"name": "mobile_1",
                                       "traj": "traj_mobile_1_lidar_in_cam.tum",
                                       "optical_frame": "zed_left_camera_optical_frame"}]}}
    work = os.path.join(data, "work", "20260101")
    json.dump(dict(base, dataset={"bag": os.path.join(work, "mapping_A", "x_merged"),
                                  "traj": "traj_lidar_refined.txt", "machine": "mobile_1"}),
              open(os.path.join(stages, "pipeline_config.json"), "w"))
    json.dump({"extends": "pipeline_config.json",
               "dataset": {"bag": os.path.join(work, "survey_1", "x_merged"), "traj": None,
                           "reference_pass": "mapping_A"}},
              open(os.path.join(stages, "pipeline_config_survey_1.json"), "w"))


def merged_tree(inputs, cfg, out, static_yaml):
    from scoop import merge
    import yaml
    edges = tftree.tf_edges(inputs)
    st, _ = tftree.attach(tftree.load_calibrations(yaml.safe_load(open(static_yaml))), edges)
    for p, c, t, q in st:
        edges[c] = tftree.Edge(p, c, True, tftree.matrix(t, q))
    drop, more, extra, after = pass_tf.plan(pass_tf.load_pipeline(cfg), edges, inputs,
                                            log=lambda *_: None)
    merge.merge_bags(inputs, out, static_tf=st + more, extra_tf=extra, log=lambda *_: None,
                     drop_tf=tftree.edge_matcher(["%s->%s" % d for d in drop]) if drop else None)
    tftree.tf_edges([out])                                   # one parent per frame, or raises
    return read_tf_and_info(out, [], want_tf=True, verbose=False)[1], after


def close(A, B, what, f, mm=0.5, deg=0.02):
    if A is None:
        f.append("%s: no lookup" % what)
        return
    d = np.linalg.norm(A[:3, 3] - B[:3, 3]) * 1000
    a = np.degrees(np.arccos(np.clip((np.trace(A[:3, :3].T @ B[:3, :3]) - 1) / 2, -1, 1)))
    if d > mm or a > deg:
        f.append("%s: off by %.2f mm / %.3f deg" % (what, d, a))


def main():
    tmp = tempfile.mkdtemp()
    f = []
    try:
        data = os.path.join(tmp, "data")
        stages = os.path.join(tmp, "stages")
        os.makedirs(stages)
        static_yaml = os.path.join(tmp, "static_tf.yaml")
        T_cam_sensor = outputs(data)
        open(static_yaml, "w").write(
            "transforms:\n  - {parent: zed_left_camera_optical_frame, child: os_sensor, "
            "translation: %s, rotation_xyzw: %s}\n" % tuple(map(json.dumps, tq(T_cam_sensor)))
            + "  - {parent: arducam_optical_frame, child: infra1_link, translation: %s, "
            "rotation_xyzw: %s}\n" % tuple(map(json.dumps, tq(T_RADAR))))
        configs(stages, data)
        raw_run = os.path.join(data, "raw", "20260101", "survey_1")
        raw_map = os.path.join(data, "raw", "20260101", "mapping_A")
        if os.path.basename(pass_tf.find_config(raw_run, stages) or "") != \
                "pipeline_config_survey_1.json":
            f.append("find_config survey_1: %s" % pass_tf.find_config(raw_run, stages))
        if os.path.basename(pass_tf.find_config(raw_map, stages) or "") != "pipeline_config.json":
            f.append("find_config mapping_A: %s" % pass_tf.find_config(raw_map, stages))
        if pass_tf.find_config(os.path.join(data, "raw", "20260101", "coop_1"), stages):
            f.append("find_config found a config for a pass without one")
        inputs, stamps = write_bags(raw_run)

        tree, after = merged_tree(inputs, os.path.join(stages, "pipeline_config_survey_1.json"),
                                  os.path.join(tmp, "run_merged"), static_yaml)
        print(tftree.format_tree(after))
        for t in stamps[::7]:
            close(tree.lookup("map", "zed_left_camera_optical_frame", T0 + t), true_cam(t),
                  "run: mobile_1 at %.1f s" % t, f)
            close(tree.lookup("map", "os_sensor", T0 + t), true_cam(t) @ T_cam_sensor,
                  "run: os_sensor at %.1f s" % t, f)
        t = T0 + 5
        close(tree.lookup("map", "map_zed", t), T_MAP_MAPZED, "run: map_zed", f)
        close(tree.lookup("map", "arducam_optical_frame", t), T_ARDU, "run: arducam", f)
        close(tree.lookup("map", "infra1_link", t), T_ARDU @ T_RADAR, "run: infra1_link", f)
        close(tree.lookup("map", "camera_color_optical_frame", t), T_RS, "run: realsense", f)
        close(tree.lookup("map", "board", t), np.eye(4), "run: board (reference's)", f)
        close(tree.lookup("map", "board_b", t), rotz(45, (-7.2, 12.6, 0.03)), "run: board_b", f)
        close(tree.lookup("map", "board_rs", t), rotz(10, (-24.4, 2.2, 0.1)), "run: board_rs", f)
        if tree.lookup("map_zed", "odom_zed", t) is None:
            f.append("run: map_zed ~~ odom_zed (the ZED's own topics) is gone")
        if after["zed_camera_link"].parent != "map_zed":
            f.append("run: zed_camera_link hangs from %s" % after["zed_camera_link"].parent)
        if after["camera_link"].parent != "map_realsense" or \
                after["map_realsense"].parent != "map":
            f.append("run: camera_link hangs from %s" % after["camera_link"].parent)
        close(tree.lookup("map", "map_realsense", t),
              T_RS @ np.linalg.inv(RS_CHAIN[0][2] @ RS_CHAIN[1][2]), "run: map_realsense", f)

        tree, after = merged_tree(inputs, os.path.join(stages, "pipeline_config.json"),
                                  os.path.join(tmp, "map_merged"), static_yaml)
        for t in stamps[::7]:
            close(tree.lookup("map", "os_sensor", T0 + t), true_cam(t) @ T_cam_sensor,
                  "mapping: os_sensor at %.1f s" % t, f)
            close(tree.lookup("map", "zed_left_camera_optical_frame", T0 + t), true_cam(t),
                  "mapping: ZED at %.1f s" % t, f)
        close(tree.lookup("map", "map_zed", T0 + 5), rotz(77), "mapping: map_zed", f)

        # 09 on a bag merged before the map stages (only os_sensor in from static_tf.yaml)
        from scoop import merge, bag as sbag
        import yaml
        cfg = os.path.join(stages, "pipeline_config_survey_1.json")
        c = json.load(open(cfg))
        c["09_publish"] = {"static_tf": static_yaml}
        json.dump(c, open(cfg, "w"))
        old = os.path.join(data, "work", "20260101", "survey_1", "x_merged")
        cal = tftree.load_calibrations(yaml.safe_load(open(static_yaml)))[:1]
        merge.merge_bags(inputs, old, static_tf=tftree.attach(cal, tftree.tf_edges(inputs))[0],
                         log=lambda *_: None)
        r = subprocess.run([sys.executable, os.path.join(ROOT, "map_stages", "09_publish_poses.py"),
                            cfg], capture_output=True, text=True)
        print(r.stdout[-1500:])
        if r.returncode:
            f.append("09: " + r.stdout[-1000:] + r.stderr[-1500:])
        else:
            b09 = os.path.join(data, "processed", "20260101", "survey_1", "bags",
                               "survey_1_20260101_best_poses")
            tp = sbag.open_bag(b09).topics()
            if sorted(tp) != ["/mobile_1/global_pose", "/mobile_1/local_pose", "/tf", "/tf_static"]:
                f.append("09 topics: %s" % sorted(tp))
            tftree.tf_edges([b09])
            tree = read_tf_and_info(b09, [], want_tf=True, verbose=False)[1]
            for t in stamps[::9]:
                close(tree.lookup("map", "os_sensor", T0 + t), true_cam(t) @ T_cam_sensor,
                      "09: os_sensor at %.1f s" % t, f)
            close(tree.lookup("map", "infra1_link", T0 + 5), T_ARDU @ T_RADAR, "09: infra1_link", f)
            close(tree.lookup("map", "camera_color_optical_frame", T0 + 5), T_RS, "09: realsense", f)
            close(tree.lookup("map", "board_b", T0 + 5), rotz(45, (-7.2, 12.6, 0.03)), "09: board_b", f)
            if tree.lookup("map_zed", "odom_zed", T0 + 5) is None:
                f.append("09: the bag's map_zed ~~ odom_zed is missing")
            ts_ = rosmsg.typestore()
            par = set()
            for _, _, pl, dec in sbag.open_bag(b09).iter_raw(["/tf"]):
                par |= {tr.header.frame_id for tr in dec(pl).transforms
                        if tr.child_frame_id == "zed_camera_link"}
            if par != {"map_zed"}:
                f.append("09: zed_camera_link's parents on /tf: %s" % sorted(par))

        # the CLI: the check prints the tree from the outputs it finds
        cli = subprocess.run([sys.executable, os.path.join(ROOT, "map_stages", "pass_tf.py"),
                              os.path.join(stages, "pipeline_config_survey_1.json")] + inputs,
                             capture_output=True, text=True)
        if cli.returncode or "map" not in cli.stdout:
            f.append("pass_tf.py CLI: " + cli.stdout[-800:] + cli.stderr[-800:])
    except Exception as e:                                        # noqa: BLE001
        import traceback
        f.append(traceback.format_exc()[-2500:])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("\n".join(f) or "pass tf ok")
    sys.exit(1 if f else 0)


if __name__ == "__main__":
    main()
