#!/usr/bin/env python
"""Self-test for run_pass.py on the synthetic pass of tests/test_pass_tf.py.

  run       --status: the run's done steps (04, 07) and the ones to do (10,
            finalize); --only 10_poses then --only finalize: the merged bag
            ends up with the complete TF and the poses, and --status says so;
  mapping   --dry: 01 --seed, 02, 01, 05, ... in that order, no 03/04/09;
  mapping   10's track for a mapping pass: the refined trajectory in map, as
            the camera optical frame;
  configs   a block under its old stage number is read as the new one.

    python scoop_pipeline/tests/test_run_pass.py
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
sys.path.insert(0, HERE)
import test_pass_tf as T                                            # noqa: E402
from scoop import merge, tftree                                     # noqa: E402
from scoop import bag as sbag                                       # noqa: E402
from pipeline_common import load_config                             # noqa: E402
from pipeline_boards import read_tf_and_info                        # noqa: E402

RUNNER = os.path.join(ROOT, "run_pass.py")


def runner(*args):
    r = subprocess.run([sys.executable, RUNNER, *args], capture_output=True, text=True)
    return r.returncode, r.stdout + r.stderr


def main():
    tmp = tempfile.mkdtemp()
    f = []
    try:
        data = os.path.join(tmp, "data")
        stages = os.path.join(tmp, "stages")
        os.makedirs(stages)
        static_yaml = os.path.join(tmp, "static_tf.yaml")
        T_cam_sensor = T.outputs(data)
        open(static_yaml, "w").write(
            "transforms:\n  - {parent: zed_left_camera_optical_frame, child: os_sensor, "
            "translation: %s, rotation_xyzw: %s}\n" % tuple(map(json.dumps, T.tq(T_cam_sensor))))
        T.configs(stages, data)
        cfg = os.path.join(stages, "pipeline_config_survey_1.json")
        c = json.load(open(cfg))
        c["10_publish"] = {"static_tf": static_yaml}
        json.dump(c, open(cfg, "w"))
        raw = os.path.join(data, "raw", "20260101", "survey_1")
        inputs, stamps = T.write_bags(raw)
        merged = os.path.join(data, "work", "20260101", "survey_1", "x_merged")
        merge.merge_bags(inputs, merged, log=lambda *_: None, static_tf=tftree.attach(
            tftree.load_calibrations({"transforms": [{
                "parent": "zed_left_camera_optical_frame", "child": "os_sensor",
                "translation": T.tq(T_cam_sensor)[0],
                "rotation_xyzw": T.tq(T_cam_sensor)[1]}]}), tftree.tf_edges(inputs))[0])

        od = os.path.join(data, "processed", "20260101", "survey_1", "odometry",
                          "reference_survey_1_20260101")
        shutil.copy(os.path.join(od, "traj_mobile_1_lidar_in_cam.tum"),
                    os.path.join(od, "traj_mobile_1_lidar.tum"))        # 04's two outputs
        rc, out = runner(raw, "--config", cfg, "--status")
        print(out)
        want = {"04_reference": "done", "07_cameras": "done", "10_poses": "todo",
                "finalize": "todo", "01_seed": "-", "03_init": "todo"}
        for step, state in want.items():
            line = next((ln for ln in out.splitlines() if ln.strip().startswith(step + " ")), "")
            if line.split()[1:2] != [state]:
                f.append("run --status: %s is %r, not %s" % (step, line, state))
        rc, out = runner(raw, "--config", cfg, "--only", "10_poses")
        if rc:
            f.append("--only 10_poses: " + out[-1500:])
        rc, out = runner(raw, "--config", cfg, "--only", "finalize")
        print(out[-600:])
        if rc:
            f.append("--only finalize: " + out[-1500:])
        else:
            tp = sbag.open_bag(merged).topics()
            for t in ("/mobile_1/global_pose", "/mobile_1/local_pose", "/tf", "/tf_static"):
                if t not in tp:
                    f.append("finalize: %s missing from the merged bag" % t)
            if os.path.exists(merged + "_tf"):
                f.append("finalize left %s_tf behind" % merged)
            tree = read_tf_and_info(merged, [], want_tf=True, verbose=False)[1]
            for t in stamps[::9]:
                T.close(tree.lookup("map", "os_sensor", T.T0 + t), T.true_cam(t) @ T_cam_sensor,
                        "finalize: os_sensor at %.1f s" % t, f)
            rc, out = runner(raw, "--config", cfg, "--status")
            line = next((ln for ln in out.splitlines() if ln.strip().startswith("finalize")), "")
            if "done" not in line:
                f.append("--status after finalize: %r" % line)

        rc, out = runner(os.path.join(data, "raw", "20260101", "mapping_A"), "--config",
                         os.path.join(stages, "pipeline_config.json"), "--dry", "--from", "01_seed")
        print(out[-1500:])
        order = [ln.split("== ")[1].split(" ==")[0].strip() for ln in out.splitlines() if ln.startswith("== ")]
        if order[:4] != ["01_seed", "02_refine", "01_map", "05_anchor"] or \
                any(s in order for s in ("03_init", "04_reference", "09_cloud")):
            f.append("mapping --dry order: %s" % order)
        if "01_build_map.py %s --seed" % os.path.join(stages, "pipeline_config.json") not in out:
            f.append("mapping --dry: no 01 --seed")

        # a mapping pass has no 04: 10 writes its track in map from the refined
        # trajectory through T_N_world, under 04's names
        import importlib
        sys.path.insert(0, os.path.join(ROOT, "map_stages"))
        from pipeline_common import load_pipeline, load_traj
        calib = os.path.join(tmp, "calib.json")
        t_, q_ = T.tq(np.linalg.inv(T_cam_sensor))
        json.dump({"camera": {"intrinsics": [500, 500, 320, 240]},
                   "results": {"T_lidar_camera": t_ + q_}}, open(calib, "w"))
        mcfg = os.path.join(stages, "pipeline_config.json")
        c = json.load(open(mcfg))
        c["dataset"]["calib_json"] = calib
        json.dump(c, open(mcfg, "w"))
        P = load_pipeline(mcfg)
        p_c = importlib.import_module("10_publish_poses").mapping_track(P)
        if not p_c or not os.path.exists(p_c):
            f.append("mapping track: not written (%s)" % p_c)
        else:
            ts, Tc = load_traj(p_c)
            for i in range(0, len(ts), 17):
                T.close(Tc[i], T.true_cam(ts[i] - T.T0), "mapping track at %.1f s"
                        % (ts[i] - T.T0), f)

        old = os.path.join(tmp, "old.json")
        json.dump({"dataset": {"bag": "x"}, "08_reference": {"name": "a"}, "09_publish": {}},
                  open(old, "w"))
        c = load_config(old)
        if c.get("04_reference") != {"name": "a"} or "10_publish" not in c or "08_reference" in c:
            f.append("old stage keys: %s" % sorted(c))
    except Exception:                                             # noqa: BLE001
        import traceback
        f.append(traceback.format_exc()[-2500:])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("\n".join(f) or "run pass ok")
    sys.exit(1 if f else 0)


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
