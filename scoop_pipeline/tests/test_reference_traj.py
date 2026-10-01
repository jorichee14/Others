#!/usr/bin/env python
"""Self-test for map_stages/08_reference_traj.py: a run's scans registered to
the reference pass's frozen anchored map, seeded by its GLIM trajectory placed
with 06's session anchor.

The scene, run and reference map are tests/test_reloc.py's (a room with
boxes; the run's GLIM world is the map turned 137 deg and moved 6.3 m). The
session anchor is the true camera pose at the start moved 4 cm and turned
0.6 deg -- a start that is close but not exact, as from the board -- and the
GLIM trajectory drifts 0.5 deg / 6 cm over the run. 08 must bring every scan
back to its true pose.

    python scoop_pipeline/tests/test_reference_traj.py
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import test_reloc as R                                              # noqa: E402

STAGES = R.STAGES


def main():
    tmp = tempfile.mkdtemp()
    failed = []
    try:
        cfg, data, T_lc = R.build(os.path.join(tmp, "run"), R.BOXES)
        # GLIM with drift: the run's own odometry is close, not exact
        glim = os.path.join(data, "work", "20260101", "survey_1", "mobile_1", "glim",
                            "traj_lidar.txt")
        stamps = np.arange(0.05, R.DUR, 0.1)
        T_run_map = np.linalg.inv(R.T_MAP_RUN)
        R.write_tum(glim, R.T0 + stamps,
                    [T_run_map @ R.rotz(0.5 * t / R.DUR, (0.06 * t / R.DUR, 0, 0))
                     @ R.true_pose(t) for t in stamps])
        # 06's session anchor: the camera at the start, 4 cm / 0.6 deg off
        t_a = R.T0 + 0.05
        T_cam = R.rotz(0.6, (0.03, -0.0265, 0.0)) @ R.true_pose(0.05) @ T_lc
        from scipy.spatial.transform import Rotation
        fr = os.path.join(data, "processed", "20260101", "survey_1", "frames")
        os.makedirs(fr, exist_ok=True)
        json.dump({"cameras": {"zed": {
            "mode": "lidar_odom", "dwell_t_end": t_a,
            "map_to_cam": {"xyz": T_cam[:3, 3].tolist(),
                           "qxyzw": Rotation.from_matrix(T_cam[:3, :3]).as_quat().tolist()}}}},
            open(os.path.join(fr, "session_anchor.json"), "w"))
        c = json.load(open(cfg))
        c["08_reference"] = {"name": "mobile_1_lidar", "points_topic": "/mobile_1/ouster/points",
                             "anchor_cam": "zed", "rounds": 2}
        json.dump(c, open(cfg, "w"))
        out = subprocess.run([sys.executable, os.path.join(STAGES, "08_reference_traj.py"), cfg],
                             capture_output=True, text=True, cwd=STAGES)
        print(out.stdout[-2500:])
        if out.returncode:
            raise AssertionError(out.stdout[-1500:] + out.stderr[-2000:])
        od = os.path.join(data, "processed", "20260101", "survey_1", "odometry",
                          "reference_survey_1_20260101")
        d = np.loadtxt(os.path.join(od, "traj_mobile_1_lidar.tum"))
        pe, re = [], []
        for row in d:
            G = R.true_pose(row[0] - R.T0)
            T = np.eye(4)
            T[:3, :3] = Rotation.from_quat(row[4:8]).as_matrix()
            T[:3, 3] = row[1:4]
            pe.append(np.linalg.norm(T[:3, 3] - G[:3, 3]) * 100)
            re.append(np.degrees(np.arccos(np.clip((np.trace(T[:3, :3].T @ G[:3, :3]) - 1) / 2,
                                                   -1, 1))))
        pe, re = np.array(pe), np.array(re)
        print(f"08 vs truth: median {np.median(pe):.2f} cm, p95 {np.percentile(pe, 95):.2f} cm, "
              f"max {pe.max():.2f} cm | rotation median {np.median(re):.3f} deg, "
              f"max {re.max():.3f} deg")
        if not (np.median(pe) < 0.5 and np.percentile(pe, 95) < 1.0 and np.median(re) < 0.05):
            failed.append(f"poses off: {np.median(pe):.2f} / {np.percentile(pe, 95):.2f} cm, "
                          f"{np.median(re):.3f} deg")
        cam = np.loadtxt(os.path.join(od, "traj_mobile_1_lidar_in_cam.tum"))
        if len(cam) != len(d):
            failed.append("in_cam trajectory has another length")
        q = open(os.path.join(od, "quality_mobile_1_lidar.csv")).read().splitlines()
        ok = sum(1 for ln in q[1:] if ",ok," in ln)
        print(f"quality: {ok}/{len(q) - 1} scans registered")
        if ok < 0.95 * (len(q) - 1):
            failed.append(f"only {ok} of {len(q) - 1} scans registered")
    except AssertionError as e:
        failed.append(str(e)[-2500:])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("\n".join(failed) or "reference trajectory ok")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
