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


def read_poses(path):
    from scipy.spatial.transform import Rotation
    d = np.loadtxt(path)
    T = np.tile(np.eye(4), (len(d), 1, 1))
    T[:, :3, :3] = Rotation.from_quat(d[:, 4:8]).as_matrix()
    T[:, :3, 3] = d[:, 1:4]
    return d[:, 0], T


def corridor_case(tmp):
    """A 40 m corridor: features at both ends, bare walls in the middle, the
    lidar limited to 6 m so the end walls are out of reach there. GLIM drifts
    30 cm along the corridor over the run. The middle scans must come out
    seed-held along the corridor (+-x), with the segment's drift estimate close
    to GLIM's drift across it, and still right across the corridor."""
    from scipy.spatial.transform import Rotation
    saved = (R.ROOM, R.BOXES, R.true_pose, R.DUR)
    R.ROOM = np.array([[-20.0, -1.2, 0.0], [20.0, 1.2, 3.0]])
    R.BOXES = np.array([[[-19.0, -1.2, 0.0], [-18.0, -0.6, 1.5]],
                        [[-16.5, 0.7, 0.0], [-16.0, 1.2, 3.0]],
                        [[16.0, -1.2, 0.0], [16.6, -0.7, 3.0]],
                        [[18.0, 0.6, 0.0], [19.0, 1.2, 1.0]]])
    R.DUR = 30.0

    def tp(t):
        return R.rotz(0.0, (-17.0 + 34.0 * t / R.DUR, 0.0, 0.6))
    R.true_pose = tp
    try:
        cfg, data, T_lc = R.build(os.path.join(tmp, "corridor"), R.BOXES)
        stamps = np.arange(0.05, R.DUR, 0.1)
        drift = lambda t: 0.30 * t / R.DUR                       # noqa: E731
        glim = os.path.join(data, "work", "20260101", "survey_1", "mobile_1", "glim",
                            "traj_lidar.txt")
        T_run_map = np.linalg.inv(R.T_MAP_RUN)
        R.write_tum(glim, R.T0 + stamps,
                    [T_run_map @ R.rotz(0, (drift(t), 0, 0)) @ tp(t) for t in stamps])
        T_cam = tp(0.05) @ T_lc
        fr = os.path.join(data, "processed", "20260101", "survey_1", "frames")
        os.makedirs(fr, exist_ok=True)
        json.dump({"cameras": {"zed": {
            "mode": "lidar_odom", "dwell_t_end": R.T0 + 0.05,
            "map_to_cam": {"xyz": T_cam[:3, 3].tolist(),
                           "qxyzw": Rotation.from_matrix(T_cam[:3, :3]).as_quat().tolist()}}}},
            open(os.path.join(fr, "session_anchor.json"), "w"))
        c = json.load(open(cfg))
        c["01_build_map"]["lidar_max"] = 6.0
        c["08_reference"] = {"name": "mobile_1_lidar", "points_topic": "/mobile_1/ouster/points",
                             "anchor_cam": "zed", "rounds": 2}
        json.dump(c, open(cfg, "w"))
        out = subprocess.run([sys.executable, os.path.join(STAGES, "08_reference_traj.py"), cfg],
                             capture_output=True, text=True, cwd=STAGES)
        print(out.stdout[-2200:])
        if out.returncode:
            raise AssertionError(out.stdout[-1500:] + out.stderr[-2000:])
        od = os.path.join(data, "processed", "20260101", "survey_1", "odometry",
                          "reference_survey_1_20260101")
        segs = np.atleast_2d(np.loadtxt(os.path.join(od, "seed_held_mobile_1_lidar.csv"),
                                        delimiter=",", skiprows=1))
        g = segs[np.argmax(segs[:, 2])]                          # the longest run
        t0, t1 = g[0] - R.T0, g[1] - R.T0
        weak = g[8:11]
        off0, off1 = g[12], g[13]
        ts, T = read_poses(os.path.join(od, "traj_mobile_1_lidar.tum"))
        inside = (ts - R.T0 >= t0) & (ts - R.T0 <= t1)
        err = np.array([T[i, :3, 3] - tp(ts[i] - R.T0)[:3, 3] for i in np.flatnonzero(inside)])
        out_ = ~inside & ((ts - R.T0 < 3) | (ts - R.T0 > R.DUR - 3))
        err_ends = np.array([np.linalg.norm(T[i, :3, 3] - tp(ts[i] - R.T0)[:3, 3])
                             for i in np.flatnonzero(out_)]) * 100
        print(f"corridor: seed-held {t0:.1f}..{t1:.1f} s ({g[3]:.0f} scans), open direction "
              f"{np.round(weak, 2).tolist()}, seed off at the edges {off0:.1f} / {off1:.1f} cm "
              f"(true {drift(t0) * 100:.1f} / {drift(t1) * 100:.1f}); inside: along "
              f"{np.abs(err[:, 0]).max() * 100:.1f} cm max, "
              f"across {np.abs(err[:, 1:]).max() * 100:.2f} cm max; ends "
              f"{np.median(err_ends):.2f} cm")
        f = []
        if not (abs(weak[0]) > 0.95 and t1 - t0 > 8):
            f.append(f"corridor: open direction {weak} over {t1 - t0:.1f} s")
        if not (abs(abs(off0) - drift(t0) * 100) < 2.0 and abs(abs(off1) - drift(t1) * 100) < 2.0):
            f.append(f"corridor: edge estimates {off0:.1f} / {off1:.1f} cm")
        if not (np.abs(err[:, 1:]).max() < 0.02 and np.median(err_ends) < 1.0):
            f.append("corridor: constrained directions off")
        if not np.abs(err[:, 0]).max() * 100 <= g[14] + 2.0:
            f.append("corridor: along-corridor error beyond what the segment reports")
        return f
    finally:
        R.ROOM, R.BOXES, R.true_pose, R.DUR = saved


def main():
    tmp = tempfile.mkdtemp()
    failed = []
    try:
        failed += corridor_case(tmp)
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
        held = sum(1 for ln in q[1:] if ln.endswith(",1"))
        print(f"room with boxes: {held} seed-held scans")
        if held > 0.02 * (len(q) - 1):
            failed.append(f"room with boxes: {held} scans seed-held")
    except AssertionError as e:
        failed.append(str(e)[-2500:])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("\n".join(failed) or "reference trajectory ok")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
