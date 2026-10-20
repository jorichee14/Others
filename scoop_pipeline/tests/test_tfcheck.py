#!/usr/bin/env python
"""Self-test for analysis/tfcheck.py (check.py tf) on a made-up poses bag:
map -> board -> origin ~~ base (on /tf) -> camera_link -> optical, base ->
os_sensor -> os_lidar, the reference LiDAR and camera trajectories in map, a
calibration file and /robot/global_pose. Right, it must read consistent; with
the base -> os_sensor edge turned 1 deg it must be caught, against both the
calibration and the reference trajectories.

    python scoop_pipeline/tests/test_tfcheck.py
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
from scipy.spatial.transform import Rotation

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from scoop import rosmsg, tftree                                    # noqa: E402
from scoop.bagwrite import BagWriter                                # noqa: E402


def T(xyz=(0, 0, 0), rpy=(0, 0, 0)):
    M = np.eye(4)
    M[:3, :3] = Rotation.from_euler("xyz", rpy, degrees=True).as_matrix()
    M[:3, 3] = xyz
    return M


T_MAP_BOARD = T((0.2, -0.1, 0.0), (0, 0, 30))
T_BOARD_ORIGIN = T((0.4, 0.0, 0.2), (0, 8, 180))
T_BASE_LINK = T()                                                   # base on camera_link
T_LINK_OPT = T((0.0, -0.06, 0.0), (-90, 0, -90))
T_BASE_SENSOR = T((-0.27, 0.03, 0.15), (0, 0, 180))
T_SENSOR_LIDAR = T((0, 0, 0.036), (0, 0, 180))
T0 = 1.79e9


def motion(t):
    return T((0.4 * t, 0.3 * np.sin(t / 3), 0.0), (0, 0, 20 * np.sin(t / 5)))


def write_tum(fn, ts, Ts):
    q = Rotation.from_matrix(Ts[:, :3, :3]).as_quat()
    np.savetxt(fn, np.column_stack([ts, Ts[:, :3, 3], q]), fmt="%.9f")


def build(tmp, tilt=0.0):
    base_dir = os.path.join(tmp, "processed", "20261015", "mapping")
    ref = os.path.join(base_dir, "odometry", "reference_mapping_20261015")
    os.makedirs(ref, exist_ok=True)
    ts = T0 + np.arange(0, 20, 0.1)
    T_mo = T_MAP_BOARD @ T_BOARD_ORIGIN
    base = np.array([T_mo @ motion(t - T0) for t in ts])
    T_base_lidar = T_BASE_SENSOR @ T_SENSOR_LIDAR
    T_base_opt = T_BASE_LINK @ T_LINK_OPT
    write_tum(os.path.join(ref, "traj_mobile_2_lidar.tum"), ts, base @ T_base_lidar)
    write_tum(os.path.join(ref, "traj_mobile_2_lidar_in_cam.tum"), ts, base @ T_base_opt)
    calib = os.path.join(tmp, "calib.json")
    tl, ql = tftree.to_tq(np.linalg.inv(T_base_lidar) @ T_base_opt)      # T_lidar_camera
    json.dump({"camera": {"intrinsics": [427, 427, 423, 240]}, "results": {"T_lidar_camera": tl + ql}},
              open(calib, "w"))
    bag = os.path.join(base_dir, "bags", "mapping_20261015_best_poses")
    os.makedirs(os.path.dirname(bag), exist_ok=True)
    sensor = T_BASE_SENSOR @ T((0, 0, 0), (0, 0, tilt))
    st = [("map", "board", T_MAP_BOARD), ("board", "origin", T_BOARD_ORIGIN),
          ("base", "camera_link", np.linalg.inv(T_BASE_LINK)), ("camera_link", "optical", T_LINK_OPT),
          ("base", "os_sensor", sensor), ("os_sensor", "os_lidar", T_SENSOR_LIDAR)]
    with BagWriter(bag) as w:
        t_ns0 = int(ts[0] * 1e9)
        w.write("/tf_static", rosmsg.tf_message([(p, c, *tftree.to_tq(M)) for p, c, M in st], t_ns0), t_ns0)
        for t in ts:
            t_ns = int(round(t * 1e9))
            Mo = motion(t - T0)
            w.write("/tf", rosmsg.tf_message([("origin", "base", *tftree.to_tq(Mo))], t_ns), t_ns)
            tg, qg = tftree.to_tq(T_mo @ Mo)
            w.write("/mobile_2/global_pose", rosmsg.pose_stamped(t_ns, "map", tg, qg), t_ns)
    cfg = os.path.join(tmp, "pipeline_config.json")
    json.dump({"dataset": {"bag": os.path.join(tmp, "work", "20261015", "mapping", "mobile_2", "x"),
                           "processed": base_dir, "machine": "mobile_2", "calib_json": calib,
                           "traj": "traj_lidar_ba_glim.txt"},
               "10_publish": {"robots": [{"name": "mobile_2", "optical_frame": "optical",
                                          "base_frame": {"name": "base", "measured_in": "camera_link",
                                                         "children": ["camera_link", "os_sensor"]}}]}},
              open(cfg, "w"))
    return cfg


def main():
    f = []
    for tilt, want in ((0.0, None), (1.0, 2)):            # calibration + LiDAR in map
        tmp = tempfile.mkdtemp()
        try:
            r = subprocess.run([sys.executable, os.path.join(ROOT, "check.py"), "tf", build(tmp, tilt),
                                "--lidar", "os_lidar", "--every", "5"], capture_output=True, text=True)
            out = r.stdout + r.stderr
            if os.environ.get("SHOW"):
                print(out)
            if want is None:
                if r.returncode or "the TF is consistent" not in out:
                    f.append("right TF not consistent:\n" + out[-2500:])
            else:
                bad = [ln for ln in out.splitlines() if ln.startswith("  !")]
                if r.returncode != 1 or len(bad) != want or not any("calib" in b for b in bad):
                    f.append("a 1 deg wrong base -> os_sensor not caught (%d !):\n%s" % (len(bad), out[-2500:]))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    print("\n".join(f) or "tf check ok")
    sys.exit(1 if f else 0)


if __name__ == "__main__":
    main()
