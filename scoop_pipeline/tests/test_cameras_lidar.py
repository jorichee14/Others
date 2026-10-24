#!/usr/bin/env python
"""Self-test for 07_build_cameras.py "source": "lidar": a parked robot's camera
placed by registering its own LiDAR scans to the map, seeded by its board
placement, which stays as the check.

tests/test_reloc.py's room (boxes, so no turn maps it onto itself) is the map;
the robot's LiDAR scans it from one pose, its camera hangs from the LiDAR by a
calibration in the bag's /tf_static. The board placement is given 15 cm and
3 deg off the truth. Cases:

  parked   the camera from the LiDAR within 1 cm / 0.1 deg of the truth, the
           board placement reported ~15 cm / 3 deg from it;
  no board the camera never sees one: the robot is placed from its own LiDAR
           searched over the whole map ("reloc"), same accuracy, no board;
  moving   the robot drives during the scans: the scans disagree, the board
           placement is kept.

    python scoop_pipeline/tests/test_cameras_lidar.py
"""
import importlib
import os
import shutil
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "map_stages"), os.path.dirname(os.path.abspath(__file__))]

import test_reloc as R                                              # noqa: E402
from scoop import rosmsg                                            # noqa: E402
from scoop.bagwrite import BagWriter                                # noqa: E402
from scipy.spatial.transform import Rotation                        # noqa: E402

C = importlib.import_module("07_build_cameras")
CAM = "camera_color_optical_frame"
TOPIC = "/mobile_2/ouster/points"


def T_lc():
    """The camera in the LiDAR frame (optical axes)."""
    T = np.eye(4)
    T[:3, :3] = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], float)
    T[:3, 3] = [0.05, 0.02, 0.12]
    return T


def bag_of(path, poses):
    rng = np.random.default_rng(3)
    w = BagWriter(path)
    Tc = T_lc()
    q = Rotation.from_matrix(Tc[:3, :3]).as_quat()
    w.write("/tf_static", rosmsg.tf_message([("os_lidar", CAM, Tc[:3, 3], q)], R.T0 * R.S),
            R.T0 * R.S)
    for i, T in enumerate(poses):
        ns = R.T0 * R.S + int((i + 1) * 0.1 * R.S)
        w.write(TOPIC, R.scan(T, rng, ns, R.BOXES), ns)
    w.close()


def run_case(tmp, poses, T_true_cam, board=True):
    import open3d as o3d
    mp = os.path.join(tmp, "map.pcd")
    if not os.path.exists(mp):
        o3d.io.write_point_cloud(mp, o3d.geometry.PointCloud(
            o3d.utility.Vector3dVector(R.surfaces(R.BOXES))))
    bag = os.path.join(tmp, "bag_%d" % len(os.listdir(tmp)))
    bag_of(bag, poses)
    off = R.rotz(3.0, (0.12, -0.09, 0.0))                        # the board's error
    T_board = T_true_cam @ off
    board_rec = {"name": "realsense", "child_frame": CAM, "parent_frame": "map",
                 "source": "board_avg", "board": "rs_anchor",
                 "translation": T_board[:3, 3].tolist(),
                 "quaternion_xyzw": Rotation.from_matrix(T_board[:3, :3]).as_quat().tolist()}
    # the board part is 07's own; `board` False is a camera that never sees one
    C.camera_from_board = lambda cam, ctx: (dict(board_rec) if board else None)
    cam = {"name": "realsense", "child_frame": CAM, "source": "lidar", "board": "rs_anchor",
           "points_topic": TOPIC, "n_scans": 8}
    ctx = {"default_bag": bag, "map_file": mp, "map_frame": "map"}
    rec = C.camera_from_lidar(cam, ctx, None)
    if rec is None:
        return None, float("nan"), float("nan")
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat(rec["quaternion_xyzw"]).as_matrix()
    T[:3, 3] = rec["translation"]
    err_mm = np.linalg.norm(T[:3, 3] - T_true_cam[:3, 3]) * 1000
    err_deg = np.degrees(np.linalg.norm(Rotation.from_matrix(
        T[:3, :3].T @ T_true_cam[:3, :3]).as_rotvec()))
    return rec, err_mm, err_deg


def main():
    tmp = tempfile.mkdtemp(prefix="cam_lidar_")
    fails = []
    try:
        T_lidar = R.rotz(40.0, (1.5, -2.0, 0.6))
        T_cam = T_lidar @ T_lc()
        rec, e_mm, e_deg = run_case(tmp, [T_lidar] * 20, T_cam)
        print("parked: %s, %.1f mm / %.3f deg from the truth, board check %s mm / %s deg"
              % (rec["source"], e_mm, e_deg, rec.get("board_check_mm"), rec.get("board_check_deg")))
        if rec["source"] != "lidar_avg" or e_mm > 10 or e_deg > 0.1:
            fails.append("parked: %s %.1f mm / %.3f deg" % (rec["source"], e_mm, e_deg))
        if not (130 < rec.get("board_check_mm", 0) < 170 and 2.5 < rec.get("board_check_deg", 0) < 3.5):
            fails.append("parked: board check %s mm / %s deg, want ~150 / 3"
                         % (rec.get("board_check_mm"), rec.get("board_check_deg")))
        # no board in sight: the robot is placed from its own LiDAR alone
        rec, e_mm, e_deg = run_case(tmp, [T_lidar] * 20, T_cam, board=False)
        print("no board: %s, %.1f mm / %.3f deg from the truth, %.0f %% of the window "
              "on the map" % (rec["source"], e_mm, e_deg, 100 * rec.get("reloc_fitness", 0)))
        if rec["source"] != "reloc_avg" or e_mm > 10 or e_deg > 0.1:
            fails.append("no board: %s %.1f mm / %.3f deg" % (rec["source"], e_mm, e_deg))

        moving = [R.rotz(40.0 + 2 * i, (1.5 + 0.15 * i, -2.0, 0.6)) for i in range(20)]
        rec, _, _ = run_case(tmp, moving, T_cam)
        print("moving: %s" % rec["source"])
        if rec["source"] != "board_avg":
            fails.append("moving: %s, want the board placement kept" % rec["source"])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    for f in fails:
        print("  FAIL  " + f)
    print("cameras lidar " + ("FAILED" if fails else "ok"))
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
