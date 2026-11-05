#!/usr/bin/env python
"""Self-tests for processing/place_cameras.py: a synthetic L-shaped room, the
robot's still cloud taken from a known pose with the LiDAR tilted, its tilt
known from a mapping trajectory. The robot is placed back to its pose, and a
room camera follows through T_lidar_colour and the ChArUco extrinsic. Needs
numpy, scipy, open3d.

    python tests/test_place_cameras.py
"""
import os
import sys
import tempfile
import shutil
import traceback

import numpy as np
from scipy.spatial.transform import Rotation as R

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "processing"))

import place_cameras as pc                                              # noqa: E402

FAILED = []


def plane(c, u, v, nu, nv, step=0.04):
    a, b = np.meshgrid(np.arange(0, nu, step), np.arange(0, nv, step))
    return np.asarray(c) + a.ravel()[:, None] * u + b.ravel()[:, None] * v


def room():
    """L-shaped room, 2.6 m high, a cabinet: no symmetry to confuse heading."""
    X, Y, Z = np.eye(3)
    P = [plane([0, 0, 0], X, Y, 10, 6), plane([0, 0, 2.6], X, Y, 10, 6),
         plane([0, 0, 0], X, Z, 10, 2.6), plane([0, 0, 0], Y, Z, 6, 2.6),
         plane([10, 0, 0], Y, Z, 3, 2.6), plane([6, 3, 0], X, Z, 4, 2.6),
         plane([6, 3, 0], Y, Z, 3, 2.6), plane([0, 6, 0], X, Z, 6, 2.6),
         plane([2, 4, 0], X, Z, 1.5, 1.2), plane([2, 4.6, 0], X, Z, 1.5, 1.2),
         plane([2, 4, 0], Y, Z, 0.6, 1.2), plane([2, 4, 1.2], X, Y, 1.5, 0.6)]
    P = np.vstack(P)
    keep = ~((P[:, 0] > 6.02) & (P[:, 1] > 3.02))                     # the cut-out corner
    return P[keep]


def test_place(d):
    rng = np.random.default_rng(1)
    M = room()
    # LiDAR pose in the map: at (3, 2, 0.5), heading 35 deg, pitch 1.5, roll -1
    T_ml = np.eye(4)
    T_ml[:3, :3] = R.from_euler("ZYX", [35, 1.5, -1.0], degrees=True).as_matrix()
    T_ml[:3, 3] = [3.0, 2.0, 0.5]
    scan_map = M[np.linalg.norm(M - T_ml[:3, 3], axis=1) > 1.0]
    scan_map = scan_map[rng.random(len(scan_map)) < 0.7]
    scan = (scan_map - T_ml[:3, 3]) @ T_ml[:3, :3]                     # in the LiDAR frame
    scan += rng.normal(0, 0.01, scan.shape)
    # mapping trajectory: camera poses with the same LiDAR tilt, many headings
    T_lc = pc.T_of([-0.0856, -0.0101, -0.0274], [0.515573, 0.513620, -0.483613, -0.486310])
    rows = []
    for k, yaw in enumerate(np.linspace(0, 300, 40)):
        Tl = np.eye(4)
        Tl[:3, :3] = R.from_euler("ZYX", [yaw, 1.5, -1.0], degrees=True).as_matrix()
        Tl[:3, 3] = [1 + 0.1 * k, 2, 0.5]
        Tc = Tl @ T_lc
        rows.append([k] + list(Tc[:3, 3]) + list(R.from_matrix(Tc[:3, :3]).as_quat()))
    tr = os.path.join(d, "traj.tum")
    np.savetxt(tr, np.array(rows))
    L, pitch, roll, spread = pc.level_from_traj(tr, T_lc)
    assert abs(pitch - 1.5) < 0.05 and abs(roll + 1.0) < 0.05 and spread < 0.05, (pitch, roll, spread)
    T, fit, rmse, riv, cands = pc.place_robot(scan, M, L)
    dt = np.linalg.norm(T[:3, 3] - T_ml[:3, 3])
    da = np.degrees(R.from_matrix(T[:3, :3].T @ T_ml[:3, :3]).magnitude())
    assert dt < 0.03 and da < 0.3, (dt, da, T)
    assert fit > 0.8, fit
    assert riv is None or riv["unique"], riv
    # the room camera follows: 2 m ahead of the colour camera, looking back at it
    T_cc = np.eye(4)
    T_cc[:3, :3] = R.from_euler("y", 180, degrees=True).as_matrix()
    T_cc[:3, 3] = [0, 0, 2.0]
    want = T_ml @ T_lc @ T_cc
    got = T @ T_lc @ T_cc
    assert np.linalg.norm(got[:3, 3] - want[:3, 3]) < 0.04, (got, want)


def test_frustum_and_tq():
    T = pc.T_of([1, 2, 3], R.from_euler("z", 90, degrees=True).as_quat())
    t, q = pc.tq(T)
    assert np.allclose(pc.T_of(t, q), T, atol=1e-5)
    F = pc.frustum(T, 0.3)
    assert np.linalg.norm(F - [1, 2, 3], axis=1).max() < 0.5


def test_find_bag(d):
    for name in ("ed305_calib_1", "ed305_calib_11", "ed305_dataset_calibration_cam10", "ed305_extrinsics_20261105"):
        os.makedirs(os.path.join(d, name))
        open(os.path.join(d, name, "metadata.yaml"), "w").write("x")
    b = lambda n: [os.path.basename(x) for x in pc.find_bag(d, n)]       # noqa: E731
    assert b(1) == ["ed305_calib_1"], b(1)
    assert b(11) == ["ed305_calib_11"], b(11)
    assert b(10) == ["ed305_dataset_calibration_cam10"], b(10)
    assert b(2) == [], b(2)


def check(name, fn, *args):
    try:
        fn(*args)
        print("  ok    %s" % name)
    except Exception:
        FAILED.append(name)
        print("  FAIL  %s" % name)
        traceback.print_exc()


def main():
    d = tempfile.mkdtemp(prefix="place_cameras_")
    tests = [(k, f) for k, f in sorted(globals().items()) if k.startswith("test_")]
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
