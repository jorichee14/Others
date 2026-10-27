#!/usr/bin/env python
"""Self-test for processing/vslam_loc.py, no ROS, no GPU: the hook script
compiles; a map's keyframes, in the camera base frame and the map's own frame,
fitted onto a reference track in the twin give back the map's pose in the twin
(residual 0); the poses in the map are map -> odom (the latest at or before
each stamp) composed with the odometry, from the localization's moment on;
TUM round trip.

    python scoop_pipeline/tests/test_vslam_loc.py
"""
import os
import sys
import tempfile

import numpy as np
from scipy.spatial.transform import Rotation

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "processing"))
import vslam_loc as VL                                              # noqa: E402


def T_of(t, rv):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_rotvec(rv).as_matrix()
    T[:3, 3] = t
    return T


def main():
    f = []
    compile(VL.SRV, "vslam_srv.py", "exec")
    # a reference T_twin_lidar: a 4 m loop with some heading change, 10 Hz
    t = 1000.0 + np.arange(300) * 0.1
    a = t - t[0]
    R = np.array([T_of([2 * np.cos(0.2 * x), 2 * np.sin(0.2 * x), 0.3 + 0.02 * np.sin(x)], [0.01, 0.0, 0.2 * x + 1.6])
                  for x in a])
    A = T_of([5.0, -3.0, 0.2], [0.0, 0.0, 0.7])                       # T_twin_cmap
    T_bl = T_of([0.1, 0.0, -0.2], [0.0, 0.05, 3.1])                   # T_base_lidar
    K = np.linalg.inv(A) @ R @ np.linalg.inv(T_bl)                   # keyframes T_cmap_base
    Af, res, n = VL.align(t[::10], K[::10], t, R, T_bl)
    if np.abs(Af - A).max() > 1e-6 or res.max() > 1e-6 or n != 30:
        f.append("align: got\n%s\nwant\n%s, residual %s, %d" % (Af, A, res.max(), n))
    # a bent map: the second half of the drive is off by another 2 m / 5 deg; a run near the first half
    # gets the first half's transform back from align_near, exactly
    d = tempfile.mkdtemp()
    far = np.array([T_of([20.0 + 0.1 * k, 0.0, 0.0], [0, 0, 0]) for k in range(150)])   # the drive's second part
    Rm = np.concatenate([R[:150], far])                                  # reference of the mapping drive
    A2 = T_of([2.0, 1.0, 0.0], [0.0, 0.0, 0.09]) @ A
    Km = np.concatenate([np.linalg.inv(A) @ Rm[:150] @ np.linalg.inv(T_bl),
                         np.linalg.inv(A2) @ Rm[150:] @ np.linalg.inv(T_bl)])
    VL.write_tum(os.path.join(d, "keyframes.tum"), t, Km)
    VL.write_tum(os.path.join(d, "ref.tum"), t, Rm)
    aj = {"reference": os.path.join(d, "ref.tum"), "residual_cm": {"rmse": 100.0}}
    Al = VL.align_near(d, aj, T_bl, R[:150], 1.0, os.path.join(d, "out", "vslam_loc"))
    if np.abs(Al - A).max() > 1e-4:
        f.append("align_near: got\n%s\nwant\n%s" % (Al, A))
    # map -> odom: identity until 1010 s (not localized), then a correction, then another
    C1, C2 = T_of([1.0, 0.0, 0.0], [0, 0, 0.1]), T_of([1.1, 0.0, 0.0], [0, 0, 0.1])
    tf = [(1000.0, np.eye(4)), (1010.0, C1), (1020.0, C2)]
    O = np.array([T_of([0.1 * x, 0.0, 0.0], [0, 0, 0]) for x in a])
    tm, Tm = VL.compose(t, O, tf, 1010.0)
    k15 = int(np.argmin(np.abs(t - 1015.0)))
    k25 = int(np.argmin(np.abs(t - 1025.0)))
    j15, j25 = int(np.argmin(np.abs(tm - 1015.0))), int(np.argmin(np.abs(tm - 1025.0)))
    if tm[0] < 1010.0 - 1e-9 or np.abs(Tm[j15] - C1 @ O[k15]).max() > 1e-9 or np.abs(Tm[j25] - C2 @ O[k25]).max() > 1e-9:
        f.append("compose: first %s, at 1015 / 1025 s not map->odom @ odom" % tm[0])
    # outputs beside the robot's: no second <machine>/ for a bag already in the robot's folder
    class P:
        def __init__(self, bag):
            self.dataset, self.cfg = {"bag": bag}, {}
    for bag, want in (("/d/work/x/coop/site_1/coop_bag", "/d/work/x/coop/site_1/mobile_2/vslam_loc"),
                      ("/d/work/x/mapping/mobile_2/map_bag", "/d/work/x/mapping/mobile_2/vslam_loc")):
        if VL.out_for(P(bag), "mobile_2", "vslam_loc") != want:
            f.append("out_for %s: %s" % (bag, VL.out_for(P(bag), "mobile_2", "vslam_loc")))
    p = os.path.join(tempfile.mkdtemp(), "x.tum")
    VL.write_tum(p, t[:5], R[:5])
    t2, R2 = VL.load_tum(p)
    if np.abs(R2 - R[:5]).max() > 1e-5 or np.abs(t2 - t[:5]).max() > 1e-6:
        f.append("tum round trip")
    if f:
        print("FAIL\n  " + "\n  ".join(f))
        sys.exit(1)
    print("vslam_loc ok")


if __name__ == "__main__":
    main()
