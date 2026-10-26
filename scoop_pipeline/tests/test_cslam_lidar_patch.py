#!/usr/bin/env python
"""Self-test for slam/patches/cslam_lidar_lc_direction.py: on a copy of
Swarm-SLAM's compute_transform (cslam at fccf8876), the patched function must
return the inverse of TEASER++'s solution (the pose of dst in src), the patch
must be a no-op the second time, and refuse code it does not know.

    python scoop_pipeline/tests/test_cslam_lidar_patch.py
"""
import importlib.util
import os
import shutil
import sys
import tempfile

import numpy as np
from scipy.spatial.transform import Rotation

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "slam", "patches"))
import cslam_lidar_lc_direction as P                                # noqa: E402

# the function as cslam has it, around a stand-in solver: dst = R src + t
CODE = '''import numpy as np
SOLUTION = None

def solve_teaser(src, dst, voxel_size, min_inliers):
    R, t = SOLUTION
    return True, t, R

def compute_transform(src, dst, voxel_size, min_inliers):
    """..."""
    valid, translation, rotation = solve_teaser(src, dst, voxel_size,
                                                min_inliers)
    success = valid
    transform = (translation, rotation)
    return transform, success
'''


def load(path):
    spec = importlib.util.spec_from_file_location("icp_utils_t", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    tmp = tempfile.mkdtemp()
    f = []
    try:
        d = os.path.join(tmp, "Swarm-SLAM", "install", "cslam", "cslam", "lidar_pr")
        os.makedirs(d)
        p = os.path.join(d, "icp_utils.py")
        open(p, "w").write(CODE)
        if P.patch(p) != "patched" or P.patch(p) != "already":
            f.append("not patched once and then left alone")
        m = load(p)
        R = Rotation.from_euler("zyx", [30, 5, -3], degrees=True).as_matrix()
        t = np.array([1.0, -0.5, 0.2])
        m.SOLUTION = (R, t)
        (tt, RR), ok = m.compute_transform(None, None, 0.5, 60)
        T_dst_src = np.eye(4)
        T_dst_src[:3, :3], T_dst_src[:3, 3] = R, t
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = RR, tt
        if not ok or np.abs(T - np.linalg.inv(T_dst_src)).max() > 1e-12:
            f.append("compute_transform does not return the pose of dst in src:\n%s" % T)
        q = os.path.join(d, "other.py")
        open(q, "w").write("def compute_transform(): pass\n")
        if P.patch(q) != "not found":
            f.append("unknown code patched")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    if f:
        print("FAIL\n  " + "\n  ".join(f))
        sys.exit(1)
    print("cslam lidar patch ok")


if __name__ == "__main__":
    main()
