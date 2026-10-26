#!/usr/bin/env python
"""Self-test for processing/vslam.py's score_monitor: a cuVSLAM run with
VSLAM_MONITOR=1 leaves a monitor whose run.json names only the key topic; it
must be completed with the bag, the robot and the pose frame, and scored as
slam/odometry/score.py scores the other methods. On test_odometry's bag, an
estimate of a frame /tf_static links to the LiDAR, in an odom frame of its own,
must score ATE 0 against the reference.

    python scoop_pipeline/tests/test_vslam_score.py
"""
import json
import os
import shutil
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "processing"))
import test_odometry as TO                                          # noqa: E402
import vslam as V                                                   # noqa: E402


def main():
    tmp = tempfile.mkdtemp()
    f = []
    try:
        bag = os.path.join(tmp, "bag")
        TO.write_bag(bag)
        mon = os.path.join(tmp, "vslam", "odometry", "monitor")
        os.makedirs(mon)
        json.dump({"method": "cuvslam", "key_topic": "/mobile_1/ouster/points"}, open(os.path.join(mon, "run.json"), "w"))
        tr, Tr = TO.ref_traj()
        ref = os.path.join(tmp, "ref.tum")
        TO.write_tum(ref, tr, Tr)
        # the estimate: the IMU frame (standing in for the camera's base frame) in its own odom frame
        T_off = TO.T_of([1.0, -2.0, 0.3], TO.Rotation.from_euler("z", 40, degrees=True).as_quat())
        T_imu_lidar = np.linalg.inv(TO.T_of(*TO.T_SENSOR_IMU)) @ TO.T_of(*TO.T_SENSOR_LIDAR)
        TO.write_tum(os.path.join(mon, "trajectory.tum"), tr,
                     np.einsum("ij,njk,kl->nil", np.linalg.inv(T_off), Tr, np.linalg.inv(T_imu_lidar)))
        c = dict(V.MACHINES["mobile_1"], base_frame="os_imu_mobile1")
        out = V.score_monitor(os.path.join(tmp, "vslam", "odometry"), bag, c, ref)
        rj = json.load(open(os.path.join(mon, "run.json")))
        if rj.get("machine") != "mobile_1" or rj.get("tum_frame") != "os_imu_mobile1" or rj.get("bag") != os.path.abspath(bag):
            f.append("run.json not completed: %s" % rj)
        if not out or "error" in out:
            f.append("not scored: %s" % out)
        elif out["ate"]["position_cm"]["rmse"] > 1e-3:
            f.append("ATE %.4f cm for a perfect estimate" % out["ate"]["position_cm"]["rmse"])
        if not os.path.exists(os.path.join(mon, "accuracy.json")):
            f.append("no accuracy.json")
        if V.score_monitor(os.path.join(tmp, "none"), bag, c, ref) is not None:
            f.append("a run without a monitor was scored")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    if f:
        print("FAIL\n  " + "\n  ".join(f))
        sys.exit(1)
    print("vslam score ok")


if __name__ == "__main__":
    main()
