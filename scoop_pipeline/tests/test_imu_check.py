#!/usr/bin/env python
"""Self-test for analysis/slam/imu_check.py on a made-up export: the IMU
standing 26 s, then yawing; its gyro and accelerometer in a frame turned
90 deg about x from the ground truth's, the gyro 0.1 s late.

It must find gravity 90 deg off, and the gyro fitted by a 90 deg rotation
and a +0.10 s shift; with the IMU in the ground truth's frame and on time,
gravity in line and no rotation or shift.

    python scoop_pipeline/tests/test_imu_check.py
"""
import os
import shutil
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "analysis", "slam"))
import imu_check as IC                                              # noqa: E402

RX = np.array([[1.0, 0, 0], [0, 0, -1], [0, 1, 0]])                 # 90 deg about x


def make(d, R, late):
    for s in ("cam0", "imu0", "state_groundtruth_estimate0"):
        os.makedirs(os.path.join(d, "mav0", s))
    t = np.arange(0, 70, 0.005)
    yaw = np.where(t > 26, 0.4 * np.sin(0.5 * (t - 26)), 0.0)
    wz = np.gradient(yaw, t)
    q = np.c_[np.cos(yaw / 2), 0 * t, 0 * t, np.sin(yaw / 2)]
    ns = (10 ** 18 + np.round(t * 1e9)).astype(np.int64)
    np.savetxt(os.path.join(d, "mav0", "cam0", "data.csv"), np.c_[ns[::14]], fmt="%d", header="t")
    gy = np.c_[0 * t, 0 * t, np.interp(t - late, t, wz)] @ R
    acc = np.tile(np.array([0, 0, 9.81]) @ R, (len(t), 1))
    np.savetxt(os.path.join(d, "mav0", "imu0", "data.csv"), np.c_[ns, gy, acc],
               fmt=["%d"] + ["%.6f"] * 6, delimiter=",", header="t")
    np.savetxt(os.path.join(d, "mav0", "state_groundtruth_estimate0", "data.csv"),
               np.c_[ns, np.zeros((len(t), 3)), q, np.zeros((len(t), 9))],
               fmt=["%d"] + ["%.6f"] * 16, delimiter=",", header="t")


def main():
    f = []
    tmp = tempfile.mkdtemp()
    try:
        for name, R, late, ang, shift in (("wrong", RX, 0.1, 90.0, 0.10), ("right", np.eye(3), 0.0, 0.0, 0.0)):
            d = os.path.join(tmp, name)
            make(d, R, late)
            rep = IC.check(d)
            s, dr = rep["still"], rep["driving"]
            if abs(s["angle_deg"] - ang) > 0.5:
                f.append("%s: gravity %.1f deg apart, not %.0f" % (name, s["angle_deg"], ang))
            if abs(dr["rotation_deg"] - ang) > 1.0 or abs(dr["shift_s"] - shift) > 0.011 or dr["fitted_rms"] > 0.01:
                f.append("%s: fit %.1f deg, %+.2f s, rms %.3f" % (name, dr["rotation_deg"], dr["shift_s"],
                                                                  dr["fitted_rms"]))
            if "best fit" not in IC.report(rep):
                f.append("%s: report" % name)
    except Exception:                                              # noqa: BLE001
        import traceback
        f.append(traceback.format_exc())
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("\n".join(f) or "imu check ok")
    sys.exit(1 if f else 0)


if __name__ == "__main__":
    main()
