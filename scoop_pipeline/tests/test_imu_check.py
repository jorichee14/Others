#!/usr/bin/env python
"""Self-test for analysis/slam/imu_check.py on made-up exports: the IMU (x
forward, z up) standing 26 s, then yawing at random; the left camera looking
along x (T_BS in params/), its images a texture sliding by fx times the yaw;
the ground truth on the images' clock.

  late    the gyro's stamps 2.1 s behind: gyro vs images and vs ground truth
          +2.10 s, ground truth vs images 0, all correlated, scale 1;
          gravity in line and no rotation
  turned  the IMU's readings in a frame turned 90 deg about x, on time:
          gravity 90 deg off, the gyro fitted by a 90 deg rotation

    python scoop_pipeline/tests/test_imu_check.py
"""
import os
import shutil
import sys
import tempfile

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "analysis", "slam"))
import imu_check as IC                                              # noqa: E402

RX = np.array([[1.0, 0, 0], [0, 0, -1], [0, 1, 0]])                 # 90 deg about x
R_BS = np.array([[0.0, 0, 1], [-1, 0, 0], [0, -1, 0]])              # camera axes in the IMU frame
FX, W, H = 200.0, 240, 136


def make(d, R, late):
    for s in ("cam0/data", "imu0", "state_groundtruth_estimate0"):
        os.makedirs(os.path.join(d, "mav0", s))
    os.makedirs(os.path.join(d, "params"))
    T = np.eye(4)
    T[:3, :3] = R_BS
    open(os.path.join(d, "params", "LeftCameraParams.yaml"), "w").write(
        "%%YAML:1.0\nT_BS:\n  cols: 4\n  rows: 4\n  data: [%s]\nintrinsics: [%f, %f, 120.0, 68.0]\n"
        % (", ".join("%.6f" % x for x in T.ravel()), FX, FX))
    t = np.arange(0, 75, 0.005)
    rng = np.random.default_rng(1)
    wz = np.convolve(rng.normal(0, 1, len(t)), np.ones(400) / 400, "same") * 6
    wz[t < 26] = 0
    wz = np.clip(wz, -0.4, 0.4)
    yaw = np.cumsum(wz) * 0.005
    q = np.c_[np.cos(yaw / 2), 0 * t, 0 * t, np.sin(yaw / 2)]
    ns = (10 ** 18 + np.round(t * 1e9)).astype(np.int64)
    gy = np.c_[0 * t, 0 * t, np.interp(t - late, t, wz)] @ R
    acc = np.tile(np.array([0, 0, 9.81]) @ R, (len(t), 1))
    np.savetxt(os.path.join(d, "mav0", "imu0", "data.csv"), np.c_[ns, gy, acc],
               fmt=["%d"] + ["%.6f"] * 6, delimiter=",", header="t")
    np.savetxt(os.path.join(d, "mav0", "state_groundtruth_estimate0", "data.csv"),
               np.c_[ns, np.zeros((len(t), 3)), q, np.zeros((len(t), 9))],
               fmt=["%d"] + ["%.6f"] * 16, delimiter=",", header="t")
    # the camera turns by -yaw about its y axis: the scene slides right by fx * yaw
    pad = int(FX * (np.abs(yaw).max() + 0.1)) + 5
    tex = cv2.GaussianBlur(np.uint8(rng.integers(0, 255, (H, W + 2 * pad))), (5, 5), 1.5)
    with open(os.path.join(d, "mav0", "cam0", "data.csv"), "w") as fh:
        fh.write("#timestamp [ns],filename\n")
        for k in range(0, len(t), 14):
            x0 = int(round(pad - FX * yaw[k]))
            cv2.imwrite(os.path.join(d, "mav0", "cam0", "data", "%d.png" % ns[k]), tex[:, x0:x0 + W])
            fh.write("%d,%d.png\n" % (ns[k], ns[k]))


def main():
    f = []
    tmp = tempfile.mkdtemp()
    try:
        d = os.path.join(tmp, "late")
        make(d, np.eye(3), 2.1)
        rep = IC.check(d, span=(26, 70), max_shift=5)
        for k, s in (("gyro vs ground truth", 2.1), ("gyro vs images", 2.1), ("ground truth vs images", 0.0)):
            x = rep["timing"][k]
            if abs(x["shift_s"] - s) > 0.05 or x["correlation"] < 0.9 or abs(x["scale"] - 1) > 0.15:
                f.append("late: %s %s, not %+.2f s" % (k, x, s))
        if rep["still"]["angle_deg"] > 0.5 or rep["rotation"]["deg"] > 3:
            f.append("late: gravity %.1f deg, rotation %.1f deg" % (rep["still"]["angle_deg"], rep["rotation"]["deg"]))
        if "+2.10 s" not in IC.report(rep):
            f.append("late: report\n" + IC.report(rep))
        d = os.path.join(tmp, "turned")
        make(d, RX, 0.0)
        rep = IC.check(d, span=(26, 70), max_shift=5, images=False)
        if abs(rep["still"]["angle_deg"] - 90) > 0.5 or abs(rep["rotation"]["deg"] - 90) > 3:
            f.append("turned: gravity %.1f deg, rotation %.1f deg" % (rep["still"]["angle_deg"],
                                                                      rep["rotation"]["deg"]))
    except Exception:                                              # noqa: BLE001
        import traceback
        f.append(traceback.format_exc())
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("\n".join(f) or "imu check ok")
    sys.exit(1 if f else 0)


if __name__ == "__main__":
    main()
