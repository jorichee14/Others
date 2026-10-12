#!/usr/bin/env python3
"""Does a datasets/euroc.py export's IMU agree with its ground truth? Checks
the IMU's link to the cameras (T_BS) and its timing before Kimera-VIO runs.

    python scoop_pipeline/analysis/slam/imu_check.py <export>/<robot> [--still 5 25] [--drive 26 60]

The ground truth (mav0/state_groundtruth_estimate0: the IMU in map, through
the same T_BS as the parameters) is compared with the IMU itself:
  still   (--still, seconds after the first stereo pair, the robot standing)
          the mean accelerometer reading against gravity turned into the
          IMU frame by the ground truth: the angle between them
  driving (--drive, the robot turning) the gyro against the ground truth's
          rotation over 0.5 s windows; then the rotation and the time shift
          of the gyro that fit them best. A right export fits with a few
          degrees and a few hundredths of a second; a wrong T_BS shows as a
          large rotation, a clock offset between IMU and camera as a shift.
"""
import argparse
import os

import numpy as np


def rot(q):
    """Rotation matrix of a quaternion (w, x, y, z)."""
    w, x, y, z = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def log(M):
    """Rotation vector of a rotation matrix."""
    a = np.arccos(np.clip((np.trace(M) - 1) / 2, -1, 1))
    w = np.array([M[2, 1] - M[1, 2], M[0, 2] - M[2, 0], M[1, 0] - M[0, 1]])
    return w * (0.5 if a < 1e-6 else a / (2 * np.sin(a)))


def fit(a, b):
    """The rotation Q with a ~ Q b (rows are vectors)."""
    U, _, Vt = np.linalg.svd(b.T @ a)
    return (U @ np.diag([1, 1, np.linalg.det(U @ Vt)]) @ Vt).T


def rms(x):
    return float(np.sqrt(np.mean(np.sum(x ** 2, 1)))) if len(x) else float("nan")


def check(robot_dir, still=(5.0, 25.0), drive=(26.0, 60.0), window=0.5):
    m0 = os.path.join(robot_dir, "mav0")
    c = np.loadtxt(os.path.join(m0, "cam0", "data.csv"), delimiter=",", comments="#", dtype=np.int64, usecols=0)
    u = np.loadtxt(os.path.join(m0, "imu0", "data.csv"), delimiter=",", comments="#", ndmin=2)
    g = np.loadtxt(os.path.join(m0, "state_groundtruth_estimate0", "data.csv"), delimiter=",", comments="#", ndmin=2)
    tu, tg = (u[:, 0] - c[0]) * 1e-9, (g[:, 0] - c[0]) * 1e-9
    rep = {}

    m = (tu > still[0]) & (tu < still[1])
    acc = u[m, 4:7].mean(0)
    up = rot(g[min(np.searchsorted(tg, np.mean(still)), len(g) - 1), 4:8]).T @ [0.0, 0.0, 9.81]
    rep["still"] = {"accelerometer": acc.tolist(), "gravity_by_ground_truth": up.tolist(),
                    "angle_deg": float(np.degrees(np.arccos(np.clip(
                        acc @ up / np.linalg.norm(acc) / np.linalg.norm(up), -1, 1))))}

    def windows(shift):
        a, b = [], []
        for t in np.arange(drive[0], drive[1], window):
            i, j = np.searchsorted(tg, [t, t + window])
            k = (tu >= t + shift) & (tu < t + window + shift)
            if j >= len(tg) or j <= i or k.sum() < 10:
                continue
            a.append(log(rot(g[i, 4:8]).T @ rot(g[j, 4:8])) / (tg[j] - tg[i]))
            b.append(u[k, 1:4].mean(0))
        return np.array(a).reshape(-1, 3), np.array(b).reshape(-1, 3)

    a, b = windows(0.0)
    Q = fit(a, b)

    def off(shift):
        a, b = windows(shift)
        return rms(a - b @ Q.T)

    best = min(np.arange(-0.5, 0.501, 0.02), key=off)
    a2, b2 = windows(best)
    Q = fit(a2, b2)
    big = np.argsort(-np.linalg.norm(a2, axis=1))[:3]
    rep["driving"] = {"windows": len(a), "ground_truth_rms": rms(a), "as_exported_rms": rms(a - b),
                      "shift_s": round(float(best), 2), "rotation_deg": float(np.degrees(np.linalg.norm(log(Q)))),
                      "rotation": Q.tolist(), "fitted_rms": rms(a2 - b2 @ Q.T),
                      "examples": [(a2[k].tolist(), b2[k].tolist()) for k in big]}
    return rep


def report(rep):
    s, d = rep["still"], rep["driving"]
    r = lambda v: "[%s]" % " ".join("%6.3f" % x for x in v)
    lines = ["still: accelerometer %s (|a| %.2f m/s^2), gravity by the ground truth %s: %.1f deg apart"
             % (r(s["accelerometer"]), np.linalg.norm(s["accelerometer"]), r(s["gravity_by_ground_truth"]),
                s["angle_deg"]),
             "driving (%d windows): the ground truth turns at rms %.3f rad/s; the gyro is off it by rms %.3f "
             "rad/s as exported" % (d["windows"], d["ground_truth_rms"], d["as_exported_rms"]),
             "best fit: the gyro rotated %.1f deg and shifted %+.2f s, then off by rms %.3f rad/s"
             % (d["rotation_deg"], d["shift_s"], d["fitted_rms"])]
    lines += ["    " + r(row) for row in d["rotation"]]
    lines += ["  ground truth %s   gyro %s" % (r(a), r(b)) for a, b in d["examples"]]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("robot_dir", help="<euroc export>/<robot> (with mav0/)")
    ap.add_argument("--still", nargs=2, type=float, default=(5.0, 25.0), metavar=("FROM", "TO"))
    ap.add_argument("--drive", nargs=2, type=float, default=(26.0, 60.0), metavar=("FROM", "TO"))
    a = ap.parse_args()
    print(report(check(a.robot_dir, a.still, a.drive)))


if __name__ == "__main__":
    main()
