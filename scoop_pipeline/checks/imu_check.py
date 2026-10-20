#!/usr/bin/env python3
"""Does a datasets/euroc.py export's IMU agree with its images and ground
truth? Checks the IMU's timing and its link to the cameras (T_BS) before
Kimera-VIO runs.

    python3 check.py euroc <export>/<robot> [--still 5 25] [--span 26 200]
        [--max-shift 10]

Times in seconds after the first stereo pair. Three measures of how fast the
left camera turns about its own vertical (y) axis, over --span:
  images        the left images' horizontal shift between frames (phase
                correlation, OpenCV) over the focal length
  gyro          the IMU's angular rate turned into the camera by T_BS
                (params/LeftCameraParams.yaml); against the ground truth
                its size (in line whether T_BS is right or not)
  ground truth  mav0/state_groundtruth_estimate0's rotation (the IMU in map,
                through the same T_BS)
Each pair is lined up by the time shift (up to --max-shift) that correlates
them best: "gyro +2.10 s" means the gyro's stamps run 2.10 s behind the
images'. Kimera-VIO needs the gyro on the images' clock: shift about 0.
Then:
  still   (--still, the robot standing) the mean accelerometer reading
          against gravity turned into the IMU frame by the ground truth
  rotation the rotation that best maps the gyro onto the ground truth's
          rate, at their shift: a few degrees when T_BS is right
"""
import argparse
import os
import re

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


def camera_params(path):
    """(R of T_BS: the camera's axes in the IMU frame, fx) of a Kimera-VIO camera yaml."""
    txt = open(path).read()
    T = np.array([float(x) for x in re.search(r"data:\s*\[([^\]]*)\]", txt).group(1).split(",")]).reshape(4, 4)
    fx = float(re.search(r"intrinsics:\s*\[([^,\]]*)", txt).group(1))
    return T[:3, :3], fx


def image_rate(cam_dir, fx, t0, span):
    """(times s, rad/s about the camera's y axis) from the left images' shift between frames."""
    import cv2
    rows = [ln.strip().split(",") for ln in open(os.path.join(cam_dir, "data.csv")) if not ln.startswith("#")]
    rows = [(int(r[0]), r[1]) for r in rows if len(r) >= 2]
    rows = [(t, f) for t, f in rows if span[0] - 1 <= (t - t0) * 1e-9 <= span[1] + 1]
    ts, rate, prev, win, ds = [], [], None, None, 1
    for t, f in rows:
        im = cv2.imread(os.path.join(cam_dir, "data", f), cv2.IMREAD_GRAYSCALE)
        if im is None:
            continue
        if win is None:
            ds = max(1, im.shape[1] // 240)
        im = np.float32(cv2.resize(im, (im.shape[1] // ds, im.shape[0] // ds), interpolation=cv2.INTER_AREA))
        if win is None:
            win = cv2.createHanningWindow(im.shape[::-1], cv2.CV_32F)
        if prev is not None:
            (dx, _), _ = cv2.phaseCorrelate(prev[1], im, win)
            dt = (t - prev[0]) * 1e-9
            # a turn by +a about y (down) moves the scene left by fx * a
            ts.append(((t + prev[0]) / 2 - t0) * 1e-9)
            rate.append(-dx * ds / fx / dt)
        prev = (t, im)
    return np.array(ts), np.array(rate)


def smooth(T, t, x, width=0.5):
    """x(t) on the grid T, averaged over `width` seconds."""
    y = np.interp(T, t, x)
    n = max(1, int(round(width / (T[1] - T[0]))))
    return np.convolve(y, np.ones(n) / n, "same")


def line_up(T, a, tb, b, max_shift):
    """(shift s, correlation, scale): b's stamps run `shift` behind a's."""
    def at(s):
        return np.interp(T + s, tb, b)

    def corr(s):
        x = at(s)
        return np.corrcoef(a, x)[0, 1] if x.std() > 0 and a.std() > 0 else -1.0

    step = tb[1] - tb[0]
    s = max(np.arange(-max_shift, max_shift + step / 2, step), key=corr)
    x = at(s)
    return float(s), float(corr(s)), float((a @ x) / (a @ a)) if a @ a > 0 else float("nan")


def check(robot_dir, still=(5.0, 25.0), span=(26.0, 200.0), max_shift=10.0, images=True):
    m0 = os.path.join(robot_dir, "mav0")
    c = np.loadtxt(os.path.join(m0, "cam0", "data.csv"), delimiter=",", comments="#", dtype=np.int64, usecols=0)
    u = np.loadtxt(os.path.join(m0, "imu0", "data.csv"), delimiter=",", comments="#", ndmin=2)
    g = np.loadtxt(os.path.join(m0, "state_groundtruth_estimate0", "data.csv"), delimiter=",", comments="#", ndmin=2)
    R_bs, fx = camera_params(os.path.join(robot_dir, "params", "LeftCameraParams.yaml"))
    tu, tg = (u[:, 0] - c[0]) * 1e-9, (g[:, 0] - c[0]) * 1e-9
    rep = {"span": list(span)}

    m = (tu > still[0]) & (tu < still[1])
    acc = u[m, 4:7].mean(0)
    up = rot(g[min(np.searchsorted(tg, np.mean(still)), len(g) - 1), 4:8]).T @ [0.0, 0.0, 9.81]
    rep["still"] = {"accelerometer": acc.tolist(), "gravity_by_ground_truth": up.tolist(),
                    "angle_deg": float(np.degrees(np.arccos(np.clip(
                        acc @ up / np.linalg.norm(acc) / np.linalg.norm(up), -1, 1))))}

    # angular rate in the IMU frame on a fine grid: gyro and ground truth (+-0.1 s differences)
    dt = 0.02
    lo, hi = max(tu[0], tg[0]) + 0.2, min(tu[-1], tg[-1]) - 0.2
    G = np.arange(lo, hi, dt)
    w_gyro = np.stack([smooth(G, tu, u[:, 1 + k]) for k in range(3)], 1)
    i0 = np.clip(np.searchsorted(tg, G - 0.1), 0, len(tg) - 1)
    i1 = np.clip(np.searchsorted(tg, G + 0.1), 0, len(tg) - 1)
    w_gt = np.array([log(rot(g[a, 4:8]).T @ rot(g[b, 4:8])) / max(tg[b] - tg[a], 1e-3) for a, b in zip(i0, i1)])
    w_gt = np.stack([smooth(G, G, w_gt[:, k]) for k in range(3)], 1)
    # about the camera's y axis
    y_gyro, y_gt = (w_gyro @ R_bs)[:, 1], (w_gt @ R_bs)[:, 1]
    T = np.arange(span[0], span[1], dt)
    T = T[(T - max_shift >= lo) & (T + max_shift <= hi)]
    gt_T = np.interp(T, G, y_gt)
    # gyro against ground truth by the rate's size: in line even when T_BS is wrong
    pairs = {"gyro vs ground truth": line_up(T, np.interp(T, G, np.linalg.norm(w_gt, axis=1)), G,
                                             np.linalg.norm(w_gyro, axis=1), max_shift)}
    if images:
        try:
            ti, ri = image_rate(os.path.join(m0, "cam0"), fx, c[0], span)
            im_T = smooth(T, ti, ri)
            pairs["gyro vs images"] = line_up(T, im_T, G, y_gyro, max_shift)
            pairs["ground truth vs images"] = line_up(T, im_T, G, y_gt, max_shift)
        except ImportError:
            rep["images"] = "no OpenCV (cv2): run in the scoop environment for the images"
    rep["timing"] = {k: {"shift_s": round(s, 2), "correlation": cc, "scale": sc} for k, (s, cc, sc) in pairs.items()}

    # T_BS: the rotation that maps the gyro onto the ground truth, the gyro lined up
    s = pairs["gyro vs ground truth"][0]
    W = np.stack([np.interp(G + s, G, w_gyro[:, k]) for k in range(3)], 1)
    keep = (G >= span[0]) & (G <= span[1]) & (G + s <= hi)
    Q = fit(w_gt[keep], W[keep])
    rep["rotation"] = {"deg": float(np.degrees(np.linalg.norm(log(Q)))), "matrix": Q.tolist()}
    return rep


def report(rep):
    s = rep["still"]
    r = lambda v: "[%s]" % " ".join("%6.3f" % x for x in v)
    lines = ["timing, %g-%g s, turn rate about the camera's vertical axis (+ = the second runs behind):"
             % tuple(rep["span"])]
    for k, d in rep["timing"].items():
        lines.append("  %-24s %+6.2f s  correlation %.2f  scale %.2f" % (k, d["shift_s"], d["correlation"], d["scale"]))
    if "images" in rep:
        lines.append("  " + rep["images"])
    lines.append("still: accelerometer %s (|a| %.2f m/s^2), gravity by the ground truth %s: %.1f deg apart"
                 % (r(s["accelerometer"]), np.linalg.norm(s["accelerometer"]), r(s["gravity_by_ground_truth"]),
                    s["angle_deg"]))
    lines.append("rotation fitting the gyro to the ground truth (lined up): %.1f deg" % rep["rotation"]["deg"])
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("robot_dir", help="<euroc export>/<robot> (with mav0/ and params/)")
    ap.add_argument("--still", nargs=2, type=float, default=(5.0, 25.0), metavar=("FROM", "TO"))
    ap.add_argument("--span", nargs=2, type=float, default=(26.0, 200.0), metavar=("FROM", "TO"))
    ap.add_argument("--max-shift", type=float, default=10.0)
    a = ap.parse_args()
    print(report(check(a.robot_dir, a.still, a.span, a.max_shift)))


if __name__ == "__main__":
    main()
