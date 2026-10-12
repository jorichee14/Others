#!/usr/bin/env python3
"""Are a robot's camera and LiDAR on the same clock? Read straight from the
recorded (merged) bag: no pipeline output, no calibration beyond fx.

    python scoop_pipeline/data_quality_validation/camera_lidar_sync.py <bag> [--robot mobile_1]
        [--image TOPIC --info TOPIC --lidar TOPIC] [--span FROM TO] [--segment 100]
        [--tol 0.05] [--max-shift 5] [--plot out.png]

How fast the robot turns (about the vertical), measured twice:
  camera  the image's horizontal shift between frames (phase correlation of
          the downsized grey images) over fx: a turn left by a moves the
          scene right by fx * a
  LiDAR   the scan's shift in azimuth between scans (phase correlation of the
          mean range per 0.5 deg of azimuth, points within 0.3-30 m): a turn
          left by a moves every point's azimuth by -a
Both assume the camera's y axis and the LiDAR's z axis about vertical (a
level mount). Times are header stamps, in seconds after the first image.

The two are lined up per --segment seconds by the time shift (up to
--max-shift) that correlates them best; "+0.30 s" means the LiDAR's stamps
run 0.30 s behind the images'. A segment counts when the robot turns in it
(rms above 0.05 rad/s) and the two correlate (above 0.5). PASS when every
counted segment is within --tol of 0 (and at least one counts), else FAIL
(exit 1). The scale (LiDAR over camera) sits near 1, lower for the camera
where driving forward adds image motion.
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "analysis", "slam"))
from imu_check import line_up, smooth                              # noqa: E402

ROBOTS = {"mobile_1": {"image": "/mobile_1/zed/left/image_rect_color", "info": "/mobile_1/zed/left/camera_info",
                       "lidar": "/mobile_1/ouster/points"}}
BINS = 720                                     # 0.5 deg of azimuth
MIN_TURN = 0.05                                # rad/s rms: a segment where the robot turns
MIN_CORR = 0.5


def first_msg(reader, topic):
    for _, _, payload, decode in reader.iter_raw([topic]):
        return decode(payload)
    return None


def shift_1d(a, b):
    """s with b(x) ~ a(x - s), circular, in bins (sub-bin by a parabola)."""
    F = np.conj(np.fft.rfft(a)) * np.fft.rfft(b)
    c = np.fft.irfft(F / (np.abs(F) + 1e-9), len(a))
    k = int(np.argmax(c))
    y0, y1, y2 = c[k - 1], c[k], c[(k + 1) % len(c)]
    d = 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2) if (y0 - 2 * y1 + y2) != 0 else 0.0
    s = k + d
    return s - len(a) if s > len(a) / 2 else s


def lidar_profile(xyz):
    """Mean range per azimuth bin, less its mean (0 where no point)."""
    r = np.hypot(xyz[:, 0], xyz[:, 1])
    k = (r > 0.3) & (r < 30.0)
    az = np.arctan2(xyz[k, 1], xyz[k, 0])
    b = ((az + np.pi) / (2 * np.pi) * BINS).astype(int) % BINS
    n = np.bincount(b, minlength=BINS)
    s = np.bincount(b, weights=r[k], minlength=BINS)
    p = np.where(n > 0, s / np.maximum(n, 1), 0.0)
    return np.where(n > 0, p - p[n > 0].mean(), 0.0) if n.any() else p


def lidar_rate(reader, topic, t0, span):
    """(times s, rad/s about the vertical) from consecutive scans."""
    from scoop import bag as B
    lo, hi = t0 + int((span[0] - 1) * 1e9), t0 + int((span[1] + 1) * 1e9)
    ts, rate, prev = [], [], None
    for sc in B.iter_scans(reader, topic, select=lambda s: 0 if lo <= s <= hi else None, min_points=50,
                           log=lambda *_: None):
        p = lidar_profile(np.asarray(sc.xyz, np.float64))
        if prev is not None:
            dt = (sc.stamp_ns - prev[0]) * 1e-9
            if 0 < dt < 0.5:
                ts.append(((sc.stamp_ns + prev[0]) / 2 - t0) * 1e-9)
                rate.append(-shift_1d(prev[1], p) * 2 * np.pi / BINS / dt)
        prev = (sc.stamp_ns, p)
    return np.array(ts), np.array(rate)


def camera_rate(reader, topic, fx, t0, span):
    """(times s, rad/s about the vertical) from consecutive images."""
    import cv2
    from scoop import bag as B
    lo, hi = t0 + int((span[0] - 1) * 1e9), t0 + int((span[1] + 1) * 1e9)
    ts, rate, prev, win, ds = [], [], None, None, 1
    for fr in B.iter_images(reader, topic, select=lambda s: 0 if lo <= s <= hi else None, mode="gray"):
        im = fr.image
        if win is None:
            ds = max(1, im.shape[1] // 240)
        im = np.float32(cv2.resize(im, (im.shape[1] // ds, im.shape[0] // ds), interpolation=cv2.INTER_AREA))
        if win is None:
            win = cv2.createHanningWindow(im.shape[::-1], cv2.CV_32F)
        if prev is not None:
            dt = (fr.stamp_ns - prev[0]) * 1e-9
            if 0 < dt < 0.5:
                (dx, _), _ = cv2.phaseCorrelate(prev[1], im, win)
                ts.append(((fr.stamp_ns + prev[0]) / 2 - t0) * 1e-9)
                rate.append(dx * ds / fx / dt)
        prev = (fr.stamp_ns, im)
    return np.array(ts), np.array(rate)


def first_stamp(reader, topic):
    from scoop import bag as B
    for _, log_ns, payload, _ in reader.iter_raw([topic]):
        return B.stamp_from_cdr(payload) or log_ns
    return None


def check(bag, image, info, lidar, span=None, segment=100.0, tol=0.05, max_shift=5.0, plot=None):
    from scoop import bag as B
    reader = B.open_bag(bag)
    ci = first_msg(reader, info)
    if ci is None:
        raise SystemExit("no %s in %s" % (info, bag))
    fx = float(ci.p[0] or ci.k[0])
    t0 = first_stamp(reader, image)
    if t0 is None:
        raise SystemExit("no %s in %s" % (image, bag))
    span = span or (0.0, 1e9)
    tc, rc = camera_rate(reader, image, fx, t0, span)
    tl, rl = lidar_rate(reader, lidar, t0, span)
    if len(tc) < 10 or len(tl) < 10:
        raise SystemExit("too little data: %d camera and %d LiDAR rates" % (len(tc), len(tl)))
    dt = 0.02
    lo, hi = max(tc[0], tl[0]) + 0.5, min(tc[-1], tl[-1]) - 0.5
    G = np.arange(lo, hi, dt)
    cam, lid = smooth(G, tc, rc), smooth(G, tl, rl)
    rep = {"bag": os.path.abspath(bag), "image": image, "lidar": lidar, "fx": fx,
           "camera_rates": len(tc), "lidar_rates": len(tl), "from_s": float(lo), "to_s": float(hi), "segments": []}
    a = lo
    while a < hi - 2 * max_shift:
        b = min(a + segment, hi)
        T = np.arange(a, b, dt)
        T = T[(T - max_shift >= lo) & (T + max_shift <= hi)]
        if len(T) > 50:
            c_T = np.interp(T, G, cam)
            turn = float(np.sqrt(np.mean(c_T ** 2)))
            s, cc, sc = line_up(T, c_T, G, lid, max_shift)
            rep["segments"].append({"from_s": float(T[0]), "to_s": float(T[-1]), "turn_rms": turn,
                                    "shift_s": round(s, 2), "correlation": cc, "scale": sc,
                                    "counted": bool(turn > MIN_TURN and cc > MIN_CORR)})
        a = b
    counted = [x for x in rep["segments"] if x["counted"]]
    rep["pass"] = bool(counted) and all(abs(x["shift_s"]) <= tol for x in counted)
    rep["tol_s"] = tol
    if counted:
        rep["median_shift_s"] = float(np.median([x["shift_s"] for x in counted]))
    if plot:
        draw(G, cam, lid, rep, plot)
    return rep


def draw(G, cam, lid, rep, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 1, figsize=(11, 6), gridspec_kw={"height_ratios": [2, 1]})
    s = rep.get("median_shift_s", 0.0)
    ax[0].plot(G, cam, lw=0.8, label="camera")
    ax[0].plot(G, lid, lw=0.8, label="LiDAR")
    ax[0].plot(G - s, lid, lw=0.8, ls="--", label="LiDAR moved by %+.2f s" % -s)
    ax[0].set_ylabel("turn rate (rad/s)")
    ax[0].legend(fontsize=8)
    seg = rep["segments"]
    ax[1].scatter([(x["from_s"] + x["to_s"]) / 2 for x in seg], [x["shift_s"] for x in seg],
                  c=["C2" if x["counted"] else "0.7" for x in seg])
    ax[1].axhspan(-rep["tol_s"], rep["tol_s"], color="C2", alpha=0.15)
    ax[1].set_ylabel("LiDAR behind camera (s)")
    ax[1].set_xlabel("time after the first image (s)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def report(rep):
    lines = ["%s vs %s: %d camera and %d LiDAR rates, %.0f-%.0f s (fx %.1f)"
             % (rep["image"], rep["lidar"], rep["camera_rates"], rep["lidar_rates"], rep["from_s"], rep["to_s"],
                rep["fx"])]
    for x in rep["segments"]:
        lines.append("  %6.0f-%4.0f s  LiDAR %+6.2f s  correlation %.2f  scale %.2f  turning %.3f rad/s%s"
                     % (x["from_s"], x["to_s"], x["shift_s"], x["correlation"], x["scale"], x["turn_rms"],
                        "" if x["counted"] else "  (not counted)"))
    if "median_shift_s" in rep:
        lines.append("median shift %+.2f s over the counted segments" % rep["median_shift_s"])
    lines.append("%s: camera and LiDAR %s within %.2f s" % ("PASS" if rep["pass"] else "FAIL",
                                                          "in sync" if rep["pass"] else "NOT in sync", rep["tol_s"]))
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bag", help="a recorded or merged bag with the robot's images and LiDAR")
    ap.add_argument("--robot", default="mobile_1", choices=sorted(ROBOTS))
    ap.add_argument("--image", help="image topic (default: the robot's left camera)")
    ap.add_argument("--info", help="its camera_info topic")
    ap.add_argument("--lidar", help="PointCloud2 topic (default: the robot's Ouster)")
    ap.add_argument("--span", nargs=2, type=float, default=None, metavar=("FROM", "TO"))
    ap.add_argument("--segment", type=float, default=100.0, help="seconds per segment")
    ap.add_argument("--tol", type=float, default=0.05, help="largest shift that passes (s)")
    ap.add_argument("--max-shift", type=float, default=5.0)
    ap.add_argument("--plot", default=None, help="write the rates and per-segment shifts to this png")
    a = ap.parse_args()
    r = ROBOTS[a.robot]
    rep = check(a.bag, a.image or r["image"], a.info or r["info"], a.lidar or r["lidar"], a.span, a.segment,
                a.tol, a.max_shift, a.plot)
    print(report(rep))
    sys.exit(0 if rep["pass"] else 1)


if __name__ == "__main__":
    main()
