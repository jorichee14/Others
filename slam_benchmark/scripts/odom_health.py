#!/usr/bin/env python3
"""Tracked fraction and where it broke, read off a run's console log.

`docs/PLAN.md` stage 1 gates on "at least one backbone tracks >= 90% of the run
on each platform", and rule 8 says a failure is a result with its number beside
it. Both need the same thing: not "it failed" but WHEN, for HOW LONG, and with
what warning on the way in.

RTAB-Map prints an `Odom: quality=N` line per frame, where N is the inlier count
and 0 means the frame was not registered at all. That series is the measurement.

    python3 scripts/odom_health.py --log /tmp/rtabmap-m1.log
"""
from __future__ import annotations

import argparse
import re
import sys

import numpy as np

QUALITY = re.compile(r"Odom:\s*quality=(\d+)")
# The warnings worth counting, because each one names a different cause and a
# run usually has only one of them in quantity.
CAUSES = {
    "lost: too few inliers": re.compile(r"Not enough inliers"),
    "lost: no correspondences": re.compile(r"Missing correspondences"),
    "lost: guess projected outside the image": re.compile(r"All projected points are outside"),
    "tf: imu unavailable at msg time": re.compile(r"Could not transform IMU msg"),
    "tf: extrapolation into the future": re.compile(r"require extrapolation into the future"),
    "tf: frame does not exist": re.compile(r"does not exist"),
    "sync: rgb/depth stamps far apart": re.compile(r"time difference between rgb and depth"),
    "imu dropped before init": re.compile(r"Dropping imu data"),
    # NOT the same thing as losing tracking. The frame never reached the
    # estimator at all, so it is missing from the denominator below rather than
    # failing in the numerator -- and a tracked fraction that quietly shrinks its
    # own denominator is the most flattering error a benchmark can make.
    "image dropped: no newer IMU sample": re.compile(
        r"We didn't receive IMU newer than previous image"),
}

# Not a warning -- a DISCONTINUITY. Each one resumes from the last good pose with
# large covariance, so the trajectory continues but the motion across the reset
# was never observed. The count belongs beside the ATE, not buried in a log.
RESET = re.compile(r"Odometry automatically reset")


def runs_of_zero(q: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous (start, length) stretches of quality 0.

    Stretches, not a count, because they break a tracker differently: scattered
    zeros are frames a system rides through, while one long stretch is where the
    trajectory leaves the room. Same reasoning as depth_health.py's starved
    stretches, applied to the estimator instead of the sensor.
    """
    out: list[tuple[int, int]] = []
    start = None
    for i, v in enumerate(q):
        if v == 0 and start is None:
            start = i
        elif v != 0 and start is not None:
            out.append((start, i - start))
            start = None
    if start is not None:
        out.append((start, len(q) - start))
    return out


def main() -> int:                                             # pragma: no cover
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", required=True)
    ap.add_argument("--rate", type=float, default=14.9,
                    help="frames per second, to turn frame indices into seconds")
    ap.add_argument("--expected-frames", type=int, default=None,
                    help="how many frames the STREAM carries (bag_probe prints it). "
                         "Without it the tracked fraction is of frames the estimator "
                         "processed, which is not the same as the fraction of the run.")
    args = ap.parse_args()

    text = open(args.log, errors="replace").read()
    q = np.array([int(m) for m in QUALITY.findall(text)])
    if len(q) == 0:
        print("no `Odom: quality=` lines in this log — either the method is not "
              "RTAB-Map or odometry never started", file=sys.stderr)
        return 2

    tracked = int(np.sum(q > 0))
    print(f"\n{len(q)} odometry frames, {tracked} tracked "
          f"({100.0 * tracked / len(q):.1f}% of frames PROCESSED)")
    if args.expected_frames:
        print(f"  of the whole run           {tracked}/{args.expected_frames} "
              f"({100.0 * tracked / args.expected_frames:.1f}%) — "
              f"{args.expected_frames - len(q)} frames never reached the estimator")
    if tracked:
        good = q[q > 0]
        print(f"  inliers while tracking   median {int(np.median(good))}, "
              f"p05 {int(np.percentile(good, 5))}, min {int(good.min())}")

    # How it did BEFORE anything went wrong, which a whole-run fraction hides:
    # a run that is perfect for 12 s and then dead reads the same as one that is
    # mediocre throughout.
    zeros = runs_of_zero(q)
    first_loss = min((s for s, _ in zeros if s > 0), default=len(q))
    if first_loss < len(q):
        head = q[:first_loss]
        print(f"  before the first loss      {first_loss} frames "
              f"({first_loss / args.rate:.1f} s), "
              f"{100.0 * np.sum(head > 0) / max(len(head), 1):.1f}% tracked")

    resets = len(RESET.findall(text))
    print(f"  automatic resets           {resets}"
          + ("   — each is an unobserved jump; report the count beside the ATE"
             if resets else ""))

    stretches = sorted(zeros, key=lambda s: -s[1])
    print(f"  lost-tracking stretches    {len(stretches)}")
    for start, length in stretches[:5]:
        print(f"      from frame {start:>5} ({start / args.rate:7.1f} s)  "
              f"{length:>5} frames ({length / args.rate:6.1f} s)"
              f"{'   — never recovered' if start + length == len(q) else ''}")

    print("\n  warnings seen")
    for name, pat in CAUSES.items():
        n = len(pat.findall(text))
        if n:
            print(f"      {name:<42} {n}")

    # The gate, stated rather than implied.
    # The gate is on the RUN, so it uses the run's frame count when one is known.
    # Scoring against frames the estimator happened to accept would let a method
    # pass by dropping the hard ones.
    denominator = args.expected_frames or len(q)
    frac = tracked / denominator
    if not args.expected_frames:
        print("\n  NOTE: --expected-frames not given, so the gate below is over "
              "PROCESSED frames. A method that drops frames before the estimator "
              "sees them scores better on that than it deserves.")
    if frac >= 0.90:
        print(f"\n  GATE: PASS — {frac * 100:.1f}% tracked, at or above the 90% "
              f"stage-1 bar.")
    else:
        worst = stretches[0] if stretches else (0, 0)
        print(f"\n  GATE: FAIL — {frac * 100:.1f}% tracked. The longest gap starts at "
              f"{worst[0] / args.rate:.1f} s and runs {worst[1] / args.rate:.1f} s. "
              f"Per rule 8 that is a result: report it with the number, and read the "
              f"warning counts above for which cause dominates before changing anything.")
    return 0


if __name__ == "__main__":                                     # pragma: no cover
    raise SystemExit(main())
