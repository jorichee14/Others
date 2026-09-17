#!/usr/bin/env python3
"""How much of the run had usable depth, and where it did not.

The question this answers is not "is the depth good" but "for how long at a
stretch was it absent", because those two break a SLAM system differently. A
uniform 30% dropout is noise a tracker rides through; a four-second blackout is
where a trajectory leaves the room and never comes back.

On coop2 this is the FIRST thing to run. Reflective walls mean the ZED's stereo
depth drops out, and that single fact re-orders the method shortlist: every
entry that leans on depth inherits the dropout, while the monocular, inertial,
radar and infrastructure entries do not. Until the dropout is a number, that
re-ordering is a hunch.

    python3 scripts/depth_health.py --bag <bag> \
        --topic /mobile_1/zed/depth/depth_registered \
        --range 0.3 12.0 --out runs/depth_health_mobile_1.csv
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "..", "rosbag_to_opv2v"))


def decode_depth(msg) -> np.ndarray:
    """A sensor_msgs/Image depth frame -> float32 metres, NaN where invalid.

    The encoding decides a factor of 1000 and the run completes either way, so
    it is read off the message rather than configured.
    """
    enc = str(msg.encoding).lower()
    h, w = int(msg.height), int(msg.width)
    buf = bytes(msg.data)
    if enc in ("32fc1", "32f"):
        a = np.frombuffer(buf, dtype=np.float32).reshape(h, -1)[:, :w].astype(np.float64)
    elif enc in ("16uc1", "mono16"):
        a = np.frombuffer(buf, dtype=np.uint16).reshape(h, -1)[:, :w].astype(np.float64) * 1e-3
    else:
        raise SystemExit(f"unhandled depth encoding {msg.encoding!r}. "
                         f"Add it here rather than assuming a scale.")
    a = a.copy()
    a[a == 0.0] = np.nan            # 0 is "no return", not "at the camera"
    return a


def runs_of(mask: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous [start, end) index ranges where `mask` is True."""
    out, i, n = [], 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            out.append((i, j))
            i = j
        else:
            i += 1
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bag", required=True)
    ap.add_argument("--topic", required=True)
    ap.add_argument("--range", nargs=2, type=float, default=[0.3, 12.0],
                    metavar=("MIN", "MAX"),
                    help="the sensor's usable range. A return outside it is not a "
                         "measurement, whatever the driver put in the pixel.")
    ap.add_argument("--starved-below", type=float, default=0.15,
                    help="valid fraction under which a frame counts as starved "
                         "(default 0.15). Report the threshold with the number: it "
                         "is a choice and it moves the answer.")
    ap.add_argument("--out", default=None, help="per-frame CSV")
    args = ap.parse_args()

    from ros2opv2v.bagreader import BagReader

    lo, hi = args.range
    reader = BagReader(args.bag, stamp_source="header")
    stamps, frac, med = [], [], []

    for _, stamp_ns, msg in reader.iter_messages([args.topic]):
        d = decode_depth(msg)
        good = np.isfinite(d) & (d >= lo) & (d <= hi)
        stamps.append(stamp_ns * 1e-9)
        frac.append(float(good.mean()))
        med.append(float(np.median(d[good])) if good.any() else float("nan"))

    if not stamps:
        raise SystemExit(f"no messages on {args.topic}")

    t = np.asarray(stamps); f = np.asarray(frac); m = np.asarray(med)
    t0 = t - t[0]
    starved = f < args.starved_below
    blocks = runs_of(starved)
    dur = [(t0[b] - t0[a], t0[a]) for a, b in
           [(a, min(b, len(t) - 1)) for a, b in blocks]]
    dur.sort(reverse=True)

    print(f"\n{args.topic}")
    print(f"  frames                {len(t)}  over {t0[-1]:.1f} s "
          f"({len(t)/max(t0[-1], 1e-9):.1f} Hz)")
    print(f"  valid-depth fraction  median {np.median(f):6.1%}   "
          f"p10 {np.percentile(f, 10):6.1%}   p90 {np.percentile(f, 90):6.1%}")
    print(f"  median valid range    {np.nanmedian(m):.2f} m  "
          f"(sensor limit {lo}-{hi} m)")
    for thr in (0.50, 0.30, args.starved_below):
        print(f"  frames below {thr:5.0%}     {np.mean(f < thr):6.1%}")

    print(f"\n  STARVED STRETCHES (valid < {args.starved_below:.0%}) — "
          f"the number that predicts a tracking failure")
    print(f"    count               {len(blocks)}")
    if dur:
        tot = sum(d for d, _ in dur)
        print(f"    total               {tot:.1f} s  ({tot/t0[-1]:.1%} of the run)")
        print(f"    longest             {dur[0][0]:.2f} s  starting at t+{dur[0][1]:.1f} s")
        print(f"    top 5:              " + ", ".join(
            f"{d:.2f}s@{s:.1f}" for d, s in dur[:5]))
    else:
        print("    none — the dropout is uniform, not episodic")

    print(f"\n  READ IT LIKE THIS")
    print(f"    a uniform dropout   every depth-dependent method degrades a little")
    print(f"    long stretches      those are where the trajectory left the room;")
    print(f"                        cross them against where the ZED odometry went")
    print(f"                        wrong before blaming any estimator")
    print(f"    if starved > ~20%   the RGB-D arm is structurally handicapped on")
    print(f"                        this sequence, and the monocular, inertial,")
    print(f"                        radar and infrastructure rows are the plan")

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
        with open(args.out, "w") as fh:
            fh.write("t_rel_s,stamp_s,valid_fraction,median_valid_range_m,starved\n")
            for i in range(len(t)):
                fh.write(f"{t0[i]:.6f},{t[i]:.6f},{f[i]:.6f},{m[i]:.4f},{int(starved[i])}\n")
        print(f"\n  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
