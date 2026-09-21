#!/usr/bin/env python3
"""Put the bag's clock back on a trajectory that was estimated from a folder.

MASt3R-SLAM's folder loader numbers its frames `index / 30.0` and writes those
numbers into the trajectory it saves. They are not times; the recording is
14.7 Hz, not 30. Left alone they would produce a trajectory that starts near
zero, associates against nothing, and -- if the evaluator ever fell back to
index association -- scored.

So the map is inverted exactly rather than approximately. Every timestamp in the
input must land on an integer index (to well within half a frame) and inside the
range of stamps that `bag_to_frames.py` recorded. Anything else means the loader
that actually ran was not the one assumed here, and that is a refusal: a
resynchronisation that "mostly works" is a time offset nobody will find later.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def restamp(lines: list[str], stamps: list[float], fps: float = 30.0) -> list[str]:
    out: list[str] = []
    for ln in lines:
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        parts = s.split()
        if len(parts) != 8:
            raise ValueError(f"expected 8 TUM columns, got {len(parts)}: {s!r}")
        exact = float(parts[0]) * fps
        idx = round(exact)
        if abs(exact - idx) > 0.05:
            raise ValueError(
                f"timestamp {parts[0]} is {exact:.4f} frames at {fps} Hz, which is not "
                f"an integer index. The loader that ran did not fabricate stamps the "
                f"way this script inverts, so the mapping would be a guess.")
        if not 0 <= idx < len(stamps):
            raise ValueError(f"frame index {idx} is outside the {len(stamps)} frames "
                             f"exported from the bag")
        out.append(" ".join([f"{stamps[idx]:.9f}"] + parts[1:]))
    if not out:
        raise ValueError("no pose rows in the input trajectory")
    return out


def main() -> int:                                           # pragma: no cover
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", required=True)
    ap.add_argument("--stamps", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=float, default=30.0)
    args = ap.parse_args()

    stamps = [float(x) for x in Path(args.stamps).read_text().split()]
    rows = restamp(Path(args.traj).read_text().splitlines(), stamps, args.fps)
    Path(args.out).write_text("# timestamp tx ty tz qx qy qz qw\n" + "\n".join(rows) + "\n")
    print(f"{len(rows)} keyframe poses restamped onto the bag clock "
          f"({stamps[0]:.3f} .. {stamps[-1]:.3f})")
    return 0


if __name__ == "__main__":                                   # pragma: no cover
    sys.exit(main())
