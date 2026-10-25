#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A mapping trajectory as one clean circuit, for `lod3dgs.py fly --path`.

`fly` makes its own loop, inset from the walls. That loop is geometric: it goes
where the floor is free, not where the robot could actually see. A mapping run
went where the operator could walk, looked at what was worth mapping, and is
already in the mesh's frame -- but it is the wrong shape for a flythrough. A
mapping run revisits: it goes up a corridor and back, circles a hall twice to
close the loop, pauses, shuffles. Flown as recorded, the camera doubles back on
itself and jerks at every pause.

So: resample the trajectory evenly, drop the revisits so each place is passed
once, and smooth what is left. The result is a single circuit in the order the
operator walked it, which a camera can follow without doubling back.

    python3 scoop_pipeline/datasets/fly_path.py <traj.tum> [--out path.txt]
        [--spacing 0.4] [--revisit 2.5] [--smooth 2.0] [--keep-ends]

Writes the waypoints as `X,Y;X,Y;...`, which is what `fly --path` reads:

    python3 scoop_pipeline/datasets/lod3dgs.py fly <mesh.ply> out.mp4 \\
        --path "$(cat path.txt)" --spin 0 --speed 0.8 --ahead 3

`--spin 0` matters: fly's default turns a full circle on the spot before
setting off, which is exactly the sudden movement a walkthrough should not have.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def read_tum_xy(path: str) -> np.ndarray:
    """The plan positions (N x 2) of a TUM trajectory, every pose."""
    rows = np.loadtxt(os.path.expanduser(path), comments="#", ndmin=2)
    if rows.shape[1] < 4 or not len(rows):
        raise SystemExit(f"{path}: not a TUM trajectory (t x y z qx qy qz qw)")
    return rows[:, 1:3]


def resample(points: np.ndarray, spacing: float, closed: bool = False) -> np.ndarray:
    """Evenly spaced along the path, so a pause stops being many points.

    A stationary stretch is thousands of poses in one spot. Left in, it
    dominates every later step -- the lap search sees a crowd, the smoother
    averages over nothing, and the flight sits still for a third of the video.

    `closed`: the path already returns to its start, and the spacing is divided
    into its length exactly rather than stepped from the beginning. Stepping
    leaves a remainder at the join, and a remainder is a stray short step --
    which reads as a turn of hundreds of degrees per metre however gentle the
    corner really is.
    """
    step = np.linalg.norm(np.diff(points, axis=0), axis=1)
    keep = step > 1e-9
    points = np.vstack([points[:1], points[1:][keep]])
    if len(points) < 2:
        return points
    along = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    if closed:
        count = max(3, int(round(along[-1] / float(spacing))))
        want = np.linspace(0.0, along[-1], count, endpoint=False)
    else:
        want = np.arange(0.0, along[-1], float(spacing))
    return np.c_[np.interp(want, along, points[:, 0]), np.interp(want, along, points[:, 1])]


def longest_lap(points: np.ndarray, radius: float, close: float,
                min_length: float) -> np.ndarray:
    """The longest stretch walked that comes back without passing anywhere twice.

    Skipping revisits in place looks right and is not: dropping a stretch leaves
    the kept points discontinuous, and smoothing renders that jump as a diagonal
    straight through a wall. A there-and-back spur has no non-doubling path
    through it at all, so half of it cannot be kept either.

    What can be kept is a sub-path of the real walk -- the operator was in a
    corridor the whole time, so anything they walked is walkable. This finds the
    longest contiguous one that ends within `close` of where it started and
    never comes within `radius` of its own earlier self. That is a lap: it
    closes without a seam, and `fly` joining its ends adds nothing.
    """
    n = len(points)
    if n < 8:
        return points
    step = max(1e-6, _median_step(points))
    skip = max(2, int(round(2.0 * radius / step)))          # the stretch just walked
    need = max(skip + 2, int(round(min_length / step)))
    distance = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)

    best = None
    for start in range(n - need):
        end = None
        for here in range(start + skip + 1, n):
            if distance[here, start:here - skip].min() < radius:
                break                                        # passed its own path
            if here - start >= need and distance[start, here] < close:
                end = here                                   # a lap, so far the longest
        if end is not None and (best is None or end - start > best[1] - best[0]):
            best = (start, end)
    if best is None:
        return points[:0]
    return points[best[0]:best[1] + 1]


def _median_step(points: np.ndarray) -> float:
    return float(np.median(np.linalg.norm(np.diff(points, axis=0), axis=1)))


def smooth(points: np.ndarray, metres: float, closed: bool = True) -> np.ndarray:
    """A moving average over `metres` of path, so the camera turns, never jerks.

    The ends are joined when `closed`: the circuit's last point runs into its
    first, and a seam there is as visible as any other jerk.
    """
    if metres <= 0 or len(points) < 3:
        return points
    half = max(1, int(round(metres / max(1e-6, _median_step(points)) / 2)))
    window = 2 * half + 1
    if window >= len(points):
        return points
    if closed:
        padded = np.vstack([points[-half:], points, points[:half]])
    else:
        padded = np.vstack([np.repeat(points[:1], half, axis=0), points,
                            np.repeat(points[-1:], half, axis=0)])
    kernel = np.ones(window) / window
    return np.c_[np.convolve(padded[:, 0], kernel, mode="valid"),
                 np.convolve(padded[:, 1], kernel, mode="valid")]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("traj", help="the mapping trajectory (TUM), in the mesh's frame")
    parser.add_argument("--out", default=None, help="write the waypoints here (default: stdout)")
    parser.add_argument("--spacing", type=float, default=0.4,
                        help="resample the trajectory to this spacing (m)")
    parser.add_argument("--revisit", type=float, default=2.5,
                        help="the lap may not come within this of its own earlier self (m)")
    parser.add_argument("--close", type=float, default=3.0,
                        help="the lap ends within this of where it started (m)")
    parser.add_argument("--smooth", type=float, default=2.0,
                        help="average the way over this much of its length (m; 0: not)")
    parser.add_argument("--keep-ends", action="store_true",
                        help="do not treat the circuit as closed when smoothing")
    parser.add_argument("--min-length", type=float, default=15.0,
                        help="the lap is at least this long (m), and a shorter result is refused")
    args = parser.parse_args()

    raw = read_tum_xy(args.traj)
    even = resample(raw, args.spacing)
    once = longest_lap(even, args.revisit, args.close, args.min_length)
    if not len(once):
        print(f"no lap found: nothing the operator walked came back within "
              f"{args.close} m of its start without crossing itself ({args.revisit} m). "
              f"Raise --close, lower --revisit, or lower --min-length.", file=sys.stderr)
        return 1
    # Resampled again after smoothing, and over the CLOSED loop: `fly` joins the
    # last waypoint back to the first, so that segment is part of the way and has
    # to be spaced like the rest. Left as a remainder it is a stray 3 cm or 90 cm
    # step, and a short step between two waypoints reads as a turn of hundreds of
    # degrees per metre however gentle the corner really is. Even spacing also
    # means the camera holds one speed.
    eased = smooth(once, args.smooth, closed=not args.keep_ends)
    way = resample(np.vstack([eased, eased[:1]]), args.spacing, closed=True)

    def length(points):
        return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())

    print(f"{len(raw)} poses, {length(raw):.1f} m walked\n"
          f"  resampled at {args.spacing} m  -> {len(even)} points\n"
          f"  longest lap (clear of itself by {args.revisit} m, closing within "
          f"{args.close} m) -> {len(once)} points, {length(once):.1f} m\n"
          f"  smoothed over {args.smooth} m -> {len(way)} waypoints, {length(way):.1f} m circuit",
          file=sys.stderr)
    if length(way) < args.min_length:
        print(f"the circuit is {length(way):.1f} m: the revisit radius has collapsed it. "
              f"Lower --revisit.", file=sys.stderr)
        return 1

    text = ";".join(f"{x:.2f},{y:.2f}" for x, y in way)
    if args.out:
        with open(os.path.expanduser(args.out), "w", encoding="utf-8") as handle:
            handle.write(text + "\n")
        print(f"wrote {args.out}", file=sys.stderr)
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
