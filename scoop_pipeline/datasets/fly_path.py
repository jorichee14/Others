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


def prune_excursions(points: np.ndarray, radius: float, max_loop: float) -> np.ndarray:
    """Cut out the short excursions: the little loops, spurs and shuffles.

    A mapping run around a building is one big ring with small circles at the
    corners, a spur into a doorway, a shuffle while the operator looks at
    something. Those are the loops to remove. The ring is not: it is the shape
    of the place.

    So the test is length, not shape. Where the walk comes back within `radius`
    of a point it left less than `max_loop` of path ago, the excursion between
    is dropped. The cut joins two points `radius` apart, so it cannot cross
    anything the operator did not. The ring survives because returning to its
    own start takes far more than `max_loop` of walking.
    """
    if len(points) < 3:
        return points
    step = max(1e-6, _median_step(points))
    ahead = max(3, int(round(max_loop / step)))
    kept, index, total = [], 0, len(points)
    while index < total:
        kept.append(points[index])
        high = min(total, index + ahead + 1)
        if high > index + 2:
            near = np.nonzero(
                np.linalg.norm(points[index + 2:high] - points[index], axis=1) < radius)[0]
            if len(near):
                index = index + 2 + int(near[-1])     # on past the whole excursion
                continue
        index += 1
    return np.asarray(kept)


def first_lap(points: np.ndarray, close: float, min_length: float) -> np.ndarray:
    """One time round: up to where the walk first comes back to where it began.

    After the excursions are gone what is left is the ring, walked once or
    several times. A flythrough wants it once.
    """
    if len(points) < 4:
        return points
    along = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    far = along >= min_length
    back = np.linalg.norm(points - points[0], axis=1) < close
    hit = np.nonzero(far & back)[0]
    return points[:hit[0] + 1] if len(hit) else points


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


def draw(raw, lap, way, path: str, args) -> None:
    """The walk, and the lap taken out of it, side by side.

    A lap that comes out far shorter than the building is the usual outcome of
    a `--revisit` radius wider than the corridors: a circuit of a floor passes
    near itself at every junction, so nothing large qualifies as simple. That is
    obvious in a picture and invisible in a length.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(13, 5.4), facecolor="#0F1419")
    for axis, (points, title) in zip(axes, [
            (raw, f"the mapping run — {len(raw)} poses"),
            (way, f"the lap — {len(way)} waypoints, "
                  f"revisit {args.revisit} m, close {args.close} m")]):
        axis.set_facecolor("#0F1419")
        axis.plot(raw[:, 0], raw[:, 1], color="#39464F", lw=0.6, zorder=1)
        if points is not raw:
            axis.plot(points[:, 0], points[:, 1], color="#F2994A", lw=1.8, zorder=3)
            axis.scatter(*points[0], c="#A3E635", s=45, zorder=5)
        axis.set_title(title, color="#E6E9EC", fontsize=10, loc="left")
        axis.set_aspect("equal")
        axis.tick_params(colors="#8A949E", labelsize=7)
        axis.set_xlabel("x (m)", color="#8A949E", fontsize=8)
        axis.set_ylabel("y (m)", color="#8A949E", fontsize=8)
        for spine in axis.spines.values():
            spine.set_color("#2A3137")
    figure.tight_layout()
    figure.savefig(path, dpi=110, facecolor="#0F1419")
    plt.close(figure)
    print(f"wrote {path}", file=sys.stderr)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("traj", help="the mapping trajectory (TUM), in the mesh's frame")
    parser.add_argument("--out", default=None, help="write the waypoints here (default: stdout)")
    parser.add_argument("--spacing", type=float, default=0.4,
                        help="resample the trajectory to this spacing (m)")
    parser.add_argument("--revisit", type=float, default=1.5,
                        help="coming back within this of an earlier point closes an excursion (m)")
    parser.add_argument("--max-loop", type=float, default=25.0,
                        help="an excursion shorter than this is cut out; the ring, which takes "
                             "far longer to come back to itself, is kept (m)")
    parser.add_argument("--close", type=float, default=4.0,
                        help="the lap ends within this of where it started (m)")
    parser.add_argument("--smooth", type=float, default=2.0,
                        help="average the way over this much of its length (m; 0: not)")
    parser.add_argument("--plot", default=None, metavar="PNG",
                        help="draw the walk and the lap chosen out of it, to pick the "
                             "settings by eye instead of by guessing")
    parser.add_argument("--keep-ends", action="store_true",
                        help="do not treat the circuit as closed when smoothing")
    parser.add_argument("--min-length", type=float, default=15.0,
                        help="the lap is at least this long (m), and a shorter result is refused")
    args = parser.parse_args()

    raw = read_tum_xy(args.traj)
    even = resample(raw, args.spacing)
    pruned = prune_excursions(even, args.revisit, args.max_loop)
    once = first_lap(pruned, args.close, args.min_length)
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
          f"  excursions under {args.max_loop} m cut out -> {len(pruned)} points, "
          f"{length(pruned):.1f} m\n"
          f"  one time round (closing within {args.close} m) -> {len(once)} points, "
          f"{length(once):.1f} m\n"
          f"  smoothed over {args.smooth} m -> {len(way)} waypoints, {length(way):.1f} m circuit",
          file=sys.stderr)
    if length(way) < args.min_length:
        print(f"the circuit is {length(way):.1f} m: the revisit radius has collapsed it. "
              f"Lower --revisit.", file=sys.stderr)
        return 1

    if args.plot:
        draw(raw, once, way, os.path.expanduser(args.plot), args)

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
