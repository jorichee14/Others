#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The flythrough seen from above: where the camera is, and what it is facing.

A flythrough shows what the place looks like and nothing about where you are
in it. Fifteen seconds down a corridor could be any corridor, and a viewer who
loses the thread cannot get it back. This draws the same flight in plan -- the
route faint, the part already flown solid, the camera and the cone it is
looking down -- so the two read together.

    python3 scoop_pipeline/datasets/fly_overhead.py <fly_path.json> [out.mp4]
        [--cloud map.ply] [--reach 6] [--stride 8] [--every 1]

    python3 scoop_pipeline/datasets/fly_overhead.py --path lap.txt out.mp4 \
        --cloud map.ply [--speed 0.8] [--fps 30] [--ahead 3]
        [--write-json lap_path.json --eye-z 1.1]     # for splat_walk --fly

The json is the one `lod3dgs.py fly` writes beside its video
(`<out>_path.json`): the camera's position and look-at point for every frame
it rendered, at the fps it rendered them. Taking the motion from there rather
than re-walking the waypoints is the point -- a second implementation of the
same path would drift out of step with the video it is meant to accompany,
and the drift would be invisible until someone watched them together.

With the default --every 1 the frames correspond one to one, so the two play
in sync and can be put side by side:

    ffmpeg -i fly.mp4 -i fly_overhead.mp4 -filter_complex hstack both.mp4
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                  # noqa: E402
import numpy as np                                               # noqa: E402
from matplotlib.colors import LinearSegmentedColormap            # noqa: E402
from matplotlib.patches import Polygon                           # noqa: E402


def read_xy(path: str, stride: int):
    """The plan positions of a mesh's vertices or a cloud's points."""
    try:
        import open3d as o3d
    except ImportError:                                          # pragma: no cover
        raise SystemExit("this needs open3d (the environment lod3dgs.py runs in)")
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
    path = os.path.expanduser(path)
    mesh = o3d.io.read_triangle_mesh(path)
    points = (np.asarray(mesh.vertices) if len(mesh.triangles)
              else np.asarray(o3d.io.read_point_cloud(path).points))
    if not len(points):
        raise SystemExit(f"{path}: no vertices and no points")
    return points[::max(1, stride), :2]


def read_waypoints(path: str) -> np.ndarray:
    """The `X,Y;X,Y;...` lap fly_path.py writes, as (n, 2)."""
    text = open(os.path.expanduser(path), encoding="utf-8").read().strip()
    pairs = [p for p in text.replace("\n", ";").split(";") if p.strip()]
    return np.array([[float(v) for v in pair.split(",")[:2]] for pair in pairs])


def walk_loop(waypoints, speed: float, fps: float, ahead: float, frames: int = 0):
    """The camera along a closed lap: (eye, look) for every frame.

    The same walk `fly` does with the same path -- densified to 20 cm, closed,
    then stepped at a constant speed along its arc length, looking a fixed
    distance further on. Driving the plan from the lap itself rather than from
    a flight's record is what lets this be made before, or without, any
    flythrough of it.
    """
    loop = np.vstack([waypoints, waypoints[:1]])
    legs = np.linalg.norm(np.diff(loop, axis=0), axis=1)
    dense = np.vstack([np.linspace(loop[i], loop[i + 1],
                                   max(2, int(np.ceil(legs[i] / 0.2))), endpoint=False)
                       for i in range(len(legs))])
    closed = np.vstack([dense, dense[:1]])
    along = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(closed, axis=0), axis=1))]
    total = float(along[-1])

    def at(distance):
        d = np.asarray(distance, dtype=float) % total
        return np.c_[np.interp(d, along, closed[:, 0]), np.interp(d, along, closed[:, 1])]

    # A frame count is the exact way to match another video: one lap spread
    # over that many frames, whatever speed that implies. Asking for a speed
    # that happens to divide the lap into the right number is the same thing
    # arrived at by arithmetic, and wrong by a frame or two when it is not.
    if frames:
        steps, step = max(2, int(frames)), total / max(2, int(frames))
    else:
        steps, step = max(2, int(total / max(1e-6, speed) * fps)), speed / fps
    travelled = np.arange(steps) * step
    return at(travelled), at(travelled + ahead), total, step * fps


def camera_height(scene: str, anchor: str):
    """Where the training cameras actually sat, in the anchored frame.

    The one height a splat render is sharp at is the height it was trained
    from: a Gaussian is fitted to look right from the views that saw it, and
    seen from a metre above them the flat ones turn to streaks. The cameras
    are on disk, so the number is there to be read rather than guessed at a
    nudge per render -- and it has to cross into the anchored frame, because
    that is the frame a flight record is written in.
    """
    centres = []
    with open(os.path.join(os.path.expanduser(scene), "sparse", "0", "images.txt"),
              encoding="utf-8") as handle:
        for line in handle:
            v = line.split()
            if len(v) >= 8 and v[0].isdigit():
                w, x, y, z = (float(c) for c in v[1:5])
                t = np.array([float(c) for c in v[5:8]])
                rotation = np.array(
                    [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
                centres.append(-rotation.T @ t)       # world-to-camera, so the centre
    if not centres:
        raise SystemExit(f"{scene}: no cameras in sparse/0/images.txt")
    with open(os.path.expanduser(anchor), encoding="utf-8") as handle:
        T = np.array(json.load(handle)["T_N_world"], dtype=float)
    anchored = np.asarray(centres) @ T[:3, :3].T + T[:3, 3]
    return anchored[:, 2], len(centres)


def cone(eye, look, reach: float, half_angle: float):
    """The wedge the camera is looking down, as a polygon in plan."""
    forward = np.asarray(look[:2]) - np.asarray(eye[:2])
    norm = float(np.hypot(*forward))
    if norm < 1e-6:
        return np.repeat(np.asarray(eye[:2])[None, :], 3, axis=0)
    heading = np.arctan2(forward[1], forward[0])
    edges = heading + np.radians(half_angle) * np.array([-1.0, 1.0])
    return np.vstack([eye[:2],
                      eye[:2] + reach * np.c_[np.cos(edges), np.sin(edges)]])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path_json", nargs="?", default=None,
                        help="the _path.json lod3dgs fly wrote (or use --path)")
    parser.add_argument("--path", default=None, metavar="FILE",
                        help="the lap fly_path.py wrote (X,Y;X,Y;...), walked here "
                             "the way fly walks it; needs --cloud")
    parser.add_argument("--speed", type=float, default=1.0, help="--path: m/s")
    parser.add_argument("--fps", type=float, default=30.0, help="--path: frames a second")
    parser.add_argument("--ahead", type=float, default=2.0, metavar="M",
                        help="--path: look this far along the way, as fly does")
    parser.add_argument("--frames", type=int, default=0, metavar="N",
                        help="--path: spread the lap over exactly N frames instead "
                             "of using --speed, to match another video's length")
    parser.add_argument("--write-json", default=None, metavar="FILE",
                        help="--path: also write the walk as a flight record, the "
                             "same form lod3dgs fly writes, so splat_walk --fly "
                             "(or another fly) can follow this exact lap")
    parser.add_argument("--eye-from-scene", default=None, metavar="SCENE",
                        help="--write-json: take the height from the training "
                             "cameras of this scene folder (needs --anchor); the "
                             "height a splat model is sharp at is the one it saw")
    parser.add_argument("--anchor", default=None, metavar="FILE",
                        help="frames/anchor_frame.json, to bring those cameras "
                             "into the frame a flight record is written in")
    parser.add_argument("--eye-z", type=float, default=None, metavar="Z",
                        help="--write-json: the camera's height in the map frame "
                             "(not over the floor); read it off an existing "
                             "flight's json to match one")
    parser.add_argument("out", nargs="?", default=None)
    parser.add_argument("--cloud", default=None,
                        help="the map to draw underneath (default: the one the json names)")
    parser.add_argument("--reach", type=float, default=6.0, help="how far the cone reaches (m)")
    parser.add_argument("--fov", type=float, default=70.0,
                        help="the flight's field of view, for the cone's width")
    parser.add_argument("--stride", type=int, default=8, help="draw every Nth map point")
    parser.add_argument("--every", type=int, default=1,
                        help="render every Nth frame (the fps is divided to match, so "
                             "the result still runs in step with the flythrough)")
    parser.add_argument("--trail", type=float, default=0.0,
                        help="show only the last N seconds of the route flown (0: all)")
    parser.add_argument("--pad", type=float, default=2.0)
    parser.add_argument("--width", type=float, default=9.0, help="inches")
    parser.add_argument("--match-height", type=int, default=0, metavar="PX",
                        help="make the video exactly this tall, so it stacks "
                             "beside the flythrough without rescaling")
    parser.add_argument("--dpi", type=int, default=120)
    parser.add_argument("--keep-frames", action="store_true")
    args = parser.parse_args()

    if not args.path_json and not args.path:
        print("give either a flight's _path.json or --path (fly_path.py's lap)",
              file=sys.stderr)
        return 2
    if args.path:
        eye, look, total, pace = walk_loop(read_waypoints(args.path), args.speed,
                                           args.fps, args.ahead, args.frames)
        fps, stem = args.fps, os.path.splitext(os.path.expanduser(args.path))[0]
        source = args.cloud
        print(f"lap {os.path.basename(args.path)}: {total:.0f} m in {len(eye)} "
              f"frames at {fps:g} fps ({len(eye) / fps:.0f} s, {pace:.2f} m/s)")
        if not source:
            print("--path needs --cloud: there is no mesh named anywhere else",
                  file=sys.stderr)
            return 2
    else:
        with open(os.path.expanduser(args.path_json), encoding="utf-8") as handle:
            flight = json.load(handle)
        eye = np.asarray(flight["eye"], dtype=float)
        look = np.asarray(flight["look"], dtype=float)
        fps = float(flight.get("fps", 30))
        source = args.cloud or flight.get("mesh")
        stem = os.path.splitext(os.path.expanduser(args.path_json))[0].replace("_path", "")
        if not source:
            print("the json names no mesh: pass --cloud", file=sys.stderr)
            return 2
    out = os.path.expanduser(args.out or stem + "_overhead.mp4")

    if args.write_json:
        if not args.path:
            print("--write-json goes with --path: a json already is one",
                  file=sys.stderr)
            return 2
        eye_z = args.eye_z
        if args.eye_from_scene:
            if not args.anchor:
                print("--eye-from-scene needs --anchor (frames/anchor_frame.json)",
                      file=sys.stderr)
                return 2
            heights, how_many = camera_height(args.eye_from_scene, args.anchor)
            eye_z = float(np.median(heights))
            print(f"{how_many} training cameras: z from {heights.min():.2f} to "
                  f"{heights.max():.2f}, median {eye_z:.2f} -- flying at the median")
        if eye_z is None:
            print("--write-json needs --eye-z or --eye-from-scene: the lap is a "
                  "plan route and says nothing about height", file=sys.stderr)
            return 2
        # fly's own form, so anything that flies a flight record flies this.
        record = {"fps": fps, "mesh": source or "",
                  "eye": np.c_[eye, np.full(len(eye), eye_z)].round(4).tolist(),
                  "look": np.c_[look, np.full(len(look), eye_z)].round(4).tolist()}
        with open(os.path.expanduser(args.write_json), "w", encoding="utf-8") as handle:
            json.dump(record, handle)
        print(f"wrote {args.write_json}: {len(eye)} frames at {fps:g} fps, "
              f"eye at z={eye_z:.3f}")

    xy = read_xy(source, args.stride)
    print(f"{len(eye):,} frames at {fps:g} fps; map {os.path.basename(source)}, "
          f"{len(xy):,} points binned")

    # The extent covers the map and the flight both: a route that leaves the
    # frame is exactly the moment a viewer needs it.
    span = np.vstack([xy, eye[:, :2]])
    low, high = span.min(axis=0) - args.pad, span.max(axis=0) + args.pad
    ground = float(high[0] - low[0]) / float(high[1] - low[1])
    def even(inches):
        """h264 with yuv420p refuses an odd pixel count in either direction."""
        return max(2.0, round(inches * args.dpi / 2.0) * 2.0 / args.dpi)

    if args.match_height:
        height = even(args.match_height / args.dpi)
        width = even(height * ground)
    else:
        width, height = even(args.width), even(args.width / ground)
    figure = plt.figure(figsize=(width, height), facecolor="white")
    axis = figure.add_axes([0.0, 0.0, 1.0, 1.0])
    axis.set_facecolor("white")
    axis.set_xlim(low[0], high[0])
    axis.set_ylim(low[1], high[1])
    axis.set_aspect("equal")
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)

    pixels = int(round(width * args.dpi))
    bins = (np.linspace(low[0], high[0], max(2, pixels // 2)),
            np.linspace(low[1], high[1], max(2, int(round(height * args.dpi)) // 2)))
    counts = np.histogram2d(xy[:, 0], xy[:, 1], bins=bins)[0]
    shade = np.log1p(counts) / max(1e-9, np.log1p(counts).max())
    axis.imshow(shade.T, origin="lower", extent=[low[0], high[0], low[1], high[1]],
                cmap=LinearSegmentedColormap.from_list(
                    "map", ["#FFFFFF", "#C7CDD4", "#6B7682"]),
                vmin=0.0, vmax=0.75, interpolation="nearest", zorder=1,
                aspect="auto")
    axis.plot(eye[:, 0], eye[:, 1], color="#9AA3AD", lw=1.0, ls=(0, (4, 3)), zorder=2)
    flown, = axis.plot([], [], color="#1B6FA8", lw=2.2, zorder=3)
    wedge = Polygon(np.zeros((3, 2)), closed=True, facecolor="#1B6FA8", alpha=0.22,
                    edgecolor="none", zorder=4)
    axis.add_patch(wedge)
    here, = axis.plot([], [], marker="o", ms=8, color="#C2410C", mec="white",
                      mew=1.2, ls="none", zorder=5)
    clock = axis.text(0.015, 0.985, "", transform=axis.transAxes, color="#111827",
                      fontsize=11, va="top", family="monospace", zorder=6)

    frames_dir = os.path.splitext(out)[0] + "_frames"
    os.makedirs(frames_dir, exist_ok=True)
    picks = range(0, len(eye), max(1, args.every))
    trail = int(args.trail * fps) if args.trail else 0
    for number, index in enumerate(picks):
        first = max(0, index - trail) if trail else 0
        flown.set_data(eye[first:index + 1, 0], eye[first:index + 1, 1])
        wedge.set_xy(cone(eye[index], look[index], args.reach, args.fov / 2.0))
        here.set_data([eye[index, 0]], [eye[index, 1]])
        clock.set_text(f"{index / fps:5.1f} s")
        figure.savefig(os.path.join(frames_dir, f"{number:06d}.png"),
                       dpi=args.dpi, facecolor="white")
        if number % 200 == 0:
            print(f"  {number}/{len(picks)}")
    plt.close(figure)

    rate = fps / max(1, args.every)
    command = ["ffmpeg", "-y", "-framerate", f"{rate:g}",
               "-i", os.path.join(frames_dir, "%06d.png"),
               "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", out]
    try:
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL,
                       stderr=subprocess.PIPE)
    except FileNotFoundError:
        print("ffmpeg not found; the frames are written, encode them with:")
        print("  " + " ".join(command))
        return 0
    except subprocess.CalledProcessError as error:
        print("ffmpeg failed:", error.stderr.decode("utf-8", "replace")[-400:])
        return 1
    print(f"wrote {out}  ({len(picks)} frames at {rate:g} fps)")
    print(f"side by side with the flythrough ({int(round(height * args.dpi))} px "
          f"tall; --match-height makes that the flythrough's own height):")
    print(f"  ffmpeg -i <fly>.mp4 -i {out} -filter_complex hstack both.mp4")
    if not args.keep_frames:
        for name in os.listdir(frames_dir):
            os.remove(os.path.join(frames_dir, name))
        os.rmdir(frames_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
