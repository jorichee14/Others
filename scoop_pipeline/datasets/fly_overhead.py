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
from matplotlib.patches import Polygon                           # noqa: E402


def read_xy_rgb(path: str, stride: int):
    """The plan positions, and colours where there are any, of a mesh or cloud."""
    try:
        import open3d as o3d
    except ImportError:                                          # pragma: no cover
        raise SystemExit("this needs open3d (the environment lod3dgs.py runs in)")
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
    path = os.path.expanduser(path)
    mesh = o3d.io.read_triangle_mesh(path)
    if len(mesh.triangles):
        points = np.asarray(mesh.vertices)
        colours = np.asarray(mesh.vertex_colors) if mesh.has_vertex_colors() else None
    else:
        cloud = o3d.io.read_point_cloud(path)
        points = np.asarray(cloud.points)
        colours = np.asarray(cloud.colors) if cloud.has_colors() else None
    if not len(points):
        raise SystemExit(f"{path}: no vertices and no points")
    keep = slice(None, None, max(1, stride))
    return points[keep, :2], (colours[keep] if colours is not None and len(colours) else None)


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
    parser.add_argument("path_json", help="the _path.json lod3dgs fly wrote")
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

    with open(os.path.expanduser(args.path_json), encoding="utf-8") as handle:
        flight = json.load(handle)
    eye = np.asarray(flight["eye"], dtype=float)
    look = np.asarray(flight["look"], dtype=float)
    fps = float(flight.get("fps", 30))
    source = args.cloud or flight.get("mesh")
    if not source:
        print("the json names no mesh: pass --cloud", file=sys.stderr)
        return 2
    out = os.path.expanduser(args.out or
                             os.path.splitext(os.path.expanduser(args.path_json))[0]
                             .replace("_path", "") + "_overhead.mp4")

    xy, rgb = read_xy_rgb(source, args.stride)
    print(f"{len(eye):,} frames at {fps:g} fps; map {os.path.basename(source)}, "
          f"{len(xy):,} points drawn")

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

    axis.scatter(xy[:, 0], xy[:, 1], s=0.5, linewidths=0, zorder=1,
                 c=(rgb if rgb is not None else "#C7CDD4"))
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
