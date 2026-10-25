#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The pair moving over the site, seen from above, zoomed to where they went.

The composite figure's animation shows the map beside its time panels, at the
size the whole figure allows. This is the map alone, framed on the ground the
robots actually covered, so a reader can see which of them was where and when
-- which is the thing a cooperative-perception dataset has to show and a
detection metric cannot.

    python3 scripts/bev_movie.py --root <split> --out site1.mp4
        [--cloud mapping/cloud_in_ref_<tag>.pcd]   # the map under them
        [--seconds 40] [--fps 20] [--trail 0]      # 0: the whole path stays

Everything comes from the converted tree: both agents' poses, the labelled
objects, and the frame clock. The cloud is only a backdrop -- it is drawn once
and never moves.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib                                               # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                 # noqa: E402
import numpy as np                                              # noqa: E402
import yaml                                                     # noqa: E402
from matplotlib.patches import Polygon                          # noqa: E402

from ros2opv2v.geometry import x_to_world                        # noqa: E402
from ros2opv2v.writers import read_pcd                           # noqa: E402

BACKGROUND = "#0F1419"
MAP_POINTS = "#2B3740"
AGENT = ("#5BB0E8", "#F2994A")          # blue / orange: the one pair every kind
INK = "#E6E9EC"                         # of colour blindness keeps apart
MUTED = "#8A949E"
BOX_EDGE = "#A3E635"


def agent_dirs(scenario: str):
    return sorted((d for d in os.listdir(scenario)
                   if os.path.isdir(os.path.join(scenario, d))), key=int)


def read_track(agent_dir: str):
    """Every frame's (key, x, y, yaw) for one agent, in the map frame."""
    keys = sorted(f[:-5] for f in os.listdir(agent_dir) if f.endswith(".yaml"))
    track, boxes = [], {}
    for key in keys:
        with open(os.path.join(agent_dir, f"{key}.yaml"), encoding="utf-8") as handle:
            params = yaml.safe_load(handle)
        pose = params["lidar_pose"]
        track.append((key, float(pose[0]), float(pose[1]), float(pose[4])))
        boxes.update(params.get("vehicles") or {})
    return track, boxes


def box_corners(entry):
    """The four plan corners of a labelled box: centre, half-sizes, yaw."""
    cx, cy = float(entry["location"][0]), float(entry["location"][1])
    ex, ey = float(entry["extent"][0]), float(entry["extent"][1])
    yaw = np.radians(float(entry["angle"][1]))
    local = np.array([[-ex, -ey], [ex, -ey], [ex, ey], [-ex, ey]])
    rotation = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    return local @ rotation.T + np.array([cx, cy])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", required=True, help="split dir holding case_*/")
    parser.add_argument("--out", default="bev.mp4")
    parser.add_argument("--cloud", default=None, help="a map cloud to draw underneath")
    parser.add_argument("--scenario", default=None)
    parser.add_argument("--seconds", type=float, default=40.0, help="how long the video runs")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--trail", type=float, default=0.0,
                        help="show only the last N seconds of each path (0: all of it)")
    parser.add_argument("--pad", type=float, default=3.0, help="margin round the paths (m)")
    parser.add_argument("--cloud-stride", type=int, default=12,
                        help="draw every Nth map point (a 20 M point map needs thinning)")
    parser.add_argument("--rate", type=float, default=10.0,
                        help="the frame grid's rate (Hz), for the clock")
    parser.add_argument("--dpi", type=int, default=120)
    parser.add_argument("--keep-frames", action="store_true")
    args = parser.parse_args()

    root = os.path.expanduser(args.root)
    scenarios = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
    scenario = os.path.join(root, args.scenario or scenarios[0])
    agents = agent_dirs(scenario)
    if len(agents) < 2:
        print(f"{scenario}: {len(agents)} agent(s); this needs two", file=sys.stderr)
        return 2

    tracks, boxes = {}, {}
    for agent in agents:
        track, found = read_track(os.path.join(scenario, agent))
        tracks[agent] = track
        boxes.update(found)
    keys = [k for k, *_ in tracks[agents[0]]]
    print(f"{len(agents)} agents, {len(keys)} frames, {len(boxes)} labelled object(s)")

    figure, axis = plt.subplots(figsize=(10, 9), facecolor=BACKGROUND)
    axis.set_facecolor(BACKGROUND)

    if args.cloud:
        cloud = read_pcd(os.path.expanduser(args.cloud))[::max(1, args.cloud_stride), :2]
        axis.scatter(cloud[:, 0], cloud[:, 1], s=0.25, c=MAP_POINTS, linewidths=0, zorder=1)
        print(f"map: {len(cloud):,} points drawn")

    for entry in boxes.values():
        axis.add_patch(Polygon(box_corners(entry), closed=True, fill=False,
                               edgecolor=BOX_EDGE, lw=1.4, zorder=4))

    # Framed on the ground the robots covered and the objects they were there
    # for, not on the map: a corridor they never entered is not what the video
    # is about, and a labelled object out of shot defeats the purpose.
    seen = [[x, y] for agent in agents for _, x, y, _ in tracks[agent]]
    seen += [[float(e["location"][0]), float(e["location"][1])] for e in boxes.values()]
    walked = np.asarray(seen)
    low, high = walked.min(axis=0) - args.pad, walked.max(axis=0) + args.pad
    axis.set_xlim(low[0], high[0])
    axis.set_ylim(low[1], high[1])
    axis.set_aspect("equal")
    axis.tick_params(colors=MUTED, labelsize=8)
    axis.set_xlabel("x (m)", color=MUTED, fontsize=9)
    axis.set_ylabel("y (m)", color=MUTED, fontsize=9)
    for spine in axis.spines.values():
        spine.set_color("#2A3137")

    drawn = {}
    for index, agent in enumerate(agents):
        colour = AGENT[index % len(AGENT)]
        path, = axis.plot([], [], color=colour, lw=2.0, zorder=5,
                          label=f"agent {agent}" + (" (ego)" if index == 0 else ""))
        dot, = axis.plot([], [], "o", color=colour, ms=11, mec=INK, mew=1.4, zorder=8)
        drawn[agent] = (path, dot, colour)
    link, = axis.plot([], [], color=MUTED, lw=1.0, ls="--", zorder=6)
    caption = axis.text(0.015, 0.985, "", transform=axis.transAxes, color=INK,
                        fontsize=11, va="top", family="monospace", zorder=10)
    legend = axis.legend(loc="lower right", fontsize=9, framealpha=0.0)
    for text in legend.get_texts():
        text.set_color(MUTED)

    frames_dir = os.path.splitext(os.path.expanduser(args.out))[0] + "_frames"
    os.makedirs(frames_dir, exist_ok=True)
    count = max(2, int(args.seconds * args.fps))
    picks = np.linspace(0, len(keys) - 1, count).astype(int)
    trail = int(args.trail * args.rate) if args.trail else 0

    for number, index in enumerate(picks):
        for agent in agents:
            track = tracks[agent]
            upto = track[max(0, index - trail):index + 1] if trail else track[:index + 1]
            xs = [p[1] for p in upto]
            ys = [p[2] for p in upto]
            path, dot, _ = drawn[agent]
            path.set_data(xs, ys)
            dot.set_data([track[index][1]], [track[index][2]])
        a, b = tracks[agents[0]][index], tracks[agents[1]][index]
        link.set_data([a[1], b[1]], [a[2], b[2]])
        apart = float(np.hypot(a[1] - b[1], a[2] - b[2]))
        caption.set_text(f"frame {keys[index]}   {index / args.rate:6.1f} s\n"
                         f"apart  {apart:5.1f} m")
        figure.savefig(os.path.join(frames_dir, f"{number:06d}.png"),
                       dpi=args.dpi, facecolor=BACKGROUND)
        if number % 50 == 0:
            print(f"  {number}/{count}")
    plt.close(figure)

    out = os.path.expanduser(args.out)
    command = ["ffmpeg", "-y", "-framerate", str(args.fps),
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
    print(f"wrote {out}")
    if not args.keep_frames:
        for name in os.listdir(frames_dir):
            os.remove(os.path.join(frames_dir, name))
        os.rmdir(frames_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
