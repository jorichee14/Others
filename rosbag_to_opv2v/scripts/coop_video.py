#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A video of what collaborative perception sees, from a converted tree.

One panel per thing the pair actually has: the ego's camera, the collaborator's
camera, and the bird's-eye view both robots' LiDAR returns land in once the
collaborator's cloud is carried into the ego's frame by the poses.  That last
step is the whole claim of cooperative detection -- if the registration is
wrong, the two clouds disagree about where the walls are, and it shows here
long before it shows in a mAP number.

    python3 scripts/coop_video.py --root <split> --out coop.mp4
    python3 scripts/coop_video.py --root <split> --from-key 000500 --limit 300

Agent folders are OPV2V's: the ego is the smallest non-negative id unless
--ego says otherwise.  A camera slower than the frame grid (the ZED at 1.3 Hz
on a 10 Hz grid) has no image in most frames; its panel then holds the last one
it had and says so, rather than going blank or pretending the frame is fresh.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib                                               # noqa: E402
matplotlib.use("Agg")
import matplotlib.image as mpimg                                # noqa: E402
import matplotlib.pyplot as plt                                 # noqa: E402
import numpy as np                                              # noqa: E402
import yaml                                                     # noqa: E402
from matplotlib.patches import Polygon                          # noqa: E402

from ros2opv2v.geometry import invert, x_to_world                # noqa: E402
from ros2opv2v.writers import read_pcd                           # noqa: E402

# Blue/orange: the one categorical pair that survives every kind of colour
# blindness, which matters when the whole point is telling two robots apart.
EGO_POINTS = "#5BB0E8"
COOP_POINTS = "#F2994A"
BOX_EDGE = "#A3E635"
BACKGROUND = "#0F1419"
INK = "#E6E9EC"
MUTED = "#8A949E"

DEFAULT_RANGE = [0.0, -11.2, -1.0, 22.4, 11.2, 3.0]


def agent_dirs(scenario: str):
    return sorted((d for d in os.listdir(scenario)
                   if os.path.isdir(os.path.join(scenario, d))), key=int)


def frame_keys(agent_dir: str):
    return sorted(f[:-5] for f in os.listdir(agent_dir) if f.endswith(".yaml"))


def box_corners(entry):
    """The four BEV corners of one `vehicles` entry, in world xy.

    `location` is the centre, `extent` half-sizes, `angle[1]` yaw in degrees --
    the convention points_in_box reads, so a box drawn here is the box the
    loader scores against.
    """
    cx, cy = float(entry["location"][0]), float(entry["location"][1])
    ex, ey = float(entry["extent"][0]), float(entry["extent"][1])
    yaw = np.radians(float(entry["angle"][1]))
    local = np.array([[-ex, -ey], [ex, -ey], [ex, ey], [-ex, ey]])
    rotation = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    return local @ rotation.T + np.array([cx, cy])


def to_ego(points_xyz, ego_from_world, world_from_source=None):
    """Points in some frame -> the ego's LiDAR frame."""
    points = np.asarray(points_xyz, dtype=np.float64)
    if points.size == 0:
        return points.reshape(0, 3)
    homogeneous = np.column_stack([points[:, :3], np.ones(len(points))])
    transform = ego_from_world if world_from_source is None \
        else ego_from_world @ world_from_source
    return (homogeneous @ transform.T)[:, :3]


def inside(points, box):
    keep = ((points[:, 0] >= box[0]) & (points[:, 0] <= box[3]) &
            (points[:, 1] >= box[1]) & (points[:, 1] <= box[4]) &
            (points[:, 2] >= box[2]) & (points[:, 2] <= box[5]))
    return points[keep]


def load_image(agent_dir, key, held):
    """This frame's camera image, or the last one seen, or nothing.

    Returns (image, is_fresh). A camera slower than the grid is the normal case,
    not an error -- the caller says so on the panel instead of hiding it.
    """
    path = os.path.join(agent_dir, f"{key}_camera0.png")
    if os.path.exists(path):
        return mpimg.imread(path), True
    return held, False


def draw(axes, ego_points, coop_points, boxes, coop_xy, box_range, key, label):
    bev, ego_cam, coop_cam = axes
    bev.clear()
    bev.set_facecolor(BACKGROUND)

    # The detection range, so a viewer sees what the model is asked to cover.
    bev.add_patch(Polygon(
        [(box_range[0], box_range[1]), (box_range[3], box_range[1]),
         (box_range[3], box_range[4]), (box_range[0], box_range[4])],
        closed=True, fill=False, edgecolor=MUTED, linewidth=1.0, linestyle=":"))

    if len(ego_points):
        bev.scatter(ego_points[:, 0], ego_points[:, 1], s=0.6,
                    c=EGO_POINTS, linewidths=0, label="ego (RealSense)")
    if len(coop_points):
        bev.scatter(coop_points[:, 0], coop_points[:, 1], s=0.6,
                    c=COOP_POINTS, linewidths=0, label="collaborator (ZED)")

    for corners in boxes:
        bev.add_patch(Polygon(corners, closed=True, fill=False,
                              edgecolor=BOX_EDGE, linewidth=1.6))

    bev.plot(0, 0, marker="o", markersize=9, color=EGO_POINTS,
             markeredgecolor=INK, markeredgewidth=1.2, zorder=5)
    if coop_xy is not None:
        bev.plot(coop_xy[0], coop_xy[1], marker="o", markersize=9,
                 color=COOP_POINTS, markeredgecolor=INK, markeredgewidth=1.2,
                 zorder=5)
        bev.plot([0, coop_xy[0]], [0, coop_xy[1]], color=MUTED,
                 linewidth=0.8, linestyle="--", zorder=4)

    margin = 2.0
    bev.set_xlim(box_range[0] - margin, box_range[3] + margin)
    bev.set_ylim(box_range[1] - margin, box_range[4] + margin)
    bev.set_aspect("equal")
    bev.set_xlabel("x forward (m)", color=MUTED, fontsize=8)
    bev.set_ylabel("y left (m)", color=MUTED, fontsize=8)
    bev.tick_params(colors=MUTED, labelsize=7)
    for spine in bev.spines.values():
        spine.set_color("#2A3137")
    bev.set_title(f"shared bird's-eye view — frame {key}   {label}",
                  color=INK, fontsize=10, loc="left")
    legend = bev.legend(loc="upper right", fontsize=7, framealpha=0.0,
                        markerscale=8)
    for text in legend.get_texts():
        text.set_color(MUTED)
    return bev, ego_cam, coop_cam


def panel(axis, image, title, fresh):
    axis.clear()
    axis.set_facecolor(BACKGROUND)
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_color("#2A3137")
    if image is None:
        axis.text(0.5, 0.5, "no image yet", ha="center", va="center",
                  color=MUTED, fontsize=9, transform=axis.transAxes)
    else:
        axis.imshow(image, alpha=1.0 if fresh else 0.45)
    suffix = "" if fresh or image is None else "   · held (camera slower than grid)"
    axis.set_title(title + suffix, color=INK if fresh else MUTED,
                   fontsize=9, loc="left")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", required=True, help="split dir holding case_*/")
    parser.add_argument("--out", default="coop.mp4", help="output video")
    parser.add_argument("--ego", default=None, help="ego agent folder (default: lowest id)")
    parser.add_argument("--scenario", default=None, help="scenario folder (default: first)")
    parser.add_argument("--from-key", default=None, help="start at this frame key")
    parser.add_argument("--limit", type=int, default=0, help="how many frames (0 = all)")
    parser.add_argument("--stride", type=int, default=1, help="keep every Nth frame")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--dpi", type=int, default=110)
    parser.add_argument("--range", type=float, nargs=6, default=DEFAULT_RANGE,
                        metavar=("XMIN", "YMIN", "ZMIN", "XMAX", "YMAX", "ZMAX"))
    parser.add_argument("--keep-frames", action="store_true",
                        help="leave the PNGs beside the video")
    args = parser.parse_args()

    root = os.path.expanduser(args.root)
    scenarios = sorted(d for d in os.listdir(root)
                       if os.path.isdir(os.path.join(root, d)))
    if not scenarios:
        print(f"no scenario folders under {root}", file=sys.stderr)
        return 2
    scenario = os.path.join(root, args.scenario or scenarios[0])

    agents = agent_dirs(scenario)
    if len(agents) < 2:
        print(f"{scenario} has {len(agents)} agent(s); this video needs two",
              file=sys.stderr)
        return 2
    ego = args.ego or min((a for a in agents if int(a) >= 0), key=int)
    coop = next(a for a in agents if a != ego)
    ego_dir = os.path.join(scenario, ego)
    coop_dir = os.path.join(scenario, coop)

    keys = frame_keys(ego_dir)
    if args.from_key:
        keys = [k for k in keys if k >= args.from_key]
    keys = keys[::max(1, args.stride)]
    if args.limit:
        keys = keys[:args.limit]
    if not keys:
        print("no frames selected", file=sys.stderr)
        return 2

    out = os.path.expanduser(args.out)
    frames_dir = os.path.splitext(out)[0] + "_frames"
    os.makedirs(frames_dir, exist_ok=True)

    figure = plt.figure(figsize=(13, 6.2), facecolor=BACKGROUND)
    grid = figure.add_gridspec(2, 2, width_ratios=[1.75, 1.0], hspace=0.22,
                               wspace=0.12, left=0.055, right=0.985,
                               top=0.92, bottom=0.08)
    bev = figure.add_subplot(grid[:, 0])
    ego_cam = figure.add_subplot(grid[0, 1])
    coop_cam = figure.add_subplot(grid[1, 1])

    held_ego = held_coop = None
    written = 0
    for index, key in enumerate(keys):
        ego_yaml = os.path.join(ego_dir, f"{key}.yaml")
        coop_yaml = os.path.join(coop_dir, f"{key}.yaml")
        if not (os.path.exists(ego_yaml) and os.path.exists(coop_yaml)):
            continue
        with open(ego_yaml, encoding="utf-8") as handle:
            ego_params = yaml.safe_load(handle)
        with open(coop_yaml, encoding="utf-8") as handle:
            coop_params = yaml.safe_load(handle)

        world_from_ego = x_to_world(ego_params["lidar_pose"])
        world_from_coop = x_to_world(coop_params["lidar_pose"])
        ego_from_world = invert(world_from_ego)

        ego_cloud = read_pcd(os.path.join(ego_dir, f"{key}.pcd"))[:, :3]
        coop_cloud = read_pcd(os.path.join(coop_dir, f"{key}.pcd"))[:, :3]
        # The ego's own returns are already in its frame; the collaborator's
        # have to be carried through the world. This is the pose chain the
        # fusion relies on, drawn rather than asserted.
        ego_points = inside(ego_cloud, args.range)
        coop_points = inside(
            to_ego(coop_cloud, ego_from_world, world_from_coop), args.range)

        vehicles = dict(ego_params.get("vehicles") or {})
        vehicles.update(coop_params.get("vehicles") or {})
        boxes = []
        for entry in vehicles.values():
            corners = box_corners(entry)
            z = float(entry["location"][2])
            local = to_ego(np.column_stack([corners, np.full(4, z)]),
                           ego_from_world)
            boxes.append(local[:, :2])

        coop_xy = to_ego(np.array([world_from_coop[:3, 3]]), ego_from_world)[0, :2]

        image_ego, fresh_ego = load_image(ego_dir, key, held_ego)
        image_coop, fresh_coop = load_image(coop_dir, key, held_coop)
        held_ego = image_ego if image_ego is not None else held_ego
        held_coop = image_coop if image_coop is not None else held_coop

        label = f"{len(boxes)} labelled object(s)"
        draw((bev, ego_cam, coop_cam), ego_points, coop_points, boxes,
             coop_xy, args.range, key, label)
        panel(ego_cam, image_ego, f"agent {ego} — ego camera", fresh_ego)
        panel(coop_cam, image_coop, f"agent {coop} — collaborator camera",
              fresh_coop)

        figure.savefig(os.path.join(frames_dir, f"{index:06d}.png"),
                       dpi=args.dpi, facecolor=BACKGROUND)
        written += 1
        if written % 50 == 0:
            print(f"  {written}/{len(keys)} frames")

    plt.close(figure)
    if not written:
        print("no frames drawn", file=sys.stderr)
        return 1
    print(f"{written} frames in {frames_dir}")

    command = ["ffmpeg", "-y", "-framerate", str(args.fps),
               "-i", os.path.join(frames_dir, "%06d.png"),
               "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", out]
    try:
        subprocess.run(command, check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    except FileNotFoundError:
        print("ffmpeg not found; the frames are written, encode them with:")
        print("  " + " ".join(command))
        return 0
    except subprocess.CalledProcessError as error:
        print("ffmpeg failed:", error.stderr.decode("utf-8", "replace")[-500:])
        return 1

    print(f"wrote {out}")
    if not args.keep_frames:
        for name in os.listdir(frames_dir):
            os.remove(os.path.join(frames_dir, name))
        os.rmdir(frames_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
