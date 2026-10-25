#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The pair moving over the site, from above, with what each robot saw.

The composite figure animates the map beside its time panels, at the size the
whole figure leaves it. This is the map large and framed on the ground the
robots covered, with the two cameras and a link trace beside it -- so a reader
can see which robot was where, what it was looking at, and what the radio was
doing, at one glance.

    python3 scripts/bev_movie.py --root <split> --out site1.mp4
        [--cloud mapping/cloud_in_ref_<tag>.pcd]   # the map under them
        [--cameras] [--comms wifi.rssi] [--theme dark]
        [--seconds 45] [--fps 20] [--trail 0]      # 0: the whole path stays

The access points, picked in each site's own anchored cloud (and note the
leading minus on site 2: pass it as --ap=-28.249,-7.207 or the shell reads it
as a flag):

    site 1   --ap 4.392,6.831
    site 2   --ap=-28.249,-7.207

A camera slower than the frame grid (the ZED at 1.3 Hz on a 10 Hz grid) has no
image in most frames. Its panel holds the last one it had and says so, rather
than going black -- a black panel for ten frames in eleven reads as a broken
sensor, which it is not.

`--comms` takes a dotted path into the frame yaml's comms block, e.g.
`wifi.rssi` or `links.csi_mobile_2_to_mobile_1.rssi_median`. Run without it and
the paths actually present are printed.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
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
from ros2opv2v.statics import box_corners as box_corners_3d      # noqa: E402
from ros2opv2v.statics import points_in_box                      # noqa: E402
from ros2opv2v.writers import read_pcd                           # noqa: E402

# statics.box_corners' sign order: sx, sy, sz each -1 then +1, z fastest.
BOX_EDGES = [(0, 1), (0, 2), (1, 3), (2, 3), (4, 5), (4, 6), (5, 7), (6, 7),
             (0, 4), (1, 5), (2, 6), (3, 7)]

# InCoP's class order, as isaacsim_class_map.json indexes it. A class keeps its
# colour whichever site is drawn and whichever classes happen to be present, so
# two videos can be read side by side.
CLASS_NAMES = ["potted_plant", "chair", "medical_bag", "traffic_cone",
               "wet_floor_sign", "fire_extinguisher", "trash_can"]

THEMES = {
    # Blue and orange in both: the one categorical pair every kind of colour
    # blindness keeps apart, which is the whole job here -- telling the two
    # robots apart is what the video is for.
    "light": dict(bg="#FFFFFF", panel="#F4F6F8", map="#CBD4DA", ink="#1F2933",
                  muted="#67737D", grid="#E2E8ED", box="#15803D", ap="#111827",
                  agents=("#1B6FA8", "#C2410C"),
                  classes=("#15803D", "#7C3AED", "#0E7490", "#B45309",
                           "#BE123C", "#4D7C0F", "#334155")),
    "dark": dict(bg="#0F1419", panel="#151C22", map="#2B3740", ink="#E6E9EC",
                 muted="#8A949E", grid="#2A3137", box="#A3E635", ap="#F8FAFC",
                 agents=("#5BB0E8", "#F2994A"),
                 classes=("#86EFAC", "#C4B5FD", "#67E8F9", "#FCD34D",
                          "#FDA4AF", "#BEF264", "#CBD5E1")),
}


def read_backdrop_xy(path: str) -> np.ndarray:
    """The x and y of any PCD, for drawing under the paths.

    Not ros2opv2v.writers.read_pcd, which understands exactly the header the
    converter writes and refuses everything else -- the right strictness for
    ground truth and the wrong one for a backdrop. A map cloud is `x y z`, a
    coloured one `x y z rgb`, and here neither has to be read exactly: the
    fields are located by the header's own offsets and the rest ignored.
    """
    with open(os.path.expanduser(path), "rb") as handle:
        header = {}
        while True:
            line = handle.readline()
            if not line:
                raise ValueError(f"{path}: truncated PCD header")
            text = line.decode("ascii", errors="replace").strip()
            if not text or text.startswith("#"):
                continue
            key, _, rest = text.partition(" ")
            header[key.upper()] = rest.split()
            if key.upper() == "DATA":
                break
        fields = header.get("FIELDS", [])
        sizes = [int(v) for v in header.get("SIZE", [])]
        types = [t.upper() for t in header.get("TYPE", [])]
        counts = [int(v) for v in header.get("COUNT", ["1"] * len(fields))]
        total = int(header.get("POINTS", header.get("WIDTH", [0]))[0])
        layout = header["DATA"][0].lower()
        for axis in ("x", "y"):
            if axis not in fields:
                raise ValueError(f"{path}: no {axis} field, got {fields}")
        if layout == "ascii":
            rows = np.loadtxt(handle, ndmin=2)
            return rows[:, [fields.index("x"), fields.index("y")]]
        if layout != "binary":
            raise ValueError(f"{path}: DATA {layout} is not read here; "
                             f"save it as binary or ascii")
        stride = sum(size * count for size, count in zip(sizes, counts))
        kinds = {("F", 4): "f4", ("F", 8): "f8", ("U", 4): "u4", ("I", 4): "i4"}
        spec, offset = [], 0
        for name, size, kind, count in zip(fields, sizes, types, counts):
            if name in ("x", "y") and count == 1 and (kind, size) in kinds:
                spec.append((name, kinds[(kind, size)], offset))
            offset += size * count
        if len(spec) != 2:
            raise ValueError(f"{path}: x and y are not plain scalars here")
        dtype = np.dtype({"names": [n for n, _, _ in spec],
                          "formats": [f for _, f, _ in spec],
                          "offsets": [o for _, _, o in spec],
                          "itemsize": stride})
        raw = np.frombuffer(handle.read(total * stride), dtype=dtype, count=total)
        return np.c_[raw["x"].astype(np.float64), raw["y"].astype(np.float64)]


def numeric_paths(block, prefix: str = "") -> dict:
    """Every number in a nested dict, by its dotted path.

    What a comms block holds depends on the radios and the topics the run
    recorded, so the series to plot is discovered rather than assumed.
    """
    found = {}
    if isinstance(block, dict):
        for key, value in block.items():
            found.update(numeric_paths(value, f"{prefix}{key}."))
    elif isinstance(block, (int, float)) and not isinstance(block, bool):
        if np.isfinite(block):
            found[prefix[:-1]] = float(block)
    return found


def dig(block, path: str):
    """One dotted path out of a nested dict, or None."""
    node = block
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return float(node) if isinstance(node, (int, float)) and not isinstance(node, bool) else None


def agent_dirs(scenario: str):
    return sorted((d for d in os.listdir(scenario)
                   if os.path.isdir(os.path.join(scenario, d))), key=int)


def read_track(agent_dir: str, comms_paths, want_camera: bool = False):
    """Per frame: key, x, y, yaw, its camera block; plus the boxes and comms."""
    keys = sorted(f[:-5] for f in os.listdir(agent_dir) if f.endswith(".yaml"))
    track, boxes, series, seen = [], {}, {p: [] for p in comms_paths}, {}
    for key in keys:
        with open(os.path.join(agent_dir, f"{key}.yaml"), encoding="utf-8") as handle:
            params = yaml.safe_load(handle)
        pose = params["lidar_pose"]
        track.append((key, float(pose[0]), float(pose[1]), float(pose[4]),
                      params.get("camera0") if want_camera else None,
                      list(pose) if want_camera else None))
        boxes.update(params.get("vehicles") or {})
        comms = params.get("comms") or {}
        if not seen:
            seen = numeric_paths(comms)
        for path in comms_paths:
            series[path].append(dig(comms, path))
    return track, boxes, series, seen


def arrow(x: float, y: float, yaw_deg: float, size: float):
    """A heading arrow at (x, y): where the robot is AND which way it faces.

    A circle says where; half of what a cooperative recording has to show is
    which way each robot was looking, since that is what decides whether an
    object was in its view at all.
    """
    yaw = np.radians(yaw_deg)
    nose, back, wing = size, -0.45 * size, 0.55 * size
    local = np.array([[nose, 0.0], [back, wing], [-0.15 * size, 0.0], [back, -wing]])
    rotation = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    return local @ rotation.T + np.array([x, y])


def project(points_world: np.ndarray, cords, intrinsic) -> np.ndarray:
    """(N, 3) map points -> (N, 3) [u, v, depth] through a frame's camera block.

    The camera block as the converter wrote it: `cords` is the camera's optical
    frame in the map, `intrinsic` the matrix of the image as written. Nothing
    here goes through the model's loader, so a box that lands on its object is
    the dataset's own word, not the loader's reading of it.
    """
    world_from_camera = x_to_world(cords)
    K = np.asarray(intrinsic, dtype=np.float64)
    local = np.c_[points_world, np.ones(len(points_world))] @ invert(world_from_camera).T
    depth = local[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = K[0, 0] * local[:, 0] / depth + K[0, 2]
        v = K[1, 1] * local[:, 1] / depth + K[1, 2]
    return np.c_[u, v, depth]


def class_colour(entry, theme):
    """The colour of a labelled box's class, by its name.

    A chair and a potted plant are different things and a reader should not
    have to guess which green box is which.
    """
    name = entry.get("class_name") or entry.get("obj_type")
    if name in CLASS_NAMES:
        return theme["classes"][CLASS_NAMES.index(name)]
    return theme["box"]


def seen_by(cloud_world, boxes, min_points: int) -> set:
    """Which labelled boxes this agent's own returns actually fall inside.

    The test visibility_split.py uses, and the only honest one: a box in front
    of the lens may be through a wall, round a corner, or past the LiDAR's
    range. Drawing it there claims the robot sees something it does not.
    """
    if cloud_world is None or not len(cloud_world):
        return set()
    return {key for key, entry in boxes.items()
            if int(points_in_box(cloud_world, entry).sum()) >= min_points}


def draw_boxes_on(panel, boxes, camera, theme, width, height,
                  only=None, solid=None):
    """The labelled boxes put back into the image they were labelled against.

    ``only`` restricts the drawing to boxes something actually saw; of those,
    the ones outside ``solid`` are this agent's blind spots, filled in by its
    collaborator, and are drawn dashed to say so.
    """
    if not camera or "cords" not in camera or "intrinsic" not in camera:
        return
    for key, entry in boxes.items():
        if only is not None and key not in only:
            continue
        mine = solid is None or key in solid
        uvz = project(box_corners_3d(entry), camera["cords"], camera["intrinsic"])
        colour = class_colour(entry, theme)
        for a, b in BOX_EDGES:
            if uvz[a, 2] <= 0.05 or uvz[b, 2] <= 0.05:
                continue            # behind the lens: no line to draw
            panel.plot([uvz[a, 0], uvz[b, 0]], [uvz[a, 1], uvz[b, 1]],
                       color=colour, lw=1.3 if mine else 0.9,
                       ls="-" if mine else (0, (3, 2)),
                       alpha=1.0 if mine else 0.7, zorder=5)
    panel.set_xlim(0, width)
    panel.set_ylim(height, 0)


def box_corners(entry):
    """The four plan corners of a labelled box: centre, half-sizes, yaw."""
    cx, cy = float(entry["location"][0]), float(entry["location"][1])
    ex, ey = float(entry["extent"][0]), float(entry["extent"][1])
    yaw = np.radians(float(entry["angle"][1]))
    local = np.array([[-ex, -ey], [ex, -ey], [ex, ey], [-ex, ey]])
    rotation = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    return local @ rotation.T + np.array([cx, cy])


def load_image(agent_dir: str, key: str, held):
    """This frame's camera image, or the last one seen.

    Returns (image, fresh). A camera slower than the grid is the normal case,
    not a fault, so the caller says "held" rather than showing black.
    """
    path = os.path.join(agent_dir, f"{key}_camera0.png")
    if os.path.exists(path):
        return mpimg.imread(path), True
    return held, False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", required=True, help="split dir holding case_*/")
    parser.add_argument("--out", default="bev.mp4")
    parser.add_argument("--cloud", default=None, help="a map cloud to draw underneath")
    parser.add_argument("--scenario", default=None)
    parser.add_argument("--cameras", action="store_true", help="a panel per agent's camera")
    parser.add_argument("--comms", action="append", default=[], metavar="PATH",
                        help="a dotted path in the comms block to trace; repeat for more")
    parser.add_argument("--theme", choices=sorted(THEMES), default="light")
    parser.add_argument("--ap", default=None, metavar="X,Y",
                        help="mark the access point here (m, map frame)")
    parser.add_argument("--ap-label", default=None, metavar="NAME",
                        help="mark the access point where this labelled object is "
                             "(needs --labels); the AP stood where one of them stood")
    parser.add_argument("--labels", default=None,
                        help="the labels json, for --ap-label (label_static.py's output)")
    parser.add_argument("--boxes-on-camera", action="store_true",
                        help="draw the labelled boxes into the camera panels too")
    parser.add_argument("--only-seen", choices=("self", "either"), default=None,
                        help="draw a box only where it is actually seen -- returns "
                             "inside it, not merely in front of the lens. 'self': "
                             "what this agent's own sweep found. 'either': what the "
                             "pair found between them, the ones this agent cannot "
                             "see itself drawn dashed")
    parser.add_argument("--min-points", type=int, default=3,
                        help="--only-seen: returns inside a box to count as seen")
    parser.add_argument("--max-range", type=float, default=0.0, metavar="M",
                        help="draw a box only within this far of either robot "
                             "(0: no limit); a box 20 m down a corridor is a few "
                             "pixels of clutter, not a reading of the frame")
    parser.add_argument("--seconds", type=float, default=45.0)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--rate", type=float, default=10.0, help="the frame grid's rate (Hz)")
    parser.add_argument("--trail", type=float, default=0.0,
                        help="show only the last N seconds of each path (0: all of it)")
    parser.add_argument("--pad", type=float, default=3.0, help="margin round the paths (m)")
    parser.add_argument("--arrow", type=float, default=0.0,
                        help="heading arrow length (m; 0: from the site's size)")
    parser.add_argument("--cloud-stride", type=int, default=12)
    parser.add_argument("--side-width", type=float, default=0.52, metavar="F",
                        help="the side column's width as a fraction of the map's")
    parser.add_argument("--comms-height", type=float, default=0.55, metavar="F",
                        help="a comms panel's height as a fraction of a camera's")
    parser.add_argument("--dpi", type=int, default=120)
    parser.add_argument("--keep-frames", action="store_true")
    args = parser.parse_args()

    theme = THEMES[args.theme]
    root = os.path.expanduser(args.root)
    scenarios = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
    scenario = os.path.join(root, args.scenario or scenarios[0])
    agents = agent_dirs(scenario)
    if len(agents) < 2:
        print(f"{scenario}: {len(agents)} agent(s); this needs two", file=sys.stderr)
        return 2

    ap = None
    if args.ap:
        ap = [float(v) for v in args.ap.split(",")][:2]
    elif args.ap_label:
        if not args.labels:
            print("--ap-label needs --labels (the json label_static.py wrote)",
                  file=sys.stderr)
            return 2
        import json
        with open(os.path.expanduser(args.labels), encoding="utf-8") as handle:
            known = json.load(handle)
        known = known.get("labels", known) if isinstance(known, dict) else known
        match = [l for l in known if l.get("name") == args.ap_label]
        if not match:
            names = ", ".join(sorted(str(l.get("name", "?")) for l in known))
            print(f"no label named {args.ap_label!r}; the file has: {names}",
                  file=sys.stderr)
            return 2
        ap = [float(match[0]["location"][0]), float(match[0]["location"][1])]
        print(f"AP at {args.ap_label}: {ap[0]:.2f}, {ap[1]:.2f}")

    tracks, series, boxes, available = {}, {}, {}, {}
    for agent in agents:
        track, found, got, seen = read_track(os.path.join(scenario, agent), args.comms,
                                             want_camera=args.boxes_on_camera)
        tracks[agent], series[agent] = track, got
        boxes.update(found)
        available.update(seen)
    keys = [k for k, *_ in tracks[agents[0]]]
    print(f"{len(agents)} agents, {len(keys)} frames, {len(boxes)} labelled object(s)")
    if available:
        print(f"comms available: {', '.join(sorted(available))}")
    if args.comms:
        missing = [p for p in args.comms if all(v is None for v in series[agents[0]][p])]
        if missing:
            print(f"no values for {missing} — see the list above", file=sys.stderr)
            return 2

    # ------------------------------------------------------------------ layout
    # A panel per comms series, never two on one axis: RSSI in dBm and RTT in
    # ms share no scale, and drawing them together squashes one flat against
    # the frame. Two y-axes would be worse -- the reader cannot tell which
    # line belongs to which.
    shown_cameras = len(agents) if args.cameras else 0
    panels = shown_cameras + len(args.comms)
    width, height = (17.0 if panels else 10.0), 9.5
    figure = plt.figure(figsize=(width, height), facecolor=theme["bg"])
    if panels:
        # The map and the camera frames are what a reader actually reads; the
        # comms traces are the reference beside them, so they get the height
        # a reference needs and no more.
        heights = [1.0] * shown_cameras + [args.comms_height] * len(args.comms)
        print(f"figure {width:.1f}x{height:.1f} in at {args.dpi} dpi; side column "
              f"{args.side_width:.2f} of the map, comms rows "
              f"{args.comms_height:.2f} of a camera row")
        grid = figure.add_gridspec(panels, 2, width_ratios=[1.0, args.side_width],
                                   height_ratios=heights,
                                   hspace=0.30, wspace=0.09,
                                   left=0.045, right=0.985, top=0.955, bottom=0.06)
        axis = figure.add_subplot(grid[:, 0])
        side = [figure.add_subplot(grid[i, 1]) for i in range(panels)]
    else:
        axis = figure.add_subplot(111)
        side = []
    axis.set_facecolor(theme["bg"])

    if args.cloud:
        cloud = read_backdrop_xy(args.cloud)[::max(1, args.cloud_stride)]
        axis.scatter(cloud[:, 0], cloud[:, 1], s=0.3, c=theme["map"], linewidths=0, zorder=1)
        print(f"map: {len(cloud):,} points drawn")

    shown = set()
    for entry in boxes.values():
        colour = class_colour(entry, theme)
        name = entry.get("class_name") or entry.get("obj_type") or "object"
        axis.add_patch(Polygon(box_corners(entry), closed=True, fill=False,
                               edgecolor=colour, lw=1.8, zorder=4,
                               label=None if name in shown else name))
        shown.add(name)

    if ap is not None:
        axis.plot([ap[0]], [ap[1]], marker="*", ms=20, color=theme["ap"],
                  mec=theme["bg"], mew=1.2, zorder=9, ls="none", label="AP")
        axis.annotate("AP", xy=(ap[0], ap[1]), xytext=(6, 6),
                      textcoords="offset points", color=theme["ap"],
                      fontsize=10, fontweight="bold", zorder=9)

    seen_xy = [[x, y] for agent in agents for _, x, y, _, _, _ in tracks[agent]]
    seen_xy += [[float(e["location"][0]), float(e["location"][1])] for e in boxes.values()]
    if ap is not None:
        seen_xy.append(ap)
    walked = np.asarray(seen_xy)
    low, high = walked.min(axis=0) - args.pad, walked.max(axis=0) + args.pad
    axis.set_xlim(low[0], high[0])
    axis.set_ylim(low[1], high[1])
    axis.set_aspect("equal")
    axis.tick_params(colors=theme["muted"], labelsize=8)
    axis.set_xlabel("x (m)", color=theme["muted"], fontsize=9)
    axis.set_ylabel("y (m)", color=theme["muted"], fontsize=9)
    axis.grid(True, color=theme["grid"], lw=0.6, zorder=0)
    for spine in axis.spines.values():
        spine.set_color(theme["grid"])
    size = args.arrow or max(0.5, 0.018 * float(max(high - low)))

    drawn = {}
    for index, agent in enumerate(agents):
        colour = theme["agents"][index % 2]
        path, = axis.plot([], [], color=colour, lw=2.0, zorder=5,
                          label=f"agent {agent}" + (" (ego)" if index == 0 else ""))
        nose = Polygon(np.zeros((4, 2)), closed=True, facecolor=colour,
                       edgecolor=theme["bg"], lw=1.2, zorder=8)
        axis.add_patch(nose)
        drawn[agent] = (path, nose, colour)
    link, = axis.plot([], [], color=theme["muted"], lw=1.0, ls="--", zorder=6)
    caption = axis.text(0.015, 0.985, "", transform=axis.transAxes, color=theme["ink"],
                        fontsize=11, va="top", family="monospace", zorder=10,
                        bbox=dict(boxstyle="round,pad=0.4", fc=theme["panel"],
                                  ec=theme["grid"]))
    legend = axis.legend(loc="lower right", fontsize=9, framealpha=0.0)
    for text in legend.get_texts():
        text.set_color(theme["muted"])

    cameras, comms_axis, cursors = [], None, []
    slot = 0
    if args.cameras:
        for index, agent in enumerate(agents):
            panel = side[slot]
            slot += 1
            panel.set_facecolor(theme["panel"])
            panel.set_xticks([])
            panel.set_yticks([])
            for spine in panel.spines.values():
                spine.set_color(theme["grid"])
            cameras.append((agent, panel, None))
    clock = np.arange(len(keys)) / args.rate
    for number, path_name in enumerate(args.comms):
        panel = side[slot]
        slot += 1
        panel.set_facecolor(theme["panel"])
        for index, agent in enumerate(agents):
            values = np.array([np.nan if v is None else v
                               for v in series[agent][path_name]], dtype=float)
            if np.all(np.isnan(values)):
                continue
            panel.plot(clock, values, lw=1.2, color=theme["agents"][index % 2],
                       label=f"agent {agent}")
        panel.set_xlim(clock[0], clock[-1])
        panel.set_title(path_name, color=theme["ink"], fontsize=9, loc="left")
        panel.tick_params(colors=theme["muted"], labelsize=7)
        panel.grid(True, color=theme["grid"], lw=0.5)
        for spine in panel.spines.values():
            spine.set_color(theme["grid"])
        if number == len(args.comms) - 1:
            panel.set_xlabel("time (s)", color=theme["muted"], fontsize=8)
        if number == 0:
            legend = panel.legend(fontsize=7, framealpha=0.0, loc="upper right")
            for text in legend.get_texts():
                text.set_color(theme["muted"])
        cursors.append(panel.axvline(0, color=theme["ink"], lw=1.0))

    # ------------------------------------------------------------------ frames
    frames_dir = os.path.splitext(os.path.expanduser(args.out))[0] + "_frames"
    os.makedirs(frames_dir, exist_ok=True)
    count = max(2, int(args.seconds * args.fps))
    picks = np.linspace(0, len(keys) - 1, count).astype(int)
    trail = int(args.trail * args.rate) if args.trail else 0
    held = {agent: (None, None, None) for agent in agents}  # image, camera, frame
    world_clouds = OrderedDict()    # (agent, frame) -> its returns in the map frame

    def returns_of(agent, frame):
        """That agent's sweep for a frame, in the map frame.

        Only the handful in play are kept: every frame of every agent is tens
        of gigabytes, and the frames being drawn only move forward.
        """
        if (agent, frame) not in world_clouds:
            points = read_pcd(os.path.join(scenario, agent,
                                           f"{keys[frame]}.pcd"))[:, :3]
            pose = x_to_world(tracks[agent][frame][5])
            world_clouds[(agent, frame)] = (points.astype(np.float64)
                                            @ pose[:3, :3].T + pose[:3, 3])
            while len(world_clouds) > 2 * len(agents):
                world_clouds.popitem(last=False)
        world_clouds.move_to_end((agent, frame))
        return world_clouds[(agent, frame)]

    for number, index in enumerate(picks):
        for agent in agents:
            track = tracks[agent]
            upto = track[max(0, index - trail):index + 1] if trail else track[:index + 1]
            path, nose, _ = drawn[agent]
            path.set_data([p[1] for p in upto], [p[2] for p in upto])
            _, x, y, yaw, _camera, _pose = track[index]
            nose.set_xy(arrow(x, y, yaw, size))
        a, b = tracks[agents[0]][index], tracks[agents[1]][index]
        link.set_data([a[1], b[1]], [a[2], b[2]])
        lines = [f"frame {keys[index]}   {index / args.rate:6.1f} s",
                 f"apart  {np.hypot(a[1] - b[1], a[2] - b[2]):5.1f} m"]
        if ap is not None:
            # The distance each robot is from the AP, because that is what the
            # RSSI and RTT traces beside the map are a function of -- the
            # reader can see the link fall off as the number grows.
            for agent, where in zip(agents, (a, b)):
                lines.append(f"AP-{agent}  "
                             f"{np.hypot(where[1] - ap[0], where[2] - ap[1]):5.1f} m")
        caption.set_text("\n".join(lines))
        for position, (agent, panel, _) in enumerate(cameras):
            image, fresh = load_image(os.path.join(scenario, agent), keys[index],
                                      held[agent][0])
            # The boxes go through the camera block of the frame the SHOWN image
            # came from. A held image is of where the robot was then, so
            # projecting this frame's pose onto it slides the boxes off their
            # objects -- and at 1.3 Hz that is most frames.
            camera = tracks[agent][index][4] if fresh else held[agent][1]
            source = index if fresh else held[agent][2]
            if fresh:
                held[agent] = (image, camera, index)
            panel.clear()
            panel.set_xticks([])
            panel.set_yticks([])
            panel.set_facecolor(theme["panel"])
            for spine in panel.spines.values():
                spine.set_color(theme["grid"])
            if image is None:
                panel.text(0.5, 0.5, "no image yet", ha="center", va="center",
                           color=theme["muted"], fontsize=9, transform=panel.transAxes)
            else:
                panel.imshow(image)
                if args.boxes_on_camera:
                    # Where the camera that took THIS image puts them, and only
                    # the ones that image's own LiDAR sweep actually found: if
                    # they land on the objects, the dataset agrees with itself.
                    only = solid = None
                    if (args.only_seen or args.max_range) and source is not None:
                        if args.only_seen:
                            solid = seen_by(returns_of(agent, source), boxes,
                                            args.min_points)
                            only = set(solid)
                            if args.only_seen == "either":
                                for other in agents:
                                    if other != agent:
                                        only |= seen_by(returns_of(other, source),
                                                        boxes, args.min_points)
                        else:
                            only = set(boxes)
                        if args.max_range:
                            # Near EITHER robot, measured where the pair stood
                            # when the SHOWN image was taken: an object the
                            # collaborator is standing next to is exactly the
                            # one worth drawing on a camera that cannot see it.
                            where = [tracks[other][source][1:3] for other in agents]
                            near = {key for key, entry in boxes.items()
                                    if min(np.hypot(entry["location"][0] - hx,
                                                    entry["location"][1] - hy)
                                           for hx, hy in where) <= args.max_range}
                            only &= near
                            if solid is not None:
                                solid &= near
                    draw_boxes_on(panel, boxes, camera, theme,
                                  image.shape[1], image.shape[0],
                                  only=only, solid=solid)
            role = "ego" if position == 0 else "collaborator"
            panel.set_title(f"agent {agent} — {role}" + ("" if fresh else "   · held"),
                            color=theme["ink"] if fresh else theme["muted"],
                            fontsize=9, loc="left")
        for cursor in cursors:
            cursor.set_xdata([index / args.rate, index / args.rate])
        figure.savefig(os.path.join(frames_dir, f"{number:06d}.png"),
                       dpi=args.dpi, facecolor=theme["bg"])
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
