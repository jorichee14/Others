#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage D: label the static objects once, in the map frame, from the run's own map.

A chair does not move, so it has ONE pose for the whole recording. Label it once
against the accumulated map cloud and the converter writes it into all 1330
frames of all three agents, already projected — every frame's pose is known, so
projection is exact and free.

That also makes a hand-placed chair better ground truth than an agent-derived
box. A box computed from an agent's own pose is right by construction and tests
nothing. A chair boxed against the map is an INDEPENDENT fact, so "do both
agents' returns land inside it" becomes a real check on the extrinsics, the
anchoring and the synchronisation at once — which is what `--check` does.

    # 1. is the map cloud in the same frame as the dataset? (always do this first)
    python3 scripts/label_static.py --pcd map.pcd --dataset ~/cpfa/data/OPV2V_mirc/test

    # 2. find candidate objects standing on the floor, so you can read off seeds
    python3 scripts/label_static.py --pcd map.pcd --dataset <root>/test \\
        --propose --map-image /tmp/coop2_top_down.png

    # 3. fit a box at a seed (a click position, or a proposal's centre)
    python3 scripts/label_static.py --pcd map.pcd --dataset <root>/test \
        --seed 3.4,-6.1 --name chair_1 --out labels/coop2_statics.json

    # 4. verify against the real clouds, then convert with them
    python3 scripts/label_static.py --pcd map.pcd --dataset <root>/test \
        --labels labels/coop2_statics.json --check

Then add to the converter config and re-run `convert_rosbag.py`:

    labels_file: labels/coop2_statics.json

`--interactive` opens an Open3D window to pick seeds by shift-click, if open3d
is installed; it is never required — a seed is just an x,y you can read off
`--propose` or any viewer.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
from typing import Dict, List, Optional

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ros2opv2v.statics import (StaticsError, cluster_at, fit_box,      # noqa: E402
                               footprint_corners, ground_level,
                               points_in_box, read_pcd_xyz, seed_report)
from ros2opv2v.preview import Canvas, height_ramp                      # noqa: E402
from ros2opv2v.geometry import x_to_world                              # noqa: E402
from ros2opv2v.writers import read_pcd                                 # noqa: E402

RESERVED_MIN = 10000     # agent-derived ids start here; ours must stay below


# ------------------------------------------------------------------ dataset
def load_dataset_poses(root: str, limit: int = 0) -> Dict[str, List[dict]]:
    """{agent_id: [{key, lidar_pose, dir}, ...]} from a converted dataset."""
    out: Dict[str, List[dict]] = {}
    for scenario in sorted(glob.glob(os.path.join(root, "*"))):
        if not os.path.isdir(scenario):
            continue
        for agent in sorted(os.listdir(scenario)):
            agent_dir = os.path.join(scenario, agent)
            if not os.path.isdir(agent_dir):
                continue
            keys = sorted(f[:-5] for f in os.listdir(agent_dir) if f.endswith(".yaml"))
            if limit:
                keys = keys[::max(1, len(keys) // limit)]
            rows = []
            for key in keys:
                with open(os.path.join(agent_dir, key + ".yaml")) as handle:
                    params = yaml.safe_load(handle)
                rows.append({"key": key, "lidar_pose": params["lidar_pose"], "dir": agent_dir})
            out.setdefault(agent, []).extend(rows)
    if not out:
        raise SystemExit(f"no converted frames under {root}")
    return out


def verify_frame(cloud: np.ndarray, poses: Dict[str, List[dict]]) -> bool:
    """Is the map cloud in the same frame as the dataset?

    Everything downstream assumes it is, and nothing else would notice if it were
    not: a cloud in the mapping pipeline's pre-anchor frame looks equally
    plausible and puts every box metres out. The test that actually discriminates
    is whether the agents' trajectories lie INSIDE the cloud's footprint — they
    were recorded in the same room the cloud is of, so they must.
    """
    lo, hi = cloud.min(axis=0), cloud.max(axis=0)
    print("map cloud   : %d points, x %.2f..%.2f  y %.2f..%.2f  z %.2f..%.2f"
          % (len(cloud), lo[0], hi[0], lo[1], hi[1], lo[2], hi[2]))
    # Only MOVING agents are evidence. A static infrastructure node legitimately
    # sits outside the mapped floor and looks into it — the MIRC Arducam is at
    # x = -5.5 m, 2 m up, outside the area the carts drove through. Failing the
    # check on it would reject a correct map.
    moving, verdicts = [], []
    for agent, rows in sorted(poses.items()):
        track = np.array([r["lidar_pose"][:3] for r in rows], dtype=np.float64)
        t_lo, t_hi = track.min(axis=0), track.max(axis=0)
        span = float(np.linalg.norm(t_hi[:2] - t_lo[:2]))
        inside = ((track[:, 0] >= lo[0]) & (track[:, 0] <= hi[0])
                  & (track[:, 1] >= lo[1]) & (track[:, 1] <= hi[1]))
        frac = float(inside.mean())
        is_static = span < 0.5
        note = ("  (static — outside the mapped floor is normal for infrastructure)"
                if is_static else ("   <-- OUTSIDE the cloud" if frac <= 0.95 else ""))
        print("  agent %-4s trajectory x %.2f..%.2f  y %.2f..%.2f   %.0f%% inside the "
              "cloud footprint%s"
              % (agent, t_lo[0], t_hi[0], t_lo[1], t_hi[1], 100 * frac, note))
        if not is_static:
            moving.append(agent)
            verdicts.append(frac > 0.95)
    if not verdicts:
        print("  -> no moving agent to test against; the frame cannot be verified this "
              "way. Check by eye that the cloud covers where the agents were.")
        return True
    if all(verdicts):
        print("  -> the moving agents' trajectories (%s) sit inside the cloud: same frame."
              % ", ".join(moving))
        return True
    print("  -> !! a MOVING agent's trajectory does not sit inside the cloud. This map is "
          "in a different frame from the dataset (the mapping pipeline's pre-anchor "
          "frame, most likely). Boxes drawn in it would be wrong everywhere. Use the "
          "ANCHORED cloud, the one whose origin is the surveyed anchor board.")
    return False


# ----------------------------------------------------------------- proposing
def trajectory_roi(poses: Dict[str, List[dict]], margin: float) -> Optional[np.ndarray]:
    """The area the MOVING agents covered, expanded by `margin`, as [xlo, ylo, xhi, yhi].

    A mapping session usually covers far more of a building than one run does —
    this map is 33 x 54 m while the carts moved through about 8 x 15 m of it. Only
    objects the agents could actually observe can be ground truth for this
    dataset, so proposals are restricted to their neighbourhood; furniture in
    rooms nobody visited is noise in the list, and a box there would be a label
    no agent can ever see.
    """
    tracks = []
    for rows in poses.values():
        track = np.array([r["lidar_pose"][:2] for r in rows], dtype=np.float64)
        if float(np.linalg.norm(track.max(axis=0) - track.min(axis=0))) >= 0.5:
            tracks.append(track)
    if not tracks:
        return None
    allpts = np.vstack(tracks)
    return np.r_[allpts.min(axis=0) - margin, allpts.max(axis=0) + margin]


def start_roi(poses: Dict[str, List[dict]], margin: float) -> Optional[np.ndarray]:
    """The box spanned by the MOVING agents' first poses, expanded by `margin`.

    A run that starts with the carts on opposite sides of the objects they are
    there to observe makes those start points a tighter and more meaningful
    bound than the whole driven path: the objects are between them by
    construction, and the path wanders off to places they are not.
    """
    starts = []
    for rows in poses.values():
        track = np.array([r["lidar_pose"][:2] for r in rows], dtype=np.float64)
        if float(np.linalg.norm(track.max(axis=0) - track.min(axis=0))) >= 0.5:
            starts.append(track[0])
    if len(starts) < 2:
        return None
    pts = np.array(starts)
    return np.r_[pts.min(axis=0) - margin, pts.max(axis=0) + margin]


class HeightMap(object):
    """The map seen from above: per XY cell, how tall the tallest return is.

    Everything downstream — the proposals and the picture — reads this one
    structure, so what you are shown is exactly what was searched.
    """

    def __init__(self, cloud: np.ndarray, ground_z: float, cell: float,
                 roi: Optional[np.ndarray] = None, floor_clearance: float = 0.05,
                 ceiling: Optional[float] = None):
        above = cloud[:, 2] - ground_z
        keep = above > floor_clearance
        if roi is not None:
            keep &= ((cloud[:, 0] >= roi[0]) & (cloud[:, 0] <= roi[2])
                     & (cloud[:, 1] >= roi[1]) & (cloud[:, 1] <= roi[3]))
        # Anything overhead — a ceiling, a beam, a pipe run, a mezzanine — is the
        # tallest return over the cells beneath it, and "tallest per cell" is the
        # whole definition of this map. Left in, a scanned ceiling makes every
        # cell under it structure, and the furniture standing on that floor cannot
        # be proposed at all: not filtered out, never seen. A wall is still taller
        # than the object band once the ceiling is gone, so the cutoff costs
        # nothing that the search needs.
        self.overhead = 0
        if ceiling is not None:
            overhead = keep & (above > ceiling)
            self.overhead = int(overhead.sum())
            keep &= ~overhead
        pts, hgt = cloud[keep], above[keep]
        self.cell, self.points = cell, len(pts)
        self.ceiling = ceiling
        if len(pts) < 200:
            self.width = self.depth = 0
            return
        grid = np.floor(pts[:, :2] / cell).astype(np.int64)
        self.origin = grid.min(axis=0)
        grid -= self.origin
        self.width = int(grid[:, 0].max()) + 1
        self.depth = int(grid[:, 1].max()) + 1
        flat = grid[:, 0] * self.depth + grid[:, 1]
        self.top = np.zeros(self.width * self.depth, dtype=np.float64)
        np.maximum.at(self.top, flat, hgt)
        self.counts = np.bincount(flat, minlength=self.width * self.depth)

    def empty(self) -> bool:
        return self.width == 0

    def cell_xy(self, flat_cells: np.ndarray) -> np.ndarray:
        """Map-frame lower corner of each cell, in metres."""
        cells = np.stack(np.divmod(flat_cells, self.depth), axis=1) + self.origin
        return cells * self.cell

    def bounds(self):
        lo = self.origin * self.cell
        hi = (self.origin + np.array([self.width, self.depth])) * self.cell
        return float(lo[0]), float(lo[1]), float(hi[0]), float(hi[1])

    def as_image_grid(self) -> np.ndarray:
        """(depth, width) — row = y index, column = x index, for rendering."""
        return self.top.reshape(self.width, self.depth).T

    def structure_fraction(self, max_height: float) -> float:
        """Of the cells that saw anything, how many are taller than the band.

        If nearly all of them are, the search has no floor left to find objects
        on, and the cause is almost always something overhead rather than a room
        made entirely of walls.

        Counted over EVERY cell of the searched grid, not only the ones that got a
        return. A bare floor gives no returns at all above the clearance, so
        measuring against occupied cells alone would compare walls to walls and
        report a room made mostly of open floor as mostly structure.
        """
        if not len(self.top):
            return 0.0
        return float((self.top > max_height).mean())


def propose(cloud: np.ndarray, ground_z: float, args,
            roi: Optional[np.ndarray] = None,
            hmap: Optional["HeightMap"] = None) -> List[dict]:
    """Candidate free-standing objects, so a seed can be read off rather than hunted.

    Not a detector, and it does not need to be — it turns "find two chairs in a
    33 x 54 m cloud" into "pick two rows from a short list".

    It works on a HEIGHT MAP rather than on the points directly, because
    clustering the points in a height band does not separate a chair from the
    wall behind it: connected components merge them, and the list fills up with
    wall segments whose only common feature is that they are big. Per XY cell the
    tallest point above the floor is what matters, and then a free-standing object
    is a compact patch of cells of the right height surrounded by floor. A wall is
    a patch of cells that are TALLER than any chair, so it never enters the
    candidate mask at all.

    What is left over after that exclusion is a wall's FRINGE: the cells along its
    base where the beam only reached partway up, which are short and so look like
    an object. A chair pushed against a wall touches structure too, so touching is
    not the discriminator — how MUCH of the blob's outline is structure is. A wall
    fringe is walled along most of its perimeter; a chair, even a chair against a
    wall, is open on the other three sides. That fraction is `wall_contact`, and
    it is what the list is sorted by.
    """
    hmap = hmap or HeightMap(cloud, ground_z, args.cell, roi,
                             ceiling=getattr(args, "ceiling", None))
    if hmap.empty():
        print("  only %d points above the floor in that area" % hmap.points)
        return []
    width, depth = hmap.width, hmap.depth
    top, counts = hmap.top, hmap.counts

    # A cell belongs to a candidate object when its TALLEST point is inside the
    # object height band. A wall's cells are taller than max_height, so they are
    # excluded here rather than having to be filtered out of a merged cluster
    # afterwards.
    candidate = ((top >= args.min_height) & (top <= args.max_height) & (counts >= 2))
    tall = top > args.max_height

    def neighbours(flat_cell: int):
        cx, cy = divmod(flat_cell, depth)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                nx, ny = cx + dx, cy + dy
                if (0 <= nx < width and 0 <= ny < depth) and (dx or dy):
                    yield nx * depth + ny

    proposals, seen = [], np.zeros(width * depth, dtype=bool)
    order = np.argsort(-np.where(candidate, counts, 0))
    for start_cell in order:
        if not candidate[start_cell] or seen[start_cell]:
            continue
        if counts[start_cell] == 0:
            break
        stack, members = [int(start_cell)], []
        seen[start_cell] = True
        while stack:
            current = stack.pop()
            members.append(current)
            for nxt in neighbours(current):
                if candidate[nxt] and not seen[nxt]:
                    seen[nxt] = True
                    stack.append(nxt)
        member_arr = np.array(members)
        n_points = int(counts[member_arr].sum())
        if n_points < args.min_points:
            continue
        cells_xy = hmap.cell_xy(member_arr)
        lo = cells_xy.min(axis=0)
        hi = cells_xy.max(axis=0) + hmap.cell
        span = hi - lo
        if max(span) > args.max_footprint or max(span) < args.min_footprint:
            continue

        # The outline: every cell adjacent to the blob but not in it. How much of
        # that outline is structure says whether this is furniture or a wall's foot.
        inside = set(int(m) for m in members)
        outline = set()
        for member in members:
            for nxt in neighbours(int(member)):
                if int(nxt) not in inside:
                    outline.add(int(nxt))
        outline_arr = np.array(sorted(outline)) if outline else np.zeros(0, dtype=np.int64)
        wall_contact = (float(tall[outline_arr].mean()) if len(outline_arr) else 0.0)

        proposals.append({
            "centre": [round(float(0.5 * (lo[0] + hi[0])), 3),
                       round(float(0.5 * (lo[1] + hi[1])), 3)],
            "points": n_points,
            "footprint_m": [round(float(span[0]), 2), round(float(span[1]), 2)],
            "top_m": round(float(top[member_arr].max()), 2),
            "cells": int(len(members)),
            "wall_contact": round(wall_contact, 2),
            "against_structure": bool(wall_contact > 0.0),
            "extent_xy": [round(float(lo[0]), 3), round(float(lo[1]), 3),
                          round(float(hi[0]), 3), round(float(hi[1]), 3)],
        })
    # Open on all sides first: that is what a chair standing in a room looks like,
    # and it is the only ordering that puts furniture above wall feet.
    limit = getattr(args, "max_wall_contact", 1.0)
    proposals = [p for p in proposals if p["wall_contact"] <= limit]
    proposals.sort(key=lambda p: (p["wall_contact"], -p["points"]))
    return proposals[:args.max_proposals]


def render_seeds(path: str, cloud: np.ndarray, ground_z: float, labels: List[dict],
                 args, poses: Optional[Dict[str, List[dict]]] = None) -> None:
    """Draw the fitted footprints on the floor around them, so a fit can be seen.

    A box printed as nine numbers is not checkable by eye. The same box drawn on
    the map, with the returns it was fitted to underneath it, is: either the
    outline sits on an object-shaped patch of floor or it obviously does not.
    """
    if not labels:
        return
    centres = np.array([label["location"][:2] for label in labels], dtype=np.float64)
    pad = max(3.0, 2.0 * args.radius)
    roi = np.r_[centres.min(axis=0) - pad, centres.max(axis=0) + pad]
    hmap = HeightMap(cloud, ground_z, args.cell, roi, ceiling=args.ceiling)
    if hmap.empty():
        print("\n  ! nothing to draw around those seeds")
        return
    render_map(path, hmap, [], args, poses, labels=labels)


def render_map(path: str, hmap: "HeightMap", rows: List[dict], args,
               poses: Optional[Dict[str, List[dict]]] = None,
               labels: Optional[List[dict]] = None) -> None:
    """Draw the searched area from above with the proposals numbered on it.

    The list can only describe a blob; this shows its shape, and the shape is how
    a person tells a chair from the foot of a wall in one glance.
    """
    x0, y0, x1, y1 = hmap.bounds()
    canvas = Canvas(x0, y0, x1, y1, hmap.cell, scale=args.map_scale)
    grid = hmap.as_image_grid()[::-1]          # row 0 = highest y, as on screen

    occupied = grid > 0.0
    tall = grid > args.max_height
    band = occupied & ~tall
    canvas.blit_cells(band, height_ramp(np.clip(grid, 0, args.max_height),
                                        0.0, args.max_height))
    canvas.blit_cells(tall, np.broadcast_to(np.array([70, 70, 78], dtype=np.uint8),
                                            grid.shape + (3,)))

    for _agent_id, frames in sorted((poses or {}).items()):
        for frame in frames[::3]:
            pose = frame["lidar_pose"] if "lidar_pose" in frame else frame["xyz"]
            canvas.dot(pose[0], pose[1], (255, 255, 255), 0)

    for index, row in enumerate(rows):
        ex = row["extent_xy"]
        colour = (220, 0, 0) if row["wall_contact"] <= 0.25 else (150, 60, 200)
        canvas.box(ex[0] - hmap.cell, ex[1] - hmap.cell,
                   ex[2] + hmap.cell, ex[3] + hmap.cell, colour)
        canvas.text(ex[2] + 2 * hmap.cell, ex[3], str(index + 1), colour,
                    size=max(1, args.map_scale // 2))

    for label in (labels or []):
        corners = footprint_corners(label)
        canvas.polygon(corners, (255, 40, 40))
        canvas.text(float(corners[:, 0].max()) + 2 * hmap.cell,
                    float(corners[:, 1].max()), str(label["id"]), (255, 40, 40),
                    size=max(1, args.map_scale // 2))

    canvas.save(path)
    print("\n  wrote %s  (%d x %d px, %.2f m per cell)"
          % (path, canvas.cols, canvas.rows, hmap.cell))
    print("    blue-green-orange = height above the floor, dark grey = taller than "
          "%.2f m (structure)," % args.max_height)
    if rows:
        print("    white dots = where the agents drove, red boxes = open on all sides, "
              "purple = touching structure.")
    else:
        print("    white dots = where the agents drove, red outlines = the fitted "
              "boxes, numbered by id.")


# -------------------------------------------------------------------- checks
def check_labels(labels: List[dict], root: str, poses: Dict[str, List[dict]],
                 stride: int, margin: float) -> None:
    """Do the agents' real clouds fall inside the hand-placed boxes?

    This is the whole justification for labelling statics rather than deriving
    boxes from poses. A pose-derived box cannot be wrong and cannot check
    anything. A surveyed box can be wrong, and when it is not, agreement between
    two independent agents' returns inside it is evidence that the extrinsics,
    the anchoring and the frame timing are all right at once.
    """
    print("\npoints inside each labelled box, per agent "
          "(margin %.2f m, every %dth frame)" % (margin, stride))
    for label in labels:
        print("  %s  id=%d  at %s  extent %s  yaw %.1f deg"
              % (label.get("name", "?"), label["id"], label["location"],
                 label["extent"], label["angle"][1]))
        for agent, rows in sorted(poses.items()):
            counts, seen_frames = [], 0
            for row in rows[::stride]:
                pcd_path = os.path.join(row["dir"], row["key"] + ".pcd")
                if not os.path.exists(pcd_path):
                    continue
                cloud = read_pcd(pcd_path)[:, :3].astype(np.float64)
                world = cloud @ x_to_world(row["lidar_pose"])[:3, :3].T \
                    + x_to_world(row["lidar_pose"])[:3, 3]
                counts.append(int(points_in_box(world, label, margin).sum()))
                seen_frames += 1
            if not counts:
                continue
            counts_arr = np.array(counts)
            hit = counts_arr > 0
            print("    agent %-4s %4d frames | sees it in %3d (%.0f%%) | points inside: "
                  "median %5.1f  max %5d"
                  % (agent, seen_frames, int(hit.sum()), 100 * hit.mean(),
                     float(np.median(counts_arr[hit])) if hit.any() else 0.0,
                     int(counts_arr.max())))
        print()
    print("Read it this way: an agent that drives past a box and never has a point "
          "inside it, while another agent does, means the box is wrong OR that agent's "
          "geometry is. An agent whose sensor cannot reach (infra radar, a few dozen "
          "points per frame) legitimately sees little.")


# ---------------------------------------------------------------------- main
def parse_seeds(raw: Optional[List[str]]) -> List[List[float]]:
    """[x, y] or [x, y, z] per --seed; z stays absent so callers can supply it."""
    out = []
    for text in (raw or []):
        parts = [float(v) for v in text.replace(";", ",").split(",")]
        if len(parts) not in (2, 3):
            raise SystemExit("--seed wants X,Y or X,Y,Z — got %r" % text)
        out.append(parts)
    return out


def probe_clouds(paths: List[str], args) -> int:
    """Which of these clouds actually contains the objects at the seeds?

    A map pipeline leaves a directory of stages — raw, filtered, downsampled,
    anchored — that all look alike and differ in exactly the way that matters
    here: whether the furniture survived. Rather than guess, read each one and
    say what is standing at the seeds.
    """
    seeds = parse_seeds(args.seed)
    print("\nprobing %d clouds for %d seed(s). A cloud is usable when returns stand\n"
          "well above the floor at every seed." % (len(paths), len(seeds)))
    for path in paths:
        try:
            cloud = read_pcd_xyz(path, max_points=args.max_points)
        except (StaticsError, OSError, ValueError) as exc:
            print("\n%-52s  unreadable: %s" % (os.path.basename(path), exc))
            continue
        ground, _ginfo = ground_level(cloud)
        lo, hi = cloud.min(axis=0), cloud.max(axis=0)
        print("\n%s" % os.path.basename(path))
        print("  %8d points   x %.2f..%.2f  y %.2f..%.2f   floor z = %.3f"
              % (len(cloud), lo[0], hi[0], lo[1], hi[1], ground))
        if not seeds:
            continue
        for index, parts in enumerate(seeds):
            seed = np.array(parts if len(parts) == 3 else parts + [ground + 0.5])
            look = seed_report(cloud, seed, ground, radius=args.radius)
            if not look["points"]:
                print("    seed %d  nothing within %.1f m" % (index + 1, args.radius))
                continue
            standing = sum(c for band_lo, _hi, c in look["bands"]
                           if band_lo >= args.cluster_floor)
            print("    seed %d  %6d standing above %.2f m, tallest %.2f m, nearest "
                  "%.3f m   %s"
                  % (index + 1, standing, args.cluster_floor, look["tallest_m"],
                     look["nearest_m"], "OK" if standing >= 20 else "-- empty"))
    print("\nRe-run with the single cloud that shows OK at both seeds.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pcd", action="append", required=True,
                        help="the run's accumulated map cloud. Repeat it, or give a "
                             "directory, to probe several clouds for the seeds "
                             "instead of fitting")
    parser.add_argument("--dataset", default=None,
                        help="converted dataset root (<out>/test) — used to verify the "
                             "frame and to check labels against the real clouds")
    parser.add_argument("--out", default=None, help="labels json to write/append to")
    parser.add_argument("--labels", default=None, help="labels json to read for --check/--list")
    parser.add_argument("--seed", action="append", default=None, metavar="X,Y[,Z]",
                        help="fit a box at this map position")
    parser.add_argument("--obj-type", default="chair",
                        help="class name written into each label; InCoP's loader "
                             "reads it and defaults to potted_plant without one")
    parser.add_argument("--name", action="append", default=None,
                        help="name for the fitted object; repeat once per --seed")
    parser.add_argument("--id", type=int, default=None,
                        help="object id (must stay below %d)" % RESERVED_MIN)
    parser.add_argument("--radius", type=float, default=1.2,
                        help="how far around the seed to look for the object")
    parser.add_argument("--cluster-voxel", type=float, default=0.06)
    parser.add_argument("--cluster-floor", type=float, default=0.15,
                        help="ignore returns within this of the floor when clustering: "
                             "a real floor is centimetres thick and connects every "
                             "object standing on it to every other one")
    parser.add_argument("--no-ground-extend", action="store_true",
                        help="do NOT extend the box down to the floor (a LiDAR sees a "
                             "chair's seat, not its legs, so the default extends)")
    parser.add_argument("--propose", action="store_true",
                        help="list candidate free-standing objects and their seeds")
    parser.add_argument("--check", action="store_true",
                        help="count each agent's real points inside each labelled box")
    parser.add_argument("--check-stride", type=int, default=25)
    parser.add_argument("--check-margin", type=float, default=0.10)
    parser.add_argument("--interactive", action="store_true",
                        help="pick seeds by shift-click in an Open3D window")
    parser.add_argument("--min-height", type=float, default=0.30,
                        help="proposals: an object's top must be at least this far "
                             "above the floor")
    parser.add_argument("--max-height", type=float, default=1.30,
                        help="proposals: and no higher than this — anything taller is "
                             "wall, door or column, and is excluded rather than "
                             "clustered")
    parser.add_argument("--min-footprint", type=float, default=0.15)
    parser.add_argument("--max-footprint", type=float, default=1.50)
    parser.add_argument("--min-points", type=int, default=80)
    parser.add_argument("--max-proposals", type=int, default=25)
    parser.add_argument("--roi-margin", type=float, default=3.0,
                        help="proposals: metres around the agents' path to search "
                             "(the map usually covers far more building than one run)")
    parser.add_argument("--whole-map", action="store_true",
                        help="propose over the entire cloud, not just where the agents were")
    parser.add_argument("--ceiling", type=float, default=2.0,
                        help="ignore returns more than this far above the floor: a "
                             "scanned ceiling is the tallest thing over every cell "
                             "beneath it and hides the furniture standing there")
    parser.add_argument("--max-wall-contact", type=float, default=1.0,
                        help="drop proposals whose outline is more than this "
                             "fraction structure (0.3 keeps furniture only)")
    parser.add_argument("--roi-from-start", action="store_true",
                        help="search the box spanned by the agents' FIRST poses "
                             "rather than their whole path")
    parser.add_argument("--map-image", default=None, metavar="PATH.png",
                        help="write a top-down picture of the searched area with the "
                             "proposals numbered on it")
    parser.add_argument("--map-scale", type=int, default=4,
                        help="pixels per height-map cell in --map-image")
    parser.add_argument("--cell", type=float, default=0.10,
                        help="proposals: height-map cell size")
    parser.add_argument("--max-points", type=int, default=4000000)
    args = parser.parse_args()

    clouds: List[str] = []
    for entry in args.pcd:
        if os.path.isdir(entry):
            clouds.extend(sorted(glob.glob(os.path.join(entry, "*.pcd"))))
        else:
            clouds.append(entry)
    if not clouds:
        print("no .pcd found in %s" % ", ".join(args.pcd))
        return 2
    if len(clouds) > 1:
        return probe_clouds(clouds, args)
    args.pcd = clouds[0]
    cloud = read_pcd_xyz(args.pcd, max_points=args.max_points)
    poses = load_dataset_poses(args.dataset, limit=400) if args.dataset else {}

    print("=" * 78)
    if poses:
        if not verify_frame(cloud, poses):
            return 2
    else:
        lo, hi = cloud.min(axis=0), cloud.max(axis=0)
        print("map cloud   : %d points, x %.2f..%.2f  y %.2f..%.2f  z %.2f..%.2f"
              % (len(cloud), lo[0], hi[0], lo[1], hi[1], lo[2], hi[2]))
        print("  (pass --dataset to verify this cloud is in the dataset's frame)")

    ground_z, ginfo = ground_level(cloud)
    print("\nfloor       : z = %+.3f m in the map frame  (%s)" % (ground_z, ginfo["method"]))
    print("  The map's z origin is the surveyed anchor board, not the ground, so this is")
    print("  the number `cloud.ground_lift` needs, and nothing else in the dataset")
    print("  supplies it.")
    if poses:
        for agent, rows in sorted(poses.items()):
            sensor_z = float(np.median([r["lidar_pose"][2] for r in rows]))
            height = sensor_z - ground_z
            print("    agent %-4s sensor at map z %+.3f = %.2f m above the floor"
                  "   -> ground_lift: %.2f  (1.9 - %.2f, to match OPV2V's car-roof prior)"
                  % (agent, sensor_z, height, max(1.9 - height, 0.0), height))
        print("    Set it per agent in the config only if you intend to run "
              "OPV2V-pretrained\n    checkpoints; 0 is a valid, documented choice otherwise.")

    labels: List[dict] = []
    source = args.labels or args.out
    if source and os.path.exists(source):
        with open(source) as handle:
            labels = json.load(handle)
        print("\nlabels      : %d loaded from %s" % (len(labels), source))
        for label in labels:
            print("  id=%-5d %-14s at %s  extent %s  yaw %6.1f"
                  % (label["id"], label.get("name", ""), label["location"],
                     label["extent"], label["angle"][1]))

    if args.propose:
        roi, roi_kind = None, ""
        if poses and not args.whole_map:
            if args.roi_from_start:
                roi, roi_kind = start_roi(poses, args.roi_margin), "started"
            if roi is None:
                roi, roi_kind = trajectory_roi(poses, args.roi_margin), "were"
        if roi is not None:
            print("\nsearching only where the agents %s: x %.2f..%.2f  y %.2f..%.2f "
                  "(plus %.1f m). --whole-map searches the entire cloud."
                  % (roi_kind, roi[0], roi[2], roi[1], roi[3], args.roi_margin))
        print("candidate free-standing objects, top between %.2f and %.2f m above the "
              "floor\n(anything taller is structure and is excluded, not clustered):"
              % (args.min_height, args.max_height))
        hmap = HeightMap(cloud, ground_z, args.cell, roi, ceiling=args.ceiling)
        if hmap.overhead:
            print("ignored %d returns above %.2f m (--ceiling): overhead structure is "
                  "the tallest\nthing over the floor it covers, and would hide "
                  "everything standing on it."
                  % (hmap.overhead, args.ceiling))
        if not hmap.empty():
            structure = hmap.structure_fraction(args.max_height)
            print("%.0f%% of the mapped cells here are taller than %.2f m." %
                  (100 * structure, args.max_height))
            if structure > 0.6:
                print("  ! that is most of the area. Free-standing objects need floor "
                      "around them,\n    so lower --ceiling until this drops — "
                      "something overhead is still in the map.")
        rows = propose(cloud, ground_z, args, roi, hmap)
        if not rows:
            print("  none — widen --min-footprint/--max-footprint or lower --min-points")
        print("   #  seed              pts   footprint     top     walled")
        for i, row in enumerate(rows):
            print("  %2d. %-16s %5d  %-12s  %.2f m  %3d%%%s"
                  % (i + 1, "%.2f,%.2f" % tuple(row["centre"]), row["points"],
                     "%.2f x %.2f" % tuple(row["footprint_m"]), row["top_m"],
                     int(round(100 * row["wall_contact"])),
                     "" if row["wall_contact"] <= 0.25 else
                     "   <- mostly wall foot" if row["wall_contact"] >= 0.5 else
                     "   (against structure)"))
        print("\n  'walled' is how much of the blob's outline is taller-than-%.2f m "
              "structure." % args.max_height)
        print("  A chair standing in the room reads 0%%; against a wall, up to ~30%%; "
              "a wall's own")
        print("  foot reads 50%% and up and is not furniture.")
        if args.map_image and not hmap.empty():
            render_map(args.map_image, hmap, rows, args, poses)
        print("\n  Pick the two chairs and fit them:")
        print("    --seed <x,y> --name chair_1 --out labels/coop2_statics.json")

    seeds: List[np.ndarray] = [
        np.array(parts if len(parts) == 3 else parts + [ground_z + 0.5], dtype=np.float64)
        for parts in parse_seeds(args.seed)]
    if args.interactive:
        seeds.extend(pick_interactive(cloud))
    names = list(args.name or [])
    if names and len(names) not in (1, len(seeds)):
        print("\n! %d --name for %d --seed: give one name each, or none"
              % (len(names), len(seeds)))
        return 2

    for index, seed in enumerate(seeds):
        # The same cap the search uses. A chair standing under a ceiling, a beam
        # or a shelf is only separated from it by empty air, and empty air is all
        # the flood fill needs — until a pole, a cable tray or the wall behind it
        # bridges the gap, and then the box swallows the building. Gating the
        # height costs nothing: the object is below the cap by definition.
        ceiling_z = (ground_z + args.ceiling) if args.ceiling else None
        look = seed_report(cloud, seed, ground_z, radius=args.radius)
        print("\nat seed %s (%.2f m above the floor), within %.1f m:"
              % ("%.3f,%.3f,%.3f" % tuple(seed), look["seed_height_m"], args.radius))
        if not look["points"]:
            print("  nothing at all — the seed is not in this cloud")
            return 3
        print("  %d points, nearest %.3f m from the seed, tallest %.2f m above the floor"
              % (look["points"], look["nearest_m"], look["tallest_m"]))
        for lo, hi, count in look["bands"]:
            if count:
                bar = "#" * min(40, 1 + count // max(1, look["points"] // 40))
                print("    %4.2f-%-5s m %6d  %s"
                      % (lo, "inf" if hi > 100 else "%.2f" % hi, count, bar))
        # A seed picked in one cloud and clustered in another is otherwise silent:
        # the fill finds the floor, the box comes back floor-shaped, and nothing
        # says the object was never here.
        standing = sum(c for lo, _hi, c in look["bands"] if lo >= args.cluster_floor)
        if standing < 20:
            print("  ! only %d returns stand more than %.2f m off the floor here."
                  % (standing, args.cluster_floor))
            print("    Nothing is at this seed IN THIS FILE. If the coordinates came "
                  "from a viewer,\n    check it was this same .pcd — a seed read off "
                  "another cloud lands on bare floor.")
            return 3
        try:
            cluster, cinfo = cluster_at(cloud, seed, radius=args.radius,
                                        voxel=args.cluster_voxel,
                                        z_min=ground_z + args.cluster_floor,
                                        z_max=ceiling_z)
        except StaticsError as exc:
            print("\n! %s" % exc)
            return 3
        box = fit_box(cluster, ground_z=ground_z,
                      sit_on_ground=not args.no_ground_extend)
        name = (names[index] if len(names) == len(seeds) else
                "%s_%d" % (names[0], index + 1) if names else None)
        # A name is the object's identity across runs. Re-fitting chair_1 with a
        # better seed must REPLACE chair_1, not stand a second box beside it —
        # every duplicate becomes a phantom object in all 1330 frames, and it is
        # a phantom that scores as a miss for every detector that gets it right.
        prior = [label for label in labels if name and label.get("name") == name]
        if prior:
            print("  replacing %s (id %s) from a previous run"
                  % (name, ", ".join(str(p["id"]) for p in prior)))
            labels = [label for label in labels if label.get("name") != name]
        used = {label["id"] for label in labels}
        if args.id is not None:
            box["id"] = args.id + index
        elif prior:
            box["id"] = min(p["id"] for p in prior)
        else:
            box["id"] = next(i for i in range(1, RESERVED_MIN) if i not in used)
        box["name"] = name or ("object_%d" % box["id"])
        near = [label for label in labels
                if float(np.linalg.norm(np.array(label["location"][:2])
                                        - np.array(box["location"][:2]))) < 0.25]
        if near:
            print("  ! %s stands within 0.25 m of %s (id %d) — two labels on one "
                  "object?" % (box["name"], near[0].get("name", "?"), near[0]["id"]))
        box["obj_type"] = args.obj_type
        box["source"] = {"pcd": os.path.abspath(args.pcd),
                         "seed": [round(float(v), 3) for v in seed],
                         "cluster": cinfo, "ground_z": round(ground_z, 4),
                         "ceiling": args.ceiling}
        width, length, height = (2 * box["extent"][1], 2 * box["extent"][0],
                                 2 * box["extent"][2])
        print("\nfitted %s (id %d)" % (box["name"], box["id"]))
        print("  centre   %s   (map frame)" % box["location"])
        print("  extent   %s   = %.2f x %.2f x %.2f m"
              % (box["extent"], length, width, height))
        print("  yaw      %.1f deg" % box["angle"][1])
        print("  fitted from %d points of %d within %.1f m; %s"
              % (box["fit"]["points"], cinfo["points_in_radius"], args.radius,
                 "extended down to the floor" if box["fit"]["extended_to_ground"]
                 else "not extended (points already reach the floor)"))
        # A merge with the wall or the floor shows up as a box that is not
        # chair-shaped, and it shows up here rather than in the trained model.
        if max(length, width) > args.max_footprint:
            print("  ! %.2f m across, wider than --max-footprint %.2f: the cluster "
                  "probably ran\n    into a wall or the floor. Shrink --radius (now "
                  "%.2f) or --cluster-voxel (now %.2f)."
                  % (max(length, width), args.max_footprint, args.radius,
                     args.cluster_voxel))
        if height < 0.25:
            print("  ! %.2f m tall: that is a slab on the floor, not an object. The "
                  "cluster ran\n    across the ground plane — raise --cluster-floor "
                  "(now %.2f)." % (height, args.cluster_floor))
        if height > args.ceiling:
            print("  ! %.2f m tall: something overhead is in the cluster. "
                  "Lower --ceiling (now %.2f)." % (height, args.ceiling))
        if box["id"] >= RESERVED_MIN:
            print("  ! id >= %d collides with agent-derived boxes" % RESERVED_MIN)
        labels = [label for label in labels if label["id"] != box["id"]] + [box]

    if seeds and args.map_image:
        render_seeds(args.map_image, cloud, ground_z, labels, args, poses)

    if args.out and seeds:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as handle:
            json.dump(labels, handle, indent=2)
        print("\nwrote %s (%d object(s))" % (args.out, len(labels)))
        print("  add to the converter config, then re-run convert_rosbag.py:")
        print("    labels_file: %s" % args.out)

    if args.check:
        if not labels:
            print("\nnothing to check — pass --labels")
            return 1
        if not poses:
            print("\n--check needs --dataset")
            return 1
        check_labels(labels, args.dataset, poses, args.check_stride, args.check_margin)
    return 0


def pick_interactive(cloud: np.ndarray) -> List[np.ndarray]:
    """Shift-click seeds in an Open3D window. Optional: a seed is just an x,y."""
    try:
        import open3d as o3d
    except ImportError:
        print("\n! --interactive needs open3d (pip install open3d). Use --propose "
              "instead, or read an x,y off any viewer.")
        return []
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cloud)
    print("\nShift+click each object, then close the window (q).")
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window(window_name="pick a point on each static object")
    vis.add_geometry(pcd)
    vis.run()
    vis.destroy_window()
    return [cloud[i] for i in vis.get_picked_points()]


if __name__ == "__main__":
    raise SystemExit(main())
