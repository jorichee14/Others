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
    python3 scripts/label_static.py --pcd map.pcd --dataset <root>/test --propose

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
                               ground_level, points_in_box, read_pcd_xyz)
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


def propose(cloud: np.ndarray, ground_z: float, args,
            roi: Optional[np.ndarray] = None) -> List[dict]:
    """Candidate free-standing objects, so a seed can be read off rather than hunted.

    Deliberately crude — everything in a height band above the floor, voxelised,
    connected, filtered by footprint. It is not a detector and does not need to
    be: it turns "find the chairs in a 16 m cloud" into "pick two rows from a
    short list", and the operator decides which are real.
    """
    z_lo = ground_z + args.min_height
    z_hi = ground_z + args.max_height
    keep = (cloud[:, 2] > z_lo) & (cloud[:, 2] < z_hi)
    if roi is not None:
        keep &= ((cloud[:, 0] >= roi[0]) & (cloud[:, 0] <= roi[2])
                 & (cloud[:, 1] >= roi[1]) & (cloud[:, 1] <= roi[3]))
    band = cloud[keep]
    if len(band) < 50:
        print("no points in the %.2f..%.2f m band above the floor" % (args.min_height,
                                                                     args.max_height))
        return []
    voxel = args.voxel
    grid = np.floor(band[:, :2] / voxel).astype(np.int64)
    grid -= grid.min(axis=0)
    keys = (grid[:, 0].astype(np.int64) << 21) | grid[:, 1]
    unique, inverse, counts = np.unique(keys, return_inverse=True, return_counts=True)
    coords = np.stack([(unique >> 21) & 0x1FFFFF, unique & 0x1FFFFF], axis=1)
    lookup = {int(k): i for i, k in enumerate(unique)}

    seen = np.zeros(len(unique), dtype=bool)
    proposals = []
    for start in np.argsort(-counts):
        if seen[start]:
            continue
        stack, members = [int(start)], []
        seen[start] = True
        while stack:
            current = stack.pop()
            members.append(current)
            cx, cy = coords[current]
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    nxt = lookup.get(int(((cx + dx) << 21) | (cy + dy)))
                    if nxt is not None and not seen[nxt]:
                        seen[nxt] = True
                        stack.append(nxt)
        mask = np.isin(inverse, members)
        blob = band[mask]
        if len(blob) < args.min_points:
            continue
        span = blob[:, :2].max(axis=0) - blob[:, :2].min(axis=0)
        if max(span) > args.max_footprint or max(span) < args.min_footprint:
            continue
        proposals.append({
            "centre": [round(float(v), 3) for v in blob[:, :2].mean(axis=0)],
            "points": int(len(blob)),
            "footprint_m": [round(float(span[0]), 2), round(float(span[1]), 2)],
            "top_m": round(float(blob[:, 2].max() - ground_z), 2),
        })
    proposals.sort(key=lambda p: -p["points"])
    return proposals[:args.max_proposals]


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
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pcd", required=True, help="the run's accumulated map cloud")
    parser.add_argument("--dataset", default=None,
                        help="converted dataset root (<out>/test) — used to verify the "
                             "frame and to check labels against the real clouds")
    parser.add_argument("--out", default=None, help="labels json to write/append to")
    parser.add_argument("--labels", default=None, help="labels json to read for --check/--list")
    parser.add_argument("--seed", default=None, metavar="X,Y[,Z]",
                        help="fit a box at this map position")
    parser.add_argument("--name", default=None, help="name for the fitted object")
    parser.add_argument("--id", type=int, default=None,
                        help="object id (must stay below %d)" % RESERVED_MIN)
    parser.add_argument("--radius", type=float, default=1.2,
                        help="how far around the seed to look for the object")
    parser.add_argument("--cluster-voxel", type=float, default=0.06)
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
    parser.add_argument("--min-height", type=float, default=0.15,
                        help="proposals: metres above the floor to start looking")
    parser.add_argument("--max-height", type=float, default=1.60)
    parser.add_argument("--min-footprint", type=float, default=0.15)
    parser.add_argument("--max-footprint", type=float, default=1.50)
    parser.add_argument("--min-points", type=int, default=80)
    parser.add_argument("--max-proposals", type=int, default=25)
    parser.add_argument("--roi-margin", type=float, default=3.0,
                        help="proposals: metres around the agents' path to search "
                             "(the map usually covers far more building than one run)")
    parser.add_argument("--whole-map", action="store_true",
                        help="propose over the entire cloud, not just where the agents were")
    parser.add_argument("--voxel", type=float, default=0.08)
    parser.add_argument("--max-points", type=int, default=4000000)
    args = parser.parse_args()

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
        roi = trajectory_roi(poses, args.roi_margin) if poses and not args.whole_map else None
        if roi is not None:
            print("\nsearching only where the agents were: x %.2f..%.2f  y %.2f..%.2f "
                  "(their path plus %.1f m). --whole-map searches the entire cloud."
                  % (roi[0], roi[2], roi[1], roi[3], args.roi_margin))
        print("candidate free-standing objects between %.2f and %.2f m above the floor:"
              % (args.min_height, args.max_height))
        rows = propose(cloud, ground_z, args, roi)
        if not rows:
            print("  none — widen --min-footprint/--max-footprint or lower --min-points")
        for i, row in enumerate(rows):
            print("  %2d. seed %-18s %5d pts  footprint %-12s top %.2f m above floor"
                  % (i + 1, "%.2f,%.2f" % tuple(row["centre"]), row["points"],
                     "%.2f x %.2f" % tuple(row["footprint_m"]), row["top_m"]))
        print("\n  Pick the two chairs and fit them:")
        print("    --seed <x,y> --name chair_1 --out labels/coop2_statics.json")

    seeds: List[np.ndarray] = []
    if args.seed:
        parts = [float(v) for v in args.seed.split(",")]
        if len(parts) == 2:
            parts.append(ground_z + 0.5)
        seeds.append(np.array(parts, dtype=np.float64))
    if args.interactive:
        seeds.extend(pick_interactive(cloud))

    for seed in seeds:
        try:
            cluster, cinfo = cluster_at(cloud, seed, radius=args.radius,
                                        voxel=args.cluster_voxel,
                                        z_min=ground_z + 0.03)
        except StaticsError as exc:
            print("\n! %s" % exc)
            return 3
        box = fit_box(cluster, ground_z=ground_z,
                      sit_on_ground=not args.no_ground_extend)
        used = {label["id"] for label in labels}
        box["id"] = args.id if args.id is not None else next(
            i for i in range(1, RESERVED_MIN) if i not in used)
        box["name"] = args.name or ("object_%d" % box["id"])
        box["source"] = {"pcd": os.path.abspath(args.pcd),
                         "seed": [round(float(v), 3) for v in seed],
                         "cluster": cinfo, "ground_z": round(ground_z, 4)}
        print("\nfitted %s (id %d)" % (box["name"], box["id"]))
        print("  centre   %s   (map frame)" % box["location"])
        print("  extent   %s   = %.2f x %.2f x %.2f m"
              % (box["extent"], 2 * box["extent"][0], 2 * box["extent"][1],
                 2 * box["extent"][2]))
        print("  yaw      %.1f deg" % box["angle"][1])
        print("  fitted from %d points; %s"
              % (box["fit"]["points"],
                 "extended down to the floor" if box["fit"]["extended_to_ground"]
                 else "not extended (points already reach the floor)"))
        if box["id"] >= RESERVED_MIN:
            print("  ! id >= %d collides with agent-derived boxes" % RESERVED_MIN)
        labels = [label for label in labels if label["id"] != box["id"]] + [box]

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
