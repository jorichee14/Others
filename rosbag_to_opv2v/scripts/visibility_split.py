#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Who sees what: the ego-only / partner-only / shared split of the ground truth.

InCoP's headline is not its mAP, it is `visibility_subset_metrics`: on their
hospital set 22.6% of labelled objects are visible to the partner and not to
the ego, and cooperative fusion is what recovers them.  A dataset where the two
robots mostly look at the same things cannot support that claim however many
objects it labels, so this measures the split before any of that work is done.

An object counts as seen by an agent when it is inside that agent's detection
range AND that agent's own returns fall inside the box -- in range but occluded
is not seen, and that distinction is the whole point here.

    python3 scripts/visibility_split.py --root <split> --stride 10
    python3 scripts/visibility_split.py --root <split> --min-points 1 --range 0 -11.2 -1 22.4 11.2 3
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np                                              # noqa: E402
import yaml                                                     # noqa: E402

from ros2opv2v.geometry import invert, x_to_world                # noqa: E402
from ros2opv2v.writers import read_pcd                           # noqa: E402

DEFAULT_RANGE = [0.0, -11.2, -1.0, 22.4, 11.2, 3.0]


def points_in_box(points, box, margin=0.0):
    """Mask of points inside a labelled box, in the frame the box is given in.

    The same convention label_static.py verifies against: `location` is the
    centre, `extent` half-sizes, `angle[1]` yaw in degrees.
    """
    if len(points) == 0:
        return np.zeros(0, dtype=bool)
    yaw = np.radians(float(box["angle"][1]))
    cos, sin = np.cos(yaw), np.sin(yaw)
    rel = np.asarray(points, dtype=np.float64)[:, :3] - \
        np.asarray(box["location"], dtype=np.float64)
    local = np.stack([rel[:, 0] * cos + rel[:, 1] * sin,
                      -rel[:, 0] * sin + rel[:, 1] * cos,
                      rel[:, 2]], axis=1)
    limits = np.asarray(box["extent"], dtype=np.float64) + margin
    return (np.abs(local) <= limits).all(axis=1)


def centre_in_range(box, world_from_agent, box_range):
    """Is the box centre inside this agent's detection range?"""
    centre = np.append(np.asarray(box["location"], dtype=np.float64), 1.0)
    local = invert(world_from_agent) @ centre
    return (box_range[0] <= local[0] <= box_range[3] and
            box_range[1] <= local[1] <= box_range[4] and
            box_range[2] <= local[2] <= box_range[5])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", required=True, help="split dir holding case_*/")
    parser.add_argument("--ego", default=None, help="ego agent folder (default: lowest id)")
    parser.add_argument("--stride", type=int, default=10,
                        help="read every Nth frame (reading two clouds a frame is the cost)")
    parser.add_argument("--min-points", type=int, default=1,
                        help="returns inside the box for an agent to count as seeing it")
    parser.add_argument("--range", type=float, nargs=6, default=DEFAULT_RANGE,
                        metavar=("XMIN", "YMIN", "ZMIN", "XMAX", "YMAX", "ZMAX"))
    args = parser.parse_args()

    root = os.path.expanduser(args.root)
    scenarios = sorted(d for d in os.listdir(root)
                       if os.path.isdir(os.path.join(root, d)))
    totals = Counter()
    per_object = defaultdict(Counter)
    frames_read = 0

    for scenario_name in scenarios:
        scenario = os.path.join(root, scenario_name)
        agents = sorted((d for d in os.listdir(scenario)
                         if os.path.isdir(os.path.join(scenario, d))), key=int)
        if len(agents) < 2:
            print(f"{scenario_name}: {len(agents)} agent(s), skipped", file=sys.stderr)
            continue
        ego = args.ego or min((a for a in agents if int(a) >= 0), key=int)
        partner = next(a for a in agents if a != ego)
        ego_dir, partner_dir = os.path.join(scenario, ego), os.path.join(scenario, partner)

        keys = sorted(f[:-5] for f in os.listdir(ego_dir) if f.endswith(".yaml"))
        for key in keys[::max(1, args.stride)]:
            paths = {a: os.path.join(scenario, a, key) for a in (ego, partner)}
            if not all(os.path.exists(p + ".yaml") and os.path.exists(p + ".pcd")
                       for p in paths.values()):
                continue
            params, clouds = {}, {}
            for agent, stem in paths.items():
                with open(stem + ".yaml", encoding="utf-8") as handle:
                    params[agent] = yaml.safe_load(handle)
                world = x_to_world(params[agent]["lidar_pose"])
                points = read_pcd(stem + ".pcd")[:, :3]
                # The agent's returns carried into the world, where the boxes live.
                homogeneous = np.column_stack([points, np.ones(len(points))])
                clouds[agent] = (homogeneous @ world.T)[:, :3], world
            frames_read += 1

            vehicles = dict(params[ego].get("vehicles") or {})
            vehicles.update(params[partner].get("vehicles") or {})
            for object_id, box in vehicles.items():
                seen = {}
                for agent in (ego, partner):
                    points, world = clouds[agent]
                    if not centre_in_range(box, world, args.range):
                        seen[agent] = False
                        continue
                    seen[agent] = int(points_in_box(points, box).sum()) >= args.min_points
                if seen[ego] and seen[partner]:
                    bucket = "shared"
                elif seen[ego]:
                    bucket = "ego_only"
                elif seen[partner]:
                    bucket = "partner_only"
                else:
                    bucket = "neither"
                totals[bucket] += 1
                per_object[object_id][bucket] += 1

    counted = sum(totals[b] for b in ("ego_only", "partner_only", "shared"))
    print(f"\n{frames_read} frames read (stride {args.stride}), ego={args.ego or 'lowest id'}")
    if not counted:
        print("no object was seen by either agent in any sampled frame")
        return 1
    print(f"union GT observations: {counted}  "
          f"(+{totals['neither']} in no agent's view)\n")
    for bucket in ("ego_only", "partner_only", "shared"):
        share = 100.0 * totals[bucket] / counted
        print(f"  {bucket:<13} {totals[bucket]:6d}  {share:5.1f}%")
    print(f"\n  InCoP hospital for reference: ego_only 45.0%, partner_only 22.6%, "
          f"shared 32.4%")

    print("\nper object:")
    for object_id in sorted(per_object, key=lambda k: str(k)):
        counts = per_object[object_id]
        total = sum(counts[b] for b in ("ego_only", "partner_only", "shared"))
        if not total:
            print(f"  {str(object_id):<8} never seen")
            continue
        print(f"  {str(object_id):<8} ego_only {counts['ego_only']:5d}  "
              f"partner_only {counts['partner_only']:5d}  shared {counts['shared']:5d}"
              f"   (partner_only {100.0 * counts['partner_only'] / total:4.1f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
