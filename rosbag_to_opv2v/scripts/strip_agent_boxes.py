#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Drop agent boxes from a converted tree's ground truth, and name the classes.

Written for trees produced before ``include_agents_in_vehicles`` existed.  The
converter now leaves the agents out at write time, so this is for data already
on disk: re-running the conversion gives the same result and costs a bag read.

Two edits per frame yaml, both confined to ``vehicles``:

* remove the agent-derived boxes.  The converter reserves ids 10000+ for CAVs
  and 20000+ for RSUs precisely so they can be told apart from a labelling
  tool's small ids, so the id *is* the discriminator -- no guessing by shape.
  They carry no class, and a class-less box is not read as unknown: InCoP's
  loader resolves it to class id 0, so an agent left in is a potted plant.
* copy ``obj_type`` to ``class_name``, which the loader reads first.

    python scripts/strip_agent_boxes.py --root .../train --dry-run
"""

from __future__ import annotations

import argparse
import os

import yaml

CAV_ID_FLOOR = 10000            # ros2opv2v.labels reserves 10000+/20000+


def frame_yamls(root: str):
    for scenario in sorted(os.listdir(root)):
        scenario_path = os.path.join(root, scenario)
        if not os.path.isdir(scenario_path):
            continue
        for agent in sorted(os.listdir(scenario_path)):
            agent_path = os.path.join(scenario_path, agent)
            if not os.path.isdir(agent_path):
                continue
            for name in sorted(os.listdir(agent_path)):
                if name.endswith(".yaml"):
                    yield os.path.join(agent_path, name)


def fix_one(path: str, dry_run: bool):
    with open(path, encoding="utf-8") as handle:
        params = yaml.safe_load(handle) or {}
    vehicles = params.get("vehicles")
    if not isinstance(vehicles, dict):
        return 0, 0

    dropped, named = 0, 0
    for object_id in list(vehicles):
        try:
            numeric = int(object_id)
        except (TypeError, ValueError):
            numeric = -1
        if numeric >= CAV_ID_FLOOR:
            del vehicles[object_id]
            dropped += 1
            continue
        entry = vehicles[object_id]
        if isinstance(entry, dict) and entry.get("obj_type") \
                and not entry.get("class_name"):
            entry["class_name"] = entry["obj_type"]
            named += 1

    if (dropped or named) and not dry_run:
        with open(path, "w", encoding="utf-8") as handle:
            yaml.safe_dump(params, handle, default_flow_style=False)
    return dropped, named


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", required=True, help="split directory")
    parser.add_argument("--dry-run", action="store_true",
                        help="report what would change, write nothing")
    args = parser.parse_args()

    root = os.path.expanduser(args.root)
    if not os.path.isdir(root):
        print(f"not a directory: {root}")
        return 2

    files = dropped = named = 0
    for path in frame_yamls(root):
        files += 1
        one_dropped, one_named = fix_one(path, args.dry_run)
        dropped += one_dropped
        named += one_named

    verb = "would drop" if args.dry_run else "dropped"
    print(f"{files} frame yamls: {verb} {dropped} agent box(es), "
          f"named {named} class(es)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
