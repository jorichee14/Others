#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The two JSON files InCoP's isaacsim loader reads beside a dataset.

Neither is derivable from the frames, and both fail quietly rather than loudly:
a missing modality entry raises a KeyError deep inside the loader, and a missing
class map sends every object to class id 0 -- `potted_plant` -- so a dataset of
chairs trains and evaluates as a dataset of plants.

    python3 scripts/make_incop_sidecars.py --root ~/.../InCoP/mirc_coop2

Writes <root>/heter_modality_assign.json and <root>/isaacsim_class_map.json,
covering every scenario folder found under every split.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

# InCoP's own indoor classes, in the order its loader indexes them. `chair` is 1.
CLASS_NAMES = ["potted_plant", "chair", "medical_bag", "traffic_cone",
               "wet_floor_sign", "fire_extinguisher", "trash_can"]

SPLITS = ("train", "validate", "test")


def scenarios(root: str):
    """{scenario_name: [agent folder, ...]} across every split under `root`."""
    found = {}
    for split in SPLITS:
        split_dir = os.path.join(root, split)
        if not os.path.isdir(split_dir):
            continue
        for name in sorted(os.listdir(split_dir)):
            path = os.path.join(split_dir, name)
            if not os.path.isdir(path):
                continue
            agents = sorted(a for a in os.listdir(path)
                            if os.path.isdir(os.path.join(path, a)))
            # A scenario appearing in two splits must hold the same agents, or
            # one modality map cannot describe both.
            if name in found and found[name] != agents:
                raise SystemExit(
                    "scenario %r has agents %s in one split and %s in another; "
                    "one modality assignment cannot cover both"
                    % (name, found[name], agents))
            found[name] = agents
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", required=True,
                        help="dataset root holding train/ validate/ test/")
    parser.add_argument("--modality", default="m1",
                        help="modality name every agent is assigned (default m1, "
                             "the multimodal LiDAR+camera branch)")
    parser.add_argument("--training-mode", default="class_agnostic")
    args = parser.parse_args()

    root = os.path.expanduser(args.root)
    if not os.path.isdir(root):
        print("no such directory: %s" % root, file=sys.stderr)
        return 2

    found = scenarios(root)
    if not found:
        print("no scenarios under %s/{%s}" % (root, ",".join(SPLITS)), file=sys.stderr)
        return 2

    assign = {name: {agent: args.modality for agent in agents}
              for name, agents in found.items()}
    class_map = {
        "class_names": list(CLASS_NAMES),
        "class_to_id": {name: i for i, name in enumerate(CLASS_NAMES)},
        "detection_class": "object",
        "training_mode": args.training_mode,
    }

    for filename, payload in (("heter_modality_assign.json", assign),
                              ("isaacsim_class_map.json", class_map)):
        path = os.path.join(root, filename)
        with open(path, "w") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        print("wrote %s" % path)

    for name, agents in sorted(found.items()):
        print("  %-16s agents %s -> %s" % (name, agents, args.modality))
    print("\n  chair is class id %d" % CLASS_NAMES.index("chair"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
