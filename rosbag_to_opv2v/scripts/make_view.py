#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A view of a converted dataset: the frames that satisfy a condition, linked
into a second tree the loaders read like any other.

One conversion on the LiDAR grid holds every frame; a camera far slower than
the grid (the ZED at 1.3 Hz on a 10 Hz grid) has its image in one frame of
eight, and the yaml says which (``camera0.image_present``). The camera arm
trains and validates on those frames only, so this script builds a tree of
them. Frames keep their keys, so a frame in the view is the same frame in the
full tree, and every agent's folder is filtered by the same keys, which keeps
the folders aligned (OpenCOOD indexes every agent by the ego's keys).

    python3 scripts/make_view.py --root <tree> --out <tree>_zed \\
        --require-image 1:0            # agent folder 1 must have camera0's image
        [--from-key 000000 --to-key 001485]   # a time range: the keys run in order
        [--split train] [--as-split validate] # the split folder to read and to write
        [--copy]                              # copy instead of symlink

A time split costs nothing this way: one conversion holds every frame, and
train and validate are two views of it, each a folder of symlinks with the
frames' own keys. Frames 10 Hz apart are near copies, so a random split is
the training set with noise; a key range is a time range.

Sidecars at the root (heter_modality_assign.json, isaacsim_class_map.json)
and the root's comms/ folder are linked too, and conversion_report.json
is written beside a view_report.json saying what was kept.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ros2opv2v.writers import read_pcd                            # noqa: E402

DEFAULT_RANGE = [0.0, -11.2, -1.0, 22.4, 11.2, 3.0]

SPLITS = ("train", "validate", "test")


def _link(src: str, dst: str, copy: bool) -> None:
    if os.path.lexists(dst):
        return
    if copy:
        if os.path.isdir(src):
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
    else:
        os.symlink(os.path.abspath(src), dst)


def frame_keys(agent_dir: str):
    return sorted(f[:-5] for f in os.listdir(agent_dir) if f.endswith(".yaml"))


def points_in_range(path: str, box) -> int:
    """Returns inside the detection range, in the cloud's own frame."""
    points = read_pcd(path)[:, :3]
    if not len(points):
        return 0
    keep = ((points[:, 0] >= box[0]) & (points[:, 0] <= box[3]) &
            (points[:, 1] >= box[1]) & (points[:, 1] <= box[4]) &
            (points[:, 2] >= box[2]) & (points[:, 2] <= box[5]))
    return int(keep.sum())


def keep_frame(scenario_dir: str, key: str, require_image, from_key=None, to_key=None,
               min_points=0, box=None) -> bool:
    if from_key and key < from_key:
        return False
    if to_key and key > to_key:
        return False
    if min_points > 0:
        # A LiDAR has a blind radius (1.3 m on these Ousters), so a robot nose
        # to a wall returns nothing inside a forward detection range: the wall
        # is in the blind zone and there is nothing beyond it. The model cannot
        # form a feature from no points -- it raises an IndexError assembling
        # the batch -- so such a frame is dropped rather than widening the range
        # to hide it. Every agent is checked: one empty agent breaks the frame.
        for agent in sorted(os.listdir(scenario_dir)):
            agent_dir = os.path.join(scenario_dir, agent)
            if not os.path.isdir(agent_dir):
                continue
            pcd = os.path.join(agent_dir, f"{key}.pcd")
            if not os.path.exists(pcd):
                return False
            if points_in_range(pcd, box) < min_points:
                return False
    for agent, slot in require_image:
        path = os.path.join(scenario_dir, agent, f"{key}.yaml")
        if not os.path.exists(path):
            return False
        with open(path) as handle:
            params = yaml.safe_load(handle) or {}
        cam = params.get(f"camera{slot}") or {}
        if not cam.get("image_present", True):
            return False
        if not os.path.exists(os.path.join(scenario_dir, agent, f"{key}_camera{slot}.png")):
            return False
    return True


def make_view(root: str, out: str, require_image, splits=SPLITS, copy=False, log=print,
              from_key=None, to_key=None, as_split=None, min_points=0, box=None) -> dict:
    root, out = os.path.expanduser(root), os.path.expanduser(out)
    report = {"root": os.path.abspath(root), "require_image": [f"{a}:{s}" for a, s in require_image],
              "from_key": from_key, "to_key": to_key, "splits": {}}
    os.makedirs(out, exist_ok=True)
    for name in os.listdir(root):
        src = os.path.join(root, name)
        if os.path.isfile(src) or name == "comms":    # sidecars, and the link tables
            _link(src, os.path.join(out, name), copy)
    for split in splits:
        split_dir = os.path.join(root, split)
        if not os.path.isdir(split_dir):
            continue
        for scenario in sorted(os.listdir(split_dir)):
            sdir = os.path.join(split_dir, scenario)
            if not os.path.isdir(sdir):
                continue
            agents = sorted(d for d in os.listdir(sdir)
                            if os.path.isdir(os.path.join(sdir, d)) and _is_int(d))
            if not agents:
                continue
            keys = frame_keys(os.path.join(sdir, agents[0]))
            kept = [k for k in keys if keep_frame(sdir, k, require_image, from_key, to_key,
                                                  min_points, box)]
            odir = os.path.join(out, as_split or split, scenario)
            for agent in agents:
                os.makedirs(os.path.join(odir, agent), exist_ok=True)
                for k in kept:
                    for f in os.listdir(os.path.join(sdir, agent)):
                        if f.startswith(k + ".") or f.startswith(k + "_"):
                            _link(os.path.join(sdir, agent, f), os.path.join(odir, agent, f), copy)
            report["splits"].setdefault(split, {})[scenario] = {
                "frames_in": len(keys), "frames_kept": len(kept), "agents": agents}
            log(f"{split}/{scenario}: {len(kept)} of {len(keys)} frames kept, agents {agents}")
        src_report = os.path.join(split_dir, "conversion_report.json")
        if os.path.exists(src_report):
            os.makedirs(os.path.join(out, as_split or split), exist_ok=True)
            _link(src_report, os.path.join(out, as_split or split, "conversion_report.json"), copy)
    with open(os.path.join(out, "view_report.json"), "w") as handle:
        json.dump(report, handle, indent=2)
    return report


def _is_int(s: str) -> bool:
    try:
        int(s)
        return True
    except ValueError:
        return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="the converted dataset (holds train/validate/test)")
    ap.add_argument("--out", required=True, help="the view to write")
    ap.add_argument("--require-image", action="append", default=[], metavar="AGENT:SLOT",
                    help="keep only frames where this agent folder has camera SLOT's image")
    ap.add_argument("--split", action="append", default=None, help="limit to these splits")
    ap.add_argument("--from-key", default=None, help="keep frames from this key on (e.g. 000000)")
    ap.add_argument("--to-key", default=None, help="keep frames up to this key (e.g. 001485)")
    ap.add_argument("--as-split", default=None,
                    help="write into this split folder instead of the source's name")
    ap.add_argument("--copy", action="store_true", help="copy files instead of symlinking")
    ap.add_argument("--min-points-in-range", type=int, default=0, metavar="N",
                    help="keep only frames where EVERY agent has at least N returns "
                         "inside --range (a LiDAR's blind radius empties a forward "
                         "range when the robot is nose to a wall)")
    ap.add_argument("--range", type=float, nargs=6, default=DEFAULT_RANGE,
                    metavar=("XMIN", "YMIN", "ZMIN", "XMAX", "YMAX", "ZMAX"),
                    help="the detection range --min-points-in-range counts inside")
    a = ap.parse_args()
    req = []
    for item in a.require_image:
        agent, _, slot = item.partition(":")
        req.append((agent, int(slot or 0)))
    if not req and not (a.from_key or a.to_key) and not a.min_points_in_range:
        print("nothing to filter on: give --require-image AGENT:SLOT, --from-key, "
              "--to-key or --min-points-in-range", file=sys.stderr)
        return 2
    make_view(a.root, a.out, req, splits=tuple(a.split) if a.split else SPLITS, copy=a.copy,
              from_key=a.from_key, to_key=a.to_key, as_split=a.as_split,
              min_points=a.min_points_in_range, box=a.range)
    return 0


if __name__ == "__main__":
    sys.exit(main())
