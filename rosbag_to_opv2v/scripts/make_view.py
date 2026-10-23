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
        [--split train] [--copy]       # copy instead of symlink

Sidecars at the root (heter_modality_assign.json, isaacsim_class_map.json)
and each scenario's comms/ folder are linked too, and conversion_report.json
is written beside a view_report.json saying what was kept.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

import yaml

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


def keep_frame(scenario_dir: str, key: str, require_image) -> bool:
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


def make_view(root: str, out: str, require_image, splits=SPLITS, copy=False, log=print) -> dict:
    root, out = os.path.expanduser(root), os.path.expanduser(out)
    report = {"root": os.path.abspath(root), "require_image": [f"{a}:{s}" for a, s in require_image],
              "splits": {}}
    os.makedirs(out, exist_ok=True)
    for name in os.listdir(root):
        src = os.path.join(root, name)
        if os.path.isfile(src):                       # sidecars
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
            kept = [k for k in keys if keep_frame(sdir, k, require_image)]
            odir = os.path.join(out, split, scenario)
            for agent in agents:
                os.makedirs(os.path.join(odir, agent), exist_ok=True)
                for k in kept:
                    for f in os.listdir(os.path.join(sdir, agent)):
                        if f.startswith(k + ".") or f.startswith(k + "_"):
                            _link(os.path.join(sdir, agent, f), os.path.join(odir, agent, f), copy)
            for extra in ("comms",):
                if os.path.isdir(os.path.join(sdir, extra)):
                    _link(os.path.join(sdir, extra), os.path.join(odir, extra), copy)
            report["splits"].setdefault(split, {})[scenario] = {
                "frames_in": len(keys), "frames_kept": len(kept), "agents": agents}
            log(f"{split}/{scenario}: {len(kept)} of {len(keys)} frames kept, agents {agents}")
        src_report = os.path.join(split_dir, "conversion_report.json")
        if os.path.exists(src_report):
            os.makedirs(os.path.join(out, split), exist_ok=True)
            _link(src_report, os.path.join(out, split, "conversion_report.json"), copy)
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
    ap.add_argument("--copy", action="store_true", help="copy files instead of symlinking")
    a = ap.parse_args()
    req = []
    for item in a.require_image:
        agent, _, slot = item.partition(":")
        req.append((agent, int(slot or 0)))
    if not req:
        print("nothing to filter on: give --require-image AGENT:SLOT", file=sys.stderr)
        return 2
    make_view(a.root, a.out, req, splits=tuple(a.split) if a.split else SPLITS, copy=a.copy)
    return 0


if __name__ == "__main__":
    sys.exit(main())
