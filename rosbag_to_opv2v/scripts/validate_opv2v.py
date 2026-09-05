#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage C: check a converted dataset the way OpenCOOD will actually read it.

The checks mirror assumptions in ``opencood.data_utils.datasets.basedataset``
that are silent until they are violated:

* every agent folder holds the *same* timestamp keys — OpenCOOD reads the list
  from the first agent and indexes the rest with it, so a gap is a KeyError
* every timestamp has both a ``.yaml`` and a ``.pcd``
* ``lidar_pose`` is a finite 6-vector and ``vehicles`` is a mapping
* each pcd is readable and carries points
* agent ids are integers, with roadside units negative

It also reports the geometry that decides whether the dataset is usable at all:
inter-agent distance (must be inside the model's ``comm_range``) and how many GT
boxes actually fall inside each ego's detection range.

    python scripts/validate_opv2v.py --root ~/cpfa/data/OPV2V_from_bag/test
    python scripts/validate_opv2v.py --root ... --with-open3d --with-opencood
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np                                              # noqa: E402
import yaml                                                     # noqa: E402

from ros2opv2v.geometry import invert, x_to_world                # noqa: E402
from ros2opv2v.writers import read_pcd                           # noqa: E402

DEFAULT_RANGE = [-140.8, -40.0, -3.0, 140.8, 40.0, 1.0]          # OPV2V GT_RANGE


class Findings:
    def __init__(self):
        self.errors, self.warnings, self.notes = [], [], []

    def error(self, message):
        self.errors.append(message)

    def warn(self, message):
        self.warnings.append(message)

    def note(self, message):
        self.notes.append(message)


def scenario_dirs(root: str):
    return sorted(os.path.join(root, d) for d in os.listdir(root)
                  if os.path.isdir(os.path.join(root, d)))


def agent_dirs(scenario: str):
    return sorted((d for d in os.listdir(scenario)
                   if os.path.isdir(os.path.join(scenario, d))),
                  key=lambda d: (0, int(d)) if _is_int(d) else (1, 0))


def _is_int(text: str) -> bool:
    try:
        int(text)
        return True
    except ValueError:
        return False


def check_scenario(scenario: str, findings: Findings, args) -> dict:
    name = os.path.basename(scenario)
    agents = agent_dirs(scenario)
    summary = {"name": name, "agents": agents, "frames": 0, "boxes_in_range": []}

    if not agents:
        findings.error(f"{name}: no agent folders")
        return summary
    for agent in agents:
        if not _is_int(agent):
            findings.error(f"{name}/{agent}: agent folder name must be an integer id "
                           f"(OpenCOOD calls int() on it)")
            return summary

    timestamps = {}
    for agent in agents:
        path = os.path.join(scenario, agent)
        keys = sorted(f[:-5] for f in os.listdir(path)
                      if f.endswith(".yaml") and "additional" not in f)
        timestamps[agent] = keys
        if not keys:
            findings.error(f"{name}/{agent}: no yaml files")

    reference = timestamps[agents[0]]
    for agent in agents[1:]:
        missing = set(reference) - set(timestamps[agent])
        extra = set(timestamps[agent]) - set(reference)
        if missing:
            findings.error(
                f"{name}/{agent}: missing {len(missing)} timestamps present in "
                f"agent {agents[0]} (e.g. {sorted(missing)[:3]}) — OpenCOOD indexes "
                f"every agent with the first agent's keys and will KeyError")
        if extra:
            findings.warn(f"{name}/{agent}: {len(extra)} timestamps not in agent "
                          f"{agents[0]}; they will never be read")

    summary["frames"] = len(reference)
    if not reference:
        return summary

    ids = [int(a) for a in agents]
    if min(ids) < 0 and len([i for i in ids if i >= 0]) == 0:
        findings.error(f"{name}: every agent id is negative — OpenCOOD would have "
                       f"no ego")

    # OpenCOOD's basedataset takes the FIRST agent folder, in lexicographic order,
    # as the ego — it does not read the `ego` flag we write. A roadside unit named
    # "-1" sorts before "1" because '-' (0x2D) precedes '1' (0x31), so the RSU
    # silently becomes the ego: fusion is then computed around a sensor that may
    # not even reach the objects, and nothing errors.
    declared = _declared_ego(scenario, agents, reference[0])
    if declared is not None and agents[0] != declared:
        findings.warn(
            f"{name}: the yaml marks agent {declared} as ego, but {agents[0]} sorts "
            f"first. OpenCOOD's basedataset treats the first agent folder as the ego "
            f"and ignores the flag, so it would fuse around {agents[0]}. Either rename "
            f"the folders so {declared} sorts first (RSU ids of 100+ instead of "
            f"negatives), or confirm your OpenCOOD fork honours the `ego` key "
            f"(V2XSet's does; vanilla OpenCOOD does not).")

    sample_keys = reference[::max(1, len(reference) // args.sample)][:args.sample]
    distances, box_counts, point_counts = [], [], defaultdict(list)

    for key in sample_keys:
        poses, vehicles_union = {}, {}
        for agent in agents:
            yaml_path = os.path.join(scenario, agent, f"{key}.yaml")
            pcd_path = os.path.join(scenario, agent, f"{key}.pcd")
            if not os.path.isfile(yaml_path):
                findings.error(f"{name}/{agent}/{key}.yaml missing")
                continue
            if not os.path.isfile(pcd_path):
                findings.error(f"{name}/{agent}/{key}.pcd missing")
                continue

            with open(yaml_path, "r") as handle:
                params = yaml.safe_load(handle)

            pose = params.get("lidar_pose")
            if not isinstance(pose, list) or len(pose) != 6 or \
                    not all(isinstance(v, (int, float)) and math.isfinite(v) for v in pose):
                findings.error(f"{name}/{agent}/{key}: lidar_pose must be 6 finite "
                               f"numbers, got {pose!r}")
                continue
            poses[agent] = pose

            if "ego_speed" not in params:
                findings.error(f"{name}/{agent}/{key}: ego_speed missing "
                               f"(intermediate fusion reads it)")
            vehicles = params.get("vehicles")
            if not isinstance(vehicles, dict):
                findings.error(f"{name}/{agent}/{key}: vehicles must be a mapping")
            else:
                vehicles_union.update(vehicles)

            if args.with_open3d or args.read_pcd:
                cloud = _read_cloud(pcd_path, args.with_open3d)
                point_counts[agent].append(cloud.shape[0])
                if cloud.shape[0] == 0:
                    findings.warn(f"{name}/{agent}/{key}: empty point cloud")

        if len(poses) >= 2:
            keys_list = list(poses)
            for i, a in enumerate(keys_list):
                for b in keys_list[i + 1:]:
                    distances.append(math.dist(poses[a][:2], poses[b][:2]))

        ego = min((a for a in poses if int(a) >= 0), key=int, default=None)
        if ego is not None and vehicles_union:
            box_counts.append(_boxes_in_range(vehicles_union, poses[ego], args.range))

    if distances:
        findings.note(f"{name}: inter-agent distance min {min(distances):.1f} m, "
                      f"mean {np.mean(distances):.1f} m, max {max(distances):.1f} m")
        if max(distances) > 70:
            findings.warn(f"{name}: agents up to {max(distances):.0f} m apart — "
                          f"OpenCOOD drops collaborators beyond comm_range (70 m by "
                          f"default), so those frames become single-agent")
    if box_counts:
        summary["boxes_in_range"] = box_counts
        mean_boxes = float(np.mean(box_counts))
        findings.note(f"{name}: {mean_boxes:.2f} GT boxes per frame inside the ego's "
                      f"detection range (of {len(box_counts)} sampled frames)")
        if mean_boxes == 0:
            findings.warn(f"{name}: no GT box falls inside the detection range — AP "
                          f"will be undefined. Expected if you converted without labels.")
    for agent, counts in point_counts.items():
        findings.note(f"{name}/{agent}: {np.mean(counts):.0f} points/frame "
                      f"(min {min(counts)}, max {max(counts)})")
    return summary


def _declared_ego(scenario: str, agents, key: str):
    """The agent whose first frame carries `ego: true`, if any."""
    for agent in agents:
        path = os.path.join(scenario, agent, f"{key}.yaml")
        try:
            with open(path) as handle:
                params = yaml.safe_load(handle) or {}
        except OSError:
            continue
        if params.get("ego"):
            return agent
    return None


def _read_cloud(path: str, use_open3d: bool) -> np.ndarray:
    if use_open3d:
        import open3d as o3d
        pcd = o3d.io.read_point_cloud(path)
        xyz = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)
        intensity = colors[:, :1] if colors.size else np.zeros((xyz.shape[0], 1))
        return np.hstack([xyz, intensity]).astype(np.float32)
    return read_pcd(path)


def _boxes_in_range(vehicles: dict, lidar_pose, limits) -> int:
    """How many GT box centres land inside the detection range, in the ego frame."""
    world_to_lidar = invert(x_to_world(lidar_pose))
    inside = 0
    for content in vehicles.values():
        location = np.asarray(content["location"], dtype=np.float64)
        center = np.asarray(content.get("center", [0, 0, 0]), dtype=np.float64)
        point = world_to_lidar @ np.append(location + center, 1.0)
        if (limits[0] <= point[0] <= limits[3] and limits[1] <= point[1] <= limits[4]
                and limits[2] <= point[2] <= limits[5]):
            inside += 1
    return inside


def check_with_opencood(root: str, findings: Findings, config_path=None,
                        samples: int = 3) -> None:
    """Build OpenCOOD's own dataset over the tree and pull frames through it.

    Every other check here re-implements what OpenCOOD is believed to do. This
    one asks OpenCOOD, which is the only way to be sure — the loader's real
    behaviour on ego selection, timestamp alignment, pose order and the pcd
    reader is whatever the installed version does, not whatever its docs say.
    """
    try:
        from opencood.data_utils.datasets.late_fusion_dataset import LateFusionDataset
        from opencood.hypes_yaml.yaml_utils import load_yaml as load_opencood_yaml
    except Exception as error:                          # noqa: BLE001
        findings.warn(f"--with-opencood skipped, OpenCOOD not importable here: {error}")
        return

    if not config_path:
        findings.warn(
            "--with-opencood needs --opencood-config pointing at the model hypes "
            "yaml you intend to run (e.g. opencood/hypes_yaml/point_pillar_late_"
            "fusion.yaml). Without it the preprocessor, anchors and lidar range "
            "would be guesses, and a dataset that loads under guessed settings "
            "proves nothing about the ones you will train with.")
        return

    try:
        params = load_opencood_yaml(config_path)
    except Exception as error:                          # noqa: BLE001
        findings.error(f"--opencood-config {config_path} did not load: {error}")
        return

    # Point every directory key this version might read at OUR tree.
    for key in ("root_dir", "validate_dir", "test_dir"):
        params[key] = root

    try:
        dataset = LateFusionDataset(params, visualize=False, train=False)
    except Exception as error:                          # noqa: BLE001
        findings.error(f"OpenCOOD's LateFusionDataset refused this tree: "
                       f"{type(error).__name__}: {error}")
        return

    count = len(dataset)
    if not count:
        findings.error("OpenCOOD built a dataset of length 0 over this tree — it "
                       "found no usable frames")
        return
    findings.note(f"OpenCOOD LateFusionDataset built over {root}: {count} samples")

    indices = sorted({0, count // 2, count - 1})[:max(1, samples)]
    for index in indices:
        try:
            item = dataset[index]
        except Exception as error:                      # noqa: BLE001
            findings.error(f"OpenCOOD failed reading sample {index}: "
                           f"{type(error).__name__}: {error}")
            return
        ego = item.get("ego", item) if isinstance(item, dict) else {}
        gt = ego.get("object_bbx_mask")
        if gt is not None and float(np.sum(np.asarray(gt))) == 0.0:
            findings.warn(f"OpenCOOD sample {index} carries no GT boxes after its own "
                          f"range filter — check the config's cav_lidar_range against "
                          f"where the objects actually are")
    findings.note(f"OpenCOOD read samples {indices} without error — the loader "
                  f"accepts this dataset")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", required=True,
                        help="split directory, e.g. .../OPV2V_from_bag/test")
    parser.add_argument("--sample", type=int, default=40,
                        help="frames per scenario to open (0 checks structure only)")
    parser.add_argument("--read-pcd", action="store_true",
                        help="read each sampled pcd with the built-in reader")
    parser.add_argument("--with-open3d", action="store_true",
                        help="read pcds through open3d, exactly as OpenCOOD does")
    parser.add_argument("--opencood-config", default=None,
                        help="OpenCOOD model hypes yaml to build the dataset with; "
                             "its validate_dir is overridden to --root")
    parser.add_argument("--with-opencood", action="store_true",
                        help="also try importing OpenCOOD")
    parser.add_argument("--range", type=float, nargs=6, default=DEFAULT_RANGE,
                        metavar=("XMIN", "YMIN", "ZMIN", "XMAX", "YMAX", "ZMAX"),
                        help="detection range used for the GT-in-range count")
    args = parser.parse_args()

    root = os.path.expanduser(args.root)
    if not os.path.isdir(root):
        print(f"not a directory: {root}", file=sys.stderr)
        return 2

    findings = Findings()
    scenarios = scenario_dirs(root)
    if not scenarios:
        print(f"no scenario folders under {root}", file=sys.stderr)
        return 2

    print(f"validating {len(scenarios)} scenario(s) under {root}\n")
    total_frames = 0
    for scenario in scenarios:
        summary = check_scenario(scenario, findings, args)
        total_frames += summary["frames"]
        print(f"  {summary['name']:<34} agents={summary['agents']} "
              f"frames={summary['frames']}")

    if args.with_opencood:
        check_with_opencood(root, findings, args.opencood_config)

    print(f"\ntotal frames: {total_frames}")
    for note in findings.notes:
        print(f"  . {note}")
    for warning in findings.warnings:
        print(f"  ! {warning}")
    for error in findings.errors:
        print(f"  X {error}")

    print()
    if findings.errors:
        print(f"FAILED: {len(findings.errors)} error(s), "
              f"{len(findings.warnings)} warning(s)")
        return 1
    print(f"PASSED ({len(findings.warnings)} warning(s))")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
