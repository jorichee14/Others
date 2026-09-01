# -*- coding: utf-8 -*-
"""
The ``vehicles`` block of a frame yaml — OPV2V's detection ground truth.

A rosbag carries no annotations, so the only labels that can be produced
automatically are the agents themselves: every robot's own SLAM pose is a 3D box
that the *other* agents should be able to detect.  That is exact (as exact as the
localisation), needs no human, and is genuinely the object cooperative perception
is supposed to help with — but it is only two or three boxes per frame, so it is
a sanity signal, not a benchmark.  Anything else has to come from a labelling
pass; :func:`merge_external_labels` is where that plugs in.

Box convention (read off ``opencood.utils.box_utils.project_world_objects``)::

    object_pose = location + center           # summed componentwise, unrotated
    corners     = x1_to_x2(object_pose, lidar_pose) @ create_bbx(extent)

so ``extent`` is *half* dimensions in the object frame and ``angle`` is
``[roll, yaw, pitch]`` in degrees.  Because ``center`` is added without being
rotated, this module folds the body-frame box offset into ``location`` itself and
always writes ``center: [0, 0, 0]`` — the two are equivalent for OPV2V's
near-level vehicles, but only the folded form stays correct when a robot pitches.
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional

import numpy as np

from .geometry import matrix_to_opencood_pose


def agent_box(world_from_base: np.ndarray, extent, center=(0.0, 0.0, 0.0)) -> dict:
    """One ``vehicles`` entry for an agent at ``world_from_base``."""
    offset = np.asarray(center, dtype=np.float64)
    location = (world_from_base @ np.append(offset, 1.0))[:3]
    pose = matrix_to_opencood_pose(world_from_base)
    return {
        "angle": [pose[3], pose[4], pose[5]],
        "center": [0.0, 0.0, 0.0],
        "extent": [float(extent[0]), float(extent[1]), float(extent[2])],
        "location": [float(location[0]), float(location[1]), float(location[2])],
    }


def vehicles_for_viewer(agent_poses: Dict[str, np.ndarray],
                        agent_objects: Dict[str, dict],
                        viewer: Optional[str],
                        include_self: bool = False) -> dict:
    """Build the ``vehicles`` dict written into one agent's frame yaml.

    ``agent_objects`` maps agent name -> ``{'object_id': int, 'extent': [...],
    'center': [...]}``.  By default an agent does not list itself, mirroring
    OPV2V, where a CAV's own box comes from the *other* CAVs' yaml files (OpenCOOD
    merges every agent's dict before generating targets).
    """
    vehicles = {}
    for name, pose in agent_poses.items():
        if pose is None:
            continue
        spec = agent_objects.get(name)
        if spec is None:
            continue
        if name == viewer and not include_self:
            continue
        vehicles[int(spec["object_id"])] = agent_box(
            pose, spec["extent"], spec.get("center", (0.0, 0.0, 0.0)))
    return vehicles


def merge_external_labels(vehicles: dict, external: Optional[Iterable[dict]]) -> dict:
    """Fold externally annotated boxes into a ``vehicles`` dict.

    Each entry must supply ``id``, ``location`` (world xyz of the box centre),
    ``extent`` (half-dimensions) and ``angle`` (``[roll, yaw, pitch]`` degrees in
    the same world frame the poses use).  Ids must not collide with the
    agent-derived ones; the converter reserves 10000+ for CAVs and 20000+ for
    RSUs precisely so a labelling tool's ids can stay small.
    """
    if not external:
        return vehicles
    for item in external:
        object_id = int(item["id"])
        if object_id in vehicles:
            raise ValueError(f"external label id {object_id} collides with an "
                             f"agent-derived box id")
        location = [float(v) for v in item["location"]]
        extent = [float(v) for v in item["extent"]]
        angle = [float(v) for v in item.get("angle", (0.0, 0.0, 0.0))]
        vehicles[object_id] = {
            "angle": angle,
            "center": [float(v) for v in item.get("center", (0.0, 0.0, 0.0))],
            "extent": extent,
            "location": location,
        }
    return vehicles
