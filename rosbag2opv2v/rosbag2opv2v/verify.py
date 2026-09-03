"""Validate a converted dataset the way OpenCOOD will read it.

Structural checks mirror ``opencood.data_utils.datasets.basedataset``; the
geometric checks re-derive, from the exported files alone, the two things a
cooperative-perception dataset silently gets wrong:

* whether the agents' point clouds actually land on top of each other once
  projected into the ego frame with the exported poses (extrinsics/pose sanity),
* whether the ground-truth boxes sit where the ego LiDAR sees something
  (annotation sanity).
"""

from __future__ import annotations

import os
from collections import OrderedDict
from typing import Dict, List, Tuple

import numpy as np
import yaml

from .pcd_io import read_pcd
from .transforms import opv2v_pose_to_matrix, invert

GT_RANGE = [-140, -40, -3, 140, 40, 1]   # opencood.data_utils.datasets.GT_RANGE


def load_points(path: str) -> np.ndarray:
    """(N,4) x,y,z,intensity -- via Open3D when available (exactly OpenCOOD's
    ``pcd_to_np``), else via the bundled reader."""
    try:
        import open3d as o3d
    except ImportError:
        return read_pcd(path)
    pcd = o3d.io.read_point_cloud(path)
    xyz = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    intensity = colors[:, 0] if colors.size else np.zeros(len(xyz))
    return np.hstack([xyz, intensity[:, None]]).astype(np.float32)


def _x1_to_x2(x1, x2) -> np.ndarray:
    return invert(opv2v_pose_to_matrix(x2)) @ opv2v_pose_to_matrix(x1)


def _box_corners(extent) -> np.ndarray:
    e = np.asarray(extent, dtype=float)
    signs = np.array([[1, -1, -1], [1, 1, -1], [-1, 1, -1], [-1, -1, -1],
                      [1, -1, 1], [1, 1, 1], [-1, 1, 1], [-1, -1, 1]])
    return signs * e


def _points_in_box(points: np.ndarray, box_to_lidar: np.ndarray,
                   extent) -> int:
    """Count LiDAR points inside an oriented box given as (pose, half-extent)."""
    if points.size == 0:
        return 0
    local = (points[:, :3] - box_to_lidar[:3, 3]) @ box_to_lidar[:3, :3]
    e = np.asarray(extent, dtype=float)
    inside = np.all(np.abs(local) <= e, axis=1)
    return int(inside.sum())


def _voxel_overlap(a: np.ndarray, b: np.ndarray, size: float = 0.25) -> float:
    """Fraction of B's occupied voxels that A also occupies."""
    if a.size == 0 or b.size == 0:
        return 0.0
    ka = set(map(tuple, np.floor(a[:, :3] / size).astype(np.int64)))
    kb = set(map(tuple, np.floor(b[:, :3] / size).astype(np.int64)))
    return len(ka & kb) / max(1, len(kb))


class Verifier:
    def __init__(self, root: str, sample: int = 20, verbose: bool = True):
        self.root = os.path.abspath(root)
        self.sample = sample
        self.verbose = verbose
        self.errors: List[str] = []
        self.notes: List[str] = []

    def log(self, message: str) -> None:
        if self.verbose:
            print(message, flush=True)

    def error(self, message: str) -> None:
        self.errors.append(message)
        print("ERROR: " + message, flush=True)

    # ------------------------------------------------------------------
    def scenarios(self) -> List[Tuple[str, str]]:
        out = []
        for split in sorted(os.listdir(self.root)):
            split_dir = os.path.join(self.root, split)
            if not os.path.isdir(split_dir):
                continue
            for scenario in sorted(os.listdir(split_dir)):
                if os.path.isdir(os.path.join(split_dir, scenario)):
                    out.append((split, scenario))
        return out

    def cav_list(self, scenario_dir: str) -> List[str]:
        cavs = sorted([d for d in os.listdir(scenario_dir)
                       if os.path.isdir(os.path.join(scenario_dir, d))])
        for cav in cavs:
            try:
                int(cav)
            except ValueError:
                self.error("cav folder '%s' in %s is not an integer name; "
                           "OpenCOOD calls int() on it" % (cav, scenario_dir))
        # OpenCOOD moves the negative (RSU) ids to the end and takes the first
        # entry as ego
        numeric = [c for c in cavs if c.lstrip("-").isdigit()]
        if numeric and int(numeric[0]) < 0:
            numeric = numeric[1:] + [numeric[0]]
        return numeric

    def run(self) -> Dict[str, object]:
        scenarios = self.scenarios()
        if not scenarios:
            self.error("no split/scenario folders under %s" % self.root)
            return {"ok": False, "errors": self.errors}

        summary = {"scenarios": len(scenarios), "frames": 0, "checked": 0,
                   "overlap": [], "gt_hit_rate": [], "points": []}
        for split, scenario in scenarios:
            scenario_dir = os.path.join(self.root, split, scenario)
            cavs = self.cav_list(scenario_dir)
            if not cavs:
                self.error("%s has no cav folders" % scenario_dir)
                continue
            stamps: Dict[str, List[str]] = OrderedDict()
            for cav in cavs:
                cav_dir = os.path.join(scenario_dir, cav)
                stamps[cav] = sorted(
                    f[:-5] for f in os.listdir(cav_dir)
                    if f.endswith(".yaml") and "additional" not in f)
            reference = stamps[cavs[0]]
            for cav in cavs[1:]:
                if stamps[cav] != reference:
                    self.error("%s: cav %s has %d timestamps but ego has %d "
                               "(OpenCOOD indexes every cav by the ego's list)"
                               % (scenario, cav, len(stamps[cav]),
                                  len(reference)))
            summary["frames"] += len(reference)
            self.log("%s/%s: %d cavs %s, %d frames" %
                     (split, scenario, len(cavs), cavs, len(reference)))

            picks = reference
            if self.sample and len(picks) > self.sample:
                step = max(1, len(picks) // self.sample)
                picks = picks[::step][:self.sample]
            for stamp in picks:
                self._check_frame(scenario_dir, cavs, stamp, summary)

        result = {
            "ok": not self.errors,
            "scenarios": summary["scenarios"],
            "frames": summary["frames"],
            "frames_checked": summary["checked"],
            "errors": self.errors,
        }
        if summary["overlap"]:
            result["mean_cross_agent_voxel_overlap"] = float(
                np.mean(summary["overlap"]))
        if summary["gt_hit_rate"]:
            result["gt_boxes_with_ego_points"] = float(
                np.mean(summary["gt_hit_rate"]))
        if summary["points"]:
            result["mean_points_per_cloud"] = float(np.mean(summary["points"]))
        return result

    def _check_frame(self, scenario_dir: str, cavs: List[str], stamp: str,
                     summary: Dict[str, object]) -> None:
        params: Dict[str, dict] = {}
        clouds: Dict[str, np.ndarray] = {}
        for cav in cavs:
            cav_dir = os.path.join(scenario_dir, cav)
            yaml_path = os.path.join(cav_dir, stamp + ".yaml")
            pcd_path = os.path.join(cav_dir, stamp + ".pcd")
            if not os.path.isfile(pcd_path):
                self.error("missing point cloud %s" % pcd_path)
                continue
            with open(yaml_path, "r") as handle:
                content = yaml.safe_load(handle)
            for key in ("lidar_pose", "vehicles", "ego_speed", "true_ego_pos"):
                if key not in content:
                    self.error("%s is missing the '%s' key" % (yaml_path, key))
            pose = content.get("lidar_pose")
            if not isinstance(pose, list) or len(pose) != 6:
                self.error("%s: lidar_pose must be a list of 6 numbers"
                           % yaml_path)
                continue
            params[cav] = content
            clouds[cav] = load_points(pcd_path)
            if clouds[cav].size == 0:
                self.error("%s contains no points" % pcd_path)
            summary["points"].append(len(clouds[cav]))

        if len(params) < 1:
            return
        summary["checked"] += 1
        ego = cavs[0]
        if ego not in params:
            return
        ego_pose = params[ego]["lidar_pose"]
        ego_points = clouds[ego]

        # cross-agent alignment
        for cav in cavs[1:]:
            if cav not in params:
                continue
            matrix = _x1_to_x2(params[cav]["lidar_pose"], ego_pose)
            projected = clouds[cav][:, :3] @ matrix[:3, :3].T + matrix[:3, 3]
            summary["overlap"].append(_voxel_overlap(ego_points, projected))

        # ground-truth boxes vs. the ego's own points
        for object_id, content in (params[ego].get("vehicles") or {}).items():
            location = np.asarray(content["location"], dtype=float)
            center = np.asarray(content["center"], dtype=float)
            angle = content["angle"]
            pose = [location[0] + center[0], location[1] + center[1],
                    location[2] + center[2], angle[0], angle[1], angle[2]]
            box_to_lidar = _x1_to_x2(pose, ego_pose)
            corners = _box_corners(content["extent"])
            corners = corners @ box_to_lidar[:3, :3].T + box_to_lidar[:3, 3]
            if np.any(corners[:, 0] < GT_RANGE[0]) or \
               np.any(corners[:, 0] > GT_RANGE[3]) or \
               np.any(corners[:, 1] < GT_RANGE[1]) or \
               np.any(corners[:, 1] > GT_RANGE[4]):
                self.notes.append(
                    "%s frame %s: object %s falls outside OpenCOOD's default "
                    "GT_RANGE and would be filtered at evaluation time"
                    % (os.path.basename(scenario_dir), stamp, object_id))
            hit = _points_in_box(ego_points, box_to_lidar, content["extent"])
            summary["gt_hit_rate"].append(1.0 if hit > 0 else 0.0)


def verify(root: str, sample: int = 20, verbose: bool = True) -> Dict[str, object]:
    verifier = Verifier(root, sample=sample, verbose=verbose)
    result = verifier.run()
    if verifier.notes and verbose:
        seen = set()
        for note in verifier.notes:
            head = note.split(":")[0]
            if head not in seen:
                seen.add(head)
                print("NOTE: " + note)
    return result
