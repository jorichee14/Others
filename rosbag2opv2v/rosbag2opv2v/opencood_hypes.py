"""Emit an OpenCOOD training config matched to a converted dataset.

OpenCOOD's stock hypes are sized for outdoor traffic (140 m range, 3.9 m long
cars).  An indoor robot dataset needs a different LiDAR range, voxel size and
anchor box, and getting those wrong silently produces a model that predicts
nothing.  This module derives them from the exported data.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml

from .verify import load_points

_TEMPLATE = """name: {name}
root_dir: "{root_dir}"
validate_dir: "{validate_dir}"

yaml_parser: "load_point_pillar_params"
train_params:
  batch_size: &batch_size 4
  epoches: 40
  eval_freq: 2
  save_freq: 2
  max_cav: &max_cav {max_cav}

fusion:
  core_method: '{dataset}' # LateFusionDataset, EarlyFusionDataset, IntermediateFusionDataset
  args: []

# preprocess-related
preprocess:
  core_method: 'SpVoxelPreprocessor'
  args:
    voxel_size: &voxel_size [{vx}, {vy}, {vz}]
    max_points_per_voxel: 32
    max_voxel_train: 32000
    max_voxel_test: 70000
  # lidar range for each individual cav, derived from the converted clouds
  cav_lidar_range: &cav_lidar [{x_min}, {y_min}, {z_min}, {x_max}, {y_max}, {z_max}]

data_augment:
  - NAME: random_world_flip
    ALONG_AXIS_LIST: [ 'x' ]
  - NAME: random_world_rotation
    WORLD_ROT_ANGLE: [ -0.78539816, 0.78539816 ]
  - NAME: random_world_scaling
    WORLD_SCALE_RANGE: [ 0.95, 1.05 ]

# anchor box related -- sized from the exported agent extents
postprocess:
  core_method: 'VoxelPostprocessor'
  anchor_args:
    cav_lidar_range: *cav_lidar
    l: {anchor_l}
    w: {anchor_w}
    h: {anchor_h}
    r: [0, 90]
    feature_stride: 2
    num: &achor_num 2
  target_args:
    # anchors sit on a {anchor_pitch} m grid while the targets are under a metre
    # long, so the best achievable BEV IoU is far below the outdoor 0.6 default
    pos_threshold: 0.45
    neg_threshold: 0.30
    score_threshold: 0.20
  order: 'hwl'
  max_num: {max_num}
  nms_thresh: 0.15

# model related
model:
  core_method: {model}
  args:
    voxel_size: *voxel_size
    lidar_range: *cav_lidar
    anchor_number: *achor_num

    pillar_vfe:
      use_norm: true
      with_distance: false
      use_absolute_xyz: true
      num_filters: [64]
    point_pillar_scatter:
      num_features: 64

    base_bev_backbone:
      layer_nums: [3, 5, 8]
      layer_strides: [2, 2, 2]
      num_filters: [64, 128, 256]
      upsample_strides: [1, 2, 4]
      num_upsample_filter: [128, 128, 128]
      compression: 0

    anchor_num: *achor_num

loss:
  core_method: point_pillar_loss
  args:
    cls_weight: 1.0
    reg: 2.0

optimizer:
  core_method: Adam
  lr: 0.002
  args:
    eps: 1e-10
    weight_decay: 1e-4

lr_scheduler:
  core_method: multistep
  gamma: 0.1
  step_size: [20, 32]
"""

_FUSION = {
    "intermediate": ("IntermediateFusionDataset", "point_pillar_intermediate"),
    "early": ("EarlyFusionDataset", "point_pillar"),
    "late": ("LateFusionDataset", "point_pillar"),
}


def _round_range(low: float, high: float, step: float) -> Tuple[float, float]:
    """Snap a range outward to a multiple of ``step`` (keeps the BEV grid even)."""
    low = float(np.floor(low / step) * step)
    high = float(np.ceil(high / step) * step)
    if high - low < step:
        high = low + step
    return round(low, 3), round(high, 3)


def survey(root: str, max_clouds: int = 40) -> Dict[str, object]:
    """Sample the exported clouds and agent metadata."""
    clouds: List[str] = []
    extents: List[List[float]] = []
    cav_counts: List[int] = []
    splits: List[str] = []

    for split in sorted(os.listdir(root)):
        split_dir = os.path.join(root, split)
        if not os.path.isdir(split_dir):
            continue
        splits.append(split)
        for scenario in sorted(os.listdir(split_dir)):
            scenario_dir = os.path.join(split_dir, scenario)
            if not os.path.isdir(scenario_dir):
                continue
            meta_path = os.path.join(scenario_dir, "scenario_meta.yaml")
            if os.path.isfile(meta_path):
                with open(meta_path, "r") as handle:
                    meta = yaml.safe_load(handle) or {}
                for info in (meta.get("agents") or {}).values():
                    if info.get("is_object") is False:
                        continue
                    if info.get("extent"):
                        extents.append([float(v) for v in info["extent"]])
            cavs = [d for d in sorted(os.listdir(scenario_dir))
                    if os.path.isdir(os.path.join(scenario_dir, d))]
            cav_counts.append(len(cavs))
            for cav in cavs:
                cav_dir = os.path.join(scenario_dir, cav)
                found = sorted(f for f in os.listdir(cav_dir)
                               if f.endswith(".pcd"))
                if found:
                    step = max(1, len(found) // 5)
                    clouds.extend(os.path.join(cav_dir, f)
                                  for f in found[::step][:5])

    if len(clouds) > max_clouds:
        step = max(1, len(clouds) // max_clouds)
        clouds = clouds[::step][:max_clouds]

    lows, highs = [], []
    for path in clouds:
        points = load_points(path)
        if points.size == 0:
            continue
        lows.append(np.percentile(points[:, :3], 1, axis=0))
        highs.append(np.percentile(points[:, :3], 99, axis=0))
    if not lows:
        raise RuntimeError("no readable point clouds under %s" % root)
    return {
        "low": np.min(np.asarray(lows), axis=0),
        "high": np.max(np.asarray(highs), axis=0),
        "extents": extents,
        "max_cav": max(cav_counts) if cav_counts else 1,
        "splits": splits,
        "clouds_sampled": len(clouds),
    }


def build(root: str, fusion: str = "intermediate",
          voxel_xy: float = 0.1, name: Optional[str] = None) -> str:
    """Return the text of an OpenCOOD hypes yaml for the dataset at ``root``."""
    if fusion not in _FUSION:
        raise ValueError("fusion must be one of %s" % sorted(_FUSION))
    info = survey(root)
    dataset, model = _FUSION[fusion]

    grid = voxel_xy * 8.0     # keep W/H divisible by the backbone strides
    # x/y are symmetric around the ego: in cooperative fusion the other agents'
    # clouds are projected into this frame and can land on either side.
    x_half = _round_range(0.0, max(abs(info["low"][0]), abs(info["high"][0])),
                          grid)[1]
    y_half = _round_range(0.0, max(abs(info["low"][1]), abs(info["high"][1])),
                          grid)[1]
    x_min, x_max = -x_half, x_half
    y_min, y_max = -y_half, y_half
    z_min, z_max = _round_range(info["low"][2], info["high"][2], 0.5)
    vz = round(z_max - z_min, 3)   # a single pillar in z, as PointPillars wants

    extents = info["extents"] or [[0.4, 0.3, 0.4]]
    biggest = np.max(np.asarray(extents, dtype=float), axis=0) * 2.0

    root = os.path.abspath(root)
    splits = info["splits"]
    train_dir = os.path.join(root, "train" if "train" in splits else splits[0])
    val_split = ("validate" if "validate" in splits
                 else ("test" if "test" in splits else splits[0]))
    return _TEMPLATE.format(
        name="point_pillar_%s_%s" % (fusion, name or os.path.basename(root)),
        root_dir=train_dir,
        validate_dir=os.path.join(root, val_split),
        dataset=dataset, model=model,
        max_cav=max(2, int(info["max_cav"])),
        vx=voxel_xy, vy=voxel_xy, vz=vz,
        x_min=x_min, y_min=y_min, z_min=z_min,
        x_max=x_max, y_max=y_max, z_max=z_max,
        anchor_pitch=round(voxel_xy * 2, 3),
        anchor_l=round(float(biggest[0]), 3),
        anchor_w=round(float(biggest[1]), 3),
        anchor_h=round(float(biggest[2]), 3),
        max_num=20,
    )
