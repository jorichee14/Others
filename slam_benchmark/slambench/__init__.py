"""slambench — evaluation for the coop2 SLAM benchmark.

Pure Python (numpy + pyyaml). No ROS, no GPU, no open3d: the SLAM systems run
in their own containers and write two files each, and everything in this package
reads those two files. That split is the point — a method is added by writing a
config and a Dockerfile, never by touching the evaluation.
"""
from .ablate import (Ablation, apply as ablate, crop_fov, decimate_beams,
                     select_rate, subsample, truncate_range)
from .align import Alignment, fit
from .collab import (RelativeResult, agent_separation, completeness_gain,
                     observation_residual, place_overlap, predict_observation,
                     relative_trajectory_error)
from .config import (DatasetConfig, MethodConfig, extrinsic_matrix, load_dataset,
                     load_method, to_reference_frame)
from .map_metrics import MapResult, evaluate_map
from .observability import (Observability, auto_voxel, estimate_normals,
                            observability)
from .pointcloud import (crop_box, load_cloud, nearest_distances, save_ply,
                         transform_points, voxel_downsample)
from .traj_metrics import (AteResult, ErrorStats, RpeResult, absolute_check, ate,
                           revisit_residual, rpe)
from .trajectory import Trajectory, associate, load_tum, save_tum

__all__ = [
    "Ablation", "ablate", "crop_fov", "truncate_range", "decimate_beams",
    "subsample", "select_rate",
    "RelativeResult", "relative_trajectory_error", "agent_separation",
    "place_overlap", "completeness_gain", "predict_observation",
    "observation_residual",
    "Observability", "observability", "estimate_normals", "auto_voxel",
    "Alignment", "fit", "DatasetConfig", "MethodConfig", "load_dataset", "load_method",
    "extrinsic_matrix", "to_reference_frame", "MapResult", "evaluate_map", "load_cloud",
    "save_ply", "voxel_downsample", "crop_box", "nearest_distances", "transform_points",
    "ErrorStats", "AteResult", "RpeResult", "ate", "rpe", "absolute_check",
    "revisit_residual", "Trajectory", "load_tum", "save_tum", "associate",
]
__version__ = "0.1.0"
