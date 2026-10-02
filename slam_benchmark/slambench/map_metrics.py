"""Map quality: accuracy, completeness, Chamfer, F-score — and the two things
that decide whether those numbers mean anything.

Both are choices, both go in the write-up:

  the evaluation volume V   A method that maps the whole room is not worse than
                            one that maps a corridor; it is scored over more
                            space. V is fixed once, the same for every method,
                            and both clouds are cropped to it before anything
                            is measured.
  the resolution rho        Both clouds are voxel-downsampled at rho first, so
                            a dense method cannot buy completeness with points.
                            rho is the scale below which this benchmark makes no
                            claim, and on coop2 it cannot honestly go below the
                            reference map's own anchoring spread (~3 mm at the
                            best board, 15 mm at the worst).

F-score thresholds are indoor-scale (2, 5, 10 cm), matching the collaborative
benchmark's T1. Outdoor SLAM papers quote 20-50 cm; using those here would put
every method at F = 1.0 and measure nothing.
"""
from __future__ import annotations

import dataclasses

import numpy as np

from .pointcloud import crop_box, nearest_distances, voxel_downsample

__all__ = ["MapResult", "evaluate_map"]


@dataclasses.dataclass
class MapResult:
    n_est: int                    # after crop + downsample
    n_ref: int
    resolution_m: float
    truncate_m: float
    accuracy_m: dict              # est -> ref: how wrong the points it produced are
    completeness_m: dict          # ref -> est: how much of the room it missed
    chamfer_m: float              # mean of the two means
    fscore: dict                  # threshold (m) -> {precision, recall, f}
    accuracy_truncated: float     # fraction of est points with no ref within truncate_m
    completeness_truncated: float
    volume: dict

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)


def _stats(d: np.ndarray) -> dict:
    return {"mean": float(np.mean(d)), "median": float(np.median(d)),
            "p90": float(np.percentile(d, 90)), "rmse": float(np.sqrt(np.mean(d ** 2)))}


def evaluate_map(est_points: np.ndarray, ref_points: np.ndarray,
                 volume: dict | None = None, resolution_m: float = 0.02,
                 truncate_m: float = 0.50,
                 thresholds_m=(0.02, 0.05, 0.10)) -> MapResult:
    """Score a reconstructed cloud against a reference cloud, both in one frame.

    Putting them in one frame is the caller's job and is not a detail: a method
    evaluated with its own ATE alignment applied to its map measures map quality
    given perfect re-localisation, while one evaluated in its raw frame measures
    map quality including its own drift. Both are legitimate; they are different
    experiments, and `scripts/eval_run.py` reports the first and names it.
    """
    est = np.asarray(est_points, dtype=np.float64).reshape(-1, 3)
    ref = np.asarray(ref_points, dtype=np.float64).reshape(-1, 3)
    vol = dict(volume) if volume else None
    if vol:
        est = crop_box(est, vol["min"], vol["max"])
        ref = crop_box(ref, vol["min"], vol["max"])
    est = voxel_downsample(est, resolution_m)
    ref = voxel_downsample(ref, resolution_m)
    if not len(ref):
        raise ValueError("the reference cloud is empty inside the evaluation volume — "
                         "check that the map and the volume are in the same frame")
    if not len(est):
        raise ValueError("the estimated cloud is empty inside the evaluation volume — "
                         "a method whose map is elsewhere usually means an unapplied "
                         "alignment, not a failed reconstruction")

    d_acc, tr_acc = nearest_distances(est, ref, truncate_m)   # est -> ref
    d_cmp, tr_cmp = nearest_distances(ref, est, truncate_m)   # ref -> est

    f = {}
    for th in thresholds_m:
        p = float(np.mean(d_acc <= th))
        r = float(np.mean(d_cmp <= th))
        f[f"{th:.2f}"] = {"precision": p, "recall": r,
                          "f": float(2 * p * r / (p + r)) if p + r > 0 else 0.0}

    return MapResult(
        n_est=len(est), n_ref=len(ref), resolution_m=resolution_m, truncate_m=truncate_m,
        accuracy_m=_stats(d_acc), completeness_m=_stats(d_cmp),
        chamfer_m=float((np.mean(d_acc) + np.mean(d_cmp)) / 2.0), fscore=f,
        accuracy_truncated=tr_acc, completeness_truncated=tr_cmp,
        volume=vol or {"min": None, "max": None},
    )
