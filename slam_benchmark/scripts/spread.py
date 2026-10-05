#!/usr/bin/env python3
"""Score every run in one cell and report the spread, not a number.

    python3 scripts/spread.py --config configs/coop2.yaml \
        --method configs/methods/rtabmap_rgbd_imu.yaml --stream mobile_1.zed_rgbd \
        --cell runs/coop2_20260828/rtabmap_rgbd_imu/mobile_1.zed_rgbd \
        --reference runs/reference/mobile_1.tum [--trajectory-file odometry.tum]

A cell is `runs/<dataset>/<method>/<stream>/`; each run is a subdirectory
(scripts/run_method.py makes one per launch). RTAB-Map's front-end is not
deterministic run to run, so one run is one sample: the cell is reported as
median [min, max] over N. Fewer than three runs is printed as the samples it
is, with no summary line. Runs that were interrupted or left no trajectory
are listed and left out. Nothing is written: eval_run.py owns metrics.json.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from slambench import ate, load_dataset, load_method, load_tum, to_reference_frame  # noqa: E402

MIN_RUNS = 3


def run_dirs(cell: Path) -> list[Path]:
    """Real run directories under a cell, oldest first; `latest` is a symlink."""
    return sorted(p for p in cell.iterdir() if p.is_dir() and not p.is_symlink())


def score(cell: Path, trajectory_file: str, scorer) -> tuple[list[dict], list[tuple[str, str]]]:
    """`scorer(path) -> dict of numbers` per run. Returns (scored, skipped-with-reason)."""
    scored, skipped = [], []
    for d in run_dirs(cell):
        if (d / "INTERRUPTED").exists():
            skipped.append((d.name, "interrupted"))
            continue
        f = d / trajectory_file
        if not f.exists():
            skipped.append((d.name, f"no {trajectory_file}"))
            continue
        scored.append({"run": d.name, **scorer(f)})
    return scored, skipped


def summarise(scored: list[dict], keys: tuple[str, ...]) -> dict | None:
    """median/min/max per key over the scored runs; None below MIN_RUNS."""
    if len(scored) < MIN_RUNS:
        return None
    out = {"n": len(scored)}
    for k in keys:
        v = np.array([s[k] for s in scored], dtype=float)
        out[k] = {"median": float(np.median(v)), "min": float(v.min()), "max": float(v.max())}
    return out


KEYS = ("rmse_mm", "p90_mm", "rot_deg", "scale", "drift_pct")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--method", required=True)
    ap.add_argument("--stream", required=True)
    ap.add_argument("--cell", required=True, help="runs/<dataset>/<method>/<stream>")
    ap.add_argument("--reference", required=True)
    ap.add_argument("--trajectory-file", default="trajectory.tum")
    ap.add_argument("--alignment", default=None)
    args = ap.parse_args()

    cfg, mcfg = load_dataset(args.config), load_method(args.method)
    stream = cfg.stream(args.stream)
    agent = stream["agent"]
    E = mcfg.sensor_extrinsic(cfg, args.stream)
    ref = load_tum(args.reference, name=f"reference/{agent}")
    ev = cfg.eval
    mode = args.alignment or ev.get("alignment", "se3")

    def scorer(path: Path) -> dict:
        est = load_tum(path, name=f"{mcfg.name}/{args.stream}")
        est_ref = to_reference_frame(est, E, cfg.agent_reference(agent).get("frame", ""))
        a = ate(est_ref, ref, mode=mode, max_gap_s=ev.get("max_gap_s", 0.25),
                n_first=ev.get("align_n_first"),
                reference_uncertainty_m=cfg.reference_uncertainty_m())
        return {"poses": len(est), "lost": est.lost,
                "rmse_mm": a.trans.rmse * 1e3, "p90_mm": a.trans.p90 * 1e3,
                "rot_deg": a.rot.rmse, "scale": a.alignment.scale_observed,
                "drift_pct": a.drift_percent}

    cell = Path(args.cell)
    scored, skipped = score(cell, args.trajectory_file, scorer)
    print(f"{mcfg.name} / {args.stream} / {args.trajectory_file}  align={mode}")
    print(f"{'run':<18} {'poses':>6} {'lost':>5} {'RMSE mm':>8} {'p90 mm':>7} "
          f"{'rot deg':>7} {'scale':>7} {'drift %':>7}")
    for s in scored:
        print(f"{s['run']:<18} {s['poses']:>6} {s['lost']:>5} {s['rmse_mm']:>8.1f} "
              f"{s['p90_mm']:>7.1f} {s['rot_deg']:>7.2f} {s['scale']:>7.4f} {s['drift_pct']:>7.2f}")
    for name, why in skipped:
        print(f"{name:<18} skipped: {why}")
    summ = summarise(scored, KEYS)
    if summ is None:
        print(f"N={len(scored)} < {MIN_RUNS}: samples only, no spread reported")
        return 0 if scored else 1
    print(f"N={summ['n']}  median [min, max]")
    for k, unit in (("rmse_mm", "mm"), ("p90_mm", "mm"), ("rot_deg", "deg"),
                    ("scale", ""), ("drift_pct", "%")):
        m = summ[k]
        p = 4 if k == "scale" else (2 if k in ("rot_deg", "drift_pct") else 1)
        print(f"  {k:<10} {m['median']:.{p}f} [{m['min']:.{p}f}, {m['max']:.{p}f}] {unit}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
