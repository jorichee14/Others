#!/usr/bin/env python3
"""The error as a distribution with an interval, not as one RMSE.

    python3 scripts/error_distribution.py --config configs/coop2.yaml \
        --run runs/coop2_20260828/pseudo_gt_v2c/mobile_1/<stamp>

`199 mm` is an RMSE and an RMSE hides its own tail. A construction with a 5 cm
median and a 2 m worst stretch is unusable for what a pseudo-GT is used for,
and no single number separates that from a uniform 20 cm. This prints what
`docs/PSEUDO_GT.md` V3(a) actually asks for: percentiles, the worst stretch a
user would hit, and a confidence interval from a BLOCK bootstrap over time
segments.

The block part is the whole point. Trajectory error is heavily autocorrelated,
so ~1500 poses are a handful of independent failure episodes rather than 1500
samples, and a per-pose interval would be too narrow by roughly an order of
magnitude. The block length is read off the error series' own autocorrelation
and the number of independent episodes is printed beside the interval, because
an interval from five blocks is a statement about five things.

ALIGNMENT is taken from the run, never chosen here. A v2c construction is
scored `none` -- unaligned, in the surveyed map frame -- and re-fitting it
would hand back the reference it is being measured against.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from slambench import fit, load_dataset, load_tum, se3                 # noqa: E402
from slambench.blockstats import describe                              # noqa: E402


def error_series(est, ref, mode: str, max_gap_s: float = 0.2):
    """Per-pose translation and rotation error, exactly as `ate()` computes them.

    Duplicated rather than returned from `ate()` because AteResult carries
    summary statistics only. The caller CHECKS the rmse against the run's own
    metrics.json, so a divergence is caught rather than quietly reported.
    """
    ref_i = ref.interpolate(est.stamps, max_gap_s=max_gap_s)
    est_i = est.subset(np.searchsorted(est.stamps, ref_i.stamps))
    est_a = fit(est_i, ref_i, mode=mode).apply(est_i)
    e_t = np.linalg.norm(est_a.positions - ref_i.positions, axis=1)
    e_r = np.array([np.degrees(se3.rotation_angle(
        ref_i.poses[i][:3, :3].T @ est_a.poses[i][:3, :3])) for i in range(len(ref_i))])
    return est_a.stamps, e_t, e_r


def show(title: str, unit: str, scale: float, d) -> None:
    p = d.percentiles
    print(f"\n{title}")
    print(f"  n {d.n} over {d.duration_s:.1f} s")
    print(f"  rmse      {d.rmse * scale:8.1f} {unit}")
    print(f"  median    {p[50] * scale:8.1f} {unit}      p90 {p[90] * scale:8.1f}      "
          f"p95 {p[95] * scale:8.1f}      p99 {p[99] * scale:8.1f}      "
          f"max {p[100] * scale:8.1f}")
    w = d.worst_window
    print(f"  worst {w['duration_s']:.1f} s   median {w['median'] * scale:.1f} {unit}, "
          f"max {w['max'] * scale:.1f} {unit}, starting t+{w['start_s'] - 0:.0f} "
          f"(absolute stamp)")
    print(f"  correlation time {d.correlation_time_s:.1f} s  ->  "
          f"{d.effective_n:.1f} independent episodes, block {d.block_s:.1f} s")
    if d.ci:
        for k, (lo, hi) in d.ci.items():
            print(f"  95% CI {k:<7} [{lo * scale:.1f}, {hi * scale:.1f}] {unit}")
    for wmsg in d.warnings:
        print(f"  WARNING: {wmsg}")


def main() -> int:                                                 # pragma: no cover
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--run", required=True, help="a run directory with run.json")
    ap.add_argument("--reference", default=None,
                    help="override the reference TUM (default: from run.json's agent)")
    ap.add_argument("--window-s", type=float, default=5.0,
                    help="length of the worst-stretch window (default 5 s). Report it "
                         "with the number: it is a choice and it moves the answer.")
    ap.add_argument("--block-s", type=float, default=None,
                    help="bootstrap block length. Default 2x the measured correlation "
                         "time, which is the honest choice.")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--json", default=None, help="also write the numbers here")
    args = ap.parse_args()

    run = Path(args.run)
    rj = json.loads((run / "run.json").read_text())
    agent = rj.get("agent") or rj.get("stream", "").split(".")[0]
    cfg = load_dataset(args.config)

    est_file = next((run / n for n in ("pseudo_gt.tum", "graph.tum", "odometry.tum")
                     if (run / n).exists()), None)
    if est_file is None:
        raise SystemExit(f"no trajectory in {run}")
    ref_file = Path(args.reference) if args.reference else \
        run.parents[2] / "reference" / f"{agent}.tum"
    if not ref_file.exists():
        raise SystemExit(f"reference not found: {ref_file}. Pass --reference.")

    mfile = run / "metrics.json"
    m = json.loads(mfile.read_text()) if mfile.exists() else {}
    mode = (m.get("ate", {}).get("alignment", {}) or {}).get("mode")
    if mode is None:
        raise SystemExit(f"{mfile} has no recorded alignment. Re-score with "
                         f"scripts/rescore.py; choosing one here would be a guess.")

    est, ref = load_tum(est_file), load_tum(ref_file)
    t, e_t, e_r = error_series(est, ref, mode)

    print(f"\n=== {run.name}   {rj.get('method', '?')} x {agent}")
    print(f"  estimate  {est_file.name}  ({len(est)} poses)")
    print(f"  reference {ref_file}")
    print(f"  alignment {mode}" + ("   (unaligned -- surveyed map frame)"
                                   if mode == "none" else "   (FITTED to the reference)"))

    recorded = (m.get("ate", {}).get("ate_trans_m", {}) or {}).get("rmse")
    if recorded is not None and abs(recorded - float(np.sqrt(np.mean(e_t ** 2)))) > 1e-6:
        raise SystemExit(
            f"recomputed rmse {np.sqrt(np.mean(e_t ** 2)) * 1e3:.3f} mm disagrees with "
            f"metrics.json's {recorded * 1e3:.3f} mm. The series below would not be the "
            f"error that was reported; fix that before reading anything from it.")

    show("TRANSLATION", "mm", 1e3, describe(t, e_t, args.window_s, args.block_s, args.n_boot))
    show("ROTATION", "deg", 1.0, describe(t, e_r, args.window_s, args.block_s, args.n_boot))

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(
            {"run": str(run), "alignment": mode,
             "translation": describe(t, e_t, args.window_s, args.block_s,
                                     args.n_boot).as_dict(),
             "rotation": describe(t, e_r, args.window_s, args.block_s,
                                  args.n_boot).as_dict()}, indent=2))
        print(f"\n  wrote {args.json}")

    print("\nHOW TO QUOTE THIS. The median is what a typical pose costs; the p95 and the "
          "worst stretch are what a USER hits and are the numbers a pseudo-GT lives or "
          "dies by. The interval is over independent episodes, so quote it with that "
          "count -- it is not an n=poses result and must not be read as one.")
    return 0


if __name__ == "__main__":                                          # pragma: no cover
    sys.exit(main())
