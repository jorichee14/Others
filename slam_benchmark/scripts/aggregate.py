#!/usr/bin/env python3
"""Collect every metrics.json under a runs root into one markdown report.

    python3 scripts/aggregate.py --runs runs/coop2_20260828 \
        --config configs/coop2.yaml --out results/coop2.md
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from slambench import load_dataset                                     # noqa: E402
from slambench.report import load_runs, map_table, trajectory_table    # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = load_dataset(args.config)
    runs = load_runs(args.runs)
    if not runs:
        print(f"no metrics.json under {args.runs} — run scripts/eval_run.py first",
              file=sys.stderr)
        return 1

    u = cfg.reference_uncertainty_m()
    forced = [r for r in runs if (r.get("run") or {}).get("forced")]
    rates = {(r.get("run") or {}).get("replay_rate", 1.0) for r in runs}

    parts = [
        f"# {cfg.name} — SLAM benchmark", "",
        f"{len(runs)} runs. Reference: **{cfg.reference.get('kind')}** "
        f"(uncertainty {u * 1e3:.0f} mm).", "",
        "> The reference trajectory is the offline mapping pipeline's own output. "
        "Tier-1 numbers below are **agreement with that pipeline**, not accuracy. "
        "The anchor table is the only tier whose error bar is independent of the "
        "thing being measured.", "",
    ]
    if len(rates) > 1:
        parts += [f"> ⚠ Runs in this table used different bag replay rates ({sorted(rates)}). "
                  f"Runtime columns are not comparable across them, and a rate above 1.0 "
                  f"changes which frames a method saw.", ""]
    if forced:
        parts += [f"> ⚠ {len(forced)} run(s) were forced past preflight: "
                  f"{', '.join(sorted({r['method'] + '/' + r['agent'] for r in forced}))}. "
                  f"Their configuration was known to be incomplete before they started.", ""]

    parts += ["## Trajectory", "", trajectory_table(runs, u), "", "## Map", "", map_table(runs)]

    # Tier 2 rolls up across runs; it is the same anchors for all of them.
    rows = [(r, a) for r in runs if isinstance(r.get("absolute_check"), list)
            for a in r["absolute_check"] if a.get("residual_m") is not None]
    if rows:
        parts += ["", "## Absolute check (tier 2, independent)", "",
                  "| method | agent | anchor | residual (mm) | survey σ (mm) | verdict |",
                  "|---|---|---|---:|---:|---|"]
        for r, a in sorted(rows, key=lambda x: x[1]["residual_m"]):
            parts.append(f"| {r['method']} | {r['agent']} | {a['anchor']} "
                         f"| {a['residual_m'] * 1e3:.1f} "
                         f"| {(a.get('uncertainty_m') or 0) * 1e3:.0f} | {a['verdict']} |")
    else:
        parts += ["", "## Absolute check (tier 2, independent)", "",
                  "_Silent: no anchor has a dwell window in the dataset config. "
                  "Until one does, every number in this report is relative to a "
                  "reference that was itself estimated._"]

    text = "\n".join(parts) + "\n"
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text)
        print(f"-> {args.out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
