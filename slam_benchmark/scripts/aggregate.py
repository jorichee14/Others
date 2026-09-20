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

    # ------------------------------------------------- tier 2, the honest tier
    # Read from metrics*.json rather than the tier-1 `runs` list, so the
    # front-end's rows sit beside the graph's and a row whose scale was
    # borrowed from the reference is marked as what it is.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from board_table import collect                                   # noqa: E402
    brows = collect(Path(args.runs))
    scored = [r for r in brows if r.get("residual_m") is not None]
    parts += ["", "## Absolute check (tier 2)", "",
              "The only tier whose error bar does not come from the thing being "
              "measured: a surveyed marker seen by the camera, against where the "
              "estimate says the camera was. Where this disagrees with the "
              "trajectory table, believe this one — and say so.", ""]
    if scored:
        parts += ["| method | agent | from | anchor | residual (mm) | survey σ (mm) "
                  "| n | rot (deg) | caveat |",
                  "|---|---|---|---|---:|---:|---:|---:|---|"]
        for r in sorted(scored, key=lambda x: (x["agent"] or "", x["residual_m"])):
            caveat = []
            if not r["independent"]:
                caveat.append("**scale from the reference: not independent**")
            if r.get("interp_gap_s"):
                caveat.append(f"interpolated across {r['interp_gap_s']:.1f} s")
            if r.get("stale"):
                caveat.append("**stale: re-run scripts/rescore.py**")
            deg = "—" if r["residual_deg"] is None else f"{r['residual_deg']:.2f}"
            parts.append(f"| {r['method']} | {r['agent']} | {r['from']} | {r['anchor']} "
                         f"| {r['residual_m'] * 1e3:.1f} "
                         f"| {(r.get('uncertainty_m') or 0) * 1e3:.0f} | {r['n']} "
                         f"| {deg} | {'; '.join(caveat)} |")
    else:
        parts += ["_Silent: no anchor produced a residual. Until one does, every "
                  "number in this report is relative to a reference that was itself "
                  "estimated._"]
    empty = [r for r in brows if r.get("residual_m") is None]
    if empty:
        seen, lines = set(), []
        for r in empty:
            key = (r["agent"], r["anchor"], r["verdict"][:60])
            if key in seen:
                continue
            seen.add(key)
            lines.append(f"- `{r['agent']}` / `{r['anchor']}`: {r['verdict']}")
        parts += ["", "### Anchors that produced nothing", "",
                  "Each is evidence not collected, and each is a config line or a "
                  "detector run away.", ""] + lines

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
