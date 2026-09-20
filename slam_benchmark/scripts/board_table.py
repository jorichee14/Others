#!/usr/bin/env python3
"""Tier 2 across every scored run: the board residual per method, side by side.

    python3 scripts/board_table.py --runs runs/coop2_20260828 [--agent mobile_1]

ATE says how far an estimate is from the reference trajectory, and on this
sequence the reference is itself a pipeline output -- on mobile_2 it is derived
from the very odometry being scored. The board residual is the only number in
this benchmark whose error bar does NOT come from the thing being measured: a
surveyed marker at 3-15 mm, seen by the camera, compared against where the
estimate says the camera was. So it belongs in its own table, read separately
from ATE, and a method that scores well on one and badly on the other is
telling you something about the reference.

Reads the `absolute_check` and `anchor_alignment` blocks that eval_run.py has
already written into each metrics.json. Nothing is recomputed here.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def collect(runs_root: Path, agent: str | None = None) -> list[dict]:
    """One flat row per (run, anchor), newest metrics last."""
    rows = []
    for p in sorted(runs_root.rglob("metrics*.json")):
        m = json.loads(p.read_text())
        if agent and m.get("agent") != agent:
            continue
        ac = m.get("absolute_check")
        if not isinstance(ac, list):
            continue
        aa = m.get("anchor_alignment") or {}
        for r in ac:
            rows.append({
                "method": m.get("method"), "agent": m.get("agent"),
                "stream": m.get("stream"), "run": Path(m.get("run_dir", "")).name,
                "anchor": r.get("anchor"), "n": r.get("n", 0),
                "residual_m": r.get("residual_m"), "residual_deg": r.get("residual_deg"),
                "uncertainty_m": r.get("uncertainty_m"), "spread_m": r.get("spread_m"),
                "interp_gap_s": r.get("interp_gap_s"),
                "align": aa.get("mode", "stale"),
                "independent": aa.get("independent", True),
                # A metrics.json written before the tier-2 fixes carries no
                # anchor_alignment block. Its residual was computed under the
                # old rules and must not sit in a table beside the new ones.
                "stale": not aa,
                "verdict": r.get("verdict", ""),
            })
    return rows


def _mm(v) -> str:
    return "—" if v is None else f"{v * 1e3:.1f}"


def render(rows: list[dict]) -> str:
    if not rows:
        return ("no absolute_check in any metrics.json under that root. Score a run "
                "first (scripts/eval_run.py), and check that the dataset config "
                "declares reference.anchors[*].windows_by_agent.")
    out = [f"{'method':<18} {'agent':<9} {'anchor':<10} {'resid':>8} {'σ':>6} "
           f"{'n':>4} {'deg':>6} {'align':>5}  note"]
    for r in sorted(rows, key=lambda r: (r["method"] or "", r["agent"] or "", r["anchor"] or "")):
        note = []
        if r["stale"]:
            note.append("STALE: scored before the tier-2 fixes; re-run eval_run.py")
        if not r["independent"]:
            note.append("scale from reference: NOT independent")
        if r["interp_gap_s"]:
            note.append(f"interpolated across {r['interp_gap_s']:.2f} s")
        if not r["n"]:
            note.append(r["verdict"])
        deg = "—" if r["residual_deg"] is None else f"{r['residual_deg']:.2f}"
        out.append(f"{r['method']:<18} {r['agent']:<9} {r['anchor'] or '—':<10} "
                   f"{_mm(r['residual_m']):>8} {_mm(r['uncertainty_m']):>6} "
                   f"{r['n']:>4} {deg:>6} {r['align']:>5}  {'; '.join(note)}")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--agent", default=None)
    ap.add_argument("--json", action="store_true", help="the rows, unformatted")
    args = ap.parse_args()
    rows = collect(Path(args.runs), args.agent)
    if args.json:
        print(json.dumps(rows, indent=2))
    else:
        print(render(rows))
    return 0 if rows else 1


if __name__ == "__main__":
    sys.exit(main())
