#!/usr/bin/env python3
"""The construction's own table: one row per build, with what made it.

    python3 scripts/construction_table.py --runs runs/coop2_20260828

`aggregate.py` prints every `pseudo_gt_*` build as an indistinguishable row
called `pseudo_gt_v2c`, which is useless the moment there is more than one --
after the first certification sweep there were eight, differing in which board
was held out and what sigma was declared, and nothing on screen said which was
which. A construction row without its provenance is not a result.

Each row carries:

  variant   which boards went IN, which were held OUT, and any hand-declared
            sigma. This is the experiment.
  ATE/rot   against the held-out LiDAR reference. For v2c this is scored
            UNALIGNED (the method config declares `alignment: none`), so it is
            the error in the surveyed map frame with no use of the reference.
  RPE 1 m   local accuracy, which the anchors are expected to COST.
  moved     how far the solve carried the trajectory from its initial
            placement, and the whitened residuals.
  tier 2    per board, marked `self` where that build CONSUMED the board
            (self-agreement: the residual is the standard error of the
            observations it was fitted to, not evidence) and `INDEP` where it
            did not. Every v1 row is independent on both boards, since the
            no-anchor floor consumes none.

`under` flags a build the solver reported as under-constrained: near-zero
whitened residuals together with a large move, which means a family of
solutions fit and this is merely one of them.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def collect(runs_root: Path) -> list[dict]:
    rows = []
    for gj in sorted(runs_root.glob("pseudo_gt_*/*/*/graph.json")):
        d = gj.parent
        if d.is_symlink():
            continue
        try:
            g = json.loads(gj.read_text())
        except (OSError, json.JSONDecodeError) as e:
            rows.append({"dir": d, "error": f"unreadable graph.json ({e})"})
            continue
        m = {}
        mf = d / "metrics.json"
        if mf.exists():
            try:
                m = json.loads(mf.read_text())
            except (OSError, json.JSONDecodeError) as e:
                rows.append({"dir": d, "error": f"unreadable metrics.json ({e})"})
                continue
        consumed = [c["anchor"] for c in g.get("anchors", {}).get("consumed", [])]
        held = list(g.get("anchors", {}).get("held_out", []))
        sig = g.get("sigma", {})
        rows.append({
            "dir": d, "rung": g.get("rung"), "source": Path(g.get("source_run", "")).name,
            "consumed": consumed, "held_out": held,
            "declared_sigma": not sig.get("reference_informed", True),
            "sigma_t_mm": sig.get("sigma_t_m", float("nan")) * 1e3,
            "sigma_r_deg": sig.get("sigma_r_deg", float("nan")),
            "moved_mm": g.get("moved_from_initial_mm", {}).get("median"),
            "rms": g.get("solve", {}).get("final_rms", {}),
            "under": bool(g.get("under_constrained")),
            "metrics": m,
        })
    return rows


def variant(r: dict) -> str:
    if r["rung"] == "v1":
        return "odometry only"
    v = "+".join(r["consumed"]) or "NO BOARD"
    if r["held_out"]:
        v += f"  (held out {','.join(r['held_out'])})"
    if r["declared_sigma"]:
        v += f"  [sigma {r['sigma_t_mm']:.1f}mm/{r['sigma_r_deg']:.3f}deg]"
    return v


def main() -> int:                                                 # pragma: no cover
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    rows = collect(Path(args.runs))
    if args.json:
        print(json.dumps([{k: (str(v) if isinstance(v, Path) else v) for k, v in r.items()}
                          for r in rows], indent=2, default=str))
        return 0
    if not rows:
        print(f"no construction builds under {args.runs}. Run scripts/build_pseudo_gt.py first.")
        return 0

    print(f"{'rung':<5} {'variant':<52} {'ATE mm':>8} {'rot':>6} {'RPE1m':>7} "
          f"{'moved mm':>9} {'rms o/a':>11}  tier 2 per board")
    for r in sorted(rows, key=lambda x: (x.get("rung") or "", str(x.get("consumed")),
                                         str(x.get("declared_sigma")))):
        if r.get("error"):
            print(f"{'?':<5} {r['dir'].name:<46} {r['error']}")
            continue
        m = r["metrics"]
        if not m or not m.get("ate"):
            print(f"{r['rung']:<5} {variant(r):<52} {'not scored -- run scripts/rescore.py':>8}")
            continue
        ate = m["ate"]["ate_trans_m"]["rmse"] * 1e3
        rot = m["ate"]["ate_rot_deg"]["rmse"]
        r1 = next((x for x in m.get("rpe", []) if x["unit"] == "m" and x["delta"] == 1), None)
        rpe = (r1 or {}).get("rpe_trans_m", {}).get("rmse")
        rms = r["rms"] or {}
        t2 = []
        for row in (m.get("absolute_check") or []):
            if not isinstance(row, dict) or row.get("residual_m") is None:
                continue
            # Independent iff this build did NOT consume the board. v1
            # consumes none, so ALL of its board rows are independent
            # evidence -- labelling them `self` would hide the only tier-2
            # numbers the no-anchor floor has.
            tag = "self" if row["anchor"] in r["consumed"] else "INDEP"
            t2.append(f"{row['anchor']} {row['residual_m'] * 1e3:.1f}/{row['residual_deg']:.2f}deg {tag}")
        mark = " UNDER-CONSTRAINED" if r["under"] else ""
        print(f"{r['rung']:<5} {variant(r):<52} {ate:8.1f} {rot:6.2f} "
              f"{(rpe * 1e3 if rpe else float('nan')):7.1f} "
              f"{(r['moved_mm'] or 0):9.1f} "
              f"{rms.get('between', float('nan')):.2f}/{rms.get('prior', float('nan')):.2f}".ljust(102)
              + "  " + "; ".join(t2) + mark)

    print("\nALIGNMENT: v1 rows are se3-fitted TO the reference; v2c rows are scored "
          "UNALIGNED in the surveyed map frame. A v2c ATE is therefore the error a user "
          "of the trajectory sees, and it is not the same measurement as a v1 ATE.")
    print("TIER 2: `self` means that board was consumed by this build, so the residual is "
          "the standard error of the observations it was fitted to -- not evidence. "
          "`INDEP` is the only independent board number, and the LiDAR ATE is the other.")
    return 0


if __name__ == "__main__":                                          # pragma: no cover
    sys.exit(main())
