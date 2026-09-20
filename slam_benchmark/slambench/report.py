"""Result tables. Markdown, one row per (method, agent, stream).

Two rules the formatter enforces so the reader cannot miss them:

  * modality blocks are never merged into one ranking. A LiDAR method and an
    RGB-D method on the same sequence are not competitors; putting them in one
    sorted table invites exactly the comparison the input budgets forbid.
  * an ATE below the reference's own uncertainty is printed with a dagger and
    footnoted, not as a win.
"""
from __future__ import annotations

import json
from pathlib import Path

__all__ = ["trajectory_table", "map_table", "write_json", "load_runs"]

_DAGGER = "†"


def _fmt(v, nd=3, scale=1.0):
    if v is None:
        return "—"
    try:
        return f"{float(v) * scale:.{nd}f}"
    except (TypeError, ValueError):
        return str(v)


def _blocks(runs: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for r in runs:
        out.setdefault("+".join(r.get("modality", ["unknown"])), []).append(r)
    return out


def _size(scale_observed) -> str:
    """`scale_observed` multiplies the ESTIMATE to reach the reference, so it
    reads backwards. This column is the estimate's size relative to truth:
    +5% means the estimate is 5% too big. Both are printed because the raw
    figure is the one in metrics.json, and the derived one is the one that
    was misread for a day."""
    if scale_observed in (None, 0):
        return "—"
    d = (1.0 / scale_observed - 1.0) * 100
    if abs(d) < 0.5:
        return "≈true"
    return f"{d:+.1f}%"


def trajectory_table(runs: list[dict], reference_uncertainty_m: float) -> str:
    lines, footnote = [], False
    for block, rows in sorted(_blocks(runs).items()):
        lines += [f"### {block}", "",
                  "| method | agent | stream | align | ATE RMSE (mm) | ATE p90 (mm) "
                  "| rot RMSE (deg) | RPE 1 m (mm) | drift (%) | scale obs. "
                  "| est. size | cov. |",
                  "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        rows = sorted(rows, key=lambda r: (r.get("ate") or {}).get("ate_trans_m", {})
                      .get("rmse", float("inf")))
        for r in rows:
            a = r.get("ate")
            if not a:
                lines.append(f"| {r['method']} | {r['agent']} | {r.get('stream','—')} "
                             f"| — | _{r.get('status','no result')}_ ||||||||")
                continue
            t = a["ate_trans_m"]
            mark = _DAGGER if a.get("below_reference_uncertainty") else ""
            footnote = footnote or bool(mark)
            r1 = next((x for x in r.get("rpe", []) if x["unit"] == "m" and x["delta"] == 1), None)
            lines.append(
                f"| {r['method']} | {r['agent']} | {r.get('stream','—')} "
                f"| {a['alignment']['mode']} | {_fmt(t['rmse'],1,1e3)}{mark} "
                f"| {_fmt(t['p90'],1,1e3)} | {_fmt(a['ate_rot_deg']['rmse'],2)} "
                f"| {_fmt((r1 or {}).get('rpe_trans_m',{}).get('rmse'),1,1e3)} "
                f"| {_fmt(a['drift_percent'],2)} | {_fmt(a['alignment']['scale_observed'],4)} "
                f"| {_size(a['alignment']['scale_observed'])} "
                f"| {_fmt(a['coverage'],2)} |")
        lines.append("")
    lines += ["_`scale obs.` multiplies the ESTIMATE to reach the reference, so a value "
              "above 1 means the estimate is SMALLER than truth. `est. size` states the "
              "same fact the other way round and is the one to quote._", ""]
    if footnote:
        lines += [f"{_DAGGER} ATE RMSE is below the reference trajectory's own stated "
                  f"uncertainty ({reference_uncertainty_m * 1e3:.0f} mm). The method is "
                  f"indistinguishable from the reference on this sequence, not better "
                  f"than the methods above it — read the RPE column instead.", ""]
    return "\n".join(lines)


def map_table(runs: list[dict]) -> str:
    rows = [r for r in runs if r.get("map")]
    if not rows:
        return "_No run produced a map; nothing to score._\n"
    res = {r["map"]["resolution_m"] for r in rows}
    lines = [f"Voxel resolution {'/'.join(f'{x*100:.0f}' for x in sorted(res))} cm; "
             f"distances truncated at {rows[0]['map']['truncate_m']*100:.0f} cm.", "",
             "| method | agent | acc. mean (mm) | compl. mean (mm) | Chamfer (mm) "
             "| F@2cm | F@5cm | F@10cm | trunc. acc/compl | points |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in sorted(rows, key=lambda r: r["map"]["chamfer_m"]):
        m = r["map"]
        lines.append(
            f"| {r['method']} | {r['agent']} | {_fmt(m['accuracy_m']['mean'],1,1e3)} "
            f"| {_fmt(m['completeness_m']['mean'],1,1e3)} | {_fmt(m['chamfer_m'],1,1e3)} "
            f"| {_fmt(m['fscore']['0.02']['f'],3)} | {_fmt(m['fscore']['0.05']['f'],3)} "
            f"| {_fmt(m['fscore']['0.10']['f'],3)} "
            f"| {_fmt(m['accuracy_truncated'],2)}/{_fmt(m['completeness_truncated'],2)} "
            f"| {m['n_est']} |")
    return "\n".join(lines) + "\n"


def write_json(obj: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=False, default=str) + "\n")


def load_runs(root: str | Path) -> list[dict]:
    """Every `metrics.json` under `root`, newest layout first."""
    return [json.loads(p.read_text()) for p in sorted(Path(root).rglob("metrics.json"))]
