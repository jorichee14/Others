#!/usr/bin/env python3
"""Which absolute sources reach into the gaps, and what gap is left after each.

    python3 scripts/source_coverage.py --config configs/coop2.yaml --agent mobile_2 \
        --reference runs/coop2_20260828/reference/mobile_2.tum \
        --peer-reference runs/coop2_20260828/reference/mobile_1.tum

The question this answers, and it is the one that decides whether the
sensor-constrained agent's ground truth can be trusted at all.

`mobile_2`'s construction is anchored by two board dwells with a **125.1 s gap
between them**. Inside that gap nothing observes the trajectory: the poses are
whatever the backbone's relative motion says, stretched between two pins, and
the error there is neither measured nor measurable from the boards -- a residual
at a board the construction CONSUMED is sigma/sqrt(n) self-agreement, not
evidence.

So the useful question is not "how accurate is the construction" (unanswerable
today) but **"which available source reaches into the gap, and how much gap is
left once it does".** A source whose coverage sits inside the existing dwells
adds precision where there already was some and nothing where there is none.
A source that spans the gap is the difference between an error bar and no
error bar.

Per source this reports coverage in TIME and in PATH, the union with the
boards, and the longest remaining blind interval. `infra_1` is computed from
its surveyed pose and its measured range ceiling; the peer from the two
reference trajectories' separation; prior-map registration is reported as
UNAVAILABLE while `reference.map_cloud` is null, because its coverage depends
on where the map actually has geometry and guessing that would be the kind of
plausible-and-wrong number this harness exists to refuse.

Nothing here scores anything. It says where evidence exists.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from slambench import load_dataset, load_tum, se3                      # noqa: E402


def intervals_from_mask(t: np.ndarray, m: np.ndarray,
                        min_s: float = 0.0) -> list[tuple[float, float]]:
    """Contiguous [t_start, t_end] where `m` holds, dropping runs under min_s."""
    out, i = [], None
    for k, flag in enumerate(m):
        if flag and i is None:
            i = k
        elif not flag and i is not None:
            if t[k - 1] - t[i] >= min_s:
                out.append((float(t[i]), float(t[k - 1])))
            i = None
    if i is not None and t[-1] - t[i] >= min_s:
        out.append((float(t[i]), float(t[-1])))
    return out


def union(spans: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if not spans:
        return []
    s = sorted(spans)
    out = [list(s[0])]
    for a, b in s[1:]:
        if a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def gaps(spans: list[tuple[float, float]], t0: float,
         t1: float) -> list[tuple[float, float]]:
    """The intervals of [t0, t1] NOT covered, including the leading and
    trailing stretches -- an uncovered END is a lever and an uncovered
    INTERIOR is a bridge, and they fail differently."""
    u = union(spans)
    out, prev = [], t0
    for a, b in u:
        if a > prev:
            out.append((prev, a))
        prev = max(prev, b)
    if prev < t1:
        out.append((prev, t1))
    return out


def path_between(traj, a: float, b: float) -> float:
    """Metres of path inside [a, b]. A gap in seconds understates the risk on a
    fast stretch and overstates it on a slow one; the metres are what the
    drift actually accumulates over."""
    m = (traj.stamps >= a) & (traj.stamps <= b)
    if m.sum() < 2:
        return 0.0
    p = traj.positions[m]
    return float(np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1)))


def describe(name: str, spans, traj, t0, t1) -> dict:
    cov_s = sum(b - a for a, b in union(spans))
    cov_m = sum(path_between(traj, a, b) for a, b in union(spans))
    g = gaps(spans, t0, t1)
    interior = [x for x in g if x[0] > t0 + 1e-6 and x[1] < t1 - 1e-6]
    lead = next((b - a for a, b in g if a <= t0 + 1e-6), 0.0)
    trail = next((b - a for a, b in g if b >= t1 - 1e-6), 0.0)
    return {"source": name, "n_spans": len(union(spans)),
            "cov_s": cov_s, "cov_frac": cov_s / max(t1 - t0, 1e-9), "cov_m": cov_m,
            "longest_interior_s": max((b - a for a, b in interior), default=0.0),
            "longest_interior_m": max((path_between(traj, a, b) for a, b in interior),
                                      default=0.0),
            "leading_s": lead, "trailing_s": trail, "spans": union(spans)}


def row(d: dict) -> str:
    return (f"  {d['source']:<26} {d['n_spans']:>3} spans  "
            f"{d['cov_s']:7.1f} s ({d['cov_frac']:5.1%})  {d['cov_m']:6.1f} m   "
            f"worst interior gap {d['longest_interior_s']:7.1f} s / "
            f"{d['longest_interior_m']:5.1f} m   "
            f"ends {d['leading_s']:.1f}/{d['trailing_s']:.1f} s")


def main() -> int:                                                 # pragma: no cover
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--agent", required=True)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--peer-reference", default=None,
                    help="the OTHER agent's reference TUM, for co-observation coverage")
    ap.add_argument("--peer-range-m", type=float, default=10.0,
                    help="separation under which the peer could plausibly observe this "
                         "agent. A DECLARED threshold, not a measurement -- report it "
                         "with the number.")
    ap.add_argument("--min-span-s", type=float, default=1.0)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    cfg = load_dataset(args.config)
    ref = load_tum(args.reference)
    t0, t1 = float(ref.stamps[0]), float(ref.stamps[-1])
    print(f"\n=== source coverage   {args.agent}   {t1 - t0:.1f} s, "
          f"{ref.path_length():.1f} m\n")

    rows, board_spans = [], []
    for a in cfg.raw["reference"].get("anchors", []):
        w = (a.get("windows_by_agent") or {}).get(args.agent) or a.get("window")
        if not w:
            continue
        lo, hi = float(w[0]), float(w[1])
        if hi < t0 or lo > t1:
            continue
        board_spans.append((max(lo, t0), min(hi, t1)))
    if board_spans:
        rows.append(describe("boards (consumed)", board_spans, ref, t0, t1))

    # infra_1: a surveyed static observer, in range iff inside its MEASURED
    # ceiling. Not consumed by any construction, which is what makes it a check.
    infra = (cfg.raw.get("streams") or {}).get("infra_1.radar") or {}
    pose = infra.get("static_world_pose")
    rmax = infra.get("max_range_measured_m")
    if pose and rmax:
        c = np.array([pose["x"], pose["y"], pose["z"]], float)
        d = np.linalg.norm(ref.positions - c, axis=1)
        spans = intervals_from_mask(ref.stamps, d <= float(rmax), args.min_span_s)
        r = describe(f"infra_1 (<= {rmax} m)", spans, ref, t0, t1)
        r["range_median_m"] = float(np.median(d))
        r["range_min_m"] = float(d.min())
        rows.append(r)
    else:
        print("  infra_1                    NO static_world_pose or max_range_measured_m\n")

    if args.peer_reference:
        peer = load_tum(args.peer_reference)
        at = peer.interpolate(ref.stamps, max_gap_s=0.5)
        i = np.searchsorted(ref.stamps, at.stamps)
        sep = np.linalg.norm(ref.positions[i] - at.positions, axis=1)
        spans = intervals_from_mask(at.stamps, sep <= args.peer_range_m, args.min_span_s)
        r = describe(f"peer (<= {args.peer_range_m:g} m)", spans, ref, t0, t1)
        r["separation_median_m"] = float(np.median(sep))
        r["separation_min_m"] = float(sep.min())
        rows.append(r)

    if cfg.raw["reference"].get("map_cloud") is None:
        print("  prior-map registration     UNAVAILABLE -- reference.map_cloud is null.\n"
              "                             Its coverage is wherever the map HAS geometry,\n"
              "                             which the map alone can answer. Not guessed.\n")

    for r in rows:
        print(row(r))

    if len(rows) > 1:
        allspans = [s for r in rows for s in r["spans"]]
        u = describe("UNION of all above", allspans, ref, t0, t1)
        print("\n" + row(u))
        b = next((r for r in rows if r["source"].startswith("boards")), None)
        if b:
            print(f"\n  The boards alone leave a {b['longest_interior_s']:.1f} s interior gap. "
                  f"Together the sources leave {u['longest_interior_s']:.1f} s.")
            if u["longest_interior_s"] < 0.5 * max(b["longest_interior_s"], 1e-9):
                print("  VERDICT: the extra sources reach into the gap. A held-out residual "
                      "there is\n           the first independent evidence this agent can "
                      "have -- wire them as CHECKS\n           (never as factors, or the "
                      "residual becomes self-agreement).")
            else:
                print("  VERDICT: the extra sources do NOT materially shorten the blind "
                      "interval. They\n           add precision where there already was "
                      "some. The gap needs a DENSE source\n           (prior-map "
                      "registration or visual localisation against a reference model),\n"
                      "           not another sparse one.")

    print("\nNOT AN ACCURACY STATEMENT. Coverage says where evidence EXISTS, not how good "
          "it is.\nA source must also be held OUT of the construction to be evidence about "
          "it.")
    if args.json:
        import json
        Path(args.json).write_text(json.dumps(rows, indent=2, default=float))
        print(f"\n  wrote {args.json}")
    return 0


if __name__ == "__main__":                                          # pragma: no cover
    sys.exit(main())
