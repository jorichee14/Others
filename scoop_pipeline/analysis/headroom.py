#!/usr/bin/env python3
"""Is the testbed finer than what it measures? (Paper A, experiment E1
"Headroom"): every error term of the testbed, in metres, against the errors
at which collaborative fusion breaks, on one log axis.

    python scoop_pipeline/analysis/headroom.py configs/headroom_terms.json
        [--traj traj.tum ...] [--speed-p 95] [--out headroom.png] [--min-margin 10]

The JSON lists the terms. A term is either a length in metres or a time in
seconds; a time becomes a length through the agent speed, by default the
--speed-p percentile of the speed of the --traj trajectories (TUM files of
the reference tracks), else the "speed_mps" in the JSON:

  {"speed_mps": 1.0,
   "breaking_points": [{"name": "pose: late fusion below ego-only", "m": 0.2, "source": "..."},
                       {"name": "delay: one LiDAR frame", "s": 0.1, "source": "..."}],
   "terms": [{"name": "clock offset, measured", "s": 0.00099, "source": "coop_2 ntp"},
             {"name": "RGB-D cart vs held-out boards", "m": 0.031, "source": "coop_2"},
             {"name": "radar reflector residual", "m": null, "source": "TODO"}]}

A term with null is drawn as missing and listed in the report, never dropped
silently. The pass criterion: every measured term at least --min-margin times
below the smallest breaking point. Prints the margin per term, writes the
figure, and the same numbers to <out>.json.
"""
import argparse
import json
import math
import os
import sys

import numpy as np


def p_speed(paths, p=95.0):
    """The p-th percentile of speed over TUM trajectories (t tx ty tz ...)."""
    v = []
    for path in paths:
        rows = [list(map(float, ln.split())) for ln in open(path)
                if ln.strip() and not ln.startswith("#")]
        if len(rows) < 2:
            continue
        a = np.array(rows)
        a = a[np.argsort(a[:, 0])]
        dt = np.diff(a[:, 0])
        ok = dt > 1e-3
        d = np.linalg.norm(np.diff(a[:, 1:4], axis=0), axis=1)
        v.append(d[ok] / dt[ok])
    if not v:
        return None
    return float(np.percentile(np.concatenate(v), p))


def to_metres(item, speed):
    if item.get("m") is not None:
        return float(item["m"])
    if item.get("s") is not None:
        return float(item["s"]) * speed if speed else None
    return None


def evaluate(cfg, speed, min_margin=10.0):
    """Each term is compared with the breaking point of its own kind: a length
    term with the pose point, a time term with the delay point (both in
    metres at the given speed). The cross-kind margin is reported too, since
    a clock error also displaces a measurement in space, but the pass is
    judged like against like."""
    bps = []
    for b in cfg["breaking_points"]:
        kind = b.get("kind") or ("delay" if b.get("s") is not None else "pose")
        bps.append({"name": b["name"], "m": to_metres(b, speed), "kind": kind,
                    "source": b.get("source", "")})
    smallest = {k: min([b["m"] for b in bps if b["kind"] == k and b["m"] is not None] or [None])
                for k in ("pose", "delay")}
    terms = []
    for t in cfg["terms"]:
        m = to_metres(t, speed)
        kind = t.get("kind") or ("delay" if t.get("s") is not None else "pose")
        own = smallest.get(kind)
        other = smallest.get("delay" if kind == "pose" else "pose")
        margin = (own / m) if (m and own) else None
        terms.append({"name": t["name"], "m": m, "kind": kind, "source": t.get("source", ""),
                      "from_time": t.get("s") is not None,
                      "margin": margin,
                      "margin_other_kind": (other / m) if (m and other) else None,
                      "pass": (margin >= min_margin) if margin is not None else None})
    return {"speed_mps": speed, "min_margin": min_margin, "breaking_points": bps,
            "smallest_breaking_point_m": smallest, "terms": terms}


def print_report(rep, log=print):
    sm = rep["smallest_breaking_point_m"]
    log(f"speed used for time terms: {rep['speed_mps']} m/s; breaking points: pose "
        f"{sm.get('pose')} m, delay {sm.get('delay')} m; pass at >= {rep['min_margin']:g}x "
        f"against the term's own kind")
    log("  %-44s %-5s %10s %8s %9s %s" % ("term", "kind", "metres", "margin", "x-kind", ""))
    for t in rep["terms"]:
        if t["m"] is None:
            log("  %-44s %-5s %10s %8s %9s MISSING (%s)" % (t["name"], t["kind"], "--", "--", "--",
                                                            t["source"]))
            continue
        xo = "%.1fx" % t["margin_other_kind"] if t["margin_other_kind"] else "--"
        log("  %-44s %-5s %10.4f %7.1fx %9s %s" % (t["name"], t["kind"], t["m"], t["margin"], xo,
                                                    "pass" if t["pass"] else "SHORT"))
    missing = [t["name"] for t in rep["terms"] if t["m"] is None]
    short = [t["name"] for t in rep["terms"] if t["pass"] is False]
    log("  verdict: %s" % ("passes on every measured term" if not short
                           else "short on: " + ", ".join(short)))
    if missing:
        log("  still missing: " + ", ".join(missing))


def plot(rep, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    terms = [t for t in rep["terms"] if t["m"] is not None]
    miss = [t for t in rep["terms"] if t["m"] is None]
    names = [t["name"] for t in terms] + [t["name"] + " (missing)" for t in miss]
    vals = [t["m"] for t in terms] + [np.nan] * len(miss)
    fig, ax = plt.subplots(figsize=(7, 0.35 * len(names) + 1.6))
    y = np.arange(len(names))
    ax.barh(y[:len(terms)], vals[:len(terms)],
            color=["#4a7ebb" if t["kind"] == "pose" else "#e59866" for t in terms])
    for yi in y[len(terms):]:
        ax.text(1e-4, yi, "missing", va="center", fontsize=8, color="grey")
    for b in rep["breaking_points"]:
        if b["m"]:
            ax.axvline(b["m"], color="#c0392b" if b["kind"] == "pose" else "#d68910", ls="--", lw=1)
            ax.text(b["m"], len(names) - 0.4, b["name"], rotation=90, va="top", ha="right",
                    fontsize=7, color="#c0392b")
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("error (m, log); time terms at %.2f m/s" % rep["speed_mps"])
    ax.invert_yaxis()
    ax.grid(axis="x", which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("terms_json")
    ap.add_argument("--traj", nargs="*", default=[], help="TUM reference tracks for the speed percentile")
    ap.add_argument("--speed-p", type=float, default=95.0)
    ap.add_argument("--min-margin", type=float, default=10.0)
    ap.add_argument("--out", default=None, help="figure path (default next to the JSON)")
    a = ap.parse_args()
    cfg = json.load(open(a.terms_json))
    speed = p_speed(a.traj, a.speed_p) if a.traj else cfg.get("speed_mps")
    if speed is None:
        sys.exit("no speed: give --traj files or speed_mps in the JSON")
    rep = evaluate(cfg, speed, a.min_margin)
    rep["speed_source"] = ("p%g of %s" % (a.speed_p, ", ".join(a.traj))) if a.traj else "speed_mps in JSON"
    print_report(rep)
    out = a.out or os.path.splitext(a.terms_json)[0] + ".png"
    plot(rep, out)
    json.dump(rep, open(os.path.splitext(out)[0] + ".json", "w"), indent=1)
    print(f"  wrote {out} and {os.path.splitext(out)[0]}.json")


if __name__ == "__main__":
    main()
