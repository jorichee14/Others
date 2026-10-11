#!/usr/bin/env python3
"""Does a setting reproduce its link? (Paper A, experiment E4 "Same Knob, Same
Link"): the distance between two passes' link distributions.

    python scoop_pipeline/analysis/comms/link_repeat.py <processed pass A> <processed pass B>
        [--label-a survey] [--label-b coop] [--grid 0.5] [--out DIR]

From each pass's comms tables (processing/comms_tables.py), per machine
present in both passes:

  rtt_ms        answered pings (reply == 1), per machine
  goodput_mbps  successful iperf tests, up (reverse == 0) and down
                (reverse == 1) separately, per machine
  rssi_dbm      Wi-Fi status signal_dbm, per machine
  rssi on grid  signal_dbm median per --grid m cell in map (the same cells
                comms_map.py draws); compared only over the cells both passes
                occupied, so a different route is not counted as a different
                link

and for each the 1-D Wasserstein distance W1 (the mean absolute difference
between the two quantile functions, in the unit of the quantity) and the
Kolmogorov-Smirnov statistic (max CDF difference, 0..1), with both medians
and sample counts. The two numbers answer different questions: W1 says by how
much, KS says whether the shapes differ at all.

The pass criterion of E4 compares two such distances: the distance between
two DAYS of the same mode must be below half the distance between the two
MODES (survey against cooperation) on the same route. This script computes
one distance per call; run it once for the mode pair and once for each day
pair and compare the JSONs (or pass --against JSON to print the ratio).

Writes <out>/link_repeat_<label-a>_vs_<label-b>.json (default: next to pass
B under comms/repeat/).
"""
import argparse
import csv
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


# ---- tables -----------------------------------------------------------------
def rows_of(proc, sub, name):
    p = os.path.join(proc, "comms", sub, name)
    if not os.path.exists(p):
        return []
    return list(csv.DictReader(open(p)))


def num(r, k):
    v = r.get(k, "")
    try:
        return float(v)
    except (TypeError, ValueError):
        return math.nan


def arr(rows, k):
    return np.array([num(r, k) for r in rows], float)


def by_machine(rows):
    out = {}
    for r in rows:
        out.setdefault(r.get("machine", "?"), []).append(r)
    return out


def finite(a):
    a = np.asarray(a, float)
    return a[np.isfinite(a)]


# ---- distances --------------------------------------------------------------
def wasserstein1(a, b, n=1001):
    """W1 between two empirical 1-D distributions: mean |Q_a(u) - Q_b(u)|."""
    a, b = np.sort(finite(a)), np.sort(finite(b))
    if len(a) == 0 or len(b) == 0:
        return math.nan
    u = (np.arange(n) + 0.5) / n
    qa = np.quantile(a, u)
    qb = np.quantile(b, u)
    return float(np.mean(np.abs(qa - qb)))


def ks_stat(a, b):
    """Two-sample Kolmogorov-Smirnov statistic (no p-value)."""
    a, b = np.sort(finite(a)), np.sort(finite(b))
    if len(a) == 0 or len(b) == 0:
        return math.nan
    x = np.concatenate([a, b])
    fa = np.searchsorted(a, x, side="right") / len(a)
    fb = np.searchsorted(b, x, side="right") / len(b)
    return float(np.max(np.abs(fa - fb)))


def compare(a, b):
    a, b = finite(a), finite(b)
    return {"n_a": int(len(a)), "n_b": int(len(b)),
            "median_a": float(np.median(a)) if len(a) else None,
            "median_b": float(np.median(b)) if len(b) else None,
            "w1": wasserstein1(a, b), "ks": ks_stat(a, b)}


# ---- per-quantity samples ---------------------------------------------------
def rtt_samples(proc):
    out = {}
    for m, rows in by_machine(rows_of(proc, "wifi", "ping.csv")).items():
        reply = arr(rows, "reply") == 1
        out[m] = arr(rows, "rtt_ms")[reply]
    return out


def goodput_samples(proc):
    out = {}
    for m, rows in by_machine(rows_of(proc, "wifi", "iperf.csv")).items():
        ok = arr(rows, "success") == 1
        rev = arr(rows, "reverse") == 1
        g = arr(rows, "bitrate_mbps")
        out[m + " up"] = g[ok & ~rev]
        out[m + " down"] = g[ok & rev]
    return out


def rssi_samples(proc):
    return {m: arr(rows, "signal_dbm")
            for m, rows in by_machine(rows_of(proc, "wifi", "wifi.csv")).items()}


def rssi_grid(proc, cell):
    """{machine: {(ix, iy): median signal_dbm}} over cells with >= 3 samples."""
    out = {}
    for m, rows in by_machine(rows_of(proc, "wifi", "wifi.csv")).items():
        x, y, v = arr(rows, "x"), arr(rows, "y"), arr(rows, "signal_dbm")
        ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(v)
        if ok.sum() == 0:
            continue
        cells = {}
        for i, j, s in zip(np.floor(x[ok] / cell).astype(int),
                           np.floor(y[ok] / cell).astype(int), v[ok]):
            cells.setdefault((int(i), int(j)), []).append(s)
        out[m] = {k: float(np.median(vs)) for k, vs in cells.items() if len(vs) >= 3}
    return out


def grid_compare(ga, gb):
    """Over the cells both passes occupied: mean |dA - dB|, RMS, W1 of the
    per-cell medians, and the shared-cell count."""
    shared = sorted(set(ga) & set(gb))
    if not shared:
        return {"shared_cells": 0}
    da = np.array([ga[k] for k in shared])
    db = np.array([gb[k] for k in shared])
    d = da - db
    return {"shared_cells": len(shared), "cells_a": len(ga), "cells_b": len(gb),
            "mean_abs_diff_db": float(np.mean(np.abs(d))),
            "rms_diff_db": float(np.sqrt(np.mean(d ** 2))),
            "median_diff_db": float(np.median(d)),
            "w1_db": wasserstein1(da, db), "ks": ks_stat(da, db)}


# ---- run --------------------------------------------------------------------
def run(pa, pb, cell=0.5, label_a=None, label_b=None, log=print):
    rep = {"pass_a": pa, "pass_b": pb, "label_a": label_a or os.path.basename(os.path.normpath(pa)),
           "label_b": label_b or os.path.basename(os.path.normpath(pb)), "grid_m": cell,
           "rtt_ms": {}, "goodput_mbps": {}, "rssi_dbm": {}, "rssi_grid": {}}
    for key, fn in (("rtt_ms", rtt_samples), ("goodput_mbps", goodput_samples),
                    ("rssi_dbm", rssi_samples)):
        A, B = fn(pa), fn(pb)
        for m in sorted(set(A) & set(B)):
            rep[key][m] = compare(A[m], B[m])
        for m in sorted(set(A) ^ set(B)):
            rep[key][m] = {"note": "only in %s" % ("A" if m in A else "B")}
    GA, GB = rssi_grid(pa, cell), rssi_grid(pb, cell)
    for m in sorted(set(GA) & set(GB)):
        rep["rssi_grid"][m] = grid_compare(GA[m], GB[m])
    return rep


def fmt(v, nd=2):
    return "--" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{nd}f}"


def print_report(rep, log=print):
    log(f"\n{rep['label_a']} vs {rep['label_b']} (grid {rep['grid_m']:g} m)")
    log("  %-22s %-14s %7s %7s %9s %9s %8s %6s" % ("quantity", "machine", "n_a", "n_b",
                                                     "median_a", "median_b", "W1", "KS"))
    for key in ("rtt_ms", "goodput_mbps", "rssi_dbm"):
        for m, c in rep[key].items():
            if "note" in c:
                log("  %-22s %-14s %s" % (key, m, c["note"]))
                continue
            log("  %-22s %-14s %7d %7d %9s %9s %8s %6s" % (
                key, m, c["n_a"], c["n_b"], fmt(c["median_a"]), fmt(c["median_b"]),
                fmt(c["w1"]), fmt(c["ks"], 3)))
    for m, c in rep["rssi_grid"].items():
        if c.get("shared_cells", 0) == 0:
            log("  rssi on grid           %-14s no shared cells" % m)
            continue
        log("  rssi on grid           %-14s %d shared cells (of %d / %d): mean |d| %s dB, "
            "RMS %s dB, W1 %s dB, KS %s" % (m, c["shared_cells"], c["cells_a"], c["cells_b"],
                                            fmt(c["mean_abs_diff_db"]), fmt(c["rms_diff_db"]),
                                            fmt(c["w1_db"]), fmt(c["ks"], 3)))


def ratio_against(rep, other, log=print):
    """E4 pass check: this call's W1 (day pair) against another JSON's W1
    (mode pair), per quantity and machine; pass when the ratio < 0.5."""
    log(f"\n  ratio of W1: ({rep['label_a']} vs {rep['label_b']}) / "
        f"({other['label_a']} vs {other['label_b']})   pass < 0.5")
    for key in ("rtt_ms", "goodput_mbps", "rssi_dbm"):
        for m, c in rep[key].items():
            o = other[key].get(m)
            if not o or "w1" not in c or "w1" not in o or not o["w1"]:
                continue
            r = c["w1"] / o["w1"]
            log("  %-22s %-14s %6.2f  %s" % (key, m, r, "pass" if r < 0.5 else "FAIL"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pass_a")
    ap.add_argument("pass_b")
    ap.add_argument("--label-a", default=None)
    ap.add_argument("--label-b", default=None)
    ap.add_argument("--grid", type=float, default=0.5, help="RSSI grid cell (m)")
    ap.add_argument("--against", default=None,
                    help="another link_repeat JSON (the mode pair) to print the W1 ratio against")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rep = run(a.pass_a, a.pass_b, a.grid, a.label_a, a.label_b)
    print_report(rep)
    if a.against:
        ratio_against(rep, json.load(open(a.against)))
    out = a.out or os.path.join(a.pass_b, "comms", "repeat")
    os.makedirs(out, exist_ok=True)
    p = os.path.join(out, "link_repeat_%s_vs_%s.json" % (rep["label_a"], rep["label_b"]))
    json.dump(rep, open(p, "w"), indent=1)
    print(f"\n  wrote {p}")


if __name__ == "__main__":
    main()
