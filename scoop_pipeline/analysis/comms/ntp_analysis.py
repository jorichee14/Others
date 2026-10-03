#!/usr/bin/env python3
"""How well the machines' clocks agreed during a pass, from comms/ntp/ntp.csv
(processing/comms_tables.py).

    python scoop_pipeline/analysis/comms/ntp_analysis.py <processed pass folder | comms dir | ntp.csv>
        [--reference mobile_1] [--no-plots]

Per machine (chrony's view of its own clock against its source):
  coverage      status messages, their time span, gaps (publisher stalls), lost seq
  sync          synchronized share, sources, stratum, reachability below 100 %
  polls         the offset MEASURED at each poll (last_offset, once per new
                reference_time): median / p95 / max |offset|
  error bound   chrony's maximum error of the clock against the root,
                |offset| + root_delay / 2 + root_dispersion, per message:
                p50 / p95 / max
  steps         clock steps (clock_stepped), when and by how much
  frequency     the crystal's error (ppm) over the run, its range, and how it
                follows the SoC temperature (slope ppm/degC, correlation)
Between machines: the clients are synchronized to the reference (mobile_1,
the NTP server), so two clients disagree by at most the sum of their bounds,
a client and the reference by its own bound. That table is the timing
uncertainty of aligning their sensors.

Writes comms/ntp/ntp_report.json, ntp_offsets.png, ntp_frequency.png.
"""
import argparse
import csv
import json
import math
import os
import sys

import numpy as np


def load(path):
    if os.path.isdir(path):
        for c in (os.path.join(path, "comms", "ntp", "ntp.csv"), os.path.join(path, "ntp", "ntp.csv"),
                  os.path.join(path, "ntp.csv")):
            if os.path.exists(c):
                path = c
                break
    if not os.path.isfile(path):
        sys.exit(f"no ntp.csv at {path} (run processing/comms_tables.py)")
    rows = list(csv.DictReader(open(path)))
    by = {}
    for r in rows:
        by.setdefault(r["machine"], []).append(r)
    return path, by


def col(rows, name, dtype=float):
    out = []
    for r in rows:
        v = r.get(name, "")
        if v in ("", None):
            out.append(math.nan)
            continue
        try:
            out.append(float(dtype(v)) if dtype is not bool else float(v in ("1", "True", "true")))
        except ValueError:
            out.append(math.nan)
    return np.array(out, float)


def stats(a, scale=1.0):
    a = a[np.isfinite(a)] * scale
    if not len(a):
        return None
    return {"median": float(np.median(a)), "p95": float(np.percentile(a, 95)),
            "max": float(a.max()), "min": float(a.min())}


def spans(t, mask):
    """[(start, end)] of the runs where mask holds."""
    out, start = [], None
    for ti, m in zip(t, mask):
        if m and start is None:
            start = ti
        if not m and start is not None:
            out.append((start, ti))
            start = None
    if start is not None:
        out.append((start, t[-1]))
    return out


def machine_report(rows):
    rows = sorted(rows, key=lambda r: float(r["t"]))
    t = col(rows, "t")
    rep = {"hostname": rows[0].get("hostname"), "role": rows[0].get("role"),
           "messages": len(rows), "t_start": float(t[0]), "t_end": float(t[-1]),
           "duration_s": float(t[-1] - t[0])}
    dt = np.diff(t)
    if len(dt):
        med = float(np.median(dt))
        big = dt > 3 * med
        rep["interval_s"] = med
        rep["gaps"] = [{"at_s": float(t[i]), "length_s": float(dt[i])} for i in np.flatnonzero(big)]
    seq = col(rows, "seq")
    if np.isfinite(seq).all() and len(seq) > 1:
        d = np.diff(seq)
        rep["lost_messages"] = int(np.sum(d[d > 1] - 1))
        rep["publisher_restarts"] = int(np.sum(d < 0))
    sync = col(rows, "synchronized", bool)
    rep["synchronized_percent"] = 100 * float(np.nanmean(sync))
    rep["sources"] = sorted({r.get("sync_source", "") for r in rows})
    rep["stratum"] = sorted({r.get("stratum", "") for r in rows})
    leap = sorted({r.get("leap_indicator", "") for r in rows} - {"no_warning", ""})
    if leap:
        rep["leap_indicator"] = leap
    reach = col(rows, "reachability_percent")
    low = np.isfinite(reach) & (reach < 100)
    rep["reach_below_100_percent_of_time"] = 100 * float(np.mean(low))
    rep["reach_below_100_spans"] = [{"from_s": float(a), "to_s": float(b)} for a, b in spans(t, low)]

    # polls: a sample the daemon accepted moves reference_time
    ref = col(rows, "reference_time")
    meas = col(rows, "last_offset_seconds")
    new = np.r_[True, np.diff(ref) != 0] if np.isfinite(ref).any() else \
        np.r_[True, np.diff(meas) != 0]
    rep["polls"] = int(new.sum())
    rep["poll_interval_s"] = stats(np.diff(t[new]))
    rep["measured_offset_ms"] = stats(np.abs(meas[new]), 1e3)
    rep["steered_offset_ms"] = stats(np.abs(col(rows, "offset_seconds")), 1e3)
    rep["rms_offset_ms"] = stats(col(rows, "rms_offset_seconds"), 1e3)
    rep["jitter_ms"] = stats(col(rows, "jitter_seconds"), 1e3)
    bound = np.abs(col(rows, "offset_seconds")) + col(rows, "root_delay") / 2 \
        + col(rows, "root_dispersion")
    rep["error_bound_ms"] = stats(bound, 1e3)

    st = col(rows, "clock_stepped", bool) == 1
    rep["clock_steps"] = [{"at_s": float(t[i]), "by_ms": float(col([rows[i]], "offset_delta_seconds")[0] * 1e3)}
                          for i in np.flatnonzero(st)]

    f = col(rows, "frequency_error_ppm")
    T = col(rows, "temperature_c")
    rep["frequency_error_ppm"] = stats(f)
    if rep["frequency_error_ppm"]:
        rep["frequency_range_ppm"] = rep["frequency_error_ppm"]["max"] - rep["frequency_error_ppm"]["min"]
    rep["temperature_c"] = stats(T)
    ok = np.isfinite(f) & np.isfinite(T)
    if ok.sum() > 10 and np.std(T[ok]) > 0.1 and np.std(f[ok]) > 0:
        slope = np.polyfit(T[ok], f[ok], 1)[0]
        rep["frequency_vs_temperature"] = {"ppm_per_degC": float(slope),
                                           "correlation": float(np.corrcoef(T[ok], f[ok])[0, 1])}
    rep["cpu_load_1min"] = stats(col(rows, "cpu_load_1min"))
    return rep, t, meas, new, bound, f, T


def pairwise(reports, reference):
    """Worst-case disagreement of two clocks: each client is bounded against
    the reference; two clients by the sum of their bounds."""
    names = sorted(reports)
    out = {}
    for q in ("p95", "max"):
        tab = {}
        for a in names:
            for b in names:
                if a >= b:
                    continue
                ba = 0.0 if a == reference else (reports[a]["error_bound_ms"] or {}).get(q, math.nan)
                bb = 0.0 if b == reference else (reports[b]["error_bound_ms"] or {}).get(q, math.nan)
                tab[f"{a} ~ {b}"] = ba + bb
        out[q] = tab
    return out


def plots(series, out_dir, t0):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (no matplotlib: plots skipped)")
        return []
    names = sorted(series)
    fig, axs = plt.subplots(len(names), 1, figsize=(10, 2.4 * len(names)), sharex=True,
                            squeeze=False)
    for ax, m in zip(axs[:, 0], names):
        t, meas, new, bound = series[m][:4]
        ax.fill_between(t - t0, -bound * 1e3, bound * 1e3, color="0.85", lw=0,
                        label="error bound (|offset| + root delay/2 + root dispersion)")
        ax.plot(t[new] - t0, meas[new] * 1e3, ".", ms=3, color="C0", label="measured offset at polls")
        ax.axhline(0, color="0.4", lw=0.5)
        ax.set_ylabel(f"{m}\nms")
    axs[0, 0].legend(loc="upper right", fontsize=7)
    axs[-1, 0].set_xlabel("s from the first status message")
    fig.tight_layout()
    p1 = os.path.join(out_dir, "ntp_offsets.png")
    fig.savefig(p1, dpi=130)
    plt.close(fig)
    fig, axs = plt.subplots(len(names), 1, figsize=(10, 2.4 * len(names)), sharex=True,
                            squeeze=False)
    for ax, m in zip(axs[:, 0], names):
        t, _, _, _, f, T = series[m]
        ax.plot(t - t0, f, color="C0", lw=1)
        ax.set_ylabel(f"{m}\nppm", color="C0")
        ax2 = ax.twinx()
        ax2.plot(t - t0, T, color="C3", lw=1)
        ax2.set_ylabel("degC", color="C3")
    axs[-1, 0].set_xlabel("s from the first status message")
    fig.tight_layout()
    p2 = os.path.join(out_dir, "ntp_frequency.png")
    fig.savefig(p2, dpi=130)
    plt.close(fig)
    return [p1, p2]


def fmt(st, k="median", unit="", nd=3):
    return "-" if not st else f"{st[k]:.{nd}g}{unit}"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path")
    ap.add_argument("--reference", default="mobile_1", help="the machine the others sync to")
    ap.add_argument("--no-plots", action="store_true")
    a = ap.parse_args()
    path, by = load(a.path)
    out_dir = os.path.dirname(path)
    reports, series = {}, {}
    for m, rows in sorted(by.items()):
        rep, t, meas, new, bound, f, T = machine_report(rows)
        reports[m], series[m] = rep, (t, meas, new, bound, f, T)
    t0 = min(r["t_start"] for r in reports.values())
    pw = pairwise(reports, a.reference)
    report = {"source": path, "reference": a.reference, "machines": reports, "pairwise_bound_ms": pw}
    with open(os.path.join(out_dir, "ntp_report.json"), "w") as fh:
        json.dump(report, fh, indent=2)

    print(f"NTP, {path}\n")
    print("%-18s %6s %7s %6s %-16s %6s %10s %10s %9s %9s %6s %10s" % (
        "machine", "msgs", "span s", "sync%", "source", "polls", "|meas| med", "|meas| max",
        "bound p95", "bound max", "steps", "reach<100%"))
    for m, r in reports.items():
        print("%-18s %6d %7.0f %6.1f %-16s %6d %10s %10s %9s %9s %6d %9.1f%%" % (
            m, r["messages"], r["duration_s"], r["synchronized_percent"], ",".join(r["sources"])[:16],
            r["polls"], fmt(r["measured_offset_ms"], unit=" ms"), fmt(r["measured_offset_ms"], "max", " ms"),
            fmt(r["error_bound_ms"], "p95", " ms"), fmt(r["error_bound_ms"], "max", " ms"),
            len(r["clock_steps"]), r["reach_below_100_percent_of_time"]))
    print("\nfrequency (crystal error) and temperature:")
    for m, r in reports.items():
        fv = r.get("frequency_vs_temperature")
        print("  %-18s %s ppm (range %s ppm), %s degC%s" % (
            m, fmt(r["frequency_error_ppm"]), "%.3g" % r.get("frequency_range_ppm", math.nan),
            fmt(r["temperature_c"], nd=3) if r["temperature_c"] else "-",
            f", {fv['ppm_per_degC']:+.3f} ppm/degC (r = {fv['correlation']:.2f})" if fv else ""))
    notes = []
    for m, r in reports.items():
        for g in r.get("gaps", []):
            notes.append(f"{m}: no status for {g['length_s']:.1f} s at {g['at_s'] - t0:.0f} s")
        for s in r["clock_steps"]:
            notes.append(f"{m}: clock stepped by {s['by_ms']:.3f} ms at {s['at_s'] - t0:.0f} s")
        for sp in r["reach_below_100_spans"]:
            notes.append(f"{m}: source reachability below 100% from {sp['from_s'] - t0:.0f} "
                         f"to {sp['to_s'] - t0:.0f} s")
        if r.get("lost_messages"):
            notes.append(f"{m}: {r['lost_messages']} status messages missing (seq)")
        if r["synchronized_percent"] < 100:
            notes.append(f"{m}: unsynchronized {100 - r['synchronized_percent']:.1f}% of the time")
        if len(r["sources"]) > 1:
            notes.append(f"{m}: source changed: {', '.join(r['sources'])}")
    if notes:
        print("\nevents:")
        for n in notes:
            print("  " + n)
    print(f"\nworst-case clock disagreement between machines (bounds added; {a.reference} is "
          f"the reference, its own bound against its upstream not counted), ms:")
    for pair in pw["max"]:
        print(f"  {pair:34s} p95 {pw['p95'][pair]:7.3f}   max {pw['max'][pair]:7.3f}")
    worst = max(pw["max"].values()) if pw["max"] else math.nan
    print(f"\nlargest: {worst:.3f} ms (a 30 Hz camera frame is 33 ms, a 10 Hz scan 100 ms)")
    if not a.no_plots:
        for p in plots(series, out_dir, t0):
            print(f"  {p}")
    print(f"  {os.path.join(out_dir, 'ntp_report.json')}")


if __name__ == "__main__":
    main()
