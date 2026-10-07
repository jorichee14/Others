#!/usr/bin/env python3
"""What a synthetic channel hides (Paper A, experiment "Sim Gap"): messages of
three sizes sent through the link as it was measured, against the synthetic
channels of the collaborative perception literature.

    python scoop_pipeline/analysis/comms/sim_gap.py <processed pass folder> [<pass> ...]
        [--rate 10] [--frame-ms 100] [--out DIR]

From comms/wifi/ping.csv and iperf.csv (processing/comms_tables.py), per robot
that ran iperf, a message every 1 / --rate s over the pass, of each size

    boxes 1 KB, features 100 KB, clouds 1.5 MB,

arrives after the logged channel's delay (Sec. VII)

    d(t) = RTT(t) / 2 + s / G(t)

  RTT(t)   the most recent ping (by ICMP seq) sent at or before t that got a
           reply, its reply counted even when late; none within 0.5 s: no
           measurement, the message is left out
  G(t)     the goodput of the robot's most recent successful iperf up test
           (robot to server; down when it ran none) started at or before t
  outage   a run of pings that never got a reply lasting >= --frame-ms: a
           message sent in it is lost (one lost ping is not an outage)

Reported per pass, robot and size: messages, lost (outage), the share that
arrives within one LiDAR frame (--frame-ms), delay p50 / p95 / max; the same
split into messages sent while an iperf test ran (the channel loaded) and the
rest. Next to them the synthetic channels, which have no outage and no size
dependence:

  V2X-ViT     a fixed 100 ms delay (its noisy setting), no loss
  Where2comm  delivery within a bandwidth budget: no delay, no loss

Writes comms/sim_gap/: sim_gap.json, sim_gap.csv (one row per message),
sim_gap.png (delay CDF per size, logged per robot against the synthetic
channels). With several passes also <--out>/sim_gap_all.png and .json (by
default next to the first pass).
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

SIZES = [("boxes", 1e3), ("features", 1e5), ("clouds", 1.5e6)]      # bytes
SYNTH = {"V2X-ViT": {"delay_ms": 100.0, "loss": 0.0,
                     "note": "fixed 100 ms delay (noisy setting), no loss, no size dependence"},
         "Where2comm": {"delay_ms": 0.0, "loss": 0.0,
                        "note": "delivery within a bandwidth budget: no delay, no loss"}}
RTT_AGE_S = 0.5


def load(path):
    if not os.path.exists(path):
        return []
    return list(csv.DictReader(open(path)))


def num(r, k):
    v = r.get(k, "")
    if v in ("", None):
        return math.nan
    if v in ("True", "true"):
        return 1.0
    if v in ("False", "false"):
        return 0.0
    try:
        return float(v)
    except ValueError:
        return math.nan


def unwrap_seq(seq):
    """16-bit ICMP seqs in message order -> ints (as wifi_analysis)."""
    out = np.array(seq, float)
    ref = None
    for i, s in enumerate(seq):
        if not np.isfinite(s):
            continue
        c = s if ref is None else s + 65536 * np.round((ref - s) / 65536)
        out[i] = c
        ref = c if ref is None else max(ref, c)
    return out


def ping_seqs(rows):
    """Per ICMP seq: (send t, rtt ms or nan when never answered), by send t."""
    rows = sorted(rows, key=lambda r: num(r, "t_log"))
    seq = unwrap_seq([num(r, "icmp_seq") for r in rows])
    first, rtt = {}, {}
    for r, s in zip(rows, seq):
        if not np.isfinite(s):
            continue
        t = num(r, "t")
        first.setdefault(s, t)
        if num(r, "reply") == 1 and np.isfinite(num(r, "rtt_ms")):
            rtt.setdefault(s, num(r, "rtt_ms"))
    ss = sorted(first, key=lambda s: first[s])
    return (np.array([first[s] for s in ss]), np.array([rtt.get(s, math.nan) for s in ss]))


def outages(ts, rtt, min_s):
    """[(t0, t1)]: runs of unanswered pings from the first one sent to the next
    answered one, min_s or longer."""
    out, i, n = [], 0, len(ts)
    while i < n:
        if np.isfinite(rtt[i]):
            i += 1
            continue
        j = i
        while j < n and not np.isfinite(rtt[j]):
            j += 1
        t1 = ts[j] if j < n else ts[-1]
        if t1 - ts[i] >= min_s:
            out.append((float(ts[i]), float(t1)))
        i = j
    return out


def iperf_tests(rows):
    """[(t0, t1, goodput Mbit/s, up)] of the successful tests, by start."""
    out = []
    for r in rows:
        t, g = num(r, "t"), num(r, "bitrate_mbps")
        if num(r, "success") == 1 and np.isfinite(t) and np.isfinite(g) and g > 0:
            d = num(r, "duration_s")
            out.append((t, t + (d if np.isfinite(d) else 0.0), g, num(r, "reverse") != 1))
    return sorted(out)


def robot_messages(ping, tests, busy, rate, frame_ms):
    """One row per message per size for one robot."""
    ts, rtt = ping_seqs(ping)
    if not len(ts):
        return [], [], math.nan
    outs = outages(ts, rtt, frame_ms / 1e3)
    use = [x for x in tests if x[3]] or tests                 # up (robot -> server), else down
    st = np.array([x[0] for x in use])
    ok = np.isfinite(rtt)
    ta, ra = ts[ok], rtt[ok]
    rows = []
    for t in np.arange(ts[0], ts[-1], 1.0 / rate):
        lost = any(a <= t < b for a, b in outs)
        k = int(np.searchsorted(ta, t, side="right")) - 1
        r = ra[k] if k >= 0 and t - ta[k] <= RTT_AGE_S else math.nan
        if not lost and not np.isfinite(r):
            continue                                          # no measurement: left out
        j = int(np.searchsorted(st, t, side="right")) - 1
        g = use[max(j, 0)][2]
        loaded = any(a <= t <= b for a, b in busy)
        for name, s in SIZES:
            d = math.nan if lost else r / 2 + s * 8 / (g * 1e6) * 1e3
            rows.append({"t": float(t), "size": name, "bytes": s, "lost": int(lost), "loaded": int(loaded),
                         "rtt_ms": r, "goodput_mbps": g, "delay_ms": d})
    return rows, outs, float(ts[0])


def stats(rows, frame_ms):
    n = len(rows)
    if not n:
        return None
    d = np.array([r["delay_ms"] for r in rows])
    lost = np.array([r["lost"] for r in rows]) == 1
    dd = d[~lost]
    return {"messages": n, "lost_percent": 100.0 * lost.mean(),
            "within_frame_percent": 100.0 * float(np.mean(~lost & (np.nan_to_num(d, nan=np.inf) <= frame_ms))),
            "delay_ms": {"p50": float(np.median(dd)) if len(dd) else None,
                         "p95": float(np.percentile(dd, 95)) if len(dd) else None,
                         "max": float(dd.max()) if len(dd) else None}}


def synth_stats(frame_ms):
    return {k: {"within_frame_percent": 100.0 * (1 - v["loss"]) * (v["delay_ms"] <= frame_ms),
                "delay_ms": v["delay_ms"], "lost_percent": 100.0 * v["loss"], "note": v["note"]}
            for k, v in SYNTH.items()}


def run_pass(proc, rate=10.0, frame_ms=100.0, log=print):
    proc = os.path.abspath(os.path.expanduser(proc))
    wifi = os.path.join(proc, "comms", "wifi")
    ping, iperf = load(os.path.join(wifi, "ping.csv")), load(os.path.join(wifi, "iperf.csv"))
    if not ping or not iperf:
        log(f"  {proc}: no ping.csv / iperf.csv (run processing/comms_tables.py)")
        return None
    by = lambda rows: {m: [r for r in rows if r["machine"] == m] for m in sorted({r["machine"] for r in rows})}  # noqa: E731
    ping, iperf = by(ping), by(iperf)
    busy = [(a, b) for rows in iperf.values() for a, b, _, _ in iperf_tests(rows)]
    rep = {"pass": proc, "rate_hz": rate, "frame_ms": frame_ms,
           "sizes_bytes": dict(SIZES), "synthetic": synth_stats(frame_ms), "robots": {}}
    table = []
    for m in sorted(ping):
        tests = iperf_tests(iperf.get(m, []))
        if not tests:
            log(f"  {m}: no successful iperf test, no goodput: left out")
            continue
        rows, outs, t_first = robot_messages(ping[m], tests, busy, rate, frame_ms)
        r = {"first_ping_t": t_first,
             "outages": [{"t0": a, "at_s": a - t_first, "duration_s": b - a} for a, b in outs],
             "outage_s_total": float(sum(b - a for a, b in outs)), "sizes": {}}
        for name, _ in SIZES:
            rr = [x for x in rows if x["size"] == name]
            r["sizes"][name] = {"all": stats(rr, frame_ms),
                                "loaded": stats([x for x in rr if x["loaded"]], frame_ms),
                                "idle": stats([x for x in rr if not x["loaded"]], frame_ms)}
        rep["robots"][m] = r
        table += [dict(x, machine=m) for x in rows]
    return rep, table


def fmt(v, nd=1):
    return "-" if v is None or (isinstance(v, float) and not np.isfinite(v)) else "%.*f" % (nd, v)


def print_report(rep, log=print):
    log(f"\n{rep['pass']}: a message every {1e3 / rep['rate_hz']:.0f} ms, frame {rep['frame_ms']:.0f} ms")
    log("  %-9s %-9s %-7s %6s %7s %10s %8s %8s %8s   %s" % (
        "robot", "size", "load", "msgs", "lost %", "in frame %", "p50 ms", "p95 ms", "max ms", ""))
    for m, r in rep["robots"].items():
        for name, _ in SIZES:
            for part in ("all", "idle", "loaded"):
                s = r["sizes"][name][part]
                if s is None:
                    continue
                d = s["delay_ms"]
                log("  %-9s %-9s %-7s %6d %7s %10s %8s %8s %8s" % (
                    m, name, part, s["messages"], fmt(s["lost_percent"]), fmt(s["within_frame_percent"]),
                    fmt(d["p50"]), fmt(d["p95"]), fmt(d["max"])))
        o = r["outages"]
        log("  %-9s outages: %d, %.1f s in all%s" % (
            m, len(o), r["outage_s_total"],
            (" (" + ", ".join("%.1f s at %.0f s" % (x["duration_s"], x["at_s"]) for x in o[:8])
             + (", ..." if len(o) > 8 else "") + "; from its first ping)") if o else ""))
    for k, v in rep["synthetic"].items():
        log("  %-9s every size: lost %s %%, delay %s ms, in frame %s %%  (%s)" % (
            k, fmt(v["lost_percent"]), fmt(v["delay_ms"]), fmt(v["within_frame_percent"]), v["note"]))


def plot(groups, frame_ms, path, title):
    """groups: [(label, rows)]; one panel per size, the delay CDF of each label
    (lost messages at the top: the CDF ends below 1), the synthetic channels as
    steps."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, len(SIZES), figsize=(4.6 * len(SIZES), 3.8), squeeze=False)
    for ax, (name, s) in zip(axs[0], SIZES):
        for label, rows in groups:
            rr = [x for x in rows if x["size"] == name]
            if not rr:
                continue
            d = np.sort([x["delay_ms"] for x in rr if not x["lost"]])
            y = np.arange(1, len(d) + 1) / len(rr)
            ax.step(np.maximum(d, 0.05), y, where="post", lw=1.2, label=label)
        for k, v in SYNTH.items():
            x = max(v["delay_ms"], 0.05)
            ax.step([0.05, x, x, 1e5], [0, 0, 1 - v["loss"], 1 - v["loss"]], where="post",
                    ls="--", lw=1.0, color="0.4" if k == "V2X-ViT" else "0.7", label=k)
        ax.axvline(frame_ms, color="r", lw=0.8)
        ax.set_xscale("log")
        ax.set_xlim(0.1, 2e3)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("arrival delay (ms); red: one frame")
        ax.set_title(f"{name}, {s / 1e3:g} KB" if s < 1e6 else f"{name}, {s / 1e6:g} MB", fontsize=9)
        ax.grid(color="#e6e6e6", lw=0.5, which="both")
    axs[0, 0].set_ylabel("share of messages arrived")
    axs[0, -1].legend(fontsize=7, loc="lower right")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write(rep, table, out):
    os.makedirs(out, exist_ok=True)
    json.dump(rep, open(os.path.join(out, "sim_gap.json"), "w"), indent=1)
    cols = ["machine", "t", "size", "bytes", "lost", "loaded", "rtt_ms", "goodput_mbps", "delay_ms"]
    with open(os.path.join(out, "sim_gap.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(table)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("processed", nargs="+")
    ap.add_argument("--rate", type=float, default=10.0, help="messages per second per robot")
    ap.add_argument("--frame-ms", type=float, default=100.0, help="one LiDAR frame")
    ap.add_argument("--out", default=None, help="with several passes: where the combined figure goes")
    ap.add_argument("--no-plots", action="store_true")
    a = ap.parse_args()
    allg, summary = [], {}
    for p in a.processed:
        r = run_pass(p, a.rate, a.frame_ms)
        if r is None:
            continue
        rep, table = r
        out = os.path.join(rep["pass"], "comms", "sim_gap")
        write(rep, table, out)
        print_report(rep)
        tag = "/".join(os.path.normpath(rep["pass"]).split(os.sep)[-2:])
        groups = [(f"{m}", [x for x in table if x["machine"] == m]) for m in rep["robots"]]
        if not a.no_plots:
            plot(groups, a.frame_ms, os.path.join(out, "sim_gap.png"), f"{tag}: logged channel vs synthetic")
            print(f"  {os.path.join(out, 'sim_gap.png')}")
        print(f"  {os.path.join(out, 'sim_gap.json')}")
        allg += [(f"{tag.split('/')[-1]} {lab}", rows) for lab, rows in groups]
        summary[tag] = rep
    if len(summary) > 1:
        out = a.out or os.path.join(os.path.dirname(os.path.abspath(os.path.expanduser(a.processed[0]))),
                                    "sim_gap")
        os.makedirs(out, exist_ok=True)
        json.dump(summary, open(os.path.join(out, "sim_gap_all.json"), "w"), indent=1)
        if not a.no_plots:
            plot(allg, a.frame_ms, os.path.join(out, "sim_gap_all.png"), "logged channel vs synthetic")
        print(f"\n  {out}/sim_gap_all.{{json,png}}")


if __name__ == "__main__":
    main()
