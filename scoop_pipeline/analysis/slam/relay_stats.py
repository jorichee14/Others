#!/usr/bin/env python3
"""What a channel did to the messages of a run: slam/channel_relay.py's
relay.csv of slam/run_swarm_slam.sh or slam/run_hydra.sh runs, side by side.

    python scoop_pipeline/analysis/slam/relay_stats.py <run dir> [<run dir> ...]
        [--to server] [--plot age.png]

Per run and stream (sender -> receiver):
  messages, MB      everything the sender put on the link
  lost              dropped by the channel (an outage)
  delay             t_arrive - t_send of the delivered ones: p50 / p95 / max
  backlog           most updates of the stream on the link at once (sent, not
                    yet arrived): above 1 means they queue
  map age           (receivers given by --to, e.g. Hydra's server) how old the
                    receiver's newest copy is, sampled every 0.1 s from the
                    first arrival to the last send: now - t_send of the newest
                    update arrived by now; p50 / p95 / max. With an ideal link
                    it is the update period; what a channel adds is the cost
--plot: the map age just before each update arrives (the oldest the
receiver's copy gets) over the run, one panel per sender, one line per run.

Writes relay_stats.json in every run.
"""
import argparse
import csv
import json
import os
import sys

import numpy as np


def load(run):
    path = os.path.join(run, "relay.csv")
    if not os.path.exists(path):
        sys.exit("no %s" % path)
    rows = []
    for r in csv.DictReader(open(path)):
        if not r["t_send"]:
            continue                                   # before the clock: not on the link
        rows.append({"t_send": float(r["t_send"]), "src": r["src"], "dst": r["dst"],
                     "bytes": int(r["bytes"]), "lost": r["lost"] == "1",
                     "t_arrive": float(r["t_arrive"]) if r["t_arrive"] else None})
    return rows


def pct(v, p):
    return float(np.percentile(v, p)) if len(v) else None


def stream_stats(rows, age=False, step=0.1):
    sent = sorted(rows, key=lambda r: r["t_send"])
    got = [r for r in sent if not r["lost"] and r["t_arrive"] is not None]
    d = np.array([r["t_arrive"] - r["t_send"] for r in got])
    backlog = 0
    for r in sent:
        if not r["lost"]:                              # this one, and the earlier ones still on the link
            backlog = max(backlog, 1 + sum(1 for o in got if o is not r and
                                           o["t_send"] <= r["t_send"] < o["t_arrive"]))
    out = {"messages": len(sent), "mb": sum(r["bytes"] for r in sent) / 1e6,
           "lost": sum(r["lost"] for r in sent),
           "delay_p50_s": pct(d, 50), "delay_p95_s": pct(d, 95), "delay_max_s": float(d.max()) if len(d) else None,
           "backlog_max": backlog}
    if age and got:
        arr = sorted((r["t_arrive"], r["t_send"]) for r in got)
        ta = np.array([a for a, _ in arr])
        newest = np.maximum.accumulate(np.array([s for _, s in arr]))
        tau = np.arange(ta[0], max(sent[-1]["t_send"], ta[0]) + 1e-9, step)
        k = np.searchsorted(ta, tau, side="right") - 1
        a = tau - newest[k]
        peak = ta[1:] - newest[:-1]                   # the age just before each arrival
        out.update({"age_p50_s": pct(a, 50), "age_p95_s": pct(a, 95), "age_max_s": float(a.max()),
                    "age_t": (ta[1:] - ta[0]).tolist(), "age_s": peak.tolist()})
    return out


def run_stats(run, to=()):
    rows = load(run)
    streams = {}
    for r in rows:
        streams.setdefault((r["src"], r["dst"]), []).append(r)
    rep = {"run": os.path.abspath(run), "streams": {}}
    for (s, d), rr in sorted(streams.items()):
        rep["streams"]["%s->%s" % (s, d)] = stream_stats(rr, age=d in to)
    json.dump({"run": rep["run"], "streams": {k: {kk: vv for kk, vv in v.items() if kk not in ("age_t", "age_s")}
                                              for k, v in rep["streams"].items()}},
              open(os.path.join(run, "relay_stats.json"), "w"), indent=1)
    return rep


def fmt(v, scale=1.0, nd=1):
    return "-" if v is None else "%.*f" % (nd, v * scale)


def table(reps):
    lines = ["%-20s %-22s %8s %8s %6s %10s %10s %8s %10s %10s" % (
        "run", "stream", "msgs", "MB", "lost", "delay p50", "p95 (ms)", "backlog", "age p95", "max (s)")]
    for rep in reps:
        name = os.path.basename(rep["run"].rstrip("/"))
        for k, s in rep["streams"].items():
            lines.append("%-20s %-22s %8d %8.1f %6d %10s %10s %8d %10s %10s" % (
                name, k, s["messages"], s["mb"], s["lost"], fmt(s["delay_p50_s"], 1e3),
                fmt(s["delay_p95_s"], 1e3), s["backlog_max"], fmt(s.get("age_p95_s"), 1, 2),
                fmt(s.get("age_max_s"), 1, 2)))
    return "\n".join(lines)


def plot(reps, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    senders = sorted({k for rep in reps for k, s in rep["streams"].items() if "age_s" in s})
    if not senders:
        sys.exit("no map age to plot: give --to <receiver>")
    colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
    ink2, grid = "#52514e", "#e6e5e1"
    fig, axes = plt.subplots(len(senders), 1, figsize=(9, 3.2 * len(senders)), sharex=True, squeeze=False)
    for ax, k in zip(axes[:, 0], senders):
        for c, rep in zip(colors, reps):
            s = rep["streams"].get(k)
            if s and "age_s" in s:
                ax.plot(s["age_t"], s["age_s"], color=c, lw=2, label=os.path.basename(rep["run"].rstrip("/")))
        ax.set_title(k, loc="left", fontsize=10, color=ink2)
        ax.set_ylabel("Map age before an update (s)", color=ink2)
        ax.grid(True, color=grid, lw=0.8)
        ax.set_axisbelow(True)
        ax.set_ylim(bottom=0)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0, 0].legend(fontsize=8, frameon=False)
    axes[-1, 0].set_xlabel("Bag time since the first arrival (s)", color=ink2)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor="#fcfcfb")
    plt.close(fig)
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="run folders (with relay.csv)")
    ap.add_argument("--to", nargs="*", default=[], help="receivers whose map age to report, e.g. server")
    ap.add_argument("--plot", default=None, help="map age over the run, to this PNG")
    a = ap.parse_args()
    reps = [run_stats(r, set(a.to)) for r in a.runs]
    print(table(reps))
    if a.plot:
        print("  wrote %s" % plot(reps, a.plot))


if __name__ == "__main__":
    main()
