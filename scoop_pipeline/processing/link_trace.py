#!/usr/bin/env python3
"""A pass's measured link as traces for scoop/channel.py (the "logged" and
"fit" channels), from the comms tables (processing/comms_tables.py).

    python scoop_pipeline/processing/link_trace.py <processed pass folder> [--outage-ms 100]

Writes comms/link/, per machine that pinged:
  ping_<machine>.csv    t (send, s), rtt_ms (empty: never answered; a late
                        reply counts), own_load (1: sent while this machine's
                        own iperf test ran), any_load (1: any machine's)
  iperf_<machine>.csv   t0, t1 (s), mbps, up (1: machine -> server, 0: server
                        -> machine), successful tests only
  outage_<machine>.csv  t0, t1: runs of unanswered pings at least --outage-ms
                        long, from the first one sent to one ping interval
                        after the last; cut where no ping was logged for over
                        0.5 s (a logging stall is no evidence of the link)
  link.json             what was made, from what, with which settings

Pings are taken per ICMP sequence number, not per message (the monitor
publishes a timeout and later the late reply of the same seq).
"""
import argparse
import csv
import json
import math
import os

import numpy as np

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
    """16-bit ICMP seqs in message order -> ints."""
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
        first.setdefault(s, num(r, "t"))
        if num(r, "reply") == 1 and np.isfinite(num(r, "rtt_ms")):
            rtt.setdefault(s, num(r, "rtt_ms"))
    ss = sorted(first, key=lambda s: first[s])
    return np.array([first[s] for s in ss]), np.array([rtt.get(s, math.nan) for s in ss])


def outages(ts, rtt, min_s):
    """[(t0, t1)] runs of unanswered pings, min_s or longer (see the module doc)."""
    if len(ts) < 2:
        return []
    dt = float(np.median(np.diff(ts)))
    out, i, n = [], 0, len(ts)
    while i < n:
        if np.isfinite(rtt[i]):
            i += 1
            continue
        j = i
        while j + 1 < n and not np.isfinite(rtt[j + 1]) and ts[j + 1] - ts[j] <= RTT_AGE_S:
            j += 1
        if ts[j] + dt - ts[i] >= min_s:
            out.append((float(ts[i]), float(ts[j] + dt)))
        i = j + 1
    return out


def iperf_tests(rows):
    """[(t0, t1, mbps, up)] of the successful tests, by start."""
    out = []
    for r in rows:
        t, g = num(r, "t"), num(r, "bitrate_mbps")
        if num(r, "success") == 1 and np.isfinite(t) and np.isfinite(g) and g > 0:
            d = num(r, "duration_s")
            out.append((t, t + (d if np.isfinite(d) else 0.0), g, int(num(r, "reverse") != 1)))
    return sorted(out)


def inside(t, wins):
    """Bool per t: inside any [a, b]."""
    t = np.asarray(t, float)
    m = np.zeros(len(t), bool)
    for a, b in wins:
        m |= (t >= a) & (t <= b)
    return m


def write_csv(path, cols, rows):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for r in rows:
            w.writerow(["" if (isinstance(v, float) and not np.isfinite(v)) else
                        ("%.9f" % v if isinstance(v, float) else v) for v in r])


def make(proc, outage_ms=100.0, log=print):
    proc = os.path.abspath(os.path.expanduser(proc))
    wifi = os.path.join(proc, "comms", "wifi")
    ping, iperf = load(os.path.join(wifi, "ping.csv")), load(os.path.join(wifi, "iperf.csv"))
    if not ping:
        raise SystemExit(f"no {wifi}/ping.csv (run processing/comms_tables.py)")
    by = lambda rows: {m: [r for r in rows if r["machine"] == m] for m in sorted({r["machine"] for r in rows})}  # noqa: E731
    ping, iperf = by(ping), by(iperf)
    tests = {m: iperf_tests(rows) for m, rows in iperf.items()}
    any_win = [(a, b) for v in tests.values() for a, b, _, _ in v]
    out = os.path.join(proc, "comms", "link")
    os.makedirs(out, exist_ok=True)
    meta = {"pass": proc, "from": [os.path.join(wifi, "ping.csv"), os.path.join(wifi, "iperf.csv")],
            "outage_ms": outage_ms, "rtt_age_s": RTT_AGE_S, "machines": {}}
    for m in sorted(ping):
        ts, rtt = ping_seqs(ping[m])
        own = [(a, b) for a, b, _, _ in tests.get(m, [])]
        outs = outages(ts, rtt, outage_ms / 1e3)
        write_csv(os.path.join(out, "ping_%s.csv" % m.replace("/", "_")),
                  ["t", "rtt_ms", "own_load", "any_load"],
                  zip(ts, rtt, inside(ts, own).astype(int), inside(ts, any_win).astype(int)))
        write_csv(os.path.join(out, "iperf_%s.csv" % m.replace("/", "_")), ["t0", "t1", "mbps", "up"],
                  tests.get(m, []))
        write_csv(os.path.join(out, "outage_%s.csv" % m.replace("/", "_")), ["t0", "t1"], outs)
        dt = float(np.median(np.diff(ts))) if len(ts) > 1 else math.nan
        meta["machines"][m] = {"pings": int(len(ts)), "answered": int(np.isfinite(rtt).sum()),
                               "ping_interval_s": dt, "t0": float(ts[0]) if len(ts) else None,
                               "t1": float(ts[-1]) if len(ts) else None, "iperf_tests": len(tests.get(m, [])),
                               "outages": len(outs), "outage_s": float(sum(b - a for a, b in outs))}
        log("  %-18s %6d pings (%.0f Hz), %3d iperf tests, %2d outages (%.1f s)"
            % (m, len(ts), 1 / dt if dt == dt and dt > 0 else 0, len(tests.get(m, [])), len(outs),
               sum(b - a for a, b in outs)))
    json.dump(meta, open(os.path.join(out, "link.json"), "w"), indent=1)
    log("  %s" % out)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("processed")
    ap.add_argument("--outage-ms", type=float, default=100.0, help="shortest outage (one LiDAR frame)")
    a = ap.parse_args()
    make(a.processed, a.outage_ms)


if __name__ == "__main__":
    main()
