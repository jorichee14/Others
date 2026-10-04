#!/usr/bin/env python3
"""The Wi-Fi links of a pass, from comms/wifi/{wifi,ping,iperf}.csv
(processing/comms_tables.py).

    python scoop_pipeline/analysis/comms/wifi_analysis.py <processed pass folder | comms dir>
        [--no-plots]

Per machine:
  link      RSSI, link quality, tx / rx PHY rate, MCS, streams, width, PHY mode,
            BSSIDs / channels (roams), driver drops, the traffic the interface
            carried (from its byte counters)
  ping      counted per ICMP sequence number, not per message: the monitor
            publishes a timeout and, later, the late reply of the same seq, so
            per message a late reply would count as lost. sent = seqs seen
            (16-bit wrap undone), lost = seqs that never got a reply; late =
            replies to a seq that had timed out. RTT over the replies.
  iperf     tests up / down, failures and their errors, goodput, the slowest
            second inside a test, retransmits, TCP RTT; goodput against the
            RSSI at the time
  contention  ping RTT and loss while an iperf test runs (any machine's: the
            channel is shared) against the rest of the time
With positions (comms tables made with the pass's config): RSSI along the
robot's path, iperf goodput where it was measured.

Writes comms/wifi/wifi_report.json, wifi_timeline.png, wifi_map.png.
"""
import argparse
import csv
import json
import math
import os
import sys

import numpy as np


def find(path, name):
    for c in (os.path.join(path, "comms", "wifi", name), os.path.join(path, "wifi", name),
              os.path.join(path, name)):
        if os.path.exists(c):
            return c
    return None


def load(path, name):
    p = find(path, name)
    if p is None:
        return None, {}
    by = {}
    for r in csv.DictReader(open(p)):
        by.setdefault(r["machine"], []).append(r)
    for rows in by.values():
        rows.sort(key=lambda r: num(r, "t"))
    return p, by


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


def col(rows, k):
    return np.array([num(r, k) for r in rows], float)


def stats(a, scale=1.0):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)] * scale
    if not len(a):
        return None
    return {"median": float(np.median(a)), "p5": float(np.percentile(a, 5)),
            "p95": float(np.percentile(a, 95)), "min": float(a.min()), "max": float(a.max()),
            "mean": float(a.mean())}


def counts(xs):
    out = {}
    for x in xs:
        if x not in ("", None, "-1", "nan"):
            out[x] = out.get(x, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def increase(rows, k):
    a = col(rows, k)
    a = a[np.isfinite(a) & (a >= 0)]
    if len(a) < 2:
        return None
    d = np.diff(a)
    return float(d[d > 0].sum())             # counter resets (negative steps) skipped


def link_report(rows):
    t = col(rows, "t")
    rep = {"messages": len(rows), "duration_s": float(t[-1] - t[0]),
           "interface": rows[0].get("interface"), "mac": rows[0].get("mac_address"),
           "associated_percent": 100 * float(np.nanmean(col(rows, "associated"))),
           "rssi_dbm": stats(col(rows, "signal_dbm")),
           "link_quality_ratio": stats(col(rows, "link_quality_ratio")),
           "tx_phy_mbps": stats(col(rows, "tx_bitrate_mbps")),
           "rx_phy_mbps": stats(col(rows, "rx_bitrate_mbps")),
           "tx_mcs": counts(r.get("tx_mcs") for r in rows),
           "rx_mcs": counts(r.get("rx_mcs") for r in rows),
           "tx_nss": counts(r.get("tx_nss") for r in rows),
           "rx_nss": counts(r.get("rx_nss") for r in rows),
           "width_mhz": counts(r.get("rx_width_mhz") for r in rows),
           "phy_mode": counts(r.get("rx_phy_mode") for r in rows),
           "channels": counts(r.get("channel") for r in rows),
           "bssids": counts(r.get("bssid") for r in rows)}
    bss = [r.get("bssid") for r in rows]
    rep["roams"] = int(sum(a != b for a, b in zip(bss, bss[1:])))
    for k in ("rx_dropped", "tx_dropped", "rx_errors", "tx_errors", "missed_beacon",
              "tx_excessive_retries", "tx_retries", "tx_failed"):
        v = increase(rows, k)
        if v is not None:
            rep[k + "_during_pass"] = v
    for d in ("tx", "rx"):
        b = col(rows, f"{d}_bytes")
        ok = np.isfinite(b) & np.isfinite(t)
        if ok.sum() > 2:
            rate = np.diff(b[ok]) * 8 / 1e6 / np.maximum(np.diff(t[ok]), 1e-3)
            rate = rate[rate >= 0]
            rep[f"{d}_traffic_mbps"] = stats(rate)
    return rep


def unwrap_seq(seq):
    """16-bit ICMP sequence numbers, in message order -> ints: each one in the
    65536-cycle nearest the highest seen so far, so a late reply to a seq from
    before a wrap stays in its own cycle."""
    out = np.array(seq, float)
    ref = None
    for i, s in enumerate(seq):
        if not np.isfinite(s):
            continue
        c = s if ref is None else s + 65536 * np.round((ref - s) / 65536)
        out[i] = c
        ref = c if ref is None else max(ref, c)
    return out


def ping_report(rows, busy):
    t = col(rows, "t")
    seq = unwrap_seq(col(rows, "icmp_seq"))
    reply = col(rows, "reply") == 1
    rtt = col(rows, "rtt_ms")
    ok = np.isfinite(seq)
    seqs = np.unique(seq[ok])
    replied = np.unique(seq[ok & reply])
    timed_out = np.unique(seq[ok & ~reply])
    late = np.intersect1d(replied, timed_out)
    rep = {"messages": len(rows), "targets": counts(r.get("target") for r in rows),
           "sent": int(len(seqs)), "replied": int(len(replied)),
           "lost": int(len(seqs) - len(replied)),
           "loss_percent": 100 * float(1 - len(replied) / max(len(seqs), 1)),
           "timeouts_reported": int(len(timed_out)), "late_replies": int(len(late)),
           "loss_percent_per_message": 100 * float(1 - reply.mean()) if len(rows) else None,
           "rate_hz": float(len(seqs) / max(t[-1] - t[0], 1e-6)),
           "rtt_ms": stats(rtt[reply])}
    if busy is not None:
        inb = in_windows(t, busy)
        for name, m in (("during_iperf", inb), ("no_iperf", ~inb)):
            s_all = np.unique(seq[ok & m])
            s_rep = np.unique(seq[ok & m & reply])
            rep[name] = {"share_of_time_percent": 100 * float(m.mean()),
                         "loss_percent": 100 * float(1 - len(s_rep) / max(len(s_all), 1)),
                         "rtt_ms": stats(rtt[m & reply])}
    return rep, t, seq, reply, rtt


def in_windows(t, wins):
    m = np.zeros(len(t), bool)
    for a, b in wins:
        m |= (t >= a) & (t <= b)
    return m


def iperf_report(rows, wifi_rows):
    out = {}
    wt = col(wifi_rows, "t") if wifi_rows else np.array([])
    wr = col(wifi_rows, "signal_dbm") if wifi_rows else np.array([])
    for d, rev in (("down", 1.0), ("up", 0.0)):
        sel = [r for r in rows if num(r, "reverse") == rev]
        if not sel:
            continue
        ok = [r for r in sel if num(r, "success") == 1]
        g = col(ok, "bitrate_mbps")
        rep = {"tests": len(sel), "failed": len(sel) - len(ok),
               "errors": counts(r.get("error") for r in sel if num(r, "success") != 1),
               "protocol": counts(r.get("protocol") for r in sel),
               "goodput_mbps": stats(g),
               "slowest_second_mbps": stats(col(ok, "bitrate_mbps_min")),
               "within_test_std_mbps": stats(col(ok, "bitrate_mbps_std")),
               "retransmits": float(np.nansum(col(ok, "retransmits"))),
               "tcp_rtt_ms": stats(col(ok, "rtt_ms_mean"))}
        if len(wt) > 2 and len(ok) > 3:
            tt = col(ok, "t") + col(ok, "duration_s") / 2
            rssi = np.interp(tt, wt, wr)
            fin = np.isfinite(rssi) & np.isfinite(g)
            if fin.sum() > 3 and np.std(rssi[fin]) > 0:
                rep["goodput_vs_rssi"] = {
                    "mbps_per_db": float(np.polyfit(rssi[fin], g[fin], 1)[0]),
                    "correlation": float(np.corrcoef(rssi[fin], g[fin])[0, 1])}
        out[d] = rep
    return out


def iperf_windows(iperf):
    wins = []
    for rows in iperf.values():
        for r in rows:
            t, d = num(r, "t"), num(r, "duration_s")
            if np.isfinite(t) and np.isfinite(d):
                wins.append((t, t + d))
    return wins


def per_second(t, v, fn):
    if not len(t):
        return np.array([]), np.array([])
    b = np.floor(t)
    keys = np.unique(b)
    return keys, np.array([fn(v[b == k]) for k in keys])


def plots(out_dir, t0, links, pings, iperf, busy):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (no matplotlib: plots skipped)")
        return []
    names = sorted(set(links) | set(pings))
    fig, axs = plt.subplots(len(names), 1, figsize=(12, 3.2 * len(names)), sharex=True,
                            squeeze=False)
    for ax, m in zip(axs[:, 0], names):
        for a, b in busy:
            ax.axvspan(a - t0, b - t0, color="0.92", lw=0)
        if m in links:
            rows = links[m]
            ax.plot(col(rows, "t") - t0, col(rows, "signal_dbm"), color="C0", lw=1, label="RSSI dBm")
        ax.set_ylabel(f"{m}\nRSSI dBm", color="C0")
        ax2 = ax.twinx()
        if m in pings:
            t, seq, reply, rtt = pings[m]
            k, med = per_second(t[reply], rtt[reply], np.median)
            ax2.semilogy(k - t0, med, ".", ms=2, color="C1", label="ping RTT median / s")
            ks, lost = per_second(t, (~reply).astype(float), np.mean)
            ax2.plot(ks - t0, np.maximum(lost * 1000, 1e-3), ".", ms=2, color="C3", alpha=0.4,
                     label="timeouts per s (x1000 = all)")
        ax2.set_ylabel("ms", color="C1")
        if m in iperf:
            for r in iperf[m]:
                if num(r, "success") == 1:
                    ax.annotate("%.0f" % num(r, "bitrate_mbps"), (num(r, "t") - t0, ax.get_ylim()[1]),
                                fontsize=5, color="C2" if num(r, "reverse") == 1 else "C4",
                                rotation=90, va="top")
    axs[0, 0].set_title("grey: an iperf test running; numbers: iperf goodput Mbit/s "
                        "(green down, purple up)", fontsize=8)
    axs[-1, 0].set_xlabel("s from the first status message")
    fig.tight_layout()
    p1 = os.path.join(out_dir, "wifi_timeline.png")
    fig.savefig(p1, dpi=130)
    plt.close(fig)
    out = [p1]
    have = {m: rows for m, rows in links.items() if "x" in rows[0] and rows[0]["x"] != ""}
    if have:
        fig, ax = plt.subplots(figsize=(9, 8))
        sc = None
        for m, rows in have.items():
            x, y, rssi = col(rows, "x"), col(rows, "y"), col(rows, "signal_dbm")
            moving = np.nanstd(x) > 0.2 or np.nanstd(y) > 0.2
            if moving:
                sc = ax.scatter(x, y, c=rssi, s=6, cmap="viridis", label=f"{m} RSSI")
            else:
                ax.plot(np.nanmedian(x), np.nanmedian(y), "k^", ms=9)
                ax.annotate(f"{m}\n{np.nanmedian(rssi):.0f} dBm", (np.nanmedian(x), np.nanmedian(y)),
                            fontsize=8, xytext=(5, 5), textcoords="offset points")
        for m, rows in iperf.items():
            ok = [r for r in rows if num(r, "success") == 1 and r.get("x")]
            if ok:
                dn = [r for r in ok if num(r, "reverse") == 1]
                ax.scatter(col(dn, "x"), col(dn, "y"), s=col(dn, "bitrate_mbps") / 8, facecolors="none",
                           edgecolors="C3", label=f"{m} iperf down (size: Mbit/s)")
        if sc is not None:
            fig.colorbar(sc, ax=ax, label="RSSI dBm")
        ax.set_aspect("equal")
        ax.set_xlabel("map x m")
        ax.set_ylabel("map y m")
        ax.legend(fontsize=7, loc="best")
        fig.tight_layout()
        p2 = os.path.join(out_dir, "wifi_map.png")
        fig.savefig(p2, dpi=130)
        plt.close(fig)
        out.append(p2)
    return out


def f(st, k="median", nd=3):
    return "-" if not st else f"{st[k]:.{nd}g}"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path")
    ap.add_argument("--no-plots", action="store_true")
    a = ap.parse_args()
    pw, links = load(a.path, "wifi.csv")
    pp, pings = load(a.path, "ping.csv")
    pi, iperf = load(a.path, "iperf.csv")
    if not (pw or pp or pi):
        sys.exit(f"no comms/wifi tables below {a.path} (run processing/comms_tables.py)")
    out_dir = os.path.dirname(pw or pp or pi)
    busy = iperf_windows(iperf)
    t0 = min(num(rows[0], "t") for d in (links, pings, iperf) for rows in d.values())
    report = {"machines": {}, "iperf_windows": len(busy),
              "iperf_busy_percent": None}
    series = {}
    for m in sorted(set(links) | set(pings) | set(iperf)):
        r = {}
        if m in links:
            r["link"] = link_report(links[m])
        if m in pings:
            r["ping"], t, seq, reply, rtt = ping_report(pings[m], busy if busy else None)
            series[m] = (t, seq, reply, rtt)
        if m in iperf:
            r["iperf"] = iperf_report(iperf[m], links.get(m))
        report["machines"][m] = r
    if busy and pings:
        tt = np.concatenate([s[0] for s in series.values()])
        report["iperf_busy_percent"] = 100 * float(in_windows(np.sort(tt), busy).mean())
    with open(os.path.join(out_dir, "wifi_report.json"), "w") as fh:
        json.dump(report, fh, indent=2)

    print(f"Wi-Fi, {out_dir}\n")
    print("link:")
    print("  %-10s %8s %8s %8s %10s %10s %8s %6s %8s %6s %10s %10s" % (
        "machine", "RSSI med", "RSSI p5", "RSSI min", "tx PHY med", "rx PHY med", "tx MCS",
        "NSS", "width", "roams", "tx traffic", "rx traffic"))
    for m, r in report["machines"].items():
        L = r.get("link")
        if not L:
            continue
        top = lambda d: next(iter(d), "-")                              # noqa: E731
        print("  %-10s %8s %8s %8s %10s %10s %8s %6s %8s %6d %10s %10s" % (
            m, f(L["rssi_dbm"]), f(L["rssi_dbm"], "p5"), f(L["rssi_dbm"], "min"),
            f(L["tx_phy_mbps"]), f(L["rx_phy_mbps"]), top(L["tx_mcs"]),
            "%s/%s" % (top(L["tx_nss"]), top(L["rx_nss"])), top(L["width_mhz"]) + " MHz",
            L["roams"], f(L.get("tx_traffic_mbps"), "mean") + " Mb/s",
            f(L.get("rx_traffic_mbps"), "mean") + " Mb/s"))
    print("\nping (per ICMP seq):")
    print("  %-10s %7s %7s %7s %8s %9s %9s %8s %8s %8s" % (
        "machine", "rate Hz", "sent", "lost", "loss %", "per msg %", "late", "RTT med",
        "RTT p95", "RTT max"))
    for m, r in report["machines"].items():
        P = r.get("ping")
        if not P:
            continue
        print("  %-10s %7.0f %7d %7d %8.2f %9.1f %9d %8s %8s %8s" % (
            m, P["rate_hz"], P["sent"], P["lost"], P["loss_percent"],
            P["loss_percent_per_message"] or 0, P["late_replies"], f(P["rtt_ms"]),
            f(P["rtt_ms"], "p95"), f(P["rtt_ms"], "max")))
    if busy:
        print(f"\ncontention: {len(busy)} iperf tests, the channel busy with one "
              f"{report['iperf_busy_percent']:.0f}% of the time")
        for m, r in report["machines"].items():
            P = r.get("ping")
            if not P or "during_iperf" not in P:
                continue
            di, no = P["during_iperf"], P["no_iperf"]
            print("  %-10s ping during iperf: loss %5.2f%%, RTT med %s ms p95 %s ms | "
                  "otherwise: loss %5.2f%%, RTT med %s ms p95 %s ms" % (
                      m, di["loss_percent"], f(di["rtt_ms"]), f(di["rtt_ms"], "p95"),
                      no["loss_percent"], f(no["rtt_ms"]), f(no["rtt_ms"], "p95")))
    for m, r in report["machines"].items():
        for d, I in (r.get("iperf") or {}).items():
            gv = I.get("goodput_vs_rssi")
            print("\niperf %s %s: %d tests (%d failed%s), goodput median %s Mbit/s (p5 %s, min %s), "
                  "slowest second median %s, retransmits %d, TCP RTT %s ms%s" % (
                      m, d, I["tests"], I["failed"],
                      (": " + "; ".join(I["errors"])) if I["errors"] else "",
                      f(I["goodput_mbps"]), f(I["goodput_mbps"], "p5"), f(I["goodput_mbps"], "min"),
                      f(I["slowest_second_mbps"]), I["retransmits"], f(I["tcp_rtt_ms"]),
                      f", {gv['mbps_per_db']:+.1f} Mbit/s per dB RSSI (r = {gv['correlation']:.2f})"
                      if gv else ""), end="")
    print()
    if not a.no_plots:
        for p in plots(out_dir, t0, links, series, iperf, busy):
            print(f"  {p}")
    print(f"  {os.path.join(out_dir, 'wifi_report.json')}")


if __name__ == "__main__":
    main()
