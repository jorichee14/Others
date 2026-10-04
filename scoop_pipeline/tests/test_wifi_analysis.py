#!/usr/bin/env python
"""Self-test for analysis/comms/wifi_analysis.py on synthetic comms/wifi tables.

Ping at 100 Hz for 900 s, so the 16-bit ICMP sequence wraps; 2 % of the
seqs never answered, 10 % time out first and get their reply late (published
as a second message, the way the ping monitor does). RTT 5 ms, 40 ms while an
iperf test runs. iperf down every 15 s, goodput 300 + 10 Mbit/s per dB of
RSSI; two tests fail. The report must count the loss per seq (2 %, not the
12 % a per-message count gives), find the late replies, split ping by iperf,
and give the goodput's RSSI slope back.

    python scoop_pipeline/tests/test_wifi_analysis.py
"""
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
T0 = 1_790_229_700.0


def write(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cols = []
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def main():
    rng = np.random.default_rng(5)
    tmp = tempfile.mkdtemp()
    f = []
    try:
        d = os.path.join(tmp, "processed", "20260924", "survey_1", "comms", "wifi")
        iperf, busy = [], []
        for i, t in enumerate(np.arange(T0 + 10, T0 + 890, 15.0)):
            rssi = -50 - 10 * np.sin(t / 60)
            ok = i not in (3, 7)
            iperf.append({"machine": "mobile_1", "peer": "", "topic": "/mobile_1/wifi/iperf_down",
                          "t": t, "t_log": t + 5, "protocol": "TCP", "reverse": 1, "duration_s": 4.0,
                          "success": int(ok), "error": "" if ok else "unable to connect",
                          "bitrate_mbps": 300 + 10 * (rssi + 50) if ok else "",
                          "bitrate_mbps_min": 250 if ok else "", "bitrate_mbps_std": 10 if ok else "",
                          "retransmits": 1, "rtt_ms_mean": "", "x": 0.0, "y": 0.0, "z": 0.0})
            busy.append((t, t + 4))
        wifi = []
        for t in np.arange(T0, T0 + 900, 0.5):
            for m in ("mobile_1", "mobile_2"):
                rssi = -50 - 10 * np.sin(t / 60) if m == "mobile_1" else -60.0
                wifi.append({"machine": m, "peer": "", "topic": f"/{m}/wifi/status", "t": t,
                             "t_log": t, "interface": "wlx", "mac_address": m, "associated": 1,
                             "bssid": "AA", "channel": 149, "signal_dbm": rssi,
                             "link_quality_ratio": 0.7, "tx_bitrate_mbps": 433, "rx_bitrate_mbps": 780,
                             "tx_mcs": 9, "rx_mcs": 9, "tx_nss": 1, "rx_nss": 2, "rx_width_mhz": 80,
                             "rx_phy_mode": "VHT", "rx_dropped": int((t - T0) // 100),
                             "tx_bytes": int((t - T0) * 1.25e6), "rx_bytes": int((t - T0) * 2.5e6),
                             "x": (t - T0) / 30 if m == "mobile_1" else 5.0, "y": 1.0, "z": 0.3})
        ping = []
        n = 90000
        lost = rng.random(n) < 0.02
        late = ~lost & (rng.random(n) < 0.10)
        for i in range(n):
            t = T0 + i * 0.01
            inb = any(a <= t <= b for a, b in busy)
            rtt = 40.0 if inb else 5.0
            row = {"machine": "mobile_1", "peer": "", "topic": "/mobile_1/wifi/ping",
                   "target": "192.168.51.11", "icmp_seq": i % 65536, "window": 20}
            if lost[i] or late[i]:
                ping.append(dict(row, t=t + 1.0, t_log=t + 1.0, reply=0, rtt_ms=""))
            if not lost[i]:
                tr = t + (1.2 if late[i] else rtt / 1000)
                ping.append(dict(row, t=tr, t_log=tr, reply=1, rtt_ms=1200.0 if late[i] else rtt))
        ping.sort(key=lambda r: r["t"])
        write(os.path.join(d, "ping.csv"), ping)
        write(os.path.join(d, "wifi.csv"), wifi)
        write(os.path.join(d, "iperf.csv"), iperf)
        r = subprocess.run([sys.executable, os.path.join(ROOT, "analysis", "comms", "wifi_analysis.py"),
                            os.path.join(tmp, "processed", "20260924", "survey_1")],
                           capture_output=True, text=True)
        print(r.stdout[-3500:])
        if r.returncode:
            raise AssertionError(r.stderr[-2500:])
        rep = json.load(open(os.path.join(d, "wifi_report.json")))["machines"]
        P = rep["mobile_1"]["ping"]
        if P["sent"] != n or abs(P["loss_percent"] - 100 * lost.mean()) > 1e-6:
            f.append("ping sent / loss: %s %.3f vs %.3f" % (P["sent"], P["loss_percent"],
                                                            100 * lost.mean()))
        if P["late_replies"] != int(late.sum()) or P["loss_percent_per_message"] < 10:
            f.append("late replies: %s vs %d" % (P["late_replies"], late.sum()))
        if abs(P["during_iperf"]["rtt_ms"]["median"] - 40) > 1e-6 or \
                abs(P["no_iperf"]["rtt_ms"]["median"] - 5) > 1e-6:
            f.append("contention split: %s / %s" % (P["during_iperf"]["rtt_ms"], P["no_iperf"]["rtt_ms"]))
        I = rep["mobile_1"]["iperf"]["down"]
        if I["failed"] != 2 or "unable to connect" not in I["errors"]:
            f.append("iperf failures: %s %s" % (I["failed"], I["errors"]))
        if abs(I["goodput_vs_rssi"]["mbps_per_db"] - 10) > 0.5 or I["goodput_vs_rssi"]["correlation"] < 0.99:
            f.append("goodput vs RSSI: %s" % I["goodput_vs_rssi"])
        L = rep["mobile_1"]["link"]
        if abs(L["tx_traffic_mbps"]["mean"] - 10) > 0.1 or abs(L["rx_traffic_mbps"]["mean"] - 20) > 0.1:
            f.append("traffic: %s %s" % (L["tx_traffic_mbps"], L["rx_traffic_mbps"]))
        if L["rx_dropped_during_pass"] != 8 or L["roams"] != 0:
            f.append("drops / roams: %s %s" % (L.get("rx_dropped_during_pass"), L["roams"]))
        try:
            import matplotlib                                            # noqa: F401
            for png in ("wifi_timeline.png", "wifi_map.png"):
                if not os.path.exists(os.path.join(d, png)):
                    f.append("no " + png)
        except ImportError:
            pass
    except AssertionError as e:
        f.append(str(e))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("\n".join(f) or "wifi analysis ok")
    sys.exit(1 if f else 0)


if __name__ == "__main__":
    main()
