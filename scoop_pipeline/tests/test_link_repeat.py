#!/usr/bin/env python3
"""Self-test for analysis/comms/link_repeat.py: two synthetic passes with
known distributions; the distances must come out where arithmetic puts them.

    python scoop_pipeline/tests/test_link_repeat.py
"""
import csv
import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "analysis", "comms"))
import link_repeat as LR                                            # noqa: E402


def write(path, rows, cols):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def make_pass(root, name, rtt_shift, rssi_shift, goodput, rng):
    proc = os.path.join(root, "20260924", name)
    n = 400
    t = np.arange(n) * 0.01
    x = np.linspace(0, 10, n)
    y = np.zeros(n)
    ping = [{"machine": "mobile_1", "t": ti, "reply": 1 if i % 20 else 0,
             "rtt_ms": 5 + rtt_shift + rng.normal(0, 0.1), "x": xi, "y": yi}
            for i, (ti, xi, yi) in enumerate(zip(t, x, y))]
    write(os.path.join(proc, "comms", "wifi", "ping.csv"), ping,
          ["machine", "t", "reply", "rtt_ms", "x", "y"])
    wifi = [{"machine": "mobile_1", "t": ti, "signal_dbm": -50 + rssi_shift - 2 * xi + rng.normal(0, 0.1),
             "x": xi, "y": yi} for ti, xi, yi in zip(t, x, y)]
    wifi += [{"machine": "infra_1", "t": ti, "signal_dbm": -60.0, "x": 3.0, "y": 3.0} for ti in t[:50]]
    write(os.path.join(proc, "comms", "wifi", "wifi.csv"), wifi,
          ["machine", "t", "signal_dbm", "x", "y"])
    iperf = [{"machine": "mobile_1", "t": ti, "success": 1, "reverse": i % 2, "bitrate_mbps": goodput[i % 2]}
             for i, ti in enumerate(t[::20])]
    write(os.path.join(proc, "comms", "wifi", "iperf.csv"), iperf,
          ["machine", "t", "success", "reverse", "bitrate_mbps"])
    return proc


def main():
    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory() as d:
        A = make_pass(d, "survey_1", 0.0, 0.0, (280.0, 600.0), rng)
        B = make_pass(d, "coop_2", 20.0, -6.0, (140.0, 300.0), rng)
        C = make_pass(d, "survey_1_day2", 0.5, -0.5, (275.0, 590.0), rng)

        mode = LR.run(A, B, 0.5, "survey", "coop")
        day = LR.run(A, C, 0.5, "survey", "survey_day2")

        # RTT: a 20 ms shift in the mode pair, 0.5 ms in the day pair
        assert abs(mode["rtt_ms"]["mobile_1"]["w1"] - 20.0) < 0.1, mode["rtt_ms"]
        assert abs(day["rtt_ms"]["mobile_1"]["w1"] - 0.5) < 0.1, day["rtt_ms"]
        assert mode["rtt_ms"]["mobile_1"]["ks"] > 0.99
        # only answered pings count
        assert mode["rtt_ms"]["mobile_1"]["n_a"] == 380
        # goodput: up and down separated
        assert abs(mode["goodput_mbps"]["mobile_1 up"]["w1"] - 140.0) < 1e-6
        assert abs(mode["goodput_mbps"]["mobile_1 down"]["w1"] - 300.0) < 1e-6
        # RSSI: the shift shows per machine; infra_1 identical
        assert abs(mode["rssi_dbm"]["mobile_1"]["w1"] - 6.0) < 0.1
        assert mode["rssi_dbm"]["infra_1"]["w1"] == 0.0
        # RSSI on grid: same route, so every cell shared; shift recovered per cell
        g = mode["rssi_grid"]["mobile_1"]
        assert g["shared_cells"] == g["cells_a"] == g["cells_b"] > 10, g
        assert abs(g["median_diff_db"] - 6.0) < 0.2, g
        # the E4 ratio: day / mode well below 0.5
        r = day["rtt_ms"]["mobile_1"]["w1"] / mode["rtt_ms"]["mobile_1"]["w1"]
        assert r < 0.5, r
        LR.print_report(mode)
        LR.ratio_against(day, mode)
    print("\nOK link_repeat")


if __name__ == "__main__":
    main()
