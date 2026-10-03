#!/usr/bin/env python
"""Self-test for analysis/comms/ntp_analysis.py on a synthetic comms/ntp/ntp.csv.

Four machines, 600 s of 1 Hz status, chrony polling every 8 s: measured
offsets of known size, an error bound of known parts, one clock step on
infra_1, reachability lost for 30 s on the sniffer, a 20 s gap in mobile_2's
status, and the sniffer's frequency following its temperature at
0.5 ppm/degC. The report must give those numbers back, and the pairwise
bound the sum of the two machines' bounds (the reference's own counting 0).

    python scoop_pipeline/tests/test_ntp_analysis.py
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
COLS = ["machine", "peer", "topic", "t", "t_log", "frame_id", "role", "hostname", "synchronized",
        "sync_source", "stratum", "offset_seconds", "last_offset_seconds", "jitter_seconds",
        "root_delay", "root_dispersion", "frequency_error_ppm", "reachability_percent",
        "reference_time", "leap_indicator", "seq", "clock_stepped", "offset_delta_seconds",
        "rms_offset_seconds", "temperature_c", "cpu_load_1min"]
T0 = 1_790_229_700.0


def write(path):
    rng = np.random.default_rng(3)
    rows = []
    meas = {"mobile_1": 20e-6, "mobile_2": 60e-6, "infra_1": 100e-6, "mobile_1/sniffer": 80e-6}
    for m, size in meas.items():
        last, ref = 0.0, T0
        for i in range(600):
            t = T0 + i + 0.2
            if m == "mobile_2" and 300 <= i < 320:
                continue                                       # publisher stalled 20 s
            if i % 8 == 0:
                last = size * (1 if (i // 8) % 2 else -1)       # |measured| = size every poll
                ref = t
            temp = 50 + 20 * i / 600
            rows.append({"machine": m, "peer": "", "topic": f"/{m}/ntp/client/status",
                         "t": t, "t_log": t + 0.01, "frame_id": m, "role": "client" if m != "mobile_1"
                         else "server", "hostname": m, "synchronized": 1,
                         "sync_source": "mobile1" if m != "mobile_1" else "7F7F0101", "stratum": 9,
                         "offset_seconds": 0.0, "last_offset_seconds": last,
                         "jitter_seconds": 5e-4, "root_delay": 0.004, "root_dispersion": 0.0005,
                         "frequency_error_ppm": (20 + 0.5 * (temp - 50)) if m == "mobile_1/sniffer"
                         else 15 + rng.normal(0, 0.001),
                         "reachability_percent": 50 if m == "mobile_1/sniffer" and 100 <= i < 130 else 100,
                         "reference_time": ref, "leap_indicator": "no_warning", "seq": i,
                         "clock_stepped": 1 if m == "infra_1" and i == 200 else 0,
                         "offset_delta_seconds": 0.0123 if m == "infra_1" and i == 200 else 0.0,
                         "rms_offset_seconds": size, "temperature_c": temp, "cpu_load_1min": 1.0})
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)


def main():
    tmp = tempfile.mkdtemp()
    f = []
    try:
        p = os.path.join(tmp, "processed", "20260924", "survey_1", "comms", "ntp", "ntp.csv")
        write(p)
        r = subprocess.run([sys.executable, os.path.join(ROOT, "analysis", "comms", "ntp_analysis.py"),
                            os.path.join(tmp, "processed", "20260924", "survey_1")],
                           capture_output=True, text=True)
        print(r.stdout[-3000:])
        if r.returncode:
            raise AssertionError(r.stderr[-2000:])
        rep = json.load(open(os.path.join(os.path.dirname(p), "ntp_report.json")))
        M = rep["machines"]
        if abs(M["infra_1"]["measured_offset_ms"]["median"] - 0.1) > 1e-6 or M["infra_1"]["polls"] != 75:
            f.append("infra_1 polls / measured: %s %s" % (M["infra_1"]["polls"],
                                                         M["infra_1"]["measured_offset_ms"]))
        if abs(M["mobile_2"]["error_bound_ms"]["max"] - 2.5) > 1e-6:
            f.append("bound: %s (0.004/2 + 0.0005 s = 2.5 ms)" % M["mobile_2"]["error_bound_ms"])
        st = M["infra_1"]["clock_steps"]
        if len(st) != 1 or abs(st[0]["by_ms"] - 12.3) > 1e-6:
            f.append("steps: %s" % st)
        if len(M["mobile_2"]["gaps"]) != 1 or abs(M["mobile_2"]["gaps"][0]["length_s"] - 21) > 1e-6:
            f.append("gaps: %s" % M["mobile_2"].get("gaps"))
        if M["mobile_2"]["lost_messages"] != 20:
            f.append("lost: %s" % M["mobile_2"]["lost_messages"])
        sp = M["mobile_1/sniffer"]["reach_below_100_spans"]
        if len(sp) != 1 or abs(sp[0]["to_s"] - sp[0]["from_s"] - 30) > 1e-6:
            f.append("reach spans: %s" % sp)
        fv = M["mobile_1/sniffer"]["frequency_vs_temperature"]
        if abs(fv["ppm_per_degC"] - 0.5) > 1e-6 or fv["correlation"] < 0.999:
            f.append("frequency vs temperature: %s" % fv)
        if not M["mobile_1"]["local_reference"] or M["mobile_1"]["reach_below_100_spans"]:
            f.append("mobile_1 local reference: %s" % M["mobile_1"]["local_reference"])
        pm = rep["pairwise_measured_ms"]["max"]
        if abs(pm["infra_1 ~ mobile_2"] - 0.16) > 1e-6 or abs(pm["infra_1 ~ mobile_1"] - 0.1) > 1e-6:
            f.append("pairwise measured: %s" % pm)
        pw = rep["pairwise_bound_ms"]["max"]
        if abs(pw["infra_1 ~ mobile_2"] - 5.0) > 1e-6 or abs(pw["infra_1 ~ mobile_1"] - 2.5) > 1e-6:
            f.append("pairwise: %s" % pw)
        for png in ("ntp_offsets.png", "ntp_frequency.png"):
            try:
                import matplotlib                                        # noqa: F401
            except ImportError:
                break
            if not os.path.exists(os.path.join(os.path.dirname(p), png)):
                f.append("no " + png)
    except AssertionError as e:
        f.append(str(e))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("\n".join(f) or "ntp analysis ok")
    sys.exit(1 if f else 0)


if __name__ == "__main__":
    main()
