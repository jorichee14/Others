#!/usr/bin/env python
"""Self-test for analysis/comms/sim_gap.py on comms tables with a known link.

mobile_1 pings at 50 Hz for 60 s: RTT 4 ms, 80 ms while its iperf up test
runs (20-30 s, 100 Mbit/s); no reply for 1 s at 40 s (an outage) and one
lone lost ping at 10 s (not one); no ping at all 50-52 s (left out); no
reply 54-54.2 s and then no ping logged until 56 s (an outage of 0.2 s
only: the stall is no evidence of the link, its messages left out). A
message every 100 ms must then take RTT / 2 + s / G: boxes 2.08 ms idle,
clouds 2 + 120 = 122 ms (never within a frame); the ten messages of the
outage lost; the stall's left out once its last ping is 0.5 s old.

    python scoop_pipeline/tests/test_sim_gap.py
"""
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def write(path, rows):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    tmp = tempfile.mkdtemp()
    f = []
    try:
        proc = os.path.join(tmp, "processed", "20990101", "coop_x")
        wd = os.path.join(proc, "comms", "wifi")
        os.makedirs(wd)
        T0 = 1000.0
        ping = []
        for i, t in enumerate(np.arange(0, 60, 0.02)):
            if 50 <= t < 52 or 54.2 <= t < 56:
                continue
            lost = (40 <= t < 41) or (54 <= t < 54.2) or i == 500
            rtt = 80.0 if 20 <= t <= 30 else 4.0
            ping.append({"machine": "mobile_1", "t": T0 + t, "t_log": T0 + t + 0.001,
                         "icmp_seq": i % 65536, "reply": "False" if lost else "True",
                         "rtt_ms": "" if lost else rtt})
        write(os.path.join(wd, "ping.csv"), ping)
        write(os.path.join(wd, "iperf.csv"), [
            {"machine": "mobile_1", "t": T0 + 20, "t_log": T0 + 30, "success": "True",
             "bitrate_mbps": 100.0, "reverse": "False", "duration_s": 10.0}])
        r = subprocess.run([sys.executable, os.path.join(ROOT, "analysis", "comms", "sim_gap.py"), proc],
                           capture_output=True, text=True)
        print(r.stdout[-3000:])
        if r.returncode:
            raise AssertionError(r.stderr[-2500:])
        rep = json.load(open(os.path.join(proc, "comms", "sim_gap", "sim_gap.json")))
        m = rep["robots"]["mobile_1"]
        dur = [round(o["duration_s"], 2) for o in m["outages"]]
        if dur != [1.0, 0.2]:
            f.append("outages %s s: want 1 and 0.2 (a lone lost ping is not one, a stall adds "
                     "nothing)" % dur)
        box = m["sizes"]["boxes"]
        if abs(box["idle"]["delay_ms"]["p50"] - 2.08) > 0.01:
            f.append("boxes idle p50 %s ms, want 2.08" % box["idle"]["delay_ms"]["p50"])
        if abs(box["loaded"]["delay_ms"]["p50"] - 40.08) > 0.01:
            f.append("boxes loaded p50 %s ms, want 40.08" % box["loaded"]["delay_ms"]["p50"])
        cl = m["sizes"]["clouds"]["all"]
        if abs(cl["delay_ms"]["p50"] - 122.0) > 0.01 or cl["within_frame_percent"] != 0:
            f.append("clouds: %s (want p50 122 ms, none within a frame)" % cl)
        n_all = box["all"]["messages"]
        lost = round(box["all"]["lost_percent"] * n_all / 100)
        if lost != 12:
            f.append("%d box messages lost, want the 10 + 2 of the outages" % lost)
        if n_all != 570:                                    # 600 minus each stall's last 15
            f.append("%d box messages, want 570 (the stalls left out after 0.5 s)" % n_all)
        if abs(m["outages"][0]["at_s"] - 40.0) > 0.05 if m["outages"] else False:
            f.append("outage at %s s, want 40" % m["outages"][0]["at_s"])
        v = rep["synthetic"]["V2X-ViT"]
        if v["delay_ms"] != 100.0 or v["lost_percent"] != 0:
            f.append("V2X-ViT synthetic %s" % v)
        try:
            import matplotlib                                    # noqa: F401
            if not os.path.exists(os.path.join(proc, "comms", "sim_gap", "sim_gap.png")):
                f.append("no sim_gap.png")
        except ImportError:
            pass
    except AssertionError as e:
        f.append(str(e))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("\n".join(f) or "sim gap ok")
    sys.exit(1 if f else 0)


if __name__ == "__main__":
    main()
