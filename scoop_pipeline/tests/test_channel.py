#!/usr/bin/env python
"""Self-test for processing/link_trace.py and scoop/channel.py.

A pass with two robots pinging at 50 Hz for 60 s:
  mobile_1  RTT 4 ms; 80 ms during its own iperf up test (20-30 s, 100 Mbit/s;
            down 200 Mbit/s at 32-34 s); 30 ms while mobile_2's test runs
            (5-15 s: cross traffic); no reply 40-41 s (an outage); no ping
            logged 50-52 s (a stall)
  mobile_2  RTT 10 ms; its iperf up 50 Mbit/s at 5-15 s, down 80 Mbit/s at 36-38 s

The logged channel must give: one hop to the server RTT/2 + s/G_up; the
robot's own test not counted twice (4 ms held through 20-30 s); the other
robot's test counted (30 ms at 5-15 s); robot to robot two hops (RTT_1/2 +
RTT_2/2, s/G_up(1) + s/G_down(2)); lost in the outage; not measured in the
stall; with ,queue a second message waits for the first. The synthetic
channels: V2X-ViT 100 ms, CoBEVFlow's binomial mean, loss and burst lengths
as asked. The fitted channel: the logged latency and outage share.

    python scoop_pipeline/tests/test_channel.py
"""
import csv
import os
import shutil
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "processing"))
import link_trace as LT                                             # noqa: E402
from scoop import channel as CH                                     # noqa: E402

T0 = 1000.0


def write(path, rows):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def make_pass(tmp):
    proc = os.path.join(tmp, "processed", "20990101", "coop_x")
    wd = os.path.join(proc, "comms", "wifi")
    os.makedirs(wd)
    ping = []
    for m in ("mobile_1", "mobile_2"):
        for i, s in enumerate(np.arange(0, 60, 0.02)):
            if m == "mobile_1" and 50 <= s < 52:
                continue
            lost = m == "mobile_1" and 40 <= s < 41
            if m == "mobile_1":
                rtt = 80.0 if 20 <= s <= 30 else 30.0 if 5 <= s <= 15 else 4.0
            else:
                rtt = 10.0
            ping.append({"machine": m, "t": T0 + s, "t_log": T0 + s + 0.001, "icmp_seq": i % 65536,
                         "reply": "False" if lost else "True", "rtt_ms": "" if lost else rtt})
    write(os.path.join(wd, "ping.csv"), ping)
    it = lambda m, t0, d, g, rev: {"machine": m, "t": T0 + t0, "t_log": T0 + t0 + d, "success": "True",  # noqa: E731
                                   "bitrate_mbps": g, "reverse": rev, "duration_s": d}
    write(os.path.join(wd, "iperf.csv"), [it("mobile_1", 20, 10, 100.0, "False"),
                                          it("mobile_1", 32, 2, 200.0, "True"),
                                          it("mobile_2", 5, 10, 50.0, "False"),
                                          it("mobile_2", 36, 2, 80.0, "True")])
    return proc


def close(a, b, tol=1e-6):
    return a is not None and b is not None and abs(a - b) <= tol


def main():
    tmp = tempfile.mkdtemp()
    f = []
    try:
        proc = make_pass(tmp)
        LT.make(proc, log=lambda *_: None)
        ch = CH.make("logged:" + proc)
        S = 1e6                                                   # bytes
        # one hop, idle: 2 ms + 1 MB at 100 Mbit/s (mobile_1's up test)
        d = ch.deliver("mobile_1", "server", T0 + 45.0, S)
        if not close(d.delay_s, 0.002 + 0.08):
            f.append("one hop idle: %s s, want 0.082" % d.delay_s)
        # own test: 80 ms not counted, 4 ms held
        d = ch.deliver("mobile_1", "server", T0 + 25.0, S)
        if not close(d.delay_s, 0.002 + 0.08) or not d.measured:
            f.append("own test: %s s measured %s, want 0.082 (held 4 ms)" % (d.delay_s, d.measured))
        d = ch.deliver("mobile_1", "server", T0 + 30.0 + 1e-9, S)
        if not d.measured or not close(d.delay_s, 0.002 + 0.08):
            f.append("just after its own test: %s s measured %s, want 0.082" % (d.delay_s, d.measured))
        # cross traffic: mobile_2's test, mobile_1's 30 ms counted
        d = ch.deliver("mobile_1", "server", T0 + 10.0, S)
        if not close(d.delay_s, 0.015 + 0.08):
            f.append("cross traffic: %s s, want 0.095" % d.delay_s)
        # robot to robot: 2 + 5 ms, 1 MB at 100 up then 80 down
        d = ch.deliver("mobile_1", "mobile_2", T0 + 45.0, S)
        if not close(d.delay_s, 0.002 + 0.005 + 0.08 + 0.1):
            f.append("two hops: %s s, want 0.187" % d.delay_s)
        # the other way: mobile_2 up 50, mobile_1 down 200
        d = ch.deliver("mobile_2", "mobile_1", T0 + 45.0, S)
        if not close(d.delay_s, 0.005 + 0.002 + 0.16 + 0.04):
            f.append("two hops back: %s s, want 0.207" % d.delay_s)
        d = ch.deliver("mobile_1", "mobile_2", T0 + 40.5, S)
        if not d.lost:
            f.append("not lost in the outage")
        d = ch.deliver("mobile_1", "server", T0 + 51.5, S)
        if d.lost or d.measured:
            f.append("stall: lost %s measured %s (want neither)" % (d.lost, d.measured))
        q = CH.make("logged:%s,queue" % proc)
        a = q.deliver("mobile_1", "server", T0 + 45.0, S)
        b = q.deliver("mobile_1", "server", T0 + 45.0, S)
        if not close(b.t_arrive - a.t_arrive, 0.08):
            f.append("queue: second %s s after the first, want 0.08" % (b.t_arrive - a.t_arrive))
        # synthetic
        if not close(CH.make("v2xvit").deliver("a", "b", 5.0, 1e9).delay_s, 0.1):
            f.append("v2xvit not 100 ms")
        cb = CH.make("cobevflow:0.3", seed=1)
        m = np.mean([cb.deliver("a", "b", 0.0, 1).delay_s for _ in range(4000)])
        if abs(m - 0.3) > 0.01:
            f.append("cobevflow:0.3 mean %.3f s, want 0.3" % m)
        sy = CH.make("synthetic:delay_ms=20,mbps=80,loss=0.1", seed=2)
        r = [sy.deliver("a", "b", t, 1e5) for t in np.arange(0, 400, 0.1)]
        lost = np.mean([x.lost for x in r])
        if abs(lost - 0.1) > 0.02 or not close([x for x in r if not x.lost][0].delay_s, 0.02 + 0.01):
            f.append("synthetic: lost %.3f (0.1), delay %s (0.03)" % (lost, [x.delay_s for x in r][:2]))
        ge = CH.make("synthetic:loss=0.1,burst_s=2", seed=3)
        bad = np.array([ge.deliver("a", "b", t, 1).lost for t in np.arange(0, 4000, 0.1)])
        runs = np.diff(np.r_[0, bad.astype(int), 0])
        mean_len = bad.sum() * 0.1 / max((runs == 1).sum(), 1)
        if abs(bad.mean() - 0.1) > 0.03 or abs(mean_len - 2.0) > 0.5:
            f.append("burst loss: share %.3f (0.1), mean outage %.2f s (2)" % (bad.mean(), mean_len))
        # fit: mobile_1 to server
        fit = CH.make("fit:" + proc, seed=4)
        p = fit.fit("mobile_1", "server")
        lat = np.exp(p["mu"])
        if not 0.002 <= lat <= 0.008 or not 0.01 < p["outage_share"] < 0.03:
            f.append("fit: latency %.4f s (2-8 ms), outage share %.3f (~1/60)" % (lat, p["outage_share"]))
    except Exception:                                              # noqa: BLE001
        import traceback
        f.append(traceback.format_exc())
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("\n".join(f) or "channel ok")
    sys.exit(1 if f else 0)


if __name__ == "__main__":
    main()
