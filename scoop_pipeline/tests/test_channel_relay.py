#!/usr/bin/env python
"""Self-test for slam/channel_relay.py's scheduling (the Relay class; the ROS
glue needs a ROS 2 install and is checked in Swarm-SLAM's container).

Through test_channel's pass and the logged channel, robot 0 (mobile_1)
sends robot 1 (mobile_2) a 1 MB message at 45 s and one during its outage
at 40.5 s, and a 1 KB message to itself and robot 1 at 45 s; through
v2xvit, a message at 5 s. It must hold each until its arrival (0.187 s
later for the 1 MB one over two hops), drop the outage one, never send a
robot its own message, release everything at once before there is a
clock, and log every message.

    python scoop_pipeline/tests/test_channel_relay.py
"""
import csv
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "processing"))
sys.path.insert(0, os.path.join(ROOT, "slam"))
import channel_relay as CR                                          # noqa: E402
import link_trace as LT                                             # noqa: E402
import test_channel as TC                                           # noqa: E402
from scoop import channel as CH                                     # noqa: E402


def main():
    tmp = tempfile.mkdtemp()
    f = []
    try:
        proc = TC.make_pass(tmp)
        LT.make(proc, log=lambda *_: None)
        log = os.path.join(tmp, "relay.csv")
        r = CR.Relay(CH.make("logged:" + proc), ["mobile_1", "mobile_2"], log)
        T = TC.T0
        r.send(T + 45.0, 0, "/cslam/pose_graph", b"x" * 1000000, [1])
        r.send(T + 40.5, 0, "/cslam/pose_graph", b"y" * 10, [1])
        r.send(T + 45.0, 0, "/r1/cslam/heartbeat", b"z" * 1000, [0, 1])
        if r.due(T + 45.0):
            f.append("released before any arrival")
        got = r.due(T + 45.0 + 0.008)
        if [g[1] for g in got] != ["/r1/cslam/heartbeat"] or got[0][0] != 1:
            f.append("at +8 ms want only the 1 KB message to robot 1: %s" % [(g[0], g[1]) for g in got])
        if r.due(T + 45.0 + 0.186):
            f.append("1 MB released before 0.187 s")
        got = r.due(T + 45.0 + 0.188)
        if len(got) != 1 or len(got[0][2]) != 1000000:
            f.append("1 MB not released at 0.187 s: %s" % [(g[0], g[1]) for g in got])
        if r.due(T + 100.0):
            f.append("the outage message was delivered")
        r.close()
        rows = list(csv.DictReader(open(log)))
        if len(rows) != 3 or sum(int(x["lost"]) for x in rows) != 1:
            f.append("log: %d rows, %s lost (want 3, 1)" % (len(rows), [x["lost"] for x in rows]))
        if any(x["src"] == x["dst"] for x in rows):
            f.append("a robot was sent its own message")
        v = CR.Relay(CH.make("v2xvit"), ["a", "b"])
        v.send(5.0, 1, "/cslam/x", b"q", [0])
        if v.due(5.099) or len(v.due(5.1001)) != 1:
            f.append("v2xvit: not 100 ms")
        n = CR.Relay(CH.make("v2xvit"), ["a", "b"])
        n.send(None, 0, "/cslam/x", b"q", [1])
        if len(n.due(None)) != 1:
            f.append("before a clock: not released at once")
    except Exception:                                              # noqa: BLE001
        import traceback
        f.append(traceback.format_exc())
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("\n".join(f) or "channel relay ok")
    sys.exit(1 if f else 0)


if __name__ == "__main__":
    main()
