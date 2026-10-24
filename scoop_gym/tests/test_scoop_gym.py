#!/usr/bin/env python
"""Self-test for scoop_gym on the fake pass of scoop_pipeline/tests/test_channel.py
(two robots, 60 s, mobile_1 with an outage at 40-41 s and a stall at 50-52 s).

Checks: the episode covers the span both robots have a trace for, one step
per 100 ms frame; under the ideal link every message is on time; the logged
link agrees with scoop.channel's own delivery, message by message; a 1.5 MB
cloud mobile_2 -> mobile_1 at 50 Mbit/s up misses the 100 ms frame with
cause "transmission" while boxes arrive; a message in the outage is lost
with cause "outage"; the deadline rule beats always-clouds on return under
the logged link; the link monitor reaches the observation; the gymnasium
view runs an episode; the run CLI writes its logs.

    python scoop_gym/tests/test_scoop_gym.py
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
GYM = os.path.dirname(HERE)
PIPE = os.path.join(os.path.dirname(GYM), "scoop_pipeline")
sys.path[:0] = [GYM, os.path.join(PIPE, "tests"), os.path.join(PIPE, "processing"), PIPE]
import link_trace as LT                                             # noqa: E402
import test_channel as TC                                           # noqa: E402
from scoop import channel as CH                                     # noqa: E402
from scoop_gym import CommSchedule, gymnasium_env                   # noqa: E402
from scoop_gym import baselines                                     # noqa: E402
from scoop_gym.run import episode                                   # noqa: E402


def main():
    tmp = tempfile.mkdtemp()
    f = []
    try:
        proc = TC.make_pass(tmp)
        LT.make(proc, log=lambda *_: None)
        logged = "logged:" + proc
        snd = [("mobile_2", "mobile_1")]

        env = CommSchedule(logged, senders=snd)
        t0, t1 = env.times[0], env.times[-1]
        if not (TC.T0 <= t0 < TC.T0 + 0.1 and t1 > TC.T0 + 59.0):
            f.append("span %.2f-%.2f s, want the pass's 0-60 s" % (t0 - TC.T0, t1 - TC.T0))
        if not np.allclose(np.diff(env.times), 0.1):
            f.append("steps are not 100 ms frames")

        ideal = CommSchedule("ideal", senders=snd)
        episode(ideal, baselines.make("always_clouds"))
        s = ideal.record.summary()
        if s["in_deadline"] != s["messages"] or s["messages"] != len(ideal.times):
            f.append("ideal: %s" % s)
        v = CommSchedule("v2xvit", senders=snd)
        episode(v, baselines.make("always_clouds"))
        if v.record.summary()["in_deadline_share"] != 1.0:
            f.append("v2xvit's fixed 100 ms not inside the 100 ms deadline: %s" % v.record.summary())

        episode(env, baselines.make("always_clouds"))
        ch = CH.make(logged)
        for r in env.record.rows[::50]:
            d = ch.deliver(r["src"], r["dst"], r["t_send"], r["bytes"])
            if d.lost != r["lost"] or (not d.lost and abs(d.t_arrive - r["t_arrive"]) > 1e-9):
                f.append("logged: gym %s, channel %s at %.2f s" % (r["t_arrive"], d.t_arrive, r["t_send"] - TC.T0))
                break

        at = {round(r["t_send"] - TC.T0, 1): r for r in env.record.rows}
        r = at.get(45.0)
        # mobile_2 up 50 Mbit/s, mobile_1 down 200: 0.24 + 0.06 s of transmission
        if r is None or r["cause"] != "transmission" or abs(r["delay_ms"] - (5 + 2 + 240 + 60)) > 1e-6:
            f.append("1.5 MB at 45 s: %s, want 307 ms, cause transmission" % r)
        r = at.get(40.5)
        if r is None or not r["lost"] or r["cause"] != "outage":
            f.append("in the outage: %s, want lost, cause outage" % r)

        env_b = CommSchedule(logged, senders=snd)
        episode(env_b, baselines.make("always_boxes"))
        sb = env_b.record.summary()
        if sb["in_deadline_share"] < 0.9:
            f.append("boxes on the logged link: %.2f in deadline, want most" % sb["in_deadline_share"])

        rets = {}
        for p in ("always_clouds", "deadline_120"):
            e = CommSchedule(logged, senders=snd)
            rets[p] = episode(e, baselines.make(p))["mobile_2"]
        if not rets["deadline_120"] > rets["always_clouds"]:
            f.append("deadline rule %.1f not above always clouds %.1f" % (rets["deadline_120"], rets["always_clouds"]))

        obs, _ = env.reset()
        if obs["mobile_2"][4] < 0 or obs["mobile_2"][5] < 0:
            f.append("link monitor missing from the observation: %s" % obs["mobile_2"])

        try:
            import gymnasium  # noqa: F401
            g = gymnasium_env(CommSchedule(logged, senders=snd))
            o, _ = g.reset(seed=0)
            n = 0
            while True:
                o, rr, term, trunc, _ = g.step(g.action_space.sample())
                n += 1
                if term or trunc:
                    break
            if n != len(env.times) or not g.observation_space.contains(o):
                f.append("gymnasium view: %d steps, obs %s" % (n, o))
        except ImportError:
            print("gymnasium not installed: its view not tested")

        sc = os.path.join(tmp, "sc.yaml")
        with open(sc, "w") as fh:
            json.dump({"link": logged, "senders": [{"src": "mobile_2", "dst": "mobile_1"}]}, fh)
        out = os.path.join(tmp, "out")
        subprocess.run([sys.executable, "-m", "scoop_gym.run", sc, "--links", "ideal", logged,
                        "--policies", "always_boxes", "deadline_120", "--out", out],
                       cwd=GYM, check=True, capture_output=True)
        if len(json.load(open(os.path.join(out, "summary.json")))) != 4:
            f.append("run CLI: summary.json does not hold 4 runs")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    for x in f:
        print("FAIL", x)
    print("OK" if not f else "%d failure(s)" % len(f))
    return 1 if f else 0


if __name__ == "__main__":
    sys.exit(main())
