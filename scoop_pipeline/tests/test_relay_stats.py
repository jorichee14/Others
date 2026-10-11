#!/usr/bin/env python
"""Self-test for analysis/slam/relay_stats.py on two made-up runs of one
robot streaming an update a second to a server:

  ideal   every update arrives at once: delay 0, backlog 1, map age the
          update period (at most 1 s)
  slow    each update takes 1.5 s (so two are on the link at once: backlog
          2), the 4th is lost: delay 1.5 s, map age up to 3.5 s (after the
          lost one, the newest copy is the 3rd until the 5th arrives)

plus a robot-to-robot message (no map age without --to) and a row sent
before the clock (left out). Writes relay_stats.json in each run and the
figure.

    python scoop_pipeline/tests/test_relay_stats.py
"""
import json
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "analysis", "slam"))
import relay_stats as RS                                            # noqa: E402

HEAD = "t_send,topic,src,dst,bytes,lost,measured,t_arrive,t_published\n"


def write(run, rows):
    os.makedirs(run)
    with open(os.path.join(run, "relay.csv"), "w") as fh:
        fh.write(HEAD)
        for r in rows:
            fh.write(",".join("" if v is None else str(v) for v in r) + "\n")


def main():
    f = []
    tmp = tempfile.mkdtemp()
    try:
        T = 1000.0
        ideal, slow = os.path.join(tmp, "ideal"), os.path.join(tmp, "slow")
        write(ideal, [[None, "/r0/x", "mobile_1", "server", 10, 0, 1, None, None]] +
              [[T + k, "/r0/hydra/backend/dsg", "mobile_1", "server", 1000000, 0, 1, T + k, T + k]
               for k in range(10)] +
              [[T + 0.5, "/cslam/x", "mobile_1", "mobile_2", 500, 0, 1, T + 0.51, T + 0.51]])
        rows = []
        for k in range(10):
            lost = k == 3
            rows.append([T + k, "/r0/hydra/backend/dsg", "mobile_1", "server", 1000000, int(lost), 1,
                         None if lost else T + k + 1.5, None if lost else T + k + 1.5])
        write(slow, rows)
        reps = [RS.run_stats(ideal, {"server"}), RS.run_stats(slow, {"server"})]
        a, b = reps[0]["streams"]["mobile_1->server"], reps[1]["streams"]["mobile_1->server"]
        if a["messages"] != 10 or a["lost"] != 0 or a["delay_p95_s"] != 0 or a["backlog_max"] != 1:
            f.append("ideal %s" % {k: v for k, v in a.items() if not k.startswith("age_") or k.endswith("_s")})
        if not (0.85 <= a["age_p95_s"] <= 1.0 and a["age_max_s"] <= 1.0 + 1e-9):
            f.append("ideal age p95 %s max %s" % (a["age_p95_s"], a["age_max_s"]))
        if b["lost"] != 1 or abs(b["delay_p50_s"] - 1.5) > 1e-9 or b["backlog_max"] != 2 or b["mb"] != 10.0:
            f.append("slow %s" % {k: v for k, v in b.items() if k not in ("age_t", "age_s")})
        if not (3.4 <= b["age_max_s"] <= 3.5 + 1e-9):
            f.append("slow age max %s, want 3.5" % b["age_max_s"])
        rr = reps[0]["streams"]["mobile_1->mobile_2"]
        if "age_p95_s" in rr or rr["messages"] != 1:
            f.append("robot to robot: %s" % rr)
        j = json.load(open(os.path.join(slow, "relay_stats.json")))
        if "age_s" in j["streams"]["mobile_1->server"] or j["streams"]["mobile_1->server"]["lost"] != 1:
            f.append("json %s" % j)
        txt = RS.table(reps)
        if "mobile_1->server" not in txt or "slow" not in txt:
            f.append("table:\n" + txt)
        png = RS.plot(reps, os.path.join(tmp, "age.png"))
        if not os.path.getsize(png) > 1000:
            f.append("no figure")
    except Exception:                                              # noqa: BLE001
        import traceback
        f.append(traceback.format_exc())
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("\n".join(f) or "relay stats ok")
    sys.exit(1 if f else 0)


if __name__ == "__main__":
    main()
