#!/usr/bin/env python
"""Self-test for analysis/slam/dsg_stream.py's plot side: a CSV of updates
(out of order, one repeated stamp) must give sizes in MB, rates in Mbit/s
from the bag time between updates (none where it does not advance), x from
the first update, and a figure.

    python scoop_pipeline/tests/test_dsg_stream.py
"""
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "analysis", "slam"))
import dsg_stream as D                                              # noqa: E402


def main():
    f = []
    tmp = tempfile.mkdtemp()
    try:
        p = os.path.join(tmp, "s.csv")
        rows = [(102.0, 1_000_000), (100.0, 250_000), (101.0, 500_000), (104.0, 2_000_000), (104.0, 2_100_000)]
        with open(p, "w") as fh:
            fh.write("t,t_recv,bytes\n")
            for t, b in rows:
                fh.write("%.6f,%.6f,%d\n" % (t, t + 0.01, b))
        t, b = D.load(p)
        x, mb, rate = D.series(t, b)
        if x != [0.0, 1.0, 2.0, 4.0, 4.0] or mb[:4] != [0.25, 0.5, 1.0, 2.0]:
            f.append("x %s mb %s" % (x, mb))
        want = [None, 4.0, 8.0, 8.0]
        if any((a is None) != (w is None) or (a is not None and abs(a - w) > 1e-9) for a, w in zip(rate, want)) \
                or rate[4] is not None:
            f.append("rate %s, want %s + [None]" % (rate, want))
        out = D.plot(p, goodput=175)
        if not os.path.getsize(out) > 1000:
            f.append("no figure")
        # several streams in one file: --topic picks one, none given refuses
        p2 = os.path.join(tmp, "server.csv")
        with open(p2, "w") as fh:
            fh.write("t,t_recv,bytes,topic\n")
            for k in range(4):
                fh.write("%d.0,%d.5,%d,/r0/hydra/backend/dsg\n" % (100 + k, 100 + k, 1000 * (k + 1)))
                fh.write("%d.2,%d.7,%d,/r1/hydra/backend/dsg\n" % (100 + k, 100 + k, 5000 * (k + 1)))
        t1, b1 = D.load(p2, "/r1/hydra/backend/dsg")
        if b1 != [5000, 10000, 15000, 20000] or abs(t1[0] - 100.2) > 1e-9:
            f.append("--topic r1: %s %s" % (t1, b1))
        try:
            D.load(p2)
            f.append("two streams without --topic did not refuse")
        except SystemExit:
            pass
    except Exception:                                              # noqa: BLE001
        import traceback
        f.append(traceback.format_exc())
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("\n".join(f) or "dsg stream ok")
    sys.exit(1 if f else 0)


if __name__ == "__main__":
    main()
