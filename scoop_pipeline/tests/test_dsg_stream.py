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
    except Exception:                                              # noqa: BLE001
        import traceback
        f.append(traceback.format_exc())
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("\n".join(f) or "dsg stream ok")
    sys.exit(1 if f else 0)


if __name__ == "__main__":
    main()
