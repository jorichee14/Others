#!/usr/bin/env python3
"""Self-test for analysis/headroom.py: margins, missing terms, and the speed
percentile from a TUM track.

    python scoop_pipeline/tests/test_headroom.py
"""
import json
import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "analysis"))
import headroom as H                                                # noqa: E402


def main():
    with tempfile.TemporaryDirectory() as d:
        # a track at 0.8 m/s for 90 % of the time and 2.0 m/s for 10 %
        t = np.arange(0, 100, 0.1)
        v = np.where((t % 10) < 9, 0.8, 2.0)
        x = np.concatenate([[0], np.cumsum(v[:-1] * 0.1)])
        tum = os.path.join(d, "ref.tum")
        with open(tum, "w") as fh:
            for ti, xi in zip(t, x):
                fh.write("%f %f 0 0 0 0 0 1\n" % (ti, xi))
        sp = H.p_speed([tum], 95)
        assert abs(sp - 2.0) < 1e-6, sp
        sp50 = H.p_speed([tum], 50)
        assert abs(sp50 - 0.8) < 1e-6, sp50

        cfg = {"speed_mps": 1.0,
               "breaking_points": [{"name": "pose", "m": 0.2}, {"name": "delay", "s": 0.1}],
               "terms": [{"name": "clock", "s": 0.001}, {"name": "cart", "m": 0.031},
                         {"name": "radar", "m": None}]}
        rep = H.evaluate(cfg, 1.0, 10.0)
        # like with like: time terms against the delay point (0.1 m at 1 m/s),
        # length terms against the pose point (0.2 m)
        assert rep["smallest_breaking_point_m"] == {"pose": 0.2, "delay": 0.1}
        by = {t["name"]: t for t in rep["terms"]}
        assert abs(by["clock"]["margin"] - 100.0) < 1e-9 and by["clock"]["pass"]
        assert abs(by["clock"]["margin_other_kind"] - 200.0) < 1e-9
        assert abs(by["cart"]["margin"] - 0.2 / 0.031) < 1e-9 and by["cart"]["pass"] is False
        assert abs(by["cart"]["margin_other_kind"] - 0.1 / 0.031) < 1e-9
        assert by["radar"]["m"] is None and by["radar"]["pass"] is None
        H.print_report(rep)
        out = os.path.join(d, "h.png")
        H.plot(rep, out)
        assert os.path.getsize(out) > 1000

        # the shipped config loads and evaluates
        shipped = json.load(open(os.path.join(ROOT, "configs", "headroom_terms.json")))
        rep2 = H.evaluate(shipped, shipped["speed_mps"], 10.0)
        assert any(t["m"] is None for t in rep2["terms"]), "shipped config should still list TODO terms"
        assert all(t["pass"] for t in rep2["terms"] if t["m"] is not None and t["name"].startswith("clock offset, measured"))
    print("\nOK headroom")


if __name__ == "__main__":
    main()
