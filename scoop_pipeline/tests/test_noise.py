#!/usr/bin/env python
"""Self-test for check.py noise: a bag of 20 scans of an 8 x 6 x 3 m room
from its middle, every point off its surface by a normal 1.4 cm along the
surface's normal. One scan's thickness on walls and floors must read
3.29 x 1.4 = 4.6 cm, with no trajectory and no map.

    python tests/test_noise.py
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from scoop import rosmsg                                               # noqa: E402
from scoop.bagwrite import BagWriter                                   # noqa: E402

TOPIC = "/ouster/points"


def scan(rng, n=60000, sigma=0.014):
    """Points on the room's walls / floor / ceiling, in the frame of a sensor
    at (4, 3, 1), each off its surface by sigma along the normal."""
    face = rng.integers(0, 6, n)
    u, v = rng.random(n), rng.random(n)
    P = np.zeros((n, 3))
    nrm = np.zeros((n, 3))
    for k, (axis, val, a, b, la, lb) in enumerate(
            [(0, 0, 1, 2, 6, 3), (0, 8, 1, 2, 6, 3), (1, 0, 0, 2, 8, 3),
             (1, 6, 0, 2, 8, 3), (2, 0, 0, 1, 8, 6), (2, 3, 0, 1, 8, 6)]):
        m = face == k
        P[m, axis] = val
        P[m, a], P[m, b] = u[m] * la, v[m] * lb
        nrm[m, axis] = 1
    P += nrm * rng.normal(0, sigma, (n, 1))
    return P - np.array([4.0, 3.0, 1.0])


def main():
    fails = []
    d = tempfile.mkdtemp(prefix="noise_")
    try:
        rng = np.random.default_rng(0)
        bag = os.path.join(d, "bag")
        with BagWriter(bag) as w:
            for k in range(20):
                ps = scan(rng)
                pts = np.zeros(len(ps), [("x", "<f4"), ("y", "<f4"), ("z", "<f4")])
                pts["x"], pts["y"], pts["z"] = ps.T
                ns = 1_800_000_000 * 10**9 + k * 10**8
                w.write(TOPIC, rosmsg.pointcloud2(pts, ns, "os_lidar"), ns)
        r = subprocess.run([sys.executable, os.path.join(ROOT, "check.py"), "noise", bag,
                            "--scans", "10", "--patches", "150"], capture_output=True, text=True)
        print(r.stdout[-2000:] + r.stderr[-1500:])
        got = {}
        for name in ("walls", "floors / ceilings"):
            m = re.search(r"^\s+%s\s+([\d.]+) cm" % re.escape(name), r.stdout, re.M)
            got[name] = float(m.group(1)) if m else float("nan")
        if r.returncode or not all(abs(v - 4.6) < 0.5 for v in got.values()):
            fails.append(f"one scan's thickness {got}, want 4.6 cm")
        else:
            print(f"  ok    walls {got['walls']} cm, floors {got['floors / ceilings']} cm "
                  "(3.29 x 1.4 = 4.6)")
    finally:
        shutil.rmtree(d, ignore_errors=True)
    for f in fails:
        print("  FAIL  " + f)
    print("noise ok" if not fails else "noise FAILED")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
