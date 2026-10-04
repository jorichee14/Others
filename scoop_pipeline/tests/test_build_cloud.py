#!/usr/bin/env python
"""Self-test for map_stages/09_build_coop_cloud.py through the shared,
deskewed scan reader (pipeline_common.iter_scans, frame="world").

The scene and the sweeping lidar are tests/test_scan_consistency.py's: a
16 x 12 x 3 m room, a robot driving a square loop with turns on the spot at
45 deg/s, a 48 x 768 lidar sweeping over 100 ms with per-point times. With
the true trajectory, 09 with deskew must put the walls as thin as the range
noise; without deskew the turns smear them.

    python scoop_pipeline/tests/test_build_cloud.py
"""
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import test_scan_consistency as SC                                  # noqa: E402
from scoop.bagwrite import BagWriter                                # noqa: E402


def read_pcd(path):
    with open(path, "rb") as f:
        n = 0
        while True:
            ln = f.readline().decode()
            if ln.startswith("POINTS"):
                n = int(ln.split()[1])
            if ln.startswith("DATA"):
                break
        return np.frombuffer(f.read(n * 12), np.float32).reshape(n, 3)


def wall_thickness(P):
    """5-95 % spread of the points on the four walls (within 0.15 m of each),
    at mid height, in cm."""
    lo, hi = SC.ROOM
    mid = (P[:, 2] > 0.8) & (P[:, 2] < 2.2)
    d = []
    for ax, c in ((0, lo[0]), (0, hi[0]), (1, lo[1]), (1, hi[1])):
        o = (ax + 1) % 2
        sel = mid & (np.abs(P[:, ax] - c) < 0.15) & (P[:, o] > lo[o] + 0.5) & (P[:, o] < hi[o] - 0.5)
        d.append(P[sel, ax] - c)
    d = np.concatenate(d)
    return float(np.percentile(d, 95) - np.percentile(d, 5)) * 100


def run07(bag, traj, out, *extra):
    r = subprocess.run([sys.executable, os.path.join(ROOT, "map_stages", "09_build_coop_cloud.py"),
                        "--bag", bag, "--traj", traj, "--out", out,
                        "--topic", "/mobile_1/ouster/points", "--voxel", "0.01",
                        "--min-count", "1", "--range-min", "0.3", *extra],
                       capture_output=True, text=True)
    if r.returncode:
        raise AssertionError(r.stdout[-1500:] + r.stderr[-1500:])
    print(r.stdout[-600:])
    return read_pcd(out)


def main():
    rng = np.random.default_rng(1)
    tmp = tempfile.mkdtemp()
    failed = []
    try:
        bag = os.path.join(tmp, "bag")
        stamps = np.arange(0.2, 24.0, 0.1)              # 4 straights and 3 turns
        w = BagWriter(bag)
        for ts in stamps:
            ns = int(round((SC.T0 + ts) * SC.S))
            w.write("/mobile_1/ouster/points", SC.scan(ts, rng), ns)
        w.close()
        traj = os.path.join(tmp, "traj.tum")
        SC.write_traj(traj, stamps)
        on = wall_thickness(run07(bag, traj, os.path.join(tmp, "on.pcd")))
        off = wall_thickness(run07(bag, traj, os.path.join(tmp, "off.pcd"), "--no-deskew"))
        print(f"wall thickness 5-95 %: deskew {on:.1f} cm, no deskew {off:.1f} cm "
              f"(range noise 5 mm sigma -> ~1.6 cm)")
        if not (on < 2.5 and off > 2 * on):
            failed.append(f"deskew {on:.1f} cm vs {off:.1f} cm without")
    except AssertionError as e:
        failed.append(str(e))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("\n".join(failed) or "build cloud ok")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
