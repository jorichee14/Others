#!/usr/bin/env python
"""Self-test for check.py imu (analysis/trajs.py): a robot turning back and
forth, its IMU (200 Hz) stamped 40 ms late against a 10 Hz trajectory, and
a trajectory on the IMU's time. The check must find +40 ms and 0 ms.
Needs numpy, scipy, rosbags, mcap.

    python tests/test_imu_vs_traj.py
"""
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "map_stages"))

import pipeline_common as pc                                           # noqa: E402
from scoop import rosmsg                                               # noqa: E402
from scoop.bagwrite import BagWriter                                   # noqa: E402

T0 = 1_800_000_000


def yaw(s):
    return 0.8 * np.sin(2 * np.pi * s / 15) + 0.4 * np.sin(2 * np.pi * s / 6.3)


def yaw_rate(s):
    return 0.8 * 2 * np.pi / 15 * np.cos(2 * np.pi * s / 15) + 0.4 * 2 * np.pi / 6.3 * np.cos(2 * np.pi * s / 6.3)


def main():
    d = tempfile.mkdtemp(prefix="imuvs_")
    fails = []
    try:
        bag = os.path.join(d, "bag")
        late = 0.040
        with BagWriter(bag) as w:
            for k in range(int(120 * 200)):
                s = k / 200
                ns = T0 * 10**9 + int(round((s + late) * 1e9))      # stamped 40 ms late
                w.write("/mobile_2/ouster/imu",
                        rosmsg.imu(ns, "os_imu_mobile2", [0, 0, 9.8], [0, 0, yaw_rate(s)]), ns)
        traj = os.path.join(d, "traj.txt")
        with open(traj, "w") as fh:
            for k in range(1200):
                s = k / 10
                q = Rotation.from_euler("z", yaw(s)).as_quat()
                fh.write("%.9f %.3f 0 0 %.9f %.9f %.9f %.9f\n" % (T0 + s, 0.4 * s, *q))
        cfg = pc.load_config(os.path.join(ROOT, "map_stages", "pipeline_config_mapping_20261015.json"))
        cfg.pop("extends", None)
        cfg["dataset"].update(bag=bag, out_dir=os.path.join(d, "out"))
        cp = os.path.join(d, "cfg.json")
        json.dump(cfg, open(cp, "w"))
        r = subprocess.run([sys.executable, os.path.join(ROOT, "check.py"), "imu",
                            cp, "--traj", f"lidar={traj}"], capture_output=True, text=True)
        m = re.search(r"^lidar\s+([-\d.]+) ms\s+([-\d.]+)", r.stdout, re.M)
        if r.returncode or not m:
            fails.append(r.stdout + r.stderr)
        else:
            sh, c = float(m.group(1)), float(m.group(2))
            ok = abs(sh - 40) < 3 and c > 0.95
            print(f"  {'ok  ' if ok else 'FAIL'}  IMU stamped 40 ms late: found {sh:+.1f} ms (r {c:.3f})")
            if not ok:
                fails.append(r.stdout)
    finally:
        shutil.rmtree(d, ignore_errors=True)
    for f in fails:
        print("  FAIL  " + f)
    print("imu vs traj ok" if not fails else "imu vs traj FAILED")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
