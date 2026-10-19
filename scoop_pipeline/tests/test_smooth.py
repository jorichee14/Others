#!/usr/bin/env python
"""Self-test for analysis/smooth.py (check.py smooth): a smooth synthetic path
(turns, speed changes and a 20 s stop, 10 Hz) and the same path with 3 mm /
0.1 deg of white jitter. The smooth one must read near zero, the jittered one
about 0.72 / 0.89 x its jitter (5 / 11-pose windows), still and moving alike,
and rougher in jerk and angular acceleration.

    python scoop_pipeline/tests/test_smooth.py
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
from scipy.spatial.transform import Rotation

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RNG = np.random.default_rng(7)
SIG_P, SIG_R = 0.003, 0.1          # m, deg


def path():
    t = np.arange(0, 200, 0.1)
    s = np.clip(t - 20, 0, None)                          # stands still for 20 s
    yaw = 0.6 * np.sin(s / 15.0) + 0.02 * s
    v = 0.4 + 0.1 * np.sin(s / 7.0)
    v[t < 20] = 0
    x = np.cumsum(v * np.cos(yaw)) * 0.1
    y = np.cumsum(v * np.sin(yaw)) * 0.1
    T = np.tile(np.eye(4), (len(t), 1, 1))
    T[:, :3, :3] = Rotation.from_euler("z", yaw[:, None]).as_matrix()
    T[:, 0, 3], T[:, 1, 3], T[:, 2, 3] = x, y, 0.4
    return t, T


def write(fn, t, T):
    q = Rotation.from_matrix(T[:, :3, :3]).as_quat()
    np.savetxt(fn, np.column_stack([t, T[:, :3, 3], q]), fmt="%.9f")


def main(show=os.environ.get("SHOW")):
    tmp = tempfile.mkdtemp()
    f = []
    try:
        t, T = path()
        J = T.copy()
        J[:, :3, 3] += RNG.normal(0, SIG_P / np.sqrt(3), (len(t), 3))
        J[:, :3, :3] = T[:, :3, :3] @ Rotation.from_rotvec(
            RNG.normal(0, np.radians(SIG_R) / np.sqrt(3), (len(t), 3))).as_matrix()
        write(os.path.join(tmp, "smooth.txt"), t, T)
        write(os.path.join(tmp, "jitter.txt"), t, J)
        cfg = os.path.join(tmp, "pipeline_config.json")
        json.dump({"dataset": {"bag": os.path.join(tmp, "none.mcap"), "out_dir": tmp,
                               "traj": os.path.join(tmp, "smooth.txt")}}, open(cfg, "w"))
        out = os.path.join(tmp, "s.json")
        r = subprocess.run([sys.executable, os.path.join(ROOT, "check.py"), "smooth", cfg,
                            "--traj", "smooth=" + os.path.join(tmp, "smooth.txt"),
                            "--traj", "jitter=" + os.path.join(tmp, "jitter.txt"), "--out", out],
                           capture_output=True, text=True)
        if show:
            print(r.stdout)
        if r.returncode:
            f.append("check.py smooth failed:\n" + r.stdout[-2000:] + r.stderr[-3000:])
        else:
            res = json.load(open(out))["results"]
            for w, k in (("0.5s", 0.72), ("1s", 0.89)):
                s, j = res["smooth"][w], res["jitter"][w]
                if not s["pos_mm"] < 0.25 * k * SIG_P * 1e3 or not s["rot_deg"] < 0.1 * k * SIG_R:
                    f.append(f"smooth path reads {s} at {w}")
                for key, sig in (("pos_mm", SIG_P * 1e3), ("pos_still_mm", SIG_P * 1e3),
                                 ("rot_deg", SIG_R), ("rot_still_deg", SIG_R)):
                    if not 0.8 * k * sig < j[key] < 1.2 * k * sig:
                        f.append(f"jitter {key} at {w}: {j[key]:.3f}, want {k * sig:.3f}")
            if not res["jitter"]["jerk_ms3"] > 3 * res["smooth"]["jerk_ms3"]:
                f.append("jerk does not tell jitter: %s" % res)
            if not 5 < res["smooth"]["still_pct"] < 15:
                f.append("still %.0f %%, want 10 %%" % res["smooth"]["still_pct"])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("\n".join(f) or "smooth ok")
    sys.exit(1 if f else 0)


if __name__ == "__main__":
    main()
