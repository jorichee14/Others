#!/usr/bin/env python
"""Self-test for map_stages/pose_time_check.py and mapping_report.py on
synthetic trajectories of a robot turning back and forth for 300 s:

  reference   the truth (02's bundle-adjusted trajectory)
  GLIM        the truth 46 ms late (its own dump, as GLIM computes it)
  corrected   GLIM's poses moved +46 ms (traj_lidar.txt, glim.yaml says so)
  late        the corrected one 30 ms late again (an offset that changed)
  scan-to-map the truth with 1 cm of noise

The check must find the corrected trajectory on time and flag the late one
as 30 ms off with 0.076 s to set; the report must list GLIM, + correction,
+ scan-to-map and + BA with GLIM's 46 ms in its pose-time column.

    python tests/test_pose_time_check.py
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STAGES = os.path.join(ROOT, "map_stages")
sys.path.insert(0, STAGES)

T0 = 1.8e9


def yaw(t):
    return 0.8 * np.sin(2 * np.pi * t / 15) + 0.4 * np.sin(2 * np.pi * t / 6.3)


def write(path, t, late=0.0, noise=0.0, seed=0):
    """The truth at t + late (a pose for t that is the robot's at t + late)."""
    rng = np.random.default_rng(seed)
    s = t - T0 + late
    q = Rotation.from_euler("z", yaw(s)[:, None]).as_quat()
    p = np.stack([0.4 * s, np.sin(s / 20), np.zeros_like(s)], 1) + rng.normal(0, noise, (len(t), 3))
    with open(path, "w") as fh:
        for row in zip(t, p, q):
            fh.write("%.9f %.6f %.6f %.6f %.9f %.9f %.9f %.9f\n" % (row[0], *row[1], *row[2]))


def main():
    fails = []
    d = tempfile.mkdtemp(prefix="ptc_")
    try:
        work = os.path.join(d, "work", "mobile_2")
        out = os.path.join(d, "out")
        os.makedirs(os.path.join(work, "glim"))
        os.makedirs(out)
        t = T0 + np.arange(0, 300, 0.1)
        write(os.path.join(work, "glim", "traj_lidar.txt"), t, late=0.046)
        write(os.path.join(out, "traj_lidar.txt"), t)
        write(os.path.join(out, "traj_late.txt"), t, late=0.030)
        write(os.path.join(out, "traj_lidar_ba.txt"), t)
        write(os.path.join(out, "traj_lidar_refined.txt"), t, noise=0.01)
        with open(os.path.join(out, "glim.yaml"), "w") as fh:
            fh.write("pose_time_offset_s: 0.046\n")
        cfg = os.path.join(d, "cfg.json")
        json.dump({"dataset": {"bag": os.path.join(work, "bag"), "out_dir": out,
                               "machine": "mobile_2", "traj": "traj_lidar_ba.txt"},
                   "01_build_map": {"fusion": "tsdf"}}, open(cfg, "w"))
        run = lambda *a: subprocess.run([sys.executable, *a], capture_output=True,  # noqa: E731
                                        text=True, cwd=STAGES)

        p = run(os.path.join(STAGES, "pose_time_check.py"), cfg)
        rep = os.path.join(out, "pose_time_check.json")
        h = json.load(open(rep)) if os.path.isfile(rep) else {}
        line = f"corrected GLIM: {h.get('dt_ms', float('nan')):+.1f} ms, r {h.get('r_timing', float('nan')):.2f}"
        if p.returncode or h.get("on_time") is not True or abs(h["dt_ms"]) > 3:
            fails.append(line + " not on time\n" + p.stdout[-1500:] + p.stderr[-1500:])
        else:
            print("  ok    " + line + ", on time")

        p = run(os.path.join(STAGES, "pose_time_check.py"), cfg, "--traj",
                os.path.join(out, "traj_late.txt"), "--out", os.path.join(out, "late.json"))
        h = json.load(open(os.path.join(out, "late.json"))) if not p.returncode else {}
        line = (f"30 ms late again: {h.get('dt_ms', float('nan')):+.1f} ms, "
                f"set {h.get('offset_to_set_s')}")
        if p.returncode or h.get("on_time") is not False or abs(h["dt_ms"] - 30) > 3 \
                or h["offset_to_set_s"] != 0.076 or "OFF by +30 ms" not in p.stdout:
            fails.append(line + " not flagged\n" + p.stdout[-1500:] + p.stderr[-1500:])
        else:
            print("  ok    " + line + ", flagged")

        p = run(os.path.join(STAGES, "mapping_report.py"), cfg, "--no-loop")
        md = os.path.join(out, "mapping_report.md")
        text = open(md).read() if os.path.isfile(md) else ""
        rows = [ln for ln in text.splitlines() if ln.startswith("| GLIM") or ln.startswith("| **GLIM")]
        glim = next((ln for ln in rows if ln.startswith("| GLIM |")), "")
        dt = float(glim.split("|")[6].split()[0]) if glim else float("nan")
        if p.returncode or len(rows) != 4 or abs(dt - 46) > 4 or "TSDF" not in text:
            fails.append(f"report: {len(rows)} trajectory rows, GLIM {dt} ms\n"
                         + p.stdout[-2500:] + p.stderr[-1500:])
        else:
            print(f"  ok    report: 4 trajectories, GLIM's pose time {dt:+.1f} ms, maps listed")
            for ln in rows:
                print("        " + ln)
    finally:
        shutil.rmtree(d, ignore_errors=True)
    for f in fails:
        print("  FAIL  " + f)
    print("pose time check ok" if not fails else "pose time check FAILED")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
