#!/usr/bin/env python
"""Self-test for check.py figures on a made-up mapping pass: GLIM as output
(46 ms late), corrected, scan-to-map, the reference and its BA report,
loop-closure reports, a TSDF and a union map of a room, and the walls
medians. Every figure but noise (no --bag) must be written, as PNG and PDF.

    python tests/test_figures.py
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_distances import room                                       # noqa: E402
from test_pose_time_check import T0, write                            # noqa: E402


def loop(path, mm, noise):
    json.dump({"anchor_board": "anchor", "boards": {"anchor": {"sections": 3, "loop_closure": {
        "from_section": 0, "to_section": 3, "dt_s": 280, "mm": mm, "deg": 1.2,
        "noise_mm": noise, "significant": False}}}}, open(path, "w"))


def main():
    fails = []
    d = tempfile.mkdtemp(prefix="figs_")
    try:
        import open3d as o3d
        work = os.path.join(d, "work", "mobile_2")
        out = os.path.join(d, "out")
        os.makedirs(os.path.join(work, "glim"))
        os.makedirs(out)
        t = T0 + np.arange(0, 300, 0.1)
        write(os.path.join(work, "glim", "traj_lidar.txt"), t, late=0.046)
        write(os.path.join(out, "traj_lidar.txt"), t, noise=0.01, seed=1)
        write(os.path.join(out, "traj_lidar_ba.txt"), t)
        write(os.path.join(out, "traj_lidar_refined.txt"), t, noise=0.005, seed=2)
        open(os.path.join(out, "glim.yaml"), "w").write("pose_time_offset_s: 0.046\n")
        json.dump({"rounds": [{"round": k, "thickness_before_cm": [4, 9 - k, 3], "thickness_after_cm":
                               [3.5, 8 - k, 2.6]} for k in (1, 2, 3)]},
                  open(os.path.join(out, "traj_lidar_ba_report.json"), "w"))
        for name, mm, noise in (("glim_raw_loop_closure.json", 10.8, 6.9), ("glim_loop_closure.json", 10.8, 6.7),
                                ("traj_lidar_refined_loop_closure.json", 11.4, 1.9),
                                ("traj_lidar_ba_loop_closure.json", 13.9, 7.7)):
            loop(os.path.join(out, name), mm, noise)
        for name, noise in (("denoised.pcd", 0.005), ("denoised_union.pcd", 0.02)):
            pc = o3d.geometry.PointCloud()
            pc.points = o3d.utility.Vector3dVector(room(np.random.default_rng(1), s=0.03, noise=noise))
            o3d.io.write_point_cloud(os.path.join(out, name), pc)
        figs = os.path.join(out, "figures")
        os.makedirs(figs)
        for k, v in (("glim", (0.047, 0.066, 0.093, 0.107, 0.121, 0.123)),
                     ("ref", (0.047, 0.067, 0.079, 0.088, 0.090, 0.090))):
            json.dump({"single": v[0], "spans": {"1.0": v[1], "10.0": v[2], "60.0": v[3], "300.0": v[4]},
                       "whole": v[5], "map": 0.12}, open(os.path.join(figs, f"walls_{k}.json"), "w"))
        cfg = os.path.join(d, "cfg.json")
        json.dump({"dataset": {"bag": os.path.join(work, "bag"), "out_dir": out, "machine": "mobile_2",
                               "traj": "traj_lidar_ba.txt"}, "01_build_map": {"fusion": "tsdf"}},
                  open(cfg, "w"))
        r = subprocess.run([sys.executable, os.path.join(ROOT, "check.py"), "figures", cfg,
                            "--samples", "1500"], capture_output=True, text=True)
        print(r.stdout[-2500:] + r.stderr[-2500:])
        for name in ("timing", "ba", "trajectories", "maps", "wall", "loop", "stages"):
            for ext in (".png", ".pdf"):
                if not os.path.isfile(os.path.join(figs, name + ext)):
                    fails.append(f"no {name}{ext}")
        if "noise: skipped" not in r.stdout:
            fails.append("noise not skipped without --bag")
        if r.returncode:
            fails.append("exit %d" % r.returncode)
        keep = os.environ.get("FIGS_KEEP")
        if keep:
            shutil.copytree(figs, keep, dirs_exist_ok=True)
        if not fails:
            print("  ok    7 figures written (PNG + PDF), noise skipped without --bag")
    finally:
        shutil.rmtree(d, ignore_errors=True)
    for f in fails:
        print("  FAIL  " + f)
    print("figures ok" if not fails else "figures FAILED")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
