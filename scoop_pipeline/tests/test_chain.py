#!/usr/bin/env python
"""Self-test for analysis/chain.py (check.py chain) on a made-up mapping pass:
GLIM's dump, its export moved +46 ms, BA's output and report, the keyframe
cache, the time check, the map / union / seed clouds and 02b, all in order,
must read consistent; then three faults, each alone, must be caught: the
export moved twice (92 ms), a denoised map older than the reference (01
would resume from it), and a cache made from another input.

    python scoop_pipeline/tests/test_chain.py
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OFF = 0.046


def traj(t):
    yaw = 0.3 * np.sin(t / 9.0)
    p = np.stack([0.4 * t * np.cos(0.1), 3 * np.sin(t / 12.0), 0.4 + 0 * t], 1)
    q = Rotation.from_euler("z", yaw[:, None]).as_quat()
    return np.column_stack([t + 1.7e9, p, q])


def write(fn, a):
    os.makedirs(os.path.dirname(fn), exist_ok=True)
    np.savetxt(fn, a, fmt="%.6f")


def shifted(a, dt):
    t = a[:, 0] - a[0, 0]
    q = np.clip(t - dt, t[0], t[-1])
    rot = Slerp(t, Rotation.from_quat(a[:, 4:8]))(q).as_quat()
    pos = np.stack([np.interp(q, t, a[:, k]) for k in (1, 2, 3)], 1)
    return np.column_stack([a[:, 0], pos, rot])


def touch(fn, t, text="x"):
    os.makedirs(os.path.dirname(fn), exist_ok=True)
    if not os.path.exists(fn):
        open(fn, "w").write(text)
    os.utime(fn, (t, t))


def build(tmp, fault=None):
    base = os.path.join(tmp, "processed", "20261015", "mapping")
    odo = os.path.join(base, "odometry", "mobile_2")
    mp = os.path.join(base, "mapping")
    dump = os.path.join(tmp, "work", "mobile_2", "glim")
    t0 = time.time() - 10000
    a = traj(np.arange(0, 300, 0.1))
    write(os.path.join(dump, "traj_lidar.txt"), a)
    write(os.path.join(odo, "traj_lidar.txt"), shifted(a, 2 * OFF if fault == "twice" else OFF))
    open(os.path.join(odo, "glim.yaml"), "w").write(
        f"bag: x\ndump: {dump}\npose_time_offset_s: {OFF}\n")
    os.makedirs(os.path.join(odo, "glim_config"), exist_ok=True)
    open(os.path.join(odo, "glim_config", "config_ros.json"), "w").write(
        '{\n  // comment\n  "imu_time_offset": 0.0,\n  "points_time_offset": 0.0\n}\n')
    seed, ref = os.path.join(odo, "traj_lidar.txt"), os.path.join(odo, "traj_lidar_ba_glim.txt")
    os.utime(seed, (t0, t0))
    write(ref, a)
    os.utime(ref, (t0 + 100, t0 + 100))
    json.dump({"input": seed, "output": ref, "rounds": [{"planes": 9000, "moved_median_cm": 1.2,
                                                          "moved_p95_cm": 4.0}]},
              open(os.path.splitext(ref)[0] + "_report.json", "w"))
    key = {"input": seed if fault != "cache" else os.path.join(odo, "other.txt"), "mtime": t0}
    kdir = os.path.join(base, "_old", "odometry", "mobile_2")      # where clean_pass.py moves it
    os.makedirs(kdir, exist_ok=True)
    np.savez(os.path.join(kdir, "traj_lidar_ba_glim_keyframes.npz"),
             points=np.zeros((1, 3)), sizes=np.array([1]), key=json.dumps(key))
    touch(os.path.join(odo, "pose_time_check.json"), t0 + 200, json.dumps({"dt_ms": 4.0}))
    tag = "mapping_20261015"
    for i, b in enumerate(("merged", "static", "denoised", "colored", "map_final")):
        tt = t0 + 300 + i
        if fault == "stale" and b == "denoised":
            tt = t0 + 50                                  # before the reference
        touch(os.path.join(mp, f"{b}_{tag}.pcd"), tt)
    for i, b in enumerate(("merged", "static", "denoised")):
        touch(os.path.join(mp, f"{b}_union_{tag}.pcd"), t0 + 400 + i)
        touch(os.path.join(mp, f"{b}_seed_{tag}.pcd"), t0 + 10 + i)
    touch(os.path.join(odo, "traj_lidar_refined.txt"), t0 + 20)
    touch(os.path.join(odo, "traj_lidar_ba_glim_v1.txt"), t0 - 500)
    touch(os.path.join(base, f"mapping_report_{tag}.md"), t0 + 500)
    cfg = os.path.join(tmp, "pipeline_config.json")
    json.dump({"dataset": {"bag": os.path.join(tmp, "work", "x.mcap"), "processed": base,
                           "traj": "traj_lidar_ba_glim.txt", "machine": "mobile_2"},
               "01_build_map": {"fusion": "tsdf"},
               "02_bundle_adjust": {"input": "traj_lidar.txt", "output": "traj_lidar_ba_glim.txt"}},
              open(cfg, "w"))
    return cfg


def run(cfg):
    return subprocess.run([sys.executable, os.path.join(ROOT, "check.py"), "chain", cfg],
                          capture_output=True, text=True)


def main():
    f = []
    for fault, want in ((None, None), ("twice", "applied twice"), ("stale", "OLDER than"),
                        ("cache", "not the seed as it is now")):
        tmp = tempfile.mkdtemp()
        try:
            r = run(build(tmp, fault))
            out = r.stdout + r.stderr
            if os.environ.get("SHOW"):
                print(f"--- {fault}\n{out}")
            if fault is None:
                if r.returncode or "the chain is consistent" not in out:
                    f.append("clean pass not consistent:\n" + out[-3000:])
                elif "traj_lidar_ba_glim_v1.txt" not in out.split("not in the chain")[-1]:
                    f.append("leftover v1 not listed:\n" + out[-1500:])
            elif r.returncode != 1 or want not in out:
                f.append(f"fault {fault} not caught (want '{want}'):\n" + out[-3000:])
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    print("\n".join(f) or "chain ok")
    sys.exit(1 if f else 0)


if __name__ == "__main__":
    main()
