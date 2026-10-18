#!/usr/bin/env python
"""Self-test for check.py trajs (analysis/trajs.py) on three synthetic
trajectories of a 200 s loop: 'glim' the truth, '02' the truth moved 5 cm
(a global offset: gone after the rigid fit, nothing in the relative
motion), 'ba' the truth with a slow yaw drift (grows with the time span).

    python tests/test_compare_trajs.py
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
sys.path.insert(0, os.path.join(ROOT, "map_stages"))
import pipeline_common as pc                                           # noqa: E402


def write(path, t, R, p):
    q = Rotation.from_matrix(R).as_quat()
    with open(path, "w") as fh:
        for row in zip(t, p, q):
            fh.write("%.9f %.6f %.6f %.6f %.9f %.9f %.9f %.9f\n" % (row[0], *row[1], *row[2]))


def turning_check(fails):
    """A trajectory stamped 30 ms late on a path that turns back and forth:
    its heading error follows the yaw rate, and the time offset is found."""
    sys.path.insert(0, os.path.join(ROOT, "analysis"))
    import trajs as ct
    t = np.arange(0, 300, 0.1)
    psi = 0.8 * np.sin(2 * np.pi * t / 15) + 0.4 * np.sin(2 * np.pi * t / 6.3)

    def poses(ps):
        T = np.tile(np.eye(4), (len(t), 1, 1))
        T[:, :3, :3] = Rotation.from_euler("z", ps[:, None]).as_matrix()
        T[:, 0, 3] = 0.4 * t
        return T
    late = 0.8 * np.sin(2 * np.pi * (t + 0.03) / 15) + 0.4 * np.sin(2 * np.pi * (t + 0.03) / 6.3)
    h = ct.heading_vs_turning(t, poses(late), t, poses(psi))
    h0 = ct.heading_vs_turning(t, poses(psi + 0.002 * np.sin(t / 7.0)), t, poses(psi))
    ok = h["r_timing"] > 0.9 and abs(h["dt_ms"] - 30) < 3 and abs(h0.get("r_timing", 0)) < 0.5
    print(f"  {'ok  ' if ok else 'FAIL'}  30 ms late: r {h['r_timing']:.2f}, offset {h['dt_ms']:.1f} ms; "
          f"error not tied to turning: r {h0.get('r_timing', 0):.2f}")
    if not ok:
        fails.append("heading vs turning")


def main():
    d = tempfile.mkdtemp(prefix="cmp_")
    fails = []
    turning_check(fails)
    try:
        t = 1.8e9 + np.arange(0, 200, 0.1)
        u = (t - t[0]) / 100 * np.pi
        p = np.c_[5 * np.cos(u), 5 * np.sin(u), np.full_like(u, 0.5)]
        R = Rotation.from_euler("z", (u + np.pi / 2)[:, None]).as_matrix()
        write(os.path.join(d, "glim.txt"), t, R, p)
        write(os.path.join(d, "o2.txt"), t, R, p + [0.05, 0, 0])
        yaw = np.radians(1.0) * (t - t[0]) / 200            # 1 deg over the run
        D = Rotation.from_euler("z", yaw[:, None]).as_matrix()
        write(os.path.join(d, "ba.txt"), t, D @ R, np.einsum("nij,nj->ni", D, p))
        cfg = pc.load_config(os.path.join(ROOT, "map_stages", "pipeline_config_mapping_20261015.json"))
        cfg.pop("extends", None)
        cfg["dataset"].update(bag=os.path.join(d, "bag"), out_dir=os.path.join(d, "out"))
        cp = os.path.join(d, "cfg.json")
        json.dump(cfg, open(cp, "w"))
        out = os.path.join(d, "r.json")
        r = subprocess.run([sys.executable, os.path.join(ROOT, "check.py"), "trajs",
                            cp, "--traj", f"glim={d}/glim.txt", "--traj", f"02={d}/o2.txt",
                            "--traj", f"ba={d}/ba.txt", "--out", out, "--plot", f"{d}/p.png",
                            "--ref", "glim"],
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise AssertionError(r.stdout + r.stderr)
        rep = {x["pair"]: x for x in json.load(open(out))["pairs"]}
        off = rep["glim vs 02"]
        if not (abs(off["position_cm"]["median"] - 5) < 0.1 and off["after_rigid_fit_cm"]["max"] < 0.1
                and off["relative"]["100s"]["translation_cm"]["max"] < 0.1):
            fails.append(f"offset pair: {off}")
        dr = rep["glim vs ba"]["relative"]
        r1, r100 = dr["1s"]["translation_cm"]["median"], dr["100s"]["translation_cm"]["median"]
        if not (r1 < 0.2 and r100 > 5 * r1 and r100 > 1.0):
            fails.append(f"drift pair relative 1 s {r1} / 100 s {r100}")
        if "ATE cm" not in r.stdout or "02 & 0.0 & 0.00" not in r.stdout:
            fails.append("no --ref table, or the offset trajectory's ATE is not 0 after the fit")
        if not os.path.isfile(f"{d}/p.png") and "matplotlib is not installed" not in r.stdout:
            fails.append("no plot")
        if not fails:
            print(f"  ok    offset: 5.0 cm, 0 after the rigid fit, 0 relative; drift: relative "
                  f"{r1:.2f} cm at 1 s -> {r100:.2f} cm at 100 s")
            print("\n".join("        " + ln for ln in r.stdout.splitlines()[-14:]))
    finally:
        shutil.rmtree(d, ignore_errors=True)
    for f in fails:
        print("  FAIL  " + f)
    print("compare trajs ok" if not fails else "compare trajs FAILED")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
