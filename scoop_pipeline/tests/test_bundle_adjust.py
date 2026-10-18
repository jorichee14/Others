#!/usr/bin/env python
"""Self-test for map_stages/02b_bundle_adjust.py: a synthetic 10 x 8 x 3 m
room (walls, floor, ceiling, two pillars, a table) scanned from a robot that
drives a loop in it twice (10 Hz, 80 s). The trajectory given to the BA has
drifted: 1.5 deg of yaw and (8, -5, 3) cm by the end, growing with time
(roll and pitch stay true, as an IMU's gravity keeps them), so
the second loop's walls land beside the first's. The BA must bring the
poses back to the truth and the planes back to the sensor's noise.
Needs numpy, scipy, rosbags, mcap.

    python tests/test_bundle_adjust.py
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
sys.path.insert(0, ROOT)
sys.path.insert(0, STAGES)

import pipeline_common as pc                                           # noqa: E402
from scoop import rosmsg                                               # noqa: E402
from scoop.bagwrite import BagWriter                                   # noqa: E402

TOPIC = "/mobile_2/ouster/points"
T0 = 1_800_000_000
DUR = 80.0


def boxes():
    """Axis-aligned boxes (lo, hi) whose faces are the scene's surfaces;
    the room is the first, seen from inside."""
    return [((0, 0, 0), (10, 8, 3)), ((3, 3, 0), (3.4, 3.4, 3)),
            ((7, 4.5, 0), (7.4, 4.9, 3)), ((5, 1.5, 0), (6, 2.5, 0.8))]


def sample(n, rng):
    faces = []
    for lo, hi in boxes():
        lo, hi = np.array(lo, float), np.array(hi, float)
        for ax in range(3):
            for side in (lo[ax], hi[ax]):
                o = [i for i in range(3) if i != ax]
                faces.append((ax, side, lo[o], hi[o], np.prod(hi[o] - lo[o])))
    area = np.array([f[4] for f in faces])
    which = rng.choice(len(faces), n, p=area / area.sum())
    P = np.empty((n, 3))
    for i, (ax, side, l, h, _) in enumerate(faces):
        m = which == i
        o = [k for k in range(3) if k != ax]
        P[m, ax] = side
        P[np.ix_(m, o)] = l + rng.random((m.sum(), 2)) * (h - l)
    return P


def truth(s):
    """A rounded rectangle loop, 40 s per lap."""
    u = (s % 40.0) / 40.0 * 2 * np.pi
    x, y = 5 + 3.2 * np.cos(u), 4 + 2.5 * np.sin(u)
    yaw = np.arctan2(2.5 * np.cos(u), -3.2 * np.sin(u))
    return Rotation.from_euler("z", yaw).as_matrix(), np.array([x, y, 0.6])


def drift(s):
    f = s / DUR
    return (Rotation.from_euler("z", 1.5 * f, degrees=True).as_matrix(),
            np.array([0.08, -0.05, 0.03]) * f)


def main():
    d = tempfile.mkdtemp(prefix="ba_")
    fails = []
    try:
        rng = np.random.default_rng(0)
        bag = os.path.join(d, "bag")
        traj = os.path.join(d, "traj_lidar_refined.txt")
        true_t = []
        with BagWriter(bag) as w, open(traj, "w") as ft:
            for k in range(int(DUR * 10)):
                s = k / 10
                R, t = truth(s)
                P = sample(5000, rng)
                ps = (P - t) @ R
                r = np.linalg.norm(ps, axis=1, keepdims=True)
                ps += ps / r * rng.normal(0, 0.01, (len(ps), 1))
                pts = np.zeros(len(ps), [("x", "<f4"), ("y", "<f4"), ("z", "<f4")])
                pts["x"], pts["y"], pts["z"] = ps.T
                ns = T0 * 10**9 + k * 10**8
                w.write(TOPIC, rosmsg.pointcloud2(pts, ns, "os_lidar_mobile2"), ns)
                Dr, Dt = drift(s)
                Ri, ti = Dr @ R, Dr @ t + Dt
                q = Rotation.from_matrix(Ri).as_quat()
                ft.write("%.9f %.6f %.6f %.6f %.9f %.9f %.9f %.9f\n" % (ns * 1e-9, *ti, *q))
                true_t.append(t)
        cfg = pc.load_config(os.path.join(STAGES, "pipeline_config_mapping_20261015.json"))
        cfg.pop("extends", None)
        cfg["dataset"].update(bag=bag, traj=traj, points_topic=TOPIC, out_dir=os.path.join(d, "out"))
        cfg["02b_bundle_adjust"] = {"input": traj, "rounds": 2}
        cp = os.path.join(d, "cfg.json")
        json.dump(cfg, open(cp, "w"))
        p = subprocess.run([sys.executable, os.path.join(STAGES, "02b_bundle_adjust.py"), cp],
                           capture_output=True, text=True, cwd=STAGES)
        log = p.stdout + p.stderr
        out = os.path.join(d, "out", "traj_lidar_ba.txt")
        if os.environ.get("BA_LOG"):
            print(log)
        if p.returncode != 0 or not os.path.exists(out):
            raise AssertionError(log[-4000:])
        _, Tb = pc.load_traj(out)
        _, Ti = pc.load_traj(traj)
        tt = np.array(true_t)
        e_in = np.linalg.norm(Ti[:, :3, 3] - tt, axis=1) * 100
        e_ba = np.linalg.norm(Tb[:, :3, 3] - tt, axis=1) * 100
        line = (f"position error vs truth: input median {np.median(e_in):.1f} / max {e_in.max():.1f} cm"
                f"  ->  BA median {np.median(e_ba):.1f} / max {e_ba.max():.1f} cm")
        th = [ln for ln in log.splitlines() if "plane thickness" in ln]
        if not (np.median(e_ba) < 1.0 and e_ba.max() < 2.5 and np.median(e_in) > 3.0):
            fails.append(line + "\n" + log[-3000:])
        else:
            print("  ok    " + line)
            for ln in th:
                print("        " + ln.strip())
        # the timing check: a drifted input with no time offset is on time
        rep = json.load(open(os.path.join(d, "out", "traj_lidar_ba_report.json")))
        tm = rep.get("timing", {})
        line = f"timing check: {tm.get('dt_ms', float('nan')):+.1f} ms, r {tm.get('r_timing', float('nan')):.2f}"
        if tm.get("on_time") is not True:
            fails.append(line + " flagged, the input has no time offset: " + json.dumps(tm))
        else:
            print("  ok    " + line + ", on time")
        # the cache: a second run reads no bag
        p2 = subprocess.run([sys.executable, os.path.join(STAGES, "02b_bundle_adjust.py"), cp,
                             "--rounds", "1"], capture_output=True, text=True, cwd=STAGES)
        if p2.returncode != 0 or "keyframes from" not in p2.stdout:
            fails.append("second run did not use the keyframe cache:\n" + p2.stdout[-2000:] + p2.stderr[-2000:])
        else:
            print("  ok    second run took the keyframes from the cache")
    finally:
        shutil.rmtree(d, ignore_errors=True)
    # the timing check flags an input whose poses are 30 ms late
    import importlib.util
    spec = importlib.util.spec_from_file_location("ba", os.path.join(STAGES, "02b_bundle_adjust.py"))
    ba = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ba)
    t = np.arange(0, 300, 0.1)
    yaw = lambda tt: 0.8 * np.sin(tt / 3.0) + 0.3 * np.sin(tt / 1.3)
    poses = lambda ps: np.array([np.block([[Rotation.from_euler("z", a).as_matrix(), np.zeros((3, 1))],
                                           [np.zeros((1, 3)), np.ones((1, 1))]]) for a in ps])
    tm = ba.timing_check(os.path.join(tempfile.gettempdir(), "none", "traj.txt"),
                         t, poses(yaw(t + 0.03)), poses(yaw(t)))
    line = f"timing check, input 30 ms late: {tm.get('dt_ms', float('nan')):+.1f} ms, r {tm.get('r_timing', 0):.2f}"
    if tm.get("on_time") is not False or abs(tm["dt_ms"] - 30) > 3:
        fails.append(line + " not flagged as 30 ms")
    else:
        print("  ok    " + line + ", flagged")
    for f in fails:
        print("  FAIL  " + f)
    print("bundle adjust ok" if not fails else "bundle adjust FAILED")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
