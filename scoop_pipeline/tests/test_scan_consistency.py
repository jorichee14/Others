#!/usr/bin/env python
"""Self-test for analysis/mapping/scan_consistency.py on synthetic sweeps.

A robot drives a square loop in a 16 x 12 x 3 m room (4 s straight at
0.8 m/s, then a 90 deg turn on the spot at 45 deg/s), scanned by a
48 x 768 sweeping lidar at 10 Hz with point times and 5 mm range noise.
Four cases: the true trajectory, one whose stamps are 50 ms off, one that
drifts in heading, and clouds whose point times run backwards (a broken
deskew); the script must tell them apart.

    python scoop_pipeline/tests/test_scan_consistency.py
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from scoop import rosmsg                                            # noqa: E402
from scoop.bagwrite import BagWriter                                # noqa: E402

T0, S = 1_790_000_000.0, 10**9
DUR, H, W = 40.0, 48, 768
ROOM = np.array([[-8.0, -6.0, 0.0], [8.0, 6.0, 3.0]])
FINE = np.arange(0, DUR + 1, 0.001)


def path():
    """yaw and xy on a 1 ms grid: repeat (4 s straight, 2 s turn of 90 deg)."""
    yaw, xy = np.zeros(len(FINE)), np.zeros((len(FINE), 2))
    y, p = 0.0, np.array([-1.6, -1.6])
    for i, t in enumerate(FINE):
        ph = t % 6.0
        if i:
            if ph < 4.0:
                p = p + 0.8 * 0.001 * np.array([np.cos(y), np.sin(y)])
            else:
                y += np.deg2rad(45) * 0.001
        yaw[i], xy[i] = y, p
    return yaw, xy


YAW, XY = path()


def pose(t):
    """(R, o) of the lidar at times t (s after T0), vectorized."""
    yw = np.interp(t, FINE, YAW)
    c, s = np.cos(yw), np.sin(yw)
    R = np.zeros((len(t), 3, 3))
    R[:, 0, 0], R[:, 0, 1], R[:, 1, 0], R[:, 1, 1], R[:, 2, 2] = c, -s, s, c, 1
    o = np.c_[np.interp(t, FINE, XY[:, 0]), np.interp(t, FINE, XY[:, 1]), np.full(len(t), 0.5)]
    return R, o


CLOUD = np.dtype({"names": ["x", "y", "z", "t"], "formats": ["<f4", "<f4", "<f4", "<u4"],
                  "offsets": [0, 4, 8, 12], "itemsize": 16})


def scan(ts, rng, reverse=False):
    tc = ts + np.linspace(0, 0.0999, W)                     # column times
    Rc, oc = pose(tc)
    az = np.linspace(np.pi, -np.pi, W, endpoint=False)
    el = np.deg2rad(np.linspace(10, -10, H))
    d = np.stack([np.cos(el)[:, None] * np.cos(az), np.cos(el)[:, None] * np.sin(az),
                  np.repeat(np.sin(el)[:, None], W, 1)], -1)          # (H, W, 3) sensor
    dw = np.einsum("wij,hwj->hwi", Rc, d)
    o = np.broadcast_to(oc, (H, W, 3))
    with np.errstate(divide="ignore"):
        tt = np.where(dw > 0, (ROOM[1] - o) / dw, (ROOM[0] - o) / dw)
    rng_ = np.min(np.where(tt > 0, tt, np.inf), axis=-1)
    rng_ = rng_ + rng.normal(0, 0.005, rng_.shape)
    a = np.zeros(H * W, CLOUD)
    p = (d * rng_[..., None]).reshape(-1, 3)                # sensor frame
    a["x"], a["y"], a["z"] = p.T
    dt = (tc - ts)[::-1] if reverse else tc - ts            # reverse: wrong point times
    a["t"] = np.broadcast_to(np.round(dt * S).astype(np.uint32), (H, W)).reshape(-1)
    return rosmsg.pointcloud2(a, int(round((T0 + ts) * S)), "os_lidar", height=H, width=W)


def write_traj(path_, stamps, shift=0.0, drift_dps=0.0):
    R, o = pose(stamps + shift)
    e = np.deg2rad(drift_dps) * stamps                      # heading drift about the origin
    c, s = np.cos(e), np.sin(e)
    D = np.zeros((len(e), 3, 3))
    D[:, 0, 0], D[:, 0, 1], D[:, 1, 0], D[:, 1, 1], D[:, 2, 2] = c, -s, s, c, 1
    R, o = np.einsum("nij,njk->nik", D, R), np.einsum("nij,nj->ni", D, o)
    from scipy.spatial.transform import Rotation
    q = Rotation.from_matrix(R).as_quat()
    np.savetxt(path_, np.c_[T0 + stamps, o, q], fmt="%.9f")


def run(cfg, traj):  # noqa: D103
    out = subprocess.run([sys.executable, os.path.join(ROOT, "analysis", "mapping",
                                                       "scan_consistency.py"), cfg,
                          "--traj", traj, "--anchors", "16", "--patches", "120",
                          "--revisit-s", "20", "--revisit-m", "0.5",
                          "--offsets=-0.05,0,0.05"],
                         capture_output=True, text=True)
    if out.returncode:
        raise AssertionError(out.stdout + out.stderr)
    return out.stdout


def rows(text):
    """section -> {row label: [numbers]}"""
    sec, out = None, {}
    for line in text.splitlines():
        if line[:2] in ("1.", "2.", "3."):
            sec = line[:2]
            out[sec] = {}
        elif sec and line.startswith("   ") and not line.strip().startswith("("):
            parts = line.replace("(", " ").replace(")", " ").split()
            nums = []
            for x in parts[1:] if sec != "2." else parts[1::2]:
                try:
                    nums.append(float(x))
                except ValueError:
                    nums.append(np.nan)
            out[sec][parts[0]] = nums
    return out


def main():
    rng = np.random.default_rng(1)
    tmp = tempfile.mkdtemp()
    bag = os.path.join(tmp, "data", "work", "20260101", "loop", "loop_merged")
    stamps = np.arange(0.2, DUR - 1, 0.1)
    w = BagWriter(bag)
    for ts in stamps:
        log = int(round((T0 + ts) * S))
        w.write("/mobile_1/ouster/points", scan(ts, rng), log)
        w.write("/broken/points", scan(ts, rng, reverse=True), log)
    w.close()

    def config(topic):
        cfg = os.path.join(tmp, f"config_{topic.strip('/').split('/')[0]}.json")
        with open(cfg, "w") as f:
            json.dump({"dataset": {"bag": bag, "traj": "unused", "machine": "mobile_1",
                                   "points_topic": topic},
                       "01_build_map": {"time_tol": 0.04, "lidar_min": 0.3,
                                        "lidar_max": 30.0, "deskew_bins": 100}}, f)
        return cfg
    failed = []
    cases = {"true": dict(), "shifted": dict(shift=0.05), "drift": dict(drift_dps=0.15),
             "broken deskew": dict()}
    for name, kw in cases.items():
        traj = os.path.join(tmp, f"{name.split()[0]}.txt")
        write_traj(traj, stamps, **kw)
        text = run(config("/broken/points" if name == "broken deskew"
                          else "/mobile_1/ouster/points"), traj)
        r = rows(text)
        print(f"===== {name}\n{text}")
        pairs, timing = r["2."], r["3."]
        reading = text.split("reading:")[1]
        try:
            # pairs rows: label -> medians of all, straight, turning, turn/rigid
            # timing rows: shift -> medians of 1s, 10s, revisit
            if name == "true":
                assert all(pairs[k][0] < 3.0 for k in ("0.1s", "1s", "10s", "revisit")), pairs
                assert "no timing, deskew or drift" in reading, reading
            if name == "shifted":
                assert timing["+50"][1] < 0.6 * timing["+0"][1], "the +50 ms shift is not found"
                assert "TIMING" in reading and "+50 ms" in reading, reading
            if name == "drift":
                assert pairs["revisit"][0] > pairs["0.1s"][0] + 3.0, "revisits not thicker"
                assert "DRIFT" in reading and "TIMING" not in reading, reading
            if name == "broken deskew":
                assert "DESKEW" in reading, reading
                assert "TIMING" not in reading and "DRIFT" not in reading, reading
        except (AssertionError, KeyError, IndexError) as e:
            failed.append(f"{name}: {e!r}")
    shutil.rmtree(tmp, ignore_errors=True)
    print("\n".join(failed) or f"all {len(cases)} cases ok")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
