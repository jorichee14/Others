#!/usr/bin/env python
"""Self-test for analysis/comms/link_clearance.py on a synthetic building.

A 20 x 12 m room, floor and four walls, split by an inner wall at x = 10
with a 2 m door at y in [5, 7]. infra_1 stands at (14, 6, 1.2); the sniffer
(on mobile_1) drives from (2, 1) to (2, 11) and back for 120 s at 1 m. CSI
RSSI = -30 - 20 log10(d) - 12 dB while the line of sight crosses the inner
wall, plus 1 dB noise. A wall-blocked window must read clearance ~0, a
window through the door open, the blocked windows must lose ~12 dB against
the distance trend, and rho(clearance, residual) must be strongly
positive. Two copies of the pass (two dates) must both be found by --all.
Control frames (0x94) must be ignored. The Wi-Fi status and iperf rows of
mobile_1 must become agent-to-AP links with --ap-file (AP at AP).

    python scoop_pipeline/tests/test_link_clearance.py
"""
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(ROOT, "analysis", "comms", "link_clearance.py")
sys.path.insert(0, os.path.join(ROOT, "analysis", "comms"))
import link_clearance as L                                         # noqa: E402

INFRA = np.array([14.0, 6.0, 1.2])
AP = [18.0, 6.0, 2.0]


def building(res=0.1):
    rng = np.arange
    pts = []
    for x in rng(0, 20 + 1e-9, res):                     # floor
        for y in rng(0, 12 + 1e-9, res):
            pts.append((x, y, 0.0))
    for z in rng(0, 2.8 + 1e-9, res):
        for x in rng(0, 20 + 1e-9, res):
            pts += [(x, 0.0, z), (x, 12.0, z)]
        for y in rng(0, 12 + 1e-9, res):
            pts += [(0.0, y, z), (20.0, y, z)]
            if not (5.0 < y < 7.0) or z > 2.2:           # inner wall with a door
                pts.append((10.0, y, z))
    return np.array(pts)


def write_pcd(path, p):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        fh.write("VERSION .7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\n"
                 "WIDTH %d\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS %d\nDATA ascii\n"
                 % (len(p), len(p)))
        np.savetxt(fh, p, fmt="%.3f")


def blocked(a, b):
    """Does a -> b cross the inner wall outside the door?"""
    if (a[0] - 10) * (b[0] - 10) > 0:
        return False
    s = (10 - a[0]) / (b[0] - a[0])
    y, z = a[1] + s * (b[1] - a[1]), a[2] + s * (b[2] - a[2])
    return not (5.0 < y < 7.0 and z < 2.2)


def write_pass(proc, seed):
    rng = np.random.default_rng(seed)
    write_pcd(os.path.join(proc, "mapping", "map_final_test_anchored.pcd"), building())
    rows = []
    t0 = 1_790_000_000.0
    for i in range(120 * 50):                             # 50 Hz for 120 s
        t = i / 50.0
        u = (t % 60) / 30.0
        y = 1 + 10 * (u if u <= 1 else 2 - u)
        a = np.array([2.0, y, 1.0])
        d = np.linalg.norm(INFRA - a)
        rssi = -30 - 20 * np.log10(d) - (12 if blocked(a, INFRA) else 0) + rng.normal(0, 1)
        fc = 0x94 if i % 5 == 0 else 0x88
        if fc == 0x94:
            rssi = 0.0                                    # garbage that must be filtered
        rows.append({"machine": "mobile_1/sniffer", "peer": "infra_1", "t": t0 + t,
                     "frame_control": fc, "rssi": round(rssi, 2),
                     "csi_power_db": round(rssi + 40, 2),
                     "x": a[0], "y": a[1], "z": a[2],
                     "peer_x": INFRA[0], "peer_y": INFRA[1], "peer_z": INFRA[2]})
    wifi = [{"machine": "mobile_1", "t": r["t"], "signal_dbm": r["rssi"] + 5,
             "x": r["x"], "y": r["y"], "z": r["z"]} for r in rows[1::10]]
    iperf = [{"machine": "mobile_1", "t": r["t"], "reverse": rev, "success": True,
              "bitrate_mbps": 200.0, "x": r["x"], "y": r["y"], "z": r["z"]}
             for i, r in enumerate(rows[::500]) for rev in (True, False)]
    for name, tb in (("wifi.csv", wifi), ("iperf.csv", iperf)):
        os.makedirs(os.path.join(proc, "comms", "wifi"), exist_ok=True)
        with open(os.path.join(proc, "comms", "wifi", name), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(tb[0]))
            w.writeheader()
            w.writerows(tb)
    os.makedirs(os.path.join(proc, "comms", "csi"), exist_ok=True)
    with open(os.path.join(proc, "comms", "csi", "csi.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    tmp = tempfile.mkdtemp(prefix="clearance_")
    try:
        M = L.MapIndex(building())
        c_wall, _ = M.clearance([2, 1, 1], INFRA)
        c_door, _ = M.clearance([2, 6, 1.2], INFRA)
        assert c_wall < 0.2, c_wall
        assert c_door > 1.0, c_door
        print(f"clearance through the wall {c_wall:.2f}, through the door {c_door:.2f}")

        root = os.path.join(tmp, "data", "processed")
        for date, seed in (("20260901", 0), ("20260915", 1)):
            write_pass(os.path.join(root, date, "survey_1"), seed)
        ap_file = os.path.join(tmp, "ap.yaml")
        with open(ap_file, "w") as fh:
            fh.write("default: [%g, %g, %g]\n" % tuple(AP))
        r = subprocess.run([sys.executable, SCRIPT, "--all", root, "--ap-file", ap_file],
                           capture_output=True, text=True)
        print(r.stdout[-1500:], r.stderr[-1500:])
        assert r.returncode == 0
        s = json.load(open(os.path.join(root, "clearance", "clearance_all.json")))
        assert len([p for p in s["passes"].values() if p]) == 2, s["passes"].keys()
        for p, rep in s["passes"].items():
            lk = rep["links"]["infra_1>mobile_1/sniffer"]
            assert lk["rssi_dbm"]["rho_residual"] > 0.5, lk
            assert 9 < lk["rssi_dbm"]["residual_drop_db"] < 15, lk
            assert lk["rssi_dbm"]["clear"]["median"] < -40, "control frames leaked in"
            assert "mobile_1>ap" in rep["links"], rep["links"].keys()
            assert "mobile_1>ap iperf down" in rep["links"], rep["links"].keys()
        pooled = s["pooled"]["csi"]["rssi_dbm"]
        assert pooled["rho_residual"] > 0.5 and 9 < pooled["residual_drop_db"] < 15, pooled
        for f in ("clearance_all.csv", "clearance_all.png"):
            assert os.path.exists(os.path.join(root, "clearance", f)), f
        assert os.path.exists(os.path.join(root, "20260901", "survey_1", "comms",
                                           "clearance", "clearance.png"))
        print("ok: pooled rho(clearance, residual) %.2f, clear-obstructed %.1f dB"
              % (pooled["rho_residual"], pooled["residual_drop_db"]))
    finally:
        shutil.rmtree(tmp)


if __name__ == "__main__":
    main()
