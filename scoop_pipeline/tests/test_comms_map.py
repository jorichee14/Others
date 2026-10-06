#!/usr/bin/env python
"""Self-test for analysis/comms/comms_map.py: the comms of tests/test_comms.py's
pass, with positions, drawn on a room's anchored map.

The figures must be written, and each grid cell must hold the median of the
samples that fall in it (the Wi-Fi RSSI grid checked against the table).

    python scoop_pipeline/tests/test_comms_map.py
"""
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import test_pass_tf as T                                            # noqa: E402
import test_comms as C                                              # noqa: E402
from scoop import merge                                             # noqa: E402


def room_pcd(path):
    import open3d as o3d
    rng = np.random.default_rng(0)
    pts = []
    for _ in range(4000):
        z = rng.uniform(-0.2, 2.8)
        s = rng.uniform(-30, 15)
        pts += [(s, -8.0, z), (s, 20.0, z), (-30.0, rng.uniform(-8, 20), z),
                (15.0, rng.uniform(-8, 20), z)]
    pts += [(rng.uniform(-30, 15), rng.uniform(-8, 20), -0.2) for _ in range(20000)]
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.array(pts)))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    o3d.io.write_point_cloud(path, pc)


def coop_case(f):
    """Two robots moving (coop): a Wi-Fi and a ping panel and grid per robot,
    each cell the median of that robot's samples in it; both paths drawn."""
    tmp = tempfile.mkdtemp()
    try:
        proc = os.path.join(tmp, "processed", "20990101", "coop_x")
        wd = os.path.join(proc, "comms", "wifi")
        os.makedirs(wd)
        os.makedirs(os.path.join(proc, "comms", "ntp"))
        rng = np.random.default_rng(1)
        wifi, ping = [], []
        for m, (x0, y0) in (("mobile_1", (0.0, 0.0)), ("mobile_2", (5.0, 3.0))):
            for k in range(200):
                x, y = x0 + 0.04 * k, y0 + 0.02 * k
                wifi.append(dict(machine=m, x=x, y=y, signal_dbm=-40 - 5 * x + rng.normal()))
                ping.append(dict(machine=m, x=x, y=y, icmp_seq=k, reply=int(k % 37 != 5),
                                 rtt_ms=1 + 0.1 * y))
        wifi.append(dict(machine="infra_1", x=-2.0, y=1.0, signal_dbm=-30.0))
        for name, rows in (("wifi.csv", wifi), ("ping.csv", ping)):
            with open(os.path.join(wd, name), "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=sorted({k for r in rows for k in r}))
                w.writeheader()
                w.writerows(rows)
        r = subprocess.run([sys.executable, os.path.join(ROOT, "analysis", "comms", "comms_map.py"),
                            proc, "--grid", "1.0"], capture_output=True, text=True)
        if r.returncode:
            f.append("coop: " + r.stdout[-1500:] + r.stderr[-2500:])
            return
        maps = os.path.join(proc, "comms", "maps")
        for n in ("comms_map.png", "rssi_mobile_1.png", "rssi_mobile_2.png", "ping_mobile_1.png",
                  "ping_mobile_2.png", "rssi_grid_mobile_1.csv", "rssi_grid_mobile_2.csv",
                  "ping_rtt_grid_mobile_2.csv"):
            if not os.path.exists(os.path.join(maps, n)):
                f.append("coop: no " + n)
        for m in ("mobile_1", "mobile_2"):
            g = os.path.join(maps, "rssi_grid_%s.csv" % m)
            if not os.path.exists(g):
                continue
            mine = [x for x in wifi if x["machine"] == m]
            for c in csv.DictReader(open(g)):
                v = [x["signal_dbm"] for x in mine
                     if np.floor(x["x"]) == np.floor(float(c["x"])) and np.floor(x["y"]) == np.floor(float(c["y"]))]
                if len(v) != int(c["n"]) or abs(np.median(v) - float(c["rssi_dbm"])) > 1e-3:
                    f.append("coop %s cell (%s, %s): %s vs %d samples" % (m, c["x"], c["y"], c["rssi_dbm"], len(v)))
        # both paths drawn, the focus's dark and the other's light
        sys.path.insert(0, os.path.join(ROOT, "analysis", "comms"))
        import comms_map
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        path = lambda x0: np.array([np.eye(4) + np.pad([[0, 0, 0, x0 + 0.1 * k]], ((0, 3), (0, 0)))  # noqa: E731
                                    for k in range(50)])
        fig, ax = plt.subplots()
        comms_map.draw_base(ax, None, None, {"mobile_1": path(0.0), "mobile_2": path(9.0)}, [], {},
                            focus="mobile_2")
        cols = sorted(ln.get_color() for ln in ax.lines if len(ln.get_xdata()) > 1)
        if cols != ["0.15", "0.6"]:
            f.append("coop: path colours %s, not one dark and one light" % cols)
        plt.close(fig)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    tmp = tempfile.mkdtemp()
    f = []
    coop_case(f)
    try:
        data = os.path.join(tmp, "data")
        stages = os.path.join(tmp, "stages")
        os.makedirs(stages)
        static_yaml = os.path.join(tmp, "static_tf.yaml")
        T_cam_sensor = T.outputs(data)
        open(static_yaml, "w").write(
            "transforms:\n  - {parent: zed_left_camera_optical_frame, child: os_sensor, "
            "translation: %s, rotation_xyzw: %s}\n" % tuple(map(json.dumps, T.tq(T_cam_sensor)))
            + "  - {parent: arducam_optical_frame, child: infra1_link, translation: %s, "
            "rotation_xyzw: %s}\n" % tuple(map(json.dumps, T.tq(T.T_RADAR))))
        T.configs(stages, data)
        cfg = os.path.join(stages, "pipeline_config_survey_1.json")
        raw = os.path.join(data, "raw", "20260101", "survey_1")
        inputs, _ = T.write_bags(raw)
        tfbag = os.path.join(tmp, "tf_merged")
        T.merged_tree(inputs, cfg, tfbag, static_yaml)
        cb = os.path.join(tmp, "comms")
        C.comms_bag(cb, T.T0, 10.0)
        merged = os.path.join(data, "work", "20260101", "survey_1", "x_merged")
        merge.merge_bags([tfbag, cb], merged, log=lambda *_: None)
        proc = os.path.join(data, "processed", "20260101", "survey_1")
        od = os.path.join(proc, "odometry", "reference_survey_1_20260101")
        shutil.copy(os.path.join(od, "traj_mobile_1_lidar_in_cam.tum"),
                    os.path.join(od, "traj_mobile_1_lidar.tum"))
        room_pcd(os.path.join(data, "processed", "20260101", "mapping_A", "mapping",
                              "map_final_mapping_A_20260101_anchored.pcd"))
        r = subprocess.run([sys.executable, os.path.join(ROOT, "processing", "comms_tables.py"),
                            merged, "--config", cfg], capture_output=True, text=True)
        if r.returncode:
            raise AssertionError(r.stdout[-1500:] + r.stderr[-2000:])
        r = subprocess.run([sys.executable, os.path.join(ROOT, "analysis", "comms", "comms_map.py"),
                            proc, "--config", cfg, "--grid", "1.0"], capture_output=True, text=True)
        print(r.stdout[-2000:])
        if r.returncode:
            raise AssertionError(r.stdout[-1500:] + r.stderr[-2500:])
        maps = os.path.join(proc, "comms", "maps")
        for n in ("comms_map.png", "comms_map.pdf", "ntp.png", "rssi.png", "ping.png", "iperf.png",
                  "rssi_grid.csv", "ping_rtt_grid.csv"):
            if not os.path.exists(os.path.join(maps, n)):
                f.append("no " + n)
        if "background: " + os.path.join(data, "processed", "20260101", "mapping_A") not in r.stdout:
            f.append("background map not the reference pass's")
        wifi = [x for x in csv.DictReader(open(os.path.join(proc, "comms", "wifi", "wifi.csv")))
                if x["machine"] == "mobile_1"]
        cells = list(csv.DictReader(open(os.path.join(maps, "rssi_grid.csv"))))
        for c in cells:
            cx, cy = float(c["x"]), float(c["y"])
            v = [float(x["signal_dbm"]) for x in wifi           # cells are [k, k + 1) m
                 if np.floor(float(x["x"])) == np.floor(cx) and np.floor(float(x["y"])) == np.floor(cy)]
            if len(v) != int(c["n"]) or abs(np.median(v) - float(c["rssi_dbm"])) > 1e-6:
                f.append("cell %s: %s vs %d samples, median %s" % (c, c["rssi_dbm"], len(v),
                                                                     np.median(v) if v else None))
        if not cells:
            f.append("no RSSI cells")
    except AssertionError as e:
        f.append(str(e))
    finally:
        keep = os.environ.get("KEEP_PNG")
        if keep and os.path.exists(os.path.join(tmp, "data")):
            src = os.path.join(tmp, "data", "processed", "20260101", "survey_1", "comms", "maps")
            if os.path.isdir(src):
                shutil.copytree(src, keep, dirs_exist_ok=True)
        shutil.rmtree(tmp, ignore_errors=True)
    print("\n".join(f) or "comms map ok")
    sys.exit(1 if f else 0)


if __name__ == "__main__":
    main()
