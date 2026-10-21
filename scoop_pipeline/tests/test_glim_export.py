#!/usr/bin/env python
"""Self-test for scoop/recording.py export_glim's glim.pose_time_offset:
traj_lidar.txt's poses moved by it in time on their own stamps, traj_imu.txt and
GLIM's dump untouched, the offset in glim.yaml; and mobile_2's setting is
read from configs/recording.yaml.

    python tests/test_glim_export.py
"""
import os
import shutil
import sys
import tempfile
from pathlib import Path

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from scoop import recording                                            # noqa: E402

LINES = ["1800000000.000000000 1.0 2.0 0.5 0 0 0 1",
         "1800000000.100000000 1.1 2.0 0.5 0 0 0.0087265 0.9999619",
         "1800000000.200000000 1.2 2.0 0.5 0 0 0.0174524 0.9998477"]


def main():
    d = Path(tempfile.mkdtemp(prefix="glimexp_"))
    fails = []
    try:
        work = d / "data" / "work" / "20261015" / "mapping" / "mobile_2"
        rec = recording.Recording(d / "data" / "raw" / "20261015" / "mapping" / "mobile_2",
                                  work / "x_retimed", work)
        dump = rec.step("glim")
        dump.mkdir(parents=True)
        (dump / "traj_lidar.txt").write_text("\n".join(LINES) + "\n")
        (dump / "traj_imu.txt").write_text("\n".join(LINES) + "\n")
        settings = recording.settings_for(recording.load_settings(), "mobile_2")
        off = settings["glim"].get("pose_time_offset")
        if off != 0.046:
            fails.append(f"mobile_2's glim.pose_time_offset is {off}, not 0.046")
        out = recording.export_glim(rec, settings, log=lambda *_: None)
        got = [ln.split() for ln in (out / "traj_lidar.txt").read_text().split("\n") if ln]
        # stamps kept; the pose at 0.1 s is the input's at 0.054 s: x 1.054
        if [g[0] for g in got] != [ln.split()[0] for ln in LINES]:
            fails.append(f"stamps changed: {[g[0] for g in got]}")
        elif abs(float(got[1][1]) - 1.054) > 1e-6 or abs(float(got[2][1]) - 1.154) > 1e-6:
            fails.append(f"poses not moved 46 ms in time: {got}")
        elif abs(float(got[0][1]) - 1.0) > 1e-6:
            fails.append(f"first pose not held: {got[0]}")
        if (out / "traj_imu.txt").read_text().split("\n")[0] != LINES[0]:
            fails.append("traj_imu.txt changed")
        if (dump / "traj_lidar.txt").read_text().split("\n")[0] != LINES[0]:
            fails.append("GLIM's dump changed")
        info = yaml.safe_load(open(out / "glim.yaml"))
        if info.get("pose_time_offset_s") != 0.046:
            fails.append(f"glim.yaml: {info.get('pose_time_offset_s')}")
        m1 = recording.settings_for(recording.load_settings(), "mobile_1")
        if m1["glim"].get("pose_time_offset") != 0.046:     # its own, the top level's
            fails.append(f"mobile_1's offset is {m1['glim'].get('pose_time_offset')}, not 0.046")
    finally:
        shutil.rmtree(d, ignore_errors=True)
    for f in fails:
        print("  FAIL  " + f)
    print("  ok    traj_lidar.txt's poses moved +46 ms in time on its own stamps; traj_imu.txt and the dump as they were"
          if not fails else "")
    print("glim export ok" if not fails else "glim export FAILED")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
