#!/usr/bin/env python3
"""How good a robot's GLIM run is: its trajectory and its loop closure on the boards.

    python3 check.py glim <raw machine folder> [--config map_stages/pipeline_config_<pass>.json]

    e.g. python3 check.py glim ~/workspaces/isaac_ros-dev/data/raw/20261015/mapping/mobile_2

The GLIM step of process_recording.py does both of these itself when GLIM
finishes; this runs them again on a GLIM run that is already there:

1. the export to processed/<date>/<pass>/odometry/<machine>/ (traj_lidar.txt,
   traj_imu.txt, glim_config/, glim.yaml): poses, time span, rate, path length;
2. the loop closure: stage 05 in its read-only check mode on GLIM's own
   trajectory (map_stages/05_anchor.py --traj --check). For every board seen
   at the start and again later -- start and end on the same board -- where
   GLIM puts it the first time and the last time, in mm / deg, against the
   detections' own noise (DRIFT or not significant), and the cross-checks of
   boards seen in one image. Report: odometry/<machine>/glim_loop_closure.json,
   summary in glim.yaml. Nothing else in processed/ is written.

The pass's map-stage config is the one in map_stages/ whose dataset is this
pass (--config picks another); its boards, camera and calibration are used.
"""
import argparse
import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.simplefilter("ignore", FutureWarning)

from scoop import recording                                         # noqa: E402
from scoop.bag import BagError                                      # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("raw_dir", help="the robot's raw folder, e.g. .../raw/<date>/<pass>/mobile_2")
    ap.add_argument("--config", default=None, help="map-stage config (default: this pass's)")
    ap.add_argument("--work-root", default=None)
    a = ap.parse_args()
    try:
        s = recording.settings_for(recording.load_settings(),
                                   os.path.basename(os.path.normpath(a.raw_dir)))
        rec = recording.find_recording(a.raw_dir, a.work_root,
                                       s["ouster"].get("packets_ns", "/ouster"),
                                       recording.zed_bags(s["zed"]))
        if not rec.done("glim"):
            sys.exit(f"no GLIM run in {rec.step('glim')}; run process_recording.py first")
        recording.export_glim(rec, s)
        lc = recording.glim_loop_check(rec, config=a.config)
    except (BagError, ValueError) as e:
        sys.exit(f"error: {e}")
    if lc is None:
        print("loop closure: not measured (see above)")


if __name__ == "__main__":
    main()
