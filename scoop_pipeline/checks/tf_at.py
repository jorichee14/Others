#!/usr/bin/env python3
"""Where the frames are in map, read back from a bag's /tf and /tf_static (10's
best_poses bag by default, or the finalized merged bag), at times counted
from mobile_1's start (the first pose of 04's LiDAR trajectory).

    python scoop_pipeline/checks/tf_at.py <pipeline config> [--bag BAG]
        [--t 0 5 60] [--frames map_zed zed_camera_link ...]

Also: map_zed against mobile_1's start (zed_camera_link at t = 0), which it
should coincide with in a run without 05.
"""
import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "map_stages"))

FRAMES = ["map_zed", "odom_zed", "zed_camera_link", "zed_left_camera_optical_frame",
          "os_sensor", "map_realsense", "camera_link", "arducam_optical_frame"]


def yaw(T):
    return float(np.degrees(np.arctan2(T[1, 0], T[0, 0])))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--bag", default=None, help="default: 10's bags/<tag>_best_poses")
    ap.add_argument("--t", type=float, nargs="+", default=[0.0, 5.0, 60.0],
                    help="seconds from mobile_1's start")
    ap.add_argument("--frames", nargs="+", default=FRAMES)
    a = ap.parse_args()
    from pipeline_common import load_pipeline
    from pipeline_boards import read_tf_and_info
    from scoop import tftree
    P = load_pipeline(a.config)
    bag = os.path.expanduser(a.bag) if a.bag else P.outp(
        (P.cfg.get("10_publish") or {}).get("output_bag") or "{tag}_best_poses")
    traj = P.lidar_track_traj("_in_cam")
    t0 = tftree.read_tum(traj)[0][0] * 1e-9
    print("bag:   %s\nstart: mobile_1's first pose in %s (t = %.3f)" % (bag, os.path.basename(traj), t0))
    tree = read_tf_and_info(bag, [], want_tf=True, verbose=False)[1]
    for dt in a.t:
        print("\nt = %.1f s from mobile_1's start" % dt)
        print("  %-32s %28s %9s" % ("frame", "xyz in map (m)", "yaw deg"))
        for fr in a.frames:
            T = tree.lookup("map", fr, t0 + dt)
            if T is None:
                print("  %-32s %28s" % (fr, "(no lookup)"))
            else:
                print("  %-32s %28s %9.2f" % (fr, "(%.3f, %.3f, %.3f)" % tuple(T[:3, 3]), yaw(T)))
    Tz = tree.lookup("map", "map_zed", t0)
    Tl = tree.lookup("map", "zed_camera_link", t0)
    if Tz is not None and Tl is not None:
        D = np.linalg.inv(Tz) @ Tl
        ang = np.degrees(np.arccos(np.clip((np.trace(D[:3, :3]) - 1) / 2, -1, 1)))
        print("\nmap_zed vs mobile_1's start (zed_camera_link at t = 0): %.1f cm, %.2f deg apart"
              % (100 * np.linalg.norm(D[:3, 3]), ang))


if __name__ == "__main__":
    main()
