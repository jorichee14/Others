#!/usr/bin/env python3
"""Are the IMU's stamps on the same time as the LiDAR's? The IMU's angular
rate against a trajectory's, lined up.

  python3 analysis/odom/imu_vs_traj.py <pipeline config> [--traj NAME=FILE ...]
      [--topic /mobile_2/ouster/imu]

Default trajectories, from odometry/<machine>/ of the config's pass: ref
(traj_lidar_ba_glim.txt: the scans made consistent, so on the LiDAR's time)
and glim (traj_lidar.txt). The IMU is the config's bag's <machine>'s
ouster/imu, or --topic.

The rigid body turns at one rate whatever the frame, so |w| from the IMU's
gyro and |w| from the trajectory (central differences of its rotations,
which carry no half-step delay) are one signal; the shift that lines them
up best (scoop/imu_sync.py line_up, the IMU-to-IMU check of the retimed
step) is how far the IMU's stamps run behind the trajectory's:

  against ref   ~0           the IMU and the LiDAR are on the same time:
                             the stamping is right
                ~ GLIM's     the IMU's stamps are off by that much: fix them
                offset       (or GLIM's imu_time_offset), and GLIM again
  against glim               the same comparison for GLIM's own poses

Reads the bag's IMU only, writes nothing.
"""
import argparse
import os
import sys

import numpy as np
from scipy.spatial.transform import Rotation

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "map_stages"))

from pipeline_common import load_pipeline, load_traj                     # noqa: E402
from scoop import bag                                                    # noqa: E402
from scoop.imu_sync import angular_rates, line_up                        # noqa: E402


def traj_rate(t, T):
    """|w| at each interior pose, from the poses either side (central)."""
    dR = np.einsum("nji,njk->nik", T[:-2, :3, :3], T[2:, :3, :3])
    ang = np.linalg.norm(Rotation.from_matrix(dR).as_rotvec(), axis=1)
    return t[1:-1], ang / (t[2:] - t[:-2])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--traj", action="append", default=[], metavar="NAME=FILE")
    ap.add_argument("--topic")
    ap.add_argument("--max-shift", type=float, default=0.5, help="search +/- this (s)")
    a = ap.parse_args()

    P = load_pipeline(a.config)
    files = {"ref": P.outp("traj_lidar_ba_glim.txt"), "glim": P.outp("traj_lidar.txt")}
    for kv in a.traj:
        name, _, f = kv.partition("=")
        files[name] = os.path.expanduser(f)
    topic = a.topic or f"/{P.machine}/ouster/imu"
    bag_path = os.path.expanduser(P.dataset["bag"])
    print(f"IMU {topic} in {bag_path}")
    ti, wi = angular_rates(bag.open_bag(bag_path), topic)
    if len(ti) < 100:
        raise SystemExit(f"{len(ti)} IMU messages on {topic}")
    ti = ti * 1e-9
    print(f"  {len(ti)} messages, {1 / np.median(np.diff(ti)):.0f} Hz")
    print(f"\n{'trajectory':12s}{'IMU behind it':>16s}{'correlation':>13s}{'scale':>8s}")
    for name, f in files.items():
        if not os.path.isfile(f):
            print(f"{name:12s}  (none at {f})")
            continue
        t, T = load_traj(f)
        tr, wr = traj_rate(t, T)
        s, c, sc, n, turn = line_up(tr, wr, ti, wi, a.max_shift)
        print(f"{name:12s}{s * 1000:13.1f} ms{c:13.3f}{sc:8.3f}")
    print("\n+ = the IMU's stamps are later than the trajectory's for the same motion.\n"
          "Against ref (on the LiDAR's time): ~0 = the stamping is right.")


if __name__ == "__main__":
    main()
