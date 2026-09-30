#!/usr/bin/env python3
"""Ouster packets bag -> points bag, offline. Replaces the replay step.

    python3 scripts/decode_ouster.py <packets_bag> <out_bag_dir>
        [--packets-ns /ouster] [--out-ns /mobile_1/ouster]
        [--driver-min-range 1.30] [--driver-max-range 1000]
        [--frame lidar|sensor] [--no-imu] [--limit N]
        [--remap SRC:=DST ...]

Does what the replay launch did (ros2 bag play --remap + ouster_ros driver +
ros2 bag record) in one pass over the ORIGINAL recording; the work is in
scoop/replay.py. Writes a rosbag2 folder with <out-ns>/points, <out-ns>/imu,
<out-ns>/metadata in SENSOR time, plus every --remap topic copied byte for
byte under its new name. Your replay launch's remaps are:

    --remap /ouster/metadata:=/mobile_1/ouster/metadata
    --remap /ouster/imu_packets:=/mobile_1/ouster/imu_packets
    --remap /ouster/lidar_packets:=/mobile_1/ouster/lidar_packets
    --remap /zed/zed_node/imu/data:=/mobile_1/zed/imu/data
    --remap /zed/zed_node/imu/mag:=/mobile_1/zed/imu/mag

Next step, in the same env (or retime_bag.py in a ROS terminal, same result):

    python3 scripts/retime.py <packets_bag> <out_bag_dir> <retimed_dir> /ouster /mobile_1/ouster
"""
import argparse
import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.simplefilter("ignore", FutureWarning)

from scoop import replay                                            # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("packets_bag")
    ap.add_argument("out_bag_dir")
    ap.add_argument("--packets-ns", default="/ouster")
    ap.add_argument("--out-ns", default="/mobile_1/ouster")
    ap.add_argument("--driver-min-range", type=float, default=1.30)
    ap.add_argument("--driver-max-range", type=float, default=1000.0)
    ap.add_argument("--frame", choices=("lidar", "sensor"), default="lidar")
    ap.add_argument("--no-imu", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="stop after N scans (for a test)")
    ap.add_argument("--remap", action="append", default=[], metavar="SRC:=DST",
                    help="copy topic SRC of the original bag into the output as DST")
    ap.add_argument("--copy-all", action="store_true",
                    help="copy every other topic of the original bag too")
    ap.add_argument("--exclude", action="append", default=[], metavar="TOPIC",
                    help="leave this topic out of the copy")
    a = ap.parse_args()
    if os.path.exists(a.out_bag_dir):
        sys.exit(f"{a.out_bag_dir} exists; pick a new output folder")
    try:
        remap = replay.parse_remaps(a.remap)
        replay.decode_ouster_bag(a.packets_bag, a.out_bag_dir, packets_ns=a.packets_ns,
                                 out_ns=a.out_ns, driver_min_range=a.driver_min_range,
                                 driver_max_range=a.driver_max_range, frame=a.frame,
                                 imu=not a.no_imu, remap=remap, copy_all=a.copy_all,
                                 exclude=a.exclude, limit=a.limit)
    except (ValueError, replay.bag.BagError) as e:
        sys.exit(f"error: {e}")
    print(f"\nnext:  python scripts/retime.py {a.packets_bag} "
          f"{a.out_bag_dir} <retimed_out_dir> {a.packets_ns} {a.out_ns}")


if __name__ == "__main__":
    main()
