#!/usr/bin/env python3
"""Rename TF frames in a bag: headers, Odometry child frames and /tf, /tf_static.

    python processing/rename_frames.py <bag> -o <new bag> [--suffix _mobile2]
        [--rename OLD:=NEW ...] [--compression zstd|none]

--suffix S   the Ouster frames, as decode_ouster.py --frame-suffix names them:
             os_lidar, os_sensor, os_imu -> os_lidar<S>, os_sensor<S>, os_imu<S>
--rename     any other frame, OLD:=NEW (both can be given)

For bags decoded before --frame-suffix, e.g. mobile_2's:

    python processing/rename_frames.py $W/merged -o $W/merged_frames --suffix _mobile2

Every message is copied byte for byte except the renamed ones; log times,
QoS and order are kept. The work is in scoop/frames.py.
"""
import argparse
import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.simplefilter("ignore", FutureWarning)

from scoop import frames, ouster                                    # noqa: E402
from scoop.bag import BagError                                      # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bag")
    ap.add_argument("-o", "--out", required=True, help="the new bag (must not exist)")
    ap.add_argument("--suffix", default="", help="on os_lidar, os_sensor, os_imu, e.g. _mobile2")
    ap.add_argument("--rename", nargs="+", default=[], metavar="OLD:=NEW")
    ap.add_argument("--compression", default="zstd", choices=["zstd", "none"])
    a = ap.parse_args()
    if os.path.exists(a.out):
        sys.exit(f"{a.out} exists; pick a new folder")
    try:
        mapping = {**ouster.frame_names(a.suffix), **frames.parse_renames(a.rename)}
        if not mapping:
            sys.exit("nothing to rename: give --suffix and/or --rename")
        frames.rename_bag(a.bag, a.out, mapping, a.compression)
    except (ValueError, BagError) as e:
        sys.exit(f"error: {e}")


if __name__ == "__main__":
    main()
