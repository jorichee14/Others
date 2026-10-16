#!/usr/bin/env python3
"""Rename TF frames in a bag: headers, Odometry child frames and /tf, /tf_static.

    python processing/rename_frames.py <bag> -o <new bag> --rename OLD:=NEW [OLD:=NEW ...]
        [--compression zstd|none]

For bags made before decode_ouster.py --frame-id, e.g. mobile_2's points:

    python processing/rename_frames.py $W/merged -o $W/merged_frames \\
        --rename os_lidar:=os_lidar_mobile2

Every message is copied byte for byte except the renamed ones; log times,
QoS and order are kept. The work is in scoop/frames.py.
"""
import argparse
import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.simplefilter("ignore", FutureWarning)

from scoop import frames                                            # noqa: E402
from scoop.bag import BagError                                      # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bag")
    ap.add_argument("-o", "--out", required=True, help="the new bag (must not exist)")
    ap.add_argument("--rename", nargs="+", required=True, metavar="OLD:=NEW")
    ap.add_argument("--compression", default="zstd", choices=["zstd", "none"])
    a = ap.parse_args()
    if os.path.exists(a.out):
        sys.exit(f"{a.out} exists; pick a new folder")
    try:
        frames.rename_bag(a.bag, a.out, frames.parse_renames(a.rename), a.compression)
    except (ValueError, BagError) as e:
        sys.exit(f"error: {e}")


if __name__ == "__main__":
    main()
