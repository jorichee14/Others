#!/usr/bin/env python3
"""ZED SVO2 -> rosbag2 on the capture clock, through the ZED ROS 2 wrapper.

    python3 scripts/svo_to_bag.py <svo> <out_bag_dir>
        [--settings configs/recording.yaml] [--keep-record]

Topics, their new names, the camera model and wrapper parameters come from
the `zed` section of the settings. The wrapper plays the SVO with its own
timestamps and `ros2 bag record` records it (scripts/run_zed.sh, which uses
the isaac_ros container when ROS is not installed here); the recording is
then restamped to capture time and renamed (scoop/zed.py).
"""
import argparse
import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.simplefilter("ignore", FutureWarning)

from scoop import recording, zed                                    # noqa: E402
from scoop.bag import BagError                                      # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("svo")
    ap.add_argument("out")
    ap.add_argument("--settings", default=None, help="default: configs/recording.yaml")
    ap.add_argument("--keep-record", action="store_true",
                    help="keep the raw recording (<out>.record) after restamping")
    a = ap.parse_args()
    try:
        z = recording.load_settings(a.settings)["zed"]
        zed.svo_to_bag(a.svo, a.out, keep_record=a.keep_record, **recording.zed_args(z))
    except (BagError, ValueError) as e:
        sys.exit(f"error: {e}")


if __name__ == "__main__":
    main()
