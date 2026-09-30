#!/usr/bin/env python3
"""Sensor-time points bag -> host capture time. retime_bag.py without ROS.

    python3 scripts/retime.py <packets_bag> <sensortime_bag> <out_bag>
                              [packets_ns] [points_ns]

Same arguments and result as retime_bag.py (packets_ns defaults to /ouster,
points_ns to /mobile_1/ouster), but runs in the scoop conda env: the header
stamps are rewritten in the message bytes, so rclpy is not needed. The work
is in scoop/retime.py.

    <packets_bag>     the ORIGINAL recording (lidar packets -> clock fit)
    <sensortime_bag>  the decoded bag (scripts/decode_ouster.py) or a replay
    <out_bag>         new folder for the retimed bag
"""
import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.simplefilter("ignore", FutureWarning)

from scoop import retime                                            # noqa: E402
from scoop.bag import BagError                                      # noqa: E402


def main():
    args = sys.argv[1:]
    if len(args) < 3 or args[0] in ("-h", "--help"):
        print(__doc__)
        sys.exit(1)
    packets, sensortime, out = args[:3]
    packets_ns = args[3] if len(args) > 3 else "/ouster"
    points_ns = args[4] if len(args) > 4 else "/mobile_1/ouster"
    if os.path.exists(out):
        sys.exit(f"{out} exists; pick a new output folder")
    try:
        retime.retime_bag(packets, sensortime, out, packets_ns, points_ns)
    except BagError as e:
        sys.exit(f"error: {e}")


if __name__ == "__main__":
    main()
