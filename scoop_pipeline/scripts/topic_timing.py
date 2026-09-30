#!/usr/bin/env python3
"""How evenly a bag's topics are spaced: intervals, gaps, repeated frames.

    python scoop_pipeline/scripts/topic_timing.py <bag dir> <topic> [<topic> ...]

e.g. left against right camera of the ZED bags:

    python scoop_pipeline/scripts/topic_timing.py <work>/<svo>_zed /mobile_1/zed/left/image_rect_color
    python scoop_pipeline/scripts/topic_timing.py <work>/<svo>_zed_right /mobile_1/zed/right/image_rect_color

Reads every message of those topics once (for images, as fast as the disk
reads them); the work is in scoop/timing.py.
"""
import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.simplefilter("ignore", FutureWarning)

from scoop import timing                                            # noqa: E402
from scoop.bag import BagError                                      # noqa: E402


def main():
    if len(sys.argv) < 3 or sys.argv[1] in ("-h", "--help"):
        sys.exit(__doc__)
    try:
        for t in timing.topic_timing(sys.argv[1], sys.argv[2:]):
            print(t.summary())
    except BagError as e:
        sys.exit(f"error: {e}")


if __name__ == "__main__":
    main()
