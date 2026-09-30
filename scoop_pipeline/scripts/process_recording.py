#!/usr/bin/env python3
"""Process one robot recording: raw bag -> decoded -> retimed -> GLIM.

    python scripts/process_recording.py <raw recording folder>
        [--until decoded|retimed|glim] [--redo decoded|retimed|glim]
        [--settings configs/recording.yaml] [--work-root <dir>] [--status]

<raw recording folder> is a folder below data/raw/, e.g.
    ~/workspaces/isaac_ros-dev/data/raw/20260924/mapping_A/mobile_1
Outputs go to the mirrored folder below data/work/ with fixed names:
    decoded/  retimed/  glim/  clock.json  process.yaml
Steps already done are skipped; --redo X runs X and everything after it
again. The GLIM step starts docker (scripts/run_glim.sh). The work is in
scoop/recording.py.
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
    ap.add_argument("raw_dir")
    ap.add_argument("--until", choices=recording.STEPS)
    ap.add_argument("--redo", choices=recording.STEPS)
    ap.add_argument("--settings", default=None, help="default: configs/recording.yaml")
    ap.add_argument("--work-root", default=None, help="default: the 'work' folder next to 'raw'")
    ap.add_argument("--status", action="store_true", help="show what is done and stop")
    a = ap.parse_args()
    try:
        settings = recording.load_settings(a.settings)
        rec = recording.find_recording(a.raw_dir, a.work_root,
                                       settings["ouster"].get("packets_ns", "/ouster"))
        print(f"raw bag: {rec.bag}\nwork:    {rec.work}")
        if a.status:
            for s, ok in rec.status().items():
                print(f"  {s:8s} {'done' if ok else '-'}")
            return
        st = recording.process(rec, settings, until=a.until, redo=a.redo)
    except (BagError, ValueError) as e:
        sys.exit(f"error: {e}")
    print("\n" + "  ".join(f"{s}: {'done' if ok else '-'}" for s, ok in st.items()))


if __name__ == "__main__":
    main()
