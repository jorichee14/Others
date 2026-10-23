#!/usr/bin/env python3
"""Process one robot recording: raw bag -> decoded -> retimed -> GLIM, and the
ZED SVO2 next to it -> a bag on the same clock.

    python processing/process_recording.py <raw recording folder>
        [--until decoded|retimed|glim|zed] [--redo decoded|retimed|glim|zed]
        [--only decoded|retimed|glim|zed]
        [--settings configs/recording.yaml] [--work-root <dir>] [--status]
    python processing/process_recording.py --save-ouster-metadata <bag or .mcap> <ns> <out.json>

<raw recording folder> is a folder below data/raw/, e.g.
    ~/workspaces/isaac_ros-dev/data/raw/20260924/mapping_A/mobile_1
Outputs go to the mirrored folder below data/work/ with fixed names:
    <bag>_decoded/  <bag>_retimed/  glim/  <svo>_zed/  <svo>_zed_right/
    clock.json  process.yaml
A folder with only an SVO (no Ouster packets bag) does the zed step alone.
--only X runs just X. Steps already done are skipped; --redo X runs X and the steps built from it
again (decoded -> retimed -> glim; zed on its own). The GLIM step starts
docker (processing/run_glim.sh); the zed step needs the ZED wrapper, here or in
the isaac_ros container (processing/run_zed.sh). The work is in
scoop/recording.py.

The Ouster's metadata (<ns>/metadata) is published once, at the start of a
recording; a recording whose first file is lost has none. Then decode and
retime read it from ouster_metadata.json in the robot's raw folder, which
--save-ouster-metadata copies out of another bag (or .mcap file) of the same
sensor, e.g. the lost first file, or the same day's mapping pass.
"""
import argparse
import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.simplefilter("ignore", FutureWarning)

from scoop import recording                                         # noqa: E402
from scoop.bag import BagError                                      # noqa: E402


def _word(ok, step=""):
    if ok is None:
        return "no svo" if step == "zed" else "no packets bag"
    return "done" if ok else "-"


def save_ouster_metadata(src, ns, out):
    from scoop import bag, ouster
    text = ouster.read_metadata(bag.open_bag(src), ns)
    with open(out, "w") as fh:
        fh.write(text)
    print(f"{ns}/metadata of {src} -> {out}")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--save-ouster-metadata":
        if len(sys.argv) != 5:
            sys.exit("usage: process_recording.py --save-ouster-metadata <bag or .mcap> <ns> <out.json>")
        try:
            save_ouster_metadata(*sys.argv[2:5])
        except BagError as e:
            sys.exit(f"error: {e}")
        return
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("raw_dir")
    ap.add_argument("--until", choices=recording.STEPS)
    ap.add_argument("--redo", choices=recording.STEPS)
    ap.add_argument("--only", choices=recording.STEPS, help="run just this step")
    ap.add_argument("--settings", default=None, help="default: configs/recording.yaml")
    ap.add_argument("--work-root", default=None, help="default: the 'work' folder next to 'raw'")
    ap.add_argument("--status", action="store_true", help="show what is done and stop")
    a = ap.parse_args()
    try:
        # the machine folder's own settings (recording.yaml machines:), e.g. mobile_2/
        settings = recording.settings_for(recording.load_settings(a.settings),
                                          os.path.basename(os.path.normpath(a.raw_dir)))
        rec = recording.find_recording(a.raw_dir, a.work_root,
                                       settings["ouster"].get("packets_ns", "/ouster"),
                                       recording.zed_bags(settings["zed"]))
        print(f"raw bag: {rec.bag or '-'}\nsvo:     {rec.svo or '-'}\nwork:    {rec.work}")
        if a.status:
            for s, ok in rec.status().items():
                print(f"  {s:8s} {_word(ok, s)}")
            return
        st = recording.process(rec, settings, until=a.until, redo=a.redo, only=a.only)
    except (BagError, ValueError) as e:
        sys.exit(f"error: {e}")
    print("\n" + "  ".join(f"{s}: {_word(ok, s)}" for s, ok in st.items()))


if __name__ == "__main__":
    main()
