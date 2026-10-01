#!/usr/bin/env python3
"""Merge two or more rosbag2 bags into one, with the topics you choose.

    python scoop_pipeline/scripts/merge_bags.py <bag> <bag> [<bag> ...] --list
    python scoop_pipeline/scripts/merge_bags.py <bag> <bag> ... -o <out bag>
        [--topics PATTERN ...] [--exclude PATTERN ...] [--pick]
        [--compression zstd|none] [--static-tf configs/static_tf.yaml]

--list         show every topic of the bags (type, messages, which bags) and stop
--topics       keep only these topics; shell patterns work: '/mobile_1/zed/*'
--exclude      leave these out (applied after --topics)
--pick         choose from a numbered list instead: 1,3,5-8 or all
               (patterns work there too)
-o             the merged bag folder (must not exist)

Messages are copied byte for byte in log-time order, QoS kept; all static
transforms go into one latched /tf_static message; zstd inside the MCAP by
default. Quote patterns so the shell leaves the * alone. The work is in
scoop/merge.py.

    merge_bags.py A B -o AB --topics '/mobile_1/ouster/points' '/mobile_1/zed/left/*' /tf /tf_static
"""
import argparse
import os
import re
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.simplefilter("ignore", FutureWarning)

from scoop import merge, tftree                                     # noqa: E402
from scoop.bag import BagError                                      # noqa: E402


def show(topics):
    w = max(len(t) for t in topics)
    for i, (t, (typ, n, bags)) in enumerate(topics.items(), 1):
        print(f"  {i:3d}  {t:{w}s}  {n:9d}  {typ:35s}  {', '.join(bags)}")


def parse_pick(text, names):
    """'1,3,5-8', 'all', or shell patterns -> topic names."""
    text = text.strip()
    if text in ("", "all", "*"):
        return list(names)
    out = []
    for part in re.split(r"[,\s]+", text):
        if not part:
            continue
        m = re.fullmatch(r"(\d+)(?:-(\d+))?", part)
        if m:
            a, b = int(m.group(1)), int(m.group(2) or m.group(1))
            if not 1 <= a <= b <= len(names):
                raise BagError(f"{part}: numbers go from 1 to {len(names)}")
            out += names[a - 1:b]
        else:
            out += merge.select_topics(names, [part])
    return list(dict.fromkeys(out))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bags", nargs="+")
    ap.add_argument("-o", "--out", help="merged bag folder (must not exist)")
    ap.add_argument("--list", action="store_true", help="list the topics and stop")
    ap.add_argument("--topics", nargs="+", default=[], metavar="PATTERN")
    ap.add_argument("--exclude", nargs="+", default=[], metavar="PATTERN")
    ap.add_argument("--pick", action="store_true", help="choose topics from a list")
    ap.add_argument("--compression", default="zstd", choices=["zstd", "none"])
    ap.add_argument("--static-tf", default="", help="calibrations to add to /tf_static")
    a = ap.parse_args()
    try:
        topics = merge.list_topics(a.bags)
        if a.list:
            show(topics)
            return
        if not a.out:
            sys.exit("error: -o <out bag> is needed (or --list)")
        if os.path.exists(a.out):
            sys.exit(f"error: {a.out} exists")
        names = list(topics)
        if a.pick:
            show(topics)
            keep = parse_pick(input("\ntopics to merge (e.g. 1,3,5-8, all, or patterns): "),
                              names)
            keep = merge.select_topics(keep, exclude=a.exclude)
        else:
            keep = merge.select_topics(names, a.topics, a.exclude)
        print(f"\nmerging {len(keep)} of {len(names)} topics:")
        for t in keep:
            print(f"    {t}  ({topics[t][1]} msgs)")
        static_tf = []
        if a.static_tf:
            import yaml
            with open(os.path.expanduser(a.static_tf)) as fh:
                cals = tftree.load_calibrations(yaml.safe_load(fh))
            if cals:
                static_tf, notes = tftree.attach(cals, tftree.tf_edges(a.bags))
                for n in notes:
                    print(f"    tf: {n}")
        res = merge.merge_bags(a.bags, a.out, compression=a.compression,
                               static_tf=static_tf, topics=keep)
        print("\n".join(f"    {t:45s} {n:9d}" for t, n in sorted(res.counts.items())))
    except (BagError, ValueError) as e:
        sys.exit(f"error: {e}")
    except (KeyboardInterrupt, EOFError):
        sys.exit("\nstopped")


if __name__ == "__main__":
    main()
