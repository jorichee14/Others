#!/usr/bin/env python3
"""Merge two or more rosbag2 bags into one, with the topics you choose.

    python scoop_pipeline/processing/merge_bags.py <bag> <bag> [<bag> ...] --list
    python scoop_pipeline/processing/merge_bags.py <bag> <bag> ... -o <out bag>
        [--topics PATTERN ...] [--exclude PATTERN ...] [--pick]
        [--source PATTERN=N ...] [--keep-duplicates]
        [--compression zstd|none]

--list         show every topic of the bags (type, messages, which bags) and stop
--topics       keep only these topics; shell patterns work: '/mobile_1/zed/*'
--exclude      leave these out (applied after --topics)
--pick         choose from a numbered list instead: 1,3,5-8 or all
               (patterns work there too)
--source       take matching topics only from bag N (N = position on the
               command line, from 1): '/mobile_1/zed/*=2'. A chosen topic
               in several bags is refused unless --source picks one (or
               --keep-duplicates keeps them all)
-o             the merged bag folder (must not exist)

Messages are copied byte for byte in log-time order, QoS kept; all static
transforms go into one latched /tf_static message; zstd-compressed (lossless)
unless --compression none. To change TF (calibrations, dropped edges, a new odometry) use
tf_edit.py on the result. Quote patterns so the shell leaves the * alone. The work is in
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

from scoop import merge                                             # noqa: E402
from scoop.bag import BagError                                      # noqa: E402


def show(topics, bags):
    for i, b in enumerate(bags, 1):
        print(f"  bag {i}: {os.path.basename(os.path.normpath(b))}")
    print()
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
    ap.add_argument("--source", nargs="+", default=[], metavar="PATTERN=N",
                    help="take matching topics only from bag N (1 = first bag)")
    ap.add_argument("--keep-duplicates", action="store_true",
                    help="keep a topic from every bag that has it")
    ap.add_argument("--compression", default="zstd", choices=["zstd", "none"],
                    help="zstd (default): lossless, about half the size; none: faster")
    a = ap.parse_args()
    try:
        topics = merge.list_topics(a.bags)
        if a.list:
            show(topics, a.bags)
            return
        if not a.out:
            sys.exit("error: -o <out bag> is needed (or --list)")
        if os.path.exists(a.out):
            sys.exit(f"error: {a.out} exists")
        names = list(topics)
        if a.pick:
            show(topics, a.bags)
            keep = parse_pick(input("\ntopics to merge (e.g. 1,3,5-8, all, or patterns): "),
                              names)
            keep = merge.select_topics(keep, exclude=a.exclude)
        else:
            keep = merge.select_topics(names, a.topics, a.exclude)
        rules = []
        for r in a.source:
            pat, _, n = r.rpartition("=")
            if not pat or not n.isdigit():
                raise BagError(f"--source {r}: expected PATTERN=N")
            rules.append((pat, int(n) - 1))
        source = merge.resolve_sources(a.bags, keep, rules)
        dup = merge.duplicated(a.bags, keep, source)
        if dup and not a.keep_duplicates:
            lines = "\n".join(f"    {t}: {', '.join(b)}" for t, b in dup.items())
            raise BagError(f"these topics are in more than one bag, so every message "
                           f"would be there twice:\n{lines}\n"
                           f"pick one with --source PATTERN=N (or --keep-duplicates)")
        names_ = [os.path.basename(os.path.normpath(b)) for b in a.bags]
        print(f"\nmerging {len(keep)} of {len(names)} topics:")
        for t in keep:
            frm = names_[source[t]] if t in source else ", ".join(topics[t][2])
            print(f"    {t}  (from {frm})")
        res = merge.merge_bags(a.bags, a.out, compression=a.compression,
                               topics=keep, source=source)
        print("\n".join(f"    {t:45s} {n:9d}" for t, n in sorted(res.counts.items())))
    except (BagError, ValueError) as e:
        sys.exit(f"error: {e}")
    except (KeyboardInterrupt, EOFError):
        sys.exit("\nstopped")


if __name__ == "__main__":
    main()
