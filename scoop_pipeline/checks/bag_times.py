#!/usr/bin/env python3
"""Do a pass's recordings fit together in time? Every robot's bags, from their
files' own time spans (the MCAP summaries; nothing is decoded).

    python3 check.py bags <raw pass folder> [--gap 2] [--spread 10]

    e.g. python3 check.py bags ~/workspaces/isaac_ros-dev/data/raw/20261015/coop/site_1

Per bag: its start, end and length; the split files it lists (_0, _1, ...)
with any number missing, a gap of more than --gap s between consecutive
files, a file listed but not on disk, a file on disk it does not list, a file
without a summary (the recording was cut off). A bag that does not start at
_0 has lost what is published once at the start (an Ouster's metadata, latched
/tf_static). Across the robots: the window they all cover, and a bag that
starts or ends more than --spread s apart from the others. An .svo2 is listed
(its time is not read). Exits 1 when anything is flagged.
"""
import argparse
import os
import re
import sys
import time

import yaml

SPLIT = re.compile(r"_(\d+)\.mcap$")


def clock(ns):
    return time.strftime("%H:%M:%S", time.localtime(ns * 1e-9))


def file_span(path):
    """(start_ns, end_ns, messages) of one .mcap from its summary, or None."""
    from mcap.reader import make_reader
    with open(path, "rb") as fh:
        try:
            s = make_reader(fh).get_summary()
        except Exception:                                     # a file cut off mid-write
            return None
    st = s.statistics if s is not None else None
    if st is None:
        return None
    return st.message_start_time, st.message_end_time, st.message_count


def read_bag(folder):
    """{"files": [(name, span or None, on_disk)], "unlisted": [...]} of a rosbag2 folder."""
    meta = yaml.safe_load(open(os.path.join(folder, "metadata.yaml"))) or {}
    info = meta.get("rosbag2_bagfile_information") or {}
    listed = list(info.get("relative_file_paths") or [])
    files = []
    for name in listed:
        p = os.path.join(folder, name)
        files.append((name, file_span(p) if os.path.isfile(p) else None, os.path.isfile(p)))
    unlisted = sorted(f for f in os.listdir(folder) if f.endswith(".mcap") and f not in listed)
    return {"files": files, "unlisted": unlisted}


def find_bags(pass_dir):
    """[(robot folder name, bag folder)] below the pass folder; and the .svo2 files."""
    bags, svos = [], []
    for robot in sorted(os.listdir(pass_dir)):
        rd = os.path.join(pass_dir, robot)
        if not os.path.isdir(rd):
            continue
        for root, dirs, fnames in os.walk(rd):
            if "metadata.yaml" in fnames:
                bags.append((robot, root))
                dirs[:] = []
            svos += [(robot, os.path.join(root, f)) for f in fnames if f.endswith(".svo2")]
    return bags, svos


def check(pass_dir, gap_s=2.0, spread_s=10.0, log=print):
    """-> the list of flagged problems (printed as it goes)."""
    bags, svos = find_bags(pass_dir)
    flags = []

    def flag(msg):
        flags.append(msg)
        log("  ! " + msg)

    if not bags:
        flag("no bag folder (metadata.yaml) below %s" % pass_dir)
        return flags
    spans = {}
    for robot, folder in bags:
        b = read_bag(folder)
        name = os.path.relpath(folder, pass_dir)
        ok = [(n, s) for n, s, disk in b["files"] if s is not None]
        if ok:
            t0 = min(s[0] for _, s in ok)
            t1 = max(s[1] for _, s in ok)
            spans[name] = (t0, t1)
            log("%s  %s .. %s  (%.0f s, %d files, %d messages)"
                % (name, clock(t0), clock(t1), (t1 - t0) * 1e-9, len(b["files"]),
                   sum(s[2] for _, s in ok)))
        else:
            log("%s  (no file with a time span)" % name)
        for n, s, disk in b["files"]:
            if not disk:
                flag("%s: %s is listed but not on disk" % (name, n))
            elif s is None:
                flag("%s: %s has no summary (the recording was cut off?)" % (name, n))
        for n in b["unlisted"]:
            flag("%s: %s is on disk but not in metadata.yaml" % (name, n))
        idx = sorted(int(m.group(1)) for n, _, _ in b["files"] for m in [SPLIT.search(n)] if m)
        if idx:
            missing = sorted(set(range(0, idx[-1] + 1)) - set(idx))
            if missing:
                flag("%s: split file(s) %s missing%s" % (
                    name, ", ".join("_%d" % i for i in missing),
                    " -- no _0: what is published once at the start (an Ouster's "
                    "metadata, latched /tf_static) is lost" if 0 in missing else ""))
        ok.sort(key=lambda x: x[1][0])
        for (n0, s0), (n1, s1) in zip(ok, ok[1:]):
            if s1[0] - s0[1] > gap_s * 1e9:
                flag("%s: %.0f s with nothing between %s (ends %s) and %s (starts %s)"
                     % (name, (s1[0] - s0[1]) * 1e-9, n0, clock(s0[1]), n1, clock(s1[0])))
    for robot, p in svos:
        log("%s  (svo, time not read)" % os.path.relpath(p, pass_dir))
    if len(spans) > 1:
        a = max(t0 for t0, _ in spans.values())
        z = min(t1 for _, t1 in spans.values())
        if z <= a:
            flag("the bags do not overlap in time")
        else:
            log("\nall bags cover %s .. %s (%.0f s)" % (clock(a), clock(z), (z - a) * 1e-9))
        first = min(t0 for t0, _ in spans.values())
        last = max(t1 for _, t1 in spans.values())
        for name, (t0, t1) in spans.items():
            if t0 - first > spread_s * 1e9:
                flag("%s starts %.0f s after the first bag" % (name, (t0 - first) * 1e-9))
            if last - t1 > spread_s * 1e9:
                flag("%s ends %.0f s before the last bag" % (name, (last - t1) * 1e-9))
    log("\n%s" % ("%d problem(s)" % len(flags) if flags else "no problems: the bags fit together"))
    return flags


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pass_dir")
    ap.add_argument("--gap", type=float, default=2.0, help="s between split files (2)")
    ap.add_argument("--spread", type=float, default=10.0,
                    help="s a bag may start / end apart from the others (10)")
    a = ap.parse_args()
    d = os.path.abspath(os.path.expanduser(a.pass_dir))
    if not os.path.isdir(d):
        sys.exit("no folder %s" % d)
    print("pass: %s\n" % d)
    sys.exit(1 if check(d, a.gap, a.spread) else 0)


if __name__ == "__main__":
    main()
