#!/usr/bin/env python
"""Self-test for checks/bag_times.py (check.py bags): a pass's bags in time.

  clean      three robots' bags covering the same minutes, split files in a
             row: no problem, the common window printed
  flawed     a bag missing _1 and _2 after _0 (a gap), a file on disk it does
             not list, a robot that starts 40 s late, a file cut off (no
             summary): each flagged

    python tests/test_bag_times.py
"""
import os
import shutil
import sys
import tempfile

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "checks")]

import bag_times                                                       # noqa: E402

S = 10 ** 9
T0 = 1_792_037_000 * S


def mcap_file(path, t0_s, t1_s, cut=False):
    from mcap.writer import Writer
    with open(path, "wb") as fh:
        w = Writer(fh)
        w.start("ros2", "test")
        ch = w.register_channel("/x", "cdr", w.register_schema("std_msgs/msg/String", "ros2msg", b"string data"))
        for t in (t0_s, (t0_s + t1_s) / 2, t1_s):
            w.add_message(ch, log_time=T0 + int(t * S), publish_time=T0 + int(t * S), data=b"\0\1\0\0")
        if cut:
            fh.flush()
            fh.truncate(fh.tell())                        # no footer, no summary
            return
        w.finish()


def bag(folder, files, extra=()):
    """files: [(index, t0_s, t1_s, cut)]; extra: indices on disk, not listed."""
    os.makedirs(folder)
    name = os.path.basename(folder)
    listed = []
    for i, a, z, cut in files:
        f = "%s_%d.mcap" % (name, i)
        mcap_file(os.path.join(folder, f), a, z, cut)
        listed.append(f)
    for i in extra:
        mcap_file(os.path.join(folder, "%s_%d.mcap" % (name, i)), 0, 1)
    yaml.safe_dump({"rosbag2_bagfile_information": {"version": 5, "storage_identifier": "mcap",
                                                    "relative_file_paths": listed}},
                   open(os.path.join(folder, "metadata.yaml"), "w"))


def main():
    tmp = tempfile.mkdtemp(prefix="bagtimes_")
    fails = []
    try:
        clean = os.path.join(tmp, "clean")
        bag(os.path.join(clean, "mobile_1", "b_mobile_1"), [(0, 0, 500, False), (1, 500.1, 1000, False)])
        bag(os.path.join(clean, "mobile_2", "b_mobile_2"), [(0, 2, 999, False)])
        bag(os.path.join(clean, "infra_1", "b_infra_1"), [(0, 1, 998, False)])
        open(os.path.join(clean, "mobile_1", "z.svo2"), "w").close()
        out = []
        flags = bag_times.check(clean, log=out.append)
        if flags:
            fails.append("clean pass flagged: %s" % flags)
        if not any("all bags cover" in ln and "(996 s)" in ln for ln in out):
            fails.append("clean pass: no common window line: %s" % out)
        if not any("svo, time not read" in ln for ln in out):
            fails.append("clean pass: the svo not listed")

        bad = os.path.join(tmp, "bad")
        bag(os.path.join(bad, "mobile_1", "b_mobile_1"), [(0, 0, 500, False), (1, 500.1, 1000, False)])
        bag(os.path.join(bad, "mobile_2", "b_mobile_2"),
            [(0, 0, 50, False), (3, 200, 600, False), (4, 600.1, 1000, False)], extra=[9])
        bag(os.path.join(bad, "infra_1", "b_infra_1"), [(0, 40, 700, False), (1, 700.1, 990, True)])
        flags = bag_times.check(bad, log=lambda *_: None)
        want = ["split file(s) _1, _2 missing", "150 s with nothing between",
                "b_mobile_2_9.mcap is on disk but not in metadata.yaml",
                "infra_1/b_infra_1 starts 40 s after the first bag",
                "b_infra_1_1.mcap has no summary"]
        for w in want:
            if not any(w in f for f in flags):
                fails.append("not flagged: %r in %s" % (w, flags))
        if any("_0" in f and "missing" in f for f in flags):
            fails.append("_0 wrongly called missing: %s" % flags)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    for f in fails:
        print("  FAIL  " + f)
    print("bag times " + ("FAILED" if fails else "ok"))
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
