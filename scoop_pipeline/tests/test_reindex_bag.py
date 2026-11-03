#!/usr/bin/env python
"""Self-tests for processing/reindex_bag.py: a split MCAP bag with no
metadata.yaml (the recorder killed while writing its last file) gets one that
the pipeline's readers take, the cut-off file recovered. Needs mcap, pyyaml.

    python tests/test_reindex_bag.py
"""
import os
import shutil
import sys
import tempfile
import traceback

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "processing"))

import reindex_bag                                                      # noqa: E402

FAILED = []
QOS = "- history: 3\n  depth: 0\n  reliability: 1\n  durability: 2\n"


def write_file(path, start_ns, n, finish=True):
    """A rosbag2-style MCAP file: two topics, n messages each, 10 ms apart."""
    from mcap.writer import CompressionType, Writer
    fh = open(path, "wb")
    w = Writer(fh, compression=CompressionType.ZSTD, chunk_size=256)
    w.start(profile="ros2", library="test")
    s = w.register_schema("std_msgs/msg/String", "ros2msg", b"string data")
    a = w.register_channel("/mobile_2/a", "cdr", s, {"offered_qos_profiles": QOS})
    b = w.register_channel("/mobile_2/ouster/metadata", "cdr", s, {"offered_qos_profiles": ""})
    for i in range(n):
        t = start_ns + i * 10_000_000
        w.add_message(a, t, b"\x00\x01\x00\x00" + bytes(40), t, i)
        w.add_message(b, t + 1, b"\x00\x01\x00\x00" + bytes(40), t + 1, i)
    if finish:
        w.finish()
    fh.close()                             # unfinished: the chunks written so far, no summary
    if not finish:                         # cut it where the recorder stopped
        size = os.path.getsize(path)
        with open(path, "r+b") as f:
            f.truncate(size - 7)


def make_bag(d, last_broken=True):
    bag = os.path.join(d, "raw", "20261103", "mapping", "mobile_2", "ed305_dataset_mapping")
    os.makedirs(bag)
    t0 = 1_793_000_000_000_000_000
    for k in range(12):                    # _0 .. _11: the plain sort puts _10 before _2
        write_file(os.path.join(bag, "ed305_dataset_mapping_%d.mcap" % k), t0 + k * 10**9, 50,
                   finish=not (last_broken and k == 11))
    return bag, t0


def test_split_bag_gets_metadata(d):
    bag, t0 = make_bag(d, last_broken=False)
    reindex_bag.reindex(bag, log=lambda *a: None)
    info = yaml.safe_load(open(os.path.join(bag, "metadata.yaml")))["rosbag2_bagfile_information"]
    assert info["relative_file_paths"][:3] == ["ed305_dataset_mapping_0.mcap",
                                               "ed305_dataset_mapping_1.mcap",
                                               "ed305_dataset_mapping_2.mcap"]
    assert len(info["files"]) == 12 and info["message_count"] == 12 * 100
    assert info["starting_time"]["nanoseconds_since_epoch"] == t0
    assert info["duration"]["nanoseconds"] == 11 * 10**9 + 49 * 10_000_000 + 1
    t = {x["topic_metadata"]["name"]: x for x in info["topics_with_message_count"]}
    assert t["/mobile_2/a"]["message_count"] == 600
    assert t["/mobile_2/a"]["topic_metadata"]["offered_qos_profiles"] == QOS
    assert t["/mobile_2/a"]["topic_metadata"]["type"] == "std_msgs/msg/String"
    # the pipeline's reader and recording finder take it
    from scoop.bag import open_bag
    r = open_bag(bag)
    assert sum(x.count for x in r.topics().values()) == 1200
    try:
        reindex_bag.reindex(bag, log=lambda *a: None)
    except SystemExit as e:
        assert "--force" in str(e)
    else:
        raise AssertionError("an existing metadata.yaml was overwritten without --force")


def test_cut_off_last_file_recovered(d):
    bag, t0 = make_bag(d, last_broken=True)
    last = os.path.join(bag, "ed305_dataset_mapping_11.mcap")
    assert reindex_bag.summary_of(last) is None
    try:
        reindex_bag.reindex(bag, recover_broken=False, log=lambda *a: None)
    except SystemExit as e:
        assert "recover" in str(e)
    else:
        raise AssertionError("--no-recover went on past a broken file")
    assert not os.path.exists(os.path.join(bag, "metadata.yaml"))
    reindex_bag.reindex(bag, log=lambda *a: None)
    assert os.path.isfile(last + ".unindexed")
    info = yaml.safe_load(open(os.path.join(bag, "metadata.yaml")))["rosbag2_bagfile_information"]
    n11 = info["files"][11]["message_count"]
    assert 0 < n11 <= 100, n11                       # what was written before the cut
    assert info["message_count"] == 11 * 100 + n11
    from scoop.bag import open_bag
    assert sum(x.count for x in open_bag(bag).topics().values()) == info["message_count"]


def check(name, fn, *args):
    try:
        fn(*args)
        print("  ok    %s" % name)
    except Exception:
        FAILED.append(name)
        print("  FAIL  %s" % name)
        traceback.print_exc()


def main():
    d = tempfile.mkdtemp(prefix="reindex_")
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    try:
        for name, fn in tests:
            args = []
            if fn.__code__.co_argcount == 1:
                args = [os.path.join(d, name)]
                os.makedirs(args[0])
            check(name, fn, *args)
    finally:
        shutil.rmtree(d, ignore_errors=True)
    print(f"\n{len(tests) - len(FAILED)}/{len(tests)} passed")
    sys.exit(1 if FAILED else 0)


if __name__ == "__main__":
    main()
