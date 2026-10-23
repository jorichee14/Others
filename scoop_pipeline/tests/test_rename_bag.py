#!/usr/bin/env python3
"""scoop.bagwrite.rename_bag: a bag folder moved under another name keeps
its files named after the folder and metadata.yaml pointing at them (the
finalize step renames <bag>_final_tf to <bag>_final; clean_pass adds a
timestamp to a folder that exists already).

    python3 tests/test_rename_bag.py
"""
import os
import sys
import tempfile

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scoop import bag, rosmsg                                        # noqa: E402
from scoop.bagwrite import BagWriter, rename_bag                     # noqa: E402


def write(path):
    w = BagWriter(path)
    for k in range(5):
        t = 1_800_000_000_000_000_000 + k * 100_000_000
        w.write("/x/pose", rosmsg.pose_stamped(t, "map", np.zeros(3), [0, 0, 0, 1]), t)
    return w.close()


def main():
    d = tempfile.mkdtemp()
    src = os.path.join(d, "run_merged_final_tf")
    write(src)
    assert os.path.exists(os.path.join(src, "run_merged_final_tf_0.mcap"))

    # the finalize case: the folder gets a new name, the file and metadata follow
    dst = rename_bag(src, os.path.join(d, "run_merged_final"))
    files = sorted(os.listdir(dst))
    assert files == ["metadata.yaml", "run_merged_final_0.mcap"], files
    m = yaml.safe_load(open(os.path.join(dst, "metadata.yaml")))["rosbag2_bagfile_information"]
    assert m["relative_file_paths"] == ["run_merged_final_0.mcap"], m["relative_file_paths"]
    assert [f["path"] for f in m["files"]] == ["run_merged_final_0.mcap"]
    assert not os.path.exists(src)
    n = sum(1 for _ in bag.iter_mcap_records(bag.open_bag(str(dst)), ["/x/pose"]))
    assert n == 5, n                                   # still readable by the pipeline's reader

    # the clean_pass case: the same name under another parent is a plain move
    old = os.path.join(d, "_old")
    os.makedirs(old)
    dst2 = rename_bag(dst, os.path.join(old, "run_merged_final"))
    assert sorted(os.listdir(dst2)) == ["metadata.yaml", "run_merged_final_0.mcap"]

    # a folder that is not a bag is just moved
    plain = os.path.join(d, "notes")
    os.makedirs(plain)
    open(os.path.join(plain, "a.txt"), "w").write("x")
    rename_bag(plain, os.path.join(d, "notes_20261023"))
    assert os.listdir(os.path.join(d, "notes_20261023")) == ["a.txt"]
    print("rename_bag ok")


if __name__ == "__main__":
    main()
