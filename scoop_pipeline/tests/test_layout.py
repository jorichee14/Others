#!/usr/bin/env python
"""Self-test for scoop/layout.py: a pass one folder below the date
(20260924/mapping) and a pass at a site, two folders (20261015/survey/site_1),
through every place that names a pass from a path: the processed folder, the
tag, the raw folder, the merged bag, another pass named in the config, the
config found for a raw folder, mcap_convert's dataset folder.

    python tests/test_layout.py
"""
import json
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "map_stages"), os.path.join(ROOT, "datasets")]

from scoop import layout, session                                       # noqa: E402
from pipeline_common import Pipeline, processed_dir_for                  # noqa: E402
import pass_tf                                                           # noqa: E402
import mcap_convert                                                      # noqa: E402


class Plan:
    prefix = "mirc_dataset"


def main():
    D = os.path.join(tempfile.mkdtemp(prefix="layout_"), "data")
    fails = []

    def eq(got, want, what):
        if got != want:
            fails.append("%s: %r, want %r" % (what, got, want))

    cases = [  # raw pass folder, a bag of it in work/, tag
        ("20261015/survey/site_1", "20261015/survey/site_1/x_merged", "survey_site_1_20261015"),
        ("20260924/mapping", "20260924/mapping/x_merged", "mapping_20260924"),
        ("20261015/mapping", "20261015/mapping/mobile_2/x_retimed", "mapping_20261015"),
        ("20260924/survey_1", "20260924/survey_1/x_merged", "survey_1_20260924"),
    ]
    stages = os.path.join(D, "stages")
    os.makedirs(stages)
    for pas, bag, tag in cases:
        raw, bag = os.path.join(D, "raw", pas), os.path.join(D, "work", bag)
        eq(processed_dir_for(bag), os.path.join(D, "processed", pas), "processed of " + pas)
        P = Pipeline({"dataset": {"bag": bag}}, ".")
        eq(P.tag, tag, "tag of " + pas)
        eq(P.raw_dir, raw, "raw folder of " + pas)
        eq(session.session_name([], Plan, raw), "mirc_dataset_" + tag, "merged bag of " + pas)
        eq(mcap_convert.dataset_dir(bag, "replica", "map"),
           os.path.join(D, "processed", pas, "datasets", "mobile_1", "replica_map_" + tag),
           "dataset folder of " + pas)
        cfg = os.path.join(stages, "pipeline_config_%s.json" % tag)
        json.dump({"dataset": {"bag": bag}}, open(cfg, "w"))
    for pas, _, tag in cases:                       # each raw folder finds its own config
        got = pass_tf.find_config(os.path.join(D, "raw", pas), stages)
        eq(os.path.basename(got or ""), "pipeline_config_%s.json" % tag, "config of " + pas)
    # site_1 and site_2 of one survey are two passes, not one
    eq(processed_dir_for(os.path.join(D, "work", "20261015", "survey", "site_2", "b")),
       os.path.join(D, "processed", "20261015", "survey", "site_2"), "site_2")
    # another pass named in the config: same date, another date, a site
    P = Pipeline({"dataset": {"bag": os.path.join(D, "work", "20261015", "survey", "site_2", "b"),
                              "reference_pass": "mapping"}}, ".")
    eq(P.ref_file("anchor"), os.path.join(D, "processed", "20261015", "mapping", "frames",
                                          "anchor_frame.json"), "reference_pass mapping")
    eq(P.pass_dir("survey/site_1"), (os.path.join(D, "processed", "20261015", "survey", "site_1"),
                                     "survey_site_1_20261015"), "pass survey/site_1")
    eq(P.pass_dir("20260924/mapping")[1], "mapping_20260924", "pass 20260924/mapping")
    # a robot's GLIM export: odometry/<machine>/ of its pass, site included
    from scoop import recording
    for pas, machine in (("20261015/survey/site_1", "mobile_1"), ("20260924/mapping", "mobile_1")):
        rec = recording.Recording(recording.Path(D, "raw", pas, machine), None,
                                  recording.Path(D, "work", pas, machine))
        eq(str(rec.odometry), os.path.join(D, "processed", pas, "odometry", machine),
           "odometry of " + pas)
    # a folder named like a site below a machine is not a site
    eq(layout.pass_parts(os.path.join(D, "work", "20261015", "mapping", "mobile_2", "x"))[2],
       ["mapping"], "mobile_2 is not a site")
    for f in fails:
        print("  FAIL  " + f)
    print("layout " + ("FAILED" if fails else "ok"))
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
