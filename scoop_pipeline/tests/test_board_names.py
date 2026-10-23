#!/usr/bin/env python
"""Self-test for the boards' names: board_1..board_4 by dictionary, the names
passes before 20261015 measured them under ("was": anchor, anchor_b,
rs_anchor) read as the new ones, a board without its size refused, and an
infra camera's extrinsic yaml found in the infra node's own folder.

    python tests/test_board_names.py
"""
import json
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STAGES = os.path.join(ROOT, "map_stages")
sys.path[:0] = [ROOT, STAGES]

import pipeline_common as pc                                            # noqa: E402
from pipeline_boards import Board                                      # noqa: E402


def main():
    fails = []

    def eq(got, want, what):
        if got != want:
            fails.append("%s: %r, want %r" % (what, got, want))

    tmp = tempfile.mkdtemp(prefix="boards_")
    data = os.path.join(tmp, "data")
    bag = os.path.join(data, "work", "20261015", "survey", "site_1", "x_merged")
    cfg = pc.load_config(os.path.join(STAGES, "pipeline_config_survey_site_1_20261015.json"))
    cfg["dataset"]["bag"] = bag
    P = pc.Pipeline(cfg, STAGES)

    eq(sorted(k for k in P.cfg["boards"] if not k.startswith("_")),
       ["board_1", "board_2", "board_3", "board_4"], "boards")
    eq({k: P.cfg["boards"][k]["dictionary"] for k in ("board_1", "board_2", "board_3", "board_4")},
       {"board_1": "DICT_4X4_50", "board_2": "DICT_5X5_50", "board_3": "DICT_6X6_50",
        "board_4": "DICT_7X7_50"}, "dictionaries")
    eq(P.board_alias(), {"anchor": "board_1", "anchor_b": "board_1b", "rs_anchor": "board_2"},
       "aliases")
    eq([P.board_name(n) for n in ("anchor", "rs_anchor", "board_4", None)],
       ["board_1", "board_2", "board_4", None], "board_name")

    # a stored pose under the old names (the 20261015 mapping pass's anchor_frame.json)
    old = {"anchor": {"design": "anchor", "frame": "board", "xyz": [0, 0, 0], "qxyzw": [0, 0, 0, 1]},
           "anchor_b": {"design": "anchor", "frame": "board", "xyz": [1, 0, 0], "qxyzw": [0, 0, 0, 1]},
           "rs_anchor": {"design": "rs_anchor", "frame": "board_rs", "xyz": [2, 0, 0],
                         "qxyzw": [0, 0, 0, 1]}}
    r = P.renamed(old)
    eq(sorted(r), ["board_1", "board_1b", "board_2"], "renamed names")
    eq([r[k]["design"] for k in ("board_1", "board_1b", "board_2")],
       ["board_1", "board_1", "board_2"], "renamed designs")
    eq(r["board_2"]["frame"], "board_rs", "a stored pose keeps its TF frame")
    # the reference (old names) and this run's 05 (new names): each board once, the
    # reference's pose standing (same day, the boards did not move)
    run = {"board_1": dict(old["anchor"], design="board_1", xyz=[0.01, 0, 0]),
           "board_4": {"design": "board_4", "frame": "board_4", "xyz": [3, 0, 0],
                       "qxyzw": [0, 0, 0, 1]}}
    b = P.boards(old, run, log=lambda *_: None)
    eq(sorted(b), ["board_1", "board_1b", "board_2", "board_4"], "merged boards")
    eq((b["board_1"]["source"], b["board_1"]["xyz"]), ("reference", [0, 0, 0]), "board_1 kept")
    eq(b["board_4"]["source"].endswith("boards_survey_site_1_20261015.json"), True,
       "board_4 from this run")

    # the survey's stages name the new boards
    eq(P.cfg["05_anchor"]["boards_used"], ["board_1", "board_2", "board_3", "board_4"],
       "05 boards_used")
    eq([c.get("board") for c in P.cfg["07_build_cameras"]["cameras"] if c["name"] == "realsense"],
       ["board_4"], "07 realsense board")
    eq([s.get("board") for s in P.cfg["03_init"]["sensors"] if s["name"] == "zed"],
       ["board_1"], "03 zed board")

    # a board whose size is not set yet: refused, named
    try:
        Board("board_4", P.cfg["boards"]["board_4"])
        fails.append("board_4 without its size was accepted")
    except SystemExit as e:
        eq("board_4" in str(e) and "square_len" in str(e), True, "unmeasured board message")
    eq(Board("board_1", P.cfg["boards"]["board_1"]).square_len, 0.02, "board_1 size")

    # the arducam's yaml in the infra node's folder
    infra = os.path.join(data, "raw", "20261015", "survey", "site_1", "infra_1")
    os.makedirs(infra)
    open(os.path.join(infra, "extrinsic_arducam.yaml"), "w").write("x: 1\n")
    eq(P.infra_yaml("extrinsic_arducam.yaml", "arducam"),
       os.path.join(infra, "extrinsic_arducam.yaml"), "yaml in infra_1/")

    # a config with only the old names (no "was"): read as before
    Q = pc.Pipeline({"dataset": {"bag": bag}, "boards": {"anchor": {"dictionary": "DICT_4X4_50"}}},
                    STAGES)
    eq((Q.board_alias(), Q.board_name("anchor")), ({}, "anchor"), "old config unchanged")

    for f in fails:
        print("  FAIL  " + f)
    print("board names " + ("FAILED" if fails else "ok"))
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
