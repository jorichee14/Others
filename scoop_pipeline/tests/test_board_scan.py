#!/usr/bin/env python
"""Self-test for map_stages/board_scan.py: a stream of frames with board_4
(the 7x7, 9 x 7 squares of 2.45 cm) rendered 0.5 m away, then 0.7 m away, then absent,
then another dictionary's board (board_1): the scan reads a pose from the
near views, finds the markers but no corners at 0.7 m (a 1.75 cm marker
is 10 px there),
counts the empty frames absent and board_1's as another board's, and writes
the contact sheet and the annotated frames.

    python scoop_pipeline/tests/test_board_scan.py
"""
import json
import os
import shutil
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "map_stages"))

import cv2                                                   # noqa: E402
from pipeline_boards import bank_from_config                 # noqa: E402
import board_scan as BS                                      # noqa: E402

W, H = 848, 480
K = np.array([[427.7, 0, 423.1], [0, 427.1, 239.7], [0, 0, 1]])
SQ_PX = 120                                                   # the printed board's pixels per square


def board_image(design):
    bw, bh = design.sx * SQ_PX, design.sy * SQ_PX
    return design.board.generateImage((bw, bh)) if hasattr(design.board, "generateImage") \
        else design.board.draw((bw, bh))


def render(board_img, design, z, rng, x=0.0, rot=(0.5, 0.7, 0.1)):
    """The board `z` m ahead of the camera, centred and moved `x` m aside,
    tilted as a real one is (seen exactly face-on its pose is a PnP mirror
    ambiguity, which the detector rightly refuses), with pixel noise."""
    R = cv2.Rodrigues(np.array(rot))[0]
    half = np.array([design.sx * design.square_len / 2, design.sy * design.square_len / 2, 0])
    t = np.array([x, 0, z]) - R @ half
    Hm = K @ np.column_stack([R[:, 0], R[:, 1], t])
    S = np.diag([design.square_len / SQ_PX] * 2 + [1.0])
    img = cv2.warpPerspective(board_img, Hm @ S, (W, H), flags=cv2.INTER_AREA, borderValue=140)
    return np.clip(img.astype(np.float32) + rng.normal(0, 2.0, img.shape), 0, 255).astype(np.uint8)


def main():
    cfg = json.load(open(os.path.join(ROOT, "map_stages", "pipeline_config.json")))
    bank = bank_from_config(cfg, ["board_4"])
    b4 = bank.designs[0]
    b1 = bank_from_config(cfg, ["board_1"]).designs[0]
    D = np.zeros(5)
    rng = np.random.default_rng(1)
    big4, big1 = board_image(b4), board_image(b1)
    frames = []
    for k in range(5):                                       # 22 cm board 0.5 m away: read
        frames.append((100.0 + 0.2 * k, render(big4, b4, 0.5 + 0.01 * k, rng)))
    for k in range(3):                                       # 0.7 m away: markers 10 px, no corners
        frames.append((110.0 + 0.2 * k, render(big4, b4, 0.7 + 0.02 * k, rng, x=0.1)))
    for k in range(4):
        frames.append((120.0 + 0.2 * k, np.full((H, W), 140, np.uint8)))     # nothing
    frames.append((130.0, render(big1, b1, 0.5, rng)))                        # another board
    tmp = tempfile.mkdtemp(prefix="board_scan_")
    try:
        lines = []
        recs, summ = BS.scan(frames, bank, b4, K, D, out=tmp, save="found", log=lines.append)
        print("\n".join(lines))
        print(summ)
        v = summ["verdicts"]
        assert summ["frames"] == 13 and v["ok"] == 5, summ
        assert all(r["verdict"] == "ok" and r["n_mine"] >= b4.n_markers - 3 and 0.4 < r["range_m"] < 0.7
                   and r["reproj"] < 1.0 for r in recs[:5]), recs[:5]
        assert all(r["verdict"] in ("few_corners", "no_corners", "absent") for r in recs[5:8]), recs[5:8]
        assert all(r["verdict"] == "absent" and r["n_markers"] == 0 for r in recs[8:12]), recs[8:12]
        assert recs[12]["verdict"] == "absent" and recs[12]["n_markers"] == 0, recs[12]   # 4x4 is not 7x7
        assert summ["seen_from_s"] == 0.0 and 10 < summ["px_median"] < 22, summ
        assert summ["absent"] == 13 - summ["with_board_markers"], summ
        files = sorted(os.listdir(tmp))
        assert "contact_sheet.jpg" in files, files
        assert sum(f.endswith("_ok.jpg") for f in files) == 5, files
        sh = cv2.imread(os.path.join(tmp, "contact_sheet.jpg"))
        assert sh is not None and sh.shape[1] == 8 * 204, sh.shape
        # a window of the stream, every frame
        recs2, summ2 = BS.scan(frames, bank, b4, K, D, t_from=9.0, t_to=21.0, out=None, log=lambda *_: None)
        assert summ2["frames"] == 7 and summ2["verdicts"]["ok"] == 0, summ2
        print("ok board_scan: %d frames, %d poses (range %.2f m, %.0f px markers), %d tiny (%s), "
              "%d absent, another dictionary's board absent; sheet and %d annotated frames written"
              % (summ["frames"], v["ok"], recs[0]["range_m"], summ["px_median"],
                 summ["with_board_markers"] - v["ok"],
                 ", ".join(sorted({r["verdict"] for r in recs[5:8]})), summ["absent"],
                 sum(f.endswith(".jpg") for f in files) - 1))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
