#!/usr/bin/env python
"""Self-tests for a single ArUco marker as a board ("type": "aruco" in the
boards registry, pipeline_boards.ArucoMarker): the registry entry makes a
marker, it is found in a rendered view and its pose comes back, it shares a
bank with a ChArUco board of the same dictionary. Needs numpy, opencv.

    python tests/test_aruco_marker.py
"""
import json
import os
import sys
import traceback

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "map_stages"))

import pipeline_boards as pb                                            # noqa: E402

FAILED = []
CFG = {"type": "aruco", "dictionary": "DICT_4X4_50", "marker_id": 43,
       "marker_len": 0.150, "tag_len": 0.200}
K = np.array([[428.0, 0, 423.0], [0, 427.6, 239.7], [0, 0, 1]])
D = np.zeros(5)


def render(R, t, marker_id=43):
    """The 200 mm tag (150 mm black square) at pose (R, t) in an 848x480 view."""
    dic = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    tag = np.full((1600, 1600), 255, np.uint8)
    tag[200:1400, 200:1400] = cv2.aruco.generateImageMarker(dic, marker_id, 1200)
    s = 0.200 / 1600
    src = np.float32([[0, 0], [1600, 0], [1600, 1600], [0, 1600]])
    obj = np.array([[(u - 200) * s, (v - 200) * s, 0.0] for u, v in src])
    ip, _ = cv2.projectPoints(obj, cv2.Rodrigues(R)[0], t, K, D)
    H = cv2.getPerspectiveTransform(src, ip.reshape(-1, 2).astype(np.float32))
    return cv2.warpPerspective(tag, H, (848, 480), flags=cv2.INTER_AREA, borderValue=128)


def test_registry_entry_is_a_marker():
    b = pb.Board("marker_43", CFG)
    assert isinstance(b, pb.ArucoMarker) and b.id_set == {43}
    assert b.size_m == (0.150, 0.150)
    assert np.allclose(b.frame_fix("ros", "center")[:3, 3], [0.075, 0.075, 0.0])
    assert "id 43" in b.describe()
    assert not isinstance(pb.Board("board_1", json.load(open(os.path.join(
        ROOT, "map_stages", "pipeline_config.json")))["boards"]["board_1"]), pb.ArucoMarker)


def test_pose_of_a_rendered_marker():
    b = pb.Board("marker_43", CFG)
    R = cv2.Rodrigues(np.array([0.0, np.radians(25), 0.0]))[0]
    t = np.array([0.1, 0.05, 1.8])
    d = b.pose_from_corners(*b.corners_full(render(R, t)), K, D)
    assert d is not None, b.reject_str()
    dt = np.linalg.norm(d.T[:3, 3] - t)
    dr = np.degrees(np.arccos(np.clip((np.trace(d.T[:3, :3].T @ R) - 1) / 2, -1, 1)))
    assert dt < 0.03 and dr < 5.0, (dt, dr)              # one marker: cm and degrees
    assert b.corners_full(render(R, t, marker_id=7)) == (None, None)


def test_in_a_bank_with_a_charuco_board():
    reg = json.load(open(os.path.join(ROOT, "map_stages", "pipeline_config.json")))["boards"]
    b = pb.Board("marker_43", CFG)
    bank = pb.BoardBank([pb.Board("board_1", reg["board_1"]), b])   # both DICT_4X4_50
    R = cv2.Rodrigues(np.array([0.0, np.radians(-20), 0.0]))[0]
    out = bank.detect(render(R, np.array([-0.1, 0.0, 1.5])), K, D)
    assert [x.design for x in out] == ["marker_43"], out


def check(name, fn):
    try:
        fn()
        print("  ok    %s" % name)
    except Exception:
        FAILED.append(name)
        print("  FAIL  %s" % name)
        traceback.print_exc()


def main():
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    for name, fn in tests:
        check(name, fn)
    print(f"\n{len(tests) - len(FAILED)}/{len(tests)} passed")
    sys.exit(1 if FAILED else 0)


if __name__ == "__main__":
    main()
