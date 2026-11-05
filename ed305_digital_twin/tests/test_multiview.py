from pathlib import Path

import numpy as np
import pytest

from ed305.camera import load_cameras
from ed305.multiview import Grid, associate, box_iou, cell_index, label_cameras, occluded_iou, support

HERE = Path(__file__).resolve().parents[1]
BOUNDS = (-4.0, 4.0, -5.0, 5.0)


@pytest.fixture(scope="module")
def grid():
    return Grid(load_cameras(HERE / "configs/cameras_nominal.yaml"), BOUNDS, step=0.1)


def test_occluded_iou_allows_hidden_legs():
    exp = np.array([[[100, 100, 200, 400]]], float)               # (S=1, N=1, 4)
    upper = np.array([[[[100, 100, 200, 250]]]], float)            # legs behind a desk
    assert occluded_iou(upper, exp)[0, 0, 0] == pytest.approx(1.0)
    assert box_iou(upper[0, 0], exp[0])[0] == pytest.approx(0.5)
    shifted = np.array([[[[300, 100, 400, 400]]]], float)
    assert occluded_iou(shifted, exp)[0, 0, 0] == 0


def scene(grid, people, rng, miss=(), fp=(), noise=4.0, conf=0.85):
    """Detections every camera would give for people [(x, y, shape)]: the expected
    boxes plus pixel noise, minus (camera, person) pairs in miss, plus fp boxes."""
    boxes = [[] for _ in grid.cams]
    confs = [[] for _ in grid.cams]
    for k, (x, y, s) in enumerate(people):
        n = cell_index(grid, np.array([x, y]))
        for c in range(len(grid.cams)):
            b = grid.boxes[c, s, n]
            if not np.isfinite(b).all() or (c, k) in miss:
                continue
            boxes[c].append(b + rng.normal(0, noise, 4))
            confs[c].append(conf)
    for c, b, cf in fp:
        boxes[c].append(np.asarray(b, float))
        confs[c].append(cf)
    return [np.asarray(b).reshape(-1, 4) for b in boxes], [np.asarray(c) for c in confs]


def test_people_found_and_misses_reported(grid):
    rng = np.random.default_rng(0)
    people = [(-2.0, 1.5, 2), (0.5, -1.0, 0), (2.5, 3.0, 1), (1.0, 2.0, 2), (-1.0, -3.0, 0)]
    miss = {(0, 1), (5, 1), (13, 4)}
    boxes, confs = scene(grid, people, rng, miss=miss)
    sup = support(grid, boxes, 0.4)
    found = associate(grid, boxes, confs, sup=sup)
    assert len(found) == len(people)
    for x, y, _ in people:
        d = min(np.linalg.norm(p.xy - [x, y]) for p in found)
        assert d < 0.3
    labels = label_cameras(grid, boxes, confs, found, sup)
    kinds = {k for L in labels for _, k, _ in L.keep}
    assert kinds == {"confirmed"}
    assert not any(L.dropped for L in labels)
    # each removed detection is reported as missed where the camera saw that person clearly
    n_missed = sum(len(L.missed) for L in labels)
    assert 1 <= n_missed <= len(miss)


def test_low_confidence_confirmed_and_false_positive_dropped(grid):
    rng = np.random.default_rng(1)
    people = [(0.0, 2.0, 2)]
    boxes, confs = scene(grid, people, rng, conf=0.15)                    # detector unsure everywhere
    boxes[3] = np.vstack([boxes[3], [1500, 100, 1600, 400]])             # a lone false positive
    confs[3] = np.append(confs[3], 0.3)
    sup = support(grid, boxes, 0.4)
    found = associate(grid, boxes, confs, sup=sup)
    assert len(found) == 1 and found[0].views >= 8
    labels = label_cameras(grid, boxes, confs, found, sup)
    assert all(k == "confirmed" for L in labels for _, k, _ in L.keep)
    assert [r for _, r, _ in labels[3].dropped] == ["unconfirmed"]


def test_confident_single_view_kept_unless_others_see_the_spot(grid):
    rng = np.random.default_rng(2)
    # a confident box only camera 0 has, on an empty floor every camera sees: refuted
    n = cell_index(grid, np.array([0.0, 0.0]))
    fake = grid.boxes[0, 2, n]
    boxes, confs = scene(grid, [], rng, fp=[(0, fake, 0.9)])
    sup = support(grid, boxes, 0.4)
    found = associate(grid, boxes, confs, sup=sup)
    assert found == []
    labels = label_cameras(grid, boxes, confs, found, sup)
    assert [r for _, r, _ in labels[0].dropped] == ["refuted"]
    # the same box with refutation off is kept as single view
    labels = label_cameras(grid, boxes, confs, found, sup, refute_views=0)
    assert [k for _, k, _ in labels[0].keep] == ["single"]


def test_crossing_lines_make_no_ghosts(grid):
    rng = np.random.default_rng(3)
    people = [(x, y, 2) for x, y in [(-2, -2), (2, -2), (-2, 2), (2, 2), (0, 0), (0, 3), (0, -3)]]
    boxes, confs = scene(grid, people, rng)
    found = associate(grid, boxes, confs)
    assert len(found) == len(people)
