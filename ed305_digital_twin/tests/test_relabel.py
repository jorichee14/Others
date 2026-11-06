from pathlib import Path

import numpy as np

from ed305.camera import load_cameras
from ed305.multiview import Grid
from ed305.tracking.model import Measurement
from ed305.tracking.relabel import label_boxes, merge_colocated
from ed305.tracking.tracker import Track, Tracklet, TrackerConfig

HERE = Path(__file__).resolve().parents[1]


def walk(tid, start, v, t0=0.0, t1=10.0, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    cfg = TrackerConfig()
    ts = np.arange(t0, t1, 0.1)
    xy = np.asarray(start) + np.outer(ts - t0, v) + rng.normal(0, noise, (len(ts), 2))
    tr = Tracklet(0, Measurement(ts[0], "cams", xy[0], np.eye(2) * 0.01), 0, cfg)
    for k in range(1, len(ts)):
        x, P = tr.predict(ts[k], cfg)
        tr.update(ts[k], Measurement(ts[k], "cams", xy[k], np.eye(2) * 0.01), k, x, P, 1.0, cfg)
    return Track(tid, [tr])


def test_duplicates_merge_neighbours_do_not():
    a = walk("p1", (0, 0), (0.5, 0), noise=0.05, seed=1)
    dup = walk("p7", (1.25, 0.1), (0.5, 0), t0=2, t1=8, noise=0.05, seed=2)     # same person, 25 cm off
    b = walk("p2", (0, 1.0), (0.5, 0), noise=0.05, seed=3)                       # someone 1 m beside
    persons = merge_colocated([a, dup, b])
    ids = sorted(sorted(m.id for m in p.members) for p in persons)
    assert ids == [["p1", "p7"], ["p2"]]
    assert {p.id for p in persons} == {"p1", "p2"}                                 # the better-observed id


def test_boxes_get_the_person_in_every_camera():
    cams = load_cameras(HERE / "configs/cameras_nominal.yaml")
    grid = Grid(cams, (-4, 4, -5, 5), 0.1)
    people = {"p1": (-1.5, 1.0), "p2": (1.0, -2.0), "p3": (0.3, 0.6)}
    persons = merge_colocated([walk(k, xy, (0, 0)) for k, xy in people.items()])
    T = 5.0
    rng = np.random.default_rng(0)
    boxes, truth = [], []
    for c in range(len(cams)):
        bs, ts = [], []
        for k, xy in people.items():
            n, _ = grid.cells(np.array([xy]))
            b = grid.boxes[c, 2, n[0]]
            if np.isfinite(b).all():
                bs.append(b + rng.normal(0, 5, 4))
                ts.append(k)
        order = rng.permutation(len(bs))                         # detections in any order
        boxes.append(np.array(bs).reshape(-1, 4)[order])
        truth.append([ts[i] for i in order])
    labels = label_boxes(grid, persons, T, boxes)
    n_boxes = sum(len(t) for t in truth)
    right = sum(l == t for L, Tr in zip(labels, truth) for l, t in zip(L, Tr))
    assert n_boxes > 30 and right >= 0.95 * n_boxes


def test_neighbours_seen_by_the_same_cameras_stay_two():
    a = walk("p1", (0, 0), (0.3, 0), noise=0.02, seed=1)
    b = walk("p2", (0.0, 0.35), (0.3, 0), noise=0.02, seed=2)        # 35 cm apart all the time
    meas = {}
    for tk, cams in ((a, ["cam_1", "cam_2", "cam_3"]), (b, ["cam_1", "cam_2", "cam_4"])):
        for k in tk.measurements:
            meas[(tk.id, k)] = cams
    class M:
        def __init__(self, t, cams): self.t, self.meta = t, {"boxes": {c: (0, 0, 1, 1) for c in cams}}
    flat = []
    for tk, cams in ((a, ["cam_1", "cam_2", "cam_3"]), (b, ["cam_1", "cam_2", "cam_4"])):
        idx = []
        for tr in tk.tracklets:
            for i, t in enumerate(tr.times):
                flat.append(M(t, cams))
                idx.append(len(flat) - 1)
            tr.meas = idx
    assert len(merge_colocated([a, b], flat)) == 2                 # same cameras see both: two people
    for tr in b.tracklets:                                         # now the cameras split between them
        for k in tr.meas:
            flat[k].meta["boxes"] = {"cam_5": (0, 0, 1, 1), "cam_6": (0, 0, 1, 1)}
    assert len(merge_colocated([a, b], flat)) == 1
