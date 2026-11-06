"""From tracks to persons, and every camera box labelled from the persons' positions.

merge_colocated(): two tracks are one person when they stay close while both
are seen AND the cameras split between them: a duplicate arises when each
camera's one box of a person goes to one track or the other, so the two are
(almost) never seen by the same camera at the same moment; two real people
side by side are each seen by the same cameras, each camera showing two boxes.

label_boxes(): at each time, each person's position (smoothed by the tracker)
gives the box where they must appear in every camera; every camera's
detections are matched to those expected boxes (one optimal assignment per
camera). So a person has the same id in every camera, and a box's id changes
only when the person really moves away from it, not when that instant's
grouping of boxes changes.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment

from ..multiview import occluded_iou


class Person:
    """One person: one or more tracks of them (duplicates merged). Same interface as Track for drawing."""

    def __init__(self, pid, members):
        self.id, self.members = pid, members
        self.t0 = min(m.t0 for m in members)
        self.t1 = max(m.t1 for m in members)

    def position_at(self, times):
        P = np.stack([m.position_at(times) for m in self.members])          # (M, N, 2)
        with np.errstate(invalid="ignore"):
            return np.nanmean(P, axis=0) if len(self.members) > 1 else P[0]

    def samples(self):
        t = np.concatenate([m.samples()[0] for m in self.members])
        x = np.concatenate([m.samples()[1] for m in self.members])
        k = np.argsort(t)
        return t[k], x[k]

    def seen_at(self, T, within=0.5):
        t = self.samples()[0]
        return bool(np.any((t <= T) & (t > T - within)))


def _camera_sets(track, meas):
    """{time: set of cameras} of a track's camera measurements (meta["boxes"])."""
    out = {}
    for k in track.measurements:
        cams = (meas[k].meta or {}).get("boxes")
        if cams:
            out.setdefault(round(meas[k].t, 3), set()).update(cams)
    return out


def merge_colocated(tracks, meas=None, dist=0.6, min_overlap=1.0, step=0.2, seen_within=0.5, max_shared=0.2):
    """Tracks -> Persons. Two tracks are merged when, over at least `min_overlap` s in which both are
    seen, their median distance is under `dist` m and 80 % of the time under 1.5 x dist, and (with
    camera measurements, `meas`) at the moments both are measured, the cameras seeing both are at
    most `max_shared` of the cameras seeing either."""
    n = len(tracks)
    csets = [_camera_sets(tk, meas) if meas is not None else {} for tk in tracks]
    parent = list(range(n))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    upd = [tk.samples()[0] for tk in tracks]
    for i in range(n):
        for j in range(i + 1, n):
            lo, hi = max(tracks[i].t0, tracks[j].t0), min(tracks[i].t1, tracks[j].t1)
            if hi - lo < min_overlap:
                continue
            ts = np.arange(lo, hi, step)
            both = np.array([np.any((upd[i] <= t) & (upd[i] > t - seen_within)) and
                             np.any((upd[j] <= t) & (upd[j] > t - seen_within)) for t in ts])
            if both.sum() * step < min_overlap:
                continue
            d = np.linalg.norm(tracks[i].position_at(ts[both]) - tracks[j].position_at(ts[both]), axis=1)
            if not (np.median(d) < dist and np.mean(d < 1.5 * dist) >= 0.8):
                continue
            common = set(csets[i]) & set(csets[j])
            if common:
                shared = np.mean([len(csets[i][t] & csets[j][t]) / len(csets[i][t] | csets[j][t]) for t in common])
                if shared > max_shared:
                    continue                                   # the same cameras see both: two people
            parent[root(i)] = root(j)
    groups = {}
    for i in range(n):
        groups.setdefault(root(i), []).append(tracks[i])
    persons = []
    for members in groups.values():
        lead = max(members, key=lambda m: len(m.measurements))           # the id of the best-observed track
        persons.append(Person(lead.id, sorted(members, key=lambda m: m.t0)))
    return persons


def label_boxes(grid, persons, T, boxes, min_iou=0.3, seen_within=1.0):
    """Labels for every camera's detections at time T.

    boxes[c]: (M_c, 4) detections of camera c. -> [list of person id or None per detection, per camera].
    Only persons seen in the last `seen_within` s take part (a person carried over a long gap does
    not take someone else's box)."""
    act = [p for p in persons if p.t0 <= T <= p.t1 and p.seen_at(T, seen_within)]
    out = [[None] * len(b) for b in boxes]
    if not act:
        return out
    pos = np.array([p.position_at([T])[0] for p in act])
    ok = ~np.isnan(pos[:, 0])
    act, pos = [p for p, k in zip(act, ok) if k], pos[ok]
    if not act:
        return out
    cells, inside = grid.cells(pos)
    for c, b in enumerate(boxes):
        b = np.asarray(b, float).reshape(-1, 4)
        if not len(b):
            continue
        E = grid.boxes[c][:, cells]                                    # (S, K, 4) expected boxes
        I = occluded_iou(b[:, None, None, :], E[None]).max(1)          # (M, K)
        I[:, ~inside] = 0
        r, k = linear_sum_assignment(-I)
        for i, j in zip(r, k):
            if I[i, j] >= min_iou:
                out[c][i] = act[j].id
    return out
