"""Person labels from many calibrated, synchronised cameras, without hand labels.

A person stands somewhere on the floor. Seen from a calibrated camera, a
person at floor point (x, y) has a predictable box: the projection of an
upright cylinder (sitting, mid, standing sizes in SHAPES). So for every cell
of a floor grid and every camera the expected boxes are computed once
(Grid). Per synchronised time step:

1. A pretrained detector runs on every camera at a low threshold.
2. associate(): a detection supports a cell when its box overlaps the
   cell's expected box (IoU, lower body allowed hidden behind a desk).
   Greedily, the cell with the most support from different cameras becomes
   a person and takes its detections (one per camera); repeat while a cell
   is supported by >= min_views cameras. Taking the detections out is what
   removes the "ghosts" where two people's view lines cross.
3. label_cameras(): per camera, boxes to keep and what is left out:
   - confirmed: a detection assigned to a person seen by >= min_views cameras,
     whatever its own score (this recovers people the detector was unsure of)
   - single: seen by this camera only, kept when its score is high, unless
     >= refute_views other cameras see that spot clearly and found nobody
   - dropped: low-score detections no other camera agrees with, and duplicates
   - missed: a person this camera sees clearly (whole box in the image, not
     behind a nearer person) but no detection; the frame is then not safe to
     use as is (left out, or the expected box added)

Room frame: z up, floor at z = 0, metres (the ED305 calibration's frame).
"""
from dataclasses import dataclass, field

import numpy as np

SHAPES = ((1.20, 0.35), (1.50, 0.30), (1.80, 0.28))   # (height, radius): sitting .. standing
RINGS = (0.0, 0.5, 1.0)                                # cylinder points at these fractions of the height
N_AROUND = 12
MIN_VISIBLE = 0.25    # a box less visible than this (after clipping to the image) does not count


def box_iou(a, b):
    """IoU of boxes a (..., 4) and b (..., 4), broadcast."""
    x0 = np.maximum(a[..., 0], b[..., 0])
    y0 = np.maximum(a[..., 1], b[..., 1])
    x1 = np.minimum(a[..., 2], b[..., 2])
    y1 = np.minimum(a[..., 3], b[..., 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    area = lambda r: (r[..., 2] - r[..., 0]) * (r[..., 3] - r[..., 1])
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.nan_to_num(inter / (area(a) + area(b) - inter))


def occluded_iou(det, exp, max_hidden=0.6):
    """IoU of a detection with an expected box whose lower part may be hidden.

    If the detection ends higher than the expected box (legs behind a desk or
    a chair), the expected box is cut there, but never more than max_hidden
    of its height. det (D, 1, 1, 4), exp (S, N, 4) -> (D, S, N)."""
    exp = np.broadcast_to(exp, np.broadcast_shapes(det.shape, exp.shape)).copy()
    h = exp[..., 3] - exp[..., 1]
    exp[..., 3] = np.clip(det[..., 3], exp[..., 1] + (1 - max_hidden) * h, exp[..., 3])
    return box_iou(det, exp)


class Grid:
    """Expected person boxes for every floor cell in every camera.

    boxes[c, s, n]: box (x0 y0 x1 y1, clipped to the image) of shape s at cell n
    in camera c, NaN where it is not in view; full[c, s, n]: the whole box is
    inside the image (so a miss there means something); dist[c, n]: floor
    distance from camera c to cell n (who is in front of whom)."""

    def __init__(self, cams, bounds, step=0.1, shapes=SHAPES):
        x0, x1, y0, y1 = bounds
        xs = np.arange(x0 + step / 2, x1, step)
        ys = np.arange(y0 + step / 2, y1, step)
        gx, gy = np.meshgrid(xs, ys)
        self.xy = np.stack([gx.ravel(), gy.ravel()], 1)
        self.cams, self.shapes, self.step = cams, shapes, step
        C, S, N = len(cams), len(shapes), len(self.xy)
        self.boxes = np.full((C, S, N, 4), np.nan)
        self.full = np.zeros((C, S, N), bool)
        self.dist = np.zeros((C, N))
        ang = np.linspace(0, 2 * np.pi, N_AROUND, endpoint=False)
        ring = np.stack([np.cos(ang), np.sin(ang)], 1)
        for c, cam in enumerate(cams):
            self.dist[c] = np.linalg.norm(self.xy - cam.center[:2], axis=1)
            for s, (H, r) in enumerate(shapes):
                xy = self.xy[:, None, None, :] + r * ring[None, None, :, :]           # (N, 1, A, 2)
                z = np.broadcast_to(np.array(RINGS)[None, :, None, None] * H, (N, len(RINGS), N_AROUND, 1))
                pts = np.concatenate([np.broadcast_to(xy, (N, len(RINGS), N_AROUND, 2)), z], -1).reshape(N, -1, 3)
                P = pts.shape[1]
                flat = pts.reshape(-1, 3)
                uv, z, r2 = cam.project(flat)
                inside, _ = cam.in_view(flat)
                ok_pt = (z > 0.1) & (r2 <= self._r2_lim(cam))
                uv, ok_pt, inside = uv.reshape(N, P, 2), ok_pt.reshape(N, P), inside.reshape(N, P)
                valid = ok_pt.all(1)                         # every point projects sensibly
                b = np.stack([uv[..., 0].min(1), uv[..., 1].min(1), uv[..., 0].max(1), uv[..., 1].max(1)], 1)
                clip = np.stack([np.clip(b[:, 0], 0, cam.width), np.clip(b[:, 1], 0, cam.height),
                                 np.clip(b[:, 2], 0, cam.width), np.clip(b[:, 3], 0, cam.height)], 1)
                area = lambda q: (q[:, 2] - q[:, 0]) * (q[:, 3] - q[:, 1])
                with np.errstate(invalid="ignore", divide="ignore"):
                    vis = valid & (area(clip) > MIN_VISIBLE * area(b))
                self.boxes[c, s, vis] = clip[vis]
                self.full[c, s] = vis & inside.all(1)

    @staticmethod
    def _r2_lim(cam):
        corners = np.array([[0, 0], [cam.width, 0], [0, cam.height], [cam.width, cam.height]], float)
        xn = (corners[:, 0] - cam.K[0, 2]) / cam.K[0, 0]
        yn = (corners[:, 1] - cam.K[1, 2]) / cam.K[1, 1]
        return 1.44 * np.max(xn * xn + yn * yn)


@dataclass
class Person:
    xy: np.ndarray
    shape: int                        # index into SHAPES
    dets: dict                        # camera index -> detection index
    score: float                      # sum of the assigned detections' confidences
    views: int = field(init=False)

    def __post_init__(self):
        self.views = len(self.dets)


def support(grid, boxes, iou_thr):
    """Per camera: (D, N) IoU of each detection with each cell's best shape, and that shape."""
    out = []
    for c, b in enumerate(boxes):
        if len(b) == 0:
            out.append((np.zeros((0, len(grid.xy))), np.zeros((0, len(grid.xy)), int)))
            continue
        iou = occluded_iou(np.asarray(b, float)[:, None, None, :], grid.boxes[c][None])   # (D, S, N)
        best = iou.argmax(1)
        m = np.take_along_axis(iou, best[:, None], 1)[:, 0]
        m[m < iou_thr] = 0
        out.append((m, best))
    return out


def associate(grid, boxes, confs, min_views=2, min_score=0.8, iou_thr=0.4, sup=None, min_sep=0.4):
    """People from detections in every camera. boxes[c]: (D_c, 4) pixels, confs[c]: (D_c,).

    No two people closer than min_sep metres: leftovers of a person whose boxes
    are off (calibration, sync) would otherwise make a second one next to it."""
    sup = sup if sup is not None else support(grid, boxes, iou_thr)
    N = len(grid.xy)
    free = [np.ones(len(b), bool) for b in boxes]
    open_cell = np.ones(N, bool)
    people = []
    while True:
        weight, conf_sum, views = np.zeros(N), np.zeros(N), np.zeros(N, int)
        best_det = []
        for c, (m, _) in enumerate(sup):
            if not free[c].any():
                best_det.append(None)
                continue
            w = m * np.asarray(confs[c])[:, None] * free[c][:, None]       # (D, N)
            j = w.argmax(0)
            wj = w[j, np.arange(N)]
            hit = wj > 0
            weight += wj
            conf_sum += np.where(hit, np.asarray(confs[c])[j], 0)
            views += hit
            best_det.append((j, hit))
        ok = (views >= min_views) & (conf_sum >= min_score) & open_cell
        if not ok.any():
            return people
        n = int(np.argmax(np.where(ok, weight, -1)))
        dets, shape_votes = {}, np.zeros(len(grid.shapes))
        for c, bd in enumerate(best_det):
            if bd is not None and bd[1][n]:
                d = int(bd[0][n])
                dets[c] = d
                free[c][d] = False
                shape_votes[sup[c][1][d, n]] += sup[c][0][d, n] * confs[c][d]
        people.append(Person(grid.xy[n].copy(), int(shape_votes.argmax()), dets, float(conf_sum[n])))
        open_cell &= np.linalg.norm(grid.xy - grid.xy[n], axis=1) >= min_sep


def covered(box, others):
    """Fraction of box covered by the union of others (rasterised on a coarse grid)."""
    if not len(others):
        return 0.0
    x = np.linspace(box[0], box[2], 16)
    y = np.linspace(box[1], box[3], 16)
    gx, gy = np.meshgrid(x, y)
    o = np.asarray(others)
    inside = ((gx[..., None] >= o[:, 0]) & (gx[..., None] <= o[:, 2])
              & (gy[..., None] >= o[:, 1]) & (gy[..., None] <= o[:, 3])).any(-1)
    return float(inside.mean())


def cell_index(grid, xy):
    return int(np.argmin(np.linalg.norm(grid.xy - xy, axis=1)))


def clearly_seen(grid, c, n, s, people, max_covered=0.5):
    """Camera c sees the whole of shape s at cell n, not hidden behind a nearer person."""
    if not grid.full[c, s, n]:
        return False
    nearer = [grid.boxes[c, p.shape, cell_index(grid, p.xy)] for p in people
              if grid.dist[c, cell_index(grid, p.xy)] < grid.dist[c, n] - 0.2]
    nearer = [b for b in nearer if np.isfinite(b).all()]
    return covered(grid.boxes[c, s, n], nearer) < max_covered


@dataclass
class CameraLabels:
    keep: list = field(default_factory=list)       # (box, kind, conf): kind confirmed / single / added
    dropped: list = field(default_factory=list)    # (box, reason, conf)
    missed: list = field(default_factory=list)     # expected boxes of people it should have found


def label_cameras(grid, boxes, confs, people, sup, single_conf=0.6, refute_views=3, dup_iou=0.5, present=None):
    """Per camera: which boxes to keep, which were dropped, which people were missed.

    present[c] False: camera c had no frame at this time step; it neither
    refutes nor misses anything."""
    present = [True] * len(boxes) if present is None else present
    out = [CameraLabels() for _ in boxes]
    assigned = [dict() for _ in boxes]            # det -> person index
    for k, p in enumerate(people):
        for c, d in p.dets.items():
            assigned[c][d] = k
    for c, (b, cf) in enumerate(zip(boxes, confs)):
        if not present[c]:
            continue
        b = np.asarray(b, float).reshape(-1, 4)
        taken = [b[d] for d in assigned[c]]
        for d in range(len(b)):
            if d in assigned[c]:
                out[c].keep.append((b[d], "confirmed", float(cf[d])))
                continue
            if taken and box_iou(b[d][None], np.asarray(taken)).max() >= dup_iou:
                out[c].dropped.append((b[d], "duplicate", float(cf[d])))
                continue
            if cf[d] < single_conf:
                out[c].dropped.append((b[d], "unconfirmed", float(cf[d])))
                continue
            m, best = sup[c]
            cells = np.flatnonzero(m[d] > 0)
            if len(cells) and refute_views:
                seen_by = [sum(clearly_seen(grid, o, n, best[d, n], people) for o in range(len(boxes)) if o != c and present[o])
                           for n in cells[:: max(1, len(cells) // 20)]]
                if min(seen_by) >= refute_views:
                    out[c].dropped.append((b[d], "refuted", float(cf[d])))
                    continue
            out[c].keep.append((b[d], "single", float(cf[d])))
        kept = np.asarray([k[0] for k in out[c].keep]).reshape(-1, 4)
        for k, p in enumerate(people):
            if c in p.dets:
                continue
            n = cell_index(grid, p.xy)
            exp = grid.boxes[c, p.shape, n]
            if len(kept) and box_iou(exp[None], kept).max() >= 0.2:
                continue                          # a kept box is already there (single view, off a bit)
            if clearly_seen(grid, c, n, p.shape, [q for j, q in enumerate(people) if j != k]):
                out[c].missed.append(exp.copy())
    return out
