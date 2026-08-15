#!/usr/bin/env python3
"""
Semantic machinery shared by stage [6] detect and stage [7] synthesize.

Everything here is BACKEND-FREE (numpy / open3d / scipy only). The GPU handles
live in 01_build_map.py, which cannot be imported -- a module name starting
with a digit is not an identifier -- so the split is deliberate rather than
stylistic: the parts that must reach `xp()` and `project_visible()` stay in the
stage script, and the parts that are pure data reduction live here where they
can be unit-tested without a bag, a card or a camera.

Three pieces:

  Detector          ultralytics YOLO-seg, wrapped so the rest of the pipeline
                    sees plain dicts and a stub can stand in for it in tests.

  VoteAccumulator   per-map-point class evidence, fused across frames. A single
                    frame's mask ALWAYS bleeds past the silhouette onto the
                    surface behind; the same point only accumulates consistent
                    votes if it really belongs to the object. This is the
                    multi-view redundancy that makes the 3D labels usable when
                    the 2D masks are not.

  planes            RANSAC peel + orientation classification, giving floor /
                    ceiling / wall / support. YOLO has no class for any of
                    them, and they are what make "the TV is ON A WALL" a
                    computable predicate rather than a guess.
"""

import json
import numpy as np
import open3d as o3d


# =============================================================================
# DETECTOR
# =============================================================================
class Detector:
    """YOLO11-seg through ultralytics, normalised to a list of dicts.

    Masks, not boxes. A box lifted into 3D takes the whole frustum column and
    drags in every surface behind the object; a mask takes the silhouette,
    which after erosion and the depth gate is tight enough that the surviving
    error is what multi-view voting is for.
    """

    def __init__(self, weights, conf=0.35, iou=0.6, imgsz=960, classes=None,
                 exclude=None, device=None, verbose=False):
        try:
            from ultralytics import YOLO
        except ImportError:
            raise SystemExit(
                "detect needs ultralytics:  pip install ultralytics\n"
                "  (and see assets/README.md for fetching the -seg weights)")
        self.model = YOLO(weights)
        self.names = dict(self.model.names)
        self.conf = float(conf)
        self.iou = float(iou)
        self.imgsz = int(imgsz)
        self.device = device
        self.verbose = bool(verbose)
        # `classes` allows, `exclude` denies, and denial wins. Filtering here
        # rather than after inference is not just tidier -- ultralytics skips
        # mask generation for suppressed classes, so excluding `person` on a
        # walk-through with people in half the frames is a real saving on top
        # of keeping them out of the map.
        self.classes = None
        allow = set(self.names.values())
        if classes:
            miss = set(classes) - allow
            if miss:
                print(f"    ! detect.classes not in the model: {sorted(miss)}")
            allow &= set(classes)
        if exclude:
            miss = set(exclude) - set(self.names.values())
            if miss:
                print(f"    ! detect.exclude not in the model: {sorted(miss)}")
            allow -= set(exclude)
        if allow != set(self.names.values()):
            self.classes = sorted(i for i, n in self.names.items()
                                  if n in allow)
            if not self.classes:
                raise SystemExit("detect.classes/exclude leave no classes at "
                                 "all; nothing could ever be detected")

    def describe(self):
        return (f"YOLO-seg {len(self.names)} classes, conf={self.conf} "
                f"iou={self.iou} imgsz={self.imgsz}"
                + (f", restricted to {len(self.classes)} classes"
                   if self.classes else ""))

    def __call__(self, img_bgr):
        r = self.model.predict(img_bgr, conf=self.conf, iou=self.iou,
                               imgsz=self.imgsz, classes=self.classes,
                               # full-resolution masks: the alternative is a
                               # 160x160 mask upsampled by us, which blurs the
                               # silhouette that erosion is trying to trim
                               retina_masks=True, verbose=self.verbose,
                               device=self.device)[0]
        if r.masks is None or len(r.boxes) == 0:
            return []
        masks = r.masks.data.cpu().numpy() > 0.5
        cls = r.boxes.cls.cpu().numpy().astype(int)
        conf = r.boxes.conf.cpu().numpy().astype(np.float32)
        out = []
        for i in range(len(cls)):
            out.append({"cls_id": int(cls[i]),
                        "name": self.names.get(int(cls[i]), str(cls[i])),
                        "conf": float(conf[i]),
                        "mask": masks[i]})
        return out


def erode_mask(mask, px):
    """Shrink a boolean mask by `px`, cheaply and without a cv2 dependency.

    The silhouette is where a mask is wrong: the boundary pixel sits on the
    object in one frame and on the wall 3 m behind it in the next. Trimming a
    couple of pixels costs a sliver of the object and removes the single
    largest source of mislabelled points.
    """
    if px <= 0:
        return mask
    m = mask
    for _ in range(int(px)):
        e = m.copy()
        e[1:, :] &= m[:-1, :]
        e[:-1, :] &= m[1:, :]
        e[:, 1:] &= m[:, :-1]
        e[:, :-1] &= m[:, 1:]
        m = e
    return m


def depth_gate(z, max_spread, min_keep=8, min_tol=0.04):
    """Keep the near-depth population of a masked point set.

    Even a perfect mask contains background: an object with a hole in it (a
    chair between its legs, a plant between leaves) shows the wall THROUGH the
    mask, and those points are genuinely inside the silhouette. Depth separates
    them -- the object occupies a narrow depth band, the leak-through does not.

    Median + MAD rather than mean + sigma, because the contaminant can be a
    large fraction of the sample and would drag a mean straight into the gap.

    `min_tol` keeps a very clean sample from gating itself down to nothing, and
    has to stay SMALL. It only binds when the MAD is near zero, i.e. when the
    object is flat and seen face-on -- a picture, a monitor, a wall-mounted TV
    standing 6 cm proud of its wall. Those are exactly the cases where the gate
    must be tight, so a comfortable-sounding 10 cm floor would wave through the
    wall behind every one of them. A bulky object never reaches the floor: its
    own depth spread sets the tolerance.
    """
    if z.size < min_keep:
        return np.ones(z.size, bool)
    med = float(np.median(z))
    mad = float(np.median(np.abs(z - med)))
    tol = max(3.0 * 1.4826 * mad, float(min_tol))   # 1.4826*MAD ~ sigma
    return z < med + min(tol, float(max_spread))


# =============================================================================
# VOTE FUSION
# =============================================================================
class VoteAccumulator:
    """Per-(map point, class) evidence, compacted so memory stays bounded.

    Votes arrive one detection at a time and are buffered as flat (key, weight,
    frame) triples with key = point_index * n_classes + class_id, then folded
    into running totals by the same sort-and-segment reduction the rest of the
    pipeline uses. A dense (N_points x N_classes) table would be 3 GB for a
    10 M-point map at 80 COCO classes; the sparse form costs only what was
    actually observed.
    """

    def __init__(self, n_points, n_classes, compact_at=20_000_000):
        self.n = int(n_points)
        self.nc = int(n_classes)
        self.compact_at = int(compact_at)
        self._k = []
        self._w = []
        self._buf = 0
        self.k = None       # sorted unique keys
        self.w = None       # summed weight
        self.f = None       # distinct frames

    def add(self, idx, cls_id, weight):
        """One detection: map point indices, its class, per-point weights."""
        if len(idx) == 0:
            return
        self._k.append(np.asarray(idx, np.int64) * self.nc + int(cls_id))
        self._w.append(np.asarray(weight, np.float32))
        self._buf += len(idx)
        if self._buf >= self.compact_at:
            self._compact()

    def _compact(self):
        if not self._k:
            return
        k = np.concatenate(self._k)
        w = np.concatenate(self._w)
        f = np.ones(k.size, np.int32)
        self._k = []; self._w = []; self._buf = 0
        if self.k is not None:
            k = np.concatenate((self.k, k))
            w = np.concatenate((self.w, w))
            f = np.concatenate((self.f, f))
        order = np.argsort(k, kind="stable")
        k = k[order]; w = w[order]; f = f[order]
        start = np.flatnonzero(np.concatenate(([True], k[1:] != k[:-1])))
        self.k = k[start]
        self.w = np.add.reduceat(w, start)
        self.f = np.add.reduceat(f, start)

    def resolve(self, min_frames=3, min_ratio=0.5):
        """Fuse to one label per point.

        Returns (cls, conf, frames) as full-length arrays; cls is -1 where the
        evidence did not clear the gates.

        min_frames is the real discriminator. Silhouette bleed lands on a
        different background point every frame because the camera moves, so a
        bled point collects one vote; the object collects one per view of it.
        min_ratio then rejects points two classes disagree about -- the seam
        where a chair meets the table it is pushed under.
        """
        self._compact()
        cls = np.full(self.n, -1, np.int16)
        conf = np.zeros(self.n, np.float32)
        frames = np.zeros(self.n, np.int32)
        if self.k is None or self.k.size == 0:
            return cls, conf, frames

        pid = self.k // self.nc
        start = np.flatnonzero(np.concatenate(([True], pid[1:] != pid[:-1])))
        total = np.add.reduceat(self.w, start)

        # winner per point: lexsort puts each point's classes in ascending
        # weight order, so the LAST row of every group is its argmax. One sort
        # instead of a per-group reduction we would have to write by hand.
        o = np.lexsort((self.w, pid))
        p2 = pid[o]
        last = o[np.flatnonzero(np.concatenate((p2[1:] != p2[:-1], [True])))]

        pts = pid[last]
        best_w = self.w[last]
        best_f = self.f[last]
        best_c = (self.k[last] % self.nc).astype(np.int16)
        ratio = best_w / np.maximum(total, 1e-9)

        ok = (best_f >= int(min_frames)) & (ratio >= float(min_ratio))
        cls[pts[ok]] = best_c[ok]
        conf[pts[ok]] = ratio[ok]
        frames[pts[ok]] = best_f[ok]
        return cls, conf, frames


class InstanceTracker:
    """Turns per-frame detections into persistent instance ids, incrementally.

    Per-point class labels alone cannot separate two chairs standing next to
    each other -- they are one connected blob of the same class, and a purely
    spatial clustering would merge them. The detector already separated them in
    every image it saw them in, so that separation is carried through instead
    of being rediscovered: a new detection joins the existing instance it most
    overlaps with, provided the overlap is a real fraction of it.

    One pass, O(points observed), no all-pairs comparison across frames.
    """

    def __init__(self, n_points, link_frac=0.30):
        self.owner = np.full(n_points, -1, np.int32)
        self.link_frac = float(link_frac)
        self.cls = []          # instance -> class id
        self.hits = []         # instance -> detections merged

    def add(self, idx, cls_id):
        if len(idx) == 0:
            return -1
        cur = self.owner[idx]
        lab = -1
        seen = cur[cur >= 0]
        if seen.size:
            u, c = np.unique(seen, return_counts=True)
            # only ever link to an instance of the SAME class: a chair mask
            # overlapping the table it is under must not swallow the table
            same = [i for i in range(len(u)) if self.cls[u[i]] == cls_id]
            if same:
                j = max(same, key=lambda i: c[i])
                if c[j] >= self.link_frac * len(idx):
                    lab = int(u[j])
        if lab < 0:
            lab = len(self.cls)
            self.cls.append(int(cls_id))
            self.hits.append(0)
        self.owner[idx] = lab
        self.hits[lab] += 1
        return lab


def build_instances(pts, cls, conf, owner, inst_cls, min_points=60,
                    eps=0.12, min_frames_seen=2, hits=None):
    """Reconcile voted labels with tracked instances into final objects.

    The tracker's ownership is provisional -- it was assigned before the votes
    were fused, and a point whose final class disagrees with its instance is
    evidence the tracker was wrong about that point, not about the instance. So
    membership is intersected with the fused label first, and only then is each
    instance split by DBSCAN, which strips the trailing bleed that survived
    every earlier gate.
    """
    out = []
    n_inst = len(inst_cls)
    if n_inst == 0:
        return out
    order = np.argsort(owner, kind="stable")
    ow = owner[order]
    first = np.flatnonzero(np.concatenate(([True], ow[1:] != ow[:-1])))
    bounds = dict()
    ends = np.concatenate((first[1:], [ow.size]))
    for a, b in zip(first, ends):
        if ow[a] >= 0:
            bounds[int(ow[a])] = order[a:b]

    for lab in range(n_inst):
        idx = bounds.get(lab)
        if idx is None or idx.size < min_points:
            continue
        if hits is not None and hits[lab] < min_frames_seen:
            continue
        c = inst_cls[lab]
        idx = idx[cls[idx] == c]              # fused label wins over the tracker
        if idx.size < min_points:
            continue
        sub = o3d.geometry.PointCloud()
        sub.points = o3d.utility.Vector3dVector(pts[idx])
        lb = np.asarray(sub.cluster_dbscan(eps=eps, min_points=10,
                                           print_progress=False))
        if lb.size == 0:
            continue
        keep = lb >= 0
        if not keep.any():
            continue
        # the largest connected component IS the object; the rest is bleed that
        # happens to share a class, e.g. the wall patch behind a monitor
        u, cnt = np.unique(lb[keep], return_counts=True)
        idx = idx[keep][lb[keep] == u[np.argmax(cnt)]]
        if idx.size < min_points:
            continue
        out.append({"instance": len(out), "cls_id": int(c), "idx": idx,
                    "conf": float(np.mean(conf[idx])),
                    "n_frames": int(hits[lab]) if hits is not None else 0})
    return out


# =============================================================================
# STRUCTURE:  floor / ceiling / wall / support
# =============================================================================
def fit_plane_models(pts, dist=0.04, min_points=20000, max_planes=12,
                     fit_voxel=0.05):
    """RANSAC plane peel -> normalised [nx, ny, nz, d] models, largest first.

    Planes are FITTED on a voxel-downsampled copy. RANSAC on 40 M points costs
    minutes per plane and buys nothing: a 5 cm sample pins a wall's plane to
    well under the sensor's own range noise.
    """
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(pts)
    small = pc.voxel_down_sample(fit_voxel) if fit_voxel > 0 else pc
    work = np.asarray(small.points)
    live = np.arange(len(work))
    scale = max(len(work) / max(len(pts), 1), 1e-9)
    need = max(int(min_points * scale), 100)
    models = []
    for _ in range(int(max_planes)):
        if live.size < need:
            break
        sub = o3d.geometry.PointCloud()
        sub.points = o3d.utility.Vector3dVector(work[live])
        try:
            model, inl = sub.segment_plane(dist, 3, 1000)
        except RuntimeError:
            break
        if len(inl) < need:
            break
        m = np.asarray(model, float)
        models.append(m / np.linalg.norm(m[:3]))
        live = np.delete(live, inl)
    return models


def assign_planes(pts, models, dist=0.04, min_points=20000):
    """Attach full-resolution points to plane models -> [(model, idx), ...].

    Split out from the fitting so a resume can rebuild the structure from the
    handful of numbers in structure.json instead of re-running RANSAC.

    A point near a wall/floor join is within tolerance of both planes; first
    match wins, and models arrive largest-first, so the dominant surface claims
    the seam rather than the two fighting over it.

    Indices are into the ORIGINAL array -- unlike stage 01's flatten(), which
    peels with select_by_index and loses the mapping. That is fine when the
    output is a new cloud and useless when the question is "which points are
    floor".
    """
    out = []
    unclaimed = np.ones(len(pts), bool)
    for m in models:
        m = np.asarray(m, float)
        r = np.abs(pts @ m[:3] + m[3])
        idx = np.flatnonzero((r < dist) & unclaimed)
        if idx.size < min_points:
            continue
        unclaimed[idx] = False
        out.append((m, idx))
    return out


def extract_planes(pts, dist=0.04, min_points=20000, max_planes=12,
                   fit_voxel=0.05):
    """fit_plane_models + assign_planes."""
    return assign_planes(pts, fit_plane_models(pts, dist, min_points,
                                               max_planes, fit_voxel),
                         dist, min_points)


class Structure:
    """Classified planes plus the queries stage [7] asks of them."""

    def __init__(self, planes, pts, tol_deg=15.0, min_wall_height=0.8,
                 support_range=(0.35, 1.30)):
        up = np.array([0.0, 0.0, 1.0])
        ct = np.cos(np.deg2rad(tol_deg))
        st = np.sin(np.deg2rad(tol_deg))
        self.planes = []
        for model, idx in planes:
            n = model[:3]
            c = abs(float(n @ up))
            p = pts[idx]
            z = float(np.median(p[:, 2]))
            rec = {"model": model, "idx": idx, "n": n, "d": float(model[3]),
                   "z": z, "n_points": int(idx.size),
                   "centroid": p.mean(0),
                   "lo": p.min(0), "hi": p.max(0), "kind": "other"}
            if c >= ct:
                rec["kind"] = "horizontal"
            elif c <= st:
                rec["kind"] = "wall" if (p[:, 2].max() - p[:, 2].min()) \
                    >= min_wall_height else "other"
            self.planes.append(rec)

        hor = [p for p in self.planes if p["kind"] == "horizontal"]
        if hor:
            zmin = min(p["z"] for p in hor)
            zmax = max(p["z"] for p in hor)
            for p in hor:
                # floor and ceiling are the extremes; a horizontal plane in
                # between at sitting/standing height is furniture -- a table or
                # shelf top, which is exactly where a vase or laptop belongs
                if p["z"] - zmin < 0.25:
                    p["kind"] = "floor"
                elif zmax - p["z"] < 0.25 and zmax - zmin > 1.5:
                    p["kind"] = "ceiling"
                elif support_range[0] <= p["z"] - zmin <= support_range[1]:
                    p["kind"] = "support"
                else:
                    p["kind"] = "other"

        self.floors = [p for p in self.planes if p["kind"] == "floor"]
        self.walls = [p for p in self.planes if p["kind"] == "wall"]
        self.supports = [p for p in self.planes if p["kind"] == "support"]
        self.ceilings = [p for p in self.planes if p["kind"] == "ceiling"]

    def summary(self):
        return (f"{len(self.floors)} floor, {len(self.walls)} wall, "
                f"{len(self.supports)} support, {len(self.ceilings)} ceiling, "
                f"{sum(1 for p in self.planes if p['kind'] == 'other')} other")

    def labels(self, n_points):
        """Per-point structure label: 0 none, 1 floor, 2 wall, 3 ceiling, 4 support."""
        lab = np.zeros(n_points, np.uint8)
        for code, group in ((3, self.ceilings), (4, self.supports),
                            (2, self.walls), (1, self.floors)):
            for p in group:
                lab[p["idx"]] = code
        return lab

    @staticmethod
    def _plane_z(p, xy):
        """Height of plane p above xy. Near-vertical planes have none."""
        n, d = p["n"], p["d"]
        if abs(n[2]) < 1e-6:
            return None
        return float(-(d + n[0] * xy[0] + n[1] * xy[1]) / n[2])

    def support_under(self, pts, prefer="floor"):
        """The plane an object rests on: (plane, z) or (None, fallback z).

        Snapping to the nearest plane by distance is wrong -- a chair beside a
        table is nearer the table top than the floor. What matters is which
        surface is directly BENEATH the footprint and below the object, so the
        candidates are filtered by that first and the highest survivor wins
        (a laptop on a desk in a room picks the desk, not the floor).
        """
        xy = pts[:, :2].mean(0)
        base = float(np.percentile(pts[:, 2], 2.0))
        cands = []
        # "any" means the asset can stand on either, so it must SEE both --
        # a potted plant is declared "any" precisely because it belongs on a
        # table as readily as on the floor, and restricting it to floors would
        # sink every one standing on furniture down to ground level
        groups = (self.supports + self.floors
                  if prefer in ("surface", "any") else self.floors)
        for p in groups:
            z = self._plane_z(p, xy)
            if z is None or z > base + 0.15:
                continue
            # the plane must actually extend under the object, not merely be
            # coplanar with something on the far side of the room
            if not (p["lo"][0] - 0.3 <= xy[0] <= p["hi"][0] + 0.3
                    and p["lo"][1] - 0.3 <= xy[1] <= p["hi"][1] + 0.3):
                continue
            cands.append((z, p))
        if not cands:
            return None, base
        z, p = max(cands, key=lambda t: t[0])
        return p, z

    def trim_wall_skirt(self, idx, pts, tol=0.025, min_standoff=0.03):
        """Drop the ring of WALL an instance's mask bled onto.

        The depth gate cannot do this job. A TV 6 cm proud of its wall, seen
        from 2.6 m and off to one side, has 40 cm of depth spread across its
        own face -- so the tolerance the gate derives from that spread is many
        times the standoff it would need to resolve, and the wall pixels just
        outside the silhouette survive at the object's own depth. Multi-view
        voting does not save it either: the bleed ring sits in nearly the same
        world place from every viewpoint that can see the object at all, so it
        accumulates votes exactly like the object does.

        What DOES separate them is the thing neither test has: the wall's
        plane. If the bulk of an instance stands proud of a wall, then the
        points sitting exactly IN that wall are not part of it.

        Self-disabling where it would be wrong: an object genuinely flush with
        the wall -- a poster, a flat panel with no standoff -- has a median
        residual near zero, fails the min_standoff test, and is left alone.
        """
        if len(idx) == 0 or not self.walls:
            return idx, None
        p = pts[idx]
        best = None
        for w in self.walls:
            r = np.abs(p @ w["n"] + w["d"])
            frac = float((r < 0.30).mean())
            if frac > 0.6 and (best is None or frac > best[0]):
                best = (frac, w, r)
        if best is None:
            return idx, None
        _, w, r = best
        standoff = float(np.median(r))
        if standoff < float(min_standoff):
            return idx, None
        keep = r > min(float(tol), 0.5 * standoff)
        return idx[keep], w

    def wall_contact(self, pts, dist=0.10, min_frac=0.55, max_thick=0.25):
        """Is this point set mounted on a wall? -> (plane, fraction) or (None, 0).

        Both halves are needed. Fraction alone flags a bookcase standing
        against a wall; thickness alone flags any thin object anywhere. A TV
        is the conjunction: most of it lies in the wall's plane AND it barely
        protrudes from it.
        """
        best = (None, 0.0)
        for p in self.walls:
            r = pts @ p["n"] + p["d"]
            frac = float((np.abs(r) < dist).mean())
            if frac < min_frac:
                continue
            if float(r.max() - r.min()) > max_thick:
                continue
            if frac > best[1]:
                best = (p, frac)
        return best


# =============================================================================
# INSPECTION
# =============================================================================
# Stage [6] produces arrays. Arrays are not reviewable: the only way to know
# whether "chair" landed on the chair or on the wall behind it is to look at
# the map with the labels painted on. semantic.pcd exists so that check takes
# one drag into a viewer rather than a bespoke script every time.
_STRUCT_RGB = {1: (0.42, 0.36, 0.30),      # floor   -- warm, dark
               2: (0.62, 0.66, 0.72),      # wall    -- cool, light
               3: (0.80, 0.82, 0.85),      # ceiling -- near white
               4: (0.60, 0.50, 0.34)}      # support -- tan
_UNLABELLED = (0.24, 0.25, 0.27)
def class_color(cid):
    """Deterministic, well-separated colour for a class id.
    Golden-ratio hue stepping: consecutive class ids land far apart on the
    wheel, so the two classes most likely to be confused (adjacent COCO ids
    like chair/couch) never come out the same colour. Deterministic matters as
    much as distinct -- a class that changes colour between runs makes two
    screenshots impossible to compare."""
    h = (int(cid) * 0.6180339887498949 + 0.11) % 1.0
    s, v = 0.72, 0.98
    i = int(h * 6.0)
    f = h * 6.0 - i
    p, q, t = v * (1 - s), v * (1 - s * f), v * (1 - s * (1 - f))
    return [(v, t, p), (q, v, p), (p, v, t),
            (p, q, v), (t, p, v), (v, p, q)][i % 6]
def semantic_colors(cls, struct_lab, names):
    """Colour every point by what stage [6] decided it is.
    Structure is deliberately muted and objects saturated: the question this
    cloud answers is "did the objects land in the right places", and a
    full-brightness wall competes with the answer. Returns (colours, legend).
    """
    col = np.tile(np.asarray(_UNLABELLED, np.float64), (len(cls), 1))
    legend = []
    for code, rgb in _STRUCT_RGB.items():
        m = struct_lab == code
        if m.any():
            col[m] = rgb
            legend.append((["", "floor", "wall", "ceiling", "support"][code],
                           rgb, int(m.sum())))
    # objects last: an object standing on a plane wins the pixel over the plane
    for cid in np.unique(cls[cls >= 0]):
        m = cls == cid
        rgb = class_color(int(cid))
        col[m] = rgb
        legend.append((coco_or_id(names, int(cid)), rgb, int(m.sum())))
    return col, legend
def hexc(rgb):
    return "#%02x%02x%02x" % tuple(int(round(255 * c)) for c in rgb)
def _basis(n):
    """Two in-plane axes for a unit normal."""
    n = np.asarray(n, float) / np.linalg.norm(n)
    a = np.array([0.0, 0.0, 1.0]) if abs(n[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(n, a); u /= np.linalg.norm(u)
    return u, np.cross(n, u)
def plane_area(pts, model, cell=0.10):
    """Occupied area of a plane's inliers, m^2.
    Counting occupied cells of an in-plane grid, not the bounding box: a wall
    with a doorway, or an L-shaped floor, has an area meaningfully smaller than
    its extent, and the bbox would overstate a room by a third. Also immune to
    the point density varying with how close the sensor passed."""
    if len(pts) == 0:
        return 0.0
    u, v = _basis(model[:3])
    q = np.stack([pts @ u, pts @ v], 1)
    g = np.floor(q / float(cell)).astype(np.int64)
    return float(len(np.unique(g[:, 0] * 1000003 + g[:, 1])) * cell * cell)
def object_context(pts, instances, struct):
    """What each object is standing on or attached to.
    The inventory is far more useful when it says a TV is on a wall and a vase
    is on a table than when it only says both exist -- and this is the same
    query stage [7]'s rules run on, so the two can never disagree about what an
    object is attached to."""
    out = {}
    for ins in instances:
        p = pts[ins["idx"]]
        rec = {"on_wall": False, "wall_fraction": 0.0,
               "support": None, "support_z": None}
        if struct is not None:
            w, frac = struct.wall_contact(p)
            rec["on_wall"] = bool(w is not None)
            rec["wall_fraction"] = round(float(frac), 3)
            plane, z = struct.support_under(p, prefer="any")
            rec["support"] = None if plane is None else plane["kind"]
            rec["support_z"] = round(float(z), 4)
        out[ins["instance"]] = rec
    return out
def _y(v):
    """Scalar -> YAML literal."""
    if isinstance(v, bool):
        return "true" if v else "false"
    if v is None:
        return "null"
    if isinstance(v, float):
        return "%.4f" % v
    if isinstance(v, str):
        return v if v and all(c.isalnum() or c in "._-/" for c in v) else '"%s"' % v
    return str(v)
def _yl(vals):
    return "[" + ", ".join(_y(x) for x in vals) + "]"
def write_inventory(path, instances, names, geom, ctx, cls, struct, pts,
                    meta=None, clouds=None):
    """A human-readable inventory of what is in the map.
    Hand-rolled YAML rather than pyyaml, matching dump_cameras_yaml in
    pipeline_common: the pipeline already writes YAML this way and it keeps
    stage 01 free of another dependency for one output file.
    """
    L = ["# objects_inventory.yaml -- written by 01_build_map.py stage [6]",
         "# every length is metres, every coordinate is in the map frame"]
    for k, v in (meta or {}).items():
        L.append("%s: %s" % (k, _y(v)))
    L.append("n_points: %d" % len(pts))
    L.append("n_labelled: %d" % int((cls >= 0).sum()))
    L.append("n_objects: %d" % len(instances))
    L.append("")
    L.append("structure:")
    if struct is None:
        L.append("  enabled: false")
    else:
        for kind, group in (("floor", struct.floors), ("wall", struct.walls),
                            ("ceiling", struct.ceilings),
                            ("support", struct.supports)):
            if not group:
                continue
            npts = sum(p["n_points"] for p in group)
            area = sum(plane_area(pts[p["idx"]], p["model"]) for p in group)
            L.append("  %s:" % kind)
            L.append("    planes: %d" % len(group))
            L.append("    points: %d" % npts)
            L.append("    area_m2: %.2f" % area)
            if kind in ("floor", "ceiling", "support"):
                L.append("    height_m: %s"
                         % _yl([round(float(p["z"]), 3) for p in group]))
        if struct.floors and struct.ceilings:
            L.append("  room_height_m: %.3f"
                     % (max(p["z"] for p in struct.ceilings)
                        - min(p["z"] for p in struct.floors)))
    L.append("")
    counts = {}
    for ins in instances:
        counts[coco_or_id(names, ins["cls_id"])] = \
            counts.get(coco_or_id(names, ins["cls_id"]), 0) + 1
    L.append("counts:")
    for k in sorted(counts, key=lambda x: (-counts[x], x)):
        L.append("  %s: %d" % (_y(k), counts[k]))
    L.append("")
    L.append("objects:")
    for ins in sorted(instances, key=lambda i: -i["idx"].size):
        g = geom[ins["instance"]]
        c = ctx.get(ins["instance"], {})
        L.append("  - id: %d" % ins["instance"])
        L.append("    class: %s" % _y(coco_or_id(names, ins["cls_id"])))
        L.append("    confidence: %.3f" % ins["conf"])
        L.append("    n_points: %d" % ins["idx"].size)
        L.append("    n_views: %d" % ins.get("n_frames", 0))
        L.append("    centroid: %s" % _yl(g["centroid"]))
        L.append("    extent: %s" % _yl(g["extent"]))
        L.append("    bbox_min: %s" % _yl(g["bbox_min"]))
        L.append("    bbox_max: %s" % _yl(g["bbox_max"]))
        L.append("    base_z: %s" % _y(g["base_z"]))
        L.append("    on_wall: %s" % _y(c.get("on_wall", False)))
        if c.get("on_wall"):
            L.append("    wall_fraction: %s" % _y(c.get("wall_fraction")))
        L.append("    support: %s" % _y(c.get("support")))
        if c.get("support") is not None:
            L.append("    support_z: %s" % _y(c.get("support_z")))
        if clouds and ins["instance"] in clouds:
            L.append("    cloud: %s" % _y(clouds[ins["instance"]]))
    if not instances:
        L.append("  []")
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")
    return path
def instance_geometry(pts, instances):
    """Where each object IS: centroid, footprint, extent, axis-aligned bbox.
    Without this, instances.json says an object exists but not where, which
    makes it unusable for anything except counting -- and unreviewable against
    a floor plan."""
    out = {}
    for ins in instances:
        p = pts[ins["idx"]]
        lo, hi = p.min(0), p.max(0)
        out[ins["instance"]] = {
            "centroid": [round(float(v), 4) for v in p.mean(0)],
            "extent": [round(float(v), 4) for v in (hi - lo)],
            "bbox_min": [round(float(v), 4) for v in lo],
            "bbox_max": [round(float(v), 4) for v in hi],
            "base_z": round(float(np.percentile(p[:, 2], 2.0)), 4)}
    return out
def coco_or_id(names, cid):
    return names.get(cid, str(cid)) if isinstance(names, dict) else str(cid)


def instance_array(n, instances):
    """Per-point instance id, -1 where a point belongs to no instance."""
    a = np.full(n, -1, np.int32)
    for ins in instances:
        a[ins["idx"]] = ins["instance"]
    return a


def reconcile(cls, instances):
    """Clear the class of every point that no instance claims.

    Voting labels points; clustering decides which of them are objects. Points
    that voted but were dropped as noise, or trimmed off as wall skirt, must
    not keep a class -- otherwise labels.npz says "chair" about points
    instances.json has never heard of, and a consumer that trusts either one
    on its own gets a different answer from the other."""
    if not instances:
        cls[:] = -1
        return cls
    owned = np.concatenate([i["idx"] for i in instances])
    cls[np.setdiff1d(np.flatnonzero(cls >= 0), owned)] = -1
    return cls


def dump_instances(path, instances, names, n_points, extra=None):
    recs = []
    for ins in instances:
        r = {"instance": ins["instance"],
             "class": coco_or_id(names, ins["cls_id"]),
             "class_id": ins["cls_id"],
             "confidence": round(float(ins["conf"]), 4),
             "n_points": int(ins["idx"].size),
             "n_frames": ins.get("n_frames", 0)}
        r.update((extra or {}).get(ins["instance"], {}))
        recs.append(r)
    with open(path, "w") as f:
        json.dump({"n_points": int(n_points),
                   "names": {str(k): v for k, v in dict(names).items()},
                   "instances": recs}, f, indent=2)
    return recs
