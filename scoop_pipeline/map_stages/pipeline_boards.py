#!/usr/bin/env python3
"""
pipeline_boards.py - shared ChArUco detection, TF, bag reading and pose
statistics for the map pipeline.

WHAT CHANGED IN THIS REVISION
-----------------------------
1. Board DESIGNS vs board INSTANCES.  A registry entry describes one printed
   design (dictionary + geometry + marker ID block).  `instances: N` says how
   many physical copies of that design are standing in the scene.  Copies are
   separated geometrically (they are more than `assoc_radius` apart), not by
   appearance -- which is the only thing that works when two boards are
   literally the same printout.

2. Marker ID offsets.  `id_offset` is now passed through to CharucoBoard, so
   two boards can share a dictionary and still be told apart directly.  Use
   this for anything you print from now on; the geometric path above is the
   fallback for bags already recorded with identical boards.

3. Duplicate-ID handling.  detectMarkers runs ONCE per dictionary per frame.
   Markers are grouped by image-space proximity.  If the same ID shows up in
   two groups (two identical boards in view) each group is cropped and run
   through CharucoDetector separately, instead of letting the detector
   interpolate corners across both boards and return a confident wrong pose.
   If the groups overlap in the image the frame is dropped for that design.

4. Reprojection error is now a true RMS in pixels.  It used to be
   norm/len(points), i.e. RMS/sqrt(N) -- roughly 6x optimistic at 36 corners,
   and the effective threshold moved with the corner count.  RETUNE max_reproj:
   an old value of 1.5 is about 0.25 in the new units.

5. avg_T uses the Markley eigenvector quaternion mean with MAD outlier
   rejection and optional weights.  The old sign(w) hemisphere flip broke for
   rotations near 180 deg, which is common once boards face several directions.

6. Streaming bag reads.  iter_images() yields frames instead of accumulating
   every decoded image in RAM, and samples on a time period rather than a frame
   count so the sampling density does not change with camera framerate.

7. Per-board rejection counters, so "no estimates" tells you WHY.
"""
import numpy as np
import cv2
from pathlib import Path
from bisect import bisect_left
from collections import OrderedDict

from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore

from pipeline_common import make_T, R_to_q, ang_deg, qR, slerp

TS = get_typestore(Stores.ROS2_HUMBLE)
_slerp = slerp                       # back-compat alias

_DICT = {
    'DICT_4X4_50': cv2.aruco.DICT_4X4_50, 'DICT_4X4_100': cv2.aruco.DICT_4X4_100,
    'DICT_4X4_250': cv2.aruco.DICT_4X4_250, 'DICT_4X4_1000': cv2.aruco.DICT_4X4_1000,
    'DICT_5X5_50': cv2.aruco.DICT_5X5_50, 'DICT_5X5_100': cv2.aruco.DICT_5X5_100,
    'DICT_5X5_250': cv2.aruco.DICT_5X5_250, 'DICT_5X5_1000': cv2.aruco.DICT_5X5_1000,
    'DICT_6X6_50': cv2.aruco.DICT_6X6_50, 'DICT_6X6_100': cv2.aruco.DICT_6X6_100,
    'DICT_6X6_250': cv2.aruco.DICT_6X6_250, 'DICT_6X6_1000': cv2.aruco.DICT_6X6_1000,
    'DICT_7X7_50': cv2.aruco.DICT_7X7_50, 'DICT_7X7_250': cv2.aruco.DICT_7X7_250,
}

REJECT_KEYS = ("absent", "no_corners", "few_corners", "ambiguous",
               "reproj", "pnp_fail", "overlap", "ok")


# --------------------------------------------------------------------------- #
#  Detection
# --------------------------------------------------------------------------- #
class Detection(object):
    """One board sighting in one image.  T is T_cam_board (board expressed in
    the camera OPTICAL frame -- z forward, what solvePnP natively returns)."""
    __slots__ = ("T", "reproj", "n", "ratio", "design", "group", "stamp")

    def __init__(self, T, reproj, n, ratio, design=None, group=0, stamp=None):
        self.T = T
        self.reproj = reproj
        self.n = n
        self.ratio = ratio
        self.design = design
        self.group = group
        self.stamp = stamp


class Board(object):
    """A named ChArUco DESIGN: dictionary, geometry, marker ID block, gates.

    `instances` / `instance_names` describe how many physical copies of this
    printout exist in the scene; separating them is the caller's job (see
    cluster_link) because identical copies cannot be told apart from one image.
    """

    def __init__(self, name, cfg):
        self.name = name
        self.cfg = dict(cfg)
        self.sx = int(cfg["squares_x"])
        self.sy = int(cfg["squares_y"])
        self.square_len = float(cfg["square_len"])
        self.marker_len = float(cfg["marker_len"])
        self.dict_name = cfg.get("dictionary", "DICT_4X4_50")
        self.id_offset = int(cfg.get("id_offset", 0))
        self.min_corners = int(cfg.get("min_corners", 8))
        # NOTE: true RMS pixels now.  ~0.25 is a tight board, ~1.0 is loose.
        self.max_reproj = float(cfg.get("max_reproj", 1.0))
        self.min_ambiguity = float(cfg.get("min_ambiguity_ratio", 1.5))
        self.instances = max(1, int(cfg.get("instances", 1)))
        self.instance_names = list(cfg.get("instance_names", []) or [])
        self.frame_prefix = cfg.get("frame", None)

        if self.marker_len >= self.square_len:
            raise SystemExit("board '%s': marker_len (%.4f) must be < square_len (%.4f)"
                             % (name, self.marker_len, self.square_len))
        if self.dict_name not in _DICT:
            raise SystemExit("board '%s': unknown dictionary '%s'. known: %s"
                             % (name, self.dict_name, ", ".join(sorted(_DICT))))

        self.n_markers = (self.sx * self.sy) // 2
        self.ids = np.arange(self.id_offset, self.id_offset + self.n_markers,
                             dtype=np.int32)
        self.id_set = set(int(i) for i in self.ids)

        self.dict = _get_dictionary(_DICT[self.dict_name])
        self.board = _make_charuco(self.sx, self.sy, self.square_len,
                                   self.marker_len, self.dict, self.ids)
        self.new_api = hasattr(cv2.aruco, "CharucoDetector")
        self.det = _make_detector(self.board) if self.new_api else None
        obj = (self.board.getChessboardCorners()
               if hasattr(self.board, "getChessboardCorners")
               else self.board.chessboardCorners)
        self.obj = np.asarray(obj, dtype=np.float64).reshape(-1, 3)
        self.reject = OrderedDict((k, 0) for k in REJECT_KEYS)

    # ------------------------------------------------------------------ #
    @property
    def size_m(self):
        return (self.sx * self.square_len, self.sy * self.square_len)

    def instance_name(self, i):
        if i < len(self.instance_names):
            return self.instance_names[i]
        return self.name if self.instances == 1 else "%s_%d" % (self.name, i)

    def frame_for(self, inst_name, is_anchor=False):
        if is_anchor:
            return "board"
        if self.frame_prefix:
            # an explicit registry "frame" is used VERBATIM for a single-instance
            # design; only genuine multi-instance designs get a suffix, so
            # frame "board_rs" stays "board_rs" and not "board_rs_rs_anchor"
            if self.instances <= 1 and inst_name == self.name:
                return self.frame_prefix
            return "%s_%s" % (self.frame_prefix, inst_name)
        return "board_%s" % inst_name

    def frame_fix(self, axes="opencv", origin="corner"):
        """Fixed correction applied to T_cam_board, as T_cam_board @ frame_fix().

        OpenCV hands back the ChArUco native frame: origin at the top-left
        board corner, x along the columns, y DOWN the rows, z = x cross y and
        therefore INTO the board (away from the camera). In RViz that reads as
        an upside-down frame stuck to a corner, and it makes the anchored map's
        +x point into the wall.

        axes="ros"     -> x = outward normal (toward the room), y = left,
                          z = up. A vertical board then has an almost identity
                          rotation relative to the map it defines.
        origin="corner"       -> leave it where OpenCV puts it. NOTE that since
                                 OpenCV 4.6 that is the TOP-left corner with y
                                 running DOWN, not the bottom-left corner of
                                 older releases. Verify with
                                 board.getChessboardCorners() if in doubt.
        origin="bottom_left"  -> the physically lower-left corner of the printed
                                 board, i.e. shifted down the native +y axis by
                                 the board height.
        origin="center"       -> the middle of the board.

        The correction is a rigid transform, so nothing about the measurement
        changes -- only which frame the same physical pose is reported in."""
        C = np.eye(3)
        if axes == "ros":
            # columns are the new axes expressed in the OpenCV board frame:
            #   x_ros = -z_cv (outward)   y_ros = x_cv   z_ros = -y_cv (up)
            C = np.array([[0.0, 1.0, 0.0],
                          [0.0, 0.0, -1.0],
                          [-1.0, 0.0, 0.0]])
        elif axes != "opencv":
            raise SystemExit("board_axes must be 'opencv' or 'ros', got %r" % axes)
        o = np.zeros(3)
        if origin == "center":
            o = np.array([self.sx * self.square_len / 2.0,
                          self.sy * self.square_len / 2.0, 0.0])
        elif origin == "bottom_left":
            # native origin is top-left with +y down, so the lower-left corner
            # is one board-height along +y
            o = np.array([0.0, self.sy * self.square_len, 0.0])
        elif origin != "corner":
            raise SystemExit("board_origin must be 'corner', 'bottom_left' or "
                             "'center', got %r" % origin)
        T = np.eye(4)
        T[:3, :3] = C
        T[:3, 3] = o
        return T

    def describe(self):
        w, h = self.size_m
        return ("board '%s' [%s ids %d..%d] %dx%d sq @ %.3f m (marker %.3f m) "
                "-> %.3f x %.3f m | instances %d | min_corners %d, "
                "max_reproj %.2f px RMS"
                % (self.name, self.dict_name, self.id_offset,
                   self.id_offset + self.n_markers - 1, self.sx, self.sy,
                   self.square_len, self.marker_len, w, h, self.instances,
                   self.min_corners, self.max_reproj))

    def reject_str(self):
        return " ".join("%s=%d" % (k, v) for k, v in self.reject.items() if v)

    # ------------------------------------------------------------------ #
    def corners_full(self, gray):
        """Whole-image ChArUco corner extraction."""
        if self.new_api:
            cc, cid, _, _ = self.det.detectBoard(gray)
            return cc, cid
        mc, mids, _ = cv2.aruco.detectMarkers(gray, self.dict)
        if mids is None or len(mids) == 0:
            return None, None
        _, cc, cid = cv2.aruco.interpolateCornersCharuco(mc, mids, gray, self.board)
        return cc, cid

    def corners_crop(self, gray, x0, y0, x1, y1):
        """Corner extraction inside a crop, returned in FULL-image pixels."""
        sub = gray[y0:y1, x0:x1]
        if sub.size == 0:
            return None, None
        cc, cid = self.corners_full(np.ascontiguousarray(sub))
        if cid is None or len(cid) == 0:
            return None, None
        cc = np.asarray(cc, dtype=np.float64).reshape(-1, 2) + np.array([x0, y0], float)
        return cc.reshape(-1, 1, 2), cid

    def pose_from_corners(self, cc, cid, K, D):
        """PnP on already-extracted corners (full-image pixel coordinates)."""
        n = 0 if cid is None else len(cid)
        if n == 0:
            # markers were found but no chessboard corners interpolated --
            # distinct from "no markers at all", which BoardBank counts
            self.reject["no_corners"] += 1
            return None
        if n < self.min_corners:
            self.reject["few_corners"] += 1
            return None
        objp = self.obj[np.asarray(cid).flatten()]
        imgp = np.asarray(cc, dtype=np.float64).reshape(-1, 1, 2)
        rvec, tvec, ratio = _solve_pnp(objp, imgp, K, D)
        # a degenerate PnP can return NaN, which passes every > / < gate below
        if rvec is None or not (np.isfinite(rvec).all() and np.isfinite(tvec).all()):
            self.reject["pnp_fail"] += 1
            return None
        if ratio < self.min_ambiguity:
            self.reject["ambiguous"] += 1      # near-frontal planar mirror flip
            return None
        proj, _ = cv2.projectPoints(objp, rvec, tvec, K, D)
        # true RMS pixel error: norm is sqrt(sum of squares) over 2N elements
        reproj = float(cv2.norm(imgp, proj.reshape(-1, 1, 2), cv2.NORM_L2)
                       / np.sqrt(len(proj)))
        if not reproj <= self.max_reproj:                  # also NaN
            self.reject["reproj"] += 1
            return None
        T = np.eye(4)
        T[:3, :3] = cv2.Rodrigues(rvec)[0]
        T[:3, 3] = np.asarray(tvec).reshape(3)
        self.reject["ok"] += 1
        return Detection(T, reproj, n, ratio, design=self.name)

    def detect(self, gray, K, D):
        """Single best sighting in the whole image, or None.  Kept for callers
        (03_init) that assume at most one copy of the board is in frame."""
        cc, cid = self.corners_full(gray)
        return self.pose_from_corners(cc, cid, K, D)


def _get_dictionary(dname):
    if hasattr(cv2.aruco, "getPredefinedDictionary"):
        return cv2.aruco.getPredefinedDictionary(dname)
    return cv2.aruco.Dictionary_get(dname)


def _make_charuco(sx, sy, sq, mk, d, ids=None):
    if hasattr(cv2.aruco, "CharucoBoard"):
        try:
            if ids is not None and int(ids[0]) != 0:
                return cv2.aruco.CharucoBoard((sx, sy), sq, mk, d, ids)
            return cv2.aruco.CharucoBoard((sx, sy), sq, mk, d)
        except Exception:
            pass
    if ids is not None and int(ids[0]) != 0:
        raise SystemExit("id_offset needs OpenCV >= 4.7 (CharucoBoard with ids)")
    return cv2.aruco.CharucoBoard_create(sx, sy, sq, mk, d)


def _make_detector(board):
    try:
        cp = cv2.aruco.CharucoParameters()
        cp.tryRefineMarkers = True
        dp = cv2.aruco.DetectorParameters()
        dp.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        return cv2.aruco.CharucoDetector(board, cp, dp)
    except Exception:
        return cv2.aruco.CharucoDetector(board)


def _solve_pnp(objp, imgp, K, D):
    """Planar PnP with explicit two-fold ambiguity reporting.

    Returns (rvec, tvec, ratio), ratio = err(2nd best) / err(best);
    inf means only one solution was produced."""
    objp = np.ascontiguousarray(objp.reshape(-1, 1, 3).astype(np.float64))
    imgp = np.ascontiguousarray(imgp.reshape(-1, 1, 2).astype(np.float64))
    rvec = tvec = None
    ratio = float("inf")
    if len(objp) >= 4:
        try:
            nsol, rvecs, tvecs, errs = cv2.solvePnPGeneric(
                objp, imgp, K, D, flags=cv2.SOLVEPNP_IPPE)
            if nsol >= 1:
                e = np.asarray(errs, dtype=float).ravel()
                order = np.argsort(e)
                rvec, tvec = rvecs[order[0]], tvecs[order[0]]
                if nsol >= 2:
                    ratio = float(e[order[1]] / max(e[order[0]], 1e-9))
        except cv2.error:
            rvec = None
    if rvec is None:
        ok, rvec, tvec = cv2.solvePnP(objp, imgp, K, D, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            return None, None, 0.0
    try:
        rvec, tvec = cv2.solvePnPRefineLM(objp, imgp, K, D, rvec, tvec)
    except (cv2.error, AttributeError):
        pass
    return rvec, tvec, ratio


# --------------------------------------------------------------------------- #
#  Multi-board bank: one marker pass per dictionary per frame
# --------------------------------------------------------------------------- #
class BoardBank(object):
    def __init__(self, designs, group_gap_factor=2.5, crop_margin=1.5):
        self.designs = list(designs)
        self.group_gap = float(group_gap_factor)
        self.crop_margin = float(crop_margin)
        self.by_dict = OrderedDict()
        for b in self.designs:
            ent = self.by_dict.setdefault(b.dict_name, [None, []])
            ent[1].append(b)
        for dn, ent in self.by_dict.items():
            ent[0] = _make_aruco_detector(_get_dictionary(_DICT[dn]))
        self._check_ids()

    def _check_ids(self):
        for dn, (_, designs) in self.by_dict.items():
            for i in range(len(designs)):
                for j in range(i + 1, len(designs)):
                    a, b = designs[i], designs[j]
                    if a.id_set & b.id_set:
                        raise SystemExit(
                            "boards '%s' and '%s' share dictionary %s AND marker "
                            "IDs %d..%d / %d..%d -- they are indistinguishable. "
                            "Give one an id_offset, or merge them into a single "
                            "design with instances: 2."
                            % (a.name, b.name, dn, a.id_offset,
                               a.id_offset + a.n_markers - 1, b.id_offset,
                               b.id_offset + b.n_markers - 1))

    def detect(self, gray, K, D, stamp=None):
        """All board sightings in one image.  Returns a list of Detection."""
        out = []
        for dn, (adet, designs) in self.by_dict.items():
            corners, ids, _ = adet.detectMarkers(gray)
            if ids is None or len(ids) == 0:
                for b in designs:
                    b.reject["absent"] += 1
                continue
            ids = np.asarray(ids).flatten()
            groups = group_markers(corners, self.group_gap)
            for b in designs:
                mine = [g for g in groups if len(set(int(i) for i in ids[g]) & b.id_set)]
                if not mine:
                    b.reject["absent"] += 1
                    continue
                dup = _has_duplicate_ids([ids[g] for g in mine], b.id_set)
                if len(mine) == 1 and not dup:
                    d = b.pose_from_corners(*(b.corners_full(gray) + (K, D)))
                    if d is not None:
                        d.stamp = stamp
                        out.append(d)
                    continue
                if dup and _groups_overlap(corners, mine):
                    b.reject["overlap"] += 1     # copies overlap in image: unusable
                    continue
                for gi, g in enumerate(mine):
                    x0, y0, x1, y1 = _crop_bbox(corners, g, gray.shape,
                                                self.crop_margin)
                    cc, cid = b.corners_crop(gray, x0, y0, x1, y1)
                    d = b.pose_from_corners(cc, cid, K, D)
                    if d is not None:
                        d.stamp = stamp
                        d.group = gi
                        out.append(d)
        return out


def _make_aruco_detector(dic):
    try:
        dp = cv2.aruco.DetectorParameters()
        dp.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_NONE   # presence only
        return cv2.aruco.ArucoDetector(dic, dp)
    except AttributeError:
        class _Legacy(object):
            def __init__(self, d):
                self.d = d

            def detectMarkers(self, gray):
                return cv2.aruco.detectMarkers(gray, self.d)
        return _Legacy(dic)


def _marker_centers_sizes(corners):
    C = np.array([np.asarray(c, float).reshape(-1, 2).mean(0) for c in corners])
    S = np.array([np.linalg.norm(np.asarray(c, float).reshape(-1, 2)[0]
                                 - np.asarray(c, float).reshape(-1, 2)[2])
                  for c in corners])
    return C, S


def group_markers(corners, gap_factor=2.5):
    """Single-link grouping of detected markers by image-space proximity."""
    C, S = _marker_centers_sizes(corners)
    r = gap_factor * max(float(np.median(S)), 1.0)
    lab = cluster_link(C, r)
    return [np.where(lab == k)[0] for k in range(int(lab.max()) + 1)]


def _has_duplicate_ids(id_groups, id_set):
    seen = set()
    for g in id_groups:
        cur = set(int(i) for i in g) & id_set
        if cur & seen:
            return True
        seen |= cur
    return False


def _crop_bbox(corners, idx, shape, margin_factor=1.5):
    C, S = _marker_centers_sizes([corners[i] for i in idx])
    pts = np.vstack([np.asarray(corners[i], float).reshape(-1, 2) for i in idx])
    m = margin_factor * max(float(np.median(S)), 1.0)
    x0 = int(max(0, np.floor(pts[:, 0].min() - m)))
    y0 = int(max(0, np.floor(pts[:, 1].min() - m)))
    x1 = int(min(shape[1], np.ceil(pts[:, 0].max() + m)))
    y1 = int(min(shape[0], np.ceil(pts[:, 1].max() + m)))
    return x0, y0, x1, y1


def _groups_overlap(corners, groups):
    boxes = []
    for g in groups:
        pts = np.vstack([np.asarray(corners[i], float).reshape(-1, 2) for i in g])
        boxes.append((pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()))
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            a, b = boxes[i], boxes[j]
            if not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1]):
                return True
    return False


# --------------------------------------------------------------------------- #
#  Clustering / sectioning
# --------------------------------------------------------------------------- #
def cluster_link(pts, radius):
    """Single-linkage clustering.  Returns an integer label per row of `pts`."""
    P = np.asarray(pts, float)
    n = len(P)
    lab = -np.ones(n, int)
    cur = 0
    for i in range(n):
        if lab[i] >= 0:
            continue
        lab[i] = cur
        stack = [i]
        while stack:
            j = stack.pop()
            d = np.linalg.norm(P - P[j], axis=1)
            for k in np.where((d <= radius) & (lab < 0))[0]:
                lab[k] = cur
                stack.append(int(k))
        cur += 1
    return lab


def sectionize(stamps, gap):
    """Split time-sorted stamps into visit sections.  Returns list of index
    arrays into the (already sorted) input."""
    t = np.asarray(stamps, float)
    if len(t) == 0:
        return []
    cuts = np.where(np.diff(t) > gap)[0] + 1
    return np.split(np.arange(len(t)), cuts)


# --------------------------------------------------------------------------- #
#  Pose averaging
# --------------------------------------------------------------------------- #
def avg_T(Ts, weights=None, mad_k=3.0, floor_m=0.002):
    """Robust mean of SE(3) poses of a STATIC thing.

    Translation: MAD-gated weighted mean.  Rotation: Markley eigenvector
    quaternion mean (sign-invariant, so it is correct near 180 deg where the
    old sign(w) hemisphere flip failed).  Do NOT use on a moving camera.

    Returns (T, n_used)."""
    Ts = list(Ts)
    ts = np.array([T[:3, 3] for T in Ts], float)
    w = np.ones(len(Ts)) if weights is None else np.asarray(weights, float).copy()
    w = np.clip(w, 1e-9, None)

    med = np.median(ts, axis=0)
    d = np.linalg.norm(ts - med, axis=1)
    mad = float(np.median(np.abs(d - np.median(d)))) * 1.4826
    # the median-distance term keeps clean Gaussian scatter from being clipped;
    # the MAD term is what actually catches a gross PnP outlier
    thr = max(mad_k * mad, 2.5 * float(np.median(d)), floor_m)
    keep = d <= thr
    if keep.sum() < max(3, int(0.5 * len(Ts))):
        keep = np.ones(len(Ts), bool)

    sel_t = ts[keep]
    sel_w = w[keep] / w[keep].sum()
    Q = np.array([R_to_q(Ts[i][:3, :3]) for i in np.where(keep)[0]], float)
    M = (Q * sel_w[:, None]).T @ Q
    _, evecs = np.linalg.eigh(M)
    q = evecs[:, -1]
    if q[3] < 0:
        q = -q
    return make_T(np.average(sel_t, axis=0, weights=sel_w), q), int(keep.sum())


def spread(Ts, Tavg):
    """(std_mm, max_mm, max_deg) scatter of a pose set about its mean."""
    tt = np.array([T[:3, 3] for T in Ts], float)
    ta = Tavg[:3, 3]
    std_mm = float(np.linalg.norm(tt.std(0)) * 1000.0)
    max_mm = float(np.linalg.norm(tt - ta, axis=1).max() * 1000.0)
    max_deg = float(max(ang_deg(Tavg[:3, :3], T[:3, :3]) for T in Ts))
    return std_mm, max_mm, max_deg


def fmt_spread(sp):
    return "std %.1f mm / max %.1f mm / max %.3f deg" % sp


def pose_delta(Ta, Tb):
    """(mm, deg) between two poses."""
    return (float(np.linalg.norm(Ta[:3, 3] - Tb[:3, 3]) * 1000.0),
            ang_deg(Ta[:3, :3], Tb[:3, :3]))


# --------------------------------------------------------------------------- #
#  TF
# --------------------------------------------------------------------------- #
def tf_to_T(transform):
    """`transform` is a geometry_msgs/Transform (tf_msg.transforms[i].transform)."""
    t = transform.translation; q = transform.rotation
    return make_T([t.x, t.y, t.z], [q.x, q.y, q.z, q.w])


class TFTree(object):
    """Minimal offline TF buffer: nearest-sample lookup with a time tolerance."""

    def __init__(self, max_gap=0.2):
        self.static = {}
        self.dyn = {}
        self.max_gap = float(max_gap)

    def add(self, parent, child, T, stamp, is_static):
        if is_static:
            self.static[child] = (parent, T)
        else:
            p, ts, Ts = self.dyn.get(child, (parent, [], []))
            ts.append(stamp); Ts.append(T)
            self.dyn[child] = (parent, ts, Ts)

    def finalize(self):
        for c, (p, ts, Ts) in list(self.dyn.items()):
            o = np.argsort(ts)
            self.dyn[c] = (p, np.array(ts)[o], [Ts[i] for i in o])

    def frames(self):
        return sorted(set(list(self.static) + list(self.dyn)))

    def _edge(self, child, stamp):
        if child in self.static:
            return self.static[child] + (0.0,)
        if child in self.dyn:
            p, ts, Ts = self.dyn[child]
            if len(ts) == 0:
                return None
            i = bisect_left(ts, stamp)
            i = min(max(i, 0), len(ts) - 1)
            if i > 0 and abs(stamp - ts[i - 1]) < abs(ts[i] - stamp):
                i -= 1
            gap = abs(stamp - ts[i])
            if gap > self.max_gap:
                return None
            return p, Ts[i], gap
        return None

    def lookup(self, root, child, stamp):
        """T_root_child at `stamp`, or None if the chain is broken / stale."""
        chain = []
        cur = child
        for _ in range(50):
            if cur == root:
                M = np.eye(4)
                for T in reversed(chain):
                    M = M @ T
                return M
            e = self._edge(cur, stamp)
            if e is None:
                return None
            parent, T, _gap = e
            chain.append(T)
            cur = parent
        return None


class PoseStream(object):
    """A sorted (stamp, T_parent_child) series from one pose topic, with linear
    + slerp interpolation.  Same role as TFTree, for a sensor whose origin is
    published only as Odometry / PoseStamped rather than a live TF edge."""

    def __init__(self):
        self._t = []; self._T = []
        self.t = np.array([]); self.T = []

    def add(self, stamp, T):
        self._t.append(stamp); self._T.append(T)

    def finalize(self):
        if not self._t:
            return
        o = np.argsort(self._t)
        self.t = np.array(self._t)[o]
        self.T = [self._T[i] for i in o]

    def __len__(self):
        return len(self.t)

    def lookup(self, stamp, max_gap=0.2):
        t = self.t
        if len(t) == 0:
            return None
        i = bisect_left(t, stamp)
        if i <= 0:
            return self.T[0] if abs(t[0] - stamp) <= max_gap else None
        if i >= len(t):
            return self.T[-1] if abs(t[-1] - stamp) <= max_gap else None
        t0, t1 = t[i - 1], t[i]
        if (stamp - t0) > max_gap and (t1 - stamp) > max_gap:
            return None
        f = 0.0 if t1 == t0 else (stamp - t0) / (t1 - t0)
        T0, T1 = self.T[i - 1], self.T[i]
        p = T0[:3, 3] * (1 - f) + T1[:3, 3] * f
        q = slerp(R_to_q(T0[:3, :3]), R_to_q(T1[:3, :3]), f)
        T = np.eye(4); T[:3, :3] = qR(q); T[:3, 3] = p
        return T


def read_pose_topics(bag, topics):
    """One pass over `bag` extracting Odometry / PoseStamped topics."""
    topics = [t for t in topics if t]
    streams = dict((t, PoseStream()) for t in topics)
    if not topics:
        return streams
    with AnyReader([Path(bag)], default_typestore=TS) as r:
        conns = [c for c in r.connections if c.topic in streams]
        for conn, _, raw in r.messages(connections=conns):
            m = r.deserialize(raw, conn.msgtype)
            p = m.pose.pose if hasattr(m.pose, "pose") else m.pose
            stamp = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
            T = make_T([p.position.x, p.position.y, p.position.z],
                       [p.orientation.x, p.orientation.y, p.orientation.z,
                        p.orientation.w])
            streams[conn.topic].add(stamp, T)
    for st in streams.values():
        st.finalize()
    return streams


# --------------------------------------------------------------------------- #
#  Bag reading
# --------------------------------------------------------------------------- #
def decode_gray(msg, msgtype=""):
    if "CompressedImage" in msgtype:
        buf = np.frombuffer(bytes(msg.data), np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise SystemExit("failed to decode a CompressedImage")
        return img
    enc = msg.encoding.lower()
    buf = np.frombuffer(bytes(msg.data), np.uint8)
    h, w = msg.height, msg.width
    ch = {"bgra8": 4, "rgba8": 4, "bgr8": 3, "rgb8": 3,
          "mono8": 1, "8uc1": 1, "mono16": 2, "16uc1": 2}.get(enc)
    if ch is None:
        raise SystemExit("unhandled encoding %s" % msg.encoding)
    step = int(getattr(msg, "step", 0) or w * ch)
    rows = buf[:h * step].reshape(h, step)[:, :w * ch]          # honour row stride
    if enc in ("mono16", "16uc1"):
        return cv2.convertScaleAbs(
            np.ascontiguousarray(rows).view(np.uint16).reshape(h, w), alpha=1 / 257.0)
    img = np.ascontiguousarray(rows).reshape(h, w, ch)
    if ch == 1:
        return img.reshape(h, w)
    img = img[:, :, :3]
    if enc in ("rgba8", "rgb8"):
        img = img[:, :, ::-1]
    return cv2.cvtColor(np.ascontiguousarray(img), cv2.COLOR_BGR2GRAY)


def read_tf_and_info(bag, info_topics=(), want_tf=True, tf_max_gap=0.2,
                     verbose=True):
    """Cheap first pass: TF tree + first CameraInfo per topic + topic list."""
    info_topics = [t for t in info_topics if t]
    infos = {}
    tree = TFTree(tf_max_gap) if want_tf else None
    with AnyReader([Path(bag)], default_typestore=TS) as r:
        available = sorted(set(c.topic for c in r.connections))
        wanted = set(info_topics) | ({"/tf", "/tf_static"} if want_tf else set())
        conns = [c for c in r.connections if c.topic in wanted]
        for conn, _, raw in r.messages(connections=conns):
            topic = conn.topic
            if topic in info_topics:
                if topic in infos:
                    continue
                m = r.deserialize(raw, conn.msgtype)
                K = np.array(m.k, dtype=float).reshape(3, 3)
                Dv = np.array(m.d, dtype=float) if len(m.d) else np.zeros(5)
                infos[topic] = (K, Dv)
            elif want_tf and topic in ("/tf", "/tf_static"):
                m = r.deserialize(raw, conn.msgtype)
                is_static = topic == "/tf_static"
                for tr in m.transforms:
                    st = tr.header.stamp.sec + tr.header.stamp.nanosec * 1e-9
                    tree.add(tr.header.frame_id, tr.child_frame_id,
                             tf_to_T(tr.transform), st, is_static)
    if tree is not None:
        tree.finalize()
    if verbose:
        for t in info_topics:
            if t not in available:
                print("  ! topic %s NOT in bag" % t)
    return infos, tree, available


def iter_images(bag, image_topic, detect_period=0.0, max_images=0, stride=1):
    """STREAM (stamp, gray) for one image topic.  Nothing is accumulated.

    detect_period: minimum seconds between yielded frames (0 = every frame).
    Time-based rather than frame-count based, so sampling density does not
    change when the camera framerate does."""
    last = None
    kept = seen = 0
    stride = max(int(stride or 1), 1)
    with AnyReader([Path(bag)], default_typestore=TS) as r:
        conns = [c for c in r.connections if c.topic == image_topic]
        if not conns:
            return
        for conn, _, raw in r.messages(connections=conns):
            # stride BEFORE deserialising -- decoding a frame only to drop it is
            # the expensive half of the loop
            if seen % stride:
                seen += 1
                continue
            m = r.deserialize(raw, conn.msgtype)
            st = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
            seen += 1
            if detect_period > 0 and last is not None and (st - last) < detect_period:
                continue
            if max_images and kept >= max_images:
                break
            last = st
            kept += 1
            yield st, decode_gray(m, conn.msgtype)


def read_bag(bag, image_topics=(), info_topics=(), want_tf=True, stride=1,
             max_images=0, tf_max_gap=0.2, verbose=True):
    """Back-compat buffering reader (used by 03_init).  Prefer iter_images for
    anything that walks a full mapping bag."""
    image_topics = [t for t in image_topics if t]
    infos, tree, available = read_tf_and_info(bag, info_topics, want_tf,
                                              tf_max_gap, verbose)
    images = {}
    stride = max(int(stride or 1), 1)
    for t in image_topics:
        images[t] = list(iter_images(bag, t, 0.0, max_images, stride))
        if verbose:
            print("  %s: %d images kept (stride %d)" % (t, len(images[t]), stride))
    return images, infos, tree, available


def pick_intrinsics(infos, info_topic, rectified, K_fallback=None, D_fallback=None):
    """CameraInfo from the bag if present, else the config/sensor fallback."""
    if info_topic in infos:
        K, D = infos[info_topic]
        K = K.copy()
        D = np.zeros(5) if rectified else D.copy()
        return K, D, True
    if K_fallback is None:
        raise SystemExit("no CameraInfo on %s and no fallback intrinsics" % info_topic)
    return (np.array(K_fallback, float).copy(),
            np.zeros(5) if rectified else np.array(D_fallback, float).copy(), False)


# --------------------------------------------------------------------------- #
#  Config
# --------------------------------------------------------------------------- #
def board_registry(cfg):
    reg = cfg.get("boards", {})
    if not reg:
        raise SystemExit(
            "pipeline_config.json has no top-level \"boards\" registry. Add one, "
            "e.g.\n"
            '  "boards": {\n'
            '    "anchor":    {"dictionary": "DICT_4X4_50", "squares_x": 7,\n'
            '                  "squares_y": 5, "square_len": 0.040,\n'
            '                  "marker_len": 0.030, "id_offset": 0,\n'
            '                  "instances": 2,\n'
            '                  "instance_names": ["anchor_a", "anchor_b"]},\n'
            '    "rs_anchor": {"dictionary": "DICT_5X5_100", "squares_x": 5,\n'
            '                  "squares_y": 5, "square_len": 0.115,\n'
            '                  "marker_len": 0.086, "id_offset": 0}\n'
            "  }")
    return reg


def boards_from_config(cfg, stage, anchor_key="board", extra_key="extra_boards"):
    """Legacy accessor: (anchor_design, [extra_designs]).  Kept for 03_init."""
    reg = cfg.get("boards", {})
    spec = stage.get(anchor_key, "anchor")
    if isinstance(spec, dict):
        anchor = Board("anchor", spec)
    else:
        if spec not in reg:
            raise SystemExit("board '%s' not in the top-level 'boards' registry" % spec)
        anchor = Board(spec, reg[spec])
    extras = []
    for nm in stage.get(extra_key, []) or []:
        if nm not in reg:
            raise SystemExit("board '%s' not in the top-level 'boards' registry" % nm)
        extras.append(Board(nm, reg[nm]))
    return anchor, extras


def bank_from_config(cfg, names=None, group_gap_factor=2.5, crop_margin=1.5):
    """Build a BoardBank over the named registry entries (default: all)."""
    reg = board_registry(cfg)
    names = list(names) if names else list(reg.keys())
    designs = []
    for nm in names:
        if nm not in reg:
            raise SystemExit("board '%s' not in the 'boards' registry (have: %s)"
                             % (nm, ", ".join(sorted(reg))))
        designs.append(Board(nm, reg[nm]))
    return BoardBank(designs, group_gap_factor, crop_margin)


# --------------------------------------------------------------------------- #
#  Output helpers
# --------------------------------------------------------------------------- #
def stp(parent, child, T, indent="  "):
    t = T[:3, 3]; q = R_to_q(T[:3, :3])
    return ("%sros2 run tf2_ros static_transform_publisher \\\n"
            "%s  --x %.6f --y %.6f --z %.6f \\\n"
            "%s  --qx %.6f --qy %.6f --qz %.6f --qw %.6f \\\n"
            "%s  --frame-id %s --child-frame-id %s"
            % (indent, indent, t[0], t[1], t[2], indent, q[0], q[1], q[2], q[3],
               indent, parent, child))


def T_record(parent, child, T):
    return {"parent": parent, "child": child,
            "xyz": np.asarray(T[:3, 3]).round(6).tolist(),
            "qxyzw": R_to_q(T[:3, :3]).round(6).tolist()}


def T_from_record(rec):
    return make_T(rec["xyz"], rec["qxyzw"])
