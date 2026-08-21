#!/usr/bin/env python3
"""
RADAR ↔ LIDAR extrinsic calibration  —  solves T_lidar_radar
=============================================================

ONE node, no board, no hand-measured offsets. The whole target is a corner
reflector on a tripod. The SOLVE lives entirely in the LIDAR frame:

    p_lidar = R · p_radar + t          (X = T_lidar_radar)

The camera is OPTIONAL and never enters the solve. Given the GLIM lidar↔camera
transform it is used for two things only: composing the deployable
T_cam_radar = T_cam_lidar · T_lidar_radar, and drawing the ZED image overlay.
So an error in the lidar↔camera calibration shows up in the composed output and
the overlay but cannot corrupt the radar↔lidar result — and re-running GLIM
later lets you recompose without recollecting any radar data.

How the reflector is found
--------------------------
LIDAR — it does not look for a "reflector". It looks for what is NEW:
  1. ~/background memorises the empty scene on a voxel grid (walls, floor AND
     the tripod — the reflector must be OFF the tripod for this),
  2. every later cloud is background-subtracted, so the only surviving points
     are the object that was not there before = the reflector,
  3. the survivors are clustered, and the APEX is localised by RANSAC-fitting
     the three plates and intersecting them (analytic corner, nothing
     measured). Fallback when the plates are too sparse: the farthest point
     along the viewing ray, which for a reflector aimed at the sensor IS the
     corner.
  The premise is "nothing else changed since the background" — hence the
  per-placement loop: background → mount reflector → step out → capture.

RADAR — background subtraction, then a ROTATION-INVARIANT range gate around
  the lidar's range (you do not know R yet, but |p| is the same in any
  orientation), |doppler|≈0 (a tripod target is truly static), clustering, and
  the SNR-weighted centroid of the best blob. Once a solve exists it tightens
  to a 3-D gate around the predicted point.

Aim feedback (before you trigger)
---------------------------------
The status marker / log shows continuously:
    radar: best 1240 (norm 1870) @ 2.31 m  OK        → capture will pass
    radar: best 41 (norm 62) @ 2.28 m  RE-AIM        → fix the aim first
    radar: no return near lidar range                → aimed way off / blocked
`norm` = snr·(r/1.5 m)^4. Received power falls as 1/r^4, so a raw SNR
threshold would wrongly refuse a well-aimed reflector at 4 m.

A capture is ATOMIC: both sensors must pass their gates or it is REFUSED with
the reason logged. Nothing half-good is ever stored.

RViz verification (plus the ZED image overlay when the camera is configured)
--------------------------------------------------
Fixed Frame = your lidar frame. Add the lidar PointCloud2 and a MarkerArray on
~/markers:
  cyan points   the foreground cluster        → must sit on the reflector only
  green sphere  the detected apex             → must sit on its corner
  amber spheres captured apexes, numbered     → your coverage map
  magenta       the radar's pick through the current solve (after first solve)
  magenta line  radar pick ↔ lidar apex, labelled with the gap in mm
  text          aim status + capture/solve state
FINAL CHECK: carry the reflector around — near/far, left/right, UP/DOWN — and
the magenta sphere must stay on the green one. If it mirrors when you raise the
reflector, the rotation is in the wrong branch: do not save.

Coverage HUD — "will these captures constrain the solve?"
---------------------------------------------------------
Six bars, top-right of the image overlay and repeated in the RViz status text
so the check survives a camera-less rig. Each is labelled with the DOF it
unlocks, so a red bar names the axis that will come out loose:

  RANGE   spread in m    → separates t from R (same-distance poses can't)
  AZ      spread in deg  → yaw
  AZ BAL  thinner side   → yaw, and kills the one-sided azimuth bias
  EL      spread in deg  → pitch AND roll (both are unobservable in a plane)
  EL BAL  thinner side   → pitch; this is the bar people leave red
  <1.5 m  count          → everything: angular error is r·sin(σ), so a close
                           capture is worth several distant ones
  CELLS   of 9           → all of the above at once (see the map)

The BAL rows exist because a spread number alone lies: 16° of elevation all
BELOW boresight fits a pitch error just as well as a vertical bias in the apex
estimate, and the two trade off. See COVERAGE_TARGETS for the geometry behind
each threshold.

Under the bars is the MAP, which answers the question the bars cannot — not
"is it enough" but "where do I put the tripod next":

  AZ × EL   the field of view as a 3×3 grid (az ±60°, el ±40°). A cell with no
            capture is SHADED AMBER — walk the tripod there. Captures are dots,
            drawn larger when they were taken close (a near capture carries more
            angular information). The live radar pick is a green ring, so you
            watch the pose land in an empty cell BEFORE you trigger.
  RANGE     the same idea in one dimension: three bands, empty ones shaded, one
            tick per capture, live range as a green caret.

RViz gets the grid as text ([X]/[ ] rows) for a camera-less rig. On a short
image the map is dropped and the bars are kept — the bars are the verdict. Below the bars sits the last solve's rot/t 1σ — the OUTCOME, as
opposed to the bars, which are the CAUSE. Green bars with a fat 1σ means the
reflector is being found badly; a tight 1σ on red bars is the over-confident fit
from a degenerate geometry, which is exactly what the bars are there to catch.
Set `show_coverage_hud:=false` to hide it.

Run
---
  ros2 run wicoms_utils radar_lidar_calib --ros-args \
    -p lidar_topic:=/ouster/points \
    -p radar_topic:=/radar1/radar/points_all -p pc_field_snr:=intensity \
    -p radar_name:=radar1 -p child_frame:=radar1_link

  ros2 topic pub -1 /radar_lidar_calib/background std_msgs/msg/Empty "{}"
  ros2 topic pub -1 /radar_lidar_calib/capture    std_msgs/msg/Empty "{}"
  ros2 topic pub -1 /radar_lidar_calib/solve      std_msgs/msg/Empty "{}"
  ros2 topic pub -1 /radar_lidar_calib/save       std_msgs/msg/Empty "{}"
  ros2 topic pub -1 /radar_lidar_calib/reset      std_msgs/msg/Empty "{}"

Camera parameters default to the GLIM result for this rig
(`lidar_camera_xyz` / `lidar_camera_quat_xyzw`, direction `lidar_camera`,
i.e. the given transform maps CAMERA points into the LIDAR frame and is
inverted internally). Set `show_image_overlay:=false` to run headless, or
override `camera_transform_is:=camera_lidar` if your file stores the other
direction.

`camera_transform_from_tf:=true` takes it from the TF tree instead, looked up
from the frame the CLOUD arrives in to `camera_frame`. Prefer it whenever a
static_transform_publisher is already broadcasting the transform: restating one
by hand fails silently in two ways — the direction is easy to invert, and it is
easy to quote it against a different frame than the cloud's (on an Ouster,
`os_sensor` and `os_lidar` are 180 deg apart in yaw plus a few cm). TF knows
both and cannot get either wrong.

The camera's intrinsics are used ONLY to draw the image overlay. An
uncalibrated camera still composes a correct `T_cam_radar` — run it with
`show_image_overlay:=false` and verify in RViz, which needs no intrinsics.

`rectify_image:=true` for a RAW feed (an Arducam publishes `/image_raw` with real
lens distortion): every frame is undistorted from `camera_info`, and projection
then uses the NEW K with zero distortion. Projecting onto a raw image without
this is not wrong in itself — projectPoints re-applies D — but the two conventions
disagree by tens of pixels near the edges, so the overlay must match whichever
image the rest of the pipeline consumes. Auto-disables with a log line if D is
already ~0. `rectify_alpha`: 0 crops to valid pixels, 1 keeps the full FoV with
black borders.

Structure: [A] cloud tools · [B] apex locator · [C] node.
"""
import json
import os
import time
from collections import deque
from itertools import combinations

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2, Image, CameraInfo
from geometry_msgs.msg import PointStamped, TransformStamped, Point
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Empty, ColorRGBA
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as Rot
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
from tf2_ros import Buffer, TransformListener

try:                                    # optional: only for the image overlay
    import cv2
    from cv_bridge import CvBridge
    _HAVE_CV = True
except ImportError:
    _HAVE_CV = False

try:                                    # flat-module or installed-package layout
    from radar_camera_calib import (robust_ml_calibrate, loo_cross_val,
                                    condition_number, cluster_points as radar_cluster,
                                    cart_to_raz)
except ImportError:
    from wicoms_utils.radar_camera_calib import (robust_ml_calibrate, loo_cross_val,
                                                 condition_number,
                                                 cluster_points as radar_cluster,
                                                 cart_to_raz)


# ────────────────────────────── [A] cloud tools ──────────────────────────────
_DT = {1: np.int8, 2: np.uint8, 3: np.int16, 4: np.uint16,
       5: np.int32, 6: np.uint32, 7: np.float32, 8: np.float64}


def cloud_fields(msg, names):
    """PointCloud2 → dict of named float32 columns (missing name → None).
    Parsed straight from the buffer; the read_points generator is far too slow
    for a 64×1024 Ouster at 10-20 Hz."""
    n = msg.width * msg.height
    if n == 0:
        return {k: None for k in names}
    step = msg.point_step
    buf = np.frombuffer(bytes(msg.data), np.uint8, count=n * step).reshape(n, step)
    offs = {f.name: (f.offset, f.datatype) for f in msg.fields}
    out = {}
    for name in names:
        if name in offs and _DT.get(offs[name][1]) is not None:
            off, dt = offs[name]
            typ = _DT[dt]
            w = np.dtype(typ).itemsize
            out[name] = buf[:, off:off + w].copy().view(typ).ravel().astype(np.float32)
        else:
            out[name] = None
    return out


def cloud_xyz(msg):
    f = cloud_fields(msg, ['x', 'y', 'z'])
    if f['x'] is None or f['y'] is None or f['z'] is None:
        return np.zeros((0, 3), np.float32)
    return np.stack([f['x'], f['y'], f['z']], 1)


# Voxel background set: keys are packed int64s, membership tested against the
# 27-neighbourhood so "within ~one voxel of any background point" counts as
# background. Vectorised with sort + searchsorted (no per-point python).
_OFF = 1 << 20
_SX, _SY = 1 << 42, 1 << 21
_NEIGH = np.array([dx * _SX + dy * _SY + dz
                   for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)],
                  dtype=np.int64)


def voxel_keys(xyz, voxel):
    idx = np.floor(xyz / voxel).astype(np.int64) + _OFF
    return idx[:, 0] * _SX + idx[:, 1] * _SY + idx[:, 2]


def foreground_mask(xyz, bg_sorted, voxel):
    keys = voxel_keys(xyz, voxel)
    hit = np.zeros(len(keys), bool)
    for d in _NEIGH:
        k = keys + d
        i = np.clip(np.searchsorted(bg_sorted, k), 0, len(bg_sorted) - 1)
        hit |= (bg_sorted[i] == k)
    return ~hit


def lidar_cluster(P, eps, min_size, cap=20000):
    """Connected components within `eps` via cKDTree. Returns point arrays,
    largest first. The foreground should be tiny (just the reflector); the cap
    is a defensive decimation for when the background went stale."""
    if len(P) == 0:
        return []
    if len(P) > cap:
        P = P[np.random.choice(len(P), cap, replace=False)]
    tree = cKDTree(P)
    lab = np.full(len(P), -1, int)
    out, cid = [], 0
    for i in range(len(P)):
        if lab[i] >= 0:
            continue
        stack, members = [i], [i]
        lab[i] = cid
        while stack:
            j = stack.pop()
            for k in tree.query_ball_point(P[j], eps):
                if lab[k] < 0:
                    lab[k] = cid
                    stack.append(k)
                    members.append(k)
        if len(members) >= min_size:
            out.append(np.array(members))
        cid += 1
    out.sort(key=len, reverse=True)
    return [P[m] for m in out]


def grow_stepwise(raw, seed, eps, r_min, r_max, step, plateau_frac, max_extent):
    """Grow outward from the tip and STOP where the object ends.

    A fixed radius cannot work: no value covers a 25 cm reflector without also
    reaching a tripod head 25 cm away. So sweep the radius outward in `step`
    increments and watch how the connected cluster grows:

        reflector filling up   -> points added every step
        reflector complete     -> increments collapse (nothing between it and
                                  the head but a thin mount)  => PLATEAU, stop
        head/legs joined       -> extent jumps past a reflector's size => stop

    Deterministic — no RANSAC, so the answer does not change frame to frame.
    Returns (points, radius_used, why).
    """
    c0 = seed.mean(0)
    pool = raw[np.linalg.norm(raw - c0, axis=1) <= r_max]
    if len(pool) < len(seed):
        return seed, 0.0, 'seed'
    tree = cKDTree(pool)
    dist = np.linalg.norm(pool - c0, axis=1)
    seed_hits = [i for p in seed for i in tree.query_ball_point(p, eps)]

    best, best_r, why, prev_n = seed, 0.0, 'cap', 0
    r = r_min
    while r <= r_max + 1e-9:
        allow = dist <= r
        seen = np.zeros(len(pool), bool)
        stack = []
        for i in seed_hits:
            if allow[i] and not seen[i]:
                seen[i] = True
                stack.append(i)
        while stack:
            for k in tree.query_ball_point(pool[stack.pop()], eps):
                if allow[k] and not seen[k]:
                    seen[k] = True
                    stack.append(k)
        P = pool[seen]
        if len(P) >= len(seed):
            ext = float(np.linalg.norm(P.max(0) - P.min(0)))
            if ext > max_extent:                       # swallowed the tripod
                why = 'size'
                break
            grew = len(P) - prev_n
            best, best_r, prev_n = P, r, len(P)
            if r > r_min and grew <= plateau_frac * max(len(P), 1):
                why = 'plateau'                        # object complete
                break
        r += step
    return best, best_r, why


def grow_from_seed(raw, seed, eps, max_r):
    """Recover the WHOLE object from a partial detection.

    Background subtraction erases everything within ~bg_voxel of a memorised
    point, so a reflector bolted to a memorised tripod head keeps only its top.
    Those surviving points are still a reliable SEED: region-grow from them back
    through the RAW cloud (single-linkage at `eps`) to pull in the erased body.

    Bounded to `max_r` of the seed centroid so the growth cannot run down the
    tripod legs — it stops after the reflector plus, at worst, the head, which
    the plane fit then rejects as outliers.
    """
    c0 = seed.mean(0)
    near = raw[np.linalg.norm(raw - c0, axis=1) <= max_r]
    if len(near) < len(seed):
        return seed
    tree = cKDTree(near)
    seen = np.zeros(len(near), bool)
    stack = []
    for p in seed:
        for i in tree.query_ball_point(p, eps):
            if not seen[i]:
                seen[i] = True
                stack.append(i)
    while stack:
        for k in tree.query_ball_point(near[stack.pop()], eps):
            if not seen[k]:
                seen[k] = True
                stack.append(k)
    return near[seen] if seen.sum() >= len(seed) else seed


# ────────────────────────────── [B] apex locator ─────────────────────────────
def ransac_planes(P, tol, iters, min_pts, rng, max_planes=6):
    """Sequentially RANSAC up to `max_planes` planes; each refined by SVD on its
    inliers, whose points are removed before the next fit. Returns (n, d, count).

    More than three because the grown cluster also contains tripod geometry —
    the trihedral is then identified by picking the mutually perpendicular
    TRIPLE, not by assuming the first three fits are the plates."""
    planes, pts = [], P.copy()
    for _ in range(max_planes):
        if len(pts) < min_pts:
            break
        best = None
        for _ in range(iters):
            a, b, c = pts[rng.choice(len(pts), 3, replace=False)]
            n = np.cross(b - a, c - a)
            nn = np.linalg.norm(n)
            if nn < 1e-9:
                continue
            n = n / nn
            inl = np.abs(pts @ n - n @ a) < tol
            if best is None or inl.sum() > best[0]:
                best = (int(inl.sum()), inl)
        if best is None or best[0] < min_pts:
            break
        Q = pts[best[1]]
        c0 = Q.mean(0)
        n = np.linalg.svd(Q - c0)[2][2]
        planes.append((n, float(n @ c0), len(Q)))
        pts = pts[~best[1]]
    return planes


def locate_apex(P, tol, iters, min_pts, perp_tol_deg, rng, seed_c=None, mode='auto'):
    """Trihedral apex from a point cluster that may also contain tripod.

    'planes3'  fit several planes, then take the mutually ~perpendicular TRIPLE
               and intersect it. Three mutually perpendicular planes are the
               signature of the trihedral, so this finds the reflector inside a
               mixed cluster instead of assuming the cluster is only reflector.
    'deepest'  fallback when no such triple exists: farthest points along the
               viewing ray. Only valid if the cluster IS the reflector, so it is
               anchored to the seed when one is given — otherwise a grown blob
               containing the tripod returns a point behind the reflector.
    """
    planes = ransac_planes(P, tol, iters, min_pts, rng) if mode in ('auto', 'planes3') else []
    budget = np.cos(np.radians(90.0 - perp_tol_deg))
    best = None
    for i, j, k in combinations(range(len(planes)), 3):
        tri = (planes[i], planes[j], planes[k])
        if any(abs(a[0] @ b[0]) >= budget for a, b in combinations(tri, 2)):
            continue
        try:
            apex = np.linalg.solve(np.stack([p[0] for p in tri]),
                                   np.array([p[1] for p in tri]))
        except np.linalg.LinAlgError:
            continue
        if np.min(np.linalg.norm(P - apex, axis=1)) > 0.10:      # must touch the cloud
            continue
        if seed_c is not None and np.linalg.norm(apex - seed_c) > 0.35:
            continue                                             # not on the reflector
        score = sum(p[2] for p in tri)
        if best is None or score > best[0]:
            best = (score, apex)
    if best is not None:
        return best[1], 'planes3'
    Q = P if seed_c is None else P[np.linalg.norm(P - seed_c, axis=1) <= 0.25]
    if len(Q) < 3:
        Q = P
    if mode == 'deepest':
        # Farthest point along the viewing ray. Only valid when the cluster is
        # PURELY reflector — if any mount or tripod is included it flips between
        # them frame to frame, which reads as tens of mm of apex jitter.
        u = Q.mean(0)
        u = u / (np.linalg.norm(u) + 1e-9)
        k = min(8, len(Q))
        return Q[np.argsort(Q @ u)[-k:]].mean(0), 'deepest'
    # Centroid: not the geometric apex, but averaging hundreds of points makes it
    # stable to a few mm, where 'deepest' rests on a single extreme point. It sits
    # a fixed distance from the true corner, so it biases the solved translation
    # rather than adding noise — the honest trade until the reflector is mounted
    # clear of the tripod and planes3 can find the real corner.
    return Q.mean(0), 'centroid'


# ──────────────────────── [B2] pose coverage (observability) ──────────────────
# Six numbers that decide whether the captures can pin all six DOF. Unlike the
# ChArUco flow there is no board here, so there is no board pitch/roll/yaw to
# spread — the ONLY lever arm is where the reflector sat in the radar's own
# (range, azimuth, elevation). Each target below is the spread at which that
# DOF stops being the limiting one; they come from the geometry, not taste.
#
# Why each row matters — a small rotation d about axis k displaces a point p by
# d(k x p), and the calibration can only see that displacement if the radar
# measures it on an axis it is GOOD at (range 5 cm, azimuth 3 deg, elevation 8 deg):
#
#   YAW   (about Z): k x p = (-p_y, p_x, 0) -> shows up as azimuth error ~ d.
#                    Needs points spread LEFT-RIGHT.               -> AZ
#   PITCH (about Y): k x p = ( p_z, 0, -p_x) -> the -p_x term is an elevation
#                    error ~ d (weak axis), but the p_z term is a RANGE error,
#                    and range is the radar's best axis. So pitch becomes sharply
#                    observable the moment points sit well ABOVE and BELOW
#                    boresight.                                    -> EL, EL BAL
#   ROLL  (about X): k x p = (0, -p_z, p_y) -> the -p_z term is an azimuth error
#                    (good axis) proportional to HEIGHT off boresight. Roll is
#                    therefore unlocked by the same elevation spread as pitch —
#                    with everything in one horizontal plane it is unobservable.
#   TRANSLATION    : a rotation about the lidar and a translation look identical
#                    when every point is at the same distance. RANGE spread is
#                    what separates them.                          -> RANGE
#
# A spread number alone lies when the coverage is one-sided: 16 deg of elevation
# all BELOW boresight fits a pitch just as well as a vertical bias in the apex
# estimate, and the two trade off. Hence the two BAL rows, which ask for real
# coverage on both sides of boresight rather than a wide one-sided smear.
#
# NEAR counts close captures because every angular error is a cross-range error
# of r*sin(sigma): one capture at 1.2 m carries the angular information of three
# at 3.6 m. It is the cheapest way to tighten rotation.
COVERAGE_TARGETS = {
    'range':  1.50,   # m,   max-min spread of radar range
    'az':    60.0,    # deg, max-min spread of radar azimuth
    'az_bal': 20.0,   # deg, coverage on the THINNER side of boresight
    'el':    30.0,    # deg, max-min spread of radar elevation
    'el_bal': 10.0,   # deg, coverage on the THINNER side of boresight
    'near':   6.0,    # count of captures closer than NEAR_RANGE_M
    'cells':  8.0,    # of the 9 az x el cells below, how many hold a capture
}
NEAR_RANGE_M = 1.5

# The az x el map. Every spread row above can be satisfied by a set that is still
# lopsided — 60 deg of azimuth all taken at eye level passes AZ, AZ BAL, EL BAL
# and still leaves pitch loose. Filling cells cannot be gamed that way: it is one
# number that implies both spreads AND both balances at once, and unlike a spread
# it says WHERE the hole is, which is what you need while the tripod is still up.
# Target is 8 of 9, not 9 — the two outer corners (high and far to the side) are
# a long walk for little extra leverage, so one may be skipped.
AZ_EDGES = (-60.0, -20.0, 20.0, 60.0)
EL_EDGES = (-40.0, -10.0, 10.0, 40.0)
# Range bands for the strip; each should hold captures. Near matters most because
# angular error is r*sin(sigma) — see the NEAR row.
RANGE_BANDS = ((0.0, 1.5), (1.5, 3.0), (3.0, 99.0))


def _cell(az, el):
    """(col, row) of a point on the 3x3 map; values past the outer edges clamp
    into the outer cell, so a wide pose still counts rather than vanishing."""
    c = 0 if az < AZ_EDGES[1] else (1 if az < AZ_EDGES[2] else 2)
    r = 0 if el < EL_EDGES[1] else (1 if el < EL_EDGES[2] else 2)
    return c, r

# What to physically do when a given bar is the worst one. Shown as the hint line
# so the answer to "it is red, now what?" is on screen instead of in a document.
# Kept short and action-first: they are drawn into a 268 px panel, so the verb
# has to survive on one line. {az}/{el} are filled in with the PHYSICAL direction
# each radar axis happens to point — see _axis_words. On an upright mount the
# radar's azimuth is left-right and its elevation is up-down, but a radar rolled
# 90 deg swaps them, and telling someone to "change height" when the axis they
# need is horizontal costs a whole collection session.
COVERAGE_HINT = {
    'range':  'move much nearer AND much farther',
    'az':     'carry the tripod wider {az}',
    'az_bal': 'one {az} side is empty - go there',
    'el':     'spread much wider {el}',
    'el_bal': 'all on one side - go both {el}',
    'near':   f'take captures closer than {NEAR_RANGE_M:.1f} m',
    'cells':  'fill the shaded boxes on the map below',
}


def pose_coverage(P_list):
    """Is this capture set able to constrain all six DOF?

    Takes the radar-frame reflector points collected so far and returns
    {name: (value, target, ok)} for the six rows above, plus 'n' and 'worst'
    (the key furthest below its target — what to fix next). See COVERAGE_TARGETS
    for why each row exists and which DOF it unlocks."""
    P = np.asarray(P_list, float).reshape(-1, 3)
    out = {'n': len(P)}
    if len(P) < 2:
        for k, tgt in COVERAGE_TARGETS.items():
            out[k] = (0.0, tgt, False)
        out['filled'] = set()
        out['bands'] = [False] * len(RANGE_BANDS)
        out['worst'] = 'range'
        return out

    raz = np.array([cart_to_raz(p) for p in P])
    rng, az, el = raz[:, 0], np.degrees(raz[:, 1]), np.degrees(raz[:, 2])

    def bal(a):
        """Coverage on the thinner side of boresight — 0 if everything is on one
        side. Deliberately NOT a count: one lonely pose at +30 deg is real
        leverage, twenty crowded at +2 deg is not."""
        return float(min(max(a.max(), 0.0), max(-a.min(), 0.0)))

    filled = {_cell(a, e) for a, e in zip(az, el)}
    vals = {'range': float(rng.max() - rng.min()),
            'az': float(az.max() - az.min()), 'az_bal': bal(az),
            'el': float(el.max() - el.min()), 'el_bal': bal(el),
            'near': float((rng < NEAR_RANGE_M).sum()),
            'cells': float(len(filled))}
    for k, v in vals.items():
        tgt = COVERAGE_TARGETS[k]
        out[k] = (v, tgt, v >= tgt)
    out['filled'] = filled
    out['bands'] = [bool(((rng >= lo) & (rng < hi)).any()) for lo, hi in RANGE_BANDS]
    out['worst'] = min(vals, key=lambda k: vals[k] / COVERAGE_TARGETS[k])
    return out


# ─────────────────────────────── [C] the node ────────────────────────────────
CYAN = ColorRGBA(r=0.1, g=0.85, b=0.95, a=1.0)
GREEN = ColorRGBA(r=0.1, g=1.0, b=0.2, a=1.0)
AMBER = ColorRGBA(r=1.0, g=0.75, b=0.1, a=0.9)
MAGENTA = ColorRGBA(r=1.0, g=0.1, b=0.9, a=1.0)
WHITE = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)


class RadarLidarCalib(Node):
    def __init__(self):
        super().__init__('radar_lidar_calib')
        dp = self.declare_parameter
        # ── topics ──
        dp('lidar_topic', '/ouster/points')
        dp('radar_topic', '/radar1/radar/points_all')
        dp('pc_field_x', 'x'); dp('pc_field_y', 'y'); dp('pc_field_z', 'z')
        dp('pc_field_snr', 'intensity'); dp('pc_field_doppler', 'doppler')
        # ── lidar detection ──
        dp('lidar_min_range', 0.3); dp('lidar_max_range', 8.0)
        dp('bg_frames_lidar', 10)
        dp('bg_voxel', 0.05)                 # background match distance (m). Anything
                                             # within this of a background point is
                                             # erased (up to 3.5x that in the worst
                                             # case), so keep it well under the gap
                                             # between reflector and tripod head.
        dp('cluster_eps', 0.12)              # foreground clustering radius (m)
        dp('min_cluster_size', 8)
        # Region-grow the seed back into the raw cloud to recover the part of the
        # reflector that background subtraction erased. Both radii scale with
        # range: lidar point spacing is angular, so it grows linearly with
        # distance (0.7 deg is 1.2 cm at 1 m but 4.9 cm at 4 m). A fixed
        # connectivity that works up close silently fails far away.
        #     effective = base + per_m * range_to_target
        # Stepwise growth from the tip: expand the radius until the cluster stops
        # growing (object complete) or outgrows a reflector (tripod joined).
        # OFF by default: growth only ever worked around a reflector bolted to its
        # tripod head, and it cannot separate the two cleanly. With the reflector
        # raised clear of the head, background subtraction returns the whole
        # reflector on its own and planes3 finds the real corner — no growth needed.
        dp('grow_max_radius', 0.0)           # 0 = growth off entirely
        dp('grow_min_radius', 0.08)
        dp('grow_step', 0.02)
        dp('grow_plateau_frac', 0.04)        # increment below this fraction => done
        dp('reflector_size', 0.25)           # SET THIS to your reflector's longest
                                             # dimension — it is the hard backstop on
                                             # how far the growth may spread
        dp('apex_method', 'auto')            # 'auto' (planes3 -> centroid) |
                                             # 'centroid' | 'deepest'
        # Isolate the reflector from the rest of the foreground. Pool the
        # background on the EMPTY room (no tripod), then wheel the whole rig in:
        # the foreground is tripod + reflector, hundreds of points, no erased halo
        # and no need to re-pool per placement. The reflector is the topmost part
        # and its size is known, so keeping everything within reflector_size of
        # the cluster's top drops the post, head and legs.
        dp('isolate_reflector', True)
        # The reflector's SIZE is constant, but locating its top is not: beam
        # spacing grows with range (0.7 deg is 1.2 cm at 1 m, 3.7 cm at 3 m), so a
        # fixed band catches a single noisy return far away. Both terms scale so
        # the top stays an average of several points, and the isolation ball gets
        # a small allowance for that top being less certain.
        dp('top_band', 0.03); dp('top_band_per_m', 0.025)
        dp('reflector_size_per_m', 0.02)
        dp('grow_radius_per_m', 0.03)        # range scaling on the max radius
        dp('grow_eps', 0.06); dp('grow_eps_per_m', 0.02)
        dp('cluster_eps_per_m', 0.02)        # same scaling for the seed clustering
        dp('plane_tol', 0.015)               # RANSAC inlier distance (m)
        dp('plane_iters', 250)
        dp('min_plane_pts', 12)              # per plate — sets max planes3 range
        dp('perp_tol_deg', 25.0)
        # ── radar detection ──
        dp('radar_min_range', 0.3); dp('radar_max_range', 8.0)
        dp('bg_frames_radar', 15); dp('bg_match_dist', 0.2)
        # Physical bound on the rig, not a prior: whatever the extrinsic turns out
        # to be, the radar's range to the target must lie within one baseline of
        # the lidar's. Rules out distant strong returns that would otherwise win on
        # raw SNR (a wall at 7 m beating the reflector at 1.7 m). Set it larger
        # than the true lidar-radar separation and forget it; <=0 disables.
        dp('max_baseline_m', 2.0)
        dp('range_gate_margin_m', -1.0)      # <=0 = OFF (default). The reflector is
                                             # the brightest NEW thing in a
                                             # background-subtracted scene, so max-SNR
                                             # needs no range assumption. Enable only
                                             # with a real prior_t_xyz: the gate is
                                             # centred on |apex - t_radar|, so a
                                             # zeroed guess on a rig with any baseline
                                             # rejects the genuine return.
        dp('max_abs_doppler', 0.15)          # tripod target is genuinely static
        dp('radar_cluster_eps', 0.20); dp('radar_min_cluster_size', 3)
        # A single radar frame is not a reliable pick: detections flicker between
        # the reflector and multipath. Pool the last N frames and cluster the
        # accumulation instead — the static reflector lands in the same place
        # every frame (dense, persistent cluster), noise appears once and moves.
        dp('radar_accum_frames', 10)
        dp('radar_min_frames', 5)            # cluster must appear in >= this many frames
        dp('gate_radius', 0.40)              # 3-D gate once a solve exists
        dp('cluster_strict', True)
        dp('min_snr', 100.0)                 # threshold on snr·(r/ref)^4
        dp('min_snr_raw', 25.0)              # absolute floor too: the r^4 scaling
                                             # inflates distant returns enormously
                                             # (snr 58 at 5.7 m normalises to 11800),
                                             # so a weak far blip would otherwise pass
        dp('snr_ref_range', 1.5)
        dp('radar_range_scale', 1.0)         # ingest correction; tune until a≈1
        dp('radar_range_bias_m', 0.0)
        # ── noise model + solver (radar-dominated; lidar apex ~1 cm ≪ these) ──
        dp('sigma_range_m', 0.05); dp('sigma_az_deg', 3.0); dp('sigma_el_deg', 8.0)
        dp('huber_f_scale', 1.5); dp('reject_sigma', 4.0); dp('reject_axis_sigma', 3.5)
        # ── radar position guess + optional solver prior ──
        #   prior_t_xyz is ALWAYS used to predict the radar's range to the target
        #   (r_exp = |apex - t|). That is pure gating, not regularisation: get it
        #   wrong by more than range_gate_margin_m and the real return is thrown
        #   away. use_extrinsic_prior controls only whether it also biases the SOLVE.
        dp('use_extrinsic_prior', False)
        dp('prior_t_xyz', [0.0, 0.0, 0.0]); dp('prior_rpy_deg', [0.0, 0.0, 0.0])
        dp('prior_t_sigma_m', 0.15); dp('prior_rot_sigma_deg', 15.0)
        # ── capture ──
        dp('capture_frames', 5)              # radar selections averaged per capture
        dp('capture_frames_lidar', 3)        # AND this many lidar apexes, see _maybe_finish
        dp('capture_timeout_s', 6.0)
        dp('lidar_std_mm', 15.0)             # apex jitter gate
        dp('radar_std_m', 0.10)              # radar point jitter gate
        dp('min_points', 12)                 # first solve at this many pairs
        dp('min_baseline', 0.15)
        # ── output ──
        dp('measured_baseline_m', -1.0)      # tape lidar→radar distance (check only)
        dp('child_frame', 'radar1_link'); dp('radar_name', 'radar1')
        dp('lidar_name', 'ouster')
        dp('output_path', ''); dp('publish_tf', True)
        dp('status_marker_xyz', [2.0, 0.0, 1.0])
        # ── OPTIONAL camera (verification + composed output only; the SOLVE never
        #    uses it, so an error here cannot corrupt the radar calibration) ──
        dp('camera_frame', 'zed_left_camera_optical_frame')
        dp('image_topic', '/zed/zed_node/left/image_rect_color')
        dp('info_topic', '/zed/zed_node/left/camera_info')
        dp('show_image_overlay', True)     # build/publish the camera overlay
        # RAW feeds (an Arducam publishes /image_raw with real lens distortion)
        # need undistorting before the overlay means anything. Projection would
        # otherwise be correct only because projectPoints re-applies D — which is
        # right on a raw image but wrong the moment anything downstream consumes a
        # rectified one, and the two disagree by tens of pixels at the edges.
        # After rectification the node projects with the new K and ZERO distortion.
        dp('rectify_image', False)
        dp('rectify_alpha', 0.0)           # 0 = crop to valid pixels, 1 = keep full FoV
        dp('show_window', True)            # ALSO pop a native cv2 window for it
        dp('debug_scale', 1.0)             # shrink the published/shown overlay
        dp('show_coverage_hud', True)      # overlay the six observability bars
        # GLIM output, os_lidar -> zed_left_camera_optical_frame (T_lidar_camera)
        dp('lidar_camera_xyz', [-0.074928, -0.066971, -0.091627])
        dp('lidar_camera_quat_xyzw', [-0.497829, -0.498035, 0.501789, 0.502329])
        dp('camera_transform_is', 'lidar_camera')   # 'lidar_camera' | 'camera_lidar'
        # Take the camera transform from the TF tree instead of the two parameters
        # above. Prefer this whenever a static_transform_publisher is already
        # broadcasting it: the parameters make you restate a transform by hand,
        # and the two ways that goes wrong are silent. Its direction is easy to
        # invert, and it is easy to quote it against a DIFFERENT frame than the
        # one the cloud arrives in — os_sensor rather than os_lidar on an Ouster,
        # which are 180 deg apart in yaw. TF knows both and cannot get either
        # wrong. Looked up once, from the cloud's own frame_id to camera_frame.
        dp('camera_transform_from_tf', False)

        g = lambda k: self.get_parameter(k).value
        self.fx, self.fy, self.fz = g('pc_field_x'), g('pc_field_y'), g('pc_field_z')
        self.fsnr, self.fdop = g('pc_field_snr'), g('pc_field_doppler')
        self.lmin, self.lmax = float(g('lidar_min_range')), float(g('lidar_max_range'))
        self.bgl_n, self.bg_voxel = int(g('bg_frames_lidar')), float(g('bg_voxel'))
        self.ceps, self.cmin = float(g('cluster_eps')), int(g('min_cluster_size'))
        self.grow_rmax, self.grow_rmin = float(g('grow_max_radius')), float(g('grow_min_radius'))
        self.grow_step, self.grow_plateau = float(g('grow_step')), float(g('grow_plateau_frac'))
        self.refl_size = float(g('reflector_size'))
        self.apex_method = str(g('apex_method'))
        self.isolate_top = bool(g('isolate_reflector'))
        self.top_band = float(g('top_band'))
        self.top_band_pm = float(g('top_band_per_m'))
        self.refl_size_pm = float(g('reflector_size_per_m'))
        self.grow_eps = float(g('grow_eps'))
        self.grow_r_pm = float(g('grow_radius_per_m'))
        self.grow_eps_pm = float(g('grow_eps_per_m'))
        self.ceps_pm = float(g('cluster_eps_per_m'))
        self.ptol, self.piters = float(g('plane_tol')), int(g('plane_iters'))
        self.pmin, self.perp = int(g('min_plane_pts')), float(g('perp_tol_deg'))
        self.rmin, self.rmax = float(g('radar_min_range')), float(g('radar_max_range'))
        self.bgr_n, self.bg_dist = int(g('bg_frames_radar')), float(g('bg_match_dist'))
        self.rmargin = float(g('range_gate_margin_m'))
        self.max_base = float(g('max_baseline_m'))
        self.max_dop = float(g('max_abs_doppler'))
        self.rceps, self.rcmin = float(g('radar_cluster_eps')), int(g('radar_min_cluster_size'))
        self.acc_n, self.min_frames = int(g('radar_accum_frames')), int(g('radar_min_frames'))
        self.gate_r, self.strict = float(g('gate_radius')), bool(g('cluster_strict'))
        self.min_snr, self.snr_r0 = float(g('min_snr')), float(g('snr_ref_range'))
        self.min_snr_raw = float(g('min_snr_raw'))
        self.rscale, self.rbias = float(g('radar_range_scale')), float(g('radar_range_bias_m'))
        self.sig_r = float(g('sigma_range_m'))
        self.sig_az = np.radians(float(g('sigma_az_deg')))
        self.sig_el = np.radians(float(g('sigma_el_deg')))
        self.huber, self.rej = float(g('huber_f_scale')), float(g('reject_sigma'))
        self.rej_axis = float(g('reject_axis_sigma'))
        self.use_prior = bool(g('use_extrinsic_prior'))
        self.t_prior = np.array(g('prior_t_xyz'), float)
        self.R_prior = Rot.from_euler('xyz', g('prior_rpy_deg'), degrees=True).as_matrix()
        self.t_psig = float(g('prior_t_sigma_m'))
        self.r_psig = np.radians(float(g('prior_rot_sigma_deg')))
        self.cap_n, self.cap_to = int(g('capture_frames')), float(g('capture_timeout_s'))
        self.cap_nl = int(g('capture_frames_lidar'))
        self.lstd_max, self.rstd_max = float(g('lidar_std_mm')), float(g('radar_std_m'))
        self.min_points, self.min_base = int(g('min_points')), float(g('min_baseline'))
        self.meas_base = float(g('measured_baseline_m'))
        self.child_frame, self.radar_name = g('child_frame'), g('radar_name')
        self.lidar_name = g('lidar_name')
        self.lidar_topic, self.radar_topic = g('lidar_topic'), g('radar_topic')
        self.out_path = g('output_path') or f'extrinsic_{self.lidar_name}__{self.radar_name}'
        self.publish_tf = bool(g('publish_tf'))
        self.status_xyz = list(g('status_marker_xyz'))
        self.camera_frame = g('camera_frame')

        # ── camera transform: store as T_cam_lidar (p_cam = R_cl·p_lidar + t_cl) ──
        self.cam_tf_wanted = bool(g('camera_transform_from_tf'))
        self.tf_buf = self.tf_lis = None
        if self.cam_tf_wanted:
            self.tf_buf = Buffer()
            self.tf_lis = TransformListener(self.tf_buf, self)
        Rg = Rot.from_quat(list(g('lidar_camera_quat_xyzw'))).as_matrix()
        tg = np.array(g('lidar_camera_xyz'), float)
        if str(g('camera_transform_is')) == 'lidar_camera':      # given maps cam -> lidar
            self.R_cl, self.t_cl = Rg.T, -Rg.T @ tg
        else:                                                    # given already maps lidar -> cam
            self.R_cl, self.t_cl = Rg, tg
        ax = {'lidar +X': self.R_cl @ [1, 0, 0], 'lidar +Y': self.R_cl @ [0, 1, 0],
              'lidar +Z': self.R_cl @ [0, 0, 1]}
        self.get_logger().info(
            'T_cam_lidar (for the composed output / overlay only): t=['
            + ' '.join(f'{v:+.4f}' for v in self.t_cl) + '] m  |  '
            + '  '.join(f'{k}->[{v[0]:+.2f} {v[1]:+.2f} {v[2]:+.2f}]' for k, v in ax.items()))

        self.rng = np.random.default_rng(0)
        self.lidar_frame = None
        self.bg_lidar = None; self.bgl_accum = []; self.bgl_want = 0
        self.bg_radar = None; self.bgr_accum = []; self.bgr_want = 0
        self.det = None                  # latest lidar detection
        self.det_t = 0.0
        self.sel = None                  # latest radar selection
        self.aim = ('starting up — no sensor data yet', 'red')
        self.lidar_stat = 'lidar: waiting'
        self.lidar_msg_t = 0.0
        self.radar_msg_t = 0.0
        self.acc = deque(maxlen=int(g('radar_accum_frames')))   # rolling radar frames
        self.frame_n = 0
        self.cap_deadline = 0.0; self.cap_lidar = []; self.cap_radar = []
        self.captures = []
        self.solution = None
        self.tfb = StaticTransformBroadcaster(self) if self.publish_tf else None

        qs = qos_profile_sensor_data
        self.create_subscription(PointCloud2, g('lidar_topic'), self._lidar, qs)
        self.create_subscription(PointCloud2, g('radar_topic'), self._radar, qs)
        self.create_subscription(Empty, '~/background', lambda _: self._bg_start(), 1)
        self.create_subscription(Empty, '~/capture', lambda _: self._arm(), 1)
        self.create_subscription(Empty, '~/solve', lambda _: self._solve(force=True), 1)
        self.create_subscription(Empty, '~/reset', lambda _: self._reset(), 1)
        self.create_subscription(Empty, '~/save', lambda _: self._save(), 1)
        self.pub_apex = self.create_publisher(PointStamped, '~/apex', 5)
        self.pub_mk = self.create_publisher(MarkerArray, '~/markers', 2)
        self.create_timer(0.1, self._markers)
        self.create_timer(1.0, self._heartbeat)

        # optional image overlay (verification only)
        self.K = self.D = self.img = None
        self.map1 = self.map2 = None
        self.rectify = bool(g('rectify_image'))
        self.rectify_alpha = float(g('rectify_alpha'))
        self.bridge = CvBridge() if _HAVE_CV else None
        self.overlay_on = bool(g('show_image_overlay')) and _HAVE_CV
        self.show_window = bool(g('show_window'))
        self.dscale = float(g('debug_scale'))
        self.show_cov = bool(g('show_coverage_hud'))
        self._rot_sig_deg = self._t_sig_mm = None  # last solve's per-DOF 1s, for the HUD
        if self.overlay_on:
            self.create_subscription(Image, g('image_topic'), self._image, qs)
            self.create_subscription(CameraInfo, g('info_topic'), self._info, qs)
            self.pub_img = self.create_publisher(Image, '~/debug_image', 2)
            self.create_timer(0.05, self._overlay)
        self.get_logger().info(
            'radar_lidar_calib ready — solves T_lidar_radar; camera used only for '
            'the composed T_cam_radar'
            + (' + image overlay.\n' if self.overlay_on else ' (overlay off).\n')
            + '  RViz: Fixed Frame = your lidar frame, add the cloud + MarkerArray '
            'on ~/markers\n'
            '  per placement: reflector OFF -> ~/background | reflector ON, aim, '
            'step out -> ~/capture')

    # ── control ──
    def _bg_start(self):
        self.bgl_accum, self.bgl_want, self.bg_lidar = [], self.bgl_n, None
        self.bgr_accum, self.bgr_want, self.bg_radar = [], self.bgr_n, None
        self.acc.clear()
        self.get_logger().info(
            f'pooling background: lidar {self.bgl_n} + radar {self.bgr_n} frames — '
            f'reflector OFF the tripod, stay out of view')

    def _arm(self):
        if self.bg_lidar is None or self.bg_radar is None:
            self.get_logger().warn('capture refused: background not pooled — ~/background first')
            return
        if self.det is None or time.time() - self.det_t > 1.0:
            self.get_logger().warn('capture refused: no live lidar detection '
                                   '(reflector mounted? in range? background stale?)')
            return
        self.cap_lidar, self.cap_radar = [], []
        self.cap_deadline = time.time() + self.cap_to
        self.get_logger().info(f'capture armed: pairing next {self.cap_n} radar frames '
                               f'(timeout {self.cap_to:.0f} s)')

    def _reset(self):
        self.captures, self.solution = [], None
        self.get_logger().info('captures cleared')

    def _current_T(self):
        if self.solution is not None:
            return self.solution['R'], self.solution['t']
        if self.use_prior:
            return self.R_prior, self.t_prior
        return None, None

    def _heartbeat(self):
        """Name a silent topic rather than letting the status line go stale —
        'waiting for X' is otherwise indistinguishable from 'X never arrived'."""
        now = time.time()
        dead = []
        if now - self.lidar_msg_t > 3.0:
            dead.append(f'LIDAR silent ({self.lidar_topic})')
        if now - self.radar_msg_t > 3.0:
            dead.append(f'RADAR silent ({self.radar_topic})')
        if dead:
            self.aim = (' | '.join(dead) + ' — check the topic name and that it is publishing',
                        'red')
        if self.cap_deadline and time.time() > self.cap_deadline:
            self._capture_timeout()
        if self.det is None:
            self.get_logger().info(self.lidar_stat, throttle_duration_sec=2.0)
        self.get_logger().info(self.aim[0], throttle_duration_sec=2.0)

    # ── lidar: background-subtract → cluster → apex ──
    def _resolve_cam_tf(self):
        """Look the camera transform up in TF, once, using the frame the cloud
        actually arrives in. Silent until it succeeds — a static publisher may
        start after this node, so failing here is normal for the first seconds."""
        if not self.cam_tf_wanted or self.lidar_frame is None:
            return
        try:
            tr = self.tf_buf.lookup_transform(
                self.lidar_frame, self.camera_frame, rclpy.time.Time()).transform
        except Exception as e:
            self.get_logger().warn(
                f'camera_transform_from_tf: no {self.lidar_frame} -> {self.camera_frame} '
                f'yet ({type(e).__name__}) — is the static publisher running?',
                throttle_duration_sec=5.0)
            return
        # lookup gives p_lidar = Rg·p_cam + tg; the node stores the inverse
        Rg = Rot.from_quat([tr.rotation.x, tr.rotation.y,
                            tr.rotation.z, tr.rotation.w]).as_matrix()
        tg = np.array([tr.translation.x, tr.translation.y, tr.translation.z])
        self.R_cl, self.t_cl = Rg.T, -Rg.T @ tg
        self.cam_tf_wanted = False                    # resolved; stop looking
        ax = {'lidar +X': self.R_cl @ [1, 0, 0], 'lidar +Y': self.R_cl @ [0, 1, 0],
              'lidar +Z': self.R_cl @ [0, 0, 1]}
        self.get_logger().info(
            f'camera transform taken from TF ({self.lidar_frame} -> {self.camera_frame}): '
            f'camera sits [' + ' '.join(f'{v:+.4f}' for v in tg) + '] m from the lidar  |  '
            + '  '.join(f'{k}->[{v[0]:+.2f} {v[1]:+.2f} {v[2]:+.2f}]' for k, v in ax.items()))

    def _lidar(self, msg):
        self.lidar_msg_t = time.time()
        self.lidar_frame = msg.header.frame_id
        if self.cam_tf_wanted:
            self._resolve_cam_tf()
        xyz = cloud_xyz(msg)
        if len(xyz) == 0:
            return
        r = np.linalg.norm(xyz, axis=1)
        xyz = xyz[np.isfinite(r) & (r > self.lmin) & (r < self.lmax)]

        if self.bgl_want > 0:
            self.bgl_accum.append(voxel_keys(xyz, self.bg_voxel))
            self.bgl_want -= 1
            if self.bgl_want == 0:
                self.bg_lidar = np.unique(np.concatenate(self.bgl_accum))
                self.bgl_accum = []
                self.get_logger().info(f'lidar background ready: {len(self.bg_lidar)} voxels')
            return
        if self.bg_lidar is None:
            self.det = None
            self.lidar_stat = f'lidar: {len(xyz)} pts in range — no background yet'
            return

        fg = xyz[foreground_mask(xyz, self.bg_lidar, self.bg_voxel)]
        # scale the seed connectivity by how far the new points are
        r_fg = float(np.median(np.linalg.norm(fg, axis=1))) if len(fg) else 0.0
        ceps_eff = self.ceps + self.ceps_pm * r_fg
        clusters = lidar_cluster(fg, ceps_eff, self.cmin)
        if not clusters:
            # the counts say WHICH stage lost it: no foreground at all means the
            # background is eating the target (reflector too close to something
            # memorised, e.g. mounted straight on the tripod head — lower
            # bg_voxel or raise it off the head); foreground but no cluster means
            # min_cluster_size is too high or cluster_eps too tight.
            self.det = None
            self.lidar_stat = (f'lidar: {len(xyz)} in range -> {len(fg)} new -> '
                               + ('NO new points (background is eating it — lower '
                                  'bg_voxel / raise the reflector off the tripod)'
                                  if len(fg) == 0 else
                                  f'no cluster >= {self.cmin} pts (lower min_cluster_size '
                                  f'or raise cluster_eps)'))
            return
        P = clusters[0]
        n_full = len(P)
        if self.isolate_top and len(P) > 3:
            # robust top: mean of the highest points, not a single extreme return
            r_c = float(np.linalg.norm(P.mean(0)))
            band = self.top_band + self.top_band_pm * r_c
            ball = self.refl_size + self.refl_size_pm * r_c
            z = P[:, 2]                                  # os_lidar +Z is up
            top = P[z >= z.max() - band].mean(0)
            iso = P[np.linalg.norm(P - top, axis=1) <= ball]
            if len(iso) >= 3:
                P = iso
        n_seed = len(P)
        r_seed = float(np.linalg.norm(P.mean(0)))
        grow_r = self.grow_rmax + self.grow_r_pm * r_seed
        grow_eps = self.grow_eps + self.grow_eps_pm * r_seed   # point spacing scales
        seed_c = P.mean(0)
        why = 'off'
        if self.grow_rmax > 0:              # tip -> whole reflector, stop at the break
            P, used_r, why = grow_stepwise(xyz, P, grow_eps, self.grow_rmin, grow_r,
                                           self.grow_step, self.grow_plateau, self.refl_size)
        else:
            used_r = 0.0
        apex, method = locate_apex(P, self.ptol, self.piters, self.pmin, self.perp,
                                   self.rng, seed_c=seed_c, mode=self.apex_method)
        self.det = dict(apex=apex, cluster=P, method=method,
                        n_fg=len(fg), n_extra=len(clusters) - 1)
        self.det_t = time.time()
        self.lidar_stat = (f'lidar: {len(fg)} new -> cluster {n_full} -> reflector {n_seed} '
                           f'@ {r_seed:.1f} m, {method}'
                           + (f', +{len(clusters)-1} EXTRA' if len(clusters) > 1 else ''))
        if self.cap_deadline > time.time():
            self.cap_lidar.append(apex.copy())
            self._maybe_finish()
        ps = PointStamped()
        ps.header = msg.header
        ps.point.x, ps.point.y, ps.point.z = map(float, apex)
        self.pub_apex.publish(ps)

    # ── radar: background-subtract → range gate → cluster → SNR centroid ──
    def _radar(self, msg):
        self.radar_msg_t = time.time()
        f = cloud_fields(msg, [self.fx, self.fy, self.fz, self.fsnr, self.fdop])
        if f[self.fx] is None:
            have = ', '.join(fl.name for fl in msg.fields)
            self.aim = (f'radar: no field "{self.fx}" — cloud has: {have}', 'red')
            return
        z = f[self.fz] if f[self.fz] is not None else np.zeros_like(f[self.fx])
        xyz = np.stack([f[self.fx], f[self.fy], z], 1)
        snr = f[self.fsnr] if f[self.fsnr] is not None else np.ones(len(xyz))
        dop = f[self.fdop]
        if self.rscale != 1.0 or self.rbias != 0.0:
            rr = np.linalg.norm(xyz, axis=1)
            ok = rr > 1e-6
            xyz[ok] *= ((self.rscale * rr[ok] + self.rbias) / rr[ok])[:, None]
        r = np.linalg.norm(xyz, axis=1)
        keep = np.isfinite(r) & (r > self.rmin) & (r < self.rmax)

        if self.bgr_want > 0:
            self.bgr_accum.append(xyz[keep])
            self.bgr_want -= 1
            if self.bgr_want == 0:
                self.bg_radar = (np.concatenate(self.bgr_accum) if self.bgr_accum
                                 else np.zeros((0, 3)))
                self.bgr_accum = []
                self.get_logger().info(f'radar background ready: {len(self.bg_radar)} points')
            return
        if self.bg_radar is None:
            self.aim = ('NO BACKGROUND — reflector OFF the tripod, then ~/background', 'red')
            return
        if len(self.bg_radar) and keep.any():
            idx = np.where(keep)[0]
            d = np.linalg.norm(xyz[idx][:, None, :] - self.bg_radar[None, :, :], axis=2).min(1)
            keep[idx[d <= self.bg_dist]] = False

        self.sel = None
        if self.det is None or time.time() - self.det_t > 1.0:
            self.aim = ('no lidar detection — mount the reflector / re-do background', 'red')
            return
        apex = self.det['apex']
        R, t = self._current_T()
        # Rotation-invariant range gate: |p_radar| = |apex − t_radar| for ANY R, so
        # only the radar's POSITION matters here. Use the solved t once available,
        # otherwise the guess — never 0, or a metre of baseline rejects every real
        # return (the symptom is 'no return near lidar range' while a strong,
        # correct return sits one baseline away).
        t_gate = t if t is not None else self.t_prior
        r_exp = np.linalg.norm(apex - t_gate)
        if self.max_base > 0:
            # |r_radar - r_lidar| <= baseline by the triangle inequality, for ANY
            # rotation and any actual baseline up to max_baseline_m.
            r_lid = float(np.linalg.norm(apex))
            keep &= np.abs(r - r_lid) <= self.max_base
        if self.rmargin > 0:
            keep &= np.abs(r - r_exp) <= self.rmargin
        if self.max_dop > 0 and dop is not None:
            keep &= np.abs(dop) <= self.max_dop
        # accumulate this frame, then work on the pooled cloud
        self.acc.append((xyz[keep], snr[keep], self.frame_n))
        self.frame_n += 1
        pts = np.concatenate([a[0] for a in self.acc]) if self.acc else np.zeros((0, 3))
        sr = np.concatenate([a[1] for a in self.acc]) if self.acc else np.zeros(0)
        fid = (np.concatenate([np.full(len(a[0]), a[2]) for a in self.acc])
               if self.acc else np.zeros(0, int))
        n_frames = len(self.acc)
        if len(pts) == 0:
            self.aim = ('radar: nothing within %.1f m of the lidar range (%.2f m) — '
                        'RE-AIM / re-do background' % (self.max_base, np.linalg.norm(apex))
                        if self.max_base > 0 else
                        'radar: nothing new after background (%d frames pooled) — '
                        'RE-AIM / re-do background' % n_frames, 'red')
            return

        pred = R.T @ (apex - t) if R is not None else None
        if pred is not None:
            # Widen the gate by what the solve's own uncertainty implies at this
            # range: a rotation 1sigma of s degrees displaces the prediction by
            # r*sin(s). Gating tighter than that refuses correct returns, which is
            # how an early, poorly-conditioned solve locks itself out of collecting
            # the very captures that would improve it.
            gate = self.gate_r
            if self.solution is not None:
                rs = np.degrees(np.sqrt(np.clip(np.diag(self.solution['cov'])[:3], 0, None))).max()
                gate += float(np.linalg.norm(apex - t) * np.sin(np.radians(min(rs, 45.0))))
            near = np.linalg.norm(pts - pred, axis=1) <= gate
            if near.any():
                pts, sr, fid = pts[near], sr[near], fid[near]
            elif self.strict and self.solution is not None:
                self.aim = (f'radar: nothing within {gate:.2f} m of prediction', 'orange')
                return
        clusters = radar_cluster(pts, self.rceps, self.rcmin)
        # persistence: how many DISTINCT frames contributed to each cluster. The
        # reflector scores ~n_frames; a one-off multipath spike scores 1.
        persist = [len(np.unique(fid[c])) for c in clusters]
        good = [i for i, k in enumerate(persist) if k >= min(self.min_frames, n_frames)]
        if not good:
            best = max(persist) if persist else 0
            self.aim = (f'radar: no persistent cluster (best {best}/{n_frames} frames, '
                        f'need {self.min_frames}) — flickering / re-aim', 'orange')
            return
        if pred is not None:
            ci = good[int(np.argmin([np.linalg.norm(pts[clusters[i]].mean(0) - pred)
                                     for i in good]))]
        else:
            ci = good[int(np.argmax([sr[clusters[i]].max() for i in good]))]
        c = clusters[ci]
        w = sr[c] / max(sr[c].sum(), 1e-9)
        p_sel = (pts[c] * w[:, None]).sum(0)
        snr_sel = float(sr[c].max())
        n_seen = persist[ci]
        r_sel = float(np.linalg.norm(p_sel))
        snr_norm = snr_sel * (r_sel / self.snr_r0) ** 4
        ok = snr_norm >= self.min_snr and snr_sel >= self.min_snr_raw
        self.sel = dict(p=p_sel, snr=snr_sel, snr_norm=snr_norm, r=r_sel, n=len(c),
                        seen=n_seen, frames=n_frames)
        # Range agreement is reported, not enforced, until a solve exists: before
        # then the baseline is unknown, so a mismatch is uninformative. After the
        # solve the 3-D prediction gate above is already doing the real work.
        gap = abs(r_sel - r_exp)
        self.aim = (f'radar: best {snr_sel:.0f} (norm {snr_norm:.0f}) @ {r_sel:.2f} m'
                    f' [{n_seen}/{n_frames} frames]'
                    + (f' (lidar {r_exp:.2f}, d {gap*100:.0f} cm)' if self.solution else '')
                    + '  ' + ('OK' if ok else 'RE-AIM'), 'green' if ok else 'orange')

        if self.cap_deadline > time.time() and ok:
            self.cap_radar.append(p_sel.copy())
            self._maybe_finish()
        elif self.cap_deadline and time.time() > self.cap_deadline:
            self._capture_timeout()

    def _maybe_finish(self):
        """Close the capture window only when BOTH sensors have their quota.

        Closing on the radar count alone was a race: the radar publishes faster
        than the lidar spins, so five radar frames can land inside a single lidar
        revolution and the capture is finalised with one or two apexes — then
        refused for 'too few lidar detections' even though the placement was
        perfect and just needed another 200 ms."""
        if len(self.cap_radar) >= self.cap_n and len(self.cap_lidar) >= self.cap_nl:
            self.cap_deadline = 0.0
            self._finish_capture()

    def _capture_timeout(self):
        """Name whichever sensor came up short — 'timeout' alone does not say
        whether to re-aim the reflector or to fix the lidar detection."""
        self.cap_deadline = 0.0
        short = []
        if len(self.cap_radar) < self.cap_n:
            short.append(f'radar {len(self.cap_radar)}/{self.cap_n} (aim: {self.aim[0]})')
        if len(self.cap_lidar) < self.cap_nl:
            short.append(f'lidar {len(self.cap_lidar)}/{self.cap_nl} ({self.lidar_stat})')
        self.get_logger().warn(
            f'capture REFUSED: timeout after {self.cap_to:.0f} s — ' + ' | '.join(short))

    # ── atomic capture: both sensors must pass ──
    def _finish_capture(self):
        if len(self.cap_lidar) < self.cap_nl:
            self.get_logger().warn(
                f'capture REFUSED: only {len(self.cap_lidar)}/{self.cap_nl} lidar '
                f'detections in the window ({self.lidar_stat})')
            return
        L, Rr = np.stack(self.cap_lidar), np.stack(self.cap_radar)
        lstd = float(np.linalg.norm(L.std(0)) * 1000)
        rstd = float(np.linalg.norm(Rr.std(0)))
        if lstd > self.lstd_max:
            self.get_logger().warn(f'capture REFUSED: lidar apex std {lstd:.1f} mm > '
                                   f'{self.lstd_max:.0f} (something still moving)')
            return
        if rstd > self.rstd_max:
            self.get_logger().warn(f'capture REFUSED: radar point std {rstd*100:.0f} cm > '
                                   f'{self.rstd_max*100:.0f} (multipath flicker — nudge or re-aim)')
            return
        p_lidar, p_radar = L.mean(0), Rr.mean(0)
        # Final geometric sanity: the two sensors are on one rig, so their ranges
        # to the same target cannot differ by more than the baseline. Re-checked
        # here on the AVERAGED pair, so a frame that slipped past the live gate
        # still cannot become a capture.
        if self.max_base > 0:
            dr = abs(np.linalg.norm(p_radar) - np.linalg.norm(p_lidar))
            if dr > self.max_base:
                self.get_logger().warn(
                    f'capture REFUSED: radar {np.linalg.norm(p_radar):.2f} m vs lidar '
                    f'{np.linalg.norm(p_lidar):.2f} m differ by {dr:.2f} m > '
                    f'max_baseline {self.max_base:.1f} m — wrong object')
                return
        for i, cp in enumerate(self.captures):
            if np.linalg.norm(np.array(cp['p_lidar']) - p_lidar) < self.min_base:
                self.get_logger().warn(f'note: {np.linalg.norm(np.array(cp["p_lidar"])-p_lidar)*100:.0f}'
                                       f' cm from capture #{i+1} — move the tripod further')
                break
        self.captures.append(dict(
            idx=len(self.captures) + 1, stamp=time.time(),
            p_lidar=[round(float(v), 4) for v in p_lidar],
            p_radar=[round(float(v), 4) for v in p_radar],
            method=self.det['method'], snr=round(float(self.sel['snr']), 1),
            radar_frames_seen=int(self.sel['seen']), radar_frames_pooled=int(self.sel['frames']),
            lidar_std_mm=round(lstd, 1), radar_std_mm=round(rstd * 1000, 1),
            # solve_from_poses_* compatibility: identity pose, apex offset zero
            board_R_quat_xyzw=[0.0, 0.0, 0.0, 1.0],
            board_t=[round(float(v), 4) for v in p_lidar]))
        self.get_logger().info(
            f'*** CAPTURED #{len(self.captures)}  lidar [{p_lidar[0]:.3f} {p_lidar[1]:.3f} '
            f'{p_lidar[2]:.3f}]  radar [{p_radar[0]:.3f} {p_radar[1]:.3f} {p_radar[2]:.3f}]  '
            f'{self.det["method"]}  snr {self.sel["snr"]:.0f} ***')
        if len(self.captures) >= self.min_points:
            self._solve()
        self._save(quiet=True)

    # ── solve: measurement-space ML, offset pinned at zero ──
    def _solve(self, force=False):
        n = len(self.captures)
        if n < (4 if force else self.min_points):
            self.get_logger().info(f'{n} captures — first solve at {self.min_points} '
                                   f'(~/solve to force)')
            return
        P = np.array([c['p_radar'] for c in self.captures])
        Q = np.array([c['p_lidar'] for c in self.captures])
        I = np.repeat(np.eye(3)[None], n, axis=0)
        res = robust_ml_calibrate(
            P, I, Q, np.zeros(3), self.sig_r, self.sig_az, self.sig_el,
            use_elevation=True, solve_offset=False,
            R_prior=self.R_prior if self.use_prior else None,
            t_prior=self.t_prior if self.use_prior else None,
            rot_prior_sigma=self.r_psig if self.use_prior else None,
            t_prior_sigma=self.t_psig if self.use_prior else None,
            huber=self.huber, reject_sigma=self.rej, reject_axis_sigma=self.rej_axis)
        self.solution = res
        R, t, mask = res['R'], res['t'], res['inlier_mask']
        Pin, Qin = P[mask], Q[mask]
        sig = np.sqrt(np.clip(np.diag(res['cov']), 0, None))
        rot1s, t1s = np.degrees(sig[:3]), sig[3:] * 1000
        self._rot_sig_deg, self._t_sig_mm = rot1s, t1s      # for the coverage HUD
        err = ((R @ Pin.T).T + t) - Qin
        bias, rms = err.mean(0) * 1000, np.sqrt((err ** 2).mean(0)) * 1000
        raz_in = np.array([cart_to_raz(p) for p in Pin])
        loo = loo_cross_val(Pin, I[mask], Qin, np.zeros(3),
                            (self.sig_r, self.sig_az, self.sig_el), True)
        cond = condition_number(Pin)
        q = Rot.from_matrix(R).as_quat()
        lid_r, rad_r = np.linalg.norm(Qin - t, axis=1), np.linalg.norm(Pin, axis=1)
        a, b = np.linalg.lstsq(np.vstack([rad_r, np.ones_like(rad_r)]).T, lid_r, rcond=None)[0]
        axes = {k: R @ v for k, v in (('X fwd', [1, 0, 0]), ('Y left', [0, 1, 0]),
                                      ('Z up', [0, 0, 1]))}
        L = [f'=== T_{self.lidar_frame or "lidar"}_{self.child_frame}  (lidar <- radar) ===',
             f'  captures {n}   inliers {res["n_in"]}/{n}   residual {res["rms_sigma"]:.2f} s'
             f'   cond {cond:.1f}',
             f'  xyz (m) : {t[0]:+.4f} {t[1]:+.4f} {t[2]:+.4f}   |t| {np.linalg.norm(t)*100:.1f} cm',
             f'  quat    : {q[0]:+.4f} {q[1]:+.4f} {q[2]:+.4f} {q[3]:+.4f}',
             f'  1s rot  : {rot1s[0]:.2f} {rot1s[1]:.2f} {rot1s[2]:.2f} deg'
             f'   1s t: {t1s[0]:.1f} {t1s[1]:.1f} {t1s[2]:.1f} mm',
             '  coverage: ' + self._coverage_text(Pin)[0],
             f'  bias mm : {bias[0]:+.0f} {bias[1]:+.0f} {bias[2]:+.0f}'
             f'   3-D RMS mm: {rms[0]:.0f} {rms[1]:.0f} {rms[2]:.0f}',
             '  radar axes in lidar frame: '
             + '  '.join(f'{k}->[{v[0]:+.2f} {v[1]:+.2f} {v[2]:+.2f}]' for k, v in axes.items())]
        if loo:
            L.append(f'  LOO CV  : {loo[0]:.2f} s (max {loo[1]:.2f})')
        if abs(a - 1) > 0.02 or abs(b) > 0.05:
            L.append(f'  range fit: lidar_r = {a:.3f}*radar_r {b:+.3f} m (want a~1) -> '
                     f'set radar_range_scale={a*self.rscale:.4f}')
        if self.meas_base > 0:
            d = abs(np.linalg.norm(t) - self.meas_base)
            L.append(f'  baseline: |t| {np.linalg.norm(t)*100:.1f} vs tape '
                     f'{self.meas_base*100:.1f} cm -> {d*100:.1f} cm '
                     f'[{"OK" if d <= 0.05 else "MISMATCH"}]')
        # ── does each angular channel actually MEASURE anything? ──
        # Regress what the radar reported against what the solved extrinsic says
        # it should have reported. A working channel gives slope ~1. A slope near
        # zero means the radar is emitting a near-constant angle, which no capture
        # plan can fix and which silently pins the matching translation axis —
        # the failure that cost three collection sessions before it was measured.
        pred_raz = np.array([cart_to_raz(R.T @ (q - t)) for q in Qin])
        chan = []
        for i, nm in ((1, 'az'), (2, 'el')):
            x, y = np.degrees(pred_raz[:, i]), np.degrees(raz_in[:, i])
            if np.ptp(x) < 5.0:                  # too little leverage to judge
                chan.append(f'{nm} slope n/a (only {np.ptp(x):.0f} deg of spread)')
                continue
            a_c = np.polyfit(x, y, 1)[0]
            chan.append(f'{nm} slope {a_c:+.2f}' + ('' if abs(a_c - 1) < 0.3 else ' !!'))
        L.append('  channels: ' + '   '.join(chan) + '   (want ~+1.00 on both)')
        dead = [nm for i, nm in ((1, 'az'), (2, 'el'))
                if np.ptp(np.degrees(pred_raz[:, i])) >= 5.0
                and abs(np.polyfit(np.degrees(pred_raz[:, i]),
                                   np.degrees(raz_in[:, i]), 1)[0] - 1) > 0.5]
        for nm in dead:
            axis = R @ ([0, 1, 0] if nm == 'az' else [0, 0, 1])
            where = 'VERTICAL' if abs(axis[2]) > 0.7 else 'HORIZONTAL'
            L.append(f'  !! the {nm.upper()} channel is not tracking the target — the radar is '
                     f'reporting a near-constant angle. That axis points {where} on this '
                     f'mount, so the {where.lower()} part of t is unmeasured and its '
                     f'coverage bar cannot be satisfied by collecting more')
        n_planes = sum(1 for c in self.captures if c.get('method') == 'planes3')
        if n_planes < 0.5 * n:
            L.append(f'  !! {n - n_planes}/{n} captures used a CENTROID, not the plate '
                     f'intersection — the lidar point sits a few cm off the true corner, '
                     f'so t carries a systematic offset no gate below will catch')
        gates = [('residual~1s', res['rms_sigma'] <= 1.5), ('cond<=5', cond <= 5),
                 ('rot1s<=4deg', rot1s.max() <= 4), ('bias<=50mm', np.abs(bias).max() <= 50)]
        L.append('  GATES   : ' + '  '.join(f'{k}[{"P" if v else "F"}]' for k, v in gates)
                 + '   + RViz up/down check before ~/save')
        cov_txt, cov_hint, cov_ok = self._coverage_text(Pin)
        if not cov_ok:
            # A loose rotation is nearly always a coverage problem, not a solver
            # one, so name the missing geometry instead of the failing gate.
            L.append(f'  NEXT    : {cov_hint}')
        Rcr, tcr = self._compose_cam_radar(R, t)          # T_cam_radar, for deployment
        qc = Rot.from_matrix(Rcr).as_quat()
        L += [f'  --- composed T_cam_radar = T_cam_lidar * T_lidar_radar ---',
              f'  xyz (m) : {tcr[0]:+.4f} {tcr[1]:+.4f} {tcr[2]:+.4f}'
              f'   |t| {np.linalg.norm(tcr)*100:.1f} cm',
              f'  quat    : {qc[0]:+.4f} {qc[1]:+.4f} {qc[2]:+.4f} {qc[3]:+.4f}',
              '  radar axes in CAMERA frame: '
              + '  '.join(f'{k}->[{v[0]:+.2f} {v[1]:+.2f} {v[2]:+.2f}]'
                          for k, v in (('X fwd', Rcr @ [1, 0, 0]), ('Y left', Rcr @ [0, 1, 0]),
                                       ('Z up', Rcr @ [0, 0, 1])))]
        self.get_logger().info('\n' + '\n'.join(L))
        if self.tfb is not None and self.lidar_frame:
            tf = TransformStamped()
            tf.header.stamp = self.get_clock().now().to_msg()
            tf.header.frame_id = self.lidar_frame
            tf.child_frame_id = self.child_frame
            (tf.transform.translation.x, tf.transform.translation.y,
             tf.transform.translation.z) = map(float, t)
            (tf.transform.rotation.x, tf.transform.rotation.y,
             tf.transform.rotation.z, tf.transform.rotation.w) = map(float, q)
            self.tfb.sendTransform(tf)

    # ── save ──
    def _save(self, quiet=False):
        g = lambda k: self.get_parameter(k).value
        out = dict(kind='radar_lidar_session', stamp=time.time(),
                   parent_frame=self.lidar_frame or 'lidar',
                   child_frame=self.child_frame,
                   lidar_name=self.lidar_name, radar_name=self.radar_name,
                   note='T_lidar_radar. Compose later: T_cam_radar = T_cam_lidar * T_lidar_radar',
                   params=dict(
                       sigma_range_m=self.sig_r, sigma_az_deg=float(np.degrees(self.sig_az)),
                       sigma_el_deg=float(np.degrees(self.sig_el)),
                       reflector_offset_x=0.0, reflector_offset_y=0.0, reflector_offset_z=0.0,
                       use_extrinsic_prior=self.use_prior,
                       prior_t_xyz=[float(v) for v in self.t_prior],
                       prior_rpy_deg=list(g('prior_rpy_deg')),
                       prior_t_sigma_m=self.t_psig,
                       prior_rot_sigma_deg=float(np.degrees(self.r_psig)),
                       radar_range_scale=self.rscale, radar_range_bias_m=self.rbias,
                       reject_sigma=self.rej, reject_axis_sigma=self.rej_axis,
                       min_snr=self.min_snr, gate_radius=self.gate_r),
                   captures=self.captures)
        if self.captures:
            # observability of what was actually collected, so a session can be
            # judged later without re-deriving where the reflector was placed
            cov = pose_coverage([c['p_radar'] for c in self.captures])
            out['coverage'] = {k: dict(value=round(cov[k][0], 2), target=cov[k][1],
                                       ok=bool(cov[k][2]))
                               for k in COVERAGE_TARGETS}
            out['coverage']['worst'] = cov['worst']
        if self.solution is not None:
            R, t = self.solution['R'], self.solution['t']
            q = Rot.from_matrix(R).as_quat()
            Rcr, tcr = self._compose_cam_radar(R, t)
            qc = Rot.from_matrix(Rcr).as_quat()
            out['result'] = dict(
                T_lidar_radar_translation=[float(v) for v in t],
                T_lidar_radar_quaternion_xyzw=[float(v) for v in q],
                n_inliers=int(self.solution['n_in']),
                residual_rms_sigma=float(self.solution['rms_sigma']),
                static_tf_cmd=('ros2 run tf2_ros static_transform_publisher '
                               + ' '.join(f'{v:.6f}' for v in t) + ' '
                               + ' '.join(f'{v:.6f}' for v in q) + ' '
                               + f'{self.lidar_frame or "lidar"} {self.child_frame}'),
                # composed with the GLIM lidar<->camera transform; this is what
                # radar_fusion_reflector.py consumes (r1_t_xyz / r1_quat_xyzw)
                T_cam_radar_translation=[float(v) for v in tcr],
                T_cam_radar_quaternion_xyzw=[float(v) for v in qc],
                T_cam_lidar_translation=[float(v) for v in self.t_cl],
                T_cam_lidar_quaternion_xyzw=[float(v) for v in
                                             Rot.from_matrix(self.R_cl).as_quat()],
                static_tf_cmd_cam=('ros2 run tf2_ros static_transform_publisher '
                                   + ' '.join(f'{v:.6f}' for v in tcr) + ' '
                                   + ' '.join(f'{v:.6f}' for v in qc) + ' '
                                   + f'{self.camera_frame} {self.child_frame}'))
        path = self.out_path + '_session.json'
        with open(path, 'w') as f:
            json.dump(out, f, indent=1)
        if not quiet:
            self.get_logger().info(f'saved {len(self.captures)} captures -> '
                                   f'{os.path.abspath(path)}')

    # ── camera composition + image overlay (verification / deployment only) ──
    def _compose_cam_radar(self, R_lr, t_lr):
        """T_cam_radar = T_cam_lidar · T_lidar_radar. The solve itself never
        touches the camera, so a wrong GLIM transform shows up here and in the
        overlay but leaves the radar↔lidar result intact — and a re-run of the
        lidar↔camera calibration can be recomposed without recollecting radar."""
        return self.R_cl @ R_lr, self.R_cl @ t_lr + self.t_cl

    def _image(self, m):
        img = self.bridge.imgmsg_to_cv2(m, 'bgr8')
        if self.map1 is not None:
            img = cv2.remap(img, self.map1, self.map2, cv2.INTER_LINEAR)
        self.img = img

    def _info(self, m):
        """Latch the intrinsics once, and build the undistort maps if asked.

        After rectification every later projection uses the NEW K with zero
        distortion, so the overlay and anything downstream that consumes a
        rectified image agree. None of this touches the solve — it is the
        overlay only."""
        if self.K is not None:
            return
        K = np.array(m.k).reshape(3, 3)
        D = np.array(m.d) if len(m.d) else np.zeros(5)
        w, h = int(m.width), int(m.height)
        if self.rectify and w and h and np.any(np.abs(D) > 1e-9):
            newK, _ = cv2.getOptimalNewCameraMatrix(K, D, (w, h), self.rectify_alpha, (w, h))
            self.map1, self.map2 = cv2.initUndistortRectifyMap(
                K, D, None, newK, (w, h), cv2.CV_16SC2)
            self.K, self.D = newK, np.zeros(5)
            self.get_logger().info(
                f'intrinsics locked ({w}x{h}) — rectifying in-node '
                f'(alpha={self.rectify_alpha}, |D| was {np.abs(D).max():.3f})')
        else:
            self.K, self.D = K, D
            if self.rectify:
                self.get_logger().info(
                    f'intrinsics locked ({w}x{h}) — rectify requested but D is ~0, '
                    f'the feed is already rectified')
            else:
                self.get_logger().info(f'intrinsics locked ({w}x{h})')

    def _proj(self, pts_lidar):
        """lidar-frame points → pixels, via T_cam_lidar and the ZED intrinsics."""
        P = np.atleast_2d(np.asarray(pts_lidar, float))
        Pc = (self.R_cl @ P.T).T + self.t_cl
        uv = np.full((len(Pc), 2), np.nan)
        ok = Pc[:, 2] > 0.05
        if ok.any():
            p, _ = cv2.projectPoints(Pc[ok].reshape(-1, 1, 3), np.zeros(3), np.zeros(3),
                                     self.K, self.D)
            uv[ok] = p.reshape(-1, 2)
        return uv

    def _overlay(self):
        if self.img is None or self.K is None:
            return
        im = self.img.copy()

        def txt(p, s, col, sc=.55):
            cv2.putText(im, s, p, cv2.FONT_HERSHEY_SIMPLEX, sc, (0, 0, 0), 3)
            cv2.putText(im, s, p, cv2.FONT_HERSHEY_SIMPLEX, sc, col, 1)

        # background is never drawn — only the live foreground + apex
        if self.det is not None and time.time() - self.det_t < 1.0:
            d = self.det
            for (u, v) in self._proj(d['cluster']):
                if np.isfinite(u):
                    cv2.circle(im, (int(u), int(v)), 2, (255, 220, 40), -1)
            au, av = self._proj(d['apex'])[0]
            if np.isfinite(au):
                au, av = int(au), int(av)
                cv2.drawMarker(im, (au, av), (0, 255, 0), cv2.MARKER_CROSS, 26, 2)
                txt((au + 12, av - 10), f'{np.linalg.norm(d["apex"]):.2f} m {d["method"]}',
                    (0, 255, 0))
        for c in self.captures:                      # pinned coverage map
            u, v = self._proj(np.array(c['p_lidar']))[0]
            if np.isfinite(u):
                cv2.circle(im, (int(u), int(v)), 6, (255, 200, 0), 2)
                txt((int(u) + 7, int(v) + 5), str(c['idx']), (255, 200, 0), .45)

        R, t = self._current_T()                     # radar pick through the solve
        if self.sel is not None and R is not None:
            p_l = R @ self.sel['p'] + t
            uv = self._proj(p_l)[0]
            if np.isfinite(uv[0]):
                u, v = int(uv[0]), int(uv[1])
                cv2.circle(im, (u, v), 7, (255, 0, 255), 2)
                if self.det is not None and time.time() - self.det_t < 1.0:
                    a = self._proj(self.det['apex'])[0]
                    if np.isfinite(a[0]):
                        cv2.line(im, (u, v), (int(a[0]), int(a[1])), (255, 0, 255), 1)
                        txt((u + 10, v + 16),
                            f'D {np.linalg.norm(p_l - self.det["apex"])*1000:.0f} mm', (255, 0, 255), .5)

        col = {'green': (0, 220, 0), 'orange': (0, 165, 255), 'red': (0, 0, 255)}[self.aim[1]]
        h = im.shape[0]
        txt((10, h - 56), self.lidar_stat,
            (200, 255, 200) if self.det is not None else (0, 165, 255))
        txt((10, h - 34), self.aim[0], col)
        state = ('NO BACKGROUND - ~/background first' if self.bg_lidar is None
                 else f'captures {len(self.captures)}'
                      + (f' | residual {self.solution["rms_sigma"]:.2f}s inl {self.solution["n_in"]}'
                         if self.solution else f'/{self.min_points} to first solve'))
        txt((10, h - 12), state, (240, 240, 240))
        if self.dscale != 1.0:
            im = cv2.resize(im, None, fx=self.dscale, fy=self.dscale)
        # AFTER the resize: the HUD is text, and shrinking a 1280-wide overlay to
        # fit a screen would shrink the panel into illegibility along with it.
        if self.show_cov:
            self._draw_coverage_hud(im)
        self.pub_img.publish(self.bridge.cv2_to_imgmsg(im, 'bgr8'))
        if self.show_window:
            cv2.imshow('radar_lidar_calib', im)
            cv2.waitKey(1)

    def _axis_words(self):
        """Which way the radar's two angular axes physically point, as words.

        Elevation is the radar's weak axis wherever it happens to aim: on an
        upright mount it is vertical, on a mount rolled 90 deg it is horizontal
        and the GOOD azimuth axis is the vertical one. The bars are labelled in
        radar coordinates (az/el) because that is what the solve uses, but the
        hints have to be in room coordinates or they send you the wrong way.
        Falls back to the upright assumption until a solve or prior exists."""
        R, _ = self._current_T()
        if R is None or self.lidar_frame is None:
            return dict(az='left/right', el='high/low')
        vertical = abs(float((R @ [0, 0, 1])[2]))     # radar +Z vs lidar up
        if vertical > 0.7:                            # elevation axis is vertical
            return dict(az='left/right', el='high/low')
        return dict(az='high/low', el='left/right')

    def _hint(self, key):
        return COVERAGE_HINT[key].format(**self._axis_words())

    def _coverage_text(self, pts=None, sep=' '):
        """The same six numbers as the HUD, as text — so the check is available
        in RViz and in the console when no camera is attached. `pts` defaults to
        every capture; the solve passes its INLIERS instead, since leverage from
        a rejected pose is leverage the fit never got."""
        cov = pose_coverage([c['p_radar'] for c in self.captures] if pts is None else pts)
        rows = [('range', 'range', 'm'), ('az', 'az', 'd'), ('az+-', 'az_bal', 'd'),
                ('el', 'el', 'd'), ('el+-', 'el_bal', 'd'), ('near', 'near', ''),
                ('cells', 'cells', '')]
        parts = []
        for label, key, unit in rows:
            v, tgt, ok = cov[key]
            num = f'{v:.1f}/{tgt:.1f}' if unit == 'm' else f'{v:.0f}/{tgt:.0f}'
            parts.append(f'{label} {num}{unit}[{"P" if ok else "F"}]')
        ok_all = all(cov[k][2] for _, k, _ in rows)
        tail = 'all DOF constrained' if ok_all else self._hint(cov['worst'])
        return sep.join(parts), tail, ok_all

    @staticmethod
    def _coverage_grid(cov):
        """The az x el map as three lines of text — the image HUD draws it, but a
        camera-less rig still needs to see WHICH cell is empty, not just how many.
        Rows run high elevation to low; columns left azimuth to right."""
        lab = ('el +10..+40', 'el -10..+10', 'el -40..-10')
        return '\n'.join(
            ' '.join('[X]' if (c, 2 - r) in cov['filled'] else '[ ]' for c in range(3))
            + '  ' + lab[r] for r in range(3))

    def _draw_coverage_hud(self, im):
        """Live 'will these captures actually constrain the solve?' panel.

        Six bars from pose_coverage(), each labelled with the DOF it unlocks, so
        a red bar names the rotation axis that will come out loose. This is the
        cue the ChArUco flow got from board tilt; here the only lever arm is
        where the reflector sat in the radar's own coordinates.

        The bars are about OBSERVABILITY, and the rot 1s line below them is the
        OUTCOME. Watch for green bars but a fat rot 1s (the reflector is being
        found badly, not placed badly) and for the reverse — a tight 1s on a red
        set, which is the classic over-confident fit from a degenerate geometry
        and is the reason the bars exist at all."""
        h, w = im.shape[:2]
        cov = pose_coverage([c['p_radar'] for c in self.captures])
        rows = [('RANGE', 'range', 'm', 't vs R'),
                ('AZ', 'az', 'deg', 'yaw'),
                ('AZ BAL', 'az_bal', 'deg', 'yaw'),
                ('EL', 'el', 'deg', 'pitch+roll'),
                ('EL BAL', 'el_bal', 'deg', 'pitch'),
                (f'<{NEAR_RANGE_M:.1f}m', 'near', 'n', 'all'),
                ('CELLS', 'cells', 'n', '')]
        pw, rh = 268, 21
        x0 = max(10, w - pw - 12)
        y0 = 40
        # The map is the useful half but the bars are the verdict, so on a short
        # image the map is what gets dropped rather than shrunk into illegibility.
        map_h = 190
        draw_map = (y0 + rh * (len(rows) + 2) + 22 + map_h) < h - 70
        panel_h = rh * (len(rows) + 2) + 22 + (map_h if draw_map else 0)
        rows[-1] = rows[-1][:3] + ('map below' if draw_map else 'az x el grid',)
        ov = im.copy()
        cv2.rectangle(ov, (x0 - 10, y0 - 26), (x0 + pw, y0 + panel_h), (0, 0, 0), -1)
        cv2.addWeighted(ov, 0.78, im, 0.22, 0, im)   # a lit room shows straight
                                                     # through anything lighter
        cv2.putText(im, f"COVERAGE  n={cov['n']}", (x0, y0 - 9),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y = y0 + 10
        all_ok = cov['n'] >= max(4, self.min_points)
        for label, key, unit, dof in rows:
            v, tgt, ok = cov[key]
            all_ok = all_ok and ok
            frac = 0.0 if tgt <= 0 else min(1.0, max(0.0, v / tgt))
            bx = x0 + 54
            bw = pw - 178
            col = (0, 200, 0) if ok else (0, 140, 255)
            cv2.putText(im, label, (x0, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.40,
                        (200, 200, 200), 1)
            cv2.rectangle(im, (bx, y - 5), (bx + bw, y + 5), (70, 70, 70), -1)
            cv2.rectangle(im, (bx, y - 5), (bx + int(bw * frac), y + 5), col, -1)
            num = f'{v:.1f}/{tgt:.1f}' if unit == 'm' else f'{v:.0f}/{tgt:.0f}'
            cv2.putText(im, num, (bx + bw + 5, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.40, col, 1)
            cv2.putText(im, dof, (bx + bw + 62, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.36,
                        (150, 150, 150), 1)
            y += rh
        if self._rot_sig_deg is not None:
            rs, ts = self._rot_sig_deg, self._t_sig_mm
            rmax = float(np.max(rs))
            rc = (0, 200, 0) if rmax <= 4.0 else ((0, 140, 255) if rmax <= 6.0 else (0, 0, 255))
            cv2.putText(im, f'rot 1s {rs[0]:.1f}/{rs[1]:.1f}/{rs[2]:.1f}d  '
                            f't {ts[0]:.0f}/{ts[1]:.0f}/{ts[2]:.0f}mm',
                        (x0, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.40, rc, 1)
            all_ok = all_ok and rmax <= 4.0
            y += rh
        if all_ok:
            cv2.putText(im, 'READY - all six DOF constrained', (x0, y + 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 220, 0), 1)
        else:
            cv2.putText(im, self._hint(cov['worst']), (x0, y + 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 165, 255), 1)
        if draw_map:
            self._draw_coverage_map(im, cov, x0, y + 20, pw)

    def _draw_coverage_map(self, im, cov, x0, y0, pw):
        """Where the captures actually are, so the answer to 'where do I put the
        tripod next' is a shaded box rather than a number to interpret.

        Top: the radar's field of view as a 3x3 az x el grid. A cell with no
        capture is shaded amber — walk the tripod there. Captures are dots sized
        by how close they were (near counts for more; angular error is r*sin(s)),
        and the live radar pick is a green ring, so you can see the pose land in
        an empty cell before you trigger.

        Bottom: the same idea in range — three bands, empty ones shaded, one tick
        per capture, live range as a caret."""
        mw = pw - 8
        mh = 118
        cw, ch = mw / 3.0, mh / 3.0
        cv2.putText(im, 'AZ x EL   shaded = no capture yet', (x0, y0),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (170, 170, 170), 1)
        top = y0 + 8

        def xy(az, el):
            """deg -> px. az grows to the RIGHT and el UPWARD, so the map reads
            like the scene in front of you rather than like an array index."""
            u = (az - AZ_EDGES[0]) / (AZ_EDGES[-1] - AZ_EDGES[0])
            v = (el - EL_EDGES[0]) / (EL_EDGES[-1] - EL_EDGES[0])
            return (int(x0 + min(max(u, 0.0), 1.0) * mw),
                    int(top + (1.0 - min(max(v, 0.0), 1.0)) * mh))

        for c in range(3):                                  # empty cells first
            for r in range(3):
                if (c, r) in cov['filled']:
                    continue
                a = (int(x0 + c * cw), int(top + (2 - r) * ch))
                b = (int(x0 + (c + 1) * cw), int(top + (3 - r) * ch))
                sub = im[a[1]:b[1], a[0]:b[0]]
                if sub.size:
                    sub[:] = (sub * 0.55 + np.array((0, 90, 140)) * 0.45).astype(im.dtype)
        for i in range(4):                                  # grid
            gx = int(x0 + i * cw); gy = int(top + i * ch)
            cv2.line(im, (gx, top), (gx, top + mh), (110, 110, 110), 1)
            cv2.line(im, (x0, gy), (x0 + mw, gy), (110, 110, 110), 1)
        bx, by = xy(0.0, 0.0)                               # boresight
        cv2.drawMarker(im, (bx, by), (140, 140, 140), cv2.MARKER_TILTED_CROSS, 9, 1)
        for c in self.captures:
            p = np.asarray(c['p_radar'], float)
            rr, a, e = cart_to_raz(p)
            u, v = xy(np.degrees(a), np.degrees(e))
            cv2.circle(im, (u, v), 4 if rr < NEAR_RANGE_M else 2, (255, 200, 0), -1)
        if self.sel is not None:
            rr, a, e = cart_to_raz(np.asarray(self.sel['p'], float))
            u, v = xy(np.degrees(a), np.degrees(e))
            lc = {'green': (0, 255, 0), 'orange': (0, 165, 255),
                  'red': (0, 0, 255)}[self.aim[1]]
            cv2.circle(im, (u, v), 7, lc, 2)
            cv2.drawMarker(im, (u, v), lc, cv2.MARKER_CROSS, 13, 1)
        for lbl, pos in ((f'{EL_EDGES[-1]:+.0f}', (x0 + 3, top + 10)),
                         (f'{EL_EDGES[0]:+.0f}', (x0 + 3, top + mh - 3)),
                         (f'az {AZ_EDGES[0]:+.0f}', (x0, top + mh + 12)),
                         (f'{AZ_EDGES[-1]:+.0f} deg', (x0 + mw - 40, top + mh + 12))):
            cv2.putText(im, lbl, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.32, (150, 150, 150), 1)

        sy = top + mh + 30                                  # range strip
        sh = 11
        # Span the range actually in use, not the lidar's 8 m gate — otherwise
        # every capture crams into the left fifth of the strip and it says nothing.
        seen = [float(np.linalg.norm(c['p_radar'])) for c in self.captures]
        if self.sel is not None:
            seen.append(float(self.sel['r']))
        lo = RANGE_BANDS[0][0]
        hi = min(self.lmax, max(4.5, (max(seen) + 0.5) if seen else 4.5))
        for i, (blo, bhi) in enumerate(RANGE_BANDS):
            a = int(x0 + mw * (blo - lo) / (hi - lo))
            b = int(x0 + mw * (min(bhi, hi) - lo) / (hi - lo))
            cv2.rectangle(im, (a, sy), (b, sy + sh),
                          (70, 70, 70) if cov['bands'][i] else (0, 90, 140), -1)
            cv2.rectangle(im, (a, sy), (b, sy + sh), (110, 110, 110), 1)
        for c in self.captures:
            rr = float(np.linalg.norm(c['p_radar']))
            u = int(x0 + mw * (min(rr, hi) - lo) / (hi - lo))
            cv2.line(im, (u, sy + 1), (u, sy + sh - 1), (255, 200, 0), 1)
        if self.sel is not None:
            u = int(x0 + mw * (min(float(self.sel['r']), hi) - lo) / (hi - lo))
            cv2.drawMarker(im, (u, sy + sh + 3), (0, 255, 0), cv2.MARKER_TRIANGLE_UP, 8, 1)
        cv2.putText(im, f'RANGE {lo:.1f}', (x0, sy - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (150, 150, 150), 1)
        cv2.putText(im, f'{hi:.1f} m', (x0 + mw - 34, sy - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (150, 150, 150), 1)

    # ── RViz markers (the verification layer) ──
    def _mk(self, ns, mid, typ, scale, color):
        m = Marker()
        m.header.frame_id = self.lidar_frame or 'lidar'
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns, m.id, m.type, m.action = ns, mid, typ, Marker.ADD
        m.pose.orientation.w = 1.0
        m.scale.x = m.scale.y = m.scale.z = scale
        m.color = color
        return m

    def _markers(self):
        if self.lidar_frame is None:
            return
        arr = MarkerArray()

        if self.det is not None and time.time() - self.det_t < 1.0:
            d = self.det
            pc = self._mk('cluster', 0, Marker.POINTS, 0.02, CYAN)
            pc.points = [Point(x=float(p[0]), y=float(p[1]), z=float(p[2])) for p in d['cluster']]
            arr.markers.append(pc)
            ap = self._mk('apex', 1, Marker.SPHERE, 0.07, GREEN)
            ap.pose.position.x, ap.pose.position.y, ap.pose.position.z = map(float, d['apex'])
            arr.markers.append(ap)
            lab = self._mk('apex_label', 2, Marker.TEXT_VIEW_FACING, 0.09, GREEN)
            lab.pose.position.x, lab.pose.position.y = float(d['apex'][0]), float(d['apex'][1])
            lab.pose.position.z = float(d['apex'][2]) + 0.15
            lab.text = (f'{np.linalg.norm(d["apex"]):.2f} m  {d["method"]}'
                        + (f'  +{d["n_extra"]} EXTRA CLUSTER' if d['n_extra'] else ''))
            arr.markers.append(lab)

        if self.captures:
            cap = self._mk('captures', 3, Marker.SPHERE_LIST, 0.06, AMBER)
            cap.points = [Point(x=float(c['p_lidar'][0]), y=float(c['p_lidar'][1]),
                                z=float(c['p_lidar'][2])) for c in self.captures]
            arr.markers.append(cap)
            for c in self.captures:
                tm = self._mk('capture_ids', 100 + c['idx'], Marker.TEXT_VIEW_FACING, 0.07, AMBER)
                tm.pose.position.x, tm.pose.position.y = float(c['p_lidar'][0]), float(c['p_lidar'][1])
                tm.pose.position.z = float(c['p_lidar'][2]) - 0.12
                tm.text = str(c['idx'])
                arr.markers.append(tm)

        # the radar's pick, mapped into the lidar frame by the current solve
        R, t = self._current_T()
        if self.sel is not None and R is not None:
            p = R @ self.sel['p'] + t
            rm = self._mk('radar_pick', 4, Marker.SPHERE, 0.07, MAGENTA)
            rm.pose.position.x, rm.pose.position.y, rm.pose.position.z = map(float, p)
            arr.markers.append(rm)
            if self.det is not None and time.time() - self.det_t < 1.0:
                ln = self._mk('delta', 5, Marker.LINE_LIST, 0.012, MAGENTA)
                ln.points = [Point(x=float(p[0]), y=float(p[1]), z=float(p[2])),
                             Point(x=float(self.det['apex'][0]), y=float(self.det['apex'][1]),
                                   z=float(self.det['apex'][2]))]
                arr.markers.append(ln)
                dt = self._mk('delta_label', 6, Marker.TEXT_VIEW_FACING, 0.08, MAGENTA)
                mid = (p + self.det['apex']) / 2
                dt.pose.position.x, dt.pose.position.y, dt.pose.position.z = map(float, mid)
                dt.text = (f'D {np.linalg.norm(p - self.det["apex"])*1000:.0f} mm '
                           f'({"solved" if self.solution else "prior"})')
                arr.markers.append(dt)

        st = self._mk('status', 7, Marker.TEXT_VIEW_FACING, 0.12, WHITE)
        st.pose.position.x, st.pose.position.y, st.pose.position.z = map(float, self.status_xyz)
        if self.bg_lidar is None or self.bg_radar is None:
            st.text = 'NO BACKGROUND — reflector OFF, then ~/background'
            st.color = ColorRGBA(r=1.0, g=0.3, b=0.2, a=1.0)
        else:
            head = (f'captures {len(self.captures)}'
                    + (f'  |  residual {self.solution["rms_sigma"]:.2f}s '
                       f'inl {self.solution["n_in"]}' if self.solution else
                       f'/{self.min_points} to first solve'))
            st.text = head + '\n' + self.lidar_stat + '\n' + self.aim[0]
            st.color = {'green': GREEN, 'orange': AMBER,
                        'red': ColorRGBA(r=1.0, g=0.3, b=0.2, a=1.0)}[self.aim[1]]
            if self.show_cov and self.captures:
                # the same observability check as the image HUD, for a camera-less rig
                cov_txt, cov_hint, cov_ok = self._coverage_text(sep='\n')
                st.text += ('\n' + cov_txt + '\naz -60      0     +60\n'
                            + self._coverage_grid(
                                pose_coverage([c['p_radar'] for c in self.captures]))
                            + '\n' + ('READY — all six DOF constrained'
                                      if cov_ok else 'NEXT: ' + cov_hint))
        if self.cap_deadline > time.time():
            st.text += f'\nCAPTURING {len(self.cap_radar)}/{self.cap_n}'
        arr.markers.append(st)
        self.pub_mk.publish(arr)


def main():
    rclpy.init()
    node = RadarLidarCalib()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node._save()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
