#!/usr/bin/env python3
"""Map checks, all in one module; run them through check.py:

  python3 check.py map    <map.pcd> [--traj T] [--save patches.ply] [--samples N]
  python3 check.py walls  <config> [--map M] [--traj T] [--probes N] [--stride N]
  python3 check.py maps   <A.pcd> <B.pcd> [--dist 0.10] [--traj T] [--save P]
  python3 check.py scans  <config> [--traj T] [--offsets ...]

  map    surface thickness of a map (planar 0.25 m patches, 5-95 % spread;
         walls / floors; a wall's two faces vs one surface mapped twice)
  walls  the same wall spots with the scans placed by a trajectory: one
         scan (the sensor), 1 s ... whole run, the map: where thickness comes from
  maps   completeness / accuracy of map B against map A
  scans  single scans and pairs of scans: the noise floor, deskew, a time
         offset of the trajectory, drift

Each command's own help (check.py <command> -h) is the section's text below.
"""
import argparse
import importlib.util
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
STAGES = os.path.join(ROOT, "map_stages")
sys.path.insert(0, STAGES)

from pipeline_common import (deskew_to_world, load_pipeline, load_traj,  # noqa: E402
                             pc2_xyzt, traj_quats)



# ============================================================================
# map: surface thickness (was analysis/mapping/map_quality.py)
# ============================================================================
QUALITY_DOC = r'''How good is the map? Measures local surface thickness across the cloud.

  python3 check.py map \
      <data>/processed/<date>/<pass>/mapping/denoised_<pass>_<date>.pcd \
      [--traj <data>/processed/<date>/<pass>/odometry/<machine>/traj_lidar_ba.txt] \
      [--save patches.ply] [--samples 4000]

--traj    the trajectory the map was built with (TUM, the map's frame): tells
          a wall's two faces from one surface mapped twice (below)
--save    every measured patch as a point at its centre with its values as
          scalar fields (ASCII PLY, named scalar_* so CloudCompare loads them
          by itself): surface_cm (colour by this), thickness_cm (the patch as
          one layer), separation_cm, kind, tilt_deg (0 = floor/ceiling,
          90 = wall), n_points
--samples measures more patches (denser)

TWO LAYERS. A patch is 50 cm across; a wall is 10-20 cm thick. Where the
robot drove on both sides of a wall the map holds both faces, and one patch
takes in both: as one layer it reads as the wall's thickness, not as error.
So each patch's points are split where there is an empty gap (>= 3 cm with
no points) along its normal, and each layer is measured on its own:

  kind 0  one layer                    surface = the patch's spread
  kind 2  two faces of one wall        the path passes on both sides of it
                                       (--traj); separation = the wall
  kind 3  one surface mapped twice     the path is on one side only: the same
                                       surface placed twice = misalignment there;
                                       separation = how far apart
  kind 1  two layers, side unknown     no --traj (or no pose within 15 m)

The verdict is on the surface thickness, every layer on its own.

Samples planar patches everywhere in the map, fits a plane to each, and
reports how thick the points sit around it. That thickness is the map's
error budget end to end: sensor range noise + pose error + any motion smear.

  ~1-3 cm   sensor noise floor for an Ouster; nothing left to fix in software
  ~3-6 cm   pose error is now the limit (drift between passes, loop closure)
  >8 cm     something systematic: motion smear, a bad extrinsic, divergence

Only planar patches are measured. Foliage, railings and clutter are not
surfaces and would inflate the number, so they are screened out by a
planarity test rather than averaged in.
'''


R = 0.25             # patch radius (m)
MIN_PTS = 30         # points needed for a meaningful plane fit
PLANARITY = 0.60     # patch is planar if thickness/width < this.
                     # NOTE: this CAPS the reportable thickness at
                     # ~3.29 * PLANARITY * R/2. At 0.25/0.25 m that
                     # was 10.3 cm, which silently truncated the
                     # distribution. 0.60 at R=0.25 caps at 24.7 cm.
N_SAMPLES = 4000     # patches to test
BIN = 0.01           # layer split: histogram bin along the normal (m)
GAP = 0.03           # an empty run at least this long splits a patch in two
EMPTY = 0.005        # a bin is empty below this share of the patch (or <= 1)
MIN_SPLIT = 100      # fewer points: too sparse to tell a gap from chance
MIN_LAYER = 0.15     # each layer holds at least this share of the patch
VIEW = 15.0          # poses this close to a patch say which sides it was seen from
CLEAR = 0.2          # ... and stand at least this far in front of a face
ONE, UNKNOWN, FACES, DOUBLED = 0, 1, 2, 3
KINDS = {ONE: "one layer", UNKNOWN: "two layers, side unknown",
         FACES: "two faces of a wall", DOUBLED: "one surface mapped twice"}


def load(path):
    if path.endswith(".npy"):
        return np.load(path)
    import open3d as o3d
    pcd = o3d.io.read_point_cloud(path)
    return np.asarray(pcd.points)


def load_positions(path):
    """The positions (x y z) of a TUM trajectory."""
    v = np.loadtxt(path, comments="#", ndmin=2)
    return v[:, 1:4]


def spread(d):
    return np.percentile(d, 95) - np.percentile(d, 5)


def split_layers(d):
    """Which points are in the upper layer (bool mask), split at the longest
    empty run (>= GAP) along the normal; None when they form one layer."""
    if len(d) < MIN_SPLIT:
        return None
    lo, hi = np.percentile(d, [2, 98])
    if hi - lo < 2 * GAP:
        return None
    h, e = np.histogram(d, bins=np.arange(lo, hi + BIN, BIN))
    empty = h <= max(1, EMPTY * len(d))
    best, run, start = (0, 0), 0, 0
    for i, z in enumerate(empty):
        run = run + 1 if z else 0
        if run == 1:
            start = i
        if run > best[0]:
            best = (run, start)
    if best[0] * BIN < GAP - 1e-9:
        return None
    up = d >= e[best[1]] + best[0] * BIN / 2
    if min(up.sum(), (~up).sum()) < MIN_LAYER * len(d):
        return None
    return up


def planar(Q):
    """Q (centred or not) is a planar patch: its thinnest spread small
    against its width."""
    if len(Q) < MIN_PTS:
        return False
    sv = np.linalg.svd(Q - Q.mean(0), compute_uv=False)
    return sv[1] > 1e-9 and sv[2] / sv[1] <= PLANARITY


def seen_from(c, n, a, b, pos, pos_tree):
    """FACES when poses near ``c`` stand in front of both layers (beyond the
    upper along +n, below the lower along -n), DOUBLED when in front of one
    only, UNKNOWN without poses."""
    if pos_tree is None:
        return UNKNOWN
    near = pos_tree.query_ball_point(c, VIEW)
    if not near:
        return UNKNOWN
    s = (pos[near] - c) @ n
    front = bool(np.any(s > np.median(b) + CLEAR))
    back = bool(np.any(s < np.median(a) - CLEAR))
    return FACES if front and back else DOUBLED


def patch_stats(P, n_samples=N_SAMPLES, r=R, seed=0, centres=None, positions=None):
    """Per patch: (thickness_m as one layer, tilt_deg of the normal from
    vertical, surface_m: the thicker layer's own spread, separation_m between
    the layers (0 for one), kind); ``positions`` (the trajectory's, N x 3)
    tell two faces from one surface mapped twice. With a list as
    ``centres``, each patch's (x, y, z, n_points) is appended to it."""
    from scipy.spatial import cKDTree
    rng = np.random.default_rng(seed)
    tree = cKDTree(P)
    pos_tree = cKDTree(positions) if positions is not None and len(positions) else None
    idx = rng.choice(len(P), size=min(n_samples, len(P)), replace=False)
    out = []
    n_sparse = n_rough = 0
    for i in idx:
        nb = tree.query_ball_point(P[i], r)
        if len(nb) < MIN_PTS:
            n_sparse += 1
            continue
        c = P[nb].mean(0)
        Q = P[nb] - c
        _, sv, vt = np.linalg.svd(Q, full_matrices=False)
        n = vt[2]
        d = Q @ n
        up = split_layers(d)
        if up is None:
            if sv[1] < 1e-9 or sv[2] / sv[1] > PLANARITY:
                n_rough += 1
                continue                  # edge, corner, clutter -- or a
                # surface so smeared it no longer looks planar. Both land here,
                # which is why the rejection count has to be reported: a high
                # n_rough means the median below is biased LOW.
        elif not (planar(Q[up]) and planar(Q[~up])):
            n_rough += 1                  # two layers, but not two planes
            continue
        # 5-95 percentile span: robust to the few stragglers a radius ball
        # always catches from an adjoining surface. Two layers (a thick
        # wall's two faces) may be further apart than one layer could be
        # and still be planar: each layer is tested on its own.
        thick = spread(d)
        if up is None:
            surface, sep, kind = thick, 0.0, ONE
        else:
            a, b = d[~up], d[up]
            surface = max(spread(a), spread(b))
            sep = float(np.median(b) - np.median(a))
            kind = seen_from(c, n, a, b, positions, pos_tree)
        out.append((thick, np.degrees(np.arccos(abs(n[2]))), surface, sep, kind))
        if centres is not None:
            centres.append((c[0], c[1], c[2], len(nb)))
    cap = 3.29 * PLANARITY * r / 2 * 100
    return np.array(out).reshape(-1, 5), dict(sparse=n_sparse, rough=n_rough, cap_cm=cap)


def summary(st):
    """patch_stats' patches -> the numbers a report needs: surface thickness
    (cm, every layer on its own) median of all, walls (tilt > 60 deg) and
    floors / ceilings (< 30 deg), and the share of patches holding one
    surface twice (kind 3, %)."""
    t, tilt, kind = st[:, 2] * 100, st[:, 1], st[:, 4].astype(int)
    med = lambda m: float(np.median(t[m])) if m.sum() > 20 else None      # noqa: E731
    return dict(patches=int(len(st)), median_cm=float(np.median(t)),
                walls_cm=med(tilt > 60), floors_cm=med(tilt < 30),
                doubled_pct=float(100 * (kind == DOUBLED).mean()))


def save_patches(path, stats, centres):
    """ASCII PLY, one vertex per measured patch with its values.
    CloudCompare takes only properties named scalar_<name> as scalar fields
    without asking (shown as <name>); others are dropped on open."""
    with open(path, "w") as fh:
        fh.write("ply\nformat ascii 1.0\nelement vertex %d\n" % len(stats))
        for f in ("x", "y", "z", "surface_cm", "thickness_cm", "separation_cm", "kind",
                  "tilt_deg", "n_points"):
            fh.write("property float %s\n" % (f if f in "xyz" else "scalar_" + f))
        fh.write("end_header\n")
        for (t, tilt, surf, sep, kind), (x, y, z, n) in zip(stats, centres):
            fh.write("%.3f %.3f %.3f %.2f %.2f %.2f %d %.1f %d\n"
                     % (x, y, z, surf * 100, t * 100, sep * 100, kind, tilt, n))


def main_quality(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] in ("-h", "--help"):
        print(QUALITY_DOC)
        return
    opt = {}
    for flag in ("--save", "--samples", "--traj"):
        if flag in args:
            i = args.index(flag)
            opt[flag] = args[i + 1]
            del args[i:i + 2]
    path = args[0] if args else "denoised.pcd"
    P = load(path)
    print(f"{path}: {len(P)} points, extent "
          f"{np.round(P.max(0) - P.min(0), 1)} m")

    n_samples = int(opt.get("--samples", N_SAMPLES))
    centres = [] if "--save" in opt else None
    positions = load_positions(opt["--traj"]) if "--traj" in opt else None
    if positions is not None:
        print(f"{opt['--traj']}: {len(positions)} poses")
    st, rej = patch_stats(P, n_samples=n_samples, centres=centres, positions=positions)
    if len(st) == 0:
        raise SystemExit("no planar patches found -- is this a surface map?")
    if centres is not None:
        save_patches(opt["--save"], st, centres)
        print(f"patches -> {opt['--save']} (colour by surface_cm)")
    raw = st[:, 0] * 100                   # cm, the patch as one layer
    tilt = st[:, 1]
    t = st[:, 2] * 100                     # cm, every layer on its own
    sep = st[:, 3] * 100
    kind = st[:, 4].astype(int)
    q = np.percentile(t, [10, 25, 50, 75, 90])

    n_try = min(n_samples, len(P))
    print(f"\n{len(st)} planar patches measured of {n_try} sampled "
          f"(radius {R} m)")
    print(f"  rejected: {rej['sparse']} too sparse, {rej['rough']} non-planar")
    print(f"  reportable ceiling {rej['cap_cm']:.1f} cm -- one layer thicker "
          f"than this lands in 'non-planar' and is NOT counted, so a large\n"
          f"  rejection count means the median below is biased low")
    print(f"  surface thickness   median {q[2]:.1f} cm")
    print(f"                      p10-p90 {q[0]:.1f} - {q[4]:.1f} cm")
    print(f"                      p25-p75 {q[1]:.1f} - {q[3]:.1f} cm")

    # walls vs floors separately: they fail in different ways. Floors are
    # measured at grazing incidence and by fewer passes; walls carry most of
    # the yaw-drift error. A big gap between the two is itself a diagnosis.
    wall = tilt > 60
    flat = tilt < 30
    for name, m in (("vertical (walls)", wall), ("horizontal (floor/ceiling)", flat)):
        if m.sum() > 20:
            print(f"  {name:28s} median {np.median(t[m]):.1f} cm "
                  f"({m.sum()} patches; {np.median(raw[m]):.1f} cm with both layers "
                  f"of a patch as one)")

    print("\nlayers (an empty gap >= %g cm along the normal splits a patch):" % (GAP * 100))
    for k in (ONE, FACES, DOUBLED, UNKNOWN):
        m = kind == k
        if not m.any():
            continue
        line = f"  {KINDS[k]:26s} {m.sum():6d} patches ({(m & wall).sum()} walls)"
        if k != ONE:
            line += (f", apart median {np.median(sep[m]):.1f} cm, each layer "
                     f"{np.median(t[m]):.1f} cm")
        print(line)
    if positions is None and (kind == UNKNOWN).any():
        print("  (--traj tells the two faces of a wall from one surface mapped twice)")

    print("\nhistogram (cm, every layer on its own):")
    h, e = np.histogram(t, bins=np.arange(0, min(30, t.max()) + 1, 1))
    peak = max(h.max(), 1)
    for a, v in zip(e[:-1], h):
        if v:
            print(f"  {a:4.0f}-{a + 1:<4.0f} {v:5d} {'#' * int(50 * v / peak)}")

    m = q[2]
    print()
    doubled = (kind == DOUBLED).sum()
    if doubled:
        print(f"DOUBLED: {doubled} patches ({100 * doubled / len(kind):.1f} %) hold one "
              f"surface twice, {np.median(sep[kind == DOUBLED]):.1f} cm apart: the map is "
              f"misaligned there (--save: kind 3)")
    if m < 3:
        print(f"VERDICT: {m:.1f} cm -- at the sensor noise floor. The map is "
              f"as good as this hardware gets.")
    elif m < 6:
        print(f"VERDICT: {m:.1f} cm -- pose error dominates. Sensor noise is "
              f"1-3 cm, so roughly {np.sqrt(max(m**2 - 2**2, 0)):.1f} cm is "
              f"coming from the trajectory.")
    else:
        print(f"VERDICT: {m:.1f} cm -- too thick to be noise or ordinary "
              f"drift. Something systematic is displacing points.")


# ============================================================================
# maps: completeness and accuracy (was analysis/mapping/compare_maps.py)
# ============================================================================
COMPARE_DOC = r'''Does one map cover what another covers? Completeness and accuracy of a map
B against a reference A, the usual reconstruction metrics:

  completeness   share of A's points with a point of B within --dist
                 (what B is missing)
  accuracy       share of B's points with a point of A within --dist
                 (what B has that A has not: new surfaces, or junk)

  python3 check.py maps A.pcd B.pcd [--dist 0.10]
      [--traj traj.txt] [--save missing.ply] [--sample 5000000]

    e.g. A = the union map (mapping/union_ba/denoised_...pcd),
         B = the TSDF map  (mapping/denoised_...pcd)

--dist    within this of each other counts as covered (m). A union map's
          surfaces are a band several cm wide; 0.10 holds the band.
--traj    also split completeness by distance from the robot's path and by
          height, to see WHERE B is missing what A has
--save    A's points B is missing, as a PLY (scalar_dist_cm: to the nearest
          point of B; scalar_range_m: to the path, with --traj)
--sample  measure on at most this many points of each map (random)
Writes nothing else.
'''


def sample(P, n, rng):
    return P if len(P) <= n else P[rng.choice(len(P), n, replace=False)]


def coverage(A, B, dist=0.10, n=5_000_000, seed=0):
    """(completeness, accuracy, As, dA, comp): the share of A's points with B
    within dist and of B's with A within dist, on at most n points of each;
    As the points of A measured, dA their distance to B, comp the mask."""
    rng = np.random.default_rng(seed)
    As, Bs = sample(A, n, rng), sample(B, n, rng)
    dA, _ = cKDTree(B).query(As, distance_upper_bound=1.0)
    dB, _ = cKDTree(A).query(Bs, distance_upper_bound=1.0)
    comp = np.isfinite(dA) & (dA <= dist)
    acc = np.isfinite(dB) & (dB <= dist)
    return float(comp.mean()), float(acc.mean()), As, dA, comp


def main_compare(argv=None):
    ap = argparse.ArgumentParser(description=COMPARE_DOC,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--dist", type=float, default=0.10)
    ap.add_argument("--traj")
    ap.add_argument("--save")
    ap.add_argument("--sample", type=int, default=5_000_000)
    a = ap.parse_args(argv)

    A, B = load(a.a), load(a.b)
    print(f"A {a.a}: {len(A)} points, extent {np.round(A.max(0) - A.min(0), 1)} m")
    print(f"B {a.b}: {len(B)} points, extent {np.round(B.max(0) - B.min(0), 1)} m")
    c, acc, As, dA, comp = coverage(A, B, a.dist, a.sample)
    print(f"\nwithin {a.dist * 100:g} cm:")
    print(f"  completeness  {c * 100:5.1f} %  of A's points have B nearby "
          f"(B misses {100 - c * 100:.1f} %)")
    print(f"  accuracy      {acc * 100:5.1f} %  of B's points have A nearby")

    rng_m = None
    if a.traj:
        pos = np.loadtxt(a.traj, comments="#", ndmin=2)[:, 1:4]
        rng_m, _ = cKDTree(pos).query(As)
        print("\ncompleteness by distance from the path (m):")
        for lo, hi in ((0, 2), (2, 4), (4, 7), (7, 12), (12, 1e9)):
            m = (rng_m >= lo) & (rng_m < hi)
            if m.any():
                print(f"  {lo:4g}-{'' if hi > 1e8 else f'{hi:g}':4s} {comp[m].mean() * 100:5.1f} %  "
                      f"({m.mean() * 100:4.1f} % of A)")
        z = As[:, 2] - pos[:, 2].min()
        print("completeness by height above the path's lowest point (m):")
        for lo, hi in ((-1e9, -1), (-1, 0), (0, 1), (1, 2), (2, 3), (3, 5), (5, 1e9)):
            m = (z >= lo) & (z < hi)
            if m.any():
                lab = f"{'' if lo < -1e8 else f'{lo:g}'} .. {'' if hi > 1e8 else f'{hi:g}'}"
                print(f"  {lab:10s} {comp[m].mean() * 100:5.1f} %  ({m.mean() * 100:4.1f} % of A)")

    if a.save:
        miss = ~comp
        cols = [As[miss], np.where(np.isfinite(dA[miss]), dA[miss], 1.0)[:, None] * 100]
        names = ["scalar_dist_cm"]
        if rng_m is not None:
            cols.append(rng_m[miss][:, None])
            names.append("scalar_range_m")
        M = np.hstack(cols)
        with open(a.save, "w") as fh:
            fh.write("ply\nformat ascii 1.0\nelement vertex %d\n" % len(M))
            for f in ["x", "y", "z"] + names:
                fh.write("property float %s\n" % f)
            fh.write("end_header\n")
            np.savetxt(fh, M, fmt="%.3f")
        print(f"\nmissing points -> {a.save} ({len(M)})")


# ============================================================================
# walls: the scans as placed at the map's wall spots (was analysis/mapping/wall_build.py)
# ============================================================================
WALLS_DOC = r'''Is the map's wall thickness made by stage 01, or already in the scans as
placed? Places the scans with 01's own code (iter_world_scans: the same
deskew, poses, range limits) and measures the same wall spots of the map
before and after 01's merge / dynamic removal / denoise.

  python3 check.py walls <pipeline config> \
      [--map <data>/processed/<date>/<pass>/mapping/denoised_<pass>_<date>.pcd] \
      [--probes 300] [--stride 1] [--traj FILE]

--map     the map to compare with (default: 01's denoised cloud of the config)
--probes  wall spots measured (planar wall patches of the map, 0.25 m radius)
--stride  every Nth scan (faster, fewer views)
--traj    place the scans with this trajectory instead of dataset.traj
--json    also write the medians (one scan, the spans, whole run, map) to a file

For each spot, every scan that hit it: where that scan puts the wall along
the wall's normal. Thickness is the 5-95 % spread, as `map`:

  one scan            the sensor's noise floor at that spot
  1 s ... whole run   the scans of that span around the spot's best view,
                      together: how the thickness grows with time and with
                      the places it was seen from
  the map             the same spot in --map

  whole run ~ map    01's merge / dynamic removal / denoise add nothing:
                     the thickness is in the scans as placed (poses, deskew,
                     the points themselves), not in how 01 builds the map
  map >> whole run   01 adds it

and where each scan puts the wall by its distance to the wall, signed away
from the sensor: a trend with distance is a range error in the points.
Reads the bag once, writes nothing.
'''


KEEP = 20                         # points kept per scan and spot for the spans
SPANS = (1.0, 10.0, 60.0, 300.0)
DIST = (0.0, 2.0, 4.0, 7.0, 12.0, 1e9)


def stage01():
    spec = importlib.util.spec_from_file_location("build_map",
                                                  os.path.join(STAGES, "01_build_map.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def wall_probes(M, n, seed=0):
    """Up to n planar one-layer wall patches of the map: centre, normal,
    thickness."""
    rng = np.random.default_rng(seed)
    tree = cKDTree(M)
    out = []
    for i in rng.choice(len(M), size=min(len(M), 20 * n), replace=False):
        nb = tree.query_ball_point(M[i], R)
        if len(nb) < MIN_PTS:
            continue
        c = M[nb].mean(0)
        Q = M[nb] - c
        _, sv, vt = np.linalg.svd(Q, full_matrices=False)
        nrm = vt[2]
        if abs(nrm[2]) > np.cos(np.radians(60)) or sv[2] / sv[1] > PLANARITY:
            continue                                  # not a wall, or not planar
        d = Q @ nrm
        if split_layers(d) is not None:
            continue                                  # two layers: not one surface
        out.append((c, nrm, spread(d)))
        if len(out) >= n:
            break
    return out


def measure(config, map_path=None, n_probes=300, stride=1, traj=None):
    """Prints the tables; returns their medians (m): single, spans {s: m},
    whole, map, by_dist [(lo, hi, median offset)]."""
    b = stage01()
    P = b.load_pipeline(config)
    s = P.stage("01_build_map")
    if traj:
        from pipeline_common import load_traj
        P.traj = load_traj(P.outp(traj))
        print(f"trajectory {P.outp(traj)}")
    else:
        P.traj = b.load_traj_cached(P)
    # 01 takes the points topic from P.sensor, which is dataset.points_topic
    # when set; this one does not need the camera P.sensor also reads
    S = SimpleNamespace(points_topic=P.points_topic)
    mpath = map_path or P.outp(P.pcd("denoised"))
    import open3d as o3d
    M = np.asarray(o3d.io.read_point_cloud(mpath).points)
    print(f"map {mpath}: {len(M)} points")
    probes = wall_probes(M, n_probes)
    del M
    if not probes:
        raise SystemExit("no planar wall patches in the map")
    C = np.array([p[0] for p in probes])
    N = np.array([p[1] for p in probes])
    map_thick = np.array([p[2] for p in probes])
    print(f"{len(probes)} wall spots, map thickness there median {np.median(map_thick) * 100:.1f} cm")
    print(f"placing the scans with 01's iter_world_scans ({len(P.traj[0])} poses, "
          f"deskew {'on' if s.get('deskew', True) else 'off'}, "
          f"range {s['lidar_min']}-{s['lidar_max']} m), every {stride} scan(s)")

    ptree = cKDTree(C)
    rng = np.random.default_rng(1)
    hits = [[] for _ in probes]      # per spot: (t, n, spread, median d, dist, sign, kept d)
    n_scan = 0
    for k, (t, wp, origin) in enumerate(b.iter_world_scans(P, S, s)):
        if k % stride:
            continue
        n_scan += 1
        wp = np.asarray(b.as_cpu(wp), dtype=np.float64)
        o = np.asarray(b.as_cpu(origin), dtype=np.float64)
        dd, ii = ptree.query(wp, distance_upper_bound=R)
        ok = np.isfinite(dd)
        if not ok.any():
            continue
        ii, pts = ii[ok], wp[ok]
        order = np.argsort(ii, kind="stable")
        ii, pts = ii[order], pts[order]
        cuts = np.flatnonzero(np.diff(ii)) + 1
        for grp, q in zip(np.split(ii, cuts), np.split(pts, cuts)):
            j = int(grp[0])
            if len(q) < 10:
                continue
            d = (q - C[j]) @ N[j]
            away = 1.0 if (C[j] - o) @ N[j] > 0 else -1.0     # +d = further from the sensor
            kept = d if len(d) <= KEEP else rng.choice(d, KEEP, replace=False)
            hits[j].append((t, len(d), spread(d) if len(d) >= MIN_PTS else np.nan,
                            float(np.median(d)), float(np.linalg.norm(C[j] - o)), away,
                            kept.astype(np.float32)))
        if n_scan % 1000 == 0:
            print(f"    {n_scan} scans", flush=True)
    print(f"    {n_scan} scans read")

    single, span_th = [], {w: [] for w in SPANS + (None,)}
    mapt, by_dist = [], [[] for _ in DIST[:-1]]
    for j, h in enumerate(hits):
        if len(h) < 3:
            continue
        ts = np.array([x[0] for x in h])
        cnt = np.array([x[1] for x in h])
        sp = np.array([x[2] for x in h])
        single += list(sp[np.isfinite(sp)])
        t0 = ts[np.argmax(cnt)]
        for w in span_th:
            sel = np.ones(len(h), bool) if w is None else np.abs(ts - t0) <= w / 2
            d = np.concatenate([h[i][6] for i in np.flatnonzero(sel)])
            if len(d) >= MIN_PTS:
                span_th[w].append(spread(d))
        mapt.append(map_thick[j])
        med = np.array([x[3] for x in h])
        away = np.array([x[5] for x in h])
        dist = np.array([x[4] for x in h])
        off = (med - np.median(med)) * away            # + = this scan puts the wall further away
        for bi in range(len(DIST) - 1):
            m = (dist >= DIST[bi]) & (dist < DIST[bi + 1])
            by_dist[bi] += list(off[m])

    def cm(v):
        return f"{np.median(v) * 100:5.1f}" if len(v) else "    -"
    print(f"\nwall thickness at the same {len(mapt)} spots, median cm (spots)")
    print(f"  one scan          {cm(single)}   (noise floor)")
    for w in SPANS:
        print(f"  {w:5g} s together  {cm(span_th[w])}   ({len(span_th[w])})")
    print(f"  whole run         {cm(span_th[None])}   ({len(span_th[None])})")
    print(f"  the map (01)      {cm(mapt)}")

    print("\nwhere a scan puts the wall by its distance to it, cm, + = further from the "
          "sensor than the spot's median (scans)")
    dist_med = []
    for bi in range(len(DIST) - 1):
        v = np.array(by_dist[bi])
        hi = "+" if DIST[bi + 1] > 1e8 else f"-{DIST[bi + 1]:g}"
        if len(v):
            dist_med.append((DIST[bi], DIST[bi + 1], float(np.median(v))))
            print(f"  {DIST[bi]:4g}{hi:5s} m   median {np.median(v) * 100:+5.1f}   "
                  f"p10-p90 {np.percentile(v, 10) * 100:+5.1f} .. {np.percentile(v, 90) * 100:+5.1f}"
                  f"   ({len(v)})")

    whole, mp = np.median(span_th[None]) if span_th[None] else np.nan, np.median(mapt)
    print()
    if np.isfinite(whole) and mp > whole + 0.02:
        print(f"READING: the map is {100 * (mp - whole):.1f} cm thicker than the scans "
              f"as placed: stage 01 adds it (merge / dynamic removal / denoise)")
    elif np.isfinite(whole):
        print(f"READING: the scans as placed are already {whole * 100:.1f} cm thick here "
              f"(map {mp * 100:.1f}): stage 01's merge adds nothing; the spans above "
              f"show when it appears, the distance table whether it is a range error")
    med = lambda v: float(np.median(v)) if len(v) else float("nan")       # noqa: E731
    return dict(single=med(single), spans={w: med(span_th[w]) for w in SPANS},
                whole=float(whole), map=float(mp), by_dist=dist_med)


def main_walls(argv=None):
    ap = argparse.ArgumentParser(description=WALLS_DOC,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--map")
    ap.add_argument("--probes", type=int, default=300)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--traj")
    ap.add_argument("--json", help="also write the medians here (check.py figures reads them)")
    a = ap.parse_args(argv)
    res = measure(a.config, a.map, a.probes, a.stride, a.traj)
    if a.json:
        import json
        with open(a.json, "w") as fh:
            json.dump(dict(res, traj=a.traj, map=a.map, spans={str(k): v for k, v in res["spans"].items()}),
                      fh, indent=1)
        print(f"medians -> {a.json}")


# ============================================================================
# scans: single scans and pairs (was analysis/mapping/scan_consistency.py)
# ============================================================================
SCANS_DOC = r'''Where does a map's wall thickness come from? Tells apart wrong point
timing (deskew), a timing offset of the trajectory, and drift, from a few
hundred scans.

  python3 check.py scans map_stages/pipeline_config.json [--traj FILE]

Uses the bag, trajectory (dataset.traj, or --traj), topic and 01_build_map
settings (time_tol, lidar_min/max, deskew_bins) of the pipeline config, and
measures wall thickness the way `map` does (5-95 % spread of the
points of 0.25 m planar patches):

  1. SINGLE scans: the sensor's own noise floor. Nothing below it is
     reachable. (Timing errors bend a whole scan smoothly, which keeps it
     locally flat, so they do not show here -- they show in 2.)
  2. PAIRS of scans 0.1 s, 1 s, 10 s apart and revisits (same place, minutes
     apart), measured only where both hit the same wall; split by whether
     the robot was turning, and the turning pairs once more without deskew:
       turning pairs thick, straight thin   ->  timing (see 3) or deskew
       turning thinner WITHOUT deskew        ->  the point times are wrong
       thickens with the separation          ->  drift
       thin until the revisits               ->  loop closure / pass to pass
  3. the 1 s / 10 s / revisit pairs again with the trajectory's stamps
     shifted by each --offsets value:
       thinnest at an offset other than 0  ->  the pose given for t is the
       pose at t + that offset

Reads only the scans it needs (--anchors x 5), writes nothing.
'''


MIN_EACH = 10       # pair patches: points needed from each scan
WALL, FLOOR = 60.0, 30.0      # normal tilt from vertical (deg): wall > 60, floor < 30


def turn_rate_and_speed(tr_t, tr_T):
    """Per pose: rotation rate (deg/s, about any axis) and speed (m/s)."""
    dt = np.maximum(np.diff(tr_t), 1e-6)
    dR = np.einsum("nji,njk->nik", tr_T[:-1, :3, :3], tr_T[1:, :3, :3])
    ang = np.degrees(np.arccos(np.clip((np.trace(dR, axis1=1, axis2=2) - 1) / 2, -1, 1)))
    v = np.linalg.norm(np.diff(tr_T[:, :3, 3], axis=0), axis=1)
    rate, speed = ang / dt, v / dt
    pad = lambda a: np.concatenate([[a[0]], 0.5 * (a[1:] + a[:-1]), [a[-1]]])
    return pad(rate), pad(speed)


def place(scan, tr_t, tr_T, tr_q, deskew, nb, shift=0.0):
    """World points of one scan; the trajectory's stamps moved by `shift` s.
    deskew: every point through the pose at its own time, as the map stages
    do (pipeline_common.deskew_to_world); else the whole scan through the
    pose at the cloud stamp."""
    t, p, dt = scan
    return deskew_to_world(p, dt if deskew else None, t, tr_t, tr_T, tr_q, nb, shift)


def patches(P, centers, owner=None):
    """(thickness m, normal tilt deg) of every planar patch around `centers`;
    with `owner` (0/1 per point) only patches holding MIN_EACH of both."""
    tree = cKDTree(P)
    out, rough = [], 0
    for nb in tree.query_ball_point(centers, R):
        if len(nb) < MIN_PTS:
            continue
        if owner is not None:
            k = int(owner[nb].sum())
            if min(k, len(nb) - k) < MIN_EACH:
                continue
        Q = P[nb] - P[nb].mean(0)
        _, sv, vt = np.linalg.svd(Q, full_matrices=False)
        if sv[1] < 1e-9 or sv[2] / sv[1] > PLANARITY:
            rough += 1
            continue
        d = Q @ vt[2]
        out.append((np.percentile(d, 95) - np.percentile(d, 5),
                    np.degrees(np.arccos(min(1.0, abs(vt[2][2]))))))
    return np.array(out).reshape(-1, 2), rough


def read_scans(bag, topic, tr_t, need, tol, lo, hi, window):
    """{pose index: (stamp s, xyz float32, point times or None)} for the
    poses in `need`. Messages whose log time is more than `window` s from
    every needed stamp are not decoded (0 = decode all)."""
    from rosbags.highlevel import AnyReader
    from rosbags.typesys import Stores, get_typestore
    want = np.array(sorted(tr_t[list(need)]))
    out = {}
    with AnyReader([Path(bag)], default_typestore=get_typestore(Stores.ROS2_HUMBLE)) as r:
        conns = [c for c in r.connections if c.topic == topic]
        if not conns:
            raise SystemExit(f"{topic} is not in {bag}")
        for conn, log_ns, raw in r.messages(connections=conns):
            if window > 0:
                i = np.searchsorted(want, log_ns * 1e-9)
                near = min(abs(want[min(i, len(want) - 1)] - log_ns * 1e-9),
                           abs(want[max(i - 1, 0)] - log_ns * 1e-9))
                if near > window:
                    continue
            msg = r.deserialize(raw, conn.msgtype)
            t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            j = int(np.clip(np.searchsorted(tr_t, t), 1, len(tr_t) - 1))
            j = j if abs(tr_t[j] - t) < abs(tr_t[j - 1] - t) else j - 1
            if j not in need or abs(tr_t[j] - t) > tol or j in out:
                continue
            p, dt = pc2_xyzt(msg)
            d = np.linalg.norm(p, axis=1)
            keep = np.isfinite(p).all(1) & (d > lo) & (d < hi)
            out[j] = (t, p[keep].astype(np.float64),
                      None if dt is None else dt[keep])
            if len(out) == len(need):
                break
    return out


def cm(a):
    return f"{np.median(a) * 100:6.1f}" if len(a) else "     -"


def main_scans(argv=None):
    ap = argparse.ArgumentParser(description=SCANS_DOC,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", nargs="?", default="pipeline_config.json")
    ap.add_argument("--traj", help="trajectory (TUM); default dataset.traj")
    ap.add_argument("--anchors", type=int, default=40, help="scans sampled along the run")
    ap.add_argument("--patches", type=int, default=200, help="patches per scan / pair")
    ap.add_argument("--turn-dps", type=float, default=15.0, help="turning above this (deg/s)")
    ap.add_argument("--straight-dps", type=float, default=3.0,
                    help="straight below this (deg/s), while moving > 0.2 m/s")
    ap.add_argument("--revisit-s", type=float, default=60.0, help="revisit: at least this apart")
    ap.add_argument("--revisit-m", type=float, default=1.0, help="revisit: within this distance")
    ap.add_argument("--offsets", default="-0.1,-0.05,-0.02,0,0.02,0.05,0.1",
                    help="trajectory stamp shifts to try (s); write --offsets=-0.1,0,0.1")
    ap.add_argument("--log-window", type=float, default=2.0,
                    help="decode only clouds logged within this of a needed stamp (0 = all)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    P = load_pipeline(a.config)
    s = P.cfg["01_build_map"]
    nb = max(2, int(s.get("deskew_bins", 100)))
    traj = a.traj or P.outp(P.dataset["traj"])
    tr_t, tr_T = load_traj(traj)
    tr_q = traj_quats(tr_T)
    topic = P.points_topic
    rate, speed = turn_rate_and_speed(tr_t, tr_T)
    step = float(np.median(np.diff(tr_t)))
    N = len(tr_t)
    print(f"trajectory {traj}: {N} poses, {tr_t[-1] - tr_t[0]:.0f} s, "
          f"one per {step * 1000:.0f} ms")

    seps = {f"{sep:g}s": int(round(sep / step)) for sep in (0.1, 1.0, 10.0)}
    far = max(seps.values())
    anchors = np.unique(np.linspace(5, N - far - 6, a.anchors).astype(int))
    pairs = {k: [(j, j + d) for j in anchors] for k, d in seps.items()}
    pos = tr_T[:, :3, 3]
    rv = []
    for j in anchors:
        dist = np.linalg.norm(pos - pos[j], axis=1)
        ok = (np.abs(tr_t - tr_t[j]) > a.revisit_s) & (dist < a.revisit_m)
        if ok.any():
            rv.append((j, int(np.flatnonzero(ok)[np.argmin(dist[ok])])))
    pairs["revisit"] = rv
    need = set(int(j) for j in anchors) | {int(k) for v in pairs.values() for _, k in v}
    print(f"reading {len(need)} scans of {topic} ({len(rv)} revisits found within "
          f"{a.revisit_m} m and > {a.revisit_s:g} s apart)")
    scans = read_scans(P.dataset["bag"], topic, tr_t, need, float(s["time_tol"]),
                       float(s["lidar_min"]), float(s["lidar_max"]), a.log_window)
    if len(scans) < len(need) // 2:
        raise SystemExit(f"only {len(scans)} of {len(need)} scans found; if the "
                         "bag's log times are not near the stamps, pass --log-window 0")
    rng = np.random.default_rng(a.seed)

    def centers(Pw, n):
        return Pw[rng.choice(len(Pw), size=min(n, len(Pw)), replace=False)]

    def state(j):
        if rate[j] > a.turn_dps:
            return "turning"
        return "straight" if rate[j] < a.straight_dps and speed[j] > 0.2 else "other"

    # 1. single scans: the noise floor ----------------------------------------
    sw, sf = [], []
    for j in anchors:
        if j in scans:
            Pw = place(scans[j], tr_t, tr_T, tr_q, True, nb)
            st, _ = patches(Pw, centers(Pw, a.patches))
            sw += list(st[st[:, 1] > WALL, 0])
            sf += list(st[st[:, 1] < FLOOR, 0])
    print("\n1. SINGLE scans (the sensor's noise floor), median cm")
    print(f"   walls {cm(sw)}   floors {cm(sf)}   ({len(sw)} + {len(sf)} patches)")

    # 2. + 3. pairs ------------------------------------------------------------
    def pair_walls(j, k, deskew, off):
        A = place(scans[j], tr_t, tr_T, tr_q, deskew, nb, off)
        B = place(scans[k], tr_t, tr_T, tr_q, deskew, nb, off)
        dB, _ = cKDTree(B).query(A, distance_upper_bound=R / 2)
        both = A[np.isfinite(dB)]             # where the two see the same surface
        if len(both) < MIN_PTS:
            return None
        owner = np.r_[np.zeros(len(A), bool), np.ones(len(B), bool)]
        st, _ = patches(np.vstack([A, B]), centers(both, a.patches), owner)
        return list(st[st[:, 1] > WALL, 0])

    offsets = sorted(set(float(x) for x in a.offsets.split(",")) | {0.0})
    cols = ("all", "straight", "turning", "turn/rigid")
    table = {n: {c: [] for c in cols} for n in pairs}
    npair = {n: {c: 0 for c in cols} for n in pairs}
    timing = {(n, off): [] for n in pairs for off in offsets}
    for name, plist in pairs.items():
        for j, k in plist:
            if j not in scans or k not in scans:
                continue
            st = {state(j), state(k)}
            kind = "turning" if "turning" in st else ("straight" if st == {"straight"} else None)
            for off in (offsets if name != "0.1s" else [0.0]):
                w = pair_walls(j, k, True, off)
                if w is None:
                    continue
                timing[(name, off)] += w
                if off == 0.0:
                    table[name]["all"] += w
                    npair[name]["all"] += 1
                    if kind:
                        table[name][kind] += w
                        npair[name][kind] += 1
            if kind == "turning":
                w = pair_walls(j, k, False, 0.0)
                if w is not None:
                    table[name]["turn/rigid"] += w
                    npair[name]["turn/rigid"] += 1
    print("\n2. PAIRS: wall thickness where both scans see the same wall, median cm "
          "(pairs)")
    print("   apart    " + "".join(f"{c:>15s}" for c in cols))
    for n in pairs:
        print(f"   {n:8s} " + "".join(f"{cm(table[n][c]):>9s} ({npair[n][c]:3d})" for c in cols))
    print(f"   turning: above {a.turn_dps:g} deg/s at either scan; straight: both below "
          f"{a.straight_dps:g} deg/s and moving; turn/rigid: the turning pairs without deskew")

    names = [n for n in pairs if n != "0.1s"]
    print("\n3. TIMING: pairs with the trajectory's stamps shifted, median cm")
    print("   shift_ms " + "".join(f"{n:>10s}" for n in names))
    for off in offsets:
        print(f"   {off * 1000:+8.0f} " + "".join(f"{cm(timing[(n, off)]):>10s}" for n in names))

    # what it points at ----------------------------------------------------------
    med = lambda v: float(np.median(v)) if len(v) else np.nan
    print("\nreading:")
    print(f"   sensor noise floor {med(sw) * 100:.1f} cm (single scans)")
    base = med(table["0.1s"]["all"])           # neighbouring scans: local consistency
    # a timing offset: one stamp shift removes most of what the pairs add over
    # their neighbours. When there is one, the other checks run on wrong poses.
    for n in names:
        b = {off: med(timing[(n, off)]) for off in offsets}
        b = {k: v for k, v in b.items() if np.isfinite(v)}
        if not b or 0.0 not in b:
            continue
        best = min(b, key=b.get)
        m0, mb = b[0.0], b[best]
        if best != 0.0 and mb < 0.8 * m0 and (m0 - mb) > (mb - base):
            print(f"   TIMING: {n} pairs are thinnest with the trajectory's stamps shifted "
                  f"{best * 1000:+.0f} ms ({mb * 100:.1f} vs {m0 * 100:.1f} cm): the pose "
                  f"given for t is the pose at t {'+' if best > 0 else '-'} "
                  f"{abs(best) * 1000:.0f} ms.\n   The deskew and drift checks are skipped: "
                  "on time-shifted poses they would only echo this. Fix it, run again")
            return
    said = False
    for n in ("1s", "10s"):
        tu, sr, rg = (med(table[n][c]) for c in ("turning", "straight", "turn/rigid"))
        if np.isfinite(tu) and np.isfinite(sr) and tu > sr + 0.02 and \
                (not np.isfinite(rg) or tu > 0.8 * rg):
            print(f"   DESKEW: {n} pairs are {100 * (tu - sr):.1f} cm thicker turning than "
                  f"straight, and deskew does not thin them ({tu * 100:.1f} cm vs "
                  f"{rg * 100:.1f} cm without): the point times or the deskew are wrong")
            said = True
            break
    # drift on straight pairs (little rotation, so point timing barely matters),
    # all pairs when there are too few straight ones
    col = "straight" if all(npair[n]["straight"] >= 3 for n in pairs) else "all"
    w = {n: med(table[n][col]) for n in pairs}
    w = {n: v for n, v in w.items() if np.isfinite(v)}
    if "0.1s" in w and len(w) > 1:
        grow = {n: v - w["0.1s"] for n, v in w.items() if n != "0.1s"}
        worst = max(grow, key=grow.get)
        if grow[worst] > 0.03:
            print(f"   DRIFT: {col} pairs thicken with separation ({w['0.1s'] * 100:.1f} cm "
                  f"at 0.1 s, {w[worst] * 100:.1f} cm at {worst}): the poses drift apart"
                  + (" -- worst at the revisits: loop closure / pass to pass"
                     if worst == "revisit" else ""))
            said = True
    if not said:
        print("   no timing, deskew or drift signature between these scans: the wall "
              "thickness of the map comes from something they do not sample")


# ============================================================================
# distances: control distances measured on site against the map
# ============================================================================
DIST_DOC = r'''Control distances: wall-to-wall lengths measured on site with a laser
meter, against the map.

  python3 check.py distances <map.pcd> --click [--zband 0.3,1.2] [--json <file>] [--new]
  python3 check.py distances <map.pcd> --picks picks.txt [--measured 6.214,4.882,...]
  python3 check.py distances <map.pcd> <distances.csv> [--picks picks.txt]
      [--radius 0.3]

--click: a floor plan opens: the map's points between --zband (m, the
map's frame: the LiDAR started at z = 0; default above desks and below the
ceiling) seen from above, the walls as dark lines. Left click near wall A,
then near wall B, for every distance; each click finds the straight wall
nearest it (within 0.5 m), fitted at once and drawn (green), and after B
the distance is drawn with its length (a red cross: no wall near the click,
click closer). Then type the meter's reading in the box under the plan (m,
8.113, or mm, 8113) and Enter: the pair's error is drawn by it and the
totals (mean, RMSE, scale) under the plan, and every pair with its picks,
map distance, reading and error and the totals are saved at once to
<map>_distances.json (--json): a session closed and opened again carries on
from its pairs (--new starts again). Scroll zooms around the cursor, a
right-button drag moves the view, h shows the whole map; u undoes the last
click; q or closing the window finishes. The picks are also saved as
<map>_picks.txt and the readings as <map>_distances.csv; the same check
runs again without clicking: check.py distances <map.pcd> <map>_distances.json.

With picks alone: the wall pairs picked in the map, two points per pair
(CloudCompare's picked points saved as ASCII; the last three numbers of each
line are taken as x y z), named d1, d2, ... in the order picked. Without
--measured each pair's map distance is printed, to match the pairs to the
meter's readings; --measured gives those readings in metres, one per pair in
the same order.

distances.csv (the other way, with names), one row per distance (a header row is fine):
  name, measured_m [, ax, ay, az, bx, by, bz]
with the two walls' picked points (the map's frame) on the row, or without
them from --picks: CloudCompare's picked points saved as ASCII, two per
distance in the csv's order (wall A, then wall B); the last three numbers of
each line are taken as x y z.

Pick A on the wall where the meter was held and B where its dot landed,
within a hand's width: the picks only say WHICH wall and WHERE along it.
Each wall is the plane through the map's points within --radius of its
pick that passes the pick and holds the most of them (RANSAC); other
surfaces in the ball (a floor, a door frame) and the strip where they meet
the wall are left out, so they cannot tilt it; then

  map distance  the meter's line: from A's pick (put on A's plane) along
                A's normal to B's plane, as a meter held flat on wall A
                reads; for parallel walls the same anywhere along them
  error         map - measured, mm, and per metre of distance
  flags         the two walls not parallel (> 2 deg: the distance changes by
                tan(angle) per metre along A, so A's pick must be where the
                meter was; left out of the totals), or a pick on no flat wall

At the end: the mean and RMS error and the scale that fits them all
(map = s x measured), to set against the meter's own 1.5-3 mm.
'''


def _ransac(Q, rng, keep, tries=300, band=0.03):
    """The plane (point, unit normal) with the most points of Q within band,
    among those keep(point, normal) accepts; None when there is none."""
    best, best_n = None, 0
    for _ in range(tries):
        a, b, c = Q[rng.choice(len(Q), 3, replace=False)]
        n = np.cross(b - a, c - a)
        if np.linalg.norm(n) < 1e-9:
            continue
        n /= np.linalg.norm(n)
        if not keep(a, n):
            continue
        k = int((np.abs((Q - a) @ n) < band).sum())
        if k > best_n:
            best, best_n = (a, n), k
    return best


def wall_plane(tree, P, p, r, seed=0):
    """(centre, unit normal, thickness m, points) of the wall the pick p is
    on, or None when there is no flat surface there. RANSAC on the map's
    points within r of p: the plane that passes within 5 cm of the pick (it
    was clicked on the wall) and holds the most points within 3 cm. Other
    surfaces in the ball (a floor, a door frame: planes more than 30 deg off
    it) and the strip where they meet it are taken out, then the wall is
    fitted to what is left by least squares."""
    idx = tree.query_ball_point(p, r)
    if len(idx) < MIN_PTS:
        return None
    Q = P[idx]
    rng = np.random.default_rng(seed)
    w = _ransac(Q, rng, lambda a, n: abs((p - a) @ n) < 0.05)
    if w is None:
        return None
    on = np.abs((Q - w[0]) @ w[1]) < 0.03
    rest = Q[~on]
    for _ in range(2):                    # up to two other surfaces in the ball
        if len(rest) < MIN_PTS:
            break
        o = _ransac(rest, rng, lambda a, n: abs(n @ w[1]) < np.cos(np.radians(30)))
        if o is None or (np.abs((rest - o[0]) @ o[1]) < 0.03).sum() < MIN_PTS:
            break
        on &= np.abs((Q - o[0]) @ o[1]) >= 0.03
        rest = rest[np.abs((rest - o[0]) @ o[1]) >= 0.03]
    Q = Q[on]
    if len(Q) < MIN_PTS:
        return None
    c = Q.mean(0)
    _, sv, vt = np.linalg.svd(Q - c, full_matrices=False)
    if sv[1] < 1e-9 or sv[2] / sv[1] > 0.3:
        return None
    n = vt[2]
    return c, n, spread((Q - c) @ n), len(Q)


def wall_distance(A, B, pa=None):
    """(distance m, angle between the walls deg) of two wall_plane()s: the
    laser meter's line, from pa (the meter on wall A, put on A's plane;
    A's centre without it) along A's normal to B's plane. For parallel walls
    that is the walls' distance anywhere; for walls an angle apart it changes
    by tan(angle) per metre along A, so pa must be where the meter was."""
    (ca, na, *_), (cb, nb, *_) = A, B
    pa = ca if pa is None else np.asarray(pa, float) - na * ((np.asarray(pa, float) - ca) @ na)
    na = na if (cb - pa) @ na > 0 else -na
    cos = float(na @ nb)
    ang = float(np.degrees(np.arccos(min(1.0, abs(cos)))))
    if abs(cos) < 0.2:
        return None, ang
    return float(((cb - pa) @ nb) / cos), ang


def per_metre(ang):
    """cm the laser line changes per metre along wall A, walls ang deg apart."""
    return 100 * np.tan(np.radians(ang))


def read_json_distances(path):
    """[(name, measured m or None, a xyz, b xyz)] from a --click session's json."""
    import json
    js = json.load(open(path))
    return [(q["name"], q.get("measured_m"), np.array(q["a"], float), np.array(q["b"], float))
            for q in js.get("pairs", [])]


def read_distances(path, picks=None):
    """[(name, measured m, a xyz, b xyz)] from the csv (and the picks file),
    or from a --click session's json."""
    import csv
    if path.lower().endswith(".json"):
        return read_json_distances(path)
    rows = []
    with open(path) as fh:
        for row in csv.reader(fh):
            row = [x.strip() for x in row if x.strip()]
            if len(row) < 2:
                continue
            try:
                vals = [float(x) for x in row[1:]]
            except ValueError:
                continue                  # the header
            rows.append((row[0], vals))
    pts = read_picks(picks) if picks else []
    out = []
    for i, (name, vals) in enumerate(rows):
        if len(vals) >= 7:
            a, b = vals[1:4], vals[4:7]
        elif len(pts) >= 2 * i + 2:
            a, b = pts[2 * i], pts[2 * i + 1]
        else:
            raise SystemExit(f"{name}: no picked points (give ax..bz on its row, or --picks "
                             f"with two points per distance: {len(pts)} points for {len(rows)} rows)")
        out.append((name, vals[0], np.array(a, float), np.array(b, float)))
    return out


def read_picks(path):
    """The picked points (x y z: the last three numbers of each line)."""
    import re
    pts = []
    for ln in open(path):
        nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", ln)
        if len(nums) >= 3:
            pts.append([float(x) for x in nums[-3:]])
    return pts


def pick_pairs(picks, measured=None):
    """[(name, measured m or None, a xyz, b xyz)] from the picks alone: d1, d2, ..."""
    pts = read_picks(picks)
    if len(pts) % 2:
        raise SystemExit(f"{picks}: {len(pts)} points, an odd number: two per distance (A, then B)")
    n = len(pts) // 2
    if measured is not None and len(measured) != n:
        raise SystemExit(f"--measured has {len(measured)} values for {n} picked pairs")
    return [(f"d{i + 1}", None if measured is None else measured[i],
             np.array(pts[2 * i], float), np.array(pts[2 * i + 1], float)) for i in range(n)]


def wall_near(S, stree, x, y, r=0.5):
    """The straight wall nearest a floor-plan click at (x, y): RANSAC on the
    slice's points within r (in plan) for a vertical plane passing within
    0.3 m of the click, refitted by least squares to its points within 3 cm
    along 1 m either side of the click.
    (centre, unit normal, thickness m, points, the click on the wall) or None."""
    idx = stree.query_ball_point([x, y], r)
    if len(idx) < MIN_PTS:
        return None
    Q = S[idx]
    q = np.array([x, y, float(np.median(Q[:, 2]))])
    rng = np.random.default_rng(0)
    w = _ransac(Q, rng, lambda a, n: abs(n[2]) < 0.17 and abs((q - a)[:2] @ n[:2]) < 0.3, tries=400)
    if w is None:
        return None
    # the wall's points along 1 m of it either side of the click (a long
    # base for its direction), refitted until the 3 cm band no longer moves
    W = S[stree.query_ball_point([x, y], 1.0)]
    c, n = w
    for _ in range(4):
        Q = W[np.abs((W - c) @ n) < 0.03]
        if len(Q) < MIN_PTS:
            return None
        c = Q.mean(0)
        n = np.linalg.svd(Q - c, full_matrices=False)[2][2]
    p = q - n * ((q - c) @ n)
    return c, n, spread((Q - c) @ n), len(Q), p


def parse_reading(text):
    """The meter's reading typed in: metres (8.113), or millimetres when over
    100 (8113); None for nothing typed. ValueError for anything else."""
    t = text.strip().lower().replace(",", ".").rstrip("m").strip()
    if not t:
        return None
    v = float(t)
    if v <= 0:
        raise ValueError(text)
    return v / 1000.0 if v > 100 else v


def distance_summary(res, max_angle=2.0):
    """The totals of distance rows ([name, measured, a, b, A, B, d, angle]):
    the measured ones with walls parallel to max_angle. None when none."""
    ok = [(r[1], r[6]) for r in res if r[1] is not None and r[6] is not None and r[7] <= max_angle]
    if not ok:
        return None
    meas, mapd = np.array(ok, float).T
    e = (mapd - meas) * 1000
    s = float(meas @ mapd / (meas @ meas))
    return {"n": len(meas), "measured_min_m": float(meas.min()), "measured_max_m": float(meas.max()),
            "mean_mm": float(e.mean()), "rmse_mm": float(np.sqrt(np.mean(e ** 2))),
            "max_abs_mm": float(np.abs(e).max()), "scale": s, "ppm": (s - 1) * 1e6,
            "rmse_after_scale_mm": float(np.sqrt(np.mean(((mapd - s * meas) * 1000) ** 2)))}


def write_distances_json(path, map_path, res, radius, zband=None):
    """The control distances, each with its picks, walls, map distance,
    reading and error, and the totals (distance_summary), to path."""
    import json
    import time
    pairs = []
    for name, m, pa, pb, A, B, d, ang in res:
        q = {"name": name, "a": [round(float(v), 4) for v in pa], "b": [round(float(v), 4) for v in pb],
             "map_m": None if d is None else round(d, 4), "measured_m": m,
             "error_mm": None if (d is None or m is None) else round((d - m) * 1000, 1),
             "angle_deg": None if ang is None else round(ang, 2),
             "wall_a_cm": None if A is None else round(A[2] * 100, 2),
             "wall_b_cm": None if B is None else round(B[2] * 100, 2)}
        if d is None:
            q["flag"] = "no flat wall at a pick"
        elif ang > 2.0:
            q["flag"] = f"not parallel: {per_metre(ang):.1f} cm per m along A; left out of the totals"
        pairs.append(q)
    out = {"map": os.path.abspath(map_path), "radius_m": radius, "zband_m": list(zband) if zband else None,
           "saved": time.strftime("%Y-%m-%d %H:%M:%S"), "pairs": pairs, "summary": distance_summary(res)}
    with open(path, "w") as fh:
        json.dump(out, fh, indent=1)


def summary_line(sm):
    if sm is None:
        return "no reading yet"
    return (f"{sm['n']} measured: error mean {sm['mean_mm']:+.1f} mm, RMSE {sm['rmse_mm']:.1f} mm, "
            f"max |{sm['max_abs_mm']:.1f}| mm, scale {sm['ppm']:+.0f} ppm")


class PlanPicker:
    """The floor-plan clicks of --click: each finds the straight wall nearest
    it (within r); pairs (A, B) give distances, each measured as the table
    does (wall_plane round each pick, wall_distance), and the meter's
    reading entered for it. res: the pairs as _distance_rows gives them."""

    def __init__(self, ax, P, tree, S, radius=0.3, r=0.5):
        self.ax, self.P, self.tree, self.radius, self.r = ax, P, tree, radius, r
        self.S, self.stree = S, cKDTree(S[:, :2])
        self.picks, self.artists = [], []           # (point on the wall, wall), [artists] per click
        self.res, self.labels = [], []               # per pair: its row, its label on the plan

    def _pair(self, pa, pb, wa, wb):
        """The pair's row: the table's walls round the picks (the floor
        plan's slice walls when those find none)."""
        A, B = wall_plane(self.tree, self.P, pa, self.radius), wall_plane(self.tree, self.P, pb, self.radius)
        if A is None or B is None:
            A, B = wa, wb
        d, ang = wall_distance(A, B, pa)
        return [f"d{len(self.res) + 1}", None, pa, pb, A, B, d, ang]

    def _label(self, row):
        name, m, _, _, _, _, d, _ = row
        if m is None:
            return f" {name}  {d:.3f} m"
        return f" {name}  map {d:.3f} · meter {m:.3f} · {(d - m) * 1000:+.0f} mm"

    def click(self, x, y):
        """A click at (x, y): a message; the pick is kept only near a straight wall."""
        w = wall_near(self.S, self.stree, x, y, self.r)
        if w is None:
            self.ax.plot(x, y, "x", color="#d03b3b", ms=10, mew=2)
            return f"no straight wall within {self.r:g} m of the click: click closer to a wall line"
        n, p = w[1], w[4]
        t = np.cross(n, [0, 0, 1.0])
        t = t / (np.linalg.norm(t) or 1)
        seg = np.array([p - 0.6 * t, p + 0.6 * t])
        art = self.ax.plot(seg[:, 0], seg[:, 1], color="#1baf7a", lw=4)
        art += self.ax.plot(p[0], p[1], "o", color="#1baf7a", ms=7)
        self.picks.append((p, w))
        k = len(self.picks)
        if k % 2:
            self.artists.append(art)
            return f"d{k // 2 + 1}: wall A found ({w[2] * 100:.1f} cm thick); now click wall B"
        (pa, wa), (pb, wb) = self.picks[-2], self.picks[-1]
        row = self._pair(pa, pb, wa, wb)
        d, ang = row[6], row[7]
        if d is None:
            self.picks.pop()
            for a in art:
                a.remove()
            return f"walls {ang:.0f} deg apart: not facing each other; click wall B again"
        A = row[4]
        na = A[1] if (row[5][0] - pa) @ A[1] > 0 else -A[1]
        q1 = pa + na * d
        art += self.ax.plot([pa[0], q1[0]], [pa[1], q1[1]], color="#eb6834", lw=2)
        lab = self.ax.text((pa[0] + q1[0]) / 2, (pa[1] + q1[1]) / 2, self._label(row),
                           color="#b4501f", fontsize=12, fontweight="bold")
        art.append(lab)
        self.artists.append(art)
        self.res.append(row)
        self.labels.append(lab)
        warn = (f"  (not parallel: {per_metre(ang):.1f} cm per m along A, so click A where the meter "
                "was; left out of the totals)") if ang > 2.0 else ""
        return f"{row[0]} = {d:.3f} m, walls {ang:.1f} deg apart{warn}; type the meter's reading below"

    def enter(self, text):
        """The meter's reading for the last pair: a message."""
        if not self.res:
            return "pick a pair of walls first"
        row = self.res[-1]
        try:
            m = parse_reading(text)
        except ValueError:
            return f"{text!r}: not a distance; type it in metres (8.113) or mm (8113)"
        row[1] = m
        self.labels[-1].set_text(self._label(row))
        if m is None:
            return f"{row[0]}: no reading; next: wall A of d{len(self.res) + 1}"
        warn = "  (walls not parallel: left out of the totals)" if row[7] > 2.0 else ""
        return (f"{row[0]}: map {row[6]:.3f} m, meter {m:.3f} m, error {(row[6] - m) * 1000:+.1f} mm{warn}"
                f"  ·  {summary_line(distance_summary(self.res))}; next: wall A of d{len(self.res) + 1}")

    def add(self, name, m, pa, pb):
        """A pair from an earlier session: drawn, measured again, its reading kept."""
        wa = wall_near(self.S, self.stree, pa[0], pa[1], self.r)
        wb = wall_near(self.S, self.stree, pb[0], pb[1], self.r)
        self.picks += [(np.asarray(pa, float), wa), (np.asarray(pb, float), wb)]
        row = self._pair(np.asarray(pa, float), np.asarray(pb, float), wa, wb)
        row[0], row[1] = name, m
        self.res.append(row)
        art = self.ax.plot([pa[0], pb[0]], [pa[1], pb[1]], color="#eb6834", lw=2)
        lab = self.ax.text((pa[0] + pb[0]) / 2, (pa[1] + pb[1]) / 2,
                           self._label(row) if row[6] is not None else f" {name}  no wall",
                           color="#b4501f", fontsize=12, fontweight="bold")
        art.append(lab)
        self.artists += [[], art]
        self.labels.append(lab)

    def undo(self):
        if not self.picks:
            return "nothing to undo"
        if len(self.picks) % 2 == 0:                   # B of a pair: the pair and its reading go
            self.res.pop()
            self.labels.pop()
        self.picks.pop()
        for art in self.artists.pop():
            art.remove()
        return f"undone; {len(self.picks)} clicks kept"

    def points(self):
        n = len(self.picks) // 2 * 2
        return [p.tolist() for p, _ in self.picks[:n]]

    def rows(self):
        """[(name, measured m or None, a, b)] of the pairs."""
        return [(r[0], r[1], r[2], r[3]) for r in self.res]


def click_plan(P, tree, zband=(0.3, 1.2), radius=0.3, cell=0.03, save=None, earlier=(), map_path=""):
    """The wall pairs clicked on a floor plan of the map, each with the
    meter's reading typed in the box under it: [(name, measured m or None,
    a xyz, b xyz)]. save: the json written after every reading and undo
    (write_distances_json); earlier: pairs of a session before, drawn and
    kept."""
    import matplotlib.pyplot as plt
    from matplotlib.widgets import TextBox
    S = P[(P[:, 2] >= zband[0]) & (P[:, 2] <= zband[1])]
    if len(S) < 1000:
        raise SystemExit(f"only {len(S)} map points between z {zband[0]} and {zband[1]} m: "
                         "choose --zband where the walls are")
    lo, hi = S[:, :2].min(0), S[:, :2].max(0)
    H, xe, ye = np.histogram2d(S[:, 0], S[:, 1], bins=np.maximum(((hi - lo) / cell).astype(int), 1),
                               range=[[lo[0], hi[0]], [lo[1], hi[1]]])
    with plt.rc_context({"toolbar": "None"}):     # no pan / zoom modes to swallow the clicks
        fig, ax = plt.subplots(figsize=(13, 9.5))
    fig.subplots_adjust(bottom=0.12)
    ax.imshow(np.log1p(H).T, origin="lower", extent=(xe[0], xe[-1], ye[0], ye[-1]), cmap="Greys",
              interpolation="nearest")
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    pk = PlanPicker(ax, P, tree, S, radius)
    for name, m, pa, pb in earlier:
        pk.add(name, m, pa, pb)
    tb = TextBox(fig.add_axes([0.32, 0.025, 0.2, 0.045]), "meter reading of the last pair (m or mm), Enter: ",
                 initial="")
    stats = fig.text(0.55, 0.04, summary_line(distance_summary(pk.res)), fontsize=11, color="#b4501f")
    head = ("left click: wall A, then wall B, then type the reading + Enter  ·  scroll: zoom  ·  "
            "right drag: move  ·  h: whole map  ·  u: undo  ·  q: done")
    home = (ax.get_xlim(), ax.get_ylim())
    drag = {}
    ax.set_title(head + "\n" + (f"{len(pk.res)} pairs from before; " if pk.res else "")
                 + f"click near a dark wall line: d{len(pk.res) + 1}, wall A", fontsize=10)

    def keep():
        stats.set_text(summary_line(distance_summary(pk.res)))
        if save:
            write_distances_json(save, map_path, pk.res, radius, zband)

    def say(msg):
        print("  " + msg)
        ax.set_title(head + "\n" + msg, fontsize=10)
        fig.canvas.draw_idle()

    def on_submit(text):
        if not text.strip() and not (pk.res and pk.res[-1][1] is None):
            return
        say(pk.enter(text))
        keep()
        tb.set_val("")
        tb.stop_typing()

    tb.on_submit(on_submit)

    def on_click(ev):
        if ev.inaxes is not ax:
            return
        if ev.button == 3:                            # right drag moves the view
            drag.update(x=ev.x, y=ev.y, xl=ax.get_xlim(), yl=ax.get_ylim())
        elif ev.button == 1:
            n = len(pk.res)
            say(pk.click(ev.xdata, ev.ydata))
            if len(pk.res) > n:                     # a pair: its reading next, typed straight in
                keep()
                tb.begin_typing()

    def on_move(ev):
        if "x" not in drag:
            return
        bb = ax.get_window_extent()
        (x0, x1), (y0, y1) = drag["xl"], drag["yl"]
        dx = (ev.x - drag["x"]) * (x1 - x0) / bb.width
        dy = (ev.y - drag["y"]) * (y1 - y0) / bb.height
        ax.set_xlim(x0 - dx, x1 - dx)
        ax.set_ylim(y0 - dy, y1 - dy)
        fig.canvas.draw_idle()

    def on_release(ev):
        drag.clear()

    def on_key(ev):
        if tb.capturekeystrokes:                      # typing a reading: the keys are its
            return
        if ev.key == "u":
            say(pk.undo())
            keep()
        elif ev.key == "h":
            ax.set_xlim(*home[0])
            ax.set_ylim(*home[1])
            fig.canvas.draw_idle()
        elif ev.key in ("q", "escape"):
            plt.close(fig)

    def on_scroll(ev):
        if ev.inaxes is not ax:
            return
        f = 0.8 if ev.button == "up" else 1.25
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        ax.set_xlim(ev.xdata + (x0 - ev.xdata) * f, ev.xdata + (x1 - ev.xdata) * f)
        ax.set_ylim(ev.ydata + (y0 - ev.ydata) * f, ev.ydata + (y1 - ev.ydata) * f)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("motion_notify_event", on_move)
    fig.canvas.mpl_connect("button_release_event", on_release)
    fig.canvas.mpl_connect("key_press_event", on_key)
    fig.canvas.mpl_connect("scroll_event", on_scroll)
    print(f"floor plan: {len(S)} points between z {zband[0]} and {zband[1]} m; " + head
          + (f"\n  every reading saved to {save}" if save else ""))
    plt.show()
    return pk.rows()


def _distance_rows(tree, P, rows, radius):
    """[(name, measured, a, b, A, B, d, angle)] with each pair's walls fitted."""
    out = []
    for name, m, pa, pb in rows:
        A, B = wall_plane(tree, P, pa, radius), wall_plane(tree, P, pb, radius)
        d, ang = wall_distance(A, B, pa) if A is not None and B is not None else (None, None)
        out.append([name, m, pa, pb, A, B, d, ang])
    return out


def _distance_table(res):
    print(f"  {'name':14s}{'measured m':>11s}{'map m':>10s}{'error mm':>10s}{'mm/m':>7s}"
          f"{'angle':>7s}{'walls cm':>11s}   flags")
    for name, m, _, _, A, B, d, ang in res:
        ms = "-" if m is None else f"{m:.3f}"
        if d is None:
            which = " and ".join(w for w, x in (("A", A), ("B", B)) if x is None)
            print(f"  {name:14s}{ms:>11s}{'-':>10s}{'':35s}   no flat wall at {which}'s pick")
            continue
        err = "" if m is None else f"{(d - m) * 1000:+10.1f}{(d - m) * 1000 / m:+7.2f}"
        print(f"  {name:14s}{ms:>11s}{d:10.3f}{err:17s}{ang:6.1f}d"
              f"{A[2] * 100:5.1f}/{B[2] * 100:<4.1f}   "
              f"{f'not parallel: {per_metre(ang):.1f} cm per m along A' if ang > 2.0 else ''}")


def main_distances(argv=None):
    ap = argparse.ArgumentParser(description=DIST_DOC,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("map")
    ap.add_argument("distances", nargs="?",
                    help="csv with names, or a --click session's json (optional: picks alone number the pairs)")
    ap.add_argument("--click", action="store_true", help="pick the wall pairs on the map in a window")
    ap.add_argument("--json", help="--click: the session's file (default <map>_distances.json; "
                                   "its pairs are drawn again and kept)")
    ap.add_argument("--new", action="store_true", help="--click: start again, not from the json's pairs")
    ap.add_argument("--picks")
    ap.add_argument("--measured", help="the meter's readings, m, one per picked pair: 6.214,4.882,...")
    ap.add_argument("--ask", action="store_true", help="ask for each unread pair's reading in the terminal")
    ap.add_argument("--radius", type=float, default=0.3)
    ap.add_argument("--zband", default="0.3,1.2",
                    help="--click: the floor plan's slice, z from,to in m (the LiDAR started at 0)")
    a = ap.parse_args(argv)
    if not (a.click or a.distances or a.picks):
        ap.error("give --click, or --picks (and --measured), or a distances csv / json")
    stem = os.path.splitext(a.map)[0]
    measured = None
    if a.measured:
        measured = [float(x) for x in a.measured.replace(",", " ").split()]
    zband = [float(x) for x in a.zband.split(",")]
    P = load(a.map)
    tree = cKDTree(P)
    js = a.json or (a.distances if a.distances and a.distances.lower().endswith(".json")
                    else stem + "_distances.json")
    if a.click:
        earlier = read_json_distances(js) if os.path.exists(js) and not a.new else []
        if earlier:
            print(f"{len(earlier)} pairs from {js} (--new starts again)")
        rows = click_plan(P, tree, zband, a.radius, save=js, earlier=earlier, map_path=a.map)
        if not rows:
            raise SystemExit("no wall pair clicked")
        if measured is not None:
            if len(measured) != len(rows):
                raise SystemExit(f"--measured has {len(measured)} values for {len(rows)} pairs")
            rows = [(n_, measured[i], pa, pb) for i, (n_, _, pa, pb) in enumerate(rows)]
        a.picks = a.picks or stem + "_picks.txt"
        with open(a.picks, "w") as fh:
            fh.writelines(f"{x:.4f},{y:.4f},{z:.4f}\n" for _, _, pa, pb in rows for x, y, z in (pa, pb))
        print(f"saved {len(rows)} pairs -> {a.picks}")
    elif a.distances:
        rows = read_distances(a.distances, a.picks)
    elif a.picks:
        rows = pick_pairs(a.picks, measured)
    print(f"{a.map}: {len(P)} points; {len(rows)} control distances\n")
    res = _distance_rows(tree, P, rows, a.radius)
    _distance_table(res)

    ask = a.ask or (a.click and sys.stdin.isatty() and all(r[1] is None for r in res))
    if ask and any(r[1] is None and r[6] is not None for r in res):
        print("\nthe meter's reading for each pair, m (Enter skips it):")
        for r in res:
            if r[6] is None or r[1] is not None:
                continue
            got = input(f"  {r[0]}  map {r[6]:.3f} m  measured: ").strip()
            if got:
                r[1] = parse_reading(got)
        print()
        _distance_table(res)
    if any(r[1] is not None for r in res):
        out = stem + "_distances.csv"
        with open(out, "w") as fh:
            fh.write("name,measured_m,ax,ay,az,bx,by,bz\n")
            for name, m, pa, pb, *_ in res:
                if m is not None:
                    fh.write(f"{name},{m:.4f}," + ",".join(f"{v:.4f}" for v in (*pa, *pb)) + "\n")
        print(f"\nsaved the readings with their picks -> {out}")
    if a.click or a.distances and a.distances.lower().endswith(".json") or any(r[1] is not None for r in res):
        write_distances_json(js, a.map, res, a.radius, zband if a.click else None)
        print(f"saved every pair, its error and the totals -> {js}")

    if all(r[1] is None for r in res):
        print("\nmap distances only: give the meter's readings in the same order with "
              "--measured 6.214,4.882,...")
        print("(walls: each wall's thickness in the map, p95-p5, around its pick)")
        return
    sm = distance_summary(res)
    if sm is None:
        raise SystemExit("\nno distance measured: check the picks (the map's frame, on the walls)")
    print(f"\n{sm['n']} distances ({sm['measured_min_m']:.2f}-{sm['measured_max_m']:.2f} m): error mean "
          f"{sm['mean_mm']:+.1f} mm, RMS {sm['rmse_mm']:.1f} mm, max |{sm['max_abs_mm']:.1f}| mm")
    print(f"scale map / measured {sm['scale']:.5f} ({sm['ppm']:+.0f} ppm = {(sm['scale'] - 1) * 100:+.3f} %); "
          f"after it, RMS {sm['rmse_after_scale_mm']:.1f} mm")
    print("(walls: each wall's thickness in the map, p95-p5, around its pick)")


# ============================================================================
# noise: one scan's own spread on flat surfaces, straight from a bag
# ============================================================================
NOISE_DOC = r'''The LiDAR's noise floor from a bag alone: no trajectory, no map.

  python3 check.py noise <bag> [--topic /ouster/points] [--scans 50]
      [--patches 400] [--min-range 1.0] [--max-range 30]

From --scans scans spread evenly over the bag, patches of 0.25 m radius
around random points of each scan, in the scan's own frame: a patch that is
one flat layer (as `map` tests it) gets a plane fitted, and its thickness is
the 5-95 % spread of its points about it, as `map` and `walls` measure it.
One scan's points at a spot are taken within a few ms of the 100 ms sweep
(the robot moves under 2 mm meanwhile), so no pose is needed: this is the
sensor alone, the "1 scan" column of `walls`.

Reported as medians: walls (normal more than 60 deg from the sensor's z;
the sensor mounted near level) and floors / ceilings (less than 30), the
walls by range and by incidence (the angle between the beam and the wall's
normal: grazing hits spread wider).
'''

INC_BINS = (0.0, 30.0, 60.0, 75.0, 90.01)
CHI_ = 3.29                       # 5-95 % spread of a normal = 3.29 sigma


def scan_noise(xyz, rng, n=400, r=R):
    """Per flat one-layer patch of one scan, in its own frame: (thickness m,
    normal tilt from z deg, range m, incidence deg)."""
    tree = cKDTree(xyz)
    out = []
    for i in rng.choice(len(xyz), size=min(n, len(xyz)), replace=False):
        nb = tree.query_ball_point(xyz[i], r)
        if len(nb) < MIN_PTS:
            continue
        Q = xyz[nb]
        c = Q.mean(0)
        _, sv, vt = np.linalg.svd(Q - c, full_matrices=False)
        if sv[1] < 1e-9 or sv[2] / sv[1] > PLANARITY:
            continue
        nrm = vt[2]
        d = (Q - c) @ nrm
        if split_layers(d) is not None:
            continue                      # two layers: an edge, not one surface
        rng_m = float(np.linalg.norm(c))
        inc = float(np.degrees(np.arccos(min(1.0, abs(c @ nrm) / max(rng_m, 1e-9)))))
        out.append((spread(d), float(np.degrees(np.arccos(min(1.0, abs(nrm[2]))))), rng_m, inc))
    return np.array(out).reshape(-1, 4)


def main_noise(argv=None):
    ap = argparse.ArgumentParser(description=NOISE_DOC,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bag")
    ap.add_argument("--topic", default="")
    ap.add_argument("--scans", type=int, default=50)
    ap.add_argument("--patches", type=int, default=400, help="patches tried per scan")
    ap.add_argument("--min-range", type=float, default=1.0, help="m (the robot itself)")
    ap.add_argument("--max-range", type=float, default=30.0)
    a = ap.parse_args(argv)
    sys.path.insert(0, ROOT)
    from scoop import bag as sbag
    reader = sbag.open_bag(a.bag)
    topic = sbag.detect_points_topic(reader, a.topic)
    total = reader.topics()[topic].count
    stride = max(1, total // max(1, a.scans))
    seen = [0]

    def every(_stamp):
        seen[0] += 1
        k = seen[0] - 1
        return k if k % stride == 0 else None
    rng = np.random.default_rng(0)
    rows, n_scans = [], 0
    print(f"{a.bag}\n{topic}: {total} scans, every {stride}th, "
          f"range {a.min_range:g}-{a.max_range:g} m")
    for scan in sbag.iter_scans(reader, topic, select=every, min_range=a.min_range,
                                max_range=a.max_range, min_points=1000):
        st = scan_noise(np.asarray(scan.xyz, np.float64), rng, a.patches)
        rows.append(st)
        n_scans += 1
    st = np.vstack(rows) if rows else np.zeros((0, 4))
    if not len(st):
        raise SystemExit("no flat patches found")
    th, tilt, rg, inc = st[:, 0] * 100, st[:, 1], st[:, 2], st[:, 3]
    wall, floor = tilt > 60, tilt < 30
    med = lambda m: f"{np.median(th[m]):5.1f} cm ({m.sum()})" if m.sum() >= 10 else "    -"  # noqa: E731
    print(f"\n{n_scans} scans, {len(st)} flat patches; one scan's thickness, median (patches)")
    sig = f"   sigma {np.median(th[wall]) / CHI_:.1f} cm per point" if wall.sum() >= 10 else ""
    print(f"  walls             {med(wall)}{sig}")
    print(f"  floors / ceilings {med(floor)}")
    print(f"  all               {med(np.ones(len(st), bool))}")
    print("\nwalls by range (m)")
    for lo, hi in zip(DIST[:-1], DIST[1:]):
        m = wall & (rg >= lo) & (rg < hi)
        print(f"  {lo:4g}-{'' if hi > 1e8 else f'{hi:g}':4s} {med(m)}")
    print("walls by incidence (deg between beam and wall normal)")
    for lo, hi in zip(INC_BINS[:-1], INC_BINS[1:]):
        m = wall & (inc >= lo) & (inc < hi)
        print(f"  {lo:4.0f}-{min(hi, 90):<4.0f} {med(m)}")

