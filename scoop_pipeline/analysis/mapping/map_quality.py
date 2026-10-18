#!/usr/bin/env python3
"""How good is the map? Measures local surface thickness across the cloud.

  python3 scoop_pipeline/analysis/mapping/map_quality.py \
      <data>/processed/<date>/<pass>/mapping/denoised_<pass>_<date>.pcd \
      [--traj <data>/processed/<date>/<pass>/odometry/<machine>/traj_lidar_refined.txt] \
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
"""
import sys
import numpy as np

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


def main():
    args = sys.argv[1:]
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


if __name__ == "__main__":
    main()
