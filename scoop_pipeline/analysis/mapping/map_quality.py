#!/usr/bin/env python3
"""How good is the map? Measures local surface thickness across the cloud.

  python3 scoop_pipeline/analysis/mapping/map_quality.py \
      <data>/processed/<date>/<pass>/mapping/denoised_<pass>_<date>.pcd [--save patches.ply]
      [--samples 4000]

--save writes every measured patch as a point at its centre with its thickness
(cm), tilt (deg, 0 = floor/ceiling, 90 = wall) and point count as scalar
fields (ASCII PLY, named scalar_* so CloudCompare loads them as scalar
fields by itself: colour by "thickness_cm"), so the thick places can be seen
on the map. --samples measures more patches (denser).

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


def load(path):
    if path.endswith(".npy"):
        return np.load(path)
    import open3d as o3d
    pcd = o3d.io.read_point_cloud(path)
    return np.asarray(pcd.points)


def patch_stats(P, n_samples=N_SAMPLES, r=R, seed=0, centres=None):
    """Per-patch (thickness_m, tilt_deg_of_normal_from_vertical); with a
    list as ``centres``, each patch's (x, y, z, n_points) is appended to it."""
    from scipy.spatial import cKDTree
    rng = np.random.default_rng(seed)
    tree = cKDTree(P)
    idx = rng.choice(len(P), size=min(n_samples, len(P)), replace=False)
    out = []
    n_sparse = n_rough = 0
    for i in idx:
        nb = tree.query_ball_point(P[i], r)
        if len(nb) < MIN_PTS:
            n_sparse += 1
            continue
        Q = P[nb]
        Q = Q - Q.mean(0)
        # SVD singular values are sqrt(N) * the spread along each axis
        sv = np.linalg.svd(Q, compute_uv=False) / np.sqrt(len(Q))
        if sv[1] < 1e-9 or sv[2] / sv[1] > PLANARITY:
            n_rough += 1
            continue                      # edge, corner, clutter -- or a
            # surface so smeared it no longer looks planar. Both land here,
            # which is why the rejection count has to be reported: a high
            # n_rough means the median below is biased LOW.
        vt = np.linalg.svd(Q, full_matrices=False)[2]
        n = vt[2]
        d = Q @ n
        # 5-95 percentile span: robust to the few stragglers a radius ball
        # always catches from an adjoining surface
        thick = np.percentile(d, 95) - np.percentile(d, 5)
        out.append((thick, np.degrees(np.arccos(abs(n[2])))))
        if centres is not None:
            c = P[nb].mean(0)
            centres.append((c[0], c[1], c[2], len(nb)))
    cap = 3.29 * PLANARITY * r / 2 * 100
    return np.array(out), dict(sparse=n_sparse, rough=n_rough, cap_cm=cap)


def save_patches(path, stats, centres):
    """ASCII PLY, one vertex per measured patch, thickness / tilt / count.
    CloudCompare takes only properties named scalar_<name> as scalar fields
    without asking (shown as <name>); others are dropped on open."""
    with open(path, "w") as fh:
        fh.write("ply\nformat ascii 1.0\nelement vertex %d\n" % len(stats))
        for f in ("x", "y", "z", "scalar_thickness_cm", "scalar_tilt_deg", "scalar_n_points"):
            fh.write("property float %s\n" % f)
        fh.write("end_header\n")
        for (t, tilt), (x, y, z, n) in zip(stats, centres):
            fh.write("%.3f %.3f %.3f %.2f %.1f %d\n" % (x, y, z, t * 100, tilt, n))


def main():
    args = sys.argv[1:]
    opt = {}
    for flag in ("--save", "--samples"):
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
    st, rej = patch_stats(P, n_samples=n_samples, centres=centres)
    if len(st) == 0:
        raise SystemExit("no planar patches found -- is this a surface map?")
    if centres is not None:
        save_patches(opt["--save"], st, centres)
        print(f"patches -> {opt['--save']} (colour by thickness_cm)")
    t = st[:, 0] * 100                     # cm
    tilt = st[:, 1]
    q = np.percentile(t, [10, 25, 50, 75, 90])

    n_try = min(n_samples, len(P))
    print(f"\n{len(st)} planar patches measured of {n_try} sampled "
          f"(radius {R} m)")
    print(f"  rejected: {rej['sparse']} too sparse, {rej['rough']} non-planar")
    print(f"  reportable ceiling {rej['cap_cm']:.1f} cm -- patches thicker "
          f"than this land in 'non-planar' and are NOT counted, so a large\n"
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
                  f"({m.sum()} patches)")

    print("\nhistogram (cm):")
    h, e = np.histogram(t, bins=np.arange(0, min(30, t.max()) + 1, 1))
    peak = max(h.max(), 1)
    for a, v in zip(e[:-1], h):
        if v:
            print(f"  {a:4.0f}-{a + 1:<4.0f} {v:5d} {'#' * int(50 * v / peak)}")

    m = q[2]
    print()
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
