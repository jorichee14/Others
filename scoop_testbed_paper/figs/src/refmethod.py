#!/usr/bin/env python3
"""What each mapping step does to a wall: figs/fig_refmethod.pdf (and .png).

    python3 figs/src/refmethod.py

Computed on a simulated room seen from above, not on 20261015: a planar
LiDAR with the Ouster's angular step (1024 per turn) and 0.9 cm range noise
(sigma of the sensor alone). Every panel shows the same wall, as the offset
of its points from the wall (cm) against the position along it (m). The
shaded bars are the 5-95 % spread of each 0.25 m patch, the paper's wall
thickness; their medians are printed, not drawn, because the simulation's
walls are perfect planes and its numbers are not the dataset's (those are
in tables/refmap_stages.tex). Top row before the step, bottom row after.

  (a) pose-time correction: scans of a walk through a turn, placed with poses
      50 ms late (GLIM labels a pose with the start of its sweep) and with the
      4 ms left after the correction.
  (b) plane bundle adjustment: one scan per 1 s keyframe around the room, the
      keyframes placed with a drifting odometry, then adjusted as in
      map_stages/02_bundle_adjust.py in 2-D: 0.5 m cubes, a cube a plane when
      thin and seen by 3 keyframes, the cost each plane's points' squared
      distance to it with the plane solved in closed form, an odometry
      factor, the first keyframe fixed, three rounds.
  (c) fusion: scans placed with a 2.7 cm spread (sigma_s of the reference)
      fused as the union (one point per 2 cm cell any scan hit) and as a TSDF
      (2 cm cells, truncation 8 cm, at least 3 rays, zero crossings), as in
      01_build_map.py. The placement errors are shared within each of 8
      visits of the wall, as pose errors are; independent errors would
      average away in the TSDF entirely.
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                         # noqa: E402
from scipy.optimize import least_squares                                # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "fig_refmethod")

INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
GREY, BLUE, ORANGE = "#8a8984", "#2a78d6", "#eb6834"     # GLIM, GLIM corrected, reference / TSDF
FS = 7.0
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": FS, "axes.edgecolor": INK2,
                     "axes.linewidth": 0.6, "xtick.color": INK2, "ytick.color": INK2,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6})

RNG = np.random.default_rng(7)
SIGMA = 0.009                          # sensor alone, m
STEP = 2 * np.pi / 1024                # Ouster angular step
PATCH = 0.25                           # thickness patch, m
WALL_X = (4.0, 6.0)                    # the wall shown: y = 0, x in this range

# room 10 x 6 m with a pillar in the middle, as segments (a, b)
_box = [(4.4, 2.6), (5.6, 2.6), (5.6, 3.4), (4.4, 3.4)]
_room = [(0, 0), (10, 0), (10, 6), (0, 6)]
SEGS = np.array([(p, q) for poly in (_room, _box) for p, q in zip(poly, poly[1:] + poly[:1])], float)


def rot(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s], [s, c]])


def scan(pose, max_range=12.0):
    """One sweep from pose (x, y, heading): points in the sensor frame."""
    phi = np.arange(0, 2 * np.pi, STEP)
    d = np.stack([np.cos(phi + pose[2]), np.sin(phi + pose[2])], 1)          # N x 2
    a, e = SEGS[:, 0], SEGS[:, 1] - SEGS[:, 0]                              # S x 2
    ao = a[None] - np.asarray(pose[:2])[None, None]                        # 1 x S x 2
    cross = lambda u, v: u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]    # noqa: E731
    den = cross(d[:, None], e[None])
    with np.errstate(divide="ignore", invalid="ignore"):
        t = cross(ao, e[None]) / den
        s = cross(ao, d[:, None]) / den
    t[~((np.abs(den) > 1e-12) & (t > 0.1) & (s >= 0) & (s <= 1))] = np.inf
    r = t.min(1)
    ok = r < max_range
    r = r[ok] + RNG.normal(0, SIGMA, ok.sum())
    return r[:, None] * np.stack([np.cos(phi[ok]), np.sin(phi[ok])], 1)


def place(p, pose):
    return p @ rot(pose[2]).T + np.asarray(pose[:2])


def on_wall(w, pad=0.15):
    """Points of the shown wall: (along, offset)."""
    keep = (np.abs(w[:, 1]) < pad) & (w[:, 0] > WALL_X[0]) & (w[:, 0] < WALL_X[1])
    return w[keep]


def thickness(w):
    """Patches of PATCH m: (x0, p5, p95) each, and the median 5-95 % spread."""
    bars = []
    for x0 in np.arange(WALL_X[0], WALL_X[1] - 1e-9, PATCH):
        y = w[(w[:, 0] >= x0) & (w[:, 0] < x0 + PATCH), 1]
        if len(y) >= 10:
            bars.append((x0, *np.percentile(y, [5, 95])))
    bars = np.array(bars)
    return bars, float(np.median(bars[:, 2] - bars[:, 1]))


# ------------------------------------------------------------------ (a) pose time
def pose_time():
    """A walk at 0.8 m/s through a left turn; scans at 10 Hz placed with the
    pose dt later than their own time."""
    ts = np.arange(0, 6, 0.001)
    w = 0.8 * np.exp(-((ts - 2.5) / 0.7) ** 2)                 # rad/s, about 80 deg in all
    th = np.cumsum(w) * 0.001
    xy = np.array([1.5, 1.2]) + np.cumsum(0.8 * np.stack([np.cos(th), np.sin(th)], 1), 0) * 0.001
    pose = lambda t: np.array([*(np.interp(t, ts, xy[:, i]) for i in (0, 1)), np.interp(t, ts, th)])  # noqa: E731
    out = {}
    times = np.arange(0.5, 5.0, 0.1)
    scans = [(t, scan(pose(t))) for t in times]
    for dt in (0.050, 0.004):
        out[dt] = np.concatenate([on_wall(place(p, pose(t + dt))) for t, p in scans])
    return out[0.050], out[0.004]


# ------------------------------------------------------------------ (b) bundle adjustment
def voxel_thin(p, v=0.05):
    k = np.floor(p / v).astype(np.int64)
    _, inv = np.unique(k, axis=0, return_inverse=True)
    inv = inv.ravel()
    n = np.bincount(inv)
    return np.stack([np.bincount(inv, p[:, i]) / n for i in (0, 1)], 1)


def compose(a, b):
    return np.array([*(place(b[None, :2], a)[0]), a[2] + b[2]])


def relative(a, b):
    return np.array([*((np.asarray(b[:2]) - a[:2]) @ rot(a[2])), b[2] - a[2]])


def find_planes(clouds, poses, c=0.5, sig_max=0.06, thin=0.3, min_kf=3):
    """Cubes of edge c whose points are flat: lists of (keyframe, point index)."""
    w = np.concatenate([place(p, T) for p, T in zip(clouds, poses)])
    kf = np.concatenate([np.full(len(p), k) for k, p in enumerate(clouds)])
    idx = np.concatenate([np.arange(len(p)) for p in clouds])
    key = np.floor(w / c).astype(np.int64)
    _, inv = np.unique(key, axis=0, return_inverse=True)
    planes = []
    for v in range(inv.max() + 1):
        m = np.flatnonzero(inv.ravel() == v)
        if len(np.unique(kf[m])) < min_kf:
            continue
        q = w[m] - w[m].mean(0)
        sig = np.sqrt(np.linalg.eigvalsh(q.T @ q / len(m)))
        if sig[0] < sig_max and sig[0] < thin * sig[1] and sig[1] > 0.1 * c:
            planes.append((kf[m], idx[m]))
    return planes


def bundle_adjust(clouds, init, odo, s_odo=(0.006, 0.006, np.radians(0.1)), rounds=3):
    K = len(clouds)
    poses = np.array(init, float)
    for _ in range(rounds):
        planes = find_planes(clouds, poses)
        pts = [np.array([clouds[k][i] for k, i in zip(kk, ii)]) for kk, ii in planes]

        def residuals(x):
            P = np.vstack([poses[:1], x.reshape(K - 1, 3)])
            res = []
            for (kk, _), p in zip(planes, pts):
                c, s = np.cos(P[kk, 2]), np.sin(P[kk, 2])
                w = np.stack([c * p[:, 0] - s * p[:, 1], s * p[:, 0] + c * p[:, 1]], 1) + P[kk, :2]
                q = w - w.mean(0)
                n = np.linalg.eigh(q.T @ q)[1][:, 0]       # the plane, in closed form
                res.append(q @ n / SIGMA)
            for k in range(K - 1):
                e = relative(P[k], P[k + 1]) - odo[k]
                e[2] = (e[2] + np.pi) % (2 * np.pi) - np.pi
                res.append(e / np.asarray(s_odo))
            return np.concatenate(res)

        x = least_squares(residuals, poses[1:].ravel(), method="trf", x_scale="jac").x
        poses = np.vstack([poses[:1], x.reshape(K - 1, 3)])
    return poses


def bundle():
    """One scan per keyframe, 1 s apart, around the pillar; odometry with
    0.6 cm and 0.1 deg noise per keyframe."""
    a = np.linspace(0, 2 * np.pi, 31)[:-1] - np.pi / 2
    truth = np.stack([5 + 3.4 * np.cos(a), 3 + 2.0 * np.sin(a), a + np.pi / 2], 1)
    clouds = [voxel_thin(scan(T)) for T in truth]
    odo = []
    for k in range(len(truth) - 1):
        d = relative(truth[k], truth[k + 1])
        odo.append(d + RNG.normal(0, [0.006, 0.006, np.radians(0.1)]))
    init = [truth[0]]
    for d in odo:
        init.append(compose(init[-1], d))
    init = np.array(init)
    ba = bundle_adjust(clouds, init, odo)
    wall = lambda P: np.concatenate([on_wall(place(p, T)) for p, T in zip(clouds, P)])  # noqa: E731
    return wall(init), wall(ba)


# ------------------------------------------------------------------ (c) fusion
def fusion(visits=8, per_visit=20, sigma_s=0.027, cell=0.02, tau=0.08, w_min=3):
    """Scans of the wall from all over the room, placed with pose errors that
    are shared within a visit (2.4 cm along the wall's normal, 0.3 deg) plus
    0.8 cm per scan: with the sensor's 0.9 cm, one scan's points spread
    sigma_s."""
    origins, hits = [], []
    for v in range(visits):
        dv, av = RNG.normal(0, 0.024), RNG.normal(0, np.radians(0.3))
        for k in range(per_visit):
            T = np.array([RNG.uniform(1, 9), RNG.uniform(0.8, 2.2), RNG.uniform(-np.pi, np.pi)])
            E = np.array([0.0, dv + RNG.normal(0, 0.008), av])        # the placement error
            w = place(place(scan(T), T), E)
            keep = (np.abs(w[:, 1]) < 0.15) & (w[:, 0] > WALL_X[0] - 0.3) & (w[:, 0] < WALL_X[1] + 0.3)
            hits.append(w[keep])
            origins.append(np.repeat(place(T[None, :2], E), keep.sum(), 0))
    p, o = np.concatenate(hits), np.concatenate(origins)

    # the union: one point (the centroid) per cell any scan hit
    key = np.floor(p / cell).astype(np.int64)
    _, inv = np.unique(key, axis=0, return_inverse=True)
    inv = inv.ravel()
    n = np.bincount(inv)
    union = np.stack([np.bincount(inv, p[:, i]) / n for i in (0, 1)], 1)

    # TSDF: every cell within tau of the hit along the ray gets r - |c - o|
    x0 = np.array([WALL_X[0] - 0.4, -0.4])
    nx, ny = int(2.8 / cell) + 1, int(0.8 / cell)
    D, W = np.zeros((nx, ny)), np.zeros((nx, ny))
    u = (p - o) / np.linalg.norm(p - o, axis=1)[:, None]
    r = np.linalg.norm(p - o, axis=1)
    s = np.arange(-tau, tau + 1e-9, cell / 2)
    q = p[:, None] + u[:, None] * s[None, :, None]                   # samples along each ray
    ij = np.floor((q - x0) / cell).astype(np.int64)
    ok = (ij[..., 0] >= 0) & (ij[..., 0] < nx) & (ij[..., 1] >= 0) & (ij[..., 1] < ny)
    ray = np.repeat(np.arange(len(p)), len(s)).reshape(len(p), len(s))
    pairs = np.unique(np.stack([ray[ok], ij[ok][:, 0], ij[ok][:, 1]], 1), axis=0)
    c = x0 + (pairs[:, 1:] + 0.5) * cell
    d = r[pairs[:, 0]] - np.linalg.norm(c - o[pairs[:, 0]], axis=1)
    m = np.abs(d) <= tau
    np.add.at(D, (pairs[m, 1], pairs[m, 2]), d[m])
    np.add.at(W, (pairs[m, 1], pairs[m, 2]), 1)
    D = np.where(W > 0, D / np.maximum(W, 1), 0)

    surf = []
    for di, dj in ((1, 0), (0, 1)):
        a, b = (slice(0, nx - di), slice(0, ny - dj)), (slice(di, nx), slice(dj, ny))
        Da, Db = D[a], D[b]
        good = (W[a] >= w_min) & (W[b] >= w_min) & (Da * Db < 0) & (np.abs(Da - Db) < tau)
        I, J = np.nonzero(good)
        f = Da[I, J] / (Da[I, J] - Db[I, J])
        ca = x0 + (np.stack([I, J], 1) + 0.5) * cell
        surf.append(ca + f[:, None] * cell * np.array([di, dj]))
    surf = np.concatenate(surf)
    return on_wall(union), on_wall(surf)


# ------------------------------------------------------------------ figure
def panel(ax, w, color, label, size=1.6):
    bars, t = thickness(w)
    for x0, lo, hi in bars:
        ax.add_patch(plt.Rectangle((x0 + 0.01, lo * 100), PATCH - 0.02, (hi - lo) * 100,
                                   fc=color, ec="none", alpha=0.16, zorder=1))
    ax.scatter(w[:, 0], w[:, 1] * 100, s=size, lw=0, color=color, alpha=0.55, zorder=2,
               rasterized=True)
    ax.axhline(0, color=INK2, lw=0.5, ls=(0, (3, 2)), zorder=0)
    ax.text(0.02, 0.95, label, transform=ax.transAxes, ha="left", va="top", color=INK, zorder=3,
            bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.5))
    ax.set_xlim(*WALL_X)
    ax.set_ylim(-12, 12)
    ax.set_yticks([-10, -5, 0, 5, 10])
    ax.grid(axis="y", color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    return t


def main():
    a0, a1 = pose_time()
    b0, b1 = bundle()
    c0, c1 = fusion()

    fig, axs = plt.subplots(2, 3, figsize=(7.16, 3.7), sharex=True, sharey=True)
    cols = [("(a) Pose-time correction", (a0, GREY, "poses 50 ms late"), (a1, BLUE, "corrected (4 ms left)")),
            ("(b) Plane bundle adjustment", (b0, BLUE, "drifting odometry"), (b1, ORANGE, "after adjustment")),
            ("(c) Fusion", (c0, GREY, "union of points"), (c1, ORANGE, "TSDF: one surface, at the rays' mean"))]
    thick = {}
    for j, (title, top, bot) in enumerate(cols):
        axs[0, j].set_title(title, loc="left", fontsize=FS + 1, fontweight="bold", color=INK, pad=6)
        thick[title] = (panel(axs[0, j], *top), panel(axs[1, j], *bot, size=3.0 if j == 2 else 1.6))
        axs[1, j].set_xlabel("along the wall (m)")
    for i, row in enumerate(("before", "after")):
        axs[i, 0].set_ylabel(f"{row}\noffset from wall (cm)")
    fig.tight_layout(h_pad=0.8, w_pad=0.6)
    fig.savefig(OUT + ".pdf", dpi=300)
    fig.savefig(OUT + ".png", dpi=300)
    for k, (t0, t1) in thick.items():
        print(f"{k}: {t0 * 100:.1f} -> {t1 * 100:.1f} cm")
    print(OUT + ".pdf", OUT + ".png")


if __name__ == "__main__":
    main()
