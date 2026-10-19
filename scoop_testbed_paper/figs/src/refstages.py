#!/usr/bin/env python3
"""What each mapping step does to a wall, at the measured spreads of pass
20261015: figs/fig_refstages.pdf (and .png).

    python3 figs/src/refstages.py

The points are simulated (the clouds are not in the repository), but every
panel is calibrated so that its median wall thickness, the p95-p5 spread
about the plane in 0.25 m patches, equals the value measured on 20261015
(tables/refmap_stages.tex, RESULTS_STATUS.md). Each scan is placed with an
offset shared by all its points, the way a pose error moves a whole scan;
the offsets are shared within each of 8 visits of the wall (drift) and vary
per scan (pose-time error, which grows with the turn rate). The dashed band
in every panel is the sensor alone, 3.0 cm (sigma 0.9 cm).

  (a) pose-time correction, scans as placed over the run:
      GLIM as output (poses 50 ms late) 19.2 cm -> corrected (4 ms left) 12.3 cm
  (b) plane bundle adjustment, scans as placed over the run:
      GLIM corrected 12.3 cm -> reference 9.0 cm
  (c) fusion, the map built from the reference poses:
      union of points 12.8 cm -> TSDF 7.8 cm
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                         # noqa: E402
from scipy.optimize import brentq                                       # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "fig_refstages")

INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
GREY, BLUE, ORANGE = "#8a8984", "#2a78d6", "#eb6834"
FS = 7.0
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": FS, "axes.edgecolor": INK2,
                     "axes.linewidth": 0.6, "xtick.color": INK2, "ytick.color": INK2,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6})

# measured on 20261015 (cm): median p95-p5 thickness of walls
SENSOR = 3.0                    # each scan against its own plane, sigma 0.9 cm
MEASURED = {
    "glim_output": 19.2,        # scans as placed over the run, GLIM as exported
    "glim_corrected": 12.3,     # same, GLIM moved +46 ms (4 ms left)
    "reference": 9.0,           # same, after plane bundle adjustment
    "union": 12.8,              # map from the reference poses, every point kept
    "tsdf": 7.8,                # map from the reference poses, TSDF zero crossings
}
SIGMA = 0.9                     # cm, sensor alone
PATCH = 0.25                    # m
WALL = (4.0, 6.0)               # m
VISITS, SCANS, PTS = 8, 20, 60  # visits of the wall, scans per visit, wall points per scan
LAG_LEFT = 4.0 / 50.0           # pose-time error left after the correction

RNG = np.random.default_rng(11)
# common random numbers, so thickness is a smooth function of each scale
X = RNG.uniform(*WALL, (VISITS, SCANS, PTS))
NOISE = RNG.normal(0, SIGMA, (VISITS, SCANS, PTS))
DRIFT = RNG.normal(0, 1, (VISITS, 1, 1))                       # shared within a visit
TILT = RNG.normal(0, 1, (VISITS, SCANS, 1))                    # small heading error per scan
TURN = RNG.laplace(0, 1, (VISITS, SCANS, 1))                   # turn rate: mostly straight, some turns
JIT = RNG.normal(0, 1, (VISITS, SCANS, 1))                     # per-scan residual
NT = int((WALL[1] - WALL[0]) / 0.02)                           # TSDF: one point per 2 cm cell ...
XT = np.repeat(WALL[0] + 0.01 + 0.02 * np.arange(NT), 6)       # ... over 6 rows of the wall's height
TS = RNG.normal(0, 1, XT.size)
TSMOOTH = np.convolve(RNG.normal(0, 1, XT.size + 40), np.ones(41) / np.sqrt(41), "valid")


def thick(x, y):
    """Median p95-p5 of y over PATCH-long patches along x (cm)."""
    t = []
    for x0 in np.arange(WALL[0], WALL[1] - 1e-9, PATCH):
        v = y[(x >= x0) & (x < x0 + PATCH)]
        if v.size >= 10:
            t.append(np.subtract(*np.percentile(v, [95, 5])))
    return float(np.median(t))


def scans(drift, lag, jitter=0.5):
    """Wall points (x, offset in cm) with drift, pose-time and per-scan errors (cm)."""
    off = drift * DRIFT + lag * TURN + jitter * JIT + 0.4 * TILT * (X - X.mean())
    return X.ravel(), (off + NOISE).ravel()


def solve(fun, target, lo=0.0, hi=40.0):
    return brentq(lambda s: thick(*fun(s)) - target, lo, hi)


def calibrate():
    # (a) and (b): one drift, one pose-time error; the correction leaves 4/50 of the lag,
    # bundle adjustment shrinks the drift
    lag, drift = 3.0, 3.0
    for _ in range(20):
        drift = solve(lambda d: scans(d, lag * LAG_LEFT), MEASURED["glim_corrected"])
        lag = solve(lambda g: scans(drift, g), MEASURED["glim_output"])
    drift_ba = solve(lambda d: scans(d, lag * LAG_LEFT), MEASURED["reference"])
    a0, a1 = scans(drift, lag), scans(drift, lag * LAG_LEFT)
    b1 = scans(drift_ba, lag * LAG_LEFT)
    # (c): the union keeps every point of the reference-placed scans; the TSDF keeps one
    # surface per cell, which still carries the spread no average removes
    c0 = scans(solve(lambda d: scans(d, lag * LAG_LEFT), MEASURED["union"]), lag * LAG_LEFT)
    tsdf = lambda s: (XT, s * (0.85 * TS + 0.45 * TSMOOTH))      # noqa: E731
    c1 = tsdf(solve(tsdf, MEASURED["tsdf"]))
    return (a0, a1), (a1, b1), (c0, c1)


def panel(ax, w, color, label, value, size=1.4):
    x, y = w
    bars = []
    for x0 in np.arange(WALL[0], WALL[1] - 1e-9, PATCH):
        v = y[(x >= x0) & (x < x0 + PATCH)]
        if v.size >= 10:
            bars.append((x0, *np.percentile(v, [5, 95])))
    for x0, lo, hi in bars:
        ax.add_patch(plt.Rectangle((x0 + 0.01, lo), PATCH - 0.02, hi - lo, fc=color, ec="none",
                                   alpha=0.16, zorder=1))
    ax.scatter(x, y, s=size, lw=0, color=color, alpha=0.5, zorder=2, rasterized=True)
    for s in (-SENSOR / 2, SENSOR / 2):
        ax.axhline(s, color=INK2, lw=0.6, ls=(0, (2, 2)), zorder=3)
    ax.text(0.02, 0.96, label, transform=ax.transAxes, ha="left", va="top", color=INK, zorder=4,
            bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.5))
    ax.text(0.98, 0.04, f"{value:.1f} cm", transform=ax.transAxes, ha="right", va="bottom",
            color=color, fontweight="bold", zorder=4, bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.5))
    ax.set_xlim(*WALL)
    ax.set_ylim(-16, 16)
    ax.set_yticks([-15, -10, -5, 0, 5, 10, 15])
    ax.grid(axis="y", color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    return thick(x, y)


def main():
    (a0, a1), (b0, b1), (c0, c1) = calibrate()
    M = MEASURED
    cols = [("(a) Pose-time correction", (a0, GREY, "GLIM as output: poses 50 ms late", M["glim_output"]),
             (a1, BLUE, "corrected: 4 ms left", M["glim_corrected"])),
            ("(b) Plane bundle adjustment", (b0, BLUE, "GLIM corrected", M["glim_corrected"]),
             (b1, ORANGE, "after adjustment (reference)", M["reference"])),
            ("(c) Fusion (map, reference poses)", (c0, GREY, "union of points", M["union"]),
             (c1, ORANGE, "TSDF surface", M["tsdf"]))]
    fig, axs = plt.subplots(2, 3, figsize=(7.16, 3.9), sharex=True, sharey=True)
    got = []
    for j, (title, top, bot) in enumerate(cols):
        axs[0, j].set_title(title, loc="left", fontsize=FS + 1, fontweight="bold", color=INK, pad=6)
        got.append((title, panel(axs[0, j], *top), panel(axs[1, j], *bot, size=3.0 if j == 2 else 1.4)))
        axs[1, j].set_xlabel("along the wall (m)")
    for i, row in enumerate(("before", "after")):
        axs[i, 0].set_ylabel(f"{row}\noffset from wall (cm)")
    fig.text(0.5, 0.005, "Bold: median wall thickness (p95$-$p5 in 0.25 m patches) measured on pass 20261015. "
             "Shaded: each patch's p5$-$p95.\nDashed: the sensor alone, 3.0 cm. "
             "Points are simulated at the measured spreads.",
             ha="center", va="bottom", color=INK2, fontsize=FS - 0.5, linespacing=1.4)
    fig.tight_layout(h_pad=0.8, w_pad=0.6, rect=(0, 0.06, 1, 1))
    fig.savefig(OUT + ".pdf", dpi=300)
    fig.savefig(OUT + ".png", dpi=300)
    for title, t0, t1 in got:
        print(f"{title}: simulated {t0:.1f} -> {t1:.1f} cm")
    print(OUT + ".pdf", OUT + ".png")


if __name__ == "__main__":
    main()
