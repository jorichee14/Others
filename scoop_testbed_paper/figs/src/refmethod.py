#!/usr/bin/env python3
"""How each mapping step works, as schematic drawings: figs/fig_refmethod.pdf
(and .png, the four panels together) and figs/fig_refmethod_{a,b,c,d}.png
(one per step, for the slides).

    python3 figs/src/refmethod.py

  (a) pose-time correction: GLIM stamps a pose with the start of its sweep,
      while the pose is the robot about half a sweep later; a scan placed
      with the pose stamped at its time lands off the walls when turning.
  (b) plane bundle adjustment: keyframe poses T_k, the points each keyframe
      sees of one plane, their distances to the plane (solved in closed
      form), all poses moved together to make every plane thin.
  (c) TSDF fusion: the rays of two scans on a grid of cells, the signed
      distance each cell within tau of a hit receives, averaged over the
      rays; the surface where it crosses zero (computed, not drawn by hand).
  (d) wall thickness: a 0.25 m patch, its plane, the offsets of the points
      along its normal; the thickness is their 5-95 % spread.
Drawn to explain, not to scale (the turn in (a) and the offsets in (b) are
exaggerated).
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                         # noqa: E402
from matplotlib.colors import LinearSegmentedColormap                   # noqa: E402
from matplotlib.patches import Polygon, Circle, Rectangle               # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "fig_refmethod")

INK, INK2, GRID = "#0b0b0b", "#52514e", "#d4d3cf"
GREY, BLUE, ORANGE = "#8a8984", "#2a78d6", "#eb6834"     # GLIM, corrected / planes, result
PLANE = "#9cc0ec"                                       # a plane, as in the classic BA drawing
FS = 7.0
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": FS, "mathtext.fontset": "dejavusans"})
RNG = np.random.default_rng(3)


def P(p, az=-30.0, el=20.0, f=14.0):
    """Perspective view of 3-D points (x right, y away, z up) -> 2-D."""
    p = np.asarray(p, float)
    x, y, z = p[..., 0], p[..., 1], p[..., 2]
    a, e = np.radians(az), np.radians(el)
    xr, yr = x * np.cos(a) - y * np.sin(a), x * np.sin(a) + y * np.cos(a)
    s = f / (f + yr * np.cos(e) - z * np.sin(e))
    return np.stack([xr * s, (z * np.cos(e) + yr * np.sin(e)) * s], -1)


def arrow(ax, p, q, color=INK, lw=0.9, style="-|>", ms=7, z=4, rad=0.0, ls="-"):
    ax.annotate("", xy=q, xytext=p, zorder=z, arrowprops=dict(
        arrowstyle=style, color=color, lw=lw, shrinkA=0, shrinkB=0, mutation_scale=ms,
        linestyle=ls, connectionstyle=f"arc3,rad={rad}"))


def callout(ax, text, xy, xytext, rad=0.3, ha="center", color=INK):
    ax.annotate(text, xy=xy, xytext=xytext, ha=ha, va="center", color=color, zorder=6,
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=0.7, mutation_scale=6,
                                shrinkA=2, shrinkB=1, connectionstyle=f"arc3,rad={rad}"))


def sensor(ax, xy, label, color=INK, hollow=False, dx=0.0, dy=-0.25, r=0.075, ha="center"):
    ax.add_patch(Circle(xy, r, fc="white" if hollow else color, ec=color, lw=1.0, zorder=5))
    if label:
        ax.text(xy[0] + dx, xy[1] + dy, label, ha=ha, va="top", color=color, zorder=6)


def frame(ax, xlim, ylim):
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.axis("off")


def rot(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s], [s, c]])


# ------------------------------------------------------------------ (a) pose time
def pose_time(ax):
    """Top view of a left turn; below it, one sweep on a time axis."""
    C, R = np.array([0.0, 2.2]), 2.2
    on = lambda phi: C + R * np.array([np.cos(phi), np.sin(phi)])       # noqa: E731
    ph = np.radians(np.linspace(-125, -8, 100))
    path = C + R * np.stack([np.cos(ph), np.sin(ph)], 1)
    ax.plot(*path.T, color=INK2, lw=0.8, ls=(0, (4, 3)), zorder=1)
    arrow(ax, path[-6], path[-1], color=INK2, lw=0.8)
    ax.text(*(path[-1] + [0.12, 0]), "path", color=INK2, ha="left", va="center")

    # the wall, its far side hatched
    xw = 3.3
    ax.plot([xw, xw], [-0.7, 2.5], color=INK, lw=1.6, zorder=2)
    ax.add_patch(Rectangle((xw, -0.7), 0.16, 3.2, fc="none", ec=GREY, lw=0, hatch="////", zorder=1))
    ax.text(xw + 0.25, 2.5, "wall", ha="left", va="top", color=INK)

    pt, pa = np.radians(-82), np.radians(-70)              # the scan's time; 50 ms later (exaggerated)
    t_pos, a_pos = on(pt), on(pa)
    t_h, a_h = pt + np.pi / 2, pa + np.pi / 2
    hits = np.array([[xw, y] for y in (-0.35, 0.25, 0.85, 1.45)])
    local = (hits - t_pos) @ rot(t_h)                      # the scan, in the sensor's frame
    ghost = local @ rot(a_h).T + a_pos                     # ... placed with the later pose

    for h in hits:
        ax.plot(*np.stack([t_pos, h]).T, color=BLUE, lw=0.7, zorder=2)
    for g in ghost:
        ax.plot(*np.stack([a_pos, g]).T, color=GREY, lw=0.7, ls=(0, (2, 2)), zorder=2)
    ax.scatter(*hits.T, s=14, color=BLUE, zorder=4, lw=0)
    ax.scatter(*ghost.T, s=14, color="white", edgecolor=GREY, lw=1.0, zorder=4)
    for h, g in zip(hits, ghost):
        arrow(ax, h, g, color=GREY, lw=0.6, ms=5, z=3)
    for pos, h, col, hol in ((t_pos, t_h, BLUE, False), (a_pos, a_h, GREY, True)):
        sensor(ax, pos, None, color=col, hollow=hol)
        arrow(ax, pos, pos + 0.55 * np.array([np.cos(h), np.sin(h)]), color=col, lw=1.4, ms=8, z=6)
    ax.text(*(t_pos + [-0.15, -0.12]), "robot at the\nscan's time $t$", ha="right", va="top", color=BLUE)
    callout(ax, "GLIM's pose\nstamped $t$", a_pos + [-0.05, 0.08], a_pos + [-0.6, 1.05], rad=0.3,
            color=INK2)
    callout(ax, "placed with it:\noff the wall", ghost[2] + [0.04, 0.0], (4.75, 1.85), rad=0.3)
    callout(ax, "corrected:\non the wall", hits[0] + [0.0, -0.02], hits[0] + [-0.6, -0.75], rad=-0.3,
            color=BLUE)

    # one sweep on a time axis
    y0, x0, x1 = -2.05, -0.9, 3.1
    ax.plot([x0, x1], [y0, y0], color=INK, lw=1.0)
    for x, lab in ((x0, "0"), ((x0 + x1) / 2, "50"), (x1, "100 ms")):  # noqa: B007
        ax.plot([x, x], [y0 - 0.06, y0 + 0.06], color=INK, lw=0.8)
        ax.text(x, y0 - 0.12, lab, ha="center", va="top", color=INK2)
    ax.scatter(np.linspace(x0 + 0.1, x1 - 0.1, 14), np.full(14, y0 + 0.17), s=4, color=INK2, lw=0)
    ax.text((x0 + x1) / 2, y0 - 0.48, "one sweep", ha="center", va="top", color=INK2)
    xm = x0 + 0.46 * (x1 - x0)
    ax.annotate("", xy=(xm, y0 + 0.36), xytext=(x0, y0 + 0.36), arrowprops=dict(
        arrowstyle="-|>", color=BLUE, lw=1.0, mutation_scale=7, shrinkA=0, shrinkB=0,
        connectionstyle="arc3,rad=-0.35"))
    ax.text(x0, y0 - 0.48, "GLIM's stamp:\nsweep start", ha="center", va="top", color=INK2)
    ax.text((x0 + xm) / 2, y0 + 0.78, "+46 ms on export", ha="center", va="bottom", color=BLUE,
            fontweight="bold")
    frame(ax, (-1.5, 5.4), (-3.0, 2.7))


# ------------------------------------------------------------------ (b) plane BA
def plane_ba(ax):
    """Three keyframes above one plane (a floor, so its normal is up); each
    keyframe's points of it, and their distances to it."""
    pr = lambda p: P(p, az=-18, el=36)                    # noqa: E731
    floor = [[-1.9, 0.3, 0], [1.9, 0.3, 0], [1.9, 3.3, 0], [-1.9, 3.3, 0]]
    ax.add_patch(Polygon(pr(floor), closed=True, fc=PLANE, ec=INK, lw=0.9, alpha=0.6, zorder=2))
    ax.text(*(pr([1.9, 0.3, 0]) + [0.05, -0.08]), "plane $f$", ha="left", va="top", color=INK,
            zorder=6)

    poses = [np.array([-2.7, 0.1, 1.7]), np.array([0.0, -0.6, 3.1]), np.array([2.7, 0.4, 1.7])]
    off = [0.42, -0.08, -0.36]                           # each keyframe's pose error, along the normal
    xy = [[(-1.45, 0.9), (-0.7, 2.5), (0.05, 1.45)],
          [(-1.0, 2.0), (0.35, 0.8), (1.0, 2.7)],
          [(-0.1, 2.9), (0.85, 1.6), (1.45, 0.75)]]
    seg = None
    for k, (T, e, pts) in enumerate(zip(poses, off, xy)):
        s2 = pr(T)
        for j, (x, y) in enumerate(pts):
            q = np.array([x, y, e + RNG.normal(0, 0.03)])
            z = 3 if q[2] > 0 else 1
            ax.plot(*np.stack([s2, pr(q)]).T, color=INK2, lw=0.55, zorder=z)
            ax.plot(*np.stack([pr(q), pr([x, y, 0])]).T, color=ORANGE, lw=1.8, zorder=z + 1)
            ax.scatter(*pr(q), s=13, color=INK, lw=0, zorder=z + 1)
            if k == 0 and j == 0:
                seg = (pr(q) + pr([x, y, 0])) / 2
        sensor(ax, s2, None, r=0.08)
        ax.text(s2[0], s2[1] + 0.3, f"$T_{k + 1}$", ha="center", va="bottom", color=INK, fontsize=FS + 1)
        for d in ((0.32, 0), (0, 0.32)):                 # the pose moves
            d = np.array(d)
            arrow(ax, s2 - d, s2 + d, style="<|-|>", lw=0.8, ms=5)

    c = np.array([0.0, 1.8, 0.0])
    arrow(ax, pr(c), pr(c + [0, 0, 0.95]), color=BLUE, lw=1.4, z=5)
    ax.text(*(pr(c + [0, 0, 0.95]) + [0.08, 0]), r"$\mathbf{n}$", color=BLUE, ha="left", va="center",
            fontsize=FS + 1.5)
    callout(ax, "distance to\nthe plane", seg, seg + [-0.9, -1.0], rad=-0.3)

    x0 = 3.5
    ax.text(x0, 2.35, "keyframe: 1 s of scans\nplane: a flat 0.5 m cube\nseen by $\\geq$3 keyframes",
            ha="left", va="top", color=INK2, linespacing=1.4)
    ax.text(x0, 0.75, "plane in closed form ($\\mathbf{n}$, $d$);\nall poses $T_k$ moved together:",
            ha="left", va="top", color=INK, linespacing=1.4)
    ax.text(x0, -0.55, r"$\min_{T}\ \sum_f\ \lambda_{\min}(\mathrm{Cov}_f)$", ha="left", va="center",
            color=INK, fontsize=FS + 1.5)
    ax.text(x0, -1.15, "every plane's thickness$^2$", ha="left", va="center", color=INK2)
    frame(ax, (-3.3, 7.0), (-1.6, 3.2))


# ------------------------------------------------------------------ (c) TSDF
def tsdf(ax):
    """Two scans' rays on 2 cm cells; D computed, surface where D = 0."""
    nx, ny, y0, tau = 16, 9, 3, 4.0                       # cells; tau = 8 cm = 4 cells
    surf = lambda x: 7.0 + 0.14 * (x - 8)                  # noqa: E731
    origins = [np.array([-1.5, -0.6]), np.array([17.5, -0.8])]
    targets = [np.arange(0.4, 16, 1.4), np.arange(1.1, 16, 1.4)]
    D, W = np.zeros((nx, ny)), np.zeros((nx, ny))
    rays = []
    for o, xs in zip(origins, targets):
        for xt in xs:
            h = np.array([xt, surf(xt)])
            u = (h - o) / np.linalg.norm(h - o)
            for _ in range(20):                            # where the ray meets the surface
                h = o + u * ((surf(h[0]) - o[1]) / u[1])
            h = h + u * RNG.normal(0, 0.35)                # range noise
            r = np.linalg.norm(h - o)
            rays.append((o, h, u))
            seen = set()
            for s in np.arange(-tau, tau + 1e-9, 0.05):
                q = h + u * s
                i, j = int(np.floor(q[0])), int(np.floor(q[1])) - y0
                if 0 <= i < nx and 0 <= j < ny and (i, j) not in seen:
                    seen.add((i, j))
                    d = r - np.linalg.norm(np.array([i + 0.5, j + y0 + 0.5]) - o)
                    if abs(d) <= tau:
                        D[i, j] += d
                        W[i, j] += 1
    D = np.where(W > 0, D / np.maximum(W, 1), np.nan)

    cmap = LinearSegmentedColormap.from_list("sd", [ORANGE, "#ffffff", BLUE])
    ax.pcolormesh(np.arange(nx + 1), np.arange(ny + 1) + y0, D.T, cmap=cmap, vmin=-tau, vmax=tau,
                  alpha=0.55, zorder=0, edgecolors="none")
    for i in range(nx + 1):
        ax.plot([i, i], [y0, y0 + ny], color=GRID, lw=0.5, zorder=1)
    for j in range(ny + 1):
        ax.plot([0, nx], [y0 + j, y0 + j], color=GRID, lw=0.5, zorder=1)

    for o, h, u in rays:
        ax.plot(*np.stack([o, h]).T, color=INK2, lw=0.5, zorder=2)
        ax.plot(*np.stack([h, h + u * tau]).T, color=INK2, lw=0.5, ls=(0, (1.5, 1.5)), zorder=2)
        ax.scatter(*h, s=10, color=INK, lw=0, zorder=4)
    sensor(ax, origins[0], None, r=0.35)
    sensor(ax, origins[1], None, r=0.35)
    ax.text(origins[0][0] - 0.6, origins[0][1], "$\\mathbf{o}_1$", ha="right", va="center", fontsize=FS + 1)
    ax.text(origins[1][0] + 0.6, origins[1][1], "$\\mathbf{o}_2$", ha="left", va="center", fontsize=FS + 1)

    # zero crossings between vertically adjacent cells (the surface)
    zc = []
    for i in range(nx):
        for j in range(ny - 1):
            a, b = D[i, j], D[i, j + 1]
            if np.isfinite(a) and np.isfinite(b) and a * b < 0 and W[i, j] >= 2 and W[i, j + 1] >= 2:
                zc.append((i + 0.5, y0 + j + 0.5 + a / (a - b)))
    zc = np.array(sorted(zc))
    ax.plot(*zc.T, color=ORANGE, lw=2.2, zorder=5)
    ax.scatter(*zc.T, s=11, color=ORANGE, lw=0, zorder=5)

    # +-tau along one ray
    o, h, u = rays[2]
    nrm = np.array([-u[1], u[0]]) * 1.0
    a, b = h - u * tau + nrm, h + u * tau + nrm
    arrow(ax, a, b, style="<|-|>", lw=0.9, ms=5, z=6)
    ax.text(-0.5, b[1] - 0.6, r"$\pm\tau$" + "\n8 cm", ha="right", va="center", color=INK, zorder=6)
    xl = nx + 0.5
    ax.text(xl, y0 + ny, "cells 2 cm", ha="left", va="top", color=INK2)
    ax.text(xl, y0 + ny - 2.3, "$D$ < 0\nbehind", ha="left", va="center", color="#b4501f")
    ax.text(xl, y0 + 4.0, "surface:\n$D$ = 0", ha="left", va="center", color=ORANGE, fontweight="bold")
    ax.text(xl, y0 + 1.3, "$D$ > 0\nin front", ha="left", va="center", color=BLUE)
    ax.text(8.0, -2.0, "each cell within $\\tau$ of a hit: $d = r - \\|\\mathbf{c}_v - \\mathbf{o}\\|$\n"
            "$D(v)$ = mean of $d$ over all rays (at least 3)",
            ha="center", va="top", color=INK, linespacing=1.5)
    frame(ax, (-3.6, 21.6), (-4.6, 12.3))


# ------------------------------------------------------------------ (d) thickness
def thickness(ax):
    """A patch seen from above (its normal up), its points' offsets along the
    normal and their histogram on the same vertical scale."""
    q = lambda u, w, n: P([u, w, n], az=-25, el=24, f=8.0)   # noqa: E731
    corners = np.array([q(0, 0, 0), q(1, 0, 0), q(1, 1, 0), q(0, 1, 0)])
    sig = 0.11
    u, w = RNG.uniform(0.05, 0.95, 60), RNG.uniform(0.05, 0.95, 60)
    n = RNG.normal(0, sig, 60)
    for uu, ww, nn in zip(u, w, n):
        if nn <= 0:
            ax.plot(*np.stack([q(uu, ww, nn), q(uu, ww, 0)]).T, color=INK2, lw=0.5, zorder=1)
            ax.scatter(*q(uu, ww, nn), s=7, color=INK2, lw=0, zorder=1)
    ax.add_patch(Polygon(corners, closed=True, fc=PLANE, ec=INK, lw=0.9, alpha=0.6, zorder=2))
    for uu, ww, nn in zip(u, w, n):
        if nn > 0:
            ax.plot(*np.stack([q(uu, ww, nn), q(uu, ww, 0)]).T, color=INK, lw=0.5, zorder=3)
            ax.scatter(*q(uu, ww, nn), s=8, color=INK, lw=0, zorder=4)
    c0, c1 = q(0.5, 0.5, 0), q(0.5, 0.5, 0.5)
    arrow(ax, c0, c1, color=BLUE, lw=1.4, z=5)
    ax.text(c1[0] + 0.04, c1[1], r"$\mathbf{n}$", color=BLUE, ha="left", va="center", fontsize=FS + 1.5)
    e0, e1 = q(0, 0, 0) + [0, -0.1], q(1, 0, 0) + [0, -0.1]
    arrow(ax, e0, e1, style="<|-|>", lw=0.7, ms=5)
    ax.text(*((e0 + e1) / 2 + [0, -0.06]), "0.25 m patch", ha="center", va="top", color=INK)
    ax.text(0.45, 0.98, "fit a plane to the patch's points", ha="center", va="bottom", color=INK)

    # histogram of the offsets along n, on the drawing's vertical scale
    sc = (c1[1] - c0[1]) / 0.5
    xh, yc = 1.45, c0[1]
    edges = np.linspace(-3 * sig, 3 * sig, 13)
    cnt, _ = np.histogram(n, edges)
    for k, e_lo, e_hi in zip(cnt, edges[:-1], edges[1:]):
        ax.add_patch(Rectangle((xh, yc + e_lo * sc), k * 0.04, (e_hi - e_lo) * sc * 0.9,
                               fc=GREY, ec="none", alpha=0.7, zorder=2))
    ax.plot([xh, xh], [yc - 3.2 * sig * sc, yc + 3.2 * sig * sc], color=INK, lw=0.8)
    ax.text(xh, yc + 3.2 * sig * sc + 0.03, "offset along $\\mathbf{n}$", ha="center", va="bottom",
            color=INK)
    p5, p95 = np.percentile(n, [5, 95])
    for p, lab in ((p5, "p5"), (p95, "p95")):
        ax.plot([xh - 0.04, xh + 0.55], [yc + p * sc] * 2, color=ORANGE, lw=1.0, ls=(0, (3, 2)))
        ax.text(xh - 0.07, yc + p * sc, lab, ha="right", va="center", color=ORANGE)
    xb = xh + 0.62
    arrow(ax, (xb, yc + p5 * sc), (xb, yc + p95 * sc), color=ORANGE, style="<|-|>", lw=1.0, ms=5)
    ax.text(xb + 0.07, yc, "thickness\n= p95 $-$ p5\n(3.29$\\sigma$)", ha="left", va="center",
            color=ORANGE, fontweight="bold")
    ax.text(1.05, -0.5, "points as placed from one scan, 1 s or the whole run;\n"
            "each scan against its own plane: the sensor alone",
            ha="center", va="top", color=INK2, linespacing=1.4)
    frame(ax, (-0.3, 2.65), (-0.85, 1.2))


PANELS = [("a", "(a) Pose-time correction", pose_time),
          ("b", "(b) Plane bundle adjustment", plane_ba),
          ("c", "(c) TSDF fusion", tsdf),
          ("d", "(d) Wall thickness", thickness)]


def main():
    fig, axs = plt.subplots(2, 2, figsize=(7.16, 5.5))
    for ax, (_, title, draw) in zip(axs.ravel(), PANELS):
        draw(ax)
        ax.set_title(title, loc="left", fontsize=FS + 1.5, fontweight="bold", color=INK)
    fig.tight_layout(h_pad=1.2, w_pad=1.0)
    fig.savefig(OUT + ".pdf")
    fig.savefig(OUT + ".png", dpi=300)
    plt.close(fig)
    for key, _, draw in PANELS:
        fig, ax = plt.subplots(figsize=(3.6, 2.95))
        draw(ax)
        fig.tight_layout(pad=0.2)
        fig.savefig(f"{OUT}_{key}.png", dpi=400)
        plt.close(fig)
    print(OUT + ".pdf", OUT + ".png", *(f"{OUT}_{k}.png" for k, _, _ in PANELS))


if __name__ == "__main__":
    main()
