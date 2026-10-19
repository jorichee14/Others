#!/usr/bin/env python3
"""Fig. 3, the reference pipeline: figs/fig_refpipeline.pdf (and .png).

    python3 figs/src/refpipeline.py

Top lane: the mapping session, once per site (GLIM with its pose-time
correction -> plane bundle adjustment -> TSDF fusion -> the frozen
reference; boards and infrastructure cameras posed in it; the checks).
Bottom lane: every sequence, registered into the frozen map per agent,
validated on held-out boards, exported in the site frame. Stage numbers are
scoop_pipeline/map_stages/.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                         # noqa: E402
from matplotlib.patches import FancyBboxPatch, Polygon                  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "fig_refpipeline")

INK, MUTED = "#14213D", "#5A6475"
LANE, LANE_EDGE = "#EEF1F5", "#A7B0C0"
PROC, OUTF, OUTE = "#FFFFFF", "#FBE6D0", "#D9822B"
FS = 6.6

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": FS})
W, H = 100.0, 40.0
fig = plt.figure(figsize=(7.16, 7.16 * H / W))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")


def lane(y0, y1, title):
    ax.add_patch(FancyBboxPatch((0.6, y0), W - 1.2, y1 - y0, boxstyle="round,pad=0,rounding_size=0.8",
                                fc=LANE, ec=LANE_EDGE, lw=0.8))
    ax.text(1.8, y1 - 1.6, title, ha="left", va="center", fontsize=FS + 0.8,
            fontweight="bold", color=INK)


def proc(x0, x1, yc, h, text, bold=False):
    ax.add_patch(FancyBboxPatch((x0, yc - h / 2), x1 - x0, h, boxstyle="round,pad=0,rounding_size=0.6",
                                fc=PROC, ec=INK, lw=0.9))
    ax.text((x0 + x1) / 2, yc, text, ha="center", va="center", color=INK,
            fontweight="bold" if bold else "normal", linespacing=1.25)
    return (x0, x1, yc, h)


def out(x0, x1, yc, h, text, bold=False):
    s = 1.2
    ax.add_patch(Polygon([(x0 + s, yc - h / 2), (x1, yc - h / 2), (x1 - s, yc + h / 2),
                          (x0, yc + h / 2)], closed=True, fc=OUTF, ec=OUTE, lw=1.0))
    ax.text((x0 + x1) / 2, yc, text, ha="center", va="center", color=INK,
            fontweight="bold" if bold else "normal", linespacing=1.25)
    return (x0, x1, yc, h)


def arrow(p, q, dashed=False, color=INK):
    ax.annotate("", xy=q, xytext=p, arrowprops=dict(
        arrowstyle="-|>", color=color, lw=0.9, shrinkA=0, shrinkB=0,
        mutation_scale=7, linestyle=(0, (3, 2)) if dashed else "-"))


def right(a, b):
    """Arrow from box a's right edge to box b's left edge, at a's centre height."""
    arrow((a[1], a[2]), (b[0], a[2]))


# ---------------------------------------------------------------- mapping session
lane(19.2, 39.4, "Mapping session (once per site)")
yc = 30.6
m1 = proc(2.0, 11.5, yc, 6.0, "Mapping\nrecording")
m2 = proc(13.5, 30.5, yc, 6.0, "LiDAR-inertial SLAM\n+ pose-time correction")
m3 = proc(32.5, 44.5, yc, 6.0, "Plane bundle\nadjustment")
m4 = proc(46.5, 56.5, yc, 6.0, "TSDF fusion")
m5 = out(58.5, 74.5, yc, 7.0, "Frozen reference\nmap + trajectory", bold=True)
b1 = proc(80.0, 98.4, 34.9, 4.8, "Board poses")
b2 = proc(80.0, 98.4, 24.9, 4.8, "Infrastructure\ncamera poses")
for a, b in ((m1, m2), (m2, m3), (m3, m4), (m4, m5)):
    right(a, b)
arrow((m5[1] - 0.5, yc + 1.0), (b1[0], b1[2]))
arrow((m5[1] - 0.5, yc - 1.0), (b2[0], b2[2]))
arrow((84.0, b1[2] - b1[3] / 2), (84.0, b2[2] + b2[3] / 2))
ax.text(85.0, (b1[2] + b2[2]) / 2, "through the boards", ha="left", va="center",
        fontsize=FS - 0.8, color=MUTED)
ax.add_patch(FancyBboxPatch((13.5, 20.6), 43.0, 3.0, boxstyle="round,pad=0,rounding_size=0.5",
                            fc="none", ec=MUTED, lw=0.7, linestyle=(0, (2, 2))))
ax.text(35.0, 22.1, "checks: wall thickness · loop closure · pose time · control distances",
        ha="center", va="center", fontsize=FS - 0.8, color=MUTED, style="italic")

# ---------------------------------------------------------------- every sequence
lane(0.6, 17.6, "Every sequence")
yc = 7.4
s1 = proc(2.0, 11.5, yc, 6.0, "Sequence\nrecording")
s2 = proc(13.5, 30.5, yc, 6.0, "LiDAR-inertial SLAM\n+ pose-time check")
s3 = proc(32.5, 56.5, yc, 8.4, "Per-agent registration\nLiDAR: scan-to-map ICP\nRGB-D: odometry + ICP + boards")
s4 = proc(58.5, 72.0, yc, 8.4, "Independent\nvalidation\n(held-out boards)")
s5 = out(74.0, 87.5, yc, 7.0, "Reference\nposes (site)", bold=True)
s6 = out(89.0, 99.0, yc, 7.0, "Labels and\ngeometry")
for a, b in ((s1, s2), (s2, s3), (s3, s4), (s4, s5), (s5, s6)):
    right(a, b)

# the frozen reference feeds every sequence
xf, ytop, ybus = 66.5, m5[2] - m5[3] / 2, 14.2
ax.plot([xf, xf], [ytop, ybus], color=OUTE, lw=1.0, linestyle=(0, (3, 2)))
ax.plot([44.5, xf], [ybus, ybus], color=OUTE, lw=1.0, linestyle=(0, (3, 2)))
arrow((44.5, ybus), (44.5, s3[2] + s3[3] / 2), dashed=True, color=OUTE)
arrow((65.25, ybus), (65.25, s4[2] + s4[3] / 2), dashed=True, color=OUTE)
ax.text(55.0, ybus + 0.6, "frozen map, boards", ha="center", va="bottom", fontsize=FS - 0.6,
        color="#9A5A14")

fig.savefig(OUT + ".pdf")
fig.savefig(OUT + ".png", dpi=300)
print(OUT + ".pdf", OUT + ".png")
