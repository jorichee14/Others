#!/usr/bin/env python3
"""The collaborative-perception gym as a simple top-to-bottom flow, in the
style of the "Dynamic Digital Twin" slide: real pass and Isaac Sim side by
side, then one column through frames, the channel knob, the detectors and
the score. Dashed = not built yet.

    python3 slides_media/gym_flow.py slides_media/gym_flow.png
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

plt.rcParams["font.family"] = "Inter"

INK, SOFT, LINK = "#1B2430", "#5A6472", "#7A5BB5"
KIND = {  # fill, shadow, edge
    "real":  ("#E9EAEC", "#DADCE0", "#B9BEC6"),
    "sim":   ("#FBEFD9", "#F1E1C2", "#E2C58E"),
    "label": ("#DFF0E3", "#CDE5D3", "#9CCBA8"),
    "data":  ("#DDE8F6", "#CCDAEE", "#9DB7DD"),
    "radio": ("#ECE6F6", "#DDD4EE", "#B7A6DA"),
}

W, H = 1100, 890
fig = plt.figure(figsize=(W / 100, H / 100), dpi=150)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis("off")


def block(x, y, w, h, title, sub, kind, built=True):
    fill, shadow, edge = KIND[kind]
    if built:
        ax.add_patch(FancyBboxPatch((x + 6, y + 6), w, h, boxstyle="round,pad=0,rounding_size=12",
                                    fc=shadow, ec="none"))
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=12",
                                fc=fill if built else "white", ec=edge if built else "#7C8594", lw=1.4,
                                linestyle="-" if built else (0, (4, 3))))
    ax.text(x + w / 2, y + h / 2 - 17, title, ha="center", va="center", fontsize=16, weight="bold", color=INK)
    ax.text(x + w / 2, y + h / 2 + 17, sub, ha="center", va="center", fontsize=12, color=SOFT)


def arrow(pts, col=INK, ls="-"):
    xs, ys = zip(*pts)
    ax.plot(xs, ys, color=col, lw=1.6, linestyle=ls, zorder=3)
    ax.annotate("", xy=pts[-1], xytext=pts[-2],
                arrowprops=dict(arrowstyle="-|>,head_length=0.65,head_width=0.34", color=col, lw=1.6,
                                shrinkA=0, shrinkB=0), zorder=4)


X0, CW, RH, GAP = 150, 860, 112, 58
HW = (CW - 40) / 2
ys = [30 + i * (RH + GAP) for i in range(5)]
mid = X0 + CW / 2

# row 1: two sources, real feeds sim
block(X0, ys[0], HW, RH, "Real coop pass", "2 robots + infra · measured link", "real")
block(X0 + HW + 40, ys[0], HW, RH, "Isaac Sim scenario", "same robots and paths + people", "sim")
arrow([(X0 + HW, ys[0] + RH / 2), (X0 + HW + 40, ys[0] + RH / 2)])

# row 2: both become labelled frames
block(X0, ys[1], CW, RH, "Labelled OPV2V frames", "objects boxed once in the map · synced 10 Hz frames", "data")
arrow([(X0 + HW / 2, ys[0] + RH), (X0 + HW / 2, ys[1])])
arrow([(X0 + HW * 1.5 + 40, ys[0] + RH), (X0 + HW * 1.5 + 40, ys[1])])

# rows 3-4: the channel sits between collaborators and the detector
block(X0, ys[2], CW, RH, "Channel knob", "ideal · synthetic · measured link", "radio")
block(X0, ys[3], CW, RH, "Collaborative detectors", "7 pretrained models: early, intermediate, late fusion", "data")
arrow([(mid, ys[1] + RH), (mid, ys[2])])
arrow([(mid, ys[2] + RH), (mid, ys[3])])

# row 5: score, and the next step beside it
block(X0, ys[4], HW, RH, "Score", "detection AP · collaboration gain", "label")
block(X0 + HW + 40, ys[4], HW, RH, "Train under the link", "next: logged link in the loop", "real", built=False)
arrow([(X0 + HW / 2, ys[3] + RH), (X0 + HW / 2, ys[4])])
arrow([(X0 + HW * 1.5 + 40, ys[3] + RH), (X0 + HW * 1.5 + 40, ys[4])], col=SOFT, ls=(0, (4, 3)))

# the measured link goes straight to the knob
lx = 95
arrow([(X0, ys[0] + RH / 2), (lx, ys[0] + RH / 2), (lx, ys[2] + RH / 2), (X0, ys[2] + RH / 2)], col=LINK)
ax.text(lx - 14, (ys[0] + ys[2] + RH) / 2, "measured link trace", rotation=90, ha="center", va="center",
        fontsize=12, color=LINK)

# legend
ly = 862
ax.add_patch(FancyBboxPatch((X0, ly - 9), 30, 18, boxstyle="round,pad=0,rounding_size=4", fc="white",
                            ec="#7C8594", lw=1.1, linestyle=(0, (3, 2))))
ax.text(X0 + 40, ly, "not built yet", ha="left", va="center", fontsize=12, color=INK)
ax.text(X0 + CW, ly, "831 channel conditions tested on OPV2V", ha="right", va="center", fontsize=12,
        style="italic", color=SOFT)

fig.savefig(sys.argv[1], facecolor="white")
print(sys.argv[1])
