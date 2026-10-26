#!/usr/bin/env python3
"""The collaborative-perception gym as one architecture and data-flow figure,
drawn in the style of the "Dynamic Digital Twin" slide: tinted panels, white
step boxes, dashed boxes for what is not built yet, a red divider for the
interface every source crosses (OPV2V frames), and a legend.

    python3 slides_media/gym_architecture.py slides_media/gym_architecture.png
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

plt.rcParams["font.family"] = "Inter"

INK, SOFT, RED = "#1B2430", "#5A6472", "#D93A32"
# panel kinds: (fill, shadow, edge)
KIND = {
    "real":   ("#E9EAEC", "#DADCE0", "#B9BEC6"),
    "sim":    ("#FBEFD9", "#F1E1C2", "#E2C58E"),
    "label":  ("#DFF0E3", "#CDE5D3", "#9CCBA8"),
    "data":   ("#DDE8F6", "#CCDAEE", "#9DB7DD"),
    "radio":  ("#ECE6F6", "#DDD4EE", "#B7A6DA"),
}

W, H = 2000, 960
fig = plt.figure(figsize=(W / 100, H / 100), dpi=150)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis("off")


def panel(x, y, w, h, title, kind, tag=None):
    fill, shadow, edge = KIND[kind]
    ax.add_patch(FancyBboxPatch((x + 7, y + 7), w, h, boxstyle="round,pad=0,rounding_size=14",
                                fc=shadow, ec="none"))
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=14",
                                fc=fill, ec=edge, lw=1.3))
    ax.text(x + 20, y + 26, title, ha="left", va="center", fontsize=15.5, weight="bold", color=INK)
    if tag:
        ax.text(x + w - 18, y + 26, tag, ha="right", va="center", fontsize=10.5, style="italic", color=SOFT)


def step(x, y, w, h, title, sub, kind, built=True):
    edge = KIND[kind][2]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=6",
                                fc="white", ec=edge if built else "#7C8594", lw=1.3,
                                linestyle="-" if built else (0, (4, 3))))
    ax.text(x + w / 2, y + 24, title, ha="center", va="center", fontsize=13.5, weight="bold", color=INK)
    ax.text(x + w / 2, y + 44, sub, ha="center", va="top", fontsize=10.5, color=SOFT, linespacing=1.3)


def arrow(pts, label=None, at=0, dx=0, dy=-10, col=INK, ls="-", ha="left"):
    xs, ys = zip(*pts)
    ax.plot(xs[:-1] + (xs[-1],), ys[:-1] + (ys[-1],), color=col, lw=1.5, linestyle=ls,
            solid_capstyle="butt", zorder=3)
    ax.annotate("", xy=pts[-1], xytext=pts[-2],
                arrowprops=dict(arrowstyle="-|>,head_length=0.6,head_width=0.32", color=col, lw=1.5,
                                shrinkA=0, shrinkB=0), zorder=4)
    if label:
        (x0, y0), (x1, y1) = pts[at], pts[at + 1]
        ax.text((x0 + x1) / 2 + dx, (y0 + y1) / 2 + dy, label, ha=ha, va="center", fontsize=11, color=col,
                bbox=dict(fc="white", ec="none", pad=1.5, alpha=0.9), zorder=5)


# ---------------------------------------------------------------- headers
LX, LW = 60, 690          # left column: sources
RX, RW = 815, 1125        # right column: the gym
DIV = 782
ax.text(LX, 22, "SOURCES · real passes and Isaac Sim, recorded the same way", ha="left", va="center",
        fontsize=13.5, weight="bold", color=INK)
ax.text(RX, 22, "GYM · workstation with RTX GPU · OpenCOOD / HEAL", ha="left", va="center",
        fontsize=13.5, weight="bold", color=INK)
ax.plot([DIV, DIV], [10, 868], color=RED, lw=2.2, linestyle=(0, (5, 4)), zorder=1)
ax.text(DIV, 888, "OPV2V frames", ha="center", va="center", fontsize=12, weight="bold", color=RED)

# ---------------------------------------------------------------- left: sources
BW, BH, G = 213, 92, 15
bx = [LX + 20, LX + 20 + BW + G, LX + 20 + 2 * (BW + G)]

panel(LX, 45, LW, 168, "Real recordings", "real", "MIRC sites 1 and 2")
step(bx[0], 100, BW, BH, "Coop pass", "2 robots + infra mast\none merged MCAP", "real")
step(bx[1], 100, BW, BH, "Reference poses", "scan-to-map + boards\ncm-level, every scan", "real")
step(bx[2], 100, BW, BH, "Link logs", "RTT 100 Hz, goodput\nRSSI, CSI, outages", "real")

panel(LX, 243, LW, 168, "Real → sim", "sim", "Isaac Sim 4.5 / 5.x")
step(bx[0], 298, BW, BH, "export_usd", "scanned mesh → USD\nin the map frame", "sim")
step(bx[1], 298, BW, BH, "make_scenario", "real robots, mounts, paths\n+ walkers, chairs", "sim")
step(bx[2], 298, BW, BH, "run_isaac", "RTX LiDAR + cameras\nMCAP + GT boxes", "sim")
arrow([(bx[0] + BW, 344), (bx[1], 344)])
arrow([(bx[1] + BW, 344), (bx[2], 344)])
arrow([(bx[1] + BW / 2, 192), (bx[1] + BW / 2, 298)], "trajectories, mounts", dx=8, dy=-2)

panel(LX, 441, LW, 168, "Labels", "label", "once per site")
step(bx[0], 496, BW, BH, "label_static", "each object boxed once\nin the map frame", "label")
step(bx[1], 496, BW, BH, "show_labels", "boxes drawn in every\nrobot's own image", "label")
step(bx[2], 496, BW, BH, "Box check", "LiDAR hits the boxes\nin 89–100% of frames", "label")
arrow([(bx[0] + BW, 542), (bx[1], 542)])
arrow([(bx[1] + BW, 542), (bx[2], 542)])

panel(LX, 639, LW, 168, "Converter · rosbag_to_opv2v", "data", "self-tested")
step(bx[0], 694, BW, BH, "Clock reconcile", "per-frame skew\nwritten into each frame", "data")
step(bx[1], 694, BW, BH, "Frame sync", "10 Hz frames\nsweep deskew", "data")
step(bx[2], 694, BW, BH, "Link per frame", "link state stored\nwith every frame", "data")
arrow([(bx[0] + BW, 740), (bx[1], 740)])
arrow([(bx[1] + BW, 740), (bx[2], 740)])
# sources into the converter, down the left edge
arrow([(LX, 150), (38, 150), (38, 740), (bx[0], 740)])
ax.plot([38, LX], [344, 344], color=INK, lw=1.5)
ax.plot([38, LX], [542, 542], color=INK, lw=1.5)

# ---------------------------------------------------------------- right: the gym
panel(RX, 45, RW, 182, "Channel knob · commchannel", "radio", "seeded · inert when off")
cw = (RW - 40 - 3 * 12) / 4
cx = [RX + 20 + i * (cw + 12) for i in range(4)]
step(cx[0], 100, cw, 102, "Delivery", "latency, i.i.d. loss\nbursty loss, bandwidth", "radio")
step(cx[1], 100, cw, 102, "Content", "stale messages, pose error\nghosts, scene swap", "radio")
step(cx[2], 100, cw, 102, "Blockage loss", "drop when an object\nblocks the radio path", "radio")
step(cx[3], 100, cw, 102, "Measured link replay", "logged RTT and outages\ndrive drop and delay", "radio",
     built=False)

panel(RX, 285, RW, 168, "Methods", "data", "pretrained, never retrained per condition")
mw = (RW - 40 - 2 * 12) / 3
mx = [RX + 20 + i * (mw + 12) for i in range(3)]
step(mx[0], 340, mw, BH, "OpenCOOD on OPV2V", "early, late, AttFuse, F-Cooper\nV2VNet, CoAlign, CoBEVT", "data")
step(mx[1], 340, mw, BH, "InCoP / HEAL", "IncopChannel wrapper\nreal MIRC + Isaac data", "data")
step(mx[2], 340, mw, BH, "Tracking", "Kalman + Hungarian MOT\nunder impairment", "data")

panel(RX, 483, RW, 168, "Scoring", "label", "resumable: one JSON per cell")
step(mx[0], 538, mw, BH, "Sweep", "831 channel conditions\n7 detectors", "label")
step(mx[1], 538, mw, BH, "Decomposition", "ego-visible vs occluded recall\nfalse-positive contamination", "label")
step(mx[2], 538, mw, BH, "Collaboration gain", "fused vs ego-only\non real hardware", "label")
arrow([(mx[0] + mw, 584), (mx[1], 584)])
arrow([(mx[1] + mw, 584), (mx[2], 584)])

panel(RX, 681, RW, 126, "Learn under the link", "real", "next")
lw2 = (RW - 40 - 12) / 2
step(RX + 20, 728, lw2, 64, "Train under the channel", "", "real", built=False)
step(RX + 20 + lw2 + 12, 728, lw2, 64, "Sim vs real, same channel", "", "real", built=False)
ax.texts[-1].set_text(""); ax.texts[-3].set_text("")
ax.text(RX + 20 + lw2 / 2, 776, "fine-tune with the logged link in the loop", ha="center", va="center",
        fontsize=10.5, color=SOFT)
ax.text(RX + 20 + lw2 * 1.5 + 12, 776, "Isaac arm vs MIRC arm, one method", ha="center", va="center",
        fontsize=10.5, color=SOFT)

# flows on the right
arrow([(bx[2] + BW, 740), (DIV + 12, 740), (DIV + 12, 396), (RX, 396)])
arrow([(RX + RW / 2, 227), (RX + RW / 2, 285)], "collaborator messages", dx=12, dy=2)
arrow([(RX + RW / 2, 453), (RX + RW / 2, 483)])
arrow([(RX + RW / 2, 651), (RX + RW / 2, 681)], col=SOFT, ls=(0, (4, 3)))
# measured link trace into the replay box, over the channel panel
ry = 37
arrow([(bx[2] + BW, 150), (DIV - 10, 150), (DIV - 10, ry), (cx[3] + 40, ry), (cx[3] + 40, 100)],
      col="#7A5BB5")
ax.text(cx[3] + 26, ry - 12, "measured link trace", ha="right", va="center", fontsize=11, color="#7A5BB5",
        bbox=dict(fc="white", ec="none", pad=1.5))

# ---------------------------------------------------------------- legend
ly = 925
items = [("real", "real hardware"), ("sim", "simulation"), ("label", "labels, scoring"),
         ("data", "data, methods"), ("radio", "channel")]
x = LX
for kind, name in items:
    fill, _, edge = KIND[kind]
    ax.add_patch(FancyBboxPatch((x, ly - 9), 30, 18, boxstyle="round,pad=0,rounding_size=4", fc=fill, ec=edge, lw=1))
    ax.text(x + 40, ly, name, ha="left", va="center", fontsize=12, color=INK)
    x += 40 + 9.2 * len(name) + 45
ax.add_patch(FancyBboxPatch((x, ly - 9), 30, 18, boxstyle="round,pad=0,rounding_size=4", fc="white",
                            ec="#7C8594", lw=1.1, linestyle=(0, (3, 2))))
ax.text(x + 40, ly, "not built yet", ha="left", va="center", fontsize=12, color=INK)
x += 40 + 9.2 * len("not built yet") + 45
ax.plot([x, x + 30], [ly, ly], color=RED, lw=2.2, linestyle=(0, (5, 4)))
ax.text(x + 40, ly, "OPV2V interface", ha="left", va="center", fontsize=12, color=INK)
ax.text(RX + RW, ly, "numbers: MIRC coop2, OPV2V sweeps", ha="right", va="center", fontsize=11,
        style="italic", color=SOFT)

fig.savefig(sys.argv[1], facecolor="white")
print(sys.argv[1])
