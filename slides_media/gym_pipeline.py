import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

out = sys.argv[1]
INK, SOFT, BLUE, BLUE_L, ORANGE, ORANGE_L, GREEN_L, GREY_L = "#17212B", "#4A5561", "#0A8FD0", "#DCEFF9", "#C2571A", "#FBE7D8", "#E3F1E4", "#EEF0EC"

fig, ax = plt.subplots(figsize=(16, 6.6), dpi=200)
ax.set_xlim(0, 16); ax.set_ylim(0, 6.6); ax.axis("off")

def box(x, y, w, h, title, sub, fill, edge=INK, bold_col=INK):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.18",
                                fc=fill, ec=edge, lw=1.6))
    ax.text(x + w / 2, y + h - 0.32, title, ha="center", va="top", fontsize=15, weight="bold", color=bold_col)
    ax.text(x + w / 2, y + h - 0.82, sub, ha="center", va="top", fontsize=11.5, color=SOFT, linespacing=1.35)

def arrow(x0, y0, x1, y1, col=INK, ls="-", rad=0.0):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=18,
                                 color=col, lw=1.8, linestyle=ls, connectionstyle=f"arc3,rad={rad}"))

# top lane: scenario generation
W, H, Y = 2.75, 1.9, 3.75
box(0.2, Y, W, H, "Real coop pass", "2 robots, real sensors\nmeasured Wi-Fi link\nreference poses", GREY_L)
box(3.45, Y, W, H, "Isaac Sim scenario", "same robots, mounts,\ntrajectories, building\n+ people, furniture", BLUE_L)
box(6.7, Y, W, H, "Sensor bag + labels", "LiDAR, cameras\nlaid out like real data\nGT boxes every frame", BLUE_L)
box(9.95, Y, W, H, "OPV2V frames", "the format\nOpenCOOD reads", BLUE_L)
arrow(2.95, Y + H / 2, 3.45, Y + H / 2)
arrow(6.2, Y + H / 2, 6.7, Y + H / 2)
arrow(9.45, Y + H / 2, 9.95, Y + H / 2)
# real data also goes straight to OPV2V
ax.annotate("", xy=(11.3, Y + H + 0.02), xytext=(1.575, Y + H + 0.02),
            arrowprops=dict(arrowstyle="-|>", color=SOFT, lw=1.4, ls="--",
                            connectionstyle="bar,fraction=-0.04", mutation_scale=16))
ax.text(6.45, 6.07, "real recordings convert directly too", ha="center", va="center", fontsize=11, color=SOFT, style="italic", bbox=dict(fc="white", ec="none", pad=3))

# bottom lane: the gym loop
Yb, Hb = 0.35, 2.15
box(9.95, Yb, W, Hb, "Detector", "early / intermediate /\nlate fusion\n(7 pretrained models)", GREEN_L)
box(6.2, Yb, 3.25, Hb, "Channel knob", "ideal  ·  synthetic  ·  measured\nlatency, loss, bursts,\nbandwidth, stale, pose error", ORANGE_L, edge=ORANGE, bold_col=ORANGE)
box(13.2, Yb, 2.6, Hb, "Score", "detection AP\ncollaboration gain\nper condition", GREY_L)
arrow(11.325, Y, 11.325, Yb + Hb)                       # frames -> detector
arrow(9.45, Yb + Hb / 2, 9.95, Yb + Hb / 2, col=ORANGE)  # channel -> detector
arrow(12.7, Yb + Hb / 2, 13.2, Yb + Hb / 2)              # detector -> score
# measured link feeds the knob
ax.add_patch(FancyArrowPatch((1.575, Y), (6.2, Yb + Hb * 0.55), arrowstyle="-|>", mutation_scale=18,
                             color=ORANGE, lw=1.8, linestyle="--", connectionstyle="arc3,rad=0.25"))
ax.text(2.2, 1.55, "measured link trace\n(RTT, goodput, outages)", ha="left", va="center", fontsize=11, color=ORANGE)

fig.savefig(out, bbox_inches="tight", facecolor="white")
print(out)
