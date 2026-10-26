#!/usr/bin/env python3
"""The gym's point in one picture: real data in, many training scenarios out,
tested back on real data. Same style as the "Dynamic Digital Twin" slide.

    python3 slides_media/gym_scenarios.py slides_media/gym_scenarios.png
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle

plt.rcParams["font.family"] = "Inter"

INK, SOFT = "#1B2430", "#5A6472"
KIND = {  # fill, shadow, edge
    "real":  ("#E9EAEC", "#DADCE0", "#B9BEC6"),
    "sim":   ("#FBEFD9", "#F1E1C2", "#E2C58E"),
    "label": ("#DFF0E3", "#CDE5D3", "#9CCBA8"),
    "data":  ("#DDE8F6", "#CCDAEE", "#9DB7DD"),
}
ACCENT = "#C2571A"

W, H = 2000, 820
fig = plt.figure(figsize=(W / 100, H / 100), dpi=150)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis("off")


def rbox(x, y, w, h, kind, r=14, white=False, lw=1.4, shadow=True):
    fill, sh, edge = KIND[kind]
    if shadow:
        ax.add_patch(FancyBboxPatch((x + 7, y + 7), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                    fc=sh, ec="none"))
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                fc="white" if white else fill, ec=edge, lw=lw))


def arrow(pts, col=INK, ls="-", lw=2.0):
    xs, ys = zip(*pts)
    ax.plot(xs, ys, color=col, lw=lw, linestyle=ls, zorder=3)
    ax.annotate("", xy=pts[-1], xytext=pts[-2],
                arrowprops=dict(arrowstyle="-|>,head_length=0.8,head_width=0.42", color=col, lw=lw,
                                shrinkA=0, shrinkB=0), zorder=4)


def heading(x, y, text, sub=None):
    ax.text(x, y, text, ha="left", va="center", fontsize=20, weight="bold", color=INK)
    if sub:
        ax.text(x, y + 34, sub, ha="left", va="center", fontsize=13, color=SOFT, style="italic")


# ------------------------------------------------------------ real data
RX, RY, RW, RH = 40, 40, 480, 600
rbox(RX, RY, RW, RH, "real")
heading(RX + 26, RY + 40, "1  Real data", "one recorded pass")
items = [("Scanned building", "reference map, cm-level"),
         ("Robots and sensors", "real LiDARs, cameras, mounts"),
         ("Real trajectories", "where the robots drove"),
         ("Measured Wi-Fi link", "RTT, goodput, outages")]
for i, (t, s) in enumerate(items):
    y = RY + 125 + i * 116
    rbox(RX + 24, y, RW - 48, 96, "real", r=8, white=True, lw=1.2, shadow=False)
    ax.add_patch(Circle((RX + 62, y + 48), 13, fc=ACCENT, ec="none"))
    ax.text(RX + 92, y + 32, t, ha="left", va="center", fontsize=15, weight="bold", color=INK)
    ax.text(RX + 92, y + 64, s, ha="left", va="center", fontsize=12.5, color=SOFT)

# ------------------------------------------------------------ the gym
GX, GY, GW, GH = 600, 40, 720, 600
rbox(GX, GY, GW, GH, "sim")
heading(GX + 26, GY + 40, "2  Gym · Isaac Sim", "change what you want, keep the rest real")
# knobs
knobs = ["+ people walking", "move furniture", "new robot paths", "harsher link"]
kx = GX + 26
for k in knobs:
    w = 24 + 9.0 * len(k)
    ax.add_patch(FancyBboxPatch((kx, GY + 108), w, 40, boxstyle="round,pad=0,rounding_size=20",
                                fc="white", ec=KIND["sim"][2], lw=1.2))
    ax.text(kx + w / 2, GY + 128, k, ha="center", va="center", fontsize=12.5, color=INK)
    kx += w + 10
# a deck of scenarios
cards = [("Scenario 3", "chairs moved · 2 people"), ("Scenario 2", "new path for mobile_2"),
         ("Scenario 1", "4 people crossing · lossy link")]
for i, (t, s) in enumerate(cards):
    off = (2 - i) * 22
    cx, cy, cw, ch = GX + 70 + off, GY + 200 + off, 560, 300
    rbox(cx, cy, cw, ch, "sim", r=10, white=True, lw=1.3, shadow=(i == 2))
front_x, front_y = GX + 70, GY + 200
ax.text(front_x + 30, front_y + 42, "Scenario 1", ha="left", va="center", fontsize=17, weight="bold", color=INK)
ax.text(front_x + 30, front_y + 76, "4 people crossing · lossy link", ha="left", va="center", fontsize=13,
        color=SOFT)
for j, (t, s) in enumerate([("LiDAR + camera data", "from every robot, like a real bag"),
                            ("Ground-truth boxes", "every object, every frame, free"),
                            ("Link per frame", "the measured or tuned channel")]):
    y = front_y + 128 + j * 52
    ax.text(front_x + 30, y, "✓", ha="left", va="center", fontsize=16, weight="bold", color="#2E7D4F")
    ax.text(front_x + 62, y, t, ha="left", va="center", fontsize=14, weight="bold", color=INK)
    ax.text(front_x + 300, y, s, ha="left", va="center", fontsize=12, color=SOFT)
ax.text(GX + GW - 30, GY + GH - 28, "× N scenarios", ha="right", va="center", fontsize=20, weight="bold",
        color=ACCENT)

# ------------------------------------------------------------ train, then test on real
TX, TW = 1400, 560
rbox(TX, 120, TW, 200, "data")
ax.text(TX + 28, 162, "3  Train", ha="left", va="center", fontsize=20, weight="bold", color=INK)
ax.text(TX + 28, 212, "collaborative perception models", ha="left", va="center", fontsize=14, color=INK)
ax.text(TX + 28, 246, "on gym scenarios built from real data", ha="left", va="center", fontsize=14, color=INK)
ax.text(TX + 28, 284, "next step", ha="left", va="center", fontsize=12, style="italic", color=SOFT)

rbox(TX, 420, TW, 220, "label")
ax.text(TX + 28, 462, "4  Test on real", ha="left", va="center", fontsize=20, weight="bold", color=INK)
ax.text(TX + 28, 512, "a held-out real recording", ha="left", va="center", fontsize=14, color=INK)
ax.text(TX + 28, 546, "with its measured link", ha="left", va="center", fontsize=14, color=INK)
ax.text(TX + 28, 590, "sim-trained vs real-trained, same scene", ha="left", va="center", fontsize=12,
        style="italic", color=SOFT)

# flows
arrow([(RX + RW, 340), (GX, 340)])
arrow([(GX + GW, 220), (TX, 220)])
arrow([(TX + TW / 2, 320), (TX + TW / 2, 420)])
# real held-out data goes straight to the test
arrow([(RX + RW / 2, RY + RH), (RX + RW / 2, 730), (TX + TW / 2, 730), (TX + TW / 2, 640)],
      col=SOFT, ls=(0, (6, 4)))
ax.text((RX + RW / 2 + TX + TW / 2) / 2, 754, "held-out real recording", ha="center", va="center",
        fontsize=13, style="italic", color=SOFT)

fig.savefig(sys.argv[1], facecolor="white")
print(sys.argv[1])
