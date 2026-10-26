#!/usr/bin/env python3
"""gym_scenarios.py as one column: real data, the gym's scenarios, train,
test on real, top to bottom.

    python3 slides_media/gym_scenarios_column.py slides_media/gym_scenarios_column.png
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

W, H = 1100, 1375
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
        ax.text(x + 9.4 * 20 * len(text) / 14 + 20, y + 2, sub, ha="left", va="center", fontsize=13,
                color=SOFT, style="italic")


X, CW = 80, 990
MID = X + CW / 2

# ------------------------------------------------------------ 1 real data
y1 = 30
rbox(X, y1, CW, 300, "real")
heading(X + 26, y1 + 40, "1  Real data", "one recorded pass")
items = [("Scanned building", "reference map, cm-level"),
         ("Robots and sensors", "real LiDARs, cameras, mounts"),
         ("Real trajectories", "where the robots drove"),
         ("Measured Wi-Fi link", "RTT, goodput, outages")]
iw = (CW - 48 - 16) / 2
for i, (t, s) in enumerate(items):
    ix, iy = X + 24 + (i % 2) * (iw + 16), y1 + 85 + (i // 2) * 104
    rbox(ix, iy, iw, 90, "real", r=8, white=True, lw=1.2, shadow=False)
    ax.add_patch(Circle((ix + 36, iy + 45), 12, fc=ACCENT, ec="none"))
    ax.text(ix + 64, iy + 30, t, ha="left", va="center", fontsize=15, weight="bold", color=INK)
    ax.text(ix + 64, iy + 60, s, ha="left", va="center", fontsize=12.5, color=SOFT)

# ------------------------------------------------------------ 2 the gym
y2 = y1 + 300 + 64
GH = 520
rbox(X, y2, CW, GH, "sim")
heading(X + 26, y2 + 40, "2  Gym · Isaac Sim", "change what you want, keep the rest real")
kx = X + 26
for k in ["+ people walking", "move furniture", "new robot paths", "harsher link"]:
    w = 24 + 9.0 * len(k)
    ax.add_patch(FancyBboxPatch((kx, y2 + 80), w, 40, boxstyle="round,pad=0,rounding_size=20",
                                fc="white", ec=KIND["sim"][2], lw=1.2))
    ax.text(kx + w / 2, y2 + 100, k, ha="center", va="center", fontsize=12.5, color=INK)
    kx += w + 10
fx, fy, fw, fh = X + 60, y2 + 160, 760, 280
for i in range(3):
    off = (2 - i) * 22
    rbox(fx + off, fy + off, fw, fh, "sim", r=10, white=True, lw=1.3, shadow=(i == 2))
ax.text(fx + 30, fy + 42, "Scenario 1", ha="left", va="center", fontsize=17, weight="bold", color=INK)
ax.text(fx + 30, fy + 76, "4 people crossing · lossy link", ha="left", va="center", fontsize=13, color=SOFT)
for j, (t, s) in enumerate([("LiDAR + camera data", "from every robot, like a real bag"),
                            ("Ground-truth boxes", "every object, every frame, free"),
                            ("Link per frame", "the measured or tuned channel")]):
    yy = fy + 128 + j * 50
    ax.text(fx + 30, yy, "✓", ha="left", va="center", fontsize=16, weight="bold", color="#2E7D4F")
    ax.text(fx + 62, yy, t, ha="left", va="center", fontsize=14, weight="bold", color=INK)
    ax.text(fx + 330, yy, s, ha="left", va="center", fontsize=12.5, color=SOFT)
ax.text(X + CW - 30, y2 + GH - 30, "× N scenarios", ha="right", va="center", fontsize=20, weight="bold",
        color=ACCENT)

# ------------------------------------------------------------ 3 train, 4 test on real
y3 = y2 + GH + 64
rbox(X, y3, CW, 150, "data")
ax.text(X + 28, y3 + 42, "3  Train", ha="left", va="center", fontsize=20, weight="bold", color=INK)
ax.text(X + 28, y3 + 88, "collaborative perception models on gym scenarios built from real data",
        ha="left", va="center", fontsize=14, color=INK)
ax.text(X + 28, y3 + 120, "next step", ha="left", va="center", fontsize=12, style="italic", color=SOFT)

y4 = y3 + 150 + 64
rbox(X, y4, CW, 150, "label")
ax.text(X + 28, y4 + 42, "4  Test on real", ha="left", va="center", fontsize=20, weight="bold", color=INK)
ax.text(X + 28, y4 + 88, "a held-out real recording with its measured link", ha="left", va="center",
        fontsize=14, color=INK)
ax.text(X + 28, y4 + 120, "sim-trained vs real-trained, same scene", ha="left", va="center", fontsize=12,
        style="italic", color=SOFT)

# flows down the middle
arrow([(MID, y1 + 300), (MID, y2)])
arrow([(MID, y2 + GH), (MID, y3)])
arrow([(MID, y3 + 150), (MID, y4)])
# held-out real data straight to the test, down the left edge
lx = 42
arrow([(X, y1 + 150), (lx, y1 + 150), (lx, y4 + 75), (X, y4 + 75)], col=SOFT, ls=(0, (6, 4)))
ax.text(lx - 22, (y1 + 150 + y4 + 75) / 2, "held-out real recording", rotation=90, ha="center", va="center",
        fontsize=13, style="italic", color=SOFT, bbox=dict(fc="white", ec="none", pad=2))

fig.savefig(sys.argv[1], facecolor="white")
print(sys.argv[1])
