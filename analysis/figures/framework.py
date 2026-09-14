#!/usr/bin/env python3
"""The dataset's framework diagram: where sensing and comms meet, and which
algorithm family sits in each slot of the replay benchmark.

    python figures/framework.py            # writes fig_framework.{pdf,png,svg}

Two recorded streams become two offline tables -- a rate-utility curve from the
clouds and a capacity trace from the radio -- and the replay harness joins them
one second at a time. Each of the four stages is a slot a published algorithm
family plugs into; only the boxed choice varies between runs.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

SENSE, COMMS, ACCENT = "#2a78d6", "#eb6834", "#1baf7a"
TEXT, TEXT2, LINE = "#16181d", "#5a6070", "#c8ccd6"
TINT = {SENSE: "#eaf2fc", COMMS: "#fdeee7", ACCENT: "#e7f6f0", TEXT2: "#f4f5f8"}


def box(ax, x0, y0, x1, y1, edge, lw=1.4, fill=None, z=2):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                                boxstyle="round,pad=0,rounding_size=0.9",
                                linewidth=lw, edgecolor=edge,
                                facecolor=TINT.get(fill or edge, "white"), zorder=z))


def arrow(ax, x0, y0, x1, y1, color, lw=1.6, z=3):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                 mutation_scale=13, linewidth=lw, color=color, zorder=z))


def stage(ax, x0, x1, num, title, sub, items, colour):
    y0, y1 = 13.0, 43.0
    box(ax, x0, y0, x1, y1, colour, lw=1.6)
    xm = (x0 + x1) / 2
    ax.text(x0 + 1.6, y1 - 2.6, num, fontsize=13, weight="bold", color=colour, va="center")
    ax.text(x0 + 5.4, y1 - 2.6, title, fontsize=11, weight="bold", color=TEXT, va="center")
    ax.text(xm, y1 - 6.0, sub, fontsize=8.4, color=TEXT2, ha="center", style="italic")
    for i, it in enumerate(items):
        ax.text(xm, y1 - 9.6 - 2.55 * i, it, fontsize=8.4, color=TEXT, ha="center")


def main() -> int:
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_xlim(0, 160); ax.set_ylim(0, 92); ax.axis("off")

    ax.text(80, 89.4, "Bridging comms and sensing with measured data",
            fontsize=16, weight="bold", color=TEXT, ha="center")

    # ---- recorded streams ----------------------------------------------------
    box(ax, 6, 74, 76, 86, SENSE)
    ax.text(41, 83.4, "SENSING  (recorded)", fontsize=11, weight="bold", color=SENSE, ha="center")
    ax.text(41, 79.9, "LiDAR clouds · RGB-D · poses · tf", fontsize=9.2, color=TEXT, ha="center")
    ax.text(41, 76.6, "what there is to share", fontsize=8.4, color=TEXT2, ha="center", style="italic")

    box(ax, 84, 74, 154, 86, COMMS)
    ax.text(119, 83.4, "COMMS  (recorded)", fontsize=11, weight="bold", color=COMMS, ha="center")
    ax.text(119, 79.9, "CSI 183 Hz · RSSI · MCS · goodput · RTT", fontsize=9.2, color=TEXT, ha="center")
    ax.text(119, 76.6, "what the link can carry", fontsize=8.4, color=TEXT2, ha="center", style="italic")

    ax.text(80, 87.2, "one recording, one clock  (NTP, 27-43 µs RMS)",
            fontsize=8.6, color=TEXT2, ha="center")

    # ---- offline products ----------------------------------------------------
    for x, c in ((41, SENSE), (119, COMMS)):
        arrow(ax, x, 74, x, 68.5, c)
        ax.text(x + 1.4, 71.2, "offline", fontsize=8, color=TEXT2, ha="left", style="italic")

    box(ax, 6, 56, 76, 68, SENSE)
    ax.text(41, 65.4, "RATE–UTILITY CURVE", fontsize=11, weight="bold", color=SENSE, ha="center")
    ax.text(41, 61.9, "bytes → IoU, per frame", fontsize=9.2, color=TEXT, ha="center")
    ax.text(41, 58.6, "fuse at full fidelity = oracle; degrade and measure",
            fontsize=8.4, color=TEXT2, ha="center", style="italic")

    box(ax, 84, 56, 154, 68, COMMS)
    ax.text(119, 65.4, "CAPACITY TRACE", fontsize=11, weight="bold", color=COMMS, ha="center")
    ax.text(119, 61.9, "bytes available(t)  ·  RTT(t)", fontsize=9.2, color=TEXT, ha="center")
    ax.text(119, 58.6, "MCS×NSS×width, calibrated by iperf",
            fontsize=8.4, color=TEXT2, ha="center", style="italic")

    arrow(ax, 41, 56, 70, 51.0, SENSE)
    arrow(ax, 119, 56, 90, 51.0, COMMS)

    # ---- harness -------------------------------------------------------------
    box(ax, 6, 10, 154, 50, TEXT2, lw=1.2, fill=TEXT2, z=1)
    ax.text(80, 47.4, "REPLAY HARNESS   —   one second at a time, nothing transmitted live",
            fontsize=11.5, weight="bold", color=TEXT, ha="center")

    stage(ax, 9, 44, "①", "BUDGET", "how many bytes?",
          ["fixed constant  (what CP papers assume)", "round-robin · max-C/I", "proportional fair",
           "knapsack (MCKP)", "Lyapunov · Whittle/AoI", "bandit / DRL on raw CSI", "MILP hindsight (ceiling)"],
          ACCENT)
    stage(ax, 47, 82, "②", "CONTENT", "what goes in those bytes?",
          ["voxel ladder  (no model)", "Where2comm", "How2comm · CoSDH", "When2com / Who2com",
           "AdaFusion", "fixed-size models:", "V2VNet, DiscoNet, CoBEVT …"], SENSE)
    stage(ax, 85, 120, "③", "LATENESS", "did it arrive in time?",
          ["bytes / capacity(t) + RTT(t)", "on time  →  full value", "late  →  stale or dropped",
           "SyncNet", "CoBEVFlow", "V2X-ViT (delay-aware)"], COMMS)
    stage(ax, 123, 151, "④", "FUSE + SCORE", "how good is the result?",
          ["OpenCOOD fusion zoo", "BEV occupancy IoU", "vs full-fidelity oracle", "(mAP if labelled)"], TEXT2)

    for x in (44.6, 82.6, 120.6):
        arrow(ax, x, 28, x + 1.8, 28, TEXT2, lw=1.4, z=4)

    ax.text(26.5, 11.4, "← the measured channel enters here", fontsize=8.6,
            weight="bold", color=ACCENT, ha="center")

    # ---- result --------------------------------------------------------------
    arrow(ax, 80, 10, 80, 6.6, TEXT2)
    box(ax, 6, 1.2, 154, 6.4, TEXT, lw=1.4, fill=TEXT2)
    ax.text(80, 4.6, "RESULTS:  mean IoU  ·  deadline-miss rate  ·  bytes used  —  one row per policy",
            fontsize=10.4, weight="bold", color=TEXT, ha="center")
    ax.text(80, 2.3, "same recording, same clouds, same fusion — only the boxed choice changes, "
                     "so every difference is scheduling",
            fontsize=8.6, color=TEXT2, ha="center", style="italic")

    out = Path(sys.argv[1] if len(sys.argv) > 1 else "figures")
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png", "svg"):
        fig.savefig(out / f"fig_framework.{ext}", dpi=200, bbox_inches="tight",
                    facecolor="white")
    print(f"wrote {out}/fig_framework.pdf, .png, .svg")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
