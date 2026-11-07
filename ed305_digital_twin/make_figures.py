#!/usr/bin/env python3
"""Paper figures from a room_track.py run.

    python make_figures.py out/run1
    python make_figures.py out/run1 --calib configs/cameras_room.yaml --persons 4 --frame-at 95

Reads <run>/persons_state.csv (and all_ids.mp4 when there). Writes to <run>/figures/,
each as PNG (300 dpi, for slides and docs) and PDF (vector, for LaTeX):

  floor_paths      small multiples, one panel per highlighted person (the N best observed):
                   their path in colour over everyone else's in grey; solid where seen,
                   dotted where carried over a gap; cameras as triangles (with --calib)
  trust_timeline   a row per highlighted person: position uncertainty (sigma_m, log) and
                   identity confidence (id_conf) over time; grey bands where not seen
  mosaic_frame     one frame of all_ids.mp4 (every camera + the floor plan), at --frame-at s
"""
import argparse
import csv
from pathlib import Path

import numpy as np

# Reference categorical palette, fixed order (dataviz skill, light mode): blue, orange, aqua, yellow,
# magenta, green. Identity follows the person, never their rank in a filtered list.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
INK, INK2, MUTED, GRID, GHOST = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#c9c8c3"


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path, help="room_track.py --out folder")
    ap.add_argument("--calib", type=Path, help="cameras YAML / json, to draw the cameras on the floor plan")
    ap.add_argument("--persons", type=int, default=4, help="people highlighted in colour (max 6, default 4)")
    ap.add_argument("--frame-at", type=float, help="time (s) of the mosaic frame (default: the middle)")
    ap.add_argument("--out", type=Path, help="figure folder (default <run>/figures)")
    return ap.parse_args()


def load_states(path):
    by = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            by.setdefault(r["person"], []).append(r)
    out = {}
    for p, rows in by.items():
        col = lambda k, typ=float: np.array([typ(r[k]) for r in rows])
        out[p] = {"t": col("t"), "x": col("x"), "y": col("y"), "sigma": col("sigma_m"),
                  "seen": col("seen", int).astype(bool), "id_conf": col("id_conf")}
    return out


def style(plt):
    plt.rcParams.update({
        "font.family": "sans-serif", "font.size": 8.5, "axes.titlesize": 9.5, "axes.labelsize": 8.5,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False, "savefig.dpi": 300,
        "figure.facecolor": "white", "axes.facecolor": "white", "pdf.fonttype": 42, "ps.fonttype": 42})


def save(fig, out, name):
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}", bbox_inches="tight", pad_inches=0.04)
    print(f"  {out / name}.png / .pdf")


def segments(t, mask):
    """Index runs where mask is True."""
    runs, start = [], None
    for i, m in enumerate(mask):
        if m and start is None:
            start = i
        if not m and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(mask)))
    return runs


def _bounds(S, cams, pad=0.3):
    xs = np.concatenate([s["x"] for s in S.values()] + [np.array([c[1] for c in cams])])
    ys = np.concatenate([s["y"] for s in S.values()] + [np.array([c[2] for c in cams])])
    return xs.min() - pad, xs.max() + pad, ys.min() - pad, ys.max() + pad


def floor_paths(plt, S, top, cams, out):
    """Small multiples: one panel per highlighted person, everyone else grey behind."""
    n = len(top)
    cols = 2 if n > 1 else 1
    rows = int(np.ceil(n / cols))
    x0, x1, y0, y1 = _bounds(S, cams)
    w = 6.8
    h = w / cols * (y1 - y0) / (x1 - x0) * rows + 0.5
    fig, axs = plt.subplots(rows, cols, figsize=(w, h), squeeze=False, sharex=True, sharey=True)
    for k, ax in enumerate(axs.flat):
        if k >= n:
            ax.axis("off")
            continue
        p, c = top[k], SERIES[k]
        for q, s in S.items():
            if q != p:
                ax.plot(s["x"], s["y"], color=GHOST, lw=0.6, zorder=1)
        s = S[p]
        ax.plot(s["x"], s["y"], color=c, lw=0.9, ls=(0, (1, 1.6)), zorder=2)          # carried over gaps
        for a, b in segments(s["t"], s["seen"]):
            ax.plot(s["x"][a:b], s["y"][a:b], color=c, lw=1.5, solid_capstyle="round", zorder=3)
        ax.plot(s["x"][-1], s["y"][-1], "o", ms=4.5, color=c, mec="white", mew=1.0, zorder=4)
        for name, x, y in cams:
            ax.plot(x, y, marker="^", ms=4, color=MUTED, ls="none", zorder=2)
        ax.set_title(f"{p}   seen {100 * s['seen'].mean():.0f} % of {s['t'][-1] - s['t'][0]:.0f} s",
                     loc="left", color=INK, fontsize=8.5)
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_aspect("equal")
    for ax in axs[-1]:
        ax.set_xlabel("x (m)")
    for ax in axs[:, 0]:
        ax.set_ylabel("y (m)")
    fig.suptitle(f"Paths of the {n} best-observed of {len(S)} people (others grey); dotted = carried over a gap"
                 + ("; triangles = cameras" if cams else ""), x=0.01, ha="left", fontsize=9.5, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save(fig, out, "floor_paths")
    plt.close(fig)


def trust_timeline(plt, S, top, out):
    """Small multiples: a row per person; left position uncertainty, right identity confidence."""
    n = len(top)
    fig, axs = plt.subplots(n, 2, figsize=(6.8, 0.95 * n + 0.8), sharex=True, squeeze=False,
                            gridspec_kw={"hspace": 0.35, "wspace": 0.22})
    smax = max(S[p]["sigma"].max() for p in top)
    smin = min(S[p]["sigma"].min() for p in top)
    for k, p in enumerate(top):
        s, c = S[p], SERIES[k]
        a1, a2 = axs[k]
        for ax in (a1, a2):
            for a, b in segments(s["t"], ~s["seen"]):                  # not seen: a quiet band
                ax.axvspan(s["t"][a], s["t"][min(b, len(s["t"]) - 1)], color=GRID, alpha=0.7, lw=0)
        a1.plot(s["t"], s["sigma"], color=c, lw=1.1)
        a2.step(s["t"], s["id_conf"], where="post", color=c, lw=1.5)
        a1.set_yscale("log")
        a1.set_ylim(smin * 0.8, smax * 1.3)
        a2.set_ylim(-0.04, 1.06)
        a2.set_yticks([0, 0.5, 1])
        a1.text(0.005, 0.95, p, transform=a1.transAxes, va="top", fontsize=8, fontweight="bold", color=INK)
    axs[0, 0].set_title("position std (m, log)", loc="left", color=INK, fontsize=8.5)
    axs[0, 1].set_title("identity confidence", loc="left", color=INK, fontsize=8.5)
    for ax in axs[-1]:
        ax.set_xlabel("time (s)")
    fig.suptitle("Uncertainty grows while a person is unseen (grey bands); identity confidence drops at "
                 "uncertain links", x=0.01, ha="left", fontsize=9.5, color=INK)
    fig.subplots_adjust(top=1 - 0.55 / (0.95 * n + 0.8))
    save(fig, out, "trust_timeline")
    plt.close(fig)


def mosaic_frame(run, at, out):
    import cv2
    path = run / "all_ids.mp4"
    if not path.exists():
        print("  (no all_ids.mp4: run room_track.py with --video all for the mosaic frame)")
        return
    cap = cv2.VideoCapture(str(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 10
    k = int(np.clip(round(at * fps), 0, max(n - 1, 0))) if at is not None else n // 2
    cap.set(cv2.CAP_PROP_POS_FRAMES, k)
    ok, f = cap.read()
    cap.release()
    if ok:
        cv2.imwrite(str(out / "mosaic_frame.png"), f)
        print(f"  {out / 'mosaic_frame.png'} (video frame {k} of {n})")


def cameras_xy(calib):
    if calib is None:
        return []
    from ed305.camera import load_any
    return [(c.name, float(c.center[0]), float(c.center[1])) for c in load_any(calib)]


def main():
    a = parse()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    style(plt)
    out = a.out or a.run / "figures"
    out.mkdir(parents=True, exist_ok=True)
    S = load_states(a.run / "persons_state.csv")
    if not S:
        raise SystemExit("persons_state.csv is empty")
    n = max(1, min(a.persons, len(SERIES)))
    top = sorted(S, key=lambda p: (-S[p]["seen"].sum(), p))[:n]     # best observed, colour fixed per person
    print(f"figures for {len(S)} people, highlighted: {', '.join(top)}")
    floor_paths(plt, S, top, cameras_xy(a.calib), out)
    trust_timeline(plt, S, top, out)
    mosaic_frame(a.run, a.frame_at, out)


if __name__ == "__main__":
    main()
