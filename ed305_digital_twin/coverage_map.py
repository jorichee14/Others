#!/usr/bin/env python3
"""Blind spots of the ED305 camera set.

    python coverage_map.py                                   # nominal layout
    python coverage_map.py --cameras configs/cameras.yaml --room configs/room.yaml
    python coverage_map.py --k 3 --without cam_4,cam_6       # what if 4 and 6 go

Writes to --out (default out/<cameras file stem>/):
  report.md      coverage at k=1..3, per camera: what it alone holds up,
                 coverage without it, and a greedy smallest set
  floor.png      top view: for each floor cell, the fewest cameras that see
                 any height of that column (0..2 m); red hatch = below k
  slices.png     one top view per target height, cameras seeing each voxel
  coverage.npz   xs, ys, zs, per-camera visibility, obstacle mask
"""
import argparse
from pathlib import Path

import numpy as np

from ed305.camera import load_cameras
from ed305.coverage import (covered, critical, greedy_subset, leave_one_out,
                            load_room, visibility)

HERE = Path(__file__).resolve().parent

# dataviz reference palette: sequential blue ramp, status critical
BLUES = ["#f0efec", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#1c5cab", "#104281", "#0d366b"]
CRITICAL = "#d03b3b"
INK, MUTED, OBSTACLE = "#1a1a19", "#6b6a66", "#b9b8b3"


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cameras", default=HERE / "configs/cameras_nominal.yaml", type=Path)
    ap.add_argument("--room", default=HERE / "configs/room_nominal.yaml", type=Path)
    ap.add_argument("--k", type=int, help="cameras needed per voxel (default: room target.k)")
    ap.add_argument("--without", default="", help="comma-separated camera names to leave out")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--no-plots", action="store_true")
    return ap.parse_args()


def pct(x):
    return f"{100 * x:.1f}%"


def report(cams, room, vis, valid, k, nz):
    names = [c.name for c in cams]
    n = vis.sum(0)
    col = n.reshape(-1, nz)
    col_valid = valid.reshape(-1, nz).all(1)
    lines = [
        f"# ED305 coverage: {len(cams)} cameras, k = {k}", "",
        f"Target: x {room.x}, y {room.y}, z {room.z[0]:.2f}..{room.z[-1]:.2f} m "
        f"every {room.z[1] - room.z[0] if len(room.z) > 1 else 0:.2f} m, "
        f"cells {room.xy_step:.2f} m, min {room.min_px_per_m:g} px/m. "
        f"{int(valid.sum())} voxels outside obstacles.", "",
        "| cameras per voxel | voxels | floor columns (every height) |", "|---|---|---|",
    ]
    for kk in sorted({1, 2, 3, k}):
        lines.append(f"| >= {kk} | {pct(np.mean(n[valid] >= kk))} | "
                     f"{pct(np.mean(col[col_valid].min(1) >= kk))} |")
    lines += ["", f"Blind (< {k}) voxels by height:", "", "| z (m) | blind |", "|---|---|"]
    vz = valid.reshape(-1, nz)
    for j, z in enumerate(room.z):
        lines.append(f"| {z:.2f} | {pct(np.mean(col[vz[:, j], j] < k))} |")

    loo = leave_one_out(vis, k, valid)
    crit = critical(vis, k, valid)
    full = np.mean(n[valid] >= k)
    lines += ["", f"Per camera (k = {k}; full set covers {pct(full)}):", "",
              "| camera | sees | only one holding k (voxels) | coverage without it | HFOV x VFOV |",
              "|---|---|---|---|---|"]
    for i, c in enumerate(cams):
        h, v = c.fov_deg()
        lines.append(f"| {c.name} | {pct(np.mean(vis[i][valid]))} | {crit[i]} | {pct(loo[i])} | {h:.0f} x {v:.0f} deg |")

    sub = greedy_subset(vis, k, valid)
    lines += ["", f"Greedy smallest set keeping everything the full set covers at k = {k}: "
              f"{len(sub)} cameras: {', '.join(names[i] for i in sub)}.",
              "Greedy is an upper bound on the true minimum; the rest are redundancy "
              "against people and robots blocking views, which this static model does not count."]
    return "\n".join(lines) + "\n"


def draw_room(ax, room, cams, label=True):
    import matplotlib.patches as mp
    for name, lo, hi in room.obstacles:
        ax.add_patch(mp.Rectangle(lo[:2], *(hi[:2] - lo[:2]), fc=OBSTACLE, ec=MUTED, lw=0.8, zorder=3))
    ax.add_patch(mp.Rectangle((room.x[0], room.y[0]), room.x[1] - room.x[0], room.y[1] - room.y[0],
                              fill=False, ec=INK, lw=1.2, zorder=4))
    for c in cams:
        p, a = c.center, c.axis
        d = a[:2] / max(np.linalg.norm(a[:2]), 1e-9) * 0.45 * np.linalg.norm(a[:2])
        ax.annotate("", xy=p[:2] + d, xytext=p[:2], zorder=6,
                    arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.2, mutation_scale=9))
        ax.plot(*p[:2], "o", ms=8, mfc="white", mec=INK, mew=1.5, zorder=7)
        if label:
            ax.text(p[0], p[1] + 0.22, c.name.replace("cam_", ""), ha="center", va="bottom",
                    fontsize=8, color=INK, zorder=8)
    ax.plot(0, 0, "+", color=INK, ms=8, zorder=5)
    ax.set_aspect("equal")
    pad = 0.6
    ax.set_xlim(room.x[0] - pad, room.x[1] + pad)
    ax.set_ylim(room.y[0] - pad, room.y[1] + pad)
    ax.tick_params(colors=MUTED, labelsize=8)
    for s in ax.spines.values():
        s.set_visible(False)


def heat(ax, xs, ys, counts, k, vmax):
    """Cameras per cell, one ramp step per count; vmax is the top bin ("vmax+")."""
    from matplotlib.colors import BoundaryNorm, ListedColormap
    cmap = ListedColormap(BLUES[: vmax + 1])
    cmap.set_bad(OBSTACLE)
    norm = BoundaryNorm(np.arange(-0.5, vmax + 1.5), cmap.N)
    s = xs[1] - xs[0] if len(xs) > 1 else 0.1
    ext = (xs[0] - s / 2, xs[-1] + s / 2, ys[0] - s / 2, ys[-1] + s / 2)
    im = ax.imshow(np.ma.minimum(counts, vmax).T, origin="lower", extent=ext, cmap=cmap, norm=norm, interpolation="nearest", zorder=1)
    # blind cells: red hatch and outline (hatch colour from rcParams, set in plots())
    blind = (counts.filled(k) < k).T.astype(float)
    if blind.any():
        X, Y = np.meshgrid(xs, ys)
        ax.contourf(X, Y, blind, levels=[0.5, 1.5], colors="none", hatches=["////"], zorder=2)
        ax.contour(X, Y, blind, levels=[0.5], colors=CRITICAL, linewidths=1.2, zorder=2)
    return im


def plots(out, cams, room, xs, ys, vis, blocked, k):
    import matplotlib as mpl
    mpl.use("Agg")
    mpl.rcParams.update({"hatch.color": CRITICAL, "hatch.linewidth": 0.8, "font.size": 9,
                         "axes.titlecolor": INK, "text.color": INK})
    import matplotlib.patches as mp
    import matplotlib.pyplot as plt

    nz = len(room.z)
    n = vis.sum(0).reshape(len(xs), len(ys), nz)
    blk = blocked.reshape(len(xs), len(ys), nz)
    vmax = min(int(n.max()), len(BLUES) - 1)
    ticks = list(range(vmax + 1))
    tick_labels = [str(t) for t in ticks[:-1]] + [f"{vmax}+" if n.max() > vmax else str(vmax)]

    # floor: worst height of each column (obstacle voxels left out)
    nn = np.where(blk, np.iinfo(int).max, n)
    colmin = nn.min(2)
    floor = np.ma.masked_where(blk.all(2), np.where(blk.all(2), 0, colmin))
    fig, ax = plt.subplots(figsize=(7, 7 * (room.y[1] - room.y[0] + 1.2) / (room.x[1] - room.x[0] + 1.2) * 0.85))
    im = heat(ax, xs, ys, floor, k, vmax)
    draw_room(ax, room, cams)
    ax.set_title(f"Fewest cameras seeing any height {room.z[0]:.1f}-{room.z[-1]:.1f} m of each floor cell",
                 loc="left", fontsize=10)
    cb = fig.colorbar(im, ax=ax, ticks=ticks, shrink=0.7, label="cameras")
    cb.ax.set_yticklabels(tick_labels)
    cb.outline.set_visible(False)
    ax.legend(handles=[mp.Patch(fc="none", ec=CRITICAL, hatch="////", label=f"blind: fewer than {k}"),
                       mp.Patch(fc=OBSTACLE, ec=MUTED, label="obstacle")],
              loc="upper left", bbox_to_anchor=(0, -0.04), ncol=2, frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "floor.png", dpi=150)
    plt.close(fig)

    cols = min(nz, 3)
    rows = int(np.ceil(nz / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(4.2 * cols, 4.6 * rows), squeeze=False)
    for j, z in enumerate(room.z):
        ax = axs[j // cols][j % cols]
        sl = np.ma.masked_where(blk[:, :, j], n[:, :, j])
        im = heat(ax, xs, ys, sl, k, vmax)
        draw_room(ax, room, cams, label=False)
        b = np.mean(sl.compressed() < k) if sl.count() else 0
        ax.set_title(f"z = {z:.2f} m  ·  {100 * b:.1f}% blind", loc="left", fontsize=9)
    for j in range(nz, rows * cols):
        axs[j // cols][j % cols].axis("off")
    cb = fig.colorbar(im, ax=axs, ticks=ticks, shrink=0.6, label="cameras")
    cb.ax.set_yticklabels(tick_labels)
    cb.outline.set_visible(False)
    fig.savefig(out / "slices.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def main():
    a = parse()
    room = load_room(a.room)
    k = a.k or room.k
    cams = load_cameras(a.cameras)
    drop = {s.strip() for s in a.without.split(",") if s.strip()}
    unknown = drop - {c.name for c in cams}
    if unknown:
        raise SystemExit(f"--without: no camera {', '.join(sorted(unknown))}")
    cams = [c for c in cams if c.name not in drop]

    xs, ys, pts = room.grid()
    vis, blocked = visibility(cams, room, pts)
    valid = ~blocked
    out = a.out or HERE / "out" / (a.cameras.stem + (f"_without_{'_'.join(sorted(drop))}" if drop else "") + f"_k{k}")
    out.mkdir(parents=True, exist_ok=True)

    text = report(cams, room, vis, valid, k, len(room.z))
    (out / "report.md").write_text(text)
    np.savez_compressed(out / "coverage.npz", xs=xs, ys=ys, zs=room.z, vis=vis, blocked=blocked,
                        names=np.array([c.name for c in cams]))
    if not a.no_plots:
        plots(out, cams, room, xs, ys, vis, blocked, k)
    print(text)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
