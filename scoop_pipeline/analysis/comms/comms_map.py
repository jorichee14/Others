#!/usr/bin/env python3
"""The comms measurements on the map, top view: an RF map of the pass.

    python scoop_pipeline/analysis/comms/comms_map.py <processed pass folder>
        [--config FILE] [--grid 0.5] [--map PCD] [--band 0.3 1.8]

Background: the anchored map (the reference pass's, for a run) seen from
above, the points between --band m above its floor (walls and furniture,
no floor or ceiling). On it: each moving robot's path with its heading every
few metres (mobile_1 from its LiDAR track, the others -- mobile_2 in a coop
run -- from their 04 tracks), the parked robots and the infra node where they
stood, the boards. Panels, from the comms tables (processing/
comms_tables.py, made with positions):

  NTP     the moving machines' clock offset measured at each poll (the CSI
          sniffer rides on mobile_1; mobile_1 is the reference), the parked
          machines labelled with their median |offset|
  Wi-Fi   RSSI of a moving robot, median per --grid m cell; the others
          labelled with theirs
  ping    RTT of a moving robot, median per cell (log scale), cells that
          lost a ping outlined
  iperf   goodput where each test ran: down circles, up triangles

With more than one robot moving (coop), a Wi-Fi and a ping panel per robot,
on one colour scale, its own path dark and the other's light.

Writes comms/maps/: comms_map.png / .pdf (all panels), one figure per panel
(ntp / rssi / ping / iperf .png; rssi_<machine> / ping_<machine> with more
than one robot moving), and <metric>_grid[_<machine>].csv (cell centre,
median, count) for other tools.
"""
import argparse
import csv
import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "map_stages"))


def rows_of(comms, sub, name):
    p = os.path.join(comms, sub, name)
    if not os.path.exists(p):
        return []
    return list(csv.DictReader(open(p)))


def num(r, k):
    v = r.get(k, "")
    if v in ("", None):
        return math.nan
    if v in ("True", "true"):
        return 1.0
    if v in ("False", "false"):
        return 0.0
    try:
        return float(v)
    except ValueError:
        return math.nan


def arr(rows, k):
    return np.array([num(r, k) for r in rows], float)


def by_machine(rows):
    out = {}
    for r in rows:
        out.setdefault(r["machine"], []).append(r)
    return out


def moving(rows):
    x, y = arr(rows, "x"), arr(rows, "y")
    return np.isfinite(x).any() and (np.nanstd(x) > 0.3 or np.nanstd(y) > 0.3)


def place(rows):
    x, y = arr(rows, "x"), arr(rows, "y")
    return (float(np.nanmedian(x)), float(np.nanmedian(y))) if np.isfinite(x).any() else None


def floor_plan(pcd_path, band, res=0.08):
    """(image, extent) of the map's points between band[0] and band[1] m above
    its floor (the floor: the most populated height in the lower 40 %)."""
    import open3d as o3d
    p = np.asarray(o3d.io.read_point_cloud(pcd_path).points)
    if not len(p):
        return None, None
    c, e = np.histogram(p[:, 2], bins=100)
    lo = max(int(len(c) * 0.4), 1)
    floor = 0.5 * (e[:-1] + e[1:])[int(np.argmax(c[:lo]))]
    q = p[(p[:, 2] > floor + band[0]) & (p[:, 2] < floor + band[1])]
    x0, y0 = q[:, 0].min(), q[:, 1].min()
    nx = int((q[:, 0].max() - x0) / res) + 1
    ny = int((q[:, 1].max() - y0) / res) + 1
    img = np.zeros((ny, nx))
    np.add.at(img, (((q[:, 1] - y0) / res).astype(int), ((q[:, 0] - x0) / res).astype(int)), 1)
    # walls are a cell or two thick: thicken them so they survive the figure's
    # downsampling, and show occupied cells, not point counts
    from scipy.ndimage import maximum_filter
    occ = maximum_filter((img >= 2).astype(float), size=2)
    return occ, (x0, x0 + nx * res, y0, y0 + ny * res)


def grid(x, y, v, cell, fn=np.nanmedian):
    """[(cx, cy, value, n)] per occupied cell."""
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(v)
    if not ok.any():
        return []
    ix, iy = np.floor(x[ok] / cell).astype(int), np.floor(y[ok] / cell).astype(int)
    out = {}
    for i, j, val in zip(ix, iy, v[ok]):
        out.setdefault((i, j), []).append(val)
    return [((i + 0.5) * cell, (j + 0.5) * cell, float(fn(np.array(vs))), len(vs))
            for (i, j), vs in sorted(out.items())]


def write_grid(path, cells, name):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["x", "y", name, "n"])
        for c in cells:
            w.writerow(["%.3f" % c[0], "%.3f" % c[1], "%.6g" % c[2], c[3]])


def context(P):
    """Map cloud, the moving robots' paths {machine: (N, 4, 4)}, the boards
    [(name, x, y)]."""
    import json
    from pipeline_common import load_traj
    pcd = P.ref_file("anchored_map") if P.reference else P.outp("map_final_{tag}_anchored.pcd")
    paths = {}
    traj = P.lidar_track_traj()
    if os.path.exists(traj):
        paths[P.dataset.get("machine") or "mobile_1"] = load_traj(traj)[1]
    for tr in (P.cfg.get("04_reference") or {}).get("tracks") or []:
        p = os.path.join(P.reference_dir(), "traj_%s.tum" % tr["name"])
        if tr.get("enabled", True) and os.path.exists(p):
            paths[tr.get("machine") or tr["name"]] = load_traj(p)[1]
    boards = []
    files = [P.anchor_frame()] + ([P.outp("boards_{tag}.json")] if P.reference else [])
    seen = set()
    for f in files:
        if os.path.exists(f):
            for n, rec in (json.load(open(f)).get("boards") or {}).items():
                if n not in seen:
                    seen.add(n)
                    boards.append((n, rec["xyz"][0], rec["xyz"][1]))
    return (pcd if os.path.exists(pcd) else None), paths, boards


def draw_base(ax, plan, extent, paths, boards, statics, focus=None):
    """paths {machine: (N, 4, 4)}: focus's (all, without one) dark, the rest light."""
    if plan is not None:
        ax.imshow(np.ma.masked_equal(plan, 0), origin="lower", extent=extent, cmap="Greys",
                  vmin=0, vmax=1.6, interpolation="antialiased", zorder=0)
    for m, path in paths.items():
        col = "0.15" if focus in (None, m) else "0.6"
        p = path[:, :2, 3]
        ax.plot(p[:, 0], p[:, 1], color=col, lw=0.6, zorder=2)
        ax.plot(p[0, 0], p[0, 1], "o", color=col, ms=4, zorder=3)
        if len(paths) > 1:
            ax.annotate(m, (p[0, 0], p[0, 1]), fontsize=7, color=col, xytext=(4, 4),
                        textcoords="offset points", zorder=6)
        d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
        for s in np.arange(2.0, d[-1], 4.0):
            i = int(np.searchsorted(d, s))
            j = min(i + 3, len(p) - 1)
            v = p[j] - p[max(i - 3, 0)]
            if np.linalg.norm(v) > 1e-3:
                v = v / np.linalg.norm(v) * 0.6
                ax.annotate("", xy=p[i] + v, xytext=p[i], zorder=3,
                            arrowprops=dict(arrowstyle="-|>", color=col, lw=0.6))
    for n, x, y in boards:
        ax.plot(x, y, "s", color="C1", ms=4, zorder=4)
        ax.annotate(n, (x, y), fontsize=6, color="C1", xytext=(3, -8), textcoords="offset points")
    for name, (xy, label) in statics.items():
        ax.plot(*xy, "^", color="k", ms=8, mfc="w", zorder=5)
        ax.annotate(f"{name}\n{label}", xy, fontsize=7, xytext=(6, 4), textcoords="offset points",
                    zorder=6, bbox=dict(boxstyle="round,pad=0.2", fc="w", ec="0.6", alpha=0.85))
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")


def draw_cells(ax, cells, cell, cmap, norm, outline=None):
    from matplotlib.patches import Rectangle
    from matplotlib.collections import PatchCollection
    patches = [Rectangle((cx - cell / 2, cy - cell / 2), cell, cell) for cx, cy, _, _ in cells]
    pc = PatchCollection(patches, cmap=cmap, norm=norm, alpha=0.9, zorder=1.5, lw=0)
    pc.set_array(np.array([c[2] for c in cells]))
    ax.add_collection(pc)
    if outline:
        for cx, cy in outline:
            ax.add_patch(Rectangle((cx - cell / 2, cy - cell / 2), cell, cell, fill=False,
                                   ec="red", lw=0.8, zorder=1.6))
    return pc


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("processed")
    ap.add_argument("--config", default=None)
    ap.add_argument("--grid", type=float, default=0.5, help="cell size (m)")
    ap.add_argument("--map", default=None, help="the cloud for the background")
    ap.add_argument("--band", type=float, nargs=2, default=[0.3, 1.8],
                    help="heights above the floor shown (m)")
    a = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, Normalize
    import pass_tf

    proc = os.path.abspath(os.path.expanduser(a.processed))
    comms = os.path.join(proc, "comms")
    out = os.path.join(comms, "maps")
    os.makedirs(out, exist_ok=True)
    cfg = a.config or pass_tf.find_config(proc)
    pcd, paths, boards = (None, {}, [])
    if cfg:
        pcd, paths, boards = context(pass_tf.load_pipeline(cfg))
    pcd = a.map or pcd
    plan, extent = floor_plan(pcd, a.band) if pcd else (None, None)
    print(f"background: {pcd or 'none'}; paths: "
          f"{', '.join('%s %d poses' % (m, len(p)) for m, p in paths.items()) or 'none'}; "
          f"boards: {len(boards)}")

    wifi = by_machine(rows_of(comms, "wifi", "wifi.csv"))
    ping = by_machine(rows_of(comms, "wifi", "ping.csv"))
    iperf = by_machine(rows_of(comms, "wifi", "iperf.csv"))
    ntp = by_machine(rows_of(comms, "ntp", "ntp.csv"))
    if not any(r and "x" in r[0] for d in (wifi, ping, ntp) for r in d.values()):
        sys.exit("the comms tables have no positions: run processing/comms_tables.py with the "
                 "pass's config (run_pass.py --only comms)")

    def statics_of(table, label):
        out = {}
        for m, rows in table.items():
            if "x" in rows[0] and not moving(rows) and place(rows):
                out[m] = (place(rows), label(rows))
        return out

    panels = []

    # NTP: the moving machine's measured offset at polls
    mv = [m for m, rows in ntp.items() if "x" in rows[0] and moving(rows)]
    pts = []
    for m in mv:
        rows = ntp[m]
        ref = arr(rows, "reference_time")
        new = np.r_[True, np.diff(ref) != 0]
        sel = [r for r, k in zip(rows, new) if k]
        pts.append((m, arr(sel, "x"), arr(sel, "y"), np.abs(arr(sel, "last_offset_seconds")) * 1e6))
    med_us = lambda rows: "%.0f us" % (np.nanmedian(np.abs(arr(rows, "last_offset_seconds"))) * 1e6)  # noqa: E731
    st = statics_of(ntp, lambda rows: ("|offset| " + med_us(rows)) if "7F7F" not in rows[0].get(
        "sync_source", "") else "reference clock")
    panels.append(("ntp", "NTP: |measured clock offset| (us)", pts, st, None))

    def movers(table):
        return [m for m, rows in table.items() if "x" in rows[0] and moving(rows)]

    def tag(key, m, ms):
        return key if len(ms) <= 1 else f"{key}_{m}"

    ms = movers(wifi)
    st = statics_of(wifi, lambda rows: "RSSI %.0f dBm" % np.nanmedian(arr(rows, "signal_dbm")))
    for m in ms or [None]:
        cells = grid(arr(wifi[m], "x"), arr(wifi[m], "y"), arr(wifi[m], "signal_dbm"), a.grid) \
            if m else []
        if cells:
            write_grid(os.path.join(out, tag("rssi_grid", m, ms) + ".csv"), cells, "rssi_dbm")
        panels.append((tag("rssi", m, ms), f"Wi-Fi: {m} RSSI (dBm), median per {a.grid:g} m cell",
                       cells, st, m))

    ms = movers(ping)
    st = statics_of(ping, lambda rows: "RTT %.1f ms" % np.nanmedian(
        arr([r for r in rows if num(r, "reply") == 1], "rtt_ms")))
    for m in ms or [None]:
        cells, lost = [], []
        if m:
            rows = ping[m]
            rep = arr(rows, "reply") == 1
            x, y, rtt = arr(rows, "x"), arr(rows, "y"), arr(rows, "rtt_ms")
            cells = grid(x[rep], y[rep], rtt[rep], a.grid)
            write_grid(os.path.join(out, tag("ping_rtt_grid", m, ms) + ".csv"), cells, "rtt_ms")
            # a cell lost a ping when one of its seqs never got a reply
            seq = arr(rows, "icmp_seq")
            got = set(seq[rep])
            miss = ~rep & np.array([s not in got for s in seq])
            lost = sorted({(c[0], c[1]) for c in grid(x[miss], y[miss], np.ones(miss.sum()), a.grid)})
        panels.append((tag("ping", m, ms),
                       f"ping: {m} RTT (ms), median per {a.grid:g} m cell; red: a ping lost",
                       (cells, lost), st, m))

    ip = []
    for m, rows in iperf.items():
        ok = [r for r in rows if num(r, "success") == 1]
        ip.append((m, arr(ok, "x"), arr(ok, "y"), arr(ok, "bitrate_mbps"), arr(ok, "reverse") == 1))
    panels.append(("iperf", "iperf goodput (Mbit/s): o down, ^ up", ip, {}, None))

    # one colour scale per metric across the robots' panels
    vals = {"rssi": [c[2] for p in panels if p[0].startswith("rssi") for c in p[2]],
            "ping": [c[2] for p in panels if p[0].startswith("ping") for c in p[2][0]]}

    def draw(ax, key, title, data, statics, focus, fig):
        draw_base(ax, plan, extent, paths, boards, statics, focus)
        ax.set_title(title, fontsize=9)
        if key == "ntp":
            allv = np.concatenate([p[3] for p in data]) if data else np.array([])
            if len(allv):
                norm = Normalize(0, np.nanpercentile(allv, 98))
                for m, x, y, v in data:
                    sc = ax.scatter(x, y, c=v, s=10, cmap="magma_r", norm=norm, zorder=4, label=m)
                fig.colorbar(sc, ax=ax, shrink=0.8, label="us")
        elif key.startswith("rssi") and data:
            v = np.array(vals["rssi"])
            pc = draw_cells(ax, data, a.grid, "viridis", Normalize(np.percentile(v, 2), np.percentile(v, 98)))
            fig.colorbar(pc, ax=ax, shrink=0.8, label="dBm")
        elif key.startswith("ping") and data[0]:
            v = np.array(vals["ping"])
            lo, hi = max(np.percentile(v, 2), 0.1), np.percentile(v, 98)
            pc = draw_cells(ax, data[0], a.grid, "plasma",
                            LogNorm(lo, hi) if hi > 1.5 * lo else Normalize(lo, hi), outline=data[1])
            cb = fig.colorbar(pc, ax=ax, shrink=0.8, label="ms")
            from matplotlib.ticker import ScalarFormatter
            cb.ax.yaxis.set_major_formatter(ScalarFormatter())
            cb.ax.yaxis.set_minor_formatter(ScalarFormatter())
        elif key == "iperf" and data:
            allv = np.concatenate([d[3] for d in data])
            if len(allv):
                norm = Normalize(np.nanmin(allv), np.nanmax(allv))
                for m, x, y, v, down in data:
                    sc = ax.scatter(x[down], y[down], c=v[down], s=40, marker="o", cmap="viridis",
                                    norm=norm, edgecolors="k", lw=0.4, zorder=4)
                    ax.scatter(x[~down], y[~down], c=v[~down], s=40, marker="^", cmap="viridis",
                               norm=norm, edgecolors="k", lw=0.4, zorder=4)
                fig.colorbar(sc, ax=ax, shrink=0.8, label="Mbit/s")

    # robots' panels side by side: rssi, ping, then ntp and iperf
    order = [p for p in panels if p[0].startswith("rssi")] + \
        [p for p in panels if p[0].startswith("ping")] + [p for p in panels if p[0] in ("ntp", "iperf")]
    nr = (len(order) + 1) // 2
    fig, axs = plt.subplots(nr, 2, figsize=(16, 6.5 * nr), squeeze=False)
    for ax, (key, title, data, statics, focus) in zip(axs.ravel(), order):
        draw(ax, key, title, data, statics, focus, fig)
    for ax in axs.ravel()[len(order):]:
        ax.axis("off")
    fig.tight_layout()
    written = []
    for ext in ("png", "pdf"):
        p = os.path.join(out, f"comms_map.{ext}")
        fig.savefig(p, dpi=200)
        written.append(p)
    plt.close(fig)
    for key, title, data, statics, focus in panels:
        fig, ax = plt.subplots(figsize=(9, 8))
        draw(ax, key, title, data, statics, focus, fig)
        fig.tight_layout()
        p = os.path.join(out, f"{key}.png")
        fig.savefig(p, dpi=200)
        plt.close(fig)
        written.append(p)
    for p in written + sorted(os.path.join(out, f) for f in os.listdir(out) if f.endswith(".csv")):
        print(f"  {p}")


if __name__ == "__main__":
    main()
