#!/usr/bin/env python3
"""One figure of a survey pass: the site from above with the robot's path
coloured by its Wi-Fi RSSI and every anchor marked, the comms over time
against where the robot was, signal against link length, the CSI of a link
over the run, and the reference's own quality. And the same as an animation.

    python3 scoop_pipeline/analysis/survey_figure.py <pipeline config>
        [--out DIR] [--ap X Y Z] [--gif] [--seconds 30] [--fps 10]
        [--band 0.3 1.8] [--near 3.0] [--link infra_1]

Panels (A4 landscape, the map on the left over both rows):
  map        the anchored map between --band m above its floor, the LiDAR
             trajectory coloured by the RSSI to the access point (by time
             without comms tables), start and end, every OTHER robot's track
             (04's other LiDAR tracks: a coop run's second robot) in its own
             colour, the boards as squares with their normal (this pass's 05,
             or the passes dataset.boards_from names), the parked robot and
             the infra camera (07, or dataset.cameras_from), the AP if --ap,
             a scale bar
  timeline   RSSI, PHY rate, ping RTT and iperf goodput of the moving robot
             against time; grey bands where it was within --near m of an
             anchor, labelled
  distance   RSSI against link length for each CSI link (and the AP with
             --ap), with the path-loss fit P = P1 - 10 n log10 d
  csi        amplitude per subcarrier over time for --link's data frames,
             the RMS delay spread above it
  quality    04's residual and correction per scan, the boards' section
             spreads and loop closures
Written to <processed>/figures/survey/<tag>_overview.png (300 dpi) and .pdf;
--gif adds <tag>_overview.gif (and .mp4 with ffmpeg): the path growing, a
heading arrow, cursors sweeping the time panels. Panels whose inputs are not
there yet (the comms tables before run_pass's comms step) are left empty
with a note, so the figure can be made as the pass progresses.
"""
import argparse
import csv
import json
import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "map_stages"))
sys.path.insert(0, os.path.join(ROOT, "analysis", "comms"))

ANCHOR_NAMES = {"realsense": "mobile_2", "arducam": "infra_1"}


# ----------------------------------------------------------------- loading --
def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return math.nan


def rows_of(path):
    return list(csv.DictReader(open(path))) if os.path.exists(path) else []


def col(rows, k):
    return np.array([num(r.get(k, "")) for r in rows], float)


def quat_yaw(q):
    x, y, z, w = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def load_site(P, band=(0.3, 1.8), ap=None, link="infra_1"):
    """Everything the figure shows, from the pass's outputs: a dict of arrays
    (missing inputs -> None), see compose()."""
    import yaml
    from comms_map import floor_plan
    from pipeline_common import load_traj
    site = {"tag": P.tag, "machine": P.dataset.get("machine") or "mobile_1", "ap": ap,
            "plan": None, "extent": None}
    pcd = P.ref_file("anchored_map") if P.reference else P.outp("map_final_{tag}_anchored.pcd")
    if os.path.exists(pcd):
        site["plan"], site["extent"] = floor_plan(pcd, band)
    traj = P.lidar_track_traj()
    site["t"], site["T"] = load_traj(traj) if os.path.exists(traj) else (None, None)
    # the boards this pass works with: its own 05, the reference pass's, and
    # other passes' (dataset.boards_from) -- a coop pass has no 05 of its own
    try:
        af = json.load(open(P.anchor_frame()))
    except (FileNotFoundError, OSError, ValueError):
        af = {}
    boards = []
    for n, rec in P.boards(af.get("boards") or {}, log=lambda *a: None).items():
        if rec.get("xyz") is None or rec.get("qxyzw") is None:
            continue
        boards.append({"name": n, "xyz": rec["xyz"], "yaw": quat_yaw(rec["qxyzw"]),
                       "n_views": rec.get("n_views", 0), "std_mm": rec.get("std_mm", math.nan),
                       "sections": rec.get("sections") or [], "lc": rec.get("loop_closure")})
    site["boards"] = sorted(boards, key=lambda b: b["name"])
    statics = []
    cys = [P.outp("cameras_in_map.yaml")]
    if not os.path.exists(cys[0]):            # 07 off (a coop pass): cameras_from's
        cys = P.cameras_from() or cys
    for cy in cys:
        if not os.path.exists(cy):
            continue
        for c in (yaml.safe_load(open(cy)) or {}).get("cameras") or []:
            statics.append({"name": ANCHOR_NAMES.get(c["name"], c["name"]), "xyz": c["translation"],
                            "source": c.get("source", "")})
    site["statics"] = statics
    # the other robots: every LiDAR track 04 built besides this machine's own
    others = []
    for tr in (P.cfg.get("04_reference") or {}).get("tracks") or []:
        if not tr.get("enabled", True) or tr.get("machine") == site["machine"]:
            continue
        f = os.path.join(P.reference_dir(), "traj_%s.tum" % tr["name"])
        if os.path.exists(f):
            ot, oT = load_traj(f)
            others.append({"name": tr.get("machine") or tr["name"], "t": ot, "T": oT})
    site["others"] = others
    comms = os.path.join(P.out_dir, "comms")
    m = site["machine"]
    wifi = [r for r in rows_of(os.path.join(comms, "wifi", "wifi.csv")) if r.get("machine") == m]
    site["wifi"] = {"t": col(wifi, "t"), "rssi": col(wifi, "signal_dbm"),
                    "rate": col(wifi, "tx_bitrate_mbps")} if wifi else None
    ping = [r for r in rows_of(os.path.join(comms, "wifi", "ping.csv")) if r.get("machine") == m]
    site["ping"] = {"t": col(ping, "t"), "rtt": np.where(col(ping, "reply") == 1, col(ping, "rtt_ms"),
                                                           np.nan)} if ping else None
    iperf = [r for r in rows_of(os.path.join(comms, "wifi", "iperf.csv")) if r.get("machine") == m]
    site["iperf"] = {"t": col(iperf, "t"), "mbps": col(iperf, "bitrate_mbps"),
                     "up": np.array(["up" in (r.get("topic") or r.get("direction") or "")
                                     for r in iperf])} if iperf else None
    csi = rows_of(os.path.join(comms, "csi", "csi.csv"))
    links = {}
    for peer in sorted({r["peer"] for r in csi if r.get("peer")}):
        rr = [r for r in csi if r["peer"] == peer and num(r.get("frame_control", "")) == 0x88]
        d, p = col(rr, "distance_m"), col(rr, "rssi")
        if np.isfinite(d).any():
            links[peer] = (d, p)
    site["links"] = links or None
    zp = os.path.join(comms, "csi", "csi_%s_to_%s_sniffer_clean.npz" % (link, m))
    site["csi"] = None
    if os.path.exists(zp):
        z = np.load(zp)
        site["csi"] = {"name": link, "t": z["t"], "sub": z["subcarrier"], "amp_db": z["amp_db"],
                       "H": z["H"]}
    q = os.path.join(P.reference_dir(), "quality_%s_lidar.csv" % m) if P.reference else ""
    qr = rows_of(q)
    site["quality"] = {"t": col(qr, "t"), "residual_cm": col(qr, "residual_cm"),
                       "correction_cm": col(qr, "last_correction_cm"),
                       "ok": np.array([r["status"] == "ok" for r in qr])} if qr else None
    return site


# ----------------------------------------------------------------- drawing --
def headings(t, T, min_speed=0.2):
    """The robot's heading per pose (rad): the registered yaw, steady to a
    tenth of a degree, plus the constant offset to its direction of travel
    fitted where it moves faster than min_speed m/s. The direction of travel
    alone swings with the pose jitter whenever the robot creeps."""
    yaw = np.arctan2(T[:, 1, 0], T[:, 0, 0])
    p = T[:, :2, 3]
    dt = np.gradient(t)
    v = np.gradient(p, axis=0) / np.maximum(dt, 1e-3)[:, None]
    speed = np.linalg.norm(v, axis=1)
    mv = speed > min_speed
    if mv.sum() < 10:
        return yaw
    d = np.arctan2(v[mv, 1], v[mv, 0]) - yaw[mv]
    off = np.arctan2(np.mean(np.sin(d)), np.mean(np.cos(d)))
    return yaw + off


def near_intervals(t, P, anchors, near):
    """[(name, t0, t1)] while the path is within `near` m of an anchor."""
    out = []
    p = P[:, :2, 3]
    for name, xyz in anchors:
        d = np.linalg.norm(p - np.asarray(xyz[:2]), axis=1)
        inside = d < near
        edges = np.flatnonzero(np.diff(np.r_[False, inside, False]))
        for a, b in zip(edges[::2], edges[1::2]):
            if t[b - 1] - t[a] > 2.0:
                out.append((name, t[a], t[b - 1]))
    return out


def draw_map(ax, site, t_max=None):
    from matplotlib.collections import LineCollection
    if site["plan"] is not None:
        ax.imshow(np.ma.masked_equal(site["plan"], 0), origin="lower", extent=site["extent"],
                  cmap="Greys", vmin=0, vmax=1.6, interpolation="antialiased", zorder=0)
    t, T = site["t"], site["T"]
    lc = None
    if T is not None:
        sel = slice(None) if t_max is None else slice(0, int(np.searchsorted(t, t_max)) + 1)
        p = T[sel, :2, 3]
        if len(p) > 1:
            segs = np.stack([p[:-1], p[1:]], axis=1)
            if site["wifi"] is not None and np.isfinite(site["wifi"]["rssi"]).any():
                w = site["wifi"]
                ok = np.isfinite(w["rssi"])
                v = np.interp(t[sel][:-1], w["t"][ok], w["rssi"][ok])
                lc = LineCollection(segs, cmap="viridis", lw=2.2, zorder=2)
                lc.set_array(v)
                lc.set_clim(np.nanpercentile(w["rssi"], 2), np.nanpercentile(w["rssi"], 98))
                lc.set_label("RSSI (dBm)")
            else:
                lc = LineCollection(segs, cmap="plasma", lw=2.2, zorder=2)
                lc.set_array((t[sel][:-1] - t[0]) / 60)
                lc.set_label("time (min)")
            ax.add_collection(lc)
        ax.plot(T[0, 0, 3], T[0, 1, 3], "o", color="k", ms=6, mfc="w", zorder=4)
        ax.annotate("start", (T[0, 0, 3], T[0, 1, 3]), fontsize=7, xytext=(5, 5),
                    textcoords="offset points")
        if t_max is None:
            ax.plot(T[-1, 0, 3], T[-1, 1, 3], "s", color="k", ms=5, mfc="w", zorder=4)
            ax.annotate("end", (T[-1, 0, 3], T[-1, 1, 3]), fontsize=7, xytext=(5, -9),
                        textcoords="offset points")
    for k, o in enumerate(site.get("others") or []):      # the other robots of a coop run
        oT = o["T"]
        if t_max is not None:
            oT = oT[:int(np.searchsorted(o["t"], t_max)) + 1]
        if len(oT) < 2:
            continue
        col = "C4" if k % 2 == 0 else "C6"
        ax.plot(oT[:, 0, 3], oT[:, 1, 3], color=col, lw=1.6, alpha=0.85, zorder=2,
                label=o["name"])
        ax.plot(oT[0, 0, 3], oT[0, 1, 3], "o", color=col, ms=5, mfc="w", zorder=4)
        ax.annotate(o["name"], (oT[0, 0, 3], oT[0, 1, 3]), fontsize=7, color=col,
                    xytext=(5, -10), textcoords="offset points", zorder=6,
                    bbox=dict(boxstyle="round,pad=0.15", fc="w", ec="none", alpha=0.8))
    for b in site["boards"]:
        x, y = b["xyz"][:2]
        ax.plot(x, y, "s", color="C1", ms=7, mec="k", mew=0.5, zorder=5)
        ax.annotate("", xy=(x + 1.2 * math.cos(b["yaw"]), y + 1.2 * math.sin(b["yaw"])), xytext=(x, y),
                    arrowprops=dict(arrowstyle="-|>", color="C1", lw=1.0), zorder=5)
        ax.annotate(b["name"], (x, y), fontsize=7, color="C1", xytext=(4, -10),
                    textcoords="offset points", zorder=6,
                    bbox=dict(boxstyle="round,pad=0.15", fc="w", ec="none", alpha=0.8))
    for s in site["statics"]:
        x, y = s["xyz"][:2]
        ax.plot(x, y, "^", color="k", ms=9, mfc="w", zorder=5)
        ax.annotate(s["name"], (x, y), fontsize=7, xytext=(6, 4), textcoords="offset points", zorder=6,
                    bbox=dict(boxstyle="round,pad=0.15", fc="w", ec="0.6", alpha=0.85))
    if site.get("ap"):
        x, y = site["ap"][:2]
        ax.plot(x, y, "*", color="C3", ms=12, mec="k", mew=0.5, zorder=5)
        ax.annotate("AP", (x, y), fontsize=7, color="C3", xytext=(6, 4), textcoords="offset points")
    if site["extent"] is not None:
        x0, x1, y0, y1 = site["extent"]
        ax.set_xlim(x0 - 1, x1 + 1)
        ax.set_ylim(y0 - 1, y1 + 1)
        sx, sy = x0 + 1.0, y0 + 1.0
        ax.plot([sx, sx + 10], [sy, sy], "k-", lw=2.5, zorder=6)
        ax.annotate("10 m", (sx + 5, sy), fontsize=7, ha="center", xytext=(0, 4), textcoords="offset points")
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    return lc


def draw_timeline(axes, site, bands):
    """axes: (rssi, ping) sharing x; the iperf on rssi's twin."""
    t0 = site["t"][0] if site["t"] is not None else 0.0
    a_r, a_p = axes
    names = sorted({b[0] for b in bands})
    for name, ta, tb in bands:                       # a strip at the top: when an anchor was within reach
        col = "C%d" % (names.index(name) % 10)
        for ax, y in ((a_r, 0.96), (a_p, 0.92)):
            ax.axvspan((ta - t0) / 60, (tb - t0) / 60, ymin=y, ymax=1.0, color=col, alpha=0.6, lw=0, zorder=0)
        if tb - ta > 20:
            a_r.annotate(name, (((ta + tb) / 2 - t0) / 60, 0.95), xycoords=("data", "axes fraction"),
                         fontsize=5.5, ha="center", va="top", color=col)
    if site["wifi"] is not None:
        w = site["wifi"]
        a_r.plot((w["t"] - t0) / 60, w["rssi"], color="C0", lw=0.7, label="RSSI")
        a_r.set_ylabel("RSSI (dBm)", color="C0")
        ax2 = a_r.twinx()
        ax2.plot((w["t"] - t0) / 60, w["rate"], color="0.5", lw=0.5, label="tx PHY rate")
        if site["iperf"] is not None:
            ip = site["iperf"]
            for up, mk, lab in ((False, "v", "iperf down"), (True, "^", "iperf up")):
                s = ip["up"] == up
                ax2.plot((ip["t"][s] - t0) / 60, ip["mbps"][s], mk, color="C3", ms=4, label=lab)
        ax2.set_ylabel("Mbit/s", color="0.4")
        ax2.legend(fontsize=6, loc="lower right", ncol=3)
    else:
        a_r.text(0.5, 0.5, "comms tables not made yet", transform=a_r.transAxes, ha="center", fontsize=8)
    if site["ping"] is not None:
        pg = site["ping"]
        a_p.plot((pg["t"] - t0) / 60, pg["rtt"], ".", color="C2", ms=1.5)
        a_p.set_ylim(0, np.nanpercentile(pg["rtt"], 99) * 1.3)
        a_p.set_ylabel("ping RTT (ms)")
    a_p.set_xlabel("time into the run (min)")


def draw_distance(ax, site):
    from csi_analysis import path_loss
    if not site["links"]:
        ax.text(0.5, 0.5, "no posed CSI yet", transform=ax.transAxes, ha="center", fontsize=8)
        return
    for k, (peer, (d, p)) in enumerate(site["links"].items()):
        ok = np.isfinite(d) & np.isfinite(p) & (d > 0.3)
        ax.plot(d[ok], p[ok], ".", color="C%d" % k, ms=1.5, alpha=0.4)
        fit = path_loss(d[ok], p[ok]) if ok.any() else None
        if fit:
            n, p1 = fit["exponent"], fit["p_at_1m"]
            dd = np.geomspace(max(d[ok].min(), 0.3), d[ok].max(), 50)
            ax.plot(dd, p1 - 10 * n * np.log10(dd), "-", color="C%d" % k, lw=1.2,
                    label="%s: n = %.1f, spread %.1f dB" % (peer, n, fit["spread_db"]))
        else:
            ax.plot([], [], "-", color="C%d" % k, label=peer)
    ax.set_xscale("log")
    ax.set_xlabel("link length (m)")
    ax.set_ylabel("RSSI (dBm)")
    ax.legend(fontsize=7)


def draw_csi(ax, site, step=None):
    c = site["csi"]
    if c is None:
        ax.text(0.5, 0.5, "no cleaned CSI yet", transform=ax.transAxes, ha="center", fontsize=8)
        return
    t0 = site["t"][0] if site["t"] is not None else c["t"][0]
    t1 = site["t"][-1] if site["t"] is not None else c["t"][-1]
    nb = 600                                           # bins over the run: packets come in bursts
    edges = np.linspace(t0, t1, nb + 1)
    k = np.clip(np.searchsorted(edges, c["t"]) - 1, 0, nb - 1)
    A = c["amp_db"] - np.nanmedian(c["amp_db"], axis=1, keepdims=True)
    B = np.full((nb, A.shape[1]), np.nan)
    cnt = np.bincount(k, minlength=nb)
    for j in range(A.shape[1]):
        B[cnt > 0, j] = np.bincount(k, A[:, j], minlength=nb)[cnt > 0] / cnt[cnt > 0]
    ax.imshow(np.ma.masked_invalid(B).T, aspect="auto", origin="lower", cmap="magma", vmin=-15, vmax=10,
              extent=(0, (t1 - t0) / 60, c["sub"][0], c["sub"][-1]), interpolation="nearest")
    ax.set_facecolor("0.92")
    ax.set_ylabel("subcarrier")
    ax.set_xlabel("time into the run (min), link %s" % c["name"])
    try:
        from csi_analysis import delay_spread
        step = step or max(len(c["t"]) // 3000, 1)
        ds = delay_spread(c["H"][::step], c["sub"])
        kk = k[::step]
        med = np.full(nb, np.nan)
        for b in np.unique(kk):
            med[b] = np.nanmedian(ds[kk == b])
        ax2 = ax.twinx()
        ax2.plot((0.5 * (edges[:-1] + edges[1:]) - t0) / 60, med, color="c", lw=0.8)
        ax2.set_ylabel("delay spread (ns), median per bin", color="c")
        ax2.set_ylim(0, np.nanpercentile(ds, 99) * 1.5)
    except Exception:
        pass


def draw_quality(ax, site):
    q = site["quality"]
    if q is None:
        ax.text(0.5, 0.5, "no 04 quality file", transform=ax.transAxes, ha="center", fontsize=8)
        return
    t0 = q["t"][0]
    tm = (q["t"] - t0) / 60
    ax.plot(tm, q["residual_cm"], color="C0", lw=0.5, label="scan residual")
    ax.plot(tm, q["correction_cm"], color="C1", lw=0.5, label="last correction")
    ax.set_ylabel("cm")
    ax.set_xlabel("time into the run (min)")
    ax.set_ylim(0, max(np.nanpercentile(q["residual_cm"], 99) * 2, 0.5))
    lines = ["%d of %d scans registered" % (q["ok"].sum(), len(q["ok"]))]
    for b in site["boards"]:
        used = [s for s in b["sections"] if not s.get("gated_out")]
        lines.append("%s: %d views, %d section%s, spread %.1f mm" % (
            b["name"], b["n_views"], len(used), "" if len(used) == 1 else "s",
            b["std_mm"] if np.isfinite(b["std_mm"]) else 0.0))
        lc = b.get("lc")
        if isinstance(lc, dict):
            mm = next((v for k, v in lc.items() if "mm" in k and isinstance(v, (int, float))), None)
            if mm is not None:
                lines[-1] += ", loop closure %.1f mm" % mm
    ax.text(0.98, 0.97, "\n".join(lines), transform=ax.transAxes, fontsize=6, va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.3", fc="w", ec="0.7", alpha=0.9))
    ax.legend(fontsize=6, loc="center right", framealpha=0.9)


def caption(site):
    bits = []
    if site["t"] is not None:
        p = site["T"][:, :2, 3]
        bits.append("%s: %.0f min, %.0f m path" % (site["machine"], (site["t"][-1] - site["t"][0]) / 60,
                                                   np.linalg.norm(np.diff(p, axis=0), axis=1).sum()))
    for o in site.get("others") or []:
        p = o["T"][:, :2, 3]
        bits.append("%s: %.0f m path" % (o["name"],
                                         np.linalg.norm(np.diff(p, axis=0), axis=1).sum()))
    if site["quality"] is not None:
        bits.append("%d scans registered to the map" % site["quality"]["ok"].sum())
    if site["boards"]:
        bits.append("%d boards" % len(site["boards"]))
    if site["csi"] is not None:
        bits.append("%d CSI data frames from %s" % (len(site["csi"]["t"]), site["csi"]["name"]))
    return "; ".join(bits)


def layout(site, near=3.0):
    import matplotlib.pyplot as plt
    from matplotlib import gridspec
    fig = plt.figure(figsize=(16.5, 9.5))
    gs = gridspec.GridSpec(2, 3, figure=fig, width_ratios=[1.7, 1, 1], height_ratios=[1, 1],
                           left=0.04, right=0.97, top=0.93, bottom=0.08, wspace=0.28, hspace=0.3)
    ax_map = fig.add_subplot(gs[:, 0])
    gs_t = gs[0, 1].subgridspec(2, 1, height_ratios=[2, 1], hspace=0.08)
    ax_rssi = fig.add_subplot(gs_t[0])
    ax_ping = fig.add_subplot(gs_t[1], sharex=ax_rssi)
    ax_dist = fig.add_subplot(gs[0, 2])
    ax_csi = fig.add_subplot(gs[1, 1])
    ax_q = fig.add_subplot(gs[1, 2])
    anchors = [(b["name"], b["xyz"]) for b in site["boards"]] + [(s["name"], s["xyz"]) for s in site["statics"]]
    bands = near_intervals(site["t"], site["T"], anchors, near) if site["t"] is not None else []
    lc = draw_map(ax_map, site)
    if lc is not None:
        cb = fig.colorbar(lc, ax=ax_map, fraction=0.03, pad=0.02)
        cb.set_label(lc.get_label())
    draw_timeline((ax_rssi, ax_ping), site, bands)
    draw_distance(ax_dist, site)
    draw_csi(ax_csi, site)
    draw_quality(ax_q, site)
    ax_map.set_title("%s: site, trajectory and anchors" % site["tag"], fontsize=10)
    ax_rssi.set_title("Wi-Fi over the run", fontsize=9)
    ax_dist.set_title("signal against link length", fontsize=9)
    ax_csi.set_title("CSI amplitude (dB from the median)", fontsize=9)
    ax_q.set_title("reference quality", fontsize=9)
    fig.text(0.5, 0.015, caption(site), ha="center", fontsize=8)
    return fig, dict(map=ax_map, rssi=ax_rssi, ping=ax_ping, dist=ax_dist, csi=ax_csi, q=ax_q)


def compose(site, out_base, near=3.0):
    """<out_base>.png and .pdf."""
    import matplotlib
    matplotlib.use("Agg")
    fig, _ = layout(site, near)
    fig.savefig(out_base + ".png", dpi=300)
    fig.savefig(out_base + ".pdf")
    import matplotlib.pyplot as plt
    plt.close(fig)
    return [out_base + ".png", out_base + ".pdf"]


def animate(site, out_base, seconds=30.0, fps=10, near=3.0, mp4=True):
    """<out_base>.gif (and .mp4 with ffmpeg): the path growing, a heading
    arrow, cursors on the time panels."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import animation
    if site["t"] is None:
        return []
    t, T = site["t"], site["T"]
    fig, axes = layout(site, near)
    for coll in list(axes["map"].collections):
        coll.remove()
    growing, = axes["map"].plot([], [], color="C0", lw=2.0, zorder=2)
    head = axes["map"].annotate("", xy=(T[0, 0, 3], T[0, 1, 3]), xytext=(T[0, 0, 3], T[0, 1, 3]),
                                arrowprops=dict(arrowstyle="-|>", color="C3", lw=2), zorder=7)
    others = []                                 # the other robots, on the same clock
    for k, o in enumerate(site.get("others") or []):
        col = "C4" if k % 2 == 0 else "C6"
        ln, = axes["map"].plot([], [], color=col, lw=1.6, alpha=0.9, zorder=2)
        dot, = axes["map"].plot([], [], "o", color=col, ms=6, mfc="w", zorder=7)
        others.append((o, ln, dot))
    clock = axes["map"].text(0.02, 0.98, "", transform=axes["map"].transAxes, fontsize=10, va="top",
                             bbox=dict(boxstyle="round", fc="w", ec="0.6"))
    cursors = [axes[k].axvline(0, color="C3", lw=1.0) for k in ("rssi", "ping", "csi", "q")]
    n = max(int(seconds * fps), 2)
    stamps = np.linspace(t[0], t[-1], n)
    hd = headings(t, T)

    def frame(k):
        tk = stamps[k]
        i = min(int(np.searchsorted(t, tk)), len(t) - 1)
        growing.set_data(T[:i + 1, 0, 3], T[:i + 1, 1, 3])
        head.set_position((T[i, 0, 3], T[i, 1, 3]))
        head.xy = (T[i, 0, 3] + 1.5 * math.cos(hd[i]), T[i, 1, 3] + 1.5 * math.sin(hd[i]))
        clock.set_text("%d:%02d" % divmod(int(tk - t[0]), 60))
        for c in cursors:
            c.set_xdata([(tk - t[0]) / 60] * 2)
        art = []
        for o, ln, dot in others:
            j = int(np.searchsorted(o["t"], tk))
            if j <= 0 or o["t"][0] > tk:        # not recording yet
                ln.set_data([], [])
                dot.set_data([], [])
            else:
                j = min(j, len(o["t"]) - 1)
                ln.set_data(o["T"][:j + 1, 0, 3], o["T"][:j + 1, 1, 3])
                dot.set_data([o["T"][j, 0, 3]], [o["T"][j, 1, 3]])
            art += [ln, dot]
        return [growing, head, clock] + cursors + art

    anim = animation.FuncAnimation(fig, frame, frames=n, blit=False)
    files = []
    anim.save(out_base + ".gif", writer=animation.PillowWriter(fps=fps), dpi=72)
    files.append(out_base + ".gif")
    if mp4 and animation.writers.is_available("ffmpeg"):
        anim.save(out_base + ".mp4", writer=animation.FFMpegWriter(fps=fps, bitrate=2500), dpi=100)
        files.append(out_base + ".mp4")
    plt.close(fig)
    return files


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--out", default=None)
    ap.add_argument("--ap", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"))
    ap.add_argument("--gif", action="store_true")
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--band", type=float, nargs=2, default=[0.3, 1.8])
    ap.add_argument("--near", type=float, default=3.0, help="an anchor is 'passed' within this (m)")
    ap.add_argument("--link", default="infra_1", help="the CSI link shown over time (its transmitter)")
    a = ap.parse_args()
    from pipeline_common import load_pipeline
    P = load_pipeline(a.config)
    site = load_site(P, tuple(a.band), a.ap, a.link)
    print("map %s; path %s; boards %d; anchors %s; wifi %s; ping %s; iperf %s; CSI links %s; csi %s; "
          "quality %s" % ("yes" if site["plan"] is not None else "none",
                          "%d poses" % len(site["t"]) if site["t"] is not None else "none",
                          len(site["boards"]), ", ".join(s["name"] for s in site["statics"]) or "none",
                          "yes" if site["wifi"] else "none", "yes" if site["ping"] else "none",
                          "yes" if site["iperf"] else "none",
                          ", ".join(site["links"]) if site["links"] else "none",
                          site["csi"]["name"] if site["csi"] else "none",
                          "yes" if site["quality"] else "none"))
    out = os.path.expanduser(a.out) if a.out else os.path.join(P.out_dir, "figures", "survey")
    os.makedirs(out, exist_ok=True)
    base = os.path.join(out, "%s_overview" % P.tag)
    files = compose(site, base, a.near)
    if a.gif:
        files += animate(site, base, a.seconds, a.fps, a.near)
    print("wrote\n  " + "\n  ".join(files))


if __name__ == "__main__":
    main()
