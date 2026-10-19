#!/usr/bin/env python3
"""How the reference is made, drawn from a mapping pass's own data at one wall
spot; run through check.py:

  python3 check.py method <pipeline config> [--spot x,y,z] [--out DIR]
      [--probes 300] [--tsdf MAP] [--union MAP] [--ref TRAJ] [--input TRAJ]
      [--keyframes NPZ]

Everything comes from the pass's outputs (no bag is read):
  TSDF map    mapping/denoised_<tag>.pcd             (01)
  union map   mapping/denoised_union_<tag>.pcd       (01 --union)
  reference   02's output (traj_lidar_ba_glim.txt)   (02)
  BA input    02's input (traj_lidar.txt, GLIM with its pose-time correction)
  keyframes   <reference>_keyframes.npz              (02's cache: each keyframe's
              points in its middle pose's frame, as the BA read them)

One wall spot is chosen for all panels: a planar one-layer wall patch of the
TSDF map, seen by keyframes from at least two visits (keyframes more than
--gap s apart), its thickness the closest to the map's median wall (or the
one given with --spot). Written to <out> (default <processed>/figures/method/),
PNG (300 dpi) and PDF each, and the spot in method_spot.json:

  method_overview   the TSDF map from above (a slice at the spot's height),
                    the reference path coloured by time, the spot and a zoom
  method_thickness  the spot's 0.25 m patch in the TSDF map: its points across
                    the wall and their offsets from its plane; thickness =
                    p95 - p5 (check.py map's measure)
  method_views      bundle adjustment at the spot: the keyframes that saw it
                    (from above, by visit), and their points' offsets from the
                    plane placed with the BA's input poses and with its output
  method_fusion     the same spot in the union of points and in the TSDF map,
                    cut across the wall
  method_submaps    bundle adjustment as submaps: each visit's keyframes are one
                    submap; the wall about the spot (2 m along, at its height) from
                    above, each submap's points across the wall, before and after
                    BA, and each submap's offset from the others
  method            all of them in one figure, with the spot's thickness at
                    each step
"""
import argparse
import json
import os
import sys

import numpy as np
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "map_stages"))

import maps                                                              # noqa: E402
import trajs                                                             # noqa: E402
from figures import COLOR, INK, INK2, plt_, save                         # noqa: E402
from pipeline_common import load_traj                                    # noqa: E402

R = maps.R                 # patch radius, as check.py map
MIN_PTS = 10               # a keyframe's points at the spot to count as a view (02's min_pts)
VISIT = ["#2a78d6", "#eb6834", "#1baf7a", "#8a5cd6", "#c9971c", "#d6336c", "#2ab0c9", "#6b8a2a"]


# -------------------------------------------------------------------- inputs
def keyframe_plan(times, period):
    """As 02_bundle_adjust: the keyframe of every pose and each keyframe's
    middle pose."""
    w = np.floor((times - times[0]) / period).astype(np.int64)
    uniq, kf_of = np.unique(w, return_inverse=True)
    anchor = np.array([int(np.flatnonzero(kf_of == k)[(kf_of == k).sum() // 2])
                       for k in range(len(uniq))])
    return kf_of, anchor


def paths(P, a):
    """The pass's files, each overridable."""
    ba = P.cfg.get("02_bundle_adjust") or P.cfg.get("02b_bundle_adjust") or {}
    ref = a.ref or trajs.reference_traj(P)
    inp = a.input or (P.outp(ba["input"]) if ba.get("input") else P.seed_traj())
    kf = a.keyframes or os.path.splitext(ref)[0] + "_keyframes.npz"
    old = os.path.join(P.out_dir, "_old", os.path.relpath(kf, P.out_dir))
    if not a.keyframes and not os.path.isfile(kf) and os.path.isfile(old):
        kf = old                                     # clean_pass.py moves 02's cache to _old/
    return dict(
        tsdf=a.tsdf or P.outp(P.pcd("denoised")),
        union=a.union or P.outp("denoised_union_%s.pcd" % P.tag if P.tag else "denoised_union.pcd"),
        ref=ref, input=inp,
        keyframes=kf, config=a.config,
        kf_period=float(ba.get("kf_period", 1.0)))


def load(f):
    """Maps, trajectories, and the keyframes placed both ways."""
    for k in ("tsdf", "union", "ref", "input"):
        if not os.path.isfile(f[k]):
            raise SystemExit(f"no {k}: {f[k]}")
    if not os.path.isfile(f["keyframes"]):
        out = os.path.splitext(os.path.basename(f["ref"]))[0] + "_kf.txt"
        raise SystemExit(
            f"no keyframes: {f['keyframes']} (02's cache; clean_pass.py moves it to _old/, so it may "
            f"have been deleted with _old/). Rebuild only the keyframes, leaving the reference as it is:\n"
            f"  python3 map_stages/02_bundle_adjust.py {f['config']} --input {f['input']} "
            f"--output {out} --rounds 0\n"
            f"then add --keyframes {os.path.join(os.path.dirname(f['ref']), out[:-4] + '_keyframes.npz')}")
    tr, Tr = load_traj(f["ref"])
    ti, Ti = load_traj(f["input"])
    z = np.load(f["keyframes"], allow_pickle=False)
    clouds = np.split(z["points"].astype(np.float64), np.cumsum(z["sizes"])[:-1])
    _, anchor = keyframe_plan(ti, f["kf_period"])
    if len(anchor) != len(clouds):
        raise SystemExit(f"{len(clouds)} keyframes in {f['keyframes']} but {len(anchor)} from "
                         f"{f['input']} at {f['kf_period']:g} s: not the BA's input or period")
    kt = ti[anchor]
    before = Ti[anchor]
    if len(tr) == len(ti) and np.allclose(tr, ti):
        after = Tr[anchor]                         # 02 writes the input's stamps
    else:
        after = trajs.interp(tr, Tr, np.clip(kt, tr[0], tr[-1]))
    return dict(M=maps.load(f["tsdf"]), U=maps.load(f["union"]), tr=tr, Tr=Tr,
                clouds=clouds, kt=kt, before=before, after=after)


def world(cloud, T):
    return cloud @ T[:3, :3].T + T[:3, 3]


# ---------------------------------------------------------------------- spot
def plane_at(M, tree, c, r=R):
    """Centre, normal, along-the-wall and up axes of the map's patch at c."""
    Q = M[tree.query_ball_point(c, r)]
    c = Q.mean(0)
    n = np.linalg.svd(Q - c, full_matrices=False)[2][2]
    u = np.cross([0.0, 0.0, 1.0], n)
    u = u / np.linalg.norm(u) if np.linalg.norm(u) > 1e-6 else np.array([1.0, 0.0, 0.0])
    return c, n, u, np.cross(n, u)


def views(d, centres, r=R):
    """Per centre: {keyframe: points within r} with the keyframes placed by
    the reference."""
    tree = cKDTree(centres)
    out = [dict() for _ in centres]
    for k, (cl, T) in enumerate(zip(d["clouds"], d["after"])):
        if not len(cl):
            continue
        dist, ii = tree.query(world(cl, T), distance_upper_bound=r)
        ii = ii[np.isfinite(dist)]
        for j, cnt in zip(*np.unique(ii, return_counts=True)):
            if cnt >= MIN_PTS:
                out[j][k] = int(cnt)
    return out


def visits_of(kt, ks, gap):
    """Visit number of each keyframe in ks (sorted by time): a new visit after
    a gap longer than gap s."""
    t = kt[ks]
    return np.concatenate([[0], np.cumsum(np.diff(t) > gap)]).astype(int)


def choose(d, n_probes, gap, spot=None):
    """The spot: (centre, normal, along, up), its keyframes (sorted) and
    their visit numbers."""
    tree = cKDTree(d["M"])
    if spot is not None:
        c, n, u, v = plane_at(d["M"], tree, np.asarray(spot, float))
        ks = np.array(sorted(views(d, c[None])[0]))
        if not len(ks):
            raise SystemExit(f"no keyframe sees {spot}")
        return (c, n, u, v), ks, visits_of(d["kt"], ks, gap), "given"
    probes = maps.wall_probes(d["M"], n_probes, seed=3)
    if not probes:
        raise SystemExit("no flat wall in the TSDF map")
    z = d["M"][:, 2]
    zlo, zhi = np.percentile(z, 5) + 0.7, np.percentile(z, 95) - 0.7   # clear of floor and ceiling
    mid = [p for p in probes if zlo <= p[0][2] <= zhi]
    probes = mid if len(mid) >= 10 else probes
    C = np.array([p[0] for p in probes])
    th = np.array([p[2] for p in probes])
    seen = views(d, C)
    med = np.median(th)
    best, why = None, ""
    dirt = np.array([clean(d["M"], tree, c) for c in C])
    for need in ((2, 6, 0.02), (2, 6, 0.05), (2, 4, 0.05), (1, 4, 0.1), (1, 1, 1.0)):   # visits, keyframes, dirt
        cand = []
        for j, s in enumerate(seen):
            ks = np.array(sorted(s))
            if (len(ks) >= need[1] and dirt[j] <= need[2]
                    and visits_of(d["kt"], ks, gap).max() + 1 >= need[0]):
                cand.append(j)
        if cand:
            best = min(cand, key=lambda j: abs(th[j] - med))
            why = (f"{len(cand)} of {len(probes)} wall spots seen on >= {need[0]} visits by >= {need[1]} "
                   f"keyframes, <= {need[2] * 100:.0f} % of the cut off the wall")
            break
    if best is None:
        raise SystemExit("no wall spot is seen by any keyframe: wrong keyframes for this map?")
    c, n, u, v = plane_at(d["M"], tree, C[best])
    ks = np.array(sorted(seen[best]))
    return (c, n, u, v), ks, visits_of(d["kt"], ks, gap), why


def keyframe_offsets(d, ks, axes, r=R):
    """Per keyframe of ks: (along, up, offset) of its points at the spot,
    placed with the BA's input (before) and its output (after); the points
    are chosen once, by the output, so both show the same points."""
    c, n, u, v = axes
    out = {"before": [], "after": []}
    for k in ks:
        pa = world(d["clouds"][k], d["after"][k])
        m = np.linalg.norm(pa - c, axis=1) <= r
        pb = world(d["clouds"][k][m], d["before"][k])
        for key, p in (("before", pb), ("after", pa[m])):
            q = p - c
            out[key].append(np.stack([q @ u, q @ v, q @ n], 1))
    return out


def tiled(Q, r=R):
    """Median p95-p5 of the offset over discs of radius r in the wall's plane,
    centred every r along and up the cut: check.py map's patches, so a cut
    reads like the map's thickness."""
    if len(Q) < 10:
        return float("nan")
    t = []
    for a0 in np.arange(Q[:, 0].min() + r, Q[:, 0].max() - r + 1e-9, r):
        for h0 in np.arange(Q[:, 1].min() + r, Q[:, 1].max() - r + 1e-9, r):
            m = np.hypot(Q[:, 0] - a0, Q[:, 1] - h0) <= r
            if m.sum() >= 10:
                t.append(spread(Q[m, 2]))
    return float(np.median(t)) if t else spread(Q[:, 2])


def clean(M, tree, c, far=0.08):
    """Share of the map's points in a box about c (the cut, 30 cm deep) that
    lie more than far off the wall's plane: floor, ceiling, a corner, clutter."""
    c, n, u, v = plane_at(M, tree, c)
    Q = map_cut(M, (c, n, u, v), across=0.3)
    return float(np.mean(np.abs(Q[:, 2]) > far)) if len(Q) else 1.0


def submaps(d, ks, vis, axes, along=1.0, up=0.15, across=0.3):
    """Each visit's keyframes as one submap: their points on the wall strip
    about the spot (|along| <= along, |up| <= up), chosen once by the
    reference placement and placed both ways, in the strip's own plane
    (fitted to the TSDF map). Returns {visit: {"before": Q, "after": Q}}."""
    c, n, u, v = axes
    q = d["M"][cKDTree(d["M"]).query_ball_point(c, float(np.hypot(along, up)) + across)] - c
    m = (np.abs(q @ u) <= along) & (np.abs(q @ v) <= up) & (np.abs(q @ n) <= 0.05)
    if m.sum() >= 30:                                  # the strip's plane, not the 0.25 m patch's
        W = q[m]
        c = c + W.mean(0)
        n2 = np.linalg.svd(W - W.mean(0), full_matrices=False)[2][2]
        n = n2 * np.sign(n2 @ n)
        u = np.cross([0.0, 0.0, 1.0], n)
        u /= np.linalg.norm(u)
        v = np.cross(n, u)
    out = {}
    for k, vi in zip(ks, vis):
        pa = world(d["clouds"][k], d["after"][k]) - c
        a, h, o = pa @ u, pa @ v, pa @ n
        m = (np.abs(a) <= along) & (np.abs(h) <= up) & (np.abs(o) <= across)
        pb = world(d["clouds"][k][m], d["before"][k]) - c
        e = out.setdefault(int(vi), {"before": [], "after": []})
        e["after"].append(np.stack([a[m], h[m], o[m]], 1))
        e["before"].append(np.stack([pb @ u, pb @ v, pb @ n], 1))
    return {vi: {key: np.vstack(e[key]) for key in e} for vi, e in out.items()
            if sum(len(x) for x in e["after"]) >= 30}


def submap_gap(S, key):
    """Max - min of the submaps' median offsets (m): how far apart they put the wall."""
    meds = [np.median(e[key][:, 2]) for e in S.values()]
    return float(max(meds) - min(meds)) if len(meds) >= 2 else float("nan")


def spread(d):
    return float(np.percentile(d, 95) - np.percentile(d, 5)) if len(d) >= 10 else float("nan")


def map_patch(cloud, axes, r=R):
    c, n, u, v = axes
    q = cloud[cKDTree(cloud).query_ball_point(c, r)] - c
    return np.stack([q @ u, q @ v, q @ n], 1)


CUT = dict(along=0.5, up=0.4, across=0.15)     # the fusion cut about the spot (m)


def map_cut(cloud, axes, along=CUT["along"], up=CUT["up"], across=CUT["across"]):
    c, n, u, v = axes
    tree = cKDTree(cloud)
    q = cloud[tree.query_ball_point(c, float(np.hypot(np.hypot(along, up), across)))] - c
    a, h, o = q @ u, q @ v, q @ n
    m = (np.abs(a) <= along) & (np.abs(h) <= up) & (np.abs(o) <= across)
    return np.stack([a[m], h[m], o[m]], 1)


# -------------------------------------------------------------------- panels
def draw_overview(ax, d, axes, zoom=True):
    c = axes[0]
    M = d["M"]
    m = np.abs(M[:, 2] - c[2]) < 0.8
    S = M[m]
    if len(S) > 1_200_000:
        S = S[np.random.default_rng(0).choice(len(S), 1_200_000, replace=False)]
    ax.scatter(S[:, 0], S[:, 1], s=0.03, color=INK2, alpha=0.5, linewidths=0, rasterized=True)
    pos = d["Tr"][:, :3, 3]
    tt = d["tr"] - d["tr"][0]
    sc = ax.scatter(pos[:, 0], pos[:, 1], c=tt, s=1.5, cmap="viridis", linewidths=0, rasterized=True)
    ax.plot(*pos[0, :2], marker="o", ms=4, color=INK, zorder=5)
    ax.plot(*c[:2], marker="s", ms=7, mfc="none", mec=COLOR["tsdf"], mew=1.6, zorder=6)
    inward = np.sign(np.median(pos[:, :2], axis=0) - c[:2])          # label on the path's side
    ax.annotate("spot", c[:2], xytext=tuple(10 * inward), textcoords="offset points", color=COLOR["tsdf"],
                fontweight="bold", fontsize=8, ha="left" if inward[0] >= 0 else "right",
                va="bottom" if inward[1] >= 0 else "top")
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.grid(False)
    length = float(np.sum(np.linalg.norm(np.diff(pos, axis=0), axis=1)))
    ax.set_title(f"reference map and path ({length:.0f} m, {tt[-1]:.0f} s)")
    cb = ax.figure.colorbar(sc, ax=ax, orientation="vertical", shrink=0.8, pad=0.02, aspect=30)
    cb.set_label("time along the path (s)")
    if zoom:
        ins = ax.inset_axes([1.3, 0.25, 0.55, 0.55])
        w = 1.2
        near = (np.abs(S[:, 0] - c[0]) < w) & (np.abs(S[:, 1] - c[1]) < w)
        ins.scatter(S[near, 0], S[near, 1], s=0.5, color=INK2, linewidths=0, rasterized=True)
        e = c[:2] + np.outer([-R, R], axes[2][:2])
        ins.plot(e[:, 0], e[:, 1], color=COLOR["tsdf"], lw=2.2)
        ins.set_xlim(c[0] - w, c[0] + w)
        ins.set_ylim(c[1] - w, c[1] + w)
        ins.set_aspect("equal")
        ins.set_xticks([])
        ins.set_yticks([])
        ins.set_title("the spot, 2.4 m across", fontsize=7)
        for s in ins.spines.values():
            s.set_visible(True)
            s.set_color(COLOR["tsdf"])


def draw_thickness(ax_pts, ax_hist, P):
    a, h, o = P[:, 0], P[:, 1], P[:, 2]
    lim = max(0.02, float(np.percentile(np.abs(o), 99)))
    ax_pts.scatter(a * 100, h * 100, c=o * 100, cmap="RdBu_r", vmin=-lim * 100, vmax=lim * 100,
                   s=3, linewidths=0, rasterized=True)
    t = np.linspace(0, 2 * np.pi, 200)
    ax_pts.plot(R * 100 * np.cos(t), R * 100 * np.sin(t), color=INK2, lw=0.6, ls="--")
    ax_pts.set_aspect("equal")
    ax_pts.set_xlabel("along the wall (cm)")
    ax_pts.set_ylabel("up (cm)")
    ax_pts.set_title(f"0.25 m patch, {len(P)} points\n(colour: offset from its plane)")
    p5, p95 = np.percentile(o, [5, 95])
    ax_hist.hist(o * 100, bins=40, orientation="horizontal", color=COLOR["tsdf"], alpha=0.75)
    for p, lab in ((p5, "p5"), (p95, "p95")):
        ax_hist.axhline(p * 100, color=INK, lw=0.9, ls="--")
        ax_hist.annotate(lab, (ax_hist.get_xlim()[1], p * 100), xytext=(-2, 2),
                         textcoords="offset points", ha="right", fontsize=7)
    ax_hist.yaxis.tick_right()
    ax_hist.yaxis.set_label_position("right")
    ax_hist.set_ylabel("offset from the plane (cm)")
    ax_hist.set_xlabel("points")
    ax_hist.set_title(f"thickness\np95 $-$ p5 = {(p95 - p5) * 100:.1f} cm")
    return float(p95 - p5)


def draw_views_top(ax, d, axes, ks, vis):
    c, n, u, _ = axes
    M = d["M"]
    m = (np.abs(M[:, 2] - c[2]) < 0.25) & (np.linalg.norm(M[:, :2] - c[:2], axis=1) < 6)
    ax.scatter(M[m, 0], M[m, 1], s=0.3, color=INK2, alpha=0.5, linewidths=0, rasterized=True)
    for k, vi in zip(ks, vis):
        o = d["after"][k][:3, 3]
        col = VISIT[vi % len(VISIT)]
        ax.plot([o[0], c[0]], [o[1], c[1]], color=col, lw=0.5, alpha=0.6)
        ax.plot(o[0], o[1], marker="o", ms=3, color=col)
    e = c[:2] + np.outer([-R, R], u[:2])
    ax.plot(e[:, 0], e[:, 1], color=INK, lw=2.4)
    ax.annotate("", xy=c[:2] + 0.6 * n[:2], xytext=c[:2],
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.0))
    pts = np.array([d["after"][k][:3, 3] for k in ks])
    lo, hi = np.minimum(pts[:, :2].min(0), c[:2]) - 0.5, np.maximum(pts[:, :2].max(0), c[:2]) + 0.5
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_aspect("equal")
    ax.grid(False)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(f"{len(ks)} keyframes on {vis.max() + 1} visits saw the spot")


def draw_views_offsets(ax, offs, vis, title, lim):
    allo = []
    for q, vi in zip(offs, vis):
        if len(q):
            ax.scatter(q[:, 0] * 100, q[:, 2] * 100, s=4, color=VISIT[vi % len(VISIT)],
                       linewidths=0, alpha=0.8, rasterized=True)
            allo.append(q[:, 2])
    o = np.concatenate(allo) if allo else np.zeros(0)
    th = spread(o)
    if len(o) >= 10:
        p5, p95 = np.percentile(o, [5, 95])
        ax.axhspan(p5 * 100, p95 * 100, color=INK2, alpha=0.1, lw=0)
    ax.axhline(0, color=INK2, lw=0.6, ls="--")
    ax.set_ylim(-lim, lim)
    ax.set_xlim(-R * 100, R * 100)
    ax.set_xlabel("along the wall (cm)")
    ax.set_title(f"{title}: {th * 100:.1f} cm")
    return th


def draw_fusion(ax, Q, color, title):
    a, h, o = Q[:, 0], Q[:, 1], Q[:, 2]
    th = tiled(Q)
    if len(o) >= 10:
        p5, p95 = np.percentile(o, [5, 95])
        ax.axvspan(p5 * 100, p95 * 100, color=color, alpha=0.15, lw=0)
    ax.scatter(o * 100, h, s=1.5, color=color, linewidths=0, rasterized=True)
    ax.set_xlim(-CUT["across"] * 100, CUT["across"] * 100)
    ax.set_xlabel("across the wall (cm)")
    ax.set_title(f"{title}: {th * 100:.1f} cm")
    return th


def draw_submap_strip(ax, S, key, title, lim):
    for vi, e in sorted(S.items()):
        Q = e[key]
        col = VISIT[vi % len(VISIT)]
        ax.scatter(Q[:, 0], Q[:, 2] * 100, s=1.2, color=col, linewidths=0, alpha=0.6, rasterized=True)
        ax.axhline(np.median(Q[:, 2]) * 100, color=col, lw=1.0)
    ax.axhline(0, color=INK2, lw=0.6, ls="--")
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("along the wall (m)")
    gap = submap_gap(S, key)
    ax.set_title(f"{title}: submaps {gap * 100:.1f} cm apart")
    return gap


def draw_submap_profiles(ax, S, lim):
    bins = np.linspace(-lim, lim, 61)
    for vi, e in sorted(S.items()):
        col = VISIT[vi % len(VISIT)]
        for key, ls in (("before", ":"), ("after", "-")):
            h, b = np.histogram(e[key][:, 2] * 100, bins=bins, density=True)
            ax.plot(h, 0.5 * (b[1:] + b[:-1]), color=col, lw=1.0, ls=ls)
    ax.plot([], [], color=INK2, ls=":", label="before BA")
    ax.plot([], [], color=INK2, ls="-", label="after BA")
    ax.legend(loc="upper right", fontsize=6.5, frameon=False)
    ax.axhline(0, color=INK2, lw=0.6, ls="--")
    ax.set_ylim(-lim, lim)
    ax.set_xticks([])
    ax.set_xlabel("points (per submap)")
    ax.set_title("across the wall")


def draw_steps(ax, steps):
    names = [s[0] for s in steps]
    vals = [s[1] * 100 for s in steps]
    cols = [s[2] for s in steps]
    y = np.arange(len(steps))
    ax.barh(y, vals, color=cols, height=0.6)
    for i, v in enumerate(vals):
        ax.annotate(f"{v:.1f} cm", (v, i), xytext=(3, 0), textcoords="offset points",
                    va="center", fontsize=7.5)
    ax.set_yticks(y, names)
    ax.invert_yaxis()
    ax.set_xlim(0, max(vals) * 1.3 if vals else 1)
    ax.set_xlabel("p95 $-$ p5 (cm)")
    ax.set_title("the spot, step by step")
    ax.grid(axis="y", visible=False)


# ---------------------------------------------------------------------- main
def make(d, axes, ks, vis, out, tag=""):
    plt = plt_()
    offs = keyframe_offsets(d, ks, axes)
    patch = map_patch(d["M"], axes)
    cut_u, cut_t = map_cut(d["U"], axes), map_cut(d["M"], axes)
    lim = max(4.0, 100 * float(np.percentile(np.abs(np.concatenate(
        [q[:, 2] for q in offs["before"] if len(q)] or [np.zeros(1)])), 98)) * 1.15)
    files = []

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    draw_overview(ax, d, axes)
    files.append(save(fig, out, "method_overview"))

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(5.2, 2.9), gridspec_kw={"width_ratios": [1.15, 1]})
    th_map = draw_thickness(a1, a2, patch)
    files.append(save(fig, out, "method_thickness"))

    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(7.16, 2.7))
    draw_views_top(a1, d, axes, ks, vis)
    th_b = draw_views_offsets(a2, offs["before"], vis, "before BA (GLIM corrected)", lim)
    th_a = draw_views_offsets(a3, offs["after"], vis, "after BA (reference)", lim)
    a2.set_ylabel("offset from the plane (cm)")
    a3.sharey(a2)
    files.append(save(fig, out, "method_views"))

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(5.2, 3.0), sharey=True)
    th_u = draw_fusion(a1, cut_u, COLOR["union"], "union of points")
    th_t = draw_fusion(a2, cut_t, COLOR["tsdf"], "TSDF (reference map)")
    a1.set_ylabel("height about the spot (m)")
    files.append(save(fig, out, "method_fusion"))

    S = submaps(d, ks, vis, axes)
    gap_b = gap_a = float("nan")
    if len(S) >= 2:
        allo = np.concatenate([e[k][:, 2] for e in S.values() for k in ("before", "after")])
        slim = max(4.0, 100 * float(np.percentile(np.abs(allo), 98)) * 1.15)
        fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(7.16, 2.8), sharey=True,
                                         gridspec_kw={"width_ratios": [1, 1, 0.45]})
        gap_b = draw_submap_strip(a1, S, "before", "before BA", slim)
        gap_a = draw_submap_strip(a2, S, "after", "after BA", slim)
        draw_submap_profiles(a3, S, slim)
        a1.set_ylabel("offset from the wall (cm)")
        fig.suptitle(f"{len(S)} submaps (one per visit) of the wall about the spot, 2 m along, from above",
                     fontsize=8, y=1.0)
        files.append(save(fig, out, "method_submaps"))

    steps = [("before BA", th_b, COLOR["glim_corr"]),
             ("after BA", th_a, COLOR["glim_ba"]),
             ("union", th_u, COLOR["union"]),
             ("TSDF", th_t, COLOR["tsdf"])]
    fig = plt.figure(figsize=(7.16, 8.4))
    gs = fig.add_gridspec(3, 12, height_ratios=[1.25, 1, 1], hspace=0.62, wspace=2.2)
    draw_overview(fig.add_subplot(gs[0, :6]), d, axes, zoom=False)
    draw_thickness(fig.add_subplot(gs[0, 7:10]), fig.add_subplot(gs[0, 10:]), patch)
    draw_views_top(fig.add_subplot(gs[1, :4]), d, axes, ks, vis)
    b2 = fig.add_subplot(gs[1, 4:8])
    b3 = fig.add_subplot(gs[1, 8:], sharey=b2)
    draw_views_offsets(b2, offs["before"], vis, "before BA", lim)
    draw_views_offsets(b3, offs["after"], vis, "after BA", lim)
    b2.set_ylabel("offset (cm)")
    c1 = fig.add_subplot(gs[2, :4])
    c2 = fig.add_subplot(gs[2, 4:8], sharey=c1)
    draw_fusion(c1, cut_u, COLOR["union"], "union")
    draw_fusion(c2, cut_t, COLOR["tsdf"], "TSDF")
    c1.set_ylabel("height about the spot (m)")
    draw_steps(fig.add_subplot(gs[2, 9:]), steps)
    if tag:
        fig.suptitle(f"how the reference is made, at one wall spot of {tag}", y=0.995, fontsize=9)
    files.append(save(fig, out, "method"))
    return files, dict(map_patch_cm=th_map * 100, keyframes_before_cm=th_b * 100,
                       keyframes_after_cm=th_a * 100, union_cut_cm=th_u * 100, tsdf_cut_cm=th_t * 100,
                       submaps=len(S), submaps_apart_before_cm=gap_b * 100,
                       submaps_apart_after_cm=gap_a * 100)


def main_method(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--spot", help="x,y,z of the wall spot (map frame); default: chosen")
    ap.add_argument("--probes", type=int, default=300, help="wall spots tried")
    ap.add_argument("--gap", type=float, default=30.0, help="s between keyframes that starts a new visit")
    ap.add_argument("--out")
    for k in ("tsdf", "union", "ref", "input", "keyframes"):
        ap.add_argument("--" + k)
    a = ap.parse_args(argv)
    from pipeline_common import load_pipeline
    P = load_pipeline(a.config)
    f = paths(P, a)
    out = a.out or os.path.join(P.out_dir, "figures", "method")
    print(f"method figures of {P.tag} -> {out}")
    for k in ("tsdf", "union", "ref", "input", "keyframes"):
        print(f"  {k:9s} {f[k]}")
    d = load(f)
    spot = [float(x) for x in a.spot.split(",")] if a.spot else None
    axes, ks, vis, why = choose(d, a.probes, a.gap, spot)
    c = axes[0]
    print(f"  spot ({c[0]:.2f}, {c[1]:.2f}, {c[2]:.2f}): {len(ks)} keyframes, {vis.max() + 1} visits ({why})")
    files, th = make(d, axes, ks, vis, out, P.tag)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "method_spot.json"), "w") as fh:
        json.dump(dict(centre=c.tolist(), normal=axes[1].tolist(), keyframes=ks.tolist(),
                       visits=int(vis.max() + 1), chosen=why, thickness=th, files=f), fh, indent=1)
    print("  thickness at the spot (cm): " + ", ".join(f"{k} {v:.1f}" for k, v in th.items()))
    print(f"  again with this spot: --spot {c[0]:.3f},{c[1]:.3f},{c[2]:.3f}")
    return files


if __name__ == "__main__":
    main_method()
