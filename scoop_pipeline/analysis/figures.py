#!/usr/bin/env python3
"""Figures of a mapping pass, one per step that is done; run through check.py:

  python3 check.py figures <pipeline config> [--only timing,ba,...] [--bag]
      [--out DIR] [--samples 8000]

Each figure is made only when the step's outputs are there (else it says
which file is missing), as PNG (300 dpi) and PDF in <processed>/figures/:

  timing        GLIM's heading error against the reference vs the yaw rate,
                as output and with its pose-time correction: the slope is the
                time offset (step: GLIM export + time check)
  ba            the reference path coloured by how far bundle adjustment
                moved GLIM's poses, and the planes' thickness round by round
                (step: 02, its <output>_report.json)
  trajectories  each trajectory's position error against the reference over
                the run, after the best rigid fit (steps: GLIM, 02b, 02)
  maps          the reference map from above, and its wall thickness against
                the union of points (step: 01 and 01 --union)
  wall          one wall cut across, in the TSDF map and in the union: the
                band that holds 90 % of its points (step: 01 and 01 --union)
  reference     a run's registration: how far each round moved the scans
                against the gate, and every board's spread against the gate
                it had to pass (steps: 04 summary_<track>.json, 05)
  loop          each trajectory's board loop closure against the detection
                noise (the <traj>_loop_closure.json reports)
  stages        a wall's thickness as the scans pile up, one scan to the
                whole run, GLIM against the reference (needs check.py walls
                --json <out>/walls_glim.json and walls_ref.json)
  noise         the sensor alone: one scan's wall thickness by range
                (--bag: reads 50 scans of the bag)
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "map_stages"))

import maps                                                              # noqa: E402
import trajs                                                             # noqa: E402
from pipeline_common import load_pipeline, load_traj                     # noqa: E402

# one colour per entity, the same in every figure (categorical slots 1-3 of
# the validated default palette; GLIM as output and the union are grey)
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
COLOR = {"glim": "#8a8984", "glim_corr": "#2a78d6", "glim_s2m": "#1baf7a",
         "glim_ba": "#eb6834", "union": "#8a8984", "tsdf": "#eb6834"}
SHORT = {"glim": "GLIM as output", "glim_corr": "GLIM corrected",
         "glim_s2m": "GLIM + scan-to-map", "glim_ba": "GLIM + BA (reference)"}
FIGS = ("timing", "ba", "trajectories", "reference", "maps", "wall", "loop",
        "stages", "noise")


def plt_():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8, "legend.fontsize": 7.5,
        "axes.edgecolor": INK2, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "lines.linewidth": 1.4, "axes.axisbelow": True,
        "savefig.bbox": "tight", "savefig.facecolor": "white"})
    return plt


def save(fig, out, name):
    os.makedirs(out, exist_ok=True)
    base = os.path.join(out, name)
    fig.savefig(base + ".png", dpi=300)
    fig.savefig(base + ".pdf")
    import matplotlib.pyplot as plt
    plt.close(fig)
    print(f"  {name}: {base}.png, .pdf")
    return base + ".png"


def need(*paths):
    """The first of paths that is missing, or None."""
    return next((p for p in paths if not p or not os.path.exists(p)), None)


def fitted_error(t, T, tr, Tr):
    """(t, position error m) of (t, T) against the reference, after the best
    rigid fit of the reference onto it (the ATE's per-pose error)."""
    lo, hi = max(t[0], tr[0]), min(t[-1], tr[-1])
    m = (t >= lo) & (t <= hi)
    t, A = t[m], T[m]
    B = trajs.interp(tr, Tr, t)
    R, tt = trajs.rigid_fit(B[:, :3, 3], A[:, :3, 3])
    return t, np.linalg.norm(A[:, :3, 3] - (B[:, :3, 3] @ R.T + tt), axis=1)


# ------------------------------------------------------------------- figures
def fig_timing(ctx):
    plt = plt_()
    rows = {k: f for k, _, f in ctx["rows"]}
    if "glim_corr" not in rows:
        return "no pose-time correction applied (glim.pose_time_offset 0)"
    miss = need(rows["glim"], rows["glim_corr"])
    if miss:
        return f"no {miss}"
    tr, Tr = ctx["ref"]
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    lim = 0
    for key in ("glim", "glim_corr"):
        t, T = load_traj(rows[key])
        _, w, e, _, _ = trajs.heading_turning_series(t, T, tr, Tr)
        h = trajs.heading_vs_turning(t, T, tr, Tr)
        wd, ed = np.degrees(w), np.degrees(e)
        k = max(1, len(wd) // 4000)
        ax.scatter(wd[::k], ed[::k], s=3, color=COLOR[key], alpha=0.35, linewidths=0, rasterized=True)
        lim = max(lim, np.percentile(np.abs(wd), 99.5))
        x = np.array([-lim, lim])
        ax.plot(x, x * h["dt_ms"] / 1000, color=COLOR[key],
                label=f"{SHORT[key]}: {h['dt_ms']:+.0f} ms (r {h['r_timing']:.2f})")
    ax.set_xlim(-lim, lim)
    ax.set_xlabel("yaw rate of the reference (deg/s)")
    ax.set_ylabel("heading error against it (deg)")
    ax.set_title("GLIM's pose time: heading error follows the turning")
    ax.legend(loc="upper left")
    return save(fig, ctx["out"], "timing")


def fig_ba(ctx):
    plt = plt_()
    rows = {k: f for k, _, f in ctx["rows"]}
    ref_f = rows["glim_ba"]
    rep = os.path.splitext(ref_f)[0] + "_report.json"
    corr = rows.get("glim_corr", rows["glim"])
    miss = need(rep, corr)
    if miss:
        return f"no {miss}"
    tr, Tr = ctx["ref"]
    t, T = load_traj(corr)
    te, err = fitted_error(t, T, tr, Tr)
    P = trajs.interp(tr, Tr, te)[:, :3, 3]
    r = json.load(open(rep))
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.16, 3.2), gridspec_kw={"width_ratios": [1.25, 1]})
    sc = a.scatter(P[:, 0], P[:, 1], c=err * 100, s=1.5, cmap="Oranges", linewidths=0,
                   vmin=0, vmax=np.percentile(err * 100, 98), rasterized=True)
    a.set_aspect("equal")
    a.set_xlabel("x (m)")
    a.set_ylabel("y (m)")
    a.set_title("how far BA moved GLIM's poses")
    cb = fig.colorbar(sc, ax=a, orientation="horizontal", shrink=0.7, pad=0.2, aspect=30)
    cb.set_label("position change (cm)")
    rounds = r.get("rounds") or []
    if rounds:
        names = ["input"] + [f"round {x['round']}" for x in rounds]
        for i, (lab, col, ls) in enumerate((("walls", INK, "-"), ("floors", INK2, "--"))):
            y = [rounds[0]["thickness_before_cm"][1 + i]] + [x["thickness_after_cm"][1 + i] for x in rounds]
            b.plot(names, y, marker="o", ms=4, color=col, ls=ls, label=lab)
            b.annotate(f"{lab} {y[-1]:.1f}", (len(names) - 1, y[-1]), xytext=(4, 0),
                       textcoords="offset points", va="center", fontsize=7.5, color=INK)
        b.set_ylim(bottom=0)
        b.set_ylabel("plane thickness (cm, 3.29 σ)")
        b.set_title("BA's planes, round by round")
        b.legend(loc="lower left")
    return save(fig, ctx["out"], "ba")


def fig_trajectories(ctx):
    plt = plt_()
    tr, Tr = ctx["ref"]
    fig, ax = plt.subplots(figsize=(7.16, 2.8))
    t0, drawn = tr[0], 0
    step = float(np.median(np.diff(tr)))
    k = max(1, int(round(1.0 / step)))                 # 1 s mean, to read the trend
    for key, label, f in ctx["rows"]:
        if key == "glim_ba" or not os.path.isfile(f):
            continue
        t, T = load_traj(f)
        te, err = fitted_error(t, T, tr, Tr)
        ate = np.sqrt(np.mean(err ** 2)) * 100
        sm = np.convolve(err, np.ones(k) / k, mode="same")
        ax.plot(te - t0, sm * 100, color=COLOR[key], lw=1.0,
                label=f"{SHORT[key]}  (ATE {ate:.1f} cm)")
        drawn += 1
    if not drawn:
        return "no trajectory besides the reference"
    ax.set_xlabel("time (s)")
    ax.set_ylabel("position error, 1 s mean (cm)")
    ax.set_ylim(bottom=0)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.02), ncol=3, borderaxespad=0)
    ax.set_title("against the reference (%s), after the best rigid fit"
                 % ("04, scan-to-map" if ctx["P"].reference else "GLIM + BA"), pad=22)
    return save(fig, ctx["out"], "trajectories")


def _summaries(ctx):
    """[(name, 04's summary dict)] of the run's tracks, in file order."""
    import glob
    out = []
    for f in sorted(glob.glob(os.path.join(ctx["P"].reference_dir(), "summary_*.json"))):
        try:
            r = json.load(open(f))
        except ValueError:
            continue
        if r.get("rounds"):
            out.append((r.get("name") or os.path.basename(f)[8:-5], r))
    return out


def fig_reference(ctx):
    """A run's own quality: what the registration had to move, and how tightly
    each board came out. Both are the questions a run (not a mapping pass)
    answers, so there is nothing to draw for a mapping pass."""
    plt = plt_()
    P = ctx["P"]
    if not P.reference:
        return "a mapping pass: its registration figure is ba"
    summ = _summaries(ctx)
    if not summ:
        return f"no summary_*.json in {P.reference_dir()} (run 04)"
    c = P.cfg.get("04_reference") or {}
    gate = float(c.get("max_shift", 0.5)) * 100
    s05 = P.stage("05_anchor") if "05_anchor" in P.cfg else {}
    boards = {}
    bf = s05.get("frame_out")
    if bf and os.path.isfile(bf):
        boards = (json.load(open(bf)).get("boards") or {})
    fuse = (s05.get("fuse") or {})
    g_std = float(fuse.get("max_std_mm", 20.0))
    g_by = fuse.get("max_std_mm_by_board") or {}

    fig, (a, b) = plt.subplots(1, 2, figsize=(7.16, 3.1),
                              gridspec_kw={"width_ratios": [1, 1], "wspace": 0.45})
    keys = ("glim_corr", "glim_s2m", "glim_ba")
    rnds = sorted({h["round"] for _, r in summ for h in r["rounds"]})
    sub = []
    for i, (name, r) in enumerate(summ):
        col = COLOR[keys[i % len(keys)]]
        rounds = r["rounds"]
        x = [h["round"] for h in rounds]
        lab = (name + ": ") if len(summ) > 1 else ""
        a.plot(x, [h["median_corr_cm"] for h in rounds], marker="o", ms=4, color=col,
               label=lab + "median")
        a.plot(x, [h["p95_corr_cm"] for h in rounds], marker="o", ms=3, color=col,
               ls="--", lw=1.0, alpha=0.8, label=lab + "p95")
        for h in rounds:                            # how many scans that round placed
            a.annotate("%d/%d" % (h["registered"], h["scans"]),
                       (h["round"], h["median_corr_cm"]), xytext=(0, 7),
                       textcoords="offset points", ha="center", fontsize=7, color=INK2)
        sf = r.get("seed_to_final_cm") or {}
        if sf:
            sub.append("%sGLIM to final %.0f cm median, %.0f max"
                       % (lab, sf["median"], sf["max"]))
    a.axhline(gate, color=INK, lw=1.0, ls=":")
    a.annotate("gate %.0f cm" % gate, (rnds[0], gate), xytext=(0, 3),
               textcoords="offset points", fontsize=7.5, color=INK)
    a.set_yscale("log")
    a.set_xticks(rnds)
    a.set_xlim(rnds[0] - 0.25, rnds[-1] + 0.25)
    a.set_xlabel("registration round")
    a.set_ylabel("correction (cm)")
    a.set_title("what the map had to move the scans"
                + ("\n" + "; ".join(sub) if sub else ""), fontsize=8.5)
    a.legend(loc="lower left", fontsize=7)

    if boards:
        names = sorted(boards)
        for i, n in enumerate(names):
            v = boards[n]
            g = float(g_by.get(n, g_std))
            b.barh(i, v["std_mm"], height=0.55, color=COLOR["glim_corr"])
            b.plot([v["max_mm"]] * 2, [i - 0.3, i + 0.3], color=INK2, lw=1.2)
            b.plot([g] * 2, [i - 0.38, i + 0.38], color=INK, lw=1.6)
            b.annotate("%.1f%s" % (v["std_mm"], " (own gate)" if n in g_by else ""),
                       (max(v["max_mm"], g), i), xytext=(4, 0),
                       textcoords="offset points", va="center", fontsize=7.5)
        b.set_yticks(range(len(names)),
                     [n + (" (anchor)" if boards[n].get("method", "").startswith("anchor")
                           else "") for n in names])
        b.invert_yaxis()
        b.set_xlim(0, max([max(boards[n]["max_mm"], float(g_by.get(n, g_std)))
                           for n in names]) * 1.9)
        b.set_xlabel("spread of a board's sightings (mm)\n"
                     "grey tick: its worst, black: the gate it had to pass")
        b.set_title("how tightly each board came out", fontsize=8.5)
        b.grid(axis="y", visible=False)
    else:
        b.set_axis_off()
        b.annotate("no %s (run 05)" % (bf or "anchor frame"), (0.5, 0.5),
                   xycoords="axes fraction", ha="center", fontsize=7.5, color=INK2)
    return save(fig, ctx["out"], "reference")


def _maps(ctx):
    P = ctx["P"]
    tsdf = P.outp(P.pcd("denoised"))
    union = P.outp("denoised_union_%s.pcd" % P.tag if P.tag else "denoised_union.pcd")
    return tsdf, union


def fig_maps(ctx):
    plt = plt_()
    tsdf, union = _maps(ctx)
    miss = need(tsdf)
    if miss:
        return f"no {miss}"
    M = maps.load(tsdf)
    rng = np.random.default_rng(0)
    sub = M[rng.choice(len(M), min(len(M), 1_500_000), replace=False)]
    have_union = os.path.isfile(union)
    fig, axs = plt.subplots(1, 2 if have_union else 1, figsize=(7.16, 3.4),
                            gridspec_kw={"width_ratios": [1.3, 1]} if have_union else None)
    a = axs[0] if have_union else axs
    z = sub[:, 2]
    lo, hi = np.percentile(z, [2, 98])
    order = np.argsort(z)                              # high points drawn on top
    sc = a.scatter(sub[order, 0], sub[order, 1], c=z[order], s=0.05, cmap="Blues",
                   vmin=lo, vmax=hi, linewidths=0, rasterized=True)
    a.set_aspect("equal")
    a.set_xlabel("x (m)")
    a.set_ylabel("y (m)")
    a.set_title(f"reference map, TSDF ({len(M) / 1e6:.1f} M points)")
    a.grid(False)
    fig.colorbar(sc, ax=a, orientation="horizontal", shrink=0.7, pad=0.2, aspect=30).set_label("height (m)")
    if have_union:
        b = axs[1]
        U = maps.load(union)
        pos = ctx["ref"][1][:, :3, 3]
        edges = np.arange(0, 25.5, 0.5)
        meds = []
        for key, cloud, lab in (("union", U, "union of points"), ("tsdf", M, "TSDF (reference)")):
            st, _ = maps.patch_stats(cloud, n_samples=ctx["samples"], positions=pos)
            th = st[st[:, 1] > 60, 2] * 100
            b.hist(th, bins=edges, histtype="step", lw=1.4, color=COLOR[key], density=True)
            meds.append((key, lab, float(np.median(th))))
        top = b.get_ylim()[1]
        for i, (key, lab, med) in enumerate(meds):
            b.axvline(med, color=COLOR[key], lw=1.0, ls="--")
            b.annotate(f"{lab}\nmedian {med:.1f} cm", (med, top * (0.55 - 0.3 * i)), xytext=(5, 0),
                       textcoords="offset points", fontsize=7.5, color=INK, va="center")
        b.set_xlabel("wall thickness (cm)")
        b.set_ylabel("share of wall patches")
        b.set_title("same scans and poses, two fusions")
        b.set_yticks([])
    return save(fig, ctx["out"], "maps")


def fig_wall(ctx):
    plt = plt_()
    from scipy.spatial import cKDTree
    tsdf, union = _maps(ctx)
    miss = need(tsdf, union)
    if miss:
        return f"no {miss}"
    M, U = maps.load(tsdf), maps.load(union)
    probes = maps.wall_probes(M, 60, seed=3)
    if not probes:
        return "no flat wall in the map"
    th = np.array([p[2] for p in probes])
    c, n, _ = probes[int(np.argmin(np.abs(th - np.median(th))))]   # a typical wall
    u = np.cross(n, [0.0, 0.0, 1.0])
    u /= np.linalg.norm(u)
    fig, axs = plt.subplots(1, 2, figsize=(5.2, 3.2), sharey=True)
    for ax, (key, cloud, lab) in zip(axs, (("union", U, "union of points"), ("tsdf", M, "TSDF (reference)"))):
        idx = cKDTree(cloud).query_ball_point(c, 0.9)
        d = cloud[idx] - c
        off, along, up = d @ n, d @ u, d[:, 2]
        m = (np.abs(off) < 0.3) & (np.abs(along) < 0.4) & (np.abs(up) < 0.6)
        lo, hi = np.percentile(off[m], [5, 95]) if m.sum() > 20 else (0, 0)
        ax.axvspan(lo * 100, hi * 100, color=COLOR[key], alpha=0.15, lw=0)
        ax.scatter(off[m] * 100, up[m], s=1.5, color=COLOR[key], linewidths=0, rasterized=True)
        ax.set_title(f"{lab}\n90 % of points within {(hi - lo) * 100:.1f} cm")
        ax.set_xlabel("across the wall (cm)")
        ax.set_xlim(-30, 30)
    axs[0].set_ylabel("height about the spot (m), a cut 0.8 m wide")
    return save(fig, ctx["out"], "wall")


def fig_loop(ctx):
    plt = plt_()
    import importlib.util
    spec = importlib.util.spec_from_file_location("check_", os.path.join(ROOT, "check.py"))
    chk = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(chk)
    items = []
    for key, label, f in ctx["rows"]:
        lp = chk.loop_closure(ctx["P"], ctx["config"], key, f, False) if os.path.isfile(f) else None
        if lp:
            items.append((key, lp))
    if not items:
        return "no loop-closure report (05 --check) beside the trajectories"
    fig, ax = plt.subplots(figsize=(5.2, 0.5 + 0.45 * len(items)))
    for i, (key, lp) in enumerate(items):
        ax.barh(i, lp["mm"], height=0.55, color=COLOR[key])
        ax.plot([lp["noise_mm"]] * 2, [i - 0.35, i + 0.35], color=INK, lw=1.4)
        ax.annotate(f"{lp['mm']:.1f} mm / {lp['deg']:.2f}°", (lp["mm"], i), xytext=(4, 0),
                    textcoords="offset points", va="center", fontsize=7.5)
    ax.set_yticks(range(len(items)), [SHORT[k] for k, _ in items])
    ax.invert_yaxis()
    ax.set_xlim(0, max(lp["mm"] for _, lp in items) * 1.35)
    ax.set_xlabel("board at the end against the start (mm); black tick: detection noise")
    ax.set_title("loop closure on the anchor board over the run")
    ax.grid(axis="y", visible=False)
    return save(fig, ctx["out"], "loop")


def fig_stages(ctx):
    plt = plt_()
    files = {k: os.path.join(ctx["out"], f"walls_{k}.json") for k in ("glim", "ref")}
    miss = need(*files.values())
    if miss:
        return (f"no {miss}: make it with check.py walls {ctx['config']} --traj "
                f"{'traj_lidar.txt' if 'glim' in miss else os.path.basename(ctx['rows'][-1][2])} "
                f"--json {miss}")
    names = ["1 scan", "1 s", "10 s", "60 s", "300 s", "run"]
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    for k, key in (("glim", "glim_corr"), ("ref", "glim_ba")):
        r = json.load(open(files[k]))
        y = [r["single"]] + [r["spans"][s] for s in ("1.0", "10.0", "60.0", "300.0")] + [r["whole"]]
        y = [v * 100 for v in y]
        ax.plot(names, y, marker="o", ms=4, color=COLOR[key], label=SHORT[key])
        ax.annotate(f"{y[-1]:.1f}", (len(names) - 1, y[-1]), xytext=(5, 0),
                    textcoords="offset points", va="center", fontsize=7.5)
    if ctx.get("noise_walls"):
        ax.axhline(ctx["noise_walls"], color=INK2, lw=1.0, ls="--")
        ax.annotate(f"sensor alone {ctx['noise_walls']:.1f}", (0, ctx["noise_walls"]),
                    xytext=(0, -10), textcoords="offset points", fontsize=7.5, color=INK2)
    ax.set_ylim(bottom=0)
    ax.set_ylabel("wall thickness (cm)")
    ax.set_xlabel("scans placed together")
    ax.set_title("a wall's thickness as the scans pile up")
    ax.legend(loc="upper left")
    return save(fig, ctx["out"], "stages")


def fig_noise(ctx):
    plt = plt_()
    if not ctx["bag"]:
        return "reads the bag: pass --bag"
    from scoop import bag as sbag
    P = ctx["P"]
    reader = sbag.open_bag(P.dataset["bag"])
    topic = sbag.detect_points_topic(reader, P.dataset.get("points_topic") or "")
    total = reader.topics()[topic].count
    stride = max(1, total // 50)
    seen = [0]

    def every(_s):
        seen[0] += 1
        return seen[0] if (seen[0] - 1) % stride == 0 else None
    rng = np.random.default_rng(0)
    rows = [maps.scan_noise(np.asarray(s.xyz, np.float64), rng, 400)
            for s in sbag.iter_scans(reader, topic, select=every, min_range=1.0, max_range=30.0,
                                     min_points=1000)]
    st = np.vstack(rows)
    wall = st[:, 1] > 60
    ctx["noise_walls"] = float(np.median(st[wall, 0]) * 100)
    edges = list(zip(maps.DIST[:-1], maps.DIST[1:]))
    labs, vals = [], []
    for lo, hi in edges:
        m = wall & (st[:, 2] >= lo) & (st[:, 2] < hi)
        if m.sum() >= 10:
            labs.append(f"{lo:g}-{hi:g} m" if hi < 1e8 else f"{lo:g}+ m")
            vals.append(float(np.median(st[m, 0]) * 100))
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    ax.bar(labs, vals, color=COLOR["glim_corr"], width=0.6)
    for i, v in enumerate(vals):
        ax.annotate(f"{v:.1f}", (i, v), xytext=(0, 3), textcoords="offset points", ha="center",
                    fontsize=7.5)
    ax.set_ylabel("one scan's wall thickness (cm)")
    ax.set_xlabel("range")
    ax.set_title(f"the sensor alone: walls {ctx['noise_walls']:.1f} cm overall")
    ax.grid(axis="x", visible=False)
    return save(fig, ctx["out"], "noise")


def main_figures(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--only", default="", help="comma-separated: " + ",".join(FIGS))
    ap.add_argument("--bag", action="store_true", help="also the noise figure (reads the bag)")
    ap.add_argument("--out")
    ap.add_argument("--samples", type=int, default=8000, help="patches per map (maps)")
    a = ap.parse_args(argv)
    import importlib.util
    spec = importlib.util.spec_from_file_location("check_", os.path.join(ROOT, "check.py"))
    chk = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(chk)
    P = load_pipeline(a.config)
    rows, _ = chk.trajectories(P)
    ref_f = rows[-1][2]
    if not os.path.isfile(ref_f):
        raise SystemExit(f"no reference trajectory at {ref_f} (run 02 first)")
    ctx = dict(P=P, config=a.config, rows=rows, ref=load_traj(ref_f), bag=a.bag,
               samples=a.samples, out=a.out or os.path.join(P.out_dir, "figures"))
    want = [f for f in FIGS if not a.only or f in a.only.split(",")]
    if "stages" in want and "noise" in want:          # the sensor line on stages
        want.remove("noise")
        want.insert(want.index("stages"), "noise")
    print(f"figures of {P.tag} -> {ctx['out']}")
    for name in want:
        r = globals()["fig_" + name](ctx)
        if r and not str(r).endswith(".png"):
            print(f"  {name}: skipped, {r}")


if __name__ == "__main__":
    main_figures()
