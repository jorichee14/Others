#!/usr/bin/env python3
"""Do the link logs agree with the building? (Paper A, experiment "Map Meets
Radio"): for every measured link sample, the first-Fresnel-zone clearance of
the path between its two ends through the pass's anchored map, against the
signal measured on it.

    python scoop_pipeline/analysis/comms/link_clearance.py <processed pass folder>
        [--config FILE] [--map PCD] [--ap X Y Z] [--window 0.5]
    python scoop_pipeline/analysis/comms/link_clearance.py --all <data/processed>
        [--ap-file FILE] [--out DIR]

Links, from the comms tables (processing/comms_tables.py, made with positions):
  csi     agent to agent: the sniffer (on mobile_1) and each transmitter it
          heard (mobile_2, infra_1), both ends positioned per packet. QoS data
          frames only. Signal: the frame's RSSI. Its CSI power goes in the
          table as it is, not tested: the CSI is taken after the receiver's
          gain control, so it does not fall with distance and the distance
          trend removed below would turn it into a length ramp.
  ap      agent to access point, only with --ap (the AP is in no TF tree):
          the Wi-Fi status RSSI of every machine, and the iperf goodput of
          the tests it ran.

Samples are binned into --window s windows per link (median signal, median
ends). For each window the straight path is sampled every 5 cm, the end
--margin m at each side dropped (the robot's own body and mast), and

    clearance = min over the path of  d_map(s) / r1(s),
    r1(s) = sqrt(lambda d1 d2 / (d1 + d2)),  lambda = c / 5745 MHz,

with d_map the distance to the nearest map point. clearance >= 1: the first
Fresnel zone is free; < 0.6: obstructed (the usual planning rule); ~0: the
line of sight crosses the map. The map is static, so people and moved
furniture are not seen: they show up as signal the clearance cannot explain.

Clearance falls with length on its own (r1 grows with it), so the test that
matters is against the signal left after the distance trend: a dB signal
minus a - 10 n log10(d), with the path-loss exponent n fixed (--exponent,
default 2, free space) and the offset a the median over the link's clear
windows. The slope is not fitted: along a route, length and obstruction move
together (the far end of a corridor is also the end behind the wall), and a
free fit takes the wall loss into the slope. Goodput has no distance trend
to remove; its residual is the goodput itself. Spearman's rho is taken
between the clearance and the residual. Reported per link: n, rho with the raw signal,
rho with the residual, median signal and residual for clear (>= 0.6) and
obstructed (< 0.6) windows, and their difference.

Writes comms/clearance/: clearance.csv (one row per window and link),
clearance_report.json, clearance.png. With --all, every pass under the
folder that has positioned comms tables, then <out>/clearance_all.csv
(every pass's rows), clearance_all.json (per pass and pooled per link kind)
and clearance_all.png.
"""
import argparse
import csv
import glob
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "map_stages"))
sys.path.insert(0, HERE)

FREQ_HZ = 5745e6                     # channel 149
LAMBDA = 299_792_458.0 / FREQ_HZ     # 5.22 cm
CLEAR = 0.6                          # obstructed below this
STEP = 0.05                          # path sampling (m)


# ---- map --------------------------------------------------------------------

def read_pcd_xyz(path):
    """(N, 3) float points of a .pcd (ascii or binary, x y z fields), with
    open3d when it is there."""
    try:
        import open3d as o3d
        return np.asarray(o3d.io.read_point_cloud(path).points, float)
    except ImportError:
        pass
    with open(path, "rb") as fh:
        head = {}
        while True:
            line = fh.readline().decode("ascii", "replace").strip()
            if not line or line.startswith("#"):
                continue
            k, *v = line.split()
            head[k.upper()] = v
            if k.upper() == "DATA":
                break
        fields = head["FIELDS"]
        size = [int(s) for s in head["SIZE"]]
        typ = head["TYPE"]
        count = [int(c) for c in head.get("COUNT", ["1"] * len(fields))]
        n = int(head["POINTS"][0])
        if head["DATA"][0] == "ascii":
            a = np.loadtxt(fh, ndmin=2)
            cols, i = {}, 0
            for f, c in zip(fields, count):
                cols[f] = i
                i += c
            return a[:n, [cols["x"], cols["y"], cols["z"]]].astype(float)
        if head["DATA"][0] != "binary":
            raise SystemExit(f"{path}: {head['DATA'][0]} PCD needs open3d")
        kinds = {"F": "f", "U": "u", "I": "i"}
        dt = np.dtype([(f if c == 1 else f + "_%d" % j, "<%s%d" % (kinds[t], s))
                       for f, s, t, c in zip(fields, size, typ, count)
                       for j in range(c)])
        a = np.frombuffer(fh.read(n * dt.itemsize), dtype=dt, count=n)
        return np.stack([a["x"], a["y"], a["z"]], 1).astype(float)


def voxel_down(p, res):
    """One point per occupied res-m voxel (its first point)."""
    if res <= 0 or not len(p):
        return p
    _, i = np.unique(np.floor(p / res).astype(np.int64), axis=0, return_index=True)
    return p[np.sort(i)]


class MapIndex:
    def __init__(self, points, res=0.05):
        from scipy.spatial import cKDTree
        self.points = voxel_down(np.asarray(points, float), res)
        self.tree = cKDTree(self.points)

    def clearance(self, a, b, margin=0.3):
        """(clearance ratio, length) of the path a -> b; nan when the path is
        shorter than twice the margin."""
        a, b = np.asarray(a, float), np.asarray(b, float)
        d = float(np.linalg.norm(b - a))
        if not np.isfinite(d) or d <= 2 * margin + STEP:
            return math.nan, d
        s = np.arange(margin, d - margin + 1e-9, STEP)
        pts = a + (b - a)[None, :] * (s / d)[:, None]
        dist, _ = self.tree.query(pts)
        r1 = np.sqrt(LAMBDA * s * (d - s) / d)
        return float(np.min(dist / r1)), d


# ---- tables -----------------------------------------------------------------

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


def rows_of(comms, sub, name):
    p = os.path.join(comms, sub, name)
    return list(csv.DictReader(open(p))) if os.path.exists(p) else []


def windows(t, cols, window):
    """{column: median per window} over the rows' times t; 'n' the count."""
    t = np.asarray(t, float)
    ok = np.isfinite(t)
    if not ok.any():
        return {}
    k = np.floor((t[ok] - t[ok].min()) / window).astype(int)
    out = {c: [] for c in cols}
    out["t"], out["n"] = [], []
    for w in np.unique(k):
        m = k == w
        out["t"].append(float(np.median(t[ok][m])))
        out["n"].append(int(m.sum()))
        for c, v in cols.items():
            vv = np.asarray(v, float)[ok][m]
            out[c].append(float(np.nanmedian(vv)) if np.isfinite(vv).any() else math.nan)
    return {c: np.array(v) for c, v in out.items()}


def csi_links(comms, window):
    """[(link, kind, {t, n, ax.., bx.., signal..})] from csi.csv, QoS data only."""
    rows = rows_of(comms, "csi", "csi.csv")
    if not rows or "peer_x" not in rows[0]:
        return []
    out = []
    for link in sorted({(r["peer"], r["machine"]) for r in rows}):
        rr = [r for r in rows if (r["peer"], r["machine"]) == link
              and (not r.get("frame_control") or num(r, "frame_control") == 0x88)]
        if not rr:
            continue
        cols = {k: [num(r, k) for r in rr] for k in ("x", "y", "z", "peer_x", "peer_y", "peer_z")}
        cols["rssi_dbm"] = [num(r, "rssi") for r in rr]
        cols["csi_power_db"] = [num(r, "csi_power_db") for r in rr]
        w = windows([num(r, "t") for r in rr], cols, window)
        if w:
            out.append(("%s>%s" % link, "csi", w))
    return out


def ap_links(comms, ap, window):
    """Agent to access point: Wi-Fi status RSSI per machine, and iperf goodput
    (one window per test) with the machine's position."""
    out = []
    if ap is None:
        return out
    wifi = rows_of(comms, "wifi", "wifi.csv")
    for m in sorted({r["machine"] for r in wifi}):
        rr = [r for r in wifi if r["machine"] == m]
        if "x" not in rr[0]:
            continue
        cols = {k: [num(r, k) for r in rr] for k in ("x", "y", "z")}
        cols["rssi_dbm"] = [num(r, "signal_dbm") for r in rr]
        w = windows([num(r, "t") for r in rr], cols, window)
        if w:
            out.append(("%s>ap" % m, "ap", w))
    iperf = rows_of(comms, "wifi", "iperf.csv")
    for m in sorted({r["machine"] for r in iperf}):
        for d, rev in (("down", 1.0), ("up", 0.0)):
            rr = [r for r in iperf if r["machine"] == m and r.get("x", "") != ""
                  and num(r, "reverse") == rev and num(r, "success") == 1]
            if not rr:
                continue
            w = {"t": np.array([num(r, "t") for r in rr]), "n": np.ones(len(rr), int),
                 "goodput_mbps": np.array([num(r, "bitrate_mbps") for r in rr])}
            for k in ("x", "y", "z"):
                w[k] = np.array([num(r, k) for r in rr])
            out.append(("%s>ap iperf %s" % (m, d), "ap_iperf", w))
    return out


# ---- statistics -------------------------------------------------------------

def rank(a):
    """Average ranks (ties share their mean rank)."""
    a = np.asarray(a, float)
    o = np.argsort(a, kind="mergesort")
    r = np.empty(len(a))
    r[o] = np.arange(len(a), dtype=float)
    s = a[o]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        if j > i:
            r[o[i:j + 1]] = (i + j) / 2.0
        i = j + 1
    return r


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 5 or np.ptp(x[ok]) == 0 or np.ptp(y[ok]) == 0:
        return None
    return float(np.corrcoef(rank(x[ok]), rank(y[ok]))[0, 1])


def logdist_residual(d, v, clr, exponent=2.0):
    """v minus a - 10 n log10(d), n fixed, a the median over the clear
    windows (all windows when none is clear); and a."""
    d, v, clr = np.asarray(d, float), np.asarray(v, float), np.asarray(clr, float)
    ok = np.isfinite(d) & np.isfinite(v) & (d > 0)
    res = np.full(len(v), math.nan)
    if not ok.any():
        return res, None
    trend = v + 10 * exponent * np.log10(np.where(d > 0, d, np.nan))
    ref = ok & (clr >= CLEAR)
    a = float(np.median(trend[ref if ref.any() else ok]))
    res[ok] = trend[ok] - a
    return res, a


def med(a):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    return float(np.median(a)) if len(a) else None


def signal_report(clr, d, v, exponent=2.0, db=True):
    if db:
        res, a = logdist_residual(d, v, clr, exponent)
    else:
        res, a = np.asarray(v, float).copy(), None
    clear, obst = clr >= CLEAR, clr < CLEAR
    rep = {"rho_signal": spearman(clr, v), "rho_residual": spearman(clr, res),
           "distance_trend": {"exponent": exponent, "a": a} if db else None,
           "clear": {"n": int((clear & np.isfinite(v)).sum()), "median": med(v[clear]),
                     "residual": med(res[clear])},
           "obstructed": {"n": int((obst & np.isfinite(v)).sum()), "median": med(v[obst]),
                          "residual": med(res[obst])}}
    a, b = rep["clear"]["residual"], rep["obstructed"]["residual"]
    rep["residual_drop_db"] = (a - b) if a is not None and b is not None else None
    return rep, res


# ---- one pass -----------------------------------------------------------------

def find_map(proc, config=None):
    """The anchored map the pass's positions are in (the reference pass's for
    a run localized in another pass's map), as comms_map.py finds it."""
    try:
        import pass_tf
        cfg = config or pass_tf.find_config(proc)
        if cfg:
            P = pass_tf.load_pipeline(cfg)
            p = P.ref_file("anchored_map") if P.reference else P.outp("map_final_{tag}_anchored.pcd")
            if os.path.exists(p):
                return p
    except (ImportError, SystemExit, KeyError, FileNotFoundError):
        pass
    c = sorted(glob.glob(os.path.join(proc, "mapping", "map_final_*_anchored.pcd")))
    return c[0] if c else None


SIGNALS = {"csi": ("rssi_dbm",), "ap": ("rssi_dbm",), "ap_iperf": ("goodput_mbps",)}
RAW = ("csi_power_db",)              # in the table, not tested (after gain control)


def run_pass(proc, map_path=None, ap=None, window=0.5, margin=0.3, config=None,
             plots=True, exponent=2.0, log=print):
    proc = os.path.abspath(os.path.expanduser(proc))
    comms = os.path.join(proc, "comms")
    map_path = map_path or find_map(proc, config)
    if not map_path:
        log(f"  {proc}: no anchored map found (--map)")
        return None
    links = csi_links(comms, window) + ap_links(comms, ap, window)
    if not links:
        log(f"  {proc}: no positioned link samples (comms_tables.py with the pass's config;"
            f" agent-to-AP links need --ap)")
        return None
    M = MapIndex(read_pcd_xyz(map_path))
    log(f"  {proc}: map {map_path} ({len(M.points)} points at 5 cm), "
        f"{len(links)} link(s)")
    out = os.path.join(comms, "clearance")
    os.makedirs(out, exist_ok=True)
    report = {"pass": proc, "map": map_path, "window_s": window, "margin_m": margin,
              "path_loss_exponent": exponent,
              "lambda_m": LAMBDA, "clear_threshold": CLEAR, "links": {}}
    table = []
    for name, kind, w in links:
        a = np.stack([w["x"], w["y"], w["z"]], 1)
        b = (np.stack([w["peer_x"], w["peer_y"], w["peer_z"]], 1) if kind == "csi"
             else np.tile(np.asarray(ap, float), (len(a), 1)))
        cl = np.array([M.clearance(p, q, margin) for p, q in zip(a, b)])
        clr, d = cl[:, 0], cl[:, 1]
        rep = {"kind": kind, "windows": int(np.isfinite(clr).sum()),
               "length_m": {"median": med(d), "min": float(np.nanmin(d)) if len(d) else None,
                            "max": float(np.nanmax(d)) if len(d) else None},
               "clearance": {"median": med(clr), "obstructed_share":
                             float(np.mean(clr[np.isfinite(clr)] < CLEAR))
                             if np.isfinite(clr).any() else None}}
        resid = {}
        for s in SIGNALS[kind]:
            if s in w and np.isfinite(w[s]).any():
                rep[s], resid[s] = signal_report(clr, d, w[s], exponent,
                                                 db=s != "goodput_mbps")
        report["links"][name] = rep
        for i in range(len(clr)):
            row = {"link": name, "kind": kind, "t": w["t"][i], "n": int(w["n"][i]),
                   "length_m": d[i], "clearance": clr[i]}
            for s in SIGNALS[kind]:
                if s in w:
                    row[s] = w[s][i]
                    row[s + "_residual"] = resid[s][i] if s in resid else math.nan
            for s in RAW:
                if s in w:
                    row[s] = w[s][i]
            table.append(row)
    cols = []
    for r in table:
        cols += [k for k in r if k not in cols]
    with open(os.path.join(out, "clearance.csv"), "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=cols)
        wr.writeheader()
        for r in table:
            wr.writerow({k: ("%.6g" % v if isinstance(v, float) else v) for k, v in r.items()})
    json.dump(report, open(os.path.join(out, "clearance_report.json"), "w"), indent=2)
    for name, rep in report["links"].items():
        for s in SIGNALS[rep["kind"]]:
            if s in rep:
                r = rep[s]
                log("    %-24s %-13s n=%-5d rho(signal)=%s rho(residual)=%s "
                    "clear-obstructed residual=%s (obstructed %s%%)" % (
                        name, s, rep["windows"], fmt(r["rho_signal"]), fmt(r["rho_residual"]),
                        fmt(r["residual_drop_db"], 1),
                        fmt(100 * rep["clearance"]["obstructed_share"], 0)
                        if rep["clearance"]["obstructed_share"] is not None else "-"))
    if plots:
        plot(table, os.path.join(out, "clearance.png"), os.path.basename(proc))
    return report, table


def fmt(v, nd=2):
    return "-" if v is None or (isinstance(v, float) and not np.isfinite(v)) else "%.*f" % (nd, v)


def plot(table, path, title):
    """Clearance against the residual signal, one panel per signal."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sig = [s for s in ("rssi_dbm", "goodput_mbps")
           if any(np.isfinite(r.get(s + "_residual", math.nan)) for r in table)]
    if not sig:
        return
    fig, axs = plt.subplots(1, len(sig), figsize=(4.2 * len(sig), 3.4), squeeze=False)
    for ax, s in zip(axs[0], sig):
        for link in sorted({r["link"] for r in table if s in r}):
            rr = [r for r in table if r["link"] == link and s in r]
            x = np.array([r["clearance"] for r in rr])
            y = np.array([r[s + "_residual"] for r in rr])
            ax.scatter(np.clip(x, 0, 3), y, s=4, alpha=0.4, label=link)
        ax.axvline(CLEAR, color="k", lw=0.8, ls="--")
        ax.set_xlabel("Fresnel clearance (d / r1, clipped at 3)")
        ax.set_ylabel(s + (" (no distance trend)" if s == "goodput_mbps"
                           else " minus distance trend"))
        ax.legend(fontsize=6, markerscale=2)
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---- every pass ---------------------------------------------------------------

def passes_under(root):
    """data/processed/<date>/<pass> folders whose comms tables carry positions."""
    out = []
    for comms in sorted(glob.glob(os.path.join(root, "*", "*", "comms"))):
        for sub, name, col in (("csi", "csi.csv", "peer_x"), ("wifi", "wifi.csv", "x")):
            p = os.path.join(comms, sub, name)
            if os.path.exists(p):
                with open(p) as fh:
                    if col in fh.readline().split(","):
                        out.append(os.path.dirname(comms))
                        break
    return out


def ap_for(ap_file, proc):
    """The AP position for a pass from a yaml {"<date>/<pass>" or "<pass>" or
    "default": [x, y, z]} (the AP is fixed per site, the map frame per pass)."""
    if not ap_file:
        return None
    import yaml
    m = yaml.safe_load(open(ap_file)) or {}
    date, pas = os.path.normpath(proc).split(os.sep)[-2:]
    for k in ("%s/%s" % (date, pas), pas, "default"):
        if k in m:
            return [float(v) for v in m[k]]
    return None


def run_all(root, out, ap_file=None, window=0.5, margin=0.3, exponent=2.0, log=print):
    ps = passes_under(os.path.abspath(os.path.expanduser(root)))
    if not ps:
        sys.exit(f"no processed pass with positioned comms tables under {root}")
    log(f"{len(ps)} pass(es) under {root}")
    allrows, summary = [], {"passes": {}, "pooled": {}}
    for p in ps:
        r = run_pass(p, ap=ap_for(ap_file, p), window=window, margin=margin,
                     exponent=exponent, log=log)
        if r is None:
            summary["passes"][p] = None
            continue
        rep, table = r
        summary["passes"][p] = rep
        tag = "/".join(os.path.normpath(p).split(os.sep)[-2:])
        allrows += [dict(row, **{"pass": tag}) for row in table]
    for kind, sigs in SIGNALS.items():
        rr = [r for r in allrows if r["kind"] == kind]
        if not rr:
            continue
        clr = np.array([r["clearance"] for r in rr])
        summary["pooled"][kind] = {"windows": int(np.isfinite(clr).sum()),
                                   "obstructed_share": float(np.mean(clr[np.isfinite(clr)] < CLEAR))
                                   if np.isfinite(clr).any() else None}
        for s in sigs:
            res = np.array([r.get(s + "_residual", math.nan) for r in rr])
            v = np.array([r.get(s, math.nan) for r in rr])
            if np.isfinite(res).any():
                c, o = clr >= CLEAR, clr < CLEAR
                drop = (med(res[c]) - med(res[o])) if med(res[c]) is not None \
                    and med(res[o]) is not None else None
                summary["pooled"][kind][s] = {
                    "rho_signal": spearman(clr, v), "rho_residual": spearman(clr, res),
                    "residual_drop_db": drop,
                    "n_clear": int((c & np.isfinite(res)).sum()),
                    "n_obstructed": int((o & np.isfinite(res)).sum())}
    os.makedirs(out, exist_ok=True)
    cols = []
    for r in allrows:
        cols += [k for k in r if k not in cols]
    with open(os.path.join(out, "clearance_all.csv"), "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=cols)
        wr.writeheader()
        for r in allrows:
            wr.writerow({k: ("%.6g" % v if isinstance(v, float) else v) for k, v in r.items()})
    json.dump(summary, open(os.path.join(out, "clearance_all.json"), "w"), indent=2)
    plot(allrows, os.path.join(out, "clearance_all.png"), "all passes")
    log("pooled:")
    for kind, s in summary["pooled"].items():
        for sig, r in s.items():
            if isinstance(r, dict):
                log("  %-9s %-13s rho(signal)=%s rho(residual)=%s clear-obstructed residual=%s"
                    " (n %d / %d)" % (kind, sig, fmt(r["rho_signal"]), fmt(r["rho_residual"]),
                                      fmt(r["residual_drop_db"], 1), r["n_clear"],
                                      r["n_obstructed"]))
    log(f"wrote {out}/clearance_all.{{csv,json,png}}")
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("processed", nargs="?", help="one processed pass folder")
    ap.add_argument("--all", default=None, help="data/processed: every pass under it")
    ap.add_argument("--out", default=None, help="with --all: where the pooled files go "
                    "(default <all>/clearance)")
    ap.add_argument("--config", default=None, help="the pass's pipeline config")
    ap.add_argument("--map", default=None, help="the anchored map (default: found as "
                    "comms_map.py does)")
    ap.add_argument("--ap", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"),
                    help="access point position in the pass's map frame")
    ap.add_argument("--ap-file", default=None, help="with --all: yaml of AP positions")
    ap.add_argument("--window", type=float, default=0.5, help="binning window (s)")
    ap.add_argument("--margin", type=float, default=0.3,
                    help="path length dropped at each end (m)")
    ap.add_argument("--exponent", type=float, default=2.0,
                    help="path-loss exponent of the distance trend removed from dB signals")
    ap.add_argument("--no-plots", action="store_true")
    a = ap.parse_args()
    if a.all:
        run_all(a.all, a.out or os.path.join(a.all, "clearance"), a.ap_file, a.window, a.margin,
                a.exponent)
    elif a.processed:
        if run_pass(a.processed, a.map, a.ap, a.window, a.margin, a.config,
                    plots=not a.no_plots, exponent=a.exponent) is None:
            sys.exit(1)
    else:
        ap.error("a processed pass folder, or --all <data/processed>")


if __name__ == "__main__":
    main()
