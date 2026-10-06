#!/usr/bin/env python3
"""The CSI links of a pass, from comms/csi/ (processing/comms_tables.py):
csi.csv (one row per packet) and csi_<tx>_to_<rx>.npz (H, amp_db per
subcarrier).

    python scoop_pipeline/analysis/comms/csi_analysis.py <processed pass folder>
        [--grid 0.5] [--no-plots]

Per link (transmitter -> sniffer):
  capture     packets, rate, gaps, frame types (802.11 frame control), QoS
              data frames per second; how their sequence numbers step (report
              only: several counters, so no share caught follows from them)
  occupancy   which subcarriers each frame type fills (median amplitude per
              subcarrier against the strongest): a VHT 80 MHz frame fills
              +-2..+-122; a legacy frame sent as a 20 MHz duplicate (control
              frames such as Block Ack) only 4 x 52, so its CSI is not
              comparable with a data frame's
  path loss   RSSI and CSI power against the link length (positions), fitted
              as P = P(1 m) - 10 n log10(d): the exponent n and the spread
              around the fit (shadowing / multipath). The CSI is taken after
              the receiver's gain control, so its power need not follow the
              distance; RSSI is the calibrated one
  delay       RMS delay spread of the channel, from the impulse response
              (inverse FFT of H, data frames, taps within 20 dB of the
              strongest; 80 MHz: 12.5 ns per tap)
  coherence   correlation of the CSI amplitude between consecutive frames of
              one type (< 0.1 s apart), over its data subcarriers (no pilots:
              +-11, +-39, +-75, +-103 at 80 MHz), against the speed of the
              faster end (receiver or transmitter: both move in a coop run):
              how fast the channel changes as they move

Writes comms/csi/csi_report.json and csi_timeline.png, csi_occupancy.png,
csi_pathloss.png, csi_coherence.png, csi_amplitude_<link>.png, csi_map.png.
"""
import argparse
import csv
import json
import math
import os
import sys
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

FC_NAMES = {(2, 8): "QoS Data", (2, 0): "Data", (2, 4): "Null", (2, 12): "QoS Null",
            (1, 9): "Block Ack", (1, 8): "Block Ack Req", (1, 5): "VHT NDP Announcement",
            (1, 11): "RTS", (1, 12): "CTS", (1, 13): "Ack",
            (0, 8): "Beacon", (0, 13): "Action", (0, 14): "Action No Ack", (0, 5): "Probe Resp",
            (0, 4): "Probe Req"}


def fc_name(fc):
    fc = int(fc)
    return FC_NAMES.get(((fc >> 2) & 3, (fc >> 4) & 15), "type %d subtype %d" % ((fc >> 2) & 3,
                                                                              (fc >> 4) & 15))


def load_csv(path, cols):
    """{column: array} for the columns there (numbers as float, else str)."""
    with open(path) as fh:
        rd = csv.reader(fh)
        head = next(rd)
        idx = {c: head.index(c) for c in cols if c in head}
        raw = {c: [] for c in idx}
        for row in rd:
            for c, i in idx.items():
                raw[c].append(row[i])
    out = {}
    for c, v in raw.items():
        try:
            out[c] = np.array([float(x) if x != "" else np.nan for x in v])
        except ValueError:
            out[c] = np.array(v)
    return out


def stats(a):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    if not len(a):
        return None
    return {"median": float(np.median(a)), "p5": float(np.percentile(a, 5)),
            "p95": float(np.percentile(a, 95)), "min": float(a.min()), "max": float(a.max())}


def seq_numbers(seq):
    """nexmon's seq is the frame's 16-bit Sequence Control: the 12-bit
    sequence number above a 4-bit fragment number. -> the sequence numbers
    (as they are, when the values are not of that form)."""
    s = np.asarray(seq, float)
    s = s[np.isfinite(s)].astype(int)
    if len(s) and np.mean(s % 16 == 0) > 0.9 and s.max() > 4095:
        return (s >> 4) & 0xFFF
    return s % 4096


def seq_coverage(seq):
    """How the QoS data frames' sequence numbers step between captured frames
    (step 1 = consecutive, k = k - 1 frames not seen). CSI is one per PPDU
    and an A-MPDU carries several MPDUs, each with its own number, and each
    TID counts on its own: the share caught is given when the median step is
    1, else the median step. Frames also come from more than one counter
    (TIDs, group-addressed frames), so no share caught is derived from them:
    only how they step (in the report)."""
    s = seq_numbers(seq)
    if len(s) < 2:
        return None
    d = np.diff(s) % 4096
    step = d[(d >= 1) & (d < 2048)]
    if not len(step):
        return None
    out = {"median_step": float(np.median(step)), "step_1_percent": 100 * float(np.mean(step == 1)),
           "retries": int((d == 0).sum()), "breaks": int((d >= 2048).sum())}
    return out


# VHT pilot subcarriers per channel width: known symbols, sent at their own
# level (some transmitters boost them), so they are no channel samples
PILOTS = {20: (7, 21), 40: (11, 25, 53), 80: (11, 39, 75, 103), 160: (25, 53, 89, 117)}


def pilot_mask(sub, bw=80):
    p = set(PILOTS.get(int(bw), ()))
    return np.array([abs(int(k)) in p for k in sub])


def occupancy(amp_db, sub):
    """Median amplitude per subcarrier, dB below the strongest; active =
    within 20 dB."""
    med = np.nanmedian(amp_db, axis=0)
    rel = med - np.nanmax(med)
    return rel, int(np.sum(rel > -20))


def path_loss(d, p):
    ok = np.isfinite(d) & np.isfinite(p) & (d > 0.3)
    if ok.sum() < 20 or np.ptp(np.log10(d[ok])) < 0.3:
        return None
    A = np.stack([np.ones(ok.sum()), -10 * np.log10(d[ok])], 1)
    (p0, n), *_ = np.linalg.lstsq(A, p[ok], rcond=None)
    res = p[ok] - A @ np.array([p0, n])
    return {"p_at_1m": float(p0), "exponent": float(n), "spread_db": float(np.std(res)),
            "distance_m": [float(d[ok].min()), float(d[ok].max())], "samples": int(ok.sum())}


def delay_spread(H, sub, nfft=256, tap_ns=12.5, chunk=8000):
    """RMS delay spread (ns) per packet from the impulse response."""
    out = np.full(len(H), np.nan)
    k = np.asarray(sub) % nfft
    for a in range(0, len(H), chunk):
        Hc = np.nan_to_num(H[a:a + chunk].astype(np.complex64))
        F = np.zeros((len(Hc), nfft), np.complex64)
        F[:, k] = Hc
        P = np.abs(np.fft.ifft(F, axis=1)) ** 2
        pk = np.argmax(P, axis=1)
        P = np.stack([np.roll(P[i], 16 - pk[i]) for i in range(len(P))])   # strongest at 16
        keep = P >= P.max(axis=1, keepdims=True) * 0.01                    # within 20 dB
        P = np.where(keep, P, 0)
        tau = (np.arange(nfft) - 16) * tap_ns
        tau = np.where(tau > nfft * tap_ns / 2, tau - nfft * tap_ns, tau)
        w = P / np.maximum(P.sum(axis=1, keepdims=True), 1e-30)
        m1 = (w * tau).sum(axis=1)
        out[a:a + chunk] = np.sqrt(np.maximum((w * tau ** 2).sum(axis=1) - m1 ** 2, 0))
    return out


def speed_at(t, x, y):
    """Speed of the receiver (m/s) at each packet, from its positions."""
    ok = np.isfinite(t) & np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 10:
        return np.full(len(t), np.nan)
    order = np.argsort(t[ok])
    tt, xx, yy = t[ok][order], x[ok][order], y[ok][order]
    g = np.arange(tt[0], tt[-1], 0.2)
    if len(g) < 3:
        return np.full(len(t), np.nan)
    gx, gy = np.interp(g, tt, xx), np.interp(g, tt, yy)
    v = np.hypot(np.gradient(gx, g), np.gradient(gy, g))
    v = np.convolve(v, np.ones(5) / 5, mode="same")
    return np.interp(t, g, v)


def coherence(amp_db, t, fc, mask_by_fc):
    """Correlation of each frame's amplitude with the previous frame of the
    same type within 0.1 s (over that type's active subcarriers)."""
    out = np.full(len(t), np.nan)
    for f, mask in mask_by_fc.items():
        idx = np.flatnonzero(fc == f)
        if len(idx) < 2:
            continue
        A = amp_db[idx][:, mask].astype(float)
        A = A - np.nanmean(A, axis=1, keepdims=True)
        A = A / np.maximum(np.sqrt(np.nansum(A ** 2, axis=1, keepdims=True)), 1e-9)
        c = np.nansum(A[1:] * A[:-1], axis=1)
        near = np.diff(t[idx]) < 0.1
        out[idx[1:][near]] = c[near]
    return out


SPEED_BINS = [0.0, 0.05, 0.2, 0.5, 1.0, 3.0]


def link_report(name, rows, z, log=print):
    t = rows["t"]
    order = np.argsort(rows["t_log"], kind="stable")
    rows = {k: v[order] for k, v in rows.items()}
    t = rows["t"]
    n = len(t)
    if len(z["t"]) != n or not np.allclose(np.sort(z["t_log"]), rows["t_log"], atol=1e-6):
        raise SystemExit(f"{name}: csi.csv and the npz do not hold the same packets "
                         f"({n} vs {len(z['t'])}): run processing/comms_tables.py again")
    zo = np.argsort(z["t_log"], kind="stable")
    sub = z["subcarrier"]
    amp = z["amp_db"][zo]
    fc = rows["frame_control"].astype(int)
    rep = {"packets": n, "duration_s": float(np.nanmax(t) - np.nanmin(t)),
           "rate_hz": float(n / max(np.nanmax(t) - np.nanmin(t), 1e-9))}
    tt = np.sort(t)
    gaps = np.diff(tt)
    rep["gaps_over_1s"] = int((gaps > 1).sum())
    rep["longest_gap_s"] = float(gaps.max()) if len(gaps) else None
    rep["longest_gap_at_s"] = float(tt[int(np.argmax(gaps))] - tt[0]) if len(gaps) else None
    rep["frames"] = {}
    masks = {}
    bw = int(z["bandwidth_mhz"]) if "bandwidth_mhz" in z.files else 80
    pil = pilot_mask(sub, bw)
    for f in sorted(set(fc)):
        m = fc == f
        rel, active = occupancy(amp[m], sub)
        masks[f] = (rel > -20) & ~pil                    # coherence over data subcarriers
        rep["frames"]["0x%02x %s" % (f, fc_name(f))] = {
            "packets": int(m.sum()), "share_percent": 100 * float(m.mean()),
            "active_subcarriers": active, "rssi_dbm": stats(rows["rssi"][m]),
            "csi_power_db": stats(rows["csi_power_db"][m]) if "csi_power_db" in rows else None}
    data = (fc >> 2 & 3) == 2
    qos = fc == 0x88
    rep["qos_data_capture"] = seq_coverage(rows["seq"][qos]) if "seq" in rows else None
    if "distance_m" in rows:
        d = rows["distance_m"]
        rep["distance_m"] = stats(d)
        rep["path_loss_rssi"] = path_loss(d[data], rows["rssi"][data])
        if "csi_power_db" in rows:
            rep["path_loss_csi_power"] = path_loss(d[data], rows["csi_power_db"][data])
    big = data & (np.array([masks[f].sum() for f in fc]) > 200)   # frames that fill the band
    ds = np.full(n, np.nan)
    if big.any():
        log(f"    {name}: delay spread of {big.sum()} data frames ...")
        H = z["H"][zo]
        ds[big] = delay_spread(H[big], sub)
        del H
        rep["rms_delay_spread_ns"] = stats(ds[big])
    nan = np.full(n, np.nan)
    spd = np.fmax(speed_at(t, rows.get("x", nan), rows.get("y", nan)),
                  speed_at(t, rows.get("peer_x", nan), rows.get("peer_y", nan)))
    coh = coherence(amp, t, fc, masks)
    rep["coherence_by_speed"] = []
    for lo, hi in zip(SPEED_BINS[:-1], SPEED_BINS[1:]):
        m = (spd >= lo) & (spd < hi) & np.isfinite(coh)
        if m.sum() >= 20:
            rep["coherence_by_speed"].append({"speed_mps": [lo, hi], "pairs": int(m.sum()),
                                              "median_correlation": float(np.median(coh[m]))})
    series = {"t": t, "rssi": rows["rssi"], "power": rows.get("csi_power_db"), "fc": fc,
              "d": rows.get("distance_m"), "ds": ds, "speed": spd, "coh": coh, "amp": amp,
              "sub": sub, "masks": masks, "x": rows.get("x"), "y": rows.get("y")}
    return rep, series


def plots(out, links, t0, proc, grid_m):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (no matplotlib: plots skipped)")
        return []
    names = sorted(links)
    files = []
    # timeline
    fig, axs = plt.subplots(len(names), 1, figsize=(12, 3.2 * len(names)), sharex=True, squeeze=False)
    for ax, n in zip(axs[:, 0], names):
        s = links[n]
        b = np.floor(s["t"] - t0)
        k, c = np.unique(b, return_counts=True)
        ax.plot(k, c, color="0.5", lw=0.8, label="packets / s")
        ax.set_ylabel(f"{n}\npackets / s")
        ax2 = ax.twinx()
        r = np.array([np.nanmedian(s["rssi"][b == kk]) for kk in k])
        ax2.plot(k, r, color="C0", lw=1, label="RSSI dBm")
        if s["d"] is not None:
            dd = np.array([np.nanmedian(s["d"][b == kk]) for kk in k])
            ax3 = ax.twinx()
            ax3.spines["right"].set_position(("axes", 1.08))
            ax3.plot(k, dd, color="C2", lw=1)
            ax3.set_ylabel("link m", color="C2")
        ax2.set_ylabel("RSSI dBm", color="C0")
    axs[-1, 0].set_xlabel("s from the first packet")
    fig.tight_layout()
    files.append(os.path.join(out, "csi_timeline.png"))
    fig.savefig(files[-1], dpi=130)
    plt.close(fig)
    # occupancy
    fig, axs = plt.subplots(1, len(names), figsize=(7 * len(names), 4), squeeze=False)
    for ax, n in zip(axs[0], names):
        s = links[n]
        for f in sorted(set(s["fc"])):
            rel, active = occupancy(s["amp"][s["fc"] == f], s["sub"])
            ax.plot(s["sub"], rel, lw=0.8, label="0x%02x %s (%d active)" % (f, fc_name(f), active))
        for k in PILOTS[80]:
            for sg in (-1, 1):
                ax.axvline(sg * k, color="0.85", lw=0.6, zorder=0)
        ax.axhline(-20, color="0.6", lw=0.5, ls="--")
        ax.set_title(n, fontsize=9)
        ax.set_xlabel("subcarrier")
        ax.set_ylabel("median amplitude, dB below the strongest")
        ax.set_title(n + "  (grey lines: VHT80 pilots)", fontsize=9)
        ax.legend(fontsize=7)
    fig.tight_layout()
    files.append(os.path.join(out, "csi_occupancy.png"))
    fig.savefig(files[-1], dpi=130)
    plt.close(fig)
    # path loss
    if any(links[n]["d"] is not None for n in names):
        fig, ax = plt.subplots(figsize=(8, 5.5))
        for i, n in enumerate(names):
            s = links[n]
            if s["d"] is None:
                continue
            data = (s["fc"] >> 2 & 3) == 2
            d, r = s["d"][data], s["rssi"][data]
            sel = np.random.default_rng(0).choice(len(d), min(len(d), 6000), replace=False) \
                if len(d) else []
            ax.scatter(d[sel], r[sel], s=2, alpha=0.25, color=f"C{i}")
            pl = path_loss(d, r)
            if pl:
                dd = np.logspace(np.log10(max(pl["distance_m"][0], 0.3)), np.log10(pl["distance_m"][1]), 50)
                ax.plot(dd, pl["p_at_1m"] - 10 * pl["exponent"] * np.log10(dd), color=f"C{i}",
                        label=f"{n}: n = {pl['exponent']:.2f}, spread {pl['spread_db']:.1f} dB")
        ax.set_xscale("log")
        ax.set_xlabel("link length m")
        ax.set_ylabel("RSSI dBm (data frames)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        files.append(os.path.join(out, "csi_pathloss.png"))
        fig.savefig(files[-1], dpi=130)
        plt.close(fig)
    # coherence
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, n in enumerate(names):
        s = links[n]
        m = np.isfinite(s["coh"]) & np.isfinite(s["speed"])
        if m.sum() < 20:
            continue
        x, y = [], []
        for lo, hi in zip(SPEED_BINS[:-1], SPEED_BINS[1:]):
            k = m & (s["speed"] >= lo) & (s["speed"] < hi)
            if k.sum() >= 20:
                x.append((lo + hi) / 2)
                y.append(np.median(s["coh"][k]))
        ax.plot(x, y, "o-", color=f"C{i}", label=n)
    ax.set_xlabel("robot speed m/s")
    ax.set_ylabel("median correlation, consecutive frames")
    ax.legend(fontsize=8)
    fig.tight_layout()
    files.append(os.path.join(out, "csi_coherence.png"))
    fig.savefig(files[-1], dpi=130)
    plt.close(fig)
    # amplitude heatmaps (QoS data)
    for n in names:
        s = links[n]
        m = s["fc"] == 0x88
        if m.sum() < 10:
            m = (s["fc"] >> 2 & 3) == 2
        if m.sum() < 10:
            continue
        A, tt = s["amp"][m], s["t"][m] - t0
        step = max(1, len(A) // 2000)
        fig, ax = plt.subplots(figsize=(12, 4))
        im = ax.imshow(A[::step].T, aspect="auto", origin="lower", cmap="viridis",
                       extent=(tt[0], tt[-1], s["sub"][0], s["sub"][-1]),
                       vmin=np.nanpercentile(A, 5), vmax=np.nanpercentile(A, 99))
        fig.colorbar(im, ax=ax, label="amplitude dB")
        ax.set_xlabel("s from the first packet")
        ax.set_ylabel("subcarrier")
        ax.set_title(f"{n}: CSI amplitude, data frames", fontsize=9)
        fig.tight_layout()
        files.append(os.path.join(out, f"csi_amplitude_{n}.png"))
        fig.savefig(files[-1], dpi=130)
        plt.close(fig)
    # on the map
    try:
        sys.path.insert(0, HERE)
        import comms_map as cm
        import pass_tf
        cfg = pass_tf.find_config(proc)
        if cfg and any(links[n]["x"] is not None for n in names):
            pcd, path, boards = cm.context(pass_tf.load_pipeline(cfg))
            plan, extent = cm.floor_plan(pcd, (0.3, 1.8)) if pcd else (None, None)
            fig, axs = plt.subplots(1, len(names), figsize=(9 * len(names), 8), squeeze=False)
            from matplotlib.colors import Normalize
            for ax, n in zip(axs[0], names):
                s = links[n]
                data = (s["fc"] >> 2 & 3) == 2
                cells = cm.grid(s["x"][data], s["y"][data], s["rssi"][data], grid_m)
                cm.draw_base(ax, plan, extent, path, boards, {})
                if cells:
                    v = np.array([c[2] for c in cells])
                    pc = cm.draw_cells(ax, cells, grid_m, "viridis",
                                       Normalize(np.percentile(v, 2), np.percentile(v, 98)))
                    fig.colorbar(pc, ax=ax, shrink=0.8, label="dBm")
                ax.set_title(f"CSI {n}: RSSI at the sniffer, median per {grid_m:g} m cell", fontsize=9)
            fig.tight_layout()
            files.append(os.path.join(out, "csi_map.png"))
            fig.savefig(files[-1], dpi=150)
            plt.close(fig)
    except Exception as e:                                         # noqa: BLE001
        print(f"  (map skipped: {e})")
    return files


def f(st, k="median", nd=3):
    return "-" if not st else f"{st[k]:.{nd}g}"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("processed")
    ap.add_argument("--grid", type=float, default=0.5)
    ap.add_argument("--no-plots", action="store_true")
    a = ap.parse_args()
    warnings.simplefilter("ignore", RuntimeWarning)          # empty seconds in the timelines
    root = os.path.dirname(os.path.dirname(HERE))
    sys.path.insert(0, root)
    sys.path.insert(0, os.path.join(root, "map_stages"))
    proc = os.path.abspath(os.path.expanduser(a.processed))
    out = os.path.join(proc, "comms", "csi")
    path = os.path.join(out, "csi.csv")
    if not os.path.exists(path):
        sys.exit(f"no {path} (run processing/comms_tables.py)")
    print(f"reading {path} ...")
    cols = load_csv(path, ["machine", "peer", "t", "t_log", "rssi", "frame_control", "seq",
                           "csi_power_db", "csi_amp_spread_db", "x", "y", "peer_x", "peer_y",
                           "distance_m"])
    links, report = {}, {}
    for peer in sorted(set(cols["peer"])):
        for machine in sorted(set(cols["machine"][cols["peer"] == peer])):
            m = (cols["peer"] == peer) & (cols["machine"] == machine)
            name = "%s_to_%s" % (peer, machine.replace("/", "_"))
            zp = os.path.join(out, f"csi_{name}.npz")
            if not os.path.exists(zp):
                print(f"  (no {zp})")
                continue
            z = np.load(zp, mmap_mode="r")
            rows = {k: v[m] for k, v in cols.items() if k not in ("machine", "peer")}
            rep, series = link_report(name, rows, z)
            report[name], links[name] = rep, series
    t0 = min(np.nanmin(s["t"]) for s in links.values())
    with open(os.path.join(out, "csi_report.json"), "w") as fh:
        json.dump(report, fh, indent=2)

    print("\ncapture:")
    print("  %-32s %8s %7s %7s %9s %10s %12s" % ("link", "packets", "rate Hz", "gaps>1s",
                                                "longest", "at s", "data/s"))
    for n, r in report.items():
        nd = r["frames"].get("0x88 QoS Data", {}).get("packets", 0)
        print("  %-32s %8d %7.1f %7d %8.2fs %10.0f %12.1f" % (
            n, r["packets"], r["rate_hz"], r["gaps_over_1s"], r["longest_gap_s"] or 0,
            r["longest_gap_at_s"] or 0, nd / max(r["duration_s"], 1e-9)))
    print("\nframe types (active = subcarriers within 20 dB of the strongest; VHT 80 MHz fills 242):")
    for n, r in report.items():
        for k, v in r["frames"].items():
            print("  %-32s %-28s %7d (%4.1f%%)  active %3d  RSSI %s dBm  CSI power %s dB" % (
                n, k, v["packets"], v["share_percent"], v["active_subcarriers"],
                f(v["rssi_dbm"]), f(v["csi_power_db"])))
    print("\npath loss (data frames, P = P(1 m) - 10 n log10 d):")
    for n, r in report.items():
        for key, lab in (("path_loss_rssi", "RSSI"), ("path_loss_csi_power", "CSI power")):
            pl = r.get(key)
            if pl:
                print("  %-32s %-9s n = %.2f, P(1 m) %.1f, spread %.1f dB over %.1f-%.1f m (%d)" % (
                    n, lab, pl["exponent"], pl["p_at_1m"], pl["spread_db"], pl["distance_m"][0],
                    pl["distance_m"][1], pl["samples"]))
    print("\nmultipath and motion:")
    for n, r in report.items():
        cs = ", ".join("%.2f-%.2f m/s: %.3f" % (c["speed_mps"][0], c["speed_mps"][1],
                                               c["median_correlation"]) for c in r["coherence_by_speed"])
        print("  %-32s RMS delay spread median %s ns (p95 %s)\n  %-32s consecutive-frame correlation: %s"
              % (n, f(r.get("rms_delay_spread_ns")), f(r.get("rms_delay_spread_ns"), "p95"), "", cs or "-"))
    if not a.no_plots:
        for p in plots(out, links, t0, proc, a.grid):
            print(f"  {p}")
    print(f"  {os.path.join(out, 'csi_report.json')}")


if __name__ == "__main__":
    main()
