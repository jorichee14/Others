#!/usr/bin/env python3
"""Pictures of the cleaned CSI (processing/csi_clean.py): one figure per link.

    python scoop_pipeline/analysis/comms/csi_view.py <processed pass folder>
        [--zoom T0 T1] [--max-cols 3000] [--delay-ns 500]

From comms/csi/csi_<link>_clean.npz, writes comms/csi/view/:

  csi_view_<link>.png / .pdf     the whole pass:
    a  link length, RSSI and the receiver's speed (standing shaded)
    b  amplitude per subcarrier, H scaled to the frame's RSSI (dBm per
       subcarrier) -- frequency-selective fading as horizontal structure,
       motion as vertical
    c  phase per subcarrier (linear part removed per frame)
    d  power delay profile: |IFFT(H)|^2 against delay (12.5 ns taps at
       80 MHz), each frame normalised to its strongest tap -- the paths
    e  amplitude across the band at the nearest, a middle and the farthest
       link length
  csi_view_<link>_zoom.png       the same over --zoom (default: 10 s in
                                 the middle of the longest stretch moving),
                                 every frame shown
"""
import argparse
import os
import sys

import numpy as np


def pdp(H, sub, nfft=256, shift=8, chunk=8000):
    """Power delay profile per frame, dB below its strongest tap, strongest
    tap moved to index `shift` (timing offsets differ per frame)."""
    k = np.asarray(sub) % nfft
    out = np.empty((len(H), nfft), np.float32)
    for a in range(0, len(H), chunk):
        F = np.zeros((min(chunk, len(H) - a), nfft), np.complex64)
        F[:, k] = np.nan_to_num(H[a:a + chunk])
        P = np.abs(np.fft.ifft(F, axis=1)) ** 2
        pk = np.argmax(P, axis=1)
        idx = (np.arange(nfft)[None, :] + pk[:, None] - shift) % nfft
        P = np.take_along_axis(P, idx, axis=1)
        out[a:a + chunk] = 10 * np.log10(np.maximum(P / P.max(axis=1, keepdims=True), 1e-6))
    return out


def speed(t, pose):
    if pose is None:
        return np.full(len(t), np.nan)
    p = pose[:, :3, 3]
    ok = np.isfinite(p).all(axis=1)
    if ok.sum() < 10:
        return np.full(len(t), np.nan)
    g = np.arange(t[ok][0], t[ok][-1], 0.2)
    gx = [np.interp(g, t[ok], p[ok, k]) for k in range(2)]
    v = np.hypot(np.gradient(gx[0], g), np.gradient(gx[1], g))
    return np.interp(t, g, np.convolve(v, np.ones(5) / 5, mode="same"))


def default_zoom(t, v, width=10.0):
    """10 s in the middle of the longest stretch above 0.1 m/s."""
    mv = np.nan_to_num(v) > 0.1
    best, cur, start = (None, 0.0), None, None
    for i in range(len(t)):
        if mv[i] and start is None:
            start = i
        if (not mv[i] or i == len(t) - 1) and start is not None:
            d = t[i] - t[start]
            if d > best[1]:
                best = ((start, i), d)
            start = None
    if best[0] is None:
        mid = t[len(t) // 2]
    else:
        mid = 0.5 * (t[best[0][0]] + t[best[0][1]])
    return mid - width / 2, mid + width / 2


def figure(z, name, sel, t0, out, title, delay_ns, max_cols):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = z["t"][sel] - t0
    sub = z["subcarrier"]
    Hs = z["H_rssi"][sel] if "H_rssi" in z.files else z["H"][sel]
    step = max(1, len(t) // max_cols)
    ts, Hc = t[::step], Hs[::step]
    amp = 20 * np.log10(np.maximum(np.abs(Hc), 1e-12)) + 30 * ("H_rssi" in z.files)   # dBm
    ph = z["phase"][sel][::step]
    P = pdp(Hc, sub)
    tap_ns = 1e3 / float(z["bandwidth_mhz"]) if "bandwidth_mhz" in z.files else 12.5   # 1 / bandwidth
    taps = int(delay_ns / tap_ns) + 8
    rx = z["rx_pose"][sel] if "rx_pose" in z.files else None
    v = speed(z["t"][sel], rx)
    d = z["distance_m"][sel] if "distance_m" in z.files else None

    fig = plt.figure(figsize=(13, 15))
    gs = fig.add_gridspec(5, 2, height_ratios=[1, 2, 2, 2, 2], width_ratios=[40, 1], hspace=0.35)
    ext = (ts[0], ts[-1]) if len(ts) > 1 else (t[0], t[0] + 1)
    a = fig.add_subplot(gs[0, 0])
    if d is not None:
        a.plot(t, d, color="C2", lw=0.8)
        a.set_ylabel("link m", color="C2")
    standing = np.nan_to_num(v, nan=1) < 0.05
    for i0 in np.flatnonzero(np.diff(np.r_[0, standing.astype(int)]) == 1):
        i1 = i0 + np.argmax(~standing[i0:]) if (~standing[i0:]).any() else len(t) - 1
        a.axvspan(t[i0], t[i1], color="0.9", lw=0, zorder=0)
    a2 = a.twinx()
    a2.plot(t, z["rssi_dbm"][sel], ".", ms=1, color="C0", alpha=0.4)
    a2.set_ylabel("RSSI dBm", color="C0")
    a.set_xlim(*ext)
    a.set_title(f"{title}  (grey: standing)", fontsize=10)

    def heat(row, img, label, cmap, vmin=None, vmax=None, extent=None, ylabel="subcarrier"):
        ax = fig.add_subplot(gs[row, 0], sharex=a)
        im = ax.imshow(img.T, aspect="auto", origin="lower", cmap=cmap, vmin=vmin, vmax=vmax,
                       extent=extent or (ext[0], ext[1], sub[0], sub[-1]), interpolation="nearest")
        cax = fig.add_subplot(gs[row, 1])
        fig.colorbar(im, cax=cax, label=label)
        ax.set_ylabel(ylabel)
        return ax
    heat(1, amp, "dBm per subcarrier" if "H_rssi" in z.files else "dB", "viridis",
         np.nanpercentile(amp, 2), np.nanpercentile(amp, 99.5))
    heat(2, ph, "rad", "twilight", -np.pi, np.pi)
    axd = heat(3, P[:, :taps], "dB below strongest", "magma", -30, 0,
               extent=(ext[0], ext[1], -8 * tap_ns, (taps - 8) * tap_ns), ylabel="delay ns")
    axd.set_ylim(-2 * tap_ns, delay_ns)
    axd.set_xlabel("s from the first frame of the pass")
    ax = fig.add_subplot(gs[4, 0])
    if d is not None and np.isfinite(d).any():
        picks = [np.nanargmin(d), np.nanargmin(np.abs(d - np.nanmedian(d))), np.nanargmax(d)]
        lab = ["nearest", "middle", "farthest"]
    else:
        picks, lab = [0, len(t) // 2, len(t) - 1], ["first", "middle", "last"]
    Hfull = z["H_rssi"][sel] if "H_rssi" in z.files else z["H"][sel]
    for i, l in zip(picks, lab):
        y = 20 * np.log10(np.maximum(np.abs(Hfull[i]), 1e-12)) + 30 * ("H_rssi" in z.files)
        ax.plot(sub, y, lw=0.9, label=f"{l}: t {t[i]:.1f} s" + (f", {d[i]:.1f} m" if d is not None else "")
                + f", RSSI {z['rssi_dbm'][sel][i]:.0f} dBm")
    ax.set_xlabel("subcarrier")
    ax.set_ylabel("dBm per subcarrier" if "H_rssi" in z.files else "dB")
    ax.legend(fontsize=8)
    fig.savefig(out + ".png", dpi=140, bbox_inches="tight")
    files = [out + ".png"]
    if not out.endswith("_zoom"):
        fig.savefig(out + ".pdf", bbox_inches="tight")
        files.append(out + ".pdf")
    plt.close(fig)
    return files


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("processed")
    ap.add_argument("--zoom", type=float, nargs=2, default=None,
                    help="s from the first frame of the pass")
    ap.add_argument("--max-cols", type=int, default=3000)
    ap.add_argument("--delay-ns", type=float, default=500.0)
    a = ap.parse_args()
    csi = os.path.join(os.path.abspath(os.path.expanduser(a.processed)), "comms", "csi")
    files = sorted(f for f in os.listdir(csi) if f.endswith("_clean.npz")) if os.path.isdir(csi) else []
    if not files:
        sys.exit(f"no *_clean.npz in {csi} (run processing/csi_clean.py)")
    out = os.path.join(csi, "view")
    os.makedirs(out, exist_ok=True)
    zs = {f[len("csi_"):-len("_clean.npz")]: np.load(os.path.join(csi, f)) for f in files}
    t0 = min(float(z["t"][0]) for z in zs.values())
    for name, z in zs.items():
        n = len(z["t"])
        print(f"{name}: {n} frames, {len(z['subcarrier'])} subcarriers")
        for p in figure(z, name, slice(None), t0, os.path.join(out, f"csi_view_{name}"),
                        f"CSI {name}, whole pass", a.delay_ns, a.max_cols):
            print(f"  {p}")
        rx = z["rx_pose"] if "rx_pose" in z.files else None
        lo, hi = a.zoom if a.zoom else [x - t0 for x in default_zoom(z["t"], speed(z["t"], rx))]
        sel = np.flatnonzero((z["t"] - t0 >= lo) & (z["t"] - t0 <= hi))
        if len(sel) > 10:
            for p in figure(z, name, sel, t0, os.path.join(out, f"csi_view_{name}_zoom"),
                            f"CSI {name}, {lo:.0f}-{hi:.0f} s, every frame", a.delay_ns, 10 ** 6):
                print(f"  {p}")


if __name__ == "__main__":
    main()
