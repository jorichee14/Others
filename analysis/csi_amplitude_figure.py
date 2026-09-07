#!/usr/bin/env python3
"""The one CSI figure for a run whose channel cannot be attributed to the robots.

    python3 csi_amplitude_figure.py --run coop2                 # heat map
    python3 csi_amplitude_figure.py --run coop2 --style lines   # overlaid frames

Two conventional views of the same thing, and they claim nothing about the
room. `heatmap`: amplitude per agent, subcarrier against time, with the
receiver's fixed per-subcarrier shape divided out. `lines`: the Intel-5300
style plot, every sampled frame's |H| across subcarriers overlaid, one colour
per agent, on an absolute dBm axis restored from each frame's RSSI; the shape
of the bundle is the receiver, its thickness the fading, its height the
signal strength, and the run median is drawn bold.

Reads the extraction made by csi_analysis.py (or extract_bag.py) and writes
results/<run>/csi/fig_csi_amplitude.{pdf,png}.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import GRID, TEXT, TEXT2, color_for  # noqa: E402
from csi_analysis import AMP_DIVERGING, FRAME_TYPES, load_csi, stack_H  # noqa: E402
from csi_core import (amplitude_db, band_mask, band_outliers,  # noqa: E402
                      effective_bandwidth_mhz, equalise_static, occupied_band,
                      usable_subcarriers)


def lines_figure(panels, out: Path, n_lines: int) -> int:
    """Every sampled frame's |H| across subcarriers, overlaid, one colour per agent.

    The y axis is absolute, in dBm per subcarrier, by the same trick the Intel
    5300 tool uses: the chip's CSI comes out after automatic gain control, so
    its raw level is the receiver's gain setting and not the channel, but the
    frame's reported RSSI is the level the receiver saw. Scale each frame so
    its mean subcarrier power equals its RSSI and the level is restored --
    a weak frame sits low on the axis, a strong one high."""
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    fig.subplots_adjust(left=0.13, right=0.97, top=0.9, bottom=0.16)
    for p in panels:
        A = p["abs_db"]
        pick = np.linspace(0, A.shape[0] - 1, min(n_lines, A.shape[0])).astype(int)
        c = color_for(p["agent"])
        ax.plot(p["idx"], A[pick].T, color=c, lw=0.4, alpha=0.08)
        ax.plot(p["idx"], np.median(A, axis=0), color=c, lw=1.8,
                label=f"{p['agent']}  ({p['frame']}, {len(pick)} of {A.shape[0]} frames)")
    ax.set_xlabel("FFT slot (subcarrier)")
    ax.set_ylabel("amplitude per subcarrier [dBm], scaled to frame RSSI")
    ax.set_title("CSI amplitude per frame; bold = run median",
                 loc="left", fontsize=8)
    ax.legend(frameon=False, fontsize=7, loc="lower center")
    ax.grid(True, color=GRID, lw=0.5)
    fig.savefig(out / "fig_csi_lines.pdf", bbox_inches="tight")
    fig.savefig(out / "fig_csi_lines.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    for p in panels:
        print(f"{p['agent']}: receiver shape {float(np.ptp(p['static_db'])):.0f} dB peak to peak, "
              f"fading spread about it {float(np.median(p['amp_db'].std(axis=1))):.1f} dB")
    print(f"wrote {out}/fig_csi_lines.pdf/png")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="run")
    ap.add_argument("--extracts", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--null-floor-db", type=float, default=20.0)
    ap.add_argument("--band-gap", type=int, default=12)
    ap.add_argument("--style", choices=["heatmap", "lines"], default="heatmap")
    ap.add_argument("--n-lines", type=int, default=400, help="frames overlaid in --style lines")
    args = ap.parse_args()

    extracts = args.extracts or Path("extracts") / args.run
    out = args.out or Path("results") / args.run / "csi"
    out.mkdir(parents=True, exist_ok=True)
    csi = load_csi(extracts)
    if not csi:
        raise SystemExit(f"no *csi.parquet in {extracts}; run csi_analysis.py --bag first")
    t0_ns = min(int(d["log_time_ns"].min()) for d in csi.values())

    panels = []
    for agent in sorted(csi):
        df = csi[agent].sort_values("log_time_ns").reset_index(drop=True)
        df["t_s"] = (df["log_time_ns"] - t0_ns) / 1e9
        sub = df.iloc[:: max(args.stride, 1)].reset_index(drop=True)
        H_all, idx_all, keep = stack_H(sub)
        sub = sub.loc[keep].reset_index(drop=True)
        bw = int(df["bandwidth_mhz"].mode().iloc[0])
        raw_slots = int(df["raw_slots"].mode().iloc[0])
        use = usable_subcarriers(H_all, args.null_floor_db)
        lo, span = occupied_band(idx_all, use, args.band_gap)
        use &= band_mask(idx_all, lo, span)
        # and, inside the band, drop the LO-leakage spike at the window centre and
        # the filter-skirt slots at the block edge: neither is a subcarrier
        use[use] &= ~band_outliers(H_all[:, use])
        # the axis covers the slots actually drawn, not the band before trimming
        lo, span = int(idx_all[use].min()), int(idx_all[use].max() - idx_all[use].min() + 1)
        H_raw = H_all[:, use]
        H, static_db = equalise_static(H_raw)
        dur = float(df["t_s"].max() - df["t_s"].min())
        fc = int(df["frame_control"].mode().iloc[0])
        panels.append(dict(
            agent=agent, t=sub["t_s"].to_numpy(), amp_db=amplitude_db(H), idx=idx_all[use],
            raw_db=amplitude_db(H_raw), static_db=static_db,
            abs_db=(20 * np.log10(np.abs(H_raw) + 1e-12)
                    - 10 * np.log10(np.maximum((np.abs(H_raw) ** 2).mean(axis=1, keepdims=True), 1e-24))
                    + sub["rssi"].to_numpy(float)[:, None]),
            lo=lo, span=span, bw=bw, eff_bw=effective_bandwidth_mhz(span, bw, raw_slots),
            rate=(len(df) - 1) / max(dur, 1e-9), frame=FRAME_TYPES.get(fc, f"0x{fc:02x}"),
            shape_db=float(np.ptp(static_db))))

    plt.rcParams.update({"font.size": 8, "axes.edgecolor": GRID, "axes.labelcolor": TEXT,
                         "xtick.color": TEXT2, "ytick.color": TEXT2, "text.color": TEXT})
    n = len(panels)
    if args.style == "lines":
        return lines_figure(panels, out, args.n_lines)
    fig = plt.figure(figsize=(4.6 * n + 1.0, 3.1))
    gs = fig.add_gridspec(1, n + 1, width_ratios=[1.0] * n + [0.05], wspace=0.24,
                          left=0.07, right=0.93, top=0.80, bottom=0.17)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("amp", AMP_DIVERGING)
    cmap.set_bad("#f7f6f3")
    vabs = float(np.percentile(np.abs(np.concatenate([p["amp_db"].ravel() for p in panels])), 98))
    im = None
    for i, p in enumerate(panels):
        ax = fig.add_subplot(gs[0, i])
        step = max(len(p["t"]) // 1200, 1)
        grid = np.full((p["amp_db"].shape[0], p["span"]), np.nan, np.float32)
        grid[:, p["idx"] - p["lo"]] = p["amp_db"]
        im = ax.imshow(grid[::step].T, aspect="auto", origin="lower", cmap=cmap,
                       vmin=-vabs, vmax=vabs,
                       extent=[p["t"][0], p["t"][-1], p["lo"] - 0.5, p["lo"] + p["span"] - 0.5])
        ax.set_xlabel("time in run [s]")
        ax.set_ylabel("FFT slot (subcarrier)" if i == 0 else "")
        ax.set_title(f"({chr(97 + i)}) {p['agent']}: {p['frame']} frames at {p['rate']:.0f} Hz, "
                     f"{len(p['idx'])} subcarriers\n"
                     f"{p['eff_bw']:.0f} MHz of an {p['bw']} MHz capture, "
                     f"{p['shape_db']:.0f} dB receiver shape removed",
                     loc="left", fontsize=8)
    cax = fig.add_subplot(gs[0, n])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("|H| relative to frame median [dB]", fontsize=7)
    cb.ax.tick_params(labelsize=7)
    cb.outline.set_visible(False)
    fig.savefig(out / "fig_csi_amplitude.pdf", bbox_inches="tight")
    fig.savefig(out / "fig_csi_amplitude.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    for p in panels:
        print(f"{p['agent']}: {p['frame']} frames, {p['rate']:.0f} Hz, {p['eff_bw']:.0f} MHz of "
              f"{p['bw']} MHz, {len(p['idx'])} subcarriers, receiver shape {p['shape_db']:.0f} dB")
    print(f"wrote {out}/fig_csi_amplitude.pdf/png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
