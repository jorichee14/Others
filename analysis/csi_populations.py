#!/usr/bin/env python3
"""Are two different channels interleaved in one CSI stream?

coop2 shows |H| correlating 0.99 between consecutive frames whenever the RSSI
word repeats and -0.4 whenever it changes, at every inter-frame gap, on a
single receive core, in both frame classes. That pattern is two stable
populations of frames taking turns, not one channel changing. This script
finds the populations without assuming what they are: k-means with k=2 on
the per-frame normalised log-amplitude, then everything the label predicts.

    python csi_populations.py --run coop2

Prints per agent: cluster sizes, RSSI per cluster, consecutive-frame
correlation within and across clusters, how many frames in a row a cluster
lasts, and whether the label follows the `seq` class or time. Writes
fig_csi_populations.png: the two cluster mean shapes and the label over time.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from csi_core import band_mask, band_outliers, occupied_band, split_populations, usable_subcarriers
from csi_analysis import load_csi, stack_H


def corr_rows(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    A = A - A.mean(1, keepdims=True); B = B - B.mean(1, keepdims=True)
    return (A * B).sum(1) / np.sqrt((A ** 2).sum(1) * (B ** 2).sum(1))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="coop2")
    ap.add_argument("--extracts", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    extracts = args.extracts or Path("extracts") / args.run
    out = args.out or Path("results") / args.run / "csi"
    out.mkdir(parents=True, exist_ok=True)

    csi = load_csi(extracts)
    fig, axes = plt.subplots(len(csi), 2, figsize=(12, 3.2 * len(csi)), squeeze=False)
    for row, agent in enumerate(sorted(csi)):
        df = csi[agent].sort_values("log_time_ns").reset_index(drop=True)
        H_all, idx_all, keep = stack_H(df)
        df = df.loc[keep].reset_index(drop=True)
        use = usable_subcarriers(H_all, 20.0)
        lo, span = occupied_band(idx_all, use, 12)
        use &= band_mask(idx_all, lo, span)
        use[use] &= ~band_outliers(H_all[:, use])
        A = np.abs(H_all[:, use])
        lab, cent, _ = split_populations(H_all[:, use])
        t = (df["log_time_ns"].to_numpy() - df["log_time_ns"].iloc[0]) / 1e9
        rssi = df["rssi"].to_numpy()
        seqcls = (df["seq"].to_numpy().astype(int) != 65535)

        r = corr_rows(A[:-1], A[1:])
        same = lab[:-1] == lab[1:]
        runs = np.diff(np.flatnonzero(np.r_[True, lab[1:] != lab[:-1], True]))
        print(f"\n{agent}: {len(df)} frames")
        print(f"  cluster sizes            {int((lab==0).sum())} / {int((lab==1).sum())}  ({100*(lab==1).mean():.1f}% in the minority one)")
        print(f"  RSSI median per cluster  {np.median(rssi[lab==0]):.0f} / {np.median(rssi[lab==1]):.0f} dBm;"
              f"  RSSI sets overlap: {len(set(rssi[lab==0]) & set(rssi[lab==1]))} shared values")
        print(f"  centre shapes differ by  {np.ptp(cent[0]-cent[1]):.1f} dB peak-to-peak across subcarriers")
        print(f"  consecutive corr         same cluster {np.nanmedian(r[same]):.2f} (n={same.sum()}),"
              f"  across clusters {np.nanmedian(r[~same]):.2f} (n={(~same).sum()})")
        print(f"  RSSI changed when        same cluster {100*np.mean(np.diff(rssi)[same]!=0):.0f}% of pairs,"
              f"  across clusters {100*np.mean(np.diff(rssi)[~same]!=0):.0f}%")
        print(f"  run length (frames)      median {np.median(runs):.0f}, p95 {np.percentile(runs,95):.0f}, max {runs.max()}")
        pct = lambda m: f"{100*lab[m].mean():.1f}%" if m.any() else "n/a"
        print(f"  minority share by seq    seq 65535 {pct(~seqcls)},  other seq {pct(seqcls)}")
        bins = np.floor(t / 10).astype(int)
        share = pd.Series(lab).groupby(bins).mean()
        print(f"  minority share per 10 s  min {100*share.min():.0f}%  median {100*share.median():.0f}%  max {100*share.max():.0f}%")

        ax = axes[row, 0]
        ax.plot(idx_all[use], cent[0], label=f"cluster 0 (n={(lab==0).sum()})")
        ax.plot(idx_all[use], cent[1], label=f"cluster 1 (n={(lab==1).sum()})")
        ax.set(title=f"{agent}: mean normalised |H| per cluster", xlabel="subcarrier slot", ylabel="dB rel. frame mean")
        ax.legend(fontsize=8)
        ax = axes[row, 1]
        ax.scatter(t, rssi, c=lab, s=2, cmap="coolwarm")
        ax.set(title=f"{agent}: RSSI over time, coloured by cluster", xlabel="s", ylabel="dBm")
    fig.tight_layout()
    fig.savefig(out / "fig_csi_populations.png", dpi=160)
    print(f"\nwrote {out}/fig_csi_populations.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
