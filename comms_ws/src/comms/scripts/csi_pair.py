#!/usr/bin/env python3
"""One figure: amplitude-vs-time and RSSI for two transmitters, shared time axis.

Same receiver, same window, same processing (csi_nexmon: fixed null list,
RSSI rescaling), one colour scale across both heat maps so the panels can be
read against each other.

    (a) mobile_1  |H| vs time
    (b) mobile_2  |H| vs time
    (c) RSSI, both

RSSI is drawn under the heat maps because it is the receiver's gain input:
where it steps, the reported amplitude shape changes; where it is flat, the
heat map shows the channel. Reading (c) against (a) and (b) is how you tell a
parked robot from a receiver artefact.

    python3 csi_pair.py --bag BAG --window 60 90
    python3 csi_pair.py --bag BAG --topics /mobile1/csi /mobile2/csi
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from csi_nexmon import (from_bag, stack, occupied_span, null_slots,  # noqa: E402
                        rssi_rescale, CONTROL_FC, GRID, TEXT)

COLOR = ["#1f4e9c", "#e08214"]


def prepare(rows, frames: str):
    H, meta = stack(rows)
    n_fft = H.shape[1]
    bw = int(meta["bw"].mode().iloc[0])
    if frames == "control":
        m = meta["is_control"].to_numpy()
    elif frames == "data":
        m = ~meta["is_control"].to_numpy()
    else:
        m = np.ones(len(meta), bool)
    H, meta = H[m], meta[m].reset_index(drop=True)
    amp = np.abs(H)
    good = (amp <= 0).mean(axis=1) < 0.5
    H, amp, meta = H[good], amp[good], meta[good].reset_index(drop=True)
    centre, occ = occupied_span(amp, n_fft)
    keep = ~null_slots(n_fft, bw, centre, occ)
    kk = (np.arange(n_fft) - centre)[keep]
    Hk = rssi_rescale(H[:, keep], meta["rssi"].to_numpy())
    A_db = 20 * np.log10(np.maximum(np.abs(Hk), 1e-12))
    frame_bw = {64: 20, 128: 40, 256: 80}.get(occ, bw)
    return dict(A=A_db, kk=kk, t=meta["t_ns"].to_numpy(), rssi=meta["rssi"].to_numpy(),
                n=len(meta), usable=int(keep.sum()), n_fft=n_fft, bw=bw, fbw=frame_bw,
                fc=meta["fc"].value_counts())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bag", type=Path, required=True)
    ap.add_argument("--topics", nargs=2, default=["/mobile1/csi", "/mobile2/csi"])
    ap.add_argument("--names", nargs=2, default=["mobile_1", "mobile_2"])
    ap.add_argument("--frames", choices=["all", "control", "data"], default="all")
    ap.add_argument("--window", type=float, nargs=2, default=None, metavar=("T0", "T1"))
    ap.add_argument("--out", type=Path, default=Path("csi_pair.png"))
    args = ap.parse_args(argv)

    per = [prepare(from_bag(args.bag, tp), args.frames) for tp in args.topics]
    t0 = min(p["t"][0] for p in per)
    for p in per:
        p["ts"] = (p["t"] - t0) / 1e9

    lo = args.window[0] if args.window else min(p["ts"][0] for p in per)
    hi = args.window[1] if args.window else max(p["ts"][-1] for p in per)
    for p in per:
        p["w"] = (p["ts"] >= lo) & (p["ts"] <= hi)
        if p["w"].sum() < 5:
            raise SystemExit(f"fewer than 5 frames in {lo}-{hi} s for one topic")

    # one colour scale for both
    allv = np.concatenate([p["A"][p["w"]].ravel() for p in per])
    vlo, vhi = np.percentile(allv, [1, 99])

    plt.rcParams.update({"font.size": 8.5, "axes.edgecolor": GRID,
                         "axes.labelcolor": TEXT, "text.color": TEXT})
    span = hi - lo
    # Width scales with the window so a full run is not compressed into the
    # same pixels as a 30 s window: ~0.4 in per second, clamped.
    width = float(np.clip(12 + 0.35 * max(span - 30, 0), 12, 40))
    fig, axes = plt.subplots(3, 1, figsize=(width, 9.4), sharex=True,
                             gridspec_kw=dict(height_ratios=[1.6, 1.6, 1.0], hspace=0.22))

    for i, (p, name) in enumerate(zip(per, args.names)):
        ax = axes[i]
        Aw, tw = p["A"][p["w"]], p["ts"][p["w"]]
        step = max(Aw.shape[0] // int(1600 * width / 12), 1)
        im = ax.imshow(Aw[::step].T, aspect="auto", origin="lower", cmap="jet",
                       vmin=vlo, vmax=vhi,
                       extent=[tw[0], tw[-1], p["kk"][0] - 0.5, p["kk"][-1] + 0.5])
        ax.set_ylabel("subcarrier k")
        fcs = ", ".join(f"0x{k:02x} {CONTROL_FC.get(k, 'DATA')} x{v}" for k, v in p["fc"].items())
        ax.set_title(f"({chr(97+i)}) {name}: |H| vs time — {int(p['w'].sum())} frames in window, "
                     f"{p['fbw']} MHz frame in {p['bw']} MHz window, "
                     f"{p['usable']}/{p['n_fft']} usable, RSSI-rescaled   [{fcs}]",
                     loc="left")
        cb = fig.colorbar(im, ax=ax, pad=0.01)
        cb.set_label("|H| [dB]")

    ax = axes[2]
    for i, (p, name) in enumerate(zip(per, args.names)):
        ax.plot(p["ts"][p["w"]], p["rssi"][p["w"]], lw=0.8, color=COLOR[i], label=name)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("RSSI [dBm]")
    ax.set_title("(c) RSSI at the receiver — the AGC input. Where it steps, the reported "
                 "amplitude shape changes; where it is flat, the heat map is the channel.",
                 loc="left")
    ax.legend(frameon=False, ncol=2)
    ax.grid(True, color=GRID, lw=0.5)
    ax.set_xlim(lo, hi)
    # keep the RSSI axis the same width as the heat maps (colour bars steal space)
    for a in axes[:2]:
        pos = a.get_position()
    pos2 = axes[2].get_position()
    axes[2].set_position([pos2.x0, pos2.y0, pos.width, pos2.height])

    fig.savefig(args.out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
