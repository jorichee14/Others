#!/usr/bin/env python3
"""Nexmon CSI (BCM43455c0) analysis, following the practice published for this chip.

Nothing here is novel. It reproduces what the nexmon_csi README, csi-explorer,
and the Pi-based sensing papers do, so results are comparable to theirs.

NULL SUBCARRIERS -- fixed indices from the nexmon_csi README, not detected
    20 MHz 802.11n/ac : -32..-29, 0, +29..+31          (57 usable of 64)
    40 MHz            : -64..-59, -1..+1, +59..+63     (114 usable of 128)
    80 MHz            : -128..-123, -1..+1, +123..+127 (242 usable of 256)
  The README says guard and null carriers "might contain arbitrary values" and
  recommends removing them. The Pi HAR paper reports 114 of 128 useful at
  40 MHz, which is exactly this list.

  Indexing: the firmware reports slots 0..N-1 in FFT order. Subcarrier k maps
  to slot (k + N/2) mod N -- i.e. slot N/2 is DC (k=0), slot 0 is k=-N/2.

NARROW FRAME IN A WIDE WINDOW
  A 20 MHz frame captured with an 80 MHz chanspec fills only one quarter of
  the 256 slots; the remainder is noise floor. The occupied quarter is found
  from the mean power profile (the only data-driven step), and the 20 MHz
  null list is then applied INSIDE it, relative to its own centre.

AGC -- RSSI rescaling, per frame
  The receiver's automatic gain control rescales every frame by an unknown
  factor. The standard mitigation (receiver-effects paper, eq. 3) rescales
  each frame so its total CSI power matches the reported RSSI:

      s = sqrt( 10^(RSSI/10) / sum_k |H_k|^2 ),   H_k <- s * H_k

  This removes the SCALE. The literature notes RSSI is coarse and the
  correction is imperfect; it does not claim to remove gain-dependent
  changes in SHAPE. On this capture, consecutive frames correlate at 0.99
  when RSSI is unchanged and ~0 when it steps, so shape changes with gain
  state are present and are reported as a known limitation, not corrected.

THE FOUR STANDARD PLOTS  (csi-explorer / csi-visualization)
    1. heat map, amplitude vs packet index x subcarrier
    2. heat map, amplitude vs time x subcarrier
    3. amplitude vs subcarrier, all packets overlaid
    4. amplitude vs time for a few subcarriers

Usage
    python3 csi_nexmon.py --bag BAG --topic /mobile1/csi
    python3 csi_nexmon.py --bag BAG --topic /mobile1/csi --window 60 90

Frames are classified on frame_control (0x94 Block Ack, 0x80 Beacon, ...);
anything else is data. seq alone is not reliable: Block Acks were observed
carrying real sequence numbers.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402
import pandas as pd               # noqa: E402

CONTROL_FC = {0x94: "Block Ack", 0x80: "Beacon", 0x84: "control",
              0x54: "control", 0x24: "control", 0xC4: "CTS", 0xD4: "ACK",
              0xB4: "RTS", 0x40: "Probe Req", 0x50: "Probe Resp"}

# nexmon_csi README: null/guard subcarrier indices (k, DC = 0)
NULLS = {
    20: list(range(-32, -28)) + [0] + list(range(29, 32)),
    40: list(range(-64, -58)) + [-1, 0, 1] + list(range(59, 64)),
    80: list(range(-128, -122)) + [-1, 0, 1] + list(range(123, 128)),
}
GRID, TEXT = "#d8d5d0", "#2b2926"


# ============================================================ null handling
def null_slots(n_fft: int, bw_mhz: int, centre_slot: int | None = None,
               occ_slots: int | None = None) -> np.ndarray:
    """Boolean mask of null slots for a frame of `bw_mhz` inside an `n_fft` FFT.

    If the frame is narrower than the window (occ_slots < n_fft), the null
    list for the FRAME's bandwidth is applied relative to the frame's own
    centre slot, and everything outside the occupied span is null too.
    """
    occ = occ_slots or n_fft
    centre = centre_slot if centre_slot is not None else n_fft // 2
    frame_bw = {64: 20, 128: 40, 256: 80}.get(occ, bw_mhz)
    ks = NULLS.get(frame_bw, NULLS[80])
    mask = np.ones(n_fft, bool)                       # start: all null
    lo, hi = centre - occ // 2, centre + occ // 2
    mask[max(lo, 0):min(hi, n_fft)] = False           # occupied span
    for k in ks:
        s = centre + k
        if 0 <= s < n_fft:
            mask[s] = True
    return mask


def occupied_span(amp: np.ndarray, n_fft: int) -> tuple[int, int]:
    """(centre_slot, occ_slots) of the frame inside the window, from power.

    The one data-driven step. Mean power per slot is median-smoothed; slots
    within 20 dB of the 90th percentile are occupied; the widest contiguous
    run is snapped to the nearest of 64 / 128 / 256 slots, since a frame can
    only be 20, 40 or 80 MHz.
    """
    p = 10 * np.log10(np.maximum((amp ** 2).mean(axis=0), 1e-30))
    k = 9
    pad = np.pad(p, k // 2, mode="edge")
    ps = np.median(np.lib.stride_tricks.sliding_window_view(pad, k), axis=1)
    live = ps > np.percentile(ps, 90) - 20.0
    best, lo = (0, 0, 0), None
    for i, v in enumerate(np.r_[live, False]):
        if v and lo is None:
            lo = i
        elif not v and lo is not None:
            if i - lo > best[0]:
                best = (i - lo, lo, i)
            lo = None
    width, a, b = best
    if width == 0:
        return n_fft // 2, n_fft
    occ = min((64, 128, 256), key=lambda x: abs(x - width))
    occ = min(occ, n_fft)
    centre = (a + b) // 2
    # Snap to DC: it is always the deepest slot inside the occupied span, and
    # the run midpoint can land a slot off, which then keeps a guard slot as
    # data at one edge and drops a real subcarrier at the other.
    lo, hi = max(centre - 4, 0), min(centre + 5, n_fft)
    centre = lo + int(np.argmin(p[lo:hi]))
    return centre, occ


# ============================================================ AGC rescaling
def rssi_rescale(H: np.ndarray, rssi_dbm: np.ndarray) -> np.ndarray:
    """Per-frame rescale so total CSI power matches RSSI (eq. 3)."""
    pw = (np.abs(H) ** 2).sum(axis=1)
    s = np.sqrt(10 ** (rssi_dbm.astype(float) / 10.0) / np.maximum(pw, 1e-30))
    return H * s[:, None]


# ============================================================ input
def from_bag(bag: Path, topic: str, limit: int | None = None):
    try:
        from rosbags.highlevel import AnyReader
    except ImportError:
        raise SystemExit("needs: pip install rosbags")
    rows = []
    with AnyReader([bag]) as reader:
        conns = [c for c in reader.connections if c.topic == topic]
        if not conns:
            have = sorted({c.topic for c in reader.connections})
            raise SystemExit(f"topic {topic} not in bag. Present: {have}")
        for i, (conn, ts, raw) in enumerate(reader.messages(connections=conns)):
            m = reader.deserialize(raw, conn.msgtype)
            rows.append({
                "t_ns": ts, "seq": int(m.seq), "rssi": int(m.rssi),
                "fc": int(m.frame_control), "src": str(m.src_mac),
                "bw": int(m.bandwidth_mhz), "core": int(m.core),
                "ss": int(m.spatial_stream),
                "re": np.asarray(m.csi_real, dtype=np.float64),
                "im": np.asarray(m.csi_imag, dtype=np.float64),
            })
            if limit and i + 1 >= limit:
                break
    return rows


def stack(rows):
    sizes = pd.Series([r["re"].size for r in rows])
    n = int(sizes.mode().iloc[0])
    keep = [r for r in rows if r["re"].size == n and r["im"].size == n]
    H = np.stack([r["re"] + 1j * r["im"] for r in keep])
    meta = pd.DataFrame([{k: r[k] for k in
                          ("t_ns", "seq", "rssi", "fc", "src", "bw", "core", "ss")}
                         for r in keep])
    meta["t_s"] = (meta["t_ns"] - meta["t_ns"].min()) / 1e9
    meta["is_control"] = meta["fc"].isin(CONTROL_FC)
    return H, meta


# ============================================================ main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bag", type=Path, required=True)
    ap.add_argument("--topic", default="/mobile1/csi")
    ap.add_argument("--out", type=Path, default=Path("csi_nexmon"))
    ap.add_argument("--frames", choices=["all", "control", "data"], default="all")
    ap.add_argument("--window", type=float, nargs=2, default=None,
                    metavar=("T0", "T1"), help="seconds to plot")
    ap.add_argument("--no-rssi-rescale", action="store_true",
                    help="skip the AGC rescaling (raw firmware amplitudes)")
    ap.add_argument("--limit", type=int, default=None, help="read at most N frames")
    args = ap.parse_args(argv)

    H, meta = stack(from_bag(args.bag, args.topic, args.limit))
    n_fft = H.shape[1]
    bw = int(meta["bw"].mode().iloc[0])

    # frame population
    if args.frames == "control":
        m = meta["is_control"].to_numpy()
    elif args.frames == "data":
        m = ~meta["is_control"].to_numpy()
    else:
        m = np.ones(len(meta), bool)
    H, meta = H[m], meta[m].reset_index(drop=True)
    if len(meta) < 10:
        raise SystemExit(f"only {len(meta)} '{args.frames}' frames")

    # empty (start-up) frames
    amp = np.abs(H)
    good = (amp <= 0).mean(axis=1) < 0.5
    n_zero = int((~good).sum())
    H, amp, meta = H[good], amp[good], meta[good].reset_index(drop=True)

    # null removal: fixed indices, relative to the frame's own centre
    centre, occ = occupied_span(amp, n_fft)
    null = null_slots(n_fft, bw, centre, occ)
    keep = ~null
    k_idx = np.arange(n_fft) - centre                 # subcarrier k of each slot
    Hk = H[:, keep]
    kk = k_idx[keep]

    # AGC
    if not args.no_rssi_rescale:
        Hk = rssi_rescale(Hk, meta["rssi"].to_numpy())
    A = np.abs(Hk)
    A_db = 20 * np.log10(np.maximum(A, 1e-12))

    # window
    t = meta["t_s"].to_numpy()
    if args.window:
        w = (t >= args.window[0]) & (t <= args.window[1])
    else:
        w = np.ones(len(t), bool)

    # ---- the gain-state check the literature does not do, reported not corrected
    a = A[:-1] - A[:-1].mean(1, keepdims=True)
    b = A[1:] - A[1:].mean(1, keepdims=True)
    r = (a * b).sum(1) / np.sqrt((a * a).sum(1) * (b * b).sum(1))
    dr = np.abs(np.diff(meta["rssi"].to_numpy()))
    r_same = float(np.nanmedian(r[dr == 0])) if (dr == 0).any() else np.nan
    r_chg = float(np.nanmedian(r[dr > 0])) if (dr > 0).any() else np.nan

    args.out.mkdir(parents=True, exist_ok=True)

    # ================================================================ figure
    plt.rcParams.update({"font.size": 8, "axes.edgecolor": GRID,
                         "axes.labelcolor": TEXT, "text.color": TEXT})
    fig = plt.figure(figsize=(13, 10))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.2, 1.2, 1.0], hspace=0.45, wspace=0.22)
    vlo, vhi = np.percentile(A_db[w], [1, 99])
    fc_mix = ", ".join(f"0x{k:02x} {CONTROL_FC.get(k, 'DATA')} x{v}"
                       for k, v in meta["fc"].value_counts().items())
    frame_bw = {64: 20, 128: 40, 256: 80}.get(occ, bw)

    # 1. amplitude vs packet index
    ax = fig.add_subplot(gs[0, 0])
    Aw = A_db[w]
    step = max(Aw.shape[0] // 1500, 1)
    im = ax.imshow(Aw[::step].T, aspect="auto", origin="lower", cmap="jet",
                   vmin=vlo, vmax=vhi,
                   extent=[0, Aw.shape[0], kk[0] - 0.5, kk[-1] + 0.5])
    ax.set_xlabel("packet index"); ax.set_ylabel("subcarrier k")
    ax.set_title("1  amplitude vs packet index", loc="left")
    fig.colorbar(im, ax=ax, label="|H| [dB]")

    # 2. amplitude vs time
    ax = fig.add_subplot(gs[0, 1])
    im = ax.imshow(Aw[::step].T, aspect="auto", origin="lower", cmap="jet",
                   vmin=vlo, vmax=vhi,
                   extent=[t[w][0], t[w][-1], kk[0] - 0.5, kk[-1] + 0.5])
    ax.set_xlabel("time [s]"); ax.set_ylabel("subcarrier k")
    ax.set_title("2  amplitude vs time", loc="left")
    fig.colorbar(im, ax=ax, label="|H| [dB]")

    # 3. amplitude vs subcarrier, all packets overlaid
    ax = fig.add_subplot(gs[1, 0])
    sub = A_db[w][::max(Aw.shape[0] // 400, 1)]
    ax.plot(kk, sub.T, lw=0.4, color="#1f4e9c", alpha=0.08)
    ax.plot(kk, np.median(A_db[w], axis=0), lw=1.6, color="#e34948", label="median")
    ax.set_xlabel("subcarrier k"); ax.set_ylabel("|H| [dB]")
    ax.set_title("3  amplitude vs subcarrier, all packets", loc="left")
    ax.legend(frameon=False); ax.grid(True, color=GRID, lw=0.5)

    # 4. a few subcarriers over time
    ax = fig.add_subplot(gs[1, 1])
    for j in np.linspace(0, len(kk) - 1, 4).astype(int):
        ax.plot(t[w], A_db[w][:, j], lw=0.6, label=f"k={kk[j]}")
    ax.set_xlabel("time [s]"); ax.set_ylabel("|H| [dB]")
    ax.set_title("4  selected subcarriers vs time", loc="left")
    ax.legend(frameon=False, ncol=4, fontsize=7); ax.grid(True, color=GRID, lw=0.5)

    # 5. RSSI over time, since it is what the AGC correction keys on
    ax = fig.add_subplot(gs[2, :])
    ax.plot(t[w], meta["rssi"].to_numpy()[w], lw=0.7, color="#e08214")
    ax.set_xlabel("time [s]"); ax.set_ylabel("RSSI [dBm]")
    ax.set_title(f"5  RSSI -- the AGC correction rescales each frame to this. "
                 f"Consecutive-frame |H| correlation: {r_same:.2f} when RSSI unchanged, "
                 f"{r_chg:.2f} when it steps", loc="left")
    ax.grid(True, color=GRID, lw=0.5)

    fig.suptitle(f"{args.topic}: {len(meta)} frames, {frame_bw} MHz frame in a "
                 f"{bw} MHz window, {keep.sum()} of {n_fft} slots usable "
                 f"(nexmon null list), RSSI-rescaled"
                 f"{' OFF' if args.no_rssi_rescale else ''}, {n_zero} empty frames "
                 f"dropped\n{fc_mix}", fontsize=9, y=0.995)
    fig.savefig(args.out / "fig_csi_nexmon.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # ================================================================ summary
    md = [f"# Nexmon CSI — `{args.topic}`", "",
          "Processing follows the practice published for the BCM43455c0: null "
          "subcarriers removed by the fixed index list in the nexmon_csi README, "
          "per-frame RSSI rescaling for AGC. Nothing else is applied.", "",
          f"- frames: {len(meta)} ({fc_mix}); {n_zero} empty frames dropped",
          f"- capture window {bw} MHz ({n_fft} slots); frame occupies {frame_bw} MHz, "
          f"centre slot {centre}",
          f"- usable subcarriers: {keep.sum()} of {n_fft} after the null list "
          f"(k = {kk[0]}..{kk[-1]})",
          f"- RSSI median {meta['rssi'].median():.0f} dBm, "
          f"{meta['rssi'].nunique()} distinct values", "",
          "## Known limitation: gain-dependent shape", "",
          f"Consecutive frames correlate at **{r_same:.2f}** when RSSI is unchanged "
          f"and **{r_chg:.2f}** when it steps, regardless of the time gap between "
          "them. RSSI rescaling corrects scale, not this. The receiver-effects "
          "literature reports the same class of artefact and notes RSSI is too "
          "coarse to remove it fully. Analyses that compare frames across RSSI "
          "steps should account for it.", ""]
    (args.out / "csi_nexmon.md").write_text("\n".join(md))

    print(f"{len(meta)} frames | {frame_bw} MHz frame in {bw} MHz window | "
          f"{keep.sum()}/{n_fft} usable | RSSI-same r={r_same:.2f}  RSSI-step r={r_chg:.2f}")
    print(f"wrote {args.out}/fig_csi_nexmon.png, csi_nexmon.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
