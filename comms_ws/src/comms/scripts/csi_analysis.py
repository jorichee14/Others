#!/usr/bin/env python3
"""Wi-Fi CSI analysis for a Nexmon capture, corrected for what this rig does.

Standalone: reads a rosbag2 bag (via `rosbags`) or the live topic, and depends
on nothing else in the tree.

WHAT THIS FIXES RELATIVE TO A GENERIC CSI SCRIPT
------------------------------------------------
1. FRAME TYPE: USE frame_control, WITH seq ONLY AS A TIE-BREAK.
   The field holds the FIRST BYTE of the captured PPDU, not a parsed 802.11
   frame-control field. For a non-aggregated frame the two coincide, so Block
   Acks show 0x94 and beacons 0x80 correctly. Bulk data is sent as an A-MPDU,
   whose first byte is an MPDU delimiter, so QoS Data NEVER appears as 0x88 --
   on this rig it shows up as 0xE0. Selecting frames on 0x88 yields an empty
   set while the data is sitting right there.

   So data frames are identified by frame_control NOT being one of the known
   control/management values (0x94 Block Ack, 0x80 Beacon, ...). On this rig
   they appear as 0xE0, with sequence numbers stepping +16 -- one A-MPDU of
   16 MPDUs per CSI record.

   `seq` alone is NOT sufficient, which cost a wrong conclusion once: in a
   capture containing only Block Acks, 7626 of 26757 of them carried a real
   sequence number rather than the 0xFFFF placeholder, and a seq-based split
   reported them as a data-frame population. Both "populations" then produced
   near-identical metrics, which was the clue. Classify on frame_control.

2. FIND THE OCCUPIED BAND BEFORE LOOKING FOR ARTEFACTS.
   A 20 MHz frame captured in an 80 MHz window fills about a quarter of the
   slots, and sits ~40 dB above the empty remainder -- a factor of 13 in
   amplitude. Any magnitude test referenced to the median of ALL slots
   therefore flags the SIGNAL as the outlier, because most slots are noise.
   Measured example: slots 132-188 carried 60-64 dB while slots 3-69 sat at
   19 dB; a median-referenced test discarded all 57 real subcarriers.
   slot_classes() finds the band first, then judges artefacts against the
   band's own median.

3. ARTEFACT SLOTS NEED TWO TESTS, NOT ONE.
   The firmware writes several slots that are not subcarriers. Some hold values
   two orders above the real ones (-32640 + 10277j against a few hundred);
   others hold small constants (111 - 80j) that sit BELOW the real subcarriers.
   A power-floor test catches only the second kind, a magnitude test only the
   first, and a variance test misses the loud slot that varies between frames
   (-32640 / -8960 / -9728 as observed). All three kinds are excluded here by
   combining a magnitude test with a zero-variance test.

   This also means the publisher's own `trim` cannot be trusted: on the same
   data it returned four different subcarrier bands across four runs, once
   keeping only the two artefact slots and discarding every real subcarrier.
   Record with `trim: false` and clean here.

4. THE TWO FRAME POPULATIONS ARE NOT INTERCHANGEABLE.
   Block Acks arrive at ~116 Hz but are short, low-rate, single-stream frames.
   Data frames arrive at ~8.5 Hz and carry the modulation whose throughput you
   are also measuring. They are reported separately throughout; averaging them
   together mixes two different channel estimates.

5. THE HEAT MAP PLOTS AMPLITUDE, NOT A PER-FRAME RESIDUAL.
   The convention across CSI datasets is time on X, subcarrier index on Y,
   |H| as colour on a sequential map. What makes it readable is the
   horizontal banding: a given subcarrier stays strong or weak while nothing
   moves, and that persistent frequency-selective structure IS the channel.
   Something moving shows as a vertical disturbance cutting across it.

   Normalising each frame by its own median (an obvious way to remove AGC)
   destroys exactly that banding, because every column gets recentred
   independently, and the picture collapses to residual noise. AGC is instead
   left alone in the figure and handled in the metrics, which are ratios
   across subcarriers and immune to it. `--norm zscore` applies the
   whole-matrix standardisation used by CSI-Bench, which rescales without
   flattening the structure.

WHAT CANNOT BE MEASURED
-----------------------
Phase is not usable as recorded: every frame carries an unknown carrier and
sampling frequency offset plus a random packet-detection delay. Nothing here
uses it. Absolute amplitude is not meaningful either (AGC), so amplitudes are
in dB relative to each frame's own median.

At 20 MHz the delay profile's tap spacing is 50 ns, and indoor delay spreads
are tens of nanoseconds -- one tap. The delay-spread column is computed but
flagged as not quotable at that bandwidth.

Usage
-----
    python3 csi_analysis.py --bag ./session_bag --topic /mobile1/csi
    python3 csi_analysis.py --live --topic /mobile1/csi --n 2000
    python3 csi_analysis.py --bag ./bag --frames data       # data only

Outputs (--out, default ./csi_out)
    csi_frames.csv     per frame: amplitude, selectivity, K, delay spread
    csi_summary.md     tables and the caveats that apply to this capture
    fig_csi.png        amplitude heat map, slot occupancy, metrics over time
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

BA_SEQ = 0xFFFF
# frame_control values that are definitely NOT data, whatever seq says.
CONTROL_FC = {0x94: "Block Ack", 0x80: "Beacon", 0x84: "control",
              0x54: "control", 0x24: "control", 0xC4: "CTS", 0xD4: "ACK",
              0xB4: "RTS", 0x40: "Probe Req", 0x50: "Probe Resp"}
BAD = "#e34948"
GRID = "#d8d5d0"
TEXT = "#2b2926"
AMP_DIVERGING = ["#104281", "#2a78d6", "#9ec5f4", "#f0efec",
                 "#f5a173", "#e34948", "#8f1f1e"]


# ============================================================ analysis core
def slot_classes(amp: np.ndarray, band_floor_db: float = 20.0,
                 gap: int = 12, loud_factor: float = 8.0,
                 var_eps: float = 1e-6, null_floor_db: float = 15.0):
    """Split slots into (usable, artefact, null, out-of-band).

    ORDER MATTERS. A narrow transmission inside a wide capture window puts
    the real subcarriers ~40 dB above the empty part of the window -- a
    factor of ~13 in AMPLITUDE. A magnitude test referenced to the median of
    all slots therefore flags the SIGNAL as the artefact, because most slots
    are noise floor and the signal is the outlier. Observed on a real
    capture: 57 real subcarriers in a 256-slot window were all discarded.

    So: find the occupied band first, then judge slots INSIDE it.

      1. frozen   -- exactly constant across frames. Magnitude-free.
      2. band     -- on a median-smoothed profile, slots within
                     `band_floor_db` of the 90th percentile; short gaps
                     bridged; the run with the most live slots wins.
      3. loud     -- inside the band, > `loud_factor` x the band median.
      4. null     -- inside the band, > `null_floor_db` BELOW the band
                     median. The DC null and pilot positions: they vary
                     and are not loud, so nothing above catches them, yet
                     they carry no channel and would sit as a permanent
                     dark line through every figure and every metric.

    Returns (usable, artefact, null, out_of_band) boolean masks.
    """
    n = amp.shape[1]
    med = np.median(amp, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel_std = np.where(med > 0, amp.std(axis=0) / med, 0.0)
    frozen = (rel_std < var_eps) | (med <= 0)

    p = 10.0 * np.log10(np.maximum((amp ** 2).mean(axis=0), 1e-30))
    p = np.where(frozen, -np.inf, p)
    if n >= 9:
        k = 9
        fill = np.nanmin(p[np.isfinite(p)]) if np.isfinite(p).any() else 0.0
        pad = np.pad(np.where(np.isfinite(p), p, fill), k // 2, mode="edge")
        ps = np.median(np.lib.stride_tricks.sliding_window_view(pad, k), axis=1)
    else:
        ps = p
    finite = ps[np.isfinite(ps)]
    empty = np.zeros(n, bool)
    if finite.size < 4:
        return ~frozen, frozen, empty, empty
    live = ps > (np.percentile(finite, 90) - band_floor_db)

    idx = np.flatnonzero(live)
    if idx.size == 0:
        return ~frozen, frozen, empty, empty
    filled = live.copy()
    for a, b in zip(idx[:-1], idx[1:]):
        if 1 < b - a <= gap + 1:
            filled[a:b] = True
    best = best_lo = best_hi = None
    lo = None
    for i, v in enumerate(np.r_[filled, False]):
        if v and lo is None:
            lo = i
        elif not v and lo is not None:
            score = int(live[lo:i].sum())
            if best is None or score > best:
                best, best_lo, best_hi = score, lo, i
            lo = None
    band = np.zeros(n, bool)
    band[best_lo:best_hi] = True

    in_med_vals = med[band & ~frozen]
    ref = np.median(in_med_vals) if in_med_vals.size else 1.0
    loud = band & (med > loud_factor * ref)
    with np.errstate(divide="ignore"):
        below_db = 20.0 * np.log10(np.maximum(med, 1e-30) / max(ref, 1e-30))
    null = band & ~frozen & ~loud & (below_db < -null_floor_db)

    artefact = (frozen | loud) & band
    usable = band & ~artefact & ~null
    return usable, artefact, null, ~band


def artefact_mask(amp: np.ndarray, factor: float = 8.0,
                  var_eps: float = 1e-6) -> np.ndarray:
    """Back-compat wrapper: everything not usable. See slot_classes."""
    usable, _, _, _ = slot_classes(amp, loud_factor=factor, var_eps=var_eps)
    return ~usable


def _otsu(v: np.ndarray, bins: int = 128) -> float:
    """Otsu threshold: the split that best separates two populations.

    Used instead of a hand-picked floor because the gap between occupied and
    empty slots varies with the capture. A fixed "15 dB below the 90th
    percentile" passes part of a band that happens to sit 15-20 dB down,
    which is exactly the case that leaves the distribution bimodal and the
    Rician estimator returning NaN.
    """
    v = v[np.isfinite(v)]
    if v.size < 8:
        return -np.inf
    hist, edges = np.histogram(v, bins=bins)
    centres = 0.5 * (edges[:-1] + edges[1:])
    total = max(hist.sum(), 1)
    w0 = np.cumsum(hist) / total                 # weight of the low class
    mu0 = np.cumsum(hist * centres) / total      # running first moment
    mu_t = mu0[-1]                               # the TOTAL MEAN, not the sum
    with np.errstate(divide="ignore", invalid="ignore"):
        between = (mu_t * w0 - mu0) ** 2 / (w0 * (1.0 - w0))
    between[~np.isfinite(between)] = -np.inf
    return float(centres[int(np.argmax(between))])


def occupied_band(amp: np.ndarray, floor_db: float = 15.0,
                  gap: int = 8, ignore: np.ndarray = None) -> np.ndarray:
    """True for slots inside the band the frames actually occupy.

    A narrower transmission captured in a wider window leaves most of the
    window empty: a 20 MHz frame fills about 64 of 256 slots and the rest is
    noise floor. Those empty slots are not loud and not frozen, so the
    artefact tests miss them entirely -- they just sit 15-20 dB down with
    noise on top.

    Leaving them in is not cosmetic. It makes the per-frame amplitude
    distribution bimodal, which inflates the 4th moment, and the Rician
    moment estimator then returns NaN or 0 for every frame however clean the
    channel is. Frequency selectivity is inflated by tens of dB for the same
    reason, since it is just the spread of |H| across slots.

    Slots within `floor_db` of the 90th-percentile slot are kept; gaps up to
    `gap` slots are bridged (a DC null sits inside the band and must not
    split it); the longest contiguous run wins.

    `ignore` MUST carry the artefact mask. The artefact slots are ~60 dB
    above every real subcarrier, so with 65 of 256 slots being artefacts the
    90th percentile lands inside the artefact population, the threshold ends
    up far above any real signal, and the only band found is the artefacts.
    Excluding them from the percentile fixes that; they are then bridged
    over as gaps, since they sit inside the band rather than beside it.
    """
    p = 10.0 * np.log10(np.maximum((amp ** 2).mean(axis=0), 1e-30))
    # Median-smooth along frequency before thresholding. A deep fade inside
    # the band can drop a single slot below the threshold, and an unsmoothed
    # profile then splits the band there, truncating it to whichever
    # fragment happens to be longest.
    if p.size >= 7:
        k = 5
        pad = np.pad(p, k // 2, mode="edge")
        p = np.median(np.lib.stride_tricks.sliding_window_view(pad, k), axis=1)
    if ignore is not None and ignore.any() and (~ignore).sum() > 4:
        clean = p[~ignore]
        p = np.where(ignore, -np.inf, p)     # never count as live themselves
    else:
        clean = p

    # Otsu first: if the slots really do form two populations it finds the
    # split without being told where it is. Fall back to the fixed floor when
    # the distribution is unimodal, i.e. the whole window is occupied and
    # Otsu would cut an arbitrary line through one population.
    thr = _otsu(clean)
    lo_pop, hi_pop = clean[clean <= thr], clean[clean > thr]
    bimodal = (lo_pop.size >= 4 and hi_pop.size >= 4
               and (hi_pop.mean() - lo_pop.mean()) >= 6.0)
    ref = np.percentile(clean, 90)
    live = p > (thr if bimodal else ref - floor_db)
    if not live.any():
        return np.ones(amp.shape[1], bool)

    # bridge short gaps so a null inside the band does not split it
    idx = np.flatnonzero(live)
    filled = live.copy()
    for a, b in zip(idx[:-1], idx[1:]):
        if 1 < b - a <= gap + 1:
            filled[a:b] = True
    # Artefact slots are transparent to the run search: they sit INSIDE the
    # band, so a contiguous block of them would otherwise split it and the
    # longest surviving fragment would win, silently discarding the rest of
    # the real band. They are excluded from the result by the artefact mask
    # anyway, so marking them here costs nothing.
    if ignore is not None:
        filled = filled | ignore

    # Best contiguous run, scored by how many REAL live slots it contains --
    # not by raw length. Scoring by length lets a solid block of artefacts
    # (transparent, so they extend any run) win outright when little else is
    # live, and the band then comes back as the artefact block itself.
    real_live = live & ~(ignore if ignore is not None else False)
    best_lo = best_hi = lo = None
    best = 0
    for i, v in enumerate(np.r_[filled, False]):
        if v and lo is None:
            lo = i
        elif not v and lo is not None:
            score = int(real_live[lo:i].sum())
            if score > best:
                best, best_lo, best_hi = score, lo, i
            lo = None
    out = np.zeros(amp.shape[1], bool)
    if best_lo is not None and best >= 4:
        out[best_lo:best_hi] = True
    else:
        out[:] = True          # no credible band: keep everything, say so
    return out


def amplitude_db(H: np.ndarray) -> np.ndarray:
    """|H| in dB relative to each frame's own median.

    For METRICS only, never for the heat map. Removing the per-frame median
    takes AGC out of the selectivity figure, which is what selectivity should
    measure -- but doing the same to the image destroys the persistent
    frequency structure the reader is looking for (see plot_amplitude).
    """
    a = np.abs(H)
    med = np.median(a, axis=1, keepdims=True)
    return 20.0 * np.log10(np.maximum(a, 1e-12) / np.maximum(med, 1e-12))


def plot_amplitude(H: np.ndarray, how: str = "db") -> tuple:
    """(image, colourbar label) for the heat map, in the conventional form.

    how="db"      20*log10|H|. The default and the convention.
    how="zscore"  whole-matrix standardisation, CSI-Bench style: one mean and
                  one std for the entire capture. Comparable across captures.
                  NOT per frame -- that is the mistake this function exists to
                  avoid.
    """
    a = np.abs(H)
    if how == "zscore":
        return (a - a.mean()) / max(a.std(), 1e-12), "|H| (z-score)"
    return 20.0 * np.log10(np.maximum(a, 1e-12)), "|H| [dB]"


def temporal_coherence(absH: np.ndarray) -> float:
    """Median correlation of |H| between consecutive frames.

    The first question to ask of any CSI stream. A physical channel changes
    slowly, so consecutive frames see almost the same multipath and correlate
    high. Near zero means each frame is an independent draw -- receiver noise,
    or an extractor emitting nothing usable for this frame format -- and no
    metric computed downstream means anything.
    """
    if absH.shape[0] < 2:
        return float("nan")
    a, b = absH[:-1], absH[1:]
    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean(axis=1, keepdims=True)
    num = (a * b).sum(axis=1)
    den = np.sqrt((a * a).sum(axis=1) * (b * b).sum(axis=1))
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(den > 0, num / den, np.nan)
    return float(np.nanmedian(r))


def rician_k(H: np.ndarray) -> np.ndarray:
    """Rician K per frame from the 2nd and 4th moments of |H| across slots.

    Moment method: with mu2 = E[|H|^2] and mu4 = E[|H|^4],
        K = sqrt(2*mu2^2 - mu4) / (mu2 - sqrt(2*mu2^2 - mu4)).
    High K means one dominant path (unobstructed); K near 0 is Rayleigh,
    diffuse multipath. Undefined where the radicand goes negative, which
    happens on noise -- returned as NaN rather than clipped, so it cannot be
    silently averaged in.
    """
    a2 = np.abs(H) ** 2
    mu2 = a2.mean(axis=1)
    mu4 = (a2 ** 2).mean(axis=1)
    rad = 2.0 * mu2 ** 2 - mu4
    with np.errstate(invalid="ignore"):
        s = np.sqrt(np.where(rad > 0, rad, np.nan))
        return s / np.maximum(mu2 - s, 1e-12)


def delay_metrics(H: np.ndarray, bw_mhz: float, min_struct_db: float = 6.0):
    """RMS delay spread per frame, and the profile's peak-to-median in dB.

    The power delay profile is |IFFT(H)|^2 across slots. A profile with no
    structure is noise, and its "delay spread" is just the window width over
    sqrt(12) -- a number about the measurement, not the room -- so frames
    below `min_struct_db` return NaN.
    """
    P = np.abs(np.fft.ifft(H, axis=1)) ** 2
    n = P.shape[1]
    tap_s = 1.0 / (bw_mhz * 1e6)
    tau = np.arange(n) * tap_s

    peak = P.max(axis=1)
    med = np.median(P, axis=1)
    struct_db = 10.0 * np.log10(np.maximum(peak, 1e-30) /
                                np.maximum(med, 1e-30))

    # Only the first half of the profile: the rest is the negative-delay image.
    half = max(n // 2, 2)
    Ph, th = P[:, :half], tau[:half]
    tot = Ph.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        mean_t = (Ph * th).sum(axis=1) / np.maximum(tot, 1e-30)
        var_t = (Ph * (th - mean_t[:, None]) ** 2).sum(axis=1) / np.maximum(tot, 1e-30)
    rms = np.sqrt(np.maximum(var_t, 0.0))
    return np.where(struct_db >= min_struct_db, rms, np.nan), struct_db


# ============================================================ input
def from_bag(bag: Path, topic: str):
    """Frames from a rosbag2 bag. Needs `pip install rosbags`."""
    try:
        from rosbags.highlevel import AnyReader
    except ImportError:
        raise SystemExit("reading a bag needs: pip install rosbags")
    rows = []
    with AnyReader([bag]) as reader:
        conns = [c for c in reader.connections if c.topic == topic]
        if not conns:
            have = sorted({c.topic for c in reader.connections})
            raise SystemExit(f"topic {topic} not in bag. Present: {have}")
        for conn, ts, raw in reader.messages(connections=conns):
            m = reader.deserialize(raw, conn.msgtype)
            rows.append({
                "t_ns": ts, "seq": int(m.seq), "rssi": int(m.rssi),
                "frame_control": int(m.frame_control),
                "src_mac": str(m.src_mac),
                "bw": int(m.bandwidth_mhz),
                "idx": np.asarray(m.subcarrier_index, dtype=np.int32),
                "re": np.asarray(m.csi_real, dtype=np.float64),
                "im": np.asarray(m.csi_imag, dtype=np.float64),
            })
    return rows


def from_live(topic: str, want: int, timeout: float):
    """Frames from the live topic."""
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import (QoSHistoryPolicy, QoSProfile,
                           QoSReliabilityPolicy)
    from comms_msgs.msg import CsiFrame

    rows = []

    class C(Node):
        def __init__(self):
            super().__init__("csi_analysis")
            qos = QoSProfile(depth=200, history=QoSHistoryPolicy.KEEP_LAST,
                             reliability=QoSReliabilityPolicy.RELIABLE)
            self.create_subscription(CsiFrame, topic, self.cb, qos)

        def cb(self, m):
            if len(rows) >= want:
                return
            rows.append({
                "t_ns": (m.header.stamp.sec * 10**9 + m.header.stamp.nanosec),
                "seq": int(m.seq), "rssi": int(m.rssi),
                "frame_control": int(m.frame_control),
                "src_mac": str(m.src_mac), "bw": int(m.bandwidth_mhz),
                "idx": np.asarray(m.subcarrier_index, dtype=np.int32),
                "re": np.asarray(m.csi_real, dtype=np.float64),
                "im": np.asarray(m.csi_imag, dtype=np.float64),
            })
            if len(rows) % 200 == 0:
                self.get_logger().info(f"  {len(rows)}/{want}")

    rclpy.init()
    node = C()
    t0 = node.get_clock().now()
    try:
        while rclpy.ok() and len(rows) < want:
            rclpy.spin_once(node, timeout_sec=0.2)
            if (node.get_clock().now() - t0).nanoseconds / 1e9 > timeout:
                node.get_logger().warn("timed out")
                break
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()
    return rows


def stack(rows: list):
    """Rows -> (H, idx, meta). Locks to the dominant slot count.

    Frame widths can change mid-capture -- a 20 MHz frame inside an 80 MHz
    window fills fewer slots. Mixing them in one array would silently change
    the delay-profile scale, so minority layouts are dropped, not padded.
    """
    if not rows:
        return None, None, None
    sizes = pd.Series([r["re"].size for r in rows])
    n = int(sizes.mode().iloc[0])
    keep = [r for r in rows if r["re"].size == n and r["im"].size == n]
    H = np.stack([r["re"] + 1j * r["im"] for r in keep])
    idx = keep[0]["idx"]
    meta = pd.DataFrame([{k: r[k] for k in
                          ("t_ns", "seq", "rssi", "frame_control", "src_mac", "bw")}
                         for r in keep])
    meta["t_s"] = (meta["t_ns"] - meta["t_ns"].min()) / 1e9
    # Data = a frame_control we do not recognise as control/management.
    # seq is a weak secondary signal only: Block Acks were observed carrying
    # real sequence numbers, so seq alone misclassifies them as data.
    meta["is_ba"] = meta["frame_control"].isin(CONTROL_FC).to_numpy()
    return H, idx, meta


# ============================================================ per-population
def analyse(H, idx, meta, label, args):
    """All metrics for one frame population. Returns (frame df, dict of state)."""
    if H.shape[0] < 5:
        return None, None
    amp = np.abs(H)

    # Zero frames FIRST, before any slot classification. The firmware emits
    # empty records at start-up; one of them puts a 40 dB spike into
    # selectivity and a -240 dB pixel into the heat map. Worse, they break
    # the frozen-slot test -- a slot that is constant for 3000 frames and
    # zero for 6 no longer has zero variance -- so the artefacts leak into
    # the band. Whole-frame zero fraction needs no knowledge of the band.
    good = (amp <= 0).mean(axis=1) < 0.5
    n_zero = int((~good).sum())
    if n_zero and good.sum() >= 5:
        H, amp, meta = H[good], amp[good], meta[good].reset_index(drop=True)

    usable, artefact, null, outband = slot_classes(
        amp, args.band_floor_db, args.band_gap,
        args.artefact_factor, args.var_eps, args.null_floor_db)
    bad = ~usable
    n_art, n_band = int(artefact.sum()), int((~outband).sum())
    n_null = int(null.sum())
    if (~bad).sum() < 4:
        print(f"  {label}: only {(~bad).sum()} slots survived "
              f"({H.shape[1]} total, {n_band} in band, {n_art} artefact). "
              "Try --band-floor-db 25 or --artefact-factor 50 and check "
              "panel (c) of the figure.", file=sys.stderr)
        return None, None

    Hg, idxg = H[:, ~bad], idx[~bad]
    bw = float(meta["bw"].mode().iloc[0])
    amp_db = amplitude_db(Hg)
    occ_bw = bw * (idxg[-1] - idxg[0] + 1) / H.shape[1]
    tau, struct = delay_metrics(Hg, occ_bw, args.min_profile_db)
    K = rician_k(Hg)

    df = pd.DataFrame({
        "population": label,
        "t_s": meta["t_s"].to_numpy(),
        "seq": meta["seq"].to_numpy(),
        "rssi_dbm": meta["rssi"].to_numpy(),
        "selectivity_db": amp_db.std(axis=1),
        "k_factor": K,
        "k_factor_db": 10 * np.log10(np.maximum(K, 1e-3)),
        "delay_spread_ns": tau * 1e9,
        "profile_structure_db": struct,
    })
    dur = float(meta["t_s"].max() - meta["t_s"].min())
    state = {
        "label": label, "amp_db": amp_db, "H": Hg, "idx": idxg,
        "bad_idx": idx[bad],
        "t": meta["t_s"].to_numpy(), "bw": bw,
        "n_frames": H.shape[0], "n_slots": H.shape[1],
        "n_usable": int((~bad).sum()),
        "n_artefact": n_art, "n_outband": int(outband.sum()),
        "n_null": n_null, "n_zero_frames": n_zero,
        "null_mask": null, "art_mask": artefact, "out_mask": outband,
        "idx_lo": int(idxg[0]), "idx_hi": int(idxg[-1]),
        # the bandwidth the frames OCCUPY, not the one the capture was set to
        "occ_mhz": round(bw * (idxg[-1] - idxg[0] + 1) / H.shape[1], 1),
        "rate_hz": (H.shape[0] - 1) / max(dur, 1e-9),
        "coherence": temporal_coherence(np.abs(Hg)),
        "tap_ns": 1e9 / (occ_bw * 1e6),
        "slot_power_db": 10 * np.log10(np.maximum((amp ** 2).mean(axis=0), 1e-30)),
        "all_idx": idx, "bad_mask": bad,
        "flat_pct": float(100 * np.mean(struct < args.min_profile_db)),
    }
    return df, state


# ============================================================ figure
def figure(states, fr, out, norm="db", window_s=None):
    """Conventional layout: time x subcarrier, |H| as colour, sequential map.

    A window is plotted rather than the whole run, because these images are
    read qualitatively -- you are meant to see individual events, and a
    ten-minute capture squeezed onto one axis shows nothing. The rest of the
    run is quantified in the tables.
    """
    plt.rcParams.update({"font.size": 8, "axes.edgecolor": GRID,
                         "axes.labelcolor": TEXT, "text.color": TEXT})
    n = len(states)
    fig = plt.figure(figsize=(6.4 * n, 9.0))
    gs = fig.add_gridspec(4, n, height_ratios=[1.3, 0.6, 1.0, 1.0],
                          hspace=0.55, wspace=0.24)

    for i, s in enumerate(states):
        ax = fig.add_subplot(gs[0, i])
        t = s["t"]
        # centre the window on the most eventful stretch: the largest change
        # in selectivity, which is where a reader would want to look
        sel = fr[fr["population"] == s["label"]]["selectivity_db"].to_numpy()
        if window_s and t[-1] - t[0] > window_s and len(sel) > 20:
            k = int(np.nanargmax(pd.Series(sel).rolling(
                max(len(sel) // 20, 3), center=True).std().to_numpy()))
            lo, hi = t[k] - window_s / 2, t[k] + window_s / 2
            m = (t >= lo) & (t <= hi)
        else:
            m = np.ones(len(t), bool)

        img, lab = plot_amplitude(s["H"][m], norm)
        step = max(img.shape[0] // 1400, 1)
        # Colour limits from the data's own 1st-99th percentile of FINITE
        # values. A single zero-valued slot is -240 dB and would otherwise
        # set vmin, leaving every real pixel the same shade.
        fin = img[np.isfinite(img)]
        vlo, vhi = (np.percentile(fin, [1, 99]) if fin.size else (None, None))
        im = ax.imshow(img[::step].T, aspect="auto", origin="lower",
                       cmap="viridis", vmin=vlo, vmax=vhi,
                       extent=[t[m][0], t[m][-1],
                               s["idx"][0] - 0.5, s["idx"][-1] + 0.5])
        ax.set_xlabel("time [s]")
        ax.set_ylabel("subcarrier index" if i == 0 else "")
        ax.set_title(f"({chr(97+i)}) {s['label']}: channel amplitude\n"
                     f"{s['n_frames']} frames at {s['rate_hz']:.0f} Hz, "
                     f"{s['n_usable']}/{s['n_slots']} slots, "
                     f"coherence {s['coherence']:.2f}"
                     + (f", {m.sum()} shown" if m.sum() < len(t) else ""),
                     loc="left", fontsize=8)
        fig.colorbar(im, ax=ax, label=lab)

        ax = fig.add_subplot(gs[1, i])
        ai, pw = s["all_idx"], s["slot_power_db"]
        ax.plot(ai, pw, lw=1.0, color="#2a78d6", zorder=1)
        for mask, mk, col, lab in ((s["out_mask"], ".", "0.6", "out of band"),
                                   (s["null_mask"], "v", "#e08214", "null"),
                                   (s["art_mask"], "x", BAD, "artefact")):
            if mask.any():
                ax.plot(ai[mask], pw[mask], mk, color=col, ms=5 if mk == "." else 7,
                        ls="none", label=f"{lab} ({int(mask.sum())})", zorder=2)
        ax.legend(fontsize=7, frameon=False, ncol=3)
        ax.set_xlabel("subcarrier index")
        ax.set_ylabel("mean power [dB]" if i == 0 else "")
        ax.set_title(f"({chr(97+n+i)}) slot occupancy: {s['n_usable']} usable "
                     f"of {s['n_slots']} -- band {s['idx_lo']}-{s['idx_hi']}, "
                     f"{s['occ_mhz']:.0f} MHz occupied",
                     loc="left", fontsize=8)
        ax.grid(True, color=GRID, lw=0.5)

    ax = fig.add_subplot(gs[2, :])
    for s in states:
        g = fr[fr["population"] == s["label"]]
        ax.plot(g["t_s"], g["k_factor_db"], lw=0.6, alpha=0.3)
        ax.plot(g["t_s"], g["k_factor_db"].rolling(25, center=True,
                min_periods=5).median(), lw=1.6, label=s["label"])
    ax.axhline(0, color=GRID, lw=0.8)
    ax.set_ylabel("Rician K [dB]")
    ax.set_title("dominant-path strength (high = one strong path, "
                 "low = diffuse)", loc="left", fontsize=8)
    ax.legend(frameon=False, fontsize=7)
    ax.grid(True, color=GRID, lw=0.5)

    ax = fig.add_subplot(gs[3, :])
    for s in states:
        g = fr[fr["population"] == s["label"]]
        ax.plot(g["t_s"], g["selectivity_db"], lw=0.6, alpha=0.3)
        ax.plot(g["t_s"], g["selectivity_db"].rolling(25, center=True,
                min_periods=5).median(), lw=1.6, label=s["label"])
    ax.set_xlabel("time [s]")
    ax.set_ylabel("frequency selectivity [dB]")
    ax.set_title("spread of |H| across subcarriers, AGC removed -- the most "
                 "robust of the three metrics", loc="left", fontsize=8)
    ax.legend(frameon=False, fontsize=7)
    ax.grid(True, color=GRID, lw=0.5)

    fig.savefig(out / "fig_csi.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


# ============================================================ main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bag", type=Path, default=None)
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--topic", default="/mobile1/csi")
    ap.add_argument("--n", type=int, default=2000, help="live: frames to take")
    ap.add_argument("--timeout", type=float, default=180.0)
    ap.add_argument("--out", type=Path, default=Path("csi_out"))
    ap.add_argument("--frames", choices=["both", "ba", "data"], default="both")
    ap.add_argument("--artefact-factor", type=float, default=10.0)
    ap.add_argument("--var-eps", type=float, default=1e-6)
    ap.add_argument("--min-profile-db", type=float, default=6.0)
    ap.add_argument("--band-floor-db", type=float, default=15.0,
                    help="slots this far below the 90th-percentile slot are "
                         "outside the occupied band")
    ap.add_argument("--null-floor-db", type=float, default=15.0,
                    help="in-band slots this far below the band median are "
                         "nulls (DC, pilots) and excluded")
    ap.add_argument("--band-gap", type=int, default=12,
                    help="gap in slots bridged when finding the band")
    ap.add_argument("--min-coherence", type=float, default=0.5)
    ap.add_argument("--norm", choices=["db", "zscore"], default="db",
                    help="heat map scale. db = 20*log10|H| (the convention); "
                         "zscore = whole-matrix standardisation, CSI-Bench "
                         "style, comparable across captures")
    ap.add_argument("--window-s", type=float, default=20.0,
                    help="seconds of the run to show in the heat map, "
                         "centred on the most eventful stretch. 0 = all")
    args = ap.parse_args(argv)

    if args.bag is None and not args.live:
        raise SystemExit("pass --bag PATH or --live")
    rows = from_bag(args.bag, args.topic) if args.bag else \
        from_live(args.topic, args.n, args.timeout)
    H, idx, meta = stack(rows)
    if H is None:
        raise SystemExit("no frames")
    args.out.mkdir(parents=True, exist_ok=True)

    # Populations split on seq, NOT on frame_control -- see the module docstring.
    groups = []
    fc_mix = meta.groupby("frame_control").size().sort_values(ascending=False)
    print("frame_control mix: " + ", ".join(
        f"0x{fc:02x} {CONTROL_FC.get(fc, 'DATA')} x{n}"
        for fc, n in fc_mix.items()))
    if args.frames in ("both", "data"):
        m = ~meta["is_ba"].to_numpy()
        if m.sum() >= 5:
            groups.append(("data frames", H[m], meta[m].reset_index(drop=True)))
        else:
            print("no data frames in this capture -- every frame is a known "
                  "control/management type. If the AP is on 802.11ax, the "
                  "BCM43455 cannot demodulate HE and will only ever see "
                  "control frames: set rtw_he_enable=0 on the transmitters.",
                  file=sys.stderr)
    if args.frames in ("both", "ba"):
        m = meta["is_ba"].to_numpy()
        if m.sum() >= 5:
            groups.append(("control frames", H[m], meta[m].reset_index(drop=True)))
    if not groups:
        raise SystemExit("no population had enough frames")

    dfs, states = [], []
    for label, Hg, mg in groups:
        df, st = analyse(Hg, idx, mg, label, args)
        if df is not None:
            dfs.append(df)
            states.append(st)
    if not states:
        raise SystemExit(
            "nothing analysable: every slot was excluded as artefact or "
            "out-of-band. See the per-population line above; --band-floor-db "
            "and --artefact-factor are the two knobs.")

    fr = pd.concat(dfs, ignore_index=True)
    fr.to_csv(args.out / "csi_frames.csv", index=False)

    inv = pd.DataFrame([{
        "population": s["label"], "frames": s["n_frames"],
        "rate_hz": round(s["rate_hz"], 1), "bandwidth_mhz": s["bw"],
        "slots": s["n_slots"], "usable": s["n_usable"],
        "band": f"{s['idx_lo']}-{s['idx_hi']}",
        "occupied_mhz": s["occ_mhz"],
        "artefact": s["n_artefact"], "null": s["n_null"],
        "out_of_band": s["n_outband"], "zero_frames": s["n_zero_frames"],
        "tap_ns": round(s["tap_ns"], 1),
        "temporal_coherence": round(s["coherence"], 3),
        "flat_profile_pct": round(s["flat_pct"], 1),
    } for s in states])

    stats = fr.groupby("population").agg(
        k_median_db=("k_factor_db", "median"),
        k_p05_db=("k_factor_db", lambda x: x.quantile(0.05)),
        selectivity_median_db=("selectivity_db", "median"),
        delay_spread_median_ns=("delay_spread_ns", "median"),
        rssi_median_dbm=("rssi_dbm", "median"),
    ).round(2).reset_index()

    figure(states, fr, args.out, args.norm,
           args.window_s if args.window_s > 0 else None)

    # ---- summary -----------------------------------------------------------
    macs = meta.groupby("src_mac").size().sort_values(ascending=False)
    fc = meta.groupby("frame_control").size().sort_values(ascending=False)
    md = [f"# CSI analysis — `{args.topic}`", "",
          "Populations are split on `seq` (0xFFFF = Block Ack) and **not** on "
          "`frame_control`, which holds the first byte of the PPDU: for an "
          "aggregated A-MPDU that is an MPDU delimiter, so data frames never "
          "appear as 0x88.", "",
          "## Capture", "", inv.to_markdown(index=False), "",
          "## Channel metrics", "", stats.to_markdown(index=False), "",
          "## Transmitters", "",
          macs.to_frame("frames").to_markdown(), "",
          "## First PPDU byte seen (`frame_control`)", "",
          fc.to_frame("frames").to_markdown(), ""]

    incoh = inv[inv["temporal_coherence"] < args.min_coherence]
    if len(incoh):
        md += ["> **This does not behave like a channel.** " +
               ", ".join(f"`{r.population}` coherence {r.temporal_coherence:.2f}"
                         for r in incoh.itertuples()) +
               f", against {args.min_coherence:.2f}. Consecutive frames should "
               "see nearly the same multipath and correlate above 0.8. Near "
               "zero means each frame is an independent draw, and every metric "
               "above is computed from that input.", ""]
    narrow = inv[inv["occupied_mhz"] < 0.8 * inv["bandwidth_mhz"]]
    if len(narrow):
        md += ["> **The frames occupy less than the capture window.** " +
               ", ".join(f"`{r.population}` slots {r.band}, "
                         f"{r.occupied_mhz:.0f} of {r.bandwidth_mhz:.0f} MHz"
                         for r in narrow.itertuples()) +
               ". The rest of the window is noise floor and has been excluded. "
               "Leaving it in makes the amplitude distribution bimodal, which "
               "inflates the 4th moment: the Rician estimator then returns NaN "
               "or 0 for every frame however clean the channel, and frequency "
               "selectivity is inflated by tens of dB. Delay-profile tap "
               "spacing follows the OCCUPIED bandwidth, not the configured "
               "one.", ""]
    coarse = inv[inv["occupied_mhz"] <= 25]
    if len(coarse):
        md += [f"> **Delay spread not quotable** at "
               f"{coarse['bandwidth_mhz'].max():.0f} MHz: tap spacing is "
               f"{coarse['tap_ns'].max():.0f} ns and indoor delay spreads are "
               "of that order, so the profile is one tap wide. K-factor and "
               "selectivity are unaffected. 80 MHz would give 12.5 ns.", ""]
    if len(states) == 2:
        md += ["> **The two populations are not interchangeable.** Block Acks "
               "are short, low-rate, single-stream control frames; data frames "
               "carry the modulation whose throughput is measured alongside. "
               "Use data frames for anything paired with throughput, and Block "
               "Acks for channel dynamics between them.", ""]
    (args.out / "csi_summary.md").write_text("\n".join(md))

    print(inv.to_string(index=False))
    print()
    print(stats.to_string(index=False))
    print(f"\nwrote {args.out}/csi_frames.csv, csi_summary.md, fig_csi.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
