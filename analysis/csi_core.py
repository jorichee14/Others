#!/usr/bin/env python3
"""CSI maths, numpy only.

Kept free of pandas and matplotlib so the live ROS 2 node can import it on a
robot without pulling in the analysis stack. `csi_analysis.py` uses the same
functions, so the offline numbers and the live display cannot drift apart.

What survives the recording, and why (see csi_analysis.py for the long version):
the phase of each frame carries an unknown constant offset and an unknown linear
ramp, so only |H| and the SHAPE of the delay profile are usable. Every quantity
here is a ratio, so the receiver's automatic gain control cancels out.
"""
from __future__ import annotations

import numpy as np


def usable_subcarriers(H: np.ndarray, floor_db: float = 20.0) -> np.ndarray:
    """Boolean mask of the columns that carry a real subcarrier.

    An 802.11 OFDM symbol does not use every FFT slot: the band edges are guard
    bands and the centre is a DC null, and at 80 MHz that is roughly 20 of 256
    slots carrying no signal. Whether the publisher removed them is recorded in
    `trimmed`, but the flag cannot be trusted over the data, so this decides
    from the mean power per slot: anything more than `floor_db` below the median
    slot is not carrying a subcarrier.

    Leaving them in is not a cosmetic problem. The amplitude distribution
    becomes bimodal -- a body of real subcarriers plus a spike near zero -- and
    that drives the 4th moment up, which makes the Rician estimator return 0
    (Rayleigh) for every frame however clean the channel is, and inflates the
    frequency selectivity by tens of dB.

    The reference is the 90th percentile of slot power, NOT the median. A 20 MHz
    frame captured in an 80 MHz window fills a quarter of the slots, so the
    median slot is itself a null and a median-referenced threshold keeps
    everything. Averaging over many frames flattens per-subcarrier fading, so
    real subcarriers sit within a few dB of that reference while nulls are tens
    of dB below it."""
    p = (np.abs(np.atleast_2d(H)) ** 2).mean(axis=0)
    pos = p[p > 0]
    if pos.size == 0:
        return np.ones(p.size, bool)
    keep = p > np.percentile(pos, 90) * 10 ** (-floor_db / 10)
    return keep if keep.sum() >= 8 else np.ones(p.size, bool)


def profile_structure_db(P: np.ndarray) -> np.ndarray:
    """Peak-to-median of each delay profile, in dB.

    A real channel puts most of its energy in a few taps, so this is large. When
    it approaches 0 dB the profile is flat -- noise, not multipath -- and any
    delay spread computed from it is just the width of the analysis window
    divided by sqrt(12), a number about the measurement rather than the room."""
    P = np.atleast_2d(P)
    med = np.median(P, axis=1)
    return 10 * np.log10(np.maximum(P.max(axis=1), 1e-30) / np.maximum(med, 1e-30))


def temporal_coherence(H: np.ndarray) -> float:
    """Median correlation of |H| across subcarriers between consecutive frames.

    This is the test for whether a CSI stream is a channel at all. A real
    channel is a physical thing that changes slowly: at ~180 frames per second
    with a robot walking pace, consecutive frames see almost the same multipath,
    so the shape of |H| across frequency repeats and the correlation sits high
    (typically > 0.8). Receiver noise, or an extractor emitting garbage for a
    frame format it does not understand, produces an independent draw every
    frame and a correlation near zero.

    Every other metric here -- K, delay spread, selectivity -- will return a
    number whether or not the input is a channel. This one says whether those
    numbers mean anything."""
    A = np.abs(np.atleast_2d(H)).astype(float)
    if A.shape[0] < 2:
        return float("nan")
    A = A - A.mean(axis=1, keepdims=True)
    num = (A[:-1] * A[1:]).sum(axis=1)
    den = np.sqrt((A[:-1] ** 2).sum(axis=1) * (A[1:] ** 2).sum(axis=1))
    r = num / np.where(den > 0, den, np.nan)
    return float(np.nanmedian(r))


def occupied_band(idx: np.ndarray, use: np.ndarray, max_gap: int = 12):
    """(first_slot, span) of the block of slots that carry signal.

    The longest RUN of usable slots, not simply the first to the last: a couple
    of noisy slots passing the threshold at opposite ends of the window would
    otherwise stretch the span across empty spectrum and overstate the occupied
    bandwidth. Gaps up to `max_gap` are bridged, since pilot and DC nulls sit
    inside a real transmission.

    The capture window and the transmitted bandwidth are not the same thing. A
    radio capturing at 80 MHz still sees a 20 MHz control frame -- an ack, say --
    filling only the primary quarter of the window, and the delay resolution that
    frame supports is set by the 20 MHz it actually occupies, not by the 80 MHz
    the capture was configured for. Taking the span from the data is the only way
    to get that right."""
    occ = np.sort(np.asarray(idx)[np.asarray(use, bool)])
    if occ.size == 0:
        return 0, int(np.asarray(idx).max()) + 1
    brk = np.flatnonzero(np.diff(occ) > max_gap + 1)
    starts = np.r_[0, brk + 1]
    ends = np.r_[brk, occ.size - 1]
    widths = occ[ends] - occ[starts] + 1
    b = int(np.argmax(widths))
    return int(occ[starts[b]]), int(widths[b])


def effective_bandwidth_mhz(span_slots: int, declared_mhz: float, raw_slots: int) -> float:
    """Bandwidth the frames actually occupy, from the span they fill."""
    if raw_slots <= 0:
        return float(declared_mhz)
    return float(span_slots) * float(declared_mhz) / float(raw_slots)


def delay_profile(H: np.ndarray, idx: np.ndarray, n_bins: int, offset: int = 0) -> np.ndarray:
    """Power delay profile per frame, each rolled so its strongest tap is first.

    H is (n_frames, n_subcarriers) complex, idx the raw firmware slot of each
    column, and the transform runs over `n_bins` slots starting at `offset` --
    pass the occupied band rather than the whole capture window, or the profile
    is interpolated across empty spectrum and its taps are finer than anything
    the frames can actually resolve.

    The firmware's slot ordering is not recorded and the two plausible
    conventions differ by a half-window circular shift of the profile; a frame's
    own symbol-timing error shifts it further. Rolling each profile onto its own
    peak removes both, and turns the delay axis into excess delay past the
    strongest path -- which is what a delay spread should be measured over."""
    H = np.atleast_2d(H)
    grid = np.zeros((H.shape[0], n_bins), dtype=complex)
    grid[:, np.clip(np.asarray(idx) - offset, 0, n_bins - 1)] = H
    P = np.abs(np.fft.ifft(grid, axis=1)) ** 2
    shift = -np.argmax(P, axis=1)
    r, c = np.ogrid[: P.shape[0], : P.shape[1]]
    return P[r, (c - shift[:, None]) % P.shape[1]]


def rms_delay_spread(P: np.ndarray, dt_s: float, window: int,
                     floor_db: float = 20.0) -> np.ndarray:
    """RMS delay spread [s] over the taps after the strongest one.

    Only the first `window` taps count (beyond that the profile wraps) and only
    those within `floor_db` of the peak, so receiver noise cannot inflate it."""
    P = np.atleast_2d(P)
    Pw = P[:, :window].copy()
    Pw[Pw < Pw.max(axis=1, keepdims=True) * 10 ** (-floor_db / 10)] = 0.0
    tau = np.arange(window) * dt_s
    tot = Pw.sum(axis=1)
    ok = tot > 0
    safe = np.where(ok, tot, 1.0)
    mean = np.where(ok, (Pw * tau).sum(axis=1) / safe, np.nan)
    m2 = np.where(ok, (Pw * tau**2).sum(axis=1) / safe, np.nan)
    return np.sqrt(np.maximum(m2 - mean**2, 0.0))


def rician_k(H: np.ndarray) -> np.ndarray:
    """Rician K per frame, from the 2nd and 4th moments of |H| across subcarriers.

    K = sqrt(2 m2^2 - m4) / (m2 - sqrt(2 m2^2 - m4)). When 2 m2^2 <= m4 the
    amplitudes are Rayleigh -- no dominant path -- and K is reported as 0. A
    ratio of moments, so the gain control cancels. It treats subcarriers as
    independent fading samples, which is why a single frame's K is noisy and
    should be smoothed over time before it is quoted."""
    A2 = np.abs(np.atleast_2d(H)) ** 2
    m2 = A2.mean(axis=1)
    m4 = (A2**2).mean(axis=1)
    root = np.sqrt(np.maximum(2 * m2**2 - m4, 0.0))
    den = m2 - root
    return np.where(den > 1e-18, root / np.where(den > 1e-18, den, 1.0), 0.0)


def amplitude_db(H: np.ndarray) -> np.ndarray:
    """|H| in dB relative to each frame's own median.

    The median, not the mean: a few deeply faded subcarriers would drag a mean
    reference down and tilt the whole frame. Referencing each frame to itself is
    what removes the automatic gain control."""
    A = 20 * np.log10(np.abs(np.atleast_2d(H)) + 1e-12)
    return A - np.median(A, axis=1, keepdims=True)


def colour_lut(hex_colors, n: int = 256) -> np.ndarray:
    """A (n,3) uint8 BGR lookup table interpolated through the given hex colours.

    BGR because that is what sensor_msgs/Image calls 'bgr8'; building the table
    once beats calling a colormap per frame."""
    rgb = np.array([[int(c[i:i + 2], 16) for i in (1, 3, 5)] for c in hex_colors], float)
    x = np.linspace(0, 1, len(rgb))
    q = np.linspace(0, 1, n)
    out = np.stack([np.interp(q, x, rgb[:, ch]) for ch in (2, 1, 0)], axis=1)
    return np.clip(out, 0, 255).astype(np.uint8)


def quantise(v: np.ndarray, lo: float, hi: float, n: int = 256) -> np.ndarray:
    """Map values onto lookup-table indices, clipping outside [lo, hi]."""
    if hi <= lo:
        hi = lo + 1e-9
    return np.clip(((v - lo) / (hi - lo) * (n - 1)), 0, n - 1).astype(np.uint16)
