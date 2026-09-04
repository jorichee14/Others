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


def delay_profile(H: np.ndarray, idx: np.ndarray, raw_slots: int) -> np.ndarray:
    """Power delay profile per frame, each rolled so its strongest tap is first.

    H is (n_frames, n_subcarriers) complex, idx the raw firmware slot of each
    column. The firmware's slot ordering is not recorded and the two plausible
    conventions differ by a half-window circular shift of the profile; a frame's
    own symbol-timing error shifts it further. Rolling each profile onto its own
    peak removes both, and turns the delay axis into excess delay past the
    strongest path -- which is what a delay spread should be measured over."""
    H = np.atleast_2d(H)
    grid = np.zeros((H.shape[0], raw_slots), dtype=complex)
    grid[:, np.clip(idx, 0, raw_slots - 1)] = H
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
