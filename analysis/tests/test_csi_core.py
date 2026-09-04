#!/usr/bin/env python3
"""Checks on the CSI maths that decide whether a run is usable.

These are the cases that produced wrong numbers on real data, kept so a change
to a default cannot quietly reintroduce them. Run: python analysis/tests/test_csi_core.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from csi_core import (band_mask, equalise_static, frame_correlation,  # noqa: E402
                      occupied_band, rician_k, temporal_coherence,
                      usable_subcarriers)

RNG = np.random.default_rng(7)
FAILED = []


def check(name, got, want, tol=0.0):
    ok = abs(got - want) <= tol if isinstance(want, (int, float)) else got == want
    print(f"  {'ok  ' if ok else 'FAIL'}  {name:38s} got {got!r}  want {want!r}")
    if not ok:
        FAILED.append(name)


def channel(n_frames, slots, taps=6, doppler=True):
    """A physical channel: taps with persistent phase, drifting slowly."""
    d = RNG.integers(0, 24, taps)
    a = RNG.random(taps) * np.exp(-d / 8.0)
    ph0 = RNG.random(taps) * 2 * np.pi
    fd = RNG.normal(0, 0.6, taps) if doppler else np.zeros(taps)
    f = np.arange(slots)[None, :]
    t = np.arange(n_frames)[:, None, None] / 180.0
    H = (a[None, None, :] * np.exp(1j * (ph0[None, None, :] + 2 * np.pi * fd[None, None, :] * t))
         * np.exp(-2j * np.pi * f[..., None] * d[None, None, :] / slots)).sum(axis=2)
    return H


print("occupied_band: a 20 MHz frame captured in an 80 MHz window")
# 64 contiguous slots plus a 3-slot DC null inside them, and two noise slots
# far outside that passed the power threshold.
idx = np.arange(256)
use = np.zeros(256, bool)
use[96:160] = True
use[126:129] = False          # DC null, must be bridged not split
use[[4, 251]] = True          # outliers at both ends of the capture window
check("first slot", occupied_band(idx, use)[0], 96)
check("span", occupied_band(idx, use)[1], 64)

print("occupied_band: an untrimmed full-width capture")
use = np.ones(256, bool)
use[:6] = use[251:] = False   # guard bands
use[127:130] = False          # DC null
first, span = occupied_band(idx, use)
check("first slot", first, 6)
check("span", span, 245)

print("occupied_band: max_gap must not bridge genuinely separate blocks")
use = np.zeros(256, bool)
use[0:40] = True
use[120:200] = True           # gap of 80 slots, far beyond max_gap
check("picks the wider block", occupied_band(idx, use), (120, 80))

print("usable_subcarriers: nulls found when most of the window is empty")
H = np.zeros((200, 256), complex)
H[:, 96:160] = channel(200, 64)
H += RNG.normal(0, 1e-4, H.shape) + 1j * RNG.normal(0, 1e-4, H.shape)
keep = usable_subcarriers(H)
check("kept count", int(keep.sum()), 64)
check("kept the right block", bool(keep[96:160].all() and not keep[:96].any()), True)

print("temporal_coherence: separates a channel from noise")
check("real channel", round(temporal_coherence(channel(300, 64)), 2), 1.0, tol=0.15)
noise = RNG.normal(0, 1, (300, 64)) + 1j * RNG.normal(0, 1, (300, 64))
check("receiver noise", abs(temporal_coherence(noise)) < 0.2, True)

print("frame_correlation: a faster-moving transmitter decorrelates sooner")
slow, fast = channel(300, 64), channel(300, 64)
# same taps, phase advancing 8x faster -- the fixture's stand-in for speed
fast = np.abs(np.fft.ifft(np.fft.fft(fast, axis=1), axis=1))  # no-op, keeps shapes explicit
r_slow = float(np.nanmedian(frame_correlation(channel(300, 64), 18)))
RNG_STATE = RNG.bit_generator.state
H_fast = channel(300, 64)
r_fast = float(np.nanmedian(frame_correlation(H_fast[::8][:37], 18)))
check("lag 18 beats lag 1 in sensitivity", float(np.nanmedian(frame_correlation(slow, 18)))
      <= float(np.nanmedian(frame_correlation(slow, 1))) + 1e-9, True)
check("faster drift, lower correlation", r_fast < r_slow, True)
check("still channel stays at 1", round(float(np.nanmedian(frame_correlation(
      channel(300, 64, doppler=False), 18))), 3), 1.0, tol=1e-3)

print("equalise_static: a fixed receiver shape must not pass as a channel")
# the coop2 signature: a few slots 20 dB off, identical in every frame
shape = np.ones(64); shape[[3, 4, 58]] = 10.0; shape[[5, 59]] = 0.1
noise_shaped = noise * shape[None, :]
check("shaped noise passes the raw gate (0.5)", temporal_coherence(noise_shaped) > 0.5, True)
check("...and fails after equalising", abs(temporal_coherence(equalise_static(noise_shaped)[0])) < 0.2, True)
ch = channel(300, 64) + 3.0
Hs, sdb = equalise_static(ch * shape[None, :])
check("shape measured", round(float(np.ptp(sdb)), 1), 40.0, tol=0.5)
check("channel survives equalising", temporal_coherence(Hs) > 0.9, True)
check("K restored", float(np.median(rician_k(Hs))) > 1.0, True)
check("K was pinned before", float(np.median(rician_k(ch * shape[None, :]))), 0.0)

print("band_mask: an isolated hot slot outside the band is not the frame")
idx = np.arange(256); use = np.zeros(256, bool); use[128:192] = True; use[0] = True
lo, span = occupied_band(idx, use)
check("band unaffected by it", (lo, span), (128, 64))
check("slot 0 excluded", bool((use & band_mask(idx, lo, span))[0]), False)
check("band kept", int((use & band_mask(idx, lo, span)).sum()), 64)

print("rician_k: nulls left in the input pin K at Rayleigh")
Hc = channel(200, 64) + 3.0          # strong dominant path -> K well above 0
check("clean input", float(np.median(rician_k(Hc))) > 1.0, True)
Hpad = np.zeros((200, 256), complex)
Hpad[:, 96:160] = Hc
check("padded with nulls", float(np.median(rician_k(Hpad))), 0.0)
check("nulls removed again", float(np.median(rician_k(Hpad[:, usable_subcarriers(Hpad)]))) > 1.0, True)

print()
if FAILED:
    print(f"{len(FAILED)} failed: {', '.join(FAILED)}")
    sys.exit(1)
print("all checks passed")
