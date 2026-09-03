"""Nexmon CSI parsing — verified against a BCM43455c0 running 7.45.189
(nexmon.org/csi: a975-1) on a Pi 4B / Ubuntu 22.04.

Header layout, established from captured data rather than documentation:

    offset  size  field
    0-1     2     magic 0x1111          <-- 2 bytes, NOT 4
    2       1     RSSI (int8, dBm)
    3       1     frame control
    4-9     6     source MAC
    10-11   2     sequence number
    12-13   2     core (bits 0-2) / spatial stream (bits 3-5)
    14-15   2     chanspec
    16-17   2     chip version
    18+     4*N   N subcarriers, int16 pairs in (imag, real) order

Two traps this module handles:

* The first five slots after the header are CONSTANT firmware fields, not
  subcarriers.  Measured across 6164 frames they had std == 0.0 and a single
  unique value each, with magnitudes up to 11498 against ~1850 for real
  subcarriers.  Left in, they dominate any amplitude scaling.
* At 80 MHz a 20 MHz transmitter fills only ~64 of the 256 slots; the rest are
  noise.  ``occupied_span`` locates the real band instead of assuming it.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np

MAGIC = 0x1111
HEADER_LEN = 18
DEFAULT_PORT = 5500

# Slots that carry fixed firmware fields rather than measurements cannot be
# identified from a single frame — a constant is only visible as constant across
# many. calibrate_constant_slots() finds them from a batch; the node runs it once
# at startup. Do NOT hardcode positions: they were at raw 0-1 and 127-131 on the
# verified node, which is not a pattern to generalise from.

NSUB_TO_BW = {64: 20, 128: 40, 256: 80}


class ParseError(ValueError):
    pass


# Broadcom chanspec bit fields (d11.h)
_BW = {0x1000: 20, 0x1800: 40, 0x2000: 80, 0x2800: 160}


def decode_chanspec(chanspec: int) -> tuple[int, int]:
    """Return (control_channel, bandwidth_mhz) from a raw Broadcom chanspec.

    The low byte is the CENTRE channel, not the control channel: 0xe02a carries
    42, while nexutil reports it as 36/80. For 40/80/160 MHz the control channel
    is derived from the centre plus the sideband bits. Verified: 0xe02a -> (36, 80).
    """
    bw = _BW.get(chanspec & 0x3800, 0)
    centre = chanspec & 0xFF
    sb = (chanspec & 0x0700) >> 8
    if bw == 20:
        return centre, 20
    if bw == 40:
        return centre - 2 + sb * 4, 40
    if bw == 80:
        return centre - 6 + sb * 4, 80
    if bw == 160:
        return centre - 14 + sb * 4, 160
    return centre, 0


@dataclass
class CsiFrame:
    src_mac: str
    rssi: int
    frame_control: int
    seq: int
    core: int
    spatial_stream: int
    chanspec: int
    chip_version: int
    csi: np.ndarray          # complex64, one entry per raw slot
    raw_slots: int

    @property
    def channel(self) -> int:
        return decode_chanspec(self.chanspec)[0]

    @property
    def bandwidth_mhz(self) -> int:
        return decode_chanspec(self.chanspec)[1]

    def occupied_span(self, exclude: np.ndarray | None = None,
                      rel_thresh: float = 0.05) -> tuple[int, int]:
        """(lo, hi) raw slot indices of the band actually carrying signal.

        ``exclude`` must list the constant-field slots. Without it the threshold
        is set relative to a fixed field several times louder than any real
        subcarrier, and every genuine slot falls below it.
        """
        amp = np.abs(self.csi).copy()
        if exclude is not None and exclude.size:
            amp[exclude] = 0
        if amp.max() <= 0:
            raise ParseError("no slot carries energy")
        loud = np.where(amp > amp.max() * rel_thresh)[0]
        return int(loud.min()), int(loud.max())

    def select(self, keep: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (raw indices, complex values) for the given slot indices."""
        return keep, self.csi[keep]


def calibrate_constant_slots(frames: list[np.ndarray],
                             tol: float = 1e-9) -> np.ndarray:
    """Slots whose value never changes across a batch — firmware fields, not
    measurements.

    A real subcarrier cannot have zero variance: thermal noise alone guarantees
    movement. Measured on the verified node, slots 127-131 held std == 0.0 and a
    single unique value across 6164 frames, with magnitudes up to 11498 against
    ~1850 for real subcarriers.

    Needs a few hundred frames to be trustworthy; with too few, a quiet slot can
    look constant by chance.
    """
    if len(frames) < 2:
        return np.array([], dtype=int)
    X = np.abs(np.stack(frames))
    return np.where(X.std(axis=0) <= tol)[0]


def _otsu_threshold(amp):
    """Otsu on log amplitude -> a linear threshold. Log because empty spectrum,
    real subcarriers and firmware artefacts span orders of magnitude; on a linear
    axis the artefacts dominate the variance and the split lands wrong."""
    v = np.log10(np.maximum(amp, 1e-3))
    lo, hi = float(v.min()), float(v.max())
    if hi - lo < 1e-9:
        return float(amp.max()) * 0.5
    hist, edges = np.histogram(v, bins=64, range=(lo, hi))
    p = hist.astype(float) / max(hist.sum(), 1)
    c = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(p); w1 = 1.0 - w0
    m0 = np.cumsum(p * c) / np.maximum(w0, 1e-12)
    mt = float((p * c).sum())
    m1 = (mt - np.cumsum(p * c)) / np.maximum(w1, 1e-12)
    return float(10 ** c[int(np.argmax(w0 * w1 * (m0 - m1) ** 2))])


def usable_slots(frames, drop_dc=True):
    """Which slots to publish, from a calibration batch. -> (keep, excluded)"""
    X = np.abs(np.stack(frames))
    amp = X.mean(axis=0); std = X.std(axis=0); n = amp.size

    # 1. flag artefacts FIRST. Doing this after band detection lets a huge
    #    artefact slot get bridged into the band by gap-closing below.
    rough = _otsu_threshold(amp)
    sig = float(np.median(amp[amp > rough])) if (amp > rough).any() else float(amp.max())
    artefact = (std <= 1e-9) | (amp > sig * 6.0)

    # 2. split noise from signal. Artefacts are EXCLUDED from the histogram, not
    #    zeroed: zeroing makes a third cluster and the split separates that.
    thr = _otsu_threshold(amp[~artefact]) if (~artefact).any() else rough
    mask = (~artefact) & (amp > thr)

    # close gaps <=3: the DC and guard nulls are below threshold by design, so
    # otherwise the band splits at the null and only half is found
    closed = mask.copy(); gap = 0
    for i in range(1, n):
        if not mask[i]:
            gap += 1
        else:
            if 0 < gap <= 3 and mask[i - gap - 1]:
                closed[i - gap:i] = True
            gap = 0
    mask = closed

    # 3. longest contiguous run is the occupied band
    best_lo = best_hi = -1; best_len = 0; i = 0
    while i < n:
        if mask[i]:
            j = i
            while j + 1 < n and mask[j + 1]:
                j += 1
            if j - i + 1 > best_len:
                best_lo, best_hi, best_len = i, j, j - i + 1
            i = j + 1
        else:
            i += 1
    if best_len == 0:
        raise ParseError("no contiguous occupied band found")

    clean = np.where(artefact, 0.0, amp)
    while best_lo < best_hi and clean[best_lo] <= thr:
        best_lo += 1
    while best_hi > best_lo and clean[best_hi] <= thr:
        best_hi -= 1

    band = np.arange(best_lo, best_hi + 1)
    keep = band[~artefact[band]]

    if drop_dc and keep.size > 8:
        w = max(2, keep.size // 20)
        win = keep[keep.size // 2 - w: keep.size // 2 + w + 1]
        keep = keep[keep != win[int(np.argmin(amp[win]))]]

    return keep, np.setdiff1d(np.arange(n), keep)


def parse_frame(payload: bytes, *, imag_first: bool = True,
                strict: bool = True) -> CsiFrame:
    """Parse one UDP payload from port 5500.

    imag_first matches this firmware's dump order. Wrong, it yields i*conj(z):
    amplitude identical but phase MIRRORED (theta -> pi/2 - theta), which cannot
    be undone by negating phase downstream. Amplitude-only work is unaffected.
    """
    if len(payload) < HEADER_LEN + 4:
        raise ParseError(f"payload too short: {len(payload)} bytes")
    magic, = struct.unpack_from("<H", payload, 0)
    if magic != MAGIC and strict:
        raise ParseError(f"bad magic 0x{magic:04x} (expected 0x{MAGIC:04x})")
    rssi, = struct.unpack_from("<b", payload, 2)
    fctl = payload[3]
    mac = payload[4:10].hex(":")
    seq, coress, chanspec, chip = struct.unpack_from("<HHHH", payload, 10)
    body = payload[HEADER_LEN:]
    if len(body) % 4:
        raise ParseError(f"CSI body {len(body)} bytes is not a multiple of 4")
    raw = np.frombuffer(body, dtype="<i2").astype(np.float32)
    a, b = raw[0::2], raw[1::2]
    csi = (b + 1j * a if imag_first else a + 1j * b).astype(np.complex64)
    return CsiFrame(
        src_mac=mac, rssi=rssi, frame_control=fctl, seq=seq,
        core=coress & 0x7, spatial_stream=(coress >> 3) & 0x7,
        chanspec=chanspec, chip_version=chip, csi=csi, raw_slots=csi.size,
    )


def build_frame(csi, *, mac="aa:bb:cc:dd:ee:ff", rssi=-40, fctl=0x94, seq=0,
                core=0, nss=0, chanspec=0xE02A, chip=0x06AE, imag_first=True):
    """Inverse of parse_frame — for tests and synthetic replay."""
    hdr = struct.pack("<H", MAGIC) + struct.pack("<b", rssi) + bytes([fctl])
    hdr += bytes.fromhex(mac.replace(":", ""))
    hdr += struct.pack("<HHHH", seq, (nss << 3) | core, chanspec, chip)
    inter = np.empty(csi.size * 2, dtype="<i2")
    if imag_first:
        inter[0::2] = np.imag(csi).astype("<i2")
        inter[1::2] = np.real(csi).astype("<i2")
    else:
        inter[0::2] = np.real(csi).astype("<i2")
        inter[1::2] = np.imag(csi).astype("<i2")
    return hdr + inter.tobytes()
