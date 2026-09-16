"""Sensor constraint as a controlled ablation.

The sensor-constrained track's problem is that `mobile_2` differs from
`mobile_1` in the sensor AND the platform AND the route AND the clock, so a gap
between them is not attributable to the sensor. coop2 offers a way out that
most datasets do not: `mobile_1` carries an Ouster and a ZED on ONE rigid body,
so the Ouster stream can be degraded toward the depth camera's envelope and
every degraded run has an exact paired control — same motion, same scene, same
reference poses.

That turns "which method wins on a weak sensor" into the question actually
worth asking: **at what point does the sensor stop supporting the task**, and
does it stop for the method or for the geometry (see `observability`).

Five axes, each a deterministic function of one cloud, each with a real sensor
it imitates:

    fov        360 -> 180 -> 120 -> 87 deg     87 is the RealSense's HFoV
    range      inf -> 20 -> 10 -> 5 -> 3 m     10 is the D4xx's usable limit
    beams      128 -> 64 -> 32 -> 16           the cheap-LiDAR axis
    rate       9.7 -> 5 -> 2 Hz                low-power / bandwidth-limited
    density    1.0 -> 0.5 -> 0.1 -> 0.01       radar-like sparsity

Applied in the SENSOR frame, before any pose is applied. Applying a FoV crop in
the world frame would carve a fixed wedge out of the room instead of a fixed
wedge out of the sensor, which is a different experiment and a much easier one.
"""
from __future__ import annotations

import dataclasses

import numpy as np

__all__ = ["Ablation", "crop_fov", "truncate_range", "decimate_beams",
           "subsample", "select_rate", "apply"]


@dataclasses.dataclass
class Ablation:
    """One cell of the constraint grid. `None` means that axis is untouched."""
    hfov_deg: float | None = None
    vfov_deg: float | None = None
    max_range_m: float | None = None
    min_range_m: float | None = None
    beams: int | None = None
    density: float | None = None
    rate_hz: float | None = None
    seed: int = 0
    name: str = ""

    def label(self) -> str:
        if self.name:
            return self.name
        bits = []
        for k, fmt in (("hfov_deg", "fov{:g}"), ("max_range_m", "r{:g}"),
                       ("beams", "b{:d}"), ("density", "d{:g}"),
                       ("rate_hz", "hz{:g}")):
            v = getattr(self, k)
            if v is not None:
                bits.append(fmt.format(v))
        return "-".join(bits) or "full"

    def as_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["label"] = self.label()
        return d


def crop_fov(points: np.ndarray, hfov_deg: float | None = None,
             vfov_deg: float | None = None, forward=(1.0, 0.0, 0.0),
             up=(0.0, 0.0, 1.0)) -> np.ndarray:
    """Keep points inside a symmetric field of view about `forward`.

    Azimuth is measured in the plane orthogonal to `up`, elevation from it.
    The default axes are the ROS body convention (x forward, z up); for a cloud
    in an OPTICAL frame pass forward=(0,0,1), up=(0,-1,0) instead, or the crop
    takes a wedge out of the wrong axis and still returns a plausible cloud.
    """
    p = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if not len(p) or (hfov_deg is None and vfov_deg is None):
        return p
    f = np.asarray(forward, float) / np.linalg.norm(forward)
    u = np.asarray(up, float)
    u = u - f * (u @ f)
    u /= np.linalg.norm(u)
    l = np.cross(u, f)                                    # left-hand axis
    x, y, z = p @ f, p @ l, p @ u
    keep = np.ones(len(p), bool)
    if hfov_deg is not None:
        keep &= np.abs(np.arctan2(y, x)) <= np.radians(hfov_deg) / 2.0
    if vfov_deg is not None:
        keep &= np.abs(np.arctan2(z, np.hypot(x, y))) <= np.radians(vfov_deg) / 2.0
    return p[keep]


def truncate_range(points: np.ndarray, min_m: float | None = None,
                   max_m: float | None = None) -> np.ndarray:
    p = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if not len(p):
        return p
    d = np.linalg.norm(p, axis=1)
    keep = np.ones(len(p), bool)
    if min_m is not None:
        keep &= d >= min_m
    if max_m is not None:
        keep &= d <= max_m
    return p[keep]


def decimate_beams(points: np.ndarray, beams: int, total: int | None = None,
                   ring: np.ndarray | None = None, up=(0.0, 0.0, 1.0),
                   ) -> np.ndarray:
    """Keep an evenly spaced subset of the sensor's scan lines.

    Uses the `ring` field when the driver supplies one. Without it, rings are
    recovered by binning elevation into `total` levels — which is what a
    spinning LiDAR's rings are, and is exact for a sensor with uniform angular
    spacing. It is NOT exact for a non-uniform beam pattern (an Ouster Gradient
    or a Velodyne VLS-128), where it approximates; pass the real ring field
    when the benchmark's claim rests on the beam count.
    """
    p = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if not len(p) or beams is None:
        return p
    if ring is not None:
        r = np.asarray(ring).reshape(-1).astype(np.int64)
        total = int(r.max()) + 1 if total is None else total
    else:
        u = np.asarray(up, float) / np.linalg.norm(up)
        el = np.arctan2(p @ u, np.linalg.norm(p - np.outer(p @ u, u), axis=1))
        total = 128 if total is None else total
        lo, hi = el.min(), el.max()
        if hi - lo < 1e-9:
            return p
        r = np.clip(((el - lo) / (hi - lo) * total).astype(np.int64), 0, total - 1)
    if beams >= total:
        return p
    keep_rings = np.unique(np.linspace(0, total - 1, beams).round().astype(np.int64))
    return p[np.isin(r, keep_rings)]


def subsample(points: np.ndarray, density: float, seed: int = 0) -> np.ndarray:
    """Keep a random `density` fraction. Seeded, so a cell is reproducible."""
    p = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if not len(p) or density is None or density >= 1.0:
        return p
    n = max(1, int(round(len(p) * float(density))))
    idx = np.random.default_rng(seed).choice(len(p), size=n, replace=False)
    return p[np.sort(idx)]


def select_rate(stamps: np.ndarray, rate_hz: float | None) -> np.ndarray:
    """Indices of the frames that survive a rate reduction.

    Greedy by elapsed time rather than by taking every k-th frame: a stream
    with dropouts has an irregular period, and every-k-th silently changes the
    realised rate wherever the source stuttered.
    """
    stamps = np.asarray(stamps, dtype=np.float64).reshape(-1)
    if rate_hz is None or rate_hz <= 0 or not len(stamps):
        return np.arange(len(stamps))
    period = 1.0 / float(rate_hz)
    keep, last = [], -np.inf
    for i, t in enumerate(stamps):
        if t - last >= period - 1e-9:
            keep.append(i)
            last = t
    return np.array(keep, dtype=np.int64)


def apply(points: np.ndarray, ab: Ablation, ring: np.ndarray | None = None,
          forward=(1.0, 0.0, 0.0), up=(0.0, 0.0, 1.0)) -> np.ndarray:
    """All spatial axes of one cell, in the order a real sensor imposes them.

    Order matters and is not arbitrary: FoV and range are optical limits, so
    they come first; beam decimation is a property of the device, so it applies
    to what the optics admitted; density stands for a detector's return rate
    and so comes last. Reversing beams and density would let the subsample
    delete whole rings and change the effective beam count.

    `rate_hz` is not here — it selects frames, not points; see `select_rate`.
    """
    p = crop_fov(points, ab.hfov_deg, ab.vfov_deg, forward, up)
    p = truncate_range(p, ab.min_range_m, ab.max_range_m)
    if ab.beams is not None:
        p = decimate_beams(p, ab.beams, ring=ring, up=up)
    if ab.density is not None:
        p = subsample(p, ab.density, ab.seed)
    return p
