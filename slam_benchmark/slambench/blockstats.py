"""Error as a DISTRIBUTION with a block bootstrap, not as one RMSE.

An ATE is a single number and it hides its own tail. A pseudo-GT with a 5 cm
median and a 2 m worst case is unusable for what pseudo-GT is used for, and no
RMSE distinguishes that from a uniform 20 cm. `docs/PSEUDO_GT.md` V3(a) asks
for percentiles, the worst stretch, and an interval -- and it asks for the
interval to come from a block bootstrap over TIME SEGMENTS rather than from
per-pose statistics. This module is that requirement.

WHY BLOCKS. Trajectory errors are heavily autocorrelated: a construction 30 cm
off at t=70.0 s is 30 cm off at t=70.1 s, and those are not two measurements of
anything. Roughly 1500 poses are a handful of independent failure episodes. A
per-pose bootstrap resamples as if every pose were an independent draw and
returns an interval too narrow by ~sqrt(poses per episode) -- an order of
magnitude here. Resampling CONTIGUOUS BLOCKS whose length exceeds the
correlation time keeps the autocorrelation inside a block, where it belongs.

The block length is MEASURED, not chosen: `correlation_time` reads it off the
error series itself, and `effective_n` says how many independent episodes the
run actually contains. When that count is small the interval is reported with
the count beside it, because an interval from five blocks is a statement about
five things and must be read as one.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Distribution:
    """What an error series is, past its mean."""
    n: int
    duration_s: float
    rmse: float
    percentiles: dict                     # {50: .., 90: .., 95: .., 99: .., 100: ..}
    worst_window: dict                    # {start_s, duration_s, median, max}
    correlation_time_s: float
    effective_n: float
    block_s: float
    ci: dict = field(default_factory=dict)      # {statistic: [lo, hi]}
    warnings: list = field(default_factory=list)

    def as_dict(self) -> dict:
        d = dict(self.__dict__)
        d["percentiles"] = {str(k): v for k, v in self.percentiles.items()}
        return d


def correlation_time(t: np.ndarray, e: np.ndarray, max_lag_fraction: float = 0.25) -> float:
    """Lag at which the error series' autocorrelation first falls below 1/e.

    This is the standard integral-scale stand-in, and the honest one to use
    here: the alternative -- summing the autocorrelation to get an integral
    time -- is dominated by the noisy long-lag tail on a series this short and
    can come out negative. The first crossing is crude and stable.

    Returns the FULL run length if the series never decorrelates, which is the
    truthful answer: a monotone drift is one episode, not many.
    """
    e = np.asarray(e, float)
    e = e - e.mean()
    n = len(e)
    if n < 8 or not np.any(e):
        return float(t[-1] - t[0]) if n > 1 else 0.0
    max_lag = max(int(n * max_lag_fraction), 2)
    denom = float(e @ e)
    dt = float(np.median(np.diff(t)))
    for lag in range(1, max_lag):
        if float(e[:-lag] @ e[lag:]) / denom < np.exp(-1.0):
            return lag * dt
    return max_lag * dt


def _blocks(t: np.ndarray, block_s: float) -> list[np.ndarray]:
    """Contiguous index blocks of `block_s` seconds, by STAMP not by count.

    By count would make a block mean different things either side of a dropout,
    and a dropout is exactly where the error changes.
    """
    edges = np.arange(t[0], t[-1] + block_s, block_s)
    idx = np.searchsorted(t, edges)
    out = [np.arange(a, b) for a, b in zip(idx[:-1], idx[1:]) if b > a]
    return out


def worst_stretch(t: np.ndarray, e: np.ndarray, window_s: float = 5.0) -> dict:
    """The window the construction is WORST over, which is what a user hits.

    Reported as a median inside the window rather than a max: one bad pose is
    an outlier, a bad five seconds is a stretch of trajectory nobody can use.
    """
    best = {"start_s": float(t[0]), "duration_s": 0.0,
            "median": float(np.median(e)), "max": float(e.max())}
    lo = np.searchsorted(t, t - window_s, side="left")
    for k in range(len(t)):
        seg = e[lo[k]:k + 1]
        if len(seg) < 3 or t[k] - t[lo[k]] < window_s * 0.5:
            continue
        m = float(np.median(seg))
        if m > best["median"]:
            best = {"start_s": float(t[lo[k]]), "duration_s": float(t[k] - t[lo[k]]),
                    "median": m, "max": float(seg.max())}
    return best


def block_bootstrap(t: np.ndarray, e: np.ndarray, block_s: float,
                    statistics=("median", "p95", "rmse"),
                    n_boot: int = 2000, alpha: float = 0.05,
                    seed: int = 0) -> dict:
    """Percentile-interval bootstrap resampling whole blocks with replacement.

    Blocks are drawn until the resample is at least as long as the original,
    so each replicate represents a run of the same duration -- not the same
    pose count, which would let a dense stretch stand in for a sparse one.
    """
    blocks = _blocks(np.asarray(t, float), block_s)
    if len(blocks) < 2:
        return {"n_blocks": len(blocks), "ci": {}, "why": "fewer than 2 blocks"}
    rng = np.random.default_rng(seed)
    fns = {"median": np.median, "p95": lambda x: np.percentile(x, 95),
           "rmse": lambda x: float(np.sqrt(np.mean(np.square(x)))),
           "p99": lambda x: np.percentile(x, 99), "max": np.max}
    draws = {s: [] for s in statistics}
    target = len(e)
    for _ in range(n_boot):
        pick, total = [], 0
        while total < target:
            b = blocks[rng.integers(len(blocks))]
            pick.append(b); total += len(b)
        sample = e[np.concatenate(pick)]
        for s in statistics:
            draws[s].append(float(fns[s](sample)))
    return {"n_blocks": len(blocks),
            "ci": {s: [float(np.percentile(draws[s], 100 * alpha / 2)),
                       float(np.percentile(draws[s], 100 * (1 - alpha / 2)))]
                   for s in statistics}}


def describe(t: np.ndarray, e: np.ndarray, window_s: float = 5.0,
             block_s: float | None = None, n_boot: int = 2000,
             seed: int = 0) -> Distribution:
    """The full picture of one error series. `block_s` defaults to 2x tau."""
    t = np.asarray(t, float)
    e = np.asarray(e, float)
    order = np.argsort(t)
    t, e = t[order], e[order]
    span = float(t[-1] - t[0])
    tau = correlation_time(t, e)
    bs = float(block_s) if block_s else max(2.0 * tau, span / 20.0)
    bs = min(bs, span / 2.0) if span > 0 else bs
    n_eff = span / max(tau, 1e-9)

    boot = block_bootstrap(t, e, bs, n_boot=n_boot, seed=seed)
    warn = []
    if n_eff < 10:
        warn.append(
            f"only {n_eff:.1f} independent episodes in {span:.0f} s (correlation time "
            f"{tau:.1f} s). The interval below is a statement about {boot['n_blocks']} "
            f"blocks and must be quoted with that number, not as an n=1500 result.")
    if boot["n_blocks"] < 5:
        warn.append(f"{boot['n_blocks']} blocks is too few to bootstrap honestly; "
                    f"report the percentiles and say the interval is unavailable.")
    if e.max() > 5.0 * np.median(e):
        warn.append(f"the worst pose ({e.max() * 1e3:.0f} mm) is more than 5x the median "
                    f"({np.median(e) * 1e3:.0f} mm) -- the RMSE is not describing this.")
    return Distribution(
        n=len(e), duration_s=span,
        rmse=float(np.sqrt(np.mean(np.square(e)))),
        percentiles={p: float(np.percentile(e, p)) for p in (50, 90, 95, 99, 100)},
        worst_window=worst_stretch(t, e, window_s),
        correlation_time_s=tau, effective_n=n_eff, block_s=bs,
        ci=boot["ci"], warnings=warn)
