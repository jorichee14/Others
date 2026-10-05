# -*- coding: utf-8 -*-
"""
Putting several hosts on one timeline before any frame is built.

Every other part of this converter assumes that two agents' ``header.stamp``
values are comparable.  On a multi-robot testbed they are not: each machine
stamps with its own clock, and a constant offset between two of them is
*invisible* in the data — it produces a dataset that loads, trains and evaluates
while one agent's every message is systematically early or late.

That failure mode matters more here than the usual timestamp bookkeeping,
because the study this converter feeds says so.  Its ``results/ANALYSIS.md``
(in ``collab_perception_failure_analysis/``) finds
100 ms of collaborator latency more damaging to fusion than 90% packet loss, and
that a *constant* delay is precisely the shape of impairment a motion model
absorbs quietly rather than flagging.  A 30 ms unnoticed clock offset is a 30 ms
latency impairment sitting inside the baseline of a latency study.

So the offsets are estimated rather than assumed, three ways:

1. **NTP monitor topics.**  Direct, but self-reported, and only present for the
   hosts that actually run the monitor.

2. **The delivery floor.**  rosbag2 records both the sender's ``header.stamp``
   and the recorder's ``log_time``.  Their difference is transit plus offset::

       D_h = log_time - stamp = transit_h + (C_rec - C_h)

   Transit is non-negative, so over a long recording ``min(D_h)`` approaches that
   link's true floor, and differencing two hosts cancels the recorder's clock
   entirely::

       min(D_h) - min(D_ref) = (transit_h - transit_ref) + (C_ref - C_h)

   leaving the wanted correction with the transit-floor asymmetry as its error.
   This is the only estimate available for a host with no NTP topic.

3. **The disagreement between the two**, where both exist.  If they tell the same
   story the correction is trustworthy; if they do not, neither number should be
   believed, and that is reported rather than averaged away.

Nothing here corrects silently.  :meth:`HostClocks.correction_ns` returns what to
apply and :meth:`HostClocks.residual_ns` returns what is left over afterwards, and
the converter writes both into every frame yaml.
"""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

NS = 1_000_000_000

# Field names an NtpStatus-like message might use for "offset from the reference
# clock", most specific first.  The message is a custom type in every testbed we
# have seen, so the name is resolved at runtime and the config can override it.
OFFSET_FIELDS: Tuple[Tuple[str, Optional[float]], ...] = (
    ("offset_ms", 1e-3), ("offset_msec", 1e-3), ("offset_millis", 1e-3),
    ("offset_sec", 1.0), ("offset_seconds", 1.0), ("offset_s", 1.0),
    ("time_offset", None), ("clock_offset", None), ("offset", None),
)
JITTER_FIELDS: Tuple[Tuple[str, Optional[float]], ...] = (
    ("jitter_ms", 1e-3), ("jitter_seconds", 1.0), ("jitter_sec", 1.0), ("jitter", None),
    ("rms_ms", 1e-3),
)
# The daemon's formal worst-case error bound (root dispersion + half root
# delay, in chrony terms). Much larger than the realised offset — several ms
# against a fraction of one — so it is reported as the BOUND, not carried as
# the residual: the residual says how far the stamps probably are, the bound
# says how far they could be.
BOUND_FIELDS: Tuple[Tuple[str, Optional[float]], ...] = (
    ("root_dispersion_seconds", 1.0), ("root_dispersion", None), ("dispersion", None),
    ("max_error_seconds", 1.0), ("max_error", None),
)
# Health flags a status message may carry. Read in verify mode: they are the
# daemon saying, per sample, whether discipline actually held.
HEALTH_FIELDS = ("synchronized", "clock_stepped", "offset_delta_seconds",
                 "reachability_percent", "reach_register", "stratum", "sync_source",
                 "leap_indicator", "warnings")

# Below this, the parsed unit is already close enough to the delivery floor that
# no rescale could matter, and "1000x smaller than nothing" is not evidence.
RESCALE_FLOOR_NS = 1_000_000

UNIT_SCALE = {"s": 1.0, "sec": 1.0, "seconds": 1.0,
              "ms": 1e-3, "msec": 1e-3, "us": 1e-6, "ns": 1e-9}


class ClockError(RuntimeError):
    pass


def resolve_field(msg, candidates):
    """``(name, values_scale)`` for the first candidate the message carries.

    A ``None`` scale means the field name did not disclose its unit; the caller
    infers it from the values or is told by the config.
    """
    for name, scale in candidates:
        if hasattr(msg, name):
            return name, scale
    return None, None


def public_fields(msg) -> List[str]:
    if hasattr(msg, "__slots__"):
        return [s for s in msg.__slots__ if not s.startswith("_")]
    return [k for k in vars(msg) if not k.startswith("_")]


def infer_unit(values: Sequence[float]) -> Tuple[float, str]:
    """Guess whether raw NTP offsets are seconds or milliseconds.

    A daemon that is synchronised at all sits within a few milliseconds of its
    reference.  Read as seconds those readings are ~1e-3; read as milliseconds
    they are ~1.  The mistake is only expensive in one direction — reading
    milliseconds as seconds inflates the correction a thousandfold — so the rule
    is conservative and the confidence is reported so a weak guess can be
    overridden instead of trusted.
    """
    finite = sorted(abs(float(v)) for v in values
                    if v is not None and float(v) == float(v))
    if not finite:
        return 1.0, "none"
    p90 = finite[min(len(finite) - 1, int(0.9 * len(finite)))]
    if p90 == 0:
        return 1.0, "all_zero"      # discloses nothing about its unit
    if p90 < 0.02:                  # <= 20 ms read as seconds: plausible, and as
        return 1.0, "high"          # milliseconds it would be a suspiciously good clock
    if p90 < 0.5:                   # genuinely ambiguous: 20..500 ms either way.
        return 1.0, "ambiguous"     # `choose_form` arbitrates with the delivery floor
    if p90 < 500.0:                 # 0.5..500 read as ms: plausible
        return 1e-3, "medium"
    return 1e-3, "low"              # implausible either way — say so loudly


@dataclass
class OffsetTrack:
    """One host's measured clock offset over time, nanoseconds, host - reference."""

    stamps: List[int] = field(default_factory=list)
    offsets_ns: List[int] = field(default_factory=list)
    jitter_ns: List[int] = field(default_factory=list)
    source: str = ""
    unit_confidence: str = "none"

    def finish(self) -> "OffsetTrack":
        order = sorted(range(len(self.stamps)), key=lambda i: self.stamps[i])
        self.stamps = [self.stamps[i] for i in order]
        self.offsets_ns = [self.offsets_ns[i] for i in order]
        if len(self.jitter_ns) == len(order):
            self.jitter_ns = [self.jitter_ns[i] for i in order]
        else:
            self.jitter_ns = []
        return self

    def __len__(self) -> int:
        return len(self.stamps)

    def at(self, t_ns: int) -> int:
        """Offset at ``t_ns``, interpolated between samples and **held flat**
        outside their range.

        Held rather than extrapolated on purpose: an NTP offset is a controlled
        variable, not a trajectory, and extrapolating a control loop's error
        invents a correction nobody measured.
        """
        if not self.stamps:
            return 0
        if t_ns <= self.stamps[0]:
            return self.offsets_ns[0]
        if t_ns >= self.stamps[-1]:
            return self.offsets_ns[-1]
        pos = bisect_left(self.stamps, t_ns)
        lo, hi = pos - 1, pos
        span = self.stamps[hi] - self.stamps[lo]
        if span <= 0:
            return self.offsets_ns[lo]
        frac = (t_ns - self.stamps[lo]) / span
        return int(round(self.offsets_ns[lo] * (1.0 - frac)
                         + self.offsets_ns[hi] * frac))

    def scale(self, sign: float) -> "OffsetTrack":
        self.offsets_ns = [int(round(sign * v)) for v in self.offsets_ns]
        return self

    def stats(self) -> dict:
        if not self.offsets_ns:
            return {"n": 0}
        signed = sorted(self.offsets_ns)
        magnitude = sorted(abs(v) for v in self.offsets_ns)
        n = len(signed)
        lo = signed[min(n - 1, int(0.05 * n))]
        hi = signed[min(n - 1, int(0.95 * n))]
        out = {
            "n": n,
            "p50_abs_ms": round(magnitude[n // 2] / 1e6, 4),
            "p95_abs_ms": round(magnitude[min(n - 1, int(0.95 * n))] / 1e6, 4),
            "max_abs_ms": round(magnitude[-1] / 1e6, 4),
            # The MAGNITUDE of an offset stops mattering once it is corrected; the
            # SPREAD does, because that is what interpolating between samples
            # cannot follow.  Reporting only the magnitude would call a perfectly
            # constant 30 ms offset a 30 ms residual, which is backwards.
            "spread_ms": round((hi - lo) / 1e6, 4),
            "drift_ms": round((self.offsets_ns[-1] - self.offsets_ns[0]) / 1e6, 4),
        }
        if self.jitter_ns:
            j = sorted(abs(v) for v in self.jitter_ns)
            out["reported_jitter_p95_ms"] = round(
                j[min(len(j) - 1, int(0.95 * len(j)))] / 1e6, 4)
        return out


@dataclass
class DeliveryStats:
    """``log_time - header.stamp`` for one host, nanoseconds.

    Two uses at once.  As a clock diagnostic only its **minimum** matters (transit
    floor plus offset).  As a physical measurement its **spread** is the point: it
    is the real one-way delivery latency of a real link carrying real perception
    traffic, which is the empirical object that the study's ``configs/matrix.yaml``
    synthetic ``latency`` grid stands in for.
    """

    host: str
    samples: List[int] = field(default_factory=list)

    def add(self, value_ns: int) -> None:
        self.samples.append(int(value_ns))

    def floor_ns(self) -> Optional[int]:
        return min(self.samples) if self.samples else None

    def summary(self) -> dict:
        if not self.samples:
            return {"n": 0}
        a = sorted(self.samples)
        n = len(a)

        def q(p):
            return round(a[min(n - 1, max(0, int(p * n)))] / 1e6, 3)

        return {"n": n, "min_ms": round(a[0] / 1e6, 3), "p50_ms": q(0.50),
                "p95_ms": q(0.95), "p99_ms": q(0.99),
                "max_ms": round(a[-1] / 1e6, 3)}


class HostClocks:
    """Every host's offset to the reference clock, and what survives correcting it."""

    def __init__(self, reference_host: str, apply_corrections: bool = False):
        self.reference_host = reference_host
        self.ntp: Dict[str, OffsetTrack] = {}
        self.delivery: Dict[str, DeliveryStats] = {}
        self.notes: List[str] = []
        self.apply_corrections = apply_corrections
        """False (the default) is VERIFY mode, and it is the default because on a
        host running chrony or ntpd the system clock is *already* disciplined:
        every header.stamp is on the corrected clock, and what the NTP status
        topic reports is the residual the daemon believes remains. Applying that
        residual again is at best redundant and, with the sign guessed, twice
        wrong. So in verify mode nothing is shifted; the NTP offset and the
        delivery floor are read as evidence that discipline held, reported, and
        carried into every frame as the clock uncertainty.

        True is CORRECT mode, for a bag whose hosts were NOT disciplined at
        record time — the estimates are then applied as shifts."""

    # ------------------------------------------------------------- estimates
    def delivery_floor_correction_ns(self, host: str) -> Optional[int]:
        """Nanoseconds to ADD to a ``host`` stamp to express it on the reference
        clock, from delivery floors alone (see the module docstring)."""
        me = self.delivery.get(host)
        ref = self.delivery.get(self.reference_host)
        if not me or not ref or not me.samples or not ref.samples:
            return None
        return me.floor_ns() - ref.floor_ns()

    def estimate_ns(self, host: str, t_ns: int) -> Tuple[int, str]:
        """The best estimate of ``(C_ref - C_host)`` at ``t_ns`` and where it came
        from — what WOULD be added in correct mode, and what verify mode judges."""
        return self._estimate(host, t_ns)

    def correction_ns(self, host: str, t_ns: int) -> Tuple[int, str]:
        """``(correction, source)`` to add to a stamp from ``host`` at ``t_ns``.
        Zero in verify mode — the stamps are already disciplined — with the
        source tagged so a reader can tell "not applied" from "not known"."""
        estimate, source = self._estimate(host, t_ns)
        if self.apply_corrections or host == self.reference_host:
            return estimate, source
        return 0, "verify:" + source

    def _estimate(self, host: str, t_ns: int) -> Tuple[int, str]:
        if host == self.reference_host:
            # Zero BY DEFINITION, not by measurement.  Nothing in the bag observes
            # the reference host's own error; whatever it has appears as an equal
            # and opposite error on every other agent.
            return 0, "reference"
        track = self.ntp.get(host)
        ref_track = self.ntp.get(self.reference_host)
        if track is not None and len(track):
            ref_offset = ref_track.at(t_ns) if ref_track is not None and len(ref_track) else 0
            return ref_offset - track.at(t_ns), "ntp"
        estimate = self.delivery_floor_correction_ns(host)
        if estimate is not None:
            return estimate, "delivery_floor"
        return 0, "UNKNOWN"

    def residual_ns(self, host: str) -> Tuple[float, str]:
        """The clock uncertainty a frame from ``host`` carries, nanoseconds.

        Verify mode (stamps already disciplined, nothing applied): the residual
        IS the daemon's reported offset — the p95 of its magnitude over the bag,
        floored by its reported jitter. That is chrony's own statement of how far
        the stamps may still be from true.

        Correct mode, NTP applied: half the offset series' 5..95 spread — what
        interpolating between samples fails to track — floored by the reported
        jitter.

        Delivery-floor only (no NTP on this host): the smaller of the two links'
        floors, as a **proxy** for the transit asymmetry the estimator cannot
        separate from the offset.  It is a proxy and not a bound; a host in this
        branch should be read as "this clock was never measured", and the fix is
        an NTP monitor on that host, not a better proxy.
        """
        if host == self.reference_host:
            return 0.0, "reference"
        track = self.ntp.get(host)
        if track is not None and len(track):
            stats = track.stats()
            jitter = stats.get("reported_jitter_p95_ms", 0.0)
            if not self.apply_corrections:
                return max(stats["p95_abs_ms"], jitter) * 1e6, "ntp_reported_offset"
            residual = max(0.5 * stats["spread_ms"], jitter) * 1e6
            return residual, "ntp_spread"
        me, ref = self.delivery.get(host), self.delivery.get(self.reference_host)
        floors = [d.floor_ns() for d in (me, ref) if d and d.samples]
        if floors:
            return float(min(floors)), "transit_floor_proxy"
        return float("inf"), "UNKNOWN"

    def cross_check(self, host: str, tolerance_ns: int) -> Tuple[str, dict]:
        """Do the NTP offset and the delivery floor tell the same story?"""
        track = self.ntp.get(host)
        estimate = self.delivery_floor_correction_ns(host)
        if track is None or not len(track) or estimate is None:
            return "unavailable", {}
        mid = (track.stamps[0] + track.stamps[-1]) // 2
        ntp_correction, _ = self._estimate(host, mid)
        difference = ntp_correction - estimate
        verdict = "agree" if abs(difference) <= tolerance_ns else "DISAGREE"
        return verdict, {
            "ntp_correction_ms": round(ntp_correction / 1e6, 3),
            "delivery_floor_correction_ms": round(estimate / 1e6, 3),
            "difference_ms": round(difference / 1e6, 3),
            "tolerance_ms": round(tolerance_ns / 1e6, 3),
        }

    def summary(self, t_ns: int, cross_check_tolerance_ns: int) -> Dict[str, dict]:
        hosts = sorted(set(self.ntp) | set(self.delivery) | {self.reference_host})
        out: Dict[str, dict] = {}
        for host in hosts:
            estimate, source = self._estimate(host, t_ns)
            applied, applied_source = self.correction_ns(host, t_ns)
            residual, residual_source = self.residual_ns(host)
            verdict, detail = self.cross_check(host, cross_check_tolerance_ns)
            entry = {
                "mode": "correct" if self.apply_corrections else "verify",
                "estimated_offset_ms": round(estimate / 1e6, 4),
                "estimate_source": source,
                "correction_ms": round(applied / 1e6, 4),
                "correction_source": applied_source,
                "residual_ms": round(residual / 1e6, 4),
                "residual_source": residual_source,
                "ntp_available": host in self.ntp,
                "cross_check": verdict,
            }
            if detail:
                entry["cross_check_detail"] = detail
            if host in self.ntp:
                entry["ntp"] = self.ntp[host].stats()
                entry["ntp_source"] = self.ntp[host].source
                entry["ntp_unit_confidence"] = self.ntp[host].unit_confidence
            if host in self.delivery:
                entry["delivery"] = self.delivery[host].summary()
            out[host] = entry
        return out


def choose_form(clocks: HostClocks, signs=(1.0, -1.0),
                scales=(1.0, 1e-3, 1e3)) -> Tuple[float, float, dict]:
    """Decide an NTP monitor's offset *sign* and *unit* from the data.

    Two things about a custom ``NtpStatus`` message cannot be read off its field
    name, and both are expensive to guess:

    * ``ntpq`` and ``chrony`` disagree about whether "offset" means
      reference-minus-local or local-minus-reference.  The wrong sign *doubles*
      the error instead of removing it — strictly worse than not correcting.
    * seconds versus milliseconds.  :func:`infer_unit` reads it off the magnitude,
      which is unambiguous for a well-disciplined host and genuinely ambiguous at
      the tens-of-milliseconds scale a wifi-connected robot fleet actually shows.

    The delivery floor settles both without needing any convention.  Since

        min(D_h) - min(D_ref) = (transit_h - transit_ref) - offset_h

    the right (sign, scale) is the pair that makes the corrected transit floors
    agree across hosts, because transit floors on one network are similar and
    clock offsets are not.  A rescale is only accepted when it wins decisively —
    otherwise the parsed unit stands and the disagreement is reported, since
    "the two estimates disagree" is a more useful output than a confidently
    rescaled wrong number.

    Returns ``(sign, scale, detail)``; ``scale`` multiplies the already-parsed
    offsets, so 1.0 means the parse was right.
    """
    hosts = [h for h in clocks.ntp
             if h != clocks.reference_host and clocks.delivery.get(h)]
    reference = clocks.delivery.get(clocks.reference_host)
    if not hosts or not reference or not reference.samples:
        return 1.0, 1.0, {"verdict": "undecidable",
                          "reason": "needs NTP and delivery samples on the reference "
                                    "host and at least one other"}
    ref_floor = reference.floor_ns()
    residuals = {}
    for sign in signs:
        for scale in scales:
            worst = 0.0
            for host in hosts:
                track = clocks.ntp[host]
                mid = track.at((track.stamps[0] + track.stamps[-1]) // 2)
                floor = clocks.delivery[host].floor_ns()
                worst = max(worst, abs(floor + sign * scale * mid - ref_floor))
            residuals[(sign, scale)] = worst

    best = min(residuals, key=lambda k: residuals[k])
    unscaled = min((k for k in residuals if k[1] == 1.0),
                   key=lambda k: residuals[k])
    others = [v for k, v in residuals.items() if k != best]
    margin = (min(others) - residuals[best]) if others else 0.0

    detail = {
        "residual_ms": {f"sign{int(k[0]):+d}_scale{k[1]:g}": round(v / 1e6, 3)
                        for k, v in sorted(residuals.items())},
        "margin_ms": round(margin / 1e6, 3),
    }
    if best[1] != 1.0:
        # A rescale contradicts the parse, so demand a decisive win before taking
        # it; a marginal one is reported and left to the cross-check to flag. Two
        # conditions, because either alone is fooled: the ratio test alone calls a
        # microsecond "1000x better than a millisecond", and an absolute test alone
        # would accept a rescale that merely halves an already-large residual.
        decisive = (residuals[best] < 0.25 * residuals[unscaled]
                    and residuals[unscaled] > RESCALE_FLOOR_NS)
        if not decisive:
            detail["verdict"] = ("rescale rejected (not decisive) — parsed unit kept; "
                                 "set clock.offset_unit if the cross-check disagrees")
            return unscaled[0], 1.0, detail
        detail["verdict"] = (f"unit corrected by the delivery floor: parsed offsets "
                             f"rescaled by {best[1]:g}. Set clock.offset_unit "
                             f"explicitly to make this deliberate.")
        return best[0], best[1], detail
    detail["verdict"] = "decided" if margin > 1e6 else "NEAR-TIE (report both)"
    return best[0], 1.0, detail


def choose_sign(clocks: HostClocks, candidates=(1.0, -1.0)) -> Tuple[float, dict]:
    """Sign only, with the unit taken as parsed (see :func:`choose_form`)."""
    sign, _scale, detail = choose_form(clocks, signs=candidates, scales=(1.0,))
    return sign, detail


def build_offset_track(rows, offset_field=None, offset_unit=None,
                       source="") -> Tuple[Optional[OffsetTrack], dict]:
    """Turn ``[(stamp_ns, msg), ...]`` from an NTP status topic into an OffsetTrack.

    Returns ``(track, meta)``; ``track`` is ``None`` when no offset-like field
    could be found, and ``meta['fields']`` then lists what the message does carry
    so the config can name the right one.
    """
    if not rows:
        return None, {"reason": "topic carried no messages"}
    sample = rows[0][1]
    if offset_field:
        if not hasattr(sample, offset_field):
            return None, {"reason": f"no field {offset_field!r}",
                          "fields": public_fields(sample)}
        name, declared_scale = offset_field, None
    else:
        name, declared_scale = resolve_field(sample, OFFSET_FIELDS)
        if name is None:
            return None, {"reason": "no offset-like field", "fields": public_fields(sample)}

    raw = [float(getattr(msg, name)) for _, msg in rows if hasattr(msg, name)]
    if offset_unit:
        if offset_unit not in UNIT_SCALE:
            raise ClockError(f"unknown clock offset unit {offset_unit!r}; "
                             f"expected one of {sorted(UNIT_SCALE)}")
        scale, confidence = UNIT_SCALE[offset_unit], "declared"
    elif declared_scale is not None:
        scale, confidence = declared_scale, "from_field_name"
    else:
        scale, confidence = infer_unit(raw)

    jitter_name, jitter_scale = resolve_field(sample, JITTER_FIELDS)
    jitter = ([int(round(float(getattr(msg, jitter_name))
                         * (jitter_scale if jitter_scale is not None else scale) * NS))
               for _, msg in rows] if jitter_name else [])
    bound_name, bound_scale = resolve_field(sample, BOUND_FIELDS)
    bound_ms = None
    if bound_name:
        vals = sorted(abs(float(getattr(msg, bound_name))
                          * (bound_scale if bound_scale is not None else scale))
                      for _, msg in rows)
        bound_ms = round(vals[min(len(vals) - 1, int(0.95 * len(vals)))] * 1e3, 4)

    track = OffsetTrack(
        stamps=[int(t) for t, _ in rows],
        offsets_ns=[int(round(v * scale * NS)) for v in raw],
        jitter_ns=jitter,
        source=f"{source}.{name}",
        unit_confidence=confidence).finish()
    return track, {"field": name, "unit_scale": scale, "unit_confidence": confidence,
                   "jitter_field": jitter_name, "bound_field": bound_name,
                   "bound_p95_ms": bound_ms, "health": status_health(rows)}


def status_health(rows) -> dict:
    """What the status messages say about whether discipline held.

    Everything is optional — read with getattr so a status type that lacks a
    field simply does not report it — but when present these are the daemon's
    own per-sample verdicts, which no estimate from the outside can improve on:
    ``synchronized`` false means free-running; ``clock_stepped`` with
    ``offset_delta_seconds`` is a discontinuity WITH its size; reachability
    below 100% means the source was dropping out.
    """
    out: dict = {"n": len(rows)}
    if not rows:
        return out
    sample = rows[0][1]
    have = {f for f in HEALTH_FIELDS if hasattr(sample, f)}
    out["fields"] = sorted(have)
    if "synchronized" in have:
        unsynced = [t for t, m in rows if not bool(getattr(m, "synchronized"))]
        out["unsynced_samples"] = len(unsynced)
        out["unsynced_stamps_ns"] = unsynced[:20]
    if "clock_stepped" in have:
        steps = []
        for t, m in rows:
            if bool(getattr(m, "clock_stepped")):
                delta = float(getattr(m, "offset_delta_seconds", 0.0)) \
                    if "offset_delta_seconds" in have else float("nan")
                steps.append({"stamp_ns": int(t), "delta_ms": delta * 1e3})
        out["steps"] = steps
    if "reachability_percent" in have:
        reach = [int(getattr(m, "reachability_percent")) for _, m in rows]
        out["reachability_min_pct"] = min(reach)
        out["samples_below_full_reach"] = sum(1 for r in reach if r < 100)
    if "stratum" in have:
        out["strata"] = sorted({int(getattr(m, "stratum")) for _, m in rows})
    if "sync_source" in have:
        out["sync_sources"] = sorted({str(getattr(m, "sync_source")) for _, m in rows})
    if "leap_indicator" in have:
        leaps = sorted({str(getattr(m, "leap_indicator")) for _, m in rows})
        out["leap_indicators"] = leaps
    if "warnings" in have:
        seen: dict = {}
        for _, m in rows:
            for w in list(getattr(m, "warnings") or []):
                seen[str(w)] = seen.get(str(w), 0) + 1
        out["daemon_warnings"] = seen
    return out
