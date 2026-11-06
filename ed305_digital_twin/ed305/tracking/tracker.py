"""Tracking as a data-association problem, in two stages.

1. build_tracklets(): online, causal. A constant-velocity Kalman filter per
   person on the floor. Every scan (one sensor, one time) is matched to the
   tracks by an optimal assignment whose costs are log-likelihood ratios:

       update  log P_D(x) + log N(z; Hx, S) - log(clutter)   [+ appearance, + identity]
       miss    log(1 - P_D(x))

   so a miss only hurts where that sensor should have seen the person (a
   radar not seeing someone sitting still costs almost nothing). A track ends
   when its score drops `drop` below its best, or after `max_coast` seconds
   without any report. A tracklet counts only once its score reaches
   `confirm`: the score already weighs each report against that sensor's
   clutter rate, so a noisy sensor needs more reports. Where two people are
   too close to tell apart
   (`ambiguity`), the tracklets are cut instead of guessed: deciding is left
   to stage 2, which sees what happens afterwards.

2. link_tracklets(): global. All tracklets at once, one assignment of
   "tracklet end -> next tracklet start" (Jaqaman et al. 2008 gap closing,
   a min-cost flow with unit capacities, solved exactly):

       link     motion over the gap (velocity that fades, spread that grows)
                + appearance + identity + how unlikely it is to have stayed unseen
                that long (the least-covered place within reach)
       no link  end here + start there, cheap only in entry zones (the door)
                and at the edges of the recording

   A tracklet chain is one identity.
"""
import itertools
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from .model import SensorModel, Zones

BIG = 1e6
H = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0]])
_k = np.arange(48) + 0.5
_DISK = np.c_[np.sqrt(_k / 48) * np.cos(_k * 2.39996), np.sqrt(_k / 48) * np.sin(_k * 2.39996)]   # unit disk


@dataclass
class TrackerConfig:
    # motion (Kalman, constant velocity with white-noise acceleration)
    q: float = 0.5                  # acceleration noise density, m^2/s^3
    v0: float = 1.0                 # velocity std of a new track, m/s
    v_max: float = 2.5              # fastest a person moves, m/s
    # stage 1: tracklets
    gate: float = 13.8              # Mahalanobis^2 gate (chi^2, 2 dof, 99.9 %)
    drop: float = 10.0              # end a tracklet this many nats below its best score
    max_coast: float = 1.5          # s without any report: end
    min_hits: int = 3               # and at least this many reports
    confirm: float = 12.0           # nats of evidence (vs clutter) before a tracklet counts as a person
    ambiguity: float = 1.5          # nats; a report this close to fitting two tracks is not used
    merge_gate: float = 0.0         # Mahalanobis^2 between two tracks to cut both (0: off; cutting in
                                    # crowds made identities worse in every simulated case)
    birth_density: float = 1e-4     # new people per m^2 per scan: a report that fits no track is
                                    # a newcomer or clutter; costs -log(birth_density / clutter) per sensor
    # appearance (cosine similarity of features), as two Gaussians
    app_weight: float = 1.0
    app_same: tuple = (0.75, 0.12)
    app_diff: tuple = (0.25, 0.15)
    app_cap: float = 12.0           # most nats appearance can add or take away
    fit_appearance: bool = True     # learn app_same / app_diff from the data itself (fit_appearance())
    ident_bonus: float = 10.0
    # stage 2: linking
    max_gap: float = 60.0           # s, longest gap bridged
    tau: float = 1.0                # s, how long a velocity is remembered over a gap
    wander: float = 0.6             # m/sqrt(s): position spread growth over a gap
    wander_cap: float = 4.0         # m: spread never more than this (the room)
    gap_miss_rate: float = 1.0      # nats per s of gap the sensors should have seen
    edge: float = 2.0               # s at the start / end of the recording: starting / ending is free


def app_llr(sim, cfg):
    """log p(sim | same person) - log p(sim | different), capped."""
    (ms, ss), (md, sd) = cfg.app_same, cfg.app_diff
    l = (-0.5 * ((sim - ms) / ss) ** 2 - np.log(ss)) - (-0.5 * ((sim - md) / sd) ** 2 - np.log(sd))
    return float(np.clip(l, -cfg.app_cap, cfg.app_cap))


def _F(dt):
    F = np.eye(4)
    F[0, 2] = F[1, 3] = dt
    return F


def _Q(dt, q):
    a, b, c = dt ** 3 / 3, dt ** 2 / 2, dt
    return q * np.array([[a, 0, b, 0], [0, a, 0, b], [b, 0, c, 0], [0, b, 0, c]])


class Tracklet:
    """One stretch of one person, without doubts. States after every update."""

    def __init__(self, uid, m, idx, cfg):
        self.uid = uid
        self.x = np.r_[m.xy, 0.0, 0.0]
        self.P = np.zeros((4, 4))
        self.P[:2, :2] = m.cov
        self.P[2:, 2:] = np.eye(2) * cfg.v0 ** 2
        self.t = m.t
        self.times, self.xs, self.Ps, self.meas = [m.t], [self.x.copy()], [self.P.copy()], [idx]
        self.ident = m.ident
        self.feat_sum = None if m.feat is None else m.feat.copy()
        self.llr = self.peak = 0.0
        self.hits = 1
        self.sensors = {m.sensor}

    @property
    def t0(self):
        return self.times[0]

    @property
    def t1(self):
        return self.times[-1]

    def confirmed(self, cfg):
        return self.hits >= cfg.min_hits and self.peak >= cfg.confirm

    @property
    def feat(self):
        if self.feat_sum is None:
            return None
        n = np.linalg.norm(self.feat_sum)
        return self.feat_sum / n if n > 0 else None

    def predict(self, t, cfg):
        dt = max(0.0, t - self.t)
        F = _F(dt)
        return F @ self.x, F @ self.P @ F.T + _Q(dt, cfg.q)

    def update(self, t, m, idx, x, P, dllr, cfg):
        S = P[:2, :2] + m.cov
        K = P @ H.T @ np.linalg.inv(S)
        x = x + K @ (m.xy - x[:2])
        P = (np.eye(4) - K @ H) @ P
        sp = np.linalg.norm(x[2:])
        if sp > cfg.v_max:
            x[2:] *= cfg.v_max / sp
        self.x, self.P, self.t = x, 0.5 * (P + P.T), t
        self.times.append(t), self.xs.append(x.copy()), self.Ps.append(self.P.copy()), self.meas.append(idx)
        self.llr += dllr
        self.peak = max(self.peak, self.llr)
        self.hits += 1
        self.sensors.add(m.sensor)
        if m.feat is not None:
            self.feat_sum = m.feat.copy() if self.feat_sum is None else self.feat_sum + m.feat
        if m.ident is not None:
            self.ident = m.ident


def _scans(meas):
    """Measurement indices grouped by (time, sensor), in time order."""
    order = sorted(range(len(meas)), key=lambda i: (meas[i].t, meas[i].sensor))
    for key, grp in itertools.groupby(order, key=lambda i: (meas[i].t, meas[i].sensor)):
        yield key[0], key[1], list(grp)


def _assign(C, new_cost):
    """Rows (tracks) to columns (measurements), each may stay unassigned (new_cost: per measurement)."""
    nT, nM = C.shape
    A = np.full((nT + nM, nM + nT), BIG)
    A[:nT, :nM] = np.where(np.isfinite(C), C, BIG)
    A[np.arange(nT), nM + np.arange(nT)] = 0.0
    A[nT + np.arange(nM), np.arange(nM)] = new_cost
    A[nT:, nM:] = 0.0
    r, c = linear_sum_assignment(A)
    return [(i, j) for i, j in zip(r, c) if i < nT and j < nM and A[i, j] < BIG / 2]


def build_tracklets(meas, sensors, cfg=None, stats=None):
    """Stage 1. meas: list of Measurement; sensors: {name: SensorModel}.

    stats: a dict to count why tracklets ended and how many reports were held back."""
    cfg = cfg or TrackerConfig()
    stats = {} if stats is None else stats
    count = lambda k, n=1: stats.__setitem__(k, stats.get(k, 0) + n)
    uid = itertools.count()
    active, done = [], []
    for t, name, idxs in _scans(meas):
        sm = sensors.get(name) or SensorModel(name)
        still = []
        for tr in active:
            if t - tr.t <= cfg.max_coast:
                still.append(tr)
            else:
                done.append(tr)
                count("end: no report for max_coast")
        active = still
        preds = [tr.predict(t, cfg) for tr in active]
        cut = set()
        for a in range(len(active) if cfg.merge_gate > 0 else 0):
            for b in range(a + 1, len(active)):
                ta, tb = active[a], active[b]
                if not (ta.confirmed(cfg) and tb.confirmed(cfg)) or (ta.ident and tb.ident):
                    continue
                d = preds[a][0][:2] - preds[b][0][:2]
                S = preds[a][1][:2, :2] + preds[b][1][:2, :2]
                if float(d @ np.linalg.solve(S, d)) < cfg.merge_gate:
                    cut |= {k for k, tk in ((a, ta), (b, tb)) if not tk.ident}   # on top of each other
        if cut:
            count("end: on top of another track", len(cut))
            done += [active[k] for k in cut]
            active = [tr for k, tr in enumerate(active) if k not in cut]
        nT, nM = len(active), len(idxs)
        C = np.full((nT, nM), np.inf)
        U = np.zeros((nT, nM))
        preds = [tr.predict(t, cfg) for tr in active]
        held = set()
        if nT:
            X = np.array([p[0] for p in preds])
            pd = sm.pd_for([tr.ident for tr in active], X[:, :2], X[:, 2:], np.array([p[1][2:, 2:] for p in preds]))
            miss = np.log1p(-pd)
            lc = np.log(max(sm.clutter, 1e-12))
            for i, (tr, (x, P)) in enumerate(zip(active, preds)):
                for j, k in enumerate(idxs):
                    m = meas[k]
                    if m.ident is not None and m.ident != tr.ident:
                        continue                  # an identity report joins only its own track
                    S = P[:2, :2] + m.cov
                    d = m.xy - x[:2]
                    d2 = float(d @ np.linalg.solve(S, d))
                    if d2 > cfg.gate:
                        continue
                    u = np.log(max(pd[i], 1e-6)) - 0.5 * d2 - 0.5 * np.log(np.linalg.det(2 * np.pi * S)) - lc
                    f = tr.feat
                    if f is not None and m.feat is not None:
                        u += cfg.app_weight * app_llr(float(f @ m.feat), cfg)
                    if m.ident is not None and m.ident == tr.ident:
                        u += cfg.ident_bonus
                    U[i, j] = u
                    C[i, j] = -(u - miss[i])
        new_cost = -np.log(cfg.birth_density / max(sm.clutter, 1e-12))
        pairs = _assign(C, new_cost) if nT and nM else []
        taken_t = {i for i, _ in pairs}
        taken_m = {j for _, j in pairs}
        ended = set()
        for i, j in pairs:
            tr = active[i]
            confirmed = tr.confirmed(cfg)
            rival_t = any(k != i and np.isfinite(C[k, j]) and C[k, j] - C[i, j] < cfg.ambiguity
                          and active[k].confirmed(cfg) for k in range(nT))
            rival_m = any(m2 != j and np.isfinite(C[i, m2]) and C[i, m2] - C[i, j] < cfg.ambiguity
                          for m2 in range(nM))
            if confirmed and (rival_t or rival_m) and meas[idxs[j]].ident is None:
                held.add(i)                     # whose report this is is unclear: do not use it
                count(f"held: unclear report ({name})")
                continue
            x, P = preds[i]
            tr.update(t, meas[idxs[j]], idxs[j], x, P, U[i, j], cfg)
        for i, tr in enumerate(active):
            if i in ended:
                continue
            if i not in taken_t and i not in held:
                tr.llr += float(miss[i])
                if tr.llr < tr.peak - cfg.drop:
                    ended.add(i)
                    count(f"end: score dropped (last miss: {name})")
        done += [active[i] for i in ended]
        active = [tr for i, tr in enumerate(active) if i not in ended]
        for j in range(nM):
            if j not in taken_m:
                active.append(Tracklet(next(uid), meas[idxs[j]], idxs[j], cfg))
    done += active
    return [tr for tr in done if tr.confirmed(cfg)]


@dataclass
class Track:
    """One identity: tracklets in time order."""
    id: str
    tracklets: list = field(default_factory=list)

    @property
    def ident(self):
        return next((tr.ident for tr in self.tracklets if tr.ident), None)

    @property
    def t0(self):
        return self.tracklets[0].t0

    @property
    def t1(self):
        return self.tracklets[-1].t1

    def samples(self):
        """times (N,), states (N, 4) at every update."""
        t = np.concatenate([tr.times for tr in self.tracklets])
        x = np.concatenate([np.array(tr.xs) for tr in self.tracklets])
        return t, x

    def position_at(self, times):
        """(N, 2) positions at times, NaN outside the track (gaps between its tracklets interpolated)."""
        t, x = self.samples()
        times = np.asarray(times, float)
        out = np.full((len(times), 2), np.nan)
        inside = (times >= t[0]) & (times <= t[-1])
        out[inside, 0] = np.interp(times[inside], t, x[:, 0])
        out[inside, 1] = np.interp(times[inside], t, x[:, 1])
        return out

    @property
    def measurements(self):
        return [k for tr in self.tracklets for k in tr.meas]


def _link_cost(a, b, sensors, cfg):
    dt = b.t0 - a.t1
    xa, Pa = a.xs[-1], a.Ps[-1]
    pb, Pb = b.xs[0][:2], b.Ps[0][:2, :2]
    pa = xa[:2]
    meas_spread = np.sqrt(np.trace(Pa[:2, :2] + Pb))
    if np.linalg.norm(pb - pa) > cfg.v_max * dt + 3 * meas_spread + 0.3:
        return BIG
    if a.ident and b.ident and a.ident != b.ident:
        return BIG
    k = cfg.tau * (1 - np.exp(-dt / cfg.tau))           # how far the last velocity still carries
    pred = pa + xa[2:] * k
    spread2 = min(cfg.wander ** 2 * dt, cfg.wander_cap ** 2)
    S = Pa[:2, :2] + Pb + Pa[2:, 2:] * k * k + np.eye(2) * spread2
    d = pb - pred
    cost = 0.5 * float(d @ np.linalg.solve(S, d)) + 0.5 * np.log(np.linalg.det(2 * np.pi * S))
    fa, fb = a.feat, b.feat
    if fa is not None and fb is not None:
        cost -= cfg.app_weight * app_llr(float(fa @ fb), cfg)
    if a.ident and a.ident == b.ident:
        cost -= cfg.ident_bonus
    if sensors and cfg.gap_miss_rate:
        # Could the person have stayed unseen? Somewhere within reach during the gap
        # (walking out and back), the place the sensors are least likely to see.
        reach = min(cfg.v_max * dt / 2, cfg.wander_cap)
        pts = np.vstack([pa, pb, (pa + pb) / 2 + reach * _DISK])
        ident = a.ident or b.ident
        used = [sm for sm in sensors.values() if sm.name in (a.sensors | b.sensors)]
        if used:
            pd = np.max([sm.pd_for([ident] * len(pts), pts, np.zeros_like(pts)) for sm in used], axis=0)
            cost += cfg.gap_miss_rate * dt * float(pd.min())
    return cost


def link_tracklets(tracklets, sensors=None, zones=None, cfg=None, t_range=None):
    """Stage 2: tracklets -> Tracks (identities), one global assignment."""
    cfg = cfg or TrackerConfig()
    zones = zones or Zones()
    sensors = sensors or {}
    T = sorted(tracklets, key=lambda tr: tr.t0)
    n = len(T)
    if n == 0:
        return []
    t_start, t_end = t_range if t_range else (min(tr.t0 for tr in T), max(tr.t1 for tr in T))
    t0 = np.array([tr.t0 for tr in T])
    t1 = np.array([tr.t1 for tr in T])
    L = np.full((n, n), BIG)
    for a in range(n):
        cand = np.flatnonzero((t0 > t1[a]) & (t0 - t1[a] <= cfg.max_gap))
        for b in cand:
            L[a, b] = _link_cost(T[a], T[b], sensors, cfg)
    start_xy = np.array([tr.xs[0][:2] for tr in T])
    end_xy = np.array([tr.xs[-1][:2] for tr in T])
    birth = np.where(t0 - t_start < cfg.edge, 0.0, -zones.log_p(start_xy))
    death = np.where(t_end - t1 < cfg.edge, 0.0, -zones.log_p(end_xy))
    M = np.full((2 * n, 2 * n), BIG)
    M[:n, :n] = L
    M[np.arange(n), n + np.arange(n)] = death
    M[n + np.arange(n), np.arange(n)] = birth
    M[n:, n:] = np.where(L.T < BIG / 2, 0.0, BIG)
    r, c = linear_sum_assignment(M)
    nxt = {a: b for a, b in zip(r, c) if a < n and b < n and M[a, b] < BIG / 2}
    has_prev = set(nxt.values())
    tracks, k = [], 1
    for a in range(n):
        if a in has_prev:
            continue
        chain = [a]
        while chain[-1] in nxt:
            chain.append(nxt[chain[-1]])
        trs = [T[i] for i in chain]
        ident = next((tr.ident for tr in trs if tr.ident), None)
        tracks.append(Track(ident or f"p{k}", trs))
        k += ident is None
    return tracks


def fit_appearance(tracklets, meas, min_pairs=50, max_pairs=20000, seed=0):
    """Appearance model from the data, no labels needed:
    same person      two features of one tracklet
    different people features of two tracklets at the same time, more than 1.5 m apart
    -> ((mean, std) same, (mean, std) different), or None if there are too few pairs."""
    rng = np.random.default_rng(seed)
    same, diff = [], []
    feats = [[meas[k].feat for k in tr.meas if meas[k].feat is not None] for tr in tracklets]
    for f in feats:
        if len(f) >= 2:
            F = np.array(f)
            i, j = rng.integers(0, len(F), (2, min(50, len(F))))
            same += [float(F[a] @ F[b]) for a, b in zip(i, j) if a != b]
    for a in range(len(tracklets)):
        for b in range(a + 1, len(tracklets)):
            ta, tb = tracklets[a], tracklets[b]
            lo, hi = max(ta.t0, tb.t0), min(ta.t1, tb.t1)
            if hi <= lo or not feats[a] or not feats[b]:
                continue
            tm = (lo + hi) / 2
            pa = np.array(ta.xs)[np.searchsorted(ta.times, tm).clip(0, len(ta.times) - 1), :2]
            pb = np.array(tb.xs)[np.searchsorted(tb.times, tm).clip(0, len(tb.times) - 1), :2]
            if np.linalg.norm(pa - pb) > 1.5:
                diff += [float(feats[a][rng.integers(len(feats[a]))] @ feats[b][rng.integers(len(feats[b]))])
                         for _ in range(5)]
    if len(same) < min_pairs or len(diff) < min_pairs:
        return None
    same, diff = np.array(same[:max_pairs]), np.array(diff[:max_pairs])
    return (float(same.mean()), max(float(same.std()), 0.03)), (float(diff.mean()), max(float(diff.std()), 0.03))


def track(meas, sensors, zones=None, cfg=None, link=True):
    """Measurements -> (tracks, {measurement index: track id})."""
    cfg = cfg or TrackerConfig()
    if cfg.fit_appearance and any(m.feat is not None for m in meas):
        fit = fit_appearance(build_tracklets(meas, sensors, cfg), meas)
        if fit is not None:
            from dataclasses import replace
            cfg = replace(cfg, app_same=fit[0], app_diff=fit[1])
    trs = build_tracklets(meas, sensors, cfg)
    if link:
        t_range = (min(m.t for m in meas), max(m.t for m in meas)) if meas else None
        tracks = link_tracklets(trs, sensors, zones, cfg, t_range)
    else:
        tracks = [Track(f"p{k + 1}", [tr]) for k, tr in enumerate(sorted(trs, key=lambda tr: tr.t0))]
    owner = {k: tk.id for tk in tracks for k in tk.measurements}
    return tracks, owner
