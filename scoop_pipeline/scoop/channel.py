"""When does a message sent over the robots' link arrive, if at all: one
model for every experiment (sim_gap, the ROS 2 relay, offline replay).

    ch = channel.make("logged:<processed pass>")       # or ideal, v2xvit, ...
    d = ch.deliver("mobile_1", "mobile_2", t_send, size_bytes)
    d.t_arrive (None when lost), d.lost, d.measured, d.delay_s

Channels (make(spec); options after a comma, e.g. "logged:<pass>,queue"):
  ideal                  arrives at once, never lost
  v2xvit                 a fixed 100 ms (V2X-ViT's async_overhead, 'sim' mode: one
                         stale 10 Hz frame), no loss, no size dependence
  where2comm             no delay, no loss (Where2comm models a bandwidth budget
                         per frame, not a delay: the method limits its own size)
  cobevflow:P            Binomial(10, P) frames of 100 ms (CoBEVFlow), no loss
  framelag:K             K frames of 100 ms (SyncNet's tau), no loss
  synthetic:k=v,...      delay_ms (constant), jitter_ms (normal, >= 0), mbps (size
                         / mbps added; none: no size dependence), loss (per
                         message), burst_s (mean outage length: loss becomes a
                         two-state Gilbert-Elliott process in time), seed
  logged:<pass>          the pass's measured link (processing/link_trace.py):
                         per hop half the ping round trip at t plus size over
                         the goodput measured at t; lost in an outage
  fit:<pass>             a synthetic channel fitted to the same pass, per link:
                         lognormal latency and median goodput from the logged
                         channel sampled every 0.1 s, outages as a
                         Gilbert-Elliott process with the measured share of
                         time and mean length; tied to neither time nor place
  ,queue                 back-to-back messages on a link wait for each other
                         (FIFO on the transmission time) instead of each getting
                         the whole goodput

The logged link, robot to robot, is two hops through the access point: the
sender's uplink then the receiver's downlink, so the latency is half of each
robot's round trip and the transmission time s / G_up(sender) + s / G_down
(receiver), both hops in the same airtime. Robot to anything else ("server",
an infra node without pings) is one hop. Round trips measured while a
robot's own iperf test runs already hold that test's queue, which the size
term then counts: a hop's latency comes from its pings outside its own tests
(the last one before a test holds through it); another robot's test is
cross traffic and stays in.
"""
import csv
import json
import math
import os
from collections import namedtuple

import numpy as np

Delivery = namedtuple("Delivery", "t_arrive lost measured delay_s")
FRAME_S = 0.1
RTT_AGE_S = 0.5


class Channel:
    name = "channel"
    queue = False

    def __init__(self):
        self._busy = {}

    def latency(self, src, dst, t, size):
        """(latency s, transmission s, lost, measured) for one message."""
        raise NotImplementedError

    def deliver(self, src, dst, t_send, size_bytes):
        lat, tx, lost, measured = self.latency(src, dst, float(t_send), float(size_bytes))
        if lost:
            return Delivery(None, True, measured, None)
        start = float(t_send)
        if self.queue:
            start = max(start, self._busy.get((src, dst), -math.inf))
            self._busy[(src, dst)] = start + tx
        t = start + tx + lat
        return Delivery(t, False, measured, t - float(t_send))


class Ideal(Channel):
    name = "ideal"

    def latency(self, src, dst, t, size):
        return 0.0, 0.0, False, True


class FrameLag(Channel):
    def __init__(self, frames, name):
        super().__init__()
        self.frames, self.name = frames, name

    def latency(self, src, dst, t, size):
        return self.frames * FRAME_S, 0.0, False, True


class CoBEVFlow(Channel):
    def __init__(self, p, n=10, seed=0):
        super().__init__()
        self.p, self.n, self.rng = float(p), n, np.random.default_rng(seed)
        self.name = "cobevflow:%g" % self.p

    def latency(self, src, dst, t, size):
        return self.rng.binomial(self.n, self.p) * FRAME_S, 0.0, False, True


class GilbertElliott:
    """Outages in time: good periods ~ Exp(mean_good_s), bad ~ Exp(mean_bad_s)."""

    def __init__(self, share, mean_bad_s, seed=0, t0=0.0):
        self.share, self.bad = float(share), float(mean_bad_s)
        self.good = self.bad * (1 - self.share) / self.share if self.share > 0 else math.inf
        self.rng = np.random.default_rng(seed)
        self.t, self.state, self.edges = t0, False, []        # [(t_start, bad)]
        self.edges.append((t0, False))

    def bad_at(self, t):
        if self.share <= 0:
            return False
        while self.t <= t:
            self.t += self.rng.exponential(self.bad if self.state else self.good)
            self.state = not self.state
            self.edges.append((self.t, self.state))
        i = max(k for k, (a, _) in enumerate(self.edges) if a <= t) if t >= self.edges[0][0] else 0
        return self.edges[i][1]


class Synthetic(Channel):
    def __init__(self, delay_ms=0.0, jitter_ms=0.0, mbps=None, loss=0.0, burst_s=None, seed=0,
                 name=None, t0=0.0):
        super().__init__()
        self.d, self.j = float(delay_ms) / 1e3, float(jitter_ms) / 1e3
        self.mbps = None if mbps in (None, "", "none") else float(mbps)
        self.loss, self.burst = float(loss), (None if burst_s in (None, "") else float(burst_s))
        self.rng = np.random.default_rng(int(seed))
        self.ge = GilbertElliott(self.loss, self.burst, int(seed) + 1, t0) if self.burst else None
        self.name = name or "synthetic"

    def latency(self, src, dst, t, size):
        lost = self.ge.bad_at(t) if self.ge else (self.rng.random() < self.loss)
        lat = self.d + (max(0.0, self.rng.normal(0.0, self.j)) if self.j > 0 else 0.0)
        tx = size * 8 / (self.mbps * 1e6) if self.mbps else 0.0
        return lat, tx, bool(lost), True


def _read(path):
    if not os.path.exists(path):
        return []
    return list(csv.DictReader(open(path)))


def _f(v):
    return math.nan if v in ("", None) else float(v)


class Trace:
    """One machine's link: pings, iperf tests, outages (processing/link_trace.py)."""

    def __init__(self, link_dir, machine):
        tag = machine.replace("/", "_")
        p = _read(os.path.join(link_dir, "ping_%s.csv" % tag))
        self.t = np.array([_f(r["t"]) for r in p])
        self.rtt = np.array([_f(r["rtt_ms"]) for r in p]) / 1e3
        own = np.array([int(r["own_load"]) for r in p], bool) if p else np.zeros(0, bool)
        ok = np.isfinite(self.rtt) & ~own
        self.ta, self.ra = self.t[ok], self.rtt[ok]
        self.tests = [(_f(r["t0"]), _f(r["t1"]), _f(r["mbps"]), int(r["up"]))
                      for r in _read(os.path.join(link_dir, "iperf_%s.csv" % tag))]
        self.own = [(a, b) for a, b, _, _ in self.tests]
        self.out = [(_f(r["t0"]), _f(r["t1"])) for r in _read(os.path.join(link_dir, "outage_%s.csv" % tag))]

    def in_outage(self, t):
        return any(a <= t < b for a, b in self.out)

    def rtt_at(self, t):
        """The last round trip outside this machine's own tests (held through a
        test), or None when too old."""
        k = int(np.searchsorted(self.ta, t, side="right")) - 1
        if k < 0:
            return None
        age = RTT_AGE_S
        for a, b in self.own:
            if a <= t <= b + RTT_AGE_S:                     # through the test and just after it
                age = max(age, (t - a) + RTT_AGE_S)
        return float(self.ra[k]) if t - self.ta[k] <= age else None

    def goodput_at(self, t, up):
        """Mbit/s of the latest test in that direction started by t (else the
        first one; else the other direction's), or None without tests."""
        for want in (up, not up):
            use = [x for x in self.tests if bool(x[3]) == bool(want)]
            if use:
                st = np.array([x[0] for x in use])
                j = int(np.searchsorted(st, t, side="right")) - 1
                return use[max(j, 0)][2]
        return None


class Logged(Channel):
    def __init__(self, proc, queue=False):
        super().__init__()
        proc = os.path.abspath(os.path.expanduser(proc))
        self.link = os.path.join(proc, "comms", "link")
        meta = os.path.join(self.link, "link.json")
        if not os.path.exists(meta):
            raise SystemExit(f"no {meta}: run processing/link_trace.py {proc} first")
        self.meta = json.load(open(meta))
        self.traces = {m: Trace(self.link, m) for m in self.meta["machines"]}
        self.queue = queue
        self.name = "logged:%s" % os.path.basename(proc) + (",queue" if queue else "")

    def hops(self, src, dst):
        h = []
        if src in self.traces:
            h.append((src, True))
        if dst in self.traces:
            h.append((dst, False))
        if not h:
            raise KeyError(f"neither {src} nor {dst} has a measured link "
                           f"(machines: {sorted(self.traces)})")
        return h

    def latency(self, src, dst, t, size):
        lat = tx = 0.0
        lost, measured = False, True
        for m, up in self.hops(src, dst):
            tr = self.traces[m]
            if tr.in_outage(t):
                lost = True
            r, g = tr.rtt_at(t), tr.goodput_at(t, up)
            if r is None or g is None:
                measured = False
                r = r if r is not None else (float(np.median(tr.ra)) if len(tr.ra) else 0.0)
                if g is None:
                    raise KeyError(f"{m} ran no iperf test: no goodput for the logged channel")
            lat += r / 2
            tx += size * 8 / (g * 1e6)
        return lat, tx, lost, measured


class Fit(Channel):
    """Per (src, dst): lognormal latency, median goodput, Gilbert-Elliott loss,
    fitted to the logged channel sampled every 0.1 s."""

    def __init__(self, proc, queue=False, seed=0):
        super().__init__()
        self.logged = Logged(proc)
        self.queue, self.seed = queue, seed
        self.params, self.models = {}, {}
        self.name = "fit:%s" % os.path.basename(os.path.abspath(proc)) + (",queue" if queue else "")

    def fit(self, src, dst):
        hops = self.logged.hops(src, dst)
        t0 = max(self.logged.traces[m].t[0] for m, _ in hops)
        t1 = min(self.logged.traces[m].t[-1] for m, _ in hops)
        lat, gp, bad = [], [], []
        for t in np.arange(t0, t1, FRAME_S):
            la, tx, lost, measured = self.logged.latency(src, dst, t, 1e6)
            if not measured and not lost:
                continue
            bad.append(lost)
            if not lost:
                lat.append(la)
                gp.append(1e6 * 8 / tx / 1e6)                     # Mbit/s of a 1 MB message
        lat = np.array(lat)
        lat = lat[lat > 0]
        bad = np.array(bad, bool)
        runs = np.diff(np.r_[0, bad.astype(int), 0])
        n_out = int((runs == 1).sum())
        share = float(bad.mean()) if len(bad) else 0.0
        mean_bad = (bad.sum() * FRAME_S / n_out) if n_out else None
        p = {"mu": float(np.mean(np.log(lat))) if len(lat) else math.log(1e-3),
             "sigma": float(np.std(np.log(lat))) if len(lat) else 0.0,
             "mbps": float(np.median(gp)) if gp else None, "outage_share": share,
             "mean_outage_s": mean_bad, "samples": int(len(bad))}
        self.params[(src, dst)] = p
        k = len(self.models)
        self.models[(src, dst)] = (np.random.default_rng(self.seed + 2 * k),
                                   GilbertElliott(share, mean_bad, self.seed + 2 * k + 1, t0) if n_out else None)
        return p

    def latency(self, src, dst, t, size):
        p = self.params.get((src, dst)) or self.fit(src, dst)
        rng, ge = self.models[(src, dst)]
        lost = ge.bad_at(t) if ge else False
        lat = float(rng.lognormal(p["mu"], p["sigma"]))
        tx = size * 8 / (p["mbps"] * 1e6) if p["mbps"] else 0.0
        return lat, tx, bool(lost), True


def make(spec, seed=0):
    """A Channel from its name (see the module doc)."""
    parts = spec.split(",")
    head, opts = parts[0], parts[1:]
    queue = "queue" in opts
    kind, _, arg = head.partition(":")
    if kind == "ideal":
        ch = Ideal()
    elif kind == "v2xvit":
        ch = FrameLag(1, "v2xvit")
    elif kind == "where2comm":
        ch = Ideal()
        ch.name = "where2comm"
    elif kind == "cobevflow":
        ch = CoBEVFlow(float(arg or 0.2), seed=seed)
    elif kind == "framelag":
        ch = FrameLag(int(arg or 1), "framelag:%s" % (arg or 1))
    elif kind == "synthetic":
        kw = dict(x.split("=", 1) for x in [arg] + [o for o in opts if "=" in o] if x)
        kw.setdefault("seed", seed)
        ch = Synthetic(name=spec, **kw)
    elif kind == "logged":
        ch = Logged(arg, queue)
    elif kind == "fit":
        ch = Fit(arg, queue, seed)
    else:
        raise ValueError(f"unknown channel {spec!r}")
    ch.queue = queue or ch.queue
    return ch
