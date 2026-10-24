"""The link engine: every message an agent sends gets an arrival time or a
drop, from one of the channels of scoop_pipeline/scoop/channel.py, and a
cause when it misses its deadline.

    link = LinkEngine("logged:<processed pass>")     # or ideal, v2xvit, fit:<pass>, ...
    m = link.send("mobile_2", "mobile_1", t, 1.5e6, deadline_s=0.1)
    m.t_arrive, m.lost, m.in_deadline, m.cause

Modes (the spec is scoop.channel.make's, so the papers and the gym share one
model):

  ideal, v2xvit, where2comm, cobevflow:P, framelag:K, synthetic:...
                     the channels of the cooperative-perception literature
  fit:<pass>         a synthetic channel fitted to a pass (tied to neither
                     time nor place): the tuned baseline
  logged:<pass>      the pass's measured link, keyed to time: valid while the
                     agents are where they were at that time (replay mode,
                     and the twin driving the recorded paths)
  positional:<grid>  the measured link keyed to place, for a twin whose agents
                     leave the recorded paths: not implemented yet (see
                     PositionalChannel)

A miss is attributed to one cause, in this order: "outage" (lost in an
outage), "unmeasured" (no measurement at t: the logged channel has no
evidence either way), "queue" (waited behind this link's previous message),
"transmission" (size over goodput alone exceeds the deadline), "latency"
(the rest). The record of every message is the attribution log
(scoop_gym.record).
"""
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

HERE = os.path.dirname(os.path.abspath(__file__))
PIPELINE = os.path.join(os.path.dirname(os.path.dirname(HERE)), "scoop_pipeline")
if PIPELINE not in sys.path:
    sys.path.insert(0, PIPELINE)
from scoop import channel as CH                                     # noqa: E402

EPS_S = 1e-9                         # a delay equal to the deadline is on time (V2X-ViT's 100 ms)


@dataclass
class Message:
    src: str
    dst: str
    t_send: float
    size_bytes: float
    deadline_s: float
    kind: str = ""
    t_arrive: Optional[float] = None
    lost: bool = False
    measured: bool = True
    latency_s: float = math.nan          # propagation and queueing outside this link's sender queue
    transmission_s: float = math.nan     # size / goodput
    queue_s: float = 0.0                 # waiting behind this link's previous message
    monitor: dict = field(default_factory=dict)

    @property
    def delay_s(self):
        return None if self.t_arrive is None else self.t_arrive - self.t_send

    @property
    def in_deadline(self):
        return self.t_arrive is not None and self.delay_s <= self.deadline_s + EPS_S

    @property
    def cause(self):
        """Why the message missed its deadline (None when it did not)."""
        if self.in_deadline:
            return None
        if self.lost:
            return "outage"
        if not self.measured:
            return "unmeasured"
        if self.queue_s > 0 and self.latency_s + self.transmission_s <= self.deadline_s + EPS_S:
            return "queue"
        if self.transmission_s > self.deadline_s + EPS_S:
            return "transmission"
        return "latency"


class PositionalChannel(CH.Channel):
    """The measured link keyed to place (twin mode, agents off the recorded
    paths). Planned: per hop, latency from the survey's ping grid at the
    sender's cell (comms_map.py's ping_grid csv), goodput from the iperf
    tests nearest that cell, scaled by the contention mode (survey vs.
    cooperation, fitted from the two passes), and an outage where the cell's
    pings were lost. Its validity is an out-of-run check: fit on one pass,
    predict another pass's logged link along that pass's path."""

    name = "positional"

    def __init__(self, grid_dir, positions):
        raise NotImplementedError("positional link: twin mode, not implemented yet")


class LinkEngine:
    """One channel for all agents, with the cause of every miss. The queue
    option (",queue" on the spec) makes back-to-back messages on a link wait
    for each other."""

    def __init__(self, spec, seed=0):
        self.spec = spec
        self.channel = CH.make(spec, seed=seed)
        self._busy = {}

    @property
    def name(self):
        return self.channel.name

    def reset(self):
        self._busy = {}

    def monitor(self, agent, t):
        """What the agent's own link monitor shows at t (logged channels only:
        the last ping round trip and the latest iperf goodput), else {}."""
        traces = getattr(self.channel, "traces", None)
        if not traces or agent not in traces:
            return {}
        tr = traces[agent]
        r, g = tr.rtt_at(t), tr.goodput_at(t, True)
        return {"rtt_ms": None if r is None else r * 1e3, "goodput_mbps": g}

    def send(self, src, dst, t_send, size_bytes, deadline_s, kind=""):
        # Channel.deliver's logic, kept apart so the latency, transmission and
        # queue terms of each message stay visible for attribution.
        lat, tx, lost, measured = self.channel.latency(src, dst, float(t_send), float(size_bytes))
        m = Message(src, dst, float(t_send), float(size_bytes), float(deadline_s), kind,
                    lost=bool(lost), measured=bool(measured), latency_s=lat, transmission_s=tx,
                    monitor=self.monitor(src, t_send))
        if lost:
            return m
        start = float(t_send)
        if self.channel.queue:
            start = max(start, self._busy.get((src, dst), -math.inf))
            self._busy[(src, dst)] = start + tx
        m.queue_s = start - float(t_send)
        m.t_arrive = start + tx + lat
        return m
