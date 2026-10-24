"""CommSchedule-v0: each sender decides, every LiDAR frame, what to send to
its receiver (nothing, boxes, features, or the raw cloud), and is paid for
what arrives within the deadline, less a cost per megabyte of airtime.

    env = CommSchedule(link="logged:<pass>", senders=[("mobile_2", "mobile_1")])
    obs, info = env.reset(seed=0)
    obs, rew, term, trunc, info = env.step({"mobile_2": 3})

Multi-agent, PettingZoo parallel style: dicts keyed by sender. Replay mode:
the frame times are the pass's, the agents' motion the recording's.

Observation per sender (6 floats, -1 where unknown):
  0  delay of the sender's latest message whose fate is known by now (ms)
  1  goodput that message implies, size / delay (Mbit/s)
  2  frames since a message last arrived within its deadline
  3  1 if that latest message was lost, else 0
  4  the sender's link monitor: last ping round trip (ms)     (logged link only)
  5  the sender's link monitor: latest iperf goodput (Mbit/s) (logged link only)
A message's fate is known at its arrival, or at its deadline when lost (the
acknowledgement itself is taken as free).

Action per sender: Discrete(4), the index into KINDS.
Reward per sender and frame: value[kind] if it arrives within the deadline,
minus cost_per_mb * size in MB, sent or not. The values are placeholders
for the detection gain of each message kind, which CoopDetect-v0 measures.
"""
import numpy as np

from .engines import ReplayEngine
from .link import LinkEngine
from .record import Recorder

KINDS = [("none", 0.0), ("boxes", 1e3), ("features", 1e5), ("clouds", 1.5e6)]
VALUE = {"boxes": 0.2, "features": 0.6, "clouds": 1.0}
OBS_DIM = 6


class CommSchedule:
    metadata = {"name": "CommSchedule-v0"}

    def __init__(self, link="ideal", senders=(("mobile_2", "mobile_1"),), frame_s=0.1,
                 deadline_frames=1, span=None, value=None, cost_per_mb=0.05, seed=0):
        self.link = link if isinstance(link, LinkEngine) else LinkEngine(link, seed=seed)
        self.senders = [tuple(s) for s in senders]
        self.agents = [s for s, _ in self.senders]
        self.dst = dict(self.senders)
        self.frame_s = float(frame_s)
        self.deadline_s = deadline_frames * self.frame_s
        self.value = dict(VALUE, **(value or {}))
        self.cost_per_mb = float(cost_per_mb)
        self.engine = ReplayEngine.from_link(self.link, self.agents, self.frame_s, span)
        self.times = self.engine.times()

    @classmethod
    def from_scenario(cls, sc):
        """From a scenario dict (scenarios/*.yaml)."""
        return cls(link=sc["link"], senders=[(s["src"], s["dst"]) for s in sc["senders"]],
                   frame_s=sc.get("frame_s", 0.1), deadline_frames=sc.get("deadline_frames", 1),
                   span=sc.get("span"), value=sc.get("value"), cost_per_mb=sc.get("cost_per_mb", 0.05),
                   seed=sc.get("seed", 0))

    # -- the reset / step contract ---------------------------------------------------

    def reset(self, seed=None):
        if seed is not None:
            self.link = LinkEngine(self.link.spec, seed=seed)
        self.link.reset()
        self.k = 0
        self.record = Recorder(self.link.name)
        self.sent = {a: [] for a in self.agents}
        self.last_ok = {a: -1 for a in self.agents}
        t = self.times[0]
        return {a: self._obs(a, t) for a in self.agents}, {a: {"t": float(t)} for a in self.agents}

    def step(self, actions):
        t = float(self.times[self.k])
        rew, infos = {}, {}
        for a in self.agents:
            kind, size = KINDS[int(actions.get(a, 0))]
            r = -self.cost_per_mb * size / 1e6
            m = None
            if size > 0:
                m = self.link.send(a, self.dst[a], t, size, self.deadline_s, kind)
                r += self.value[kind] if m.in_deadline else 0.0
                self.sent[a].append((self.k, m))
                self.record.message(self.k, m, r)
            rew[a] = r
            infos[a] = {"t": t, "message": m}
        self.k += 1
        done = self.k >= len(self.times)
        tn = float(self.times[min(self.k, len(self.times) - 1)])
        obs = {a: self._obs(a, tn) for a in self.agents}
        term = {a: False for a in self.agents}
        trunc = {a: done for a in self.agents}
        return obs, rew, term, trunc, infos

    # -- observation -------------------------------------------------------------------

    def _known(self, m, t):
        return (m.t_arrive is not None and m.t_arrive <= t) or (m.t_arrive is None and m.t_send + m.deadline_s <= t)

    def _obs(self, a, t):
        o = np.full(OBS_DIM, -1.0, np.float32)
        for k, m in reversed(self.sent.get(a, [])):
            if self._known(m, t):
                if m.t_arrive is not None:
                    o[0] = m.delay_s * 1e3
                    o[1] = m.size_bytes * 8 / 1e6 / max(m.delay_s, 1e-6)
                o[3] = float(m.lost)
                break
        for k, m in reversed(self.sent.get(a, [])):
            if m.in_deadline and self._known(m, t):
                self.last_ok[a] = max(self.last_ok[a], k)
                break
        o[2] = self.k - self.last_ok[a] if self.last_ok[a] >= 0 else -1.0
        mon = self.link.monitor(a, t)
        if mon.get("rtt_ms") is not None:
            o[4] = mon["rtt_ms"]
        if mon.get("goodput_mbps") is not None:
            o[5] = mon["goodput_mbps"]
        return o


def gymnasium_env(env, agent=None, others=None):
    """A single-agent gymnasium.Env view of one sender; the other senders follow
    `others` (a policy: obs -> action; default: send nothing)."""
    import gymnasium as gym

    agent = agent or env.agents[0]
    others = others or (lambda o: 0)

    class Single(gym.Env):
        metadata = env.metadata
        observation_space = gym.spaces.Box(-1.0, np.inf, (OBS_DIM,), np.float32)
        action_space = gym.spaces.Discrete(len(KINDS))

        def reset(self, seed=None, options=None):
            super().reset(seed=seed)
            self._obs, info = env.reset(seed=seed)
            return self._obs[agent], info[agent]

        def step(self, action):
            acts = {a: (action if a == agent else others(self._obs[a])) for a in env.agents}
            self._obs, r, term, trunc, info = env.step(acts)
            return self._obs[agent], r[agent], term[agent], trunc[agent], info[agent]

    return Single()
