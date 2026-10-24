"""What the agents observe and where they are: the two engines of SCooP-iT
Gym. Both run on one frame clock; the link engine (scoop_gym.link) is the
same for both.

  ReplayEngine   the real recordings: the frame times of a pass, and the
                 recorded sensor frames for tasks that read them. Motion is
                 the recording's, so agents control communication only. The
                 logged link is exact here (validated by E3a, replay vs. live)
  TwinEngine     the Isaac Sim twin of the measured building
                 (isaac_coop_sim): simulated sensors along the recorded paths
                 (open loop, logged link valid; validated by E3b, twin vs.
                 real) or under the agents' own motion (closed loop, needs
                 the positional link). Not wired in yet
"""
import numpy as np


class ReplayEngine:
    def __init__(self, t0, t1, frame_s=0.1):
        if not t1 > t0:
            raise ValueError(f"empty span [{t0}, {t1}]")
        self.t0, self.t1, self.frame_s = float(t0), float(t1), float(frame_s)

    @classmethod
    def from_link(cls, link, agents, frame_s=0.1, span=None):
        """The span of a logged pass in which every agent has a link trace
        (else [0, 60 s] for channels tied to no recording)."""
        if span:
            return cls(span[0], span[1], frame_s)
        traces = getattr(link.channel, "traces", None)
        if not traces:
            return cls(0.0, 60.0, frame_s)
        have = [traces[a].t for a in agents if a in traces and len(traces[a].t)]
        if not have:
            raise ValueError(f"no link trace for any of {agents}")
        return cls(max(t[0] for t in have), min(t[-1] for t in have), frame_s)

    def times(self):
        return self.t0 + self.frame_s * np.arange(int((self.t1 - self.t0) / self.frame_s))

    def observe(self, agent, t):
        """The agent's recorded sensor frame at t (tasks with a detector in the
        loop). Planned: read through the SCooP devkit's extracted frames."""
        raise NotImplementedError("sensor frames in replay mode: through the SCooP devkit, not wired in yet")


class TwinEngine:
    def __init__(self, scenario):
        raise NotImplementedError("twin mode: isaac_coop_sim scenario + run_isaac.py, not wired in yet")
