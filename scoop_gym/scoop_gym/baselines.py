"""Policies for CommSchedule-v0 (obs -> action index into KINDS).

  Always(kind)        send that kind every frame
  DeadlineRule(mbps)  send the raw cloud when the goodput the sender believes
                      it has would land it inside the deadline, else features:
                      the link monitor's goodput where there is one (logged
                      link), else the goodput the last known message implied
"""
from .comm_schedule import KINDS

INDEX = {k: i for i, (k, _) in enumerate(KINDS)}


class Always:
    def __init__(self, kind):
        self.a = INDEX[kind]
        self.name = "always_" + kind

    def __call__(self, obs):
        return self.a


class DeadlineRule:
    def __init__(self, mbps=120.0, low="features"):
        self.mbps, self.low = float(mbps), INDEX[low]
        self.name = "deadline_%g" % self.mbps

    def __call__(self, obs):
        g = obs[5] if obs[5] >= 0 else obs[1]
        if g < 0:
            return INDEX["clouds"]                 # nothing known yet: try
        return INDEX["clouds"] if g >= self.mbps else self.low


def make(name):
    if name.startswith("always_"):
        return Always(name[len("always_"):])
    if name.startswith("deadline"):
        _, _, v = name.partition("_")
        return DeadlineRule(float(v) if v else 120.0)
    raise ValueError(f"unknown policy {name!r}: always_<kind> | deadline[_<mbps>]")
