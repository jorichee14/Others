"""The attribution log of an episode: one row per message, with the link
state it met and, when it missed its deadline, the cause; and a summary of
the misses by cause.

    rec = Recorder()
    rec.message(step, m)          # a scoop_gym.link.Message
    rec.write("episode.jsonl")
    rec.summary()                 # {"messages", "in_deadline", "misses": {cause: n}}
"""
import json
import math
from collections import Counter


def _num(v):
    return None if v is None or (isinstance(v, float) and math.isnan(v)) else v


class Recorder:
    def __init__(self, link_name=""):
        self.link_name = link_name
        self.rows = []

    def message(self, step, m, reward=None):
        self.rows.append({
            "step": step, "link": self.link_name, "src": m.src, "dst": m.dst, "kind": m.kind,
            "bytes": m.size_bytes, "t_send": m.t_send, "t_arrive": _num(m.t_arrive),
            "delay_ms": None if m.delay_s is None else m.delay_s * 1e3,
            "deadline_ms": m.deadline_s * 1e3, "in_deadline": m.in_deadline, "cause": m.cause,
            "lost": m.lost, "measured": m.measured, "latency_ms": _num(m.latency_s * 1e3),
            "transmission_ms": _num(m.transmission_s * 1e3), "queue_ms": m.queue_s * 1e3,
            "monitor": m.monitor, "reward": reward})

    def summary(self):
        n = len(self.rows)
        ok = sum(r["in_deadline"] for r in self.rows)
        return {"link": self.link_name, "messages": n, "in_deadline": ok,
                "in_deadline_share": ok / n if n else math.nan,
                "misses": dict(Counter(r["cause"] for r in self.rows if r["cause"]))}

    def write(self, path):
        with open(path, "w") as fh:
            for r in self.rows:
                fh.write(json.dumps(r) + "\n")
