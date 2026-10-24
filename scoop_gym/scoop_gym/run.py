"""Run policies through CommSchedule-v0 under several links and print the
in-deadline share, the return, and the misses by cause, per policy and link:
the same policies, the link the only variable.

    python -m scoop_gym.run scenarios/comm_schedule_coop_2.yaml \
        [--links ideal v2xvit fit:<pass> logged:<pass>] \
        [--policies always_boxes always_clouds deadline_120] [--out DIR]

--links overrides the scenario's link; --out writes one attribution log
(<policy>__<link>.jsonl) per run and summary.json.
"""
import argparse
import json
import os

from . import baselines
from .comm_schedule import CommSchedule


def episode(env, policy, seed=0):
    obs, _ = env.reset(seed=seed)
    ret = {a: 0.0 for a in env.agents}
    while True:
        obs, r, term, trunc, _ = env.step({a: policy(obs[a]) for a in env.agents})
        for a in r:
            ret[a] += r[a]
        if all(trunc.values()) or all(term.values()):
            return ret


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scenario")
    ap.add_argument("--links", nargs="+")
    ap.add_argument("--policies", nargs="+", default=["always_boxes", "always_features", "always_clouds",
                                                       "deadline_120"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out")
    a = ap.parse_args()
    import yaml
    sc = yaml.safe_load(open(a.scenario))
    links = a.links or [sc["link"]]
    out = []
    for link in links:
        for pname in a.policies:
            env = CommSchedule.from_scenario(dict(sc, link=link))
            ret = episode(env, baselines.make(pname), a.seed)
            s = env.record.summary()
            s.update(policy=pname, link=link, frames=len(env.times),
                     return_per_frame={k: v / len(env.times) for k, v in ret.items()})
            out.append(s)
            print("%-16s %-28s in deadline %5.1f %%   return/frame %s   misses %s" % (
                pname, link[:28], 100 * s["in_deadline_share"],
                " ".join("%.3f" % v for v in s["return_per_frame"].values()), s["misses"]))
            if a.out:
                os.makedirs(a.out, exist_ok=True)
                env.record.write(os.path.join(a.out, "%s__%s.jsonl" % (pname, link.replace("/", "_").replace(":", "-"))))
    if a.out:
        json.dump(out, open(os.path.join(a.out, "summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
