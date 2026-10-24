# SCooP-iT Gym

Multi-agent environments in which every message an agent sends goes through
the link the SCooP-iT testbed measured. Agents reset and step as in
Gymnasium and PettingZoo; a task is scored against the testbed's ground
truth; every episode leaves an attribution log, so each missed deadline is
traced to its cause on the link.

Lingjing (arXiv 2608.08045) is the template: a Gym-like contract, scenario
files, engine-side adjudication, and replayable episodes with attribution.
What this gym adds is the link: Lingjing's message bus is ideal (a message
costs a count, it is never late or lost), here a message is late or lost as
it was on the testbed. And the world is measured, not generated: the
testbed's reference map, and in replay mode its real recordings.

## Status

| Part | State |
|---|---|
| Link engine (`scoop_gym/link.py`): every channel of `scoop_pipeline/scoop/channel.py`, plus the cause of each miss | done, tested |
| Attribution log (`scoop_gym/record.py`) | done, tested |
| Replay engine, frame clock over a pass (`scoop_gym/engines.py`) | done for tasks that need no sensor frames |
| `CommSchedule-v0` (`scoop_gym/comm_schedule.py`), baselines, runner | done, tested on a synthetic pass; to run on coop_2 and survey_1 |
| Gymnasium registration (`scoop/CommSchedule-v0`, single-sender view) | done |
| Replay engine, recorded sensor frames (through the SCooP devkit) | planned, needed by `CoopDetect-v0` |
| Twin engine (Isaac Sim, `../isaac_coop_sim`) | planned: open loop first (recorded paths, logged link), closed loop second |
| Positional link (twin agents off the recorded paths) | planned (`PositionalChannel`): survey radio map + contention mode, validated out of run |

## The contract

```python
from scoop_gym import CommSchedule
env = CommSchedule(link="logged:<processed pass>", senders=[("mobile_2", "mobile_1")])
obs, info = env.reset(seed=0)                  # dicts keyed by agent
obs, rew, term, trunc, info = env.step({"mobile_2": 3})
env.record.summary()                           # in-deadline share, misses by cause
env.record.write("episode.jsonl")              # one row per message
```

Single-agent Gymnasium view: `gymnasium.make("scoop/CommSchedule-v0", link=..., senders=[...])`
(after `import scoop_gym`), or `scoop_gym.gymnasium_env(env, agent, others)`.

## Two modes

| Mode | Observations | Agents control | Link | Trust rests on |
|---|---|---|---|---|
| replay | the real recordings | communication: what, to whom, how big | `logged:<pass>`, keyed to time | E3a (replay vs. live) |
| twin, open loop | Isaac Sim sensors along the recorded paths | communication | `logged:<pass>` (same places, same times) | E3b (twin vs. real under the same link) |
| twin, closed loop | Isaac Sim sensors | motion and communication | `positional:<grid>` | E2 (map predicts radio) and an out-of-run link check |

## Link modes

Any `scoop.channel.make` spec, so the papers and the gym share one model:
`ideal`, `v2xvit`, `where2comm`, `cobevflow:P`, `framelag:K`,
`synthetic:delay_ms=..,mbps=..,loss=..,burst_s=..`, `fit:<pass>` (synthetic,
fitted to the pass), `logged:<pass>` (measured), each with `,queue` for
back-to-back messages that wait for each other. The channel is the only
variable between runs of a policy.

A miss gets one cause: `outage`, `unmeasured` (no evidence at that time),
`queue`, `transmission` (size over goodput alone exceeds the deadline), or
`latency`.

## Tasks

| Env | Mode | Agents decide | Reward | State |
|---|---|---|---|---|
| `CommSchedule-v0` | replay | per frame and receiver: nothing, boxes (1 KB), features (100 KB), raw cloud (1.5 MB) | value of what lands within the frame, minus airtime | done |
| `CoopDetect-v0` | replay | the same, with a detector in the loop | detection gain over ego-only, minus airtime | planned |
| `OcclusionHandoff-v0` | twin | a mast warns a robot of a person behind a corner | warning lands before the robot reaches the corner | planned |
| `ConnectedCoverage-v0` | twin, closed loop | motion, and when to sync | area covered and data delivered, penalty for lost links | planned |
| `RelayPlacement-v0` | twin, closed loop | where a robot parks to relay | share of time the team stays connected | planned |

`CommSchedule-v0`'s values (boxes 0.2, features 0.6, clouds 1.0) are
placeholders for each kind's detection gain, which `CoopDetect-v0` measures.

## Scenarios

`scenarios/*.yaml`: task, mode, link spec, frame and deadline, span,
senders, value and cost. The twin's scenarios extend `isaac_coop_sim`'s
yaml (robots, sensors, actors) with the same fields.

## Run

```
python -m scoop_gym.run scenarios/comm_schedule_coop_2.yaml \
    --links ideal v2xvit fit:<pass> logged:<pass> --out runs/comm_schedule_coop_2
python scoop_gym/tests/test_scoop_gym.py
```

Needs `numpy`, `pyyaml`, the processed pass's link traces
(`scoop_pipeline/processing/link_trace.py <pass>`); `gymnasium` optional.
