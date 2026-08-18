# TCG-v0 — the minimal implementable version

**Purpose:** the smallest working system. No CSI, no optimization theory, no task
weighting, no ML. Established mechanisms only (send-on-delta + covariance clamping) —
v0 claims no novelty; it is the apparatus every research claim upgrades.
See [`FRAMING.md`](FRAMING.md) for the paper framing, [`METHOD.md`](METHOD.md) for the
full mechanism.

## One rule

Every sensing node runs the same constant-velocity predictor the twin runs. It
transmits an entity's state **only when its own observation disagrees with the shared
prediction by more than ε**. Otherwise silence — and the twin trusts its prediction.
A tiny heartbeat distinguishes "silent: nothing changed" from "silent: link dead."

## Nodes (ROS 2)

| node | host | function | est. size |
|---|---|---|---|
| `detector` | each sensing node | off-the-shelf detection (frozen) → `/detections`, world frame | config |
| `local_predictor` | each sensing node | per-entity CV-Kalman on own detections; trigger: `StateDelta` when innovation > ε; `Anchor` on entity birth/loss | ~200 loc |
| `heartbeat` | each sensing node | 1 byte, seq-numbered, every T_hb (500 ms) | ~30 loc |
| `twin` | edge | association + per-entity Kalman; continuous extrapolation; silence + live heartbeat ⇒ clamp covariance at ε-bound (silence-as-evidence, v0 form); missed heartbeats ⇒ covariance grows (outage) | ~300 loc |
| `evaluator` | edge | bytes/s per node + twin position error vs ground truth per entity | ~150 loc |

## Messages (payload discreteness from day one)

```
StateDelta: entity_id, x, y, vx, vy, cov_trace, t_meas      # ~40 B
Anchor:     entity_id, class, bbox, full_cov, t_meas        # ~200 B (+ optional crop)
```

## Trigger (entire logic)

```python
pred = kalman_predict(entity, t_now)          # same model the twin runs
if dist(obs.pos, pred.pos) > EPSILON:         # hand-set 0.3 m to start
    publish(StateDelta(...))
# else: silence == "within EPSILON of the shared prediction"
```

## v0 experiment

One plot: **bytes/s vs twin position error** — TCG-v0 vs periodic sync at 1/2/5/10 Hz,
real Wi-Fi, people walking. Expected: matched error at a fraction of traffic, gap
widening with scene staticness. This plot is proof-of-life and the baseline row of
every future table.

## Why v0 is safe

- No unverified mechanism: send-on-innovation + clamping = established theory
  (send-on-delta; Sijs-style negative information in its simplest form).
- The heartbeat is deliberately the "spends bits" resolution of silence-ambiguity —
  v0 **is** the baseline the core claim (C1) must later beat.
- Entirely classical; buildable with current skills; no gate dependencies.

## Upgrade ladder (each rung = one claim, added to a working system)

| rung | replaces | claim |
|---|---|---|
| v1: CSI/radar coherence certificate | heartbeat | C1 — zero-bit certified silence (heartbeat = ablation baseline) |
| v2: measured damage function (testbed) | hand-set ε | C2 — value-zero age, growth law, variance term |
| v3: derived thresholds + dual-RAT dispatch | fixed ε, single link | C3 — optimization layer |
| v4: task-region weighting | uniform ε | C4 headline task-regret experiment |

De-risking: any rung can fail without collapsing those below. v0+v2 alone is a
publishable measurement paper.

## Build prerequisites (from `IMPLEMENTATION.md` M1)

Clock sync with measured bound (ages are meaningless without it) · detections in a
common world frame · ground-truth method for the evaluator. Dual-RAT and CSI are NOT
prerequisites for v0.
