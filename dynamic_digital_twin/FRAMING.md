# Final framing — problem, motivation, novelty, contribution targets

**Frozen 2026-08-18.** This is the paper-level framing that sits above
[`METHOD.md`](METHOD.md) (mechanism detail) and [`IMPLEMENTATION.md`](IMPLEMENTATION.md)
(build plan). It folds in the task-oriented/AoII synthesis: the objective is derived
from a *measured correction of AoII* per consumer tier; the evaluation is reported in
*task regret* (tracker-as-task); the venue picks which is the headline.

---

## Problem

Synchronizing a dynamic digital twin over a real wireless network wastes both resources
it exists to conserve: nodes transmit continuously because no policy can tell which
updates matter (overhead), and the updates sent are heavy enough to arrive stale
(latency). Root cause — two false assumptions in every existing policy:

1. **Every update has non-negative value**, uniformly across the scene, regardless of
   age, timing pattern, or consumer.
2. **Silence on a lossy link is uninformative** — "nothing changed" and "packet lost"
   are indistinguishable, so the cheapest possible signal cannot be used.

Formally: minimize task-weighted twin incorrectness subject to a bit budget, across N
noisily-perceived kinematic entities, two links with measured delay/loss distributions,
and discrete payload classes — where update value depends on entity dynamics, age,
age-variance, and consumer tier, and where non-transmission must be made informative to
be affordable.

## Motivation

1. **The theory's damage function is measurably wrong.** AoII/goal-oriented sampling
   derives policies assuming any successful update resets the penalty (value ≥ 0,
   mean-age-based), on binary/small-Markov sources. The parent study measured the real
   damage function on perception stacks: value-zero age exists (all 7 architectures
   below the ego floor at 100 ms); age *variance* hurts a stateful consumer 4–6× beyond
   equal mean age; value is consumer-tier-dependent (detection vs tracking absorb age
   oppositely). Three measured structural violations of the field's assumptions.
2. **The applied line optimizes the wrong currency.** DT-assisted CP (arXiv 2607.09070)
   maximizes *uniform* accuracy with *periodic* sync; the task pays for error only
   where its decisions look, and the moving∩relevant sliver is small. Bits outside it
   are waste an accuracy objective cannot represent. TWIST (arXiv 2605.27205) owns
   "what to sync" for perception-centric wireless twins; *when*, and how silence is
   certified, are open.
3. **Silence is the cheapest channel and nobody can use it.** Negative-information
   estimation (Sijs 2012/2013; Trimpe 2014; Shi 2016) works only for self-measured
   sources on reliable links. On real radio the ambiguity compounds (silent-packet-loss
   line: exponential hypothesis growth) or is bought back with bits (ACKs/coding;
   Leong–Dey–Quevedo 2017 = the strongest theory baseline, drops modeled not resolved).
4. **The testbed can do what neither literature can.** Real dual-RAT wireless +
   channel-based sensing = the physics to certify silence for zero transmitted bits;
   closes the two-camp gap (real-time CP systems always send; event-triggered
   estimation never leaves simulation).

## Novelty

1. **Task-weighted coherence certificate (core mechanism).** Non-transmission as
   evidence for perceived third-party entities on lossy links; twin-predicted channel
   coherence certifies silence vs outage for zero bits and detects unmodeled motion at
   PHY timescale; audit attention prioritized by task relevance (decision-horizon
   anomalies trigger immediately; task-dead space logged). No adjacent line combines
   these; 2607.09070 names channel-coherence-aware sync as its own future work.
2. **AoII corrected by measurement.** Negative-value region + age-variance penalty +
   per-consumer tiers, presented as measured properties any objective must respect — a
   contribution *to* the AoII literature. **Deliberately no new named metric** (no
   acronym to defend in rebuttal; let others name it).
3. **Derived policy.** First formulation over entities × measured dual links × payload
   classes × consumer tiers; thresholds and link assignment derived, generalizing
   Wu 2013 / Han 2015 / Leong 2017 along every axis they hold fixed.
4. **First realization at perception scale.** Event-triggered goal-oriented sync on
   real Wi-Fi + 5G under ROS 2, evaluated in **task regret** via tracker-as-task (track
   continuity inside a region of interest — zero new system components; the
   planner-horizon apparatus is one flagship experiment, not the spine).

## Contribution targets

| # | contribution | pass/fail target |
|---|---|---|
| C1 | TCG protocol: certified silence on lossy wireless | ablation grid (negative-updates × certificate on/off): naive silence diverges under real loss, certified silence bounded; certificate = 0 transmitted bits |
| C2 | Measured damage function (AoII corrected) | indoor value-zero age, growth law, variance penalty, per-tier curves — measured on testbed, pre-registered gate applied |
| C3 | Derived scheduler | beats periodic / AoI-optimal / uniform event-triggered on task-regret-per-bit; beats the Leong-style hedging baseline that spends bits on the ambiguity |
| C4 | Real-time system on real radios | ≥10× steady-state traffic reduction at equal task regret; change-detection latency < any polling at equal bandwidth; **Wi-Fi-fed twin ≈ 5G-fed twin**; task-weighted silence beats uniform silence by ≈ the measured task-irrelevant motion fraction; p99 latency per stage/RAT, pinned DDS config, published bags |

**Framing rule — venue picks the wrapper, skeleton is fixed:** comm/theory venue
(TWC/INFOCOM): lead C2+C3 ("AoII, measured and realized"), C4 validates.
Systems/robotics venue (MobiCom/IPSN/ICRA/RA-L): lead C1+C4 ("task-oriented
certified-silence synchronization"), C2+C3 are the engine. Decide before the abstract,
not before the work.

**Honesty ledger** (standing): the relevance map is state that must itself be synced —
its bits go in the overhead accounting; all [S]-tagged reads in `METHOD.md` §4 remain
mandatory (add: **TWIST 2605.27205 in full** — it shapes the abstract's gap statement;
Sijs 2013 FUSION; Wu 2013; Han 2015; Leong 2017; the semantic-aware
POMDP tracking formulation 2311.06522 as the closest formulation shape).
