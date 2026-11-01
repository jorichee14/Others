# Dissertation plan: a dynamic digital twin for communication-aware robot teams

Working title: **Keeping a Robot Team's Shared Picture of a Changing
Building Correct in Time, over a Shared Wireless Link.**

The dissertation builds a dynamic digital twin, a shared model of the
building that stays correct where and when robots act on it, and
shows that the communication, motion and computation of the team
should all be organized around one quantity, the age of incorrect
information at the moment of a decision. Five papers, each a phase,
each standing alone, each feeding the next. The thread is the twin:
it is built in phase 1, kept valid in phase 2, fed under contention in
phase 3, acted on in phase 4, and generalized in phase 5.

## The thesis in one paragraph

A digital twin for robots is not a rendering; it is the team's
memory, and its value is whether it is correct at the place and
moment a robot acts on it. Correctness decays at a rate set by the
world (how often each place changes), is restored by observation (the
robot's own or a teammate's), and is limited by the channel (what
arrives in time). Define that correctness as AoII at the decision; then
what to send, when, by whom, how to share the channel, how to move,
and what to compute are all consequences of keeping it under a
tolerance.

---

## Phase 1: the instrument — a testbed that measures the link beside the sensors

*Paper 1 (SCooP-iT, IoT-J; exists in draft).*

**Question.** Can a heterogeneous indoor team be captured on one clock
and in one frame, sensing and communication alike, with no motion
capture, so that any collaborative method can be run against the link
the agents actually had?

**Contribution.** The testbed; mocap-free ground truth; every inter-
node link (RSSI, RTT, goodput, CSI) on the sensor clock in the map
frame; replay of the logged link into any method, validated against
live runs (E3); the finding that link indicators do not predict the
task outcome (loaded RTT 25–33 ms, yet 32–41 % of 1.5 MB messages miss
the frame).

**Twin content after this phase.** The static twin: the reference
map, the robots' poses in it, the measured radio map by cell. Built
once.

---

## Phase 2: validity — a twin that knows where it is stale

*Paper 2 (systems; MobiSys or IoT-J).*

**Question.** How long does an observation of a place stay true, how
does the twin learn that from the team's own revisits, and how much of
the team's update traffic was never needed?

**Contribution.**
- The per-cell change rate λ(x), estimated from interval-censored
  multi-sensor revisits (a two-state model with confidence per cell),
  and its map of site 1: walls at zero, corridors in seconds.
- The twin as a cache: one copy per cell with a lifetime from λ(x),
  invalidation when any node observes a change, promotion of persistent
  changes into a new map version, and robots holding copies with
  velocity and expiry.
- The waste measurement: the fraction of timer-driven update traffic
  whose content had not changed or whose receiver would see it itself,
  on the real link.
- The changed-building experiment: a cabinet moved and a door closed
  after mapping; invalidation reaches the robot before its stopping
  distance.

**Twin content after this phase.** The dynamic layer: an overlay of
changes with lifetimes, a change-rate map, version control. The twin
can say, for any cell, how likely it is to be wrong now.

---

## Phase 3: delivery — feeding the twin under contention

*Paper 3 (networking; INFOCOM, ToN or JSAC).*

**Question.** When the link cannot carry every observation, which
observations should reach which robot, in what form, sent by whom,
and when, and how should senders share the channel so that they
arrive?

**Contribution.**
- The fetch-timing problem with expiry: windows from the robot's
  approach time and the cell's lifetime, λ-weighted misses under link
  capacity; EDF in windows, multicast merge, representation downgrade;
  the clairvoyant bound from recorded paths and the logged link.
- The contention measurement: in-frame delivery versus number of
  senders and message size on a real AP; the Bianchi fit.
- Clock-slotted de-synchronization with kinematic compensation: a
  known delay traded for collisions; slot phase as a compute budget;
  arrival-order fusion at the server.
- The crossing experiment: five nodes, overlapping windows, two
  holders of one cell, a throttled link; misses at decision time
  against fetch-on-change, fetch-on-request, age-based scheduling and
  the bound.

**Twin content after this phase.** The twin is fed by a scheduler that
knows each robot's deadlines and each cell's lifetime; its staleness
is now bounded by the link rather than by a timer.

---

## Phase 4: action — the robot drives on the twin's correctness

*Paper 4 (robotics; T-RO, RA-L or ICRA/RSS).*

**Question.** What does "fresh enough" mean for a robot about to act,
does it predict task success where age and bytes do not, and how
should the robot move to stay inside it?

**Contribution.**
- AoII at the decision, defined per robot and cell with λ(x), as the
  metric; evidence from phases 2–3 that it predicts outcomes at blind
  corners where AoI and bytes do not.
- The motion loop: speed and path from the AoII of the places ahead;
  slow where the picture cannot be kept correct, route past a mast or
  a good-link place before a hidden region, fetch early where the
  radio map says the link will be good. Mission time at fixed safety
  against crawl-at-every-corner and drive-on-AoI.
- Task-agnosticism shown: the same rule over a detector, an occupancy
  map and a one-bit certificate.
- The theory: expected AoII at the decision as a function of λ, the
  delivery-delay distribution and the trigger threshold; the TTL rule,
  the speed rule, and the condition under which communication is
  unnecessary.

**Twin content after this phase.** The twin is closed-loop: robots
act on its per-cell correctness, and their motion changes what the
twin can know.

---

## Phase 5: the twin as a simulator — closing the loop for training and prediction

*Paper 5 (the Gym; venue after phase 4).*

**Question.** Can the twin, built from the recordings and kept valid
by phases 2–4, predict what a new policy will do on the real team
before it is deployed?

**Contribution.**
- The replay engine: recorded sensing and the logged link, frame
  clock, attribution of every miss to its cause (outage, queue,
  transmission, latency); the policies of phases 3–4 as Gym tasks.
- The twin engine: the static map as a simulated world, the dynamic
  layer as scripted changes, the positional link from the radio map,
  validated out of run.
- Train-test-deploy: a policy trained in the twin, tested on the
  recordings, deployed on the testbed, with the gap at each step
  measured (E3b, E3c of the SCooP plan).

**Twin content after this phase.** The twin predicts, not only
records; the dissertation's claim is complete.

---

## How the phases depend on each other

```
P1 instrument ──▶ P2 validity ──▶ P3 delivery ──▶ P4 action ──▶ P5 simulator
 (static twin)    (dynamic layer)  (fed in time)   (acted on)    (predicts)
      │                │                │               │
      └── the clock, frame and logged link are used by every phase ──┘
```

Each phase's paper stands alone: P1 is a testbed paper; P2 a systems
measurement with a mechanism; P3 a scheduling paper with a bound and
a real rig; P4 a robotics paper with a metric, a loop and a theory;
P5 a simulation-to-real paper. Together they are one argument: the
twin is the team's memory, its correctness at the decision is the
quantity, and communication, motion and computation are how a team
keeps it.

## Who does what

| Phase | Lead | Others |
|---|---|---|
| 1 | you (existing) | the group |
| 2 | person 1 (store, λ, invalidation, promotion, waste) | you: the metric page |
| 3 | you (fetch scheduler, bound) with person 2 (contention, slotting, compute) | person 1: the store it runs on |
| 4 | you | persons 1 and 2 run their switches |
| 5 | you with whoever builds the Gym | — |

## Timeline

| Months | Phase | Milestone |
|---|---|---|
| 0–3 | 1 finishing, 2 starting | SCooP-iT submitted; blind-corner instrument check; λ map |
| 3–6 | 2, 3 | changed-building result; crossing with throttle; contention map; slotting |
| 6–9 | 3 finishing, 4 | paper 3 submitted; the motion loop on the rig; AoII-predicts-outcome |
| 9–12 | 4, 5 starting | paper 4 submitted; replay engine complete |
| 12–18 | 5 | twin engine, out-of-run validation, one train-test-deploy cycle |
| 18–24 | writing | dissertation assembled from the five |

## What is deliberately not in it

A learned codec, semantic or token communication, a radio ray tracer
in the loop, vehicle benchmarks, large-language-model agents. Each is
an extension of one phase and is listed in that phase's future work,
not in the thesis.

## The sentence for the committee

*I built a digital twin of a building for a robot team, defined its
correctness as the age of incorrect information at the moment a robot
acts, measured that this is what predicts whether the robot acts well,
and showed that the team's communication, channel access, motion and
computation can all be organized to keep it correct in time over
ordinary shared Wi-Fi, on real robots.*
