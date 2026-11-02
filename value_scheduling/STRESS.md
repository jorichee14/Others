# Stress test of shadow scheduling

Adversarial, not defensive. Each attack states the hole, how bad it is,
and the number from data that decides it.

## A1. Proposition 1 is false when self-observation is imperfect

The proposition says a unit about x is worthless to j if j observes x
itself before x matters. "Observes" is binary; perception is not. A
cell seen at 40 m with 5 LiDAR returns and a cell seen by a partner at
8 m with 300 returns are both "observed". InCoP's distance cases exist
to show that a partner's closer view of a *jointly visible* target
raises mAP (0.253 → 0.452 with Dense CLC in its hospital split [F]), and
DR's and JigsawComm's gains on OPV2V come partly from such cells.

**How bad.** Fatal to the proposition as stated. The zero-value claim
holds only for noise-free self-observation. With noise, the correct
quantity is the *residual uncertainty after* self-observation, which is
not zero and depends on range, incidence, resolution and occlusion
fraction.

**What survives.** A soft shadow: w_j(x) = expected uncertainty about x
at a_j(x) after j's own observations, with a sensor-quality model. That
is an information-gain scheduler, and it is not new as a type (Rocha et
al. 2005; Krause and Guestrin's sensor placement; every active-
perception paper). The novelty shrinks to "with the receiver's future
observations as the conditioning set and approach time as the deadline".

**Decides it.** On OPV2V, split each baseline's transmitted bytes into
(i) cells j never sees itself within H, (ii) cells j sees itself but at
lower quality than the sender, (iii) cells j sees at equal or better
quality. The binary shadow claims (ii) is waste; if a detector oracle
gives (ii) substantial value, the binary shadow is wrong and only the
soft version stands.

## A2. Decisions depend on a cell before its approach time

The proposition assumes no decision of j depends on x before a_j(x).
A planner choosing its speed *now* depends on whether the corridor
ahead is clear, because its stopping distance, hence a_j(x), depends on
the speed it picks. Knowing early that x is clear lets j go faster;
that value is realized before a_j(x) and the proposition misses it
entirely. This is the whole point of the free-space certificate idea
and it contradicts the theorem that was supposed to make the method
task-agnostic.

**How bad.** Serious. The envelope E_j is defined by the plan, and the
plan depends on information; the definition is circular. Fixing it
means defining a_j(x) under the *most conservative* plan (slowest
speed), which puts nearly everything on the path into the shadow at
once and loses the ordering, or under the current plan, which
undervalues anticipatory information.

**Decides it.** On a closed-loop task (SCooP Gym `OcclusionHandoff`, or
a CARLA planner), measure task loss when cells are delivered at
a_j(x) − ε versus at the earliest possible time. If early delivery
changes the plan (speed, route), the anticipatory value is real and
the ordering key must include it.

## A3. The increment over present visibility may be small

The field's rules already exclude, implicitly or explicitly, what the
receiver sees *now*: Where2comm's request map is the ego's low-
confidence region, JigsawComm's top-1 picks the ego's own feature where
the ego is best (so nothing is transmitted for well-seen cells), AutoCast
ray-casts current visibility. What shadow scheduling adds is cells the
receiver will see *soon*. At 10 Hz and a 0.5 s horizon a vehicle moves
5 m and a robot 0.5 m; the field of view changes little except at turns
and corners.

**How bad.** Potentially fatal to the headline. If the bytes baselines
spend on "will see soon" cells are 10 % of their traffic, the result is
a footnote.

**Decides it.** Measure (b) from the test plan honestly, split by
horizon (0.1, 0.5, 1, 2 s) and by scene type (straight, intersection,
corner). If the waste outside the *present* visibility is already near
zero for JigsawComm, and the future-only increment is under ~20 % of
bytes, there is no paper in the waste number.

## A4. Path prediction error for vehicles

Shadows for vehicles depend on predicted trajectories 1–2 s ahead. At an
intersection the predicted path is ambiguous (turn or straight), and
the shadows of the two hypotheses differ in exactly the cells that
matter. Averaging over hypotheses inflates the shadow toward the union,
which is the present blind region plus everything reachable, i.e.
approximately what AutoCast already sends.

**How bad.** Moderate on roads, negligible for robots with planned
paths. It pushes the clean case indoors, where the comparison to the
vehicle literature is weaker.

**Decides it.** Shadow IoU between constant-velocity prediction and the
true future (OPV2V) by scene type; if IoU at 1 s is below ~0.6 at
intersections, vehicle shadows are not actionable at that horizon.

## A5. "Task-agnostic" hides a representation choice

The weight w_j(x) is an entropy of "the world state at x". Occupancy
entropy, class entropy and object-state entropy rank cells differently;
a detector cares about the second, a mapper about the first, a planner
about free space. Choosing which W the entropy is over is choosing a
task. The scheduler is agnostic only after that choice.

**How bad.** Honesty problem, not a fatal one: the claim must be "one
schedule per world-state representation serves every task that depends
on that representation", which is weaker and true.

**Decides it.** Run the schedule with occupancy entropy and score
detection; run it with class entropy and score mapping; report the
cross-loss. If the cross-loss is small, agnosticism holds in practice.

## A6. Over AutoCast it is one line of code

AutoCast holds the receiver's trajectory and ray-casts visibility from
the current pose. Casting from the predicted pose along the trajectory
is a change inside its visibility routine. A reviewer who knows AutoCast
will say the delta is an engineering tweak plus feature-level units
borrowed from DR.

**How bad.** The strongest novelty attack. The defense is not the
mechanism but what it enables: the sufficiency proposition (damaged by
A1, A2), the waste measurement (depends on A3), approach-time deadlines
(real but modest), and the task-agnostic claim (weakened by A5).

**Decides it.** Nothing in data; this is a framing judgment. If A1–A3
come back weak, the honest paper is a *measurement* paper ("how much
of collaborative-perception traffic is provably redundant to its
receiver, and where"), not a method paper.

## A7. Selection may not matter much at all

Reason-to-Transmit's result: all nine selection policies within
~0.5 AP, oracle barely above, fusion dominates [F]. If that holds on
OPV2V at the budgets that matter, every selection method including this
one is fighting over a small margin, and the only defensible benefit is
airtime, where JigsawComm already reports 500×.

**How bad.** Caps the attainable accuracy result at whatever the oracle
gap is; the airtime result competes with methods that are already at
tens of KB per frame.

**Decides it.** The oracle gap on OPV2V at 10–50 KB per frame with
fusion held fixed, measured first. If it is under 1 AP, accuracy claims
are off the table and the paper is about airtime and deadlines on a
contended link, which is the SCooP-iT story, not this one.

## Verdict

As claimed in `APPROACH.md` and `paper/DRAFT.md`, the method over-
reaches on three points: the proposition is false under imperfect
self-observation (A1) and under anticipatory decisions (A2), and
"task-agnostic" is conditional on a representation choice (A5). The
increment over present-visibility rules may be small in bytes (A3) and
is an incremental extension of AutoCast as a mechanism (A6).

What is defensible without the data: approach time as a deadline
source, and the *measurement* of receiver-redundant traffic from the
receiver's own future frames, which no one has reported. Whether a
method paper exists depends on A3 and A7, and both are one script on
OPV2V. Run those two before anything else; do not write the method
paper first.
