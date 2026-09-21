# Ground truth without a reference sensor — the theory, and how to test it

`PSEUDO_GT.md` argues the gap. `STAGE2_GRAPH.md` specifies the estimator.
`PLAN.md` schedules the work. **This file answers four questions that none of
them answer:** what actually makes a trajectory reference-grade when no
reference sensor is on board, what evidence licenses trusting it, how the
method is tested on datasets other than this one, and which numbers are
comparable across datasets at all.

---

## 1. The principle: error must not accumulate

This is the whole of it. Everything below is a consequence.

An odometry estimates each pose from the previous one, so its error is a sum of
increments and grows without bound. A *reference* has some mechanism that makes
each pose's error independent of how long the run has been going. The coop2
pipeline's own provenance note states this exactly: every scan is initialised
from its seed pose, never from the previous scan's refined pose, **so error
cannot accumulate along the trajectory**. That property, not the sensor, is
what makes something a reference.

There are exactly three ways to get it without a LiDAR on the platform, and all
three reduce to the same thing — an observation of something whose position is
already known:

| mechanism | what is known | error floor |
|---|---|---|
| **A. surveyed anchors** | a board's pose in `map` | the survey's uncertainty + the detector's |
| **B. prior-map registration** | a whole point cloud in `map` | the map's own accuracy + the registration's |
| **C. external observers** | a static sensor's pose in `map` | the sensor's pose uncertainty + its measurement noise |

A LiDAR-carrying platform gets B for free and continuously. A platform without
one must assemble the same property out of whatever of A, B and C the recording
contains. **That substitution is the method.**

## 2. What sets the accuracy: absolute coverage in time

Absolute observations are not continuous. Between them the backbone carries the
trajectory, and that is where the error lives. The model:

```
e(t)  ~  min over absolute observations a of  [ sigma_a  +  D(|t - t_a|) ]
```

`sigma_a` is that source's own uncertainty; `D(dt)` is the backbone's drift over
an interval `dt`. Two regimes follow, and they behave differently enough that
conflating them is the single most common way to mis-state this kind of result:

**A LEVER — unanchored at an END.** Nothing beyond `t_a` constrains anything.
Error grows monotonically with distance from the last anchor, and a rotation
inside the anchor's own sigma becomes metres of displacement far along the path.
Measured here: holding out one of two boards left the remaining chain anchored
at one end, and the solve moved **7.26 m** while reporting convergence with
whitened residuals of 0.01 / 0.08.

**A BRIDGE — unanchored BETWEEN two anchors.** Both ends are pinned, so the
accumulated drift is not free to grow; the optimiser distributes it. For a
drift that behaves as a random walk this is a Brownian bridge, whose variance
at offset `u` into a gap of length `T` is

```
Var(u)  =  sigma_D^2 * u * (T - u) / T          peak at the midpoint, where it is sigma_D^2 * T / 4
```

so peak error scales as `sqrt(T)/2`, not as `T`. For a systematic drift it
scales as `T/2`. Either way **the midpoint of the longest gap is the worst
place in the trajectory**, and its error is set by the gap's LENGTH and the
backbone's drift RATE — not by how many anchors exist.

**Consequences, and each is testable rather than asserted:**

1. **Anchor count is the wrong statistic. Anchor coverage in time is the right
   one.** Two boards at opposite ends beat six boards clustered together.
2. **The quantity to report is the longest unanchored gap**, in seconds and in
   metres of path, plus whether either END is unanchored.
3. **A better backbone helps even when the anchors do not move.** It lowers
   `sigma_D`, which is the only free parameter in the bridge. Measured here:
   the best backbone gave the best construction (199 mm) and needed the least
   correction (186 mm median move against 336-512 mm).
4. **The error model is a curve, not a scalar** — `e` against time-to-nearest-
   anchor — and that curve is what transfers between platforms, because it is
   expressed in quantities both platforms have.

### The test this predicts

Plot construction error against time-to-nearest-absolute-observation on
`mobile_1`, where held-out LiDAR gives truth everywhere. If the theory holds
the curve is flat near the anchors at roughly `sigma_a`, rises toward the gap
midpoint, and is symmetric about it. **A different shape falsifies the model**,
which is the point of writing it down.

This also replaces the covariate regression with something principled. The
primary covariate is not "dropout fraction" — it is **time to the nearest
absolute observation**, with the backbone's local drift rate as the scale
factor. Depth dropout matters only through its effect on that drift rate.

---

## 3. Trust: what licenses believing the number

Evidence about a construction comes in three tiers, and they are not
interchangeable.

| tier | what it is | available on |
|---|---|---|
| **A — held-out reference sensor** | a sensor that could have produced ground truth, deliberately not used | `mobile_1` only |
| **B — held-out absolute source** | an anchor, map or observer the construction did NOT consume | any recording with more than one source |
| **C — internal consistency** | sigma sensitivity, under-determination, revisit residuals | always |

### The rule that makes tier B possible

**Every absolute source is either a FACTOR or a CHECK. Never both.**

A residual at a source the construction consumed is the standard error of the
observations it was fitted to — 0.7 mm at `anchor` is just `7 mm / sqrt(86)` —
and it is not evidence. So the design question for a new recording is not "how
do I use everything" but **which sources do I spend on accuracy and which do I
keep for the error bar**. Spend on accuracy only what accuracy is short of.

This is the same discipline that holds the LiDAR out on `mobile_1`, applied one
level down.

### Tier C, stated as refusals rather than reports

Three failure modes that all *look like success*, each caught by a specific
check:

| failure | looks like | caught by |
|---|---|---|
| under-determined system | convergence, near-zero residuals | rms < 0.2 **together with** a large move. Low residual alone is not a good fit |
| the declared weights carrying the result | a plausible number | sigma sweep over 4x. Here it moves the answer 1.1 mm |
| a free gauge | an answer | refuse to solve when no prior factor exists |
| self-agreement quoted as validation | a small residual | tag every tier-2 row `self` or `INDEP` |

### The certification chain, end to end

```
tier A on the instrumented twin   -> the PROCEDURE is sound
tier B on the subject platform    -> the procedure worked HERE
tier C everywhere                 -> the result is determined by data, not by declarations
theory check (section 2)          -> the error model can be extrapolated, not just quoted
```

**The chain's weak link is transfer between platforms**, and it should be named
rather than papered over: different cameras fail differently, so the error
model's *function* is assumed platform-independent even though its *inputs* are
measured per platform. Tier B on the subject platform is what tests that
assumption. A recording that offers no tier B leaves it untested, and the
result is then a **prediction**, stated as one.

---

## 4. Testing on other datasets: ablate the ground truth

The obvious objection to a method validated on one recording is that the
recording chose the answer. The fix does not need another recording.

**Protocol — GT ablation.** Take any dataset with continuous ground truth.
Keep a small fraction of it, arranged in windows, and treat those poses as
"surveyed anchor observations". Hide the rest. Build the construction from a
backbone run plus the kept fraction, and score against everything hidden.

This puts the method in exactly this project's regime — sparse, intermittent
absolute evidence — on data where truth is known **everywhere**, including in
the middle of the bridge where our own recording can only guess.

It buys four things no additional recording would:

1. **The coverage sweep.** Vary the kept fraction from 100% down to 2% and
   watch the error curve. That maps the accuracy-vs-coverage relation directly
   and turns section 2's model into a measured law.
2. **Lever vs bridge, controlled.** Place the kept windows at both ends, then
   at one end only, on identical data. The 7.26 m lever we observed once
   becomes a reproducible phenomenon with a magnitude that depends on
   something.
3. **Platform independence.** Run it on datasets with different sensors and
   motions. If the error-vs-coverage curve has the same shape, the transfer
   assumption in section 3 is supported by more than an argument.
4. **Comparability.** Every number is produced by the same protocol, so the
   method's performance can be quoted against dataset properties rather than
   against other methods solving a different problem.

**Candidate datasets, by what they contribute:**

| dataset | GT source | why it is worth running |
|---|---|---|
| **EuRoC MAV** | Vicon / Leica | the field's default. Leica sequences give POSITION only, which is itself the sparse-anchor regime one axis down |
| **TUM-VI** | mocap in a subset of the volume | the closest natural analogue: sequences leave and re-enter the mocap room, so the gap structure is real rather than ablated |
| **Newer College** | survey-grade prior map + LiDAR | tests mechanism B directly, and holding the LiDAR out mirrors this project exactly |
| **Oxford Spires** | survey-grade map | outdoor, larger scale, different drift regime |
| **Hilti SLAM** | total-station control points | genuinely sparse absolute evidence in the wild — closest to this method's native input |

Start with EuRoC (smallest, most standard) to establish the protocol, then
TUM-VI for the natural gap structure, then Newer College to test mechanism B.

**One honesty requirement.** Ablated mocap is not a surveyed board: it has
different noise, no detection failures, and no incidence limits. The ablation
tests the ESTIMATOR and the coverage model, not the front end. Board detection
is validated separately, against the mapping pipeline's own detections
(8.1 mm / 0.66 deg here). Do not let one stand in for the other.

---

## 5. An optimisation that adapts to what a recording has

No dataset carries all the sources. The builder should assemble the factor set
from what is declared and **refuse when the assembled set cannot determine the
answer** — rather than defaulting, which is how a graph completes, validates
and is geometrically wrong.

### The source table

| source | factor | requires | gives |
|---|---|---|---|
| backbone odometry | between, consecutive | any SLAM front end | shape; sets `sigma_D` |
| surveyed anchors | prior on observing poses | survey + detector | absolute pose in windows |
| prior-map registration | prior per registered frame | a surveyed map cloud | absolute pose CONTINUOUSLY |
| external observer | range/bearing on the platform | a static surveyed sensor | absolute position while in range |
| inter-agent | relative pose between two agents | simultaneous co-observation | ties a weak platform to a strong one |
| IMU preintegration | between, with bias states | a noise model (Allan) | bridges dropout; constrains gravity |

### The assembly rule

```
1. odometry factors            always
2. add every source declared AVAILABLE and marked `role: factor`
3. hold back every source marked `role: check`
4. observability test:
     - at least one prior factor, else the gauge is free -> REFUSE
     - report leading / trailing unanchored time and path      -> LEVER warning
     - report the longest interior gap                          -> expected peak error, from section 2
5. solve
6. score against every `role: check` source, tagged INDEP
```

Step 3 is the one that is easy to skip and expensive to skip. Step 4's outputs
are not diagnostics printed for completeness — they are the inputs to section
2's error model, and they are what gets reported beside the result.

### What this recording supports

| source | status here | assigned role |
|---|---|---|
| backbone odometry | available, 3 backbones tried, 1 usable on `mobile_2` | factor |
| surveyed boards | 2 usable per agent (`anchor_b` unreadable: 0 of 179 frames) | **factor** |
| prior-map registration | **`map_cloud` is null** — the one missing input, and the strongest check available | **check**, once supplied |
| external observer (`infra_1`) | chain validated: `mobile_2` az +0.16 deg (sd 3.51), range +0.16 m (sd 0.53), ~600 observations in range | **check** |
| inter-agent | place overlap 100%, but SIMULTANEITY never measured | measure before claiming either way |
| IMU preintegration | blocked: noise model needs hours of static logging, bag is 156 s | unavailable |

The `infra_1` and map-registration rows correct an earlier verdict in
`RESULTS.md` that `mobile_2` has no independent check available. The scatter on
`infra_1` averages down across its observations; what does not average down is
the mast's own pose bias, so the check is good to roughly that systematic
error and should be quoted that way.

---

## 6. Metrics that are comparable across datasets

An ATE in millimetres is not comparable between a 26 m indoor lap and a 2 km
outdoor drive. Six numbers that are:

| metric | definition | why it travels |
|---|---|---|
| **absolute coverage** | fraction of run duration within the anchored windows | the input variable of section 2's model |
| **longest unanchored gap** | max interior gap, in s AND in metres of path | predicts the worst point; the metres version normalises for speed |
| **end exposure** | leading and trailing unanchored path, as a fraction of total | separates bridge from lever, which is a different failure |
| **backbone drift rate** | RPE at 1 m, from the front end alone | `sigma_D`; normalises the platform out |
| **anchored-to-bridged error ratio** | error at anchors / error at gap midpoints | the conditioning of the graph. ~1 means anchors dominate; >>1 means the backbone does |
| **construction gain** | unaligned construction ATE / se3-aligned backbone ATE | dimensionless. **< 1 means the construction beats a trajectory fitted to ground truth.** Here: 199.3 / 331.3 = **0.60** |

**Construction gain is the headline metric** and the one to put on every
dataset the protocol runs on. It is a ratio, so it carries across scales; it
compares against the fairest available baseline rather than against another
method solving a different problem; and it is < 1 only when the absolute
evidence genuinely adds information rather than merely being present.

Report beside it, always: the **local cost** (RPE at 1 m, construction vs
backbone — here 149.8 vs 91.3 mm, a real and necessary trade), the **span**,
and the **tier** of the number being quoted.

---

## 7. What this plan commits to

1. **Theory** — error is set by absolute coverage in time and the backbone's
   drift rate, with a Brownian-bridge shape between anchors and unbounded
   growth beyond the last one. Section 2 states the prediction; the
   error-vs-time-to-anchor curve on `mobile_1` tests it.
2. **Trust** — tier A certifies the procedure, tier B certifies the instance,
   tier C refuses the failures that look like success, and every source is a
   factor or a check but never both.
3. **Generality** — GT ablation on EuRoC, TUM-VI and Newer College puts the
   method in its own regime on data where truth is known everywhere, and the
   coverage sweep turns the model into a measured law.
4. **Adaptivity** — the factor set is assembled from what a recording declares,
   with an observability test that refuses rather than defaults.
5. **Comparability** — six normalised metrics, headed by construction gain.

The single external dependency is the **map cloud**, which is the difference
between `mobile_2` having no independent check and having the strongest one.
