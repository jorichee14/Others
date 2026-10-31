# Shadow Scheduling: Communicate Only What the Receiver Will Not See for Itself

Front matter: gap, motivation, introduction, related work, system model,
problem formulation, novelty. Claims about papers marked [F] come from
full reads (`../refs/`); [S] from abstracts. Bib keys in
`../references.bib`.

---

## Abstract

Agents that share perception over a wireless channel decide what to
send by the sender's confidence, by a learned estimate of what helps the
receiver, or by the age of what was last sent. Every such rule asks
what a message contains. We ask instead what the receiver will observe
for itself. A receiver's planned or predicted motion, its sensor model
and the shared map determine, by ray casting, the space-time cells it
will see on its own before its path reaches them; the complement is its
*shadow*, the only part of the world about which communication can
change its decisions. We prove that for any task whose loss depends on
the world through the receiver's belief, restricting transmissions to
the shadow loses nothing, for all tasks at once. Shadow scheduling fills
each receiver's shadow in order of approach time, one sender per cell,
priced in airtime, and transmits whatever representation the sender
holds for the cell; it has no learned component, no task in the loop and
no per-unit metadata. On OPV2V we measure the share of bytes that
confidence-, utility- and budget-driven methods spend on cells their
receivers see anyway, and show that shadow scheduling matches task-
specific selection on detection and on occupancy mapping at a fraction
of the airtime, with one schedule serving both.

---

## 1. Introduction

### 1.1 Motivation

A perception message is worth something to its receiver only if it
tells the receiver what it cannot find out alone. Collaborative
perception exists because single agents are blind behind occluders and
beyond range; the message that matters is the one about the region the
receiver cannot see. Yet the field's transmission rules never look at
what the receiver *will* see. Spatial-confidence selection sends what
the sender is sure about (Where2comm, DR [F]); utility estimators send
what a learned head predicts will raise a detector's accuracy
(JigsawComm [F]); timeliness scheduling sends what is oldest (TAMP,
Update the Unseen Only [F]); relevance rules send what might collide
with the receiver (AutoCast [F]). The nearest any of them comes to the
receiver's own perception is its present field of view: AutoCast ray-
casts visibility from the receiver's *current* pose, Update the Unseen
Only models the persistence of present unobservability statistically.
A receiver that will round the corner in half a second and see the
corridor for itself is, under every one of these rules, a receiver that
should be sent the corridor now.

The bytes this wastes are not marginal. In OPV2V scenes with two to
five agents, most of each agent's field of view overlaps a neighbour's,
and JigsawComm's own result that a second sender per cell *lowers*
accuracy [F] shows that fused representations do not even tolerate the
redundancy, let alone benefit from it. On a shared link the waste is
paid twice: in the airtime the redundant units occupy, and in the
deadline misses they cause for the units that were not redundant. On
the SCooP indoor testbed, two robots sharing one access point at a
25–33 ms round-trip time still see 32–41 % of 1.5 MB messages miss the
100 ms frame [R].

The information needed to avoid this is already on the wire. Every
agent broadcasts its pose at 10 Hz; robots know their planned path and
vehicles exchange predicted trajectories (AutoCast's control messages
[F]); the map is shared. From those, the set of cells a receiver will
observe itself over the next second is a ray cast. Its complement is
the receiver's shadow. Nothing outside the shadow can change what the
receiver decides, because the receiver will condition on its own
observation of it first.

### 1.2 Research gap

Three questions the literature answers, and the one it does not.

*What does the message contain?* Confidence, utility, information
bottleneck, codebooks, layered refinement (Sec. 2.1). All are
properties of the sender's data.

*What does the receiver want now?* Request maps, demand maps, handshakes
(Sec. 2.2); current-pose visibility and trajectory collision (AutoCast
[F]); present unobservability with statistical persistence (Update the
Unseen Only [F]). All are properties of the receiver's present.

*When is it stale?* Age of information and its scheduling theory
(Sec. 2.4). A property of the message's past.

*What will the receiver observe for itself before it matters?* Not
asked. It is a property of the receiver's future, it is computable from
state the agents already exchange, and it is, by the proposition of
Sec. 4, the only property that bounds the value of communication for
every task at once. No scheduling rule in collaborative perception, in
the V2X standards or in multi-robot map sharing uses it.

### 1.3 Contributions

1. **The shadow** as the demand signal for inter-agent perception
   traffic: the receiver's future self-coverage, cast from its path, its
   sensor model and the shared map, subtracted from its planning
   envelope and from the static map it already holds.
2. **A sufficiency proposition**: for any receiver task whose loss
   depends on the world state only through the receiver's belief,
   restricting transmissions to the shadow loses nothing. Task
   agnosticism follows, not as a design goal but as a theorem.
3. **Shadow scheduling**: fill shadows in approach-time order, one
   sender per cell, cheapest airtime first; representation-agnostic,
   learning-free, with a sender report of a few hundred bytes.
4. **A measurement of waste**: the fraction of bytes that confidence-,
   utility- and budget-driven methods spend outside their receivers'
   true shadows on OPV2V, computed from the dataset's own future frames.
5. **Evidence across two tasks**, detection and occupancy mapping, that
   one shadow schedule matches task-specific selection at equal airtime,
   and the price of its task-blindness measured against each task's
   oracle.

---

## 2. Related Work

### 2.1 Selection by what the message contains

Where2comm's spatial confidence map, EffiComm's keep ratios, How2comm's
mutual-information filter, InfoCom's bottleneck, CodeFilling's codebook
and RDComm's pragmatic rate–distortion all rank units by a property of
the sender's features [S]. Dense Coverage / Sparse Refinement [F] ranks
16×16-cell patches by a learned benefit score against counterfactual
detection gain (AUC 0.785) under a strict byte cap and names cross-agent
redundancy as its open problem. JigsawComm [F] closes that problem with
a learned per-cell utility, a symmetric pairwise overlap penalty, and a
top-1-sender-per-cell rule that is identical at every agent; its
guarantees hold for the learned proxy, and it finds that a second
sender per cell degrades accuracy under max and attention fusion. All
of these decide from the sender's side; none asks whether the receiver
is about to see the cell for itself.

### 2.2 Selection by what the receiver wants now

Who2com and When2com learn a request handshake; CoSDH gates supply by
the receiver's demand mask; LIF uses a demand map from uncertainty;
HydraCollab shares rich features only where both agents are confident
[S]. Reason-to-Transmit [F] adds a neighbour-state token so a small
transformer can estimate what the receiver likely sees now, and reports
that in its synthetic environment all nine selection policies lie
within half an AP point of each other at a 10 % budget, with the
fusion operator mattering far more than selection; its lead appears
only under high occlusion, where receiver asymmetry is largest. That
finding is the argument for a demand signal that is exactly receiver
asymmetry and nothing else.

### 2.3 Receiver-conditioned relevance at object level

AutoCast [F] is the closest prior work. Every 100 ms each vehicle
broadcasts its trajectory and an object map that marks, per object and
per other vehicle j, whether the object is visible to j by ray tracing
from j's current pose and whether it is relevant to j by extrapolated-
trajectory collision; reward (1 − visible) × relevant per byte drives a
greedy max-weight schedule replicated at every vehicle over an LTE-
Direct TDMA channel at a uniform rate. AutoCast therefore has the
receiver's present visibility and its future relevance, object point
clouds as units, and channel time at one rate. It does not cast the
receiver's *future* visibility, although it holds the trajectory that
would allow it, and its reward is a collision indicator rather than a
statement about any task. Edge-assisted relevance dissemination (Wang
and Cao) and relevance-theoretic V2X (Lusvarghi et al.) keep the
collision-relevance half [S]. Update the Unseen Only [F] schedules
infrastructure broadcasts to minimize age over areas users cannot
observe, modelling the unobserving population as an INAR(1) process
whose persistence discounts areas users will soon see; it is the
statistical, downlink, age-objective counterpart of the shadow.

### 2.4 Timeliness

Age of information (Kaul et al.; Yates et al.; Kadota et al.) and its
collaborative-perception forms (Fresh2comm, TAMP, AoI-guaranteed
scheduling) decide when to send from the age of the last update [S].
Signal-aware sampling (Sun, Polyanskiy, Uysal) and age of incorrect
information add the source's change to age. None derives a deadline
from the receiver's motion; AutoCast's starvation term and 1/TTC are
the closest [F].

### 2.5 Multi-robot map sharing

Rocha et al. share the measurements of highest entropy utility against
the shared map; Aragues et al. send increments when the local map has
changed enough; Commerge exchanges a selected subset of scans [S];
event-triggered estimation sends a state when its innovation exceeds a
threshold (Trimpe and D'Andrea). These value a unit against what the
receiver *has*; the shadow values it against what the receiver *will
have*. The static-map exclusion in the shadow is their contribution,
kept.

### 2.6 Information-theoretic foundations

Coding against the receiver's side information (Wyner and Ziv 1976;
Steinberg and Merhav 2004) is the theory of sending only what the
receiver lacks; the shadow is its geometric instance with the
receiver's future observations as the side information. Submodular
selection with greedy guarantees (Nemhauser, Wolsey, Fisher 1978;
Khuller, Moss, Naor 1999) covers the scheduler.

### 2.7 Scheduling on the channel

PRISM [F] learns a distributed MAC over semantic segments shared across
users with equal value and no deadline; SchedCP selects users by CSI
[S]; 802.11ax uplink OFDMA gives every access point a centralized
scheduler with buffer status reports. The shadow schedule is a
deterministic function of shared state and runs equally at an AP or
replicated at every agent, as AutoCast's and JigsawComm's do; the AP's
only advantage is knowing per-sender rates.

---

## 3. System Model

### 3.1 Agents, map, time

Agents i ∈ 𝒜, each with pose p_i(t), a sensor model (range r_i, field
of view Φ_i, occlusion by the map), and a motion plan π_i: a planned
path for robots, a predicted trajectory for vehicles, with uncertainty
growing over a horizon H. A shared metric map M of the static
environment, discretized into cells x ∈ 𝒳 (BEV cells or voxels). Time
in intervals of T (the perception frame, 100 ms). Agents exchange poses
and plans every T on a control channel, as in AutoCast's metadata
exchange [F].

### 3.2 Self-coverage and shadow

For receiver j and future time t′ ∈ [t, t + H], the predicted pose
p̂_j(t′) and the map give by ray casting the cells j observes at t′:

    C_j(t′) = { x : x within r_j of p̂_j(t′), within Φ_j, not occluded in M }.

The **planning envelope** E_j(t′) is the set of cells whose state can
affect j's decision at t′: the stopping-distance cone around π_j, or
the planned corridor. The **approach time** of cell x for j is

    a_j(x) = inf { t′ ≥ t : x ∈ E_j(t′) },     a_j(x) = ∞ if never.

The **shadow** of j at t is

    S_j(t) = { x : a_j(x) < ∞,  x ∉ C_j(t′) for all t′ ∈ [t, a_j(x)],  x ∉ M_static }.

A cell is in the shadow exactly when j's path will make it matter
before j's own sensors reach it, and the map does not already say what
is there. Uncertainty in π_j enters as a probability P(x ∈ S_j);
Sec. 4 uses the expectation.

### 3.3 Senders and units

At t, sender i observes cells O_i(t) ⊆ C_i(t) and holds, for each, a
unit u_i(x) in whatever representation its pipeline produces (a feature
patch, a voxel block, a point tile, a box), packetized into equal-size
units of b bytes. Sender i reports the bitmap O_i(t) (|𝒳| bits, a few
hundred bytes) on the control channel; it reports nothing about
content.

### 3.4 Channel

One shared channel with per-sender PHY rate R_i(t) and a per-interval
airtime budget A after the control traffic. A unit from i costs
b / R_i(t) plus packing overhead. A frame from one sender to several
receivers costs once (multicast). Units that have not arrived by the
interval's end are missing at the receiver.

### 3.5 Receiver belief and task

Receiver j holds a belief B_j(t) over the world state W(t) in its
envelope, formed from its own observations, the map and received units,
and runs a task f_j(B_j) with loss ℓ_j(f_j(B_j), W). The method does not
know f_j or ℓ_j; Sec. 4 shows it does not need to.

---

## 4. Problem Formulation and the Sufficiency Proposition

**Problem.** Choose, each interval, the set of (sender, cell) units to
transmit under the airtime budget so as to minimize the sum over
receivers of their task losses at their decision times:

    min_π  E[ Σ_j Σ_{t′} ℓ_j( f_j(B_j^π(t′)), W(t′) ) ]   s.t.  Σ_{(i,x) ∈ π(t)} b / R_i(t) ≤ A  ∀t.

**Proposition 1 (sufficiency of the shadow).** Let x ∉ S_j(t): either
a_j(x) = ∞, or j observes x at some t′ ≤ a_j(x), or x ∈ M_static. Then
for any task f_j whose loss depends on W only through B_j, and any unit
u(x), transmitting u(x) to j at t changes E[ℓ_j] by zero.

*Proof.* If a_j(x) = ∞, no decision of j depends on x. If x ∈ M_static,
B_j already conditions on W(x). Otherwise j conditions B_j on its own
observation of x at t′ ≤ a_j(x) before any decision depends on x; a
unit received in [t, t′) alters B_j only on an interval during which no
decision depends on x. ∎

**Corollary.** An optimal policy exists that transmits only shadow
cells, for every receiver task simultaneously. The policy's
task-dependence is confined to the *order* in which shadow cells are
filled; the shadow itself is task-free.

**Shadow scheduling** fixes that order by approach time and the sender
by airtime: sort pairs (x, j) with x ∈ S_j by a_j(x), assign each to the
sender in {i : x ∈ O_i} with the smallest b / R_i, admit while the
budget holds, and let one transmission serve every j with x ∈ S_j. Under
equal unit sizes this is greedy on a weighted matroid (cells with
deadlines, EDF feasibility) and is within 1 − 1/e of the clairvoyant
order for every budget at once; with unequal sizes no fixed order is.

**The price of task-blindness.** Let π* be the loss-optimal shadow
schedule for task f_j and π_S the approach-time schedule. The regret
E[ℓ_j(π_S)] − E[ℓ_j(π*)] is the only quantity a task-specific method can
improve, and it is measurable against a task oracle on sequential data.
Reporting it per task is the paper's evidence.

---

## 5. Evaluation plan (one paragraph)

OPV2V supplies, for every receiver, the true future self-coverage from
its own next frames, hence the true shadow without any predictor; a
constant-velocity extrapolation and the dataset's own future poses give
the predicted shadow at two fidelities. Measured: shadow-prediction IoU
by horizon; the share of transmitted bytes outside the true shadow for
Where2comm, DR and JigsawComm as published; detection AP and occupancy
error for shadow scheduling against each at equal airtime on a MAC
model that charges per-sender rate and packing; the regret against each
task's oracle. The indoor robot case (planned paths, no prediction
error) is the clean deployment and the hardware check.

---

## 6. Novelty

The novelty is a change of what the transmission decision is a function
of. Every prior rule is a function of the message (its confidence,
utility, information, age) or of the receiver's present (its request,
its current view, its current unobservability). Shadow scheduling is a
function of the receiver's *future observations*, which are computable
from state the agents already exchange and which, by Proposition 1,
bound the value of communication for every task at once.

Three things exist only once that change is made:

1. a demand signal that is task-agnostic by theorem rather than by
   design, so one schedule serves detection, mapping, tracking and
   planning without a value head for any of them;
2. a deadline per cell derived from the receiver's motion (approach
   time), replacing frame clocks and age counters;
3. a measurement no prior work can make: the fraction of the field's
   transmitted bytes that are provably worthless to their receivers,
   computed from the receivers' own future frames.

What it is not: not a better utility estimator (JigsawComm), not
current-visibility relevance (AutoCast), not statistical
unobservability (Update the Unseen Only), not a request protocol
(Who2com), not a compression scheme (DR, InfoCom). Each of those can
run *inside* the shadow to order its cells; none of them can say which
cells are outside it.
