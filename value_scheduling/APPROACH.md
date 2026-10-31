# The approach: need-before-see scheduling

One sentence: **a message unit is worth, to a given receiver, the task
loss it removes at the moment that receiver's decision depends on it,
beyond what the receiver will have perceived by itself by then; the
access point computes that value for every (unit, receiver) pair from
the poses and plans it already holds, prices it in airtime, and grants
the units with the highest value per airtime in the one round the uplink
scheduler already runs.**

This is not a plan. It is the mechanism, the quantities it introduces,
why it is a change in kind, and what would show it wrong.

---

## 1. The quantity: value as the receiver's future counterfactual

Let unit u (a DR fine patch, a box with velocity, a centroid) be held by
sender i at time t. Let receiver j have a planned or predicted motion
over a horizon H, a decision time τ_j(x) for each location x (the time
its path makes x decision-relevant: entry into its collision cone or
planned corridor, minus reaction time), and its own observations O_j(t′)
for t′ ∈ [t, t + H] (in a dataset these are its future frames; online
they are predicted by ray-casting its planned path against the map and
the predicted motion of known objects).

The value of u to j is

    V_j(u) = E[ ℓ_j( τ_j ; O_j[t, τ_j] )  −  ℓ_j( τ_j ; O_j[t, τ_j] ∪ {u at t} ) ],

the drop in j's decision-time task loss from having u now, measured
against the baseline of everything j will have seen on its own by τ_j.
Three cases make the behaviour clear:

- j will see the content of u itself before τ_j: V_j(u) ≈ 0, whatever
  the sender's confidence;
- j will not see it and τ_j is near: V_j(u) is the full detection gain,
  with deadline τ_j;
- τ_j is beyond H (j never goes there): V_j(u) = 0.

The deadline of u for j is d_j(u) = τ_j(x_u), not the frame.

**Why this is a change in kind.** Every feature-level method prices u by
a property of the sender's data (confidence, utility, information); the
object-level relevance line prices by a binary "will j see it" with no
task loss. V_j(u) is a property of the triple (unit, receiver, horizon)
with the receiver's own future as the baseline. The number being
estimated is different, its oracle is different, and its supervision is
different.

## 2. The oracle, and the training target it gives

On any sequential dataset the counterfactual is computable: fuse j's
own frames up to τ_j with and without u, score ℓ_j at τ_j. This is the
clairvoyant bound for "what to send to j" and it has never been
reported for feature units (DR's counterfactual is over the coarse layer
at time t; AutoCast's is object visibility). It also gives a supervised
target: train a value head on (sender features, receiver pose and plan,
horizon) → V_j(u), the way DR trains β on counterfactual gain over
coarse. The head learns "will j see this, and does it matter when", not
"is this patch informative".

## 3. The locus and the resource: the AP, in airtime

Only a center has, at every trigger interval, every agent's pose and
plan (from the pose flow), every sender's reported ranking (DR's β list,
which rides on the buffer status report), and every sender's PHY rate.
The AP therefore computes V_j(u) for all pairs, discounts a unit's value
by what it has already granted this interval (a cell granted from one
sender has value ≈ 0 from another *for the same receiver*; for a
different receiver it may not), prices u at airtime b_u / R_i plus
packing overhead, and runs a cost-ratio greedy under the airtime left
after the protected base-layer window, subject to each unit's deadline
d_j(u). Monotone submodular surrogate, knapsack budget, deadline
feasibility as EDF: greedy with a known bound, microseconds for tens of
agents. One round: the value report rides on the BSR, the grant on the
trigger frame. No request round, no utility-map exchange.

Redundancy (G4), granularity (G6) and the receiver's memory (G9) fall
out of the same quantity: redundancy is V_j(u) after a grant to the same
j; granularity is a centroid and a patch of the same object competing
at their own V_j and airtime; memory is the horizon-zero case of O_j.

## 4. The metric that goes with it

Frame-level AP cannot see V_j: at the frame boundary everything matters
now. The task metric must be scored at decision time:

    need-weighted timely detection = share of (object, receiver) pairs with τ_j < ∞
    that the receiver's fused perception has detected before τ_j,
    weighted by the loss of missing them,

plus the airtime used and the deadline misses of the time-critical
flows. AutoCast's "visibility into safety-critical objects" is the
object-level ancestor of this metric. Reported against the clairvoyant
bound of section 2 as a regret, so that a shortfall is attributed to
value estimation, to allocation, or to the channel.

## 5. Why it is cutting edge and not a merge

- It introduces a quantity (V_j over the receiver's future), an oracle
  (the receiver-future counterfactual), a training target, a deadline
  (τ_j from decisions), a cost (airtime at the AP) and a metric
  (need-weighted timely detection with regret to clairvoyant) that do
  not exist in the literature, each of which can be computed on today's
  datasets.
- It makes three open problems the same problem: DR's cross-agent
  redundancy, AutoCast's object-only relevance, and the AoI line's
  content-blind timeliness are all V_j(u) with different restrictions.
- It is deployable in the control loop that exists: 802.11ax uplink
  scheduling (trigger frames, buffer status reports), with grants as the
  emulation on APs that expose no scheduler.

## 6. What would show it wrong, and what it costs

- If, on OPV2V or V2V4Real, the receiver-future counterfactual oracle
  does not beat the sender-side oracle by a clear margin at equal
  airtime, the receiver's future is not where the value is and the idea
  reduces to AP-side scheduling of DR patches (still a result, a smaller
  one).
- Reason-to-Transmit's finding, read in full: on its synthetic
  environment all selection policies sit within ~0.5 AP and the oracle
  barely leads, so "what to send" can matter far less than the fusion
  operator. The approach must be shown where receiver asymmetry is high
  (occlusion, distance, blind corners) and on a channel that does not
  deliver everything, with fusion held fixed and the oracle gap reported
  first; where the oracle gap is small, so is everything built on it.
- JigsawComm, read in full, already removes cross-agent redundancy at
  feature level with a top-1-per-cell rule that is identical at every
  agent and O(1) in N. The AP is not new as a place to run such a rule;
  what it adds is per-sender airtime and per-receiver geometry over a
  horizon. The redundancy discount here must be per receiver and over
  time, or it is JigsawComm's.
- If the learned V_j head's AUC against the oracle is far below DR's
  0.785 for β, the quantity is right but not estimable from the inputs
  available online; the oracle gap says by how much.
- Vehicles need predicted trajectories for τ_j and O_j; robots have
  planned paths. The error of that prediction enters V_j directly, and
  the horizon H must be chosen where prediction is still good (1–2 s on
  roads, longer in corridors).
- The metric is unfamiliar; the paper must report frame AP alongside it
  and show where the two disagree.
- Compute: the counterfactual oracle is one fusion pass per (unit,
  receiver, horizon) on frozen checkpoints; feasible offline at a few
  hundred frames, not for every frame of every sweep.
