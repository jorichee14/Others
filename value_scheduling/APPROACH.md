# Shadow scheduling

**The whole method in one sentence.** Each receiver's *shadow* is the set
of space-time cells it will not observe for itself before its path
reaches them; the team's airtime goes to filling shadows, nearest-
approach first, one sender per cell, and nothing else is ever sent.

That is the method. There is no value head, no task loss, no learned
module, no per-unit metadata, no deadline model, no redundancy model and
no priority ladder, because each of those is what the shadow already is.

## The quantity

Receiver j has a pose, a planned (robot) or predicted (vehicle) path
over a horizon H, and a sensor model (range, field of view). From these
and the shared map, ray-cast forward: the set of cells j will observe
itself in [t, t + H] is its **self-coverage** C_j. Its **shadow** is

    S_j = { (x, t′) : x within j's planning envelope at t′,  x ∉ C_j(t′),  x not static-and-already-mapped }.

Each shadow cell carries one number: the time a_j(x) at which j's path
first makes x matter (enters its stopping distance or planning
corridor). Cells j will see for itself before a_j(x) are not in the
shadow by construction. Cells it never approaches are not in it either.

Everything the earlier drafts listed as separate features is a face of
this one set:

| Earlier feature | What it is here |
|---|---|
| "value conditional on the receiver" | x ∈ S_j or not; binary, geometric |
| "deadline from the receiver's decision" | a_j(x), the ordering key |
| "redundancy across senders" | a shadow cell is filled once; a second sender's observation of it has no shadow left to fill |
| "receiver memory as side information" | static mapped cells are excluded from the shadow |
| "future visibility, not current" | C_j is cast along the path, not from the current pose |
| "task value" | none; see the proposition |

## The scheduler

At each interval the AP (or any node that holds everyone's path and
the map; the schedule is a deterministic function of shared state) does:

1. cast C_j and S_j for every receiver j;
2. for every shadow cell, find the senders that observe it now (each
   sender reports a bitmap of observed cells, a few hundred bytes);
3. sort (cell, receiver) pairs by a_j(x) ascending;
4. walk the list, assigning each cell to the sender with the lowest
   airtime per cell (its rate, its packing), until the interval's
   airtime is spent; a cell already assigned for one receiver serves
   every receiver that shares it (multicast on Wi-Fi is one frame).

What is transmitted for a cell is whatever representation the sender
holds for it, in equal-size units: a feature patch, a voxel block, a
point tile, a box. The scheduler does not look inside. That is what
makes it task-agnostic: the same schedule feeds a detector, a mapper, a
tracker or a planner, because each of them depends on the world only in
cells the receiver cannot see for itself.

## The proposition that makes it legitimate

For any receiver task whose loss depends on the world state only
through the receiver's belief, and any unit about cell x that j will
observe itself before a_j(x), the expected loss reduction from receiving
the unit is zero. Hence the optimal schedule can be restricted to shadow
cells without loss, for every task at once.

Proof sketch: by a_j(x) the receiver's belief about x conditions on its
own observation of x; a unit received earlier about x changes the belief
only during [arrival, self-observation), when by definition no decision
depends on x. ∎

The converse is not claimed: a shadow cell may be worthless to a given
task (an empty corridor). The method fills it anyway, in approach order,
and the cost of that is the only thing a task-specific method could win
back. Measuring that cost is the experiment.

## Why it is new and why it is not a merge

- AutoCast prices an object to j by *current* visibility; Update the
  Unseen Only by the statistical persistence of unobservability;
  JigsawComm assigns one sender per cell by learned utility. None of
  them has the receiver's own future observations as the demand signal.
  Casting the future shadow is a single operation that replaces all
  three of their mechanisms, and it is possible because paths are
  planned or predicted and the map is shared, which AutoCast had and
  did not use for visibility.
- The approach-time ordering is a deadline derived from motion, not
  from a frame clock or an age counter; no scheduler in the literature
  has it.
- The claim is a sufficiency result (the shadow is a task-agnostic
  sufficient demand), not a better accuracy number. The paper's
  evidence is how little is lost against task-specific oracles across
  tasks, and how much is saved in airtime.

## What would kill it

- If, on sequential data with a detector and with a mapper, a task-
  specific oracle beats shadow filling by a large margin at equal
  airtime on *both* tasks, the shadow is too coarse a demand and the
  method is only a cheap baseline.
- If path prediction for vehicles is poor beyond 1 s, the shadow is
  wrong where it matters; robots with planned paths are the clean
  case, vehicles need the horizon chosen where prediction holds.
- If most of the shadow is empty space, airtime is spent certifying
  nothing. The fix is not a feature: a sender whose observation of a
  shadow cell is "free" sends one bit, which is the free-space
  certificate, and it is the same mechanism with a one-bit unit.

## The test that settles it first

On OPV2V (future frames give the true self-coverage): for each receiver
and frame, compute the true shadow from the dataset and the predicted
shadow from a constant-velocity path. Report (a) shadow prediction
IoU over horizons 0.5, 1, 2 s; (b) the fraction of transmitted units
in DR, JigsawComm and Where2comm that fall *outside* the true shadow,
i.e. bytes those methods spend on what the receiver sees anyway; (c)
detection AP and, separately, occupancy-map error, for shadow filling
against each at equal airtime. (b) alone is a result: it says how much
of the field's traffic is provably worthless to its receiver.

## When the map is dynamic

The shadow definition above uses the static map twice: to exclude cells
the receiver already knows, and to cast occlusions. Both break when
doors close, furniture moves and people walk. The repair is not a new
feature; it is replacing a binary exclusion with the scalar it was a
special case of.

**The scalar.** What communication can change at cell x for receiver j
is j's uncertainty about x at the moment x matters, a_j(x), given only
what j will have observed for itself by then:

    w_j(x) = H[ W(x, a_j(x)) | j's own observations up to a_j(x) ].

A static wall in the map: w = 0 (the old exclusion). A cell j will see
for itself before a_j(x): w = 0 (the old shadow test). A cell no one has
observed: w = prior entropy. A cell observed τ seconds ago whose content
changes at rate λ(x): w grows with λ(x)·τ. The shadow is the support of
w_j, and the schedule fills cells in order of a_j(x) weighted by w_j(x).
Nothing else in the scheduler changes.

**Where λ(x) comes from.** From the observations the team already has:
how often the occupancy of x changed between successive observations of
it, kept as one number per cell by whoever maintains the shared map (the
twin's change log, or simply a count). Corridors get high λ, walls zero,
doors in between. It is a statistic of the environment, not of any
task.

**Timing falls out.** A unit sent now about x is worth, at a_j(x), only
what survives the gap: w_j(x) − H[W(x, a_j) | W(x, t)]. For a volatile
cell far in the future that is near zero, so the scheduler defers it;
for a static or near cell it is the full weight, so it sends now. This
is just-in-time delivery with no deadline model: the same scalar, read
at two times.

**Occlusion through a changing map.** Self-coverage C_j is cast through
the receiver's *current belief* of the map, not the recorded one. Where
that belief is uncertain (a doorway with high λ), the cast is
conservative: assume the occluder may be there, so C_j shrinks and the
shadow grows. The cost of being wrong is a few extra units; the cost of
the other error is a receiver that was promised a view it never gets.

**The proposition survives unchanged.** It conditions on the receiver's
own observation of x before any decision depends on x; dynamics make
that observation more valuable relative to a stale unit, not less. What
dynamics change is the *weight* and the *timing* inside the shadow,
which is exactly where the corollary said task- and world-dependence
would live.

**Moving objects** are the extreme of high λ: a person in a corridor
cell will not be there at a_j(x). The unit the sender holds for that
cell is then whatever its pipeline produces, and if that is a box with
velocity, the receiver can propagate it; the scheduler still does not
look inside. Whether the representation can be propagated is the
sender's codec problem and is out of scope, deliberately.
