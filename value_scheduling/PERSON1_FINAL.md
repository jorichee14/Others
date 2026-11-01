# Person 1, final

This supersedes `PERSON1.md`, `HANDOFF_PERSON1.md` §4 (experiments) and
the week plan in `PLANS_123.md`. The technical specification of the
nodes, messages, λ(x), logging and scoring in `HANDOFF_PERSON1.md`
§1–3, §4b, §5–§10 still applies; what changes is the problem, the
experiments after week 2, and the claim.

## 1. The problem, in one sentence

Which stored observation goes to which robot at which moment, knowing
when each robot will need it and how fast it spoils, so that under a
link that cannot carry everything the fewest decisions are made on
stale or missing facts.

## 2. The problem, precisely

- Robots j with fixed scripted paths and fixed speeds; hence a known
  need-time a_j(x) for every cell x on the path (approach time,
  §2 of the handoff).
- Observations of x with a spoil time TTL(x) = τ_tol / λ(x), with
  λ(x) the measured per-cell change rate and τ_tol a tolerance given
  from outside (person 3), swept as a parameter.
- A fetch of x for j is useful only inside its window
  W_j(x) = [a_j(x) − TTL(x), a_j(x)]: earlier, it spoils before use;
  later, it misses the decision.
- Fetches cost bytes b(repr) on a shared downlink of capacity C
  (throttled with `tc`, or measured); several robots' windows overlap;
  one fetch can serve every robot whose window for x is open (multicast).
- Objective: minimize the λ-weighted count of (j, x) pairs whose
  need-time passes without a valid observation held, subject to the
  capacity. Weighted by λ because a miss on a volatile cell is a miss
  on something that may have changed; a miss on a wall is harmless.

This is scheduling with release times, deadlines and a shared
resource. Feasible instances: EDF over the windows is optimal. Over-
loaded instances: NP-hard in general; the structure here (windows
nested by λ, multicast sharing, representation as a size choice) is
what the policy exploits. Fixed motion is an assumption, stated; it is
what makes need-times inputs and the offline optimum computable.

## 3. Three policies and a bound

| Policy | Rule | Expected failure |
|---|---|---|
| Fetch-on-change | push every change to every robot at once | fetches too early for volatile cells (spoil before use); floods the link |
| Fetch-on-request | the robot asks at a_j(x) − δ | too late under load; the answer misses the decision |
| **Window scheduling (his)** | EDF over open windows, λ-weighted when overloaded, multicast merge, representation downgrade (jpeg → box → bit) when the window would otherwise be missed | — |
| Offline optimum | computed after the run from the known paths and the logged link capacity (an ILP or DP over the windows) | the bound |

The method is the third row. Its proof is the gap to the fourth row
and the margin over the first two.

## 4. Experiments

**Week 1, bench.** Nodes and logger as in the handoff; λ(x) from the
recordings; bench replay through the logged channel. Exit: the scorer
table on 20 synthetic trials.

**Week 2, S0 blind corner.** One mast, one robot, the instrument
check: representation × trigger × load. Produces the waste fraction of
timer traffic and the bytes/latency per representation. No scheduling
claim here; the link is not short.

**Weeks 3–4, S1 the overloaded store (the main experiment).** Two
robots on long scripted routes covering the site; three masts on
corridors with different λ; the robots' own LiDAR observations also
written to the store; downlink throttled with `tc` to the level where
fetch-on-change misses ~30 % of windows (found in a pilot); JPEG as
the base representation so sizes matter, box and bit as downgrades.
Routes designed so windows overlap: both robots pass the same
corridors a few seconds apart. Four runs per throttle level (three
levels), with τ_tol ∈ {1, 2, 4} s: the four policies above. Scored:
λ-weighted misses at need-time, plain misses, bytes, fetch lead time
distribution, multicast share, downgrade share. Person at a corridor
in some runs so that misses have consequences in the robot's decision
log, but the score is the miss count, not the decision.

**Week 5, S2 the changed building.** Cabinet and door as in the
handoff §4b, run under window scheduling: does invalidation plus a
scheduled refresh reach the robot before its stopping distance when
the link is throttled. Short; one session.

**Week 6, joint session.** Nodes and scorer handed over; his policy
is one of the switches.

## 5. Claim

Under a link that cannot carry every observation, scheduling fetches
inside their spoil-to-need windows, with λ-weighting, multicast and
representation downgrade, reduces λ-weighted misses at need-time by a
large factor against fetch-on-change and fetch-on-request, and comes
within a stated factor of the offline optimum, on real robots over a
real throttled Wi-Fi link. The waste fraction of timer broadcast,
measured in S0, is the motivation.

## 6. Stop rules

- After the S1 pilot: if fetch-on-change misses < 10 % of windows at
  any throttle the AP can reach, the link is not the bottleneck on this
  rig; report the waste number from S0 and the window formulation as a
  position, and stop the scheduling claim.
- After S1: if his policy is within noise of plain EDF without
  λ-weighting, multicast or downgrade, the contribution is the
  formulation and the bound, not the policy; say so.

## 7. What he does not own, and why

- **τ_tol** (how stale is too stale): set by person 3 from decision
  outcomes; he sweeps it, because owning it would let the scheduler
  loosen its own constraint.
- **Motion** (slow, reroute, fetch early where the link is good):
  person 3; he holds motion fixed, because a scheduler that can brake
  the robot is no longer measuring the link.
- **Slotting and compute-before-send**: person 2.
- **Codecs**: a size knob (jpeg / box / bit), not a thing he builds.

He states both assumptions in his paper with one sensitivity figure
over τ_tol and a pointer to person 3 for the relaxed-motion case.

## 8. Reading

Trimpe and D'Andrea 2014 (event-triggered estimation); ETSI TS 103 324
(CPM generation and redundancy rules); Update the Unseen Only and
AutoCast (refs/); a survey of mobility-aware proactive caching at RSUs;
Fofack et al. on TTL caches; one text on scheduling with release times
and deadlines (Pinedo, ch. 3–4); Belady 1966.

## 9. Day one

The corner and corridors; a Jetson and a Pi to deploy to; ROS 2 distro
and Pi camera driver; the Scout's speed topic and a waypoint follower;
the static map on both nodes; processed survey_1 and coop_2 for λ(x);
person 3's metric page (τ_tol and the belief field); `tc` on the server
or the AP side for the throttle; the sniffer Pi in monitor mode.
