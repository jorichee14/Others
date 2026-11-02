# Thesis: decision-oriented communication for collaborative robots

"Task-agnostic" and "task-oriented" stop contradicting each other once
the task is the robot's *next action* rather than a perception metric.
Semantic and task-oriented communication fix the task as detection or
segmentation and optimize bits for its loss; that is task-specific by
construction. A robot does not need detections; it needs a belief that
is correct where its next decision depends on it. That requirement is
the same whether the belief is built from boxes, features, points or a
mast's one bit, so it is agnostic to the perception task, and it is
oriented to the only task the robot actually has, acting.

**The thesis.** Communication among collaborating robots should
maintain the correctness of each robot's belief over the region its
next decision depends on, and nothing else; the measure of that
correctness is the age of incorrect information at the moment of the
decision; the robot's motion is itself a lever on it, because where
the robot goes decides what it can sense and what it can receive.

Three parts, one quantity:

1. **The quantity.** AoII_j(x, a_j(x)): for robot j and cell x, the time
   its belief about x has been wrong, evaluated at the approach time
   a_j(x) when its path makes x matter. Computable from the per-cell
   change rate λ(x), the last observation j holds (its own or received),
   and its path. Task-agnostic: it does not mention what x contains or
   which detector sees it. Task-oriented: it is zero outside the
   decision envelope and scored at the decision.

2. **The communication rule.** Send to j only observations of cells in
   its envelope whose AoII at approach would otherwise exceed tolerance,
   on change rather than on a timer, from one stored copy, before
   approach time (person 1's system is this rule's implementation;
   person 2's slotting is what makes "before approach time" hold on a
   shared link).

3. **The motion rule.** Choose speed and path so that AoII at approach
   stays under tolerance given what the robot can sense itself and what
   the link at each place lets it receive (the survey radio map). Slow
   where the belief cannot be kept correct; route past a mast or a good
   link before entering a shadow; go full speed where a valid
   certificate exists. Communication-aware robotics as a consequence of
   the same quantity, not a separate module.

**What is new, against each neighbour.**
- Semantic / task-oriented communication: the task is a perception
  loss; here it is the decision, so the method is indifferent to the
  representation and holds across perception tasks.
- AoI scheduling: age without content; here age of *incorrectness*,
  with the environment's measured change rate supplying the content
  model, scored at the decision rather than averaged over time.
- Communication-aware motion planning (Mostofi, Fink–Kumar): motion
  for connectivity or throughput; here motion for belief correctness
  at the decision, with the link as a means.
- AutoCast / Update the Unseen Only: relevance now and unseen areas by
  age; here the receiver's future and the belief's correctness, in one
  scalar.

**Claims to prove.**
- C1 AoII at approach predicts task success at blind corners; AoI and
  bytes do not (S0–S3 logs).
- C2 A robot driving on AoII completes missions faster at equal safety
  than one driving on AoI or on none, with link-aware prefetch adding
  a further gain (the speed switch in S0 and the joint session).
- C3 Expected AoII at approach has a closed form in λ, the delivery
  delay and the trigger threshold, from which the TTL, the speed rule
  and the no-communication-needed condition follow.
- C4 The same rule, unchanged, works when the perception behind the
  belief is a detector, an occupancy map or a one-bit certificate
  (task-agnosticism shown, not asserted).

**Where the lead works.** The metric page and log contract; the speed
and path rule in the robot node; the analysis of both persons'
sessions; the theory. Persons 1 and 2 supply the communication rule's
two halves and their measurements.

**Title to aim at.** *Decision-Oriented Communication: Keeping
Collaborative Robots' Beliefs Correct Where Their Next Action Depends
on Them.*
