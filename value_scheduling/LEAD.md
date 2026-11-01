# Where the lead sits: AoI, AoII and the closed loop

Person 1 removes messages; person 2 makes them arrive. Neither owns
the quantity that says whether the robot's information was *good
enough when it acted*, and neither closes the loop from information
back to motion. That is the lead's layer, and it is exactly where AoI,
AoII and communication-aware robotics live.

## 1. The quantities, defined on this rig

For receiver j and map cell x at time t:

- **AoI_j(x, t)** = t − (time of the freshest observation of x that j
  holds). Content-blind. A wall has an AoI of hours and it does not
  matter.
- **AoII_j(x, t)** = time since j's held belief about x stopped being
  true. With a change rate λ(x) per cell, the expected AoII grows at
  rate λ(x) after the last observation; for a wall λ ≈ 0 and AoII stays
  zero however old the observation; for a corridor λ is high and AoII
  grows in seconds. AoII is AoI weighted by the probability that the
  world moved, and λ(x) from the recordings is what makes it
  computable here.
- **AoII at approach time**, AoII_j(x, a_j(x)): the value of the above
  at the moment j's path makes x matter. This is the metric. Task
  success at a corner should be a function of it and of nothing else
  about the communication; that is the hypothesis to test.

Both plans feed it. Person 1's TTL is a threshold on expected AoII;
person 2's delay is AoI added on the way. The lead reads both logs and
computes the same quantity from each.

## 2. Three results the lead owns

**R1. AoII, not AoI, predicts task success.** From S0–S3 logs: per
corner approach, AoI and AoII of the corridor cell at approach time,
against the decision outcome. Expected: AoI is a poor predictor (stale
observations of an empty corridor are fine), AoII is a sharp one. This
settles which metric the two plans should be optimizing and is the
first thing to report; the Unseen Only and Fresh2comm line optimizes
AoI, and this says why that is the wrong target for perception.

**R2. The closed loop: motion from information.** Communication-aware
robotics proper. The robot sets its speed from the AoII of the cells
ahead: full speed where AoII at approach is below tolerance (a valid
certificate or a fresh observation), slow where it exceeds it, and,
using the survey radio map, it knows *where* its link will be good
enough to refresh before entering a shadow, so it can fetch early or
route past a mast. Three policies to compare: speed from no
information (always crawl at corners), speed from AoI, speed from
AoII with link-aware prefetch. Measured: mission time at a fixed
safety outcome. This is the result the robotics audience wants and
the one that makes the whole programme a robotics contribution rather
than a networking one.

**R3. The theory that links them.** AoII under λ(x) with on-change
sending and delivery before approach time: the expected AoII at
approach as a function of λ, the delivery delay distribution (person
2's measurement) and the trigger threshold (person 1's). It gives the
TTL rule from first principles (TTL ∝ 1/λ at a chosen AoII
tolerance), the speed rule (stopping distance such that AoII at
approach stays under tolerance), and the condition under which
communication is unnecessary (the receiver's own sensors keep AoII
under tolerance). Short, clean, and it is the lead's own.

## 3. What the lead needs from the others

- From person 1's log: per (receiver, cell): last-observation time,
  source, TTL, every serve and expire; the λ(x) table.
- From person 2's log: per message: send stamp, receive stamp,
  slot, size; the delivery-delay distribution per N and size.
- From both: the robot's pose from the reference and its speed
  command, per frame.

One extra field in the shared log schema from day one: the robot's
belief per cell ahead (held observation time and λ), so AoII can be
computed offline without re-running anything.

## 4. The lead's own experiment

Extend S0 by one switch: the robot's speed controller takes AoII of
the corridor cell as input (R2's three policies). Everything else is
person 1's S0. Then the same switch in S3. No new hardware, no new
nodes beyond a speed rule in the robot node.

## 5. Reading, in order

Maatouk, Kriouile, Assaad, Ephremides, *Age of Incorrect
Information* (ToN 2020); Sun, Polyanskiy, Uysal (T-IT 2020,
signal-aware sampling); Kadota et al. (ToN 2018); Update the Unseen
Only (refs/, AoI over unseen areas); Fresh2comm; Mostofi's
communication-aware motion planning line and Fink–Kumar on
communication-aware multi-robot control; the MEAN / networked
embodied intelligence surveys already cited in SCooP-iT
(`../scoop_testbed_iotj/DIRECTION.md`).

## 6. Position in one line

Person 1 says less, person 2 takes turns, the lead decides what
"fresh enough" means and lets the robot drive on it.
