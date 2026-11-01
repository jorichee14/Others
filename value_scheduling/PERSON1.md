# Person 1: brief

## Goal

Show that a robot team can share only the observations a receiver
cannot make for itself, created when something changes and delivered
before the receiver needs them, and still make the same decisions as a
team that broadcasts everything on a timer, with an order of magnitude
fewer bytes and with the time-critical messages arriving on time
because the needless ones are gone.

The one number you are after: **the fraction of today's traffic that
was never needed**, and the decision-correctness and deadline-hit rates
that go with removing it.

## The problem in one scene

A mast camera sees a corridor. A robot drives toward the corner that
hides it at 1 m/s. Today the mast broadcasts a frame every 500 ms to
everyone, whether anything moved or not, whether any robot is coming or
not; the robot acts on whatever arrived last. Under load, the frame
that mattered, the one showing a person, arrives after the robot has
committed to its speed. Your system: the mast sends when the corridor's
state changes; the server keeps that one observation with how long it
stays valid; the robot gets it, from that copy, before its stopping
distance reaches the corner, and keeps it with its velocity until it
expires.

## What you build, and nothing else

1. **Mast node** (Raspberry Pi, camera at 10 fps). Background
   subtraction against an empty-corridor reference; per frame: an
   occupied bit, a bounding box, image-plane velocity, timestamp.
   Three publishers: JPEG frame, box + velocity, one bit. A trigger
   switch: periodic (2 Hz, 10 Hz) or on change (bit flips, or box
   moves > 0.5 m).
2. **Robot node** (Jetson). Reads its scripted path and the static map;
   computes approach time per cell ahead (distance along the path over
   speed, minus reaction time). Subscribes to the mast directly or to
   the server. Memory: last observation per cell with timestamp and
   velocity; extrapolates to now; expires at TTL and reports a miss.
   Speed decision: crawl if the corridor cell is occupied or missed,
   full speed if certified clear. (Person 3 replaces this decision with
   the AoII rule later; leave the hook.)
3. **Server node**. Relay mode: forwards. Store mode: keeps the latest
   observation per cell with source, timestamp, velocity, TTL; dedups
   by content hash on upload; serves to a robot when its approach time
   for that cell falls under a threshold, or on request.
4. **λ(x)**, the per-cell change rate. From survey_1 and coop_2: for
   cells visited more than once, the fraction of revisits where
   occupancy differed, over the time between visits. One table. TTL(x)
   = tolerance / λ(x), clipped.
5. **Logger and scorer**. JSONL on the chrony clock: every create,
   send, receive, store, serve, expire, hit, miss, with bytes and
   cell; the robot's speed command and reference pose per frame;
   person-presence per trial from the trial script. The scorer joins
   these into: observation-before-approach rate, decision correctness,
   bytes up and down, latency mast-frame-to-decision, deadline hits of
   pose and control, and the waste fraction of the timer baseline.
   Include person 3's field from day one: the robot's held observation
   time and λ for each cell ahead.

Do not build: anything at the MAC, slotting, encoders, detectors
beyond background subtraction, a digital twin, a tile cache (the
static map is preloaded on every node).

## Where to start, week by week

**Week 1.** Build 1, 2 (relay only), 3 (relay mode), 5. Replay a
recorded corridor sequence through the three nodes on the bench so the
logs and the scorer work before any driving. Compute λ(x) from the
recordings. Deliverable by Friday: a bench run that produces the
scorer's table for one fake trial.

**Week 2. S0, the blind corner.** One session, about two hours. Grid:
representation (3) × trigger (periodic 2 Hz, on change) × route
(direct, via server relay), person in half the trials, each cell with
and without the second robot streaming LiDAR to the server as load. 20
trials per cell. Result: bytes and latency per representation against
decision correctness; the first waste number (timer traffic that
carried no change).

**Week 3. The store; S1.** Store mode on the server. Two robots
approaching the same corner at different times; the mast uploads once;
the server serves each robot at its approach time. Compare with
on-change broadcast to both. **Stop-rule check:** if on-change
broadcast without the store already gives ≥ 90 % of the byte saving
and the same deadline hits, the store is not earning its place; write
up the event-trigger result and stop here.

**Week 4. S2, the changed building.** Move a cabinet into a corridor
(and, separately, close a door) after the map was made. The robot
drives the route; a mast or the other robot sees the change; the store
invalidates the cell; the robot refreshes it before approach. Compare
with periodic full refresh and no refresh. Result: refresh bytes, the
fraction of refreshes that were unnecessary, and whether the robot
replans before its stopping distance.

**Week 5.** Write-up of S0–S2; hand the nodes and logs to the joint
session (week 6, with persons 2 and 3).

## Baselines you run every time

Timer broadcast at 2 Hz and 10 Hz (today's behaviour); on-change
broadcast with no store; request-on-demand with no store.

## What makes each week a result

- Week 2: bytes and correctness per representation; the one-bit
  certificate against the JPEG.
- Week 3: write once, serve many, under load; or the stop-rule
  finding, which is also a result.
- Week 4: validity from the building's own change rate against a timer.

## Prior art you must know, because reviewers will

Trimpe and D'Andrea 2014 (event-triggered estimation: send on
innovation; silence is information). ETSI TS 103 324 (the Collective
Perception Message: generation on dynamics, redundancy mitigation
rules). Update the Unseen Only and AutoCast (both in `refs/`): age over
unseen areas; relevance by current visibility. One survey of mobility-
aware proactive caching at roadside units. Your difference: validity
from the measured change rate of the place, delivery timed to the
receiver's approach, one stored copy serving many, on a real contended
link.

## What you need on day one

The corner and mast position; a Jetson and a Pi you can deploy to; the
ROS 2 distro and camera driver on the Pi; the robot's speed topic and
limits; the static map copied to both nodes; read access to survey_1
and coop_2 for λ(x); person 3's metric page for the log fields.
