# Testing the cache on the real robot, task by task

Each test is a task the robot has to complete, a cache behaviour that
decides whether it does, a pass metric scored against ground truth the
testbed already produces, and a baseline run on the same route the
same day. Rig: one or two Scout Mini robots (Jetson Orin NX, Ouster,
ZED), three Raspberry Pi masts with cameras, the Asus AP, the server,
chrony on everything, the reference map of site 1, labels in the map
frame where needed. Every run logs, on one clock: cache hits/misses
and evictions, bytes up and down per second, tile and observation
arrival times, robot pose from the localization and from the
reference, and the task outcome.

## Task 1. Drive the route on cached tiles (work package A)

**Task.** Follow the survey route at 1 m/s, localizing against map
tiles served from the server; the robot stops if localization is lost
(map→odom not updated for 2 s).

**Cache under test.** Tile cache on the Jetson with a memory cap small
enough that the whole site does not fit (set from the Pi's limit so
the same policy runs on both classes). Policies: LRU, LFU, path-
prefetch + MIN, each with prefetch depth {5, 10, 20 m}.

**Pass.** No localization stops over the route; tile bytes downloaded
≤ 60 % of LRU's at equal memory; pose error against the reference
trajectory within the testbed's own bound (3.1 cm median) for the
whole run.

**Baseline.** LRU at the same memory; and "all tiles preloaded"
(unbounded memory) as the ceiling.

**Variation that makes it a result.** Deviate from the plan on purpose
(an obstacle forces a detour of 10 m) and record the hit-rate drop and
whether the robot stops. That number, hit rate versus plan deviation,
is the paper's figure for A.

## Task 2. Blind-corner approach on served observations (work package C)

**Task.** The robot approaches a corner behind which a mast sees a
person (or a placed obstacle). The robot must slow before the corner
if the corridor is occupied and may keep speed if it is clear. Scored:
did the robot have the mast's observation of the corridor cell before
its approach time (stopping distance at current speed), and did it
choose the right speed.

**Cache under test.** The edge cache at the server: the mast uploads its
observation once; the robot pulls by region along its path; the server
serves at the robot's approach time on the downlink. Compared with the
mast re-sending to every robot on the uplink.

**Pass.** ≥ 90 % of approaches have the observation on the robot before
approach time; correct speed decision in ≥ 90 % of trials; uplink
airtime for the masts lower than the re-send baseline (sniffer).

**Baseline.** Masts broadcast on a timer at 2 Hz (today's behaviour);
and no cooperation (the robot slows at every corner).

**Trials.** 20 approaches per condition, person present in half, two
corners of the site, both robots as the approaching robot.

## Task 3. A changed building (work package B)

**Task.** Between the mapping pass and the test, move a cabinet into a
corridor or close a door on the route. The robot drives the route; the
planner uses the cached tiles; a stale tile makes it plan through the
cabinet or into the door and it must stop and replan late.

**Cache under test.** TTL coherence from λ(x): corridor cells have a
short TTL and are refreshed from the server (which saw the change via a
mast or the other robot) before approach time; wall cells are never
refreshed. Compared with no TTL (tiles valid until explicitly
invalidated) and with periodic refresh of everything.

**Pass.** The robot replans before the stopping distance in every trial
with the TTL policy; refresh bytes below 25 % of the periodic-refresh
baseline; zero refreshes of cells that did not change (walls).

**Baseline.** Periodic full refresh at 0.1 Hz; no refresh.

**Trials.** 10 drives per condition, change placed at two different
cells with different λ (a corridor and a doorway).

## Task 4. Five agents on one AP (work packages C and the contention rig)

**Task.** Both robots drive, the three masts send observations; all
agents' pose and control traffic must meet their deadlines while the
observation traffic flows. Scored on the time-critical flows (control
and pose deadline hits) and on task 2's delivery-before-approach-time
for the observations.

**Cache under test.** With the edge cache: senders upload once with
hash-first dedup, receivers pull scheduled downlink. Without: every
sender sends to every receiver.

**Pass.** Control and pose deadline misses below 1 % with the cache;
uplink collision rate (sniffer retries) at least halved; dedup rate
reported.

**Baseline.** Today's system: all flows in one FIFO, senders broadcast.

## Task 5 (optional). Delta frames (work package D)

**Task.** One robot streams its LiDAR to the server for the live map
while driving the route; the map must stay within the reference
thickness bound.

**Cache under test.** The robot sends only cells that differ from its
cached map prediction; the server reconstructs.

**Pass.** Bytes per frame ≤ 20 % of full frames on the static parts of
the route; map thickness within the reference bound; encode time per
frame under 30 ms on the Jetson.

## Instrumentation he needs to build once

- A cache library with a policy interface (get, put, evict, prefetch,
  TTL) used by both the tile cache on the robot and the observation
  cache on the server, with a JSONL log of every operation on the
  shared clock.
- A replay harness that runs the same library on recorded routes
  (survey_1, coop_2) so every policy is tried offline before a drive.
- A scorer that joins the cache log with the reference trajectory, the
  labels and the sniffer trace to produce the pass metrics above.

## Order

Task 1 first (one robot, no masts, no contention); task 3 next (adds
the server's change detection and λ); task 2 (adds masts and the
downlink schedule); task 4 last (everything at once, shared with the
contention session). Each task is one session of about two hours after
the offline replay says the policy works.
