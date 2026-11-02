# What each person focuses on

Two researches, one rig, one joint session at the end. Person 1 owns
scenario 0's shared nodes because their plan uses all of them; person
2 builds the contention rig in parallel and borrows the nodes for the
final session.

---

## Person 1: validity-timed sharing

**Focus, in one line.** Decide which observations exist, for whom, and
until when; measure how much of today's traffic was never needed.

**Owns.**
- the observation store on the server (keyed by cell: latest
  observation, source, timestamp, velocity, TTL);
- the on-change trigger on masts and robots, with hash-first dedup;
- the receiver's memory on the robot: keep, extrapolate with velocity,
  expire at TTL, treat expiry as a miss;
- approach time per cell on the robot from its path and the static map;
- λ(x), the per-cell change rate, first from the recordings' repeated
  visits, then updated online;
- the logger of every create / store / serve / expire / hit / miss on
  the shared clock, and the scorer that joins it with the reference
  trajectory, the person's presence, and the sniffer trace.

**Does not touch.** The MAC, slotting, message timing within a frame,
encoders. Uses timer broadcast as the baseline and the AP as it comes.

**Weeks.**
1. Mast node (Pi: camera, background subtraction, three
   representations), robot node (approach time, memory, speed
   decision), server relay, logger. Offline: λ(x) from survey_1 and
   coop_2.
2. S0 on the real corner: representation × trigger × route, person in
   half the trials, with and without background load. First figure:
   bytes and latency per representation against decision correctness.
3. Store on the server; S1 with two robots: write once, serve twice.
   The stop-rule check: does the store beat on-change broadcast?
4. S2 moved cabinet: TTL invalidation against periodic refresh; the
   unnecessary-refresh fraction.
5. Write-up of S0–S2; hand the nodes to the joint session.

**Deliverable.** The waste number (fraction of timer traffic never
needed) and the decision-correctness and deadline-hit rates that go
with removing it; one figure per scenario.

**Reads first.** Trimpe and D'Andrea 2014 (event-triggered estimation);
ETSI TS 103 324 generation and redundancy rules; Update the Unseen Only
(refs/); AutoCast §3–4 (refs/); one survey of mobility-aware caching
at RSUs.

---

## Person 2: de-synchronized access and compute

**Focus, in one line.** Make the messages that must be sent arrive
without colliding, by giving each node a phase in the frame and a
compute budget to match; measure contention loss first.

**Owns.**
- the contention rig: a configurable publisher (size, period, phase)
  on all five nodes, synchronized readiness on the shared clock;
  sniffer capture; per-message send and receive stamps;
- the Bianchi fit and the damage map L(N, size);
- slotted publishing (phase from id, width T/N; later width ∝ s_i/R_i
  from the AP's per-client rate);
- the message timestamp and velocity field and the receiver's
  extrapolation to the fusion instant;
- encode-time-versus-bytes curves on Jetson and Pi; slot-aware encode
  depth;
- arrival-order incremental fusion on the server, and its latency
  under burst versus staggered arrivals.

**Does not touch.** What a message contains or whether it was needed;
the store; triggers. Uses timer broadcast of fixed-size messages as
the traffic.

**Weeks.**
1. The rig and the sniffer pipeline; jitter measurement on Jetson and
   Pi (application send to air), which bounds usable N.
2. C0 damage map: N 1–5, 10 KB–1.5 MB, one far agent; Bianchi fit.
   First figure: in-frame delivery versus N and size.
3. C1 slotted versus synchronized, compensation on and off on replayed
   content with boxes and velocity. The stop-rule check: ≥ 50 % of the
   loss recovered at ≤ 2 AP cost.
4. C2 rate-aware widths; encode curves on both device classes
   (T1, T2 of COMPUTE.md).
5. C3 slot-aware encode and arrival-order fusion at N = 5; write-up;
   bring the slotted publisher to the joint session.

**Deliverable.** The contention-loss surface and how much of it
slotting recovers at what task cost; the encode curves per device;
one figure per experiment.

**Reads first.** Bianchi 2000; Heusse et al. 2003 (performance
anomaly); ETSI DCC / CAM de-synchronization; 802.11ax OFDMA and
restricted TWT basics; ProgressFTX (split inference); Dense Coverage /
Sparse Refinement Supp. F (refs/).

---

## Joint session (week 6)

Three masts, two robots driving, pose and control traffic live. Four
runs: timer broadcast over EDCA (today); person 1's system alone;
person 2's slotting alone; both. Scored on decision correctness at
corners, bytes, deadline hits of pose and control, airtime. Each
person's paper takes its own column; the system paper takes the
fourth.

## Shared, decided once

- One corner, one mast position, one scripted path, fixed for every
  session.
- One clock (chrony to the server), one log format (JSONL with the
  shared stamp), one sniffer position.
- Asus settings fixed: airtime fairness and band steering off, 5 GHz
  pinned, 80 MHz, 802.11ax off if the sniffer needs ac.
- Background load: the second robot's recording stack streaming to the
  server, so both plans see the same load.
