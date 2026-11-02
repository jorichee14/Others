# Plans for persons 1, 2 and 3

Three researches on one rig (two Scout Mini robots with Jetson Orin NX,
Ouster, ZED, radar; three Raspberry Pi masts with cameras; one Asus AP;
one server; chrony; the reference map of site 1). No digital twin, no
Sionna, no labels beyond the person's presence per trial. One shared
scenario (the blind corner) to build on, one joint session at the end.

Person 3 is the lead and owns the thesis (`THESIS.md`): communication
keeps each robot's belief correct where its next action depends on it,
measured by AoII at approach time, with motion as a lever. Person 1
implements the communication rule; person 2 makes it hold on a shared
link; person 3 defines the metric, closes the loop and writes the
theory.

---

## Person 1: validity-timed sharing

**Question.** Which observations need to exist, for whom, and until
when?

**Claim.** Same decisions as timer broadcast, an order of magnitude
fewer bytes, higher deadline hits under load, because the removed
traffic is what delayed the traffic that mattered.

**Owns.**
- Server observation store keyed by cell: latest observation, source,
  timestamp, velocity, TTL.
- On-change triggers on masts and robots with hash-first dedup.
- Robot memory: keep, extrapolate with velocity, expire at TTL, treat
  expiry as a miss.
- Approach time per cell on the robot from path and static map.
- λ(x), the per-cell change rate, from repeated visits in survey_1 and
  coop_2, then online.
- The logger (every create / store / serve / expire / hit / miss on
  the shared clock) and the scorer joining it with reference
  trajectory, person presence and sniffer trace.
- The shared scenario-0 nodes: mast (Pi camera, background
  subtraction, three representations), robot (approach time, memory,
  speed decision), server relay.

**Does not touch.** MAC, slotting, encoders, the speed rule's AoII
version (person 3's).

**Weeks.**
1. The three nodes and the logger; λ(x) offline.
2. S0 blind corner: representation × trigger × route, person in half
   the trials, with and without background load. Figure: bytes and
   latency per representation against decision correctness.
3. Store on the server; S1 one mast, two robots: write once, serve
   twice; the second robot's LiDAR as load. Stop-rule check: does the
   store beat on-change broadcast?
4. S2 moved cabinet: TTL invalidation versus periodic refresh; the
   unnecessary-refresh fraction.
5. Write-up; nodes to the joint session.

**Baselines.** Timer broadcast at 2 and 10 Hz; on-change broadcast with
no store; request-on-demand with no store.

**Stop rule.** If S1 shows on-change broadcast alone reaches ≥ 90 % of
the saving and the deadline hits, publish the event-trigger result
short and stop.

**Deliverable.** The waste number (fraction of timer traffic never
needed) with the decision and deadline rates that go with removing it.

**Reads.** Trimpe and D'Andrea 2014; ETSI TS 103 324 generation and
redundancy rules; Update the Unseen Only and AutoCast §3–4 (refs/);
one survey of mobility-aware caching at RSUs.

---

## Person 2: de-synchronized access and compute

**Question.** Given the messages that must be sent this frame, when
does each node touch the medium, and what does it compute before?

**Claim.** For compensated perception a known bounded delay is nearly
free and a collision costs a frame; clock slotting with no coordinator
recovers most of the contention loss at negligible task cost, and the
slot phase doubles as a compute budget.

**Owns.**
- The contention rig: configurable publisher (size, period, phase) on
  all five nodes with synchronized readiness; sniffer capture; per-
  message send and receive stamps.
- The Bianchi fit and the damage map L(N, size).
- Slotted publishing: phase from id, width T/N; later width ∝ s_i/R_i
  from the AP's per-client rate.
- Message timestamp and velocity fields; receiver extrapolation to the
  fusion instant (object level first).
- Encode-time-versus-bytes curves on Jetson and Pi; slot-aware encode
  depth.
- Arrival-order incremental fusion on the server; latency under burst
  versus staggered arrivals.

**Does not touch.** Content, triggers, the store, the speed rule.

**Weeks.**
1. Rig and sniffer pipeline; jitter from application send to air on
   Jetson and Pi, which bounds usable N.
2. C0 damage map: N 1–5, 10 KB–1.5 MB, one far agent; Bianchi fit.
   Figure: in-frame delivery versus N and size.
3. C1 slotted versus synchronized; compensation on and off on replayed
   content with boxes and velocity. Stop-rule check.
4. C2 rate-aware widths with one agent at low MCS; encode curves on
   both device classes.
5. C3 slot-aware encode and arrival-order fusion at N = 5; write-up;
   the slotted publisher to the joint session.

**Baselines.** Synchronized send over EDCA (today); with WMM
priorities; the edge-grant scheduler as the coordination reference.

**Stop rule.** If slotting recovers < 50 % of the contention loss or
compensation costs > 2 AP at the needed offsets, move to the grant
scheduler and report why.

**Deliverable.** The contention-loss surface and how much slotting
recovers at what task cost; encode curves per device.

**Reads.** Bianchi 2000; Heusse et al. 2003; ETSI DCC / CAM
de-synchronization; 802.11ax OFDMA and restricted TWT; ProgressFTX;
Dense Coverage / Sparse Refinement Supp. F (refs/).

---

## Person 3 (lead): decision-oriented communication

**Question.** What does "fresh enough" mean for a robot about to act,
and how should the robot move and communicate to stay there?

**Claim.** AoII at approach time predicts task success where AoI and
bytes do not; a robot that drives on it completes missions faster at
equal safety; the same rule holds whatever perception sits underneath.

**Owns.**
- The metric page: AoI, AoII and AoII at approach time defined on the
  rig with λ(x); the log contract both others must satisfy, including
  the robot's held belief per cell ahead.
- The speed and path rule in the robot node: speed = f(AoII of cells
  ahead, tolerance); three versions (none, AoI, AoII); link-aware
  prefetch from the survey radio map.
- The analysis of persons 1's and 2's sessions.
- The theory: expected AoII at approach in λ, the delivery-delay
  distribution and the trigger threshold; from it the TTL rule, the
  speed rule, and the no-communication-needed condition.
- C4: the same rule over three perception backends (detector,
  occupancy map, one-bit certificate).

**Does not touch.** Nodes, sniffer, AP, training.

**Weeks.**
1. Metric page and log contract to persons 1 and 2 before they build.
2. Speed rule in the robot node (the one switch); tolerance set at 2 s
   for corridors at 1 m/s, to be revised by S0.
3. R1 from S0 and S1 logs: AoI versus AoII at approach against
   decision outcome. Figure.
4. R2: the speed switch on in S0; mission time at fixed safety for the
   three versions. Figure. Theory drafted.
5. C4 with the backends available; theory finished.
6. Joint session: three masts, two robots driving, pose and control
   live; four runs (today's system, P1 alone, P2 alone, both), speed
   rule on; each paper takes its column, the system paper the fourth.

**Deliverable.** The metric, the closed loop, the theory; the system
paper with persons 1 and 2.

**Reads.** Maatouk et al. (AoII, ToN 2020); Sun, Polyanskiy, Uysal
(T-IT 2020); Kadota et al. (ToN 2018); Update the Unseen Only (refs/);
Fresh2comm; Mostofi's communication-aware motion planning; Fink and
Kumar; the MEAN survey cited in SCooP-iT.

---

## Shared, decided once

- One corner, one mast position, one scripted path, fixed for every
  session.
- One clock (chrony to the server); one log format (JSONL with the
  shared stamp) with person 3's belief field from day one; one sniffer
  position.
- Asus settings fixed: airtime fairness and band steering off, 5 GHz
  pinned, 80 MHz, 802.11ax off if the sniffer needs ac.
- Background load: the second robot's real recording stack streaming
  to the server.

## Dependencies

```
P3 metric page ──▶ P1 logger, P2 stamps (week 1)
P1 nodes ──▶ S0 (week 2) ──▶ P3 R1, P3 speed switch
P2 rig ──▶ C0 (week 2) ──▶ C1 ──▶ C2 ──▶ C3
P1 S1, S2 + P2 C1, C2 ──▶ P3 R2, theory
all three ──▶ joint session (week 6)
```

Each person has a result at week 2 that stands alone; nothing after
week 2 blocks anyone else until the joint session.
