# Two plans

Two distinct projects on the same rig (two Scout Mini robots with
Jetson Orin NX, Ouster, ZED, radar; three Raspberry Pi masts with
cameras; one Asus AP; one server; chrony; the reference map of site 1).
No digital twin needed for either. They share scenario 0 as the first
experiment and diverge after it. Each plan has a direction, a claim,
what gets built, what gets measured, the baselines, the prior art that
must be answered, and a stop rule.

---

## Plan 1: Validity-timed sharing (the systems / caching plan)

**Direction.** Treat every observation an agent shares as an item with
a lifetime and a set of consumers. Decide when it is created (on
change, not on a timer), how long it stays true (from the measured
change rate of the place it describes), who will need it (receivers
whose path reaches that place), and when it is delivered (before their
approach time, from one stored copy). Bits per message are not the
object; messages per task are.

**Claim.** On a shared link, a team that shares by validity and
approach time makes the same decisions as a team that broadcasts on a
timer, with an order of magnitude fewer bytes and a higher deadline-hit
rate under load, because the traffic it removes is exactly the traffic
that delayed the traffic that mattered.

**Build.**
1. A shared observation store on the server keyed by map cell, holding
   the latest observation per cell with its timestamp, source,
   representation and velocity; TTL per cell from λ(x), the per-cell
   change rate estimated from repeated visits in the recordings
   (survey_1, coop_2) and updated online.
2. Sender triggers on the masts and robots: publish on change
   (occupancy flips, a box appears or moves > 0.5 m), never on a timer;
   hash-first so a second sender of the same cell does not upload.
3. Receiver-side: each robot computes approach time per cell along its
   path from the static map and its speed; requests or is served the
   stored observations for cells it cannot see itself before approach
   time; keeps them with velocity and extrapolates until TTL.
4. Delivery scheduling on the server: serve by approach time, time-
   critical flows (pose, control) ahead of observations, on the
   downlink the AP owns.
5. Logging on the shared clock: every create, store, serve, expire,
   hit, miss; sniffer airtime.

**Measure.** Decision correctness at blind corners; observation on the
receiver before approach time (rate); bytes up and down; deadline hits
of pose and control; the fraction of timer-baseline messages that were
redundant or expired before use (the waste number); hit/miss of the
receiver's store at decision time.

**Baselines.** Timer broadcast at 2 Hz and at 10 Hz (today); on-change
broadcast without the store (event-triggered, no validity, no
approach timing); receiver request-on-demand with no store.

**Experiments, in order.**
- S0 blind corner: one mast, one robot, one person (one session).
- S1 one mast, two robots: write once, serve twice; the second robot's
  LiDAR as background load.
- S2 changed building: move a cabinet after mapping; TTL invalidation
  versus periodic refresh.
- S3 full rig: three masts, two robots driving, pose and control
  deadlines under observation traffic.

**Prior art to answer.** Event-triggered estimation (Trimpe and
D'Andrea), ETSI CPM generation and redundancy rules, mobility-aware
proactive caching at RSUs, Update the Unseen Only (AoI over unseen
areas), AutoCast (relevance and current visibility). The answer:
validity from the environment's measured change rate, delivery timed
to the receiver's approach, one stored copy serving many, tested on
a real contended link.

**Stop rule.** If S1 shows on-change broadcast without the store
already reaches ≥ 90 % of the plan's byte saving and deadline hits,
the store and the approach timing are not earning their complexity;
publish the event-trigger result as a short paper and stop.

**Who.** The caching person; the robot code is three small ROS 2 nodes
and a logger.

---

## Plan 2: De-synchronized access with compensation (the networking / compute plan)

**Direction.** Synchronized sensing makes every agent's message ready
at the same instant, and on CSMA that is the worst case: collisions,
random serialization, the last sender misses the frame. Keep sensing
synchronized, de-synchronize *transmission* by giving each agent a
phase in the frame from the shared clock, absorb the resulting known
delay with a velocity tag and extrapolation at the receiver, and let
the phase also set how much each agent computes before it sends.

**Claim.** For perception traffic with kinematic compensation, a
known, bounded delay is nearly free and a collision costs a whole
frame; so a clock-slotted schedule with no coordinator recovers most
of the contention loss at negligible task cost, and the slot phase
doubles as a compute budget that lets late agents send less.

**Build.**
1. The contention rig: all five nodes publishing a configurable
   message (size, period) on the shared clock, synchronized readiness;
   sniffer; per-message send/receive stamps.
2. Slotted publishing: agent i sends at t_k + φ_i, slot width T/N,
   φ_i from agent id; later, width ∝ s_i/R_i from the AP's per-client
   rate.
3. Message timestamp and velocity field; receiver extrapolation to
   the fusion instant (object level first: boxes with velocity).
4. Slot-aware encoding on the Jetsons: the encode-time-versus-bytes
   curve (selection threshold, quantization, decimation); pick the
   point that finishes by φ_i and fits the slot. On the Pis: decimation
   and JPEG quality only.
5. Arrival-order fusion on the server: fuse incrementally as messages
   land; read the result at the deadline.

**Measure.** In-frame delivery versus N and message size, synchronized
versus slotted; delay distribution; retries and airtime from the
sniffer; task metric (detection or tracking on replayed content) at
each slot offset with and without compensation; server fusion latency
burst versus staggered; encode time and bytes on each device class;
Bianchi-model fit.

**Baselines.** Synchronized send over EDCA (today); synchronized send
with WMM priorities; the edge-grant scheduler (coordination) as the
upper reference.

**Experiments, in order.**
- C0 damage map: N 1–5, size 10 KB–1.5 MB, one far agent (one day).
- C1 slotted versus synchronized at the same N and sizes; compensation
  on and off.
- C2 rate-aware widths with one agent at low MCS.
- C3 compute: encode curves on Jetson and Pi; slot-aware encode on;
  arrival-order fusion on; end-to-end deadline hits at N = 5.

**Prior art to answer.** TDMA over a shared clock, ETSI DCC and CAM
de-synchronization (beacons, no payloads, no compensation), Bianchi's
CSMA model, the 802.11 performance anomaly, 802.11ax OFDMA, split
inference (ProgressFTX), CPM's measurement time and kinematics. The
answer: perception payloads with compensation, the delay-versus-
collision trade measured in task loss, compute budget from slot phase,
on a real AP with heterogeneous devices.

**Stop rule.** If C1 shows slotting recovers < 50 % of the contention
loss (device jitter eats the slots), or compensation costs > 2 AP at
the needed offsets, de-synchronization is not the answer at this N;
move to the grant scheduler (coordination) and report why.

**Who.** A networking or systems person comfortable with ROS 2 timing,
a sniffer and a Jetson; the compute part needs someone who can export
a detector to the Jetson.

---

## How they relate

Plan 1 removes messages; plan 2 makes the remaining ones arrive. Both
start from scenario 0 on the same corner with the same nodes, and S3
and C3 are the same session with different switches. If both work,
the combined system is: on-change observations, served from one store
at approach time, in clock slots, with velocity tags. If only one
works, it stands alone; neither depends on the other, on a digital
twin, on Sionna, or on labels beyond the person's presence per trial.

| | Plan 1 | Plan 2 |
|---|---|---|
| Lever | which messages exist and when | when the medium is used and what is computed before |
| Object | observations with validity | frames with phase |
| First number | fraction of today's traffic that was never needed | contention loss versus N and size |
| Risk | the store adds nothing over on-change broadcast | device jitter eats the slots |
| Prior art centre | event-triggered estimation, caching | TDMA, DCC, Bianchi |
| Paper venue | MobiSys / IoT-J | INFOCOM / ToN, or JSAC |
