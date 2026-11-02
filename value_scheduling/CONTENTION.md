# Contention: a general plan

Scope: what contention on a shared channel does to collaborative
perception, and the ways out. Content selection (what to send) is a
separate question and is held fixed here. Prior-art notes are from
memory and search summaries except where marked [R] (SCooP
measurements) or [F] (read in full).

## 1. The diagnosis: fusion synchronizes the senders, and synchronized senders collide

Collaborative perception needs the agents' observations aligned in
time, so every agent's sensors tick on a shared clock (SCooP: ≤ 0.99 ms
offset [R]). Each agent's message is therefore ready at the same
instant, and on a CSMA channel N ready senders back off at random and
serialize; the last ones to win the medium miss the frame. The data
already show it: one robot on the AP lands 86 % of 1.5 MB messages in
the frame, two robots 59–68 %, with the loaded round-trip time still
25–33 ms [R]. Latency is not the problem; simultaneous readiness is.

Three further properties make perception traffic a worst case for
CSMA:

- **Periodic and phase-locked.** Every 100 ms the same collision
  recurs; random backoff never breaks the phase lock because the
  arrivals are regenerated in phase each frame.
- **Deadline-bound with no MAC deadline.** 802.11 has priorities (EDCA
  access categories) but no notion of "worthless after 100 ms"; a
  message that lost the backoff keeps retrying into the next frame and
  collides with the next frame's messages.
- **Rate-heterogeneous.** A far agent at a low MCS occupies the channel
  for as long as several near agents (the 802.11 performance anomaly,
  Heusse et al. 2003); under CSMA the slow sender is airtime-greedy by
  construction.

And one property that is the way out:

- **Staleness is cheap for perception.** A message that arrives 30 ms
  later than its sibling is not lost; with a velocity tag the receiver
  extrapolates it, and indoors at ~1 m/s the residual displacement is
  far inside the matching tolerance (the regime number Π_c ≪ 1 in
  `../task_oriented_comm/THEORY.md` §2.5). The contention literature
  treats delay as the cost; for perception with kinematic compensation
  a bounded, *known* delay costs almost nothing, while a collision
  costs a whole frame.

That asymmetry is the plan's premise: **trade a known, compensable
delay for an unknown, uncompensable loss.**

## 2. The three ways out, and what each costs

| Strategy | Mechanism | Cost | Where it exists |
|---|---|---|---|
| **Coordinate** | a center (AP trigger loop, edge grants) or a replicated schedule from shared state decides who sends when | a control round or control traffic; a single point that must see everyone | 802.11ax OFDMA trigger/BSR; AutoCast's replicated TDMA schedule [F]; JigsawComm's replicated top-1 [F]; `HARDWARE.md` grants |
| **De-synchronize** | keep sensing synchronized, offset each agent's *transmission* phase within the frame so ready times do not coincide | staleness equal to the offset, compensated at the receiver | ETSI DCC and CAM desynchronization for V2X beacons; beacon staggering in sensor networks; not applied to perception payloads with compensation |
| **Tolerate** | make the message useful under partial delivery: progressive units, rateless/network coding across agents, fusion trained for correlated loss | coding overhead; a codec that is prefix-decodable | DR's nested patches [F]; JigsawComm's ego fallback [F]; LR-V2X, masked-quantized loss-tolerant CP [S]; fountain codes (networking) |

None of the three is new alone. What has not been done is to put them
on one axis for perception, where the cost of each is measured in the
same unit, task loss at the deadline, on a real contended link.

## 3. The plan

### Phase 0. Measure the damage as a function of (N, size, period) on the real AP

Two Jetsons and three Raspberry Pis on the Asus AP, all sending a
synthetic perception message of size s at period T on the shared clock,
synchronized readiness. Sweep N ∈ {1..5}, s ∈ {10 KB, 100 KB, 300 KB,
1.5 MB}, T = 100 ms, MCS spread (one agent far). Measure per message:
in-frame delivery, delay distribution, retries (sniffer), airtime per
agent. Fit Bianchi's model with the measured parameters.

Output: the contention-loss surface L(N, s). The paper's first figure,
and the quantity every later gain is measured against. Pass bar: the
model within 10 % of the measured in-frame rate.

### Phase 1. Clock-slotted de-synchronization with kinematic compensation

Each agent i transmits its frame-k message at t_k + φ_i, with φ_i a
deterministic function of agent id (or of the previous frame's
schedule), slots of width T/N. No coordinator, no control traffic: the
clock is already shared for fusion. Each message carries its
measurement time and a velocity field (as ETSI CPM does); the receiver
extrapolates to the fusion instant.

Theory to state: expected in-frame delivery under CSMA with N
phase-locked senders (Bianchi, collision probability p(N)) versus
slotted (≈ 1 when s/R_i < T/N), and the staleness cost bound: the
displacement error from an offset φ is v·φ, compensable to σ_v·φ with a
velocity tag; the regime number says when that is negligible.

Measure on the same rig: in-frame delivery and, in replay with a
detector, AP with and without compensation at each φ. Decision: if
slotted delivery recovers ≥ 80 % of the contention loss at ≤ 1 AP cost
from staleness, de-synchronization is the primary answer and
coordination is unnecessary at this scale.

### Phase 2. Rate-aware slot widths (the performance anomaly)

Equal slots waste airtime on fast agents and starve slow ones. Set slot
width ∝ s_i / R_i from the AP's per-client rate readout, re-announced
each second on the control channel (or derived from the shared radio
map by position). This is airtime-fair TDMA on top of CSMA; the novelty
is only that the rate is predicted from where the agent will be, so
slots are planned a frame ahead.

### Phase 3. Tolerate what is left: progressive units and cross-agent rateless coding

For messages that cannot fit their slot (raw clouds, bursts), send
prefix-decodable units so that the part that lands by the deadline is
usable (the partial-delivery identity in `../task_oriented_comm/
THEORY.md` Prop. 1 gives the exact gain). Where two agents hold the
same cell, random linear combinations across them let the receiver
decode from whichever arrives: channel diversity from perception
redundancy. Test whether coding gains beat coding overhead at the
measured loss rates; expectation: it pays only for large messages
under high N.

### Phase 4. Coordination only where de-synchronization fails

When slots are too narrow (large N, large s), move to the grant design
of `HARDWARE.md`: the edge scheduler assigns slots by value per airtime.
Phase 0–1 numbers say where that boundary is; the claim is that below
it, zero-coordination slotting is enough, and above it, coordination is
what pays for its control traffic.

### Phase 5. Multiple access points and association as scheduling

Indoors with several APs, which AP an agent associates to decides which
contention domain it loads. Association by position along the planned
path (from the radio map) is scheduling at the slowest timescale.
Deferred; stated as the extension.

## 4. What is new here

- **The premise**, stated and measured: for perception with kinematic
  compensation, a known delay is nearly free and a collision is a whole
  frame, so the right design for synchronized sensing is deliberately
  de-synchronized access. CP papers assume synchronized arrival; V2X
  DCC de-synchronizes beacons but has no compensation and no payloads
  of this size.
- **One axis of comparison**: coordinate, de-synchronize, tolerate,
  all scored as task loss at the deadline on the same contended link,
  with the boundary between them located by N and s.
- **Slots planned from predicted rate**, not measured rate, using the
  radio map.

What is not new, and is used as such: TDMA over a shared clock, Bianchi's
model, the performance anomaly, EDCA priorities, rateless coding, ETSI
CPM's measurement time and kinematics, 802.11ax OFDMA.

## 5. Risks

- Jetson and Raspberry Pi Wi-Fi stacks add jitter of several
  milliseconds between application send and air; slot guard times eat
  into T/N at N = 5 (20 ms slots). Measure the jitter first; it bounds
  the usable N.
- The AP's own firmware scheduler (airtime fairness, aggregation) can
  reorder what the slots ordered; disable what can be disabled and
  measure the rest.
- Background traffic that is not slotted (map tiles, control) still
  contends; it goes in its own slot or in a lower access category.
- The velocity tag assumes the sender can estimate velocity; for
  features without a tracker, the compensation is a flow (CoBEVFlow),
  which is a learned component and should be a baseline, not the
  design.

## 6. Order

Phase 0 needs only the five devices, the AP, a sniffer and chrony: one
day of recording. Phase 1 adds a phase offset in the publisher and the
measurement-time field in the message: a week. Everything after depends
on what those two show.
