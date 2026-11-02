# Computing inside the same deadline

Companion to `CONTENTION.md`. The deadline of a perception frame is
shared three ways: sensing-to-encode on the agent, airtime on the
channel, and fusion on the server. The literature treats the first and
third as constants (DR's Supp. F: 40 ms compute plus 10 ms per trip
[F]) and optimizes the middle. On the hardware here they are not
constants, they are levers, and they couple to the channel through the
same clock.

## 1. Three observations about the rig

- **Compute is heterogeneous.** Two Jetson Orin NX (a BEV backbone at
  tens of ms, a learned selector affordable) and three Raspberry Pi 4
  (no neural encoder; only decimation, voxel downsampling, JPEG). The
  representation each agent can produce is set by its compute, not by
  a design choice: the Pis send raw-ish, the Jetsons can send features
  or detections, and the server must fuse across levels.
- **The server bursts with the channel.** With synchronized senders,
  all N messages arrive at the frame boundary and the server's fusion
  runs once on the full batch; its latency is in the deadline just like
  airtime. With de-synchronized senders (CONTENTION phase 1) messages
  arrive in slot order and the server can process each on arrival.
- **Waiting is computing.** An agent with slot offset φ_i has
  φ_i − t_sense milliseconds before it may transmit. That is time it
  would otherwise idle; spent on a deeper encode it yields a smaller
  message, which needs a narrower slot. Slot phase and encode depth are
  one decision.

## 2. The angle: slot-aware encoding and arrival-order fusion

**Agent side (slot-aware encoding).** Each agent has a measured
encode-time-versus-bytes curve e_i(b): bytes out as a function of
compute time in (deeper selection, coarser quantization, more
decimation). Given its slot start φ_i and width w_i at rate R_i, it
picks the point on the curve that finishes by φ_i and fits in w_i·R_i
bytes. Late slots compute more and send less; early slots send more and
compute less. On a Pi the curve is flat (little compute helps), so Pis
take early slots with wide widths; Jetsons take late slots with narrow
widths. The slot assignment is then a matching of agents to slots by
their curves, computable from shared state (ids, rates, device class)
with no coordination traffic.

**Server side (arrival-order fusion).** Fusion runs incrementally as
messages arrive in slot order: late fusion is trivially incremental,
max-fusion is associative, attention fusion is recomputed over the
agents present so far. At any instant the server holds a fused estimate
from the agents that have arrived; the frame's result is whatever that
is at the deadline, and a late agent's message improves the *next*
read, not none. This is the compute analogue of the partial-delivery
identity: value accrues per arrival instead of all-or-nothing at the
batch.

**Cross-level fusion (forced by the rig).** The server receives raw
points from Pis and features or boxes from Jetsons. The fusion must
accept heterogeneous levels; the simplest is to run the server-side
encoder on raw inputs and fuse at feature level (the server has the
GPU), which moves the Pis' compute to the server and charges it to the
server's share of the deadline. The question "where does each agent's
encoding run" is then answered per agent by its compute and its slot,
not globally.

## 3. What is new

- The coupling: slot phase ↔ encode depth on the agent, slot order ↔
  incremental fusion on the server, both from the one clock that fusion
  already requires. Split inference (ProgressFTX, edge offloading) picks
  a split point per device under a modeled channel; it does not
  schedule the split against other agents' slots or against the
  server's arrival order.
- Compute heterogeneity as a *design input* for the representation
  level per agent, rather than a homogeneous-agent assumption (every CP
  benchmark assumes identical encoders; HEAL-style heterogeneity is
  about sensors, not compute).
- Arrival-order anytime fusion as the server's answer to the same
  burst that contention is the channel's answer to.

Not new, used as such: split computing, pipelining encode with
transmit, incremental late fusion, TensorRT on Jetson.

## 4. What is testable on the robot now

| Test | Device | Measures | Needs |
|---|---|---|---|
| T1 encode curve | Jetson Orin NX | e(b): time vs bytes for PointPillars + Where2comm-style selection at several thresholds, and for voxel downsampling at several resolutions; with tegrastats power | OpenCOOD on Jetson (TensorRT or PyTorch), one checkpoint |
| T2 encode curve | Raspberry Pi 4 | e(b) for decimation, voxel grid, JPEG/PNG of depth and RGB | Python, no model |
| T3 server burst vs staggered | server GPU | fusion latency and GPU utilization when N messages arrive together vs in slots; incremental fusion quality vs time within the frame | the server's fusion stack, logged arrivals |
| T4 end to end | all five + AP | deadline hit rate and task metric with (a) synchronized send, batch fusion; (b) slotted send, batch fusion; (c) slotted send, slot-aware encode, incremental fusion | CONTENTION phase 0 rig plus the above |

T1 and T2 take a day each and need no network. T3 needs only the
server and a replay of logged messages with synthetic arrival times.
T4 is the CONTENTION phase 1 session with two extra switches.

## 5. Decision rule

If T4(c) does not beat T4(b) by at least 10 % on in-deadline fused
results at N = 5, the compute lever adds nothing beyond
de-synchronization and the paper's compute section is one paragraph of
measurements (T1–T3). If it does, slot-aware encoding and arrival-order
fusion are the method, and de-synchronization is what enabled them.
