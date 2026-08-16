# Implementation Tracker — Dynamic Digital Twin

**Goal:** build a dynamic digital twin of the lab space on the existing testbed
(2 AMRs + infrastructure nodes, ROS 2, Wi-Fi + 5G, edge processing), then measure the
freshness → twin-error relationship over the real network and build
divergence-triggered synchronization that beats AoI-based scheduling on
twin-error-per-bit.

**Read [`INTRODUCTION.md`](INTRODUCTION.md) first** — thesis, gaps G1–G3, contributions
C1–C6, RQ1–RQ5, positioning, references. Working rules in [`CLAUDE.md`](CLAUDE.md).

**Starting point (2026-08-16): no twin exists yet.** The plan therefore builds the
instrument first (M1–M2), measures second (M3–M4), contributes third (M5).

---

## How to use this file (do not delete)

Whenever a step finishes, update this file **in the same commit** as the work:
flip its status (`⬜ TODO` → `🟨 IN PROGRESS` → `✅ DONE` / `⛔ BLOCKED`), fill the
**Result** line with what actually happened (numbers, paths, surprises), append a dated
row to the progress log. Never mark ✅ without verifying the **Done when** criterion.

---

## Milestone 1 — Foundation (testbed research-ready, ~2–3 weeks)

### Step 1.1 — End-to-end connectivity  `⬜ TODO`
- Every sensing node (AMR 1, AMR 2, infrastructure) streams into the edge ROS 2
  environment over Wi-Fi and over 5G. Verify per-topic which network carries what.
- **Done when:** all sensor topics visible at the edge; per-topic RAT verified by
  interface counters, switchable between Wi-Fi and 5G.
- **Result:** _pending_

### Step 1.2 — Common timebase  `⬜ TODO`
- Synchronize clocks across all nodes and the edge (PTP or equivalent). Measure the
  residual per-node offset; that bound accompanies every age/latency number reported
  later. No age measurement is valid before this step is done.
- **Done when:** per-node clock-error bound measured and documented (target ≤ 1 ms
  wired, ≤ 5 ms wireless).
- **Result:** _pending_

### Step 1.3 — Static map  `⬜ TODO`
- Produce the map of the space with the existing SLAM stack. This is the twin's
  static layer; it changes rarely.
- **Done when:** map generated, stored, loadable by downstream nodes.
- **Result:** _pending_

### Step 1.4 — Decision D-GT: ground truth for dynamic-entity pose  `⬜ TODO`
- Choose how true positions of robots/objects/people will be known (mocap, tags +
  infrastructure cameras, surveyed props + odometry). Quantify the method's own
  accuracy — it caps the resolution of every result. **Gates Milestone 4.**
- **Done when:** method chosen, its error measured (target ≤ 5–10 cm), documented.
- **Result:** _pending_

## Milestone 2 — Twin v0, the instrument (~3–4 weeks)

*Keep it as simple as possible: the research is about what the network does to the
twin, not about the twin's sophistication. Tracked-blobs-on-a-map is sufficient.*

### Step 2.1 — Detections at every node  `⬜ TODO`
- Each sensing node emits "object of type T at position P at time t" instead of (in
  addition to) raw streams — pick per-sensor detectors, off the shelf where possible.
- **Done when:** detection topics from all nodes, in a common world frame, with
  source timestamps.
- **Result:** _pending_

### Step 2.2 — State keeper at the edge  `⬜ TODO`
- The heart of the twin: subscribe to all detection streams, associate detections to
  entities, maintain per-entity state (id, class, position, velocity, covariance,
  timestamp), support querying the state at an arbitrary time via constant-velocity
  extrapolation. No learning.
- **Done when:** twin runs live from ≥ 3 sources; state queryable at arbitrary *t*;
  all states logged for offline scoring.
- **Result:** _pending_

### Step 2.3 — Live visualization  `⬜ TODO`
- Map + live entity markers (RViz is enough). This is the working-demo milestone.
- **Done when:** the room's motion is watchable in the twin in real time.
- **Result:** _pending_

### Step 2.4 — Age plumbed through  `⬜ TODO`
- Every twin entity knows how old its supporting evidence is (per-source generation
  timestamps carried end-to-end; delivery age computable per update).
- **Done when:** per-entity, per-source age is loggable during live operation.
- **Result:** _pending_

## Milestone 3 — Characterization (~2–3 weeks, may overlap M2) → C1

### Step 3.1 — Measurement campaign  `⬜ TODO`
- Run the system as-is and measure: age/loss/rate distributions per network (Wi-Fi
  2.4/5, 5G), per payload type (detections, point clouds, images), per mobility state
  (static/moving AMR), per load. Record raw streams (rosbag2) in every run so later
  analyses replay offline. External anchors: CoInfra [24], 5G-vs-Wi-Fi robot
  measurements [29].
- **Done when:** per-condition distributions (median/p95/p99 age, loss, burst
  structure), plots, and trace paths committed under `results/m3/`; campaign
  re-runnable from its documentation.
- **Result:** _pending_

### Step 3.2 — Decision D-CSI: channel-state extractability  `⬜ TODO`
- **Now load-bearing** (method core per [`METHOD.md`](METHOD.md) — Twin-Coherence
  Gating): can per-packet CSI be captured from the Wi-Fi NICs and/or reference-signal
  reports from the 5G UE? One afternoon, yes/no with evidence. Do this EARLY (can run
  during M1). NO ⇒ fall back to radar-Doppler coherence (same loop); record the pivot.
- Includes the **empty-room CSI pilot**: record CSI while one AMR drives a known path
  in an otherwise empty room — if odometry does not explain the channel variation, the
  coherence mechanism dies cheaply here.
- **Done when:** verdict + pilot result recorded. **Gates the TCG gate layer.**
- **Result:** _pending_

## Milestone 4 — Core experiment (~4–5 weeks) → C2, C3

### Step 4.1 — Twin-error metric  `⬜ TODO`
- Per-entity position error vs D-GT ground truth, decomposed: static vs dynamic
  entities × strict/loose thresholds (the AP@0.5-vs-AP@0.7 analogue) × miss/ghost
  rates. State its resolution bound (from Steps 1.2 + 1.4).
- **Done when:** implemented, validated on a recorded bag, resolution bound stated.
- **Result:** _pending_

### Step 4.2 — ⚠️ PRE-REGISTER the gate, then run it  `⬜ TODO`
- **Rule to finalize in this file BEFORE the run** (draft): under natural network
  latency, **NO-GO for the dynamics-aware thesis** if oracle kinematic correction
  recovers < 50% of the gap between the stale twin and the fresh-snapshot twin.
  NO-GO ⇒ the project pivots to the measurement contribution (M3 + Step 4.3) alone.
- **Done when:** rule committed before the run; verdict written with its numbers.
- **Result:** _pending_

### Step 4.3 — The freshness → twin-error mapping  `⬜ TODO`
- Open loop, replay-first from M3 recordings: twin error across network × payload ×
  entity class; heterogeneous-age conditions (fresh infrastructure + stale AMR and
  permutations); four kinematic arms (none / oracle velocity / oracle displacement /
  tracker velocity — the design pre-registered in `temporal_messaging/HANDOFF.md` §5,
  now on real data). Test whether AoI or a kinematic divergence predictor better
  predicts measured twin error.
- **Done when:** the curves exist with seeds/runs documented — the paper's main
  evidence.
- **Result:** _pending_

## Milestone 5 — The contribution (~4 weeks) → C4

### Step 5.1 — Divergence-triggered synchronization, closed loop  `⬜ TODO`
- Entity-level rule: transmit when predicted divergence exceeds a bound. Baselines
  under identical conditions: send-everything, periodic (several rates), AoI-based.
  Metric: twin-error-per-bit on the live network. Pre-register the comparison
  protocol before the first closed-loop run.
- **Done when:** head-to-head results on the live network, protocol pre-registered.
- **Result:** _pending_

## Milestone 6 — Write and extend (ongoing)

- Paper drafting runs alongside from M3 onward (M3 = measurements section; M4 = core
  evidence; M5 = headline).
- **C5 (network-twin coupling, dual-RAT selection by the twin)** and **C6
  (channel-as-sensor updates; ⛔ auto-dropped if D-CSI = NO)** start only after M4/M5
  succeed.

---

## Progress log

| Date | Step | Notes |
|------|------|-------|
| 2026-08-16 | setup | Project skeleton: `INTRODUCTION.md` (thesis, gaps, C1–C6, RQ1–RQ5, positioning, 48 refs), `CLAUDE.md` (working rules), first tracker version. |
| 2026-08-16 | replan | Tracker restructured for the true starting point (no twin exists yet): M1 foundation → M2 twin v0 → M3 characterization → M4 core experiment → M5 policy → M6 writing/extensions. Gates D-GT (ground truth, blocks M4) and D-CSI (blocks only C6) carried over; pre-registration points marked at Steps 4.2 and 5.1. |
