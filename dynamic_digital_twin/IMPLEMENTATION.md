# Implementation Tracker — Dynamic Digital Twin

**Goal:** measure the AoI → twin-error mapping for a dynamic digital twin over a real
dual-RAT network (Wi-Fi + 5G), then build divergence-triggered synchronization that
beats AoI-optimal scheduling on twin-error-per-bit.

**Read [`INTRODUCTION.md`](INTRODUCTION.md) first** — thesis, gaps G1–G3, contributions
C1–C6, RQ1–RQ5, positioning, references. Working rules in [`CLAUDE.md`](CLAUDE.md).

---

## How to use this file (do not delete)

Whenever a step finishes, update this file **in the same commit** as the work:
flip its status (`⬜ TODO` → `🟨 IN PROGRESS` → `✅ DONE` / `⛔ BLOCKED`), fill the
**Result** line with what actually happened (numbers, paths, surprises), append a dated
row to the progress log. Never mark ✅ without verifying the **Done when** criterion.

---

## Phase 0 — Testbed characterization and gates (RQ1 → C1)

*No science yet — instrumentation and honest error bars. Everything downstream inherits
these bounds.*

### Step 0.1 — Clock synchronization across all nodes  `⬜ TODO`
- Establish a common timebase across AMR 1, AMR 2, infrastructure nodes, and the edge
  server. Preferred: PTP (`ptp4l`/`phc2sys`, or `chrony` with hardware timestamping)
  with the edge as grandmaster; wired for infrastructure nodes, wireless for AMRs.
- Quantify residual offset per node (e.g., wired-reference round-trip, or a shared
  physical event visible to two sensors). Re-measure while an AMR roams between APs.
- **Done when:** per-node clock-error bound is measured and documented (target ≤ 1 ms
  wired, ≤ 5 ms wireless; if worse, every downstream age carries the measured bound).
- **Result:** _pending_

### Step 0.2 — Per-topic RAT steering  `⬜ TODO`
- Make traffic routable per stream: pin a given ROS 2 topic to Wi-Fi 2.4, Wi-Fi 5, or
  5G (Linux policy routing / interface binding in the DDS config / separate DDS
  domains per interface). Verify with interface byte counters, not assumptions.
- **Done when:** a chosen topic demonstrably traverses a chosen RAT, switchable
  without rebooting the robot.
- **Result:** _pending_

### Step 0.3 — Age instrumentation of the pipeline  `⬜ TODO`
- Stamp every message at three points: `t_sensor` (driver), `t_pub` (publish),
  `t_recv` (edge). Log `(topic, size_bytes, t_sensor, t_pub, t_recv, seq)` per message
  via a lightweight edge logger node; compute delivery age and loss (seq gaps) offline.
- Record raw streams with rosbag2 in the same runs (working rule 6).
- **Done when:** age/loss records stream to disk for all sensor topics on all nodes;
  a one-hour smoke capture parses cleanly.
- **Result:** _pending_

### Step 0.4 — Measurement campaign  `⬜ TODO`
- Grid: payload {detections/boxes, point cloud, RGB-D image} × RAT {Wi-Fi 2.4, Wi-Fi
  5, 5G} × mobility {static AMR, moving AMR} × network load {idle, loaded}. Multiple
  runs per cell; report age distributions (median, p95, p99), loss, and burst
  structure per cell. External comparison anchors: CoInfra [24], the 5G-vs-Wi-Fi robot
  measurements [29].
- **Done when:** per-cell distributions with run counts, plots, and trace paths are
  committed under `results/phase0/`; the campaign is documented well enough to re-run.
- **Result:** _pending_

### Step 0.5 — Decision D-GT: ground truth for dynamic-entity pose  `⬜ TODO`
- Options: (a) motion capture if available; (b) ceiling/infrastructure cameras +
  AprilTags on AMRs and props; (c) surveyed static props + AMR odometry/SLAM as
  silver-standard. Quantify the chosen method's accuracy — it caps the twin-error
  metric's resolution.
- **Done when:** method chosen, its error measured (target ≤ 5–10 cm), procedure
  documented. **Gates Phase 1.**
- **Result:** _pending_

### Step 0.6 — Decision D-CSI: channel-state extractability  `⬜ TODO`
- Determine whether usable CSI/RSSI streams can be captured from the Wi-Fi NICs
  (chipset-dependent: FeitCSI/iwl, Nexmon, or AX2xx tooling) and/or the 5G UE (UE
  metrics, gNB traces). One afternoon, yes/no with evidence.
- **Done when:** verdict recorded. NO ⇒ Option C (C6) is dropped, no other step
  changes. **Gates only C6.**
- **Result:** _pending_

## Phase 1 — Twin v0 and the error metric (RQ2 → C2)

### Step 1.1 — Twin v0 at the edge  `⬜ TODO`
- Minimal object-level twin: static map from SLAM + per-entity state records
  (id, class, position, velocity, covariance, `t_state`) maintained from all sources'
  detections; constant-velocity extrapolation available at query time (query the twin
  at time *t*, get extrapolated states). No learning.
- **Done when:** twin runs live on edge from ≥ 3 sources; state queryable at
  arbitrary *t*; states logged for offline scoring.
- **Result:** _pending_

### Step 1.2 — Twin-error metric  `⬜ TODO`
- Per-entity position error vs D-GT ground truth at sampled instants, decomposed:
  static vs dynamic entities × strict/loose distance thresholds (the AP@0.5-vs-AP@0.7
  analogue from the parent study) × miss/ghost rates.
- **Done when:** metric implemented, unit-tested on a recorded bag, and its
  resolution bound (from Steps 0.1 + 0.5) stated.
- **Result:** _pending_

### Step 1.3 — ⚠️ PRE-REGISTER the Phase 1 gate, then run it  `⬜ TODO`
- **Rule to fix in this file BEFORE the run** (draft, to finalize with Phase 0 numbers
  in hand): under natural network latency, compare twin error with extrapolation
  disabled vs enabled (oracle-velocity arm). **NO-GO for the dynamics-aware thesis**
  if oracle kinematic correction recovers < 50% of the gap between the stale twin and
  the fresh-snapshot twin — displacement is then not the dominant error mechanism on
  real data, and the project pivots to the measurement contribution (C1–C3) alone.
- **Done when:** rule finalized and committed before the run; verdict written with
  the numbers that produced it.
- **Result:** _pending_

## Phase 2 — The AoI → twin-error mapping (RQ3 → C3)  `⬜ TODO`
Open loop, replay-first: recompute twin error offline from recorded campaigns across
RAT × payload × entity class; heterogeneous-age conditions (fresh infra + stale AMR
and permutations); four kinematic arms (none / oracle velocity / oracle displacement /
tracker velocity — the pre-registered design from `temporal_messaging/HANDOFF.md` §5,
on real data). Output: measured curves AoI vs twin error, and the divergence predictor
that beats AoI as an error predictor.

## Phase 3 — Divergence-triggered synchronization, closed loop (RQ4 → C4)  `⬜ TODO`
Entity-level update rule: transmit when predicted divergence exceeds a bound.
Baselines: periodic (several rates), AoI-optimal scheduling, send-everything.
Metric: twin-error-per-bit on the live network. Pre-register the comparison protocol
before the first closed-loop run.

## Phase 4 — Follow-ons  `⬜ TODO`
- **C5 (Option B):** couple to a network twin — per-link quality prediction feeding
  dual-RAT selection per source.
- **C6 (Option C):** channel-as-sensor change detection. ⛔ auto-dropped if D-CSI = NO.

---

## Progress log

| Date | Step | Notes |
|------|------|-------|
| 2026-08-16 | setup | Project skeleton: `INTRODUCTION.md` (thesis, gaps, C1–C6, RQ1–RQ5, positioning, 48 refs), `CLAUDE.md` (working rules), this tracker with Phase 0 steps and the two gating decisions (D-GT, D-CSI). |
