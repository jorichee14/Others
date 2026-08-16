# PulseSync — Freshness-Stratified Synchronization for Dynamic Digital Twins

**Status: proposed method (2026-08-16). This is the paper's contribution; the
characterization and twin-v0 work in `IMPLEMENTATION.md` are instrumentation for it.**

---

## 1. Premise

The parent study established that staleness damage is **displacement**: a stale update
is not wrong, it is mis-timed, and the error is `velocity × age`
(`collab_perception_failure_analysis/results/ANALYSIS.md`, ΔAP@0.5 vs ΔAP@0.7 =
3.9–8.8× at 100 ms) [V]. A twin fed by delayed rich sensors is therefore wrong for
exactly one reason: it does not know each entity's **velocity** well enough to
extrapolate across the delay.

The testbed carries radar on both AMRs and the infrastructure nodes. Radar measures
Doppler — instantaneous radial velocity — in a few bytes with near-zero processing
latency. **The cheapest, freshest sensor in the system directly measures the quantity
whose absence makes staleness harmful.** PulseSync is the protocol built on that
coincidence.

## 2. The method

Synchronization is split into two streams with opposite freshness/richness profiles:

| stream | content | size | age | rate |
|---|---|---|---|---|
| **Anchor** | full detections (LiDAR/RGB-D): identity, class, geometry, precise pose | heavy | high (payload + inference + link) | **event-triggered, rare** |
| **Pulse** | per-entity kinematic evidence: radar Doppler/range, odometry, optionally channel-state variance | bytes/entity | near-zero | continuous, high rate |

Three mechanisms at the twin:

### 2.1 Pulse-corrected extrapolation
The twin propagates each entity with a motion model whose **velocity is continuously
corrected from the pulse stream**, even while anchors are stale. Twin error becomes
bounded by pulse noise and maneuver rate — decoupled from anchor age. Anchors are
needed only for what pulses cannot carry: identity, class, shape, new entities.

### 2.2 Measured-divergence triggering
Age-of-Incorrect-Information scheduling has a known observability paradox: the
receiver cannot know its state is incorrect without the very update it is deciding to
request, so AoII policies fall back to *modeling* incorrectness. PulseSync makes
incorrectness **observable**: anchor requests fire on *measured* innovation —
- Doppler disagreement with the extrapolated velocity (maneuver detected),
- a radar return with no matching twin entity (birth detected),
- pulse silence where an entity is predicted (death/occlusion candidate).

Claim, stated precisely: *a cheap always-fresh side-channel converts event-triggered
twin synchronization from model-driven to measurement-driven, resolving the AoII
observability paradox physically.*

### 2.3 Validity-deadline dual-RAT dispatch
Each entity carries a validity horizon `τ_i` = time until predicted twin error
exceeds `ε` given current velocity uncertainty. Anchor dispatch chooses the transport
per entity from measured link-latency distributions: `τ_i` inside Wi-Fi's p95 latency
⇒ 5G; otherwise Wi-Fi. Bandwidth contention across entities is resolved by expiring
validity (earliest-deadline-first in `τ`). Transport selection is driven by scene
kinematics, not flow-level QoS.

### 2.4 Message format
The pulse message is essentially the kinematics + `timeOfMeasurement` subset of ETSI
CPM's Perceived Object Container — the method slots into the standards-track message
rather than inventing a format. State this in any write-up; it is a deployability
argument and an honesty requirement (the *fields* are prior art; the *protocol* is
the contribution).

## 3. Falsifiable headline target

**Latency equivalence:** make the Wi-Fi-fed twin match the 5G-fed twin's accuracy
while using a fraction of the anchor bandwidth. (Direct descendant of the parent
programme's "make F-Cooper latency-robust without modifying F-Cooper".)

Secondary target: hold twin error constant while cutting anchor bandwidth ×N
(measure N; N≈1 kills the method).

## 4. Novelty accounting (honest, per house convention)

| # | Claim | Verdict / threat |
|---|---|---|
| 1 | Freshness-stratified two-stream sync — a fresh cheap modality keeps a stale rich modality extrapolatable | **Core claim.** Nearest: Doppler-guided masking in multi-agent perception (SlimComm-family) uses Doppler for *bandwidth selection*, sender-side [S]; staleness-robust on-vehicle fusion (arXiv 2506.05780) hardens the model, not the protocol [S]; FFNet transmits motion but learned, feature-level, same-sensor [S]. Delta = protocol-side + cross-modal + zero-learning. Must be stated exactly so. |
| 2 | Measured-divergence triggering (AoII observability paradox resolved by side-channel) | **Sharpest theoretical claim. ⚠️ UNVERIFIED** — event-triggered remote estimation with a secondary observation channel may exist; targeted search on "hybrid / two-channel event-triggered estimation", "sensor scheduling with side information" is MANDATORY before asserting. |
| 3 | Validity-deadline dual-RAT dispatch | Likely novel as stated — multi-RAT steering literature is flow/QoS-based [S]; nothing found doing per-entity kinematic-deadline transport assignment. |
| 4 | Divergence/threshold-triggered sync *alone* | **NOT NOVEL — never claim.** Classical event-triggered estimation (threshold policies provably optimal, e.g. covariance-threshold results); an adaptive-fidelity event-triggered DT paper already exists [S]. PulseSync triggers must always be presented as *measurement-driven*; without the pulse stream the contribution collapses into 2015-era control theory. |

## 5. Evaluation design

- **Baselines:** periodic sync at matched bandwidth; AoI-optimal scheduling;
  **classical model-triggered sync** (identical trigger thresholds, no pulse stream —
  the ablation that isolates mechanism 2.2 and proves the method rather than the
  trigger); everything-at-full-rate over 5G as ceiling.
- **Metric:** twin-error-per-bit (per-entity error metric from `IMPLEMENTATION.md`
  Step 4.1), plus the latency-equivalence test of §3.
- **Conditions:** Wi-Fi vs 5G vs mixed; static vs moving AMRs; with/without human
  motion in scene.
- **Pre-registration:** fix the latency-equivalence success criterion and the ×N
  bandwidth threshold before the first closed-loop run.

## 6. Risks and gates

1. **D-Doppler (replaces D-CSI as the critical hardware gate):** do the radar units
   output usable per-target Doppler through their drivers? Check first, one afternoon.
   Fallback: pulses = high-rate compressed centroids from any cheap detector —
   preserves freshness stratification, weakens the elegance and the AoII claim.
2. **Two-channel estimation literature check** (novelty claim 2) — do before writing.
3. **Indoor dynamics may be too slow for staleness to matter over 5G.** This is why
   the Wi-Fi link and latency-equivalence framing are load-bearing: the method's value
   shows where the link is bad. If nothing matters even over Wi-Fi, that is the RQ2
   NO-GO firing correctly — pivot per `IMPLEMENTATION.md` Step 4.2.
4. Radial-only velocity (Doppler) observes one component; multiple radar viewpoints
   (2 AMRs + infrastructure) restore the full vector — this is quietly a second use of
   the multi-node testbed and worth a sentence in the paper.

## 7. Relation to the tracker

`IMPLEMENTATION.md` milestones map onto PulseSync as: M1–M2 build the twin and the
anchor path; M3 measures the link distributions that mechanism 2.3 consumes
(instrument calibration, not a contribution); M4's four-arm experiment measures the
extrapolation headroom PulseSync exploits (arm C ≈ pulse-corrected extrapolation with
tracker velocities); M5 = PulseSync closed loop vs the §5 baselines. Add D-Doppler to
M1 alongside D-GT.
