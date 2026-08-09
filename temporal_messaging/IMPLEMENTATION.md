# Implementation Tracker — Temporal Messaging

**Goal:** test whether making collaborator messages time-parameterized and
self-describing recovers the latency damage that no fusion operator can discount away.

**Read [`HANDOFF.md`](HANDOFF.md) first** — full context, prior findings, machine setup,
novelty accounting, and the pre-registered Phase 0 decision rule.

---

## How to use this file (do not delete)

Whenever a step finishes, update this file **in the same commit** as the work:
flip its status (`⬜ TODO` → `🟨 IN PROGRESS` → `✅ DONE` / `⛔ BLOCKED`), fill the
**Result** line with what actually happened (numbers, paths, surprises), append a dated
row to the progress log. Never mark ✅ without verifying the **Done when** criterion.

---

## Phase 0 — Go/no-go: is latency damage recoverable displacement?

### Step 0.1 — Pre-register the decision rule  `✅ DONE`
- **Result:** fixed in `HANDOFF.md` §5 before any run. NO-GO if oracle-velocity
  correction recovers <50% of the gap between uncorrected latency and the clean ceiling.

### Step 0.2 — Per-sender detection + tracking  `⬜ TODO`
- Run the single-vehicle detector on *each* CAV independently (the `nocomm` code path
  generalized from ego-only to per-agent), then a tracker per sender to obtain velocity
  estimates from data that sender actually has.
- **Done when:** per-(scenario, frame, cav) box + velocity records exist for the test split.
- **Result:** _pending_

### Step 0.3 — Four-arm latency comparison  `⬜ TODO`
- Arms: (A) zero-velocity / current, (B) oracle GT velocity, (C) sender-tracker velocity,
  (D) clean ceiling. Latency 1–10 frames. Late fusion, no retraining.
- **Done when:** AP@0.5/0.7 + precision/recall for all four arms at every latency level,
  3 seeds.
- **Result:** _pending_

### Step 0.4 — Read the verdict  `⬜ TODO`
- Apply the §5 rule. Record the B−C gap (concept vs. velocity-estimation engineering).
- Check for a velocity-error regime where correction *hurts* (misalignment valley).
- **Done when:** verdict written with the numbers that produced it.
- **Result:** _pending_

## Phase 1 — Object-state messaging (only if Phase 0 = GO)  `⬜ TODO`
Senders transmit tracked objects with velocity **and covariance**; receivers extrapolate
to their own timestamp before fusion. Evaluate on detection *and* tracking (task-agnostic
evidence). Baselines: stale late fusion, receiver-side compensation, fresh oracle.

## Phase 2 — Discountability without touching the operator  `⬜ TODO`
The sharpest claim (HANDOFF §4). Apply pre-corrected messages to **unmodified** pretrained
intermediate-fusion models. Target: **make F-Cooper latency-robust without modifying
F-Cooper** — the case where receiver-side weighting is provably ineffective.

## Phase 3 — Rate × age allocation  `⬜ TODO`
Given a fixed bit budget, fidelity now vs. extrapolatability later.
**Blocked on D1** (literature check — the novelty claim here is unverified).

## Phase 4 — Systems payoff  `⬜ TODO`
How much channel capacity does extrapolatability buy? Absorbs the deferred
freshness-scheduling direction (HANDOFF §7).

---

## Progress log

| Date | Step | Notes |
|------|------|-------|
| 2026-08-09 | setup | Branch and skeleton created; `HANDOFF.md` written with full context transfer from the completed failure-attribution study, Phase 6 negative result, machine setup, honest novelty accounting, and the pre-registered Phase 0 decision rule. |
