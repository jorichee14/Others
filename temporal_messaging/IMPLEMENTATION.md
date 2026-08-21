# Implementation Tracker — Temporal Messaging

**Goal:** test whether making collaborator messages time-parameterized and
self-describing recovers the latency damage that no fusion operator can discount away.

> **Direction revised 2026-08-09** after the literature check
> ([`docs/LITERATURE.md`](docs/LITERATURE.md)). The goal above is still the *hypothesis*,
> but the architecture it implied already exists (StreamingFlow, CVPR 2024, intra-vehicle).
> What survives is the **network-specific** half: out-of-order arrival, unbounded per-agent
> age, permanent loss, hard bit budget. See [`METHOD.md`](METHOD.md) for the current
> position, the three mechanisms (M1 out-of-sequence update, M2 predictive-residual
> messaging, M3 belief divergence), and the pre-stated bars from the literature.

**Read order:** [`METHOD.md`](METHOD.md) → [`HANDOFF.md`](HANDOFF.md) (prior findings,
machine setup) → [`docs/LITERATURE.md`](docs/LITERATURE.md) (reference base).

---

## How to use this file (do not delete)

Whenever a step finishes, update this file **in the same commit** as the work:
flip its status (`⬜ TODO` → `🟨 IN PROGRESS` → `✅ DONE` / `⛔ BLOCKED`), fill the
**Result** line with what actually happened (numbers, paths, surprises), append a dated
row to the progress log. Never mark ✅ without verifying the **Done when** criterion.

---

## Phase 0 — Go/no-go: is latency damage recoverable displacement?

> **Role changed 2026-08-09.** This is no longer "does the thesis live or die". It is now
> (a) the **linear-advection ablation the literature demands** — CoDynTrust (ICRA 2025)
> reached SOTA with plain per-ROI velocity × delay, so any learned dynamics model must be
> shown to beat linear extrapolation; and (b) a cheap sanity check on the premise. It is
> still worth one day, still pre-registered, and arm C is now the *baseline to beat* rather
> than an engineering footnote. Run it before writing any model code.

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

## Phase 0.5 — Read the four unread decisive papers  `⬜ TODO`
CoST (arXiv:2508.00359 — the "isn't this just…" paper), CooperTrim (2602.13287), the AoI
alignment paper (2602.13439), V2X-DSC (2602.00687). Plus a search on **out-of-sequence
measurement (OOSM) filtering**, which the check did not cover at all and which is M1 in
linear-Gaussian form. Also read StreamingFlow end to end.
- **Done when:** a differentiation paragraph against CoST exists, in writing, by name; and
  the OOSM prior art is either cited or shown absent. Decisions D1b, D1c closed.

## Phase 1 — Belief retention vs. drift (the experiment most likely to kill the method)  `⬜ TODO`
Build the **discrete Δt-conditioned recurrent BEV belief** (not an ODE — `METHOD.md` §5),
train it on OPV2V, and measure **M3**: divergence `d(z_i, z_j)` across agents, and belief
quality vs. time-since-last-message and vs. cumulative lossy integrations, swept through
the parent study's impairment matrix. The **ego-only floor test** is the drift detector.
- **Why first:** V2XPnP reports error accumulation when lossy intermediate features are
  transformed across time. If drift dominates, retention is a liability and this is a
  negative result — cheaper to learn now than after building M1 and M2 on top.
- **Pre-register** the drift threshold before running, per rule 3.
- **Blocked on D7** (OPV2V train split, ~100 GB; 12 GB card → small-model config).

## Phase 2 — M1: learned out-of-sequence update  `⬜ TODO`
The technical core. Three arms: rewind–replay (exact, expensive), learned retrodiction
`U_back(z(t), m, τ)` (the research object), forward-warp (what CoBEVFlow / TraF-Align do).
Trained on the arrival-order statistics the channel actually produces.
- **Bar:** TraF-Align, −4.87% AP50 at 400 ms on V2V4Real (checkpoints released, built on
  OpenCOOD). Get them and run them inside `commchannel`.
- **Mandatory ablation:** learned dynamics vs. linear advection (Phase 0 arm C).

## Phase 3 — M2: predictive-residual messaging  `⬜ TODO`
Residual against the receiver's *predicted* belief, not its last observation. Measure that
bit-rate scales with age by construction. Evaluate the O(N²) per-receiver state against the
O(N) broadcast-common-belief fallback.
- **Bar:** instance-query bandwidth, not dense transmission — INSTINCT 1/281 and 1/264,
  CoCMT 0.416 Mb. Stating headroom against dense feature maps is a straw man.

## Phase 4 — Loss and the systems payoff  `⬜ TODO`
The strongest surviving claim: loss is a non-event by construction (retention, not
reconstruction — contrast V2X-INCOP, which predicts the missing message).
- **Bar:** V2X-INCOP, +14.06% cooperative gain on OPV2V averaged over drop rates — same
  dataset, direct comparison.
- Absorbs the deferred freshness-scheduling direction (HANDOFF §7) and the time-queryable
  readout evaluated on detection **and** tracking.

---

## Progress log

| Date | Step | Notes |
|------|------|-------|
| 2026-08-09 | setup | Branch and skeleton created; `HANDOFF.md` written with full context transfer from the completed failure-attribution study, Phase 6 negative result, machine setup, honest novelty accounting, and the pre-registered Phase 0 decision rule. |
| 2026-08-09 | D1 resolved | Literature check run (16 searches + StreamingFlow full text), stored as `docs/LITERATURE.md`. **It falsified the proposed architecture's novelty**: StreamingFlow (CVPR 2024) is persistent BEV latent + GRU-ODE dynamics + async update at own timestamp + time-query readout, intra-vehicle; CoST (ICCV 2025) has memory bank + change-only transmission. Also falsified: "CP is stateless" (SCOPE/V2XPnP/CoST/V2X-INCOP), "CoBEVFlow discretises time", and the strength of the latency premise (TraF-Align: −4.87% AP50 at 400 ms). |
| 2026-08-21 | baseline audit | `docs/BASELINE_AVAILABILITY.md`. **V2X-INCOP has no public code** (verified via arXiv, authors' GitHub, general search) — cannot be run; matched its metric instead with `scripts/compare_incop_protocol.py`, since the ego-only floor *is* its "individual perception" baseline. Result: all seven unmodified baselines exceed INCOP's reported +14.06% OPV2V gain (range +20.4% to +34.2% i.i.d., +21.8% to +35.4% bursty). Recorded as a **protocol tension, not a refutation** — four questions listed that need the paper's full text. **Where2comm** is OpenCOOD-based and attachable but ships only `dair-v2x/` configs and no OPV2V checkpoint, so it needs a config plus a training run. **V2X-ViT and DiscoNet** are free from HEAL's zoo pending a spconv 1.2.1 vs 2.3.6 compatibility check. |
| 2026-08-09 | direction revised | `METHOD.md` written. Novelty relocated to the three network-specific mechanisms — M1 learned out-of-sequence update, M2 predictive-residual messaging, M3 belief divergence — with the pre-stated bars from TraF-Align / INSTINCT / V2X-INCOP / CoDynTrust. `HANDOFF.md` §4 marked superseded; phases restructured; Phase 0 repurposed as the mandatory linear-advection ablation. |
