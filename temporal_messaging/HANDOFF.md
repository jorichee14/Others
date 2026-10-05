# HANDOFF — Temporal Messaging for Communication-Aware Collaborative Perception

**Read this first.** It is the complete context transfer for a new session. It covers
what the previous study established, what the machine looks like, what the new research
direction is, which novelty claims are verified vs. unverified, and the exact first
experiment with its pre-registered decision rule.

- Branch: `claude/collab-perception-temporal-messaging-k4mz7p`
- Parent branch (completed study): `claude/collab-perception-failure-analysis-s3bsij` (PR #1)
- Prior work lives in `collab_perception_failure_analysis/`; new work in `temporal_messaging/`.

---

## 1. What the previous study established (all numbers reproducible)

831 detection cells + 15 spatial cells + 27 tracking cells over OPV2V, 7 pretrained
architectures, no retraining. Pipeline validated: every published AP@0.7 reproduced to
±0.001 before any impairment work. Full analysis: `collab_perception_failure_analysis/results/ANALYSIS.md`.

**Reference points**

| | AP@0.7 |
|---|---|
| Ego-only floor (no collaboration) | **0.575** (P 0.825 / R 0.666) |
| Late 0.781 · Early 0.801 · AttFuse 0.815 · F-Cooper 0.790 | |
| V2VNet 0.822 · CoAlign 0.833 · CoBEVT 0.862 | |

**Findings that matter for the new direction**

1. **Delivery vs content.** Packet loss never drives any method below the floor (worst
   0.579 at 90% loss). Latency drives **all seven** below the floor at **100 ms**.
   Dropping 90% of messages beats delivering them 200 ms late.
2. **Latency ≈ staleness** mechanically: their NPD columns agree within 1.5 points for
   every method.
3. **THE KEY MEASUREMENT for the new direction — latency damage is displacement, not
   information loss.** At 100 ms latency:

   | method | ΔAP@0.5 | ΔAP@0.7 | ratio |
   |---|---|---|---|
   | AttFuse | −0.045 | −0.294 | 6.5× |
   | V2VNet | −0.038 | −0.334 | 8.8× |
   | CoAlign | −0.042 | −0.287 | 6.8× |
   | CoBEVT | −0.082 | −0.416 | 5.1× |
   | Late | −0.094 | −0.455 | 4.8× |
   | F-Cooper | −0.113 | −0.437 | 3.9× |

   Objects are still detected; they land ~2 m off (20 m/s × 100 ms). Ego motion is
   already compensated by OpenCOOD's pose transform (`cur_ego_pose_flag=True` uses the
   collaborator's *delayed* pose → ego's *current* pose), so static content is placed
   correctly and only **other objects' motion** causes the error.
4. **Mechanism principle.** Each fusion operator is as content-robust as its ability to
   discount an arriving message; its specific weakness is the impairment mimicking
   evidence it was trained to trust. Maxout (F-Cooper) discounts nothing → worst under
   all content impairments. CoAlign's alignment-robust training is the only defense
   that transfers across the misalignment family.
5. **Bandwidth is free to 4 bits** (≤0.05 AP for 6 of 7 methods), cliff at 2 bits.
   ⇒ ~8× spare bit budget available.
6. **Spatial (Step 4.3).** Delivery loss is surgical: 90% loss costs 0.50–0.59 occluded
   recall but only 0.06–0.08 ego-visible (~8:1). Latency contaminates the ego's own
   field of view: ego-visible recall −0.21…−0.46, ego-visible FPs ×3.4–4.6.
7. **Tracking (Phase 5).** Burst loss costs 15–23% more IDSW than i.i.d. at matched rate
   (detection is burst-blind). Staleness explodes IDSW 4–6×; constant latency leaves
   IDSW near-clean but dominates in misses ⇒ **motion models absorb consistent delay and
   amplify oscillating staleness**.
8. **All impairments were applied UNIFORMLY across collaborators.** Heterogeneous age
   (what a scheduler actually produces) is unmeasured — an open gap.

## 2. Phase 6 (blockage) — a NEGATIVE result, do not revive

Hypothesis: the vehicle occluding an agent's lidar also blocks its radio, so lost
messages are disproportionately valuable. Audit at 1,905 links:

- Pooled E[U|blocked] 2.64 vs E[U|clear] 1.97 → the script printed **GO**.
- **But stratified within scenario it does not hold.** Sign test 5 positive / 6 negative;
  scenario 11 (78 links, 5% of data) contributes +448 of a +363 weighted total;
  **removing it flips the sign to −0.06**. The largest scenario (s3, 332 links) points
  the opposite way (−1.09). 22% of links sit in scenarios with zero within-scene
  contrast, and they bias the pooled statistic (all-blocked scenes happen to be high-U,
  all-clear scenes low-U) — a textbook density confound.
- **Verdict: not supported in OPV2V.** The audit's GO/NO-GO logic tests only the pooled
  statistic; `scripts/analyze_blockage_audit.py` (on the parent branch) does the
  stratified version. Report as a negative result if written up.

## 3. Machine and environment

Run machine `wicomsrobot`, RTX 3080 12 GB, driver CUDA 13.0.

```
conda env: opencood (python 3.8)
torch 1.13.1+cu117 · spconv-cu117 2.3.6 · cumm-cu117 0.4.11 · numpy 1.23.5
!! the plain `cumm` package must NOT be installed (shadows cumm-cu117, breaks spconv.core_cc)
OpenCOOD commit 31ba16025da27ffe4e336f011290dfbc66f9a1f1  at ~/cpfa/OpenCOOD
```

| path | contents |
|---|---|
| `~/cpfa/OpenCOOD` | framework (run all scripts from here so `import opencood` resolves) |
| `~/cpfa/data/OPV2V/test` | 16 scenarios, 2,170 ego frames, 5,985 frame-CAV pairs, 10 Hz |
| `~/cpfa/checkpoints/` | 10 checkpoints (see `env/CHECKPOINTS.md` for md5s) |
| `~/cpfa/results/` | phase1, sweeps, spatial, tracking, blockage |
| `~/cpfa/Others` | this repo |

**Operational lessons (learned the hard way):**
- **One process per method** for Shapely-heavy runs. Heap fragmentation made a third
  method in the same process ~15× slower. Use `for m in ...; do python ... --methods $m; done`.
- Use DataLoader **workers** (seed inside `__getitem__`), never a manual in-process loop
  — that was the 10× difference in the sweep runner.
- Everything is resumable per unit of work; delete a result JSON to force a re-run.
- Everything is seed-deterministic (CRC32 of seed/scenario/frame/agent). The spatial tier
  was executed twice end-to-end and reproduced digit-for-digit.
- Warnings that are expected and harmless: `nn.functional.sigmoid is deprecated`,
  shapely `invalid value encountered in intersection`.

## 4. The new direction

### Thesis

> Collaborative perception transmits **snapshots** — "here is what the world looked like
> at my timestamp" — which carry no description of their own dynamics or validity. When
> a snapshot arrives late the receiver has no principled correction, so every system
> behaves as a **zero-velocity predictor**, the worst possible choice. Make messages
> **time-parameterized and self-describing** so any receiver can evaluate them at its own
> timestamp.

Two consequences, each attacking a finding from §1:

- **Decouples value from age** (finding 3: the information is intact, merely mis-timed).
- **Makes messages discountable** (finding 4: no operator discounts well; rather than
  making every receiver smarter, make the message self-describing — a sender-side,
  architecture-agnostic fix).

**Task-agnostic angle:** intermediate-fusion features are trained under a detection loss
— a task-specific encoding masquerading as a general message. A time-parameterized state
estimate is consumer-agnostic: detector, tracker, predictor, planner each query it at
their own timestamp. Testable across detection + tracking with the existing harness.

### Novelty — honest accounting

> **⚠️ SUPERSEDED 2026-08-09 by [`RELATED_WORK.md`](RELATED_WORK.md) §8.** The literature check
> this section asks for has been done. Outcome: candidate 3 (rate×age) and the task-agnostic
> angle are **dead**; candidate 2 (discountability) is **wounded** and survives only in a
> narrowed form; candidate 1 (the displacement diagnostic) **survives and is the only
> defensible headline**. The core mechanism — time-parameterized self-describing messages — is
> prior art in ETSI CPM (`timeOfMeasurement`), FFNet (NeurIPS 2023), and SparseCoop (AAAI 2026).
> Read §8–§9 there before writing anything. The text below is kept as the original record.

**NOT novel (state plainly in any write-up):**
- Transmitting object state with velocity — ETSI CPM standardizes this.
- Latency compensation in collaborative perception — SyncNet, CoBEVFlow, mmCooper, V2X-INCOP.
- Sender-side vs receiver-side placement — a conditioning argument, not a headline.

**Candidate novelty, ranked by confidence:**

1. **The displacement diagnostic** (finding 3). Highest confidence — it is measured, in
   hand, and reframes latency from destruction to correctable mis-timing. Implies the
   field's latency methods solve an unnecessarily hard problem.
2. **Discountability as a property of the message, not the operator.** The sharpest
   claim. Two independent results show receiver-side weighting *cannot* rescue a
   non-discounting operator: this study's mechanism finding, and AgentComm-Bench's
   ablation where staleness weight λ had literally zero effect on maxout at every value.
   Pre-correction is discounting-free and works with any operator.
   **Falsifiable target: make F-Cooper latency-robust without modifying F-Cooper.**
3. **The joint rate × age allocation question** — given a fixed budget, spend bits on
   fidelity now or extrapolatability later. ⚠️ **UNVERIFIED.** The claim "nobody has
   measured accuracy as a function of both" could not be checked (arxiv blocked in the
   authoring environment) and is probably too strong — V2X-ViT evaluates both compression
   and delay and may have a relevant ablation. **Do a real literature search before
   asserting this.**

**Expected reviewer attack:** *"This is late fusion with a Kalman filter, standardized in
2019."* Answer: the object-level version is the **baseline**, not the contribution. The
contribution is (a) quantifying that the representation the learning community adopted is
strictly worse under delay than the one it abandoned, (b) the hybrid that keeps feature
richness and gains extrapolatability, (c) the bit-allocation rule.

### Related work to position against (verify details; some are from memory)

SyncNet (Lei et al., ECCV 2022) and CoBEVFlow (Wei et al., NeurIPS 2023) both compensate
**receiver-side**, inferring motion from messages that arrived stale/sparse. Where2comm
(NeurIPS 2022) transmits sender-side *spatial* confidence for bandwidth selection — not
temporal validity. mmCooper uses receiver-side confidence-guided fusion for delays.
AgentComm-Bench (arXiv 2603.20285) proposes staleness-aware fusion weights and shows they
fail on maxout. ETSI CPM is the standards-side prior art for state transmission.

## 5. Phase 0 — the go/no-go audit (do this first, one day, no retraining)

Everything above is contingent on this. **Pre-register the decision rule before looking
at results**, exactly as Phase 6 did.

**Setup.** Late fusion at latency 1–10 frames (0.1–1.0 s). Current baseline is
AP@0.7 = 0.326 at 100 ms; floor 0.575; clean ceiling 0.781.

**Four arms:**

| arm | correction applied to each stale box |
|---|---|
| A. zero-velocity (current) | none — reproduces the existing 0.326 |
| B. oracle velocity | ground-truth object velocity from consecutive GT frames |
| C. sender-tracker velocity | velocity from a tracker run on the *sender's own* detections |
| D. clean ceiling | no latency |

**Decision rule (fix before running):**
- **NO-GO** if arm B recovers < 50% of the gap between A and D. Displacement is then not
  the dominant mechanism and the thesis dies for the cost of one script.
- **GO** if B recovers ≥ 50%. The B−C gap then quantifies how much of the remaining
  programme is velocity-estimation engineering rather than concept.

**Why it is cheap.** Everything exists: `run_phase1.py`'s `nocomm` mode is per-vehicle
detection; the Phase 5 Kalman tracker gives sender-side velocities; late fusion needs no
feature buffering and no retraining. Reuse `commchannel` unchanged for the latency.

**Watch for:** the misalignment valley (finding: moderate error is worse than severe)
means there may be a velocity-error regime where correction *hurts*. That is a finding,
not a failure — report it.

## 6. Open decisions for the next session

| # | Decision | Notes |
|---|---|---|
| ~~D1~~ | ~~Literature check on the rate×age claim~~ | **CLOSED 2026-08-09 — NO-GO.** See `RELATED_WORK.md` §5. |
| D2 | Object-level only, or feature-level hybrid? | Object-level is Phase 0/1; feature-level needs retraining |
| D3 | Which downstream tasks count as the task-agnostic evidence? | Detection + tracking exist; prediction would need new harness |
| D4 | Agent density | ~1.59 collaborators/ego frame; measure the histogram if scheduling re-enters |
| D5 | Does the study stay on OPV2V, or add V2X-Sim / DAIR-V2X / V2V4Real? | Real-data validation strengthens it; costs setup |

## 7. Deferred / rejected

- **Phase 6 blockage** — negative result, do not revive without real-radio data (§2).
- **Freshness scheduling** — was the initially chosen direction; superseded because
  correctable staleness changes the scheduler's cost function. Re-enters as a later phase
  (RQ3: how much channel capacity does extrapolatability buy?). Constraints already
  identified: 10 Hz quantum means only the 100 ms–1 s regime is representable; agent
  density is thin (see D4); scheduling *message units* rather than whole messages is the
  escape if agent counts are too low.
- **Sub-100 ms transport latency** — not representable at 10 Hz. Scope out explicitly.
