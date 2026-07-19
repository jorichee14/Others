# Research Proposal

## Safety-Budgeted Collaborative Perception: Certified, Failure-Robust Communication for Multi-Agent Autonomous Driving

*Prepared 2026-07-19. Companion documents: `NOVELTY_ANALYSIS.md` (competitive landscape), `RESEARCH_DIRECTION.md` (the direction and its references).*

---

## Abstract

Collaborative perception (CP) lets connected vehicles see through occlusions by sharing intermediate features over wireless links, but every deployed system faces a bandwidth–accuracy trade-off and optimizes it as if the channel were a single, reliable, always-on pipe. It is not: vehicles carry multiple heterogeneous radios (WiFi 2.4/5 GHz today, WiFi + 5G tomorrow) that differ in rate and reliability, and any link can drop transiently during exactly the high-mobility moments perception matters most. Worse, the importance estimates that decide what to transmit come from neural networks with no calibration guarantee, so a system can silently drop a safety-critical detection and never know its own miss-rate. We propose **safety-budgeted collaborative perception**: a closed-loop framework that spends the *minimum* communication needed to hold a *certified* safety level and reallocates across heterogeneous links as conditions change. The certificate is provided by conformal risk control, but the contribution is not certification per se — it is (i) a distribution-free guarantee that **survives link failure**, made non-trivial because a dropping link is a deliberate *intervention* that breaks the exchangeability standard conformal assumes; (ii) **certified graceful degradation** — provably constant safety at variable bandwidth as links worsen; and (iii) a **joint select-and-route co-design** in which the certificate *sizes the reliable-link budget* and the budget shapes the routing, making the two inseparable. The result reframes CP's deliverable from "X% accuracy at Y kilobytes" to "provably ≤ α missed-safety-critical rate under best-effort link failure, at minimum guaranteed-link bandwidth."

---

## 1. Problem and motivation

**The scenario.** At an occluded intersection, an ego vehicle's view of a crosswalk is blocked by a parked truck; a collaborator (another vehicle or a roadside sensor) sees it, and a pedestrian is stepping out. CP shares compressed feature-map **units** (grid cells / tokens) so the ego fuses the collaborator's view and detects the pedestrian. This is real and standardized (ETSI Collective Perception Service).

**Two false assumptions that break in the field:**

1. **"One reliable channel."** Deployed agents have *multiple heterogeneous radios* with sharply different rate/reliability, and individual links **fail transiently** (fading, obstruction, congestion). The field optimizes for a single abstract pipe; reality provides neither reliability nor singularity.
2. **"Trustworthy importance scores."** What to transmit is decided by learned importance estimates that are typically **miscalibrated**. A system that drops a unit "because it looked unimportant" is trusting an uncertified number. When it is wrong, a safety-critical object is silently lost and **no current system can state the probability that this happened.**

**Why it matters, and why now.** A missed occluded pedestrian is a crash; "500× bandwidth reduction" is meaningless without a bound on what it does to miss-rate under link failure. Safety auditors and regulators need a *guarantee*, not a benchmark average. And the moment is ripe: communication-efficient CP has matured into a crowded field (Where2comm, JigsawComm, HydraCollab, CoSDH, V2X-DSC), while **conformal risk control** — a distribution-free, finite-sample guarantee machinery — matured in parallel. The two have never been connected. That intersection is the opportunity, and it is closing.

---

## 2. Background and the precise gap

**Communication-efficient CP** decides *what* to send: spatial-confidence selection (Where2comm), learned select-and-encode with cross-agent redundancy removal (JigsawComm), supply–demand region selection (CoSDH), adaptive intermediate/late hybrid on confidence overlap (HydraCollab), distributed-source-coding conditional codecs exploiting receiver side-information (V2X-DSC), diffusion-based generative reconstruction (DiffCP). **All are empirical, single-channel, and certify nothing.**

**Multi-radio / multipath networking** steers traffic across links (hybrid DSRC/C-V2X, ATSSS) but **content-blind**, or uses a second path to **duplicate** important content for *reliability* (content-aware MPTCP, MPR-QUIC) — never to *offload for capacity*.

**Distribution-free uncertainty quantification** (Conformal Risk Control; Risk-Controlling Prediction Sets; Learn-Then-Test) certifies model behavior with no distributional assumptions. In driving it has been applied **only to outputs** — cooperative trajectory prediction, multi-object tracking, bounding boxes — **never to the communication policy.**

**The unoccupied gap:** no system co-decides *what survives* and *which physical link carries it*, spends the minimum guaranteed-link bandwidth for a **certified** miss-rate, and holds that guarantee **when the best-effort link fails**. Standard conformal cannot be dropped in, because link failure is an intervention that violates exchangeability — resolving that is a research contribution, not an application.

---

## 3. Research questions and hypotheses

- **RQ1 (foundation).** Is receiver-side *reconstructability* genuinely distinct from mere cross-agent *overlap*, and does it predict which units are safe to offload to a droppable link? **H1:** calibrated reconstructability (conditional-codec proxy) diverges from HydraCollab's overlap mask and better predicts fusion-innovation loss.
- **RQ2 (jointness).** Does co-deciding selection and physical-link routing beat a staged "select-then-steer" pipeline, specifically under link failure and tight reliable budget? **H2:** yes, concentrated in the degraded / low-budget regime.
- **RQ3 (certification under intervention).** Can a distribution-free miss-rate guarantee be made to hold under a *chosen* best-effort-link-failure intervention? **H3:** yes, via link-state-mixture calibration with covariate-shift-robust conformal.
- **RQ4 (efficiency of the guarantee).** At a fixed certified safety level, does the co-design use the least guaranteed-link bandwidth vs. baselines, and does it degrade gracefully? **H4:** yes; certified constant safety at monotonically increasing bandwidth as links worsen.

---

## 4. Proposed approach

### 4.1 System model
Decentralized, asymmetric agents; directed pair `j → i` (collaborator → ego). Sender `j` decomposes features into units `u` with payload `b(u)`. Two physical radios per pair: reliable low-rate `r` (failure ≈ 0, capacity `C^r` small), best-effort high-rate `b` (failure prob `p^b`, capacity `C^b` large). Each unit is assigned `a(u) ∈ {∅ (drop), b, r}`.

### 4.2 The two borrowed signals (inputs, not claims)
- **Importance** `Împ(u)` — marginal task value (detector confidence / ablation saliency; Where2comm-style).
- **Reconstructability** `R̂ec(u) ∈ [0,1]` — receiver-side conditional predictability from ego side-information (temporal history, priors, own view), estimated with a **V2X-DSC-style conditional codec** or diffusion prior. Must be strictly broader than confidence overlap (see RQ1).

### 4.3 Contribution A — the joint select-and-route co-design (the actuator)
Route by weight `w(u) = Împ(u)·(1 − R̂ec(u))`: fill the reliable link with highest-`w` (critical *and* irreplaceable) units; offload reconstructable content to best-effort (free if it arrives, no loss if it drops — the ego regenerates it); drop the rest. This **inverts** the multipath norm: the second link is used for *capacity*, not duplication.

### 4.4 Contribution B — the certificate as controller (not a wrapper)
Define the **degraded risk**
`R(λ) = E[ ρ(A_λ) ]`, where `ρ` = fraction of safety-critical ground-truth objects the ego fails to detect after fusion, `A_λ` = units routed to the *reliable* link under threshold policy `π_λ`, and the expectation is taken **conditioned on best-effort failure**. `R(λ)` is monotone in reliable-inclusion aggressiveness. Calibrate `λ̂` by **Conformal Risk Control** (single knob) or **Learn-Then-Test** (selection + offload + budget knobs) so that, distribution-free and finite-sample:
`E[ ρ(A_{λ̂}) ] ≤ α + O(1/n)`
for operator-chosen safety level `α`. The certificate **sizes the reliable-link budget**; minimizing `Σ_{u∈A} b(u)` subject to it gives *minimum guaranteed-link bandwidth for certified safety*. This is the inseparable co-design: certificate → budget → routing.

### 4.5 Contribution C — certification under a link-failure intervention (the theory)
Vanilla conformal assumes calibration/test exchangeability; a dropping link is a deliberate intervention that breaks it. We calibrate `R(λ)` under a **mixture over link states** and correct the induced covariate shift with **weighted / online conformal** (covariate-shift conformal; adaptive conformal for the streaming V2X setting). Result: a guarantee that **survives** total or stochastic best-effort outage — the rigorous form of "task-sufficiency under failure," and a distribution-free guarantee under a controllable intervention, which is novel beyond CP.

---

## 5. Contributions (summary)

1. **Safety-budgeted communication** — a closed loop that spends minimum guaranteed-link bandwidth for a certified miss-rate; reframes CP's deliverable from accuracy-at-bandwidth to certified-safety-at-minimum-bandwidth.
2. **Conformal risk control under a communication intervention** — a distribution-free guarantee that holds when the best-effort link fails (theory).
3. **Certified graceful degradation** — provably constant safety at variable bandwidth as links worsen (system behavior).
4. **Capacity-offload joint select-and-route** — the second radio used for capacity, not duplication, as the certified actuator.

Explicitly *not* claimed: importance/redundancy selection (JigsawComm/CoSDH), reconstructability coding (V2X-DSC), two radios (hybrid V2X), conformal UQ itself (Angelopoulos et al.). The novelty is their intersection under failure.

---

## 6. Evaluation plan

**Datasets.** OPV2V, DAIR-V2X, V2X-R (real 4D-radar + LiDAR), V2X-Sim. Paired-view real data for headline results; synthetic complementary-masking only for the Phase-0 sanity check.

**Baselines.** Where2comm (importance-only), JigsawComm (select+encode), HydraCollab (adaptive intermediate/late — public code), V2X-DSC (conditional codec), and a "staged select-then-steer" ablation of our own method.

**Metrics.** Missed-safety-critical rate (primary), AP@0.7, reliable-link bytes, end-to-end latency. Conditions: **nominal** (both links up) and **degraded** (best-effort dropped, total and stochastic `p^b`).

**Go/no-go gates (run in order; each can kill the project cheaply):**
- **G1′ — the foundation (Phase 0, one afternoon).** Does calibrated `R̂ec` diverge from HydraCollab's overlap mask `1(c_i≥λ)⊙1(c_j≥λ)` and better predict fusion-innovation loss? *If no → stop.*
- **G1 — axes distinct.** `corr(Împ, R̂ec) < 0.6`; high-Imp/high-Rec cell > ~10% of important units.
- **G2 — jointness matters.** Joint ≫ staged in the degraded / low-budget regime.
- **G3 — guarantee real.** Empirical miss-rate ≤ α under outage across the α-sweep (conformal coverage plot); joint retains more task performance than importance-only at equal reliable budget.

**Money plot.** Certified vs. empirical miss-rate across α, under best-effort outage: ours stays under the line where every baseline's miss-rate blows up; companion plot shows least reliable-link bytes at each α.

---

## 7. Timeline

| Phase | Duration | Deliverable | Gate |
|---|---|---|---|
| 0 — Foundation | 2 weeks | Reconstructability-vs-overlap script; axis scatter/correlation on real frames | G1′, G1 |
| 1 — Actuator | 6 weeks | Joint select-and-route optimizer; nominal/degraded routing on one dataset | G2 |
| 2 — Certificate | 8 weeks | Conformal-risk-control calibration under link-failure mixture; coverage validation | G3 |
| 3 — Scale & robustness | 8 weeks | Full multi-dataset sweep; covariate-shift/online conformal; latency budget | — |
| 4 — Writeup | 4 weeks | Paper + released code | — |

---

## 8. Risks and mitigations

- **Exchangeability void under cross-city shift** → covariate-shift-weighted and online/adaptive conformal (long-run-average guarantee under drift); fallback to per-deployment marginal guarantee (weaker but still novel).
- **Non-monotone risk in `λ`** (fusion can hurt when a bad feature is added) → use Learn-Then-Test (no monotonicity needed) or 2026 non-monotone-loss CRC.
- **Reconstructability = overlap (G1′ fails)** → the project's true kill-switch; run first. Fallback: reposition on the certificate-under-intervention theory alone with a single-link actuator.
- **Codec latency** (conditional/diffusion Rec estimator) → report end-to-end latency explicitly; consider amortized/temporal Rec.

---

## 9. Expected outcomes and impact

- The first collaborative-perception system that **certifies its communication policy** with a distribution-free, failure-robust safety guarantee.
- A methodological result — **conformal risk control under a controllable communication intervention** — reusable beyond CP (any system that sheds load under resource failure).
- A deployment-relevant behavior — **certified graceful degradation** — that safety auditors and regulators can consume directly.
- Public code and a reproducible benchmark for *certified* CP under link failure.

---

## 10. References (verified July 2026)

**Distribution-free guarantee machinery**
- Angelopoulos, Bates, Fisch, Lei, Schuster — *Conformal Risk Control* — https://arxiv.org/abs/2208.02814
- Bates, Angelopoulos, Lei, Malik, Jordan — *Distribution-Free, Risk-Controlling Prediction Sets* (JACM) — https://dl.acm.org/doi/10.1145/3478535
- Angelopoulos, Bates, Candès, Jordan, Lei — *Learn Then Test* (AoAS 2025) — https://arxiv.org/abs/2110.01052
- Angelopoulos & Bates — *A Gentle Introduction to Conformal Prediction and Distribution-Free UQ* — https://arxiv.org/abs/2107.07511
- *Conformal Risk Control under Non-Monotone Losses* (2026, fallback) — https://arxiv.org/pdf/2604.01502

**Conformal in driving — outputs only (the gap we fill)**
- *Conformal Trajectory Prediction with Multi-View Data in Cooperative Driving* — https://arxiv.org/html/2408.00374v1
- *Collaborative Multi-Object Tracking with Conformal Uncertainty Propagation* — https://arxiv.org/pdf/2303.14346
- *Beyond Confidence: Dual-Threshold Conformal Prediction for Autonomous System Perception* — https://arxiv.org/pdf/2502.07255
- *Conformal Object Detection by Sequential Risk Control* — https://arxiv.org/html/2505.24038v1

**Communication-efficient collaborative perception (actuator / baselines)**
- Where2comm (NeurIPS 2022) — https://arxiv.org/abs/2209.12836
- JigsawComm (2025) — https://arxiv.org/abs/2511.17843
- HydraCollab (2026) — arXiv 2607.00191 · https://github.com/AICPS/HydraCollab
- CoSDH (CVPR 2025) — https://arxiv.org/abs/2503.03430
- V2X-DSC (2026, Wyner–Ziv conditional codec = the `Rec` estimator) — https://arxiv.org/abs/2602.00687
- DiffCP (diffusion generative reconstruction) — https://arxiv.org/abs/2409.19592
- Rate-Distortion Optimized Communication for CP — https://arxiv.org/pdf/2509.21994

**Communication realism / robustness (motivation)**
- ETSI CPS redundancy-mitigation survey (VoI, TS 103 324) — https://arxiv.org/abs/2501.01200
- Interruption-Aware Cooperative Perception (V2X-INCOP) — https://arxiv.org/abs/2304.11821
- MPR-QUIC (partially-reliable priority multipath, contrast) — https://www.sciencedirect.com/science/article/abs/pii/S1383762124001322
