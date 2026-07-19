# Strong Research Direction — *Risk-Controlled Collaborative Perception*

A cutting-edge, defensible reframe of the dual-link concept, grounded in the July-2026 literature.
Companion to `NOVELTY_ANALYSIS.md`. Prepared 2026-07-19.

---

## The thesis (one sentence)

> **Use conformal risk control to certify the *communication policy* of a collaborative-perception system — jointly selecting and physical-link-routing feature units so that the ego's rate of missed safety-critical objects is provably ≤ α, distribution-free and finite-sample, *even when the best-effort radio fails and the importance estimator is miscalibrated*.**

Your original dual-link idea becomes the **actuator**; the **contribution moves up a level** — from "a better joint optimizer" (crowded: HydraCollab, JigsawComm, CoSDH) to "the first CP system with a distribution-free task-risk guarantee on what it transmits" (empty).

---

## 1. Why this is the strong bet

Two mature bodies of work have **never been combined**, and their intersection is exactly the whitespace:

- **Communication-efficient collaborative perception** (what/where/how-much to send): Where2comm, JigsawComm, CoSDH, HydraCollab, V2X-DSC, DiffCP, What2Keep, InfoCom. *All empirical. None offers a guarantee.*
- **Conformal risk control / distribution-free UQ** (certify a decision with finite-sample guarantees, no distributional assumptions): Angelopoulos et al., Bates et al. (RCPS), Learn-Then-Test. *Applied in driving only to **outputs** — cooperative trajectory prediction, MOT, bounding boxes — never to the **communication policy**.*

**The move nobody has made:** conformal methods have always calibrated *what the model predicts*. Here they calibrate *what the network transmits and over which link*. That single reframe is the paper.

**Why it is defensible (hard to scoop):**
1. It requires *both* CP-systems knowledge *and* conformal-statistics machinery — few groups hold both, so the field won't casually converge on it the way it is converging on selection.
2. It **repairs the exact weakness** your note flagged as its Achilles heel. Your §6 rider: *"Imp/Rec are estimated with error δ, so the real bound is ε + g(δ) and demands calibrated estimators."* Conformal risk control **eliminates that dependency**: the guarantee `R ≤ α` holds regardless of how badly the importance/reconstructability estimators are calibrated, because calibration is done post-hoc on held-out data. You go from a *model-dependent* bound to a *distribution-free* one. That is a category upgrade, not an increment.
3. Safety framing gives it impact: the certified quantity is *"probability the ego fails to perceive an occluded pedestrian after fusion,"* under a link that can drop — a metric a regulator/AV-safety audience actually wants.

---

## 2. The precise problem

Decentralized agents; directed pair `j → i`. Sender `j` decomposes features into units `u`, each with importance-proxy `Împ(u)` and reconstructability-proxy `R̂ec(u)` (estimated via a **V2X-DSC-style conditional codec** — do *not* claim these as novel; cite them). Two physical radios: reliable low-rate `r` (failure ≈ 0), best-effort high-rate `b` (failure prob `p^b`). A communication policy `π_λ` parameterized by threshold(s) `λ` maps each unit to `{drop, b, r}`.

**Certified risk.** Let the safety-critical miss-rate under receiver-held set `R` be
`ρ(R) = (# ground-truth safety-critical objects the ego fails to detect after fusion) / (# safety-critical objects)`.
Define the **degraded risk** `R(λ) = E[ ρ(A_λ) ]` where `A_λ` = the units routed to the *reliable* link under `π_λ` — i.e. the risk **conditioned on total best-effort failure**. `R(λ)` is monotone non-increasing in "reliable-inclusion aggressiveness" (lower threshold ⇒ more critical content on the guaranteed link ⇒ fewer misses).

**Goal.** Pick `λ̂` from calibration data so that, distribution-free and finite-sample,
`E[ R(λ̂) ] ≤ α`, **while minimizing reliable-link bytes** `E[ Σ_{u∈A} b(u) ]`.

This is the rigorous replacement for the note's hand-set constraint `D(A) ≤ ε`: **`α` is the operator-chosen safety level; `λ̂` is certified by conformal risk control, not guessed.**

---

## 3. Three contributions

1. **Conformal certification of a communication policy (the core novelty).** Cast reliable-link partitioning as a monotone-risk-controlled decision and calibrate the routing threshold(s) with **Conformal Risk Control** (single knob) or **Learn-Then-Test** (multiple knobs: selection threshold, offload threshold, per-pair budget). Result: `P`-level or expectation-level guarantee on missed-critical-detection rate, holding for *any* pretrained detector and *any* Imp/Rec estimator.

2. **Failure-conditioned & shift-robust calibration (the methodological contribution).** Link failure is an *intervention*, not exchangeable noise, so vanilla CRC does not directly apply. Contribution: calibrate the degraded risk `R(λ)` under a **mixture over link states** and handle the induced covariate shift with **weighted / robust conformal** (Tibshirani et al. covariate-shift conformal; PID/adaptive conformal for the online V2X stream). This yields a guarantee that *survives the best-effort link dropping* — the rigorous form of "task-sufficiency under failure."

3. **The joint select-and-route actuator (your original idea, now instrumented).** The optimizer that realizes `π_λ̂` across `{drop, b, r}` — greedy on `w(u)=Împ(u)(1−R̂ec(u))` to fill `r` up to the certified threshold, offload reconstructable content to `b`, drop the rest. Now it is not "a heuristic we hope works" — it is *the mechanism the certificate is attached to*.

---

## 4. The central result (theorem shape)

> **Proposition (informal).** Fix target risk `α ∈ (0,1)`. Given a calibration set of `n` scene/link-state samples exchangeable with test conditions, the conformal-risk-controlled routing threshold `λ̂` satisfies
> `E[ ρ(A_{λ̂}) ] ≤ α + O(1/n)`
> over the randomness of scenes *and best-effort link failure*, with **no assumptions on the detector, the Imp/Rec estimators, or the feature distribution.** In particular, total best-effort outage raises the ego's safety-critical miss-rate to at most `α` in expectation.

Contrast with everything cited in `NOVELTY_ANALYSIS.md`: JigsawComm/HydraCollab/CoSDH give AP numbers; V2X-DSC gives a rate-distortion story; none gives a certified miss-rate. This proposition is the differentiator, and it is *provable*, not aspirational — the note's original `ε + g(δ)` becomes `α + O(1/n)`, trading an uncontrolled model-error term for a controlled sample-size term.

---

## 5. Positioning (drop-in for the paper)

> Communication-efficient collaborative perception decides *what* to transmit; multi-radio steering decides *which link*; both are optimized empirically and certify nothing. Distribution-free uncertainty quantification certifies model *outputs* — cooperative trajectory prediction, tracking, boxes — but never the *communication decision*. We are the first to **conformally certify the communication policy itself**: select-and-route feature units so the ego's missed-safety-critical-object rate is provably ≤ α, distribution-free and finite-sample, and — via failure-conditioned calibration — robust to best-effort link outage. Importance and reconstructability are borrowed estimators (Where2comm / V2X-DSC); the contribution is the *certificate on the transmission policy* and its *survival guarantee under link failure*.

---

## 6. Experiments (what proves it)

- **Datasets:** OPV2V, DAIR-V2X, V2X-R (real 4D-radar + LiDAR), V2X-Sim. Use paired-view real data, not the synthetic-masking trick, for the headline results.
- **Baselines:** Where2comm (importance-only), JigsawComm (select+encode), HydraCollab (adaptive intermediate/late — run their public code), V2X-DSC (DSC codec), and a "staged select-then-steer" ablation.
- **Headline plot:** certified vs. *empirical* miss-rate across `α` sweep, **under simulated best-effort outage** — show the guarantee holds (empirical ≤ α) where all baselines' miss-rate blows up on link failure.
- **Efficiency plot:** reliable-link bytes vs. `α`. Claim: *at a fixed safety level, we use the least guaranteed-link bandwidth.*
- **Ablation (the four go/no-go tests from the note + one new):**
  - G1 `corr(Împ, R̂ec)` low and the high-Imp/high-Rec cell frequent.
  - **G1′ (new, from HydraCollab tension):** calibrated `R̂ec` (conditional-codec) predicts fusion-innovation-loss **better than HydraCollab's overlap mask** `1(c_i≥λ)⊙1(c_j≥λ)`. If it doesn't, offloading overlap content is unsafe — kill.
  - G2 joint ≫ staged specifically in the degraded / low-budget regime.
  - G3 certified miss-rate ≤ α under outage; coverage plot validates the conformal guarantee.

---

## 7. Risks & kill-criteria (be honest early)

- **Exchangeability is the load-bearing assumption.** Cross-scene / cross-city shift can void the guarantee. Mitigation: covariate-shift-weighted and *online/adaptive* conformal (guarantee holds in the long-run average even under drift). If neither holds empirically, retreat to a *marginal* (per-deployment) guarantee — still novel, weaker.
- **Monotonicity of the risk in `λ`.** CRC needs it; if routing interacts non-monotonically (fusion can *hurt* when a bad feature is added), use **Learn-Then-Test** (no monotonicity needed) or the 2026 non-monotone-loss CRC. Cited fallback exists — this is not a dead end.
- **The HydraCollab tension (G1′) is the real kill-switch.** If mutually-observed content is fusion-valuable rather than reconstructable, the offload premise fails. Run G1′ *first*.
- **Latency of the codec.** A V2X-DSC/DiffCP-style Rec estimator adds compute; report end-to-end latency, don't hide it.

---

## 8. The arc (why it's a program, not one paper)

1. **Paper 1 (this):** conformal certification of the dual-link select-and-route policy; failure-conditioned guarantee; single directed pair.
2. **Paper 2:** decentralized allocation of the scarce reliable budget `C^r` across contending senders at one receiver (auction / dual decomposition) — *network-level* risk control.
3. **Paper 3:** generative reconstructability — replace the DSC codec with a **diffusion prior** (DiffCP-style) so `Rec` = "what the receiver's generative model can hallucinate," and certify against hallucination error. Closes the loop between generative CP and safety guarantees.

---

## 9. References (verified July 2026)

**Conformal risk control / distribution-free UQ (the new machinery):**
- Angelopoulos, Bates, Fisch, Lei, Schuster — *Conformal Risk Control* — https://arxiv.org/abs/2208.02814
- Bates, Angelopoulos, Lei, Malik, Jordan — *Distribution-Free, Risk-Controlling Prediction Sets* (JACM) — https://dl.acm.org/doi/10.1145/3478535
- Angelopoulos, Bates, Candès, Jordan, Lei — *Learn Then Test: Calibrating Predictive Algorithms to Achieve Risk Control* (AoAS 2025) — https://arxiv.org/abs/2110.01052
- Angelopoulos & Bates — *A Gentle Introduction to Conformal Prediction and Distribution-Free UQ* — https://arxiv.org/abs/2107.07511
- *Conformal Risk Control under Non-Monotone Losses* (2026, fallback for non-monotonic risk) — https://arxiv.org/pdf/2604.01502

**Conformal in driving — applied to OUTPUTS only (the gap we fill):**
- *Conformal Trajectory Prediction with Multi-View Data in Cooperative Driving* — https://arxiv.org/html/2408.00374v1
- *Collaborative Multi-Object Tracking with Conformal Uncertainty Propagation* — https://arxiv.org/pdf/2303.14346
- *Beyond Confidence: Dual-Threshold Conformal Prediction for Autonomous System Perception* — https://arxiv.org/pdf/2502.07255
- *Conformal Object Detection by Sequential Risk Control* — https://arxiv.org/html/2505.24038v1
- *Adaptive Bounding Box Uncertainties via Two-Step Conformal Prediction* (ECCV 2024) — https://arxiv.org/html/2403.07263v1
- *Probabilistic Object Detection with Conformal Prediction* — https://arxiv.org/abs/2605.07549

**Communication-efficient collaborative perception (the actuator / baselines):**
- Where2comm (NeurIPS 2022) — https://arxiv.org/abs/2209.12836
- JigsawComm (2025) — https://arxiv.org/abs/2511.17843
- HydraCollab (2026, adaptive intermediate/late) — arXiv 2607.00191 · https://github.com/AICPS/HydraCollab
- CoSDH (CVPR 2025) — https://arxiv.org/abs/2503.03430
- V2X-DSC (2026, Wyner–Ziv conditional codec = the `Rec` estimator) — https://arxiv.org/abs/2602.00687
- DiffCP (diffusion generative reconstruction, for Paper 3) — https://arxiv.org/abs/2409.19592
- Rate-Distortion Optimized Communication for CP — https://arxiv.org/pdf/2509.21994

**Communication-realism / robustness (motivation):**
- ETSI CPS redundancy-mitigation survey (VoI, TS 103 324) — https://arxiv.org/abs/2501.01200
- Interruption-Aware Cooperative Perception (V2X-INCOP) — https://arxiv.org/abs/2304.11821
- MPR-QUIC (partially-reliable priority multipath, contrast) — https://www.sciencedirect.com/science/article/abs/pii/S1383762124001322

---

### TL;DR
Keep your dual-link mechanism. Change the *claim*. Stop competing on "better selection/routing" (crowded, HydraCollab-adjacent) and instead deliver **the first collaborative-perception system that conformally certifies its own communication policy with a distribution-free, failure-robust safety guarantee.** It is novel because two mature fields have never met here, defensible because it needs both skill-sets, and it converts your formulation's weakest point (estimator calibration) into its strongest (distribution-free certification).
