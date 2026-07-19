# Core Task (scoped) — Joint Selection-and-Link-Routing for Collaborative Perception

*Certification deferred. This is the joint co-design as a standalone systems/optimization contribution.*
*Companion: `NOVELTY_ANALYSIS.md`, `PROPOSAL.md`. Prepared 2026-07-19.*

---

## Problem

Connected vehicles improve perception by sharing compressed feature-map **units** over wireless links, under a bandwidth budget. Two facts that current systems ignore:

1. Vehicles have **multiple heterogeneous radios** — one **reliable but low-rate** (`r`), one **fast but failure-prone** (`b`). They are not one abstract channel.
2. The fast link can **drop transiently** (fading, obstruction, congestion) during exactly the high-mobility moments perception matters most.

**The open problem:** given per-unit *importance* and *reconstructability*, decide for each unit whether to **drop it, send it on the best-effort link, or send it on the reliable link** — as a *single joint decision* — so that (a) the reliable link's scarce budget is spent only on content that is both critical and irreplaceable, (b) the best-effort link is used to *offload* reconstructable content for extra accuracy when it survives, and (c) the reliable partition **alone** keeps task loss bounded if the best-effort link fails.

No existing method does this. Selection methods assume one channel; multi-radio methods steer content-blind or duplicate for reliability.

---

## Motivation

- A missed occluded pedestrian is a crash; a bandwidth reduction is worthless if the fast link drops and the content you kept was the wrong content.
- Real deployments already carry two radios and already suffer transient link loss — the single-reliable-channel assumption everyone optimizes under is simply false.
- The two decisions — *what to send* and *which link* — are currently made by two separate communities that each ignore the other's variable. Coupling them is untapped and, as shown below, is a genuine joint problem rather than a pipeline.

---

## Novelty (precise)

**One sentence:** the first framework to make *content-value selection* and *physical-link routing* a **single joint decision** that uses a lossy link for **capacity offload** of reconstructable content — the inverse of duplication-for-reliability — while guaranteeing the reliable partition alone is **task-sufficient under best-effort failure**.

Three specific, defensible novelties:

1. **Jointness (not a two-stage pipeline).** The routing option *changes the optimal selection*. In a single-channel world you **drop** reconstructable content (it wastes budget); in a dual-link world you **keep it on best-effort** (nearly free, adds accuracy if it arrives). A staged "select-then-steer" system commits to selection before it knows the routing and structurally cannot produce this partition. Two coupling effects it misses: *routing-induced retention* (a unit a selector would drop is kept on best-effort) and *failure-conditioned promotion* (a reconstructable-but-critical unit is promoted onto the reliable link to satisfy sufficiency).

2. **Capacity-offload inversion.** Every prior multipath system uses a second link to **duplicate** important content for *reliability*. We use it for **capacity** — offloading reconstructable content to free the reliable link. Opposite purpose, opposite decision rule.

3. **Failure-survival by construction.** A design constraint `D(A) ≤ ε` on the reliable partition `A` means total best-effort outage raises task loss by at most `ε`. This is provable-by-construction (the constraint *is* the guarantee), and it is what forces the joint solution — the sufficiency requirement is what promotes critical-reconstructable units.

**Explicitly not claimed:** importance/redundancy selection (JigsawComm, CoSDH, Where2comm), reconstructability coding (V2X-DSC), two radios (hybrid DSRC/C-V2X, ATSSS), priority multipath (content-aware MPTCP). Novelty = their intersection, which is empty.

**Distinction from the closest work (HydraCollab):** HydraCollab chooses *representation* (rich feature vs. cheap detection output) over **one** channel, keyed on confidence *overlap*, with **no physical-link model, no failure, no capacity offload** — and it treats overlap content as *fusion-valuable*, the **opposite** of our offload premise. Reconciling that (reconstructability ≠ overlap) is our first experiment.

---

## Approach

**Signals (borrowed inputs).** Per unit `u`: importance `Împ(u)` (detector confidence / ablation saliency) and reconstructability `R̂ec(u) ∈ [0,1]` (receiver-side conditional predictability from ego side-information — inpainting proxy now, V2X-DSC-style conditional codec later). `R̂ec` must be strictly broader than confidence overlap.

**Decision.** For each `u`, choose `a(u) ∈ {∅, b, r}`. Let `A = {u : a=r}`, `B = {u : a=b}`.

**Residual task distortion** when receiver holds set `R`:
`D(R) = Σ_{u ∉ R} Împ(u)·(1 − R̂ec(u))` — importance-weighted over units that are absent *and* non-reconstructable.

**Per-pair program (solved locally by the sender):**
```
minimize    Σ_{u ∈ A} b(u)                       # reliable-link load
subject to  D(A) ≤ ε                             # survives best-effort failure
            Σ_{u∈A} b(u) ≤ C^r,  Σ_{u∈B} b(u) ≤ C^b   # link capacities
```

**Solver.** A unit only hurts `D(A)` through weight `w(u) = Împ(u)·(1 − R̂ec(u))`. Greedy: place highest-`w` units (critical AND irreplaceable) on `r` until `D(A) ≤ ε`; offload remaining high-value-but-reconstructable units to `b`; drop the rest. This *derives* the 2×2 routing table, including offloading high-Imp/high-Rec content (its `w` is small because `1−Rec ≈ 0`).

**Validation (offline, simulated links, off-the-shelf detector — no radios, no training):**
- **Phase 0 / G1, G1′.** Per-cell `Împ` vs `R̂ec` on real paired frames: correlation, 2×2 quadrant frequency, and whether `R̂ec` diverges from an overlap mask. *Kill if axes collapse.*
- **Phase 1 / G2, G3.** Three policies (importance-only, staged, joint) at fixed reliable budget, measured nominal and best-effort-dropped. *Win = joint ≥ others under failure at equal reliable budget.*
- **Phase 2.** Sweep reliable budget over 20–50 frames; report joint-vs-staged gap, concentrated in the degraded / low-budget regime.

**Datasets.** DAIR-V2X / OPV2V / V2X-R (real paired views) for headline; single self-contained image with complementary masking for the Phase-0 sanity check.
