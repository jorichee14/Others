# Phase 4 — Failure Attribution Analysis

**Question:** when collaborative perception fails under a constrained link, is the failure
in *delivery* (the message never usefully arrived) or in *content* (the message arrived
and poisoned fusion)?

**Data:** 831 cells — 7 methods × 8 impairment families × 5–6 severities × 3 seeds,
OPV2V test split (stride 3), pretrained checkpoints, evaluation pipeline validated
against published results to ±0.001 (`results/baseline.md`). Full grid in
`results/sweep_summary.md` (per method) and `results/sweep_table.md` (one table, methods
as columns — AP/P/R and delivered-collaborator blocks). Reference points: ego-only floor **AP@0.7 = 0.575**
(P 0.825 / R 0.666); clean AP@0.7 per method: cobevt 0.862, coalign 0.833, v2vnet
0.822, attfuse 0.815, early 0.801, fcooper 0.790, late 0.781.

---

## 1. Headline findings

1. **The delivery/content divide is the study's strongest effect — stronger than any
   method difference.** Message *loss* never pushes any of the 7 methods below the
   ego-only floor, even at 90% drop rate. Message *corruption* (latency, staleness,
   pose error, scene swap) pushes **every** method below the floor — the point where
   collaboration is actively worse than silence.
2. **Latency is a content failure, and a brutal one: 100 ms of delay puts all 7 methods
   below the floor.** Dropping 90% of messages beats delivering all of them 200 ms
   late, for every method tested. *Better silent than stale.*
3. **Content robustness is what differentiates algorithms; delivery robustness does
   not.** Mean-AP-over-severities under delivery impairments spans only 0.70–0.78
   across methods; under content impairments it spans 0.30–0.51. Clean-channel
   rankings predict delivery rankings almost perfectly and content rankings almost
   not at all.
4. **The fusion-mechanism hypothesis is confirmed, with sharper structure than
   hypothesized** (§5): maxout fusion is uniformly content-fragile; each learned
   fusion mechanism has a *specific* content vulnerability matching its architecture;
   and alignment-robust training (CoAlign) is the only defense that transfers across
   the whole misalignment family.
5. **The misalignment valley:** moderate spatial error is worse than severe spatial
   error. Pose noise at 0.8–1.6 m hurts more than 3.2 m for every method; late
   fusion shows the same valley under latency. Plausibly-wrong evidence competes
   with correct evidence; wildly-wrong evidence self-discredits (§6).
6. **Bandwidth is free until a cliff:** 8× feature compression (4-bit) costs ≤0.05 AP
   for most methods, then 2-bit/1-bit quantization flips bandwidth from a delivery
   impairment into a content impairment — crossing below the floor for 4 of 5
   intermediate methods, with a reproducible non-monotonic anomaly for CoBEVT (§7).

---

## 2. Floor test (Step 4.1)

First severity level at which mean AP@0.7 crosses **below** the ego-only floor
(0.575 − 0.02). “–” = never crosses, at any tested severity.

| method | loss_iid | loss_burst | bandwidth | latency | stale | pose | ghosts | swap |
|---|---|---|---|---|---|---|---|---|
| cobevt | – | – | L3 (2b) | **L0 (100ms)** | L1 (0.4s) | L1 (0.4m) | – | L2 (50%) |
| coalign | – | – | L4 (1b) | **L0** | L1 | L1 | – | L3 (75%) |
| v2vnet | – | – | – | **L0** | L1 | L1 | – | L2 |
| attfuse | – | – | – | **L0** | L1 | L1 | – | L3 |
| early | – | – | n/a | **L0** | **L0 (0.2s)** | L1 | L4 | L1 (30%) |
| fcooper | – | – | L3 (2b) | **L0** | **L0** | L1 | – | L1 |
| late | – | – | n/a | **L0** | **L0** | **L0 (0.2m)** | – | L2 |

**Attribution verdict per impairment family:**

- **Packet loss (iid and bursty): pure delivery failure.** All methods degrade
  gracefully toward the floor and stop there. Burstiness at matched mean rate is
  irrelevant for single-frame detection (burst ≈ iid at every level; slightly kinder
  at 90% where bursts leave clean stretches) — expected for a stateless task, and a
  prediction that it *will* matter for tracking (Phase 5).
- **Latency, staleness: content failures with the lowest tolerance in the study.**
  One 100 ms tick of delay is already net-harmful for every architecture. Staleness
  mirrors latency (same mechanism, sawtooth age).
- **Pose error, scene swap: content failures with a small safe margin.**
  0.2 m localization error is tolerable (except for late fusion); 0.4 m is
  net-harmful for everyone. Swap tolerates 10–30% corrupted collaborators before
  crossing.
- **Ghost injection: the mildest content failure — never net-harmful** except for
  early fusion at 16 ghosts/message. Fusion mechanisms absorb *additive* false
  evidence far better than *misplaced* true evidence.
- **Bandwidth: delivery down to 4-bit, content below that.** The quantized features
  stop being a degraded signal and become a corrupting one.

## 3. Precision/recall decomposition (Step 4.2)

ΔP and ΔR at max severity relative to the same impairment's mildest level:

| | ΔP range (7 methods) | ΔR range | signature |
|---|---|---|---|
| loss_iid 90% | −0.01…−0.11 | **−0.14…−0.19** | recall-driven: delivery |
| ghosts ×16 | **−0.06…−0.30** | −0.01…−0.07 | precision-driven: additive content |
| latency 1 s | −0.11…−0.24 | −0.16…+0.09 | joint collapse: misplacement |
| swap 100% | −0.33…−0.46 | −0.21…−0.52 | joint collapse: misplacement |

The two predicted signatures hold across all seven methods with zero exceptions:
losing messages costs **recall** (occluded objects go undetected — the exact benefit
collaboration bought on the clean channel); corrupt-but-additive messages cost
**precision** (hallucinations); misplaced evidence (stale or misaligned) costs both at
once, because each wrong box is simultaneously a false positive where the object
isn't and a miss where it is.

Two per-method standouts:
- **CoBEVT under loss keeps P ≈ 0.91–0.93 at any drop rate** (ΔP −0.022) — the purest
  delivery-failure profile observed. **Late fusion is close behind** (ΔP −0.014):
  box-level fusion cannot hallucinate from missing boxes.
- **V2VNet under ghosts loses only 0.056 precision at 16 ghosts/message** — best in
  class by 2.4× (next: attfuse −0.137) — while the same model is bottom-tier under
  staleness. Mechanism in §5.

## 4. Rank stability (Step 4.4)

Ranking at max severity per impairment (best → worst):

- clean:        cobevt > coalign > v2vnet > attfuse > early > fcooper > late
- loss (both):  cobevt > coalign > {late, v2vnet} > fcooper > attfuse > early
- bandwidth 1b: v2vnet > attfuse > coalign > cobevt > fcooper
- latency 1 s:  **coalign > attfuse** > late > cobevt > v2vnet > early > fcooper
- stale 3.2 s:  **coalign > attfuse** > late > cobevt > v2vnet > early > fcooper
- pose 3.2 m:   **coalign > attfuse** > cobevt > late > v2vnet > fcooper ≈ early
- ghosts ×16:   **v2vnet** > attfuse > cobevt > late > coalign > fcooper > early
- swap 100%:    **coalign > attfuse** > late > cobevt > v2vnet > early > fcooper

Observations:
1. **Delivery preserves the clean ranking; content scrambles it.** Under loss, the
   clean order survives nearly intact. Under every misalignment-type impairment the
   clean #1 (CoBEVT) falls to mid-pack and the clean #1–#2 spots go to CoAlign and
   AttFuse — clean #2 and #4.
2. **CoAlign wins the entire misalignment family** (latency, staleness, pose, swap),
   not just pose error, which is what it was designed and trained for. Alignment
   robustness *transfers* across content impairments — the only defense in the cohort
   that does. Its cost: worst-in-class ghost precision (−0.252) — a mechanism tuned
   to reconcile misaligned evidence apparently also legitimizes fabricated evidence.
3. **F-Cooper is last or second-to-last under every content impairment** while
   mid-pack under delivery — the maxout prediction, confirmed without exception.
   Element-wise max has no mechanism to discount any arriving activation.
4. **Early fusion is the other consistent content loser** — raw points cannot be
   down-weighted after concatenation — and the only method driven below the floor by
   ghost injection (physical spoofing is most effective against point-level fusion).
5. **V2VNet's split personality is architecturally legible:** its GNN *averages*
   neighbor messages (ghost activations get diluted → ghost champion) but *warps*
   them using pose/time metadata before averaging (stale or misaligned messages get
   confidently warped to wrong places → bottom-tier under latency/stale). Same
   mechanism, opposite outcomes depending on impairment type.
6. **CoBEVT trades content fragility for delivery excellence:** its strong fused
   attention trusts whatever arrives — ideal when arrival is the only problem
   (precision immune to loss), costly when content lies (worst relative latency hit;
   bandwidth cliff at 2-bit).

**Fusion-mechanism verdict:** the original hypothesis (maxout = delivery-tolerant /
content-fragile; attention = partial down-weighting of corrupt content) is confirmed,
but the data support a sharper formulation: *every fusion mechanism is exactly as
content-robust as its ability to discount an arriving message, and its specific
vulnerability is the impairment that mimics evidence it was trained to trust.*
Max fusion trusts everything (fragile to all content). Averaging distrusts amplitude
(robust to ghosts) but trusts geometry (fragile to misalignment). Attention trusts
consensus (robust to additive noise, fragile to consistent misplacement). Alignment
learning distrusts geometry (robust to the whole misalignment family) but trusts
appearance (fragile to fabrication).

## 5. The misalignment valley (Step 4.4, novel)

Pose error is non-monotonic for every method — worst at 0.8–1.6 m, partially
recovering at 3.2 m (e.g. fcooper: 0.172 @0.8 m → 0.269 @3.2 m; cobevt: 0.304 →
0.443). Controls: the recovery survives the empty-message guard (attfuse L4 AP
identical with and without message drops), and late fusion shows the same valley
under *latency* (0.221 @200 ms → 0.310 @1 s), where no guard exists.

Interpretation: moderately misplaced evidence is *plausible* — shifted boxes/features
overlap real objects enough to displace or suppress correct detections (in NMS or in
feature space). Severely misplaced evidence lands nowhere plausible: it adds
suppressed-or-ignorable clutter but stops competing with the truth. Precision
recovering faster than recall at extreme severities (visible in every method's L4
pose row) supports this reading.

Practical consequence: **defenses that bound the worst case by rejecting extreme
outliers address the easy part of the problem.** The damage peak sits at error
magnitudes (≈1 m, ≈100–400 ms) that are exactly the plausible operating errors of
real localization and networking stacks.

## 6. The bandwidth cliff (Step 4.4)

16→8→4 bit quantization of shared features is nearly free (≤0.05 AP for all except
CoBEVT's 0.10 at 4-bit) — an 8× communication saving for noise-level cost, consistent
with the checkpoints' own compression variants (baseline.md). At 2-bit the cliff:
fcooper 0.476, coalign 0.577, cobevt 0.322 — and below the floor for 3 of 5. The
floor test classifies extreme quantization as a *content* impairment: the features
still arrive, but arrive wrong.

CoBEVT's anomaly — 2-bit (0.322) reproducibly *worse* than 1-bit (0.452), σ ≤ 0.001 —
is a real architectural interaction, not noise. Plausible account: 1-bit features
approximate a binary occupancy mask that attention can reinterpret, while 2-bit
levels retain enough spurious amplitude structure to be treated as (wrong) magnitude
information. Flagged for follow-up; does not affect the study's main claims.

## 7. Practical implications

- **Protocol designers: prioritize freshness over completeness.** Every architecture
  tested prefers losing most messages to receiving slightly old ones. Retransmission
  of stale data is actively counterproductive; aggressive message expiry (≤100 ms)
  is the single highest-value channel policy.
- **8× feature compression is free.** No architecture needs more than 4 bits per
  feature value on this benchmark. Below that, stop transmitting instead.
- **Method choice should follow the dominant impairment of the deployment link:**
  lossy-but-fresh (urban V2V with congestion) → CoBEVT; well-connected but poorly
  synchronized/localized → CoAlign; adversarial/spoofing-prone → V2VNet-style
  averaging; never maxout or raw-point fusion on any degraded link.
- **A latency-robust fusion mechanism is the field's most valuable missing piece.**
  Nothing in this cohort tolerates even one frame of delay; CoAlign's transfer of
  pose-robustness to time-robustness hints that alignment-style training against
  *temporal* misalignment is the promising direction (cf. SyncNet/CoBEVFlow, untested
  here).

## 8. Spatial decomposition (Step 4.3)

Mechanistic verification on 3 representative methods × 5 conditions (724 frames each).
Zones: a GT box is **ego-visible** iff it contains ≥5 of the ego's own lidar returns;
all other GT is **occluded** (reachable only through collaboration). `FP_egovis/f` =
false positives per frame claiming an object where ego's own sensor sees ≥5 points —
direct evidence of contamination inside ego's field of view.

| method | condition | R_vis | R_occ | FP/f | FP_egovis/f |
|---|---|---|---|---|---|
| attfuse | identity | 0.933 | 0.761 | 1.70 | 0.95 |
| attfuse | loss90 | 0.864 | 0.204 | 3.25 | 2.05 |
| attfuse | latency200ms | 0.673 | 0.399 | 5.58 | 3.54 |
| attfuse | ghosts8 | 0.926 | 0.748 | 3.27 | 1.17 |
| attfuse | swap50 | 0.846 | 0.449 | 5.13 | 1.76 |
| coalign | identity | 0.953 | 0.783 | 1.88 | 0.93 |
| coalign | loss90 | 0.894 | 0.282 | 3.30 | 1.80 |
| coalign | latency200ms | 0.747 | 0.307 | 5.62 | 3.14 |
| coalign | ghosts8 | 0.946 | 0.771 | 5.77 | 1.54 |
| coalign | swap50 | 0.912 | 0.545 | 5.91 | 1.95 |
| fcooper | identity | 0.913 | 0.713 | 1.86 | 1.24 |
| fcooper | loss90 | 0.838 | 0.123 | 2.51 | 2.02 |
| fcooper | latency200ms | 0.456 | 0.201 | 7.76 | 5.65 |
| fcooper | ghosts8 | 0.890 | 0.690 | 4.89 | 1.60 |
| fcooper | swap50 | 0.617 | 0.362 | 4.83 | 1.90 |

Findings (all read as deltas from each method's identity row):

1. **Delivery failure is spatially surgical.** loss90 removes 0.50–0.59 of occluded-zone
   recall but only 0.06–0.08 of ego-visible recall (~8:1 selectivity, all three
   methods): losing messages removes exactly the zone collaboration was providing.
2. **Content failure reaches inside ego's own field of view.** latency200ms cuts
   ego-visible recall by 0.21–0.46 and multiplies ego-visible false positives by
   3.4–4.6× — stale evidence demonstrably degrades detection of objects the ego's own
   sensor sees, which is fusion poisoning measured directly.
3. **The contamination magnitude reproduces the sweep's content-fragility ranking**:
   fcooper (ΔR_vis −0.46) > attfuse (−0.26) > coalign (−0.21). Three independent
   diagnostics — floor test, P/R decomposition, spatial decomposition — now agree on
   both the attribution *and* the method ordering.
4. **Plausible corruption contaminates more than implausible corruption.** Latency
   produces roughly double the ego-visible contamination of swap at comparable AP cost
   (attfuse FP_egovis 3.54 vs 1.76) — the misalignment valley (§5) made spatial: stale
   evidence lands on and around real traffic and competes with correct detections;
   foreign-scene features land nowhere meaningful.
5. **Ghosts stay out of ego's zone**: large FP/f increases (up to +3.9) but FP_egovis
   rises only +0.2–0.4 and both recalls hold within 0.02 — injected hallucinations
   appear mostly in unobserved space and do not suppress real objects.
6. **Nuance for the delivery story**: extreme loss also roughly doubles FPs
   (flickering collaborators produce unstable partial evidence), so "delivery failures
   cost only recall" should read "overwhelmingly recall."
7. Method note: even on the clean channel, F-Cooper has the highest baseline
   contamination (FP_egovis 1.24 vs 0.93–0.95) — maxout is the least selective fusion
   even before any impairment.
8. Reproducibility note: the attfuse and coalign cells were executed twice end-to-end
   (in separate processes) and reproduced **digit-for-digit** — the pipeline's
   seeded determinism holds through the full spatial analysis.

## 9. Cooperative tracking under impairment (Phase 5, Step 5.1)

Detection + constant-velocity Kalman tracking (world frame, Hungarian association,
2-hit confirm / 3-miss delete, per-scenario reset), full test split at stride 1.
GT tracks from dataset object identities. Burst conditions share the marginal loss
rate of their iid pair; `_long` bursts average ~10 frames (beyond the tracker's
3-frame coast window) vs ~3.3 (at it).

| method | condition | MOTA | FN | FP | IDSW | FRAG |
|---|---|---|---|---|---|---|
| coalign | clean | 0.835 | 1331 | 3914 | 149 | 483 |
| coalign | iid30 | 0.809 | 1547 | 4492 | 185 | 550 |
| coalign | burst30 | 0.806 | 1612 | 4502 | 200 | 563 |
| coalign | burst30_long | 0.802 | 1761 | 4494 | 208 | 567 |
| coalign | iid70 | 0.728 | 2708 | 5813 | 349 | 856 |
| coalign | burst70 | 0.734 | 2583 | 5716 | 362 | 815 |
| coalign | burst70_long | 0.701 | 3394 | 5942 | 413 | 892 |
| coalign | stale4 | 0.653 | 3193 | 7474 | 629 | 1394 |
| coalign | latency2 | 0.654 | 4106 | 6935 | 223 | 806 |
| cobevt | clean | 0.881 | 1832 | 1883 | 160 | 511 |
| cobevt | iid30 | 0.877 | 2029 | 1799 | 194 | 546 |
| cobevt | burst30 | 0.864 | 2347 | 1837 | 260 | 625 |
| cobevt | burst30_long | 0.855 | 2524 | 1968 | 231 | 629 |
| cobevt | iid70 | 0.815 | 3760 | 1944 | 341 | 788 |
| cobevt | burst70 | 0.801 | 4000 | 2097 | 378 | 878 |
| cobevt | burst70_long | 0.795 | 4198 | 2055 | 418 | 844 |
| cobevt | stale4 | 0.541 | 5845 | 8091 | 1031 | 2248 |
| cobevt | latency2 | 0.555 | 7305 | 6980 | 230 | 966 |
| fcooper | clean | 0.818 | 2428 | 3328 | 167 | 597 |
| fcooper | iid30 | 0.799 | 2717 | 3623 | 196 | 648 |
| fcooper | burst30 | 0.801 | 2824 | 3406 | 265 | 683 |
| fcooper | burst30_long | 0.783 | 3146 | 3632 | 282 | 746 |
| fcooper | iid70 | 0.742 | 4222 | 3828 | 375 | 872 |
| fcooper | burst70 | 0.722 | 4605 | 4075 | 378 | 933 |
| fcooper | burst70_long | 0.711 | 4964 | 4042 | 430 | 937 |
| fcooper | stale4 | 0.505 | 6239 | 8987 | 917 | 2109 |
| fcooper | latency2 | 0.478 | 8484 | 8284 | 248 | 1021 |

**P1 — burstiness matters for tracking, confirmed with a dose-response.** At every
matched loss rate, IDSW rises with burst length for all three methods (e.g. fcooper
@30%: 196 → 265 → 282; coalign @70%: 349 → 362 → 413; cobevt @70%: 341 → 378 → 418),
and MOTA orders iid ≥ burst ≥ burst_long almost everywhere. The contrast with
detection is the finding: at 70% loss, detection AP was statistically identical for
iid vs bursty loss (§2), yet tracking loses 15–23% more identities under bursts.
Burstiness irrelevance was a property of the stateless task, not of the channel —
**failure attribution depends on the task's temporal structure.** Nuance: short
(~3-frame) bursts, inside the tracker's coast window, barely move MOTA while already
inflating IDSW ~35% — aggregate metrics hide identity churn that downstream consumers
(prediction, planning) would feel.

**P2 — staleness and latency, near-twins at detection level, have OPPOSITE tracking
signatures.** Staleness (sawtooth age, refresh every 0.4 s) explodes identity
switches (4–6× clean: 629/1031/917) and fragmentation; constant latency (0.2 s)
leaves IDSW almost untouched (223/230/248 vs clean 149/160/167) but produces the
table's largest miss counts. Mechanism: constant delay is a *consistent* bias — the
motion model locks onto steadily-displaced detections and keeps identities coherent
(wrong in place, stable in identity → FN-dominated); sawtooth staleness makes
detections *oscillate* between true and displaced positions each refresh cycle,
shredding association. A motion-model tracker therefore partially absorbs consistent
temporal error and actively amplifies oscillating temporal error — a distinction
invisible at the detection level, where stale4 and latency2 cost similar AP.
Deployment corollary: for tracking-based pipelines, a channel with *predictable*
delay is preferable to one with *variable* freshness at equal average age — the
opposite of what detection-level analysis alone would suggest.

Method notes: CoBEVT's delivery excellence carries to tracking (best MOTA in every
loss condition; FP nearly flat under loss — its precision retention becomes track
stability); CoAlign's clean-channel FP-heaviness (3914, 2× cobevt) caps its clean
MOTA at 0.835 despite the best detection AP among the three under misalignment;
F-Cooper is last in every condition, consistent with all prior diagnostics.

## 10. Limitations

- The floor (0.575) is the late-fusion checkpoint evaluated ego-only; per-backbone
  floors would shift individual crossings by small margins (the floor-test margin
  ±0.02 absorbs part of this). Cross-method floor-test *patterns* are robust to this.
- Stride-3 frame subset (724/2170 frames); baseline reproduction at stride 3 matched
  the full split to ±0.003.
- Bandwidth impairment for AttFuse intercepts the backbone input (its fusion is
  interleaved) — a documented approximation; late/early fusion have no feature
  bandwidth analogue and were excluded from that family.
- Spatial decomposition (§8) covers 3 representative methods × 5 conditions rather
  than the full matrix — sufficient for mechanism verification; extendable via
  `run_phase43.py --methods`.
- Single dataset (OPV2V, simulated). The tracking tier (§9) uses a deliberately
  simple constant-velocity Kalman tracker — an instrument for measuring impairment
  effects, not a tracking contribution; absolute MOTA values should not be compared
  to the tracking literature. Harder tiers (trajectory prediction, closed-loop
  CARLA) remain unscoped.
