# Literature Review — Temporal Messaging in Collaborative Perception

Field survey conducted 2026-08-09 to resolve **D1** (`IMPLEMENTATION.md`) and to re-audit the
novelty accounting in [`HANDOFF.md`](HANDOFF.md) §4. The parent study's authoring environment
could not reach arxiv, so four claims were carried as ⚠️ UNVERIFIED. This document replaces
that placeholder with evidence.

**Headline: two of the four candidate novelty claims are dead, one is wounded, one survives
and is now the only defensible headline.** Details in §8. Phase 0 is still worth running —
arguably more so — but the framing around it has to change. See §9.

---

## 1. How this review was done, and how far to trust it

Search coverage is good; primary-text verification is not. The container's egress proxy
permits `github.com` / `raw.githubusercontent.com` and the search index, and **blocks
`arxiv.org`, `openreview.net`, `openaccess.thecvf.com`, `semanticscholar.org`,
`ieeexplore.ieee.org`** and the paper-summary mirrors. Every attempt to fetch a primary PDF
was refused by the proxy.

So each claim below is tagged:

| tag | meaning |
|---|---|
| **[V]** | verified against primary text or an official repo README |
| **[S]** | from search-engine summaries of the abstract — reliable for *what a paper claims*, not for its numbers or experimental protocol |
| **[I]** | my inference from [V]/[S] material, not stated by any author |

The field map itself is **[V]**: it comes from
[Little-Podi/Collaborative_Perception](https://github.com/Little-Podi/Collaborative_Perception),
the canonical maintained digest, read in full at commit-time. It indexes through CVPR 2026 /
ICLR 2026 / AAAI 2026, so the coverage is current rather than training-cutoff-bound.

**Nothing here should be written into a paper without reading the primary text.** The numbers
in §3 and §8 in particular are exactly the kind of thing search summaries get subtly wrong.
Treat this as a map of where to read, plus a set of threats that must be answered.

---

## 2. The field in one paragraph

Collaborative perception splits by *what is transmitted*: raw points (early), feature maps
(intermediate), detections (late). The learning community standardized on intermediate fusion
because it wins the accuracy-per-bit trade-off, and the last five years of work optimize inside
that choice — better fusion operators, fewer bits, robustness to pose error, heterogeneity, and
adversaries. Temporal asynchrony is recognized as a first-class failure mode and has its own
sub-literature. The thesis in `HANDOFF.md` — that the *representation* is the problem, because
a feature map is a snapshot with no description of its own dynamics or validity — is a coherent
position, and it is **not** an unoccupied one.

---

## 3. Thread 1 — Temporal asynchrony and latency compensation

This is the most crowded thread and the one the thesis lives in.

### 3.1 Receiver-side compensation (the dominant paradigm)

The receiver holds a history of what arrived and infers the motion it missed.

- **V2X-ViT** (ECCV 2022) — delay-aware positional encoding inside the fusion transformer.
  Reported as degrading <4% per additional 100 ms on V2XSet, versus early/late fusion losing
  48.6% / 54.4% AP at 200 ms **[S]**. Note how closely that late-fusion collapse tracks the
  parent study's own late-fusion result (AP@0.7 0.781 → 0.326 at 100 ms).
- **SyncNet** (ECCV 2022) — the original "latency-aware collaborative perception"; a plug-in
  module that synchronizes features to a common timestamp **[S]**.
- **CoBEVFlow** (NeurIPS 2023) — BEV motion-vector field; reassigns asynchronous features to
  corrected positions, handles irregular (non-quantized) timestamps, and deliberately only
  *moves* existing features rather than generating new ones, to avoid injecting noise. Ships
  the IRV2V dataset for irregular asynchrony **[V]**, [repo](https://github.com/MediaBrain-SJTU/CoBEVFlow).
- **TraF-Align** (CVPR 2025) — **the most direct technical competitor.** Learns the feature-level
  *trajectory* of objects from past observations up to the ego's current time, then attends
  along temporally ordered sample points on those paths. Reports AP50 drops of only **4.87%
  (V2V4Real) and 5.68% (DAIR-V2X-Seq) at 400 ms latency** **[S]**,
  [repo](https://github.com/zhyingS/TraF-Align).
- **CoDynTrust** (ICRA 2025) — models aleatoric + epistemic uncertainty per ROI to produce a
  "dynamic feature trust modulus" that selectively suppresses or retains stale features **[S]**,
  [repo](https://github.com/CrazyShout/CoDynTrust).
- Also in this family: **SCOPE** (ICCV 2023), **How2comm** (NeurIPS 2023, flow-guided delay
  compensation), **CoST** (ICCV 2025), **MRCNet** (CVPR 2024, motion-aware), **CTCE**,
  **DATA** (domain-and-time alignment), **StreamLTS**-style streaming work **[S]**.

### 3.2 Sender-side prediction — the space is occupied

`HANDOFF.md` §4 treats sender-side placement as "a conditioning argument, not a headline."
That was the right call, and it is now load-bearing:

- **FFNet** (NeurIPS 2023) — transmits **feature flow** rather than a feature map: a
  first-order, time-parameterized object that the receiver *evaluates at its own timestamp*.
  The flow generator is trained self-supervised to predict future features. Claims to beat
  prior cooperative detectors at ≤1/10 the raw-data transmission cost on DAIR-V2X when
  asynchrony exceeds 200 ms **[V]** via [repo](https://github.com/haibao-yu/FFNet-VIC3D), numbers **[S]**.

  This is the thesis's core mechanism — "make the message a function of time, not a snapshot" —
  already published at the feature level, sender-side, three years ago.

- **SparseCoop** (AAAI 2026) — **the closest prior art that exists.** A fully sparse framework
  for detection *and* tracking whose message is a "kinematic-grounded instance query" carrying
  an **explicit state vector with 3D geometry and velocity**, used for spatio-temporal
  alignment across asynchronous viewpoints **[S]**. That is a time-parameterized,
  self-describing, consumer-agnostic message unit, learned, evaluated on both detection and
  tracking. Tsinghua/NTU/PolyU/HKU/Penn.

### 3.3 Instance- and query-level message units

The "message unit" question (`HANDOFF.md` §7 raises scheduling message units) is active:
**QUEST** (query stream), **CoopDETR**, **INSTINCT** (ICCV 2025), **TransIFF** (ICCV 2023),
**CPPC** (ICLR 2025 — point cluster as a compact message unit), **RefPtsFusion**,
**SlimComm** (ICCV 2025, Doppler-guided sparse queries — note: *Doppler*, i.e. sender-side
velocity, used for bandwidth selection) **[S]**.

---

## 4. Thread 2 — Rate: bandwidth-efficient messaging

Long, mature line: **When2com** / **Who2com** (2020) → **DiscoNet** (NeurIPS 2021) →
**Where2comm** (NeurIPS 2022, spatial confidence maps) → **UMC**, **What2comm** (2023) →
**CodeFilling** (CVPR 2024, codebook) → **ERMVP** (CVPR 2024) → **CoSDH** (CVPR 2025,
intermediate-late hybridization) → **ReVQom** (residual VQ), **QuantV2X** (full quantization),
**DiffCP** (ultra-low-bit via diffusion), **InfoCom** (AAAI 2026, kilobyte-scale via information
bottleneck), **WhisperNet**, **JigsawComm**, **DinoLink**, **CoLC** (CVPR 2026) **[S]**.

Two entries matter specifically:

- **RDComm** (ICLR 2026, "Rate-Distortion Optimized Communication for Collaborative Perception")
  — builds a *pragmatic rate-distortion theory* for multi-agent collaboration, with a
  task-driven "pragmatic distortion" metric and task-entropy discrete coding that assigns
  codeword lengths by task relevance **[S]**. This is the principled rate-side framework.
- **HyComm** — hybrid: transmits compact perceptual outputs *and* raw observations, with
  adaptable compression, and explicitly uses **standardized, model-independent message formats**
  so agents with different detectors interoperate **[S]**. This is the "hybrid keeps richness"
  and "consumer-agnostic message" argument, already made.

The parent study's own finding — bandwidth is free down to 4 bits, cliff at 2, so ~8× spare
budget — is consistent with this literature and is a *measurement* the literature mostly asserts.

---

## 5. Thread 3 — Rate × age jointly: **D1 is resolved, and the answer is no**

The pre-registered question was whether "nobody has measured accuracy as a function of both
rate and age" holds. **It does not hold.** The joint problem is active, mostly in the
networking-side literature rather than the CV venues the handoff was sampling:

- **Fresh2comm** (Feb 2025) — incorporates Age of Information into collaborative perception and
  builds an **AoI-based optimization that allocates communication resources** to control system
  AoI, explicitly to study "high transmission delay and inconsistent delay" **[S]**.
- **"Spatiotemporal Feature Alignment and Weighted Fusion … Network Synchronization and Age of
  Information"** (Feb 2026) — closes the loop explicitly: a feature's **AoI is computed by
  estimating network delay from the given feature size and link quality**, and that AoI then
  drives the fusion weight **[S]**. Feature size → delay → age → fusion weight *is* the rate×age
  coupling, modeled end to end.
- **"Update the Unseen Only: Minimizing AoI for Collaborative Perception through Online
  Learning"** (Jul 2026) — opens on "limited bandwidth can cause severe shared data staleness",
  and schedules updates by what the receiver *cannot already sense itself* (LocMW) **[S]**.
- **V2X-ReaLO** (2025) — real vehicles + infrastructure, online; measures perception accuracy
  against **message size and communication latency** together, for early/late/intermediate **[S]**.
- **V2X-ViT** already sweeps compression (32×, 128×) and delay, and the field's standard framing
  is that compression *buys* latency **[S]**.

**Verdict on D1: NO-GO on the novelty claim.** Do not assert that the joint rate×age
allocation question is unasked. Phase 3 is not dead — but it must be positioned as *measuring*
a trade-off others *optimize under assumptions*, and it needs a literature-anchored delta.
See §9.

Note also the direct overlap with the parent study's Step 4.3: "Update the Unseen Only"
schedules by whether the ego can already see the region, which is precisely the mechanism the
parent study measured (90% loss costs 0.50–0.59 occluded recall but only 0.06–0.08 ego-visible,
~8:1). Their policy principle is the parent study's empirical result. That is a genuine
citation-level connection in both directions.

---

## 6. Thread 4 — Standards: the message the field abandoned

**ETSI TS 103 324 (Collective Perception Service, Release 2, 2023-06)** — the Perceived Object
Container carries the object's kinematic and attitude dynamics *and* a **`timeOfMeasurement`**,
defined as the offset between the object's detection time and the message's generation time
**[S]**, with the ASN.1 available at
[forge.etsi.org](https://forge.etsi.org/rep/ITS/asn1/cpm_ts103324).

So the standardized V2X message is *already* time-parameterized and self-describing in exactly
the sense the thesis proposes. This is the single most important framing fact in this review:
**the thesis's proposed message is the standards-track message, and the learning literature
walked away from it.** That is a good story — but it is a story about *quantifying a known
trade-off*, not about inventing a representation.

---

## 7. Thread 5 — Task-agnostic messaging, and Thread 6 — robustness benchmarks

**Task-agnostic** is an established goal, not an open angle:
**STAR** (CoRL 2022) — "the first task-agnostic collaborative perception paradigm", a
self-supervised spatiotemporal autoencoder trained via multi-robot scene completion **[S]**;
**STAMP** (ICLR 2025) — "Scalable Task- And Model-Agnostic Collaborative Perception" **[S]**;
**HyComm** — model-independent message formats (§4). Downstream-task breadth is likewise
covered: **V2XPnP** (perception *and* prediction), **CoopTrack** (ICCV 2025, end-to-end
cooperative tracking), **TurboTrain** (ICCV 2025, multi-task perception+prediction), **CMP**,
**Co-MTP**, **V2X-Graph** (NeurIPS 2024), **R&B-POP** (ICLR 2025, "Learning 3D Perception from
Others' *Predictions*") **[S]**.

**Robustness benchmarking** — the parent study's home turf, and it is getting crowded:
**AgentComm-Bench** (arXiv 2603.20285) **verified to exist**, and its abstract matches the
handoff's summary: six impairment dimensions, and the finding that *perception fusion is immune
to transport failures but amplifies corrupted data (>85% NPD)* **[S]** — an independent
replication of the parent study's delivery-vs-content split. Also **RCP-Bench** (CVPR 2025,
robustness under diverse corruptions), **CP-FREEZER** (latency *attacks*), and
**"When Autonomous Vehicle Meets V2X Cooperative Perception: How Far Are We?"** (ASE 2025),
whose "**misleading cooperative errors** — cooperation degrades otherwise-correct ego
perception" is an independent confirmation of the parent study's Step 4.3 finding that latency
contaminates the ego's own field of view **[S]**.

---

## 8. Re-audit of the four novelty claims

| # | Claim (HANDOFF §4) | Verdict |
|---|---|---|
| 1 | **The displacement diagnostic** — latency damage is mis-timing, not information loss (ΔAP@0.5 vs ΔAP@0.7, 3.9–8.8×) | **SURVIVES — and is now the only headline.** No paper found reports the AP@0.5-vs-AP@0.7 decomposition under latency across an architecture family. TraF-Align reports **AP50** at 400 ms; V2X-ViT reports aggregate AP. The whole field measures the metric that *hides* this effect. ⚠️ Must confirm by reading TraF-Align, CoBEVFlow, SyncNet, FFNet primary texts for any AP@0.7-under-latency table. |
| 2 | **Discountability is a property of the message, not the operator** | **WOUNDED.** CoDynTrust does receiver-side per-ROI suppression of stale features and reports SOTA under asynchrony. The claim only survives in a *precise* form: weights applied **inside** a permutation-invariant non-discounting aggregator (maxout) are provably inert — which is what AgentComm-Bench's λ ablation shows — whereas **masking/suppressing before aggregation** is a different intervention and it works. State it that way or it is simply false. The falsifiable target (*make F-Cooper latency-robust without modifying F-Cooper*) remains a good experiment. |
| 3 | **Joint rate × age allocation** | **DEAD as novelty.** See §5. Fresh2comm, the AoI+synchronization paper, "Update the Unseen Only", and V2X-ReaLO all occupy it. Do not assert. |
| 4 | *(implicit)* **Task-agnostic / consumer-agnostic messaging** | **DEAD as novelty.** STAR (2022), STAMP (2025), HyComm. Usable as motivation, never as contribution. |

And the thesis's central mechanism — time-parameterized, self-describing messages — is prior
art three times over: **ETSI CPM** (standard, object-level, with `timeOfMeasurement`),
**FFNet** (feature level, sender-side, NeurIPS 2023), **SparseCoop** (instance level, geometry
+ velocity, detection *and* tracking, AAAI 2026).

The expected reviewer attack in `HANDOFF.md` — *"this is late fusion with a Kalman filter"* — was
correctly anticipated but under-estimated. The sharper version is: **"this is FFNet, or
SparseCoop, and you did not cite either."**

---

## 9. What this means for the plan

**Phase 0 should still run, unchanged.** It is one day of work, its decision rule is
pre-registered, and its value went *up*: it measures how much of the latency gap is recoverable
by pure constant-velocity correction with **zero learning**. That is the missing baseline for
every method in §3.1 — none of which, as far as this review can tell, reports what a trivial
sender-side kinematic correction would have achieved. A paper whose contribution is *"the
learned latency-compensation literature is measured against a baseline that does not exist,
and here it is"* is a real contribution, and Phase 0 is exactly that experiment.

Concrete adjustments:

1. **Re-frame from "new representation" to "diagnostic + missing baseline."** Contribution (a)
   in `HANDOFF.md` §4 was always the strongest; it is now the only one. Lead with the
   displacement diagnostic and the zero-learning correction baseline.
2. **Add arm B′ (oracle displacement) to Phase 0.** With B (constant-velocity from consecutive
   GT frames) *and* B′ (exact GT displacement over the delay), the B′−B gap separates
   *velocity-estimation error* from *constant-velocity model error*. This is cheap and it is
   the number that tells you whether learned flow (FFNet/TraF-Align/CoBEVFlow) can beat
   kinematics at all.
3. **Report AP@0.5 and AP@0.7 side by side everywhere.** That contrast is the contribution;
   the field's AP50-only convention is what hides it.
4. **Phase 2 must retarget.** "Discountability" needs the precise statement from §8 row 2, and
   CoDynTrust becomes a required baseline, not an unmentioned neighbour.
5. **Phase 3 is blocked, not dead.** D1 came back NO-GO. Either drop it, or re-scope it to
   something the AoI papers assume rather than measure — e.g. *is the rate→delay→age chain they
   model actually monotonic in detection accuracy, given the 4-bit bandwidth floor the parent
   study measured?* That is a real question and the parent study already owns half the answer.
6. **Read before writing.** Priority primary texts, in order: **SparseCoop**, **FFNet**,
   **TraF-Align**, **CoDynTrust**, **RDComm**, **Fresh2comm**, the **AoI+synchronization**
   paper, **HyComm**, **AgentComm-Bench**. Specifically check each for an AP@0.7-under-latency
   table — claim 1 lives or dies on that.
7. **Dataset decision (D5) gets easier.** TraF-Align, FFNet and CoBEVFlow all report on
   **DAIR-V2X-Seq** and/or **V2V4Real**. Staying OPV2V-only makes the work incomparable to the
   closest competitors.

---

## 10. Reference index

Field digest (read in full): [Little-Podi/Collaborative_Perception](https://github.com/Little-Podi/Collaborative_Perception).

**Asynchrony / latency** — [SyncNet](https://arxiv.org/abs/2207.08560) ECCV'22 ·
[V2X-ViT](https://arxiv.org/abs/2203.10638) ECCV'22 ·
[CoBEVFlow](https://github.com/MediaBrain-SJTU/CoBEVFlow) NeurIPS'23 ·
[FFNet](https://github.com/haibao-yu/FFNet-VIC3D) NeurIPS'23 ·
[How2comm](https://github.com/ydk122024/How2comm) NeurIPS'23 ·
[SCOPE](https://arxiv.org/abs/2307.13929) ICCV'23 ·
[MRCNet](https://github.com/IndigoChildren/collaborative-perception-MRCNet) CVPR'24 ·
[TraF-Align](https://github.com/zhyingS/TraF-Align) CVPR'25 ·
[CoDynTrust](https://github.com/CrazyShout/CoDynTrust) ICRA'25 ·
[CoST](https://github.com/tzhhhh123/CoST) ICCV'25 ·
[SparseCoop](https://arxiv.org/abs/2512.06838) AAAI'26

**Rate / bandwidth** — [Where2comm](https://github.com/MediaBrain-SJTU/where2comm) NeurIPS'22 ·
[CodeFilling](https://github.com/PhyllisH/CodeFilling) CVPR'24 ·
[CoSDH](https://github.com/Xu2729/CoSDH) CVPR'25 ·
[SlimComm](https://github.com/fzi-forschungszentrum-informatik/SlimComm) ICCV'25 ·
[RDComm](https://openreview.net/forum?id=920RxFvsMx) ICLR'26 ·
[InfoCom](https://github.com/fengxueguiren/InfoCom) AAAI'26 · HyComm · QuantV2X · ReVQom · DiffCP

**Rate × age / AoI** — [Fresh2comm](https://arxiv.org/abs/2502.07852) ·
[AoI + network synchronization](https://arxiv.org/abs/2602.13439) ·
[Update the Unseen Only](https://arxiv.org/abs/2607.20967) ·
[V2X-ReaLO](https://arxiv.org/abs/2503.10034)

**Standards** — ETSI TS 103 324 v2.1.1 Collective Perception Service
([ASN.1](https://forge.etsi.org/rep/ITS/asn1/cpm_ts103324)) · ETSI TR 103 562

**Task-agnostic** — [STAR](https://github.com/coperception/star) CoRL'22 ·
[STAMP](https://github.com/taco-group/STAMP) ICLR'25 ·
[V2XPnP](https://github.com/Zewei-Zhou/V2XPnP) · [CoopTrack](https://github.com/zhongjiaru/CoopTrack) ICCV'25 ·
[R&B-POP](https://github.com/jinsuyoo/rnb-pop) ICLR'25

**Robustness benchmarks** — [AgentComm-Bench](https://arxiv.org/abs/2603.20285) ·
[RCP-Bench](https://github.com/LuckyDush/RCP-Bench) CVPR'25 ·
[How Far Are We?](https://arxiv.org/abs/2509.24927) ASE'25 ·
[CP-FREEZER](https://github.com/WiSeR-Lab/CP-FREEZER)
