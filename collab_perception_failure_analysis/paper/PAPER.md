# Better Silent Than Stale: Attributing Collaborative Perception Failures to Delivery vs. Content Under Constrained Links

*Working paper. All numbers trace to `results/` in this repository and are reproducible
from `configs/matrix.yaml`. Author list and venue formatting pending; reference details
should be verified against publisher records before submission.*

---

## Abstract

Collaborative perception methods are trained and benchmarked on ideal communication
channels, yet deployed links drop, delay, compress, and misalign messages. We ask not
*how much* accuracy degrades under impairment, but *why*: does a method fail because
messages fail to **arrive** (delivery failure), or because messages arrive **wrong** and
poison fusion (content failure)? We instrument seven collaborative 3D detectors — early
fusion, late fusion, AttFuse, F-Cooper, V2VNet, CoAlign, and CoBEVT — with a channel
wrapper that injects eight impairment families at controlled severity, and evaluate 831
conditions on OPV2V using pretrained checkpoints that reproduce published accuracy to
±0.001 AP. Four independent diagnostics — degradation relative to an ego-only *floor*,
precision/recall decomposition, spatial decomposition into ego-visible and occluded
zones, and task-level validation through multi-object tracking — agree on every
attribution.

We find: (1) packet loss never drives any method below the ego-only floor, even at 90%
loss, whereas latency does so at **100 ms** for all seven methods — dropping 90% of
messages is preferable to delivering all of them 200 ms late; (2) content robustness,
not delivery robustness, differentiates architectures (mean-AP spread 0.30–0.51 vs
0.70–0.78), and clean-channel rankings fail to predict it; (3) each fusion mechanism's
specific weakness is the impairment that mimics evidence it was trained to trust;
(4) moderate spatial and temporal error is *worse* than severe error, a misalignment
valley peaking at the magnitudes real localization and networking stacks produce; and
(5) failure attribution depends on the downstream task's temporal structure — burst
losses, provably irrelevant to single-frame detection, cost 15–23% more identity
switches than i.i.d. losses at matched rates in tracking, and constant latency and
oscillating staleness produce opposite tracking failures.

Our results replicate the delivery/content asymmetry reported by AgentComm-Bench
[1] on real neural fusion, but **overturn its latency finding** (they report 0%
normalized degradation for perception under latency; we find latency the most
destructive impairment tested) and **rescale its corruption finding** (their 85.4%
collapse matches our maxout method at 82.4% but overstates the damage to
attention-based fusion by ≈1.5×). We trace both differences to identifiable properties
of their grid-world instantiation, and derive deployment guidance: prioritize freshness
over completeness, compress shared features 8× for free, and avoid maxout or raw-point
fusion on degraded links.

---

## 1. Introduction

Vehicle-to-vehicle (V2V) perception improves detection of occluded and distant objects
by sharing sensor data, intermediate features, or detections between agents
[2, 3, 4]. Published results report these gains on effectively ideal channels.
Robustness studies, where they exist, typically address one impairment — latency
[5, 6], pose error [7], lossy links [8, 9], or adversarial agents [10] — and
conclude that the proposed method degrades more gracefully than baselines.

Aggregate accuracy under impairment conflates two mechanistically distinct failures.
A message that never arrives **removes a benefit**. A message that arrives corrupted
**removes a benefit and adds harm**. The two demand different remedies — network
engineering versus fusion design — and they are distinguishable only if the experiment
is built to distinguish them.

We build that experiment. Our contributions:

1. **An attribution methodology** (§4) with four independent diagnostics whose agreement
   is itself evidence, anchored by an **ego-only floor** that separates "collaboration
   stopped helping" from "collaboration started hurting."
2. **A transparent channel instrument** (§5) validated to bitwise identity when
   disabled, covering delivery impairments (latency, i.i.d. and bursty loss, bandwidth
   quantization) and content impairments (staleness, pose error, ghost injection, scene
   swap), applied uniformly across early, late, and intermediate fusion.
3. **An 831-cell empirical study** (§7) with three seeds per cell, plus 15
   spatial-decomposition cells and 27 tracking cells, all from pretrained checkpoints
   validated against published numbers.
4. **A direct comparison** (§8) with the closest prior protocol, AgentComm-Bench [1],
   which confirms one of its headline findings, corrects a second, and explains both in
   terms of substrate differences.

---

## 2. Background: how collaborative perception works

All evaluated methods share the same detection backbone — **PointPillars** [11] —
and differ only in *what* is transmitted and *how* it is fused. This is what makes the
cohort comparable: differences under impairment are attributable to the fusion
mechanism, not the detector.

**The pipeline.** Each connected autonomous vehicle (CAV) voxelizes its LiDAR sweep into
vertical pillars, encodes each pillar with a small PointNet, and scatters the results
into a 2D bird's-eye-view (BEV) feature map. A convolutional backbone produces
multi-scale BEV features; a detection head regresses 3D boxes and classification scores;
non-maximum suppression (NMS) yields the final detections. Collaboration inserts a
message exchange at one of three points, defining the three **fusion stages**:

| Stage | What is transmitted | Bandwidth | Sensitivity |
|---|---|---|---|
| Early | Raw point clouds | Highest | Corruption enters before any learned filtering |
| Intermediate | BEV feature maps | Tunable | Corruption enters the learned fusion operator |
| Late | 3D boxes + scores | Lowest | Corruption enters only the box merge |

### 2.1 Evaluated algorithms

**No-Comm (the floor).** The ego vehicle detects using only its own LiDAR. Not a
collaborative method but the reference point for the entire study (§4.1).

**Early fusion (Cooper) [12].** Every CAV transmits its raw point cloud. The ego
transforms all received points into its own coordinate frame, concatenates them into one
cloud, and runs a single detector over the union. Fusion is a set union of measurements
— maximally informative, maximally bandwidth-hungry, and with no mechanism to discount a
contributing agent.

**Late fusion.** Each CAV runs the full detector locally and transmits 3D boxes with
confidence scores. The ego transforms received boxes into its frame and merges the pooled
set by NMS. Fusion is a spatial vote among box proposals; only box-level information
crosses the channel.

**AttFuse (OPV2V baseline) [2].** Each CAV encodes its cloud to a BEV feature map and
transmits it; the ego spatially aligns received maps using relative pose and fuses them
with a **single-head self-attention** operator applied per spatial location: each
BEV cell attends across the agents that observe it, producing a weighted combination.
The attention weights give the mechanism a way, in principle, to discount an
uninformative agent at a given location.

**F-Cooper [13].** Same encoder and alignment, but fusion is **element-wise maximum**
across agents' aligned feature maps ("maxout"). Cheap and permutation-invariant, but
the operator has no capacity to discount: the largest activation wins regardless of
which agent produced it or how plausible it is. This is the neural analogue of the
`np.maximum` grid fusion used in AgentComm-Bench [1], and it is the study's
content-fragility extreme.

**V2VNet [3].** Fusion is a **spatially-aware graph neural network**. Each agent is a
node holding its BEV feature map; on each message-passing round, a neighbor's features
are *warped* into the receiver's frame with a differentiable affine transform derived
from relative pose, concatenated with the receiver's own features, passed through a
message CNN, and aggregated (mean or max) before a GRU-style update. The warp makes
V2VNet's geometry-aware — and, as we show, geometry-*dependent*.

**CoAlign [7].** Designed for pose-error robustness. It combines an agent-object pose
graph optimization that corrects relative-pose estimates before fusion with a
**multi-scale attention-with-warp** module applied at each backbone scale. It is the
only method in the cohort with an explicit alignment defense.

**CoBEVT [4].** Fusion is a **fused axial attention (FAX) transformer** over the stacked
multi-agent BEV tensor, attending both within local windows and across a sparse global
grid, so each output cell integrates evidence from all agents and a wide spatial context.
It is the strongest method on a clean channel and, as we show, the most trusting of what
arrives.

### 2.2 Why fusion stage and operator should matter under impairment

Each operator implies a different response to a corrupted message. Maxout takes the
largest activation, so a spurious high activation propagates unchecked. Averaging
dilutes any single agent's anomaly by 1/N. Attention can, in principle, learn to
down-weight an inconsistent agent — but only along dimensions its training exposed it
to. Warping trusts the *metadata* (pose, timing) that positions a message. §7.3 turns
these intuitions into a measured account.

---

## 3. Related work

**Collaborative perception architectures.** Early fusion [12], late fusion, and
intermediate feature fusion [3] span the bandwidth/accuracy trade-off; intermediate
fusion dominates recent work through attention [2, 4, 14], graph message passing
[3], knowledge distillation [15], handshake protocols [16, 17], and spatial
confidence maps [18]. Benchmarks and datasets include OPV2V [2], V2X-Sim [19],
DAIR-V2X [20] (real, vehicle–infrastructure), and V2V4Real [21] (real, V2V).

**Single-impairment robustness.** Latency compensation is addressed by SyncNet [5]
and CoBEVFlow [6]; pose error by CoAlign [7] and V2XPnP [22]; lossy links by
V2VAM [8] and V2X-INCOP [9]; delay-aware fusion by mmCooper [23]; adversarial
agents by ROBOSAC [10]. Each targets one failure mode on one architecture family,
which makes cross-impairment and cross-architecture comparison impossible from the
literature alone.

**Communication-aware evaluation protocols.** AgentComm-Bench [1] is the closest
prior work and the direct motivation for this study: it defines six impairment
dimensions (latency, packet loss, bandwidth collapse, asynchronous updates, stale
memory, conflicting sensor evidence), three grid-world task families (cooperative
perception, navigation, search), five communication strategies including a redundant
coding + staleness-weighting wrapper (ResilientComm), and a metric suite (normalized
performance drop, robustness curves, area under robustness curve, rank stability). Its
central perception finding is an asymmetry: "perception fusion is immune to packet loss
but amplifies corrupted data." Its stated most-important future direction is applying
the protocol "to standard cooperative perception datasets with neural perception
pipelines, where communication impairments will interact with learned representations in
ways our lightweight simulations cannot capture." **This paper is that study**, and §8
reports where the neural instantiation agrees with, contradicts, and explains the
grid-world result.

---

## 4. Method: attributing failures

### 4.1 The ego-only floor

Define the **floor** as the same detector evaluated with all collaborator messages
withheld, scored against the *unchanged* collaborative ground truth (the union of all
agents' annotations, transformed to the ego frame). Then for any impairment condition:

- degradation **toward** the floor ⇒ collaboration's benefit is being lost — a
  **delivery-type** failure;
- degradation **below** the floor ⇒ the ego would do better with its radio switched off
  — collaboration is actively harmful, a **content-type** failure.

Two design requirements make this measurable. The ground truth must **not** shrink when
agents are dropped (otherwise dropping messages removes hard targets and inflates
scores), so all conditions are scored against GT built by a parallel clean pipeline.
And the task must exhibit a **real clean-channel collaboration benefit**, or the floor
coincides with clean performance and only harm is observable. On OPV2V the floor is
AP@0.7 = 0.575 (P 0.825 / R 0.666) against clean collaborative performance of
0.781–0.862 — a benefit of 0.21–0.29 AP.

The floor also calibrates *what collaboration buys*: precision barely moves
(0.825 → 0.85–0.94) while recall jumps 0.666 → 0.87–0.92. Collaboration is almost purely
a **recall** mechanism, which fixes the expected signature of its loss.

### 4.2 Four diagnostics

**D1 — Floor test.** Classify every (method, impairment, severity) cell as
above / at / below floor (margin ±0.02 AP).

**D2 — Precision/recall decomposition.** Predictions: delivery failure ⇒ recall loss,
precision retained (the objects only a collaborator could reveal go missing); additive
content corruption ⇒ precision collapse, recall retained (hallucinations); *misplaced*
evidence ⇒ both collapse, since each wrong box is simultaneously a false positive where
the object is not and a miss where it is.

**D3 — Spatial decomposition.** Partition ground truth into **ego-visible** (box
contains ≥5 of the ego's own LiDAR returns) and **occluded** zones. Delivery failure
should be confined to the occluded zone; content failure should contaminate the
ego-visible zone. We additionally count false positives asserting an object where the
ego's own sensor sees ≥5 points — direct evidence of fusion contamination inside the
ego's field of view.

**D4 — Task-level validation.** Feed impaired detections to a multi-object tracker and
measure MOTA, misses, false positives, identity switches (IDSW), and fragmentation,
testing whether attributions survive a task with temporal state.

---

## 5. The channel instrument

A wrapper intercepts each collaborator message between retrieval and fusion. It attaches
to a built dataset without modifying the underlying framework, so the identical
impairment logic applies to early, late, and intermediate fusion.

### 5.1 Impairment families

**Delivery.**
- **Latency** — collaborator messages are delivered from *k* frames in the past
  (10 Hz data; k ∈ {1,2,4,6,8,10} = 0.1–1.0 s).
- **Packet loss (i.i.d.)** — each (frame, collaborator) message is dropped with
  probability *p* ∈ {0.1,…,0.9}.
- **Packet loss (bursty)** — a two-state Gilbert–Elliott chain [24, 25] per
  collaborator per scenario, with transition probabilities calibrated so the stationary
  loss rate matches the i.i.d. condition, isolating *burst structure* from *rate*.
- **Bandwidth** — uniform quantization of transmitted BEV features to
  {16, 8, 4, 2, 1} bits, applied by forward hooks at each model's fusion input
  (intermediate fusion only; late/early fusion have no feature-level analogue).

**Content.**
- **Stale memory** — messages refresh only every *N* frames; between refreshes the last
  transmitted frame is re-delivered, so message age follows a sawtooth
  (N ∈ {2,…,32} frames = 0.2–3.2 s).
- **Pose error** — Gaussian noise (σ ∈ {0.2,…,3.2} m, with yaw noise coupled at
  2°/m) on the collaborator's reported pose, from which the cav→ego transform is
  recomputed; ground-truth transforms are left untouched, so GT stays anchored to the
  true pose.
- **Ghost injection (fabricated evidence)** — 1–16 vehicle-shaped point clusters
  (220 points sampled on the visible surfaces of a 4.5 × 1.9 × 1.6 m box,
  ground-supported, placed 8–60 m from the collaborator) appended to the transmitted
  cloud.
- **Scene swap (conflicting evidence)** — with probability *q* ∈ {0.1,…,1.0} the
  collaborator's cloud is replaced by one drawn from a *different scenario*: the message
  claims this collaborator's pose but carries another world's content.

### 5.2 Determinism and validity

Every random decision derives from a CRC32 hash of (seed, scenario, frame, agent), so a
(config, seed) pair produces an identical impairment realization regardless of data-loader
worker count or query order, and any cell reproduces exactly. Two independent
executions of the full spatial tier reproduced digit-for-digit.

**Inertness gate.** With all impairments disabled, the collated model inputs must be
*bitwise identical* to the stock pipeline. Verifying this required first controlling the
framework's own test-time stochasticity (point-cloud shuffling), then comparing every
tensor in the collated batch. The gate passes 100/100 frames for the late-fusion,
early-fusion, and intermediate-fusion code paths — establishing that any measured
degradation is caused by the impairment, not the instrument.

---

## 6. Experimental protocol (reproduction steps)

**Step 1 — Environment.** OpenCOOD at commit `31ba160`, PyTorch 1.13.1+cu117,
spconv-cu117 2.3.6, NumPy 1.23.5, single RTX 3080 (12 GB). Exact versions in
`env/VERSIONS.md`.

**Step 2 — Data.** OPV2V [2] test split: 16 scenarios, 2,170 ego frames, 5,985
frame-CAV pairs, 2–7 CAVs per scene (mean 2.59 agents delivering, i.e. 1.59
collaborators per ego frame). Detection range ±140.8 m longitudinal, ±40 m lateral.

**Step 3 — Models.** Seven public pretrained checkpoints (OpenCOOD model zoo and the
CoAlign/CoBEVT releases), all PointPillars-based, evaluated **without retraining** —
the deployment-realistic question is what happens to a perfect-channel-trained model on
an imperfect channel. Checkpoint sources and MD5s in `env/CHECKPOINTS.md`.

**Step 4 — Baseline validation (gate).** Evaluate all checkpoints on the clean channel
and compare with published AP@0.7. Every published value reproduces to **±0.001**
(Table 1). Only after this gate does impairment work begin.

**Step 5 — Instrument validation (gate).** Attach the channel with all impairments
disabled; require bitwise-identical model inputs (§5.2).

**Step 6 — Sweep.** 7 methods × 8 impairment families × 5–6 severities × 3 seeds =
**831 cells**, each evaluated on every third frame (724 frames/cell), writing one record
per cell. Frozen configuration in `configs/matrix.yaml`.

**Step 7 — Aggregate.** Mean ± std over seeds; classify each cell by the floor test.

**Step 8 — Spatial tier.** 3 representative methods (AttFuse, CoAlign, F-Cooper) × 5
conditions × 724 frames, computing zone-wise recall and ego-visible contamination.

**Step 9 — Tracking tier.** 3 methods (CoAlign, CoBEVT, F-Cooper) × 9 conditions over
contiguous frames (stride 1, per-scenario tracker reset).

### 6.1 Metrics

AP@0.3/0.5/0.7 by the VOC-style accumulation used by the framework [26], plus overall
precision and recall at the deployed operating point (score threshold + NMS). For
cross-study comparison we also report **normalized performance drop**,
NPD = (P_clean − P_impaired)/P_clean × 100%, as defined in [1]. Tracking uses
CLEAR-MOT accounting [27].

### 6.2 Tracker (D4 instrument)

A deliberately simple AB3DMOT-style [28] tracker: constant-velocity Kalman filter
[29] in world coordinates, Hungarian assignment [30] on BEV centre distance
(3 m gate), 2-hit confirmation, 3-miss deletion, reset per scenario. Ground-truth tracks
come from the dataset's persistent object identities. This is an *instrument*, not a
contribution: absolute MOTA is not comparable to the tracking literature; only
condition-to-condition contrasts are interpreted.

---

## 7. Results

### 7.1 Clean channel and validation

**Table 1 — Perfect-channel baseline (full 2,170-frame test split).**

| method | fusion | AP@0.5 | AP@0.7 | P@0.7 | R@0.7 | published AP@0.7 |
|---|---|---|---|---|---|---|
| No-Comm (floor) | — | 0.698 | 0.575 | 0.825 | 0.666 | — |
| Late | late | 0.859 | 0.781 | 0.847 | 0.871 | 0.781 |
| Early | early | 0.892 | 0.801 | 0.858 | 0.897 | 0.800 |
| AttFuse | intermediate | 0.905 | 0.815 | 0.889 | 0.900 | 0.815 |
| F-Cooper | intermediate | 0.887 | 0.790 | 0.878 | 0.874 | 0.790 |
| V2VNet | intermediate | 0.917 | 0.822 | 0.885 | 0.913 | 0.822 |
| CoAlign | intermediate | 0.903 | 0.833 | 0.880 | 0.920 | 0.833 |
| CoBEVT | intermediate | 0.914 | 0.862 | 0.934 | 0.909 | 0.861 |

Collaboration adds 0.21–0.29 AP@0.7 over the floor, and the gain is **almost entirely
recall** (+0.20…+0.25) at near-constant precision — the calibration D2 depends on.

### 7.2 Delivery is survivable; content is not

**Table 2 — NPD (%) at maximum tested severity.** Loss = 90%; latency = 1.0 s;
stale = 3.2 s; pose = 3.2 m; swap = 100%; ghosts = 16/message. Bold = below floor.

| method | loss (i.i.d.) | loss (burst) | latency | stale | pose | swap | ghosts |
|---|---|---|---|---|---|---|---|
| F-Cooper | 24.7 | 19.7 | **74.8** | **74.3** | **65.9** | **82.4** | 29.2 |
| Early | 27.7 | 22.2 | **70.3** | **71.3** | **67.2** | **79.3** | **37.7** |
| Late | 21.9 | 18.1 | **60.3** | **61.8** | **48.7** | **64.8** | 19.8 |
| AttFuse | 28.1 | 23.1 | **56.0** | **54.7** | **43.6** | **57.2** | 16.0 |
| V2VNet | 25.9 | 21.2 | **68.2** | **68.4** | **55.8** | **71.4** | 9.2 |
| CoBEVT | 20.8 | 16.9 | **68.1** | **67.9** | **48.6** | **69.1** | 21.2 |
| CoAlign | 25.7 | 21.4 | **53.1** | **51.5** | **38.7** | **53.8** | 27.9 |

**Packet loss never crosses the floor** for any method at any tested rate. **Latency
crosses for all seven at its mildest setting, 100 ms.** Staleness crosses at 0.2–0.4 s,
pose error at 0.2–0.4 m, scene swap at 30–75% corrupted agents. Ghost injection never
crosses except for early fusion at 16 ghosts/message. Bandwidth behaves as delivery
degradation down to 4 bits and as content corruption at 2 bits and below.

The practical inversion: AttFuse scores 0.586 at 90% i.i.d. loss but 0.399 at 200 ms
latency. **Better silent than stale.**

Latency and staleness are *mechanically the same failure*: their NPD columns agree to
within 1.5 points for every method (F-Cooper 74.8 vs 74.3; CoAlign 53.1 vs 51.5;
CoBEVT 68.1 vs 67.9). Both deliver a message describing a world that has moved.

### 7.3 Signatures and the fusion-mechanism account

**D2.** Both predicted signatures hold for all seven methods without exception:
loss costs recall (ΔR −0.14…−0.19) with precision retained (ΔP −0.01…−0.11);
ghosts cost precision (ΔP −0.06…−0.30) with recall retained (ΔR −0.01…−0.07);
latency and swap collapse both.

**D3 (Table 3, spatial).** Under 90% loss, occluded-zone recall falls 0.50–0.59 while
ego-visible recall falls only 0.06–0.08 (≈8:1 selectivity). Under 200 ms latency,
ego-visible recall falls 0.21–0.46 and ego-visible false positives rise 3.4–4.6×:
corrupted messages measurably degrade detection of objects the ego sees perfectly well
by itself. The contamination magnitude orders **F-Cooper (−0.46) > AttFuse (−0.26) >
CoAlign (−0.21)**, reproducing the aggregate content-fragility ranking from an
independent measurement.

| method | condition | R_visible | R_occluded | FP/frame | FP_egovisible/frame |
|---|---|---|---|---|---|
| AttFuse | identity | 0.933 | 0.761 | 1.70 | 0.95 |
| AttFuse | loss 90% | 0.864 | 0.204 | 3.25 | 2.05 |
| AttFuse | latency 200 ms | 0.673 | 0.399 | 5.58 | 3.54 |
| AttFuse | ghosts ×8 | 0.926 | 0.748 | 3.27 | 1.17 |
| AttFuse | swap 50% | 0.846 | 0.449 | 5.13 | 1.76 |
| CoAlign | identity | 0.953 | 0.783 | 1.88 | 0.93 |
| CoAlign | loss 90% | 0.894 | 0.282 | 3.30 | 1.80 |
| CoAlign | latency 200 ms | 0.747 | 0.307 | 5.62 | 3.14 |
| CoAlign | ghosts ×8 | 0.946 | 0.771 | 5.77 | 1.54 |
| CoAlign | swap 50% | 0.912 | 0.545 | 5.91 | 1.95 |
| F-Cooper | identity | 0.913 | 0.713 | 1.86 | 1.24 |
| F-Cooper | loss 90% | 0.838 | 0.123 | 2.51 | 2.02 |
| F-Cooper | latency 200 ms | 0.456 | 0.201 | 7.76 | 5.65 |
| F-Cooper | ghosts ×8 | 0.890 | 0.690 | 4.89 | 1.60 |
| F-Cooper | swap 50% | 0.617 | 0.362 | 4.83 | 1.90 |

**The account.** Mean AP over severities spans 0.70–0.78 across methods for delivery
impairments but 0.30–0.51 for content impairments: content robustness differentiates
architectures ≈2.6× more strongly. Under loss the clean ranking survives nearly intact;
under every misalignment-type impairment it scrambles — the clean leader (CoBEVT) falls
to mid-pack while CoAlign and AttFuse take the top places. The governing principle:
**each fusion mechanism is as content-robust as its ability to discount an arriving
message, and its specific weakness is the impairment that mimics evidence it was trained
to trust.**

- *Maxout* (F-Cooper) cannot discount anything: last or second-to-last under every
  content impairment, mid-pack under delivery, and the highest baseline contamination
  even on a clean channel (FP_egovis 1.24 vs 0.93–0.95).
- *Graph averaging* (V2VNet) dilutes amplitude anomalies — best ghost robustness in the
  cohort (9.2% NPD, 2.4× better precision retention than the next method) — but warps
  messages using pose and timing metadata, making it bottom-tier under staleness and
  latency. Same mechanism, opposite outcomes.
- *Fused attention* (CoBEVT) trusts consensus: precision immune to loss (ΔP −0.022,
  the purest delivery profile observed) at the cost of the largest relative latency
  penalty and a bandwidth cliff at 2 bits.
- *Alignment-robust training* (CoAlign) is the only defense that **transfers**: it wins
  latency, staleness, pose, and swap — not merely the pose error it targets — at the
  cost of worst-in-class ghost precision (a mechanism that reconciles misaligned
  evidence also legitimizes fabricated evidence).

### 7.4 The misalignment valley

Pose error is **non-monotonic for every method**: damage peaks at 0.8–1.6 m and partially
recovers at 3.2 m (F-Cooper 0.172 → 0.269 AP; CoBEVT 0.304 → 0.443; CoAlign 0.454 →
0.511). Late fusion shows the same valley under *latency* (0.221 at 200 ms → 0.310 at
1 s). Controls: the recovery survives with and without dropping messages rendered empty
by extreme displacement (AttFuse AP identical either way), and appears in late fusion
where no such filtering exists.

Interpretation: moderately displaced evidence is **plausible** — it overlaps real objects
and competes with correct detections in NMS and in feature space. Severely displaced
evidence lands nowhere plausible and becomes ignorable clutter. Precision recovers
faster than recall at extreme severities, consistent with this reading.

Consequence: **the damage peak sits at exactly the error magnitudes real localization and
networking stacks produce** (≈1 m, ≈100–400 ms). Defenses that bound the worst case by
rejecting extreme outliers address the easy part of the problem.

### 7.5 Bandwidth: free, then catastrophic

16 → 8 → 4-bit quantization of shared features costs ≤0.05 AP for six of seven methods —
an 8× communication saving for noise-level cost. At 2 bits the cliff arrives (F-Cooper
0.476, CoAlign 0.577, CoBEVT 0.322) and the floor test reclassifies bandwidth from a
delivery to a content impairment. One reproducible anomaly: CoBEVT scores *worse* at
2 bits (0.322) than at 1 bit (0.452), σ ≤ 0.001 — plausibly because 1-bit features
approximate an occupancy mask its attention can reinterpret, while 2-bit features retain
misleading amplitude structure.

### 7.6 Attribution depends on the downstream task (tracking)

**Table 4 — Tracking (full split, stride 1).** Burst conditions share the marginal loss
rate of their i.i.d. pair; `_long` bursts average ~10 frames (beyond the tracker's
3-frame coast window) vs ~3.3 frames (at it).

| method | condition | MOTA | FN | FP | IDSW | FRAG |
|---|---|---|---|---|---|---|
| CoAlign | clean | 0.835 | 1331 | 3914 | 149 | 483 |
| CoAlign | iid 30% | 0.809 | 1547 | 4492 | 185 | 550 |
| CoAlign | burst 30% | 0.806 | 1612 | 4502 | 200 | 563 |
| CoAlign | burst 30% long | 0.802 | 1761 | 4494 | 208 | 567 |
| CoAlign | iid 70% | 0.728 | 2708 | 5813 | 349 | 856 |
| CoAlign | burst 70% | 0.734 | 2583 | 5716 | 362 | 815 |
| CoAlign | burst 70% long | 0.701 | 3394 | 5942 | 413 | 892 |
| CoAlign | stale 0.4 s | 0.653 | 3193 | 7474 | 629 | 1394 |
| CoAlign | latency 0.2 s | 0.654 | 4106 | 6935 | 223 | 806 |
| CoBEVT | clean | 0.881 | 1832 | 1883 | 160 | 511 |
| CoBEVT | iid 70% | 0.815 | 3760 | 1944 | 341 | 788 |
| CoBEVT | burst 70% long | 0.795 | 4198 | 2055 | 418 | 844 |
| CoBEVT | stale 0.4 s | 0.541 | 5845 | 8091 | 1031 | 2248 |
| CoBEVT | latency 0.2 s | 0.555 | 7305 | 6980 | 230 | 966 |
| F-Cooper | clean | 0.818 | 2428 | 3328 | 167 | 597 |
| F-Cooper | iid 70% | 0.742 | 4222 | 3828 | 375 | 872 |
| F-Cooper | burst 70% long | 0.711 | 4964 | 4042 | 430 | 937 |
| F-Cooper | stale 0.4 s | 0.505 | 6239 | 8987 | 917 | 2109 |
| F-Cooper | latency 0.2 s | 0.478 | 8484 | 8284 | 248 | 1021 |

*(Full 27-cell table in `results/ANALYSIS.md` §9.)*

**Burstiness matters for tracking but not detection.** At matched loss rates, i.i.d. and
bursty loss differ by ≤0.005 AP in detection but cost 15–23% more identity switches in
tracking, with a clean dose-response in burst length (F-Cooper at 30% loss: 196 → 265 →
282 IDSW). Short bursts that fit inside the tracker's coast window barely move MOTA while
already inflating IDSW ~35% — aggregate metrics hide identity churn that prediction and
planning would feel. **Burstiness irrelevance is a property of the stateless task, not
of the channel.**

**Constant latency and oscillating staleness produce opposite tracking failures.** They
cost similar detection AP, yet staleness explodes identity switches 4–6× above clean
(629 / 1031 / 917) while constant latency leaves IDSW near-clean (223 / 230 / 248) and
instead dominates in misses. A constant delay is a *consistent* bias — the motion model
locks onto steadily displaced detections and identities stay coherent; sawtooth staleness
makes detections *oscillate* between true and displaced positions each refresh cycle,
shredding association. **A motion model absorbs consistent temporal error and amplifies
oscillating temporal error.** For tracking-based stacks, a predictably delayed channel is
preferable to a variably fresh one at equal average age — the opposite of the
detection-level recommendation.

---

## 8. Comparison with AgentComm-Bench [1]

AgentComm-Bench evaluates cooperative perception as four agents observing a static
20 × 20 grid of 30 objects through 90° quadrant fields of view, sharing flattened
detection grids fused by element-wise maximum, scored by F1. Our study is the neural
instantiation its conclusion calls for. Three outcomes:

**(a) Confirmed — the delivery/content asymmetry.** They report perception "immune to
packet loss but amplifies corrupted data" (0% NPD transport, 85.4% content). We measure
21–28% NPD under loss with *no method ever falling below the floor*, versus 39–82% NPD
under content corruption with *every method falling below it*. The asymmetry survives the
transition to learned representations and real sensor geometry — and gains a sharper
criterion, since the floor distinguishes lost benefit from added harm, which NPD alone
cannot.

**(b) Contradicted — latency.** They report **0% NPD for perception under latency** at
up to 500 ms and group latency as a transport-layer impairment. We find latency the
**most destructive** impairment tested: all seven architectures below the floor at
100 ms, 53–75% NPD at 1 s. The discrepancy is explained by their substrate: in a
**static** scene a delayed message is *identical* to a fresh one, so 0% NPD measures
scene staticness, not fusion tolerance. In driving, agents and objects move at
10–20 m/s, so a 200 ms-old message describes a world displaced by 2–4 m and latency
*becomes* content corruption — which is why our latency and staleness columns coincide
(§7.2), while in their setup only staleness (their D5, which corrupts the agent-state
model rather than delaying a static observation) shows damage. The practical
recommendation reverses: their result implies perception designers may deprioritize
latency; ours makes it the first thing to engineer against.

**(c) Rescaled — the magnitude of content collapse.** Their 85.4% NPD is a property of
`np.maximum` fusion, as their own ablation demonstrates (staleness weighting λ has *zero*
effect because "the fusion is a hard selector"). Our F-Cooper — the neural maxout
analogue — reaches **82.4% NPD**, within three points. But CoAlign reaches 53.8% and
AttFuse 57.2%. Generalizing the grid-world figure to "cooperative perception" therefore
overstates the damage to attention- and alignment-based systems by ≈1.5×. We also
partially answer their open question — they call for "a fusion mechanism that could
detect and reject corrupted inputs" — with evidence that alignment-robust *training*
(CoAlign) achieves this and transfers across the whole misalignment family.

**Why the floor test is possible here and not there.** Their CP task has **no
clean-channel collaboration benefit** (No-Comm = Full-Comm = 95.8 F1, acknowledged as a
limitation). A task where collaboration buys nothing can measure harm but never lost
benefit, which is also why their bandwidth-collapse NPD is 0% — full collapse merely
returns their CP task to a No-Comm baseline that was already equal. Our floor sits
0.21–0.29 AP below collaborative performance, making both directions observable.

**Complementary strengths.** They cover three task families (navigation and search
degrade >96%, beyond anything perception exhibits), propose a defense (redundant coding +
staleness weighting), and run 29,700 episodes in 3.5 minutes on one CPU core — an
accessibility we cannot match. We add architectural depth (seven published models,
validated to ±0.001), the floor criterion, spatial decomposition, and the tracking tier.
One transfer note: their ResilientComm's redundancy targets packet loss, which our floor
test shows was never the dangerous failure for perception; its staleness-aware half
targets the failure that actually inverts collaboration's sign — but their own ablation
shows weighting cannot repair maxout fusion.

---

## 9. Deployment implications

1. **Prioritize freshness over completeness.** Every architecture prefers losing most
   messages to receiving slightly old ones. Retransmitting stale data is
   counterproductive; aggressive expiry (≤100 ms) is the highest-value channel policy.
2. **Compress features 8×.** No method needs more than 4 bits per feature value; below
   that, withhold rather than transmit.
3. **Match fusion to link.** Lossy-but-fresh links favor attention fusion (CoBEVT);
   poorly synchronized or localized fleets favor alignment-robust training (CoAlign);
   spoofing risk favors averaging (V2VNet); maxout and raw-point fusion are poor choices
   on any degraded link.
4. **Choose channel discipline by downstream task.** Detection stacks should minimize
   message *age*; tracking stacks should additionally minimize age *variance* and burst
   length.
5. **The open problem is temporal alignment.** No method tolerates a single frame of
   delay. CoAlign's transfer from spatial to temporal robustness suggests
   alignment-style training against *temporal* misalignment — the direction taken by
   SyncNet [5] and CoBEVFlow [6], which this cohort does not include — as the most
   promising path.

---

## 10. Limitations

Single dataset (OPV2V, simulated) and a single detector backbone family (PointPillars);
real-world validation on DAIR-V2X [20] or V2V4Real [21] remains future work. The
floor derives from the late-fusion checkpoint evaluated ego-only rather than
per-backbone floors; the ±0.02 classification margin absorbs part of this, and cross-method
*patterns* are robust to it. Sweeps use every third frame (baseline reproduction at this
stride matches the full split to ±0.003). Spatial and tracking tiers use representative
method subsets. The AttFuse bandwidth impairment intercepts backbone input because its
fusion is interleaved with the backbone. The tracker is intentionally simple, so absolute
MOTA is not comparable to the tracking literature. Two dedicated robustness methods
(SyncNet, CoBEVFlow) and heterogeneous V2I settings are not evaluated. Closed-loop
driving, which would connect these failures to behavioral consequences, is unaddressed.

---

## 11. Conclusion

Collaborative perception fails in two mechanistically distinct ways, and conflating them
misdirects engineering effort. Losing messages is survivable: it removes the
occluded-zone recall collaboration provides and never makes an agent worse off than
driving alone. Receiving *wrong* messages is not: corruption reaches inside the ego's own
field of view, and 100 ms of latency makes every architecture we tested worse than
silence. Which corruption hurts most is set by the fusion mechanism — each architecture's
weakness is the impairment that mimics evidence it was trained to trust — and by the
downstream task's temporal structure, which decides whether burst structure and freshness
variance matter at all. The resulting guidance is simple and, in one respect,
counterintuitive: on a constrained link, silence is better than staleness.

---

## References

[1] A. Bansal and I. Gangwani. "AgentComm-Bench: Stress-Testing Cooperative Embodied AI
Under Latency, Packet Loss, and Bandwidth Collapse." arXiv:2603.20285, 2026.

[2] R. Xu, H. Xiang, X. Xia, X. Han, J. Li, and J. Ma. "OPV2V: An Open Benchmark Dataset
and Fusion Pipeline for Perception with Vehicle-to-Vehicle Communication." *IEEE
International Conference on Robotics and Automation (ICRA)*, 2022.

[3] T.-H. Wang, S. Manivasagam, M. Liang, B. Yang, W. Zeng, and R. Urtasun. "V2VNet:
Vehicle-to-Vehicle Communication for Joint Perception and Prediction." *European
Conference on Computer Vision (ECCV)*, 2020.

[4] R. Xu, Z. Tu, H. Xiang, W. Shao, B. Zhou, and J. Ma. "CoBEVT: Cooperative Bird's Eye
View Semantic Segmentation with Sparse Transformers." *Conference on Robot Learning
(CoRL)*, 2022.

[5] Z. Lei, S. Ren, Y. Hu, W. Zhang, and S. Chen. "Latency-Aware Collaborative
Perception." *European Conference on Computer Vision (ECCV)*, 2022.

[6] S. Wei, Y. Hu, Y. Lu, Y. Zhong, S. Chen, et al. "Asynchrony-Robust Collaborative
Perception via Bird's Eye View Flow." *Advances in Neural Information Processing Systems
(NeurIPS)*, 2023.

[7] Y. Lu, Q. Li, B. Liu, M. Dianati, C. Feng, S. Chen, and Y. Wang. "Robust
Collaborative 3D Object Detection in Presence of Pose Errors." *IEEE International
Conference on Robotics and Automation (ICRA)*, 2023. (CoAlign)

[8] J. Li, R. Xu, X. Liu, J. Ma, Z. Chi, J. Ma, and H. Yu. "Learning for
Vehicle-to-Vehicle Cooperative Perception under Lossy Communication." *IEEE Transactions
on Intelligent Vehicles*, 2023. (V2VAM)

[9] S. Ren, Z. Lei, Z. Wang, S. Chen, and W. Zhang. "Robust Collaborative Perception
against Communication Interruption." arXiv preprint, 2023. (V2X-INCOP)

[10] Y. Li, Q. Fang, J. Bai, S. Chen, F. Juefei-Xu, and C. Feng. "Among Us: Adversarially
Robust Collaborative Perception by Consensus." *IEEE/CVF International Conference on
Computer Vision (ICCV)*, 2023. (ROBOSAC)

[11] A. H. Lang, S. Vora, H. Caesar, L. Zhou, J. Yang, and O. Beijbom. "PointPillars:
Fast Encoders for Object Detection from Point Clouds." *IEEE/CVF Conference on Computer
Vision and Pattern Recognition (CVPR)*, 2019.

[12] Q. Chen, S. Tang, Q. Yang, and S. Fu. "Cooper: Cooperative Perception for Connected
Autonomous Vehicles Based on 3D Point Clouds." *IEEE International Conference on
Distributed Computing Systems (ICDCS)*, 2019.

[13] Q. Chen, X. Ma, S. Tang, J. Guo, Q. Yang, and S. Fu. "F-Cooper: Feature Based
Cooperative Perception for Autonomous Vehicle Edge Computing System." *ACM/IEEE Symposium
on Edge Computing (SEC)*, 2019.

[14] R. Xu, H. Xiang, Z. Tu, X. Xia, M.-H. Yang, and J. Ma. "V2X-ViT:
Vehicle-to-Everything Cooperative Perception with Vision Transformer." *European
Conference on Computer Vision (ECCV)*, 2022.

[15] Y. Li, S. Ren, P. Wu, S. Chen, C. Feng, and W. Zhang. "Learning Distilled
Collaboration Graph for Multi-Agent Perception." *Advances in Neural Information
Processing Systems (NeurIPS)*, 2021. (DiscoNet)

[16] Y.-C. Liu, J. Tian, N. Glaser, and Z. Kira. "When2com: Multi-Agent Perception via
Communication Graph Grouping." *IEEE/CVF Conference on Computer Vision and Pattern
Recognition (CVPR)*, 2020.

[17] Y.-C. Liu, J. Tian, C.-Y. Ma, N. Glaser, C.-W. Kuo, and Z. Kira. "Who2com:
Collaborative Perception via Learnable Handshake Communication." *IEEE International
Conference on Robotics and Automation (ICRA)*, 2020.

[18] Y. Hu, S. Fang, Z. Lei, Y. Zhong, and S. Chen. "Where2comm:
Communication-Efficient Collaborative Perception via Spatial Confidence Maps." *Advances
in Neural Information Processing Systems (NeurIPS)*, 2022.

[19] Y. Li, D. Ma, Z. An, Z. Wang, Y. Zhong, S. Chen, and C. Feng. "V2X-Sim: Multi-Agent
Collaborative Perception Dataset and Benchmark for Autonomous Driving." *IEEE Robotics
and Automation Letters*, 2022.

[20] H. Yu, Y. Luo, M. Shu, Y. Huo, Z. Yang, Y. Shi, Z. Guo, H. Li, X. Hu, J. Yuan, and
Z. Nie. "DAIR-V2X: A Large-Scale Dataset for Vehicle-Infrastructure Cooperative 3D Object
Detection." *IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*,
2022.

[21] R. Xu, X. Xia, J. Li, H. Li, S. Zhang, Z. Tu, Z. Meng, H. Xiang, X. Dong, R. Song,
H. Yu, B. Zhou, and J. Ma. "V2V4Real: A Real-World Large-Scale Dataset for
Vehicle-to-Vehicle Cooperative Perception." *IEEE/CVF Conference on Computer Vision and
Pattern Recognition (CVPR)*, 2023.

[22] Z. Zhou, et al. "V2X-PnP: Perception and Planning for Collaborative
Vehicle-to-Everything." *IEEE/RSJ International Conference on Intelligent Robots and
Systems (IROS)*, 2023.

[23] B. Zhang, et al. "mmCooper: Collaborative 3D Object Detection with Confidence-Guided
Multi-Modality Fusion." *AAAI Conference on Artificial Intelligence*, 2024.

[24] E. N. Gilbert. "Capacity of a Burst-Noise Channel." *Bell System Technical Journal*,
39(5):1253–1265, 1960.

[25] E. O. Elliott. "Estimates of Error Rates for Codes on Burst-Noise Channels." *Bell
System Technical Journal*, 42(5):1977–1997, 1963.

[26] M. Everingham, L. Van Gool, C. K. I. Williams, J. Winn, and A. Zisserman. "The
Pascal Visual Object Classes (VOC) Challenge." *International Journal of Computer
Vision*, 88(2):303–338, 2010.

[27] K. Bernardin and R. Stiefelhagen. "Evaluating Multiple Object Tracking Performance:
The CLEAR MOT Metrics." *EURASIP Journal on Image and Video Processing*, 2008.

[28] X. Weng, J. Wang, D. Held, and K. Kitani. "3D Multi-Object Tracking: A Baseline and
New Evaluation Metrics." *IEEE/RSJ International Conference on Intelligent Robots and
Systems (IROS)*, 2020. (AB3DMOT)

[29] R. E. Kalman. "A New Approach to Linear Filtering and Prediction Problems."
*Journal of Basic Engineering*, 82(1):35–45, 1960.

[30] H. W. Kuhn. "The Hungarian Method for the Assignment Problem." *Naval Research
Logistics Quarterly*, 2(1–2):83–97, 1955.

[31] A. Dosovitskiy, G. Ros, F. Codevilla, A. López, and V. Koltun. "CARLA: An Open Urban
Driving Simulator." *Conference on Robot Learning (CoRL)*, 2017.

*Verification note: reference [9] is an arXiv preprint whose venue may have changed;
[22] and [23] should be checked against publisher records for author lists and page
numbers before submission.*

---

## Appendix A — Artifact map

| Component | Path |
|---|---|
| Channel instrument | `commchannel/{config,schedule,channel,feature_hooks}.py` |
| Frozen experiment grid | `configs/matrix.yaml` |
| Environment gates | `scripts/verify_phase0.py`, `docs/PHASE0_SETUP.md` |
| Baseline runner | `scripts/run_phase1.py` |
| Instrument inertness gate | `scripts/run_phase2_identity.py` |
| Sweep runner / aggregator | `scripts/run_phase3.py`, `scripts/aggregate_sweeps.py` |
| Spatial decomposition | `scripts/run_phase43.py` |
| Tracking tier | `scripts/run_phase5_tracking.py` |
| Unit tests | `scripts/test_commchannel.py` |
| Frozen baseline table | `results/baseline.md` |
| Master sweep summary (277 rows) | `results/sweep_summary.md` |
| Full grid, methods as columns | `results/sweep_table.md`, `scripts/pivot_sweep_table.py` |
| Full analysis | `results/ANALYSIS.md` |
| Experimental history and gates | `IMPLEMENTATION.md` |

## Appendix B — Severity grids

| Impairment | Levels |
|---|---|
| Latency | 1, 2, 4, 6, 8, 10 frames (0.1–1.0 s @ 10 Hz) |
| Packet loss (i.i.d.) | 10, 30, 50, 70, 90 % |
| Packet loss (bursty) | matched marginal rates 10–90 %, p(bad→good) = 0.3 |
| Bandwidth | 16, 8, 4, 2, 1 bits (intermediate fusion only) |
| Stale memory | refresh every 2, 4, 8, 16, 32 frames |
| Pose error | σ = 0.2, 0.4, 0.8, 1.6, 3.2 m (yaw coupled 2°/m) |
| Ghost injection | 1, 2, 4, 8, 16 vehicles per message |
| Scene swap | 10, 30, 50, 75, 100 % of messages |
| Tracking bursts | p(bad→good) = 0.3 (~3.3 frames) and 0.1 (~10 frames) |
