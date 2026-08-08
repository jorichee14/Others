# Better Silent Than Stale: Attributing Collaborative Perception Failures to Delivery vs. Content

*Draft — author list, venue formatting, and figures pending. All numbers trace to
`results/` in this repository; every cell is reproducible from `configs/matrix.yaml`.*

---

## Abstract

Collaborative perception methods are trained and benchmarked on perfect communication
channels, yet deployed links drop, delay, compress, and misalign messages. We ask not
*how much* performance degrades under impairment but *why*: does a method fail because
messages fail to **arrive** (delivery), or because messages arrive **wrong** and poison
fusion (content)? We instrument seven collaborative 3D detectors (early, late, AttFuse,
F-Cooper, V2VNet, CoAlign, CoBEVT) with a transparent channel wrapper that injects eight
impairment families at controlled severity, and evaluate 831 conditions on OPV2V using
pretrained checkpoints. Four independent diagnostics — degradation relative to an
ego-only floor, precision/recall decomposition, spatial decomposition into ego-visible
and occluded zones, and task-level validation through multi-object tracking — agree on
every attribution. We find that (i) packet loss never drives any method below the
ego-only floor even at 90% loss, while latency does so at **100 ms** for all seven
methods: dropping 90% of messages is preferable to delivering them 200 ms late;
(ii) content robustness, not delivery robustness, differentiates architectures (spread
0.30–0.51 vs 0.70–0.78 mean AP), and clean-channel rankings do not predict it;
(iii) each fusion mechanism's specific vulnerability is the impairment that mimics
evidence it was trained to trust; (iv) moderate spatial/temporal error is *worse* than
severe error — a misalignment valley peaking at exactly the magnitudes real localization
and networking stacks produce; and (v) attribution depends on the task's temporal
structure: burst losses, provably irrelevant to single-frame detection, cost 15–23% more
identity switches than i.i.d. losses at matched rates in tracking, and constant latency
and oscillating staleness — near-twins at detection level — produce opposite tracking
failures. We derive concrete deployment guidance: prioritize freshness over
completeness, compress shared features 8× for free, and avoid maxout or raw-point fusion
on degraded links.

## 1. Introduction

Vehicle-to-vehicle perception improves detection of occluded and distant objects by
sharing sensor data, features, or detections between agents. The literature reports
these gains on effectively ideal channels; robustness studies, when present, typically
report aggregate accuracy under a single impairment (usually latency or pose error) and
conclude that the proposed method degrades more gracefully than baselines.

Aggregate accuracy conflates two mechanistically distinct failures. A message that never
arrives removes a benefit; a message that arrives corrupted removes a benefit **and adds
harm**. These call for different remedies — network engineering versus fusion design —
and they are distinguishable only if the analysis is built to distinguish them.

We build that analysis. Our contributions:

1. **An attribution methodology** with four independent diagnostics whose agreement (or
   disagreement) is itself evidence, anchored by an ego-only *floor* that separates
   "collaboration stopped helping" from "collaboration started hurting."
2. **A transparent channel instrument** validated to bitwise identity when disabled,
   covering delivery (latency, i.i.d./bursty loss, bandwidth quantization) and content
   (staleness, pose error, ghost injection, scene swap) impairments uniformly across
   early, late, and eight intermediate-fusion architectures.
3. **An 831-cell empirical study** with 3 seeds per cell, plus 15 spatial-decomposition
   cells and 27 tracking cells, all from pretrained checkpoints that reproduce published
   numbers to ±0.001.
4. **Findings** that revise common assumptions: latency is a *content* failure, not a
   bandwidth-like degradation; the damage peak sits at plausible rather than extreme
   error magnitudes; and burstiness matters only for tasks with temporal state.

## 2. Related work (to be expanded with citations)

Collaborative perception architectures span early fusion (raw point sharing), late
fusion (detection sharing), and intermediate fusion (feature sharing), the last
dominating recent work via attention (AttFuse, V2X-ViT, CoBEVT), graph message passing
(V2VNet), spatial confidence (Where2comm), and alignment-robust training (CoAlign).
Robustness studies address latency compensation (SyncNet, CoBEVFlow), pose error
(CoAlign), lossy links (V2VAM, V2X-INCOP), and adversarial agents (ROBOSAC), typically
one impairment per paper and one architecture family per study. Our contribution is
orthogonal: a uniform, multi-impairment attribution framework applied to a cohort, which
lets us compare *failure mechanisms* rather than robustness scores.

## 3. Method

### 3.1 The floor: separating lost benefit from added harm

We define the **ego-only floor** as the same detector evaluated with collaborator
messages withheld, scored against the *unchanged* collaborative ground truth (the union
of all agents' annotations). Degradation *toward* the floor means collaboration's
benefit is being lost; degradation *below* it means collaboration is actively harmful —
the ego would be better off alone. On OPV2V this floor is AP@0.7 = 0.575 (P 0.825 /
R 0.666), against clean collaborative performance of 0.781–0.862.

The floor also calibrates what collaboration *buys* on a perfect channel: precision
barely moves (0.825 → 0.85–0.94) while recall jumps 0.666 → 0.87–0.92. Collaboration is
almost purely a recall mechanism, which sets the expected signature of its loss.

### 3.2 Channel instrument

A wrapper intercepts per-agent messages between retrieval and fusion, implementing
delivery impairments (constant latency; i.i.d. Bernoulli loss; Gilbert-Elliott bursty
loss calibrated to matched marginal rates; uniform feature quantization at 16/8/4/2/1
bits) and content impairments (staleness as sawtooth message age; Gaussian pose error
with coupled yaw; injection of car-shaped ghost point clusters; replacement of a
collaborator's cloud with one from a different scenario). All randomness derives from
hashed (seed, scenario, frame, agent) keys, making every cell exactly reproducible and
worker-count independent.

Two validity requirements shaped the design. First, **ground truth must not move**:
because the framework builds GT as a union over present agents, dropping messages would
otherwise shrink the target set and inflate scores; we therefore always score against
GT from a parallel clean pipeline. Second, **the instrument must be provably inert**: with
all impairments disabled, the collated model inputs are bitwise identical to the stock
pipeline (verified across 100 frames for late, early-style, and intermediate paths).

### 3.3 Diagnostics

- **D1 Floor test.** Classify each (method, impairment, severity) as above / at / below
  floor.
- **D2 Precision–recall decomposition.** Delivery failure predicts recall loss with
  precision retained; additive content failure predicts precision collapse with recall
  retained; misplacement predicts joint collapse.
- **D3 Spatial decomposition.** Partition GT into *ego-visible* (≥5 of the ego's own
  lidar returns inside the box) and *occluded* zones. Delivery failure should be
  confined to the occluded zone; content failure should contaminate the ego-visible
  zone, measured additionally as false positives asserting objects where the ego's own
  sensor sees points.
- **D4 Task-level validation.** Feed impaired detections to a constant-velocity Kalman
  tracker and measure MOTA, misses, false positives, identity switches, fragmentation —
  testing whether attributions survive a task with temporal state.

### 3.4 Experimental protocol

OPV2V test split (2,170 ego frames, 16 scenarios), pretrained checkpoints evaluated
without retraining — the deployment-realistic question is what happens to a
perfect-channel-trained model on an imperfect channel. Seven methods × eight impairment
families × 5–6 severities × 3 seeds = 831 detection cells (stride 3); spatial and
tracking tiers use representative subsets at full or stride-1 resolution. Our pipeline
reproduces every published AP@0.7 in the cohort to ±0.001.

## 4. Results

### 4.1 Delivery is survivable; content is not

Packet loss — i.i.d. or bursty — never drives any of the seven methods below the floor,
even at 90% loss (worst case 0.579, floor 0.575). Latency drives **all seven** below the
floor at its mildest tested setting, 100 ms. Staleness, pose error, and scene swap cross
below at 0.2–0.4 s, 0.2–0.4 m, and 30–75% corrupted-agent rates respectively. Ghost
injection never crosses (except early fusion at 16 ghosts/message). Bandwidth
quantization behaves as delivery degradation down to 4 bits and as content corruption at
2 bits and below.

The practical inversion: at 90% i.i.d. loss AttFuse scores 0.586; at 200 ms latency it
scores 0.399. **Better silent than stale.**

### 4.2 The signatures separate cleanly

D2 confirms both predicted signatures across all seven methods without exception: loss
costs recall (ΔR −0.14…−0.19) with precision retained (ΔP −0.01…−0.11); ghosts cost
precision (ΔP −0.06…−0.30) with recall retained (ΔR −0.01…−0.07); latency and swap
collapse both, the signature of *misplaced* evidence, which is simultaneously a false
positive where the object is not and a miss where it is.

D3 localizes these mechanisms in space. Under 90% loss, occluded-zone recall falls
0.50–0.59 while ego-visible recall falls only 0.06–0.08 (≈8:1 selectivity). Under 200 ms
latency, ego-visible recall falls 0.21–0.46 and ego-visible false positives rise
3.4–4.6×: corrupted messages measurably degrade detection of objects the ego sees
perfectly well by itself. The contamination magnitude orders F-Cooper > AttFuse >
CoAlign — the same content-fragility ranking the aggregate sweep produced, recovered by
an independent diagnostic.

### 4.3 Content robustness differentiates architectures; delivery robustness does not

Mean AP over severities spans 0.70–0.78 across methods for delivery impairments but
0.30–0.51 for content impairments. Under loss, the clean-channel ranking survives nearly
intact; under every misalignment-type impairment it scrambles — the clean-channel leader
(CoBEVT) falls to mid-pack, and CoAlign and AttFuse take the top two places.

**Fusion-mechanism account.** Each mechanism is as content-robust as its ability to
discount an arriving message, and its specific weakness is the impairment that mimics
evidence it was trained to trust:

- *Maxout* (F-Cooper) cannot discount anything: last or second-to-last under every
  content impairment, mid-pack under delivery, and the highest baseline contamination
  even on a clean channel.
- *Averaging over a graph* (V2VNet) dilutes amplitude anomalies — best ghost precision
  by 2.4× — but warps messages using pose and timing metadata, making it bottom-tier
  under staleness and latency. Same mechanism, opposite outcomes.
- *Fused attention* (CoBEVT) trusts consensus: precision immune to loss (ΔP −0.022, the
  purest delivery profile observed) but the largest relative latency penalty.
- *Alignment-robust training* (CoAlign) is the only defense that transfers: it wins
  latency, staleness, pose, and swap — not just the pose error it targets — at the cost
  of worst-in-class ghost precision (a mechanism that reconciles misaligned evidence
  also legitimizes fabricated evidence).

### 4.4 The misalignment valley

Pose error is non-monotonic for every method: damage peaks at 0.8–1.6 m and partially
recovers at 3.2 m (F-Cooper 0.172 → 0.269; CoBEVT 0.304 → 0.443). Late fusion shows the
same valley under latency (0.221 at 200 ms → 0.310 at 1 s). Moderately displaced
evidence is *plausible* — it overlaps real objects and competes with correct detections;
severely displaced evidence lands nowhere plausible and is effectively ignorable. The
damage peak therefore sits at precisely the error magnitudes real localization and
networking stacks produce (≈1 m, ≈100–400 ms), and defenses that bound the worst case by
rejecting extreme outliers address the easy part of the problem.

### 4.5 Bandwidth is nearly free until it is catastrophic

16→4-bit quantization of shared features costs ≤0.05 AP for six of seven methods — an 8×
communication saving for noise-level cost. At 2 bits the cliff arrives, and the floor
test reclassifies the impairment from delivery to content. One reproducible anomaly:
CoBEVT scores *worse* at 2 bits (0.322) than at 1 bit (0.452), σ ≤ 0.001, plausibly
because 1-bit features approximate an occupancy mask its attention can reinterpret while
2-bit features retain misleading amplitude structure.

### 4.6 Attribution depends on the task's temporal structure

At matched loss rates, bursty and i.i.d. loss are indistinguishable for detection
(≤0.005 AP). In tracking, bursts cost 15–23% more identity switches, with a clean
dose-response in burst length (e.g. F-Cooper at 30% loss: 196 → 265 → 282 IDSW for
i.i.d., ~3-frame, and ~10-frame bursts). Short bursts that fit inside the tracker's
coast window barely move MOTA while already inflating identity switches ~35% —
aggregate metrics hide churn that downstream consumers would feel.

More strikingly, staleness and constant latency — near-twins in detection AP — produce
**opposite** tracking failures. Staleness (sawtooth freshness) explodes identity
switches 4–6× above clean; constant latency leaves them near-clean but dominates in
misses. A motion model absorbs a *consistent* bias (steadily displaced detections remain
associable) and amplifies an *oscillating* one (detections alternating between true and
stale positions shred association). For tracking-based stacks, a predictably delayed
channel is preferable to a variably fresh one at equal average age — the opposite of the
detection-level recommendation.

## 5. Deployment implications

1. **Prioritize freshness over completeness.** Retransmitting stale data is
   counterproductive; aggressive expiry (≤100 ms) is the highest-value channel policy.
2. **Compress features 8×.** No method in the cohort needs more than 4 bits per feature
   value; below that, withhold rather than transmit.
3. **Match fusion to link.** Lossy-but-fresh links favor attention-style fusion;
   poorly synchronized or localized fleets favor alignment-robust training; spoofing
   risk favors averaging; maxout and raw-point fusion are poor choices on any degraded
   link.
4. **Choose channel discipline by task.** Detection stacks should minimize age;
   tracking stacks should additionally minimize *age variance* and burst length.
5. **The open problem is temporal alignment.** No method tolerates a single frame of
   delay. CoAlign's transfer from spatial to temporal robustness suggests
   alignment-style training against temporal misalignment as the promising direction.

## 6. Limitations

Single dataset (OPV2V, simulated) and a single detector backbone family; the ego-only
floor is derived from the late-fusion checkpoint rather than per-backbone floors
(absorbed by the ±0.02 classification margin); spatial and tracking tiers use
representative method subsets; the AttFuse bandwidth impairment intercepts backbone
input because its fusion is interleaved; the tracker is a deliberately simple
constant-velocity Kalman filter — an instrument, not a contribution, so absolute MOTA
should not be compared against the tracking literature. Closed-loop driving evaluation,
which would connect these failures to behavioral consequences, remains future work.

## 7. Reproducibility

All code, configurations, and results are in this repository: `commchannel/` (channel
instrument), `configs/matrix.yaml` (frozen experiment grid), `scripts/` (baseline,
sweep, spatial, tracking runners and aggregator), `results/` (frozen baseline,
per-cell records, master summary, analysis). Every cell is keyed by
(method, impairment, level, seed) and reproduces exactly; two independent full
executions of the spatial tier reproduced digit-for-digit. `IMPLEMENTATION.md` records
the complete experimental history, including the four instrumentation bugs found and
fixed during the study and the gates that caught them.
