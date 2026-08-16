# Freshness-Aware Dynamic Digital Twins over Real Wireless Networks
## From Age of Information to Age of Twin Validity — Introduction, Motivation, References

**Scope of this document.** Paper-grade introduction and motivation for the selected
research plan (Option A + D core: dynamics-aware twin synchronization with heterogeneous
per-source age; Option B follow-on: coupled scene-twin × network-twin; Option C
opportunistic: channel-as-sensor updates). Reference list at the end, grouped by thread.

**Verification convention** (inherited from `temporal_messaging/RELATED_WORK.md`):
**[V]** verified against primary text or official repo; **[S]** from search-engine
summaries of abstracts — reliable for what a paper *claims*, not its numbers;
**[I]** our inference, not stated by any author. Claims from the parent study
(`collab_perception_failure_analysis/`) are **[V]** — they are our own measurements.
⚠️ Nothing tagged [S] goes into a submitted manuscript before the primary text is read.

---

## 1. Introduction

A digital twin is a virtual replica of a physical system, coupled to it by a live data
flow from the physical to the virtual and an information flow back [1], [2]. What
separates a twin from a simulation is precisely this coupling: definitional surveys
converge on *real-time update* as the distinguishing component [3]. For robotic and
smart-space deployments the twin is the substrate for monitoring, prediction, and
control — collision checking, task verification, human-safety reasoning, fleet
coordination [4], [5]. Every one of those functions consumes the twin's estimate of the
*current* world state, and every one of them fails quietly when that estimate is out of
date.

For the static structure of an environment, twinning is a mapping problem, and it is
largely solved: SLAM and offline reconstruction produce floorplans and meshes that
change on the timescale of days. The open problem is the **dynamic scene state** — the
people, robots, and movable objects whose positions change on the timescale of hundreds
of milliseconds. Keeping this state synchronized is not a modeling problem but a
*communication* problem: sensor nodes and mobile robots observe the scene and must push
updates to the edge-hosted twin across wireless links with finite rate, non-zero loss,
and — decisively — non-zero, time-varying latency. The networking literature has
recognized this and formalized twin maintenance as an update-scheduling and
resource-allocation problem, with **Age of Information (AoI)** — the time elapsed since
the newest delivered update was generated — as the standard freshness objective
[6]–[14].

AoI, however, is **content-agnostic**: it measures how old the newest update is,
independent of what has changed since. Goal-oriented refinements exist — Age of
Incorrect Information (AoII) alerts only when the monitor's state is actually wrong
[18], Value-of-Information and goal-oriented metrics weight age by task impact
[19]–[21] — but in the digital-twin literature these objectives are validated almost
exclusively in *simulation*, against abstract source models (Markov state chains,
synthetic sensor processes) rather than against a measured twin of a real dynamic scene
maintained over a real radio access network [6], [9]–[13] [S]. The empirical question
underneath the entire optimization stack — **what error does a stale twin actually
commit, and does AoI predict it?** — is, to the best of our knowledge, unmeasured.

There is direct experimental evidence that the answer is *no*, from our own prior
controlled study of communication impairments in cooperative perception (7 fusion
architectures, 831 evaluated conditions, OPV2V) [37] **[V]**. Four of its findings bear
on twin synchronization:

1. **Delivery failure is benign; staleness is not.** Losing 90% of collaborator
   messages never pushed any architecture below the no-collaboration floor
   (worst AP@0.7 0.579 vs floor 0.575), while a single 100 ms of latency pushed *all
   seven* below it. Dropping 90% of messages beats delivering all of them 200 ms late —
   *better silent than stale*.
2. **The damage is displacement, not information loss.** At 100 ms the loss at a
   loose localization threshold (AP@0.5) was 3.9–8.8× smaller than at a strict one
   (AP@0.7): objects were still detected, but placed where they used to be. Stale
   evidence is *mis-timed*, not destroyed — which means it is, in principle,
   correctable by a consumer that knows the scene's dynamics.
3. **The temporal error pattern matters more than its magnitude for a stateful
   consumer.** For a motion-model tracker, constant 200 ms latency left identity
   switches nearly untouched (223/230/248 vs clean 149/160/167 across three
   architectures) while sawtooth staleness at comparable mean age exploded them 4–6×
   (629/1031/917). A consistent delay is absorbed by prediction; an oscillating one is
   amplified.
4. **Heterogeneous age was never tested.** Like the rest of the controlled-impairment
   literature, that study applied every impairment *uniformly* across sources. A real
   network does the opposite: each source has its own age process, set by its radio
   technology, link quality, and payload.

A digital twin is exactly the consumer these findings describe: a *stateful* estimator
holding per-entity motion models, fed by *heterogeneously aged* updates. Findings 1–3
jointly predict that AoI is the wrong control variable for dynamic-scene twinning — two
update policies with identical mean age can produce radically different twin error
depending on *which entities* are stale, *how dynamic* they are, and *in what temporal
pattern* the staleness arrives [I]. Finding 4 marks the regime a real testbed uniquely
provides. Yet these predictions come from simulation (OPV2V/CARLA), from a vehicular
setting, and from a detection/tracking consumer rather than a maintained twin. Whether
they transfer to a real indoor robotic space, over real Wi-Fi and 5G, into an actual
edge-hosted twin, is precisely what has not been measured — by us or, as far as our
survey shows, by anyone [S]/[I].

Three concrete gaps follow:

- **G1 — The AoI → twin-error mapping is unmeasured on any real network.** The closest
  works either measure timeliness–fidelity for 3D scene *representations* without
  per-entity dynamic state [15] [S], optimize the accuracy–timeliness trade-off over
  modeled channels [16] [S], or perform goal-oriented temporal selection for a single
  controlled entity (a robot arm) rather than a multi-agent dynamic scene [17] [S].
- **G2 — Heterogeneous per-source age is unstudied as a fusion condition.** Real
  deployments measure order-of-magnitude payload- and RAT-dependent latency spreads —
  24 ms for boxes vs 2,190 ms for compressed images in a 14-node outdoor system [24]
  [S]; ~80 ms mean on 5G vs ~380 ms on Wi-Fi for mobile robots [29] [S] — but no
  controlled study relates twin or fusion quality to the *distribution* of age across
  sources rather than its mean [I].
- **G3 — The zero-learning kinematic baseline is missing.** The latency-compensation
  literature [32]–[36] reports learned receiver-side or sender-side correction, but
  none of it reports what plain constant-velocity extrapolation of tracked state —
  the correction a twin gets for free from its own motion models — would have
  achieved. (Established in `temporal_messaging/RELATED_WORK.md` §9 [V]; it is the
  baseline this plan carries onto real hardware.)

**What we propose.** Using a real indoor testbed — two autonomous mobile robots
(3D LiDAR, RGB-D, radar) and static infrastructure nodes (RGB, radar), publishing over
ROS 2 across *both* Wi-Fi (2.4/5 GHz) and a private 5G network (gNB → 5GCN) into an
edge-hosted processing stack — we will measure, rather than model, the freshness
behavior of a dynamic digital twin, and then use the measurements to build a
dynamics-aware synchronization policy. Contributions, in plan order:

- **C1 (Phase 0 — characterization).** A measurement campaign of per-RAT, per-payload,
  per-mobility age and loss processes (boxes vs point clouds vs images; Wi-Fi vs 5G;
  static node vs moving AMR), released as traces. This is the empirical substrate the
  DT-scheduling literature currently assumes.
- **C2 (Phase 1 — metric).** A per-entity twin-error metric that decomposes static
  vs dynamic entities and localization-strictness levels — the digital-twin analogue
  of the displacement diagnostic of [37]. Pre-registered gate: if dynamic-entity
  displacement does not dominate twin error under real latency, the dynamics-aware
  thesis dies here, cheaply.
- **C3 (Phase 2 — the mapping).** The measured AoI → twin-error mapping across RAT ×
  payload × entity class, including *heterogeneous-age* conditions (G2) and the
  zero-learning kinematic-extrapolation arms (G3: no correction / oracle velocity /
  oracle displacement / tracker velocity, the four-arm design pre-registered in
  `temporal_messaging/HANDOFF.md` §5, now on real data).
- **C4 (Phase 3 — the policy).** Divergence-triggered synchronization — update an
  entity when its *predicted* twin divergence exceeds a bound, an AoII-style rule with
  a kinematic predictor — evaluated in closed loop against periodic and AoI-optimal
  scheduling, reported as twin-error-per-bit on the real network.
- **C5 (follow-on, Option B).** Coupling the scene twin with a network twin: the twin
  predicts both scene dynamics and per-link quality, and chooses per-source what to
  send and over *which RAT*. Dual-RAT selection made by the twin itself is, per our
  survey, unoccupied [S]/[I]; the network-DT literature twins the network alone
  [38]–[43].
- **C6 (opportunistic, Option C).** Channel-based sensing as a zero-age change
  detector: when perception updates are stale or lost, the radio itself signals
  "something moved," triggering targeted re-synchronization. ISAC × DT is currently
  conceptual or simulated [44]–[47]; a real, even coarse, implementation would be a
  first [S]/[I]. Gated on CSI extractability from our radios (verify in Phase 0).

---

## 2. Motivation

**M1 — Twin staleness is a safety problem, and it fails silently.** A stale twin does
not look broken; it shows a *plausible* world in which every moving entity is displaced
along its recent trajectory. At indoor speeds (~1.4 m/s human walking, 1–2 m/s AMR),
the measured Wi-Fi latencies of [29] (~380 ms mean) translate to 0.5–0.8 m of
displacement per dynamic entity — enough to put a person on the wrong side of an aisle
or an AMR across a doorway threshold — while measured 5G (~80 ms) keeps it near 0.1 m.
The parent study's spatial decomposition sharpens the danger: stale evidence does not
merely degrade the twin *where the twin is blind* — it contaminated perception inside
the ego sensor's own field of view (recall −0.21…−0.46, false positives ×3.4–4.6) [37]
[V]. The corresponding twin failure — a stale remote update overriding a fresh local
one — is exactly the kind of fault a safety argument must exclude, and today there is
no measurement to exclude it with.

**M2 — The optimization literature has outrun the measurement literature.** There are
by now many schedulers that minimize AoI or its variants for twin maintenance
[6]–[13], all resting on an assumed monotone age → error link. Our simulation evidence
says the link is not monotone in the relevant sense: error depends on the age
*pattern* (finding 3), on *which* source is aged (finding 4 / G2), and on entity
dynamics (finding 2) — and there is even a measured regime where *more* impairment
hurts *less* (the misalignment valley of [37]: moderate displacement competes with
truth, extreme displacement self-discredits). If any of this transfers to real
networks, AoI-optimal schedulers are optimizing the wrong objective, and the fix —
divergence-based scheduling with kinematic prediction — is implementable with zero
learning. Either outcome of the measurement is publishable: a confirmed transfer
rewrites the objective; a refutation validates AoI empirically for the first time on
real hardware [I].

**M3 — Real networks produce exactly the regime simulation cannot.** The parent
study's 10 Hz simulation quantum made sub-100 ms latency unrepresentable and forced
uniform impairment (its §7 explicitly scopes both out) [V]. The testbed inverts this:
sub-100 ms is the *native* 5G regime; heterogeneous age arises for free from the
Wi-Fi/5G split, payload sizes, and AMR mobility; and burst structure comes from real
contention and real interference (Wi-Fi degrading under electromagnetic interference
where 5G held [29] [S]) rather than from a two-state Markov model. Nothing about the
campaign requires network emulation — the network *is* the experimental condition.

**M4 — The testbed spans both sides of a loop no one has closed.** Because the
platform carries perception sensors *and* channel-based sensing on the same
infrastructure, it can host the scene twin and the network twin in one system —
enabling C5 (the twin schedules its own updates across RATs) and C6 (the channel
itself as a twin sensor). Existing network digital twins model the network in
isolation [38]–[43]; existing perception/scene twins treat the network as a given.
Coupling them is repeatedly named as a 6G target [44], [45], [47] [S] but, per our
survey, has no real implementation. A modest working loop on real hardware would
outweigh a large simulated one.

**M5 — Methodological readiness.** The parent study contributes a validated working
method, not just findings: pre-registered go/no-go rules before expensive runs,
seed-determinism and per-unit resumability, attribution via impairment families, and a
reproduction record (published baselines to ±0.001; spatial tier reproduced
digit-for-digit) [37] [V]. The plan transplants this discipline to a domain — DT
synchronization — where evaluation practice is currently ad hoc and
simulation-bound [I].

**Research questions.**

- **RQ1** What are the real age/loss processes of a dynamic-twin update pipeline, per
  RAT, payload, and mobility state? (Phase 0 → C1)
- **RQ2** Is dynamic-entity displacement the dominant component of twin error under
  real network latency, as simulation predicts? (Phase 1 → C2, pre-registered gate)
- **RQ3** How well does AoI predict measured twin error, compared to a
  kinematics-aware divergence predictor — and how does the answer change under
  heterogeneous per-source age? (Phase 2 → C3)
- **RQ4** How much twin accuracy per transmitted bit does divergence-triggered
  synchronization buy over periodic and AoI-optimal policies, in closed loop on the
  real network? (Phase 3 → C4)
- **RQ5** *(follow-on)* Does coupling the scene twin to a network twin (dual-RAT
  selection, C5) or to channel-based change detection (C6) further improve the
  error-per-bit frontier?

---

## 3. Positioning against the closest prior art

| Work | What it does | What we add |
|---|---|---|
| Timeliness–Fidelity in 3D scenes [15] [S] | Delay vs reconstruction-fidelity trade-off, cameras + edge | Per-entity *dynamic state* error, not scene fidelity; real dual-RAT network; scheduling policy |
| Accuracy–timeliness DT framework [16] [S] | Optimizes the trade-off over modeled channels | Measures the trade-off on a real network; shows where the modeled objective diverges |
| Robot-arm DT semantic comms [17] [S] | Goal-oriented feature/temporal selection, single controlled entity | Multi-entity dynamic scene; uncontrolled dynamics (humans); heterogeneous sources |
| Fresh2comm / AoI+sync / Update-the-Unseen [6]–[8] [S] | AoI-based allocation for cooperative perception, simulated | The measured age→error mapping their optimizations assume; real heterogeneous age |
| Indoor delay-aware CP nodes [26] [S] | Real-time indoor CP with delay handling | The twin as the consumer; freshness *policy*, not point solution; dual-RAT measurement |
| NDT for 5G robots [38] [S] | Twins the network for robot ops | Couples network twin with the *scene* twin (C5); scene-side error metric |
| SyncNet/CoBEVFlow/FFNet/TraF-Align/CoDynTrust [32]–[36] [S] | Learned latency compensation | The zero-learning kinematic baseline none of them report (G3), on real data |

**Standing threats** (carry into every write-up):
(i) all [S] entries must be verified against primary texts — the authoring
environment's proxy blocks arXiv/IEEE/CVF PDFs, HTML mirrors sometimes pass;
(ii) the AoII literature [18] could subsume C4's policy if any DT work already
implements kinematic-predictor AoII on hardware — explicitly search for this before
claiming C4;
(iii) if CSI extraction fails on our radios, C6 is dropped without affecting C1–C4;
(iv) OPV2V-derived expectations may not transfer indoors — that is RQ2, a gate, not
an assumption.

---

## References

**Digital-twin foundations**
[1] M. Grieves, J. Vickers — *Origins of the Digital Twin Concept.* [researchgate.net](https://www.researchgate.net/publication/307509727_Origins_of_the_Digital_Twin_Concept)
[2] *Revisiting Digital Twins: Origins, Fundamentals and Practices.* [arXiv:2203.12867](https://arxiv.org/pdf/2203.12867)
[3] *What is a Digital Twin Anyway? Deriving the Definition from over 15,000 Publications.* [arXiv:2409.19005](https://arxiv.org/html/2409.19005v2)
[4] *The Concept of a Digital Twin of an Autonomous Mobile Robot.* [Springer](https://link.springer.com/chapter/10.1007/978-3-032-03722-0_15)
[5] *Digital Twin Enabled Runtime Verification for Autonomous Mobile Robots under Uncertainty.* [arXiv:2412.09913](https://arxiv.org/pdf/2412.09913)

**Digital-twin synchronization & Age of Information**
[6] *Fresh2comm: AoI-based communication resource allocation for cooperative perception.* [arXiv:2502.07852](https://arxiv.org/abs/2502.07852)
[7] *Spatiotemporal Feature Alignment and Weighted Fusion … Network Synchronization and Age of Information.* [arXiv:2602.13439](https://arxiv.org/abs/2602.13439)
[8] *Update the Unseen Only: Minimizing AoI for Collaborative Perception through Online Learning.* [arXiv:2607.20967](https://arxiv.org/abs/2607.20967)
[9] *Digital Twin Synchronization Over Mobile Embodied AI Network With Agentic Intelligence.* [arXiv:2605.14625](https://arxiv.org/html/2605.14625)
[10] *Toward Efficient Deployment and Synchronization in Digital-Twins-Empowered Networks.* [arXiv:2604.00566](https://arxiv.org/html/2604.00566)
[11] *Optimizing Wireless Resource Management and Synchronization in Digital Twin Networks.* [arXiv:2502.05116](https://arxiv.org/pdf/2502.05116)
[12] *A UAV-Aided Digital Twin Framework for IoT Networks with High Accuracy and Synchronization.* [arXiv:2504.15967](https://arxiv.org/pdf/2504.15967)
[13] *Age-of-Information and Energy Optimization in Digital Twin Edge Networks.* [researchgate.net](https://www.researchgate.net/publication/384115673_Age-of-Information_and_Energy_Optimization_in_Digital_Twin_Edge_Networks)
[14] *AI-driven digital twin networks for future wireless systems: a survey.* [Springer](https://link.springer.com/article/10.1007/s44443-026-00522-y)

**Timeliness, fidelity, and goal-oriented metrics**
[15] *Timeliness–Fidelity Tradeoff in 3D Scene Representations.* [arXiv:2407.16575](https://arxiv.org/abs/2407.16575)
[16] *Intelligent Digital Twin Communication Framework for the Accuracy–Timeliness Tradeoff in Resource-Constrained Networks.* IEEE TCCN. [pdf](https://cerc-ngct.ca/wp-content/uploads/2025/03/J-2024-IEEE-TCCN-Intelligent-Digital-Twin-Communication-Framework-for-Addressing-Accuracy-and-Timeliness-Tradeoff.pdf)
[17] *Goal-oriented Semantic Communication for Robot Arm Reconstruction in Digital Twin: Feature and Temporal Selections.* [arXiv:2411.08835](https://arxiv.org/pdf/2411.08835)
[18] *Semantics-Empowered Communications Through the Age of Incorrect Information.* [researchgate.net](https://www.researchgate.net/publication/362642087_Semantics-Empowered_Communications_Through_the_Age_of_Incorrect_Information)
[19] *Goal-oriented Tensor: Beyond AoI Towards Semantics-Empowered Goal-Oriented Communications.* [arXiv:2307.00535](https://arxiv.org/pdf/2307.00535)
[20] *Towards Goal-Oriented Semantic Communications: New Metrics, Framework, and Open Challenges.* [arXiv:2304.00848](https://arxiv.org/abs/2304.00848)
[21] *AoI Minimization in Goal-Oriented Communication with Processing and Cost-of-Actuation-Error Constraints.* [arXiv:2508.07865](https://arxiv.org/pdf/2508.07865)
[22] *Resource requirements of an edge-based digital twin service: an experimental study.* [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S2096579622000468)
[23] *Digital Twins at the Edge: A High-Availability Framework for Resilient Data Processing in IoT Sensor Networks.* [MDPI](https://doi.org/10.3390/fi18030137)

**Real-world cooperative perception & testbeds**
[24] *CoInfra: A Large-Scale Cooperative Infrastructure Perception System and Dataset.* [arXiv:2507.02245](https://arxiv.org/pdf/2507.02245)
[25] *Cooperative Infrastructure Perception.* [arXiv:2207.08930](https://arxiv.org/pdf/2207.08930)
[26] *Enhancing Indoor Mobility with Connected Sensor Nodes: A Real-Time, Delay-Aware Cooperative Perception Approach.* [arXiv:2411.02624](https://arxiv.org/pdf/2411.02624)
[27] *Real-World Deployment of Cloud-based Autonomous Mobility Systems for Outdoor and Indoor Environments.* [arXiv:2505.21676](https://arxiv.org/pdf/2505.21676)
[28] *V2X-ReaLO: Real vehicles + infrastructure, online cooperative perception.* [arXiv:2503.10034](https://arxiv.org/abs/2503.10034)
[29] *Latency-Sensitive Wireless Communication in Dynamically Moving Robots for Urban Mobility Applications.* [MDPI Smart Cities](https://www.mdpi.com/2624-6511/8/4/105)
[30] *SERN: Bandwidth-Adaptive Cross-Reality Synchronization for Simulation-Enhanced Robot Navigation.* [arXiv:2410.16686](https://arxiv.org/pdf/2410.16686)
[31] *Simulation to Reality: Testbeds and Architectures for Connected and Automated Vehicles.* [arXiv:2505.03472](https://arxiv.org/pdf/2505.03472)

**Latency compensation (baselines and the missing-baseline argument)**
[32] SyncNet — *Latency-aware collaborative perception.* [arXiv:2207.08560](https://arxiv.org/abs/2207.08560)
[33] CoBEVFlow — *Asynchrony-robust collaborative perception via BEV flow.* [repo](https://github.com/MediaBrain-SJTU/CoBEVFlow)
[34] FFNet — *Feature-flow transmission for cooperative detection.* [repo](https://github.com/haibao-yu/FFNet-VIC3D)
[35] TraF-Align — *Trajectory-aware feature alignment under latency.* [repo](https://github.com/zhyingS/TraF-Align)
[36] CoDynTrust — *Dynamic feature trust for asynchronous cooperative perception.* [repo](https://github.com/CrazyShout/CoDynTrust)
[37] *Better Silent Than Stale* — parent study, this repository: `collab_perception_failure_analysis/paper/PAPER.md`, full analysis in `results/ANALYSIS.md`.

**Network digital twins (Option B)**
[38] *Network Digital Twin for 5G-Enabled Mobile Robots.* [arXiv:2502.02253](https://arxiv.org/html/2502.02253)
[39] *DT-RaDaR: Digital-Twin-Assisted Robot Navigation using Differential Ray-Tracing.* [arXiv:2411.12284](https://arxiv.org/html/2411.12284v1)
[40] *AdaPTwin: Adaptive Multi-Fidelity Predictive Digital Twin for Proactive Radio Resource Management.* [arXiv:2605.21897](https://arxiv.org/pdf/2605.21897)
[41] *Digital Twin Online Channel Modeling: Challenges, Principles, and Applications.* [arXiv:2501.08680](https://arxiv.org/pdf/2501.08680)
[42] *Open Wireless Digital Twin: End-to-End 5G Mobility Emulation with OpenAirInterface and Ray Tracing.* [arXiv:2503.12177](https://arxiv.org/pdf/2503.12177)
[43] *Colosseum: The Open RAN Digital Twin.* [arXiv:2404.17317](https://arxiv.org/pdf/2404.17317)

**ISAC × digital twin (Option C)**
[44] *Integrated Sensing and Communication Driven Digital Twin for Intelligent Machine Network.* [arXiv:2402.05390](https://arxiv.org/pdf/2402.05390)
[45] *Toward Deeper Environmental Understanding: Event-Level Sensing for Intelligent 6G ISAC.* [arXiv:2606.14223](https://arxiv.org/pdf/2606.14223)
[46] *SynthSoM-Twin: A Multi-Modal Sensing-Communication Digital-Twin Dataset for Sim2Real Transfer.* [arXiv:2511.11503](https://arxiv.org/pdf/2511.11503)
[47] *Digital Twin and ISAC Applications in IoT: Taxonomy, Implications, and Open Issues.* [MDPI](https://doi.org/10.3390/app16010073)
[48] *When SLAM Meets Wireless Communications: A Survey.* [arXiv:2602.06995](https://arxiv.org/pdf/2602.06995)
