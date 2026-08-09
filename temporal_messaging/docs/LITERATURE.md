# Literature Check: Asynchronous Continuous-Time Collaborative Belief

**Prepared:** 9 August 2026
**Purpose:** verify the novelty claim in the method note before drafting, and assemble an annotated reference base covering the four literatures the proposal claims to subsume (latency, loss, bandwidth, asynchrony) plus continuous-time deep learning.

---

## 0. Headline verdict — read this first

The proposal's stated novelty claim was:

> *Continuous-time deep learning (Neural ODEs, Neural CDEs, latent ODEs) … has not been applied to the asynchronous multi-agent perception problem, even though asynchronous message arrival is an irregularly-sampled time series.*

**This claim is half true, and the half that is false is the load-bearing half.**

The specific architecture — a **persistent BEV latent state, evolved between observations by a learned neural-ODE dynamics model, updated asynchronously at each measurement's own timestamp, and read out by querying the state at an arbitrary requested time** — already exists in the literature. It is **StreamingFlow** (Shi et al., CVPR 2024). StreamingFlow applies it to *asynchronous multi-sensor fusion within one vehicle* (LiDAR at one rate, cameras at another), not across a communication network between agents.

What that means concretely:

| Proposal component | Status |
|---|---|
| 1. Persistent latent BEV belief | **Taken** — StreamingFlow; also memory-bank variants in CP (CoST) |
| 2. Learned continuous dynamics conditioned on Δt | **Taken** — StreamingFlow (SpatialGRU-ODE, built on GRU-ODE-Bayes) |
| 3. Asynchronous update at message's own timestamp | **Taken** — StreamingFlow's trigger-mode predict-update; also Neural CDE robotics work |
| 4. Time-query readout at arbitrary timestamp | **Taken** — StreamingFlow explicitly demonstrates arbitrary-interval and zero-shot horizon extension |
| 5. Delta messaging (send what changed) | **Substantially taken** — CoST (ICCV 2025) and CooperTrim (2026) both transmit only dynamic/changed content against a retained history |
| Channel-in-the-loop end-to-end training | **Common practice** — described as "distortion-in-the-loop" training, used by several works; SyncNet uses curriculum over latency |
| **The composition of all of these across a multi-agent communication network, with per-agent age, arrival order, and loss handled natively** | **Open — no prior work found** |

So the paper is still viable, but the framing in the note ("one architecture subsumes four separate literatures — that's the claim") will not survive review as written, because a reviewer who knows StreamingFlow will read the dynamics model as a port rather than an invention. The defensible claim is narrower and, I'd argue, more interesting: *the multi-agent case is not a trivial port of the multi-sensor case, because the network introduces per-agent age, adversarial arrival order, permanent loss, and a bandwidth budget — none of which exist inside a single vehicle's sensor rig.* Section 13 works through how to frame this.

Also note: a June 2026 survey (Hu et al., arXiv:2606.13840) has already coined **"Shared World Models"** as the organizing frame for exactly this direction and names *"verifiable shared-state maintenance"* as an open research priority. That is good news for positioning — you can cite it as independent confirmation the gap is recognised — but it also means the framing is no longer unclaimed territory.

---

## 1. How this search was conducted

Sixteen web searches plus one full-text fetch, covering: asynchrony/latency compensation in CP; loss and interruption robustness; bandwidth and communication efficiency; temporal and persistent state in CP; the neural-differential-equation family; continuous-time methods applied to driving perception and robotics; task-agnostic and query-based CP; datasets and testbeds; and 2025–2026 surveys.

**Coverage caveats.** Search returns snippets and abstracts; for most entries below I have the abstract, contributions list, and headline results, but not full experimental tables. Where I state a number, it came from the source. Where I could not verify a number, I say so rather than estimating. I fetched StreamingFlow in full because it is the decisive prior-art item. Sections 14 lists what remains unchecked.

---

## 2. Strand A — Foundations: the fusion paradigms your study benchmarks

These are the "seven baselines" family. Included for completeness and because the related-work section needs them, not because they threaten novelty.

### A1. V2VNet — Wang, Manivasagam, Liang, Yang, Zeng, Urtasun. ECCV 2020.
*V2VNet: Vehicle-to-Vehicle Communication for Joint Perception and Prediction.*

- **Motivation.** Single-vehicle perception is occlusion-limited and degrades at range where returns are sparse or absent. Aggregating views from nearby vehicles lets a vehicle see through occlusions and detect distant actors.
- **Method.** Intermediate fusion: each vehicle transmits compressed deep feature-map activations; a spatially-aware graph neural network aggregates received messages, with the message warped by the relative pose between sender and receiver. Joint perception *and* motion forecasting head.
- **Results.** Reported high accuracy while meeting communication-bandwidth constraints; established compressed intermediate features as the dominant transmission unit for the field.
- **Relevance.** This is where the "warp by relative pose, then fuse, then discard" pattern originates. Note for your framing: V2VNet's latency handling assumes a *global* misalignment caused by ego-motion between query and receipt — which implicitly assumes agents sample synchronously. Dao et al. (2307.01462) make this critique explicitly.

### A2. DiscoNet — Li, Ren, Wu, Chen, Feng, Zhang. NeurIPS 2021.
*Learning Distilled Collaboration Graph for Multi-Agent Perception.*

- **Motivation.** Trade off the accuracy of early (raw-data) fusion against the bandwidth of intermediate fusion.
- **Method.** Knowledge distillation from an early-fusion teacher into an intermediate-fusion student, with a learned, matrix-valued collaboration graph controlling per-region inter-agent attention.
- **Relevance.** DiscoNet is the standard synchronous-assumption baseline. Dao et al. note it assumes a shared clock, identical sampling rates, and zero transmission latency — a clean citation for your "statelessness plus synchrony assumption" argument.

### A3. When2com / Who2com — Liu, Tian, Ma, Glaser, Kuo, Kira. CVPR 2020 / ICRA 2020.
- **Motivation.** Not every agent should talk to every other agent every frame; bandwidth should be spent on useful links.
- **Method.** Learnable handshake communication (Who2com) and communication-graph grouping (When2com) to decide *whom* to talk to and *when*.
- **Relevance.** These are the "when to communicate" ancestors of Where2comm's "where to communicate". Your delta-messaging contribution is a third axis — *what has changed* — which is worth explicitly distinguishing from both.

### A4. V2X-ViT — Xu, Xiang, Tu, Xia, Yang, Ma. ECCV 2022.
*V2X-ViT: Vehicle-to-Everything Cooperative Perception with Vision Transformer.*

- **Motivation.** Vehicles and infrastructure are heterogeneous agents with different sensor placement and capability; V2X also suffers pose error and asynchronous sharing.
- **Method.** A unified transformer alternating heterogeneous multi-agent self-attention with multi-scale window self-attention. Handles asynchrony *implicitly*, by training under simulated delay rather than by modelling time.
- **Results.** Introduced the V2XSet dataset (CARLA + OpenCDA).
- **Relevance.** V2X-ViT is the canonical example of the "compensate implicitly by training on noisy data" school. It's the right foil for your argument that time should be a modelled variable rather than a nuisance to be trained through.

### A5. CoBEVT — Xu, Tu, Xiang, Shao, Zhou, Ma. CoRL 2022.
*CoBEVT: Cooperative Bird's Eye View Semantic Segmentation with Sparse Transformers.* Sparse-transformer fusion for cooperative BEV segmentation. Frequently used as a baseline alongside V2X-ViT and Where2comm.

### A6. F-Cooper — Chen, Tang, Yang, Fu. SEC 2019.
Early feature-level cooperative perception via maxout fusion of voxel features; the historical origin of intermediate fusion in this field.

### A7. Practical Collaborative Perception — Dao, Berrio, Vu, et al. arXiv:2307.01462.
*A Framework for Asynchronous and Multi-Agent 3D Object Detection.*

- **Motivation.** Explicitly attacks the synchrony assumption in mid-collaboration methods, and the architectural invasiveness of fusion modules.
- **Results.** On V2X-Sim, reports 76.72 mAP — stated as roughly 99% of early-collaboration performance at late-collaboration bandwidth (~0.01 MB average).
- **Relevance.** **Cite this early in your intro.** It is the cleanest existing statement of the critique your paper opens with, and it pre-empts a reviewer asking "hasn't someone said this already?" Better to cite it and then say what it *didn't* do (it didn't build a persistent state) than to be caught not knowing it.

---

## 3. Strand B — Latency and asynchrony compensation

This is the strand your "dynamics are the model" claim competes with most directly. It is more crowded than the note assumes: there are now at least six distinct approaches, two of them from 2025.

### B1. SyncNet — Lei, Ren, Hu, Zhang, Chen. ECCV 2022. arXiv:2207.08560.
*Latency-Aware Collaborative Perception.*

- **Motivation.** Existing CP methods assume ideal communication; real latency degrades performance, and in safety-critical driving that is unacceptable. The paper positions itself as the first to formulate the latency problem in CP from a machine-learning standpoint.
- **Method.** Feature-level synchronisation: a latency compensation module that actively adapts asynchronous perceptual features from multiple agents onto a common timestamp. Two mechanisms — *feature-attention symbiotic estimation* (jointly estimating the intermediate feature map and the collaboration attention so each improves the other) and *time modulation*. A dual-branch pyramid LSTM extrapolates the current feature from a sequence of past observations. System has four stages: encode → compensate → fuse → decode.
- **Training.** Curriculum learning over latency — latency is increased by one step every ten epochs up to ten steps — because training loss rises sharply with latency and destabilises optimisation.
- **Results.** Reported ~15.6% improvement over the prior state of the art in the latency scenario, and maintains collaborative-over-single-agent superiority under severe latency. Evaluated on V2X-Sim 2.0 via the coperception codebase.
- **Relevance.** Two things matter here. (a) SyncNet is *extrapolation from a history buffer*, discarded each frame — your "nothing persists" critique lands. (b) Their curriculum-over-latency training is prior art for your channel-in-the-loop training, so present yours as *sampling arbitrary latency/loss/age jointly* rather than as *training under a channel*, which is not new.

### B2. CoBEVFlow — Wei, Wei, Hu, Lu, Zhong, Chen, Zhang. NeurIPS 2023. arXiv:2309.16940.
*Asynchrony-Robust Collaborative Perception via Bird's Eye View Flow.*

- **Motivation.** Temporal asynchrony from communication delay, interruption, and clock misalignment is unavoidable, and causes information mismatch that undermines fusion. Framed as misplacement of moving objects in the received message.
- **Method.** Estimate a *BEV flow* — a field of motion vectors over spatial locations — and use it to reassign asynchronous features to their correct positions at the receiver's time. An ROI-generation module restricts compensation to object-containing regions so background features are not corrupted with noise. Flow is estimated from historical ROI sets using multi-head attention with time encoding, which makes it robust to irregular timestamps. Message packing sends ROIs plus sparse features.
- **Two claimed advantages.** (i) It handles messages at irregular, continuous timestamps *without discretisation*; (ii) it warps original features rather than synthesising new ones, avoiding generation noise.
- **Results.** Introduced **IRV2V**, the first synthetic CP dataset with varied temporal asynchrony. Beats baselines on IRV2V and DAIR-V2X, and is stated to remain robust in extremely asynchronous settings. In a trade-off analysis at 300 ms expected latency on IRV2V, it beats Where2comm at substantially lower communication volume. Also reported robust to combined asynchrony and Gaussian pose noise.
- **Relevance — important.** **CoBEVFlow already claims continuous, non-discretised irregular timestamp handling.** Your Table's row "dynamics are the model; updates applied at their own timestamp" needs to be sharper than "CoBEVFlow discretises", because it does not. The real distinction is: CoBEVFlow *warps a received message to now, then discards it*; you *integrate the message into a state that persists*. Frame the difference as **state vs. transform**, not as **continuous vs. discrete**.

### B3. FFNet — Yu, Tang, Xie, Mao, Luo, Nie. NeurIPS 2023. arXiv:2311.01682 (earlier: 2303.10552).
*Flow-Based Feature Fusion for Vehicle-Infrastructure Cooperative 3D Object Detection.*

- **Motivation.** VIC3D suffers *both* uncertain temporal asynchrony and limited bandwidth; treating them separately leaves fusion misaligned and infrastructure data underused.
- **Method.** Transmit *feature flow* rather than a still-frame feature map, exploiting temporal coherence across sequential infrastructure frames so the receiver can generate a feature aligned to its own timestamp on the fly. A self-supervised training scheme gives the flow generator its prediction ability, using a pre-trained infrastructure model's features from randomly chosen future frames as targets. Attention masks and quantisation compress the flow.
- **Results.** On DAIR-V2X at 200 ms latency: 62.87% mAP@BEV, reported as state of the art, at roughly 1/100 the transmission cost of early fusion / raw point clouds. One model covers all latencies. Latency in DAIR-V2X is simulated by replacing the current infrastructure frame with one from *k* frames earlier (k×100 ms); note the dataset's native pairwise time difference is within ±30 ms.
- **Relevance — important for your bandwidth claim.** FFNet is the closest existing thing to "send a temporal object rather than a snapshot". It is not a delta against a shared belief, but it *is* a departure from snapshot transmission, and it already reports ~100× cost reduction. If your delta-messaging contribution is pitched on bandwidth alone, FFNet is the number you must beat or contextualise.

### B4. TraF-Align — Song, Yang, Wen, Li. CVPR 2025. arXiv:2503.19391.
*Trajectory-aware Feature Alignment for Asynchronous Multi-agent Perception.*

- **Motivation.** Latency causes *two* distinct failures, which prior work conflates: **spatial** misalignment (the object has moved) and **semantic** misalignment (features from different times are not recognised as the same object). Delayed messages can also lose the feature entirely.
- **Method.** Predict a feature-level *trajectory field* — the flow path of features up to the ego's current time — from past observations. Generate temporally ordered sampling points along each path, then direct attention from the current-time query to the historical features lying along that trajectory, reconstructing the current-time feature and enforcing cross-agent semantic consensus. Field loss + offset loss; end-to-end trainable. Built on OpenCOOD.
- **Results.** On V2V4Real and DAIR-V2X-Seq at **400 ms latency**, AP50 drops of only **4.87%** and **5.68%** respectively — reported as a new benchmark for asynchronous CP.
- **Relevance — this is your strongest baseline.** Those degradation numbers are the bar. A persistent-belief architecture that cannot beat ~5% AP50 drop at 400 ms will not read as a paradigm improvement, regardless of how much cleaner the formulation is. Get these checkpoints (they are released) and run them in your CommChannel.

### B5. CoDynTrust — Xu, Li, Wang, Yang, Wu, Chen, Wang. ICRA 2025. arXiv:2502.08169.
*Robust Asynchronous Collaborative Perception via Dynamic Feature Trust Modulus.*

- **Motivation.** Asynchrony produces *information mismatch*; prior compensation methods treat all compensated features as equally reliable, so errors propagate silently, and "patchy" collaboration creates safety risk.
- **Method.** Rather than compensating better, quantify how much to trust each compensated region. A **Dynamic Feature Trust Modulus (DFTM)** per ROI, derived from modelled aleatoric and epistemic uncertainty, selectively suppresses or retains single-vehicle features. Multi-scale fusion module consumes DFTM-weighted multi-scale maps. Notably, motion prediction uses **linear extrapolation** (per-ROI velocity × expected delay) rather than a learned model — a deliberate simplicity choice.
- **Results.** Reported to significantly reduce asynchrony-induced degradation across multiple datasets, with state-of-the-art detection under asynchrony. Uncertainty is propagated to downstream planning/control.
- **Relevance.** Two implications. (a) CoDynTrust's use of *linear extrapolation* beating learned compensators is a direct challenge to your premise that the network must learn to advect features — you should test a linear-advection ablation of your dynamics model and report it honestly. (b) Uncertainty-aware readout is a natural extension of a probabilistic belief; if your belief carries variance, say so, because that's a differentiator over CoDynTrust's per-ROI scalar trust.

### B6. Time Compensation Late Fusion (TCLF) — in DAIR-V2X, Yu et al. CVPR 2022.
The late-fusion baseline for delay: match boxes across successive frames, estimate velocity, linearly interpolate to the ego timestamp, then fuse. Included because it is the zero-velocity-predictor-plus-one-correction baseline your framing argues against, and it is cheap to run.

### B7. V2X-PC — Liu, Ding, Fu, Li, Chen, Zhang, Zhou. arXiv:2403.16635.
*Vehicle-to-Everything Collaborative Perception via Point Cluster.* Avoids dependence on historical feature buffers by using low-level point-cluster coordinates to predict positions at the current timestamp directly. A useful contrast: it makes latency compensation cheap by moving it to a lower level of representation.

### B8. Spatiotemporal Feature Alignment with Network Synchronization and Age of Information — arXiv:2602.13439 (2026).
- **Motivation.** Prior work (CoBEVFlow, SyncNet) treats transmitted timestamps as accurate and ignores network-level synchronisation error; frames the problem partly in terms of **Age of Information (AoI)**.
- **Results.** Reports higher mAP@0.5 and mAP@0.7 than both CoBEVFlow and SyncNet.
- **Relevance.** This is the closest existing work to your "per-agent age handled by construction" line. AoI is a well-developed communications-theory concept and connecting your per-agent age term to it explicitly would strengthen the systems framing considerably. Worth fetching in full.

---

## 4. Strand C — Loss, interruption, and lossy communication

Smaller literature than latency, and the note's characterisation ("feature recovery modules") is accurate. This is the strand where your "nothing to recover — the belief persists" argument is strongest.

### C1. V2VAM / LCRN — Li, Xu, Liu, Ma, Chi, Ma, Yu. IEEE T-IV 8(4), 2023. arXiv:2212.08273.
*Learning for Vehicle-to-Vehicle Cooperative Perception under Lossy Communication.*

- **Motivation.** Every prior CP algorithm assumed ideal V2V communication. The paper first *measures* the detection drop caused by lossy communication, then mitigates it.
- **Method.** Two components. **LCRN** (LC-aware Repair Network): an encoder–decoder with skip connections, inspired by image denoising, that generates tensor-wise filters and applies them to the damaged incoming features to repair them. **V2VAM** (V2V Attention Module): intra-vehicle attention over the ego's own features plus *uncertainty-aware* inter-vehicle attention, to weight collaborators according to transmission-induced uncertainty.
- **Results.** On OPV2V (CARLA digital twin), LiDAR 3D detection; reports significant AP improvement over prior CP algorithms under lossy communication. Code at github.com/jinlong17/V2VLC.
- **Critique available to you.** A 2023 survey (arXiv:2310.03525) observes that LCRN mitigates feature *degradation* but does not address feature *loss* — i.e., it can denoise a corrupted message but cannot conjure a message that never arrived. **That is precisely your argument, already made in the literature, which makes it safe to lean on.**

### C2. V2X-INCOP — Ren, Lei, Wang, Dianati, Wang, Chen, Zhang. IEEE T-IV 9(4), 2024. arXiv:2304.11821.
*Interruption-Aware Cooperative Perception for V2X Communication-Aided Autonomous Driving.*

- **Motivation.** Interruption — messages simply not arriving — is common and distinct from corruption, and had not been addressed. Claims to be the first work on communication interruption in CP.
- **Method.** Recover the missing information from *historical* cooperation information via a communication-adaptive multi-scale spatial-temporal prediction model, which conditions its feature extraction on the observed communication conditions. Trained with knowledge distillation (explicit supervision for the prediction model) and curriculum learning (training stability).
- **Results.** Cooperative gain over individual perception, averaged across packet-drop rates: **14.06%** on OPV2V, **13.9%** on V2X-Sim, **12.07%** on DAIR-V2X. Stated future work: delays and attacks.
- **Relevance.** V2X-INCOP is the single most direct competitor to your loss story, and it *does* use history. The distinction you need: V2X-INCOP predicts *the missing message* from history — a reconstruction target. Your architecture has no missing message to reconstruct, because information from prior messages was already integrated into the state. Frame it as **reconstruction vs. retention**. That is a real and defensible difference, but you must say it precisely.

### C3. QPoint2Comm — Xu et al. arXiv:2602.21667 (2026).
*Send Less, Perceive More: Masked Quantized Point Cloud Communication for Loss-Tolerant Collaborative Perception.*

- **Method.** Skip feature transmission entirely: quantise raw LiDAR into discrete codebook indices (Discrete Point Cloud Representation) against a shared codebook and send only indices, preserving voxel-aligned geometry. A **masked training strategy** simulates random packet loss, with a learnable feature-filling mechanism so the model learns to reason from partially missing pillars. Pyramid-scale cascade attention fusion.
- **Results.** Codebook ablation: best at codebook size 2048, vector dim 1024. Reports minimal degradation across 0–0.4 s delay and 0–0.4 m localisation error. Claims SOTA on accuracy, communication efficiency, and packet-loss resilience.
- **Relevance.** Its masked-loss training is close to your channel-in-the-loop proposal, and it's from 2026 — cite it as concurrent rather than ignore it.

### C4. Other robustness variants worth a line each.
- **CoAlign** (Lu et al., ICRA 2023) — robust CP under pose error via agent-object pose graph consistency.
- **FreeAlign** (ICRA 2024) — robust CP with no external localisation or clock devices. *Relevant:* clock-free operation is adjacent to your asynchrony story.
- **ROBOSAC / "Among Us"** (Li et al., ICCV 2023) — adversarially robust CP by consensus sampling.
- **CP-Guard** (AAAI 2025) — malicious agent detection in collaborative BEV perception.
- **RoCooper** (INFOCOM 2025) — robust cooperative perception under V2V impairments.
- **AFFormer** (arXiv:2605.01888, 2026) — adaptive feature fusion transformer for V2X under channel impairments.

---

## 5. Strand D — Bandwidth and communication efficiency

**This strand contains the most serious threat to your delta-messaging contribution.** The note characterises the field's bandwidth response as "spatial confidence selection (Where2comm)". That was true in 2022. It is not true in 2026: two recent papers explicitly transmit only what changed against retained history.

### D1. Where2comm — Hu, Fang, Lei, Zhong, Chen. NeurIPS 2022 (Spotlight). arXiv:2209.12836.
- **Motivation.** Perception performance vs. bandwidth is a fundamental trade-off; perceptual information is spatially heterogeneous, so uniform transmission wastes bandwidth on empty space.
- **Method.** A **spatial confidence map** identifies perceptually critical regions; agents share spatially sparse but critical features ("where to communicate"). Multi-round, multi-modality, multi-agent framework: observation encoder → spatial confidence generator → confidence-aware communication → confidence-aware message fusion (multi-head attention) → detection decoder. Bandwidth is adjustable at inference by varying the spatial area shared.
- **Results.** Evaluated on OPV2V, V2X-Sim, DAIR-V2X, and their own CoPerception-UAVs, across camera/LiDAR and cars/drones. Consistently better performance-bandwidth trade-off than prior methods.

### D2. CoST — Tang et al. ICCV 2025. arXiv:2508.00359.
*Efficient Collaborative Perception From a Unified Spatiotemporal Perspective.*

**This is the paper that overlaps your delta messaging most directly. Read it in full before writing.**

- **Motivation.** Re-transmitting the whole scene every frame is redundant, because most of the scene is static and was already transmitted.
- **Method.** Three parts. **STT (Spatio-temporal Transmission)** — transmit *only dynamic object features*; static objects are retrieved from prior transmissions via pose-projected reuse from a **memory bank**. **USTF (Unified Spatio-temporal Fusion)** — treat historical agents as time-delayed duplicates of current agents, so spatial and temporal fusion collapse into a single fusion operation instead of two separate stages. **MADA** as the concrete fusion module, with recurrent modelling to cut compute.
- **Results.** Evaluated on V2XSet, V2V4Real, DAIR-V2X; reported superior to existing methods, with the spatio-temporal modules claimed to drop into prior methods to improve accuracy and reduce bandwidth simultaneously.
- **Overlap with your proposal.** CoST has: a persistent memory bank at the receiver; transmission of change only; and a unification of the temporal and cross-agent axes. Three of your five components, in one ICCV 2025 paper. **What it does not have:** continuous-time dynamics (its reuse is pose-projection of static content, not learned advection of a latent state), an arbitrary-time query interface, or native handling of per-agent age. Your differentiation must be stated against CoST specifically, by name, in the intro.

### D3. CooperTrim — arXiv:2602.13287 (2026).
*Adaptive Data Selection for Uncertainty-Aware Cooperative Perception.*
- **Motivation.** Per-frame subset selection still stresses real wireless links; go proactive.
- **Method.** Exploit temporal continuity to identify features that capture environment *dynamics*, avoiding repetitive transmission of static content; temporal awareness lets agents adapt the sharing quantity to scene complexity.
- **Relevance.** Same core idea as your "bandwidth scales with scene dynamics rather than scene size", stated in almost the same words. Cite it as concurrent work.

### D4. What2comm — Yang et al. ACM MM 2023.
Feature *decoupling*: transmit exclusive vs. common feature maps separately across heterogeneous agents; plus a spatio-temporal collaboration module that integrates collaborator information with temporal ego cues, aimed at robustness to transmission delay and localisation error.

### D5. How2comm — Yang, Yang, Wang, et al. NeurIPS 2023.
Communication-efficient and collaboration-pragmatic multi-agent perception; the "how to fuse" complement to When2com/Where2comm/What2comm.

### D6. Select2Col — arXiv:2307.16517.
Leverages the **spatial-temporal importance** of semantic information for collaborator selection and efficient transmission.

### D7. Compression-side work (a cluster, cite as a group).
- **ReVQom** (Shenkut & Kumar, ICASSP 2026; arXiv:2509.21464) — multi-stage **residual vector quantization** codec preserving spatial identity; transmits per-pixel code indices only, cutting from 8192 bits/pixel (32-bit float features) to **6–30 bpp per agent** with minimal accuracy loss.
- **InfoCom** (arXiv:2512.10305) — information-bottleneck framing; pushes CP bandwidth from MB to **KB** scale. Explicitly names adding *temporal cues* to the message unit as future work — i.e., someone is planning your delta contribution.
- **V2X-DSC** (arXiv:2602.00687, 2026) — **distributed source coding**: since agents' latents are correlated across viewpoints, transmission cost should be governed by *conditional* information content given the receiver's own representation, not marginal complexity. **This is information-theoretically the same intuition as delta messaging, generalised to the spatial rather than temporal axis** — a strong citation for motivating why deltas are principled rather than merely engineering.
- **WhisperNet** (arXiv:2603.01708, 2026) — argues the field overlooked *channel-dimension* redundancy, as opposed to spatial.
- **QuantV2X** (arXiv:2509.03704) — fully quantized multi-agent system; reports **3.2× system-level latency reduction** and **+9.5 mAP30** over full-precision baselines under deployment-oriented metrics. Useful because it evaluates in *system* terms, which is the register your CommChannel work lives in.
- **CoSDH** (arXiv:2503.03430, 2025) — supply-demand awareness and intermediate-late hybridisation.

### D8. PragComm — Hu et al.
Pragmatic communication: task-critical selection giving spatially *and temporally* sparse feature vectors, task-adaptive dictionary approximation so messages become integer indices, plus collaborator pruning. Covers detection and tracking.

---

## 6. Strand E — Temporal context and persistent state in collaborative perception

The note treats temporal state in CP as essentially absent ("V2XPnP and some recurrent variants use ego temporal context at fixed rate"). This understates it. There are now several works maintaining state, and one 2025 paper (CoST, §D2) with an explicit memory bank.

### E1. SCOPE — Yang, Yang, Zhang, Li, Liu, Liu, Wang, Sun, Song. ICCV 2023. arXiv:2307.13929.
*Spatio-Temporal Domain Awareness for Multi-Agent Collaborative Perception.*

- **Motivation.** Single-frame point clouds are sparse and under-represent moving objects; the temporal context of the ego agent was being thrown away.
- **Method.** Three components: (i) **context-aware information aggregation** — selective information filtering plus spatio-temporal feature integration to pull semantic cues from the ego's preceding frames; (ii) **confidence-aware cross-agent collaboration** with multi-scale feature interaction to survive localisation error; (iii) **importance-aware adaptive fusion** across sources.
- **Claim.** States it is the first to consider the ego agent's temporal context in a CP system.
- **Results.** On DAIR-V2X, V2XSet, OPV2V (AP@0.5/0.7): improves SOTA by **3.63%** on V2XSet and **7.41%** on OPV2V. Better performance–communication trade-off than Where2comm across bandwidth settings. Robust to localisation and heading error.
- **Follow-up:** **SCOPE++** (*Robust Multi-Agent Collaborative Perception via Spatio-Temporal Awareness*), extending to communication efficiency and collaboration robustness jointly.
- **Relevance.** SCOPE's temporal state is **ego-only and fixed-rate** — it does not accumulate collaborator information across time. That's a genuine gap you fill. Say it that specifically.

### E2. V2XPnP — Zhou, Xiang, Zheng, Zhao, Lei, Zhang, Cai, Liu, Liu, Bajji, Xia, Huang, Zhou, Ma. ICCV 2025. arXiv:2412.01812.
*Vehicle-to-Everything Spatio-Temporal Fusion for Multi-Agent Perception and Prediction.*

- **Motivation.** Prior CP is single-frame: it fuses across space but ignores time and temporal tasks. Systematically separates three design axes — **when to transmit** (one-step vs. multi-step communication), **what to transmit** (early/late/intermediate), and **how to fuse**.
- **Method.** Intermediate fusion within one-step communication, chosen because it balances accuracy against transmission load and because intermediate spatio-temporal features are shareable across tasks. Unified transformer with four attention types: temporal, self-spatial, multi-agent spatial, and map attention. Each agent first extracts inter-frame and self-spatial features locally (which reduces communication load and supports single-vehicle operation), then multi-agent spatial attention fuses across agents. Multi-stage training: single-agent multi-task pretraining, then multi-agent spatio-temporal learning with dynamic loss weighting to stop perception dominating prediction.
- **Contributions.** Benchmark of **11 fusion models** across all fusion/communication combinations, and the **V2XPnP Sequential Dataset** — first large-scale real-world V2X sequential dataset covering all collaboration modes (V2V, I2I, vehicle-centric, infrastructure-centric) with perception data, object trajectories, and map data.
- **Reported finding worth noting.** Early fusion benefits from the ego fusing *lossless raw data*, avoiding error accumulation from lossy intermediate information transformed across the temporal dimension. That is a warning about your architecture: **repeatedly integrating lossy messages into a persistent latent risks compounding error over time.** You should measure drift explicitly (e.g., belief quality vs. time since last high-quality observation).
- **Follow-up:** **TurboTrain** (Zhou, Zhao, Cai, Huang, Zhou, Ma, 2025) — masked-reconstruction pretraining plus conflict-suppressing gradient balancing to stabilise end-to-end multi-agent perception-and-prediction training without hand-designed multi-stage schedules. **Directly useful to you**: end-to-end training of a joint belief across tasks is exactly the optimisation problem you'll hit.

### E3. CATNet — arXiv:2603.05255 (2026).
*Collaborative Alignment and Transformation Network.* Evaluates robustness by randomly dropping packets within a historical window (e.g. 600 ms). Reports OPV2V staying above 78% and V2XSet above 65.0% AP@0.5 at delays up to 600 ms. Token-retention ablation. Relevant as a recent robustness bar and an example of the "historical window" evaluation protocol your CommChannel generalises.

### E4. OnlineBEV — Koh et al. arXiv:2507.08644 (2025).
*Recurrent Temporal Fusion in BEV Representations for Multi-Camera 3D Perception.* Single-agent, but architecturally close to your persistent-state idea: recurrent BEV accumulation increasing effective feature count at low memory, with an **MBFNet** extracting motion features from consecutive BEV frames to dynamically align historical BEV to current, supervised by a **Temporal Consistency Learning Loss**. **63.9% NDS** on nuScenes test, SOTA camera-only. Cite as evidence that recurrent BEV state works and that *explicit alignment supervision is necessary* — the consistency loss is a design lesson for your dynamics model.

### E5. CoMamba — arXiv:2409.10699.
Real-time cooperative perception with state space models. Relevant because SSMs are the other continuous-time-adjacent formalism, and a reviewer may ask why ODE rather than SSM. Have an answer.

### E6. CEST — Chen, Shu, Lu, Zhang, Wang. IEEE T-ITS 27(5), 2026.
*Enhancing Multi-Agent Perception via Communication-Efficient Spatial–Temporal Fusion.* Recent T-ITS entry in the same space; verify what it does before submission.


---

## 7. Strand F — Continuous-time deep learning (the formalism you'd import)

Mature, well-cited, and — as the note assumed — mostly developed outside perception. This is the strand where citation is straightforward.

### F1. Neural ODEs — Chen, Rubanova, Bettencourt, Duvenaud. NeurIPS 2018. arXiv:1806.07366.
- **Idea.** Instead of a discrete stack of layers, parameterise the *derivative* of the hidden state with a neural network and compute the output with a black-box ODE solver.
- **Properties.** Constant memory cost, input-adaptive evaluation strategy, explicit precision/speed trade-off. Backpropagation through any solver via the adjoint sensitivity method, without accessing solver internals — which is what makes end-to-end training inside a larger model feasible.
- **Known limitation directly relevant to you.** A pure Neural ODE's solution is fully determined by the initial condition, so *later-arriving observations cannot influence the trajectory*. This is why Neural ODEs are almost always combined with an RNN-style update when data arrives (see F2, F3). **Your "asynchronous update operator" is not optional — it is the standard fix, and you should present it as such rather than as a novelty.**

### F2. Latent ODEs and ODE-RNN — Rubanova, Chen, Duvenaud. NeurIPS 2019. arXiv:1907.03907.
*Latent ODEs for Irregularly-Sampled Time Series.*
- **Motivation.** RNNs fit regularly-sampled data; the standard workaround of binning an irregular series into fixed intervals and imputing destroys information — specifically information carried by *when* measurements occurred.
- **Method.** **ODE-RNN** generalises RNN state transitions to continuous-time dynamics between observations, with a discrete RNN update at each observation. **Latent ODE** uses ODE-RNN as its recognition network. Both handle arbitrary gaps natively, and can model observation times explicitly with a Poisson process.
- **Results.** Outperform RNN counterparts on irregularly-sampled data.
- **Relevance.** This is the canonical citation for "asynchronous arrival is an irregularly-sampled time series". Your framing sentence should cite this paper.

### F3. GRU-ODE-Bayes — De Brouwer, Simm, Arany, Moreau. NeurIPS 2019.
*Continuous Modeling of Sporadically-Observed Time Series.*
- **Relevance — critical.** This is the exact formalism StreamingFlow lifted into BEV space, and therefore the one you would be re-lifting. Continuous GRU dynamics between observations plus a Bayesian update at each observation, with proven boundedness of the state. If you build on SpatialGRU-ODE you are three citations deep in someone else's stack; be upfront about that.

### F4. Neural CDEs — Kidger, Morrill, Foster, Lyons. NeurIPS 2020.
*Neural Controlled Differential Equations for Irregular Time Series.* Constructs a continuous control path from discrete observations and evolves the latent state *against that path*, so incoming data continuously steers the trajectory rather than only resetting it. Extended by **Neural RDEs** via rough path theory. **Architecturally, a Neural CDE is arguably a better fit for your problem than an ODE-plus-update**, because messages arriving from multiple agents are precisely a control signal; consider it and justify your choice.

### F5. ContiFormer — Chen et al. arXiv:2402.10635.
*Continuous-Time Transformer for Irregular Time Series Modeling.* Extends transformer relation-modelling to continuous time, combining Neural-ODE continuous dynamics with attention. Argues Neural ODEs and variants fail to capture intricate correlations *within* the sequence and shows many specialised irregular-time-series transformers are special cases. Relevant if your belief needs cross-region attention as well as pointwise dynamics.

### F6. Supporting / adjacent.
- **Neural Delay Differential Equations** (Zhu, Guo, Lin, ICLR 2021) — DDEs learn behaviours ODEs cannot. Conceptually apt for a system whose defining feature *is* delay.
- **Spatial-Temporal Delay Differential Equations for traffic forecasting** (arXiv:2402.01231) — nearest existing use of delay-DEs to a traffic-domain spatiotemporal problem.
- **Liquid Time-Constant networks** (Hasani et al., 2021) and **Closed-form Continuous-time (CfC)** (Hasani et al., 2022) — CfC gives an analytic solution to LTC dynamics at roughly 20× speedup while keeping continuous-time properties. **Worth serious consideration for real-time on-vehicle inference**, since ODE-solver latency is your biggest deployment risk.
- **UFO / U-Former ODE** (arXiv:2602.11738, 2026) — first *time-parallel* Neural CDE, ~15× inference-time reduction, attacking the sequential-computation bottleneck. Again, a latency-risk mitigation.
- **ANCDE, CADN** — attention-augmented Neural CDE / ODE-RNN.
- **Graph ODEs survey** (arXiv:2503.23167) — for multi-agent structure over continuous dynamics.
- **Coupled Graph ODE** (Huang, Sun, Wang, KDD 2021) and *Learning continuous system dynamics from irregularly-sampled partial observations* (NeurIPS 2020) — multi-agent interacting systems in continuous time. **Closest existing "multi-agent + continuous-time" work**, though on abstract dynamical systems, not perception.

---

## 8. Strand G — Continuous-time applied to perception and robotics

**This is the decisive section for your novelty claim.**

### G1. StreamingFlow — Shi, Jiang, Wang, Li, Wang, Yang, Yang. CVPR 2024. arXiv:2302.09585.
*Streaming Occupancy Forecasting with Asynchronous Multi-modal Data Streams via Neural Ordinary Differential Equation.*

**Read this paper in full before writing a single line of your own.** Code: github.com/synsin0/StreamingFlow.

- **Motivation.** Existing occupancy predictors output *uniform snapshots* at fixed frequencies inherited from sensor rates (LiDAR ~10 Hz, cameras ~30 FPS), and require strictly synchronised sensor data for fusion. Continuous prediction at any timestamp would cut latency and relax synchronisation. Two obstacles identified: labels exist only at sparse fixed times, and mainstream architectures aren't built for streaming.
- **Method.** Three phases: per-modality BEV encoders (PillarNet for LiDAR; Lift-Splat-Shoot with depth supervision for cameras) → asynchronous multi-sensor fusion → streaming occupancy prediction. The core is **SpatialGRU-ODE**, a spatial GRU whose state derivative follows GRU-ODE-Bayes form, operating on a BEV state of shape [B, C, H, W] (internally compressed to H/4, W/4 for memory). The BEV state starts from zeros and fuses each incoming BEV feature in **trigger mode rather than matching mode** — i.e., the state propagates under the ODE until a new observation arrives, at which point it is updated, then continues propagating. Update path is a dual-pathway SpatialGRU that propagates observed and predicted features separately, mixes their distributions, and combines them through a trust gate and softmax. Auxiliary **KL-divergence loss** between the predicted state distribution and the measured feature distribution. Euler or midpoint solver; fixed or variable ODE step.
- **Results.** *nuScenes* future instance segmentation: StreamingFlow-base 53.9 IoU / 52.8 PQ, beating FusionAD (51.5 / 51.1); StreamingFlow-tiny 47.8 / 46.1, beating camera-only BEVerse-Swin-small (40.8 / 36.1) by +7.0 IoU / +10.0 VPQ and LiDAR-only BE-STI by +7.7 IoU / +5.1 VPQ. *Lyft L5*: 56.9 IoU / 55.9 PQ, +20.6 IoU / +23.5 VPQ over ST-P3. *BEV segmentation:* 50.8 vehicle IoU / 37.2 pedestrian IoU, +10.7 / +22.7 over ST-P3.
- **The streaming results that matter most to you.**
  - **Horizon extension:** trained on 2 s, extends zero-shot to 8 s with graceful decay (2 s: 47.8/46.1 → 8 s: 32.5/29.8). Zero-shot 8 s is only **−0.9 IoU / −1.8 VPQ** below a model fully supervised at 8 s.
  - **Arbitrary query times:** predicts at any requested interval; results stable, with slight decline as prediction becomes denser (0.5 s interval: 47.8 IoU; 0.25 s: 43.4; 0.6 s: 45.6).
  - **Asynchronous streams:** LiDAR at 5 Hz + camera at 2 Hz gives 47.8 IoU; denser inputs (10/2, 10/4) give similar accuracy.
  - **Fusion ablation (be aware of this one).** On nuScenes, conventional *synchronised* spatial-then-temporal fusion actually **beat** SpatialGRU-ODE (50.2 vs 47.8 IoU, 47.0 vs 46.1 PQ). Only on Lyft did the ODE win (56.9 vs 54.6 IoU). **The continuous-time formulation was not free accuracy — it bought flexibility.** Expect the same in your setting, and pre-empt it: your gain must come from what a persistent shared belief does under *degraded communication*, not from clean-channel accuracy.
  - **Solver / step ablations:** midpoint beats Euler on VPQ (+3.2 nuScenes, +0.8 Lyft); finer ODE steps generally help (0.05 s > 0.1 s > 0.5 s); variable step is a good accuracy/latency compromise.
  - **Cost.** Inference 0.1968 s/sample with variable ODE step (faster than FIERY 0.6436 and StretchBEV 0.6469), but ~0.5 s/sample for the 40-frame fine-granularity setting — **prediction density directly costs latency**. Training memory: GRU-ODE 13 GB vs GRU-base 11 GB at 4 supervised frames; at 40 supervised frames GRU-ODE went **out of memory** on a 48 GB A6000 while GRU-base fitted in 39 GB. **Densely-supervised ODE training is memory-hostile. Plan for this.**
- **What they say about CoBEVFlow.** They name it as the closest approach — predicting BEV flow to interpolate asynchronous roadside-to-vehicle timestamps — and argue short-term flow cannot support long-term prediction. **So the ODE-vs-flow argument you want to make has already been made, by them, against your closest CP baseline.** You can cite it rather than re-derive it, but you cannot claim it.
- **What is genuinely left for you.** StreamingFlow is a *single-vehicle* system. Its "asynchronous streams" are its own sensors: they never go permanently missing, never arrive out of order from an adversarial network, carry no per-source age that must be reasoned about, and cost nothing to transmit. Everything hard about the multi-agent case is absent. **That is your paper.**

### G2. Multiscale Sensor Fusion and Continuous Control with Neural CDEs — Singh, Ramirez, Varley, Zeng, Sindhwani. arXiv:2203.08715.
- **Motivation.** Robots receive asynchronous modalities at very different rates (30 Hz video, 100 Hz proprioception, 500 Hz force-torque); the standard fix — batching into fixed windows and encoding — is the wrong abstraction for near-continuous feedback control.
- **Method.** A policy grounded in Neural CDEs: a latent state conditioned on images and *driven* by higher-frequency modalities, fusing each reading as it arrives.
- **Relevance.** Independent confirmation that "asynchronous multi-rate streams → continuous-time latent state" is an established design pattern in robotics. Cite it in related work to show you know the pattern isn't new; that is much better than being told so by a reviewer.

### G3. Conditional Latent ODEs for Motion Prediction in Autonomous Driving — arXiv:2405.19183.
Latent ODEs for multi-agent motion prediction via imitation learning. Object-level, not scene-level, and single-viewpoint — but establishes latent ODEs are already in the driving-prediction literature.

### G4. Physics-guided spatio-temporal Neural ODE for multi-vehicle trajectory prediction — *Information Sciences*, 2026.
Embeds a kinematic backbone into the ODE function and augments with bounded residual dynamics, so basic vehicle-motion consistency is preserved while interaction-induced deviations remain learnable. **Directly applicable design idea:** rather than learning advection from scratch, give your dynamics model a kinematic prior with a learned residual. Cheap to implement, likely to help, and a good ablation.

### G5. Streaming Motion Forecasting — Pang, Ramanan, Li, Wang. IROS 2023.
Argues streaming forecasting has intrinsic safety advantages over discrete snapshot prediction. Useful for motivating the query interface in safety terms rather than aesthetic ones.

### G6. ODE-GS — arXiv (June 2025).
Transformer latent ODEs as continuous-time deformation models for 3D Gaussian Splatting; reports up to 10 dB PSNR improvement and halved perceptual error on long-horizon forecasting. Evidence latent ODEs scale to high-dimensional scene representations.

---

## 9. Strand H — Task-agnostic formulations and query interfaces

Relevant to component 4 ("task-agnosticism becomes structural rather than rhetorical"). The word "task-agnostic" is already claimed, though for a different axis.

### H1. STAMP — Gao, Xu, Li, Wang, Fan, Tu. ICLR 2025. arXiv:2501.18616.
*Scalable Task- and Model-agnostic Collaborative Perception.*
- **Motivation.** Agent heterogeneity — different sensors, model architectures, and *tasks* — blocks collaboration. Also model security: agents shouldn't have to share models.
- **Method.** Lightweight **adapter–reverter pairs** map BEV features between each agent's private domain and a shared **protocol domain**, so features are exchangeable without retraining or model sharing.
- **Results.** On OPV2V and V2V4Real, comparable or better accuracy than SOTA with significantly lower training-resource growth as heterogeneous agents are added. Bills itself as the first task- and model-agnostic CP framework.
- **Relevance — naming conflict.** STAMP owns "task-agnostic" in this field, but along the *agent-heterogeneity* axis. Yours is along the *readout-time* axis. Use different language — "**query interface**" or "**time-queryable scene state**" — or you will be read as a weaker STAMP.

### H2. QUEST — Fan, Wang, Huo, Wang, Liu. ICRA 2024. arXiv:2308.01804.
*Query Stream for Practical Cooperative Perception.*
- **Motivation.** Existing paradigms are either interpretable (result cooperation) or flexible (feature cooperation), never both.
- **Method.** **Query cooperation** — a stream of instance-level queries flows between agents; co-aware instances are fused, unaware instances complemented.
- **Results.** On DAIR-V2X-Seq (camera-based V2I), effective and notably **robust to packet dropout**, with transmission flexibility. The paper also discusses the paradigm's limitations, which is worth reading for how the community reasons about query-based CP.

### H3. INSTINCT — Xu et al. ICCV 2025.
*Instance-Level Interaction Architecture for Query-Based Collaborative Perception.* Quality-aware instance filtering, dual-branch routing separating collaboration-relevant from collaboration-irrelevant instances, and cross-agent local instance fusion. Reports **+13.23% / +33.08%** accuracy on DAIR-V2X and V2V4Real while cutting bandwidth to **1/281** and **1/264** of SOTA.
**Relevance:** these bandwidth ratios are the real bar for your delta-messaging claim. If your "8× headroom" argument is stated against dense feature maps, an instance-query baseline at 1/281 will make it look modest. Position deltas against *instance-level* methods, not against dense transmission.

### H4. CoopDETR — arXiv:2502.19313 (2025).
Unified cooperative perception via object queries across multiple agents, generalising QUEST beyond the single-vehicle-plus-infrastructure case.

### H5. CoCMT — Cross-modal transformer with Efficient Query Transformer (EQFormer). On V2V4Real with top-50 object queries: **0.416 Mb** bandwidth, stated as 83× less than SOTA, with +1.1 AP70.

---

## 10. Strand I — Datasets, benchmarks, and codebases

| Dataset | Year | Type | Real? | Modes | Scale | Notes |
|---|---|---|---|---|---|---|
| **OPV2V** (Xu et al., ICRA 2022) | 2022 | Sim (CARLA+OpenCDA) | No | V2V | ~11K frames, ~232K 3D boxes, 2–7 agents (avg 3), 8 towns, 70+ scenarios | The default simulated V2V benchmark |
| **V2XSet** (Xu et al., ECCV 2022) | 2022 | Sim (CARLA+OpenCDA) | No | V2V + V2I | 11,447 frames (6,694/1,920/2,833), 73 scenes, 2–7 agents, 5 road types | Includes noise simulation |
| **V2X-Sim** (Li et al., RA-L 2022) | 2022 | Sim | No | V2V + I | ~10K point-cloud frames, 47.2K samples, 5 Hz, 3 CARLA towns | Detection + tracking + segmentation benchmarks |
| **DAIR-V2X-C** (Yu et al., CVPR 2022) | 2022 | Real | Yes | V2I | 38,845 LiDAR + 38,845 camera frames, ~464K boxes, 10 classes, 9,311 cooperative-annotated pairs, 28 intersections | Native pair time difference within ±30 ms; latency usually simulated by frame substitution |
| **V2V4Real** (Xu et al., CVPR 2023) | 2023 | Real | Yes | V2V | 20K LiDAR + 40K RGB, 240K boxes, 5 classes, 410 km, HD maps | Detection, tracking, sim2real |
| **V2X-Seq** (Yu et al., CVPR 2023) | 2023 | Real | Yes | V2I | ~15K point clouds/images, 9 classes, track IDs | Sequential perception (SPD) + forecasting splits |
| **IRV2V** (Wei et al., NeurIPS 2023) | 2023 | Sim | No | V2V | — | **First dataset built specifically for varied temporal asynchrony.** Your natural home benchmark |
| **TUMTraf-V2X** (Zimmer et al., CVPR 2024) | 2024 | Real | Yes | V2I | 2.0K point clouds, 5.0K images, 8 classes, 200 m range | Day+night, traffic violations, OpenLABEL format |
| **V2X-Real** (ECCV 2024) | 2024 | Real | Yes | V2X | — | Large-scale real V2X |
| **V2XPnP-Seq** (Zhou et al., ICCV 2025) | 2025 | Real | Yes | All (V2V, I2I, VC, IC) | — | First real sequential dataset covering all modes, with trajectories + maps |
| **V2X-Traj** | 2024 | Real | Yes | V2X | — | First real V2X motion-forecasting dataset with multiple AVs + infrastructure per scene |
| **RCooper** (CVPR 2024) | 2024 | Real | Yes | Roadside | — | Roadside cooperative perception |

**Codebases:** OpenCOOD (V2X-ViT, TraF-Align, STAMP built on it), CoPerception (SyncNet, Where2comm), V2XPnP (11-model zoo). Paper lists: `Siheng-Chen/CollaborativePerception_paper`, `frankwnb/Collaborative-Perception-Datasets-for-Autonomous-Driving`.

**Recommendation.** Your evaluation should run **IRV2V** (asynchrony is native, not simulated) + **DAIR-V2X-Seq or V2V4Real** (real, and where TraF-Align's numbers exist for direct comparison) + **V2XPnP-Seq** (only dataset supporting the multi-task query readout your component 4 promises). If you cannot show the time-query interface serving detection *and* tracking *and* prediction, component 4 is rhetoric — which is exactly what the note says it should not be.

---

## 11. Strand J — Surveys and positioning

### J1. Multi-Agent Embodied Autonomous Driving: From V2X Information Exchange to Shared World Models — Hu et al., arXiv:2606.13840 (June 2026).
- **What it is.** A 380+ publication survey organised around **Shared World Models (SWMs)**: predictive cross-agent representations maintained across vehicles, infrastructure, and other participants.
- **Its criteria.** C1 shared-state alignment (satisfied by joint latent states, shared latent maps, or a maintained V2X knowledge pool; *not* satisfied by broadcasting BSMs, CAMs, detections, or raw features); C2 intent/plan alignment; C3 coordinated downstream action.
- **Its stated gaps.** Evaluation is concentrated in simulation and offline protocols; foundation-model coordination lacks real-time safety guarantees. Priorities named: **verifiable shared-state maintenance**, robust intent/plan alignment, safe coordinated action under communication, latency, and deployment constraints.
- **Why it matters to you — both ways.** It is the best possible citation for "the field recognises this gap": your persistent belief is close to a textbook C1 satisfier and the survey says maintaining shared state is unsolved. But it also means your framing already has a name in the literature, so you should adopt or explicitly relate to "shared world model" language rather than coining a competing term.

### J2. Towards Vehicle-to-everything Autonomous Driving: A Survey on Collaborative Perception — arXiv:2308.16714.
Organises latency compensation into two families: (i) history-based temporal-prediction methods (SyncNet); (ii) implicit delay-aware modules trained on simulated latency, or transmission strategies (AutoCast, AVR, V2X-ViT, DAIR-V2X). Also runs its own lossy-communication robustness study, training under ideal communication and fine-tuning under loss, simulating loss by uniformly sampling a retention probability on intermediate features. **Their protocol is a reasonable reference point for validating your CommChannel's loss model.**

### J3. Vehicle-to-Everything Cooperative Perception for Autonomous Driving — arXiv:2310.03525.
Contains the taxonomy sentence you should borrow: first-order expansion works over very short intervals but has inherent limits; SyncNet used a dual-branch pyramid LSTM; HFP used multidimensional convolutional mixing; **these did not effectively address irregular delays**; CoBEVFlow treated each message as an irregular sample but still depended on past data for compensation; V2X-PC bypassed that dependency via low-level coordinates. Also the source of the LCRN critique (repairs degradation, not loss).

### J4. Others.
- *Collaborative Perception in Autonomous Driving: Methods, Datasets and Challenges* — Han, Zhang, Li, Jin, Lang, Li. IEEE ITS Magazine 15(6), 2023.
- *Multi-agent Collaborative Perception for Robotic Fleet: A Systematic Review* — arXiv:2405.15777.
- *Systematic Literature Review on Vehicular Collaborative Perception — A Computer Vision Perspective* — arXiv:2504.04631.
- *A Survey on Deep Multi-Task Learning in Connected Autonomous Vehicles* — arXiv:2508.00917. Contains a comparison table of V2X cooperative prediction works.

---

## 12. The novelty ledger, claim by claim

The note's core assertion is: *"One architecture subsumes four separate literatures. That's the claim."* Here is what survives.

### Claim 1 — "Every collaborative perception system is stateless snapshot exchange."
**False as of 2023.** SCOPE (ICCV 2023) maintains ego temporal context; V2XPnP (ICCV 2025) does multi-frame spatio-temporal fusion; CoST (ICCV 2025) maintains an explicit memory bank; V2X-INCOP retains historical cooperation information; What2comm integrates temporal ego cues.
**Survivable rewrite:** *no existing system maintains a shared, continuously-defined belief that all agents' messages are integrated into at their own timestamps.* That is still true and is the honest version.

### Claim 2 — "Latency: dynamics are the model; updates applied at their own timestamp."
**Partially novel.** CoBEVFlow explicitly claims irregular-continuous-timestamp handling without discretisation. StreamingFlow already applies updates at their own timestamps under learned ODE dynamics. What is new is doing it *across agents*, where the sender's timestamp is arbitrarily old and the age varies per agent.
**Survivable rewrite:** state-based integration vs. per-message warping — retention vs. transformation.

### Claim 3 — "Loss: nothing to recover — the belief persists."
**Strongest claim in the proposal.** No prior work makes loss a non-event by construction; V2X-INCOP and LCRN both frame it as something to be repaired or predicted. Even CoST's memory bank retains *static* content by pose projection, not *all* prior information.
**Caveat to handle:** V2XPnP's finding that error accumulates when lossy intermediate features are transformed across time. A persistent belief could compound this. Measure belief drift explicitly.

### Claim 4 — "Bandwidth: deltas; cost scales with change, not scene."
**Substantially anticipated.** CoST transmits dynamic objects only; CooperTrim selects by temporal continuity; V2X-DSC argues the same conditional-information principle spatially; InfoCom names temporal cues as its next step. And instance-query methods already hit 1/281 bandwidth.
**Survivable rewrite:** deltas *against a jointly-evolved continuous-time belief* rather than against a static-scene memory — meaning the delta is against the receiver's *predicted* state, not its last observation. That is a real distinction and worth making precisely.

### Claim 5 — "Async: native, no synchrony assumption exists."
**Novel in the multi-agent setting.** No CP work found handles arrival order, arrival rate, and per-agent age purely by construction. The AoI paper (2602.13439) is the nearest and it still compensates rather than dissolving the problem.

### Claim 6 — "Task-agnosticism becomes structural via a time-query readout."
**Novel in CP as stated, but the vocabulary is taken** (STAMP) and the multi-task evidence bar is high (V2XPnP-Seq). Also, StreamingFlow already demonstrates the time-query readout — you're extending, not inventing.

### Claim 7 — "Channel in the training loop."
**Not novel.** Called distortion-in-the-loop training; SyncNet uses latency curriculum; QPoint2Comm uses masked packet-loss training; the 2308.16714 survey fine-tunes under simulated loss. Present as methodology, not contribution.

### Suggested honest framing
> Collaborative perception has converged on per-message compensation: each received message is repaired, warped, or reweighted toward the receiver's present, then discarded. We instead maintain a shared continuous-time scene belief into which each message is integrated at its own timestamp. Continuous-time latent dynamics have been shown effective for asynchronous fusion *within* a single agent's sensor suite [StreamingFlow]; we show that extending this to a *communication network* is not a straightforward port — network asynchrony additionally involves permanent message loss, unbounded and heterogeneous per-agent age, adversarial arrival order, and a hard bandwidth budget, none of which arise intra-vehicle. We address these with an asynchronous update operator over agent-tagged messages, delta encoding against the receiver's *predicted* belief, and end-to-end training under a sampled channel.

That claim is defensible and still worth a strong venue. The original framing is not.

---

## 13. Concrete recommendations

1. **Read StreamingFlow end to end today.** Then decide: build on SpatialGRU-ODE (fast, honest, cites cleanly) or diverge to a Neural CDE formulation (better theoretical fit for multi-source control signals, more work, stronger novelty). Justify whichever you pick.
2. **Read CoST (ICCV 2025) in full** and write the differentiation paragraph before writing anything else. It is the paper most likely to be a reviewer's "isn't this just…".
3. **Get TraF-Align checkpoints** (released, V2V4Real + V2X-Seq). Their 400 ms numbers (−4.87% / −5.68% AP50) are the bar.
4. **Add a linear-advection ablation.** CoDynTrust achieved SOTA with linear extrapolation. If your learned dynamics don't beat linear advection, your displacement finding doesn't imply what the note says it implies.
5. **Measure belief drift.** Plot detection quality vs. time-since-last-message per agent, and vs. cumulative number of lossy integrations. This is the experiment that either validates or kills the persistence argument, and V2XPnP's error-accumulation finding says it's a live risk.
6. **Budget for ODE cost early.** StreamingFlow: 0.1968 s/sample at variable step, ~0.5 s at fine granularity, and OOM at dense supervision on a 48 GB card. Look at CfC (~20× speedup) or the time-parallel Neural CDE (~15×) if latency becomes the blocker — which for an on-vehicle real-time system it likely will.
7. **Re-benchmark bandwidth against instance-query methods**, not dense feature maps. 1/281 (INSTINCT) and 0.416 Mb (CoCMT) are the real comparators; "8× headroom" measured against dense transmission will read as a straw man.
8. **Adopt or explicitly relate to "shared world model" language** (arXiv:2606.13840) rather than coining a competing term. Cite it for the gap.
9. **Connect per-agent age to Age of Information** formally (see 2602.13439). It gives the systems half of the paper a principled vocabulary.
10. **Consider a kinematic prior with learned residual** in the dynamics function (per the 2026 physics-guided Neural ODE work) — likely a cheap accuracy win and a clean ablation.

---

## 14. What this check did *not* cover

Be aware of these gaps before you rely on the verdict:

- **Full-text read** was done only for StreamingFlow. Everything else rests on abstracts, contribution lists, and snippets. In particular, **CoST**, **CooperTrim**, **the AoI paper (2602.13439)**, and **V2X-DSC** should be read in full — all four touch your contributions directly.
- **Very recent preprints (mid-2026)** are the highest scoop risk and the least indexed. Set up arXiv alerts on cs.CV + cs.RO for: collaborative perception, cooperative perception, V2X, asynchronous, continuous-time.
- **Non-driving multi-agent perception** — multi-robot SLAM, distributed estimation, sensor networks — was not searched. Classical distributed Bayesian filtering (consensus Kalman, delayed-state filters, out-of-sequence measurement handling) is a *very* old literature that solves structurally the same problem, and a reviewer from robotics will ask about it. **Out-of-sequence measurement (OOSM) filtering is essentially your asynchronous update operator in linear-Gaussian form.** Search that before submitting.
- **Patents** — one USPTO filing on Neural ODEs for irregularly-sampled time series appeared in results (US 12,462,146). Not perception-specific, but worth a glance if commercialisation matters.
- **Numerical verification** of results tables was not possible; treat every number here as "as reported by the source", not independently checked.

---

## 15. Compact reference list

**Foundations**
1. Wang, Manivasagam, Liang, Yang, Zeng, Urtasun. V2VNet. ECCV 2020.
2. Li, Ren, Wu, Chen, Feng, Zhang. Learning Distilled Collaboration Graph (DiscoNet). NeurIPS 2021.
3. Liu, Tian, Glaser, Kira. When2com. CVPR 2020. / Liu, Tian, Ma, Glaser, Kuo, Kira. Who2com. ICRA 2020.
4. Xu, Xiang, Tu, Xia, Yang, Ma. V2X-ViT. ECCV 2022.
5. Xu, Tu, Xiang, Shao, Zhou, Ma. CoBEVT. CoRL 2022.
6. Chen, Tang, Yang, Fu. F-Cooper. SEC 2019.
7. Dao et al. Practical Collaborative Perception. arXiv:2307.01462.

**Latency / asynchrony**
8. Lei, Ren, Hu, Zhang, Chen. Latency-Aware Collaborative Perception (SyncNet). ECCV 2022. arXiv:2207.08560.
9. Wei, Wei, Hu, Lu, Zhong, Chen, Zhang. CoBEVFlow. NeurIPS 2023. arXiv:2309.16940.
10. Yu, Tang, Xie, Mao, Luo, Nie. FFNet. NeurIPS 2023. arXiv:2311.01682.
11. Song, Yang, Wen, Li. TraF-Align. CVPR 2025. arXiv:2503.19391.
12. Xu, Li, Wang, Yang, Wu, Chen, Wang. CoDynTrust. ICRA 2025. arXiv:2502.08169.
13. Liu et al. V2X-PC. arXiv:2403.16635.
14. Spatiotemporal Feature Alignment with Network Synchronization and Age of Information. arXiv:2602.13439.

**Loss / robustness**
15. Li, Xu, Liu, Ma, Chi, Ma, Yu. V2VAM / LCRN. IEEE T-IV 8(4), 2023. arXiv:2212.08273.
16. Ren, Lei, Wang, Dianati, Wang, Chen, Zhang. V2X-INCOP. IEEE T-IV 9(4), 2024. arXiv:2304.11821.
17. Xu et al. QPoint2Comm. arXiv:2602.21667.
18. Lu et al. CoAlign. ICRA 2023. / FreeAlign. ICRA 2024.
19. Li, Fang, Bai, Chen, Juefei-Xu, Feng. Among Us (ROBOSAC). ICCV 2023.

**Bandwidth**
20. Hu, Fang, Lei, Zhong, Chen. Where2comm. NeurIPS 2022. arXiv:2209.12836.
21. Tang et al. CoST. ICCV 2025. arXiv:2508.00359.
22. CooperTrim. arXiv:2602.13287.
23. Yang et al. What2comm. ACM MM 2023. / How2comm. NeurIPS 2023.
24. Select2Col. arXiv:2307.16517.
25. Shenkut, Kumar. ReVQom. ICASSP 2026. arXiv:2509.21464.
26. InfoCom. arXiv:2512.10305. / V2X-DSC. arXiv:2602.00687. / WhisperNet. arXiv:2603.01708.
27. QuantV2X. arXiv:2509.03704.

**Temporal / persistent state**
28. Yang, Yang, Zhang, Li, Liu, Liu, Wang, Sun, Song. SCOPE. ICCV 2023. arXiv:2307.13929. (+ SCOPE++)
29. Zhou et al. V2XPnP. ICCV 2025. arXiv:2412.01812. (+ TurboTrain, 2025)
30. Koh et al. OnlineBEV. arXiv:2507.08644.
31. CoMamba. arXiv:2409.10699.
32. CATNet. arXiv:2603.05255.

**Continuous-time deep learning**
33. Chen, Rubanova, Bettencourt, Duvenaud. Neural ODEs. NeurIPS 2018. arXiv:1806.07366.
34. Rubanova, Chen, Duvenaud. Latent ODEs / ODE-RNN. NeurIPS 2019. arXiv:1907.03907.
35. De Brouwer, Simm, Arany, Moreau. GRU-ODE-Bayes. NeurIPS 2019.
36. Kidger, Morrill, Foster, Lyons. Neural CDEs. NeurIPS 2020.
37. Chen et al. ContiFormer. arXiv:2402.10635.
38. Zhu, Guo, Lin. Neural Delay Differential Equations. ICLR 2021.
39. Hasani et al. Liquid Time-Constant Networks, 2021. / Closed-form Continuous-time (CfC), 2022.
40. Kuleshov, Marusov, Zaytsev. U-Former ODE. arXiv:2602.11738.
41. Huang, Sun, Wang. Coupled Graph ODE. KDD 2021. / Learning continuous system dynamics from irregularly-sampled partial observations. NeurIPS 2020.

**Continuous-time in perception / robotics**
42. **Shi, Jiang, Wang, Li, Wang, Yang, Yang. StreamingFlow. CVPR 2024. arXiv:2302.09585.** ← the key one
43. Singh, Ramirez, Varley, Zeng, Sindhwani. Multiscale Sensor Fusion with Neural CDEs. arXiv:2203.08715.
44. Conditional Latent ODEs for Motion Prediction. arXiv:2405.19183.
45. Physics-guided spatio-temporal Neural ODE trajectory prediction. Information Sciences, 2026.
46. Pang, Ramanan, Li, Wang. Streaming Motion Forecasting. IROS 2023.

**Task-agnostic / query**
47. Gao, Xu, Li, Wang, Fan, Tu. STAMP. ICLR 2025. arXiv:2501.18616.
48. Fan, Wang, Huo, Wang, Liu. QUEST. ICRA 2024. arXiv:2308.01804.
49. Xu et al. INSTINCT. ICCV 2025.
50. CoopDETR. arXiv:2502.19313.
51. Wang, Zhang, Wang, Zhao, Zhou. CORE: Cooperative Reconstruction for Multi-Agent Perception. ICCV 2023.
52. Lu, Hu, Zhong, Wang, Chen, Wang. HEAL: Extensible Framework for Open Heterogeneous Collaborative Perception. ICLR 2024. arXiv:2401.13964.

**Datasets**
53. Xu, Xiang, Xia, Han, Li, Ma. OPV2V. ICRA 2022.
54. Li, Ma, An, Wang, Zhong, Chen, Feng. V2X-Sim. RA-L 2022. arXiv:2202.08449.
55. Yu et al. DAIR-V2X. CVPR 2022. / V2X-Seq. CVPR 2023.
56. Xu et al. V2V4Real. CVPR 2023.
57. Zimmer et al. TUMTraf-V2X. CVPR 2024. arXiv:2403.01316.
58. V2X-Real. ECCV 2024.

**Surveys**
59. Hu et al. Multi-Agent Embodied Autonomous Driving: From V2X Information Exchange to Shared World Models. arXiv:2606.13840.
60. Towards V2X Autonomous Driving: A Survey on Collaborative Perception. arXiv:2308.16714.
61. Vehicle-to-Everything Cooperative Perception for Autonomous Driving. arXiv:2310.03525.
62. Han, Zhang, Li, Jin, Lang, Li. Collaborative Perception in Autonomous Driving. IEEE ITS Magazine 15(6), 2023.
