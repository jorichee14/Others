# A Co-Calibrated Infrastructure Sensing Dataset for Perception Under Spatially-Varying Connectivity

*Research direction, motivation, contributions, related work, ground-truth model, and benchmark suite.*

> **Status:** Draft research-direction document for a dataset currently being recorded. Citations were web-verified during drafting; entries still marked *(verify)* need a final check before external publication. Numeric ground-truth values quoted here are taken from the calibration session records referenced in the appendix and are subject to update as recording completes.

---

## 0. TL;DR

We are recording an **indoor, fixed-infrastructure, multi-modal sensing dataset** in which three modalities — a **multi-view infrastructure camera network**, a **multi-radar mmWave network** (TI IWR6843ISK), and a **Wi-Fi / RF link-quality sensor suite** — are all expressed in a **single metric world frame** and shipped with **independently-bounded, per-sensor ground-truth uncertainty**.

The dataset's distinguishing feature is not any single modality but their **co-registration plus an honest uncertainty model**, which together open a research axis that existing datasets do not cover: **perception coupled to communication** — how perception quality and wireless connectivity co-vary through space, and how a system should behave when the link it depends on degrades exactly where perception is hardest.

---

## 1. Introduction

Most multi-sensor perception datasets are built for one of two worlds. The first is **automotive**: a vehicle drives through the world carrying camera, LiDAR, and (increasingly) radar, and the research question is ego-centric perception in motion (nuScenes, View-of-Delft, K-Radar, TJ4DRadSet, RADIATE). The second, newer world is **roadside / infrastructure**: sensors are bolted to poles or gantries over an intersection, and the research question is bird's-eye monitoring and vehicle-to-infrastructure cooperation (DAIR-V2X, Rope3D, IPS300+, TUMTraf/A9).

Both worlds share three limits that this dataset is designed to escape:

1. **They are outdoor and vehicle/traffic-centric.** Indoor infrastructure perception — the setting for warehouses, hospitals, factories, and service robots — is comparatively under-served by *radar-inclusive* multi-view datasets.
2. **They treat ground truth as an oracle.** Poses and boxes are published as "truth" with, at best, a single aggregate accuracy figure. When the ground truth's errors are correlated with the system under evaluation, the benchmark flatters that system instead of exposing it.
3. **They ignore the network.** A perception system deployed on real infrastructure runs on a *wireless link* whose quality varies dramatically through space. No mainstream perception dataset records where the link is good, where it collapses, and how that correlates with where perception itself is hard.

This dataset attacks all three. It is **indoor and infrastructure-mounted**; it carries **per-sensor, independently-bounded uncertainty as first-class metadata**; and it adds a **co-located, spatially-registered Wi-Fi/RF modality** so that perception and connectivity can be studied together for the first time in one frame.

## 2. Motivation

### 2.1 Why indoor infrastructure perception

Fixed cameras and radars watching a shared indoor space are the sensing substrate for a growing class of deployments — automated warehouses, elder-care facilities, smart manufacturing cells, retail analytics — where the sensors do not move but the agents they watch do. These deployments need **metric, multi-view, privacy-aware** perception, and they need it to keep working through occlusion and poor lighting. That is exactly where **mmWave radar complements cameras**: radar sees through darkness and smoke, measures radial velocity directly (Doppler), and is inherently privacy-preserving because it does not form a recognizable image of a person. A dataset that co-registers fixed cameras and fixed radars indoors, with certified extrinsics, is the missing substrate for benchmarking this class of system.

### 2.2 Why connectivity belongs in a perception dataset

Infrastructure perception is almost never fully on-device. Frames are streamed to an edge server; tracks are fused in a central node; a robot offloads heavy inference over Wi-Fi. **The wireless link is part of the perception pipeline** — and it is the least reliable part. Link quality varies by tens of dB across a single room, throughput can collapse behind a metal shelf, and dead zones are common precisely in the cluttered areas where perception matters most.

Today a researcher who wants to study *connectivity-aware perception* — adaptive offloading, graceful degradation, coverage-aware sensor placement, communication-aware planning — has **no dataset** that provides perception and connectivity in the same spatial frame. They must either simulate the radio or ignore it. This dataset removes that gap: every Wi-Fi/RF sample is timestamped and pose-registrable, so a coverage/throughput/latency field can be built over the exact space the cameras and radars observe.

### 2.3 Why honest, bounded ground truth

The calibration procedures underpinning this dataset were deliberately designed around one principle, stated in the source documents and adopted here as a dataset-wide commitment:

> Ground truth is an **estimate**, not an oracle. Its value comes from having error that is **bounded**, **per-sensor**, and — critically — **independent of the system it is later used to evaluate**, so the two fail differently and the ground truth can expose the system's mistakes rather than flatter them.

Concretely, camera world-poses come from ChArUco handshakes (bounded, drift-free, mm-level) carried by a GLIM trajectory (drift-prone, growing with path length), and each camera's pose ships with a residual bound tied to the trajectory length that fed it. Radar-camera extrinsics come with **per-DOF covariance and observability**, and are cross-validated by an independent physical invariant (the shared rig apex offset). This turns "ground truth" from a number into a **distribution with a stated shape**, which is what makes uncertainty-aware benchmarking possible.

## 3. Contributions

1. **A single-frame, tri-modal infrastructure dataset.** Multi-view fixed cameras + multi-radar mmWave + Wi-Fi/RF link quality, all co-registered into one metric world frame with a published transform tree.
2. **The first perception dataset with a co-located, pose-registered RF-connectivity modality**, enabling joint study of perception and communication (to our knowledge, no prior public dataset provides both in one spatial frame — *to be confirmed against the related-work survey in §5*).
3. **Uncertainty-as-metadata ground truth.** Every ground-truth pose carries an independent, per-sensor error bound with a stated derivation (ChArUco reprojection residual, GLIM loop-closure drift, radar per-DOF covariance), not a single blanket accuracy claim.
4. **A reproducible, measurement-first calibration methodology** for both camera-network and radar-camera extrinsics, released with the data so the ground truth is auditable rather than asserted.
5. **A benchmark suite of seven tasks** spanning calibration, multi-view 3D tracking, radar-camera fusion, privacy-preserving sensing, RF coverage mapping, connectivity-aware perception, and calibration-robustness — each specified with inputs, ground truth, metrics, and a provided baseline.
6. **An anisotropic-covariance radar fusion baseline** with measured performance (fused 1σ ≈ [53, 69, 29] mm vs. ≈[112, 325] / [287, 112] mm single-radar), demonstrating that the released extrinsics and uncertainty model are usable end-to-end.

## 4. Background: the sensing stack

### 4.1 Multi-view infrastructure camera network

*N* fixed RGB cameras (optionally depth) observe a shared indoor space. Their extrinsics are recovered in a common world frame by the **GLIM-anchored, ChArUco-certified** procedure:

- A **reference camera** moves through the space; **GLIM** (LiDAR-inertial SLAM, optionally visual) estimates its continuous trajectory.
- At each fixed camera, a **ChArUco board handshake** gives a direct, drift-free, mm-level relative pose between the reference camera and that fixed camera at a shared instant.
- A fixed **origin board** defines the world frame; **revisits** to it measure and correct GLIM drift, and a **final loop closure** reports total trajectory drift.
- World pose of a fixed camera: `T_world_infra = T_world_ref(t) · T_ref_infra`, where the handshake board cancels out of the relative pose and `T_world_ref(t)` is interpolated from the globally-optimised trajectory (SLERP for rotation, lerp for translation).

The key property: **the network is board-defined, not drift-defined**, and accuracy is reported *per camera*, tied to the trajectory-segment length feeding it.

### 4.2 Multi-radar mmWave network

Three TI IWR6843ISK mmWave radars (radar1, radar2, radar_infra) are calibrated to the ZED left camera by a **ChArUco + trihedral-corner-reflector rig**. The estimator is **measurement-first**: measure the rig apex offset and a tape-measured extrinsic prior, seed them as Bayesian priors, collect one diverse pose set, and solve by **maximum likelihood in the radar's native (range, azimuth, elevation) space**, weighted by each axis's real sensor σ, with Huber robust loss, σ-gated outlier rejection, and joint MAP refinement of the rig offset.

Radar noise is **anisotropic and range-dependent**: range is precise (≈cm), angle is coarse (degrees), and cross-range error grows as ≈ range·σ_az. radar1 and radar2 are mounted with **orthogonal soft axes** (radar2 rolled ~90°), so their weak directions are perpendicular and fusion constrains every axis. radar_infra is not yet finalized (see §8).

### 4.3 Wi-Fi / RF link-quality suite

Three independent ROS 2 nodes sweep three dimensions of the wireless link:

- **Passive RF monitor** (up to ~5 Hz): RSSI, SNR (when a real noise floor is reported), negotiated rate, MCS/NSS/width, retries/failures, channel utilization, error counters — read-only, no traffic injected, NaN/−1 for genuinely unknown values.
- **Active throughput monitor** (iperf3, ~1 Hz continuous or periodic bursts): achievable goodput, retransmits, and — in continuous mode — kernel-socket TCP RTT from the loaded connection.
- **Latency/loss monitor** (ping, 1 Hz, cheap enough to run during live operation): per-ping RTT and rolling-window loss.

Every message is timestamped for **offline time-join to pose**, producing coverage / throughput / latency / dead-zone maps over the observed space.

## 5. Related Work

*The claims of novelty in §3 rest on this survey. Entries are grouped by theme; the gap this dataset fills is stated at the end of each group.*

### 5.1 Automotive radar-camera (and radar-LiDAR-camera) datasets

The dominant line of radar-inclusive perception datasets is vehicle-mounted and outdoor:

- **nuScenes** (Caesar et al., CVPR 2020) — 6 cameras, 5 automotive radars, 1 LiDAR; the reference multimodal AD dataset, but radar is sparse 2D and the setting is ego-motion on roads.
- **View-of-Delft (VoD)** (Palffy et al., RA-L 2022) — 4D (elevation-capable) radar + stereo camera + LiDAR, urban driving.
- **TJ4DRadSet** (Zheng et al., ITSC 2022) — 4D radar + camera + LiDAR with tracking labels.
- **K-Radar** (Paek et al., NeurIPS 2022) — 4D radar tensor dataset with adverse-weather emphasis.
- **RADIATE** (Sheeny et al., ICRA 2021) — scanning radar + camera + LiDAR in adverse weather.
- **aiMotive** (Matuszka et al., 2022) — long-range multimodal AD dataset.
- **CARRADA** (Ouaknine et al., ICPR 2020), **RadarScenes** (Schumann et al., 2021), **Astyx HiRes2019** (Meyer & Kuschk, 2019) — radar-focused sets with camera/annotation of varying richness.

*Gap:* all are vehicle-mounted, outdoor, ego-motion; none are fixed-infrastructure indoor, and none include an RF-connectivity modality.

### 5.2 Roadside / infrastructure perception datasets

The closest existing work in *fixed-sensor* spirit is roadside/ITS:

- **DAIR-V2X** (Yu et al., CVPR 2022) — the first large vehicle-infrastructure-cooperative dataset; infrastructure side has camera + LiDAR.
- **Rope3D** (Ye et al., CVPR 2022) — roadside monocular 3D detection.
- **IPS300+** (Wang et al., ICRA 2022) — dense roadside multi-modal (camera + LiDAR).
- **TUMTraf / A9 (Providentia++)** (Cress et al., 2022–2024) — highway/intersection infrastructure sensing.
- **LUMPI** (Busch et al., 2022) — multi-perspective intersection dataset.

*Gap:* these establish the fixed-infrastructure paradigm but are **outdoor traffic, camera+LiDAR** — radar is rare and Wi-Fi/RF absent; indoor is out of scope.

### 5.3 Radar-camera extrinsic calibration

The dataset's ground truth depends on radar-camera calibration, so the method is positioned against this literature:

- **Target-based** dual-purpose rigs pairing a corner reflector with a checkerboard/ChArUco board are an established pattern. Recent tools include **4D-CAAL** (Yao et al., arXiv:2601.21454, 2026 — checkerboard-for-camera + rear reflector-for-radar, plus auto-labeling), the **3D-UPnP** uncertainty-aware PnP tool (Cao et al., arXiv:2507.19829, 2025), and the open-source multi-sensor tool of **Domhof et al.** (ICRA 2019; extended in IEEE T-IV 6(3), 2021).
- **Targetless** methods align radar and camera without a fiducial: via **trajectory alignment** (Durmaz & Cevikalp, *Sensors* 2025), **track-to-track association** (Liu et al., *Sensors* 25(3):949, 2025), learned **common features** (Cheng & Cao, NAECON 2023 / arXiv:2309.00787), or azimuth-angle + multi-frame tracking (**Fusion calib**, Zhang et al., *Scientific Reports* 2025 — note the distinct same-nicknamed PRL 2023 road-plane method).
- **Anisotropic / measurement-space** weighting — modelling the non-uniform spherical-to-Cartesian radar noise rather than isotropic Kabsch/Umeyama alignment — is an emerging best practice, made explicit in the 3D-uncertainty PnP formulation above.
- **Doppler-consistency** correspondence is recognized in radar odometry/scan-matching (Kim et al., *Doppler Correspondence*, arXiv:2502.11461, 2025; Dynamic-ICP; DGRO).
- **Surveys:** Shi et al. (*Radar and Camera Fusion … A Comprehensive Survey*, arXiv:2410.19872, 2024/IEEE 2025) and Han et al. (*4D Millimeter-Wave Radar in Autonomous Driving: A Survey*, arXiv:2306.04242, 2023) frame the field.

*Position:* our procedure is **consistent with, not ahead of,** this trend. Its differentiation is a *specific engineering combination* (uniform moderate translation prior that auto-dominates the blind axis; joint MAP estimation of the rig's own reflector-to-board offset; camera-predicted-radial-velocity Doppler gating for hand-held sweeps; live pose-diversity feedback), not a new paradigm. See §7 for the honest novelty stance.

### 5.4 Multi-camera extrinsics, SLAM, and geometric primitives

- **GLIM** (Koide, Yokozuka, Oishi & Banno, *Robotics and Autonomous Systems* 179:104750, 2024) — GPU-accelerated **range/LiDAR-inertial** SLAM with **global factor-graph optimization** (registration-cost minimization; explicit loop closure via the `glim_ext` add-on), optionally accepting multi-camera visual-feature constraints; provides the continuous carrier trajectory.
- **ArUco / ChArUco** (Garrido-Jurado et al., *Pattern Recognition* 47(6):2280–2292, 2014; ChArUco has no standalone paper — it is an OpenCV construct, Bradski 2000) — the fiducial that provides drift-free metric handshakes.
- **Umeyama** (*IEEE TPAMI* 13(4):376–380, 1991) / **Kabsch** (*Acta Cryst.* A32, 1976, corr. 1978) — closed-form point-set rigid alignment used as estimator initialization.
- **Rotation averaging** (Hartley, Trumpf, Dai & Li, *IJCV* 103(3):267–305, 2013) and **SLERP** (Shoemake, SIGGRAPH 1985) for correct SO(3) mean and interpolation.
- **Huber** (*Ann. Math. Stat.* 35(1):73–101, 1964) M-estimation for robustness; MAP estimation with priors is standard Bayesian estimation (Thrun et al., *Probabilistic Robotics*, 2005).

*Gap:* these are the tools; the contribution is composing them into a **certified, auditable, per-sensor-bounded** ground-truth pipeline released with the data.

### 5.5 Wi-Fi / RF sensing, coverage mapping, and connectivity-aware robotics

- **RSSI/CSI indoor localization** — RADAR (Bahl & Padmanabhan, INFOCOM 2000), Horus, SpotFi (Kotaru et al., SIGCOMM 2015), and surveys thereof.
- **Radio maps / REM / radio environment maps** and Gaussian-process signal-strength interpolation for coverage prediction.
- **Communication-aware / connectivity-aware motion planning** — robots planning to maintain bandwidth/coverage (surveys of comm-aware robotics).
- **Computation offloading under varying link quality** — edge/cloud robotics where offload decisions depend on the link.
- **Wi-Fi CSI human sensing** datasets (activity recognition, presence) — contrasted here with our *radar-based* privacy-preserving sensing.

*Gap:* Wi-Fi sensing and perception are studied in **separate communities with separate datasets**. We are not aware of a public dataset that provides **perception (camera/radar) and wireless link quality in one metric frame** — the union this dataset targets. *(To be confirmed by the survey; stated as the central novelty claim.)*

## 6. Ground Truth and Its Uncertainty Model

Ground truth is released **with its error model**, per modality.

### 6.1 Camera-network world poses

| GT product | Derivation | Reported uncertainty |
|---|---|---|
| Per-camera world pose `T_world_infra` | ChArUco handshake ∘ interpolated GLIM reference pose, anchored to origin board | **ChArUco reprojection residual** (mm/deg), *per camera* |
| Trajectory quality | GLIM final loop-closed trajectory | **Loop-closure residual** (end-vs-origin drift), the headline trajectory figure |
| Cross-pass consistency | Repeat passes registered | **RMSE** across passes |
| Local consistency | Short-window relative motion | **Relative Pose Error (RPE)** |

**Accuracy is explicitly non-uniform**: a camera reached by a short trajectory is essentially as good as its ChArUco handshake; a camera at the end of a long path inherits GLIM's accumulated drift. Each camera's bound is tied to its trajectory-segment length — this is published, not hidden behind an average.

### 6.2 Radar-camera extrinsics

| GT product | Derivation | Reported uncertainty |
|---|---|---|
| `T_cam_radar` per radar | Measurement-space MLE (range/az/el, σ-weighted, Huber, MAP offset) | **Per-DOF covariance** → rotation 1σ (deg) and translation 1σ (mm) |
| Systematic accuracy | Signed bias per axis | few-mm bias on both finalized radars |
| Independent cross-check | Shared rig **apex offset** recovered by each radar independently | **In-plane agreement within combined 1σ** (radar1 vs radar2: X 256 vs 250 mm, Y 539 vs 544 mm) |
| Fused track quality | Anisotropic-covariance Kalman fusion of radar1+radar2 | **Fused 1σ ≈ [53, 69, 29] mm** |

A crucial honesty note carried from the calibration procedure: **large per-axis 3-D RMS on a radar's soft axis is random angular noise, not calibration error** — judge by signed bias, per-DOF 1σ, and live overlay, and let it average out in fusion. The dataset documents which axis is soft for each radar so users weight it correctly.

### 6.3 What "independent" buys the benchmark

Because camera GT error originates in **pixel detection** and radar GT error in **per-detection angular noise**, both are **independent of any downstream system that consumes tracks or maps**. A perception system evaluated against this GT fails *differently* from the GT, so the benchmark can expose the system's errors instead of correlating with them. This independence is the reason the uncertainty model is a contribution and not just bookkeeping.

## 7. Justification of Approaches (and honest novelty)

- **GLIM + ChArUco instead of motion capture.** No survey-grade or mocap system is assumed. The design gets drift-free metric anchors from boards and a continuous carrier from SLAM, and *certifies* the result internally from residuals and loop closure. This is cheaper, deployable in the actual operating space, and — importantly — yields *stated bounds* rather than an unverifiable claim.
- **Measurement-space radar MLE instead of isotropic Kabsch.** Radar error is anisotropic (precise range, coarse angle, range-growing cross-range error). Cartesian least-squares mis-weights this and is biased toward the noisy geometry; measurement-space σ-weighted MLE trusts range far more than angle. Monte-Carlo and live re-solves in the source records show a large margin (≈1.6× rotation, ≈3× translation over Kabsch).
- **Measurement-first priors instead of a no-prior bootstrap.** A single-reflector radar has one translation axis it physically cannot observe; no pose set fixes it. A tape-measured prior anchors that blind axis, while a uniform moderate prior width is automatically overridden on the observable axes — so the operator never has to diagnose which axis is blind.
- **Joint MAP of the rig apex offset.** A fixed hand-measured target geometry silently biases every pose; making it a MAP-refined free parameter both removes that bias and yields a shared physical invariant for cross-radar validation.
- **Orthogonal radar mounting + anisotropic fusion.** Perpendicular soft axes mean each radar contributes its sharp axis; the fused covariance is tight in every direction — a deliberate design choice the dataset lets others reproduce and beat.
- **Passive + active + latency Wi-Fi triad.** Passive monitoring is safe to run continuously and captures link *state*; active iperf captures true *capacity* but saturates the link and is confined to survey passes; ping captures *responsiveness* cheaply. Together they give link *quality, capacity, and responsiveness* on one time base — no single tool does.
- **Recording locally, not over the measured link.** Bags are written to the robot's own disk so data is not lost precisely when the link (the thing under study) degrades.

**Honest novelty stance (carried verbatim in spirit from the calibration doc).** Individually, the building blocks have precedent. This dataset does **not** claim a new calibration paradigm. What is defensibly new is (a) the **modality union** — camera + radar + RF connectivity in one indoor infrastructure frame — which the survey in §5 finds unoccupied; (b) the **uncertainty-as-metadata** ground-truth discipline applied across all modalities; and (c) the **specific calibration engineering combination** in §5.3. Any stronger claim (e.g. "first ever") is scoped to the literature check performed and should be re-verified with a formal prior-art search before a paper or patent.

## 8. Benchmark Suite

Each task lists **inputs → ground truth → metrics → provided baseline**. Tasks are designed so the uncertainty model in §6 is usable (e.g. metrics reported relative to the per-camera bound).

### B1 — Radar↔Camera extrinsic calibration
- **Inputs:** synchronized radar point clouds + camera frames of the ChArUco+reflector rig.
- **GT:** finalized `T_cam_radar` for radar1/radar2 with per-DOF covariance; independent apex-offset invariant.
- **Metrics:** rotation geodesic error, translation error, per-DOF within-1σ rate, apex-offset agreement; LOO-CV residual.
- **Baseline:** the measurement-space MLE estimator; isotropic Kabsch as the lower bound.

### B2 — Multi-view 3D person localization & tracking
- **Inputs:** synchronized multi-view infrastructure camera streams.
- **GT:** world-frame camera poses (with per-camera bounds); radar group-tracker tracks as an auxiliary cross-modal reference.
- **Metrics:** 3D MOTA/MOTP, IDF1, localization error **reported against the per-camera pose bound** (so a method is not penalized below GT uncertainty).
- **Baseline:** multi-view triangulation from 2D detections through the released extrinsics.

### B3 — Radar-camera fusion for detection & tracking
- **Inputs:** radar1+radar2 detections + camera.
- **GT:** fused-track reference (anisotropic Kalman, ≈[53,69,29] mm) and camera-derived tracks.
- **Metrics:** track accuracy, jitter (frame-to-frame), covariance calibration (are predicted σ's honest?).
- **Baseline:** the provided anisotropic-covariance constant-velocity fusion node.

### B4 — Privacy-preserving radar-only sensing vs. camera
- **Inputs:** radar-only streams.
- **GT:** camera-derived person tracks (as the higher-fidelity reference).
- **Metrics:** detection/counting accuracy, localization error, degradation vs. camera; occlusion/low-light robustness.
- **Baseline:** radar group-tracker + fusion; camera tracker as the reference ceiling.

### B5 — Wi-Fi / RF coverage mapping & prediction
- **Inputs:** time-joined RSSI/SNR/throughput/latency/loss + pose.
- **GT:** dense measured coverage on held-out survey passes.
- **Metrics:** map RMSE / MAE on held-out locations, dead-zone detection F1, calibration of predictive variance.
- **Baseline:** Gaussian-process / kriging interpolation of a radio map from pose-tagged samples.

### B6 — Connectivity-aware perception *(flagship)*
- **Inputs:** perception streams **plus** the co-registered link-quality field.
- **GT:** perception GT (B2/B3) + measured throughput/latency/loss along the trajectory.
- **Tasks & metrics:**
  - *Adaptive offloading:* choose on-device vs. offloaded inference per location; score end-to-end accuracy **under the real measured link budget** (accuracy achieved per byte / per ms of latency).
  - *Graceful degradation:* accuracy retained as the link drops (measured, not simulated).
  - *Coverage-aware sensor/relay placement:* place a sensor/relay to maximize joint perception+connectivity coverage; score against measured maps.
- **Baseline:** a link-agnostic always-offload policy vs. a link-aware policy driven by the B5 coverage map — quantifying the value of knowing the radio.

### B7 — Calibration robustness & drift re-estimation
- **Inputs:** sequences with induced mount perturbation / long-trajectory cameras / re-visits.
- **GT:** before/after extrinsics with bounds; loop-closure drift.
- **Metrics:** re-estimation accuracy, drift-detection latency, degradation vs. trajectory length.
- **Baseline:** re-run the calibration pipelines; report per-camera accuracy vs. path length.

## 9. Limitations and Honest Scope

- **radar_infra is not finalized.** It has only a round-1 bootstrap (10 poses; rotation 1σ ≈ 7°, translation 1σ ≈ 80–106 mm) and **must not be treated as deployable**. It will be released as *provisional* with a clear flag, or completed before the dataset freezes (round-2 target: rot 1σ ≲ 3°, t 1σ ≲ 40 mm, all diversity bars green, session logging enabled).
- **Camera resolution bottleneck.** Sessions at 960×540 degrade ChArUco detection past ~0.4–1 m; this bounds handshake range and is documented per sequence.
- **Non-uniform camera-pose accuracy.** Distant cameras inherit GLIM drift; this is a *feature of the honest GT*, not a defect, but users must weight per-camera bounds accordingly.
- **Radar soft axes.** Each radar has a physically weak axis; single-radar localization on that axis is noisy by hardware design. The dataset documents which axis and relies on fusion to constrain it.
- **Novelty is scoped to a literature check**, not a formal prior-art/patent search (§7). The "first to combine" claim in §3 must be re-verified before formal publication.
- **Wi-Fi coverage is site- and hardware-specific.** Radio maps depend on the AP placement, chipset (some report no noise floor → no true SNR), and environment; generalization across sites is itself a research question, not an assumption.

## 10. Data Format, Splits, and Release (proposed)

- **Container:** ROS 2 bags (native), with exported per-modality files (images + camera_info, radar point clouds with x/y/z/doppler/intensity, Wi-Fi/iperf/ping messages) and a static transform tree.
- **Ground-truth package:** extrinsic YAML/JSON per sensor + `*_session.json` reproducibility records + a machine-readable **uncertainty manifest** (per-camera bound, per-radar covariance, loop-closure drift, apex-offset agreement).
- **Splits:** by session/site and by task; held-out survey passes reserved for B5/B6; a calibration-only split for B1/B7.
- **Tooling:** the calibration pipelines (`radar_camera_calib*`, `general_charuco`) and the Wi-Fi monitor stack shipped so ground truth is auditable and reproducible.
- **Licensing / ethics:** indoor human subjects imply consent and privacy handling; radar-only tracks are highlighted as the privacy-preserving alternative.

## 11. Open Questions to Resolve Before Freeze

1. Finalize radar_infra (round 2) or ship it flagged as provisional.
2. Confirm the §5 novelty gap with the completed related-work survey (agents in progress).
3. Decide camera resolution for the recording (raise above 960×540 if handshake range matters for the target tasks).
4. Fix the site/AP topology for the Wi-Fi survey so B5/B6 have a well-defined radio environment.
5. Define exact synchronization guarantees across the three modalities (clock domains, max sync dt) and record them per sequence.

---

### Appendix A — Source documents
- *Multi-Camera Extrinsic Calibration — GLIM-Anchored, ChArUco-Certified Procedure.*
- *ROS 2 Wi-Fi Monitor — System Documentation.*
- *Standardized Radar↔Camera Calibration Procedure (IWR6843ISK ×3 ↔ ZED).*
- Repository module `radar_camera_calibration/` (README, SPRINT_UPDATE, session records).

### Appendix B — Finalized extrinsic snapshots (from session records, 2026-07-22)
- **radar1:** t = [+0.2368, +0.0190, −0.0542] m (|t|=24.4 cm), range-scale 0.958, rot 1σ [4.85, 3.62, 4.19]°, t 1σ [27.3, 39.8, 30.5] mm, soft axis vertical.
- **radar2:** t = [−0.1194, −0.0096, −0.0157] m (|t|=12.1 cm), range-scale 0.967, rot 1σ [3.06, 4.08, 3.58]°, t 1σ [32.7, 29.1, 23.0] mm, soft axis horizontal.
- **radar_infra:** round-1 bootstrap only — **not deployable**.
- **Apex-offset cross-check:** radar1 vs radar2 in-plane X 256/250 mm, Y 539/544 mm (match); Z −20/−55 mm (weak axis on both, expected).
