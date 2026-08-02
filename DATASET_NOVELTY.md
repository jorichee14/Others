# A Co-Calibrated Infrastructure + Mobile-Robot Sensing Dataset for Cooperative Perception Under Spatially-Varying Connectivity

*Research direction, motivation, contributions, related work, ground-truth model, and benchmark suite.*

> **Status:** Draft research-direction document for a dataset currently being recorded. Related-work citations were web-verified during drafting (see §5 and Appendix C); a handful of author-list / volume-issue details behind paywalled or proxy-blocked hosts remain flagged in Appendix C and need a final check before external publication. Numeric ground-truth values quoted here are taken from the calibration session records referenced in the appendix and are subject to update as recording completes.

---

## 0. TL;DR

We are recording an **indoor, multi-modal sensing dataset with a hybrid infrastructure-plus-mobile-agent topology**:

- **Fixed infrastructure** — a **multi-view camera network** and a fixed infrastructure radar (`radar_infra`, TI IWR6843ISK) that watch a shared indoor space.
- **Two operational mobile robots** that move through that space:
  - **Robot A (fully instrumented ego agent)** — ZED stereo camera + two mmWave radars (radar1/radar2, fused) + a **Wi-Fi/RF link-quality** monitor, with **GLIM** (LiDAR-inertial SLAM) providing its globally-optimised trajectory.
  - **Robot B (lightweight secondary agent)** — an **RGBD camera only**; its ground-truth pose comes from **point-cloud registration** (RGBD depth against the LiDAR reference map) or **fiducial markers**, not GLIM.

Everything is expressed in a **single metric world frame** and shipped with **independently-bounded, per-sensor ground-truth uncertainty**. Two features distinguish it:

1. **Perception coupled to communication.** Robot A measures Wi-Fi link quality along its path, co-registered to the same frame the perception lives in — so one can study how perception quality and wireless connectivity co-vary through space, and how a system should behave when the link it depends on degrades exactly where perception is hardest.
2. **Certified, independent *dynamic* ground truth.** The robots *are* the tracked targets, and their true positions are known by methods that fail **differently** from the cameras and radars being evaluated — Robot A by LiDAR-inertial GLIM, Robot B by point-cloud matching / fiducials. This makes the dataset a self-contained **ego ↔ infrastructure cooperative-perception** benchmark: an indoor, radar- and RF-inclusive analogue to vehicle-infrastructure datasets like DAIR-V2X.

---

## 1. Introduction

Most multi-sensor perception datasets are built for one of two worlds. The first is **automotive**: a vehicle drives through the world carrying camera, LiDAR, and (increasingly) radar, and the research question is ego-centric perception in motion (nuScenes, View-of-Delft, K-Radar, TJ4DRadSet, RADIATE). The second, newer world is **roadside / infrastructure**: sensors are bolted to poles or gantries over an intersection, and the research question is bird's-eye monitoring and vehicle-to-infrastructure cooperation (DAIR-V2X, Rope3D, IPS300+, TUMTraf/A9).

Both worlds share three limits that this dataset is designed to escape:

1. **They are outdoor and vehicle/traffic-centric.** Indoor infrastructure perception — the setting for warehouses, hospitals, factories, and service robots — is comparatively under-served by *radar-inclusive* multi-view datasets.
2. **They treat ground truth as an oracle.** Poses and boxes are published as "truth" with, at best, a single aggregate accuracy figure. When the ground truth's errors are correlated with the system under evaluation, the benchmark flatters that system instead of exposing it.
3. **They ignore the network.** A perception system deployed on real infrastructure runs on a *wireless link* whose quality varies dramatically through space. No mainstream perception dataset records where the link is good, where it collapses, and how that correlates with where perception itself is hard.

This dataset attacks all three. It is **indoor**; it carries **per-sensor, independently-bounded uncertainty as first-class metadata**; and it adds a **co-located, spatially-registered Wi-Fi/RF modality** so that perception and connectivity can be studied together for the first time in one frame.

It also **does not stop at fixed infrastructure**. Two mobile robots operate inside the instrumented space — Robot A with a full ego suite (radar + camera + Wi-Fi) and a GLIM-certified trajectory, and Robot B a lighter RGBD-only agent whose pose is recovered by point-cloud matching or fiducials. This gives the dataset a **cooperative, multi-viewpoint** character that a purely fixed rig cannot: the infrastructure sees both robots; Robot A senses the space (and Robot B); and every viewpoint is expressed in one world frame. Both robots' certified trajectories double as **dynamic ground truth** for the infrastructure's tracking of moving objects — the indoor, radar+RF counterpart to the vehicle-infrastructure cooperative paradigm (DAIR-V2X, §5.2), which no existing dataset provides indoors with radar and connectivity.

## 2. Motivation

### 2.1 Why indoor infrastructure perception

Fixed cameras and radars watching a shared indoor space are the sensing substrate for a growing class of deployments — automated warehouses, elder-care facilities, smart manufacturing cells, retail analytics — where the sensors do not move but the agents they watch do. These deployments need **metric, multi-view, privacy-aware** perception, and they need it to keep working through occlusion and poor lighting. That is exactly where **mmWave radar complements cameras**: radar sees through darkness and smoke, measures radial velocity directly (Doppler), and is inherently privacy-preserving because it does not form a recognizable image of a person. A dataset that co-registers fixed cameras and fixed radars indoors, with certified extrinsics, is the missing substrate for benchmarking this class of system.

### 2.2 Why connectivity belongs in a perception dataset

Infrastructure perception is almost never fully on-device. Frames are streamed to an edge server; tracks are fused in a central node; a robot offloads heavy inference over Wi-Fi. **The wireless link is part of the perception pipeline** — and it is the least reliable part. Link quality varies by tens of dB across a single room, throughput can collapse behind a metal shelf, and dead zones are common precisely in the cluttered areas where perception matters most.

Today a researcher who wants to study *connectivity-aware perception* — adaptive offloading, graceful degradation, coverage-aware sensor placement, communication-aware planning — has no dataset that provides *perception* and *link quality* in the same spatial frame. Indoor RF datasets that include radar and cameras (MM-Fi, XRF55) use Wi-Fi CSI as a sensing signal, not as a communications link measurement, and robot radio-mapping datasets (Milosheski et al. 2026) pair RSSI with LiDAR only — no imaging perception and no active throughput/latency (§5.5). So today one must either simulate the radio or ignore it. This dataset removes that gap: every Wi-Fi/RF sample is timestamped and pose-registrable, so a coverage/throughput/latency field can be built over the exact space the cameras and radars observe.

### 2.3 Why honest, bounded ground truth

The calibration procedures underpinning this dataset were deliberately designed around one principle, stated in the source documents and adopted here as a dataset-wide commitment:

> Ground truth is an **estimate**, not an oracle. Its value comes from having error that is **bounded**, **per-sensor**, and — critically — **independent of the system it is later used to evaluate**, so the two fail differently and the ground truth can expose the system's mistakes rather than flatter them.

Concretely, camera world-poses come from ChArUco handshakes (bounded, drift-free, mm-level) carried by a GLIM trajectory (drift-prone, growing with path length), and each camera's pose ships with a residual bound tied to the trajectory length that fed it. Radar-camera extrinsics come with **per-DOF covariance and observability**, and are cross-validated by an independent physical invariant (the shared rig apex offset). This turns "ground truth" from a number into a **distribution with a stated shape**, which is what makes uncertainty-aware benchmarking possible.

## 3. Contributions

1. **A single-frame, hybrid infrastructure + mobile-agent dataset.** Multi-view fixed cameras and a fixed radar, plus **two mobile robots** — Robot A with ego radar+camera perception and a Wi-Fi/RF monitor, Robot B a lighter RGBD-only agent — all co-registered into one metric world frame with a published transform tree.
2. **The first dataset to make the communication budget a *measured, pose-tied variable* beside independent perception.** It co-registers multi-view radar+camera perception with **end-to-end measured link quality** (RSSI/SNR + iperf throughput + ping latency/loss) in one indoor metric frame. The closest real perception+RF datasets sense *with* the channel (DeepSense 6G; LuViRA — beam power / CSI as the target), and the bandwidth-aware collaborative-perception line (Where2comm, DiscoNet …) trades accuracy against an *abstract, simulated* byte budget; none pairs independent perception with a **measured** channel (§5.5–§5.6). This turns "communication cost" from a bit-count proxy into a real, location-dependent quantity.
3. **Certified, independent *dynamic* ground truth from the mobile robots.** Each robot's trajectory gives a bounded estimate of where it actually was — Robot A from GLIM (LiDAR-inertial, loop-closure-checked), Robot B from RGBD point-cloud registration / fiducials — a moving-target GT for the infrastructure's and Robot A's tracking that is produced by a **different modality** than the systems under test, so it fails differently and can expose their errors.
4. **An indoor ego ↔ infrastructure cooperative-perception benchmark.** Fixed infrastructure and two moving agents observe the same scene from different viewpoints in one frame — the radar- and RF-inclusive indoor analogue of vehicle-infrastructure cooperative datasets (DAIR-V2X), enabling cross-view fusion, handoff, and cooperative tracking.
5. **Uncertainty-as-metadata ground truth.** Every ground-truth pose carries an independent, per-sensor error bound with a stated derivation (ChArUco reprojection residual, GLIM loop-closure drift, radar per-DOF covariance), not a single blanket accuracy claim.
6. **A reproducible, measurement-first calibration methodology** for both camera-network and radar-camera extrinsics, released with the data so the ground truth is auditable rather than asserted.
7. **A curated, tiered benchmark suite** headlining one **flagship** task — connectivity-aware cooperative perception under a *measured* channel (F1) — plus two signature tasks (uncertainty-aware cross-view tracking; geometry-fused connectivity mapping) and a compact enabling tier (calibration, anisotropic fusion, privacy-preserving radar, re-localization). Selection is deliberate: we headline only what no other public dataset can support (§8).
8. **An anisotropic-covariance radar fusion baseline** with measured performance (fused 1σ ≈ [53, 69, 29] mm vs. ≈[112, 325] / [287, 112] mm single-radar), demonstrating that the released extrinsics and uncertainty model are usable end-to-end.

## 4. Background: the sensing stack

### 4.0 Platforms and frames

The dataset spans **fixed infrastructure** and **two mobile robots**, all tied into one world frame:

| Platform | Mobility | Sensors it carries | Role |
|---|---|---|---|
| **Infrastructure camera network** (*N* cameras) | Fixed | RGB (optionally depth) | Multi-view overhead perception; world-frame extrinsics are the calibration target |
| **`radar_infra`** (IWR6843ISK) | Fixed | 1 mmWave radar | Infrastructure-side radar perception *(calibration pending — §8)* |
| **Robot A** (fully instrumented ego agent) | Mobile | ZED stereo + radar1 + radar2 (fused) + Wi-Fi/RF monitor + odometry; **GLIM** onboard | Ego perception + connectivity survey; GLIM trajectory = its pose GT |
| **Robot B** (lightweight secondary agent) | Mobile | **RGBD camera only** | Second moving agent + secondary RGBD viewpoint; pose GT via point-cloud matching or fiducials |

Three consequences drive the rest of the design:

- **Radar1 and radar2 are ego (Robot-A-mounted) and fused on the robot; `radar_infra` is the fixed one.** The finalized extrinsics in Appendix B are the **ego** ZED↔radar1/radar2 transforms; `radar_infra`'s extrinsic into the world frame is the outstanding calibration item.
- **Only Robot A carries radar and the Wi-Fi monitor.** So the ego radar-camera fusion (E2), radar-only privacy sensing (E3), and connectivity tasks (F1/S2) run from Robot A; Robot B contributes an RGBD viewpoint and a second moving target.
- **Both robots' trajectories are certified, bounded, and independent** of the camera/radar perception being evaluated — Robot A via GLIM (loop-closure residual), Robot B via point-cloud registration / fiducials (§6.3). So the robots serve simultaneously as *sensing agents* and as *moving ground-truth targets* for the infrastructure and for each other.

### 4.1 Multi-view infrastructure camera network

*N* fixed RGB cameras (optionally depth) observe a shared indoor space. Their extrinsics are recovered in a common world frame by the **GLIM-anchored, ChArUco-certified** procedure:

- A **reference camera** (carried by a mobile robot, which runs **GLIM** onboard — the same platform that later operates as an ego agent) moves through the space; GLIM (LiDAR-inertial SLAM, optionally visual) estimates its continuous trajectory.
- At each fixed camera, a **ChArUco board handshake** gives a direct, drift-free, mm-level relative pose between the reference camera and that fixed camera at a shared instant.
- A fixed **origin board** defines the world frame; **revisits** to it measure and correct GLIM drift, and a **final loop closure** reports total trajectory drift.
- World pose of a fixed camera: `T_world_infra = T_world_ref(t) · T_ref_infra`, where the handshake board cancels out of the relative pose and `T_world_ref(t)` is interpolated from the globally-optimised trajectory (SLERP for rotation, lerp for translation).

The key property: **the network is board-defined, not drift-defined**, and accuracy is reported *per camera*, tied to the trajectory-segment length feeding it.

### 4.2 mmWave radar (ego-fused + infrastructure)

Three TI IWR6843ISK mmWave radars are calibrated to the ZED left camera by a **ChArUco + trihedral-corner-reflector rig**. **radar1 and radar2 are the robot-mounted (ego) pair, fused on the robot in the ZED camera frame; `radar_infra` is the fixed infrastructure radar.** The estimator is **measurement-first**: measure the rig apex offset and a tape-measured extrinsic prior, seed them as Bayesian priors, collect one diverse pose set, and solve by **maximum likelihood in the radar's native (range, azimuth, elevation) space**, weighted by each axis's real sensor σ, with Huber robust loss, σ-gated outlier rejection, and joint MAP refinement of the rig offset.

Radar noise is **anisotropic and range-dependent**: range is precise (≈cm), angle is coarse (degrees), and cross-range error grows as ≈ range·σ_az. The ego pair radar1/radar2 are mounted with **orthogonal soft axes** (radar2 rolled ~90°), so their weak directions are perpendicular and fusion constrains every axis. `radar_infra`'s calibration into the world frame is not yet finalized (see §8).

### 4.3 Wi-Fi / RF link-quality suite (robot-mounted)

The Wi-Fi/RF suite runs **on the mobile robots**, measuring each robot's own link to the access point as it moves — so coverage is sampled by the moving agents along their GLIM-certified paths. Three independent ROS 2 nodes sweep three dimensions of the wireless link:

- **Passive RF monitor** (up to ~5 Hz): RSSI, SNR (when a real noise floor is reported), negotiated rate, MCS/NSS/width, retries/failures, channel utilization, error counters — read-only, no traffic injected, NaN/−1 for genuinely unknown values.
- **Active throughput monitor** (iperf3, ~1 Hz continuous or periodic bursts): achievable goodput, retransmits, and — in continuous mode — kernel-socket TCP RTT from the loaded connection.
- **Latency/loss monitor** (ping, 1 Hz, cheap enough to run during live operation): per-ping RTT and rolling-window loss.

Every message is timestamped for **offline time-join to pose**, producing coverage / throughput / latency / dead-zone maps over the observed space.

## 5. Related Work

*The claims of novelty in §3 rest on this survey. Entries are grouped by theme; the gap this dataset fills is stated at the end of each group.*

### 5.1 Automotive radar-camera (and radar-LiDAR-camera) datasets

The dominant line of radar-inclusive perception datasets is vehicle-mounted and outdoor:

- **nuScenes** (Caesar et al., CVPR 2020) — 6 cameras, **5 automotive radars** (Continental ARS408, 2D/Doppler, no elevation), 1 LiDAR; the reference multimodal AD benchmark, 1000 scenes.
- **View-of-Delft (VoD)** (Palffy et al., *RA-L* 7(2):4961–4968, 2022) — 3+1D (4D) radar + stereo camera + 64-layer LiDAR, urban VRU detection.
- **TJ4DRadSet** (Zheng et al., ITSC 2022) — 4D radar + camera + LiDAR with tracking IDs, 7757 frames.
- **K-Radar** (Paek et al., NeurIPS D&B 2022) — dense **4D radar *tensor*** (range/az/el/Doppler power) + LiDAR + camera, adverse weather.
- **RADIATE** (Sheeny et al., ICRA 2021) — Navtech 360° **scanning radar** + camera + LiDAR in fog/rain/snow/night.
- **aiMotive** (Matuszka et al., ICLR-W 2023) — 360° long-range camera + LiDAR + radar for highway perception.
- **CRUW** (Wang et al., WACV 2021) — camera + raw radar RA heatmaps, no LiDAR.
- **RadarScenes** (Schumann et al., **FUSION 2021**), **CARRADA** (Ouaknine et al., ICPR 2020/21), **Astyx HiRes2019** (Meyer & Kuschk, EuRAD 2019), **Oxford Radar RobotCar** (Barnes et al., ICRA 2020, scanning-radar odometry), **Dual Radar** (Zhang et al., *Sci. Data* 2025, two 4D radars) — radar-focused sets of varying modality richness.

*Gap:* all are vehicle-mounted, outdoor, ego-motion; none are fixed-infrastructure indoor, and none record an RF-connectivity/link-quality modality.

### 5.2 Roadside / infrastructure perception datasets

The closest existing work in *fixed-sensor* spirit is roadside/ITS:

- **DAIR-V2X** (Yu et al., CVPR 2022) — the first large vehicle-infrastructure-cooperative dataset; infrastructure side has camera + LiDAR. V2X communication is the *application*, but no RF link-quality is recorded as a data modality.
- **Rope3D** (Ye et al., CVPR 2022) — roadside monocular 3D detection (camera, LiDAR-derived GT).
- **IPS300+** (Wang et al., ICRA 2022) — dense roadside intersection perception (camera + 80-layer LiDAR).
- **A9-Dataset** (Creß et al., IEEE IV 2022) — gantry-mounted highway sensing that **does include radar** alongside cameras and LiDARs; and **TUMTraf Intersection** (Zimmer et al., ITSC 2023) — the urban camera+LiDAR intersection set (no radar).
- **LUMPI** (Busch et al., IEEE IV 2022) — multi-perspective (up to 5 LiDARs + 3 cameras) intersection dataset.

*Gap:* these establish the fixed-infrastructure and **vehicle-infrastructure cooperative** paradigm (DAIR-V2X) but are **outdoor traffic**; radar appears only in the A9 highway set, none are indoor, and none record a Wi-Fi/RF link-quality modality. Our hybrid topology — fixed infrastructure plus two mobile agents in one indoor frame — is the radar- and RF-inclusive **indoor** counterpart to that cooperative paradigm (cross-view fusion/handoff appears in tasks F1/S1). The V2X benchmark suites and the bandwidth-aware collaborative-perception line are treated in detail in §5.6.

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

- **RSSI/CSI indoor localization** — RADAR (Bahl & Padmanabhan, INFOCOM 2000), Horus (Youssef & Agrawala, MobiSys 2005), SpotFi (Kotaru et al., SIGCOMM 2015); survey: Ma, Zhou & Wang, *WiFi Sensing with CSI: A Survey*, ACM CSUR 52(3), 2019.
- **Radio maps / REM and Gaussian-process signal fields** — Ferris, Hähnel & Fox (*GP for signal-strength location estimation*, RSS 2006) introduced GP models of the RSS field with predictive uncertainty; Fink & Kumar (*Online radio signal mapping with mobile robots*, ICRA 2010) is the classic robot RF-mapping reference; GP+path-loss hybrids bound variance far from samples.
- **Communication-/connectivity-aware motion planning** — Yan & Mostofi (*Communication-aware motion planning*, IEEE TAC / TWC 2013); best review: Muralidharan & Mostofi, *Communication-Aware Robotics*, Annual Review of Control, Robotics, and Autonomous Systems 4:115–139, 2021; resilient-connectivity planners (RCAMP, IROS 2017).
- **Computation offloading under location-varying bandwidth** — edge/cloud robotics where offload decisions track link quality (recent works tie cloud-inference offloading to spatially heterogeneous connectivity; edge-robotics surveys, 2025).

**Indoor RF-sensing datasets that DO combine radar + camera + Wi-Fi — and why they are not the same thing.** Two indoor human-sensing datasets already pair mmWave radar, camera, and Wi-Fi:
- **MM-Fi** (Yang et al., NeurIPS D&B 2023) — RGB, depth, LiDAR, mmWave radar point cloud, **and Wi-Fi CSI**, 40 subjects, fixed indoor rig.
- **XRF55** (Wang et al., ACM IMWUT 8(1), 2024) — 9 Wi-Fi CSI links + mmWave radar + RFID + Azure Kinect RGB-D, 39 subjects.

Crucially, in both, **Wi-Fi is a *sensing signal*** — CSI subcarrier amplitude/phase treated as an imaging modality for human activity/pose recognition — **not a *communications link-quality* measurement**. Neither records throughput, latency, loss, RSSI/SNR-as-link-health co-registered to the scene. The nearest thing to a link-quality-plus-perception dataset is **Milosheski et al.** (*multimodal indoor radio mapping with 3D point clouds and RSSI*, Data in Brief / arXiv:2511.00494, 2026) — but that is **LiDAR point clouds + passive RSSI only**: no camera/radar imaging perception and no active throughput/latency.

- **Contrast for privacy framing:** Wi-Fi CSI human-sensing datasets (Widar3.0, MobiSys 2019; SignFi, IMWUT 2018; UT-HAR; SenseFi benchmark, *Patterns* 2023) recognize activity from RF echoes; we instead offer **radar-based** privacy-preserving sensing and treat Wi-Fi purely as the communications link.

*Gap (the central novelty claim, now precisely scoped):* **no public dataset pairs camera/radar *perception* with a co-located Wi-Fi/RF *link-quality* modality (RSSI/SNR + active throughput + latency/loss) time-joined to pose in one metric frame.** Datasets that combine radar+camera+Wi-Fi treat Wi-Fi as a sensing signal (MM-Fi, XRF55); the one dataset that treats RF as link quality alongside perception uses passive RSSI + LiDAR only (Milosheski 2026). This dataset fills that specific gap.

### 5.6 The closest work: perception × communication

This is where the dataset's central claim must survive contact with the literature. Two lines of work approach the perception–communication intersection from opposite sides, and **each misses on a different axis**:

**(a) Real datasets that co-locate perception and RF — but treat the channel as the *sensing target*, not a measured link.**
- **DeepSense 6G** (Alkhateeb et al., *IEEE Comms Mag.* 61(9), 2023; arXiv:2211.09769) — the single closest real dataset: co-located camera, LiDAR, **radar**, GPS, and mmWave (28/60 GHz) wireless, outdoor vehicular. But its "communication" signal is **mmWave beam / received power used as the prediction target** for sensing-aided communication (beam/blockage prediction) — the channel *is* the thing being predicted, not an independently measured link quality logged beside perception. Single Tx–Rx link; no throughput/latency/loss; no certified GT uncertainty.
- **LuViRA** (Yaman et al., ICRA 2024; arXiv:2302.05309) — the closest **indoor** cousin: robot-mounted camera + depth + IMU + **5G massive-MIMO channel** + audio, with 0.5 mm 6-DOF pose GT. But the radio is **CSI/channel response for localization**, single-agent, radar-free, and carries no link-quality (throughput/latency/loss/RSSI-as-health) measurement.
- **ISAC / RF-sensing sets** (mmHSense, DISC) sense *using* the comm waveform (CIR → activity/gait); no independent perception task, no link-quality co-registration.

**(b) Cooperative-perception benchmarks that model the communication budget — but never measure a channel.**
- **The V2X family** — OPV2V (R. Xu et al., ICRA 2022), V2X-Sim (Y. Li et al., RA-L 2022, RSU+vehicles — the closest simulated infrastructure↔agent analogue), V2V4Real (R. Xu et al., CVPR 2023), DAIR-V2X / V2X-Seq (Yu et al., CVPR 2022/2023), TUMTraf V2X (Zimmer et al., CVPR 2024) — define the cooperative-perception paradigm we bring indoors, but are **outdoor AD, LiDAR/camera, with communication idealized or simulated**.
- **Bandwidth-aware collaborative perception** — Where2comm (Hu et al., NeurIPS 2022), When2com (Liu et al., CVPR 2020), Who2com (Liu et al., ICRA 2020), DiscoNet (Y. Li et al., NeurIPS 2021), CoSDH (Xu et al., CVPR 2025) — the **direct methodological competitors to our flagship task**. They report **accuracy vs. communication-volume** curves, deciding *what/when/whom* to communicate under a budget. But that budget is an **abstract byte count over an idealized channel**; none ties it to a measured throughput/latency/loss.
- **Real-link edge perception** is essentially a tooling gap: PEERNet (Narayanan et al., IROS 2024) profiles end-to-end latency of networked robotics over real hardware, but it is instrumentation, not a released co-registered perception+GT dataset.

**Where this dataset sits.** It is the first to make the communication budget a **measured, pose-tied variable** sitting next to independent perception: co-registering multi-view perception (fixed infrastructure + mobile radar/camera) with **end-to-end measured link quality** (iperf throughput, ping latency/loss, RSSI/SNR) in one indoor metric frame, with **certified per-sensor and dynamic GT uncertainty**. DeepSense 6G and LuViRA have the *real-RF-with-perception* half but sense with the channel; the Where2comm/DiscoNet line has the *perception-under-a-budget* half but simulates the channel. This dataset is what lets those methods be evaluated against a **real** channel rather than a bit-count proxy — the precise wedge, and the substrate for benchmark F1 below.

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

### 6.3 Mobile-robot trajectories as dynamic ground truth

| Robot | GT trajectory `T_world_robot(t)` | Reported uncertainty |
|---|---|---|
| **Robot A** | **GLIM** (LiDAR-inertial) final, globally-optimised, loop-closed trajectory | **Loop-closure residual** (drift bound), plus cross-pass RMSE / RPE as in §6.1 |
| **Robot B** | **RGBD point-cloud registration** against the LiDAR reference map, and/or **fiducial-marker** detection (markers on the robot seen by the calibrated infrastructure cameras, or environment markers seen by the robot) | Registration residual / fiducial reprojection residual — bounded and reported per pose |

This is the piece that makes dynamic evaluation possible without hand-labelling. When the infrastructure (or Robot A) tracks a moving agent, the agent's *true* position is that robot's own certified pose — a metric, timestamped, bounded reference. Crucially the GT is produced by a **different modality** than the tracker under test — LiDAR-inertial (Robot A) or depth-registration/fiducials (Robot B) versus the camera/radar systems being scored — so they fail differently (§6.4). Mounting a ChArUco/reflector target on each robot further ties the dynamic GT back to the same fiducial machinery as the static calibration and disambiguates the infrastructure's detection.

*Note:* Robot B's GT method (point-cloud matching vs. fiducials, or both) is a design decision still to be fixed (§11.7); point-cloud registration is drift-free but needs a good reference map and depth quality, while fiducials are drift-free and simple but require marker visibility.

### 6.4 What "independent" buys the benchmark

Because camera GT error originates in **pixel detection** and radar GT error in **per-detection angular noise**, both are **independent of any downstream system that consumes tracks or maps**. A perception system evaluated against this GT fails *differently* from the GT, so the benchmark can expose the system's errors instead of correlating with them. This independence is the reason the uncertainty model is a contribution and not just bookkeeping.

## 7. Justification of Approaches (and honest novelty)

- **GLIM + ChArUco instead of motion capture.** No survey-grade or mocap system is assumed. The design gets drift-free metric anchors from boards and a continuous carrier from SLAM, and *certifies* the result internally from residuals and loop closure. This is cheaper, deployable in the actual operating space, and — importantly — yields *stated bounds* rather than an unverifiable claim.
- **Measurement-space radar MLE instead of isotropic Kabsch.** Radar error is anisotropic (precise range, coarse angle, range-growing cross-range error). Cartesian least-squares mis-weights this and is biased toward the noisy geometry; measurement-space σ-weighted MLE trusts range far more than angle. Monte-Carlo and live re-solves in the source records show a large margin (≈1.6× rotation, ≈3× translation over Kabsch).
- **Measurement-first priors instead of a no-prior bootstrap.** A single-reflector radar has one translation axis it physically cannot observe; no pose set fixes it. A tape-measured prior anchors that blind axis, while a uniform moderate prior width is automatically overridden on the observable axes — so the operator never has to diagnose which axis is blind.
- **Joint MAP of the rig apex offset.** A fixed hand-measured target geometry silently biases every pose; making it a MAP-refined free parameter both removes that bias and yields a shared physical invariant for cross-radar validation.
- **Orthogonal radar mounting + anisotropic fusion.** Perpendicular soft axes mean each radar contributes its sharp axis; the fused covariance is tight in every direction — a deliberate design choice the dataset lets others reproduce and beat.
- **Passive + active + latency Wi-Fi triad.** Passive monitoring is safe to run continuously and captures link *state*; active iperf captures true *capacity* but saturates the link and is confined to survey passes; ping captures *responsiveness* cheaply. Together they give link *quality, capacity, and responsiveness* on one time base — no single tool does.
- **Recording locally, not over the measured link.** Bags are written to the robot's own disk so data is not lost precisely when the link (the thing under study) degrades.

**Honest novelty stance (carried in spirit from the calibration doc).** Individually, the building blocks have precedent. This dataset does **not** claim a new calibration paradigm, and it is **not** the first to put radar + camera + Wi-Fi in one indoor rig — MM-Fi and XRF55 already do, using Wi-Fi CSI as a *sensing* signal (§5.5). What is defensibly new is (a) the **specific modality union** — camera + radar perception + a Wi-Fi/RF *link-quality* (communications-performance) modality in one indoor infrastructure frame — which the survey in §5.5 finds unoccupied; (b) the **uncertainty-as-metadata** ground-truth discipline applied across all modalities; and (c) the **specific calibration engineering combination** in §5.3. Any stronger claim (e.g. "first ever") is scoped to the literature check performed (via web search, not a formal prior-art/patent search) and should be re-verified before a paper or patent — in particular, directly inspecting Milosheski et al. (2026) to confirm it carries no active throughput/latency.

## 8. Benchmark Suite

**Selection principle.** We do not benchmark everything the data *can* support; we headline only what **no other public dataset can**. A task earns *flagship* status only if it requires the dataset's capabilities *simultaneously* — independent perception **and** a measured link **and** certified independent dynamic GT **and** cooperative infra+ego viewpoints. Tasks the automotive / V2X / radio-map communities already serve well are demoted to *enabling*: they establish and validate the ground truth, but they are not the reason this dataset exists. The result is one flagship, two signature tasks, and a compact enabling tier.

Each task lists **inputs → ground truth → metrics → baseline**, with metrics stated relative to the §6 uncertainty bounds.

---

### Tier 1 — Flagship (unique to this dataset)

#### F1 — Connectivity-aware cooperative perception under a *measured* channel
The one task that only this dataset enables. It is the Where2comm/DiscoNet problem — cooperative perception under a communication budget — with the **abstract byte budget replaced by the real, pose-tied, measured link** (throughput/latency/loss/RSSI at each location), and with certified GT to score against.

- **Inputs:** cooperative perception streams — Robot A ego (ZED + radar1/radar2 fusion), Robot B RGBD, and infrastructure (cameras + `radar_infra`) — **plus** the co-registered Robot-A link-quality field over the space.
- **GT:** the observed robot's certified trajectory as moving-target GT (Robot A: GLIM; Robot B: point-cloud/fiducial — §6.3), plus per-viewpoint extrinsic bounds; the **measured** throughput/latency/loss along the path.
- **Tasks & metrics:**
  - *Bandwidth-constrained cooperative fusion:* decide *what/when/whom* to share between ego and infrastructure **under the measured link**; report the **accuracy-vs-measured-cost curve** (3D AP / MOTA against deliverable bytes and ms of latency at each pose) — the real-channel analogue of the AP-vs-communication-volume curves in Where2comm/DiscoNet.
  - *Adaptive offloading:* per-location on-device vs. offloaded inference, scored end-to-end **under the real link budget** (accuracy per byte / per ms).
  - *Graceful degradation:* accuracy retained as the link drops — **measured, not simulated (`netem`)**.
- **Baselines:** (i) a link-agnostic always-share/always-offload policy; (ii) a link-aware policy driven by the measured field; (iii) a bandwidth-aware method (e.g. a Where2comm-style selector) evaluated first on its assumed budget and then on the **measured** channel — quantifying the sim-to-real gap the dataset exposes.
- **Why it's flagship:** the entire bandwidth-aware collaborative-perception line (§5.6) evaluates on idealized channels; DeepSense 6G / LuViRA have real RF but sense *with* it. F1 needs measured-link + cooperative perception + certified GT at once — no other dataset has all three.

---

### Tier 2 — Signature (rare; showcases the certified-GT and cooperative capabilities)

#### S1 — Uncertainty-aware cross-view 3D tracking with independent dynamic GT
Highlights the **certified, independent, hand-label-free dynamic GT**: the robots are the moving targets, localized by a modality that fails differently from the trackers under test.
- **Inputs:** multi-view infrastructure cameras + `radar_infra`, and each robot as a moving target (optionally the other robot's ego view).
- **GT:** each robot's certified trajectory (§6.3); per-camera pose bounds.
- **Metrics:** 3D MOTA/MOTP, IDF1, and localization error **reported relative to the GT bound** (no method penalized below GT uncertainty); **handoff** continuity across camera FoV boundaries / occlusion (ID consistency, gap-bridging error).
- **Baseline:** multi-view triangulation through the released extrinsics; late-fused ego+infrastructure tracking vs. single-viewpoint, quantifying the cooperative gain.
- **Why it's signature:** cooperative 3D tracking exists in the V2X family, but outdoor, with hand-labelled GT and simulated comms; here it is indoor, radar-inclusive, and scored against *independent, bounded* dynamic GT.

#### S2 — Connectivity mapping fused with scene geometry
Highlights that link quality is co-registered with **perceived structure**, not just pose — so blockage can be *explained*, not merely mapped.
- **Inputs:** Robot-A RSSI/SNR/throughput/latency/loss time-joined to its GLIM trajectory, alongside the perceived scene (infrastructure + ego geometry/occupancy).
- **GT:** dense measured coverage on held-out survey passes.
- **Metrics:** coverage-map RMSE/MAE at held-out poses, dead-zone F1, **predictive-variance calibration**, and the lift from conditioning the radio map on scene geometry vs. pose alone (does seeing the metal shelf predict the dead zone?).
- **Baseline:** GP/kriging radio map from pose only vs. a geometry-conditioned predictor.
- **Why it's signature:** radio-map datasets give RSSI+pose; only here is the map co-registered with independent radar+camera scene perception, enabling geometry-aware connectivity prediction.

---

### Tier 3 — Enabling (foundational; establish and validate the GT — cited with real numbers, not headlined)

- **E1 — Radar↔camera extrinsic calibration.** GT: finalized `T_cam_radar` (radar1/radar2) with per-DOF covariance + the apex-offset invariant. Metrics: rotation/translation error, within-1σ rate, LOO-CV. Baseline: measurement-space MLE vs. isotropic Kabsch (the dataset ships the ≈1.6×/≈3× margin as reference).
- **E2 — Anisotropic radar-camera fusion.** GT: fused-track reference (≈[53,69,29] mm). Metrics: track accuracy, jitter, **covariance calibration** (are predicted σ's honest?). Baseline: the provided anisotropic constant-velocity fusion node.
- **E3 — Privacy-preserving radar-first sensing.** Radar-only detection/counting/localization vs. camera-derived reference, under occlusion/low-light; quantifies the privacy-vs-fidelity trade the deployment setting cares about.
- **E4 — Robot re-localization & calibration-drift.** Ego re-localization against the calibrated infrastructure (APE vs. GLIM, recall/latency after occlusion), and re-estimation accuracy vs. trajectory length / induced mount perturbation.

## 9. Limitations and Honest Scope

- **radar_infra is not finalized.** It has only a round-1 bootstrap (10 poses; rotation 1σ ≈ 7°, translation 1σ ≈ 80–106 mm) and **must not be treated as deployable**. It will be released as *provisional* with a clear flag, or completed before the dataset freezes (round-2 target: rot 1σ ≲ 3°, t 1σ ≲ 40 mm, all diversity bars green, session logging enabled).
- **Camera resolution bottleneck.** Sessions at 960×540 degrade ChArUco detection past ~0.4–1 m; this bounds handshake range and is documented per sequence.
- **Non-uniform camera-pose accuracy.** Distant cameras inherit GLIM drift; this is a *feature of the honest GT*, not a defect, but users must weight per-camera bounds accordingly.
- **Radar soft axes.** Each radar has a physically weak axis; single-radar localization on that axis is noisy by hardware design. The dataset documents which axis and relies on fusion to constrain it.
- **Novelty is scoped to a literature check**, not a formal prior-art/patent search (§7). The "first to combine" claim in §3 must be re-verified before formal publication.
- **Dynamic GT quality differs by robot.** Robot A's moving-target GT (§6.3) inherits GLIM drift (bounded by loop-closure residual); Robot B's depends on point-cloud registration quality (needs a good reference map + clean depth) or fiducial visibility. Both are reported per pose rather than hidden, and evaluation metrics are stated relative to the applicable bound.
- **Cross-platform time sync.** With two robots plus fixed infrastructure on independent clocks, the cooperative and cross-view tasks (F1/S1) depend on tight synchronization; clock domains and max sync-dt must be recorded per sequence (open question §11.5) or cross-view association degrades.
- **Wi-Fi coverage is single-robot and site/hardware-specific.** Only Robot A samples the link, so the radio map reflects Robot A's radio and the session's AP placement/chipset (some chipsets report no noise floor → no true SNR); generalization across radios and sites is itself a research question, not an assumption.

## 10. Data Format, Splits, and Release (proposed)

- **Container:** ROS 2 bags (native) — **per-robot** bags recorded on each robot's own disk (avoiding the measured Wi-Fi link) plus the infrastructure bag — with exported per-modality files (images + camera_info, radar point clouds with x/y/z/doppler/intensity, Wi-Fi/iperf/ping messages), each robot's **GT trajectory** (Robot A: GLIM; Robot B: point-cloud-registration / fiducial poses), and a transform tree linking fixed and mobile frames to the world frame.
- **Ground-truth package:** extrinsic YAML/JSON per sensor + `*_session.json` reproducibility records + both robots' GT trajectories + a machine-readable **uncertainty manifest** (per-camera bound, per-radar covariance, Robot A loop-closure drift, Robot B registration/fiducial residuals, apex-offset agreement).
- **Splits:** by session/site and by task; held-out survey passes reserved for F1/S2; a calibration-only split for E1/E4.
- **Tooling:** the calibration pipelines (`radar_camera_calib*`, `general_charuco`) and the Wi-Fi monitor stack shipped so ground truth is auditable and reproducible.
- **Licensing / ethics:** indoor human subjects imply consent and privacy handling; radar-only tracks are highlighted as the privacy-preserving alternative.

## 11. Open Questions to Resolve Before Freeze

1. Finalize radar_infra (round 2) or ship it flagged as provisional.
2. ~~Confirm the §5 novelty gap with the related-work survey.~~ **Done** — survey completed; novelty reframed precisely as *perception + RF link-quality* (MM-Fi/XRF55 use Wi-Fi as a sensing signal; Milosheski 2026 is RSSI+LiDAR only). Remaining: directly inspect Milosheski et al. (2026) to confirm it carries no active throughput/latency.
3. Decide camera resolution for the recording (raise above 960×540 if handshake range matters for the target tasks).
4. Fix the site/AP topology for the Wi-Fi survey so F1/S2 have a well-defined radio environment.
5. Define exact synchronization guarantees across the modalities **and across the two robots + fixed infrastructure** (clock domains, max sync dt) and record them per sequence — flagship/signature tasks F1/S1 depend on it.
6. ~~Confirm platform composition.~~ **Resolved** — Robot A is fully instrumented (ZED + radar1 + radar2 + Wi-Fi + GLIM); Robot B is RGBD-only. Radar/Wi-Fi tasks (E2/E3/F1/S2) run from Robot A; Robot B is a secondary RGBD viewpoint + moving target.
7. **Fix Robot B's pose-GT method:** point-cloud registration against the LiDAR reference map, fiducial markers, or both (with one as a cross-check). Decide before recording so the reference map / marker layout is in place (§6.3).
8. Decide whether to also mount a ChArUco/reflector target on **each** robot so the infrastructure's detection of them anchors to the same fiducial machinery as the static calibration.

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

### Appendix C — Verified references and flagged items

**Closest prior art (cite explicitly):**
- MM-Fi — Yang et al., *MM-Fi: Multi-Modal Non-Intrusive 4D Human Dataset for Versatile Wireless Sensing*, NeurIPS Datasets & Benchmarks 2023 (arXiv:2305.10345).
- XRF55 — Wang et al., *XRF55: A Radio Frequency Dataset for Human Indoor Action Analysis*, Proc. ACM IMWUT 8(1), 2024, DOI 10.1145/3643543.
- Milosheski et al., *Multimodal indoor radio mapping with 3D point clouds and RSSI*, Data in Brief 2026 (arXiv:2511.00494) — LiDAR + passive RSSI only.

**Datasets:** nuScenes (Caesar et al., CVPR 2020, arXiv:1903.11027); VoD (Palffy et al., RA-L 7(2):4961–4968, 2022); TJ4DRadSet (Zheng et al., ITSC 2022, arXiv:2204.13483); K-Radar (Paek et al., NeurIPS D&B 2022, arXiv:2206.08171); RADIATE (Sheeny et al., ICRA 2021); aiMotive (Matuszka et al., ICLR-W 2023, arXiv:2211.09445); CRUW/RODNet (Wang et al., WACV 2021); RadarScenes (Schumann et al., FUSION 2021, arXiv:2104.02493); CARRADA (Ouaknine et al., ICPR 2020/21); Astyx HiRes2019 (Meyer & Kuschk, EuRAD 2019); Oxford Radar RobotCar (Barnes et al., ICRA 2020); Dual Radar (Zhang et al., Sci. Data 2025); DAIR-V2X (Yu et al., CVPR 2022); Rope3D (Ye et al., CVPR 2022); IPS300+ (Wang et al., ICRA 2022); A9-Dataset (Creß et al., IEEE IV 2022, radar-inclusive); TUMTraf Intersection (Zimmer et al., ITSC 2023); LUMPI (Busch et al., IEEE IV 2022).

**Calibration / SLAM / geometry:** GLIM (Koide, Yokozuka, Oishi & Banno, *Robotics and Autonomous Systems* 179:104750, 2024); ArUco (Garrido-Jurado et al., *Pattern Recognition* 47(6):2280–2292, 2014); Umeyama (*IEEE TPAMI* 13(4):376–380, 1991); Kabsch (*Acta Cryst.* A32, 1976); Huber (*Ann. Math. Stat.* 35(1), 1964); SLERP (Shoemake, SIGGRAPH 1985); Rotation Averaging (Hartley et al., *IJCV* 103(3):267–305, 2013).

**Radar-camera calibration:** 4D-CAAL (Yao et al., arXiv:2601.21454, 2026); 3D-UPnP (Cao et al., arXiv:2507.19829, 2025); Domhof et al. (ICRA 2019; IEEE T-IV 6(3), 2021); Durmaz & Cevikalp (*Sensors* 2025, trajectory alignment); Liu et al. (*Sensors* 25(3):949, 2025, track-to-track); Cheng & Cao (NAECON 2023, arXiv:2309.00787); Fusion calib (Zhang et al., *Sci. Reports* 2025 — distinct from the PRL 2023 nickname collision); Doppler Correspondence (Kim et al., arXiv:2502.11461, 2025); surveys: Shi et al. (arXiv:2410.19872, 2024) and Han et al. (arXiv:2306.04242, 2023).

**Wi-Fi/RF sensing & connectivity-aware robotics:** RADAR (Bahl & Padmanabhan, INFOCOM 2000); Horus (Youssef & Agrawala, MobiSys 2005); SpotFi (Kotaru et al., SIGCOMM 2015); CSI survey (Ma, Zhou & Wang, ACM CSUR 52(3), 2019); GP RSS field (Ferris, Hähnel & Fox, RSS 2006); robot RF mapping (Fink & Kumar, ICRA 2010); comm-aware planning (Yan & Mostofi, IEEE TAC/TWC 2013); review (Muralidharan & Mostofi, Annu. Rev. Control Robot. Auton. Syst. 4:115–139, 2021); RCAMP (IROS 2017); CSI-HAR datasets Widar3.0 (MobiSys 2019), SignFi (IMWUT 2018), UT-HAR, SenseFi (*Patterns* 2023).

**Perception × communication (the intersection — §5.6):** DeepSense 6G (Alkhateeb et al., *IEEE Comms Mag.* 61(9), 2023, arXiv:2211.09769 — real camera/LiDAR/radar + mmWave, but channel-as-target); LuViRA (Yaman et al., ICRA 2024, arXiv:2302.05309 — indoor vision + 5G CSI + 0.5 mm GT, CSI-for-localization); mmHSense (arXiv:2509.21396), DISC (arXiv:2306.09469 — ISAC/RF human-sensing). Cooperative-perception benchmarks: OPV2V (R. Xu et al., ICRA 2022), V2X-Sim (Y. Li et al., RA-L 2022), V2V4Real (R. Xu et al., CVPR 2023), DAIR-V2X / V2X-Seq (Yu et al., CVPR 2022/2023), TUMTraf V2X (Zimmer et al., CVPR 2024). Bandwidth-aware collaborative perception: Where2comm (Hu et al., NeurIPS 2022), When2com (Liu et al., CVPR 2020), Who2com (Liu et al., ICRA 2020), DiscoNet (Y. Li et al., NeurIPS 2021), CoSDH (Xu et al., CVPR 2025). Real-link edge robotics: PEERNet (Narayanan et al., IROS 2024, arXiv:2409.06078).

**Flagged in §5.6 set — confirm before camera-ready:** DeepSense 6G comm-data semantics (beam/received power vs. link metric) verified via secondary sources only (primary PDF proxy-blocked); confirm exact venue/volume for the Shi et al. fusion survey and TUMTraf V2X page numbers.

**Flagged — confirm before camera-ready** (author lists / volume-issue behind blocked or paywalled hosts): Zendar (venue/authors, likely Mostajabi et al., CVPRW 2020); Fusion calib (Sci. Reports) author list; Trajectory-Alignment (*Sensors*) volume/article number; Shi et al. fusion survey IEEE journal name/volume; RCAMP / JCMP author lists; REM survey (2014) and RAS 2018 GP+path-loss author lists; recent offloading preprints (arXiv:2606.31497 authors); Widar3.0/SignFi author lists; Yan & Mostofi TAC exact vol/issue/pages. **Also directly inspect Milosheski et al. (2026)** to confirm no active throughput/latency and whether robot pose is released as an explicit stream.
