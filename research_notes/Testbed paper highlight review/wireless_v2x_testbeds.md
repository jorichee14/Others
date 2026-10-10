# Wireless-network testbeds, CAV/V2X testbeds and co-simulation platforms: headline claims and fidelity validation

Scope note: about 15 tool calls were used. arxiv.org and cisl.ucr.edu could not be fetched from this sandbox (DNS failure), so several details come from search-result abstracts and snippets rather than full text. Items marked **[PK]** come from prior knowledge and were NOT re-verified this session. Their venue/year/DOI is given so the writer can check it, and should be treated as "likely but unverified".

Positioning reminder: the target testbed is indoor, multi-robot, with every inter-agent Wi-Fi link measured (RSSI 5 Hz, ping 100 Hz, iperf goodput, CSI 104-141 Hz) alongside robot sensors, on one clock and one map frame, so collaborative perception can be replayed against the real link.

## Q1. Wireless testbeds (Colosseum, POWDER, AERPAW, COSMOS, ORBIT, NITOS) and how Colosseum validated channel-emulation fidelity

### Takeaway
The NSF PAWR / Colosseum family emphasises scale, programmability and remote access. Fidelity is shown mainly by "twinning": the same experiment is run over-the-air and in emulation and KPI time series are compared. The best-documented case is Colosseum-as-digital-twin of the indoor Arena testbed (TMC 2024). It reports throughput/SINR similarity of about 0.98-0.99 in its final arXiv version, and about 92.5% throughput / 80% SINR "accuracy" in its first version. These testbeds produce network KPIs, not co-registered robot/vehicle perception data.

### Cited Findings
**Colosseum (base platform)**
- Bonati et al., "Colosseum: Large-Scale Wireless Experimentation Through Hardware-in-the-Loop Network Emulation", arXiv 2110.10617 (Oct 2021). Northeastern, with Greenfly SAU LLC and Cerbo IO LLC. It is an open-access testbed of 256 SDRs plus a Massive Channel Emulator (MCHEM). RF scenarios are reproduced by FPGA-based finite-impulse-response (FIR) filters that apply channel taps to the signals the radio nodes generate — [arXiv 2110.10617](https://arxiv.org/abs/2110.10617v1)
- Scale reported elsewhere: 21 server racks, more than 170 servers, 256 USRP X310s, and over 65K emulated channels — [search summary of Colosseum papers / NSF PAR](https://par.nsf.gov/servlets/purl/10298726)
- Venue: IEEE DySPAN 2021 **[PK, verify]**. Headline: "world's largest wireless network emulator with hardware-in-the-loop".

**Colosseum as a Digital Twin (fidelity validation)**
- Villa et al., "Colosseum as a Digital Twin: Bridging Real-World Experimentation and Wireless Network Emulation", IEEE Transactions on Mobile Computing, vol. 23, no. 10, Oct 2024. It is an extended version of an ACM WiNTECH 2022 paper — [WINES BibTeX](https://ece.northeastern.edu/wineslab/wines_bibtex/villa2024dt.txt); [arXiv 2303.17063](https://arxiv.com/abs/2303.17063)
- What is built: a digital twin of the publicly available indoor sub-6 GHz OTA testbed "Arena". It uses CaST (Channel emulation scenario generator and Sounder Toolchain): ray-traced/measured channels are converted into emulator taps, and channel sounding is used to check the emulated taps — [arXiv 2303.17063](https://arxiv.org/pdf/2303.17063); [WINES digital-twin page](https://ece.northeastern.edu/wineslab/digitaltwin.php)
- Validation method: the same experiments were run on both the real and the digital system, covering cellular network scenarios and jamming of Wi-Fi nodes. Throughput and SINR time series were compared — [arXiv 2303.17063v6](https://arxiv.org/html/2303.17063v6)
- Fidelity numbers by version:
  - v6 / final: normalised cross-correlation similarity of 0.987 (throughput) and 0.982 (SINR).
  - Earlier version: up to 0.986 / 0.989.
  - v1 and the Northeastern news release: "average accuracy up to 92.5% in throughput and 80% in SINR".
  - Cite the TMC version — [arXiv 2303.17063](https://arxiv.org/pdf/2303.17063); [Northeastern INSI news](https://insi.northeastern.edu/news/colosseum-as-a-digital-twin-bridging-real-world-experimentation-and-wireless-network-emulation/)
- Admitted limitation / motivation: emulation results are only as reliable as the input channel model. The twin is proposed as a way to bound emulation-model error. Sounding verifies the emulator against the *input model*, not against the physical environment itself — [arXiv 2303.17063](https://arxiv.org/pdf/2303.17063)
- Follow-up: "Colosseum: The Open RAN Digital Twin" (Polese et al., IEEE Open Journal of the Communications Society, 2024) **[PK, verify]**. Later Northeastern work frames digital and physical open platforms together — [arXiv 2601.19027](https://arxiv.org/pdf/2601.19027)

**POWDER**
- Sim-to-real gap study: McManus et al. built a virtual POWDER in UBSim and compared policies trained on synthetic data with policies trained on over-the-air data. The size of the gap depends strongly on the chosen path-loss model — [arXiv 2408.14465](https://arxiv.org/html/2408.14465v2)
- Base paper: Breen et al., "POWDER: Platform for Open Wireless Data-driven Experimental Research", ACM WiNTECH 2020 / Computer Networks 2021 **[PK, verify]**. It is a city-scale (Salt Lake City) outdoor SDR testbed with remote access.

**AERPAW (aerial)**
- AERPAW digital twin: the drone emulation runs the same autopilot firmware as the real vehicles ("identical autopilot behavior"), so the claim is control fidelity. The RF part of the twin uses a physical USRP through a Keysight PROPSIM channel emulator — [arXiv 2410.09648](https://arxiv.org/pdf/2410.09648)
- ACHEM (2026): a real-time digital twin with channel and radio emulation, validated at AERPAW's Lake Wheeler site. The same trajectory and software stack were run OTA and in the twin, and LTE/5G/handover behaviour was compared ("system-level fidelity") — [arXiv 2604.04742](https://arxiv.org/pdf/2604.04742)
- AERPAW localization challenge, which uses the real testbed — [arXiv 2407.12180](https://arxiv.org/pdf/2407.12180)

**COSMOS, ORBIT, NITOS**
- COSMOS: "Challenge: COSMOS: A City-Scale Programmable Testbed for Experimentation with Advanced Wireless", ACM MobiCom 2020 (Raychaudhuri et al.) **[PK, verify]**. It is in West Harlem (NYC) with mmWave, optical x-haul and edge compute. The search found no COSMOS emulation-vs-real twin validation — [search result noting COSMOS as an ultra-dense NYC testbed](https://arxiv.org/pdf/2301.03359)
- ORBIT: a 400-node indoor radio grid at Rutgers WINLAB, IEEE WCNC 2005 (Raychaudhuri et al.) **[PK, verify]**. Its emphasis is reproducibility and remote access (OMF).
- NITOS: a Wi-Fi/LTE/SDR testbed at the University of Thessaly, part of FIRE/Fed4FIRE. A commonly cited paper is ITC 2014 **[PK, verify]**.

### Inferences
- The fidelity metric of record in this community is KPI time-series similarity (throughput, SINR) between twin and real runs. Per-link channel ground truth logged at high rate next to the application is uncommon. The target testbed's dense, multi-rate link logging (RSSI, RTT, goodput, CSI) is a measurement-first counterpart to this emulation-first lineage.
- Colosseum's own validation compares emulated against real KPIs for one indoor room (Arena) with static SDR nodes. It does not involve mobile robots or perception workloads. This is an opening for an indoor *mobile* multi-agent dataset with measured links.
- These testbeds stress scale (Colosseum: 256 SDRs, 65K channels), remote access and programmability. They do not provide co-registered robot sensing.

### Gaps
- Could not fetch full texts, so the per-scenario fidelity tables in Villa et al. and the exact DySPAN 2021 venue/DOI for Bonati et al. are not confirmed.
- No reliable source was found on COSMOS or NITOS twin/emulation validation.
- The POWDER base paper's venue was not verified.

## Q2. CAV/V2X platforms and datasets: OpenCDA, CARLA+network co-simulation, Veins/Artery, Mcity, and real V2X datasets with or without measured communication

### Takeaway
Simulation frameworks (OpenCDA, Veins, Artery, CDASim, ms-van3t-CARLA) show realism through bidirectional coupling and standards-compliant network stacks, not through comparison with real measurements. The large real cooperative-perception datasets (DAIR-V2X, V2V4Real, V2X-Real, TUMTraf V2X) record sensors but no per-frame measured link. Downstream papers therefore inject *modelled* latency and loss. Only recent work (CooperScene 2026, V2X-ReaLO 2025, COOPERNAUT radio measurements, AutowareV2X field tests) pairs perception with real communication traces. CooperScene is the closest outdoor analogue to the target testbed.

### Cited Findings
**Simulation / co-simulation**
- OpenCDA: Xu et al., "OpenCDA: An Open Cooperative Driving Automation Framework Integrated with Co-Simulation", IEEE ITSC 2021, arXiv 2107.06260 — [arXiv 2107.06260](https://arxiv.org/pdf/2107.06260)
  - What is built: CARLA (rendering and dynamics) plus SUMO (traffic) co-simulation with a full-stack CDA prototype (perception, localization, planning, platooning).
  - ns-3 is named only as a *future* extension: the docs say "SUMO and NS3 will be shown in OpenCDA in the next version v0.2" — [OpenCDA docs](https://opencda-documentation.readthedocs.io/en/feature-doc_v01/md_files/introduction.html); [Why OpenCDA](https://opencda-documentation.readthedocs.io/en/feature-rst_doc/md_files/introduction.html)
  - Whether an ns-3 link shipped was not confirmed.
- ms-van3t-CARLA (IFIP WONS 2024): the authors state that although frameworks such as OpenCDA extend CARLA for cooperative perception, "a significant gap remains in the use of accurate communication models". They couple ns-3 (ms-van3t, ETSI C-ITS stack) with CARLA sensors and physics and OpenCDA's fusion/control — [WONS 2024 PDF](https://dl.ifip.org/db/conf/wons/wons2024/1570976495.pdf); [EURECOM page](https://www.eurecom.edu/fr/node/4809294)
- CDASim (FHWA/TRB): CARLA + SUMO + ns-3 + CARMA Platform co-simulation, with ns-3 as the V2X communication simulator — [TRB session page](https://trb.secure-platform.com/a/solicitations/109/sessiongallery/schedule/items/1822/application/10945)
- Veins: Sommer, German, Dressler, "Bidirectionally Coupled Network and Road Traffic Simulation for Improved IVC Analysis", IEEE TMC 10(1), 2011, DOI 10.1109/TMC.2010.133 **[PK, verify]**. It is OMNeT++ + SUMO via TraCI. The headline is that bidirectional coupling changes IVC results versus pre-computed mobility traces. Its validity argument is coupling plus calibrated models, not field comparison.
- Artery: Riebl et al., "Artery: Extending Veins for VANET applications", MT-ITS 2015 **[PK, verify]**. It adds an ETSI ITS-G5 (Vanetza) stack on top of Veins.
- CarlaViz is a visualisation tool, not a network simulator **[PK]**. Mcity (U. Michigan): a 32-acre closed test facility with DSRC/C-V2X roadside units **[PK]**. No Mcity fidelity paper was found this session.

**Real cooperative-perception datasets (sensors only; no measured link per frame)**
- DAIR-V2X: real-world V2I cooperative dataset with 71,254 LiDAR and camera frames — [survey arXiv 2404.14022](https://arxiv.org/pdf/2404.14022)
  - CVPR 2022, arXiv 2204.05575 **[PK, verify]**.
  - It addresses temporal asynchrony with its Time Compensation Late Fusion baseline rather than measured link data **[PK, verify]**.
- V2V4Real: two vehicles with cameras, LiDAR and GPS/IMU, focused on V2V perception — [survey arXiv 2404.14022](https://arxiv.org/pdf/2404.14022). CVPR 2023, arXiv 2303.07601 **[PK, verify]**.
- V2X-Real (Xiang et al., UCLA Mobility Lab, ECCV 2024, arXiv 2403.16034) — [arXiv 2403.16034](https://arxiv.org/abs/2403.16034v2); [ECCV 2024 poster](https://eccv2024.ecva.net/virtual/2024/poster/988)
  - Setup: 2 CAVs and 2 smart infrastructure units, each with LiDAR and multi-view cameras.
  - Scale: about 33K LiDAR frames, 171K images and over 1.2M 3D boxes in 10 classes.
  - Splits: vehicle-centric, infrastructure-centric, V2V and I2I.
  - Headline gap claimed: no prior real dataset covered both V2V and V2I. No measured communication is reported in the abstract.
- TUMTraf V2X (Zimmer et al., arXiv 2403.01316; CVPR 2024 **[PK, verify]**): one RSU and one CAV at a four-way intersection. No communication-latency measurements are mentioned — [arXiv 2403.01316](https://arxiv.org/pdf/2403.01316)
- Downstream practice: a 2026 paper using V2V4Real and OPV2V builds a "simplified latency model" from another study's C-V2X latency and drop measurements. Latency is therefore modelled on top of the dataset, not logged in it — [arXiv 2604.22973](https://arxiv.org/pdf/2604.22973)
- Simulated cooperative datasets (V2X-Sim 2.0, OPV2V) are built in CARLA/SUMO/OpenCDA — [survey arXiv 2405.16973](https://arxiv.org/pdf/2405.16973)

**Datasets and studies that measure real V2X communication**
- CooperScene: "Multi-Modal Cooperative Autonomy Benchmark with C-V2X Communication Characterization", UC Riverside, arXiv 2606.31219 (Jul 2026, venue unknown) — [arXiv 2606.31219](https://arxiv.org/pdf/2606.31219); [project page](https://cisl.ucr.edu/CooperScene)
  - Platform: 3 CAVs and 1 RSU, each with multi-modal sensors and COTS C-V2X radios.
  - Data: 59K frames, 344K objects, 10 Hz labels. Each scene carries synchronised C-V2X latency, throughput, packet loss and jitter.
  - Sync and localisation: PTP time sync; GNSS-RTK + ICP with about 0.2 m average RMSE.
  - Claim: "in contrast to huge performance gains with infeasible communication assumptions, our practical benchmark raises awareness of the significant challenges in balancing model efficacy and communication efficiency." Packet loss and latency fluctuate in real channels because of mobility, interference and contention.
  - The per-model numbers on the project page were garbled in search output and are not quoted here.
- SEE-V2X: an earlier UC Riverside C-V2X direct-communication dataset — [SEE-V2X](https://cisl.ucr.edu/SEE-V2X/)
- V2X-ReaLO (arXiv 2503.10034, 2025): an open *online* cooperative perception framework on real vehicles and infrastructure, with a dataset for perception accuracy plus communication latency. It argues that offline evaluation on static or simulated datasets misses transmission latency — [arXiv 2503.10034](https://arxiv.org/html/2503.10034v1)
- COOPERNAUT (Cui et al., CVPR 2022 **[PK venue]**, arXiv 2205.02222): measured throughput and packet loss with 3 DSRC and 3 C-V2X radios on moving vehicles. C-V2X loss was below 5%, versus above 90% for 802.11n/ac, which are not designed for mobility. These measurements set the bandwidth budget for the end-to-end driving policy, which was trained in CARLA — [arXiv 2205.02222](https://arxiv.org/pdf/2205.02222)
  - This 802.11n/ac result is directly relevant to positioning: a Wi-Fi-based testbed should report its measured loss and RTT regime.
- AutowareV2X (field tests): end-to-end latency of shared perception messages (CPMs) is about 30 ms. Dual-channel CPM delivery keeps delivery reliable when one channel has heavy packet loss — [AutowareV2X page](https://tlab.hongo.wide.ad.jp/portfolio/extending-autoware-for-cooperative-driving-integrating-perception-planning-and-coordination/)
- Hawlader et al. (U. Luxembourg), "Cooperative Perception Using V2X Communications: An Experimental Study": CPMs over ITS-G5 and C-V2X, measuring end-to-end delay against detection quality — [ORBilu PDF](https://orbilu.uni.lu/bitstream/10993/63217/1/Cooperative_Perception_Using_V2X_Communications_An_Experimental_Study.pdf)
- "When Autonomous Vehicle Meets V2X Cooperative Perception: How Far Are We?" (arXiv 2509.24927): online evaluation over 32 routes under normal and abnormal communication, including injected latency and pose errors. It reports vulnerabilities in cooperative perception components — [arXiv 2509.24927](https://arxiv.org/pdf/2509.24927)
- CISTER (ISEP Porto), ETSI ITS-enabled robotic-scale testbed: 1/10-scale robotic vehicles with 802.11p OBUs and RSUs. It notes that V2X work "often resorts to static wireless testbeds" with little attention to sensors and processing. This is the closest *small-scale robotic* V2X analogue found — [CISTER PDF](https://cister.isep.ipp.pt/docs/an_etsi_its_enabled_robotic_scale_testbed_for_network_aided_safety_critical_scenarios/1927/attach.pdf)

### Inferences
- The main real-world V2X perception datasets from 2022-2024 (DAIR-V2X, V2V4Real, V2X-Real, TUMTraf V2X) appear to log sensors only. Latency and loss are then modelled synthetically downstream, e.g. the V2X-ViT-style latency and pose-noise injection **[PK]**. A testbed with a measured link per frame fills this gap directly. CooperScene (2026) is the main competitor making the same argument, outdoors with C-V2X.
- The target testbed can differ from CooperScene through: indoor and multi-robot operation, Wi-Fi with CSI (CooperScene reports network-level KPIs only, with no PHY-level CSI that was found), and higher-rate link sampling (ping at 100 Hz).
- COOPERNAUT's finding that 802.11n/ac fails under vehicular mobility suggests stressing that indoor robot speeds and distances suit Wi-Fi. It also suggests reporting measured link quality distributions.

### Gaps
- The DAIR-V2X, V2V4Real and TUMTraf papers were not opened to confirm that none of them logs real link metrics. No source explicitly says "communication not measured" for them, so this is an inference.
- The CooperScene sampling rates for its C-V2X metrics and its benchmark numbers were not retrievable. Its venue is unknown (arXiv, July 2026).
- No Mcity fidelity or validation paper was found. Whether OpenCDA's ns-3 integration was ever released was not confirmed.

## Q3. Robot-network co-simulation (ROS-NetSim, Gazebo+ns-3, FlyNetSim, digital twins)

### Takeaway
Robot-network co-simulators emphasise time synchronisation and transparency to ROS, and are agnostic to the physics and network simulator used. They do not validate their channel output against real measured links. Later work (BotNet) criticises them on scale and on lacking standards-compliant communication models.

### Cited Findings
- ROS-NetSim: Calvo-Fullana et al. (UPenn and collaborators), "ROS-NetSim: A Framework for the Integration of Robotic and Network Simulators", arXiv 2101.10113 (Jan 2021) — [arXiv 2101.10113](https://www.arxiv.org/abs/2101.10113); [GitHub](https://github.com/NESTLab/ros-net-sim)
  - What it is: a ROS package that interfaces robot and network simulators for Perception-Action-Communication loops in multi-robot systems.
  - Design goals: transparent to the ROS application, agnostic to the physics and network simulator, tunable in fidelity and complexity.
  - The code repository was archived (read-only) on 16 Mar 2022.
  - Venue: IEEE RA-L 2021 is likely but was **not confirmed** by search.
- BotNet (arXiv 2108.13606) criticises it: ROS-NetSim "is not designed to handle high agent counts nor standards compliant communication models" — [arXiv 2108.13606](https://arxiv.org/pdf/2108.13606)
- FlyNetSim (Baidya, Shaikh, Levorato, UC Irvine, 2018, arXiv 1808.04967): ns-3 + ArduPilot/DroneKit SITL, not Gazebo. A middleware layer provides time synchronisation and publish/subscribe data paths. The target is UAV swarms in an urban IoT setting — [FlyNetSim ar5iv](https://ar5iv.arxiv.org/html/1808.04967)
  - Venue: ACM MSWiM 2018 **[PK, verify]**.
- CORNET (IISc, COMSNETS 2020): Gazebo/ROS + ns-3 middleware for multi-UAV systems, focused on synchronising time and position across the two simulators — [CORNET PDF](https://ece.iisc.ac.in/~htyagi/papers/comsnet-20.pdf); [COMSNETS 2020 slides](https://ece.iisc.ac.in/~parimal/slides/2020/comsnets.pdf)
- FANS (Flying Ad-hoc Network Simulator) reuses ns-3 + Gazebo + ROS — [NITK](https://idr.nitk.ac.in/items/0a6842b5-8870-42ba-8474-407166dbf78e/full)
- A low-latency synchronising middleware for heterogeneous multi-robot co-simulation — [arXiv 2211.05359](https://arxiv.org/pdf/2211.05359)

### Inferences
- None of these co-simulators reports a comparison between its simulated link (RSSI, latency, loss) and real measured robot links. Measured indoor multi-robot Wi-Fi traces could serve as the missing ground truth to calibrate or validate them, for example by replaying measured traces as a "trace-driven channel" inside ROS-NetSim-style pipelines.

### Gaps
- The ROS-NetSim venue (RA-L 2021) was not verified. A Gazebo-integrated FlyNetSim was not found and may not exist.
- No robotics-network digital twin with an explicit measured-vs-simulated link fidelity metric was found in this session.

## Q4. Cross-cutting: what these papers emphasise and what gaps they admit

### Takeaway
Wireless testbeds emphasise scale, remote access, programmability and, more recently, twin fidelity as KPI similarity. V2X simulators emphasise coupling and standards-compliant stacks. Perception datasets emphasise sensor scale and labels. The explicitly admitted gaps are three: accurate communication models are missing in CARLA-based cooperative perception; offline datasets miss transmission latency; and emulators are only as good as their channel model. A synchronised, measured inter-agent link is acknowledged as missing until very recent outdoor work (CooperScene, V2X-ReaLO).

### Cited Findings
- Emulation fidelity is bounded by the channel model, which motivates a real-vs-twin comparison — [Villa et al., arXiv 2303.17063](https://arxiv.org/pdf/2303.17063)
- The sim-to-real gap depends on the propagation (path-loss) model — [POWDER twin study, arXiv 2408.14465](https://arxiv.org/html/2408.14465v2)
- "A significant gap remains in the use of accurate communication models" for CARLA/OpenCDA cooperative perception — [ms-van3t-CARLA, WONS 2024](https://dl.ifip.org/db/conf/wons/wons2024/1570976495.pdf)
- Offline evaluation on simulated or static datasets misses transmission latency — [V2X-ReaLO](https://arxiv.org/html/2503.10034v1)
- Gains under "infeasible communication assumptions" collapse under real C-V2X loss and latency — [CooperScene](https://arxiv.org/pdf/2606.31219)
- V2X work often uses static wireless testbeds with little attention to sensors and processing — [CISTER robotic-scale testbed](https://cister.isep.ipp.pt/docs/an_etsi_its_enabled_robotic_scale_testbed_for_network_aided_safety_critical_scenarios/1927/attach.pdf)
- Robot-network co-simulators lack scale and standards-compliant models — [BotNet](https://arxiv.org/pdf/2108.13606)

### Inferences
Positioning statement supported by the above: the target testbed sits where three gaps meet.
1. Wireless twins validate KPIs, not perception.
2. Cooperative perception datasets have sensors but synthetic links.
3. Robot-network co-simulators have no measured ground truth.

Its differentiators against the nearest competitor (CooperScene) are indoor multi-robot Wi-Fi, PHY-level CSI, high-rate RTT probing, and a single clock and map frame for exact replay.

### Gaps
- No found paper combines measured Wi-Fi CSI with multi-robot collaborative perception. This absence comes from limited searching and is not proof that none exists.
- The sampling rates of measured link metrics in CooperScene and V2X-ReaLO were not obtained, so a head-to-head comparison of rates (e.g., ping at 100 Hz versus their rates) cannot be made from these notes.
