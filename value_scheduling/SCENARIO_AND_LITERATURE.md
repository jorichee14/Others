# The scenario, and the literature by category

## 1. The scenario

A robot team on shared Wi-Fi in a changing building, keeping each
robot's picture of the places its next move depends on correct, in
time.

Some robots move through the building on jobs; some fixed cameras
watch parts of it; all share one access point with the building's
other traffic. People walk, doors open and close, things get moved.
Each robot is about to act somewhere: turn a corner, enter a room,
cross a junction. To act well it needs a correct picture of the places
its next move depends on. Some it will see for itself in time; some
are hidden and it can only learn about them from another robot or a
camera that sees them now. Every observation is about a place, taken
at a time, and stays true for a while: a wall forever, a parked cart
for minutes, a walking person for a second or two.

The problem the team faces at every moment:

> Of everything anyone has seen, which observations should go to
> which robot, in what form, sent by whom, and when, so that every
> robot has a correct picture of the places its next move depends on
> by the time it makes that move, given that the link cannot carry
> everything and the robots keep moving?

Four things make it hard, and they are the four axes of the
literature below:

1. **Deadlines come from motion.** When a robot needs a place is set
   by how fast it moves toward it.
2. **Validity comes from the world.** How long an observation stays
   true depends on what is in it and how busy the place is.
3. **Need comes from the receiver's own future.** A place the robot
   will see itself before it matters needs no message.
4. **The channel is shared and short.** One AP; messages collide; bulk
   traffic delays small urgent messages.

The quantity everything is scored by: **AoII at the decision**, how
long a robot's belief about a place has been wrong at the moment its
move depends on that place.

The one approach: decide what to send by when it is needed and how
long it stays true; take turns on the channel; let the robot move to
keep its picture right; score at the moment it acts.

Experiments, in order of size: one blind corner (one mast, one robot,
a loading robot); the crossing (three masts, two robots, a throttled
link); the changed building (a moved cabinet, a closed door); the
full rig under pose and control deadlines.

Rig: two Scout Mini (Jetson Orin NX, Ouster, ZED, radar), three
Raspberry Pi masts with cameras, one Asus RT-BE82U, one server, chrony
(offset ≤ 0.99 ms measured), one sniffer, the reference map of site 1
preloaded on every node. No digital twin, no ray tracer, no labels
beyond who was where per trial.

---

## 2. The literature, by category

Each category owns one of the four axes and assumes the others away.
Tags: [F] read in full in this project (PDFs in `refs/`); [V] venue
checked; [S] abstract or summary only, check before citing.

### 2.1 Collaborative perception: what to send

Owns: which cells, features or objects, and how few bytes. Assumes
away: the channel (a byte budget or a constant delay), the receiver's
future, deadlines, other senders.

| Paper | Venue | Note |
|---|---|---|
| Where2comm (Hu, Fang, Lei, Zhong, Chen) | NeurIPS 2022 | spatial confidence + request map; the canonical baseline |
| V2X-ViT (Xu et al.) | ECCV 2022 | delay-aware fusion; constant 100 ms delay |
| CoBEVT (Xu et al.) | CoRL 2022 | sparse BEV fusion |
| CodeFilling (Hu et al.) | CVPR 2024 [S] | codebook messages |
| CoSDH (Xu et al.) | CVPR 2025 [S] | supply–demand masks, intermediate–late hybrid |
| ERMVP (Zhang et al.) | CVPR 2024 [S] | efficient and robust messages |
| InfoCom | AAAI 2026 [S] | kilobyte messages by information bottleneck |
| RDComm (Liu, Hu, Hu, Zhang, Chen) | ICLR 2026 [V] | pragmatic rate–distortion |
| WhisperNet (Chen, Zhang, Zhao) | CVPR 2026 [S] | sender importance + receiver coordination at 1 % bandwidth |
| Dense Coverage, Sparse Refinement (Yazgan, Mueller, Zoellner) | arXiv 2609.29456, 2026 [F] | coarse base + 198-byte patches ranked by learned benefit; strict byte caps; redundancy stated as open |
| JigsawComm (Wang, Li, Li, Wen) | arXiv 2511.17843, 2026 [F] | learned per-cell utility, top-1 sender per cell, O(1) in N; decision identical at every agent; "more is worse" |
| SparseCoop (Wang et al.) | AAAI 2026 [V] | instance queries with explicit position and velocity |
| TraF-Align (Song, Yang, Wen, Li) | CVPR 2025 [V] | trajectory-aware feature alignment for latency |
| CoBEVFlow (Wei et al.) | NeurIPS 2023 | BEV flow for asynchrony |
| FFNet (Yu et al.) | NeurIPS 2023 [S] | sender-side feature flow |
| SyncNet (Lei et al.) | ECCV 2022 | latency-aware fusion |
| Reason-to-Transmit (Bansal, Gangwani) | arXiv 2603.20308, 2026 [F] | transformer selection with neighbour-state token; finds selection policies within ~0.5 AP of each other, fusion dominates |
| InCoP (anon., under review) | AAAI 2027 submission [F] | indoor ground-robot benchmark; CGRF density-guided residual fusion |

### 2.2 Task-oriented and semantic communication: how to encode

Owns: bits that preserve a task loss over a noisy physical channel.
Assumes away: the MAC (no contention, no queues), the receiver's
state, time.

| Paper | Venue | Note |
|---|---|---|
| Gündüz et al., Beyond Transmitting Bits | JSAC 2023 | the survey |
| Shao, Mao, Zhang, Task-oriented communication for edge inference | JSAC 2022 | information bottleneck |
| Shao, Mao, Zhang, multi-device cooperative edge inference | TWC 2023 | distributed IB |
| Lan, Zeng, Popovski, Gündüz, Huang, ProgressFTX | TWC 2022 [S] | progressive feature transmission, threshold stopping |
| SComCP (Gan et al.) | arXiv 2507.00895, 2025 [V] | semantic comm for CP over a PHY channel |
| CoDS (Gan, Liang, Zhang, Guo, Jin) | arXiv 2512.22513, 2025 [V] | digital semantic comm for CP |
| R-ACP (Fang et al.) | JSAC 2025 [V] | capacity-adaptive compression, priority scheduling, real vehicles |
| Token Communications (Qiao et al.) | IEEE Wireless Commun. 2025 [S] | tokens as the unit |
| Equitz–Cover; Steinberg–Merhav | T-IT 1991; 2004 | successive refinement; with side information |
| Tian, Steiner, Shamai, Diggavi | T-IT 2008 | broadcast approach, expected distortion |

### 2.3 Timeliness: when to update

Owns: age, freshness, scheduling theory with bounds. Assumes away:
content (every update ages alike), the receiver's own sensing (partly
fixed by Unseen Only), decision deadlines.

| Paper | Venue | Note |
|---|---|---|
| Kaul, Yates, Gruteser | INFOCOM 2012 | the origin of AoI |
| Yates, Sun, Brown, Kaul, Modiano, Ulukus | JSAC 2021 | survey |
| Kadota, Sinha, Uysal-Biyikoglu, Singh, Modiano | ToN 2018 | AoI scheduling, Whittle and max-weight |
| Sun, Polyanskiy, Uysal | T-IT 2020 [V] | signal-aware sampling beats age-optimal |
| Maatouk, Kriouile, Assaad, Ephremides, AoII | ToN 2020 | age of incorrect information |
| Fresh2comm (Wu, Peng, Yu) | arXiv 2502.07852, 2025 [S] | AoI for CP |
| TAMP (Zhu, Sun, Jia, Chen, Ai, Zhou) | arXiv 2601.04542, 2026 [V] | Lyapunov region scheduling with an age-to-AP penalty |
| AoI-guaranteed scheduling for CP (Wang/Xu, Sun, Zhou, Niu) | WiOpt 2023 [S] | center-side RU allocation under age constraints |
| Han, Wang, Liwang, Niyato | TCCN 2026 [V] | AoI from message size and link, weighting fusion |
| Update the Unseen Only (Ma, Zhao, Fang, An, Chen, Fang) | arXiv 2607.20967, 2026 [F] | age over areas users cannot sense; INAR persistence; LocMW with sublinear regret |
| WiSwarm (Tripathi, Kadota, Tal, Rahman, Warren, Karaman, Modiano) | INFOCOM 2023 [S] | AoI scheduling on a real multi-UAV Wi-Fi rig |
| Tripathi et al., computation and communication co-design | WiOpt 2021 [S] | joint compute and channel scheduling by AoI, occupancy mapping |

### 2.4 Mobile systems: who sends to whom on a real radio

Owns: relevance, visibility, bandwidth partitioning, measured
latency, real radios. Assumes away: spoil times from content,
decision-time metrics, motion as a lever.

| Paper | Venue | Note |
|---|---|---|
| AutoCast (Qiu, Huang, Asavisanu, Liu, Psounis, Govindan) | MobiSys 2022 [F] | per-object, per-receiver visibility (current pose) and trajectory relevance; greedy by reward per byte; LTE-Direct TDMA; DSRC replay. The closest prior work |
| EMP (Zhang, Zhang, Sun, Zhu, Guo, Qian, Mao) | MobiCom 2021 [S] | edge-assisted multi-vehicle perception, bandwidth-adaptive point-cloud partition |
| AVR (Qiu, Ahmad, Bai, Gruteser, Govindan) | MobiSys 2018 [S] | augmented vehicular reality, V2V point clouds |
| Zhang et al., robust real-time multi-vehicle collaboration on asynchronous sensors | MobiCom 2023 [S] | asynchrony in real time |
| Wang, Cao, edge-assisted relevance-aware dissemination | IEEE 2024 [S] | send only objects whose trajectories intersect the receiver's |
| Lusvarghi et al., The Search for Relevance | TVT 2026 [V] | relevance theory for V2X content |
| V2X-ReaLO (Xiang et al.) | arXiv 2503.10034, 2025 [S] | online fusion over a live link |
| CooperScene | arXiv 2606.31219, 2026 [S] | per-scene C-V2X traces outdoors |
| SCooP-iT (this group) | in preparation | every link logged on the sensor clock and in the map frame, indoors, replay validated |

### 2.5 Wireless access and resource allocation: who gets the channel

Owns: scheduling users, slots, OFDMA, DCC, MARL for access, MEC
offloading. Assumes away: what messages mean, when they are needed.

| Paper | Venue | Note |
|---|---|---|
| Bianchi, performance analysis of 802.11 DCF | JSAC 2000 | the CSMA model |
| Heusse, Rousseau, Berger-Sabbatel, Duda, performance anomaly | INFOCOM 2003 | slow senders hog airtime |
| IEEE 802.11ax / 802.11be | standards | uplink OFDMA, trigger frames, BSR, restricted TWT |
| ETSI TS 103 324, TR 103 562 | standards | CPM generation and redundancy rules; DCC |
| Lyu et al., VoI-based prioritisation for CPM | VNC 2025 [S] | value of information in the standard |
| SchedCP | arXiv 2502.10456, 2025 [S] | DRL user scheduling by CSI and semantics |
| PRISM (Mazandarani, Shokrnezhad, Taleb) | TNSM 2026 [F] | distributed MADRL MAC over shared semantic segments |
| Yahya, Naeem, Alsisi, Butt, Ejaz, robust task offloading in DT-enabled UAV-IoT | OJ-COMS 2026 [F] | MINLP offloading with box-uncertain positions; content-free tasks; orthogonal channels |
| Hady et al., MARL for resource allocation: a survey | AI Review 2025 [S] | the MARL line |
| Neely, Stochastic Network Optimization | 2010 | drift-plus-penalty |
| Kelly, Maulloo, Tan | JORS 1998 | network utility maximization |

### 2.6 Communication-aware robotics: where to move for the link

Owns: connectivity, throughput, radio maps as inputs to motion.
Assumes away: what is sent, validity of content, perception tasks.

| Paper | Venue | Note |
|---|---|---|
| Muralidharan, Mostofi, Communication-aware robotics | Annu. Rev. Control Robot. Auton. Syst. 2021 [V] | the survey |
| Fink, Ribeiro, Kumar | Proc. IEEE 2012 | robust control for mobility and wireless communication |
| Stephan, Fink, Kumar, Ribeiro | T-RO 2017 | concurrent control of mobility and communication |
| Mostofi, Malmirchegini, Ghaffarkhah | ICRA 2010 | channel estimation in robotic networks |
| FERMI (Luo et al.) | RSS 2025 [V] | multi-robot radio mapping |
| Amigoni, Banfi, Basilico | IEEE Intell. Syst. 2017 | communication-restricted exploration survey |
| Gielis, Shankar, Prorok | Current Robotics Reports 2022 | critical review of communications in multi-robot systems |
| Yang et al., online learning and connectivity maintenance | arXiv 2410.05798, 2024 [S] | connectivity-aware coordination |

### 2.7 Multi-robot mapping and estimation: sharing a map

Owns: novelty against a shared map, increments, event triggering.
Assumes away: deadlines, the channel, dynamic objects.

| Paper | Venue | Note |
|---|---|---|
| Rocha, Dias, Carvalho | ICRA 2005 | entropy-utility measurement sharing |
| Kimera-Multi (Tian et al.) | T-RO 2022 | bandwidth-parsimonious distributed SLAM |
| Swarm-SLAM (Lajoie, Beltrame) | RA-L 2024 | sparse inter-robot sharing |
| Commerge | arXiv 2606.25386, 2026 [S] | scan-subset exchange for merging |
| Trimpe, D'Andrea, event-based state estimation | TAC 2014 | send on innovation; silence is information |
| Khronos (Schmid, Abate, Chang, Carlone) | RSS 2024 | spatio-temporal metric-semantic SLAM with change |

### 2.8 Networked control and co-design

Owns: compute + channel + control under AoI. Assumes away: perception
content, real contended radios, heterogeneous teams.

| Paper | Venue | Note |
|---|---|---|
| Tripathi et al., co-design for monitoring and control | WiOpt 2021 [S] | Whittle co-scheduling of processing and channel |
| Hahn, Stursberg, uncertain AoI in stochastic control | arXiv 2021 [S] | AoI as control input |
| AoI-aware task-oriented scheduling for NCS on SDR | arXiv 2022 [S] | argues AoI needs semantic measures |
| Park, Sahai et al., wireless network design for control systems | COMST 2018 | survey |

### 2.9 Caching and edge systems (person 1's shelf)

Owns: TTLs, invalidation, prefetch, compute reuse. Assumes away:
validity from the environment, access patterns known from plans.

| Paper | Venue | Note |
|---|---|---|
| Belady | IBM Syst. J. 1966 | optimal offline eviction |
| Fofack, Nain, Neglia, Towsley, TTL caches | (check venue) | TTL as a cache model |
| CrossRoI (Guo et al.) | MMSys 2021 [S] | cross-camera RoI masks to cut compute and traffic |
| Spatula (Jain et al.) | SEC 2020 [S] | cross-camera correlations, 8–86× compute reduction |
| Mobility-aware proactive caching at RSUs | surveys 2024–25 [S] | prefetch by predicted trajectory |

---

## 3. Where the gaps are, read off the categories

| Axis of the scenario | Nearest prior | What is missing |
|---|---|---|
| deadline from motion | AutoCast's 1/TTC relevance; AoI's age | a per-place deadline from the receiver's stopping distance |
| validity from the world | AoII (assumed source model); ETSI dynamics rules (per object) | a per-place change rate measured from revisits, used as the lifetime |
| need from the receiver's future | AutoCast's current visibility; Unseen Only's statistical persistence | the receiver's own future view cast from its plan and the map |
| shared, short channel with content on it | MobiSys rigs (no content model); MAC work (no meaning); CV (no channel) | a measured contended AP where the content decision changes what arrives |

Two consequences no category reports: a clairvoyant bound from the
known paths and the logged link, and the fraction of the field's
traffic that its receivers never needed.

## 4. Positioning in one paragraph

The CV community fixes the channel, the communication-theory community
fixes the content, the AoI community fixes the meaning, the mobile-
systems community fixes the time, and the robotics community fixes the
message; each gets a clean problem by freezing what its neighbour
studies. The scenario above leaves all four open at once, and the one
quantity that ties them, AoII at the decision, is computable from
state the agents already exchange. AutoCast is the paper to position
against first; WiSwarm and Tripathi's co-design are the scheduling
ancestors; EMP the systems ancestor; Mostofi and Fink–Kumar the motion
ancestors; Trimpe's event triggering and ETSI's generation rules the
ancestors of "send on change".
