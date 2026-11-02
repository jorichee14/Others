# Reference papers

These are the references for [`PLAN.md`](PLAN.md),
[`THEORY.md`](THEORY.md) and [`OBJECTIVE.md`](OBJECTIVE.md), grouped by
the role each plays. BibTeX keys are in
[`references.bib`](references.bib).

Each entry has a status:

| Status | Meaning |
|---|---|
| **C** | Classical; the record is well known |
| **V** | Title, authors and venue checked against an arXiv, proceedings or publisher listing on 30 Oct 2026 |
| **S** | Known from search summaries only; fields marked "confirm" in the bib still need checking |

None of the V or S papers was read in full here. Read the primary text
before citing any claim about its content.

## 1. What to read first

These are the closest prior work. The novelty claims stand or fall on
them.

| Key | Paper | Threat to | Status |
|---|---|---|---|
| `lan2022progressftx` | Lan, Zeng, Popovski, Gündüz, Huang, *Progressive Feature Transmission for Split Classification at the Wireless Edge*, IEEE TWC | Proposition 3, the stopping rule | S |
| `gan2025scomcp` | Gan et al., *SComCP: Task-Oriented Semantic Communication for Collaborative Perception*, arXiv 2507.00895 | C1, the formulation | V (arXiv) |
| `zhu2026tamp` | Zhu, Sun, Jia, Chen, Ai, Zhou, *Timeliness-Oriented Scheduling and Resource Allocation in Multi-Region Collaborative Perception*, arXiv 2601.04542 | The objective's AoI special case; Lyapunov | V |
| `fang2025racp` | Fang et al., *R-ACP: Real-Time Adaptive CP Leveraging Robust Task-Oriented Communications*, IEEE JSAC 2025 | Adaptive compression with priority scheduling | V |
| `tian2008successive` | Tian, Steiner, Shamai, Diggavi, *Successive Refinement via Broadcast*, IEEE T-IT 2008 | C2, the PHY analogue | C |
| `fouladi2018salsify` | Fouladi et al., *Salsify*, NSDI 2018 | C4, prediction against progressivity | C |
| `ma2026unseen` | Ma et al., *Update the Unseen Only*, arXiv 2607.20967 | Value conditional on the receiver (visibility) | V |

## 2. Task-oriented and semantic communication: motivation and framing

| Key | Paper | Used for | Status |
|---|---|---|---|
| `shannon1949mtc` | Shannon and Weaver, *The Mathematical Theory of Communication*, 1949 | Level C, the effectiveness problem: the origin of "task-oriented" | C |
| `gunduz2023beyond` | Gündüz et al., *Beyond Transmitting Bits*, IEEE JSAC 2023 | Survey and framing | C |
| `strinati2021sixg` | Strinati and Barbarossa, *6G Networks: Beyond Shannon*, Computer Networks 2021 | Framing | C |
| `shao2022ib` | Shao, Mao, Zhang, *Task-Oriented Communication for Edge Inference: An Information Bottleneck Approach*, IEEE JSAC 2022 | The task-oriented baseline formulation | C |
| `kurka2019successive` | Kurka and Gündüz, *Successive Refinement of Images with Deep JSCC*, SPAWC 2019 | Learned progressive codes (prior art) | C |
| `qiao2025tokencom` | Qiao et al., *Token Communications*, IEEE Wireless Commun. 2025 | Tokens as the unit (Proposition 2) | S |
| `gan2025cods` | Gan, Liang, Zhang, Guo, Jin, *CoDS*, arXiv 2512.22513 | Semantic comm for CP over a PHY channel | V |
| `sheng2024simharq` | *Semantic Communication for Cooperative Perception Using HARQ*, arXiv 2409.09042 | Incremental revisions (prior art for progressivity) | S |

## 3. Information theory: the foundation of THEORY §2.1–2.2

| Key | Paper | Used for | Status |
|---|---|---|---|
| `equitz1991successive` | Equitz and Cover, *Successive Refinement of Information*, IEEE T-IT 1991 | Progressive codes | C |
| `rimoldi1994successive` | Rimoldi, IEEE T-IT 1994 | Achievable rates for refinement | C |
| `lastras2001nearly` | Lastras and Berger, *All Sources Are Nearly Successively Refinable*, IEEE T-IT 2001 | A bound on the refinement penalty ε under MSE | C |
| `wyner1976wz` | Wyner and Ziv, IEEE T-IT 1976 | The receiver's view (or the twin) as decoder side information | C |
| `steinberg2004wzsr` | Steinberg and Merhav, *On Successive Refinement for the Wyner–Ziv Problem*, IEEE T-IT 2004 | Progressive coding with side information | C |
| `tian2007multistage` | Tian and Diggavi, *Multistage Successive Refinement for Wyner–Ziv*, IEEE T-IT 2007 | Same, multistage | C |
| `shamai1997broadcast`, `shamai2003broadcast` | Shamai; Shamai and Steiner, *The Broadcast Approach*, ISIT 1997 and IEEE T-IT 2003 | Layering when the channel state is unknown | C |
| `ng2009distortion` | Ng, Gündüz, Goldsmith, Erkip, IEEE T-IT 2009 | Minimizing expected distortion over a continuum of states | C |

## 4. Optimization: Propositions 2–3 and the solution of the objective

| Key | Paper | Used for | Status |
|---|---|---|---|
| `nemhauser1978analysis` | Nemhauser, Wolsey, Fisher, Math. Prog. 1978 | The 1 − 1/e greedy bound for every prefix (Proposition 2) | C |
| `fisher1978analysis` | Fisher, Nemhauser, Wolsey, Math. Prog. Study 1978 | The ½ greedy bound under a matroid (OBJECTIVE Property 2) | C |
| `khuller1999budgeted` | Khuller, Moss, Naor, IPL 1999 | Why unequal unit sizes break the guarantee | C |
| `cormen2009clrs` | Cormen et al., §16.5 | Unit tasks with deadlines form a matroid ("select by value, send by deadline") | C |
| `neely2010stochastic` | Neely, *Stochastic Network Optimization*, 2010 | Drift-plus-penalty, the O(1/V) and O(V) bounds | C |
| `kelly1998rate` | Kelly, Maulloo, Tan, JORS 1998 | NUM, the throughput-utility special case | C |

## 5. Timeliness: Proposition 6 and the AoI special case

| Key | Paper | Used for | Status |
|---|---|---|---|
| `kaul2012aoi` | Kaul, Yates, Gruteser, INFOCOM 2012 | The origin of AoI | C |
| `yates2021aoi` | Yates et al., *Age of Information: An Introduction and Survey*, IEEE JSAC 2021 | Survey | C |
| `sun2020wiener` | Sun, Polyanskiy, Uysal, IEEE T-IT 2020 | Signal-aware sampling beats age-optimal (the analogue of Proposition 6) | V |
| `maatouk2020aoii` | Maatouk et al., *Age of Incorrect Information*, IEEE/ACM ToN 2020 | Content-aware timeliness | C |
| `kadota2018scheduling` | Kadota et al., IEEE/ACM ToN 2018 | AoI scheduling (Whittle, max-weight) | C |
| `tripathi2023wiswarm` | Tripathi et al., *WiSwarm*, INFOCOM 2023 | AoI on a real multi-robot Wi-Fi network | S |

## 6. Networking: codec–transport co-design and Wi-Fi

| Key | Paper | Used for | Status |
|---|---|---|---|
| `winstein2013sprout` | Winstein, Sivaraman, Balakrishnan, *Sprout*, NSDI 2013 | Stochastic capacity forecasts for a delay target | C |
| `fouladi2018salsify` | Fouladi et al., *Salsify*, NSDI 2018 | Choosing an encoding by predicted capacity | C |
| `schwarz2007svc` | Schwarz, Marpe, Wiegand, *H.264/SVC*, IEEE TCSVT 2007 | Layered video | C |
| `heusse2003anomaly` | Heusse et al., *Performance Anomaly of 802.11b*, INFOCOM 2003 | Airtime sharing under contention (THEORY §2.7) | C |
| `ieee80211e` | IEEE 802.11e-2005 (WMM) | Priority access categories (Proposition 6) | C |

## 7. Collaborative perception: what the objective generalizes

| Key | Paper | Special case of DTCV | Status |
|---|---|---|---|
| `where2comm` | Hu et al., *Where2comm*, NeurIPS 2022 | Fixed bit budget | C |
| `hu2024codefilling` | Hu et al., *CodeFilling*, CVPR 2024 | Fixed bit budget | S |
| `liu2026rdcomm` | Liu, Hu, Hu, Zhang, Chen, *Rate-Distortion Optimized Pragmatic Communication for CP*, ICLR 2026 | Pragmatic rate–distortion | V |
| `infocom2026` | *InfoCom*, AAAI 2026, arXiv 2512.10305 | Information bottleneck at kilobyte scale | S |
| `wu2025fresh2comm` | Wu, Peng, Yu, *Fresh2comm*, arXiv 2502.07852 | AoI | S |
| `wang2023aoicp` | Wang et al., *AoI Guaranteed Scheduling ... in CP*, arXiv 2310.04813 | AoI | V (authors differ from the WiOpt version) |
| `han2026aoisync` | Han, Wang, Liwang, Niyato, IEEE TCCN 2026 | Size → delay → AoI → fusion weight | V |
| `schedcp2025` | *DRL-Based User Scheduling for CP (SchedCP)*, arXiv 2502.10456 | Scheduling on current CSI | S |
| `v2xvit` | Xu et al., *V2X-ViT*, ECCV 2022 | Constant 100 ms delay | C |
| `lei2022syncnet` | Lei et al., *SyncNet*, ECCV 2022 | Latency compensation | C |
| `wei2023cobevflow` | Wei et al., *CoBEVFlow*, NeurIPS 2023 | Latency compensation | C |
| `yu2023ffnet` | Yu et al., *FFNet*, NeurIPS 2023 | Sender-side prediction | S |
| `song2025trafalign` | Song, Yang, Wen, Li, *TraF-Align*, CVPR 2025 | Latency compensation | V |
| `wang2026sparsecoop` | Wang et al., *SparseCoop*, AAAI 2026 | Instance queries with velocity (the kinematic tag) | V |
| `codyntrust2025` | *CoDynTrust*, ICRA 2025, arXiv 2502.08169 | Discounting stale features | S |

## 8. Measured links, radio prediction and twins

| Key | Paper | Used for | Status |
|---|---|---|---|
| `cooperscene` | *CooperScene*, arXiv 2606.31219 (full entry in `../scoop_testbed_iotj/references.bib`) | The only other perception dataset with a measured link (outdoor C-V2X) | V |
| `v2xrealo` | *V2X-ReaLO*, arXiv 2503.10034 (full entry there too) | Online CP over a real link | V |
| `hoydis2023sionnart` | Hoydis et al., *Sionna RT*, Globecom Workshops 2023 | Predicting the link from the twin | C |
| `luo2025fermi` | Luo et al., *FERMI*, RSS 2025 | Robot radio mapping | V |
| `amigoni2017survey` | Amigoni, Banfi, Basilico, IEEE Intelligent Systems 2017 | Communication maps for multi-robot planning | C |
