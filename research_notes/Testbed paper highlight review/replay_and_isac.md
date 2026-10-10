# Network trace replay / record-and-replay literature and sensing-plus-communication (ISAC) datasets: how replay fidelity was validated, and positioning for "channel-in-the-loop replay"

Research method note (read first). In this session, arxiv.org, usenix.org, ecva.net, neurips.cc, stanford.edu and acm.org could not be fetched directly: the egress proxy returned 403, and WebFetch could not resolve DNS. Evidence therefore comes from (1) web-search snippets of the primary papers and (2) **the official GitHub repos, cloned and read directly**. The repos are the authoritative source for the exact synthetic-channel parameters below. Items marked **[unverified, from background knowledge]** could not be checked against a primary source in this session. Confirm them against the PDF before citing them in the paper.

The user's replay model, for reference: arrival = t + d(t) + s/G(t), where d is measured one-way delay and G is measured goodput. Messages are dropped during outages.

---

## Q1. Classic trace-based emulation and record-and-replay (Noble 1997, Mahimahi, Cellsim/Sprout, Pantheon, NetEm, Mininet, fidelity studies)

### Takeaway
Every canonical trace-replay system checked fidelity the same way. Each ran the same application or benchmark live and under emulation, then reported the % difference in an application-level metric:
- Noble '97: Web, FTP and Andrew benchmark times.
- Mahimahi: page-load time, 12.4% median error.
- Pantheon: throughput and delay, 17% mean error, 13–25% per path.

The recurring limitation is that a replayed trace does not react to the system under test. It has no cross-traffic or feedback coupling: the channel's capacity is fixed by the trace and cannot change in response to what the tested sender does. This is acceptable when the application load is small relative to capacity, and it breaks down for contention MACs such as Wi-Fi and for heavy senders. Noble et al.'s "trace modulation" model (per-interval latency + per-byte bandwidth cost + loss probability) is structurally almost identical to the user's d(t) + s/G(t) model. It is the most direct precedent and should be cited as such.

### Cited Findings

**Noble, Satyanarayanan, Nguyen, Katz, "Trace-based mobile network emulation", ACM SIGCOMM 1997 (pp. 51–61), DOI 10.1145/263105.263140**
- Headline: "trace modulation" re-creates the observed end-to-end characteristics of a real wireless network in a controlled, repeatable way. It is transparent to applications and covers all traffic sent or received by the system under test — [ACM DL](https://dl.acm.org/doi/10.1145/263105.263140); [SIGCOMM'97 PDF](http://conferences.sigcomm.org/sigcomm/1997/papers/p214.pdf)
- Validation: three benchmarks (Web browsing, FTP transfers, Andrew Benchmark), each run live over the wireless network and under modulation. The authors claim trace modulation "accurately reproduces the original wireless environment" and that the technique "is indeed capable of reproducing wireless network performance faithfully" — [ACM DL](https://dl.acm.org/doi/10.1145/263105.263140)
- Method **[unverified, from background knowledge]**:
  - A mobile host runs a known probe workload while traversing a route.
  - The collected trace is "distilled" into a replay trace: a sequence of time-sliced tuples of (latency, per-byte bandwidth cost, loss probability).
  - A kernel modulation layer delays each packet by latency + size × per-byte cost and drops packets with the given probability.
  - The model treats the network as symmetric and derives its parameters from round-trip probe measurements, not from one-way synchronized clocks.
  - Check Sec. 2–3 of the PDF before relying on this.
- Gap: exact live-vs-modulated error percentages were not retrievable from the snippets.

**Netravali, Sivaraman, Das, Goyal, Winstein, Mickens, Balakrishnan, "Mahimahi: Accurate Record-and-Replay for HTTP", USENIX ATC 2015** ([PDF](https://www.usenix.org/system/files/conference/atc15/atc15-paper-netravali.pdf), [ACM DL 10.5555/2813767.2813798](https://dl.acm.org/doi/10.5555/2813767.2813798); an earlier toolkit version appeared in CCR 2014, [DOI 10.1145/2740070.2631455](https://dl.acm.org/doi/10.1145/2740070.2631455))
- Headline: more accurate replay because it emulates the multi-server structure of web pages, which is present in 98% of the Alexa US Top 500 — [Mahimahi PDF](http://mahimahi.mit.edu/mahimahi_atc.pdf)
- Validation metric: error = |% difference| between mean page-load time (over 25 runs) inside the emulator and on the live Internet — [Mahimahi PDF](https://www.usenix.org/system/files/conference/atc15/atc15-paper-netravali.pdf)
- Setup: pages were loaded inside LinkShell with a 5 Mbit/s trace, plus DelayShell with a 100 ms minimum RTT — [Mahimahi PDF](http://mahimahi.mit.edu/mahimahi_atc.pdf)
- Results, per a secondary source (CS244 reproduction blog):
  - Median error: 12.4% for multi-server ReplayShell, 20.5% for single-server ReplayShell, 36.7% for Google web-page-replay.
  - DelayShell/LinkShell add about 0.3% error.
  - Repeat runs on different machines differ by less than 0.5%.
  - Source: [CS244'17 reproduction](https://reproducingnetworkresearch.wordpress.com/2017/06/05/cs244-17-mahimahi-accurate-record-and-replay-for-http/). Verify against the paper's tables.
- LinkShell replays a packet-delivery trace and loops it at the end — [Mahimahi PDF](http://mahimahi.mit.edu/mahimahi_atc.pdf)
- Trace format **[background knowledge]**: one millisecond timestamp per line, each an opportunity to deliver one MTU (1500 B). It is packet-granular, not a fluid model.

**Winstein, Sivaraman, Balakrishnan, "Stochastic Forecasts Achieve High Throughput and Low Delay over Cellular Networks" (Sprout + Cellsim), USENIX NSDI 2013** — [PDF](https://www.usenix.org/system/files/conference/nsdi13/nsdi13-final113.pdf)
- Recording ("Saturator"): saturates the uplink and downlink with MTU-sized packets so the base station always has data queued, and records each packet's arrival time. The timestamps form a packet-delivery-opportunity trace — [Sprout PDF](https://www.usenix.org/system/files/conference/nsdi13/nsdi13-final113.pdf)
- Replay ("Cellsim"):
  - A PC with two Ethernet interfaces delays packets by a configurable propagation delay, then enqueues them.
  - At each delivery instant in the trace, it dequeues one MTU's worth of bytes, accounted byte by byte.
  - Source: [Sprout PDF](https://www.usenix.org/system/files/conference/nsdi13/nsdi13-final113.pdf)
- Headline: traces from Verizon LTE/3G, AT&T LTE and T-Mobile 3G. Compared with Skype, Hangout, Facetime and TCP variants, Sprout cut self-inflicted delay 7.9× at 2.2× the bit rate on average — [Sprout PDF](https://www.usenix.org/system/files/conference/nsdi13/nsdi13-final113.pdf)
- Fidelity: I found no self-contained live-vs-Cellsim validation in the search results. Later work questions the saturation premise: "saturating the network fundamentally alters the network's behavior" — [NeuralEmu, arXiv 2604.26080](https://arxiv.org/pdf/2604.26080)

**Yan, Ma, Hong, Dong, Winstein et al., "Pantheon: the training ground for Internet congestion-control research", USENIX ATC 2018** — [PDF](https://www.usenix.org/system/files/conference/atc18/atc18-yan-francis.pdf); [mirror](https://pantheon.stanford.edu/static/pantheon/documents/pantheon-paper.pdf)
- Headline: a shared testbed of real Internet paths plus "calibrated emulators". These are low-parameter Mahimahi-style emulators fitted to each real path.
- Emulator parameters: bottleneck link rate, propagation delay, queue size and random loss only. Jitter, reordering and other detailed mechanisms are deliberately not modeled — [Pantheon PDF](https://pantheon.stanford.edu/static/pantheon/documents/pantheon-paper.pdf)
- Fidelity claim: each scheme's throughput and delay on the emulator falls within about 17% of the real-path values on average — [Pantheon FAQ](https://pantheon.stanford.edu/faq); [Pantheon paper](https://www.usenix.org/system/files/conference/atc18/atc18-yan-francis.pdf). An earlier FAQ wording says "within 20%".
- Per-path errors (paper table, via search snippet):
  - Nepal→AWS India (Wi-Fi, 1 flow): 19.1%
  - AWS Brazil→Colombia (cellular): 13.0%
  - Mexico→AWS California (cellular): 25.1%
  - Source: [Pantheon PDF](https://fyy.cs.illinois.edu/documents/pantheon-atc18.pdf). Verify against the table.
- Emulators for all-wired paths are more faithful than those for partly-wireless paths. Leave-one-out cross-validation (fit on n−1 schemes, predict the held-out one) gives about 20% error — [Pantheon PDF](https://pantheon.stanford.edu/static/pantheon/documents/pantheon-paper.pdf)
- Calibration procedure **[unverified, from background knowledge]**: parameters were searched with Bayesian optimization to minimize the mean throughput/delay error across the set of congestion-control schemes. The emulators are "replicas" of a path's aggregate behaviour. They do not model the path's cross traffic explicitly.

**NetEm (Hemminger, "Network Emulation with NetEm", linux.conf.au 2005)**
- **[unverified, from background knowledge]**: a Linux qdisc that adds delay (with jitter and distributions), loss, duplication, corruption and reordering, and later rate limiting. It uses statistical parameters, not traces.
- MoonEm (CoNEXT 2025) states that the performance limitations of tools like NetEm "can alter network measurements" — [TU Berlin entry](https://www.tkn.tu-berlin.de/bib/lachnit2025moonem/)
- No primary-source accuracy study of NetEm was retrieved.

**Mininet / Mininet-HiFi (Handigol, Heller, Jeyakumar, Lantz, McKeown, "Reproducible network experiments using container-based emulation", ACM CoNEXT 2012, DOI 10.1145/2413176.2413206)**
- Premise: the performance fidelity of container-based emulation was "largely unproven" — [Academia mirror](https://www.academia.edu/111159861/Reproducible_network_experiments_using_container_based_emulation); [Crossref](https://api.crossref.org/works/10.1145%2F2413176.2413206)
- Validation: reproduced published results (DCTCP, Hedera, router buffer sizing) on Mininet-HiFi and compared them against the original papers' hardware results — [Bonaventure blog](https://perso.uclouvain.be/olivier.bonaventure/blog/html/2012/12/14/mininet.html); [Academia](https://www.academia.edu/111159861/Reproducible_network_experiments_using_container_based_emulation)
- Known limitation: shared host resources can violate timing realism through contention, scheduling serialization and background load — [TUM 2025](https://www.net.in.tum.de/fileadmin/TUM/NET/NET-2025-05-1/NET-2025-05-1_08.pdf)

**Trace-driven emulation fidelity studies**
- "The Challenges of Trace-Driven Wi-Fi Emulation" (arXiv 2002.03905) — [arXiv](https://arxiv.org/html/2002.03905)
  - Checks whether Saturator-style recording fills the link: the throughput/capacity ratio is mostly close to 1.
  - Saturation recording fails when the feedback path shares the saturated interface, because the queueing delay it adds becomes significant.
  - **[background knowledge]**: the paper's broader thesis is that Wi-Fi capacity depends on the offered load and contention. A fixed delivery-opportunity trace therefore cannot capture the channel's reaction to the tested sender.
- iBox (Ashok et al., HotNets 2020, DOI 10.1145/3422604.3425935) — [ACM DL](https://dl.acm.org/doi/abs/10.1145/3422604.3425935)
  - **[background knowledge]**: learns path models from traces that include a cross-traffic estimate. The motivation is that pure replay ignores how cross traffic reacts.
- NeuralEmu (arXiv 2604.26080, 2026): ML-based, measurement-driven 5G emulation, motivated by the claim that saturation-based record-and-replay distorts the network — [arXiv](https://arxiv.org/pdf/2604.26080)

### Inferences
- **Direct precedent.** The user's d(t) + s/G(t) model is a per-message fluid version of Noble's latency + per-byte-cost + loss model. Mahimahi and Cellsim are packet-level (MTU delivery opportunities, queue). Differences worth stating explicitly:
  - d(t) is measured **one-way with synchronized clocks**, whereas Noble's parameters are derived from round trips.
  - G is measured goodput, not saturated capacity.
  - The model works at message granularity (whole perception messages) and has an outage/drop rule.
  - No queue is modeled. Back-to-back messages do not queue behind each other unless G(t) is shared across messages; the paper should say whether it is.
- **Validation template the field expects.** Run the same application live and under replay, then report the % error of an application-level metric:
  - collaborative perception AP, or end-to-end message age, or the fraction of messages fused within the deadline;
  - include per-condition breakdowns, as in Pantheon's per-path table;
  - include a leave-one-out style check, as in Pantheon.

  Anything at or below the field's numbers (Mahimahi about 12% median page-load-time error, Pantheon about 17% mean throughput/delay error) is a defensible bar.
- **Limitations the paper should pre-empt:**
  - (a) Fluid vs packet: no MTU segmentation, no head-of-line blocking or queue build-up between consecutive messages.
  - (b) No feedback: the replayed channel does not react to the tested message rate or size. This matters if G was measured under a different offered load, as in the Wi-Fi emulation critique.
  - (c) No cross-traffic coupling.
  - (d) Goodput measured with one traffic pattern may not transfer to other message sizes, because small messages are latency-dominated.

  Measuring G with the same message sizes and rate as the replayed application partially mitigates (b) and (d).

### Gaps
- Exact live-vs-modulated numbers for Noble '97, and exact Pantheon calibration objective/optimizer: PDFs blocked. Retrieve from [SIGCOMM PDF](http://conferences.sigcomm.org/sigcomm/1997/papers/p214.pdf) and Pantheon Sec. 4.
- Mahimahi error figures come from a secondary reproduction source; verify against ATC'15 Sec. 4.
- No primary NetEm accuracy study was found.

---

## Q2. Synthetic channels in collaborative perception (V2X-ViT, Where2comm, SyncNet, CoBEVFlow, lossy-communication works)

### Takeaway
Every major collaborative-perception paper models the channel synthetically, with no measured link:
- **V2X-ViT**: a constant 100 ms delay (quantized to one 10 Hz frame) plus Gaussian pose noise with σ = 0.2 m on x/y/z and 0.2° on yaw only.
- **Where2comm**: models bandwidth only, as log2(bytes) communication volume. It has no latency model, and pose noise is swept over σ ∈ [0, 0.6] m.
- **SyncNet**: per-agent integer frame lags.
- **CoBEVFlow**: Binomial(n = 10, p) frame lags.
- **Lossy-communication papers**: random feature/packet corruption.

Latency in all of these is either constant or i.i.d. random and quantized to sensor frames. None ties delay to message size and a time-varying measured rate, and none models outages from real traces.

### Cited Findings

**V2X-ViT (Xu, Xiang, Tu, Xia, Yang, Ma), ECCV 2022** — [arXiv 2203.10638](https://arxiv.org/pdf/2203.10638); [ECVA PDF](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136990106.pdf); official code [DerrickXuNu/v2x-vit](https://github.com/DerrickXuNu/v2x-vit)
- **Exact config (verified in repo `v2xvit/hypes_yaml/point_pillar_v2xvit.yaml`, `wild_setting`):**
  ```
  async_mode: 'sim'
  async_overhead: 100
  seed: 25
  xyz_std: 0.2
  ryp_std: 0.2
  data_size: 1.06   # Mb
  transmission_speed: 27   # Mbps
  backbone_delay: 10   # ms
  ```
  The `async` and `loc_err` flags default to false; they are switched on for the "Noisy Setting". Same values in [OpenCOOD](https://github.com/DerrickXuNu/OpenCOOD) `opencood/hypes_yaml/point_pillar_v2xvit.yaml`.
- **Delay model (repo `data_utils/datasets/basedataset.py`, `time_delay_calculation`):**
  - `'sim'` mode: the delay is a constant `async_overhead` (100 ms).
  - `'real'` mode: delay = U(0, async_overhead) + data_size/transmission_speed × 1000 + backbone_delay. With the values above that is U(0,100) + about 39 ms + 10 ms.
  - Either way the delay is then integer-divided by 100. The code comment says "the data is 10 hz for both opv2v and v2x-set".
  - Net effect: 100 ms becomes exactly one stale frame for each non-ego agent. The ego has zero delay.
  - Note: the "1.06 Mb / 27 Mbps" transmission term is a fixed message-size / fixed-rate fluid term, a constant analogue of s/G.
  - Source: [v2x-vit repo](https://github.com/DerrickXuNu/v2x-vit); [OpenCOOD repo](https://github.com/DerrickXuNu/OpenCOOD)
- **Pose-noise model (`add_loc_noise`):** Gaussian N(0, 0.2 m) added to x, y, z, and N(0, 0.2°) added to **yaw only**. Roll and pitch are left unchanged even though the parameter is called `ryp_std`. A fixed seed is used — [OpenCOOD basedataset.py](https://github.com/DerrickXuNu/OpenCOOD)
- **Paper claims:**
  - Under the Noisy Setting, V2X-ViT drops less than 5% AP@0.5 and less than 10% AP@0.7 relative to the Perfect Setting.
  - The pose-error sweep covers σ_xyz ∈ [0, 0.5] m and σ_heading ∈ [0°, 1.0°]. In the "normal range" (σ_xyz ≤ 0.2 m, σ_heading ≤ 0.4°) the drop is less than 3%.
  - Source: [arXiv 2203.10638](https://arxiv.org/pdf/2203.10638) (via search snippet)
- **Uncertainty to flag:**
  - The text statement "std 0.2 m / 0.2°, 100 ms delay" for the Noisy Setting is confirmed by the released config, but the exact section/table wording could not be read. **[background knowledge]**: it is stated in Sec. 4.1 (Experimental setup), and Table 2 reports Perfect vs Noisy results. The delay ablation figure sweeps 0–400 ms.
  - A survey reports V2X-ViT degrades less than 4% per additional 100 ms on V2XSet — [survey arXiv 2308.16714](https://arxiv.org/pdf/2308.16714)
  - V2XSet communication range (70 m) is **[background knowledge]**.

**Where2comm (Hu, Fang, Li, Chen, Chen), NeurIPS 2022** — [arXiv 2209.12836](https://arxiv.org/pdf/2209.12836); [NeurIPS PDF](https://papers.neurips.cc/paper_files/paper/2022/file/1f5c5cd01b864d53cc5fa0a3472e152e-Paper-Conference.pdf); official code [MediaBrain-SJTU/Where2comm](https://github.com/MediaBrain-SJTU/Where2comm)
- **Bandwidth model:**
  - Communication volume follows DiscoNet's metric but with log base 2, which is about 3.32× the base-10 number. It is defined as "the message size by byte in log scale with base 2" (appendix) — [arXiv 2209.12836](https://arxiv.org/pdf/2209.12836)
  - The trade-off is controlled by thresholding a spatial confidence map. In code, `thre` (example 0.01) is applied after an optional Gaussian smoothing (k_size 5, c_sigma 1.0), and `communication_rate` = fraction of the H×W BEV cells sent — [repo `opencood/models/comm_modules/where2comm.py`](https://github.com/MediaBrain-SJTU/Where2comm)
  - There is no rate-over-time or bandwidth-in-Mbps model. "Bandwidth" means a per-frame message-size budget.
- **Pose noise:**
  - Follows the V2VNet / V2X-ViT setting: Gaussian with mean 0 and std 0 to 0.6 m on all three datasets. Claimed more robust than prior SOTA because intermediate features have low spatial resolution — [arXiv 2209.12836](https://arxiv.org/pdf/2209.12836)
  - Repo DAIR config: `noise_setting: add_noise: False, pos_std: 0.2, rot_std: 0.2` — [repo](https://github.com/MediaBrain-SJTU/Where2comm)
- **Latency:** no dedicated latency experiment in the NeurIPS paper. Latency is named as a realistic issue and left to future work (a prediction module) — [arXiv 2209.12836](https://arxiv.org/pdf/2209.12836)
- **Third-party latency result:** CoRA reports Where2comm AP@0.5 falling from 0.7629 at 0 ms to 0.4322 at 400 ms — [CoRA arXiv 2512.13191](https://arxiv.org/pdf/2512.13191). This is not the authors' own result.
- **Exact section/figure numbers:** not retrieved (PDF blocked). **[background knowledge]**: the pose-robustness figure is in Sec. 5 experiments, and the communication-volume definition is in the appendix.

**SyncNet / "Latency-Aware Collaborative Perception" (Lei, Ren, Hu, Zhang, Chen), ECCV 2022** — official code [MediaBrain-SJTU/SyncNet](https://github.com/MediaBrain-SJTU/SyncNet)
- Billed as the first latency-aware collaborative perception system. It aligns asynchronous features to a common timestamp via feature-attention symbiotic estimation and time modulation.
- Claimed +15.6% over SOTA in the latency scenario, and it stays above single-agent perception "under severe latency" — [SyncNet README](https://github.com/MediaBrain-SJTU/SyncNet)
- **Latency model (verified in repo):** `--tau` is an integer per-agent latency in **frames** on V2X-Sim:
  - default for training: `[2,0,2,2,2,2]`, ego = 0;
  - the test script uses `--tau 5 0 5 5 5 5`;
  - a curriculum in `train_LAcodet.py` raises tau gradually and clamps it to 1–10.
  - Delays are constant and the same for every non-ego agent within a run, with no randomness or size dependence — [repo `LA-det/*.py`, `test_LA.sh`](https://github.com/MediaBrain-SJTU/SyncNet)
- Conversion of tau to ms depends on V2X-Sim's frame rate. **[unverified]** whether this is 5 Hz (200 ms/frame) or 10 Hz.

**CoBEVFlow (Wei, Wei, Chen et al.), NeurIPS 2023** — [arXiv 2309.16940](https://arxiv.org/abs/2309.16940); [NeurIPS PDF](https://proceedings.neurips.cc/paper_files/paper/2023/file/5a829e299ebc1c1615ddb09e98fb6ce8-Paper-Conference.pdf); official code [MediaBrain-SJTU/CoBEVFlow](https://github.com/MediaBrain-SJTU/CoBEVFlow)
- Targets asynchrony from communication delays, interruptions and clock misalignment. A BEV flow field moves features to where they should be at ego time, with no time discretization of the fusion. Introduces the IRV2V synthetic dataset; also evaluated on DAIR-V2X — [arXiv 2309.16940](https://arxiv.org/abs/2309.16940)
- **Delay model (verified in repo `intermediate_fusion_dataset_dair_irregular.py`):**
  - Each collaborator's lag = sum of `binomial_n` Bernoulli(`binomial_p`) trials, i.e. Binomial(n = 10, p) **frames**. Expected delay = n·p frames, e.g. p = 0.3 gives 3 frames ≈ 300 ms at 10 Hz.
  - Configs use `binomial_n: 10` with `binomial_p` ∈ {0, 0.3, …}.
  - A separate `time_delay:` integer appears in the baseline configs.
  - Source: [CoBEVFlow repo](https://github.com/MediaBrain-SJTU/CoBEVFlow)
- A later paper reports CoBEVFlow at expected intervals of 0/300/500 ms on DAIR-V2X, V2XSet and OPV2V — [CoDynTrust arXiv 2502.08169](https://arxiv.org/pdf/2502.08169)

**Lossy / interrupted communication works**
- Li et al., "Learning for Vehicle-to-Vehicle Cooperative Perception under Lossy Communication" — [arXiv 2212.08273](https://arxiv.org/pdf/2212.08273); IEEE T-IV 2023 **[venue from background knowledge]**
  - Evaluated on OPV2V (CARLA). Proposes an LC-aware Repair Network (encoder-decoder with skip connections) and a V2V attention module.
  - Attributes loss to signal fluctuation from obstacles, interference and tampering.
  - The exact loss model was not retrieved. **[background knowledge]**: it is applied as random corruption of received feature maps.
  - Code repo [jinlong17/V2VLC](https://github.com/jinlong17/V2VLC) contains only a README ("code is coming soon"), so the loss model cannot be checked in code.
- "Interruption-Aware Cooperative Perception for V2X Communication-Aided Autonomous Driving" (V2X-INCOP) — [arXiv 2304.11821](https://arxiv.org/html/2304.11821v2). Synthetic communication interruptions.
- LR-V2X (arXiv 2610.11264): models feature-level packet loss at the spatial-packet level with up to 90% loss, on DAIR-V2X and V2X-Real. Loss is simulated, not recorded — [arXiv 2610.11264](https://arxiv.org/html/2610.11264)

### Inferences
- Gap the testbed fills: no surveyed collaborative-perception method uses (a) measured one-way delay, (b) delay that grows with message size through a measured time-varying goodput, or (c) trace-driven outages.
  - V2X-ViT's `'real'` mode is the closest analogue. Its delay term is a *constant* fluid model, s/G with s = 1.06 Mb and G = 27 Mbps, plus uniform overhead. The paper's headline results use `'sim'` mode, a constant 100 ms.
  - Frame quantization (//100 ms) means any delay under 100 ms is effectively zero in these benchmarks. A continuous-time arrival rule exposes this artifact.
- Where2comm's "bandwidth" is a size budget, not a channel. Under s/G(t) replay, Where2comm's sparsity should translate into lower *latency*, which its own paper never measures.

### Gaps
- Exact section/table numbers for V2X-ViT and Where2comm could not be read from the PDFs; the parameter values are confirmed from the official configs/code.
- SyncNet frame-rate-to-ms conversion: unverified.
- Exact loss model in Li et al.: not retrieved, and no released code.

---

## Q3. Sensing-plus-communication (ISAC) and cooperative-perception datasets with measured links

### Takeaway
The ISAC datasets (DeepSense 6G, ViWi) pair sensors with **PHY-layer** radio measurements such as mmWave beam power vectors. Their target tasks are beam prediction and blockage prediction, not end-to-end message delay or goodput for multi-agent perception. Real cooperative-perception datasets (DAIR-V2X, V2X-Real) carry no link measurements.

The closest prior work is **CooperScene** (UC Riverside, arXiv 2606.31219, 2026). It records synchronized C-V2X latency, throughput, packet-loss rate and jitter alongside multi-agent LiDAR/camera. Its benchmark appears to estimate transmission latency from message size and measured throughput, which overlaps with the user's s/G(t) idea. Positioning must address it directly.

### Cited Findings

**DeepSense 6G (Alkhateeb et al.)** — [IEEE DataPort](https://ieee-dataport.org/documents/deepsense-6g-large-scale-real-world-multi-modal-sensing-and-communication-dataset); [ResearchGate](https://www.researchgate.net/publication/371325115_DeepSense_6G_A_Large-Scale_Real-World_Multi-Modal_Sensing_and_Communication_Dataset)
- Real-world, 40+ scenarios across locations and times of day. Time-synchronized mmWave and sub-6 GHz RF, LiDAR, cameras, radar and GPS. Covers V2V, V2I, drone and fixed settings — [ResearchGate fig.](https://www.researchgate.net/figure/The-DeepSense-6G-dataset-comprises-40-scenarios-with-realistic-datasets-to-encourage-the_fig2_371325115); [EmergentMind summary](https://www.emergentmind.com/topics/deepsense6g-dataset)
- Typical use: beam prediction. For example, scenarios 31–34 have a roadside unit with an RGB camera, radar, LiDAR and a mmWave receiver; labels are the optimal index in a 64-beam codebook — [arXiv 2609.10200](https://arxiv.org/pdf/2609.10200); [arXiv 2506.22469](https://arxiv.org/pdf/2506.22469)
- Extension: DeepSense-V2V — [ResearchGate](https://www.researchgate.net/publication/381736480_DeepSense-V2V_A_Vehicle-to-Vehicle_Multi-Modal_Sensing_Localization_and_Communications_Dataset)
- The IEEE Communications Magazine 2023 venue is **[background knowledge; not verified in this session]**. Only the DataPort entry was found.

**ViWi (Vision-Wireless)**
- A synthetic multi-modal dataset (communication + visual + LiDAR). Lacks UAV scenarios — [survey arXiv 2601.03181](https://arxiv.org/pdf/2601.03181)
- **[background knowledge]**: Alrabeiah et al., IEEE VTC 2020-Spring; ray-tracing plus game-engine rendering.

**CooperScene (UC Riverside; Roy-Chowdhury group), arXiv 2606.31219, 2026** — [arXiv](https://arxiv.org/html/2606.31219); [project page](https://cisl.ucr.edu/CooperScene/)
- 3 vehicles + 1 infrastructure unit; 59K synchronized LiDAR frames, 53K images, 344K 3D boxes — [project page](https://cisl.ucr.edu/CooperScene/)
- Each scene includes synchronized C-V2X measurements (latency, throughput, PLR, jitter) alongside LiDAR, camera and GNSS-RTK. The authors claim no existing dataset records real-world C-V2X throughput — [arXiv](https://arxiv.org/html/2606.31219)
- Benchmark: perception accuracy under measured C-V2X vs an unlimited-network baseline, for 3D detection and motion prediction.
- Reported example: V2VNet's estimated transmission latency is 64,349 ms (vehicle–infrastructure) and 107,432 ms (full cooperation). This implies latency was estimated from message size / measured throughput.
- Uses PTP and spatial-temporal ICP for sync and calibration.
- Source: [arXiv](https://arxiv.org/pdf/2606.31219); [Pith](https://pith.science/paper/2606.31219)
- Snippet-reported link figures: about 38% PLR, about 31 ms latency, about 0.72 Mbps throughput. Flagged by the search tool as hard to read; verify against the paper.
- Whether CooperScene re-times individual messages and drops them during outages, versus applying aggregate estimates, is **unknown** (full text not read).

**SEE-V2X (UC Riverside)**: a C-V2X direct-communication trace dataset (throughput, PLR, latency, jitter; indoor and parking lot) with **no perception data** — [project page](https://cisl.ucr.edu/SEE-V2X/)

**V2X-ReaLO (arXiv 2503.10034)**: an online framework on V2X-Real with synchronized ROS bags, giving real-time evaluation of bandwidth, latency and accuracy. Per the search summary, latency is injected/emulated rather than recorded as a separate network trace — [arXiv](https://arxiv.org/pdf/2503.10034)

**V2X-Real (Xiang et al., ECCV 2024)** — [arXiv 2403.16034](https://arxiv.org/abs/2403.16034)
- **[background knowledge; not verified]**: 2 CAVs + 2 smart-infrastructure units; about 33K LiDAR frames, about 171K camera images, about 1.2M 3D boxes.
- It contains no measured link metrics. Search results confirm that DAIR-V2X and V2X-Real lack measured link metrics and that loss experiments on them are simulated — [LR-V2X](https://arxiv.org/html/2610.11264)

**Other measured-link work**
- "Cooperative Perception Using V2X Communications: An Experimental Study" (Univ. of Luxembourg) — [ORBilu PDF](https://orbilu.uni.lu/bitstream/10993/63217/1/Cooperative_Perception_Using_V2X_Communications_An_Experimental_Study.pdf). This is an experimental V2X cooperative-perception study; dataset release status unknown.
- A survey notes that few datasets annotate PQoS indicators jointly with perception labels — [arXiv 2512.00490](https://arxiv.org/pdf/2512.00490)

### Inferences
- **Positioning against ISAC datasets:** they answer "can sensing predict the radio channel?" (beam/blockage). The testbed answers "what does the measured channel do to multi-agent perception?" These are complementary layers: PHY measurements versus application-level d(t)/G(t) message timing.
- **Positioning against CooperScene, the main threat to novelty:**
  - Differentiate on: (1) per-message continuous-time replay with outage drops versus aggregate latency estimates; (2) one-way delay from synchronized clocks; (3) validation of replay against live closed-loop runs.
  - The third point is a fidelity check that, in the material retrieved, none of the collaborative-perception datasets reports. It is also the standard set by Noble, Mahimahi and Pantheon.
  - Confirm CooperScene's exact replay mechanism before claiming (1).
- No dataset found pairs a measured robot/MIMO CSI link with multi-agent perception for *ground robots*, as opposed to vehicles. CSI-with-robots datasets were not found in this search budget (see Gaps).

### Gaps
- MIMO/CSI datasets collected with mobile robots (e.g., robot-mounted CSI collections) were not searched within budget, so whether any pairs CSI with multi-robot perception is unconfirmed.
- CooperScene's full results tables and replay mechanics were not read (PDF blocked). Its arXiv metadata/date should be checked.
- DeepSense 6G Comm Mag 2023 citation details (vol/issue/DOI) and V2X-Real statistics were not verified in this session.
