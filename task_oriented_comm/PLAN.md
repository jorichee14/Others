# Task-oriented, timely communication for an indoor robot team on SCooP

The question: given the SCooP data, which communication-aware direction is
both new and timely? The candidates are compression, semantic communication,
token communication, and approaches beyond the channel itself. This file
states the open problem in the literature, assesses each family against what
SCooP measured, and ranks the options. Each option comes with a first
experiment and a decision rule fixed before any data are read.

The theoretical foundation, the formal results and the novelty claims are
in [`THEORY.md`](THEORY.md).

Literature tags follow `temporal_messaging/RELATED_WORK.md`. **[S]** means
the claim comes from abstracts or search summaries only. **[R]** means it
comes from this repository's own measurements. Read the primary text before
citing any [S] claim.

---

## 1. Recommendation

**Predict the link, then fit the message to the deadline.** The sender looks
ahead at the delivery time its message will face. It predicts that time from
the map (Fresnel clearance), the CSI, and the contention state. It then sends
a progressive, task-ordered message, sized so that the part which lands
before the receiver's decision time carries the most task value. The
objective is expected task value at decision time, not distortion at a fixed
rate and not average Age of Information (AoI).

This merges options O1 (deadline-matched progressive messages) and O2 (a
predicted link) below, scored with the metric of O3 (value at decision
time). Each option is one-third of the method, and each is published only in
isolation. Only SCooP can evaluate the combination, because only SCooP logs
the link on the sensor clock and in the map frame.

---

## 2. What SCooP measured that matters here [R]

| Fact | Value | Source |
|---|---|---|
| 1.5 MB messages that land within one 100 ms LiDAR frame | 86 % with one robot (survey_1); 59–68 % with two robots (coop_2) | `sim_gap.py` |
| Same messages under the synthetic channels the CP literature uses | 100 % (V2X-ViT constant 100 ms; Where2comm has no delay) | `sim_gap.py` |
| RTT, idle to loaded | 5–10 to 25–33 ms (24 Sep); 3.3 to 60–109 ms (28 Aug) | `wifi_analysis.py` |
| Single-stream capacity per robot | 275–282 Mbit/s up, 598–676 Mbit/s down | `wifi_analysis.py` |
| Real outages | 3, covering 12 % of frames (coop2_0828) | |
| Map predicts the radio | Spearman ρ 0.60, 5.4 dB clear-minus-obstructed (mobile_2 link); ρ 0.23 on infra_1 | `link_clearance.py` |
| CSI rate | 104–141 Hz, valid up to about 1.2 m/s relative speed | `csi_analysis.py` |
| Clock | ≤ 0.99 ms measured offset, so one-way delay is observable | `ntp_analysis.py` |
| Agent speed | about 1 m/s | E1 headroom |

From the parent OPV2V study (`temporal_messaging/HANDOFF.md`):

- At 100 ms latency, every method falls below ego-only. Packet loss never
  does.
- Latency damage there is displacement: ΔAP@0.7 is 3.9–8.8× ΔAP@0.5.
- Bandwidth is free down to 4 bits.

**Implication.** The RTT sits well inside the frame, yet a third of large
messages miss it. Message size times contended goodput, not latency, decides
whether a message lands. The link indicators the literature optimizes do not
predict the task outcome. This is already contribution 4 of the SCooP-iT
paper.

**Hypothesis H0 (to check, section 6, step 0).** Indoors at about 1 m/s, a
100 ms delay moves an object about 0.1 m. That is below the 0.2 m pose
breaking point, but it is close to a person's IoU-0.7 tolerance (0.11 m).
`THEORY.md` §2.5 makes this precise with a displacement number Π per class
and IoU threshold. Uncorrected indoor people sit near Π ≈ 1. With a
velocity tag in the first unit of every message, they sit at Π ≈ 0.2, and
the displacement mechanism that dominated OPV2V at 20 m/s becomes
second-order. What costs the task is then whether, and how much of, a
message arrives in time. That moves the problem away from latency
compensation, where the vehicle literature is crowded, and toward
delivery-time-aware transmission.

---

## 3. The problem in the literature

Four communities each own one piece of timely collaborative perception, and
each abstracts away the pieces the others own.

| Community | What it optimizes | What it assumes about the link | What it ignores |
|---|---|---|---|
| **Bandwidth-efficient CP** (Where2comm, CodeFilling, CoSDH, InfoCom, RDComm, CoPLOT) | Task accuracy per bit; RDComm builds a pragmatic rate-distortion theory [S] | A per-frame bit budget, with no delay or a constant delay | That delivery time depends on size, and that the budget changes from frame to frame |
| **Semantic / task-oriented comm for CP** (SComCP, CoDS, R-ACP, SimHARQ, the domain-generalized SemCom paper) | Robustness of the feature to the PHY channel: SNR, fading, HARQ rounds [S] | AWGN or Rayleigh channels at the symbol level, single link, no medium-access contention | That on Wi-Fi the loss is at the MAC and queue (frame misses and outages), not bit errors |
| **AoI / timeliness scheduling** (Fresh2comm, TAMP 2601.04542, Update the Unseen Only, AoI-guaranteed scheduling, WiSwarm) | Average age, or an empirical age-to-AP penalty, through Lyapunov, Whittle or online learning [S] | Atomic messages on modeled or orthogonal resources | What the message contains; age is content-blind, so a static wall and a person turning a corner age alike |
| **Latency compensation** (SyncNet, CoBEVFlow, FFNet, TraF-Align, CoDynTrust, SparseCoop, Temporal Residual Bottleneck) | Accuracy given a delayed message, by receiver-side flow or sender-side prediction | Constant or Binomial frame lags, applied uniformly; the message always arrives whole | Choosing what to send; vehicle speeds where displacement dominates |
| **Token communication** (TokenCom, UniToCom, emergent VLM tokens) | Generative reconstruction with large-model context [S] | PHY-level channel; no deadline | Real-time perception tasks and multi-agent contention |

**The gap in one sentence.** No method chooses what to send, and how much,
from the delivery time it will actually face on a shared, contended link.
None of them can be checked against a real link either, because no dataset
logs one alongside the sensors indoors. CooperScene logs C-V2X outdoors for
vehicles only.

**The formal problem.** At time t, sender i chooses a message m from a
progressive representation, either a prefix of task-ordered units or none.
m reaches receiver j at the delivery time D, distributed as
p(D | |m|, link state x_t, the others' traffic). Receiver j decides at time
τ_j. The sender maximizes

    E_D[ V_j( prefix of m delivered by τ_j , τ_j − t ) ]  −  λ·airtime(m)

Here V is task value: detection gain over ego-only, or a hazard warning
received before the robot's decision point. Three choices make this new:

1. The value depends on how much of the message lands by the deadline, not
   on whether all of it lands.
2. p(D | ·) is predicted from map, CSI and contention, not assumed.
3. V is scored at the receiver's decision time, not as age.

Each published piece fixes the other two.

---

## 4. Assessment of each family against SCooP

| Family | Promise on SCooP | Why | Verdict |
|---|---|---|---|
| **Pure compression** (quantization, VQ, codebooks) | Low as a headline | The bit-depth floor is already measured (free to 4 bits), and the field is saturated (RDComm, InfoCom, QuantV2X, ReVQom). Smaller messages miss fewer frames, but that is an operating point, not a contribution | Use as a component: the progressive codec inside O1 |
| **Semantic comm (PHY-level, JSCC)** | Low on this testbed | SCooP's link is commodity Wi-Fi with MAC retransmissions. The receiver sees whole packets late or lost, never noisy symbols, so JSCC's graceful-degradation argument has no channel to act on. CSI is logged, but the PHY cannot be changed | Do not pursue on SCooP |
| **Token communication** | Medium as a unit, low as a headline | Tokens are a natural ordered, prefix-decodable unit, which is what O1 needs. LLM-driven TokenCom targets generative reconstruction, not 100 ms detection. Instance tokens for CP already exist (CoPLOT, SparseCoop queries) | Use tokens as the message unit in O1; do not lead with "TokenCom" |
| **AoI scheduling** | Medium | Directly about timeliness, but content-blind and evaluated on synthetic channels. SCooP can show where AoI and task value disagree | Fold into O3 as the baseline to beat |
| **Latency compensation** | Low indoors if H0 holds | At about 1 m/s the displacement is under 0.2 m; the vehicle-speed case is crowded (section 3) | Keep as the receiver's cheap kinematic correction; not a contribution |
| **Beyond comms: link prediction from map and CSI (ISAC)** | High, unique to SCooP | The map predicts RSSI (ρ 0.60). CSI is logged at 104–141 Hz on the sensor clock. Radio-map work in robotics (FERMI, communication maps) drives motion planning, not message selection [S] | Core of O2 |
| **Beyond comms: motion (relay, connectivity-aware paths)** | Medium, later | Needs the closed-loop twin and a positional link, neither of which exists yet | Extension (`RelayPlacement-v0`) |

---

## 5. Options, ranked

### O1. Deadline-matched progressive messages (an "anytime" message)

- **Idea.** The sender encodes into an ordered sequence of units (tokens,
  instance queries, or residual-VQ layers), most task value per byte first:
  a box and velocity, then the occluded-region features, then the rest. It
  sends the prefix that its predicted goodput can deliver by τ_j. The
  receiver fuses whatever prefix has arrived at its frame boundary. Training
  uses nested dropout (Matryoshka style) over the units, so every prefix is
  decodable.
- **Why new.** Progressive pieces exist without a deadline: SimHARQ
  revisions, WaveComm wavelet levels, InfoCom progressive decoding [S]. No
  paper found sizes a prefix-decodable CP message to a predicted delivery
  time. Check "Matryoshka" + "collaborative perception" before claiming.
- **Why SCooP.** The measured cost of a 1.5 MB message is a 14–41 % miss
  rate. A prefix policy converts those misses into partial value.
- **Risk.** It needs retraining of an intermediate-fusion model, and labels
  for coop_2 are not done yet.

### O2. A predicted link: map + CSI + contention → delivery-time distribution

- **Idea.** Predict goodput G and latency d over the next 0.1–1 s along the
  agent's known path. The predictor uses clearance along the planned path
  from the reference map, recent CSI power and Doppler, and the other
  robot's traffic. The prediction drives O1's prefix size. It also triggers
  prefetching: send occluded-region content before entering a shadow or an
  outage.
- **Why new.** CP schedulers use current CSI (SchedCP) or AoI [S]. Radio
  maps in robotics plan motion. Nobody has used a map-predicted link to
  choose perception payloads, and no other dataset can test it, because it
  needs link, pose and map on one clock and frame.
- **Why SCooP.** E2 already shows that the signal exists (ρ 0.60).
- **Risk.** ρ 0.60 is moderate, and infra_1 fails (ρ 0.23). The predictor
  must beat "persistence" (the last measured value), out of run.

### O3. Value at decision time instead of age (the task metric)

- **Idea.** Score a message by the task value it delivers at the receiver's
  decision time, after the receiver's kinematic correction. The value is
  high for content the receiver cannot sense itself (the occluded region, a
  person about to emerge) and near zero for a static map it already has.
  Schedule by this value, using the Whittle index or a Lyapunov drift
  formulation, with AoI and TAMP as baselines.
- **Why new.** AoII and Urgency of Information add semantics in theory [S].
  "Update the Unseen Only" uses visibility [S]. Neither is evaluated against
  a real link or an event task with a deadline.
- **Why SCooP.** `OcclusionHandoff-v0` (a mast warns a robot of a person
  behind a corner) is exactly a decision-time task.
- **Risk.** Alone, it is a metric-plus-scheduler paper close to TAMP. It is
  stronger as the objective of O1 and O2 than as a standalone contribution.

### O4. Contention-aware coordination between agents sharing an AP

- **Idea.** When two robots share an AP, one robot's payload changes the
  other's delay. The 86 % to 59–68 % drop is that interaction. A joint
  policy (multi-agent RL in `CommSchedule-v0` with `,queue`, or a simple
  token-passing rule) chooses sizes jointly.
- **Why new.** CP scheduling papers allocate orthogonal resources from a
  base station. CSMA contention is not modeled [S].
- **Risk.** A logged trace cannot replay a counterfactual: the other robot's
  load is baked into the measurements. That needs a contention model fitted
  across survey_1 (alone) and coop_2 (shared), plus a live check. This is
  the weakest link in the evidence chain, so O4 is ranked last and treated
  as an extension.

### O5. LLM / VLM token communication between robots (not recommended now)

Language or scene-graph tokens between heterogeneous agents (LiDAR robot,
RGB-D cart, camera masts) are timely as a topic. But the labeled tasks in
SCooP are 3D detection and segmentation at 10 Hz, and a VLM in the loop
would itself blow the 100 ms budget. Revisit only if a language-level task
is added to the Gym.

---

## 6. Plan with pre-registered decision rules

Steps 0–2 need no labels and no training. They run on the existing passes
with `scoop_pipeline/scoop/channel.py` and `scoop_gym`. Their thresholds are
fixed here, before any data are read.

| Step | Experiment | Decision rule |
|---|---|---|
| **0. H0: does displacement matter indoors?** | Π and Π_c (`THEORY.md` §2.5) per class at θ = 0.5 and 0.7: p95 speed and velocity-estimate error from the reference tracks, times the measured age of messages that landed in time | If Π_c < 0.5 for every class, the plan proceeds as written, with the velocity tag in unit 1. If Π_c ≥ 1 for any class, a learned receiver-side compensation becomes a required component for that class |
| **1. Oracle headroom for O1** | Per frame and sender on coop_2 and survey_1: the largest size that would land by the frame boundary, s*(t) = (T − d(t))·G(t), under `logged`. Compare the in-deadline value of an oracle size-matched policy against the best fixed kind in `CommSchedule-v0`. Use placeholder values (0.2 / 0.6 / 1.0), with a value proportional to delivered bytes for the progressive case | **NO-GO for O1** if the oracle gains < 15 % over the best fixed kind on coop_2. Report the same number under `fit:` and `v2xvit`: if synthetic channels show ≈ 0 headroom, that contrast is itself a result |
| **2. Predictability for O2** | Predict, at horizons 0.1 / 0.5 / 1.0 s, whether a 100 KB and a 1.5 MB message would land in-frame. Features: clearance along the recorded future path, CSI power, recent goodput. Baseline: persistence. Train on survey_1, test on coop_2 and the reverse | **GO for O2** if the predictor beats persistence by ≥ 5 points of balanced accuracy at 0.5 s, out of run. Otherwise O2 drops to a reactive policy (recent goodput only), and the map is reported as not predictive at the time scale of a message |
| **3. Policy on replay** | A size-matching sender (O1 without learned features: choose among boxes, features and cloud by predicted s*) against fixed kinds, AoI-optimal, and TAMP-style schedulers. Run each under ideal, `v2xvit`, `fit:`, and `logged` | Headline result if the ranking of policies under `logged` differs from their ranking under synthetic channels |
| **4. Learned progressive message** | Once coop_2 labels and `CoopDetect-v0` exist: train a prefix-decodable intermediate-fusion model (nested dropout over token or instance units), then rerun step 3 with a real detection value V | GO if prefix fusion at matched in-deadline bytes is within 5 AP of the full message, and beats truncation of a non-progressive model |
| **5. Event task (O3)** | `OcclusionHandoff-v0`: share of warnings that land before the robot's decision point, under value-at-decision scheduling against AoI scheduling | Reported as a demonstration; no pass threshold |

**Dependencies.** Steps 1 and 3 trust the logged channel's size term,
d(t) + s/G(t). That term is a model, and E3 (the replay-versus-live session)
is what validates it. Run steps 0–2 now, and state the E3 dependency in any
write-up until the session is done.

---

## 7. What the resulting paper claims

1. **Problem.** Task-oriented communication for collaborative perception
   optimizes rate, PHY robustness or age under channels that never miss a
   frame. On a measured indoor link, delivery time is set by size and
   contention, and the existing objectives rank policies wrongly (step 3).
2. **Method.** Predict the delivery time from map, CSI and contention.
   Send the task-ordered prefix that fits the deadline (steps 2 and 4).
3. **Evidence.** Gains under the logged link, with the synthetic-channel
   rankings shown to mislead. All evidence is on real recordings, with E3
   bounding the replay error.

**Venues.** IEEE JSAC or TWC for the comms framing (O2 + O3 theory), or
IEEE IoT-J or INFOCOM as a follow-up to SCooP-iT. If step 4 succeeds,
CVPR/ICCV/CoRL with the learned codec as the lead.

---

## Sources (read the primary text before citing; most are [S])

- SComCP, task-oriented semantic comm for CP, IEEE TVT 2026: https://www.researchgate.net/publication/399855625_SComCP_Task-Oriented_Semantic_Communication_for_Collaborative_Perception
- R-ACP, robust task-oriented comm for real-time CP: https://arxiv.org/pdf/2410.04168
- CoDS, digital semantic comm for CP: https://arxiv.org/pdf/2512.22513
- Domain-generalized adaptive SemCom for CP: https://arxiv.org/pdf/2608.00056
- Task-oriented wireless comm for CP in unmanned systems, IEEE Network 2024: https://network.ee.tsinghua.edu.cn/niulab/wp-content/uploads/2024/12/Task-Oriented_Wireless_Communications_for_Collaborative_Perception_in_Intelligent_Unmanned_Systems.pdf
- Multi-robot task-oriented comm, shared 3D semantics: https://arxiv.org/html/2602.08624
- Semantic comm for cooperative perception with HARQ (SimHARQ): https://arxiv.org/pdf/2409.09042
- Token Communications (Qiao et al.): https://arxiv.org/abs/2502.12096 ; UniToCom: https://arxiv.org/pdf/2507.01728
- TAMP, timeliness-oriented multi-region scheduling: https://arxiv.org/abs/2601.04542
- Update the Unseen Only: https://arxiv.org/abs/2607.20967
- AoI-guaranteed scheduling for CP: https://ar5iv.labs.arxiv.org/html/2310.04813
- AoI + network synchronization fusion: https://arxiv.org/abs/2602.13439
- SchedCP, DRL scheduling with CSI for CP: https://arxiv.org/pdf/2502.10456
- Temporal Residual Bottleneck, asynchronous CP: https://arxiv.org/html/2610.10090v1
- From freshness to semantics of information (survey): https://arxiv.org/pdf/2512.12758 ; AoII: https://ar5iv.labs.arxiv.org/html/2012.13214
- WiSwarm and the AoI thesis (Tripathi, MIT): https://www.mit.edu/~vishrant/tripathi-vishrant-phd-2023-thesis.pdf
- FERMI, multi-robot radio mapping, RSS 2025: https://roboticsconference.org/2025/program/papers/95/
- Earlier CP literature map (latency compensation, rate, AoI, ETSI CPM): `../temporal_messaging/RELATED_WORK.md`
