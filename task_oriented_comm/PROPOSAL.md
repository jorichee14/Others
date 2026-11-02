# Proposal: deadline-matched, twin-aware communication for an indoor robot team

This file assesses the material in `PLAN.md`, `THEORY.md` and
`OBJECTIVE.md`, fixes one scope, and states the motivation, the system
model and the plan in their final form. Where this file and the earlier
files disagree, this file wins.

---

## 1. Assessment of the material so far

### What holds

| Item | Verdict |
|---|---|
| The gap: no method chooses what to send from the delivery time it will face on a shared link, and none is tested on a measured indoor link | Holds. Supported by the literature map and by SCooP's own numbers |
| The measurement that motivates it: RTT 25–33 ms, yet 59–68 % of 1.5 MB messages in-frame with two robots, 86 % alone, 100 % under the field's synthetic channels | Holds [R] |
| Proposition 1: the gain of a progressive message equals the partial-delivery mass, which is zero under every synthetic CP channel | Holds; elementary but not in the CP literature |
| The objective (DTCV) nests rate–distortion CP, AoI scheduling, latency compensation and remote estimation as special cases | Holds as a formulation |
| Select by value, send by deadline (matroid structure) | Holds for equal-size tokens and known per-window capacity |

### What is weak, and the fix adopted here

| Weakness | Fix |
|---|---|
| **Scope drift.** `PLAN.md` is about perception messages between robots; `OBJECTIVE.md` covers every flow of the robot–server twin system | One scope: the robot–server twin system of the diagram, with four flows (section 3). Robot-to-robot perception is a special case of the keyframe flow and is not treated separately |
| **The value function F_k is the hard part, and the theory treats it as given.** At the sender, "loss reduction for the twin given what the twin holds" is not computable | Use proxies the sender can compute, and validate them offline against the true loss (step 2). Proxies: fraction of a scan not explained by the robot's twin tiles (change), co-visibility from the map, and pose innovation |
| **Circularity.** The robot must know what the twin holds to value its own data, but that knowledge arrives over the same link | Versioning, as the diagram already has (twin "version n", tiles). The robot values data against the twin version it holds; the server corrects on the next tile. Stale side information lowers the value estimate, it does not break the scheme |
| **The age factor ρ_k(a) assumes value factorizes as content × age.** It does not in general | Keep ρ only as a first-order model; measure the residual error after kinematic correction directly in step 0, and report the factorization as an approximation |
| **The evidence chain rests on the logged channel's size term d + s/G**, a model until E3 validates it | Run E3 before any result goes in a paper; state the dependency until then |
| **Contention counterfactuals cannot be replayed from traces.** A policy that sends less changes what the other robot sees, and the trace holds the other robot's load fixed | Replay gives a bound: each robot's policy against the recorded cross traffic. The joint effect needs the fitted contention model or a live run (step 5). Say so |
| **R-ACP is closer than the earlier notes allowed.** It has channel-aware calibration, adaptive compression and priority scheduling | Read it first. The expected difference is that R-ACP's priorities and compression are set per stream and the value is not conditional on the receiver's state; confirm before claiming |
| **Sionna in the loop is heavy and may not beat clearance** | Clearance from the map is the primary predictor; Sionna is the optional upgrade, tested only if clearance beats persistence |
| **The theory oversells.** Propositions 1–6 are short derivations | Present them as the structure of the problem, not as the contribution. The contribution is the formulation, the system, and the measured evidence |

### Novelty, after the assessment

| Claim | Confidence |
|---|---|
| The formulation: expected task value at each consumer's decision time, conditional on the receiver's (twin) state, over a shared no-recall queue with a predicted deadline capacity | Medium-high, pending a full read of R-ACP, SComCP, ProgressFTX |
| Synthetic channels make progressive and predictive sending worthless by construction (Proposition 1), so the CP literature could not have found the effect | High |
| The twin as side information for its own update traffic, and the twin's map as the predictor of the link that carries it | Medium-high; no prior work found, search to confirm |
| First measurement of these effects on a real indoor team | High, conditional on E3 |

---

## 2. Motivation

**The system.** Two robots and a server keep one shared world: a map, a
live Gaussian-splat scene, a radio twin. Every message that keeps it
shared crosses one Wi-Fi access point: control commands down, poses up at
10 Hz, keyframe packs up at 0.5–1 MB/s per robot, map tiles of 1–5 MB
down, corrections down.

**Three facts from the SCooP recordings.**

1. **Size, not latency, decides whether a message arrives in time.** With
   two robots the loaded RTT is 25–33 ms, far inside a 100 ms frame, yet
   only 59–68 % of 1.5 MB messages land within the frame (86 % with one
   robot). A keyframe pack or a tile is in this size range.
2. **Bulk traffic delays the time-critical flows.** RTT rises from 3.3 ms
   idle to 60–109 ms under load. A tile in front of a control command in
   the same FIFO queue makes the command late. Outages covering 12 % of
   frames were recorded on one pass.
3. **The world the team maps predicts the link it uses.** Fresnel
   clearance from the reference map gives Spearman ρ 0.60 against RSSI,
   and CSI is logged at 104–141 Hz on the sensor clock.

**Why the literature does not cover this.** Rate–distortion CP assumes a
fixed budget; semantic communication for CP assumes a noisy PHY on one
link; AoI scheduling ignores content; latency compensation assumes every
message arrives whole. Each of them evaluates on channels that never
deliver part of a message, and under such channels a progressive or
predictive sender gains exactly nothing (THEORY Prop. 1). The effect could
not have been seen on the data the field uses. No indoor dataset logs the
link beside the sensors; CooperScene logs C-V2X outdoors for vehicles.

**The idea.** The server's twin is what the receiver already knows, so it
is the side information that prices every byte a robot sends. The twin's
map, with the robot's planned path, predicts the link those bytes will
cross. A sender that knows both can send the most valuable part of its
data first, in the amount the deadline allows, and keep the time-critical
flows ahead of the bulk. The twin predicts the channel that keeps the twin
current.

**What the paper claims.** On a measured indoor link, the objectives the
literature optimizes rank communication policies wrongly. A policy that
values bytes against the receiver's state and sizes them to a predicted
deadline capacity reduces the communication regret of every consumer,
and the theory says when each part of it pays.

---

## 3. System model

### Agents

| Agent | Role | Holds |
|---|---|---|
| Robot i ∈ {1, 2} | sensing, local odometry, map localization | its odometry, its scans, map tiles of twin version v_i, its planned path |
| Server | Swarm-SLAM, live map, scene update, Isaac Sim, Sionna RT | the twin at version v, the reference map, the radio map |
| Access point | one shared 802.11ac channel, DCF | per-access-category FIFO queues, no recall |

### Flows

Each flow f has a source, a receiver, a consumer k(f), and a decision
schedule 𝒯_k. Every flow is packetized into tokens of one MTU (β ≈ 1.4 KB).

| f | Direction | Size, rate | Consumer k | Decision times | Deadline | Value shape |
|---|---|---|---|---|---|---|
| control | server → robot | 1 token, event | drive base | every control period | tens of ms | step: full value in time, zero late |
| pose | robot → server | 1 token, 10 Hz | sim and radio prims | 10 Hz | 100 ms | falls with residual error after correction (ρ) |
| keyframe | robot → server | 200–350 tokens, 2 Hz | Swarm-SLAM, live map, change detection | per SLAM / scene update | about 1 s | conditional on the twin: new regions, changes, loop closures |
| tile | server → robot | 700–3500 tokens, rare | map localization | when the robot's path enters an area it lacks | seconds | conditional on the robot's tiles and planned path |
| correction | server → robot | 1 token, ~0.1 Hz | map→odom | per correction | seconds | pose drift bounded |

### Link

The `logged` model of `scoop_pipeline/scoop/channel.py`, kept as is. A
prefix of B bytes committed at t by robot i arrives at

    t + δ_i(t) + W_i(t) + B / G_i(t)

with δ the one-way latency, W = R/G the wait behind the backlog R, and G
the goodput of the hop (two hops through the AP for robot-to-robot). In an
outage, G = 0. With WMM, each access category has its own queue; bulk
tokens behind a control token do not delay it, and the contention between
categories follows the EDCA parameters.

**Deadline capacity** for consumer k from robot i:

    S_k(t) = ( T_k − δ_i(t) − W_i(t) )⁺ · G_i(t)

S is random given the sender's information x_t.

### Predictive information

    x_t = ( planned path over the horizon, clearance along it from the map,
            CSI power and Doppler, recent goodput and RTT,
            the other robot's recent load )

The predictor returns the distribution of S_k over the next 0.1–1 s, or a
chosen quantile of it. Three predictors, in order of cost: persistence
(the last value), clearance from the map, Sionna RT on the twin mesh.

### Receiver state (side information)

| Receiver | M_k(τ) |
|---|---|
| Server, SLAM and scene | the twin at version v, with which regions are mapped, when each was last observed, and what changed |
| Robot i, localization | tiles of version v_i, its planned path |
| Robot i, drive base | the last command |
| Sim and radio | the last pose and velocity |

### Value and loss

For consumer k, the true loss ℓ_k is scored against SCooP's ground truth:

| k | ℓ_k |
|---|---|
| control | deadline missed |
| pose | pose error at τ against the reference trajectory; Sionna CIR error against measured CSI |
| SLAM | ATE and RPE against the reference |
| twin | time from a scene change to its appearance in the twin; divergence from the reference map |
| localization | map→odom failures; pose error against the reference |

The sender cannot compute ℓ_k. It uses a proxy value per token,
ΔF̂_k(u | M), validated against ℓ_k offline (step 2):

| Flow | Proxy |
|---|---|
| keyframe | fraction of the scan not explained by the robot's twin tiles (change detection on the robot); co-visibility from the map; loop-closure candidates |
| tile | distance along the planned path to the first unmapped area |
| pose | innovation of the pose against the server's last prediction |
| control, correction | constant per token |

### Decision

At each slot, each sender commits a set of tokens to the AP queues.
Token (f, m) has index

    I_{f,m}(t) = w_k · ρ_k(â) · ΔF̂_k(u_{f,m} | M_k) − λ · β / Ĝ(t)

where w_k converts consumer k's loss to a common unit, ρ_k(â) is the share
of value left after kinematic correction at the expected age â, and λ is
the airtime price (a Lyapunov virtual queue over the long run).

The rule: commit tokens with I > 0 in descending I while the q-quantile
deadline capacity for their window still admits them (earliest-deadline-
first feasibility), and transmit the committed set in EDF order. Priority
across flows is an output: control and pose always come first because
their index per byte is largest and their deadline earliest. The mapping
to WMM categories (control, pose, correction → AC_VO; keyframe → AC_BE;
tile → AC_BK) implements the EDF order at the MAC.

### Objective

    max_π  lim inf (1/T) E[ Σ_k w_k Σ_{τ ∈ 𝒯_k} ρ_k(a_τ) F_k( 𝒜_k^π(τ) | M_k(τ) ) − λ Air^π(T) ]

subject to causality, no recall, the shared queue, and side-information
updates. Reported as the communication regret R(π) = J(ideal) − J(π) per
consumer. `OBJECTIVE.md` has the special cases and the structure;
`THEORY.md` has Propositions 1–6.

---

## 4. Plan

Six steps. Steps 0–3 run on the existing passes (survey_1, coop_2) with
`channel.py` and `scoop_gym`; they need no labels and no new hardware.
Thresholds are fixed here, before any data are read.

| Step | What | Needs | Decision rule |
|---|---|---|---|
| **0. Regime check** | Π and Π_c per class (robots, people) at IoU 0.5 and 0.7 from the reference tracks and the measured age of in-time messages | tracks | Π_c < 0.5 for every class: proceed, with the velocity tag in the first token. Any class with Π_c ≥ 1: learned receiver-side compensation becomes required for it |
| **1. FIFO against priority, with the real traffic mix** | Replay the diagram's five flows at their rates through the logged link with `,queue`, FIFO and per-category queues. Deadline hits for control and pose; delay of keyframes and tiles | traces | If control or pose misses under FIFO exceed 5 % on coop_2, the priority split is justified and reported. Prop. 6's backlog price is measured here |
| **2. Value proxies and the partial-delivery mass** | (a) On static passes, the fraction of each keyframe pack's bytes that add nothing to the twin (change detection against the tiles). (b) Σ Δ_m P(B_m ≤ S < B_n) from the traces, under `logged`, `fit:`, `v2xvit`. (c) Correlation of each proxy with the true ℓ_k on the reference | traces, change detection (to build) | (a) ≥ 50 % redundant bytes: the conditional value pays. (b) Gap ≈ 0 under synthetic, > 0 under logged, larger on coop_2: Prop. 1 confirmed. (c) proxy–loss Spearman ρ ≥ 0.5 for keyframes, or the proxy is replaced |
| **3. Link prediction** | Predict in-frame delivery of 100 KB and 1.5 MB messages 0.1 / 0.5 / 1 s ahead. Persistence, clearance, (Sionna). Train on survey_1, test on coop_2, and the reverse | traces, map, (Sionna chain) | Clearance beats persistence by ≥ 5 points of balanced accuracy at 0.5 s out of run: GO for prediction. Otherwise the policy is reactive and the paper says the map is not predictive at the message time scale |
| **4. Policies in replay** | Send-all-FIFO (today), fixed WMM, AoI scheduler, TAMP-style, DTCV with each predictor; under `ideal`, `fit:`, `logged`. Regret per consumer | steps 1–3 | Headline if the ranking under `logged` differs from the ranking under synthetic channels. Report the one-robot bound (cross traffic fixed) explicitly |
| **5. Live check** | E3 (replay against live) for the size term; then one session with DTCV running on the robots, both robots sharing the AP, against send-all-FIFO on the same route | payload node, policy on the Jetsons, one session each | Replay error on arrival times within the E3 criterion (≥ 90 % frame agreement). Live regret within 20 % of replayed regret; the joint contention effect reported from the live run only |
| **6. Learned progressive keyframe** (optional) | Prefix-decodable keyframe encoding (nested dropout over tokens) for the live map and change detection | labels, training | Within 5 % of the full pack's twin-update quality at matched in-time bytes, and better than truncating a non-progressive pack |

**Milestones.**

| M | Done when | Yields |
|---|---|---|
| M1 | steps 0–2 | the regime, the priority case, the partial-delivery mass, the value proxies: the motivation section with numbers |
| M2 | step 3 | the predictor and its out-of-run accuracy |
| M3 | step 4 | the policy comparison, regret per consumer |
| M4 | step 5 | E3 validation and the live run: the result the paper leads with |
| M5 | step 6 | the learned codec, if time allows |

**Dependencies.** Change detection on the robot (step 2a) is the one new
component before M1. Sionna is optional. Labels are needed only for step 6.

**Writing.** Lead with the measurement (section 2's three facts and
step 2b), then the system model, then the policy and its regret, then the
live run. The theory goes in one section as structure, with proofs in the
appendix. Venue: IEEE JSAC or IEEE IoT-J as the SCooP-iT follow-up;
INFOCOM if the live run is in hand.
