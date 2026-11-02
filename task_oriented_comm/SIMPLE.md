# The problem in its simplest form: deadline-adaptive task-oriented compression

One sender, one receiver, one task, one deadline per frame. This is the
smallest problem that contains the effect, and it is the starting point
of the paper. `OBJECTIVE.md` generalizes it in two steps (section 4).

---

## 1. Setting

At each frame t the sender holds an observation X_t (a LiDAR scan, a
keyframe, a feature map). It must choose a **compression level**

$$r_t\in\{r_1<r_2<\dots<r_L\},$$

where level r gives a message of size b(r) bytes and, if the receiver
gets it in time, a task value V(r), nondecreasing in r (detection gain,
map-update quality, pose error reduction). V(0) = 0 means "send nothing".

The receiver acts at the frame boundary, T after capture. The message
arrives at

$$t+\delta_t+\frac{b(r_t)}{G_t}$$

with latency δ_t and goodput G_t set by the link at that moment. So the
message counts only if

$$b(r_t)\;\le\;S_t\;:=\;(T-\delta_t)^+\,G_t ,$$

the **deadline capacity**. On a shared Wi-Fi link, S_t is random: it
changes with the other robot's traffic, the agent's position, and
outages. SCooP measures it; the synthetic channels of the literature fix
it.

The sender knows x_t (recent goodput, RTT, CSI, map clearance along its
path) and hence a predictive distribution P(S_t ≤ s | x_t).

## 2. The problem

$$\boxed{\;\max_{r(\cdot)}\;\;\mathbb E\Big[\,V\big(r(x_t)\big)\,\mathbf 1\{b(r(x_t))\le S_t\}\;-\;\lambda\,b\big(r(x_t)\big)\Big]\;}$$

Per frame, with the expectation over S_t given x_t:

$$r^*(x_t)=\arg\max_{r}\;\Big[\,V(r)\,P\big(S_t\ge b(r)\,\big|\,x_t\big)\;-\;\lambda\,b(r)\Big].$$

**In words:** pick the compression level whose task value, discounted by
the chance it beats the deadline, best exceeds its airtime cost. That is
the whole problem.

## 3. What fixed compression gets wrong

Fixed task-oriented compression (Where2comm, RDComm, InfoCom) solves

$$\max_r\;V(r)-\lambda\,b(r)\quad\text{s.t.}\quad b(r)\le B,$$

with a known budget B. It is the special case S_t ≡ B: the indicator is
1 below B and 0 above, so the level is chosen once and never changes.

Two failures when S_t is random:

1. **A fixed level too high misses.** If b(r) > S_t, the message is worth
   V(r)·0 = 0, not V(r). On coop_2 a 1.5 MB message sits above S_t in
   32–41 % of frames; under V2X-ViT's constant delay it never does.
2. **A fixed level too low wastes.** When S_t is large, a small message
   leaves value on the table.

The loss of a fixed level r_f against the adaptive rule is

$$\mathbb E\big[\max_r V(r)P(S\ge b(r)\mid x)-\lambda b(r)\big]-\big[V(r_f)P(S\ge b(r_f))-\lambda b(r_f)\big]\;\ge 0,$$

and it is zero exactly when S_t is deterministic or when x_t carries no
information about S_t. Both are measurable on the traces.

## 4. The three measurable quantities

| Quantity | What it is | From |
|---|---|---|
| V(r) | task value per compression level, in-time delivery | run the task at each level under the ideal channel |
| P(S_t ≥ b \| x_t) | in-time probability of a message of size b, given what the sender sees | the logged link: for each frame, which sizes would have landed |
| r*(x_t) vs r_f | the adaptive rule against the best fixed level | the two previous rows |

The paper's first figure is V(r) against b(r) with the curve of
P(S ≥ b) under `logged`, `fit:` and `v2xvit` overlaid: the adaptive
optimum moves with the link, the fixed optimum does not.

## 5. Two steps to the full objective

1. **Make the message progressive.** Replace the menu {r_l} by an ordered
   token stream and let the receiver use the prefix that arrived. The
   indicator 1{b ≤ S} becomes the delivered prefix N(min(S, B)), and the
   objective becomes Σ_m Δ_m P(S ≥ B_m) − λB (THEORY Prop. 1). The sender
   now chooses *how many* tokens to commit rather than *which level*, and
   a miss delivers partial value instead of none.
2. **Make the value conditional and the queue shared.** V(r) becomes
   F_k(· | M_k), the value given what the receiver already holds (the
   twin); several flows with different deadlines share one no-recall
   queue; age enters through residual error. That is DTCV
   (`OBJECTIVE.md`).

The simple problem is DTCV with one flow, an atomic message, M fixed,
ρ ≡ 1. Everything the paper claims is already visible in it: the random
deadline capacity, the prediction, and the zero gain under synthetic
channels.

---

## 6. Current literature on this simple problem (checked 30 Oct 2026)

Adaptive compression for collaborative perception is an occupied line.
These are the papers that set the compression level from the channel
rather than from a fixed budget. All [S]: from abstracts and search
summaries, since arXiv is not reachable from this session; read before
citing.

| Paper | What adapts, and to what | Channel at decision time | Deadline | Evaluation |
|---|---|---|---|---|
| SmartCooper (Fang et al., arXiv 2402.00321) | per-agent compression ratio from the allocated sub-channel capacity; a "judger" drops agents whose data would not help | known: capacity allocated by the base station | none stated | simulation |
| PACP (Fang et al., IEEE TMC, arXiv 2404.06891) | priority per collaborator from BEV match; transmission rate, link and compression ratio jointly by submodular optimization; adaptive autoencoder from CSI feedback | known CSI | delay minimized in the related ACC-DA, not a per-message deadline | simulation |
| ACC-DA (Hu, Fang et al., GLOBECOM 2024, arXiv 2310.00013) | communication graph and rate–distortion trade-off to minimize average transmission delay | known | average delay, not a deadline | simulation |
| R-ACP (Fang et al., JSAC 2025) | video compression rate to channel capacity; drops links with high loss; priority scheduling | measured current capacity | none stated | multi-camera ground vehicles, emulated loss |
| EffiComm (arXiv 2507.19354) | per-vehicle keep ratio from role, scene complexity and instantaneous network load, by a graph attention network | measured current load | none | OPV2V |
| BEVCooper (Hou et al., arXiv 2512.19082) | online selection of collaborators and feature fusion under changing utility and V2V channel quality, bandit style; claims a real testbed | measured, learned online | latency reported, not a deadline | real testbed (details unverified) |
| WhisperNet (Chen et al., CVPR 2026, arXiv 2603.01708) | spatial and channel-importance scoring at the sender for a given bandwidth; motivates by "fixed-rate encoders adapt poorly" | budget given | none | OPV2V, DAIR-V2X |
| Variable-rate deep JSCC (DeepJSCC-V and successors), TCS-JSCC (WCL 2026), AdaSem (arXiv 2401.08360) | code length from target quality and SNR; AdaSem trades accuracy against end-to-end latency | SNR known | AdaSem: latency, not a deadline | single link, PHY simulation |

**What they share.** The compression level is matched to a capacity that
is known at decision time: allocated by a base station, read from CSI,
or measured as the current load. The decision is then deterministic:
pick the largest level that fits, b(r) ≤ C_now. In the notation of
section 2 this is P(S ≥ b | x) = 1{b ≤ C_now}, the persistence predictor
with no uncertainty. None of them counts a message that misses its
frame as worth zero, none evaluates on a shared CSMA link where the
other agent's traffic moves S between frames, and none predicts S ahead
from where the agent is going.

**What the simple problem still owns, after this check.**

1. **The in-time probability, not the capacity, is the decision
   variable.** V(r)·P(S ≥ b(r) | x) with S random. The matched-capacity
   rule is the corner P ∈ {0, 1}. When S is dispersed (coop_2: a 1.5 MB
   message lands in 59–68 % of frames at the same RTT) the two rules
   choose differently, and the matched rule misses whenever S drops
   below C_now within the frame.
2. **Prediction ahead of the message, from geometry.** x_t includes the
   planned path and the map's clearance along it, so P(S ≥ b | x) can
   change before the measured capacity does. No entry above uses
   anything but the current channel.
3. **A measured, contended link as the test.** Every entry evaluates on
   an allocated or simulated channel, except BEVCooper, whose testbed
   details are unverified. SCooP supplies the logged S_t per frame and
   the replay check (E3).

So the simple form is not new as "adaptive compression"; it is new as
**deadline-probabilistic, predictive compression on a measured shared
link**, and the paper must say so in those words. The progressive step
(section 5, step 1) is where the formulation leaves the adaptive-
compression line entirely, since none of the entries has a message that
is worth something when only part of it arrives.
