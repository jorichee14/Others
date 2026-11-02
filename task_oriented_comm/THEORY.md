# Theory and novelty: deadline-matched task messages over a predicted link

This file is the theoretical companion to [`PLAN.md`](PLAN.md). The
approach it covers: the sender predicts the link, sends a task-ordered,
prefix-decodable message, and is scored by its task value at the receiver's
decision time.

The sections below cover:

- the model;
- six results, with proofs;
- the established theory each result rests on;
- what is new and what is not;
- the measurement on SCooP that tests each result.

Tags: **[S]** marks prior work known from abstracts or search summaries.
Read the primary text before citing it. Classical results (Equitz–Cover,
Nemhauser–Wolsey–Fisher, Neely) are standard and untagged.

---

## 1. Model

**Agents and time.** Sender i captures an observation at time t. Receiver j
uses whatever has arrived at its decision time τ = t + T. In the frame-
synchronous case, T is the 100 ms LiDAR period. In the event case, T is set
by the robot's motion (section 2.6).

**Message.** The message is an ordered sequence of units u_1, …, u_N, for
example tokens, BEV cells, or instance queries. Unit n has size b_n; the
first n units have cumulative size B_n = b_1 + … + b_n. Units are sent as
one byte stream, in order. A prefix is decodable by itself. Where2comm-style
sparse cell maps already are: a subset of cells is a valid sparse map.

**Delivery.** This matches the `logged` channel in
`scoop_pipeline/scoop/channel.py`. A prefix of B bytes arrives at
t + δ + B/G. Here δ is the one-way latency plus any wait behind earlier
traffic, W = R/G, where R is the backlog. G is the effective goodput over
the two hops:

$$1/G = 1/G_\text{up} + 1/G_\text{down}$$

In an outage, G = 0.

**Deadline capacity.** This is the number of bytes that can arrive by τ:

$$S \;=\; (T-\delta)^+\,G .$$

The receiver gets the prefix of N(S) = max{n : B_n ≤ S} units. S is random
given the sender's information x_t, which is the map, the planned path, the
CSI, the recent goodput, and the other robots' traffic.

**Value.** V(n) is the task value of the first n units at the decision
time: the gain over the receiver's ego-only result. It is computed given
the receiver's own observation Y_j, which plays the role of side
information. V(0) = 0 and V is nondecreasing. Let Δ_n = V(n) − V(n−1).

**Sender's problem (one frame).** Choose the code, its order, and the
number of bytes B to commit. Maximize

$$J(B)\;=\;\mathbb{E}\big[V(N(\min(S,B)))\big]\;-\;\lambda B\;-\;\mu\,\mathbb{E}\big[(B-C)^+\big],$$

where:

- λ is the price of airtime;
- C = T·G is what the link drains in one frame;
- μ is the price of the backlog (B − C)^+ left for later traffic.

**Three literatures are special cases of this problem:**

| Special case | Literature |
|---|---|
| S deterministic, a fixed per-frame budget | bandwidth-efficient CP and pragmatic rate–distortion (Where2comm, RDComm, InfoCom) |
| V blind to content, a single unit, value falling with age | AoI scheduling (Fresh2comm, TAMP, WiSwarm) |
| S = ∞ (everything arrives), value falling with age | latency compensation (CoBEVFlow, FFNet, TraF-Align) |

None of them has S random, units ordered, and V conditional on the
receiver's side information, all at once. That combination is the problem.

---

## 2. Results

### 2.1 The value of a progressive message is the partial-delivery mass

**Proposition 1.** Take any distribution of S and any fixed unit order.
Suppose the progressive and atomic codes have equal prefix values. Then:

- A progressive message of n units earns
  $$\mathbb{E}[V(N(\min(S,B_n)))] = \sum_{m\le n}\Delta_m\,P(S\ge B_m).$$
- An atomic message of the same n units, useful only if it arrives whole,
  earns $$V(n)\,P(S\ge B_n).$$
- The difference is
  $$\mathrm{Gap}_n=\sum_{m\le n}\Delta_m\,P(B_m\le S<B_n)\;\ge 0 .$$

*Proof.* N(min(S, B_n)) ≥ m exactly when S ≥ B_m, for m ≤ n, so
V(N(min(S, B_n))) = Σ_{m≤n} Δ_m 1{S ≥ B_m}. Take the expectation. The
atomic value is Σ_{m≤n} Δ_m P(S ≥ B_n). Subtract, and use
P(S ≥ B_m) − P(S ≥ B_n) = P(B_m ≤ S < B_n). ∎

**Corollary 1a: zero under the synthetic CP channels.** Gap_n = 0 exactly
when S puts no mass in [B_1, B_n), the range where a partial message would
arrive. Two channel families have no such mass:

- **Deterministic S.** Examples are a constant delay with no size term, as
  in V2X-ViT, or a fixed bit budget, as in Where2comm. Here the sender sets
  B_n ≤ S and loses nothing.
- **On/off S.** An outage, or more than enough capacity.

Every synthetic channel in `channel.py` except `synthetic:…,mbps=…` falls
in one of these two cases. This is a formal reason why the CP literature
never needed progressive messages: its channels cannot produce partial
delivery. On SCooP's logged link, 14–41 % of 1.5 MB messages miss the frame,
and a positive fraction of those would have delivered a useful prefix.

**Corollary 1b: when progressivity pays.** A real progressive code gives
up some value against the best code designed for each size, V*(n). Write
V_P(n) = V*(n) − ε(n), where ε is the refinement penalty. Progressive beats
the best atomic choice exactly when

$$\mathbb{E}[V^*(N(S))]-\max_n V^*(n)P(S\ge B_n)\;>\;\mathbb{E}[\varepsilon(N(S))].$$

The left side is computable from the logged traces alone (`PLAN.md`
step 1). The right side comes from training.

### 2.2 Equal-size tokens give an ordering that needs no prediction

The value of a set of units given the receiver's view is F(A). A natural
choice is a weighted coverage of cells or objects the receiver cannot see
itself. Weighted coverage is monotone submodular. Overlap with the
receiver's own view removes value, which is the Wyner–Ziv side information
of section 3.

**Proposition 2.** Let all units have the same size β, and let g be the
greedy order, which repeatedly takes the unit with the largest marginal
gain. Then for every distribution of S,

$$\mathbb{E}\big[F(g_{1:\lfloor S/\beta\rfloor})\big]\;\ge\;(1-1/e)\,\mathbb{E}\big[\mathrm{OPT}(\lfloor S/\beta\rfloor)\big],$$

where OPT(k) = max over sets A with |A| ≤ k of F(A). OPT is a clairvoyant
sender that knows S before it encodes. If F is additive, the bound becomes
equality.

*Proof.* Nemhauser, Wolsey and Fisher (1978) show that for every k, the
first k greedy elements satisfy F(g_{1:k}) ≥ (1 − 1/e)·OPT(k). The greedy
order does not depend on k, so the inequality holds for every k at once.
It therefore holds for K = ⌊S/β⌋ for every realization of S, and so in
expectation. For additive F, the top-k units by value are optimal for every
k. ∎

**Remarks.**

- With unequal sizes, the greedy-by-ratio prefix has no constant guarantee
  between its breakpoints. The known fix (Khuller–Moss–Naor 1999) compares
  greedy with the best single unit, and that choice depends on the budget,
  so it breaks the ordering's independence from S. **This is the
  theoretical reason to use fixed-size tokens for time-critical messages.**
  It gives the "token communication" idea a precise role.
- The sender needs F, so it needs a model of what the receiver already
  sees. The shared reference map and the receiver's pose give that by ray
  casting. SCooP releases this co-visibility per frame.
- The greedy marginal gains are nonincreasing, so V is discretely concave
  along g. Proposition 3 uses this.

### 2.3 How many bytes to commit, and where prediction matters

**Proposition 3.** Suppose V has nonincreasing increments along the order
(as in 2.2), and treat B as continuous. Then J is concave in B. The optimal
commitment B* is the full message if J′(B_max) ≥ 0. Otherwise it is the
unique solution of

$$V'(B^*)\;P(S>B^*\mid x)\;=\;\lambda\;+\;\mu\,P(C<B^*\mid x).$$

In words: stop when the marginal unit's value, times the probability that
it beats the deadline, falls below the price of airtime plus the price of
the backlog it leaves.

*Proof.* J′(B) = V′(B)P(S > B) − λ − μP(C < B). Each term is nonincreasing
in B: V′ is nonincreasing, P(S > B) is nonincreasing, and P(C < B) is
nondecreasing. So J′ is nonincreasing and J is concave. ∎

**Corollary 3a: prediction alone is worth nothing without commitment
costs.** If λ = μ = 0, then B* = B_max for every x: stream the whole ordered
message and let the deadline cut it. The value of prediction is

$$\mathrm{VoP}=\mathbb{E}_x\big[\max_B J(B\mid x)\big]-\max_B\mathbb{E}_x\big[J(B\mid x)\big]\;\ge 0 .$$

VoP is zero when B* does not depend on x, and in particular when λ = μ = 0.
So prediction pays for one reason only: bytes committed to the network
cannot be recalled. They cost airtime, delay the next frame, and delay other
robots. That is the case on Wi-Fi: once a frame is in the driver or the AP
queue, the application cannot cancel it.

**Corollary 3b: prediction and progressivity substitute for each other.**
Apply Proposition 1 conditionally on x:

$$\mathrm{Gap}_n(x)\le V(n)\,P(B_1\le S<B_n\mid x).$$

- With a perfect predictor (S | x degenerate), the gap is zero. An atomic
  code sized to the prediction then wins by the refinement penalty ε. This
  is the Salsify design: pick among encodings by the predicted capacity.
- The more uncertain S is given x, the more progressivity earns.

**Progressivity pays for the uncertainty that the predictor cannot
remove.** Which side wins on SCooP is an empirical question, and
`PLAN.md` steps 1–2 answer it.

### 2.4 Deadline capacity on the measured link

With the two-hop model and a backlog R:

$$S=\Big(T-\tfrac12\mathrm{RTT}_\text{up}-\tfrac12\mathrm{RTT}_\text{down}-R/G\Big)^+\Big/\big(1/G_\text{up}+1/G_\text{down}\big).$$

**Remark.** S depends on the RTT only through the subtraction from T. When
RTT ≪ T, S is set almost entirely by G and R.

On coop_2, the loaded RTT is 25–33 ms against T = 100 ms. Yet 32–41 % of
1.5 MB messages miss the frame. That is the formula's prediction: a
dispersed G, not latency, puts mass on S < 1.5 MB.

This also explains why AoI and latency indicators fail to predict the task
outcome, which is contribution 4 of SCooP-iT. Those indicators measure δ,
whereas the task depends on the distribution of S.

### 2.5 When age matters: a regime number

A box of length L, shifted by Δ along its length, keeps IoU (L − Δ)/(L + Δ)
with the true box. So a match at IoU θ tolerates

$$\Delta_{\max}(\theta,L)=L\,\frac{1-\theta}{1+\theta}.$$

Define the displacement number of a message used at age a (from capture to
use) for an object at speed v:

$$\Pi=\frac{v\,a}{\Delta_{\max}(\theta,L)}\qquad\text{(uncorrected)},\qquad \Pi_c=\frac{\sigma_v\,a+\tfrac12 \bar a_c\,a^2}{\Delta_{\max}}\qquad\text{(after kinematic correction)},$$

where σ_v is the error of the sender's velocity estimate and ā_c is a
typical acceleration.

| Object | L | Δ_max at θ 0.5 / 0.7 | Π at v·a = speed × 0.1 s |
|---|---|---|---|
| Car, OPV2V, 20 m/s | 4.5 m | 1.50 / 0.79 m | 1.3 / 2.5 |
| Person, SCooP, 1.2 m/s | 0.6 m | 0.20 / 0.11 m | 0.6 / 1.1 |
| Person, corrected, σ_v = 0.2 m/s | 0.6 m | 0.20 / 0.11 m | 0.1 / 0.2 |

**Proposition 5 (regimes).**

- **Π ≳ 1: displacement regime.** Age destroys matches. Value depends on a,
  and latency compensation is the lever. Vehicles sit here; the parent
  study's ΔAP@0.7 being 3.9–8.8× ΔAP@0.5 is this regime.
- **Π_c ≪ 1: delivery regime.** After a cheap kinematic correction,
  V(n, a) ≈ V(n). The only lever left is how much of the message arrives,
  which is Propositions 1–3.

Uncorrected indoor people sit near Π ≈ 1 at strict IoU. With a velocity
tag in unit 1, they sit well inside the delivery regime. So the kinematic
tag belongs in the first unit of every message. It costs a few bytes and
moves the problem into the regime where this theory applies.

`PLAN.md` step 0 measures Π and Π_c per class from the reference tracks.

### 2.6 Events: the backlog price comes from the next warning

Suppose a person appears behind a corner at t_e. A warning of size b_w ≪ R
is useful only if it arrives before the robot's decision time
t_dec = t_e + (distance to corner)/(speed) − t_proc. Let s = t_dec − t_e be
the slack. Under FIFO behind a backlog R,

$$P_\text{in time}=P\big(\delta+(R+b_w)/G\le s\big).$$

**Proposition 6.** Suppose warnings arrive at rate h and each is worth V_w.
Then the backlog price in Proposition 3 is

$$\mu \;=\; h\,V_w\,\Big|\tfrac{\partial}{\partial R}P_\text{in time}\Big| \;=\; h\,V_w\,\mathbb{E}\big[f_\delta\big(s-(R+b_w)/G\big)/G\big].$$

Here f_δ is the density of δ. This assumes R and δ are independent and
holds for small R. With a priority access category (802.11e/WMM AC_VO for
warnings), R drops out of the warning's delay and μ → 0. Bulk sizing then
needs only λ.

*Proof sketch.* Committing dR more bulk bytes delays every warning queued
behind them by dR/G. The expected loss rate is h V_w times the change in
P_in time. Differentiate under the expectation. ∎

This is a perception analogue of Sun–Polyanskiy–Uysal (2020), where
signal-aware sampling beats age-optimal sampling for remote estimation.
Here the event-aware price beats any age-based schedule, because the value
is a step function of arrival time against a deadline set by the robot's
own motion, which age does not see.

### 2.7 Two senders on one AP (remark)

When both robots are backlogged, DCF splits airtime roughly evenly
(Heusse et al. 2003, the performance anomaly). One robot's commitment B_i
lowers the other's S_j. The social optimum adds a Pigouvian term to
Proposition 3:

$$V_i'(B_i)P(S_i>B_i)=\lambda+\mu_i+\sum_{j\ne i}\partial\,\mathbb{E}[\text{loss}_j]/\partial B_i .$$

A selfish sender drops the last term and oversends. The drop from 86 %
in-frame (one robot) to 59–68 % (two robots) is the measured size of the
interaction. Testing this needs a fitted contention model (`PLAN.md` O4).

---

## 3. Where each piece comes from

| Layer | Established result | Used for | What this work changes |
|---|---|---|---|
| Information theory | Successive refinement: Equitz–Cover 1991; Rimoldi 1994; Lastras–Berger 2001 (every source is within 0.5 bit/sample per stage under MSE) | The progressive code and its penalty ε | Task (pragmatic) distortion instead of MSE; ε measured, not bounded |
| | Wyner–Ziv successive refinement: Steinberg–Merhav 2004 | The receiver's own view as decoder side information | Side information estimated from the shared map and pose (co-visibility) |
| | Broadcast approach (Shamai 1997; Shamai–Steiner 2003) and expected distortion over fading (Tian–Steiner–Shamai–Diggavi; Ng–Gündüz–Goldsmith–Erkip) [S] | Layered coding when the transmitter does not know the channel state | The unknown state is the deadline capacity S at the packet layer, not the PHY SNR. Layering is free (a byte prefix, no power split), so the only loss is ε (Corollary 1b) |
| Optimization | Nemhauser–Wolsey–Fisher 1978; Khuller–Moss–Naor 1999 | Token order with a guarantee for every budget at once | Applied to equal-size tokens under random S (Proposition 2) |
| | Threshold stopping in progressive feature transmission, ProgressFTX (Lan, Zeng, Popovski, Gündüz, Huang, TWC 2022) [S] | Stopping when the marginal gain falls below a price | Stopping on deadline arrival probability and backlog, with side information, multi-agent |
| | Lyapunov drift-plus-penalty (Neely 2010) | μ over frames through a virtual backlog queue | Standard; TAMP uses it for AoI |
| Timeliness | AoI → remote estimation, signal-aware beats age-optimal (Sun–Polyanskiy–Uysal, T-IT 2020); AoII; Urgency of Information [S] | Decision-time value instead of age | Perception events with a deadline set by robot motion (Proposition 6) |
| Networking | Sprout (stochastic capacity forecasts for a delay target, NSDI 2013); Salsify (codec–transport co-design, NSDI 2018); scalable video coding | Prediction-driven sizing; choosing among encodings | Task value conditional on the receiver, not visual quality; prediction from geometry ahead, not only past throughput |
| Robotics | Radio maps and communication maps (FERMI, RSS 2025; Amigoni et al.) [S] | Predicting the link along a planned path | Used to size perception payloads, not to plan motion |

---

## 4. Novelty, claim by claim

| # | Claim | Closest prior work | Difference | Confidence |
|---|---|---|---|---|
| C1 | **The formulation**: CP messaging as maximizing expected decision-time task value over the deadline-capacity distribution of a measured, contended link, with receiver side information. It nests rate–distortion CP, AoI scheduling and latency compensation as special cases (section 1) | Tian et al. (PHY, MSE, no deadline, no side information); ProgressFTX (one device, a known channel model, confidence stopping); SComCP and R-ACP (PHY robustness) [S] | The random quantity is the bytes that beat the deadline; value is task value conditional on what the receiver sees | Medium-high. Confirm that SComCP and ProgressFTX do not model a random deadline capacity |
| C2 | **Proposition 1 and Corollary 1a**: the gain of progressivity equals the partial-delivery mass, which is zero under every synthetic channel the CP literature uses | Broadcast-approach expected distortion (the PHY analogue) | Packet-layer form, exact and computable from traces; explains why CP never needed progressive messages | High for CP; the math is elementary, the insight is not in the CP literature |
| C3 | **Proposition 2**: with equal-size tokens, one ordering is within 1 − 1/e of a clairvoyant sender for every deadline-capacity distribution; with unequal sizes, no ordering is | NWF 1978 (the math); TokenCom (tokens as the unit, with no timing argument) [S] | A precise reason to tokenize time-critical messages | Medium. A corollary, to be presented as one |
| C4 | **Corollaries 3a and 3b**: prediction pays only through commitment costs; prediction and progressivity substitute; progressivity pays for the residual uncertainty | Salsify (prediction plus a choice of encodings), SVC streaming | States when each lever wins, and how to test it on traces | Medium-high |
| C5 | **Proposition 5**: the displacement number Π per class and IoU threshold separates the displacement regime (vehicles) from the delivery regime (indoor robots with a kinematic tag) | Latency-compensation papers, which assume the displacement regime implicitly | Places the existing literature in one regime and this work in the other | Medium |
| C6 | **Proposition 6**: the backlog price on bulk perception data equals event rate × warning value × miss sensitivity; priority access removes it | AoI and AoII scheduling; Sun et al. 2020 | Ties bulk sizing to time-critical events through one number | Medium |
| E | **Evidence**: the first test of these results on a measured indoor link, with S predicted from map, CSI and contention | CooperScene (outdoor C-V2X traces, no prediction, no progressive messages) | Only SCooP logs the link on the sensor clock and in the map frame | High, conditional on E3 validating the replay |

**What is not new.** These must be cited, never claimed:

- progressive and layered transmission;
- successive refinement and the broadcast approach;
- threshold stopping;
- Lyapunov scheduling;
- "signal-aware beats age-aware";
- codec–transport co-design;
- tokenized or instance-level CP messages;
- radio maps.

The contribution is the formulation, results C2–C6 in this setting, and the
measured evidence.

**Answers to the three reviewer objections to expect:**

- *"This is SVC or Salsify for perception."* Those maximize visual quality
  for one viewer. Here, value is conditional on what the receiver already
  sees, which makes the order a Wyner–Ziv problem (Proposition 2). Section 2
  also says when prediction beats progressivity and when it does not.
- *"This is ProgressFTX."* ProgressFTX stops on inference confidence for
  one device over a modeled channel. Here, stopping uses the probability of
  beating a deadline on a contended link, and the price of the backlog
  (Propositions 3 and 6).
- *"The broadcast approach did this in 2003."* That is the PHY analogue.
  At the packet layer, layering is free, so the trade-off collapses to the
  refinement penalty against partial-delivery mass (Corollary 1b). That
  mass is measurable only on a real link.

---

## 5. What each result predicts on SCooP

| Result | Prediction | Test | Needs |
|---|---|---|---|
| Proposition 1 | Gap > 0 under `logged`, ≈ 0 under `v2xvit` and `where2comm`, and larger on coop_2 than on survey_1 | Σ Δ_m P(B_m ≤ S < B_n) from the traces, with Δ ∝ bytes and with the `CommSchedule` placeholder values | Traces only (`PLAN.md` step 1) |
| Proposition 2 | Greedy token order within 1 − 1/e of the clairvoyant optimum; ratio order worse with unequal units | Replay with oracle values from labels | coop_2 labels |
| Proposition 3 | The optimal B* varies with x when λ, μ > 0; VoP > 0 only then | Size-matching policy against send-all, swept over λ | Traces plus the predictor (step 2) |
| Corollary 3b | The gain of progressivity shrinks as the predictor sharpens | Gap_n(x) under persistence, map and map+CSI predictors | Step 2 |
| Proposition 5 | Π_c ≪ 1 for SCooP people and robots with a velocity tag; Π ≈ 1 without | p95 speed × measured age per class, θ ∈ {0.5, 0.7} | Reference tracks (step 0) |
| Proposition 6 | Warning in-time rate falls with bulk backlog under FIFO and recovers under priority | `OcclusionHandoff-v0` replay with `,queue` | Event annotations |
