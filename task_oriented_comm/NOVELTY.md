# Novelty of the objective, reference by reference

This file writes each reference objective in its own notation, shows the
restriction of DTCV that recovers it, and states what DTCV keeps that the
reference drops. The published objectives are reconstructed from the
primary texts where the record is classical (C) and from abstracts where
it is not (S, see `REFERENCES.md`). An S reconstruction is a best reading
and must be checked against the paper before it is written up.

**DTCV, for reference:**

$$J(\pi)=\liminf_{T\to\infty}\frac1T\,\mathbb E\!\left[\sum_{k} w_k\!\!\sum_{\tau\in\mathcal T_k}\!\rho_k(a_\tau)\,F_k\big(\mathcal A^\pi_k(\tau)\,\big|\,M_k(\tau)\big)\;-\;\lambda\,\mathrm{Air}^\pi(T)\right]$$

with 𝒜_k^π(τ) the tokens that arrived by τ through the shared no-recall
queue, M_k(τ) the receiver's state, and the deadline capacity
S_k = (T_k − δ − W)⁺G random given x_t.

Five elements of DTCV are at stake. Each reference keeps at most three.

| Element | Symbol |
|---|---|
| E1 | value conditional on the receiver's state: ∂F_k/∂M_k ≠ 0 |
| E2 | ordered tokens with partial delivery: F evaluated on the arrived prefix, S random |
| E3 | a per-consumer deadline: value counted at τ ∈ 𝒯_k only |
| E4 | one shared, no-recall queue coupling all flows and all consumers |
| E5 | age only through residual error after correction: ρ_k(a), not a raw age penalty |

---

## 1. Rate-budget collaborative perception

### Where2comm (Hu et al., NeurIPS 2022) [C]

$$\max_{\mathbf M}\;\mathrm{Perf}\big(\Phi(X_\text{ego},\,\mathbf M\odot F_\text{helper})\big)\quad\text{s.t.}\quad\|\mathbf M\|_0\le B,$$

with 𝐌 a binary cell mask chosen from the helper's confidence map and
the ego's request map.

**Restriction of DTCV.** One consumer (the ego detector), one decision
time, S ≡ B deterministic, Air ≡ ‖𝐌‖₀, ρ ≡ 1, no queue. Then
E[F(𝒜(τ) | M)] = Perf(Φ(X_ego, 𝐌 ⊙ F)) − Perf(Φ(X_ego)), and DTCV is
Where2comm's problem.

**Kept by Where2comm.** E1, in part: the request map is the receiver's
state entering the value.

**Dropped.** E2 (S is a known budget, so a partial message never occurs),
E3, E4, E5. Under S ≡ B, Proposition 1 gives Gap ≡ 0: a progressive
message and an atomic one are identical in value, so the question DTCV
asks cannot be posed.

### RDComm (Liu et al., ICLR 2026) [V, objective from the abstract]

Pragmatic rate–distortion:

$$\min_{\text{enc},\,\text{dec}}\;\mathbb E\big[d_\text{task}(Y,\hat Y)\big]+\beta\,R .$$

**Restriction of DTCV.** The Lagrangian form of the Where2comm
restriction, with λ = β and F = −d_task. Same elements dropped. RDComm
also optimizes the code; DTCV takes the token code as given and optimizes
what is committed and when. The two compose: RDComm's code can be the
token source for DTCV. DTCV is not a generalization of the codec design.

### Information bottleneck (Shao, Mao, Zhang, JSAC 2022) [C]

$$\min_{p(z|x)}\;-I(Z;Y)+\beta\,I(Z;X).$$

Same restriction as RDComm with mutual information as the proxy. Single
link, no deadline, no receiver state.

---

## 2. Age of information

### AoI scheduling (Kadota et al., ToN 2018; Kaul et al. 2012) [C]

$$\min_\pi\;\lim_{T\to\infty}\frac1T\,\mathbb E\!\left[\sum_t\sum_k w_k\,A_k(t)\right],\qquad A_k(t)=t-u_k(t),$$

with u_k(t) the generation time of the freshest update delivered to k.

**Restriction of DTCV.** Take F_k(𝒜 | M) = c_k·1{𝒜 ≠ ∅} (an update is
worth c_k whatever it contains and whatever k holds), ρ_k(a) = 1 − a/Ā,
𝒯_k every slot. Then

$$\sum_\tau \rho_k(a_\tau)F_k=c_k|\mathcal T_k|-\frac{c_k}{\bar A}\sum_\tau A_k(\tau),$$

so maximizing DTCV is minimizing the weighted time-average age with
w_k c_k / Ā as the weights.

**Kept.** E4 (a shared channel, one transmission per slot) and E3 in a
weak form.

**Dropped.** E1 (∂F/∂M = 0 by construction: a redundant update and a
novel one age alike), E2 (updates are atomic), E5 (age enters raw, not
through the error it leaves). The AoI special case is exactly the
content-blind, atomic corner of DTCV.

### Age of incorrect information (Maatouk et al., ToN 2020) [C]

$$\min_\pi\;\lim\frac1T\,\mathbb E\!\left[\sum_t f(t-V(t))\,g\big(X_t,\hat X_t\big)\right],$$

with V(t) the last time the estimate was correct. This adds content
(g ≠ 0 only when the estimate is wrong) to age. It is DTCV with
F = −g(X, X̂) for a single scalar source, ρ·F replaced by a product of a
time penalty and an error indicator, atomic updates, one consumer.
Dropped: E1 beyond the scalar estimate, E2, E4 across flows.

### TAMP (Zhu et al., arXiv 2601.04542) [S, from the abstract]

Lyapunov drift-plus-penalty over regions r, with an empirical penalty
P_r(AoI_r, volume_r) fitted to detection AP:

$$\min_\pi\;\lim\frac1T\,\mathbb E\!\left[\sum_t\sum_r P_r\big(A_r(t),\,v_r(t)\big)\right]\quad\text{s.t. resource constraints.}$$

**Restriction of DTCV.** F_r a function of age and volume only, through
the fitted table; consumers are regions; the queue is an orthogonal
resource allocation. The per-slot Lyapunov decomposition is the same
machinery as `OBJECTIVE.md` Property 4.

**Kept.** E3, E4 in the resource-allocation sense, and volume as a crude
E2.

**Dropped.** E1 (the penalty does not depend on what the receiver holds
or sees), E2 in the ordered-prefix sense (volume is a scalar, not an
ordered message with a random cut), E5. TAMP is the nearest scheduling
objective and must be a baseline in step 4.

### Update the Unseen Only (Ma et al., arXiv 2607.20967) [V, from the abstract]

$$\min_\pi\;\lim\frac1T\,\mathbb E\!\left[\sum_t\sum_k\sum_{r\notin \mathcal V_k(t)} A_{k,r}(t)\right],$$

the age summed only over regions r the vehicle k cannot sense itself,
𝒱_k(t).

**Restriction of DTCV.** F_k(𝒜 | M_k) = c·1{𝒜 covers a region ∉ 𝒱_k}:
the first and only published objective here with ∂F/∂M ≠ 0, where M_k is
k's own field of view. Then ρ linear and atomic updates give the AoI
form above.

**Kept.** E1 (in the visibility form), E3, E4 (downlink from one base
station).

**Dropped.** E2 (atomic updates, no partial delivery, S not random in the
size sense), E5. This is the closest point of contact on E1. The
difference DTCV adds: M_k is the whole twin (what is mapped, when, and
what changed), not visibility alone; the update is an ordered prefix cut
by a random S; the queue is CSMA with no recall, not an allocated
downlink.

---

## 3. Remote estimation

### Sun, Polyanskiy, Uysal (T-IT 2020) [V]

$$\min_\pi\;\limsup\frac1T\,\mathbb E\!\left[\int_0^T (W_t-\hat W_t)^2\,dt\right]\quad\text{s.t. a sampling-rate constraint,}$$

with W a Wiener process, samples delivered after a random delay Y, and
Ŵ_t = W_{S_i} the last delivered sample. Signal-blind sampling gives
E[(W_t − W_{S_i})²] = t − S_i: the MSE **is** the age. Signal-aware
sampling is optimal and is a threshold on |W_t − W_{S_i}|.

**Restriction of DTCV.** One flow, one consumer, 𝒯 continuous,
F(𝒜 | M) = reduction in MSE given M = the last delivered sample, ρ through
the age, atomic samples, a single-server queue (E4 in the one-flow form).

**Kept.** E1 (the sampling decision depends on the realized signal
against the receiver's estimate: this is exactly "value conditional on
the receiver's state"), E4, E5 in the sense that the error, not the age,
is the cost.

**Dropped.** E2 (samples are atomic), E3 (no decision deadline; the cost
is integrated), multiple consumers. DTCV's Proposition 6 is the
event-with-deadline analogue of their threshold result; their theorem is
the strongest precedent for "signal-aware beats age-aware", and DTCV
cites it as such.

---

## 4. Progressive transmission and layering

### ProgressFTX (Lan et al., TWC 2022) [S, from the abstract and preprint]

Transmit feature blocks over slots n = 1, 2, …; stop at n to minimize

$$\min_{n}\;\;c\,n+\big(1-U_n\big),\qquad U_n=\text{inference confidence after } n \text{ blocks},$$

with the optimal rule: continue while U_{n+1} − U_n > c (Theorem 1 in the
preprint, for the Gaussian case).

**Restriction of DTCV.** One flow, one consumer, M fixed (the server has
nothing), S deterministic per slot (the channel is modeled, delivery is
certain), ρ ≡ 1, Air = n. Then the per-slot index of `OBJECTIVE.md`,
I_n = ΔF_n · P(S > B_n) − λβ/G, reduces to ΔU_n − c.

**Kept.** E2 (ordered blocks, stop early).

**Dropped.** E1, E3 (the stopping rule has no deadline; the cost is the
latency incurred, not a cut at a decision time), E4, E5. The factor
P(S > B | x) in DTCV's index is what ProgressFTX lacks: the marginal
block is discounted by the chance it misses the consumer's deadline on a
contended link, and the order itself is set by the receiver's state.

### Broadcast approach, expected distortion (Tian, Steiner, Shamai, Diggavi, T-IT 2008; Ng et al. 2009) [C]

Layers l = 1..L with rates R_l and powers P_l over a fading channel with
unknown gain h; layer l decodes iff h ≥ h_l. Minimize

$$\min_{\{P_l\}}\;\mathbb E_h\Big[D\Big(\sum_{l:\,h_l\le h}R_l\Big)\Big]\quad\text{s.t.}\quad\sum_l P_l\le P,$$

with D the distortion–rate function of the source.

**Restriction of DTCV.** Proposition 1 is the packet-layer form of this
expectation: with S in the role of h, tokens in the role of layers,
V = −D,

$$\mathbb E\big[V(N(\min(S,B)))\big]=\sum_m\Delta_m\,P(S\ge B_m)\;\;\leftrightarrow\;\;\mathbb E_h\Big[-D\Big(\sum_{l:h_l\le h}R_l\Big)\Big].$$

**Kept.** E2 exactly.

**Dropped and changed.** At the PHY, layering costs power, so the problem
is the allocation {P_l}. At the packet layer a prefix costs nothing, so
that optimization disappears and what remains is the choice of B
(Proposition 3) and the order (Proposition 2), both of which the
broadcast approach does not have because its layers are fixed by the
source code. E1 (no side information), E3 (h is the channel state, not a
consumer's deadline), E4, E5 are absent. The distortion is MSE, not task
loss.

### Salsify (Fouladi et al., NSDI 2018) [C]

$$\max_{e\in\{e_1,e_2\}}\;Q(e)\quad\text{s.t.}\quad |e|\le\hat C_t ,$$

choose between two encodings of the frame by a predicted capacity Ĉ_t.

**Restriction of DTCV.** One flow, one consumer, atomic message (no
prefix), S replaced by its point prediction, F = visual quality
independent of the receiver, ρ ≡ 1. Corollary 3b: with a sharp predictor
the atomic choice loses only the refinement penalty, which is Salsify's
design point.

**Kept.** The prediction-driven choice of size (the sizing half of E2).

**Dropped.** E1, E2 in the prefix sense, E3 (a target delay, not a
consumer's decision time), E4 across flows, E5.

### Scalable video coding (Schwarz et al., 2007) [C]

A representation, not an objective: a layered bitstream decodable at
every prefix. It supplies E2's code and nothing else.

---

## 5. Network utility maximization (Kelly, Maulloo, Tan, 1998) [C]

$$\max_{x\ge 0}\;\sum_s U_s(x_s)\quad\text{s.t.}\quad \sum_{s\ni l} x_s\le C_l\;\;\forall l .$$

**Restriction of DTCV.** F_k concave in throughput alone, no deadlines
(𝒯_k = every slot, ρ ≡ 1), content-blind. Keeps E4 (shared capacity),
drops E1, E2, E3, E5.

---

## 6. Semantic and task-oriented communication for CP

### SComCP (Gan et al., arXiv 2507.00895) and CoDS (Gan et al., arXiv 2512.22513) [S]

From the abstracts: learned encoders and decoders of CP features over a
PHY channel (SNR, fading, in CoDS a digital constellation), trained for
detection loss:

$$\min_{\text{enc},\,\text{dec}}\;\mathbb E_{\text{channel}}\big[\mathcal L_\text{det}\big(\Phi(X_\text{ego},\text{dec}(\text{ch}(\text{enc}(F))))\big)\big]\quad\text{s.t. a rate or SNR.}$$

**Restriction of DTCV.** The Where2comm restriction with the channel
inside the expectation at the symbol level. The randomness is in the
symbols, not in how many bytes beat a deadline; the message is atomic in
time. Dropped: E2 (no partial delivery; the whole frame is decoded with
noise), E3, E4, E5; E1 only through the ego features in the fusion. On
802.11 the MAC delivers whole packets or none, so the symbol-level
randomness these objectives optimize over does not reach the
application; DTCV's randomness is the one the application sees.

### R-ACP (Fang et al., JSAC 2025) [S, from the abstract]

Adaptive compression with priority-based scheduling and channel-aware
calibration, maximizing detection accuracy at a communication cost under
packet loss. A best reading:

$$\max_{\text{rate}_s,\,\text{prio}_s}\;\mathrm{Acc}\big(\{\text{stream}_s\}\big)-\gamma\sum_s\text{bits}_s\quad\text{under a loss process.}$$

**Kept, if the reading is right.** Parts of E2 (rate adaptation) and E4
(priorities across streams).

**Dropped, to confirm.** E1 (priorities and rates are per stream or
region, not conditional on what the receiver holds), E3 (accuracy, not
value at a decision time), E5. This is the closest systems paper, and
the claim that it lacks E1 and E3 must be checked against the full text
before it is written.

---

## 7. Summary table

| Objective | E1 receiver state | E2 ordered tokens, random S | E3 consumer deadline | E4 shared no-recall queue | E5 residual-error age |
|---|---|---|---|---|---|
| Where2comm, RDComm, IB | partial (request map) | – | – | – | – |
| AoI scheduling | – | – | weak | ✓ | – |
| AoII | scalar estimate | – | – | – | partial |
| TAMP | – | volume only | ✓ | allocation | – |
| Update the Unseen Only | ✓ (visibility) | – | ✓ | allocation | – |
| Sun–Polyanskiy–Uysal | ✓ (last sample) | – | – | one flow | ✓ |
| ProgressFTX | – | ✓ | – | – | – |
| Broadcast approach | – | ✓ (PHY) | – | – | – |
| Salsify | – | sizing only | target delay | – | – |
| NUM | – | – | – | ✓ | – |
| SComCP, CoDS | ego features | – | – | – | – |
| R-ACP (to confirm) | – | rate | – | priorities | – |
| **DTCV** | ✓ (the twin) | ✓ | ✓ | ✓ | ✓ |

No row other than the last has more than three of the five. Each
published objective is recovered from DTCV by fixing the elements it
lacks, as derived above.

---

## 8. The mathematical objects that are new

1. **The deadline capacity** S_k = (T_k − δ − W)⁺G as the random variable
   the sender optimizes against, in place of a bit budget (rate CP), a
   fading gain (broadcast), a delay target (Salsify) or an age (AoI). It
   is the quantity SCooP measures and the synthetic channels fix.
2. **The partial-delivery identity** (Proposition 1),
   E[V(N(min(S, B)))] − V(n)P(S ≥ B_n) = Σ_m Δ_m P(B_m ≤ S < B_n), and
   its corollary that the gain is identically zero under every channel in
   the CP literature. The identity is the packet-layer form of the
   broadcast-approach expectation; the corollary has no counterpart there
   because fading always has mass between layer thresholds.
3. **The index** I = w ρ(â) ΔF(u | M) P(S > B | x) − λβ/Ĝ, which carries
   all five elements: ΔF(· | M) is E1, the prefix order and P(S > B) are
   E2 and E3, λ and the Lyapunov queue are E4, ρ is E5. ProgressFTX's
   rule is the case ρ ≡ 1, P ≡ 1, M fixed; the AoI max-weight rule is the
   case ΔF ≡ c, ρ linear.
4. **The prefix-order guarantee under random S** (Proposition 2): the
   greedy order is within 1 − 1/e of the clairvoyant sender for every
   distribution of S at once, if and only if the units have equal size.
   NWF 1978 gives the bound for one budget; the "for every budget at
   once, hence for random S" step and the equal-size condition are the
   addition.
5. **Communication regret** R(π) = J(ideal) − J(π) per consumer, measurable
   because the same pass can be replayed under the ideal and the logged
   link and scored against one ground truth. No reference reports the
   gap to an ideal link per consumer, because none has both a measured
   link and a ground truth for the task.

**What is not new, stated once.** Progressive codes (Equitz–Cover),
side-information coding (Wyner–Ziv, Steinberg–Merhav), layering under an
unknown channel (Shamai), threshold stopping (Lan et al.), max-weight
and drift-plus-penalty (Neely, Kadota), signal-aware sampling (Sun et
al.), prediction-driven sizing (Salsify), visibility-conditioned updates
(Ma et al.). DTCV assembles them into one objective whose five elements
no reference holds together, and its new results are the ones that only
exist once they are together: items 1–5 above.
