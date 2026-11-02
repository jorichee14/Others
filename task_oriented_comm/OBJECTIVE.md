# The objective: decision-time conditional value (DTCV)

This file gives one objective for everything that crosses SCooP's shared
Wi-Fi link: control, poses, keyframe packs, map tiles and corrections. The
receiver and the server's twin are its consumers.

The objective has three properties no published objective combines:

- each byte's worth depends on what its receiver already knows;
- a byte counts only if it arrives by that receiver's own decision time;
- every flow competes in one queue that cannot be recalled.

Its gap to an ideal link, the **communication regret**, can be measured on
SCooP. The model and Propositions 1–6 are in [`THEORY.md`](THEORY.md).

---

## 1. Ingredients

| Symbol | Meaning | In the robot–server system |
|---|---|---|
| f ∈ 𝓕 | a flow: source, receiver, consumer task k(f) | control ↓, pose ↑, keyframe ↑, tiles ↓, corrections ↓, peer |
| u_{f,m} | token m of flow f; every token has the same size β, one MTU packet (about 1.4 KB) | pose and control fill one token; a keyframe pack is about 200–350 tokens |
| 𝒯_k | the decision times of consumer k | control period; 10 Hz sim and radio step; SLAM and twin updates; localization |
| M_k(τ) | what consumer k already holds at τ: its **side information** | the twin (map, splats, scene), the robot's own tiles and scan, the last pose |
| 𝒜_k^π(τ) | the tokens for k that have **arrived by τ** under policy π | set by the shared queue and the link (G_t, δ_t, outages) |
| a | age of the content when used | capture to τ |
| ρ_k(a) ∈ [0, 1] | the share of value left after the receiver's kinematic correction: P(residual error < Δ_max,k) | ≈ 1 when the displacement number Π_c ≪ 1 (THEORY §2.5) |
| ℓ_k(b, X) | consumer k's task loss for belief b against the true world X | SLAM ATE, detection miss, twin divergence from the reference map, missed control deadline |
| x_t | the sender's predictive information | planned path, Sionna or clearance radio map, CSI, recent goodput, the other robot's load |

**Conditional value of a delivered set.** This is the expected loss
reduction from fusing the tokens in A, corrected to the age a, with what k
already holds:

$$F_k(\mathcal A\mid M)\;=\;\mathbb E\big[\ell_k(\Phi_k(M),X)\;-\;\ell_k(\Phi_k(M\cup\mathcal A),X)\big].$$

Here Φ_k is the consumer's fusion: SLAM, the splat mapper, change
detection, or the controller.

---

## 2. The objective

$$\boxed{\;\max_{\pi}\;\;\liminf_{T\to\infty}\frac1T\,\mathbb E\!\left[\sum_{k} w_k\!\!\sum_{\tau\in\mathcal T_k\cap[0,T]}\!\!\rho_k(a_\tau)\,F_k\!\big(\mathcal A^\pi_k(\tau)\,\big|\,M_k(\tau)\big)\;-\;\lambda\,\mathrm{Air}^\pi(T)\right]\;}$$

The maximization is subject to four conditions:

1. **Causality.** π chooses, from x_t and the past, which tokens to commit
   and when.
2. **No recall.** Once a token is committed to the driver or the AP queue,
   it cannot be cancelled.
3. **The shared queue.** 𝒜_k^π(τ) is what the AP delivers by τ, given every
   flow's committed tokens, the queue discipline, and the link
   (G_t, δ_t, outage) ~ P(· | x_t).
4. **Side information updates.** M_k changes with what arrives, so a change
   already delivered is worth nothing the second time.

**The weights.** w_k puts every consumer's loss in one unit. One choice is
mission cost per failure: a missed stop command, a lost localization, an
undetected person. Another is to report the Pareto front over w.

**Communication regret.** This is the form to report:

$$\mathcal R(\pi)\;=\;J(\text{ideal link})-J(\pi)\;\ge 0 .$$

The regret is zero exactly when communication costs the task nothing. It
can be measured, because SCooP replays the same pass under `ideal` and
`logged` and scores ℓ_k against the reference trajectory, the labels and
the reference map.

---

## 3. Why it is new: the published objectives are special cases

| Restriction of DTCV | Gives | Literature |
|---|---|---|
| One consumer, one decision, capacity S deterministic, ρ ≡ 1, M fixed | max F(A) − λ\|A\| subject to \|A\|β ≤ S | Rate–distortion and pragmatic CP (RDComm, Where2comm, InfoCom) |
| F_k(A \| M) = c·1{A ≠ ∅} (blind to content), ρ_k(a) = 1 − f(a)/c | minimize the time average of f(age) | AoI scheduling (Fresh2comm, TAMP, WiSwarm) |
| 𝒜 = everything (the link always delivers), optimization over Φ_k only | best fusion of late data | Latency compensation (CoBEVFlow, FFNet, TraF-Align) |
| One flow, scalar Gauss–Markov source, F = reduction in MSE | signal-aware sampling | Remote estimation (Sun–Polyanskiy–Uysal 2020) |
| F concave in throughput, no deadlines | network utility maximization | Kelly, NUM |
| One device, stop on confidence, a modeled channel | progressive feature transmission | ProgressFTX (Lan et al., TWC 2022) |

Each published objective drops at least two of these four elements:

1. **Value conditional on the receiver's state.** The twin is the side
   information, so a token's worth differs by receiver and by time.
2. **Value counted only if it arrives by each consumer's own decision
   time**, with flows whose deadlines range from milliseconds (control) to
   seconds (tiles) in one sum.
3. **One shared queue with no recall.** Priority and size are outputs of
   the optimization, not hand-set inputs.
4. **Age only through the residual error after correction.** The term is
   ρ_k(a), not f(a).

DTCV is the first objective found that keeps all four. Claim this only
after reading SComCP, ProgressFTX and TAMP in full.

---

## 4. Structure and solution

**A1 (assumption).** F_k(· | M) is monotone submodular: weighted coverage
of cells, objects or map regions that M does not already explain.

**Property 1: emergent priority.** A control token has a large
w_k·F_k/β and the earliest deadline, so the optimizer always selects it
first. The mapping to access categories (control and pose → AC_VO;
keyframes → AC_BE; tiles → AC_BK) is therefore derived, not assumed.

**Property 2: the decision for one slot is a matroid problem.**

- Take tokens of equal size and a known capacity for each deadline window.
- Call a token set feasible if an earliest-deadline-first (EDF) schedule
  delivers every token by its deadline.
- Feasible sets form a matroid: unit tasks with deadlines, a transversal
  matroid (Cormen et al., §16.5).

Hence:

- **Select by value, transmit by deadline.** Choose tokens greedily by
  marginal value w_k ρ_k ΔF_k. Send the chosen tokens in EDF order.
- This is **optimal** for additive F.
- It is a **½-approximation** for submodular F under a matroid constraint
  (Fisher–Nemhauser–Wolsey 1978).
- Selection order and transmission order differ. Ranking by deadline alone
  (EDF) or by value alone (static priority) is suboptimal.

**Property 3: random capacity.** Replace each window's capacity with its
q-quantile under P(· | x_t). The greedy step then runs on a
chance-constrained matroid. The quantile q trades value against deadline
risk, and this is where link prediction enters (THEORY Corollary 3a).

**Property 4: the long run.** Add a virtual queue for the airtime budget
and apply drift-plus-penalty (Neely 2010). Each slot then maximizes:

> Σ ρ ΔF − (λ + Q(t)/V)·airtime

The bound is an O(1/V) loss in objective for an O(V) backlog. This
guarantee covers the one-slot relaxation with M held fixed. Because M
evolves with what arrives, the full policy is greedy over time (myopic in
M). State this as a limitation.

**Index used in practice.** For token m of flow f at time t:

$$I_{f,m}(t)=w_k\,\rho_k(\hat a)\,\Delta F_k(u_{f,m}\mid M_k)\;-\;\lambda\,\beta/\hat G_t .$$

Commit tokens with I > 0 in descending I while the q-feasibility check
passes. Send them in EDF order. Recompute ΔF lazily after each selection.
For keyframes, ΔF comes from the robot comparing its scan with its twin
tiles (change detection). For tiles, it comes from the planned path
entering areas without tiles.

---

## 5. How it is measured on SCooP

| Consumer k | ℓ_k scored against | Decision times 𝒯_k |
|---|---|---|
| Control | deadline missed (indicator) | control period |
| Sim and radio pose | pose error at τ against the reference trajectory; Sionna CIR against the measured CSI | 10 Hz |
| Swarm-SLAM | ATE and RPE against the reference | each SLAM update |
| Twin / change detection | time from a scene change to its appearance in the twin; twin-to-reference-map divergence | each scene update |
| Localization (tiles) | map→odom failures; error against the reference | per tile request |

**Policies compared:**

- send everything over FIFO, as the system is built today;
- fixed WMM priority;
- an AoI scheduler;
- greedy DTCV with persistence, clearance or Sionna prediction.

Each runs under `ideal`, `fit:` and `logged`.

**Headline:** the regret per consumer, and the fact that the policies rank
differently under `logged` than under synthetic channels.
