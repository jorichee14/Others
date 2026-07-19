# A Complete Guide — Task-Conditioned Token Communication for Collaborative Perception

*Everything in one place: the problem, why it matters, what is missing in prior work, the precise
novelty, the mathematics (objective, relaxations, solvers, guarantee), the method, and the exact
steps to execute — with honest justification against existing methods at every turn.*

*Prior art verified against July-2026 literature; the two closest works (STAMP, Scene Completion)
were read in full text. Companion artifacts on this branch: `FINAL_DIRECTION.md`, `THESIS.md`,
`NOVELTY_ANALYSIS.md`, `phase0_axis_check.py`, `toy_cp_sandbox.py`.*

---

## 0. How to read this

The guide is linear. Sections 1–5 fix *what* you are doing and *why*. Sections 6–11 are the
*mathematics* — system model, the optimization objective in three equivalent forms, the relaxations
and solvers, and the certification layer. Section 12 is the *justification vs existing methods*.
Sections 13–17 are *execution*: the dual-link extension, the staged build plan with equations tied to
each step, experiments, risks, and reading. If you only read one technical section, read **§8
(the optimization objective)** — it is the intellectual core.

---

## 1. Executive summary

Collaborative perception (CP) lets connected agents share intermediate features to see through
occlusion, but bandwidth is the binding constraint, so every system must decide *what to transmit*.
The field is converging on **general, multi-task features** (one shared representation feeding
detection, segmentation, prediction). This creates an unsolved tension: **you cannot prune
"task-irrelevant" content when the representation must serve many tasks, and a feature's value is
task-dependent and unknown at send time.** Existing methods either optimize a single fixed task
(Where2comm, JigsawComm), send a task-*agnostic* message (Scene Completion, STAMP), or compress the
payload (V2X-DSC, DiffCP) — **none conditions the transmission decision on the receiver's task demand.**

This guide develops **task-conditioned token communication**: agents transmit a shared discrete
**token** stream; the receiver announces a **task-demand vector**; the sender solves a **budget-
constrained multi-task allocation** to choose the token subset that maximizes weighted task utility
(or protects the worst task), with a **per-task preservation guarantee** that each requested task
degrades by at most a controlled amount. The novelty is the *decision* (task-conditioned, jointly
allocated, certified), not the representation. It is a solid, verified-open contribution — comparable
in weight to STAMP — not a paradigm shift.

---

## 2. The problem

**Setting.** Agents $\mathcal{N}$ observe a shared scene. For a directed pair $j\to i$, sender $j$
holds a feature map decomposed into transmissible **tokens** $\mathcal{U}_j=\{u\}$, each with payload
size $b(u)$ bits. Receiver (ego) $i$ fuses received tokens with its own features and runs one or more
downstream **tasks** $t\in\mathcal{T}=\{1,\dots,M\}$ (e.g. 3D detection, BEV segmentation, motion
prediction). A link budget $B$ (bits per round) caps what can be sent.

**The decision.** For each token, transmit it or not (later: and over which link). Formally choose
$x_u\in\{0,1\}$, $S=\{u:x_u=1\}$, subject to $\sum_{u\in S} b(u)\le B$.

**The open problem.** When the receiver runs *several* tasks off one shared token stream, and a token's
worth differs across tasks, *what is the token subset to send under the budget so that no requested
task is silently starved?* No existing CP method conditions this decision on the receiver's task demand.

---

## 3. Motivation — why, and why now

- **Generality is the trend, and it breaks pruning.** CP is moving to task-agnostic / foundation
  features (Scene Completion, STAMP, DriveX-2026). Single-task selectors prune what one task ignores;
  that mechanism is invalid when the features must serve many tasks. The more general the
  representation, the *worse* the bandwidth problem — you can no longer call anything "irrelevant."
- **Value is genuinely task-dependent.** A token on a partially-occluded pedestrian is critical for
  detection, marginal for drivable-area segmentation; a token on distant lane geometry is the reverse.
  A single scalar "importance" cannot serve both — demonstrated empirically in `toy_cp_sandbox.py`
  (Rung 3: $\mathrm{corr}(V_{\text{det}},V_{\text{seg}})\approx 0$, top-K sets disjoint).
- **Silent starvation is a safety problem.** Optimizing the shared budget for detection can quietly
  destroy segmentation with no signal — the same "uncertified proxy" failure that JigsawComm and
  HydraCollab exhibit, now across tasks.
- **Timing.** The task-agnostic representation work (STAMP ICLR'25) and the payload-compression work
  (V2X-DSC 2026) both leave the *task-conditioned decision* untouched. The window is ~12–18 months.

---

## 4. Existing methods and the exact gap in each

| Method | What it does | Exact gap you exploit |
|---|---|---|
| Where2comm (NeurIPS'22) | spatial-confidence region selection | single fixed task; scalar importance; task-blind |
| JigsawComm (2025) | learned utility select + top-1 redundancy | **uncertified** proxy; single task; no per-task value; raw payload |
| HydraCollab (2026) | sensor gating + inter/late hybrid on overlap | inherits Where2comm-map error; single task; float32 payload; no link model |
| CoSDH (CVPR'25) | supply–demand region selection | "demand" is **spatial** (blind spots), **not task** demand |
| V2X-DSC (2026) | Wyner–Ziv conditional codec, VQ + entropy | payload compression only; **task-blind**; assumes keep-set already chosen |
| DiffCP / InfoCom | diffusion / info-bottleneck extreme compression | compress payload; single task; no task-conditioned selection |
| Scene Completion / STAR (CoRL'22) | task-agnostic shared feature; sub-sampled tokens | drops tokens by **spatiotemporal redundancy** (task-blind); one message for all tasks; no budget adaptation; no link failure *(full-text verified)* |
| STAMP (ICLR'25) | task/model-agnostic protocol via adapter-reverter | **uniform** compressor; **task-agnostic by design**; tests pose noise, **not** link loss *(full-text verified)* |
| SComCP / task-oriented CP | compress for one task | single-task by construction |
| *(ancestors)* layered coding + UEP; multi-RAT | base/enh over reliable/best-effort; parallel radios | fixed codec layers (not content/task-driven); content-blind or duplication-for-reliability |

**Pattern:** every method optimizes one fixed task, or sends a task-agnostic message, or compresses the
payload. **None conditions the transmit decision on the receiver's task demand, allocates one budget
across contending tasks, or guarantees per-task preservation.** That triple is the opening.

---

## 5. The core idea (thesis)

> Transmit a shared **token** stream; let the receiver declare a **task-demand vector** $w$; solve a
> **budget-constrained multi-task allocation** for the token subset $S$ that maximizes weighted task
> utility (or protects the worst task), **certified** so each requested task degrades $\le\varepsilon_t$.

Three claims, in order of defensibility:
1. **Task-conditioned transmit-value** (core): value is $V_t(u)$, a vector over tasks; the send decision
   depends on $w$.
2. **Multi-task budget allocation** (non-triviality): tasks contend for one budget; not $M$ independent
   selectors.
3. **Per-task preservation guarantee** (feature): certification that no task is silently starved.

---

## 6. System model and notation

| Symbol | Meaning |
|---|---|
| $\mathcal{U}$ | set of candidate tokens from the sender (size $N=\lvert\mathcal{U}\rvert$) |
| $b(u)$ | payload size of token $u$ (bits); uniform $b$ ⇒ budget = token count |
| $B$ | link budget (bits) |
| $\mathcal{T}=\{1..M\}$ | receiver's active tasks |
| $w=(w_1,\dots,w_M)$ | task-demand weights, $w_t\ge0,\ \sum_t w_t=1$ (declared by receiver, may vary per frame) |
| $x_u\in\{0,1\}$ | transmit token $u$ or not; $S=\{u:x_u=1\}$ |
| $P_t(S)$ | performance of task $t$ (e.g. AP, mIoU) when receiver holds $S$ |
| $v_t(u)\ge 0$ | per-token value of $u$ to task $t$ (see §11) |
| $D_t(S)$ | degradation of task $t$ vs full transmission: $D_t(S)=P_t(\mathcal{U})-P_t(S)$ |
| $\varepsilon_t$ | tolerated degradation for task $t$ |

**Modular (first-order) model.** Assume $P_t(S)\approx \sum_{u\in S} v_t(u)$ and normalize
$\bar v_t(u)=v_t(u)/\sum_{u'} v_t(u')$ so tasks are comparable. Then the *recovered utility* of task $t$
is $U_t(S)=\sum_{u\in S}\bar v_t(u)\in[0,1]$ and the degradation is the absent mass
$D_t(S)=\sum_{u\notin S}\bar v_t(u)=1-U_t(S)$. (Reality is submodular — diminishing returns — which we
handle in §9; the modular model is the tractable starting point and what the toy uses.)

---

## 7. Definitions

- **Token** $u$: a discrete, entropy-codable unit of the shared representation (a BEV patch token, or a
  VQ codeword). Tokens are the substrate — *not* the contribution.
- **Per-task value** $v_t(u)$: how much token $u$ helps task $t$ at the ego (§11 gives three estimators).
- **Task-demand vector** $w$: the receiver's declared active tasks and priorities (safety-critical tasks
  weighted higher; can change frame to frame — e.g. planning suddenly needs prediction tokens).
- **Budget** $B$: bits available this round on the link.
- **Preservation** : task $t$ is *preserved* if $D_t(S)\le\varepsilon_t$.

---

## 8. The optimization objective (the intellectual core)

We give three formulations. They are connected by Lagrangian duality (§9). Choose by intent.

### (A) Weighted-sum scalarization — the workhorse
Maximize demand-weighted total utility under the budget:
$$
\max_{x\in\{0,1\}^N}\ \sum_{t\in\mathcal{T}} w_t\, U_t(S)
=\sum_{u\in\mathcal{U}} x_u\underbrace{\Big(\sum_{t} w_t\,\bar v_t(u)\Big)}_{c(u)\ \text{(combined value)}}
\quad\text{s.t.}\quad \sum_{u} x_u\, b(u)\le B .
$$
This is a **0/1 knapsack** with item value $c(u)=\sum_t w_t\bar v_t(u)$ and weight $b(u)$. **The
contention lives in $c(u)$**: a token serving two tasks accumulates value from both, so it is preferred
— which is exactly what independent per-task selectors miss.

### (B) Max–min (egalitarian) — protect the worst task
Weighted-sum can still starve a low-weight task. To *guarantee balance*, maximize the worst task:
$$
\max_{x\in\{0,1\}^N}\ \min_{t\in\mathcal{T}} U_t(S)\qquad\text{s.t.}\qquad \sum_u x_u b(u)\le B .
$$
This is the fair formulation — it is what makes "no task is silently starved" an *objective*, not an
afterthought. It is harder (non-modular even under the additive model; see §9).

### (C) Guarantee-constrained — the deployable form
Send the *fewest bits* such that every task is preserved:
$$
\min_{x\in\{0,1\}^N}\ \sum_u x_u\, b(u)\qquad\text{s.t.}\qquad D_t(S)\le \varepsilon_t\ \ \forall t\in\mathcal{T}.
$$
This is a **multi-dimensional covering** problem (dual to A). It directly encodes claim #3 — the
$\varepsilon_t$ are the per-task guarantees — and §10 shows how conformal calibration sets the operating
point so the constraints hold *distribution-free*.

**Why this is a genuine joint problem (not $M$ selectors).** In (A), the combined value $c(u)$ couples
all tasks through one budget; in (B)/(C) the coupling is explicit in the $\min$ / the shared $x$. Running
$M$ independent top-$K$ selectors either **overspends** (union $>B$) or, if you split the budget blindly,
**double-charges** tokens that serve multiple tasks and starves the diffuse task (demonstrated in the toy:
naive weighted-sum collapsed to one task; a fixed 50/50 split was crude). The allocation *is* the
contribution.

---

## 9. Relaxations and solvers

### 9.1 Weighted-sum (A)
- **Uniform payload** $b(u)\equiv b$: the knapsack degenerates to "take the top-$\lfloor B/b\rfloor$
  tokens by $c(u)$." Exact, $O(N\log N)$ — this is the toy's `select_topk`.
- **Variable payload**: 0/1 knapsack is NP-hard but trivially approximable. **LP relaxation**
  $x_u\in[0,1]$ and the **fractional-knapsack greedy** by density $c(u)/b(u)$ gives an optimum within one
  item of the LP bound; round down the fractional item. Integrality gap $\le \max_u c(u)$.
- **Submodular reality**: true $P_t$ has diminishing returns, so $U_t$ is monotone **submodular**, and
  $\sum_t w_t U_t$ is a nonnegative weighted sum of monotone submodular functions — again monotone
  submodular. Maximizing a monotone submodular function under a knapsack constraint: the **cost–benefit
  greedy** (add $\arg\max_u \Delta(u)/b(u)$, $\Delta(u)$ = marginal gain), with partial enumeration of
  small seed sets, achieves a $(1-1/e)$ approximation (Sviridenko 2004; Nemhauser–Wolsey–Fisher 1978).
  This is the principled solver and gives you a *theorem* to state.

### 9.2 Max–min (B)
Robust/egalitarian submodular maximization is NP-hard but has good algorithms:
- **Saturate** (Krause et al. 2008): binary-search a target level $c$; at each level solve a *truncated*
  submodular cover $\sum_t \min(U_t(S),c)$ (submodular) greedily; return the largest feasible $c$.
  Guarantees the worst task reaches $\approx$ the optimum up to a bounded budget over-run.
- **Subgradient / adaptive-weights**: run (A) but *update* $w$ by a subgradient step on the $\min$ —
  increase $w_t$ for the currently-worst task, re-solve, iterate. Converges to the max–min point and is
  trivial to implement on top of the knapsack solver. (This is the clean thing to build first.)
- **Round-robin-by-shortfall greedy**: at each step add the token with the best marginal gain *for the
  currently-lowest-utility task*. Cheap, intuitive, a strong baseline.

### 9.3 Guarantee-constrained (C) via Lagrangian duality — *the elegant link*
Relax the per-task constraints with multipliers $\lambda_t\ge0$:
$$
\mathcal{L}(x,\lambda)=\sum_u x_u b(u)+\sum_t \lambda_t\big(D_t(S)-\varepsilon_t\big).
$$
For fixed $\lambda$, the inner minimization over $x$ is separable and each token is included iff its
"guarded value" exceeds its bit cost:
$$
x_u^\star=\mathbf{1}\!\left[\ \sum_t \lambda_t\,\bar v_t(u)\ \ge\ b(u)\ \right].
$$
So **the dual variables $\lambda$ are exactly the task weights**, and the operating rule is: score token
$u$ by $\sum_t\lambda_t\bar v_t(u)$ per bit, include the high scorers. Solve by **dual ascent**: raise
$\lambda_t$ for any task whose constraint is violated ($D_t>\varepsilon_t$), lower it otherwise, until all
tasks are preserved. This unifies (A), (B), (C): weighted-sum with the *right* weights *is* the guarantee.
The remaining question — *what are the right weights so the guarantee actually holds on unseen data?* — is
answered by §10.

---

## 10. The per-task guarantee (certification as a feature)

The solver hits $D_t(S)\le\varepsilon_t$ **on the estimated $\bar v_t$**. But $\bar v_t$ is a proxy; the
real per-task degradation may exceed $\varepsilon_t$. **Conformal risk control** closes this gap
distribution-free.

Let the policy have a scalar knob $\tau$ (e.g. the inclusion threshold, or the budget). Define the
per-task risk $R_t(\tau)=\mathbb{E}[\,D_t(S_\tau)\,]$, which is monotone non-increasing in "how much we
send." On a held-out **calibration set** of $n$ frames, choose $\hat\tau$ so that, for every task,
$$
\mathbb{E}\big[D_t(S_{\hat\tau})\big]\ \le\ \varepsilon_t + O(1/n)\qquad\text{distribution-free, finite-sample,}
$$
using **Conformal Risk Control** (Angelopoulos et al. 2022) per task, or **Learn-Then-Test**
(Angelopoulos et al. 2025) to control the $M$ risks *simultaneously* (it handles multiple hyper-parameters
/ multiple risks with family-wise validity). The guarantee holds *regardless of how miscalibrated
$\bar v_t$ is* — that is the point, and it is what turns "we allocated the budget" into "no task degrades
more than $\varepsilon_t$, provably." Certification is a **feature** here, not the thesis; the thesis is the
task-conditioned allocation it certifies.

---

## 11. Estimating the per-task value $v_t(u)$

Three estimators, cheap→faithful:
1. **Gradient saliency (cheapest):** $v_t(u)=\big\lVert \partial \mathcal{L}_t/\partial f_u\big\rVert$ — how
   much task-$t$ loss moves with token $u$'s features. One backward pass per task.
2. **Leave-one-out ablation (faithful):** $v_t(u)=P_t(S)-P_t(S\setminus\{u\})$ — the definition; expensive
   ($N$ forward passes), used to *validate* the cheap proxies.
3. **Learned per-task utility head (deployable):** a small head predicts $v_t(u)$ from features (à la
   JigsawComm's estimator, but **one per task**, and **calibrated** per §10). Trained with a
   bandwidth-regularized multi-task loss $\mathcal{L}=\sum_t w_t\mathcal{L}_t+\gamma\sum_u x_u b(u)$.

Novelty note: everyone estimates a single importance; you estimate a **task-indexed, calibrated** value
$\bar v_t(u)$. That vector is what makes claims #1–#3 possible.

---

## 12. Why do this instead of the existing methods (head-to-head)

- **vs single-task selectors (Where2comm, JigsawComm, HydraCollab):** they optimize one $P_t$; run a second
  task off their selection and it is starved ($\mathrm{corr}(v_{\text{det}},v_{\text{seg}})\!\approx\!0$).
  You optimize the *set* of active tasks jointly under one budget.
- **vs task-agnostic representations (Scene Completion, STAMP):** they send *one message for all tasks*,
  chosen task-blind (STAR by spatiotemporal redundancy, STAMP by a uniform compressor). Under a tight
  budget that is provably wasteful — you send the *demand-weighted* subset. You **build on** their
  representation, you do not compete with it.
- **vs payload compressors (V2X-DSC, DiffCP, InfoCom):** orthogonal — they shrink the *bits per token* and
  assume the keep-set is already right. You choose the keep-set per task; their codec can sit *under* your
  selection (stackable).
- **vs multi-RAT / layered UEP (ancestors):** they route by fixed codec layers or content-blind reliability;
  your partition is *task-value-driven*, and (in the Paper-2 extension) survives link failure per task.

The one-sentence differentiator: **prior work chooses what to send by one task, or by no task; you choose
by the receiver's task demand, allocate one budget across contending tasks, and certify each task.**

---

## 13. Extension (Paper 2) — task-conditioned routing across links

Generalize $x_u\in\{0,1\}$ to $a(u)\in\{\varnothing,\text{best-effort},\text{reliable}\}$ with two link
budgets $C^r,C^b$ and best-effort failure probability $p^b$. Add the **failure-survival** constraint so the
reliable partition alone preserves every task:
$$
\min \sum_{u:a(u)=r} b(u)\quad\text{s.t.}\quad D_t\big(\{u:a(u)=r\}\big)\le \varepsilon_t\ \forall t,\ \
\text{capacities } C^r,C^b .
$$
This adds link-failure robustness — a differentiator vs STAR/STAMP (which model no link failure) — but
requires radio/failure simulation. **Defer it**; it is not Paper 1.

---

## 14. The build plan (steps, with equations tied to each)

Each milestone is small, gated, and cheap to kill.

**M1 — Master the toy (this week).** Read/poke `toy_cp_sandbox.py`. Understand fusion, the budget curve
$U(K)$, and $\mathrm{corr}(v_{\text{det}},v_{\text{seg}})$. *Exit:* you can predict the effect of any
parameter change before running.

**M2 — Build the real allocator in the toy (days). ← invent the contribution.**
Replace the 50/50 split with: (a) the **cost–benefit greedy** for (A); (b) the **subgradient
adaptive-weight** loop for (B) max–min; (c) the **dual-ascent** loop for (C) with the per-task
$\varepsilon_t$ check. Report worst-task utility and per-task $D_t$. *Exit (G-alloc):* your allocator beats
task-blind, raw-sum, weighted-sum, and 50/50 on worst-task at equal budget.

**M3 — Cross to real features (weeks). ← the true de-risk.**
- *3a infra:* run STAMP (or OpenCOOD/Where2comm) on OPV2V on a GPU; reproduce one published number.
- *3b real G1:* attach detection + BEV-seg heads; compute $\bar v_t(u)$ per token (gradient saliency,
  validated against ablation); re-measure $\mathrm{corr}(v_{\text{det}},v_{\text{seg}})$ on **real** BEV
  features. *Exit (G1):* correlation low on real data ⇒ thesis alive; high ⇒ pivot (cheap lesson).

**M4 — Real G0/G2 (weeks). ← the headline.**
Run M2's allocator on real features; sweep $B$; compare against (i) task-blind detection selector,
(ii) task-agnostic (STAR/STAMP-style) send, (iii) staged per-task union. Plot per-task $U_t(B)$ and
worst-task. *Exit (G2):* task-blind starves segmentation; task-conditioned recovers both, gap concentrated
at low $B$.

**M5 — Guarantee + write-up.** Add conformal per-task control (§10); show empirical $D_t\le\varepsilon_t$
holds on a test split across the $B$-sweep; ablate the three value estimators; write related work (§4,
full-text-verified). Draft.

**Parallel:** read STAMP, STAR, JigsawComm, one conformal-risk-control paper, one multi-task-learning
survey; and write §8 cleanly on paper — stating the objective crisply is half the paper.

---

## 15. Experiments

- **Datasets:** OPV2V, V2V4Real (STAMP's own — det + BEV-seg, public code), DAIR-V2X, V2X-Sim (det +
  seg + tracking).
- **Tasks:** 3D detection (AP@0.5/0.7), BEV segmentation (mIoU), optionally motion prediction.
- **Baselines:** task-blind single-task selector (Where2comm/JigsawComm-style), task-agnostic send
  (STAR/STAMP-style), staged per-task union, and your task-conditioned allocator (A/B/C variants).
- **Metrics:** per-task performance vs budget $B$; **worst-task** performance; communication bits;
  fraction of frames with $D_t\le\varepsilon_t$ (guarantee coverage).
- **Headline plot:** per-task and worst-task performance vs $B$ — task-conditioned dominates in the
  low-budget regime; baselines starve a task.
- **Ablations:** value estimator (saliency vs ablation vs learned); demand vector shift (change $w$ at
  test time and show adaptation); guarantee coverage vs $\varepsilon_t$.

---

## 16. Risks and kill-criteria

- **G1 fails on real data** (tasks don't diverge) → the whole thesis dies; *run 3b before building M4.*
  Cheapest kill-switch — do it early.
- **"Just run a selector per task"** (reviewer) → defended by (B)/(C): shared-budget contention makes it a
  joint problem; independent selectors overspend or starve. Make the max-min / guarantee the centerpiece.
- **Foundation-model CP encroaches** → you optimize the *communication decision under constraint*, orthogonal
  to representation; bigger general features make the budget problem worse, not obsolete.
- **Submodularity doesn't hold** (fusion interactions) → the modular model still gives a strong baseline;
  use greedy empirically and drop the $(1-1/e)$ claim if the assumption fails.
- **Demand vector is unrealistic** → ground $w$ in a real multi-task stack (planning needs prediction now,
  parking needs segmentation) and show dynamic adaptation.

---

## 17. Reading list and references

**Build these into your head first:**
- STAMP (ICLR'25) — task/model-agnostic CP — https://arxiv.org/abs/2501.18616 · code https://github.com/taco-group/STAMP
- Multi-Robot Scene Completion / STAR (CoRL'22) — task-agnostic CP — https://openreview.net/forum?id=hW0tcXOJas2 · code https://github.com/coperception/star
- Where2comm (NeurIPS'22) — https://arxiv.org/abs/2209.12836
- JigsawComm (2025) — https://arxiv.org/abs/2511.17843
- V2X-DSC (2026) — https://arxiv.org/abs/2602.00687

**Optimization / theory:**
- Nemhauser, Wolsey, Fisher (1978) — greedy for monotone submodular maximization, $(1-1/e)$.
- Sviridenko (2004) — submodular maximization under a knapsack constraint, $(1-1/e)$.
- Krause, McMahan, Guestrin, Gupta (2008) — *Saturate*: robust (max–min) submodular optimization.

**Certification:**
- Angelopoulos, Bates, Fisch, Lei, Schuster — *Conformal Risk Control* — https://arxiv.org/abs/2208.02814
- Angelopoulos, Bates, Candès, Jordan, Lei — *Learn Then Test* (AoAS'25) — https://arxiv.org/abs/2110.01052
- Bates et al. — *Distribution-Free Risk-Controlling Prediction Sets* (JACM) — https://dl.acm.org/doi/10.1145/3478535

**Context / baselines:**
- CoSDH (CVPR'25) https://arxiv.org/abs/2503.03430 · DiffCP https://arxiv.org/abs/2409.19592 ·
  InfoCom https://arxiv.org/html/2512.10305 · SComCP https://arxiv.org/pdf/2507.00895

---

### One-paragraph statement of the whole thing
Collaborative-perception agents are moving to general, multi-task features, but bandwidth forces them to
send a subset — and today they choose that subset for one fixed task or for no task at all, silently
starving whatever task they did not optimize. We transmit a shared token stream and let the receiver
declare a task-demand vector; the sender solves a budget-constrained multi-task allocation — a knapsack
over demand-weighted per-task token value (weighted-sum), a max–min program that protects the worst task,
or, dually, the fewest bits that keep every task within $\varepsilon_t$ — and certifies each task's
preservation distribution-free with conformal risk control. The representation is borrowed; the novelty is
the task-conditioned, jointly-allocated, per-task-certified *communication decision*, which no prior CP
method makes.
