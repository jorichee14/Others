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
