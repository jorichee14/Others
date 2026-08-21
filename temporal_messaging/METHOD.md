# Method — LOOM: Latent Out-of-Order Measurement fusion for collaborative perception

*Working name. Supersedes the "Asynchronous Continuous-Time Collaborative Belief" note,
which the literature check in [`docs/LITERATURE.md`](docs/LITERATURE.md) partially
falsified. Read §1 before anything else — it is the record of what was killed and why.*

---

## 1. What the literature check killed

The proposal was: a persistent latent BEV belief, evolved between observations by learned
continuous dynamics conditioned on Δt, updated asynchronously at each message's own
timestamp, read out by querying at an arbitrary time, with delta messaging and
channel-in-the-loop training. The claim was *"one architecture subsumes four literatures."*

That architecture already exists. It is **StreamingFlow** (Shi et al., CVPR 2024,
arXiv:2302.09585): SpatialGRU-ODE over a BEV state, trigger-mode predict/update, arbitrary
time query, zero-shot horizon extension from 2 s to 8 s. It is applied to asynchronous
multi-sensor fusion *inside one vehicle*.

| Proposed component | Verdict |
|---|---|
| Persistent latent BEV belief | **Taken** — StreamingFlow; memory-bank variant in CoST (ICCV 2025) |
| Learned Δt-conditioned continuous dynamics | **Taken** — StreamingFlow, on GRU-ODE-Bayes |
| Update at the message's own timestamp | **Taken** — StreamingFlow trigger mode |
| Time-query readout | **Taken** — StreamingFlow demonstrates it explicitly |
| Delta messaging | **Substantially taken** — CoST transmits dynamic content only; CooperTrim (2026) selects by temporal continuity; V2X-DSC argues the same conditional-information principle on the spatial axis |
| Channel-in-the-loop training | **Not novel** — "distortion-in-the-loop"; SyncNet's latency curriculum; QPoint2Comm's masked packet loss |

Three further corrections to premises the earlier note relied on:

- **"Every CP system is stateless snapshot exchange" is false as of 2023.** SCOPE (ICCV
  2023) keeps ego temporal context; V2XPnP (ICCV 2025) does multi-frame spatio-temporal
  fusion; CoST keeps an explicit memory bank; V2X-INCOP retains historical cooperation
  information.
- **"CoBEVFlow discretises time" is false.** CoBEVFlow explicitly claims irregular,
  continuous-timestamp handling without discretisation. The distinction that survives is
  *retention vs. transformation*, not *continuous vs. discrete*.
- **The ODE-beats-flow argument has already been made** — by StreamingFlow, against
  CoBEVFlow. It can be cited; it cannot be claimed.

And one warning that reshapes the whole evaluation strategy:

> **StreamingFlow's own ablation: on nuScenes, conventional synchronised spatial-then-temporal
> fusion beat SpatialGRU-ODE (50.2 vs 47.8 IoU, 47.0 vs 46.1 PQ). The continuous-time
> formulation did not buy accuracy. It bought flexibility.**

So a clean-channel accuracy win is not the paper. The payoff has to be demonstrated under
degraded communication — which is exactly what the parent study's instrument measures and
nothing else in this literature does properly.

---

## 2. What survives, and what the paper is actually about

Everything hard about the multi-agent case is absent from the single-vehicle case. A
vehicle's own sensors never go permanently missing, never arrive out of order, carry no
per-source age to reason about, and cost nothing to transmit. Across a network, all four
are the normal condition.

**Thesis.** Collaborative perception has converged on *per-message compensation*: each
received message is repaired, warped, or reweighted toward the receiver's present, then
discarded. We instead maintain a **shared continuous-time scene belief** into which each
message is integrated **at its own timestamp**. Extending continuous-time asynchronous
fusion from a sensor rig to a communication network is not a port: the network adds
permanent loss, unbounded and heterogeneous per-agent age, out-of-order arrival, and a hard
bit budget. Those four are the contribution surface.

Three mechanisms carry the paper. Each one is a thing StreamingFlow, CoST, CoBEVFlow, and
TraF-Align structurally cannot have, because none of them is on a network.

### M1 — Learned out-of-sequence measurement (OOSM) update  ← the technical core

A message stamped *t−τ* arrives when the belief has already been integrated forward to *t*.
Two agents' messages arrive in an order unrelated to their timestamps. "Update at the
message's own timestamp" is not implementable by forward integration alone: the state has
moved past the measurement.

The three implementable options, in increasing cost and increasing correctness:

| Option | Mechanism | Cost |
|---|---|---|
| **Rewind–replay** | keep a ring buffer of past states, roll back to *t−τ*, apply, re-integrate to *t* | exact, but O(buffer) compute per late message; unbounded if τ is unbounded |
| **Learned retrodiction** | an operator `U_back(z(t), m, τ)` that folds a τ-old measurement directly into the current state, conditioned on how much the state has advanced since | O(1); the research object |
| **Forward-warp (what the field does today)** | advance the message to *t*, then treat it as a current measurement | what CoBEVFlow / TraF-Align do; throws away the information that the measurement is *old*, so its uncertainty is not down-weighted |

Classical estimation solved this in linear-Gaussian form — Bar-Shalom's OOSM retrodiction,
delayed-state filters, consensus Kalman. **There is no learned high-dimensional analogue,
and a robotics reviewer will ask about the classical one, so cite it and position against
it.** M1 is the load-bearing novelty: a learned retrodictive update operator over a latent
BEV state, trained on the arrival-order statistics an actual channel produces.

### M2 — Predictive-residual messaging (deltas against the *predicted* belief)

CoST sends what *changed*. That is a delta against a static-scene memory. The sharper
object is a delta against what the receiver **can already predict**.

Both agents run the same dynamics `F_θ`. The sender knows when its last message was
delivered (piggybacked acknowledgement) and therefore can compute `ẑ_rx(t)` — the
receiver's belief propagated forward under `F_θ` — and transmit only the residual
`m = Enc(z_tx(t) ⊖ ẑ_rx(t))`: the part of its observation the receiver's own dynamics
could not have anticipated.

Two consequences worth the section:

1. **Bit-rate becomes a function of age by construction.** The longer since the last
   successful delivery, the further the receiver's prediction has drifted, the larger the
   residual. This is the rate×age coupling the earlier note wanted to *measure* as a
   novelty — the honest move is not to claim the measurement is new, but to make the
   encoder *implement* the coupling. Rate then scales with **surprise**, not with scene
   size (dense transmission) and not with scene change (CoST).
2. **It degrades gracefully in the right direction.** A lost residual is not a lost
   observation; it is a residual that gets absorbed into the next one, because the next
   residual is computed against the receiver's belief as it *actually* is, not as the
   sender wishes it were. Contrast V2X-INCOP, which must *predict the missing message*.

This is predictive coding (DPCM) across agents in latent space. V2X-DSC makes the same
conditional-information argument on the spatial axis (agents' latents are correlated across
viewpoints); M2 is the temporal-and-cross-agent version. **Cite V2X-DSC as the principle,
claim the instantiation.**

Cost to be honest about: the sender must track a per-receiver predicted belief → O(N²)
state in a fully-connected group. Bounded in practice by the measured agent density (1.59
collaborators per ego frame in OPV2V test), but it must be stated, and the fallback
(residual against a *broadcast-common* belief, one state per sender) must be evaluated.

### M3 — Belief divergence as a first-class quantity

If the belief is *shared*, asymmetric loss makes agents' copies diverge silently, and a
divergent shared state is worse than no shared state. Nobody measures this. The June 2026
survey (arXiv:2606.13840) names **"verifiable shared-state maintenance"** as an open
priority under its Shared World Models framing.

Deliverables: a divergence metric `d(z_i(t), z_j(t))` tracked through the parent study's
831-cell impairment matrix; the drift curve (belief quality vs. time since last message per
agent, and vs. cumulative number of lossy integrations); and a resync mechanism triggered
by divergence exceeding a bound.

**This is also the experiment most likely to kill the method,** and it must be run early.
V2XPnP reports that early fusion wins partly because it avoids error accumulation from
lossy intermediate features transformed across the temporal dimension. A persistent belief
compounds exactly that. If drift dominates, retention is a liability and the paper is a
negative result — which the parent study has already demonstrated it is willing to publish.

---

## 3. Positioning sentence (use this, not the old framing)

> Collaborative perception has converged on per-message compensation: each received message
> is repaired, warped, or reweighted toward the receiver's present, then discarded. We
> instead maintain a shared continuous-time scene belief into which each message is
> integrated at its own timestamp. Continuous-time latent dynamics have been shown
> effective for asynchronous fusion *within* a single agent's sensor suite [StreamingFlow];
> we show that extending this to a *communication network* is not a straightforward port —
> network asynchrony additionally involves permanent message loss, unbounded and
> heterogeneous per-agent age, out-of-order arrival, and a hard bandwidth budget, none of
> which arise intra-vehicle. We address these with a learned out-of-sequence update
> operator over agent-tagged messages, residual encoding against the receiver's *predicted*
> belief, and end-to-end training under a sampled channel.

Language notes: use **"shared world model"** (arXiv:2606.13840) rather than coining a
competing term. Do **not** use "task-agnostic" — STAMP (ICLR 2025) owns it along the
agent-heterogeneity axis; say **"time-queryable scene state"** for the readout-time axis.

---

## 4. The bars this has to clear

Stated in advance so results cannot be graded on a curve.

| Bar | Number | Source | Why it binds |
|---|---|---|---|
| **Latency** | AP50 drop of only **4.87%** (V2V4Real) / **5.68%** (DAIR-V2X-Seq) at **400 ms** | TraF-Align, CVPR 2025 | Checkpoints are released and built on OpenCOOD. A persistent belief that cannot beat this is not a paradigm improvement however clean the formulation is. |
| **Bandwidth** | **1/281** and **1/264** of SOTA (INSTINCT, ICCV 2025); **0.416 Mb** (CoCMT) | instance-query methods | The residual argument must be made against *instance-query* baselines, not dense feature maps. "8× headroom vs. dense transmission" is a straw man. |
| **Loss** | **+14.06%** cooperative gain on OPV2V averaged over drop rates | V2X-INCOP, T-IV 2024 | Same dataset as this study. Direct comparison, no excuses. |
| **Dynamics model** | must beat **linear extrapolation** | CoDynTrust, ICRA 2025, reached SOTA with per-ROI velocity × delay | Mandatory ablation. If learned advection does not beat linear advection, the displacement finding does not imply what the earlier note said it implied. |

**A premise correction that follows from the first row.** The parent study found latency
crossing the ego-only floor at 100 ms for all seven methods. TraF-Align reports −4.87% AP50
at 400 ms. These are not contradictory — the parent study measured *uncompensated*
pretrained models, TraF-Align measures a *trained compensator* on different data — but
"latency is the binding unsolved constraint" is a materially weaker premise than it looked
before this check. Any intro that leans on it will be challenged. Lean on **loss + age +
out-of-order** instead, where the literature is thinner and M1/M2/M3 actually live.

---

## 5. De-risking path

Do not start with an ODE solver.

1. **Discrete Δt-conditioned recurrent belief first.** A spatial GRU whose update is
   conditioned on elapsed time. It captures the whole hypothesis — retention, age
   conditioning, asynchronous update — and trains in a fraction of the time. Only promote
   to SpatialGRU-ODE / Neural CDE if the discrete version works and the continuous version
   demonstrably buys something (arbitrary query times, irregular arrival, horizon
   extension).
2. **ODE-vs-CDE is a real choice, so justify it.** Neural CDEs (Kidger et al., NeurIPS
   2020) are arguably the better fit: messages from multiple agents *are* a control signal
   steering the trajectory, whereas a Neural ODE's trajectory is fixed by its initial
   condition and needs an RNN-style update bolted on. Also have an answer for "why not a
   state space model?" (CoMamba is in this space).
3. **Budget for solver cost early.** StreamingFlow: 0.1968 s/sample at variable step, ~0.5
   s at fine granularity, and **OOM on a 48 GB A6000** at 40 supervised frames while the
   GRU baseline fitted in 39 GB. This machine has a **12 GB RTX 3080**. Densely-supervised
   ODE training is not going to fit. Mitigations if continuous time becomes necessary: CfC
   (~20× speedup, Hasani et al. 2022), time-parallel Neural CDE (~15×, arXiv:2602.11738).
4. **Kinematic prior with a learned residual** in the dynamics function (per the 2026
   physics-guided Neural ODE work). Cheap, likely helps, clean ablation, and it is the
   principled version of the CoDynTrust linear-extrapolation comparison.

---

## 6. What the parent study contributes to this

Not results — an **instrument**, and it is the reason this is worth attempting here rather
than anywhere else.

- `commchannel` is **bitwise inert when disabled** (verified, 100/100 frames, three
  methods). Every measured difference is attributable to the impairment and nothing else.
  Nothing in the cited literature has this property; most simulate latency by frame
  substitution.
- The 831-cell matrix already spans latency × i.i.d. loss × bursty loss (Gilbert-Elliott) ×
  bandwidth quantization × staleness × pose error × ghosts × scene swap, on seven
  architectures, seed-deterministic and resumable.
- The **ego-only floor test** is the drift detector M3 needs. A belief that hallucinates
  drops below the floor, and the floor is already measured for every method.
- Detection **and** tracking harnesses exist, so the time-query readout can be evaluated on
  two consumers without new infrastructure.

---

## 7. Risks, in order of how likely they are to end this

1. **Drift dominates retention** (M3). V2XPnP's error-accumulation finding says this is
   live. Mitigated by measuring it in Phase 1, not Phase 4.
2. **Learned dynamics do not beat linear advection** (CoDynTrust). Mitigated by running the
   ablation before building anything on top of it.
3. **Training cost.** M1/M2 require retraining; the OPV2V train split is ~100 GB and this
   is a 12 GB card. The pretrained-only discipline that made the parent study cheap does
   not survive into this project. Plan the download and a small-model configuration first.
4. **Concurrent work.** Mid-2026 preprints are the least indexed and the highest scoop
   risk. Set arXiv alerts on cs.CV + cs.RO for: collaborative perception, cooperative
   perception, V2X, asynchronous, continuous-time.
5. **Unread decisive prior art.** Only StreamingFlow was read in full. **CoST**,
   **CooperTrim**, the **AoI paper (arXiv:2602.13439)**, and **V2X-DSC** all touch M1/M2
   directly and are still unread. **Out-of-sequence measurement filtering was not searched
   at all** and is structurally the same problem in linear-Gaussian form.
