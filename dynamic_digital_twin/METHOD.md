# Twin-Coherence Gating (TCG) — the method

**Status: settled direction (2026-08-16), supersedes the earlier PulseSync draft (kept in
git history). This is the paper's contribution; `IMPLEMENTATION.md`'s milestones build
the instrument for it. Zero learning by design — no component of this method is trained.**

---

## 1. One-sentence statement

A synchronization protocol for dynamic digital twins in which **not transmitting is the
default informative action**: silence is evidence that the twin's prediction is right, a
zero-bit physical-layer coherence check — the twin predicting its own radio environment —
makes silence unambiguous on lossy links and detects unmodeled scene changes at PHY
timescale, and an optimization layer with derived thresholds decides per entity when an
update is worth sending, at what payload level, over which link.

**Targets (both structural, not bolted on):**
- *Communication overhead* → silence is the steady state; monitoring rides on existing
  traffic (CSI from data/ACK frames costs zero transmitted bits).
- *Latency* → change detection at channel timescale (per-packet, ms), not
  perception-pipeline timescale (sense→infer→encode→send, 100+ ms).

## 2. The three layers

### Layer 1 — the twin (classical estimation)
Per-entity Kalman filters (position, velocity, covariance, timestamp) + Hungarian
association; constant-velocity extrapolation between updates; covariance propagation
gives the error-growth law analytically. (The parent repo's Phase-5 tracker is the
starting implementation.)

### Layer 2 — the gate (the mechanism claim)
Two coupled negative-evidence channels:

**(a) Silence as evidence.** Sender and twin share the deterministic trigger rule, so a
non-transmission is itself a message: "my observation of entity *i* is within the agreed
bound of the shared prediction." The twin *tightens* its uncertainty on silence instead
of growing it — event-based estimation with negative information (Sijs & Lazar 2012;
Shi et al. 2016), which exists for **self-measured scalar sources on reliable links**,
extended here to **noisily-perceived third-party entities (association, births, deaths)
on lossy wireless**. Each italicized condition is one the existing theory assumes away.

**(b) Channel coherence as the disambiguator and surprise detector.** The known Achilles'
heel of negative-information estimation on real networks: silence is ambiguous — "no
event" (informative) vs "packet lost" (not). The literature fixes this with
error-detecting codes / heartbeats, i.e. spends bits. TCG fixes it with physics: the twin
predicts the expected channel statistics per link (Doppler band energy, CSI variance)
from everything it believes is moving — its own AMRs via odometry, tracked entities via
their state — and runs a sequential test (CUSUM/GLR) on the residual:
- channel coherent + silence ⇒ **no event** — tighten the bound;
- channel anomalous on a link ⇒ **outage or unmodeled motion** — widen the bound and/or
  trigger a targeted perception update from the nodes whose links show the anomaly
  (intersecting link geometry coarsely localizes the change);
- crucially, this also catches what no perception-side trigger can: **entities the twin
  does not know exist yet** (births), and it **explains away known robot motion** — the
  failure mode of raw-CSI gating (Wi-Filter-style), which fires continuously in a room
  where AMRs always move.

### Layer 3 — the scheduler (the formal contribution — optimization)
> **Problem:** minimize expected twin error subject to a bit-rate budget across N
> entities and 2 links (Wi-Fi, 5G) with measured delay distributions.
> **Decision variables:** which entity updates, when, over which link, at what payload
> level (state delta vs full anchor).
> **Structure:** per-entity error growth between updates is analytic (covariance
> propagation — faster entities cost more per second of silence); links differ in
> delay/price; updates have a **measured negative-value region** (an update older than
> the value-zero age hurts more than silence); the objective carries an **age-variance
> term** for the stateful consumer.

The trigger threshold and link assignment are *derived* from this problem (index /
deadline policies; event-triggered-estimation threshold-optimality results generalized
to the multi-entity, dual-RAT, discrete-payload case), not hand-tuned. This is the
paper's theory section and it requires no learning.

**Empirical bases of the problem structure** (parent study, all [V]; constants must be
re-measured indoors — Phase 1 gate):
1. *Negative update value*: all 7 architectures below the ego floor at 100 ms; drop-90%
   beats deliver-200ms-late. Value-zero age scales as tolerance/velocity (indoors ≈ 1 s).
2. *Analytic growth / age sufficiency*: latency ≡ staleness within 1.5 NPD points —
   age is a sufficient statistic for the damage; displacement = velocity × age.
3. *Step payoff*: quantization free to 4-bit, cliff at 2-bit; and message *types* are
   inherently discrete (a delta cannot re-establish identity/geometry).
4. *Age-variance term*: constant delay IDSW ≈ clean (223/230/248) vs sawtooth staleness
   4–6× (629/1031/917) — a Kalman-type consumer absorbs consistent bias and amplifies
   oscillation. Applies to the twin **by construction**.

### ROS 2 systems layer (implementation contribution)
Implement the gate as a twin-aware QoS layer: entity validity horizon → DDS deadline
QoS; staleness bound → lifespan QoS; per-topic RAT binding via interface-bound DDS
config. "Dynamic-scene-aware QoS for ROS 2 over dual-RAT" targets the robotics systems
audience (ICRA/IROS/RA-L; networking: INFOCOM/IPSN).

## 3. Falsifiable claims (pre-register before closed-loop runs)

1. **Order-of-magnitude steady-state traffic reduction** at equal twin error vs periodic
   sync in quasi-static periods; graceful convergence to periodic under continuous change.
2. **Change-to-update latency below any polling scheme** at equal bandwidth (PHY-speed
   trigger).
3. **Raw-CSI gating collapses under AMR motion** (near-continuous false triggers) while
   TCG's false-trigger rate stays bounded — the head-to-head that proves the
   explain-away mechanism.
4. **Silence-as-evidence with channel disambiguation avoids the divergence** that naive
   negative-information updating suffers under real packet loss (ablation: negative
   updates on/off × disambiguation on/off).

Baselines: periodic sync (several rates), AoI-based scheduling, classical
model-triggered sync (no channel check), send-everything ceiling. Metric:
twin-error-per-bit + the claim-specific measures above.

## 4. Novelty accounting (verified 2026-08-16; house convention [V]/[S]/[I])

| # | claim | verdict / nearest prior art |
|---|---|---|
| 1 | Silence-as-evidence for perceived entities on lossy links, disambiguated by twin-predicted channel coherence | **Core claim — no combining work found.** Foundations: Sijs & Lazar 2012, Shi et al. 2016 (self-measured scalar, reliable links) [S→read]. Foil: error-detecting-code event scheduling (spends bits on the ambiguity TCG resolves with physics) [S]. |
| 2 | Twin predicts the channel to validate itself / gate communication | ISAC×DT literature runs the arrow the other way (channel feeds twin) [S]; DT residual-anomaly work uses application-state residuals for fault alarms, not comms gating [S]; **2607.09070 [V, primary text read]** twins the *network* for parameter tuning, periodic fixed-π sync, and leaves "π under mobility- and channel-coherence-aware criteria" explicitly to future work — the gap certificate. |
| 3 | Explain-away-known-motion CSI triggering | Wi-Filter / RF-assisted camera wake-up = raw motion detection, no world model, fails under constant robot motion [S]; Mobi2Sense/CIRSense = sensing techniques, not sync protocols [S]. |
| 4 | Derived multi-entity dual-RAT discrete-payload trigger/scheduling | Event-triggered estimation threshold-optimality = single sensor/link [S]; multi-source AoI scheduling exists but with age objectives, identical sources, no negative-value region, no payload classes [S]. Claim as *formalization + generalization*, never as "we discovered event triggering". |
| — | Dead claims — never assert | "Updates can hurt" as a discovery (SyncNet, AoI theory, When2com); threshold triggering per se (Åström, send-on-delta); silence-carries-information per se (Sijs); CSI-gates-camera per se (Wi-Filter). |

**Mandatory reads before writing** (novelty lives or dies here): Sijs & Lazar 2012;
Shi et al. 2016 hybrid estimators; frame-dropping MOT (arXiv 2308.00330);
error-detecting-code event scheduling; Wi-Filter; Mobi2Sense. AgentComm-Bench and
2607.09070 already read in primary text [V].

## 5. Gates and risks

1. **D-CSI is now load-bearing** (was opportunistic): per-packet CSI from the Wi-Fi NICs
   (chipset-dependent) and/or 5G UE reference-signal reports. Check FIRST, one
   afternoon. **Fallback: radar** — predict expected Doppler returns from twin state,
   trigger on unexplained returns; same loop, weaker elegance.
2. **Channel-prediction fidelity**: exact CSI prediction is hopeless; the bet is that
   coarse features (Doppler band energy, variance envelope) are predictable from twin
   kinematics. Phase-0 pilot: record CSI while an AMR drives a known path in an empty
   room — if odometry does not explain the channel variation, the mechanism dies cheaply.
3. **Indoor value-zero age** (~1 s at 1.4 m/s vs 100 ms vehicular): re-measured in
   Phase 1 before any scheduler constant is fixed; the pre-registered NO-GO applies.
4. **Theory depth scales with progress**: a clean formulation + provably-good simple
   policy (index/EDF) is sufficient; deepen only if it is going well.

## 6. Relation to the tracker

`IMPLEMENTATION.md`: M1 adds the D-CSI check and the empty-room CSI pilot (risk 2);
M2 builds Layer 1; M3's campaign supplies Layer 3's delay distributions; M4 re-measures
the value-zero age and growth law (gate); M5 = TCG closed loop vs §3 baselines.
