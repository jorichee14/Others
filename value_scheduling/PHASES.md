# Phases, with what each one is for

The plan is `PLAN.md`; this file is the execution order, with the reason
for each phase, what it produces, what it needs, and the exit test.
Written after reading Yazgan, Mueller, Zoellner, *Dense Coverage, Sparse
Refinement* (arXiv 2609.29456, `refs/`), which is the codec and value
predictor the plan builds on, and after fixing the hardware available:
two Jetson Orin NX mobile agents, three Raspberry Pi fixed agents, one
Asus RT-BE82U, one server.

## The facts from the DR paper that shape the phases

| Fact | Consequence |
|---|---|
| Message = coarse base layer (S_base = 4; 2,304 bits ≈ 0.29 KB on DAIR-V2X) + fine patches, each 16×16 cells = 256 SimVQ tokens × 6 bits + 48-bit header = **1,584 bits ≈ 198 B** | The scheduling unit is the fine patch, ~200 B, not an MTU. About 7 patches fit one 802.11 frame; the scheduler selects patches and packs them per agent per grant |
| Benefit score β per cell from a Pyramid Window Attention selector (1.27 M params, 28.84 GFLOPs); ROC-AUC 0.785 against counterfactual marginal detection gain; Top-N_fine is a nested prefix | The value report is the per-patch β list (a few bytes per patch); nesting means any granted prefix of an agent's ranking is valid |
| Built on OpenCOOD, PointPillars, CoBEVT / SwapFusion; trained end to end, 30–40 epochs on one RTX 4090 | Reproduction on the RTX 3080 (12 GB) is feasible at batch 2 but slow; the parent study's OpenCOOD install and OPV2V test split exist |
| Code "will be published" | Phase 0 may have to reimplement the codec and selector from the paper's details (Supp. A, D); plan for it |
| Their Fig. 7: AP falls at N = 5 from accumulated quantization artifacts across many 2 KB agents | The N sweep must separate codec artifacts from scheduling; use the oracle and a no-quantization diagnostic at each N |
| Their Supp. F latency model: T_E2E = T_compute + N_trips × 10 ms, compute 40 ms | Our timeline (encode k while k − 1 transmits) must absorb 40 ms compute on a 4090, more on a Jetson; the hardware check runs perception offline (below) |
| Their stated limitation: one-shot broadcast leaves cross-agent redundancy, "future work: redundancy-aware coordination" | This is the plan's opening; cite it as the gap in their own words |
| They model MAC cost as a fixed per-trip overhead, not airtime | Our contribution is exactly the airtime model they abstract away |

## Phase 0. Reproduce the codec and the value predictor (weeks 1–3)

**Why.** Everything downstream selects among DR's patches by DR's β.
If the reproduction is off, every later number is off. This phase also
tells us whether the selector's β is good enough to schedule on (AUC
0.785 is useful but not strong; the oracle gap in phase 2 will say how
much it costs).

**Do.** On the RTX 3080 with the parent study's OpenCOOD: CoBEVT
baseline on OPV2V; implement CoBEVT-DR (SimVQ K = 64, d = 64; coarse
S_base = 4; 16×16 fine cells; benefit selector with task-confidence
prior; Gumbel-sigmoid budgeted training; Top-N inference) from the
paper if FZI's code is not out; train on OPV2V; compute per-patch
counterfactual marginal gain ΔL (their Supp. G) on the test split, which
is also the oracle for phase 2.

**Produces.** A trained DR model; per frame and agent: coarse bytes,
ranked fine patches with β, and oracle ΔL per patch; a dump of these for
the OPV2V test split (this dump is what the scheduler and the hardware
check consume).

**Exit.** OPV2V AP@0.5 / AP@0.7 at 2 KB within ±0.02 of DR's Table 2
(0.94 / 0.85); selector ROC-AUC ≥ 0.75 against ΔL.

## Phase 1. The MAC abstraction and its ns-3 validation (weeks 2–5, overlaps phase 0)

**Why.** The claim is about airtime on a shared AP, so the harness must
charge airtime, not bytes. Running ns-3 inside every training or
scheduling step is too slow and too fragile; a Python abstraction
validated against ns-3 once per configuration gives the sweeps at
negligible cost and keeps simulation and hardware on one abstraction.

**Do.** Python model: frame period T_f = 100 ms; grant interval T_g;
per-agent PHY rate R_i from an MCS table (spread as a sweep parameter);
airtime of a packed frame of k patches = (k × 1,584 bits + MAC/PHY
overhead) / R_i plus fixed preamble; base-layer window first; EDCA mode
(DCF airtime sharing with collision loss from the standard analytical
model) for the baseline; grant mode (one sender per window, or parallel
grants) for ours; background bulk flows. ns-3 802.11ax with uplink OFDMA
(the stock `RrMultiUserScheduler`) and with EDCA, run on a grid of
N ∈ {2, 3, 5, 7} × background ∈ {0, 1 MB/s per agent} × MCS spread
∈ {none, near/far}; compare delivered bytes per interval, deadline-miss
rate and p99 delay. The replaced `MultiUserScheduler` for the trigger-
loop version is written here too, as the ideal target.

**Produces.** `mac/` package with EDCA, grant and trigger-loop modes;
the validation table.

**Exit.** Abstraction within 10 % of ns-3 on delivered bytes and miss
rate across the grid; the two-round protocol's extra channel-access
cost reproduced.

## Phase 2. The AP-side scheduler, oracle and ablations (weeks 6–9)

**Why.** This is the method. Everything before it is scaffolding.

**Do.** Value report format (agent id, patch ids in ranked order, β per
patch, cell coordinates; ≤ 1 frame). Scheduler: cost-ratio greedy over
all agents' reported patches under the airtime left after base layers,
with the redundancy discount by predicted visibility overlap (poses
known; a cell both agents see unoccluded is discounted on the second
selection; a cell only one sees is not), packing per agent per grant.
Oracle: the same greedy on true ΔL with true overlap. Ablations: no
discount; cell-identity discount instead of visibility; no MCS-aware
cost; sender-side Top-N with equal-airtime round-robin (DR as published
under our MAC); compressed-over-EDCA.

**Produces.** AP-versus-airtime curves on OPV2V for N = 3 and 5, all
policies, plus the oracle-gap decomposition (value-prediction error vs
redundancy error).

**Exit.** Pre-registered: ours ≥ +3 AP@0.7 over sender-side selection
at equal airtime with N ≥ 3, and lower miss rate; or the oracle gap
explains the shortfall and the write-up changes to a diagnosis.

## Phase 3. Stress sweeps and external baselines (weeks 10–13)

**Why.** The hypothesis says the gap widens with N and with background
load; the sweeps are the test, and the external baselines are what a
reviewer will ask for.

**Do.** N from 2 to 7 on OPV2V; background 0 / 0.5 / 1 MB/s per agent;
MCS spread; the two-round protocol model (request round + response
round, same codec, same MAC); AoI-guaranteed center scheduling and
TAMP-style region scheduling implemented on the same MAC with the same
patches; DAIR-V2X and V2X-Real at their fixed N for perception realism.
At each N, the no-quantization diagnostic to separate codec artifacts
from scheduling.

**Produces.** The sweep figures; the external-baseline table.

**Exit.** Hypothesis confirmed or refuted across the sweep with the
pre-registered thresholds; the N = 5 codec effect isolated and stated.

## Phase 4. Hardware check: perception offline, network live (weeks 14–18)

**Why.** The paper needs one result on a real AP with real contention,
and the agents available cannot run the perception stack (three of them
are Raspberry Pis). So the perception is computed once offline and the
network is live: each agent is a byte source that replays, with real
timing, the unit stream phase 0 produced for its assigned OPV2V agent,
and sends only what the edge scheduler grants. Fusion runs afterwards on
whatever arrived by each frame's deadline, which gives real AP under a
real channel.

**Do.** Grant emulation per `HARDWARE.md`: edge scheduler on the server
(Ethernet to the AP); agents send the value report in AC_VO at each
frame, transmit granted patches in their window, base layer in the
protected first window; EDCA baseline on the same AP with no grants;
round-robin grants for the sender-side baseline; the two-round model.
Agents: 2 Jetsons (mobile, may move along a route to vary MCS) and 3
Raspberry Pis (fixed, placed near and far from the AP for MCS spread);
N = 5 matches the OPV2V scenes with five agents. Background bulk flows
from the Jetsons. One monitor-mode sniffer; chrony on all nodes;
airtime fairness and band steering off on the RT-BE82U. Delivered
patches per frame are logged and fed back into the offline fusion.

**Produces.** AP-versus-airtime and miss-rate on hardware for all
policies; the sim-grant vs hardware-grant agreement table.

**Exit.** The policy ranking from the grant simulation holds on
hardware; delivered-bytes and miss-rate agreement within 20 %; the
trigger-loop vs grant gap reported from simulation as the cost of not
owning the AP.

## Phase 5. Write-up and release (weeks 19–22)

One paper: formulation (units by task gain per airtime with cross-agent
redundancy), the one-round scheduler with its guarantee and its grant
emulation, the validated harness, hardware results. Release `mac/`, the
scheduler, the unit dumps for OPV2V, and the replay agents.

## Dependencies and what runs in parallel

```
Phase 0 (codec, β, ΔL dump) ──┐
                              ├──▶ Phase 2 (scheduler) ──▶ Phase 3 (sweeps) ──▶ Phase 5
Phase 1 (MAC, ns-3 grid) ─────┘                                   │
                                                                  └──▶ Phase 4 (hardware; needs the phase 0 dump and the grant code from phase 1)
```

Phases 0 and 1 run in parallel from day 1 (GPU work vs network work).
Phase 4's replay agents and grant protocol are written during phase 1
so the hardware session needs no new code.

## What I need from you before phase 0 or 1 starts

1. Whether FZI has released the DR code (the paper says "will be
   published"); if not, phase 0 reimplements it, +1 week.
2. The intended MAC-level reference behind "PRISM": the PDF supplied
   (arXiv 2603.22754) is an LLM-reasoning paper with the same name. Drop
   it or supply the right one.
3. Confirmation that the RTX 3080 machine (`wicomsrobot`) and the
   OpenCOOD install from the parent study are available for phase 0.
4. ns-3 version on hand, if any; otherwise phase 1 installs ns-3.42+
   (802.11ax OFDMA with `MultiUserScheduler` is in the stock Wi-Fi
   module).
