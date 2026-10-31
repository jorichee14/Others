# The easiest path: an end-to-end result in three steps, no DR code, no ns-3, no hardware

The goal of the MVP is one figure: AP@0.7 versus airtime per frame on
OPV2V for N = 3 and 5, with EDCA-contended sender-side selection, round-
robin grants, value-driven AP grants, and the oracle. Everything it needs
exists on `wicomsrobot`: OpenCOOD, the OPV2V test split, the CoBEVT
checkpoint from the parent study. Nothing is trained.

The stand-in codec replaces DR until DR is reproduced; it keeps DR's
structure (coarse base + equal-size fine patches ranked by a value) with
parts that need no training:

| DR component | MVP stand-in | Why it is enough |
|---|---|---|
| SimVQ fine patch (256 tokens × 6 bits) | the raw 16×16-cell patch of the intermediate feature map, 4-bit quantized: 256 cells × C channels × 0.5 B | the parent study measured bandwidth free down to 4 bits; patches are larger (8–32 KB) but equal-size, and the airtime model scales |
| coarse base layer (S_base = 4 VQ) | 4× average-pooled feature map, 4-bit | DR's own no-VQ diagnostic ("Raw C+F") shows the coverage-refinement gain without VQ |
| learned benefit β | DR's task-confidence prior ψ: per-cell normalized channel-mean magnitude (their Eq. 2–3), pooled per patch | DR Table 1A: prior-only ranking reaches 0.713 / 0.589 against 0.736 / 0.604 learned at 2 KB; good enough to test scheduling |
| counterfactual ΔL (oracle) | zero a patch (coarse in its place), run fusion + head, record the loss change | one forward pass per patch per frame with the frozen checkpoint; offline, cached |
| dense reconstruction | coarse everywhere, fine where delivered, same fusion module | DR's design, untouched |

## Step 1 (days 1–3): the unit dump

Script `mvp/dump_units.py`, run from `~/cpfa/OpenCOOD`:

- for each OPV2V test frame and each CAV: the pooled coarse layer, the
  fine patches, ψ per patch, the CAV pose, the patch's BEV cell range;
- the oracle ΔL per patch for a subset of frames (say 200 frames × all
  CAVs), cached as a table;
- all written as one `.npz` per frame.

Done when the dump reproduces the clean CoBEVT AP when every patch is
"delivered", and ψ against ΔL gives a ROC-AUC; record it (DR's learned
selector is 0.785; the prior will be lower, and that number is the
first result).

## Step 2 (days 3–7): the MAC abstraction and the scheduler

Package `mac/` (pure Python, numpy):

- `Agent`: PHY rate from an MCS index (table for 802.11ax, 20 MHz, 1
  stream), queue of units, background flow rate;
- `airtime(bytes, rate)`: preamble + MAC header + bytes / rate, per
  packed frame (pack whole patches, as many as fit one A-MPDU up to the
  cap);
- `EDCA`: N backlogged stations share airtime by the DCF model (Heusse
  et al. 2003 equal-airtime-per-transmission-opportunity with collision
  loss from Bianchi's p for N stations), background included; returns,
  per frame, which units of each agent arrived before the deadline;
- `Grant`: a per-frame airtime budget after the protected base-layer
  window; the scheduler returns a list of (agent, unit) in order; the MAC
  serializes them and charges airtime; units past the budget miss.

Package `sched/`:

- `sender_topk`: each agent's ψ-ranked prefix up to an equal per-agent
  budget (DR under round-robin grants);
- `edca_topk`: the same prefix, but delivery decided by `EDCA`;
- `value_greedy`: all agents' patches in one pool, cost-ratio greedy on
  ψ × (1 − overlap discount) / airtime, with the discount computed from
  the agents' BEV ranges and poses (a cell already selected from another
  agent whose range covers it unoccluded discounts the second selection
  by a factor, swept 0 / 0.5 / 1);
- `oracle_greedy`: the same with ΔL in place of ψ.

Done when, on synthetic values, `value_greedy` dominates `sender_topk`
at equal airtime and `oracle_greedy` dominates both; unit tests for the
airtime arithmetic.

## Step 3 (days 7–12): the figure

Script `mvp/run.py`: for N ∈ {3, 5} (OPV2V scenes with that many CAVs),
sweep the per-frame airtime budget over 6 points, background ∈ {0,
1 MB/s per agent}, MCS spread ∈ {all MCS 7, half at MCS 3}; for each
policy, deliver units through the MAC, rebuild the dense feature map
(coarse + delivered fine), run the frozen fusion and head, score AP@0.5
and AP@0.7 and the deadline-miss rate. Plot AP versus airtime, one
panel per (N, background), four policies plus oracle.

Done when the figure exists with error bars over scenes. The reading of
it, whichever way it comes out, is the first result of the project:

- value-driven > sender-side at equal airtime: the thesis holds at the
  prior's AUC; DR's learned β can only widen it;
- no gap: the oracle panel says whether the value signal or the
  redundancy model is at fault, before anything is trained.

## What is deliberately left out of the MVP

ns-3 validation of the MAC model (phase 1 proper), DR's codec and
selector (phase 0 proper), the two-round and AoI/TAMP/PRISM baselines
(phase 3), hardware (phase 4). None of them changes the pipeline; they
replace a module or add a policy.

## What the MVP needs

`wicomsrobot` with the parent study's environment (`opencood` conda env,
OPV2V test split, the CoBEVT checkpoint; see
`../collab_perception_failure_analysis/env/`). About 20 GPU-hours for
the oracle table at 200 frames; the rest is CPU.
