# Plan (amended)

`PLAN_ORIGINAL.md` as uploaded, with the changes from `ASSESSMENT.md`
folded in. Changes are marked **[amended]**.

## Research question

When N agents share one Wi-Fi AP, which pieces of which agents'
perception messages should get airtime in each frame, so that the
team's perception is maximized under a deadline?

**Hypothesis, with thresholds [amended].** At equal airtime, an
AP-side value scheduler over layered messages gives higher detection
accuracy and fewer deadline misses than (a) sender-side compression with
standard Wi-Fi access and (b) receiver-driven multi-round protocols.
Pre-registered: at N = 5 with 1 MB/s background per agent, ours ≥ +3
AP@0.7 over (a) at equal airtime and ≤ half of (a)'s deadline-miss rate;
at N = 2 the gap may be within noise and that is reported.

## Why it is new

| Closest line | What it does | What it lacks |
|---|---|---|
| Communication-efficient CP (Where2comm, Dense Coverage/Sparse Refinement, V2X-DSC) | shrinks each message to KB scale | a private per-link budget; no contention or airtime; cross-sender redundancy remains |
| Receiver-driven CP (WhisperNet, CoSDH) | plans who sends what, globally | an extra request round that pays channel access again |
| Semantic / AoI-aware access (PRISM, SchedCP) and center-side CP scheduling (AoI-guaranteed scheduling, TAMP) **[amended]** | schedules whole agents or regions by redundancy, freshness or age | no sub-message units by task value, not in the uplink loop, generic content |

**The claim [amended]:** selection of sub-message units by task value
with cross-agent redundancy, inside the uplink scheduling loop, in one
round. The AP-side decision alone is not the novelty; the four works in
the third row are baselines or explicit contrasts.

## Approach

One loop per frame (~100 ms), one decision point: the scheduler's
selection. Agents never wait for a request; the value report rides on
the buffer status report (trigger loop) or on a one-frame AC_VO
datagram (grant emulation, `HARDWARE.md`).

**A. Layered message format.** Coarse dense base layer plus refinement
units, nested so any prefix is decodable. **[amended after reading the
DR paper]** The unit is DR's fine patch: 16×16 cells, 256 SimVQ tokens
× 6 bits + 48-bit header = 1,584 bits ≈ 198 B. About seven patches pack
into one 802.11 frame; the scheduler selects patches and packs them per
agent per grant, and airtime is charged per packed frame. Patch metadata
(cell coordinates, β) travels in the value report. Guarantee [amended]: with equal-size units the greedy order
is within 1 − 1/e of a clairvoyant order for every airtime budget at
once; the gain of a prefix-decodable message over an atomic one equals
the probability mass of partial delivery, which is zero under a fixed
budget and positive under contention.

**B. Value estimate.** The sender predicts each unit's marginal
detection gain with a small head trained on counterfactual drops.
Redundancy is handled at the scheduler, which knows every agent's pose
and the cells already selected this frame; the discount uses
**predicted visibility overlap [amended]** (a cell both agents see
unoccluded is redundant; one only one agent sees is not), not cell
identity alone. The predictor's reported AUC (0.785) is to be
reproduced before use.

**C. Scheduler.** Per trigger interval, choose units to maximize total
discounted predicted gain subject to the airtime left after base
layers; a unit's airtime is its size over the agent's current full-band
rate **[amended: RU assignment after selection, RU-aware selection as an
ablation]**. Monotone submodular surrogate under a knapsack; cost-ratio
greedy with best-single-item fallback. Base layers go first in a
protected window (fixed trigger / restricted TWT in simulation; the
first grant window on hardware). **Timeline [amended]:** encode frame k
during the transmit window of frame k − 1; the value report is due at
the trigger instant; the deadline-miss metric is defined against this
pipeline and the one-frame staleness it introduces is stated. **Encode
on grant [amended]:** base layer always; refinement units encoded only
when granted, to spare agent compute.

**Downlink [amended].** The fused result or selected units to the agents
that need them share the airtime; accounted as scheduled downlink
traffic in the budget, not ignored.

## Evaluation

**Setup.** Perception from an OpenCOOD-style pipeline on OPV2V (N up to
7) and on DAIR-V2X or V2X-Real (perception realism only; N fixed by the
dataset). Network: a **two-level harness [amended]**, a Python MAC
abstraction (trigger interval, per-agent rate, airtime budget, grant
outcome) for all sweeps, validated against ns-3 802.11ax with uplink
OFDMA on a grid of (N, load, MCS) configurations by comparing the
distribution of delivered bytes per interval; ns-3 runs once per
configuration with the replaced `MultiUserScheduler` for the trigger-
loop system and with the default scheduler plus application grants for
the grant system. Whatever has not arrived by the deadline is missing
at fusion.

**Sweeps.** N from 2 to 7; airtime budget per frame; background bulk
traffic (1 MB/s per agent); spread of agents' MCS.

**Baselines.** No collaboration; compressed features over EDCA
(CodeFilling-style); sender-side selection with equal-airtime round-
robin (Where2comm / Dense Coverage-style); receiver-driven two-round at
protocol level **[amended: protocol model, labelled as such]**; AoI-
guaranteed center scheduling and TAMP-style region scheduling
**[amended]**; ours without the redundancy discount; ours without
MCS-aware cost; oracle selection with true marginal gains.

**Metrics.** AP@0.5 / AP@0.7 versus airtime per frame (headline);
deadline-miss rate against the pipelined timeline; p99 message latency;
background throughput retained; scheduler run time per interval;
**gap to the oracle split into value-prediction error and redundancy
error [amended]**.

## Phases

1. **Harness (weeks 1–5) [amended].** Python MAC abstraction; ns-3
   802.11ax OFDMA validation grid; layered codec; EDCA and round-robin
   baselines; the grant emulation designed and its overhead measured.
   Done when baseline AP-versus-airtime curves exist and the abstraction
   matches ns-3 on the grid.
2. **Core method (weeks 6–11).** Value head (reproduce the predictor
   AUC); greedy scheduler with visibility-overlap discount; ablations;
   oracle. Done when ours beats sender-side selection at equal airtime on
   OPV2V with N ≥ 3 by the pre-registered margin, or the oracle gap
   explains why not.
3. **Stress tests (weeks 12–15).** N sweep, background load, MCS
   spread, the two-round protocol model, the AoI and TAMP baselines.
   Done when the hypothesis is confirmed or refuted across the sweep.
4. **Hardware check (weeks 16–20) [amended: designed in phase 1].** Two
   to four agents, one Asus RT-BE82U, edge scheduler with grants, a
   sniffer. Done when the method ranking from the grant simulation
   holds on hardware, and the sim-grant vs hardware-grant agreement is
   reported.
5. **Write-up (weeks 21–24).** One paper; release the harness.

## Contributions and scope

1. Collaborative perception on shared Wi-Fi as per-frame selection of
   message units by task gain per airtime with cross-agent redundancy,
   with the format guarantee and the partial-delivery identity.
2. A one-round scheduler that fits the 802.11ax trigger loop and
   carries an approximation guarantee, and its grant-based emulation
   for APs that expose no scheduler.
3. A validated two-level network-in-the-loop harness and results on a
   real AP, released.

Out of scope: radio-twin link forecasting, roaming, generative decoding,
over-the-air fusion, mesh topologies, SLAM traffic except as background
load. Link forecasting is the second paper; it plugs into the same
scheduler as a predicted rate.
