# Assessment of the plan as written

The plan under review is `PLAN_ORIGINAL.md`. The assessment keeps its
structure: question, novelty, approach (A format, B value, C scheduler),
evaluation, phases. Each point says what holds, what is at risk, and the
change adopted in `PLAN.md`.

## 1. What holds

- **One decision point, one round.** Selection at the AP, with the
  value report riding on the buffer status report (BSR) that 802.11ax
  stations already send, is the sharpest idea in the plan. It removes
  the request round that receiver-driven protocols pay, and it puts
  the decision where the demand of every agent is visible.
- **Equal-size, prefix-decodable units.** Equal size makes airtime
  comparable across agents, and nesting makes every prefix usable. This
  is the right format and it has a guarantee behind it: with equal-size
  units, one greedy order is within 1 − 1/e of a clairvoyant sender for
  every airtime budget at once, and with unequal sizes no fixed order
  is (Nemhauser–Wolsey–Fisher 1978; Khuller–Moss–Naor 1999 for the
  failure with unequal sizes). The plan should state this.
- **Cost-ratio greedy under a knapsack.** Correct tool, known bound
  (≈ (1 − 1/e)/2 with the best-single-item fallback, better with partial
  enumeration), microseconds for tens of agents.
- **Baselines and ablations** are the right set: no collaboration,
  compressed-over-EDCA, sender-side selection with round-robin OFDMA,
  receiver-driven two-round, two ablations, and an oracle. The oracle is
  what turns a negative result into a diagnosis.
- **Out-of-scope list** is explicit and keeps the paper to one method.

## 2. What is at risk, and the change adopted

### 2.1 The AP is not programmable (hardware check, weeks 15–20)

The plan says: "commodity APs do not expose the trigger scheduler, so
emulate it at the edge server with application-level airtime grants."
That is correct and it is the whole hardware story, so it must be
designed in week 1, not week 15, and the ns-3 model must match the
emulation, or the simulation and the hardware test different systems.
The Asus AP in use (RT-BE82U, Wi-Fi 7) runs OFDMA and MU-MIMO in
firmware with no per-station scheduling API; Asuswrt and Asuswrt-Merlin
expose airtime-fairness and QoS toggles only. Details and the adopted
design are in `HARDWARE.md`. Short form: the AP's selection is
implemented as **application-layer grants** from an edge scheduler
(which agent sends which units in which slot), agents transmit only on
grant, and the same grant abstraction is what ns-3 simulates. The
hardware claim becomes "scheduled, value-selected uplink against EDCA
contention on a real AP", not "inside the trigger loop"; the trigger
loop remains the simulated target and the stated deployment path.

### 2.2 The value head predicts a context-free gain

B predicts each unit's marginal detection gain at the sender, without
knowing what other agents hold. The AP's redundancy discount repairs
overlap but not complementarity (two views of the same cell from
different sides are not redundant when one is occluded). Two changes:

- discount at the AP by **predicted visibility overlap** (both agents'
  poses are known; a cell both can see unoccluded is redundant, a cell
  only one sees is not), not by cell identity alone;
- report the **gap to the oracle** as a primary diagnostic, split into
  value-prediction error and redundancy error, so that a shortfall is
  attributed.

The cited predictor AUC (0.785) is confirmed in the DR paper (Supp. G,
against counterfactual marginal detection gain, Precision@10 % = 0.51);
it must still be reproduced on our harness before it is relied on.

### 2.3 Submodularity is a surrogate property

Detection gain over a set of feature units is not guaranteed submodular;
features can be complementary (supermodular) at cell boundaries. The
greedy guarantee holds for the surrogate objective (sum of predicted
gains with overlap discount), not for AP@0.7. State it that way, and
measure the surrogate-to-true gap through the oracle.

### 2.4 The 100 ms timeline is not accounted

Per frame: sensing → encode → value report (in the BSR) → trigger →
transmit → fusion, all inside 100 ms, with the trigger interval chosen
by the AP. The plan has no timeline. Encoding alone on a Jetson-class
device is tens of milliseconds for a BEV backbone; a BSR round trip is
milliseconds; the transmit window is what is left. Adopt a pipelined
design with an explicit budget (encode frame k while frame k − 1
transmits) and state the resulting one-frame staleness, or shrink the
backbone. The deadline-miss metric must be defined against that
timeline.

### 2.5 Unit size and headers

DR's fine patch is 1,584 bits ≈ 198 B including a 48-bit header (3 %).
That is the unit; it is far below one 802.11 frame, so the scheduler
packs about seven patches per frame per agent and charges airtime per
packed frame, including the fixed PHY preamble that dominates at this
size. The original plan's "equal-size units" holds; "one unit = one
frame" does not.

### 2.6 MCS-aware cost is circular under OFDMA

Airtime = size / rate, but under OFDMA the rate depends on the resource
unit assigned, which depends on the schedule. Use the per-agent
full-band rate as the cost in the greedy step and let ns-3 assign RUs
after selection; report the loss against an RU-aware selection as an
ablation. On hardware the circularity disappears (one grant, full
band).

### 2.7 Datasets limit the N sweep

OPV2V gives N up to 7 and is simulated perception; DAIR-V2X has N = 2
and V2X-Real N = 4. The network is ns-3 in every case, so the real
datasets test perception realism, not network realism. Say so. The N
sweep is on OPV2V only.

### 2.8 Receiver-driven baselines are expensive to reproduce

A faithful WhisperNet or CoSDH on the same MAC is weeks of work.
Adopt a protocol-level model (request round, response round, same
codec, same MAC) and label it as such; it captures the cost the plan
argues against (an extra channel-access round at KB payloads).

### 2.9 Novelty threats to read in full

The AP-side decision itself is not new: AoI-guaranteed scheduling for CP
(Wang et al. 2023) allocates resource units at a center with per-region
age constraints; TAMP (Zhu et al. 2026) schedules regions by a
timeliness penalty at an RSU; SchedCP (2025) schedules users by CSI and
semantics; the "PRISM" the plan cites could not be verified: the PDF
supplied under that name (arXiv 2603.22754) is an LLM-reasoning paper,
so the MAC-level reference is either misnamed or missing. The precise claim that survives them: **selection of
sub-message units by task value with cross-agent redundancy, inside the
uplink scheduling loop, in one round.** Each of the four must be a
baseline or an explicit contrast.

### 2.10 Pre-register the hypothesis thresholds

"The gap widens as N grows and as background traffic increases" needs
numbers before the runs: for example, at N = 5 and 1 MB/s background,
ours ≥ +3 AP@0.7 over sender-side selection at equal airtime and
≤ half its deadline-miss rate; the gap at N = 2 may be within noise.

### 2.11 The harness is the schedule risk

OpenCOOD coupled online to ns-3 802.11ax with a replaced multi-user
scheduler, in four weeks, is optimistic: the scheduler depends on
values that depend on the frame, so the coupling must be online
(ns3-ai or a socket), and ns-3's OFDMA model has its own learning
curve. Adopt a two-level harness: a Python MAC abstraction (trigger
interval, per-agent rate, airtime budget, grant outcome) used for all
sweeps, validated offline against ns-3 on a grid of (N, load, MCS)
configurations by comparing delivered-bytes-per-interval
distributions. ns-3 then runs once per configuration, not once per
frame.

## 3. What is missing

- **Theory section.** The plan has a guarantee for the scheduler but
  none for the format. Two results transfer directly: the prefix-order
  guarantee under a random airtime budget (equal-size units), and the
  identity that the value of a prefix-decodable message over an atomic
  one equals the partial-delivery mass, which is zero under a fixed
  budget. The second explains why the compression literature never
  needed layered messages and why this plan does.
- **Downlink.** The plan is uplink only. The fused result or the
  selected units must reach the agents that need them; on hardware
  the downlink shares the same airtime. Account for it as background
  or schedule it; do not ignore it.
- **Energy and compute at the agent.** Encoding every unit of every
  frame to let the AP choose costs the sender compute it may not spend;
  a two-stage encode (base layer always, refinements on grant) is the
  natural fix and matches the grant design.
