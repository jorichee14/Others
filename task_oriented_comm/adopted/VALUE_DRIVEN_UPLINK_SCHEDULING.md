# Focused Plan: Value-Driven Uplink Scheduling for Collaborative Perception on Shared Wi-Fi

(uploaded 30 Oct 2026, text extracted from the .docx; the adopted plan)

Focused Plan: Value-Driven Uplink Scheduling for Collaborative Perception on Shared Wi-Fi
Oct 31, 2026 · @J
Research question
When N agents share one Wi-Fi AP, which pieces of which agents' perception messages should get airtime in each frame, so that the team's perception is maximized under a deadline?
Thesis. Today the codec decides what to send (per link, assuming a private pipe) and the AP decides who transmits (by queue size, blind to content). Moving the decision to the AP, which sees every agent's demand and already schedules uplink OFDMA, and letting it select by marginal task gain per unit airtime with cross-agent redundancy removed, beats both.
Hypothesis. At equal airtime, an AP-side value scheduler over layered messages gives higher detection accuracy and fewer deadline misses than (a) sender-side compression with standard Wi-Fi access and (b) receiver-driven multi-round protocols. The gap widens as N grows and as background traffic increases.
Why it is new
Closest line of work
What it does
What it lacks
Communication-efficient CP (Where2comm, Dense Coverage, Sparse Refinement, V2X-DSC)
Shrinks each message to KB scale
Assumes a private per-link budget; no contention, no airtime; redundancy across senders remains
Receiver-driven CP (WhisperNet, CoSDH)
Plans who sends what, globally
Extra request round; at KB payloads each round pays the channel-access cost again
Semantic / AoI-aware access (PRISM, SchedCP)
Schedules by redundancy or freshness
Whole-agent decisions on generic content; not Wi-Fi uplink OFDMA, not layered perception messages
The gap all three leave is the same: no design lets the Wi-Fi scheduler choose sub-message units by task value and redundancy inside the control loop the AP already runs.
Approach
One loop per perception frame (~100 ms), with one decision point: the AP's selection.

per-frame scheduling loop · 5 steps, one decision at the AP
Agents never wait for a request; the value report rides on the buffer status report Wi-Fi 6 already uses, so the plan costs no extra contention round.
A. Layered message format. Each agent sends a coarse dense base layer plus refinement units of equal size, as in Dense Coverage, Sparse Refinement. Equal-size units make costs comparable across agents, and the nested ranking means any prefix is decodable. Each unit carries its BEV cell, layer and deadline.
B. Value estimate. The sender predicts each unit's marginal detection gain with a small learned head, trained on counterfactual drops; the same paper reports such a predictor reaching ROC-AUC 0.785 for high-value patches. Redundancy is handled at the AP, not the sender: the AP knows every agent's pose and which cells it has already selected this frame, so a second unit covering the same cell is discounted.
C. Scheduler. Per trigger interval, choose units to maximize total discounted gain subject to the airtime left after base layers, where each unit's airtime is its size divided by that agent's current MCS rate. The objective is monotone submodular with a knapsack budget, so a cost-ratio greedy algorithm has a known approximation guarantee and runs in microseconds for tens of agents. Base layers always go first in a protected periodic slot (a fixed trigger on Wi-Fi 6, restricted TWT on Wi-Fi 7).
Evaluation
Setup. Perception from an OpenCOOD-style pipeline on OPV2V (several connected agents per scene, which allows scaling N) and on a real-world set such as DAIR-V2X or V2X-Real. Network from ns-3's 802.11ax model with uplink OFDMA, replacing its multi-user scheduler with ours. Each frame's messages pass through the simulated MAC; whatever has not arrived by the deadline is missing at fusion.
Sweeps. Number of agents (2 to 7), per-frame airtime budget, background bulk traffic (e.g. map or keyframe uploads at ~1 MB/s per agent), and spread of agents' MCS (near vs far from the AP).
Baselines.
No collaboration (lower bound)
Compressed features over default EDCA contention (CodeFilling-style)
Sender-side selection with equal-airtime round-robin OFDMA (Where2comm / Dense Coverage-style)
Receiver-driven two-round protocol on the same MAC (WhisperNet / CoSDH-style)
Ours without the redundancy discount, and ours without MCS-aware cost (ablations)
Oracle selection using true marginal gains (upper bound)
Metrics.
Detection AP@0.5 / AP@0.7 versus airtime per frame (the headline curve)
Deadline-miss rate: share of frames whose fusion input is incomplete at 100 ms
99th-percentile perception message latency
Background throughput retained
Scheduler run time per trigger interval
Phases
Durations are a suggested pace; the exit criteria are what matter.
Harness (weeks 1–4). Couple the perception pipeline to ns-3 802.11ax; implement the layered codec; run the EDCA and round-robin baselines. Done when baseline AP-versus-airtime curves exist.
Core method (weeks 5–10). Train the value head; implement the AP greedy scheduler, ablations and oracle. Done when ours beats sender-side selection at equal airtime on OPV2V with N ≥ 3, or the gap to the oracle explains why not.
Stress tests (weeks 11–14). Scale N, add background load and MCS spread, compare against the two-round protocol. Done when the hypothesis is confirmed or refuted across the sweep.
Hardware check (weeks 15–20). Two to four robots, optionally one fixed sensor, one AP. Commodity APs do not expose the trigger scheduler, so emulate it at the edge server with application-level airtime grants. Done when the method ranking from simulation holds on hardware.
Write-up (weeks 21–24). One paper on the method; release the harness.
Contributions and scope
Contributions.
A formulation of collaborative perception on shared Wi-Fi as per-frame selection of message units by task gain per airtime, with cross-agent redundancy.
A one-round AP scheduler that fits the 802.11ax trigger loop and carries an approximation guarantee.
A network-in-the-loop evaluation harness and results on real AP behavior, released openly.
Deliberately out of scope for this plan: radio-twin link forecasting, roaming, generative decoding, over-the-air fusion, mesh topologies, and SLAM traffic except as background load. Link forecasting is the natural second paper, because it plugs into the same scheduler as a predicted MCS.
Key references
Dense Coverage, Sparse Refinement: Byte-Constrained Cooperative Perception (2026) — layered message format and value predictor
WhisperNet (CVPR 2026) and CoSDH (CVPR 2025) — receiver-driven baselines
Where2comm (NeurIPS 2022) and V2X-DSC (2026) — sender-side and conditional coding
PRISM: semantic-aware multiple access (2026) — closest MAC-level work
SchedCP (2025) — semantic and channel-aware user scheduling for CP
Task-oriented communication for multi-device cooperative edge inference (TWC 2022) — distributed information bottleneck foundation
