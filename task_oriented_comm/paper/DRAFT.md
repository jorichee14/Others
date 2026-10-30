# Radio-Gated Perception: The Team's Links Decide When the Shared Twin Is Stale

Front matter for the paper: gap, motivation, introduction, related
work, system model, problem formulation, and the novelty statement.
Numbers marked [R] are from the SCooP recordings
(`scoop_testbed_paper/RESULTS_STATUS.md`). Claims about other papers
marked [S] come from abstracts and must be checked before submission.
Bib keys are in `../references.bib`.

---

## Abstract

A robot team that shares a twin of its building keeps it current by
sending keyframes on a timer, so most of what it sends describes what
the twin already holds, and all of it competes on one wireless link
with the control and pose traffic that cannot wait. We show that the
link can say when the twin is stale. Every data packet the team
exchanges yields a channel-state measurement on the sensors' clock; a
radio map recorded during the mapping pass predicts that measurement at
the places the nodes stand; the residual rises on the links whose
Fresnel zone crosses something the map does not contain. We treat the
residual as a change-detection statistic, place it in the building by
link geometry, and let it gate perception traffic: a camera transmits
only toward regions where the radio and the twin disagree. For the
static mast links the mechanism is a zero-bandwidth tripwire whose
quiet state certifies free space to a robot approaching a blind corner.
We prove that under this policy the perception bytes grow with the rate
of real changes rather than with time, and we measure on the SCooP
recordings the detection of labelled people from the mast links' CSI
alone, the bytes saved at equal twin-update delay, and the deadline hits
recovered for the time-critical flows. The sensing that decides what to
communicate is a by-product of communicating.

---

## 1. Introduction

### 1.1 Motivation

Indoor robot teams converge on one architecture: each robot keeps a
local estimate, a server keeps a shared twin of the building (a metric
map, a dense scene, a record of change), robots feed the twin with
keyframes, the twin feeds the robots with map tiles and corrections,
and all of it crosses one Wi-Fi access point together with the control
and pose traffic that keeps the robots safe. The twin is the team's
common knowledge; keeping it current is a communication problem.

That problem is timing under contention, not average load. Two robots
at 0.5–1 MB/s of keyframe packs use under a tenth of the 275 Mbit/s
measured per-robot uplink [R]. Yet with two robots on the access point
the loaded round-trip time stays at 25–33 ms, far inside a 100 ms LiDAR
frame, while only 59–68 % of 1.5 MB messages arrive within the frame,
against 86 % with one robot and 100 % under the synthetic channels the
collaborative perception literature evaluates on [R]. On an earlier pass
the round-trip time rose from 3.3 ms idle to 60–109 ms loaded and
outages covered 12 % of frames [R]. A map tile queued in front of a
control command makes the command late. The question is which bytes
deserve airtime and when, not how to compress all of them.

Most keyframe bytes deserve none. Sent on a 2 Hz timer, a keyframe
describes walls and floors the twin holds to centimetres (7.8 cm wall
thickness, 1.9 cm floors [R]). What the twin needs is what changed since
it was built: a door closed, a cabinet moved, a person present. Change
detection finds that, but only after the keyframe has crossed the link
and been processed, which is the traffic we wanted to avoid; or on the
robot, which costs its compute and still needs the robot to be looking
the right way.

The team already owns a sensor that is always on, costs no bandwidth,
and covers the building *between* the nodes rather than in front of
them: the link. Every data frame carries channel state information, and
the testbed extracts it from frames in flight at 104–141 Hz on the
sensors' clock [R]. The channel between two nodes is a measurement of
the building between them, and the testbed has shown the building is
visible in it: Fresnel clearance from the reference map predicts the
detrended signal strength with Spearman ρ = 0.60, and a wall in the
first Fresnel zone costs 5.4 dB [R]. If the map predicts the radio, a
radio that departs from its prediction says the map is wrong, and on
which link, which places the departure in the building.

### 1.2 Research gap

Three literatures each hold a piece; none holds the loop.

*Communication-efficient collaborative perception* chooses what to send
from the sender's confidence, the receiver's request, or the channel's
capacity [where2comm, liu2026rdcomm, fang2024smartcooper, fang2025racp].
The channel is a constraint on the message or noise in it, never a
measurement of the scene; timeliness scheduling adds age
[zhu2026tamp, ma2026unseen], which is blind to content. Content-selection
rules in the V2X standards and the relevance literature
[etsi103562, lyu2025voi, qiu2022autocast, lusvarghi2026relevance] decide
by object dynamics, overheard redundancy, or the receiver's predicted
path; none has a sensor that reports staleness without anything being
sent.

*Radio sensing* has shown since the device-free localization and radio
tomographic imaging work of the late 2000s that a body in a link's
Fresnel zone changes its received power and multipath profile, and that
many links can image the change between them [wilson2010rti]. The nodes
are placed for sensing, the baseline is the link's own past with the
space empty, and the output is a detection or an image. Nothing
downstream of it chooses what another sensor should transmit.

*Radio maps and radio twins* predict the channel from geometry, by ray
tracing [hoydis2023sionnart] or by robot-collected maps
[luo2025fermi, amigoni2017survey], and read the residual between
prediction and measurement as calibration error to be fitted away [S].
The residual has not been read as "the geometry changed", and radio
maps in robotics drive motion, not message selection.

The loop no one has closed: the data traffic produces channel
measurements → a radio map the robots recorded while mapping predicts
them → the residual, placed by link geometry, says where the twin is
stale → that decides which camera sends a keyframe, over the same link,
producing the next measurement. Closing it needs, on one clock and in
one frame: CSI on the links that carry the data, both endpoints' poses
at each packet, a metric map to place the change, and a radio map
measured where the links are used. No published dataset has all four
[cooperscene: outdoor C-V2X traces, no CSI or shared map; v2xrealo: a
live link, no map]. SCooP has them at ≤ 0.99 ms and 3.1 cm [R].

### 1.3 Contributions

1. **Radio-gated perception.** A policy in which the residual between
   the measured channel on the team's own data links and a radio map
   gates perception traffic: a keyframe is requested only toward regions
   where the links say the twin is stale.
2. **Staleness evidence from link geometry.** A per-link change
   statistic (a CUSUM on two residuals, received power and subcarrier
   profile) projected into the building through each link's Fresnel
   zone, giving a ranking of regions by the probability that the twin
   is stale there, with moving nodes and the map supplying the geometry
   that fixed-node tomography has to assume.
3. **Radio free-space certificates.** For static links, the quiet state
   of the statistic over a window is a certificate that the corridor the
   link crosses is clear, delivered in a few bytes to a robot approaching
   a blind corner.
4. **A scaling result.** Under the gated policy the expected perception
   bytes over a run are bounded by a constant times the number of real
   changes, while under any timer-based policy they grow linearly in
   time; the gated policy's staleness delay is bounded by the links'
   sweep time plus the keyframe's delivery time.
5. **Evidence on a measured link.** Detection of labelled people from
   the mast links' CSI alone, placement error of the evidence map,
   bytes at equal twin-update delay against periodic keyframes, and the
   control and pose deadline hits recovered, on the SCooP recordings with
   a replay validated against live runs.

---

## 2. Related Work

### 2.1 Communication-efficient and task-oriented collaborative perception

Feature maps won accuracy per bit, and the last five years optimize
inside that choice: spatial confidence and request maps [where2comm],
codebooks and kilobyte messages [hu2024codefilling, infocom2026],
pragmatic rate–distortion [liu2026rdcomm], channel-adaptive compression
ratios set from allocated capacity or current CSI
[fang2024smartcooper, fang2024pacp, hu2024accda, fang2025racp,
efficomm2025, hou2025bevcooper, chen2026whispernet] [S], and semantic
coding for a noisy physical channel [gan2025scomcp, gan2025cods]. In all
of it the channel is a budget, a capacity or a noise source. CSI, where
it appears, sets a compression ratio. It is never evidence about the
scene, and what the receiver already holds enters at most as a request
map, never as a twin of the environment.

### 2.2 Content selection and timeliness

The V2X standards generate a Collective Perception Message by object
dynamics and mitigate redundancy by frequency, distance,
self-announcement and confidence rules [etsi103562, redundancy2025survey],
moving toward a value-of-information function [lyu2025voi, etsi103324].
Relevance work sends an object only if its predicted path crosses the
receiver's [qiu2022autocast, wang2024relevance, lusvarghi2026relevance];
visibility work sends what the receiver cannot sense [ma2026unseen].
Age-of-information scheduling [kaul2012aoi, kadota2018scheduling,
zhu2026tamp] and signal-aware sampling [sun2020wiener, maatouk2020aoii]
decide when to send from age or from the sender's own innovation. Every
rule is computed from what the sender perceives or what the receiver
asks; none has an input that reports change in the building without a
perception message being formed.

### 2.3 Radio sensing and radio tomographic imaging

Device-free localization and radio tomographic imaging invert the
attenuation on many links between fixed nodes into an image of what
obstructs them [wilson2010rti]; CSI-based sensing reads motion,
occupancy and gesture from one link's subcarrier amplitudes and phases;
mobile radio tomography mounts the nodes on vehicles [S]. The baseline
is the link's own history or a calibration with the space empty; the
nodes are placed to sense; the output is a detection. Here the baseline
is a radio map indexed by place, so a moving robot compares its link
against what the same place looked like during the survey; the nodes
are wherever the team's traffic puts them; and the output is a decision
about which other sensor transmits.

### 2.4 Radio maps and digital twins of the radio environment

Communication maps predict link quality between places to plan
multi-robot exploration [amigoni2017survey]; FERMI collects a radio map
with a robot team and a hybrid propagation model [luo2025fermi];
ray-traced twins predict the channel from a mesh [hoydis2023sionnart]
and are calibrated against measurements, with the residual reported as
sim-to-real error [S]. We take the opposite reading of the residual:
after the static part is calibrated, what remains is the building's
departure from its twin, and that is the quantity we want. We also need
no ray tracer: the pass that builds the metric twin builds the radio
twin, by recording the channel at every place it visits.

### 2.5 Change detection and twin maintenance in robotics

Spatio-temporal metric-semantic SLAM separates short-term dynamics from
long-term change and keeps the scene state over time
[schmid2024khronos]; map-based change detection decides what to update.
These detect change from a camera or LiDAR already looking at it. Our
mechanism detects change on the links between nodes, which cover the
building where no camera points, and directs the camera there.

### 2.6 Sequential change detection

Page's CUSUM [page1954cusum] and its extensions are the standard for
detecting a shift in the mean of a monitored statistic with a bounded
false-alarm rate and minimal detection delay (Lorden's criterion). We
run one per link on the residuals; the geometry then combines the
alarms. Multi-sensor CUSUM with known sensor-to-hypothesis weights is
classical [S]; the weights here are the Fresnel intersections that the
map and the poses supply.

### 2.7 Measured links in perception datasets

CooperScene logs C-V2X latency, throughput and loss per scene outdoors
[cooperscene]; V2X-ReaLO runs fusion over a live link [v2xrealo].
Neither has CSI, a shared metric map, or a radio map. SCooP-iT records
every inter-node link including CSI on the sensor clock and in the map
frame and validates its replay against live runs [scoop_testbed].

---

## 3. System Model

### 3.1 Nodes, frame, clock

A team of mobile robots i ∈ 𝓜 and fixed masts i ∈ 𝓘, one access point
a, one server. All nodes share one metric frame from the reference map
(3.1 cm median for moving nodes, fixed for masts [R]) and one clock
(offset ≤ 0.99 ms [R]). Node i has pose p_i(t), known on the server from
the 10 Hz pose flow and on the robot from its localization.

### 3.2 The twin and its staleness

The server holds the twin 𝒯_v at version v. Partition the building into
regions r ∈ 𝓡 (voxels or corridor cells). The building at time t is
B_t; the twin is stale in r when B_t and 𝒯_v differ there: an object
added, removed or moved, or a person present. Let z_r(t) ∈ {0, 1} mark
that, with onset times τ_r^(1), τ_r^(2), … and N_r(T) the number of
onsets in [0, T]. The twin's update of r at time u_r clears it. The
staleness delay of an onset is u_r − τ_r. Region r has a need n_r ∈
[0, 1]: the share of the team's planned paths over the horizon that
enter r.

### 3.3 Links and channel state

Active links 𝓛(t) are directed pairs ℓ = (i → j) that carried a data
frame in the last window. Each frame on ℓ overheard by a CSI-capable
node yields h_ℓ(t) ∈ ℂ^K over K subcarriers, at 104–141 Hz [R], valid
for relative speeds up to about 1.2 m/s [R]. Per window of w = 0.5 s we
use two scalars: detrended received power P̃_ℓ(t) (the E2 quantity) and
the normalized amplitude profile a_ℓ(t) = (|h_ℓ,k|)_k.

### 3.4 The radio map and the residuals

The mapping and survey passes give, for every cell c of a grid over the
map frame (177 cells on site 1 [R]) and every link endpoint j, the
survey channel H̄(c, j) = {P̄, ā, and their spreads}; for robot–robot
links, H̄(c, c′) over cell pairs. The prediction for ℓ = (i → j) at t is
the entry at the cells of p_i(t) and p_j(t), interpolated. Residuals:

    ρ_ℓ^P(t) = P̃_ℓ(t) − P̄(c_i(t), j)
    ρ_ℓ^a(t) = 1 − corr( a_ℓ(t), ā(c_i(t), j) ).

For a mast–AP link both ends are fixed; the prediction is the link's
own survey history and no map enters. Each residual has a measured
quiet-state distribution per link, mean 0 and spread σ_ℓ on the survey;
for moving robots σ_ℓ includes interpolation between cells.

### 3.5 Per-link change statistic

On each link and residual, a two-sided CUSUM [page1954cusum]:

    g_ℓ(t) = max( 0, g_ℓ(t − w) + |ρ_ℓ(t)| / σ_ℓ − k ),

with drift k and alarm when g_ℓ(t) > h. The pair (k, h) is set per link
from the quiet-state distribution for a false-alarm rate α per hour.
The alarm time of ℓ after an onset in its zone is T_ℓ^det; Lorden's
bound gives E[T_ℓ^det] ≤ (h + o(1)) / (γ_ℓ − k) for a shift of size γ_ℓ
in units of σ_ℓ.

### 3.6 Placing the alarm: link geometry

The first Fresnel zone of ℓ at t is the ellipsoid F_ℓ(t) with foci
p_i(t), p_j(t) and radius r_F(s) = √(λ s (d − s)/d) at distance s along
the path of length d. The map gives the points inside F_ℓ(t) (the E2
clearance computation). Sensitivity of ℓ to region r:

    w_ℓr(t) = vol( F_ℓ(t) ∩ r ) / vol( F_ℓ(t) ),

zero off the path. Evidence that the twin is stale in r:

    e_r(t) = Σ_ℓ ∫_{t − T_e}^{t} w_ℓr(t′) · 1{ g_ℓ(t′) > h } dt′ ,

normalized so that Σ_r e_r = 1 when any alarm is active. With two
robots moving at about 1 m/s and three masts, {F_ℓ(t)} sweeps the
building over tens of seconds; the sweep time of region r is
T_sweep(r) = the mean time until some link with w_ℓr > 0 is active.

**Certificates.** For a static link ℓ crossing corridor segment s, the
quiet state g_ℓ(t′) ≤ h for all t′ ∈ [t − Δ, t] is a free-space
certificate for s over that window, with miss probability bounded by
the per-link missed-detection rate β_ℓ measured in Sec. 5.

### 3.7 Flows, the shared link, deadline capacity

Flows, consumers and deadlines follow `../PROPOSAL.md` §3: control (tens
of ms), pose at 10 Hz (100 ms), keyframe (about 1 s), tile and
correction (seconds). All are tokenized to one MTU and committed to the
access point's queues, from which nothing is recalled. For a flow from
node i with deadline T, the deadline capacity S_i(t) = (T − δ_i − W_i)⁺
G_i is random given the sender's information; the same radio map gives
P(S_i ≥ s | c_i, path) from the survey's goodput and RTT per cell.

### 3.8 Cameras and what a keyframe resolves

Robot or mast i's camera at p_i(t) sees regions V_i(t) (ray cast on the
reference map). A keyframe from i, once processed, updates every r ∈
V_i(t) and clears e_r. Its size is b_kf, about 200–350 tokens.

---

## 4. Problem Formulation

**State.** (z_r(t))_r unknown; (g_ℓ(t))_ℓ, (e_r(t))_r, poses, and the
twin version observed.

**Decision.** At each slot the server chooses a request set
A(t) ⊆ 𝓜 ∪ 𝓘 of nodes that send a keyframe, and the robots' control and
pose tokens are committed first. The policy π maps the observed state to
A(t).

**Cost.** Over a run of length T,

    J(π) = E[ Σ_r n_r · Σ_{onsets of r} (u_r − τ_r) ]   +   λ · E[ Σ_t Σ_{i ∈ A(t)} b_kf ]
           ───────────────────────────────────────         ─────────────────────────────
                  needed staleness delay                          perception bytes

subject to: the keyframe tokens of A(t) fit the deadline capacity of
their senders with probability at least q after the control and pose
tokens (feasibility on the shared queue); and control and pose deadline
misses do not exceed their budgets ε_c, ε_p.

**Two reference policies.** Timer: A(t) = every robot every 1/f seconds
(the system as built). Oracle: A(t) = the node that sees r whenever
z_r turns on (knows the onsets).

**The gated policy.** Request from i the keyframe with the largest
positive index

    I_i(t) = Σ_{r ∈ V_i(t)} n_r e_r(t) · P( S_i(t) ≥ b_kf | c_i ) − λ b_kf / Ĝ_i(t),

and none when no index is positive. Control and pose tokens have the
largest index per byte and the earliest deadline, so they go first;
priority is an output of the rule and is implemented at the MAC by
access categories.

**Proposition (scaling).** Assume each onset in region r shifts the
residual of every link with w_ℓr > 0 by at least γ σ_ℓ (a body in the
first Fresnel zone, measured in Sec. 5), the false-alarm rate per link is
α, and a keyframe toward r clears e_r. Then under the gated policy

    E[bytes over [0, T]] ≤ b_kf · ( Σ_r E[N_r(T)] · m + α |𝓛| T ),

where m is the mean number of requests per onset (one when the first
alarmed region's camera sees it), and

    E[u_r − τ_r] ≤ T_sweep(r) + E[T_ℓ^det] + E[D_kf],

with D_kf the keyframe delivery and processing time. Under the timer
policy E[bytes] = b_kf · f · |𝓜| · T regardless of N_r, and
E[u_r − τ_r] ≈ 1/(2f) + E[D_kf]. So the gated policy's bytes scale with
the number of changes plus the false-alarm rate; the timer's scale with
time; and the gated policy's staleness delay exceeds the timer's by the
sweep and detection times, which the static links make zero for the
corridors they cross.

*Proof sketch.* Bytes: each keyframe is requested either after a true
alarm (at most m per onset, since the request clears the evidence) or
after a false alarm (rate α per link). Delay: an onset in r is in the
zone of some active link after T_sweep(r), alarms after T_ℓ^det by
Lorden, and the next request goes to a camera seeing r, which lands
after D_kf. ∎

The proposition is the paper's claim in one line: **perception traffic
proportional to change, with a delay set by geometry, at no sensing
cost.** The quantities it depends on (γ, α, β, T_sweep, D_kf) are all
measured on SCooP.

---

## 5. Metrics and evidence (plan)

| Metric | Definition | Ground truth | Data |
|---|---|---|---|
| Shift γ, false alarms α, misses β | per link kind, residual distributions with and without a labelled person in F_ℓ | labels in the map frame | survey_1, coop_2 |
| Detection | AUC of g_ℓ against "labelled change in F_ℓ(t)" per link kind | same | coop_2 |
| Placement | distance from argmax_r e_r to the labelled change | same | coop_2 |
| Certificates | share of blind-corner approaches with a valid certificate; certificate miss rate | labels, paths | coop_2 |
| Staleness delay | u_r − τ_r, gated vs timer vs oracle | labels, twin log | replay |
| Bytes | keyframe bytes at equal needed staleness delay | link traces | replay under `logged` |
| Time-critical deadlines | control and pose tokens by deadline, gated vs timer | sniffer and receiver stamps | replay, then live |
| Regret | J(ideal link) − J(π) per consumer | replay under `ideal` and `logged` | replay |

Order: γ, α, β on the static mast links first (days; no map needed);
then the moving links against the survey map (prices the interpolation);
then detection and placement; then the policy comparison in replay; then
one live run with the gate on the robots.

---

## 6. Novelty, stated once

The novelty is not that pieces are combined. It is a new input to the
communication decision and a new reading of an existing signal:

**The channel state of the team's own data traffic is used as the
observation that decides the content of that traffic.** In every prior
system the channel is a constraint on the message (a budget, a capacity,
a noise process) and the scene is observed by perception sensors. Here
the scene is observed by the channel, through a radio map the team
recorded while mapping, and the perception sensors are dispatched by
what the channel observed. The direction of integrated sensing and
communication is reversed: not sensing to improve communication, but
communication, as a sensor, deciding what to communicate.

Three consequences exist only once that reversal is made:

1. a staleness-evidence map of the building built from link residuals
   and link geometry, with moving nodes and no sensing deployment;
2. free-space certificates that cost zero bytes to sense and a few bytes
   to deliver;
3. a perception-traffic bound proportional to the number of changes in
   the building rather than to elapsed time (Sec. 4), which no
   timer-, age-, budget- or relevance-based policy can achieve, because
   each of those must send to find out whether anything changed.

The closest prior work is radio tomographic imaging, which images
obstructions with links placed for that purpose and stops at the image;
and radio-twin calibration, which measures the same residual and fits it
away. Neither feeds the residual back into a communication decision,
and neither has had the data to try: CSI on the data links, both poses
per packet, a metric map and a radio map on one clock exist only on
SCooP.

---

## 7. Scaling to N mobile and M fixed agents

Let |𝓜| = N robots, |𝓘| = M masts, one access point per site (several
at a large site, Sec. 7.6). Everything in Secs. 3–4 was written for
general 𝓜 and 𝓘; this section says what each quantity does as N and M
grow, and where the design must change.

### 7.1 Links: sensing grows quadratically, traffic linearly

Data links are to the access point: N + M uplinks and downlinks. Sensing
links are every pair (i → j) whose frames node j can overhear with a
CSI-capable radio, up to (N + M)(N + M − 1) directed pairs; in practice
every CSI node overhears every transmitter within range, so the sensing
set grows as the square of the team while the data traffic grows
linearly. The static sub-mesh is the M mast–AP links plus up to
M(M − 1)/2 mast–mast pairs: a fixed tripwire net laid over the building
by the masts' placement, which turns Sec. 3.6's certificates from one
corridor into a coverage problem (Sec. 7.4).

**CSI budget.** A CSI node extracts at most R_csi frames per second
(104–141 Hz measured [R]), shared over the links it overhears. Per link
the rate is R_csi / L_j, with L_j the transmitters j hears. The Sec. 3.5
CUSUM needs a few samples per window; at w = 0.5 s and R_csi = 100 Hz a
node can monitor ~20 links at 2–3 samples per window before the window
must grow. Beyond that, nodes take disjoint subsets of the links (an
assignment the server fixes from the poses), or the window lengthens
and T_det grows accordingly.

### 7.2 Evidence: a region's radio signal-to-noise grows with the links that cross it

With many links, Sec. 3.6's sum becomes a log-likelihood combination.
For region r let 𝓛_r(t) = {ℓ : w_ℓr(t) > 0}. Under the Sec. 4
assumption (an onset in r shifts every ℓ ∈ 𝓛_r by ≥ γ σ_ℓ), the
per-window log-likelihood ratio for "r changed" is

    Λ_r(t) = Σ_{ℓ ∈ 𝓛_r(t)} w_ℓr(t) · ( |ρ_ℓ(t)| / σ_ℓ − γ/2 ) · γ ,

and e_r is its CUSUM. The detectability of r is γ √|𝓛_r|: doubling
the links crossing a region halves the shift a single link needs. The
false-alarm rate is controlled per region, not per link, so the
α |𝓛| T term of the scaling proposition is replaced by α_r |𝓡| T with
α_r set by the regional CUSUM threshold; it does not grow with the
number of links. This is the standard multi-sensor gain, and it is what
makes the moving robots useful as sensors even when each robot's
residual is marginal (Sec. 3.4's interpolation penalty).

### 7.3 Sweep time falls with N

T_sweep(r) is the mean time until some active link crosses r. The static
net covers a fixed set 𝓡_static of regions with T_sweep = 0. For the
rest, with N robots moving independently at speed v along the team's
planned paths, the coverage process of the Fresnel zones is a union of
N renewal processes, so

    T_sweep(r) ≈ T_sweep^(1)(r) / N

to first order, with T_sweep^(1) the single-robot sweep time measured on
the survey route. Staleness delay in the proposition therefore falls as
1/N while its other terms (T_det, D_kf) do not grow.

### 7.4 Mast placement as tripwire coverage

Given the corridor graph of the twin, each candidate mast pair (and
each mast–AP pair) defines a line of sight; a corridor segment is
covered if some static line crosses it inside the first Fresnel zone.
Choosing M mast positions to cover the segments with the highest need
n_r is weighted set cover; greedy placement is within ln(|segments|) of
optimal, and the masts already serve as perception nodes, so the cost
is their placement, not their count. The covered set 𝓡_static is the
set of corridors with certificates at zero sweep time; the paper
reports it for site 1 with M = 3.

### 7.5 Who responds: a weighted max-coverage, not a per-node index

With N + M cameras, several may see an alarmed region and one camera
may see several alarmed regions. The request set A(t) replaces the
single-node index of Sec. 4 with

    max_{A ⊆ 𝓜 ∪ 𝓘}  Σ_{r ∈ ∪_{i ∈ A} V_i(t)} n_r e_r(t)  −  λ Σ_{i ∈ A} b_kf / Ĝ_i(t)
    s.t.  the keyframe tokens of A fit the deadline capacity of their senders after control and pose,

a monotone submodular objective under a knapsack-like feasibility
constraint; greedy by marginal gain per byte, with the best single
request as a fallback, is within (1 − 1/e)/2 of optimal
[khuller1999budgeted]. The marginal-gain step is also what prevents m
in the scaling proposition from growing with N: once one camera's
keyframe is requested for r, the gain of every other camera seeing r is
zero. Election from poses alone (`../METHODS.md` M5) gives the same
result distributedly when the server is unreachable.

### 7.6 Contention: the deadline capacity shrinks as 1/N_active, the certificates do not

Under DCF the N_active backlogged robots share the airtime roughly
evenly [heusse2003anomaly], so a robot's keyframe deadline capacity is
S_i ≈ (T − δ_i − W_i)⁺ G_i / N_active and the in-time probability
P(S_i ≥ b_kf) in the index falls with N_active. The radio map carries
this through its contention mode (survey vs cooperation, measured as
86 % vs 59–68 % in-frame [R]); for general N the mode is indexed by
N_active, measured during the survey with k robots loading the link, or
modeled by the DCF share. Two consequences:

- the gated policy's advantage *grows* with N: timer traffic
  b_kf f N T competes for a capacity that shrinks per robot, while gated
  traffic stays bounded by changes;
- the time-critical flows need the priority access category more, not
  less, as N grows: N × 10 Hz pose tokens plus control stay a few
  percent of airtime, but behind bulk FIFO they would miss.

Certificates and alarms are 1-bit-per-link-per-window reports: |𝓛|
bits per 0.5 s, negligible at any N.

With several access points, each AP is a contention domain; links to
different APs do not share airtime, and the sensing links still overhear
across domains. The radio map is per AP association; the survey records
it by walking each cell associated to each AP in range.

### 7.7 The survey scales with the site, not with the team

The survey robot records, at every cell, the channel to every endpoint
that transmits during the survey: all M masts and the AP at once. Its
cost is therefore O(|cells|), independent of M. Robot–robot links need
cell pairs, O(|cells|²); the survey records those its route happens to
visit, and the rest fall back to the clearance model of E2 (a
parametric prediction from the map) with a larger σ_ℓ. Reciprocity
halves the pairs. For N robots the map is shared, not per robot, since
the radios are identical (single stream, 802.11ac, 80 MHz [R]); a
different radio class needs its own survey.

### 7.8 The scaling proposition, restated for (N, M)

Under the assumptions of Sec. 4 with per-region false-alarm control
(Sec. 7.2):

    E[bytes_gated]  ≤  b_kf ( Σ_r E[N_r(T)] + α_r |𝓡| T ),            independent of N and M
    E[bytes_timer]  =  b_kf f N T,                                       linear in N
    E[u_r − τ_r]    ≤  1{r ∉ 𝓡_static} T_sweep^(1)(r)/N + E[T_det(r)] + E[D_kf],

with T_det(r) falling as 1/(γ √|𝓛_r|) and 𝓡_static growing with M. So
adding robots adds sensing and shortens delay without adding gated
traffic, and adding masts turns corridors from swept to certified. The
only quantity that worsens with N is the per-robot deadline capacity
under contention, which hurts the timer policy far more than the gated
one.

### 7.9 What SCooP can and cannot test at scale

Site 1 has N = 2, M = 3. The quantities that depend on N (sweep time,
contention mode) are measured at N = 1 and N = 2 and extrapolated by the
1/N and DCF models; the quantities that depend on M (static coverage)
are computed on the corridor graph for any placement, since they are
geometry. The scaling claims are therefore stated as the proposition
with measured constants, and the N = 1 vs N = 2 contrast is the one
empirical scaling point. A larger team is the stated next step.
