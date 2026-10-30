# The Link Senses What to Send: Radio-Triggered Maintenance of a Shared Twin for Indoor Robot Teams

Draft of the front matter: research gap, motivation, introduction,
related work, and system model. Numbers marked [R] are from the SCooP
recordings (`scoop_testbed_paper/RESULTS_STATUS.md`); claims about other
papers marked [S] come from abstracts and must be checked against the
primary text before submission. Bib keys refer to
`../references.bib`.

---

## Abstract (draft)

A robot team that shares a twin of its building must keep the twin
current, and every update crosses the same shared wireless link that
carries the team's control and pose traffic. Today the update traffic is
sent on a timer, so most of it describes what the twin already holds,
and it competes with the traffic that cannot wait. We show that the link
itself can say when the twin is stale. Each data packet the team
exchanges yields a channel-state measurement on the sensors' clock; a
radio map measured once, during the mapping pass, predicts that
measurement for the places the nodes stand in; the residual rises on
exactly the links whose Fresnel zone crosses something the map does not
contain. We use that residual, placed in the map by the link geometry,
to decide which camera sends a keyframe and when, so that perception
traffic flows only toward the regions where the radio and the twin
disagree. On the SCooP recordings, the static mast links detect a
labelled person in their Fresnel zone with AUC [x] from the subcarrier
profile alone, and residual-triggered keyframes reach the same
twin-update delay as periodic keyframes at [y] % of the bytes, while the
control and pose deadlines, which periodic traffic missed in [z] % of
frames under contention, are met. Sensing that decides what to
communicate is here a by-product of communicating.

---

## 1. Introduction

### 1.1 Motivation

Indoor robot teams are converging on one architecture: each robot keeps
a local map and a local estimate, and a server keeps a shared twin of
the building, including a metric map, a dense scene representation, and
a record of what has changed. The robots feed the twin with keyframes;
the twin feeds the robots with map tiles and corrections; both share the
same Wi-Fi access point with the control and pose traffic that keeps the
robots safe. The twin is the team's common knowledge, and keeping it
current is a communication problem.

The communication problem is not bandwidth on average. Two robots
sending keyframe packs at 0.5–1 MB/s use well under a tenth of the
measured 275 Mbit/s per-robot uplink [R]. It is timing under contention.
On the SCooP testbed, with two robots on one access point, the loaded
round-trip time stays at 25–33 ms, far inside a 100 ms LiDAR frame, yet
only 59–68 % of 1.5 MB messages arrive within that frame, against 86 %
with one robot and 100 % under the synthetic channels the collaborative
perception literature evaluates on [R]. Under an earlier pass the
round-trip time rose from 3.3 ms idle to 60–109 ms loaded, and real
outages covered 12 % of frames [R]. A map tile in front of a control
command in the same queue makes the command late. The question is
therefore which bytes deserve the airtime and when, not how to compress
all of them.

Most of the update traffic deserves none of it. A keyframe sent on a
2 Hz timer describes, for the most part, walls and floors the twin
already holds to centimetre accuracy. What the twin needs is the part of
the building that has changed since it was built: a door closed, a
cabinet moved, a person present. Change detection against the twin can
find that part, but only after the keyframe has been sent and processed
on the server, which is the traffic we wanted to avoid, or on the robot,
which costs the robot's own compute and still requires the robot to be
looking in the right direction.

The team already has a sensor that is always on, costs no bandwidth, and
covers the building between the nodes rather than in front of them: the
link. Every data packet carries channel state information, and the
testbed extracts it from the frames in flight at 104–141 Hz on the
sensors' clock [R]. The channel between two nodes is a measurement of
the building between them. The testbed's own validation shows the map is
visible in the radio: Fresnel clearance computed from the reference map
predicts the detrended signal strength with Spearman ρ = 0.60, and a
wall in the first Fresnel zone costs 5.4 dB [R]. If the map predicts the
radio, then a radio that departs from its prediction says the map is
wrong, and says on which link, which places the departure in the
building.

### 1.2 Research gap

Three literatures each hold one piece of this and none holds the loop.

**Communication-efficient collaborative perception** decides what to
send from the sender's own view, the receiver's request, or the
channel's capacity: spatial confidence maps, codebooks, pragmatic
rate–distortion, channel-adaptive compression ratios
[where2comm, liu2026rdcomm, fang2024smartcooper, fang2025racp]. The
decision is content-driven or capacity-driven; nothing in it asks
whether the receiver already holds what is about to be sent, and
nothing in it uses the channel as evidence about the scene. Timeliness
scheduling adds age of information [zhu2026tamp, ma2026unseen], which
is content-blind: a keyframe of an unchanged wall and one of a newly
closed door age alike.

**Radio sensing** has shown for fifteen years that bodies and objects in
the Fresnel zone of a link change its received power and its multipath
profile, and that a set of links can image the change between them
(radio tomographic imaging [wilson2010rti]; CSI-based motion and
occupancy sensing). The sensing is done with fixed nodes, against the
link's own past, and the output is a detection or an image, not a
decision about what another sensor should transmit.

**Digital twins of the radio environment and radio maps** predict the
channel from geometry, by ray tracing on a mesh [hoydis2023sionnart] or
by a robot-collected map [luo2025fermi, amigoni2017survey], and the
residual between prediction and measurement is treated as calibration
error to be fitted away [S], or as an anomaly in the spectrum [S]. The
residual has not been used as evidence that the geometry changed, and
radio maps in robotics drive motion planning, not message selection.

The loop that no one has closed is: the data traffic produces channel
measurements; a radio map built from the robots' own mapping pass
predicts them; the residual, placed in the map by the link geometry,
says where the twin is stale; and that decides which camera sends a
keyframe, through the same link, which produces the next measurement.
Closing it needs four things in one frame and on one clock: a radio map
measured at the places the links are used, channel state on the links
that carry the data, the poses of both ends at the instant of each
packet, and a metric map to place the change in. No published dataset
has all four [cooperscene has measured C-V2X traces outdoors without
CSI or a shared map; V2X-ReaLO has a live link without a map]. SCooP has
all four, on one clock within 0.99 ms and in one frame to 3.1 cm [R].

### 1.3 Contributions

1. **Radio-triggered twin maintenance.** A mechanism in which the
   residual between the measured channel on the team's data links and a
   radio map measured during the mapping pass is placed in the building
   by the links' Fresnel geometry and used to schedule perception
   traffic toward the regions where the radio and the twin disagree.
2. **A measured radio map as the predictor.** The survey pass that the
   testbed already records is the radio twin: per-cell CSI and RSSI in
   the map frame, with the static mast links as a baseline that needs no
   map at all. No ray tracer is required.
3. **Two residuals and their comparison.** Detrended received power, the
   quantity existing clearance models predict, and the correlation of
   the per-subcarrier amplitude profile against the survey, which a
   body alters before it alters the total power. The second fires
   earlier; we quantify by how much.
4. **Scheduling with the same map.** The residual sets the value of a
   keyframe; the radio map sets the probability that it arrives in time;
   one index (Sec. 3.6) commits tokens to the shared queue so that
   control and pose traffic is never behind bulk traffic.
5. **Evidence on a measured link.** Detection of labelled people in the
   Fresnel zones of the static and moving links, the bytes saved at
   equal twin-update delay, and the deadline hits of the time-critical
   flows, all on SCooP recordings with a replay validated against live
   runs.

---

## 2. Related Work

### 2.1 Communication-efficient and task-oriented collaborative perception

The field transmits raw points, feature maps, or detections and has
settled on feature maps for accuracy per bit. Where2comm [where2comm]
selects spatial cells by the sender's confidence and the receiver's
request; CodeFilling [hu2024codefilling] and InfoCom [infocom2026]
compress to codebooks and kilobytes; RDComm [liu2026rdcomm] gives the
rate–distortion theory with a task-driven distortion. Channel-adaptive
variants set the compression ratio from the allocated capacity or the
current CSI: SmartCooper, PACP, ACC-DA, R-ACP, EffiComm, BEVCooper,
WhisperNet [fang2024smartcooper, fang2024pacp, hu2024accda, fang2025racp,
efficomm2025, hou2025bevcooper, chen2026whispernet] [S]. Semantic
communication for collaborative perception encodes features for a noisy
physical channel [gan2025scomcp, gan2025cods]. In all of these the
channel is a constraint on the message or a source of noise in it; it
is never a measurement of the scene. The receiver's state enters only
as a request map; what the receiver already holds, which is the whole
twin here, does not price the message. And every evaluation uses a
channel that is allocated, constant, or simulated, under which a
message's in-time delivery is certain and the effects we measure
vanish.

### 2.2 Timeliness: age of information and scheduling

Age of information [kaul2012aoi, yates2021aoi] and its scheduling
theory [kadota2018scheduling] give the time-average freshness of
content-blind updates; remote-estimation results show that signal-aware
sampling beats age-optimal sampling [sun2020wiener], and age of
incorrect information adds content to age [maatouk2020aoii]. Applied to
collaborative perception, Fresh2comm, TAMP and Update-the-Unseen-Only
[wu2025fresh2comm, zhu2026tamp, ma2026unseen] schedule by age, by an
age-to-accuracy penalty, or by what the receiver cannot sense itself
[S]. The last is the closest in spirit: it conditions on the receiver's
view. None conditions on the receiver's twin, and none has a sensor that
reports staleness without sending anything.

### 2.3 Radio sensing and radio tomographic imaging

That a person in the first Fresnel zone of a link changes its received
power is the basis of device-free localization and radio tomographic
imaging [wilson2010rti], which inverts the attenuation on many links
between fixed nodes into an image of the obstruction. CSI-based sensing
extends this to the multipath profile, detecting motion, occupancy and
gestures from a single link's subcarrier amplitudes and phases. Mobile
radio tomography has mounted the nodes on vehicles [S]. In all of it,
the baseline is the link's own history or a calibration with the space
empty, the nodes are placed for sensing rather than for communicating,
and the output is a detection. Our baseline is a radio map indexed by
place, so that a moving robot compares its link against what the same
place looked like during the survey; our nodes are where the team's
data traffic puts them; and the output is a decision about which other
sensor should transmit.

### 2.4 Radio maps and digital twins of the radio environment

Communication maps in multi-robot exploration predict link quality
between places to plan motion [amigoni2017survey]; FERMI collects a
radio map with a robot team and a hybrid propagation model
[luo2025fermi]; ray-traced twins predict the channel from a mesh
[hoydis2023sionnart], and a growing literature calibrates them against
measurements and reports the residual as sim-to-real error [S]. We take
the opposite view of that residual: once the static part is calibrated,
what remains is the building's departure from its twin, and it is the
quantity we want. We also avoid the ray tracer for now: the mapping
pass that builds the metric twin also builds the radio twin, by
recording the channel at every place it visits.

### 2.5 Change detection and twin maintenance in robotics

Spatio-temporal metric-semantic SLAM separates short-term dynamics from
long-term change and keeps the scene state over time [Khronos, RSS
2024]; change detection against a prior map decides what to update in
the twin. These systems detect change from the camera or LiDAR that is
already looking at it. Our mechanism detects change on the links
between nodes, which cover the building where no camera points, and
uses that to direct the camera.

### 2.6 Event-triggered estimation

Event-triggered estimation sends a state only when its innovation
exceeds a threshold and treats the absence of a message as information,
since the receiver knows the rule [trimpe2014et]; lossy extensions
assume an i.i.d. drop probability [S]. Our radio map gives a drop
probability that varies with place; Sec. 3.6 uses it so that silence on
a clear link means more than silence on a shadowed one.

### 2.7 Measured links in perception datasets

CooperScene logs C-V2X latency, throughput and loss per scene outdoors
[cooperscene]; V2X-ReaLO runs fusion over a live link [v2xrealo]. Neither
has CSI, a shared metric map, or a radio map, so the loop above cannot
be closed on them. SCooP-iT records every inter-node link, including
CSI, on the sensor clock and in the map frame, and validates its replay
against live runs [scoop_testbed].

---

## 3. System Model

### 3.1 Nodes, frames, clock

A team of N nodes: mobile robots i ∈ 𝓜 and fixed masts i ∈ 𝓘, one
access point, and a server. All nodes share one metric map frame, set
by the reference map built in the mapping pass, with a validated
accuracy of 3.1 cm median for the moving nodes and fixed poses for the
masts [R]; and one clock, with measured offset ≤ 0.99 ms [R]. Node i has
pose p_i(t) ∈ SE(3), known on the server from the pose flow (Sec. 3.5)
and on the robot from its localization.

### 3.2 The twin and its staleness

The server holds the twin 𝒯_v at version v: the reference map
(a TSDF surface with measured thickness 7.8 cm on walls and 1.9 cm on
floors [R]), the dense scene, and a free-space record. The building at
time t is B_t. Partition the building into regions r ∈ 𝓡 (voxels or
room cells). The twin is stale in region r at time t if B_t and 𝒯_v
differ there: an object added, removed or moved, or a person present.
Let z_r(t) ∈ {0, 1} be that indicator. The quantity to minimize is the
staleness delay, the time from z_r turning on to the twin's update of
r, over the regions the team needs (Sec. 3.6), at the lowest cost in
airtime.

### 3.3 Links and channel state

At time t the set of active links is 𝓛(t), each a directed pair
ℓ = (i → j) that carried at least one data frame in the last window.
Each frame on ℓ overheard or received by a CSI-capable node yields

    h_ℓ(t) ∈ ℂ^K,

the channel across K subcarriers, at 104–141 Hz [R]. From h we use two
scalars per window of length w = 0.5 s:

- received power P_ℓ(t) = 10 log₁₀ Σ_k |h_ℓ,k|², and its detrended form
  P̃_ℓ = P_ℓ − (free-space trend in distance), the E2 quantity;
- the amplitude profile a_ℓ(t) = (|h_ℓ,k|)_k, normalized.

The CSI is valid for relative speeds up to about 1.2 m/s at the achieved
sample rates (three samples per half wavelength) [R], which covers the
team.

### 3.4 The radio map and the residual

The mapping and survey passes, which the testbed records anyway, give
the radio map: for every cell c of a grid over the map frame (177 cells
on site 1 [R]) and every node j that was a link endpoint, the survey
channel

    H̄(c, j) = { P̄(c, j), ā(c, j), and their spreads },

with masts as fixed endpoints and the survey robot's cell as the other.
For a robot–robot link, H̄(c, c′) over pairs of cells, sparser.

The predicted channel for link ℓ = (i → j) at time t is the survey
entry at the cells of p_i(t) and p_j(t), interpolated between cells.
Two residuals:

    ρ_ℓ^P(t) = P̃_ℓ(t) − P̄(c_i(t), j)                       (power)
    ρ_ℓ^a(t) = 1 − corr( a_ℓ(t), ā(c_i(t), j) )              (profile)

For a mast–AP link both endpoints are fixed, the cell is constant, and
the prediction is the link's own survey history; no map enters. The
static residual, with nothing in the zone, has a floor σ_ℓ measured per
link on the survey; the floor for moving robots includes the cost of
interpolation between cells and is measured on the same pass.

### 3.5 Link geometry: placing the residual in the building

For link ℓ at time t, the first Fresnel zone F_ℓ(t) is the ellipsoid
with foci p_i(t), p_j(t) and radius

    r_F(s) = √( λ s (d − s) / d )

at distance s along the path of length d, λ the wavelength. The testbed
computes the zone's clearance against the reference map, which is the
E2 quantity, so the map points inside F_ℓ(t) are known. Define the
sensitivity of link ℓ to region r as

    w_ℓr(t) = volume( F_ℓ(t) ∩ r ) / volume( F_ℓ(t) ),

zero for regions off the path. A residual on ℓ is evidence for a change
somewhere in the regions with w_ℓr > 0. With the links of two moving
robots and three masts, and the robots moving at about 1 m/s, the set
{F_ℓ(t)} sweeps the building over tens of seconds, and the evidence
accumulates per region:

    e_r(t) = Σ_ℓ Σ_{t′ ∈ [t − T_e, t]} w_ℓr(t′) · 1{ |ρ_ℓ(t′)| > κ σ_ℓ },

with threshold κ and window T_e. This is a detector-based radio
tomography with moving nodes, with the map providing the geometry that
fixed-node tomography has to assume. Its output is not an image but a
ranking of regions by the probability that the twin is stale there.

### 3.6 Flows, the shared link, and the decision

The flows, their consumers and deadlines are as in `../PROPOSAL.md`
§3: control (tens of ms), pose at 10 Hz (100 ms), keyframes (about
1 s), tiles and corrections (seconds). Every flow is packetized into
tokens of one MTU and committed to the access point's queues, from which
nothing can be recalled. The deadline capacity of a flow from node i
with deadline T is

    S_i(t) = ( T − δ_i(t) − W_i(t) )⁺ · G_i(t),

random given what the sender knows; the same radio map that predicts
the channel for the residual predicts its distribution, P(S ≥ s | c_i,
path), from the survey's goodput and RTT per cell and the clearance
along the planned path.

**The keyframe decision.** Robot i's camera at pose p_i(t) sees a set
of regions V_i(t) (ray cast against the reference map). The value of a
keyframe from i is the evidence it would resolve:

    F_i(t) = Σ_{r ∈ V_i(t)} e_r(t) · (1 − u_r),

with u_r the share of r already updated since e_r rose. The index of
committing the keyframe's tokens is

    I_i(t) = w_kf · F_i(t) · P( S_i(t) ≥ b_kf ) − λ · b_kf / Ĝ_i(t),

and the server requests the keyframe from the robot with the largest
positive index, subject to the per-window feasibility of the queue with
the control and pose tokens committed first (they have the largest
index per byte and the earliest deadline, so priority is an output of
the rule, implemented at the MAC by access categories). When no
region's evidence exceeds the floor, no keyframe is sent; the twin is
maintained on the radio's word.

**Silence as a measurement (optional, Sec. 2.6).** Object states from
robot i are event-triggered on the innovation against the twin. The
server treats a frame with no state message from i as the likelihood
P(no trigger) + P(trigger)·P(lost | c_i), with P(lost | c_i) from the
same radio map, so that silence from a clear cell sharpens the server's
estimate and silence from a shadowed cell does not.

### 3.7 Objective and metrics

The objective over a run is the one of `../OBJECTIVE.md`, with the
twin's consumer loss ℓ_twin = Σ_r (staleness delay of r) · (need of r),
need being the share of the team's planned paths that enter r. Reported
metrics:

| Metric | Definition | Ground truth |
|---|---|---|
| Detection | AUC of ρ_ℓ against "labelled person or moved object in F_ℓ(t)" per link kind | labels in the map frame |
| Placement | distance from argmax_r e_r to the labelled change | same |
| Staleness delay | time from a labelled change to the twin's update of its region | labels, twin log |
| Bytes | keyframe bytes per unit of staleness delay, residual-triggered vs periodic | link traces |
| Time-critical deadlines | share of control and pose tokens delivered by their deadline | sniffer and receiver stamps |
| Regret | J(ideal link) − J(policy), per consumer | replay under `ideal` and `logged` |

All of it is computed on the recordings through the logged channel,
whose replay error against live runs is bounded in the companion paper
[scoop_testbed]; the live run of the full loop is the final experiment.
