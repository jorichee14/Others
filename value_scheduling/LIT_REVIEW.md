# What to send, to whom, when, and who decides: a review of the literature and its gaps

Scope: methods that decide the **content** of inter-agent perception
messages and the **use of the shared channel** that carries them, in
collaborative perception for vehicles and robots, plus the standards,
networking, MARL, robotics-mapping and information-theory work that bears
on the same decision. The review is organized by the decision each line
of work makes and the model of the receiver, the channel and time that
it assumes, because the gaps are in those models, not in the methods.

Confidence: **[F]** read in full (Dense Coverage / Sparse Refinement,
PRISM); **[S]** from abstracts, search summaries or secondary citations
(arXiv was unreachable from this session); classical results unmarked.
Nothing tagged [S] should be quoted in a paper without reading it.

---

## 1. The decision space

Every method in this review takes a position, often implicitly, on
seven questions:

| Axis | Question | Typical positions |
|---|---|---|
| Representation | what kind of thing is sent | raw points · BEV features · codebook indices · object states · tokens or queries · trajectories · free space |
| Selection | which parts of it | all · by sender confidence · by receiver request · by predicted relevance · by redundancy |
| Receiver model | what the sender assumes the receiver already has or will have | nothing · its own current view · its request · its predicted path · its map |
| Time | what deadline or age enters the decision | none · a frame · age of information · the receiver's decision time |
| Resource | what is charged | nothing · bytes · a per-link budget · slots · airtime |
| Locus | who decides | sender · receiver · a center (RSU, AP, edge) · a learned joint policy |
| Channel | how the network is modeled | ideal · fixed budget · constant delay · PHY noise · slotted collisions · a simulator · a measured link |

The field's methods cluster in a small region of this space: features
selected by the sender's confidence, with no receiver model beyond a
request map, time absent or as age, bytes as the resource, the sender as
locus, and a budget as the channel. Most of the gaps in section 3 are
the empty regions of the table, and two of them are axes the field has
not yet drawn.

---

## 2. Lines of work

### 2.1 Representation: from raw sensing to objects, and back toward structured features

Early, intermediate and late fusion fix the representation at the
architecture level: raw points (Cooper), feature maps (F-Cooper, V2VNet,
AttFuse, CoBEVT, V2X-ViT), detections (late fusion). The learning
community settled on feature maps for accuracy per bit; the standards
community settled on objects. ETSI's Collective Perception Message
carries perceived objects with kinematics, a `timeOfMeasurement`, and
free-space information [etsi103324] [S]. Object-first designs persist in
the learning literature as "centres first, detail on request" (Chiu and
Smith, selective communication for end-to-end driving, 2023 [S]) and in
instance-level messages: SparseCoop's kinematic-grounded queries with
explicit position and velocity (AAAI 2026) [S], CoPLOT's point-level
tokens [S], QUEST, CoopDETR, INSTINCT. Hybrids send objects where the
sender is confident and features where it is not (CoSDH, CVPR 2025;
HyComm) [S]. Layered representations, a coarse dense base plus sparse
fine refinements, appear in Dense Coverage / Sparse Refinement (Yazgan,
Mueller, Zoellner, 2026) [F], which borrows scalable video coding's base-
plus-enhancement structure and gives it a task-aware ranking.

*What this line assumes.* The representation is chosen once for the
system, not per receiver or per object. Late fusion's ceiling (0.78
against 0.86 AP@0.7 for intermediate fusion in the parent study on
OPV2V) and object messages' tiny size are two ends of one trade-off that
no method lets a sender move along per recipient.

### 2.2 Sender-side selection

The dominant line. Where2comm (NeurIPS 2022) masks BEV cells by a spatial
confidence map and a request map from the receiver. EffiComm (ITSC 2025)
sets per-vehicle keep ratios from role, scene complexity and
instantaneous network load with a graph attention network [S]. What2comm
(ACM MM 2023) decouples exclusive from common features [S]. How2comm
(NeurIPS 2023) filters by mutual information with the collaborator and
adds flow-guided delay compensation [S]. InfoCom (AAAI 2026) reaches
kilobyte messages through an information bottleneck with progressive
reconstruction [S]. CodeFilling (CVPR 2024) sends codebook indices [S].
RDComm (ICLR 2026) gives the rate–distortion theory with a task-driven
"pragmatic" distortion and assigns codeword lengths by task relevance
[S]. PragComm (Hu et al., 2024; TPAMI) selects regions and timestamps,
represents them through a task-adaptive codebook, and selects
collaborators by whether a neighbour's demanded information is present
[S]. WhisperNet (CVPR 2026) scores spatial and channel importance at the
sender and coordinates requests at the receiver, and reaches 1 %
bandwidth [S]. Dense Coverage / Sparse Refinement [F] ranks 16×16-cell
patches by a learned benefit score (Pyramid Window Attention, ROC-AUC
0.785 against counterfactual marginal detection gain), sends the
top-N_fine under a strict byte cap as a nested prefix, and shows that
the gain comes from the coverage–refinement allocation rather than from
quantization.

*What this line assumes.* A **per-link byte budget known to the sender**.
Value is the sender's estimate of what helps *a* receiver, with the
receiver present only as a request map (Where2comm, PragComm) or absent.
Time is absent: a message is either within budget or not. The channel
never partially delivers. DR states its own limit: one-shot broadcast
"does not remove cross-agent redundancy, so communication still grows
with the number of collaborators" [F].

### 2.3 Receiver-driven and interactive selection

Who2com (ICRA 2020) and When2com (CVPR 2020) learn a handshake: a
degraded agent sends a compressed query, others reply with matching
scores, the requester selects [S]. CoSDH's supply–demand masks let the
receiver's demand (from its detection confidence) gate what each sender
supplies [S]. LIF (2025) computes a demand map from the ego's and the
sender's uncertainty and transmits only where demand passes a threshold
[S]. HydraCollab (2026) gates rich feature sharing to regions where both
agents have confident overlapping information, and chooses between
intermediate and late collaboration per region [S]. Reason-to-Transmit
(2026) inserts a small transformer that reasons about what to send
rather than transmitting the sender's most confident regions, and
reports an oracle built from ground-truth object content in a synthetic
four-agent environment [S].

*What this line assumes.* A **request round** before the data round, and
the receiver's **current** uncertainty as the definition of need. The
extra round is the cost DR and the standards argue against (channel
access is paid twice at kilobyte payloads [F]), but no paper in either
camp measures that cost on a MAC; each asserts it. And the receiver's
need is its need *now*: nothing in the request says what the receiver
will see for itself a moment later.

### 2.4 Receiver-conditioned relevance and predicted visibility (object level)

This is the line the user pointed at, and it is older and better
developed than the feature-level work. AutoCast (MobiSys 2022) predicts,
from agents' future trajectories, the visibility and relevance of each
detected object to each other vehicle, and a global max-weight scheduler
allocates bandwidth to the objects that are both relevant and not
visible to the receiver; it reports 2–4× more visibility into safety-
critical objects [S]. Wang and Cao (2024) do edge-assisted dissemination
with relevance scored by the likelihood that two trajectories intersect
[S]. "The Search for Relevance" (Lusvarghi et al., TVT 2026) argues from
relevance theory that an object is relevant to a receiver only if it can
influence that receiver's driving decision, and reports a two-fold
efficiency gain [S]. ETSI's draft VoI function (Lyu et al., VNC 2025)
consolidates object inclusion rules into one value [S]. Update the
Unseen Only (2026) schedules infrastructure updates by whether the
receiver can sense the region itself, minimizing age over unseen regions
[S]. MASS and C-MASS (Jia, Sun, Zhou et al., 2023–2024) schedule *which
sensors* contribute, predicting perception gain from past observations
with a restless-bandit regret bound, and in C-MASS a second-order
topology over vehicle pairs with a greedy guarantee [S]. V2X-BEVDet4D
(VNC 2025) fuses a roadside detection only when the object is beyond
25 m and the ego's own model missed it: a receiver-side complementary
filter [S].

*What this line assumes.* **Objects as the unit**, roads as the geometry
(lanes make trajectory prediction tractable and FoVs forward-facing),
and bandwidth as a count. Two consequences. First, the value of an
object to a receiver is binary or a collision likelihood, not a task
gain; the detector downstream is not in the loop. Second, the insight
that value is *conditional on what the receiver will perceive itself*
has never been carried to the feature level, where the accuracy is. The
reason is practical: at object level, "the receiver will see it" is a
ray cast; at feature level, the equivalent counterfactual requires the
receiver's own future features, which only an offline dataset can
supply. No feature-level paper has used that offline supply even as an
oracle.

### 2.5 Redundancy across senders

The standards handle redundancy by rules evaluated per sender from what
it overhears: frequency, distance, dynamics, self-announcement,
confidence, entropy (ETSI TR 103 562) [S]; surveys catalogue them and
their failure under loss [redundancy2025survey] [S]. JigsawComm (2025–26)
is the first feature-level treatment: a Feature Utility Estimator
predicts each agent's per-location contribution, agents exchange compact
utility maps, and a top-1 policy keeps, per location, the one agent with
the highest utility, so the data-channel cost stays constant as agents
are added; its guarantees hold with respect to the learned proxy [S].
PRISM (Mazandarani, Shokrnezhad, Taleb, TNSM 2026) [F] treats redundancy
as *semantic segments shared across UEs* given by an association matrix
from an orchestration layer, defines self and assisted throughput, and
learns a distributed MAC (D3QL with value decomposition) that chooses
channel and packet length per UE; segments have equal value, time is
slotted, and the association matrix is an input the UEs do not see.
C-MASS's second-order topology is the scheduling-theory version.

*What this line assumes.* Redundancy is either a **count** (how many
announced it), an **assignment** (which one agent owns a location), or
an **association** supplied from outside. None of them is a function of
geometry and time: two views of one cell from different sides are not
redundant when one is occluded, and a cell is redundant *for a given
receiver* only until that receiver sees it. JigsawComm's exchange of
utility maps is itself a round; PRISM's association matrix is assumed
known.

### 2.6 Timeliness

Age of information (Kaul, Yates, Gruteser 2012; survey Yates et al. 2021;
scheduling Kadota et al. 2018) measures staleness of content-blind
updates. In collaborative perception, Fresh2comm optimizes AoI [S]; TAMP
(2026) schedules regions at an RSU by a Lyapunov policy with an empirical
age-and-volume-to-AP penalty [S]; AoI-guaranteed scheduling (Wang et al.
2023) allocates resource units at a center under per-region age
constraints [S]; Han et al. (TCCN 2026) derive a feature's AoI from its
size and the link and weight fusion by it [S]. Remote estimation shows
signal-aware sampling beats age-optimal sampling (Sun, Polyanskiy, Uysal
2020) and age of incorrect information adds content to age (Maatouk et
al. 2020). On the receiving side, latency compensation (SyncNet,
CoBEVFlow, FFNet, TraF-Align, CoDynTrust, DATA, Temporal Residual
Bottleneck) repairs a late message rather than deciding what to send
[S]; DR reports its own robustness to 200 ms delay without compensation
[F].

*What this line assumes.* Time enters as **age** or as **a frame
deadline**. Neither is the quantity that matters to a receiver, which is
*when its own decision will depend on the content*. A message about a
region the receiver will not enter for three seconds has a three-second
deadline; one about a person behind the corner it is turning has a
deadline of now. The content-selection line (2.4) has the geometry to
compute this and never does; the timeliness line has the scheduling
theory and no geometry.

### 2.7 The channel and the access layer

Three channel models dominate. A **budget** (Where2comm, DR, RDComm): the
sender knows its bytes. A **constant or Binomial delay** (V2X-ViT's 100 ms,
CoBEVFlow's lags): everything arrives, late. A **noisy PHY** (SComCP,
CoDS: SNR, fading, in CoDS a digital constellation) [S]: symbols are
corrupted, packets never queue. Channel-adaptive compression (SmartCooper,
PACP, ACC-DA, R-ACP, BEVCooper, WhisperNet's motivation) sets the level
from allocated capacity or current CSI [S]; all treat the capacity as
known at decision time. SchedCP selects users by CSI and semantics with
deep RL [S]. PRISM [F] is the one semantic-aware **MAC**: distributed,
slotted, collisions, no deadline. V2X-ReaLO runs fusion over a live link
and finds offline evaluation misses transmission latency [S]; CooperScene
releases per-scene C-V2X latency, throughput and loss outdoors [S].
On the Wi-Fi side, 802.11ax uplink OFDMA with trigger frames and buffer
status reports is a centralized uplink scheduler in every modern AP,
studied in the networking literature on scheduler design and modeled in
ns-3, but never connected to perception content.

*What this line assumes.* The channel is a constraint on the message or
a noise process in it. Airtime, the quantity actually shared on a Wi-Fi
or sidelink channel, which depends on each sender's rate and on how
many are sending, appears nowhere in the perception literature; DR's
Supp. F notes that at kilobyte payloads channel access and PHY overhead
dominate serialization time, and then models the channel as a constant
per trip [F]. Measured-link datasets exist but carry no CSI and no
shared map, so they cannot test anything that depends on geometry.

### 2.8 Learning to communicate (multi-agent RL)

CommNet, IC3Net (a learned binary gate on whether to speak), TarMAC
(targeted attention), Gated-ACML (message pruning under a bandwidth
limit), Wang et al. ICML 2020 (an information-bottleneck protocol plus a
scheduler deciding which agent sends what), event-triggered MARL
communication (send when the observation changed enough), CommGIB (graph
information bottleneck) [S]. The abstractions are the right ones, when,
what, to whom, under a limit, and the evaluations are StarCraft, traffic
lights and packet routing, with messages that are latent vectors and
channels that are a count of messages.

*What this line assumes.* A learned policy can discover the value of
communication from reward. For perception at kilobytes per frame with
MAC-level costs and a detector downstream, nobody has closed that loop,
and the sample cost of doing so from reward alone is the reason the
perception literature hand-designs value heads instead.

### 2.9 Robotics: sharing maps between robots

Rocha, Dias and Carvalho (2005) share the measurements with the highest
entropy-based utility and choose viewpoints that minimize mutual
information between robots' maps [S]. Aragues et al. send map increments
when the local map has changed enough [S]. Kimera-Multi and Swarm-SLAM
are bandwidth-parsimonious by design (sparse keyframe descriptors,
distributed optimization) but their sharing policy is fixed [S].
Commerge (2026) exchanges a graph-selected subset of scans for merging,
5,000× fewer [S]; CQLite shares Q-values at frontiers [S]; a 2025
aerial-to-ground paper trades how much map to send against the ground
robots' navigation cost [S]. Event-triggered estimation (Trimpe and
D'Andrea 2014) sends a state when its innovation exceeds a threshold and
lets silence carry information.

*What this line assumes.* The receiver holds a **map**, and value is
novelty against it. This is the only line in which the receiver's
accumulated knowledge is the baseline for value, and it is absent from
the collaborative-perception literature, which values each frame as if
the receiver had no memory.

### 2.10 Foundations the field cites, and what they cover

Value of information and the expected value of perfect information
(Howard 1966) bound what any information source can be worth to a
decision maker. Submodular selection with greedy guarantees (Nemhauser,
Wolsey, Fisher 1978; Krause and Guestrin for sensor placement) underlies
every greedy selector in 2.2–2.5, and the budgeted and matroid variants
(Khuller–Moss–Naor 1999; Sviridenko 2004) cover unequal costs. Wyner–Ziv
(1976) and its successive-refinement form (Steinberg–Merhav 2004) are
the information theory of coding against what the receiver already
holds. Successive refinement (Equitz–Cover 1991) and the broadcast
approach (Shamai–Steiner 2003; Tian et al. 2008) are the theory of
layered messages for unknown channels. Max-weight and drift-plus-penalty
(Neely 2010) and the AoI scheduling theory give the long-run allocation
tools. Remote estimation over random delay (Sun et al. 2020) gives
signal-aware sampling.

*What is cited and what is used.* The greedy bounds are used; VoI is
cited as motivation and never computed against a decision; Wyner–Ziv is
cited for codebooks and never for the receiver's map or view; the
broadcast approach and successive refinement are cited for layering and
never connected to a channel that partially delivers; scheduling theory
is used by the AoI line with content-blind value.

### 2.11 Evaluation practice

Datasets: OPV2V (simulated, up to seven agents), V2XSet, V2X-Sim,
DAIR-V2X (real, N = 2), V2V4Real (N = 2), V2X-Real (N = 4), V2XVerse.
Channels: a byte cap, a constant delay, or nothing. Metrics: AP@0.5 and
AP@0.7 against average KB per agent per frame. Payload accounting is
inconsistent across papers; DR's Supp. D reconstructs decodable payloads
for Where2comm, EffiComm, ERMVP, MMCooper, CoSDH and InfoCom and finds
that masks, indices and request rounds are variously omitted [F].
Oracles: DR's counterfactual marginal gain per patch [F] is the most
rigorous; Reason-to-Transmit's oracle uses ground-truth object content
in a synthetic environment [S]; JigsawComm's guarantee is relative to its
own learned proxy [S]. Measured links: V2X-ReaLO and CooperScene, both
outdoor vehicles, no CSI, no shared map, no replay validation.

*What this practice cannot see.* With a fixed budget per agent, the
in-time delivery of a message is certain, so any method that sizes
messages to delivery, or that makes them useful when partly delivered,
scores exactly the same as one that does not; the effect is zeroed by
the evaluation channel, not absent. With AP-versus-KB as the only
curve, two methods with the same bytes but different airtime (different
MCS, different packing, an extra request round) are indistinguishable.
Without the receiver's future frames in the loop, "the receiver would
have seen it anyway" cannot be scored, although the datasets contain
those frames.

---

## 3. The gaps

These are stated as gaps in *kind*: a quantity the field does not define,
an axis it has not drawn, a measurement it cannot make. Where a gap is
"A has not been combined with B", it is listed only if the combination
changes what the decision is.

### G1. Value is defined on the sender's side of the link

Every feature-level method prices a unit by the sender's estimate of its
usefulness (confidence, utility, benefit, information), at most masked by
the receiver's current request. The receiver-conditioned line (2.4)
prices an object by whether the receiver will see it or collide with it,
but stops at objects and at binary relevance. **No method defines the
value of a feature unit to a specific receiver as its marginal task gain
over what that receiver will perceive on its own before the information
matters.** That counterfactual is computable offline from any sequential
dataset (the receiver's future frames are in it) and has never been
reported even as an oracle. It is not an integration of 2.2 and 2.4: it
changes what the number being estimated *is*, from a property of the
sender's data to a property of the pair (unit, receiver, horizon).

### G2. Time is age, never need

Deadlines are frames; freshness is age. The quantity a receiver cares
about is the time at which its own decision depends on the content,
which is computable from its planned motion and the geometry and ranges
from "now" to "never". Nothing in the literature defines per-unit
deadlines from receiver decisions, so every scheduler treats a warning
about the corner the receiver is turning and an update about a corridor
it will not enter as equally urgent, and every evaluation scores them at
the frame boundary.

### G3. The resource is bytes, not airtime

On a shared channel the cost of a message is the airtime it occupies,
which depends on the sender's rate, on the packing, on PHY overhead that
dominates at kilobytes, and on how many others are sending. Every
perception paper charges bytes; DR notices the problem and models the
channel as a constant per trip [F]. The consequence is not a constant
factor: at equal bytes, airtime differs by an order of magnitude across
MCS, and the choice of *which agent* sends is a choice of how much
airtime a byte costs. No method's objective contains that term, so no
method can trade a far agent's valuable patch against a near agent's
cheaper one.

### G4. Redundancy is a count, an assignment or an input, not geometry over time

The standards count announcements; JigsawComm assigns each location to
one agent; PRISM takes the association as given. Two views of one
surface from different angles are complementary, not redundant, for
detection and more so for mapping; a region is redundant for a receiver
exactly until the receiver observes it; and the redundancy between two
senders depends on which receiver is asked. A redundancy term that is a
function of the senders' poses, the receiver's pose and the horizon has
not been written down, although every input to it is already exchanged
(poses at 10 Hz).

### G5. The locus of decision is chosen by convenience and never analysed

Senders decide because they have the data; receivers decide because they
know their need; centers decide in the scheduling line because the math
is central. No paper asks which party holds the information the decision
needs: the sender has the content, the receiver has its need, and only a
center has every agent's pose, demand and rate at once. The one place in
a deployed network where all of that is already collected every few
milliseconds is the access point's uplink scheduler (802.11ax trigger
frames and buffer status reports), and it has never been given
perception content to decide on. The interactive line argues that a
request round is expensive and the standards line that broadcast is
blind; neither measures the alternative of a center that decides in the
round the MAC already runs.

### G6. Granularity is an architecture choice, not a per-receiver decision

Whether to send an object, a box with velocity, a feature patch or raw
points is decided once per system. For a given receiver at a given
moment the right answer differs by distance, by whether the receiver is
about to gain its own view (association needs features, re-detection
needs none), and by the budget. The hybrids (CoSDH, HyComm) choose by
sender confidence per region; nobody chooses by the receiver's state,
and nobody has an objective in which a centroid and a patch of the same
object compete for the same airtime at their respective values.

### G7. There is no clairvoyant bound, so nobody knows how far any method is from optimal

Selection methods report AP against bytes and against each other. The
expected value of perfect information for "what to send" is well defined
(the selection a clairvoyant would make with the receiver's future, the
delivery outcome and the task loss known) and computable offline on
every sequential dataset; it is reported nowhere. DR's counterfactual
gain [F] is the nearest thing, for one sender and one receiver with a
known budget; JigsawComm's bound is relative to its own proxy; Reason-to-
Transmit's oracle is synthetic. Without the bound, a method's shortfall
cannot be attributed to value estimation, to redundancy, to allocation,
or to the channel, and the literature's progress is unmeasured in
absolute terms.

### G8. The evaluation channel zeroes the effects that matter

A fixed byte budget makes in-time delivery certain, so sizing to
delivery and partial-delivery usefulness are worth exactly zero under
it; a constant delay makes every message arrive whole. These are not
approximations of a shared link; they are channels under which the
decisions in G1–G6 have no consequence. The measured-link datasets
(V2X-ReaLO, CooperScene) restore the consequence but lack CSI, a shared
map and replay validation, so geometry-dependent decisions cannot be
tested on them. Payload accounting differs between papers by the cost
of masks, indices and request rounds [F]. The field therefore cannot
currently distinguish a method that sends less from one that arrives in
time, and cannot rank methods on airtime at all.

### G9. The receiver's memory is not side information

Collaborative perception values each frame as if the receiver had no
past. The robotics-mapping line values novelty against the receiver's
map, and the information theory of coding against the receiver's
knowledge (Wyner–Ziv) is forty years old. A receiver that holds a map, a
twin, or simply its own tracks from the last second already knows most
of what any sender's frame contains; the value of a unit is its
innovation against that knowledge. No perception method, and no
evaluation, conditions on it.

### G10. One message, one consumer

Every objective has one consumer, the ego detector. A robot team's link
carries messages for several consumers with different deadlines:
control, pose, detection, tracking, prediction, mapping. The value of a
byte differs by consumer and the consumers compete for the same airtime.
No objective sums them, so no method can decide that a map tile should
yield to a pose or that a detection patch should yield to a warning.

---

## 4. What the gaps imply, without a plan

Three of the gaps change what the central quantity is and are therefore
the ones worth a paper on their own:

- **G1 with G2**: the value of a unit is a property of (unit, receiver,
  horizon), its deadline is the receiver's decision time, and both are
  computable from geometry the team already exchanges. The oracle (G7)
  for this value exists in every sequential dataset and has never been
  reported.
- **G3 with G5**: the resource is airtime and the party that can price
  it per sender and allocate it in one round is the access point. The
  decision moves to where the information is.
- **G8**: until the evaluation channel can deliver partially and charge
  airtime, none of the above can be measured, which is why the DR
  authors could see the redundancy problem and not the airtime one, and
  why the AoI line could see time and not content.

The remaining gaps (G4, G6, G9, G10) are consequences: once value is
receiver-conditioned over a horizon, redundancy (G4) and granularity
(G6) become functions of the same geometry, the receiver's map (G9) is
the horizon-zero case of its future perception, and multiple consumers
(G10) are a weighted sum of the same value with different deadlines.

## 5. Closest prior work, ranked by how much of the above it already holds

| Paper | Holds | Lacks |
|---|---|---|
| AutoCast (MobiSys 2022) [S] | receiver-conditioned relevance and visibility from predicted trajectories; a central max-weight scheduler | objects only; relevance binary; bandwidth as count; roads |
| Dense Coverage / Sparse Refinement (2026) [F] | equal-size nested units; a learned task value with a counterfactual oracle; strict budgets; zero-retraining adaptation | sender-side; receiver-blind; per-link budget; redundancy stated as open; channel as per-trip constant |
| JigsawComm (2025–26) [S] | feature-level cross-agent redundancy by per-location utility assignment | a utility-map exchange round; symmetric redundancy; receiver-agnostic; bytes |
| Update the Unseen Only (2026) [S] | value conditional on the receiver's own sensing | age as the objective; atomic updates; allocated downlink |
| PRISM (TNSM 2026) [F] | semantic redundancy inside the MAC; self and assisted throughput | distributed; equal-value segments; slotted; no deadline; association given |
| TAMP / AoI-guaranteed scheduling (2023–26) [S] | central scheduling with a perception-linked penalty | age-based; region-level; orthogonal resources |
| MASS / C-MASS (2023–24) [S] | which sensors to schedule by predicted gain with regret and approximation bounds | sensors as units, not content; no receiver conditioning |
| Reason-to-Transmit (2026) [S] | a receiver-aware selection module and an oracle | synthetic environment; oracle from object content |
| Rocha et al. (2005) [S]; Trimpe and D'Andrea (2014) | value as novelty against the receiver's map; silence as information | no perception task; no shared-channel cost |

None holds G1 (value as marginal over the receiver's future), G2
(deadlines from receiver decisions) or G3 (airtime as the resource), and
none can be evaluated on G8's terms.

---

## References

Keys are in `references.bib` (this folder) and
`../task_oriented_comm/references.bib` for the timeliness, information-
theory and radio entries. Entries first used here and not yet in either
file: Chiu and Smith 2023 (selective communication, end-to-end driving);
What2comm (ACM MM 2023); How2comm (NeurIPS 2023); PragComm (Hu et al.
2024); LIF (2025); HydraCollab (2026); Reason-to-Transmit (2026);
JigsawComm (2025–26); Lusvarghi et al. (TVT 2026); Wang and Cao (2024);
MASS and C-MASS; V2X-BEVDet4D (VNC 2025); IC3Net, TarMAC, Gated-ACML,
Wang et al. ICML 2020; Rocha et al. 2005; Aragues et al.; Commerge
(2026); CQLite; Howard 1966; Krause and Guestrin. All [S] except where
marked, and all to be added with checked records before any of this is
cited.
