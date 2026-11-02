# What to send: the content-selection family, and new methods in it

The earlier files fixed the channel side (deadline capacity, progressive
messages, the twin as side information). This file is about the other
side: **which objects, regions or facts deserve to be sent at all**, the
family of methods that send a centroid instead of a cloud, that predict
whether the receiver already sees an object, that drop what the receiver
will not need. Section 1 is the catalog of what exists. Section 2 gives
new methods in that style built for an indoor team with a map, masts
and people. Section 3 picks a paper.

Literature from search summaries [S] unless marked classical; read before
citing.

---

## 1. Catalog of existing tricks

| # | Trick | Representative work | What it decides from | What it assumes |
|---|---|---|---|---|
| 1 | **Send objects, not sensors.** Centroid, box, velocity, class; the ETSI CPM Perceived Object Container with `timeOfMeasurement` | ETSI TS 103 324; late fusion; Chiu & Smith 2023 (centres first, then selective detail) | sender's detections | receiver trusts the sender's detector |
| 2 | **Dynamics-based inclusion.** Send an object only if it moved > 4 m, changed speed > 0.5 m/s, or 1 s passed | ETSI TR 103 562 generation rules; "Generation of CPMs" (Thandavarayan, Sepulcre, Gozalvez 2019) | the object's own change | every receiver wants every object |
| 3 | **Redundancy mitigation by overhearing.** Omit an object if enough neighbours already announced it (frequency, distance, self-announcement, confidence, entropy rules) | ETSI TR 103 562; survey arXiv 2501.01200; MDPI fi15020041 | what others transmitted | overhearing is reliable; all receivers alike |
| 4 | **Value of information per object.** One scalar from accuracy, novelty, dynamics, redundancy decides inclusion; now in the ETSI TS 103 324 draft | Lyu et al., VNC 2025 best paper; relative-entropy VoI; the "value-anticipating" patent | object state and confidence | VoI is receiver-independent |
| 5 | **Relevance to the receiver's trajectory.** Send an object only if its predicted path and the receiver's intersect; score by collision likelihood | AutoCast (MobiSys 2022); Wang & Cao, edge-assisted relevance dissemination 2024; "The Search for Relevance" (Lusvarghi et al., TVT 2026) | both parties' predicted paths | road geometry; prediction horizon |
| 6 | **Predicted visibility.** Send what the receiver cannot see: ego-visible vs occluded regions; visibility from the receiver's pose | Update the Unseen Only (AoI by region the receiver can't sense); AutoCast's visibility prediction; Share the Unseen (T-CST 2025) | the receiver's field of view | FoV known; occluders known |
| 7 | **Request-driven (pull).** The receiver says what it needs; senders answer: learned handshakes, request maps, demand maps from uncertainty | Who2com / When2com (Liu, Kira 2020); Where2comm request map; LIF demand map (2025); HydraCollab gating (2026) | receiver's uncertainty | a request round-trip fits the deadline |
| 8 | **Uncertainty-triggered.** Share only where the ego is unsure, or only confident detections | LIF; COOPERTRIM (2026); CoDynTrust (stale-feature trust) | detector uncertainty | calibrated uncertainty |
| 9 | **Negative information.** Share free space and "nothing here", not only objects | Tischler & Vogt 2007; CPM free-space container; three-state occupancy (free/occupied/unknown) in 2026 adversarial CP | observed-empty volumes | receiver can use free space |
| 10 | **Event-triggered state.** Send only when the innovation against a shared prediction exceeds a threshold; silence is information | Trimpe & D'Andrea 2014; event-triggered cooperative localization | innovation | shared predictor on both ends |
| 11 | **Send predictions, not states.** Trajectories with uncertainty instead of boxes; feature flow instead of features | FFNet; Collaborative Trajectory Prediction via Late Fusion (2026); V2XPnP; CMP | sender's forecast | forecast beats receiver's own |
| 12 | **Hybrid detail levels.** Boxes where confident, features where not; intermediate–late hybrids | CoSDH (CVPR 2025); HyComm | confidence per region | — |
| 13 | **Who sends.** Elect the best-view agent per region; graph grouping; scheduler assigns bandwidth by relevance | When2com; MASS / C-MASS mobility-aware sensor scheduling; AutoCast's max-weight scheduler | view quality, mobility | central or overheard coordination |
| 14 | **Novelty for maps.** Entropy-utility measurement selection; change-triggered map increments; scan-subset merging | Rocha et al. 2005; Aragues et al.; Commerge (2026, 5000× fewer scans) | information gain against the shared map | map shared |

Three regularities across the catalog. First, almost everything is
vehicle-centric: paths are lanes, FoVs are forward, occluders are other
cars. Second, the receiver's state enters as a field of view or a
request, never as a shared twin of the whole environment. Third, none of
it is evaluated on a measured shared link, so "send less" is never
weighed against "send in time".

---

## 2. New methods for an indoor team with a map, masts and people

The indoor setting changes what is clever. The map is static and shared
to centimetres; people move along corridors, so their futures are
graph-constrained; masts have fixed, known coverage; robots announce
their poses at 10 Hz anyway; occluders are walls, which the map knows,
not other agents. Each method below uses one of these.

### M1. Send only what the receiver will need before it can see it itself

For each object o the sender holds and each receiver j: from j's
planned path and the map, ray-cast the time t_see(j, o) at which j's own
sensors will first see o (walls and the object's predicted motion
included). From j's path, the time t_need(j, o) at which o first affects
j's decision (enters its collision cone or its planned corridor minus
reaction time). Send o to j only if

    t_need(j, o) < t_see(j, o) + margin,

and send it so that it lands by t_need. Everything j will see for itself
before it matters is never transmitted. Catalog items 5 and 6 each have
half of this; the map makes both halves computable indoors, and the
difference t_see − t_need is a per-object deadline that the channel side
can use directly.

*Test:* on coop_2 with labels and recorded paths, the fraction of
labelled objects per frame with t_need < t_see, and the bytes of boxes
needed, against send-all boxes; detection of people before t_need as
the task metric.

### M2. Transmit the receiver's blind volume, not the objects in it

A mast that sees a corridor sends robot j not the people in it but the
volume j is blind to that the mast currently sees as **empty**: a
free-space certificate "corridor segment s is clear until t + Δ", a few
bytes. The robot can then commit to speed through its blind corner. When
the segment is not clear, the mast sends the object (M1). Negative
information (item 9) is the cheapest message there is, and in a building
it is also the most common state. The certificate's Δ comes from the
mast's view and the maximum person speed.

*Test:* share of frames in coop_2 where a robot's planned corridor is
certifiably clear by a mast; bytes of certificates vs bytes of boxes;
effect on the OcclusionHandoff task (warning or clearance lands before
the decision point).

### M3. Map-constrained intent tokens for people

In a corridor graph, a person's future is a path through nodes. The
sender tracks the person and sends the next-edge probability vector
(one byte per edge, a handful of edges) plus the current node and speed,
instead of a trajectory. The receiver, with the same graph, regenerates
the trajectory distribution. The graph is part of the twin, so both ends
share it. This is item 11 with the map as the codebook; the forecast
indoors is far better than in open space because the graph constrains
it.

*Test:* trajectory-prediction error of edge-probability messages vs
polyline messages at matched bytes, on labelled people in coop_2.

### M4. Mast authority: robots stay silent where a mast covers

Mast coverage of the building is static and computed once from the map
(ray cast from each mast). Within a mast's covered volume the mast is
the sole announcer; robots send detections only for regions outside
every mast's coverage. Redundancy mitigation (item 3) by geometry rather
than by overhearing, with no coordination traffic and no reliance on
hearing others' messages, which on a contended link is exactly what
fails. A region leaves mast authority only when the mast's own view is
blocked, which the mast knows (M2's certificate becomes "I can no longer
certify").

*Test:* bytes of robot detections inside vs outside mast coverage on
coop_2; detection recall of people in covered regions from masts alone.

### M5. Best-view election from poses alone

Every node already knows every other node's pose at 10 Hz and the map.
For each region r, each node computes deterministically which node has
the best view (distance, incidence, occlusion from the map) and only
that node sends for r. No handshake (item 7), no overhearing (item 3),
no learned grouping (item 13): the election is a pure function of the
pose broadcast all nodes already receive. Ties break by node id.
Disagreement arises only from pose-message loss, which the radio map
predicts; a node that has not heard a pose for k frames assumes the
worst case and sends.

*Test:* duplicate detections per labelled object across senders, with
election vs without; bytes; recall.

### M6. Receiver-conditioned detail ladder

Per object and receiver, the detail level follows the same t_see,
t_need geometry as M1: centroid when t_need is far; box and velocity
when t_need is near; features (for the receiver's detector to fuse) only
in the handoff window when the receiver is about to gain its own view
and needs to associate the object. Item 12's hybrid, keyed on receiver
geometry rather than sender confidence.

### M7. Blind-cone requests, geometric

The receiver broadcasts, with its pose, the cone it is blind to along
its planned path (computable on the receiver from the map, a few bytes).
Senders answer only inside it. Who2com's handshake (item 7) without
learning and without a round-trip: the request rides on the pose message
that is sent anyway.

### M8. Just-in-time delivery at the decision point

M1 gives each object a deadline t_need. The channel side gives the
in-time probability for a message sent at t with size b. The sender
chooses the latest send time that keeps the in-time probability above a
target, so that the message carries the freshest state that still
arrives in time. Sending early wastes freshness; sending late risks the
deadline; the radio map sets the balance per place. This is where the
content side meets the deadline side.

### M9. The radio says where to look (from `CLEVER.md`)

Send camera or LiDAR data only toward regions where the links' CSI
disagrees with the radio map. Content selection by a sensor that costs
no bandwidth.

---

## 3. Which of these makes a paper

M1 + M2 + M4 are one coherent method: **relevance by geometry**. A
shared map, the receiver's path, and the masts' fixed coverage decide,
per object and receiver, whether to send at all (M1), whether a
free-space certificate replaces the object (M2), and who sends (M4).
Nothing is learned, nothing is overheard, every decision is a ray cast
in the map that both ends hold. The claim: an indoor team with a shared
map needs to transmit a small fraction of what vehicle-style CP
transmits, and the fraction is computable before sending. The evidence:
on SCooP, bytes against task value (people the receiver could not see
before it needed them, warnings and clearances landing before decision
points) under the logged link, where "send less" and "send in time" are
finally on the same axis.

M3 and M6 are the content-coding additions if the first paper lands.
M5 is a short standalone note. M8 is the bridge to the deadline work
and belongs in the same paper as section 3 of `PROPOSAL.md`. M9 is its
own paper (`paper/DRAFT.md`).

Position against: "The Search for Relevance" (Lusvarghi et al., TVT
2026), which argues for relevance-driven V2X in principle; AutoCast,
which does trajectory relevance on roads; Update the Unseen Only, which
does visibility for AoI; and the ETSI VoI draft. None has a map the
receiver shares, masts, or a measured link.
