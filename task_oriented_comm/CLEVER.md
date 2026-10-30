# Three ideas that are not gap-filling

Each is a single mechanism with a counter-intuitive claim, a reason nobody
has tried it, a way it could die, and a test that runs on the recordings
in hand. Novelty checks are from search summaries (arXiv unreachable from
this session); the closest prior work is named so it can be read first.

---

## Idea 1. The link senses what to send (recommended)

**Mechanism.** Every data packet the team already exchanges yields a CSI
measurement at 100–140 Hz on the sensor clock. The twin, with Sionna RT
on its mesh and the sender's and receiver's known poses, predicts that
CSI. The residual, measured minus predicted, is near its calibrated floor
while the building matches the twin, and rises on exactly the links
whose Fresnel zone crosses something the twin does not contain: a moved
cabinet, a closed door, a person. The scheduler requests a keyframe only
from the camera or LiDAR that sees the region where the radio and the
twin disagree. Everything else stays untransmitted.

**The claim.** Twin maintenance needs no perception traffic until the
communication signal itself says the twin is stale. The sensing that
decides what to communicate is a by-product of communicating. The
dynamic radio tomography is formed by the links in use: two moving
robots and three masts, with the mast–AP links as a static baseline
where any residual is a change.

**Why nobody has tried it.** It needs four things in one frame and on one
clock: a radio twin accurate enough that the static residual is small,
CSI on the links that carry the data, the poses of both ends at the
instant of each packet, and a map in which to place the change. Radio
tomographic imaging (Wilson and Patwari 2010) has fixed sensors and no
twin; Sionna-twin calibration papers fit the residual away rather than
use it; Wi-Fi sensing detects motion but does not know where the camera
should look. The closest search hits are a spectrum-anomaly paper using
a ray-traced twin, and a W-band industrial paper that learns the
residual, neither of which schedules anything.

**What could kill it.** Sensitivity. E2 shows walls at 5.4 dB; a body at
5 GHz is 3–10 dB, furniture less. Multipath makes the residual noisy at
the per-packet level, and Sionna's twin fidelity at 5 GHz indoors is
unproven here. If the residual cannot separate "a person crossed" from
"the robot turned" at the 0.5 s scale, the idea is dead.

**Test on the recordings (one week, no new hardware).**
1. Static baseline: the mast–AP and mast–robot links on survey_1 while
   the robot is stationary. Residual against clearance-only prediction,
   then against Sionna on the reference mesh. Pass: residual standard
   deviation below 2 dB over 0.5 s windows.
2. Detection: on coop_2, the labelled people's positions give, per
   window and link, whether a body is in the first Fresnel zone. ROC of
   the residual against that label. Pass: AUC above 0.8 on at least one
   link kind.
3. Scheduling value: compare "request a keyframe where the residual
   fires" against "send keyframes at 2 Hz" on the bytes needed to reach
   the same twin-update delay for the labelled changes.

**Where it leads.** A paper whose title could be "Communication that
knows when the map is wrong". Venue: IEEE JSAC special issue on ISAC, or
MobiCom if the live run is in hand.

---

## Idea 2. Silence is a measurement when the loss is predictable

**Mechanism.** Senders are event-triggered: a robot sends an object state
only when its innovation against the shared prediction (the twin plus
the last sent state) exceeds a threshold. The receiver, who knows the
rule, treats a frame with no message as evidence that the innovation
was small. On a lossy link that inference is ambiguous: not sent or
lost. Here the receiver also knows P(lost | x) from the map and the CSI,
so silence becomes a soft measurement with a likelihood

    P(silence | state) = P(no trigger | state) · 1 + P(trigger | state) · P(lost | x).

The receiver's posterior sharpens when the map says the link was clear
(silence means nothing happened) and stays wide when it says the link was
shadowed (silence means nothing). The trigger threshold itself is set
from P(lost | x): send with a lower threshold where the link is poor,
because silence there is worth less.

**The claim.** The radio map is part of the estimator, not only of the
scheduler. Two agents sharing a twin and a radio map can communicate
below the event-triggered rate because non-delivery carries information
whose value is known per place.

**Why nobody has tried it.** Event-triggered estimation treats silence as
information (Trimpe and D'Andrea 2014; the cooperative-localization paper
found in the search fuses "implicit" measurements), and the lossy
extensions assume an i.i.d. drop probability. Nobody has had a drop
probability that varies with place and is predictable from a map, and
collaborative perception has never used event triggering at all.

**What could kill it.** The gain is bounded by how much P(lost | x) varies.
If the in-time probability is 0.9 everywhere, silence is always
informative and the map adds nothing; if it is 0.6 everywhere, the same.
The idea pays only where the map separates clear from shadowed places,
which E2 says it does at ρ 0.60.

**Test on the recordings.** On coop_2, replay the labelled people as
sender-side tracks. Simulate event triggering at a fixed threshold, pass
the triggered messages through the logged link, and run two receivers: one
with silence treated as nothing, one with silence as the likelihood
above using the in-time probability of the logged link at that place.
Compare tracking error against the labels at matched bytes. Pass: the
map-aware receiver matches the always-send baseline at under half its
bytes.

---

## Idea 3. The deadline is the decision, not the frame

**Mechanism.** The receiver's deadline for a message is not its next
frame but the time at which its action would change because of the
message's content. With the twin and the receiver's planned path, the
sender can compute that time: a warning about a region is needed by the
time the receiver's path makes the region decision-relevant, which is
seconds away for most regions and never for most of the map. Frame-
synchronous fusion, which the entire literature assumes, imposes a
100 ms deadline on data whose true deadline is seconds.

**The claim.** On an indoor team, the fraction of perception traffic
that has a sub-frame deadline is small. The 100 ms problem the field
works on is an artifact of fusing on a frame clock rather than on
decisions. The data that does have a 100 ms deadline is a few tokens
(states, warnings) that can be made nearly loss-free with priority
access, and the heavy traffic has a deadline set by motion.

**Why nobody has tried it.** Every collaborative-perception dataset is a
sequence of frames; the receiver's decisions are not in the data, so
decision-relevant deadlines cannot be computed. SCooP has the planned
paths and the map, and the Gym has the tasks.

**What could kill it.** If the people, not the map, are what matters,
and people's decision relevance is immediate, the fraction is not
small. The test decides.

**Test on the recordings.** For each frame of coop_2 and each labelled
object or occluded region, compute the time until the receiver's
recorded path would need it (time-to-reach the region's visibility or
collision cone minus reaction time). Report the distribution. If the
median is above 1 s, the claim holds and sizing becomes trivial for most
bytes; the paper then becomes idea 1 or 2 on the remainder.

---

## Which first

Idea 1 is the one no one has tried and the one only SCooP can test. Its
risk is sensitivity, and test 1.1 settles that in days. Idea 2 is the
cleanest theory and composes with idea 1 (the same P(lost | x)). Idea 3
is a reframing that could shrink the problem; run its test first because
it costs an afternoon and tells the other two how much traffic they are
fighting over.
