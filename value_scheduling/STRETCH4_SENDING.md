# Stretch 4: what to send, formulation, approaches, literature

## Formulation

Phases k = 1..K of the pick, each with
- a deadline D_k, from motion: D_k ≈ position tolerance / speed;
- an accuracy tolerance ε_k: how wrong the answer may be (e.g. object
  position within 10 cm at approach, within 1 cm before the grasp).

For each request i in phase k the robot chooses a configuration
c ∈ C: send or not, region (full / crop), resolution, codec and
quality, where it is computed (robot / server).

Delay of a request sent at position x:

    T(c, x) = t_enc(c) + S(c) / R(x) + q + t_inf(c) + t_ret

S(c) bytes, R(x) link throughput at position x (from the ED305 link
map), q server queue, t_inf inference, t_ret answer back.

Error e(c): how wrong the answer is with configuration c (missed
detection, 3-D position error), profiled offline on the rosbag.

**Problem**

    minimize    Σ_i S(c_i)                          (bytes on the link)
    subject to  P[ T(c_i, x_i) ≤ D_k ] ≥ 1 − δ      (deadline, with link uncertainty)
                e(c_i) ≤ ε_k                         (accuracy the phase needs)

Equivalent success form: maximize Σ_k P[answer k on time and within
ε_k] under a byte budget.

**Weak spots** enter through R(x). Where the robot sends from is also
a choice: send the big one-shot images (arrival, grasp) from where the
link is good, or before entering a weak spot; downgrade the form when
the request has to be sent from a weak spot.

**Complexity.** C is small and discrete (tens of configurations), so
each request is a lookup: the cheapest c whose profiled error is within
ε_k and whose delay at the current (or predicted) R(x) fits D_k. No
optimizer, no training.

## Approaches, best fit first

| Approach | Idea | Fits | Key literature |
|---|---|---|---|
| Profile and select | profile accuracy, size, delay per configuration offline; pick the cheapest that fits at run time | the lookup above | Chameleon (SIGCOMM 2018); AWStream (SIGCOMM 2018) |
| Server-driven regions | send low quality first, server returns where to look, robot sends those regions at high quality | crop after the object is found | DDS, Du et al. (SIGCOMM 2020); EAAR, Liu et al. (MobiCom 2019) |
| Filter and track on the device | track locally between server answers; send only when the scene changed or tracking is lost | search and approach | Glimpse, Chen et al. (SenSys 2015); Reducto, Li et al. (SIGCOMM 2020) |
| Where to compute | robot vs server per frame or per model layer, from link and load | localization placement, YOLO placement | Neurosurgeon, Kang et al. (ASPLOS 2017); Chinchali et al. (RSS 2019); FogROS2, Ichnowski et al. (ICRA 2023) |
| Accuracy-aware standard encoding | a small model chooses quality per image block; codec stays H.264 | if crops are still too big at weak spots | AccMPEG, Du et al. (MLSys 2022) |
| Learned features / split computing | send intermediate features or a learned code instead of pixels | only if all of the above fails | Shao et al., IB edge inference (JSAC 2021); Matsubara et al., survey (ACM CSUR 2022); Duan et al., video coding for machines (TIP 2020) |
| Freshness at the decision | score whether the answer was right when the robot acted | the measurement in step 5 | AoI, Kaul et al. (INFOCOM 2012); AoII, Maatouk et al. (ToN 2020) |

## Do we train encoders?

Not at first.
- The largest savings come from when, where and what (10–100×);
  a trained codec adds 2–5× on top of JPEG/H.264 with crops.
- Trained codecs and split computing need YOLO retrained or cut, which
  changes 廖恩莆's pipeline, and need a GPU encoder on the robot.
- Middle ground if needed: train only a small "which blocks matter"
  model and keep the standard codec (AccMPEG style).

Train only if the profiling shows that at the weak spots even the
smallest standard configuration misses the deadline at the needed
accuracy.

## What the literature covers and what it does not

Covered: video analytics accuracy vs bandwidth (profiling, regions,
filtering), robot-vs-cloud offloading, learned features for one model.
Not covered as one problem: a robot whose deadlines and accuracy
tolerances change by phase of the task, and whose link depends on
where it is in the room.

## The contribution: the sending policy

Measurement, the link map and placement are inputs the policy uses,
not contributions on their own.

**Inputs per request.** Phase k from the stage machine (gives D_k and
ε_k); robot position and planned path, with the link expected along it
(link map + live estimate); the server's reported queue; the profile
table (bytes, error per phase, encode and inference time for each
configuration).

**Decision.** The configuration c (skip / full / crop, resolution,
quality, codec) and the moment and place to send it.

**Rule.**
1. Pick the cheapest c with e_k(c) ≤ ε_k and P[T(c, x) ≤ D_k] ≥ 1 − δ.
2. If none fits here, send earlier from a point on the path where the
   link is good enough, if there is one before the deadline.
3. If still none fits, slow the robot: D_k = tolerance / speed, so a
   lower speed extends the deadline until a configuration fits.
Server side: serve requests by deadline, drop those whose phase has
ended.

**Baselines.** Current raw stream; continuous JPEG; on-request fixed
form; link-adaptive only (bandwidth-driven, phase-blind, AWStream
style). Ablations: without position (live link only), without phase
(one deadline for all).

**Conditions.** Good link; a weak spot at the table (AP placement or
`tc` throttling); background traffic from the camera network.

**Metrics.** Pick success; answers correct and in time at the robot's
action, per phase; deadline hit rate; bytes per pick; robot time per
pick (the cost of slowing).
