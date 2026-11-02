# Stretch 4: optimization, objective, execution

## Optimization or rules?

An optimization, solved exactly by enumeration. Each decision has a
small, discrete set of options (tens), so the robot evaluates every
option and takes the best. No hand-written rules, no solver, no RL.
The earlier "rules" (send from a better spot, then slow down) are what
the optimum turns out to be in typical cases.

The comms side supplies the one quantity the optimization cannot get
from profiling: the probability that a server request comes back in
time from where the robot is. That part is estimation (a predictor),
not optimization.

## The decision

At each phase k (deadline D_k, tolerance ε_k), the robot chooses an
option o:
- **onboard:** answer from the robot's model;
- **server (c, l, x'):** request form c (crop, resolution, codec), link
  l (5 GHz, 2.4 GHz, or both), sent from position x' (here, or a
  candidate stop point on the planned path).

The onboard model always runs anyway (cheap); a server answer that is
late is replaced by the onboard one.

## Objective

    maximize over o:   P_succ(o) − λ_b · b(o) − λ_t · Δt(o)

    P_succ(onboard)  = q_on
    P_succ(server o) = p(o) · q_srv(c) + (1 − p(o)) · q_on

| Symbol | Meaning | Comes from |
|---|---|---|
| p(o) = P[T(o) ≤ D_k] | server answer back before the deadline | **the comms predictor** |
| q_srv(c) = P[e_srv(c) ≤ ε_k] | server answer accurate enough, with form c | profiling on the rosbag (compute side) |
| q_on = P[e_on ≤ ε_k] | onboard answer accurate enough | profiling (compute side) |
| b(o) | airtime / bytes the request uses on the shared AP | request size and link rate |
| Δt(o) | time lost going to send point x' or slowing | motion planner |
| λ_b, λ_t | weights for airtime and time | chosen; swept in evaluation |

D_k = ε_k / v_k (tolerance over speed). Slowing to v' raises D_k and is
one more option with its own Δt.

Over a whole pick, phases are sequential and each decision depends only
on its own phase and position, so the pick objective is the sum of the
per-phase objectives.

## The predictor's objective

The predictor outputs p(o). It is good if it is calibrated: of the
requests given p = 0.9, about 90 % come back in time.

    minimize   (1/N) Σ_i ( p_i − 1[T_i ≤ D_i] )²      (Brier score)

T = uplink + server queue + inference + downlink.
Features: request size; link map at the send point (per-cell goodput
and RTT distributions); live RTT, goodput, RSSI; AP load (camera
network on/off, airtime); server queue; link l.
Model, in increasing complexity, stopping when calibration is good:
1. per-cell empirical delay quantiles scaled by size;
2. 1 corrected by the ratio of live to map goodput;
3. quantile regression (gradient boosting) on all features.

Baselines: RSSI threshold; mean RTT; static map only (Liu et al.
2026 style).

## Executing the contributions

**1. Instrumentation**
- chrony on robot and server.
- Every request carries an ID and timestamps: created, sent, server
  received, inference start, inference end, answer sent, answer
  received.
- Link logger: `iw dev <if> station dump` (RSSI, bitrate, retries),
  ping RTT, goodput; AP load from the AP's statistics or a sniffer.

**2. Probe campaign (the predictor's data)**
- Probe node on the robot sends requests of the real sizes (full
  image, crop, lidar box points, lidar scan) to a server echo service
  that waits the measured inference time and replies.
- Drive ED305 along varied paths; stop at the table and the weak spot.
- Repeat with the camera network off / on, and on 5 GHz, 2.4 GHz, both.
- Every probe logged with the robot's pose in the USD twin frame.
- Different days for training and testing.

**3. Link map**
- Per cell of the twin: distributions of goodput and RTT, per link,
  per load level.

**4. Predictor**
- Fit models 1–3; report the Brier score and calibration curves on the
  held-out days against the baselines.

**5. Transport knobs**
- Form: `image_transport` (JPEG / H.264), crop and resolution;
  `point_cloud_transport` (Draco) for lidar box points.
- Priority: DSCP marking on the request flow (DDS transport priority
  or `iptables`/`tc` marking) so the AP puts it in a WMM high class.
- DDS: fragment size, socket buffers, reliability per Lee et al.
  (2025).
- Dual link: two interfaces (5 GHz, 2.4 GHz); a critical request is
  sent on both; the server and robot drop duplicates by request ID.
- Measure each knob's effect on p at the table and the weak spot.

**6. Decision node**
- ROS 2 node on the robot: on each phase change or request, enumerate
  the options, compute the objective with p from the predictor and q
  from the profile tables, pick the best; onboard model runs in
  parallel as the fallback.

**7. Evaluation**
- Predictor: calibration and Brier score vs baselines.
- Knobs: deadline-miss rate with and without each.
- Picks: success, object error at the moment of action, airtime and
  time per pick, for the decision driven by each predictor.
