# Communication-aware offloading for coarse-to-fine mobile manipulation (Stretch 4, ED305)

## 1. Problem

A Stretch 4 picks an object in ED305: find it, drive to it, locate it,
plan the grasp, align, close, verify. Each phase can be answered by a
small model on the robot (fast, independent of the link, less accurate)
or a heavy model on the server (accurate, slow, dependent on the Wi-Fi
at the robot's position). As the task goes from coarse to fine, the
accuracy needed rises while the time available shrinks. The robot must
decide, per phase, which model answers, and for the server, what form
of request to send and from where, so that the answer is accurate
enough and arrives before it acts.

## 2. System model

**Entities.**
- Robot: Intel NUC 15 + Jetson Orin NX; two 3-D lidars, IMU, central
  RGB (gripper workspace), two fisheye RGB; no depth camera (object
  3-D from lidar points projected into the image).
- Server: GPU workstation; also receives the ED305 camera streams.
- Link: one Wi-Fi AP, 5 GHz, shared with the camera network.
- Workspace W: ED305, in the USD twin frame. Robot position x(t) on
  planned path π.
- Transport: ROS 2 (DDS) topics: request topic and answer topic matched
  by request ID; best effort, keep last 1 for requests and streams.

**Task.** Phases k = 1..K, each with
- tolerance ε_k (object position ε_k^V; robot pose ε_k^L);
- deadline D_k: ε_k / v_k while moving at speed v_k; the dwell budget
  τ_k (how long the robot will wait before acting) while still.

**Models.** For each phase, an onboard model m_k^on and a server model
m_k^srv, with error e and inference time t_inf. Accuracy:
q_on,k = P[e_on ≤ ε_k], q_srv,k(c) = P[e_srv(c) ≤ ε_k] for request form
c (crop, resolution, codec, Draco level), measured offline.

**Request.** Form c with size S(c), sent at position x'. Response time

    T(c, x', t) = S(c) / G_up(x', t) + d_acc(x', t) + Q(t) + t_inf^srv + S_ans / G_down(x', t)

G: goodput at the position and time (depends on distance to AP,
obstacles, people, and load from other stations); d_acc: channel access
and retransmission delay; Q: server queue.

**Predictor.** p(c, x') = P̂[T(c, x', t) ≤ D_k | features]; features:
S(c), link map at x' (per-cell goodput and RTT distributions in the
twin), live RTT / goodput / RSSI, AP load, server queue.

**Localization.** Odometry on the robot; 2Fast-2Lamaa gives map→odom
corrections (onboard by default; server as an option). Pose error at
action e_L = e_loc + ρ · v · A, A = age of the last correction, ρ =
odometry drift per metre.

**Decision per phase.** a_k ∈ {onboard} ∪ {server(c, x') : c ∈ C_k,
x' ∈ candidate send points on π before the deadline point} ∪
{slower speed v' < v_k}. The onboard model always runs; a late server
answer is replaced by the onboard one.

## 3. Formulation

Per phase k:

    maximize over a:   P_succ(a) − λ_b · b(a) − λ_t · Δt(a)

    P_succ(onboard)       = q_on,k
    P_succ(server(c, x')) = p(c, x') · q_srv,k(c) + (1 − p(c, x')) · q_on,k

b(a): airtime on the shared AP; Δt(a): time to reach x' or lost by
slowing (slowing raises D_k). Localization is decided per drive segment
with q = P[e_L ≤ ε^L].

Over a pick, phases are sequential and each decision depends on its
own phase and position, so the pick objective is the sum over phases.
Each decision has tens of options and is solved exactly by
enumeration.

Predictor objective (calibration): minimize the Brier score
(1/N) Σ (p_i − 1[T_i ≤ D_i])² on held-out days.

**Metric of the task.** Error of the information the robot acts on, at
the moment it acts (object error, pose error), against ground truth
from the ED305 camera network: the age-of-incorrect-information view,
measured.

## 4. Approach

| Part | Approach | Concretely |
|---|---|---|
| Will it arrive in time? | radio-map-based latency prediction, calibrated | link map of ED305 in the USD twin + live correction → p(c, x') |
| What to send | content adaptation | crop, resolution, JPEG / H.264; Draco for lidar points |
| How it is carried | QoS + middleware | DSCP → WMM high class for requests; DDS fragment size, buffers, reliability |
| Whether to send | expected-success offloading with onboard fallback | enumeration of the options above |

Models per phase:

| Phase | Onboard | Server |
|---|---|---|
| Localization | 2Fast-2Lamaa on the NUC | 2Fast-2Lamaa on the server |
| Find the goal | closed-vocabulary YOLO (Orin NX) | open-vocabulary detector on the robot image or the room cameras |
| Drive | Nav2 | — |
| Arrive, locate | YOLO + lidar points in the box | open-vocabulary segmentation + lidar points |
| Plan grasp | heuristic grasp from lidar points | learned grasp generator (check: needs dense enough points) |
| Align, close | box tracking + gripper effort | — |
| Verify | gripper aperture / effort + image | VLM check |

## 5. Novelty

**Claimed.**
1. **The problem.** Per-phase choice between the robot's and the
   server's model in coarse-to-fine mobile manipulation, where the
   accuracy needed rises as the deadline shrinks, the deadline comes
   from tolerance and speed, and the link depends on the robot's
   position; with the onboard answer as a fallback, so offloading can
   only add success.
2. **The predictor.** A calibrated probability that a request of a given
   size, sent from a given position, returns before its deadline, on a
   real building's Wi-Fi under real load, built in the digital twin's
   frame and corrected online.
3. **The evidence.** Real picks with ground truth, scored by the error
   of the information at the moment of action.

**Against the closest work.**

| Work | What it does | What it leaves |
|---|---|---|
| Liu et al. (arXiv 2606.31497, 2026) | communication map → where to send the next cloud request | map known in advance; no onboard alternative; one request type; simulation; authors list load, people and deployed Wi-Fi as open |
| Chinchali et al. (RSS 2019) | offload when the local model is uncertain | no phases, no position, no deadline from motion |
| Pohland et al. (MSR-TR-2026-14) | measure onboard / edge / cloud for mobile manipulation | no per-phase decision |
| AWStream, Chameleon, DDS | adapt the form of a video stream to bandwidth | no robot, no onboard alternative |
| SPO, TIC-VLA, latency-tolerant VLA | hide delay inside the policy by training | policy changed; here the models are unchanged |
| FogROS2-PLR | minimise deadline-miss probability across network routes | no position, no task phase |

**Not claimed.** Compression, QoS marking, DDS tuning, slowing down:
used, not new.

## 6. Execution

1. **Both sides up.** Onboard and server models with the same message
   interface (box / mask / grasp + request ID and capture stamp).
2. **Accuracy profiles (q).** Each model and request form, per phase, on
   the rosbag, against camera-network ground truth.
3. **Instrumentation.** chrony; timestamps at every hop; link logger
   (`iw station dump`, RTT, goodput); AP load.
4. **Probe campaign (p).** Real-size requests to a server echo service,
   across ED305, camera network on and off, several days.
5. **Link map and predictor.** Per-cell distributions in the twin;
   models from per-cell quantiles to quantile regression; calibration
   on held-out days.
6. **Transport settings.** Form, DSCP, DDS; effect on p at the table and
   the weak spot.
7. **Decision node.** Enumerates options, picks the best, onboard
   fallback in the answer handler.
8. **Evaluation.**

## 7. Evaluation

| Method | Description |
|---|---|
| Always onboard | robot models only |
| Always server (today) | raw images to the server every phase |
| Always server, compressed | fixed form, no decision |
| Static-map placement | send point from a fixed map, fixed form (Liu et al. style) |
| Decision with simple predictors | RSSI threshold; mean RTT |
| Ours | calibrated predictor + per-phase decision + transport settings |

Conditions: good link; weak spot at the table; camera network loaded.
Metrics: pick success; object and pose error at the moment of each
action; deadline-miss rate; airtime per pick; time per pick; predictor
calibration.

## 8. Kill tests

1. Onboard models within tolerance in every phase → no offloading
   problem for this task.
2. Server models not clearly better where it matters → same.
3. Server answers never late in ED305, even loaded → nothing to
   predict; create the weak spot or load, or go multi-robot.

## 9. Roles

| Who | Part |
|---|---|
| 廖恩莆 | pipeline, stage machine, onboard and server models (1–2) |
| Prof. Yeh | compute placement, server, AMD PC |
| Jorich | instrumentation, probes, link map, predictor, transport, decision node, evaluation (3–8) |
