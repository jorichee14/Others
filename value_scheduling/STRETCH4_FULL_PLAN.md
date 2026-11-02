# Stretch 4 in ED305: from models to communication-aware optimization

## 1. Hardware and where things can run

| Place | Hardware | Runs |
|---|---|---|
| Robot | Intel NUC 15 (CPU), Jetson Orin NX (GPU) | control, odometry, Nav2, small models, the decision node |
| Network | ED305 Wi-Fi AP (5 GHz + 2.4 GHz, dual link) | robot ↔ server; the camera network shares it if wireless |
| Server | GPU workstation | heavy models; the ED305 camera streams; USD twin |

Sensors: two hemispherical 3-D lidars, IMU, central high-resolution RGB
(gripper workspace), two fisheye RGB; no depth camera, so 3-D object
geometry comes from lidar points projected into the image.

## 2. Models per phase: onboard option vs server option

| Phase | Needs | Onboard option | Server option | Up the link if server |
|---|---|---|---|---|
| Localization (all moving phases) | robot pose, ε^L ~ 5–10 cm | 2Fast-2Lamaa on the NUC, on the built map or USD twin point cloud | 2Fast-2Lamaa on the server | lidar keyframes (Draco), IMU |
| 1 Find the goal | which table / where the object is, from a language query | closed-vocabulary YOLO (TensorRT, Orin NX) | open-vocabulary detector (e.g. Grounding DINO / OWLv2) on the robot image, or on the ED305 camera streams | robot image (or nothing, if the room cameras answer) |
| 2 Drive | path, obstacles | Nav2 on lidar, local | — | — |
| 3 Arrive, locate | object box and 3-D position, ε ~ 2–3 cm | YOLO + lidar points in the box → centroid | open-vocabulary segmentation (e.g. Grounded-SAM 2) mask + lidar points | one full image (+ lidar box points) |
| 4 Plan grasp | grasp pose, ε ~ 1 cm | heuristic grasp from the lidar points (centroid + principal axis, top or side) | learned grasp generator (e.g. AnyGrasp as in OK-Robot, Contact-GraspNet) | crop + object points |
| 5 Align | gripper on target, mm, tens of ms | box tracking on the central camera + lidar points, local servo | — (deadline too short) | — |
| 6 Close | contact | gripper effort, local | — | — |
| 7 Verify | holding or not | gripper aperture/effort + image check | VLM check | one crop |

Check before relying on the server grasp option: learned grasp
generators expect dense depth; lidar points on a small object may be
too sparse.

## 3. Comms: what the optimization needs from the link

For any server option o (request form c, link l, sent from position
x'), the probability that its answer is back before the phase deadline:

    p(o) = P[ T_up + T_queue + T_inf + T_down ≤ D_k ]

**Predictor.** Features: request size; link map at x' (per-cell
goodput and RTT distributions in the USD twin); live RTT, goodput,
RSSI; AP load (camera network on/off, airtime); server queue; link l.
Models, simplest first: per-cell delay quantiles scaled by size → plus
live/map goodput correction → quantile regression (gradient boosting).
Judged by calibration (Brier score) on held-out days; baselines RSSI
threshold, mean RTT, static map only (Liu et al. 2026).

**Knobs that raise p.** Request form (crop, resolution, JPEG/H.264,
Draco for points); priority (DSCP → WMM high class); DDS settings for
large messages (Lee et al. 2025); dual link (send on 5 GHz and 2.4 GHz,
take the first answer).

## 4. The optimization

At each phase k (tolerance ε_k, speed v_k, deadline D_k = ε_k / v_k),
choose an option o from: onboard; server (c, l, x') for each form, link
and candidate send point; or a lower speed.

    maximize over o:   P_succ(o) − λ_b · b(o) − λ_t · Δt(o)

    P_succ(onboard)  = q_on
    P_succ(server o) = p(o) · q_srv(c) + (1 − p(o)) · q_on

- q_on, q_srv(c): probability the answer is within ε_k, profiled
  offline per phase.
- p(o): from the comms predictor.
- b(o): airtime on the shared AP; Δt(o): time to reach x' or lost by
  slowing.
- The onboard model always runs; a late server answer is replaced by
  the onboard one.

**Localization** is decided per drive segment with the same form:
q = P[pose error ≤ ε^L], pose error = e_loc + ρ · v · age of the last
correction, where age depends on p for the server option.

**Solved by enumeration** (tens of options per decision); exact, no
solver, no training of the decision.

## 5. Software architecture

```
ROBOT                                                          SERVER
stage machine ─phase─▶ DECISION NODE ◀── profile tables (q)    receiver: deadline queue,
                        │   ▲                                  drop stale, report queue
                        │   └── PREDICTOR (p) ◀── link logger,          │
                        │        link map, AP load, server queue        ▼
                        ▼                                       heavy models: open-vocab
            onboard models (always on)     request sender ──▶   detection, segmentation,
            YOLO, lidar-box points,        (form, priority,      grasp, VLM; 2Fast-2Lamaa
            heuristic grasp, servo,         dual link)           (option)
            2Fast-2Lamaa                        ◀── answer + stamps
                        │
            answer handler: pick in-time server answer or onboard answer;
            answers in the map frame (pose at capture)
```

## 6. Execution, in order

1. **Bring up both sides.** Onboard: YOLO (TensorRT), lidar–camera
   projection, heuristic grasp, 2Fast-2Lamaa. Server: open-vocabulary
   detector, segmentation, grasp generator. Same input/output messages
   for both (box / mask / grasp + capture stamp).
2. **Profile accuracy (q).** On the rosbag and ground truth from the
   camera network: each model's error per phase; each request form's
   effect on the server models' error. Gives q_on, q_srv(c).
3. **Instrument and probe (p).** chrony; request IDs with timestamps at
   every step; link logger; probe node sending real-size requests to a
   server echo service, over ED305, camera network on/off, each link;
   different days for training and testing.
4. **Link map and predictor.** Per-cell distributions in the twin; fit
   the predictor; check calibration.
5. **Knobs.** Form, priority, DDS, dual link; measure their effect on p.
6. **Decision node.** Enumerate options, compute the objective, act;
   answer handler with onboard fallback.
7. **Evaluate.**

## 7. Evaluation

| Method | Description |
|---|---|
| Always onboard | robot models only |
| Always server (today) | every phase to the server, raw images |
| Always server, compressed | fixed form, no decision |
| Static-map placement | send point from a fixed map, fixed form (Liu et al. 2026 style) |
| Ours | per-phase onboard/server choice with the calibrated predictor and knobs |

Conditions: good link; weak spot at the table; camera network loaded.
Metrics: pick success; object and pose error at the moment of each
action; deadline-miss rate; airtime per pick; time per pick; predictor
calibration.

## 8. Kill tests

1. Onboard models within tolerance in every phase → no offloading
   problem for this task.
2. Server models not clearly better where it matters → same.
3. Server answers never late in ED305, even loaded → the predictor has
   nothing to predict; create the weak spot or load, or go multi-robot.

## 9. Roles

| Who | Part |
|---|---|
| 廖恩莆 | pipeline, stage machine, onboard and server models (steps 1–2) |
| Prof. Yeh | compute placement, server and AMD PC |
| Jorich | comms: instrumentation, probes, link map, predictor, knobs, decision node, evaluation (steps 3–7) |
