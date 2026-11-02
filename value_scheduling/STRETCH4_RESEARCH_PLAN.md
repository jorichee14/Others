# Communication-aware coarse-to-fine mobile manipulation: research plan

## The idea in one paragraph

A pick is coarse-to-fine: drive and locate (server), hand over at the
table, align and grasp (robot). Each phase needs something different
from the link: while the base moves, a fresh pose (lidar up,
corrections down); at the handover, one accurate image and scan; during
the fine phase, nothing. What the link can deliver depends on where the
robot is in the room. The policy matches the two: the phase sets the
deadline and accuracy each flow must meet, the position sets what the
link can carry, and the robot chooses the form of each flow, where to
send the one-shot requests from, and, last, its speed.

**Research question.** How should a mobile manipulator decide what to
send, when and from where, so that each phase of a coarse-to-fine task
gets information that is accurate enough and in time over a shared,
position-dependent wireless link?

## Hypotheses

- **H1.** Phase-aware sending cuts bytes per pick by an order of
  magnitude at equal pick success, compared with today's streams.
- **H2.** With a weak link at the table, adding position (where to send
  from, speed) keeps pick success where phase-aware-only and
  bandwidth-adaptive sending drop.
- **H3.** The error of the robot's information at the moment it acts
  (pose error, object error) predicts pick failure better than latency
  or bandwidth.

## Stage 1. Measure (weeks 1–2)

Actions:
1. Tuesday rosbag: lidars, IMU, cameras, odometry, joint states;
   recorded on the robot's disk.
2. chrony on robot and server; timestamps at every hop.
3. Link survey of ED305 in the USD twin frame: RSSI, RTT, throughput
   with pose (link map).
4. Ground truth from the camera network (AprilTag board on the robot).
5. 10 baseline picks with the current pipeline (raw images, 2Fast-2Lamaa
   on the server with raw lidar).

Output: per phase, delay split into link / queue / inference; bytes;
the link map.
Gate: if the ED305 link never limits the pick, create the weak spot
(AP placement or `tc`) and background load (camera network); the
study is about the condition, not one lucky room.

## Stage 2. Model the task (weeks 2–4)

Actions:
1. **Tolerances per phase.** Offset the grasp target and the base pose
   on purpose and record success; the largest offset that still
   succeeds is ε_k (object) and ε_k^L (base).
2. **Odometry drift ρ** of the Stretch 4 base per metre, against ground
   truth.
3. **Profile tables** on the rosbag, offline:
   - lidar configurations (rate, one/two lidars, voxel, Draco level) →
     2Fast-2Lamaa error, bytes, time;
   - image configurations (full / crop, resolution, JPEG / H.264
     quality) → YOLO miss rate and 3-D error, bytes, time.

Output: the tolerances, ρ, and the two tables. This is the task model
the policy reads.

## Stage 3. Policy v1: phase-aware (weeks 4–6)

Actions:
1. Robot node: reads the phase from the stage machine; for each flow
   picks the cheapest configuration that meets ε_k and the deadline at
   the live link estimate; lidar pauses when the base is still.
2. Server node: corrections first while the base moves; YOLO requests
   by deadline; drop requests whose phase ended.
3. Interface with 廖恩莆's pipeline: phase topic in, box + capture stamp
   out; nothing in YOLO or 2Fast-2Lamaa changes.
4. Evaluate on bag replay over the real Wi-Fi, then live picks.

Output: H1.

## Stage 4. Policy v2: position-aware (weeks 6–8)

Actions:
1. Use the link map: choose the stop point at the table, and the send
   point of the arrival image and scan, where the link is good enough;
   send before entering a weak spot.
2. Last step: lower speed when nothing fits (longer deadline, less
   drift between corrections).
3. Evaluate with the weak spot at the table.

Output: H2.

## Stage 5. Campaign and paper (weeks 8–10)

Design: 3 conditions (good link; weak link at the table; background
traffic) × 5 methods × 20 picks.

| Method | What it is |
|---|---|
| Current | raw images, raw lidar |
| Fixed compression | JPEG / H.264 and Draco, fixed settings |
| Bandwidth-adaptive | adapts to the link, ignores the phase (AWStream-style) |
| Placement-only | chooses the send point from the link map, fixed form (Liu et al. 2026 style) |
| Ours | phase-aware form + position-aware send point + speed |

Metrics: pick success; pose and object error at each action; deadline
hit rate per phase; bytes per pick; time per pick.

Analysis: success against error-at-action, latency and bandwidth
(H3).

Output: the paper. Target: RA-L / ICRA or IROS (robotics side), or
IEEE IoT-J / an INFOCOM NetRobiCS-type workshop (networking side).

## Contributions

1. The formulation: per-phase deadline and accuracy for both the
   localization and the perception flow, over a position-dependent
   link, with speed as the coupling.
2. The policy: phase- and position-aware sending, as ROS 2 nodes, no
   training, no change to the perception or localization models.
3. The evidence: real coarse-to-fine picks with the Stretch 4 in ED305,
   with ground truth, against the four baselines.

## Roles

| Who | What |
|---|---|
| Jorich | stages 1–5: measurement, task model, policy nodes, experiments, paper |
| 廖恩莆 | pipeline, stage machine with the phase topic, robot sessions |
| Prof. Yeh | compute side (where models run, the AMD PC), co-author |

## Risks and fallbacks

- **The link is never the bottleneck in ED305:** use the weak spot and
  background load (stage 1 gate).
- **2Fast-2Lamaa does not run well with two lidars or on the server:**
  one lidar, or KISS-ICP / KISS-SLAM as the localization stand-in; the
  policy does not depend on which localizer runs.
- **The pipeline changes during the study:** freeze the interface
  (phase topic in; box, pose and capture stamp out).
