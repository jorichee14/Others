# Communication-aware coarse-to-fine pick on the Stretch 4: architecture and approach

## Architecture

```
ROBOT (Stretch 4)                                                    SERVER
────────────────────────────────────────────                         ─────────────────────────────
sensors: 2× 3-D lidar, IMU, central RGB,                             RECEIVER / SCHEDULER
fisheye RGB, joint encoders, gripper                                 - corrections first while base moves
        │                                                            - YOLO requests by deadline
        ▼                                                            - drop requests whose phase ended
┌──────────────── COMMS MANAGER ─────────────────┐                   - reports queue + inference time ──┐
│ link estimator: live RTT/goodput/RSSI          │                          │                           │
│   + link map lookup (now, along path)          │     Wi-Fi AP             ▼                           │
│ task model: phase → tolerances, speed, deadline│   (priority     2Fast-2Lamaa      YOLO               │
│ profile tables: config → error, bytes, time    │    classes)     on the map         (boxes)           │
│ SELECTOR (the policy):                         │                 (built / USD twin)                    │
│   lidar config, image config, send point,     │──── lidar L ──▶ │                  │                    │
│   speed limit                                  │──── image V ──▶ │                  │                    │
│ encoders: voxel + Draco; resize/crop + JPEG/H.264                │                  │                    │
│ tagger: capture stamp, pose at capture, phase  │◀── map→odom ────┘                  │                    │
└────────────────────────────────────────────────┘◀── box + stamp ───────────────────┘                    │
        ▲                    │                   ◀────────────────── server load feedback ──────────────┘
        │                    ▼
STAGE MACHINE ──phase──▶ ANSWER HANDLER                              DIGITAL TWIN (server)
(find, drive, arrive,    - pose = odometry ⊕ latest correction       - USD twin + built map
 plan, align, close,     - box → 3-D with local depth and the        - link map of ED305
 verify)                   pose at capture                           - ground truth from the camera
        │                - discard answers past their phase            network (evaluation)
        ▼
LOCAL LOOPS: odometry (full rate), fine alignment, grasp, gripper effort
```

## Approach

**1. Value information at the moment of action.** What matters is
how wrong the robot's pose or target is when it acts, not frame rate
or latency. Each phase k has a tolerance (ε_k^L for pose, ε_k^V for the
object) and a speed v_k; together they give a deadline D_k = ε / v_k.

**2. Split the task by what each part can tolerate.**
- Coarse (find, drive, locate): heavy models, tolerates hundreds of
  ms → server.
- Fine (align, close): millimetres in tens of ms → robot.
- The handover at the table is the one message that matters most: the
  arrival image and scan.

**3. Make late answers usable instead of fighting delay.**
- Every message carries its capture stamp and the pose at capture;
  answers come back in the map frame.
- Localization returns a map→odom correction; odometry runs locally
  at full rate, so a late correction costs only the drift
  ρ · v · age.
- Answers whose phase has ended are discarded.

**4. Let the phase decide which flow gets the link.**

The lidar has two jobs: the robot's pose (2Fast-2Lamaa, server) and
the object's 3-D position (lidar points inside the YOLO box, since the
Stretch 4 has no depth camera listed). The second job continues in the
fine phases, but it needs only the points inside the box, not the
scan, and can be done on the robot.

| Phase | Base | Lidar for robot pose (up) | Lidar for object position | Images (YOLO) |
|---|---|---|---|---|
| Find, drive | moving | keyframes, priority | — | low-rate video / off |
| Arrive | stopping | one high-quality scan, then pause | box points, first estimate | one full image |
| Plan grasp | still | paused | box points (on the robot, or to the server if it plans the grasp) | one crop |
| Align, close | still | paused | box points on the robot, every scan | nothing |
| Verify, leave | still → moving | resumes | box points once | one crop |

Box points: lidar points that project into the YOLO box (camera–lidar
extrinsics), a few hundred to a few thousand points; tens of KB with
Draco if they must cross the link. Check that the lidar's range noise
and point spacing at 0.3–0.8 m are finer than the grasp tolerance; if
not, the final centimetres come from the camera or the gripper.

**5. Decide each request by lookup, in this order.**
1. Phase → tolerances, speed, deadline.
2. Link at the robot's position (live + link map).
3. Cheapest lidar and image configuration that meets both tolerances
   and fits the link together:
   - pose: e_loc(c_L) + ρ · v · A_L(c_L, x) ≤ ε^L
   - object: e_V(c_V) ≤ ε^V and P[T_V(c_V, x) ≤ D] ≥ 1 − δ
   - rate(c_L) + rate(c_V) ≤ R(x)
4. Nothing fits → send from a better point (stop point or before a
   weak spot, from the link map).
5. Still nothing → lower the speed (longer deadline, less drift).

No training: the tables come from profiling the rosbag offline.

## Where each part comes from in the literature

| Part | Established by |
|---|---|
| Coarse on server, fine on robot | OK-Robot (RSS 2024, Stretch); Edge-SLAM; Chinchali et al. (RSS 2019); Johns, coarse-to-fine (ICRA 2021) |
| Selector over profiled configurations | Chameleon, AWStream (SIGCOMM 2018); DDS (SIGCOMM 2020); AdaptSLAM |
| Lidar compression for offloaded SLAM | Draco / KISS-SLAM over 5G (2025); FLiCR |
| Send point from a link map | Liu et al. (arXiv 2606.31497, 2026); Gordon et al. (INFOCOM 2026 workshop); RCAMP (IROS 2017) |
| Delay-tolerant answers | SPO (2026); latency-tolerant cloud–edge VLA (2026); TIC-VLA (2026) |
| Scoring at the action | AoI (INFOCOM 2012); AoII (ToN 2020) |

What is new is the combination the selector makes: the phase of the
pick sets each flow's tolerance and deadline, and the robot's position
sets what the link can carry, for both the localization and the
perception flow at once.
