# Stretch 4: the sending policy (2Fast-2Lamaa and YOLO both on the server)

## Setup assumed

Both heavy jobs run on the server:
- **L, localization:** 2Fast-2Lamaa. Up: lidar scans (two hemispherical
  3-D lidars) and IMU. Down: map→odom correction. The robot keeps
  wheel/IMU odometry locally at full rate and applies corrections when
  they arrive.
- **V, perception:** YOLO. Up: images (central RGB camera). Down: boxes;
  the robot turns a box into 3-D with its own depth.

Fine alignment and closing run on the robot (arm encoders, wrist
camera, gripper effort).

## How the literature does communication-aware robotics

| Question | How it is done | Work |
|---|---|---|
| Which loops stay on the robot | fast, fine loops local; heavy, slower jobs offloaded | Edge-SLAM, Ben Ali et al. (ACM TECS 2022): tracking local, mapping on the edge; Chinchali et al. (RSS 2019): query the cloud only when the local model is unsure; Pohland et al., *Offload or Overload* (MSR-TR-2026-14, incl. Stretch 3): tens of ms of added latency lower manipulation accuracy by more than 10 % |
| What form is sent | adapt resolution / quality / region to bandwidth, profiled offline | Chameleon, AWStream (SIGCOMM 2018); DDS (SIGCOMM 2020); EAAR (MobiCom 2019); AdaptSLAM (adapts SLAM data to bandwidth and compute) |
| Lidar over the link | compress scans (Draco, range images); fixed settings give a fixed, high bitrate | 5G-edge KISS-SLAM over L4S with Draco (arXiv 2511.15677, 2025): runs in real time but bitrate is not adaptive, authors point to adaptive encoding; FLiCR (2023); D-Compress (2026) |
| Where to send from | a communication map tells the robot where a request will make it back in time | Liu, Ohsita, Murata, Shimonishi (arXiv 2606.31497, 2026): request–response window, choose the point for the next cloud request, simulation with a known map; Gordon et al. (INFOCOM 2026 NetRobiCS): radio map → QoS risk costmap for path planning; RCAMP, Caccamo et al. (IROS 2017); Rahman et al. (GLOBECOM 2017): offload decision, path and AP chosen jointly |
| Hiding the delay | buffer or predict ahead; send redundantly | Speculative Policy Orchestration (arXiv 2603.19418, 2026): cloud precomputes waypoints into a local buffer; latency-tolerant cloud–edge VLA (arXiv 2608.00569, 2026); FogROS2-PLR (2024): choose routes to minimise deadline-miss probability; FogROS2 (ICRA 2023): H.264 for image topics |
| Speed and delay | slow down as delay grows; stop when updates stop | mostly teleoperation practice and patents; industrial AMR work keeps safety onboard and widens margins with delay |
| Scoring freshness | age of the information, or of wrong information, when used | AoI, Kaul et al. (INFOCOM 2012); AoII, Maatouk et al. (ToN 2020) |
| Offloading over Wi-Fi in practice | offload gain can be cancelled by link delay | Computation offloading for ground robots over Wi-Fi (Empirical Software Engineering, 2023) |

## Formulation

Phase k has a base/arm speed v_k and tolerances: ε_k^L for the robot's
pose, ε_k^V for the object's position.

**Localization (L).** Configuration c_L: scan rate, which lidar(s),
voxel size, Draco quantization. Pose error when the robot acts:

    e_L = e_loc(c_L) + ρ · v_k · A_L(c_L, x)

e_loc: 2Fast-2Lamaa error with that configuration (profiled on the
bag); ρ: odometry drift per metre (measured); A_L: age of the latest
correction = scan interval + uplink + server + downlink.
Constraint: e_L ≤ ε_k^L. When the base is still (v_k = 0), odometry does
not drift and the lidar can pause.

**Perception (V).** Configuration c_V: skip / full / crop, resolution,
codec, quality. Deadline D_k = ε_k^V / v_k. Constraints:
P[T_V(c_V, x) ≤ D_k] ≥ 1 − δ and e_V(c_V) ≤ ε_k^V.

**Shared link.** rate(c_L) + rate(c_V) ≤ R(x), R(x) from the ED305 link
map and the live estimate.

**Objective.** Minimize bytes on the link (equivalently, maximize
answers correct and in time when the robot acts), choosing c_L, c_V and,
for one-shot requests, the place x they are sent from; the speed v_k is
the last resort.

## What this gives per phase: the two flows take turns

| Phase | Base | L (lidar up) | V (images up) |
|---|---|---|---|
| Find the goal | moving | keyframes, downsampled + Draco, priority | low-rate video, yields to L |
| Drive | moving | keyframes, priority | off or very low |
| Arrive at the table | stopping | one high-quality scan (base pose sets the arm's reach), then pause | one full image, sent from the stop point |
| Plan the grasp | still | paused | one crop |
| Fine alignment, close | still | paused | nothing (local) |
| Verify, leave | still → moving | resumes when the base moves | one crop |

While the base moves, the lidar needs the link; while it is still, the
images do. The robot's phase decides which flow gets it.

## The policy

1. From the phase: tolerances, speed, deadline.
2. Pick the cheapest (c_L, c_V) that meets both error constraints and
   fits the link at the robot's position.
3. If none fits: send the one-shot image from a point on the path where
   the link is good (stop point chosen with the link map), or before
   entering a weak spot.
4. If still none fits: lower the speed; this extends D_k and reduces
   odometry drift between corrections.
5. Server: corrections first while the base moves, YOLO requests by
   deadline, drop requests whose phase has ended.

Each step is a lookup over a profiled table (tens of configurations
per flow); no training.

## Where it stands against the literature

Adaptive streaming chooses the form of one video stream for one
model; SLAM offloading compresses or splits one localization stream;
Liu et al. (2026) choose where to send the next request from a
communication map, in simulation. The policy here chooses the form of
both the localization and the perception streams and the place of the
one-shot requests, per phase of a real pick, with the phase setting
each flow's tolerance and deadline.

## Evaluation

- **Baselines:** current raw streams; fixed compression for both flows;
  bandwidth-adaptive only (phase-blind); communication-map placement
  only (Liu et al. style, fixed form).
- **Ablations:** no phase (one deadline); no position (live link only);
  no speed step.
- **Conditions:** good link; weak spot at the table (AP placement or
  `tc`); background traffic from the camera network.
- **Metrics:** pick success; pose error and object error at the moment
  of each action; deadline hit rate per phase; bytes per pick; time
  per pick.

## Note on sources

The 2026 arXiv entries were checked from their abstracts and search
summaries; the full PDFs could not be opened from this session
(arxiv.org is not in its allowed network domains). Read them before
citing details.
