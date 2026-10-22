# Dynamic digital twin: data flow, robot (Jetson Orin) vs server

Which block of the dynamic digital twin (slide 3/18) runs on each robot's
Jetson Orin and which on the server ("cloud": the lab GPU server, or a cloud
machine with an RTX GPU), and what goes over the network between them.

Rule that decides the split: **on the robot, only what must keep running when
the network drops, or what would cost too much bandwidth to send raw.**
Everything that needs a big GPU or more than one robot's data runs on the server.

```
 ROBOT i  (Jetson Orin, one per robot)                     SERVER  (x86 + RTX GPUs, shared)
 ────────────────────────────────────────                 ──────────────────────────────────────────────
 ZED RGB-D ─┬─► ZED depth + positional tracking ─┐        ┌─► Metric prior: back-projection
 IMU ───────┤                                    ├─► EKF  │       │ pose init, metric points
 wheels ────┴────────────────────────────────────┘   │    │       ▼
                                                odom→base │   GS-SLAM front end: tracking, rasterizer,
                                                     │    │   keyframe gate (one per robot)
 frames 5 Hz: downscale 640x360,                     │    │       │ keyframes
   RGB JPEG + depth lossless (RVL) ───── uplink ─────┼────┤       ▼
 odom pose + covariance 50 Hz ───────── uplink ──────┘    │   GS-SLAM back end: loop closure, mapping
                                                          │       │ submaps
                                                          │       ▼
                                                          │   Map fusion: inter-robot loop closure,
                                                          │   alignment T_i, shared map store → G*
                                                          │       │
                                                          │       ▼
 world→odom correction 5 Hz ◄──────── downlink ───────────┤   Scene update: retrieval, decomposition,
 control commands 20 Hz ◄──────────── downlink ───────────┤   change detection, maintenance
   │                                                      │       │ USD edits
   ▼                                                      │       ▼
 drive controller + watchdog (stop if no command          └─► Isaac Sim (robot + antenna prims,
   for 300 ms), local depth bumper                             Gaussian layer, mesh layer)
 ring buffer of frames on NVMe (last 60 s)                         │ meshes, materials, Tx/Rx poses
                                                                   ▼
                                                               Sionna RT: scene export, paths, CIR, radio map
```

## On the robot (Jetson Orin)

| block (slide) | runs as | why here |
|---|---|---|
| Sensors | ZED SDK capture and depth (GPU), IMU, wheel encoders; timestamps on the PTP/chrony clock (PLAN phase 2) | raw RGB-D is ~1 Gbit/s per robot at 720p 30 Hz; it never leaves the robot |
| Odometry: front end | ZED positional tracking (or Isaac ROS cuVSLAM) | needed every frame, also offline |
| Odometry: fusion filter | `robot_localization` EKF: visual odometry + IMU + wheels, 50-100 Hz, CPU | gives `odom → base_link`, smooth and continuous |
| Odometry: output | `T_k` and covariance, sent to the server at 50 Hz | tiny (~6 kB/s); it is the pose prior for GS tracking and the per-frame pose for Isaac Sim |
| Frame upload | 5 Hz, downscaled to 640x360, RGB JPEG on the hardware encoder, depth 16-bit mm lossless (RVL or zstd); none while odometry says the robot stands still | ~200 kB per frame, ~8 Mbit/s per robot while moving. Depth is never sent through a lossy video codec |
| Pose in world | `T_world_k = T_world_odom (from server) · T_odom_base (EKF)` | the server's correction can jump; the EKF output never does (same split as `map → odom` in ROS REP-105) |
| Drive base | executes the server's commands; watchdog stops the robot when no command for 300 ms; stop on an obstacle in the depth image closer than a set distance | safety does not depend on the network |
| Ring buffer | last 60 s of upload frames on the NVMe | after a dropout they are uploaded, so the map has no gap |

The Orin's GPU then mostly runs the ZED depth and tracking. That leaves room,
and any Orin (AGX, NX, Nano) is enough for this list; on an Orin Nano use the
ZED `PERFORMANCE` depth mode rather than `NEURAL`.

## On the server

| block (slide) | input | output |
|---|---|---|
| Metric prior: back-projection | uploaded depth, `K` (sent once), odom pose | world points, pose init |
| GS-SLAM front end: tracking, rasterizer, rendered view | uploaded frames, odom pose prior, Gaussian map `G_i` | refined pose `T_k`; `T_world_odom` sent back down |
| Keyframe gate | rendered opacity (overlap), motion | keyframes |
| GS-SLAM back end: loop closure, mapping | keyframes | submap per robot |
| Map fusion: inter-robot loop closure, alignment, shared map store | submaps, descriptors | global map `G*`, `T_i` per robot |
| Scene update: retrieval, decomposition, change detection, maintenance | `G*`, object asset library, Isaac Sim's expected view | USD edits (add, move, remove) |
| Isaac Sim | each robot's world pose every frame, USD edits | control commands, expected view, meshes/materials/Tx-Rx poses |
| Sionna RT | scene export from the USD stage | CIR per link, radio map |

Isaac Sim needs an x86 machine with an RTX GPU (it does not run on Jetson),
and GS mapping, map fusion, segmentation for the scene update and Sionna RT
all want a large GPU and the data of every robot, so all of them are here.
With two GPUs: one for GS-SLAM, map fusion and scene update, one for Isaac
Sim and Sionna RT, so rendering the twin does not slow tracking down.

## Over the network (per robot)

| direction | what | rate | size |
|---|---|---|---|
| up | odom pose `T_k` + covariance + timestamp | 50 Hz | ~6 kB/s |
| up | frame: RGB JPEG + depth RVL, 640x360, with its odom pose | 5 Hz while moving | ~1 MB/s |
| up | intrinsics `K`, camera-to-base and antenna-to-base extrinsics | once at start | - |
| down | `T_world_odom` correction | 5 Hz (after each tracked frame) | <1 kB/s |
| down | control commands (velocity or waypoints) + heartbeat | 20 Hz | <1 kB/s |

About 8 Mbit/s up per moving robot; a few robots fit on the lab Wi-Fi.
ROS 2 on both sides, bridged over Wi-Fi with Zenoh (`rmw_zenoh` or
`zenoh-bridge-ros2dds`) rather than plain DDS discovery across the access point.

## Latencies

- Robot pose to Isaac Sim: EKF output plus one network hop, ~10-30 ms, every frame.
- Map correction: upload + tracking + download, ~50-100 ms, applied through
  `T_world_odom`, so a late correction does not disturb the robot's own control.
- New objects in the twin: keyframe → mapping → scene update → USD edit, seconds;
  this path is not time-critical.

## When the network drops

The robot keeps its EKF pose (drift grows slowly), stops within 300 ms because
the commands stop, and keeps buffering frames. On reconnect it uploads the
buffer; the server tracks those frames, closes the loop and sends one new
`T_world_odom`.
