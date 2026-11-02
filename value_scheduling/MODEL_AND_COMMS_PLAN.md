# Coarse-to-fine pick: model and comms plan (general)

## Per stage

| Stage | Goal / precision | Model(s) | Runs on | Input | Output | Crosses the link (tier) | If slow or late |
|---|---|---|---|---|---|---|---|
| **A. Mapping** (offline) | photoreal map + geometry | GLIM / FAST-LIO2 (poses), NeST Gaussian surfels | robot records; server trains (~16 h) | LiDAR, IMU, colour frames | Gaussian map, LiDAR map, 2-D grid, keyframes | bags at the dock (offline) | — |
| **B. Map upkeep** (background) | keep keyframes and map current | LiDAR-inertial odometry (robot), SLAM correction (server) | robot + server | LiDAR scan, JPEG, pose | updated keyframes | keyframe packs ~2 Hz, on change (background) | buffer, backfill later |
| **C. Assets** (background) | metric object models | SAM 3D, checked against measured clouds | server (≥32 GB GPU) | keyframes with depth | textured asset, surfaces measured / generated | nothing new | runs when no pick is active |
| **0. Task** | what to find | text parsing → class | server | "give me the Pringles" | class / prompt | ↑ text (request) | retry |
| **1a. Find** | which place, ~10 cm | Perception Encoder (keyframe search) → SAM 3 → object cloud → object cache (CRISP) | server | map keyframes + live site cameras (RGB; depth from map, triangulation) | object pose in map + age; view pose | ↓ target + view pose + age (request) | refresh from site cameras if old; else last valid target |
| **1b. Drive** | reach view pose, ~10 cm | Nav2 + localization (AMCL on the grid, or LiDAR localization) | robot | 2-D/3-D LiDAR, odometry | robot at view pose | ↑ robot pose 10 Hz (request); keyframe packs at reduced rate | keep last valid target, slow down |
| **2. Locate + align** | object pose, ~1–2 cm | **YOLOE-seg** (open-vocabulary, small) → mask × robot depth → points → median / shape fit | **robot** (Orin, if fast enough); else GPU PC | head or gripper RGB-D, robot stopped | object points + centre in base frame | if on robot: nothing. If on server: ↑ 1 resized JPEG, ↓ best mask cropped (request) | retry once (new snapshot), then onboard result; keyframe packs paused |
| **3. Grasp** | gripper pose, ~5 mm | IK (pink / pinocchio) + wrist depth grasp on measured surfaces; small base moves if out of reach, re-detect after each | robot | wrist RGB-D, joints, odometry | grasp executed | nothing | — (local) |
| **4. Verify** | holding or not | gripper aperture / effort; optional YOLOE check | robot (server optional) | gripper state, wrist image | success / retry from 2 | optional ↑ crop (request) | local check only |

Rule of thumb: **coarse = big model on the server (SAM 3, search); fine = small model on the robot (YOLOE) or no model (depth + IK).**

## Comms rules (all stages)

1. **No streams during a task.** Images only on request; depth and raw LiDAR never cross the link.
2. **Every request carries an ID and capture time; every answer echoes them and carries its age.** Old answers are not used.
3. **TF at the capture time**, so answers stay correct if the robot moves while waiting.
4. **Timeout from the link,** not a fixed number: RTT + size / goodput + server time (from a heartbeat).
5. **Three tiers with priority:** request (pick) > background (keyframe packs, paused in stages 2–3) > offline (bags at the dock).
6. **Send less:** resize the JPEG to the model input, return only the best mask cropped to its box.
7. **Server GPU:** online requests (SAM 3, YOLOE) before background jobs (training, SAM 3D).
8. **Transport:** request topics best effort keep-last-1, answers reliable; DDS tuned for large messages or `rmw_zenoh`; Wi-Fi power saving off; no remote viewing of camera topics during runs.

## What to do, in order

1. **Measure their grasp as it is:** add request ID + timestamps to `Detect`; log JPEG size, mask size, link time vs GPU time, RSSI over 20 grasps.
2. **YOLOE on the robot:** run YOLOE-seg on the Orin NX (TensorRT); compare time and 3-D centre error with the GPU PC version. If within tolerance, make it the default for stages 2–3, GPU PC as fallback.
3. **Send less + retry:** resized JPEG, best mask cropped, one retry before failing.
4. **Coarse stage:** keyframe search + SAM 3 on the server, site-camera depth from the map, object cache with age, target + view pose to Nav2.
5. **Comms layer:** heartbeat + link monitor, timeouts from the link, tiers with DSCP, keyframe packs paused by stage, GPU priority.
6. **Evaluate:** picks with and without the comms layer, under a good link, a weak spot and background traffic: success, time per pick, waiting time, stale answers used, bytes.
