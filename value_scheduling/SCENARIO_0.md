# Scenario 0: the blind corner

The smallest multi-agent, multi-modal task on the rig that exercises
communication, compute and caching at once, and needs no digital twin.

## Setup

- One corner in site 1 where a corridor is hidden from the approach.
- One Raspberry Pi mast with a camera looking down the hidden corridor.
- One Scout Mini robot (Jetson Orin NX, Ouster LiDAR, ZED, radar)
  driving toward the corner at 1 m/s on a scripted path.
- One person who is either in the corridor or not, per trial.
- The Asus AP, the server, chrony on all three nodes.
- Static map: the existing reference TSDF of site 1, or the robot's own
  local occupancy grid built on the first drive. Used for two things
  only: ray casting what the robot can see, and the stopping distance
  along its path.

**The task.** The robot must slow to a crawl before the corner when the
corridor is occupied and keep 1 m/s when it is clear. It is scored on
(a) whether the mast's information was on the robot before its
approach time (distance to the corner at current speed minus reaction
time), (b) whether it made the right speed decision, (c) bytes sent,
(d) end-to-end latency from the mast's frame to the robot's decision.
Ground truth: the person's presence per trial, and the ZED/LiDAR
detection once the robot rounds the corner.

## What is varied (each axis is one of the earlier threads)

| Axis | Levels | Thread |
|---|---|---|
| What the mast sends | JPEG frame (~100 KB) · person box + velocity (~50 B) · one bit "corridor occupied" | representation / compute on the Pi |
| When it sends | every 500 ms · on change (box appears or moves > 0.5 m) · on request when the robot's path approaches | event trigger / shadow / approach time |
| How it gets there | mast → robot direct · mast → server → robot, server keeps the last observation and serves it when the robot's approach time nears | edge cache |
| What the robot keeps | nothing (acts on the latest message only) · the last observation with velocity, extrapolated to now | robot-side cache + kinematic compensation |
| Who else is on the AP | nobody · the second robot streaming its LiDAR to the server at 1 MB/s | contention |

Multi-modal in two places: the mast's RGB detection versus the robot's
LiDAR/ZED/radar once the person is visible (association across
modalities and frames); and, in the extension, a second sender of the
same corridor with a different modality (the other robot's LiDAR from
the far end), where the server has to pick one.

## Build (one week)

- **Mast node (Pi):** camera at 10 fps; person detection by a light
  model (MobileNet-SSD at ~5 fps on a Pi 4) or background subtraction
  against an empty-corridor reference; output per frame: occupied bit,
  box, image-plane velocity, timestamp; three publishers for the three
  representations.
- **Robot node (Jetson):** reads its path and the map; computes
  approach time to the corridor cell; subscribes to the mast; keeps the
  last message with timestamp and velocity; extrapolates; sets speed.
  Logs decision time and the message it used.
- **Server node:** relays, and in cache mode stores the mast's latest
  observation and pushes it to the robot when its approach time falls
  under a threshold, or on the robot's request.
- **Logger:** every message with send and receive stamps on the shared
  clock; sniffer airtime; the robot's pose from localization and from
  the reference.

## Run (one session)

20 trials per cell of the grid that matters: representation × trigger
× route, with and without the second robot's background load, person
present in half. About 2 hours of driving.

## What it shows

- The byte and latency cost of each representation against decision
  correctness (the one-bit certificate against the JPEG).
- Whether event triggering plus a cached last observation matches
  periodic sending at a fraction of the bytes.
- Whether serving from the server at approach time beats direct
  broadcast under background load (the first contention number).
- Whether the robot's extrapolated cache lets it decide correctly when
  the last message is 1–2 s old.

## Then

Scenario 1 adds the second robot as the approaching agent and the
mast as a shared sender (two receivers, one upload). Scenario 2 adds
all three masts and both robots driving (the contention rig). Nothing
in the code changes; only the number of nodes.
