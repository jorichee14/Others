# Stretch 4 collaboration: the comms-side plan

**Goal.** Make the Stretch 4 navigation-to-manipulation pipeline meet
each phase's deadline over the ED305 Wi-Fi, by changing what crosses
the link (when, which part, in what form, in what order). The
localization, YOLO and manipulation algorithms stay as they are.

## Who does what (from the action items)

| Part | Owner | Comms side uses it for |
|---|---|---|
| Pipeline (item 4) | 廖恩莆 | the flows to measure and change |
| Pipeline latency (item 5) | 廖恩莆 | split into link / queue / inference by the comms logging |
| Localization and perception bottleneck (item 9) | Prof. Yeh | the link share of the bottleneck comes from step 1 |
| Reference map (item 2), sensor recording, 2Fast-2Lamaa test | Jorich | the map frame; the rosbag for offline replay; localization cost on the robot vs the server |
| Comms measurement and changes (steps 1–4 below) | Jorich | — |

## Step 0: Tuesday on the robot

1. Record a rosbag: both lidars, IMU, the three cameras, odometry,
   joint states, while driving around ED305 and approaching the table.
2. List every topic that crosses the Wi-Fi: rate and size
   (`ros2 topic hz`, `ros2 topic bw`), direction, QoS.
3. Measure the link at 4–5 places in ED305 (table, doorway, far
   corner): RTT, throughput (`iperf3`), RSSI.
4. Run 2Fast-2Lamaa on the robot with one lidar and its IMU; note CPU,
   memory and time per scan.
5. Note the onboard CPU/GPU and the server's specs.

## Step 1: Baseline, where the time goes (weeks 1–2)

- chrony on robot and server.
- A timestamp at each hop: capture, send, server receive, YOLO done,
  answer received on the robot, command executed.
- Run the current pipeline (raw images over ROS 2) for 10 picks.

**Output.** Per phase: deadline, end-to-end delay split into link,
queue and inference, bytes sent, how many answers arrived too late to
be used. This becomes the comms part of items 5 and 9.

## Step 2: Comms changes, one at a time (weeks 2–4)

Each change is measured against the baseline: first on the replayed
rosbag, then live.

1. **Images:** resize to YOLO's input, JPEG, on request instead of a
   continuous stream; best effort, keep last 1.
2. **Depth stays on the robot:** YOLO returns a box; the robot turns it
   into 3-D with its own depth.
3. **Crop:** once the object is found, send only the region around the
   last box.
4. **Video codec** (H.264, zero-latency) for the continuous phases
   (search, drive).
5. **Traffic separation:** control, pose and YOLO requests given
   priority (separate topics, DSCP/WMM); visualisation off during runs;
   Wi-Fi power saving off, 5 GHz.
6. **Localization placement:** 2Fast-2Lamaa on the robot (only the pose
   on the link) vs on the server (downsampled lidar keyframes with
   Draco, map→odom corrections back). If the AMD Ryzen AI Max+ 395 PC
   (item 8) arrives, it is one more option for where this runs.

## Step 3: Phase-aware sending (weeks 4–6)

The robot's task phase decides what is sent: rate, region and form,
each phase with its own deadline. Before each request, the robot picks
the form that fits the deadline at the link quality it has measured.

| Phase | Deadline | Sent |
|---|---|---|
| Find the goal | seconds | low-rate frames (video) |
| Drive | 100–300 ms | nothing if localization is on the robot; otherwise corrections |
| Locate on arrival | 1–2 s | one full image |
| Plan grasp | ~0.5 s | one crop |
| Fine alignment and close | 30–100 ms | nothing (runs on the robot) |
| Verify | ~1 s | one crop |

## Step 4: Results (weeks 6–8)

Live picks in ED305, before and after:
- pick success rate and time per phase;
- deadline hit rate per phase;
- bytes on the link per pick;
- answers that arrived too late to be used;
- 2Fast-2Lamaa on the robot vs on the server: delay and bytes.

This becomes the shared report or paper with 廖恩莆 and Prof. Yeh.

## Needed from others

- Robot access (Tuesday and later sessions): 廖恩莆
- The pipeline repo and its topic list (item 7): 廖恩莆
- Access to ED305 compute and the camera network (item 1): 廖恩莆
- Server specs and permission to install chrony and a logger on both ends
