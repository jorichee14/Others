# Stretch 4 in ED305: the comms side

**Where comms fits.** Everything in the collaboration moves data
between the robot, the server and the room's cameras: images to YOLO,
lidar or poses for localization, the map (built or the USD twin), the
camera network's streams. The comms side measures those flows on the
real ED305 network, puts the link into the digital twin, decides where
localization and perception should run, and changes what is sent so
each phase of the pick meets its deadline. The algorithms (2Fast-2Lamaa,
YOLO, manipulation) are not changed.

| Collaboration piece | Owner | Comms question |
|---|---|---|
| Sensor recording, 2Fast-2Lamaa, built map vs USD twin | Jorich | robot or server? how big is each map to ship, and to update? |
| ED305 camera network | (access granted) | does it share the Wi-Fi with the robot? what does it cost on the link? |
| Navigation-to-manipulation pipeline (items 4–5) | 廖恩莆 | which flows cross the link, and do they meet each phase's deadline? |
| Bottleneck analysis (item 9) | Prof. Yeh | how much of the delay is the link and queue vs compute? |

## Step 0: Tuesday, alongside the recording

1. **Network inventory.** The AP (model, band, channel, width), the
   robot's Wi-Fi card, the server's connection, and whether the ED305
   cameras are wired or on the same Wi-Fi.
2. **Record the rosbag on the robot's disk**, not over the network
   (recording over Wi-Fi floods the link and distorts every
   measurement).
3. **Log the link while driving:** RSSI, ping RTT to the server, and a
   short `iperf3` at 4–5 spots (table, doorway, corners), timestamped.
4. **chrony** on robot and server.
5. **Topic list:** every topic the pipeline sends across the network,
   with rate and size (`ros2 topic hz`, `ros2 topic bw`).

## Step 1: A link map of ED305 in the twin

Drive the room with the link logger running. Each measurement gets the
robot's pose from localization, in the USD twin frame. Result: a map
of RSSI, RTT and throughput over the twin, with the table and pick
area marked. It shows where the link is weak, and it is the same
frame as the localization comparison.

## Step 2: Baseline of the current pipeline

Stamp each hop (capture, send, server receive, YOLO done, answer back,
command executed) and run 10 picks with the current setup (raw images
over ROS 2). Per phase: delay split into link, queue and inference,
bytes, answers that came back too late. This is the comms part of
items 5 and 9.

## Step 3: Where localization runs, as a network question

Use the Tuesday bag, replayed from the robot over the real Wi-Fi to the
server (repeatable, no need to rerun the robot).

| Option | On the link | Measure |
|---|---|---|
| On the robot | the map, once (built map or twin point cloud); then only the pose | map size and download time; later, how much a map update costs when the room changes |
| On the server | both 3-D lidars (or downsampled keyframes with Draco) and IMU up; corrections down | uplink rate, delay and jitter of the corrections, effect on the camera and YOLO traffic |

Together with the built-map vs twin comparison this gives, for each
map, both its localization accuracy and its cost on the network.

## Step 4: Change what is sent

One change at a time, each measured against the baseline:
1. images resized, JPEG, on request instead of continuous;
2. depth stays on the robot (YOLO returns a box, the robot makes it
   3-D);
3. crop around the object once it is found;
4. H.264 for the search and drive phases;
5. priority for control, pose and YOLO requests; visualisation off;
   Wi-Fi power saving off, 5 GHz;
6. localization placed as decided in step 3.

## Step 5: Phase-aware sending and results

The robot's phase sets what is sent and its deadline:

| Phase | Deadline | Sent |
|---|---|---|
| Find the goal | seconds | low-rate video |
| Drive | 100–300 ms | nothing, or corrections if localization is on the server |
| Locate on arrival | 1–2 s | one full image |
| Plan grasp | ~0.5 s | one crop |
| Fine alignment and close | 30–100 ms | nothing (on the robot) |
| Verify | ~1 s | one crop |

Live picks in ED305, before and after: pick success, deadline hit rate
per phase, bytes per pick, late answers, and the effect of the weak
spots from step 1.

## Later

The ED305 cameras can already see the table. Asking them where the
object is, instead of streaming the robot's images, would remove the
largest uplink flow in the pick.

## Needed

- Robot sessions and the pipeline repo with its topic list: 廖恩莆
- Network details (AP, how the cameras and server connect)
- Permission to install chrony and a link logger on the robot and server
