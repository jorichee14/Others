# Making the Stretch 4 communication-aware: plan and policy

Setting: room cameras are wired to the server; the robot's Wi-Fi
carries only the robot's traffic (plus other lab traffic). Localization
runs on the robot. So the robot depends on the link for one thing: the
server's answers (goal, object position, fused perception), and once,
at arrival, for sending a crop and the lidar points in the box.

The robot becomes communication-aware in three steps: it **knows** the
link, it **knows how old** each server answer is, and a **policy**
decides what to do with that.

## 1. What the robot knows

**Link state** (link monitor node, 5–10 Hz), published on `/link_state`:
- RSSI, bitrate, retries (`iw dev <if> station dump`);
- RTT to the server (ping or a tiny heartbeat topic);
- recent goodput (from bytes and timestamps of recent requests);
- server's reported inference time and queue;
- **expected answer time** for a request of size S:
  T̂(S) = RTT + S / goodput + server time;
- **link ahead**: the same values looked up in the ED305 link map
  (built once, in the USD twin frame) at points on the planned path.

**Information age**, published on `/info_age`: for each server answer
the robot holds (goal, object position), its age = now − capture time,
and its validity: how long it can be trusted (object on a table: long;
people around the table, as seen by the room cameras: short).

## 2. The policy (rule table first)

| Phase | Normal | Link degraded (T̂ > deadline) | Link lost |
|---|---|---|---|
| Find the object | ask the server (room cameras) for the goal | wait up to the deadline, then search with onboard YOLO | onboard search |
| Drive | full speed; accept goal updates from the server | keep the last goal while its age < validity; slow down near the goal | drive to the last goal at reduced speed; stop if its age > validity |
| Arrive, locate | stop where the link map says the link is good; send one crop + box points; use the fused answer | onboard only (YOLO + lidar points in the box) | onboard only |
| Plan grasp, align, close | local | local | local |
| Verify | local check; server check optional | local | local |

Rules that hold in every phase:
- **Never act on stale information:** an answer older than its
  validity is discarded.
- **Never wait past the deadline:** fall back to the onboard answer.
- **Send the arrival request from a good spot:** choose the stop point
  at the table using the link map.
- **Slow down when the robot depends on the server and the link is
  poor:** speed limit to Nav2 on `/speed_limit`.

Thresholds (deadline per phase, validity per information type, speed
levels) come from the measurements, not guesses.

## 3. The policy as an optimization (upgrade, optional)

Replace the thresholds with one rule per decision: choose the option
(wait for server / onboard / move to a better spot / slow down) that
maximizes the probability that the robot acts on correct information
in time, using T̂ from the link monitor and the measured accuracy of
the onboard and server answers. Same inputs; the table above is what
it reduces to in the typical cases.

## 4. Plan

1. **Measure.** chrony on robot and server; timestamps at capture,
   send, server receive, answer sent, answer received; link logger.
   Drive ED305 to build the link map in the twin.
2. **Link monitor node.** Publishes `/link_state` with T̂(S) and the link
   ahead.
3. **Answer handler.** Stamps every answer with capture time; publishes
   `/info_age`; drops answers older than their validity.
4. **Policy node.** Reads phase, `/link_state`, `/info_age`; outputs:
   who answers (server / onboard), when and from where to send, speed
   limit.
5. **Comms fixes** applied once: JPEG / crop instead of raw, best effort
   keep-last-1, DDS tuning, DSCP priority, Wi-Fi power saving off.
6. **Evaluate.**

## 5. Evaluation

Compare **comms-unaware** (today's pipeline: always waits for the
server, constant speed) with **comms-aware** (the policy).

Conditions: normal link; weak spot at the table (AP placement or
`tc netem` delay / loss on the robot's interface); lab traffic.

Metrics: pick success; time per pick; time spent waiting for the
server; how often the robot acted on stale information; object error at
the moment of action (ground truth from the room cameras); bytes on the
robot's Wi-Fi.

## 6. Literature behind each piece

| Piece | Work |
|---|---|
| Fallback to onboard when the server is not worth it | Chinchali et al. (RSS 2019) |
| Choose where to send from using a link map | Liu et al. (arXiv 2606.31497, 2026); Gordon et al. (INFOCOM 2026 workshop) |
| Slow down with link quality | teleoperation practice; RCAMP (IROS 2017) for link-aware motion |
| Information age and validity | AoI (Kaul et al., INFOCOM 2012); AoII (Maatouk et al., ToN 2020) |
| Image and transport fixes | AWStream / DDS (SIGCOMM); Lee et al. (2025) on ROS 2 over Wi-Fi |
