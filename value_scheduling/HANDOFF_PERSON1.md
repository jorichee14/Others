# Handoff: validity-timed sharing (person 1)

Technical document. Read `PERSON1.md` for the goal in plain words; this
is the specification. Everything here runs on the existing rig and the
existing recordings; nothing needs a digital twin, Sionna, labels, or
changes below the application layer.

## 0. Summary

Build a system in which a mast or robot shares an observation of a map
cell only when that cell's state changes, the server keeps one copy of
it with a validity time derived from how often that cell changes, and a
robot receives it before its stopping distance reaches the cell, holds
it with its velocity, and treats it as expired after the validity time.
Measure, against timer broadcast, the bytes removed, the decisions kept,
the deadline hits recovered for pose and control, and the fraction of
timer traffic that was never needed.

## 1. Environment and assets

| Item | Where / what |
|---|---|
| Robots | two Scout Mini: `mobile_1` (Jetson Orin NX, Ouster OS1, ZED 2i, two TI radars; NTP server of the rig), `mobile_2` (Jetson, RealSense, Isaac VSLAM) |
| Masts | three Raspberry Pi 4 (`infra_*`), camera; one Pi is the nexmon CSI sniffer, not a mast |
| AP | Asus RT-BE82U; settings for sessions: airtime fairness off, band steering off, 5 GHz pinned, 80 MHz, 802.11ax off if the sniffer needs ac |
| Server | the workstation that runs the twin-side stack; Ethernet to the AP |
| Clock | chrony, `mobile_1` serves; offset ≤ 0.99 ms measured. Every node logs with `rclpy` time after chrony sync; the `/ntp/*` topics are recorded as evidence |
| ROS 2 | Humble (Isaac ROS common image on the Jetsons) |
| Recording | `scoop_recording/record.yaml`, `record.sh <machine> <pass>`; mcap bags per machine, aligned offline by timestamp |
| Static map | reference TSDF of site 1 (`data/processed/<date>/<pass>/map/`); copy to every node; it is not a cache |
| Processed passes | `data/processed/<date>/<pass>/` per `scoop_pipeline/scoop/layout.py`; `comms/` holds `wifi/{wifi,ping,iperf}.csv`, `ntp/ntp.csv`, every row with the machine's map position |
| Logged channel | `scoop_pipeline/scoop/channel.py`: `make("logged:<processed pass>")`, `deliver(src, dst, t_send, size_bytes)` gives arrival time or loss from the recorded RTT, goodput and outages; use it for bench replay before any session |
| Existing analysis | `scoop_pipeline/analysis/comms/sim_gap.py` (in-frame share by message size), `wifi_analysis.py`, `comms_map.py` |

## 2. Definitions

- **Cell** x: a cell of the static map's 2-D grid (0.2 m, the BEV
  resolution used elsewhere; make it a parameter). Index by integer
  (ix, iy).
- **Observation** o = (x, source, t_obs, state, box, v, repr, bytes):
  cell, who saw it, when, occupied ∈ {0, 1}, bounding box in map frame
  (optional), velocity in map frame (optional), representation ∈
  {bit, box, jpeg}, size.
- **Change rate** λ(x) [1/s]: the rate at which the occupied state of
  x flips, estimated from the recordings (§5).
- **Validity** TTL(x) = clip(τ_tol / λ(x), TTL_min, TTL_max), with
  τ_tol the AoII tolerance from person 3's metric page (initial 2 s),
  TTL_min = 1 s, TTL_max = 600 s. Walls: λ ≈ 0 → TTL_max.
- **Approach time** a_j(x) for robot j: time until the stopping
  distance of j along its path reaches x: a_j(x) = (s_path(x) − d_stop)
  / v_j, with s_path the arc length along the path from the robot to
  the closest path point to x, d_stop = v_j·t_react + v_j²/(2·a_brake),
  t_react = 0.3 s, a_brake = 1.0 m/s². Infinite if x is off the path
  corridor (lateral distance > 1.5 m).
- **Envelope**: cells with a_j(x) < H, H = 5 s.
- **Shadow cell**: a cell in the envelope that the robot's own sensors
  will not see before a_j(x) (ray cast from path points against the
  static map; a cell is self-visible if any path point before a_j(x)
  has line of sight within sensor range). For S0 the corridor cell
  behind the corner is the only shadow cell; compute it anyway so the
  code is general.
- **Hit / miss**: at decision time the robot holds an observation of
  the shadow cell with t_now − t_obs ≤ TTL(x) (hit) or not (miss).

## 3. Architecture

```
 mast (Pi)                      server                          robot (Jetson)
 camera 10 fps                  obs_store                        path + static map
 bg subtraction ──▶ obs ──▶ relay | store ──▶ serve ──▶ memory ──▶ speed decision
   trigger: periodic / change     dedup, TTL, approach    extrapolate, expire   (hook for AoII rule)
```

All three run as ROS 2 nodes in one package `vts` (validity-timed
sharing). Transport: DDS best-effort for observations, reliable for
control. Routes: `direct` (mast publishes, robot subscribes), `relay`
(server republishes), `store` (server stores and serves).

### 3.1 Topics and messages

| Topic | Type | Notes |
|---|---|---|
| `/vts/obs/<mast>/bit` | `vts/Obs` | repr = bit, bytes ≈ 40 (header + cell + state + stamp) |
| `/vts/obs/<mast>/box` | `vts/Obs` | repr = box, bytes ≈ 80 |
| `/vts/obs/<mast>/jpeg` | `vts/ObsImage` | repr = jpeg, bytes ≈ 60–120 KB at 640×480 q75 |
| `/vts/serve/<robot>` | `vts/Obs` | server → robot, store mode |
| `/vts/request/<robot>` | `vts/Request` | robot → server, cells and approach times (optional route) |
| `/vts/robot/<robot>/belief` | `vts/Belief` | per envelope cell: t_obs held, λ, TTL, a_j; person 3's field; 10 Hz |
| `/vts/robot/<robot>/decision` | `vts/Decision` | speed command issued, reason, cell, hit/miss; on change |
| `/vts/log` | `vts/Event` | every event (§6); every node publishes; one recorder subscribes |

`vts/Obs`: `header.stamp` (t_obs, chrony clock), `string source`,
`int32 ix, iy`, `uint8 occupied`, `float32 box[6]` (x, y, z, l, w, yaw,
map frame; zeros if none), `float32 vx, vy`, `uint8 repr`, `uint32
bytes`, `uint64 hash` (FNV-1a of (ix, iy, occupied, box rounded to
0.1 m)), `float32 ttl_s` (set by server in store mode).

### 3.2 Mast node (`vts_mast`)

- Camera at 10 fps (Pi camera v2 or USB; use `v4l2` through
  `usb_cam` or `camera_ros`).
- Background subtraction: reference frame captured at session start
  with the corridor empty (`--capture-ref`), MOG2 as the fallback.
  Foreground blob → occupied bit; blob bbox → image box; map box via
  the mast's fixed extrinsic and the floor plane (the mast's pose in
  the map frame comes from the board anchoring; load from
  `static_tf.yaml`); velocity from blob centroid over 3 frames.
- Cell: the corridor cell(s) the mast's frustum covers; fixed per mast,
  from a YAML listing (ix, iy) per mast.
- Trigger: `periodic` at 2 Hz or 10 Hz; `change` when occupied flips
  or box centroid moves > 0.5 m or velocity changes > 0.5 m/s; a
  heartbeat at 0.2 Hz in `change` mode so silence is distinguishable
  from a dead mast.
- Publishes on all three representation topics every trigger (the
  route decides which one is consumed), or on one, by parameter.
- Logs `create` and `send` events.

### 3.3 Server node (`vts_server`)

- Mode `relay`: republishes to `/vts/serve/<robot>` for every robot.
- Mode `store`: table keyed by (ix, iy): latest `Obs`, source, t_obs,
  hash, TTL(x) from the λ table. On upload: if hash equals the stored
  one and the stored copy is still valid, drop (`dedup` event);
  otherwise replace (`store` event). Expiry sweep at 1 Hz (`expire`
  event).
- Serving: subscribes to each robot's `belief`; for every envelope
  cell with a_j(x) ≤ T_serve (default 3 s) and a stored observation
  the robot does not hold (robot's `t_obs` older than the stored one,
  or none), publish it once on `/vts/serve/<robot>` (`serve` event).
  Alternatively on `/vts/request`.
- Priority: observations are published in the `AC_BE` access category;
  the robots' pose and control traffic in `AC_VO` (set DSCP via the
  DDS QoS transport priority or `iptables` marking; document which).
  This is not slotting; it is only so the baseline FIFO case and the
  prioritized case can both be run.

### 3.4 Robot node (`vts_robot`)

- Inputs: scripted path (list of waypoints in the map frame), current
  pose (`/tf` map→base_link from the localization), current speed, the
  static map, the λ table, mast list.
- Every 100 ms: envelope cells and a_j(x); shadow cells by ray cast;
  publishes `belief`.
- Memory: per cell, latest `Obs` held, from direct, relay or serve;
  extrapolate box by v·(t_now − t_obs); valid while t_now − t_obs ≤
  TTL(x). `hit`/`miss` events at decision time.
- Decision (default, to be replaced by person 3's AoII rule through the
  `decide()` hook): for the nearest shadow cell with a_j(x) ≤ 1 s:
  valid observation and occupied = 0 → full speed (1.0 m/s); valid and
  occupied = 1 → crawl (0.2 m/s); miss → crawl. Publishes `Decision`
  and the speed command on the Scout's topic (`/cmd_vel`,
  `geometry_msgs/Twist`; confirm limits on the platform).
- Logs `receive`, `hit`, `miss`, `decide`.

### 3.5 Load generator

Baseline load: `mobile_2` streams its recording stack to the server
(the real thing). Fallback: a publisher of 1 MB messages at 1 Hz in
`AC_BE`. Log bytes on the wire from the sniffer, not from the
publisher.

## 4. Experiments

Common: one corner at site 1 with a corridor behind it longer than the
stopping distance at 1 m/s (≥ 1.5 m); one mast placed to see the
corridor and not the approach; a scripted path from 8 m before the
corner to 3 m past it; a person who stands 1–3 m into the corridor in
half the trials (randomized, recorded in the trial script); 20 trials
per cell. Every session is also recorded with `record.sh` so the pose
reference, link tables and CSI exist for scoring.

### S0 (week 2): one mast, one robot

Grid: representation {bit, box, jpeg} × trigger {periodic 2 Hz,
change} × route {direct, relay} × load {off, on}. 24 cells × 20 trials
= 480 approaches, about 12 s each including the return: two sessions
of two hours, or trim to representation × trigger × load with route =
relay (12 cells).

Outputs per cell: decision correctness, observation-before-approach
rate, bytes per approach (wire), latency mast-frame-to-decision
(median, p95), pose/control deadline hits (from their own stamps),
waste fraction for the periodic cells (messages whose hash equalled
the previous one).

### S1 (week 3): one mast, two robots, the store

Both robots approach the same corner 10–20 s apart on the same path.
Modes: change-broadcast (mast to both, no store), store (mast uploads
once, server serves each at a_j ≤ 3 s), request (robots ask at a_j ≤
3 s). Load on. 20 trials per mode, person in half.

Outputs: uplink bytes (one upload vs two), serve-before-approach rate
per robot, correctness, deadline hits. **Stop-rule check**: if
change-broadcast reaches ≥ 90 % of the store's byte saving and the
same deadline hits, stop after the write-up of S0–S1.

### S2 (week 4): the changed building

After mapping, place a cabinet in a corridor on the route (trial A)
and close a door (trial B). The other robot or a mast observes the
change first. Modes: no refresh; periodic refresh of every envelope
cell at 0.1 Hz; TTL (the store invalidates on the observed change and
the robot refreshes before approach). 10 drives per mode per change.

Outputs: refresh bytes, fraction of refreshes with no change
(unnecessary), time from the change being observed to the robot
holding it, whether the robot replans before its stopping distance.

### S3 (week 6, joint)

Three masts, two robots driving, pose and control live, person 2's
slotting and person 3's speed rule as switches. Four runs. Not yours to
design; bring the nodes and the scorer.

## 5. λ(x) from the recordings

Input: processed passes with repeated visits (survey_1 stops; coop_2;
the mapping passes). For each pass, the per-frame occupancy of each
cell from the decoded LiDAR (`scoop_pipeline/processing/decode_ouster.py`
→ clouds → 2-D occupancy at 0.2 m, height band 0.2–1.8 m) in the map
frame via the reference trajectory. For each cell: the sequence of
(t, occupied) across visits; a flip is a change between consecutive
visits; λ̂(x) = flips / total time between first and last visit, with
a Laplace prior (1 flip per 1000 s) for cells with few visits. Output:
`lambda.npz` (grid, λ̂, visit count). Report the histogram and a map;
sanity: walls near zero, corridor centres highest.

Online update: every observation a node makes is a visit; increment
flips and time; recompute λ̂ at 1 Hz on the server; robots pull the
table at session start.

## 6. Logging and scoring

`vts/Event`: `stamp` (chrony), `node`, `event ∈ {create, send, receive,
dedup, store, serve, expire, hit, miss, decide}`, `ix, iy`, `source`,
`repr`, `bytes`, `hash`, `t_obs`, `ttl_s`, `a_j`, `robot`, `extra`
(JSON string). One recorder writes `events.jsonl` per session.
Sniffer: `tcpdump`/`tshark` in monitor mode on the Pi, pcap per
session; airtime per station from frame durations.

Scorer (`vts/score.py`): joins `events.jsonl`, the trial script (trial
id, person present, start/end stamps), the reference pose (from the
pass's localization output), and the pcap. Produces one row per
approach: cell, mode, person, decision, correct, obs_before_approach,
bytes_uplink, bytes_downlink, latency_frame_to_decision, hit/miss,
pose_deadline_misses, control_deadline_misses, waste (periodic only).
And per cell of the grid: medians, p95, rates with Wilson intervals.

Person 3's metric page adds: per approach, AoI and AoII of the shadow
cell at a_j; your `belief` topic carries what is needed.

## 7. Bench replay before any session (week 1 exit)

`vts/replay.py`: feed a recorded corridor sequence (frames or a
synthetic occupancy script) through `vts_mast` → channel.py `Logged`
(`logged:<coop_2 processed pass>`, `,queue`) → `vts_server` →
`vts_robot` on one machine, with the robot's path and speed simulated;
produce `events.jsonl` and the scorer's table for 20 synthetic trials.
Exit criterion: the table is complete, bytes and latencies are
plausible against `sim_gap.py`'s numbers for the same sizes, and the
hit/miss logic agrees with hand-computed approach times on three
cases.

## 8. Acceptance criteria

| Week | Done when |
|---|---|
| 1 | bench replay produces the scorer table; λ̂ map exists with walls ≈ 0 |
| 2 | S0 table for ≥ 12 grid cells; the one-bit-on-change cell decides as well as jpeg-periodic (within the Wilson interval) at < 2 % of its bytes |
| 3 | S1 table; the stop rule evaluated and the outcome stated either way |
| 4 | S2 table; TTL mode replans before stopping distance in ≥ 9/10 drives per change |
| 5 | write-up with one figure per scenario; nodes tagged for the joint session |

## 9. Risks and what to do

- **Background subtraction fails under lighting change.** Capture the
  reference at session start; fall back to MOG2; if both fail, run
  MobileNet-SSD on the Pi at ~5 fps and drop the 10 fps requirement.
- **Mast extrinsic wrong → box in the wrong cell.** Only the bit and
  the cell id are needed for the decision; the box is for person 3's
  extrapolation. Verify the cell mapping with a person standing at a
  marked spot before trials.
- **Few revisits → λ̂ noisy.** Use the prior; report visit counts;
  S2's cells are chosen where visits are many.
- **Clock drift on the Pi.** Check `/ntp/*` before every session;
  reject a session if offset > 5 ms.
- **The stop rule fires.** That is a result; write it up.

## 10. Repository layout

```
value_scheduling/vts/
  package.xml, setup.py
  vts/mast.py  vts/server.py  vts/robot.py  vts/load.py
  vts/lambda_map.py   (§5)
  vts/replay.py       (§7)
  vts/score.py        (§6)
  msg/Obs.msg ObsImage.msg Request.msg Belief.msg Decision.msg Event.msg
  config/masts.yaml  paths/corner_a.yaml  params/*.yaml
  scripts/trial.py    (trial script: randomizes person presence, stamps start/end)
  sessions/<date>_<scenario>/events.jsonl, trials.csv, sniffer.pcap, scores.csv
```

Commit per week with the acceptance item in the message; raw sessions
are not committed (bags and pcaps go with the dataset).
