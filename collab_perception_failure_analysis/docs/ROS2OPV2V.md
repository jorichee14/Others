# Converting a rosbag2 recording to OPV2V (`ros2opv2v/`)

Turns a multi-agent ROS 2 recording into the directory layout OpenCOOD reads, so
a real testbed can be run through the same code path as the simulated OPV2V
dataset used everywhere else in this study.

```
python scripts/inspect_bag.py     --bag <bag> --emit-config configs/mine.yaml
python scripts/convert_rosbag.py  --config configs/mine.yaml --dry-run
python scripts/convert_rosbag.py  --config configs/mine.yaml
python scripts/validate_opv2v.py  --root <out>/test --with-open3d
python scripts/test_ros2opv2v.py                       # 62 self-tests, no bag needed
```

Dependencies: `numpy`, `pyyaml`, and `mcap` + `mcap-ros2-support` for `.mcap`
bags (`rosbags` for `.db3`). No ROS installation and no GPU. `open3d` is only
needed to cross-check the written PCDs the way OpenCOOD reads them.

---

## The output

```
<root>/<split>/<scenario>/<cav_id>/000000.pcd
                                  /000000.yaml
                                  /000000_camera0.png      (optional)
```

`cav_id` is an integer folder name. OpenCOOD sorts them, moves a leading
**negative** id to the end, and treats the first remaining one as **ego** — so
infrastructure gets a negative id and the ego is simply the lowest non-negative
one. Timestamps are sequential 6-digit indices, not wall-clock: the real sensor
time is preserved inside each yaml as `ros_stamp_ns`.

Each frame yaml carries what stock OpenCOOD actually reads —

| key | meaning |
|---|---|
| `lidar_pose` | `[x, y, z, roll, yaw, pitch]`, degrees — pose of *this agent's cloud sensor* in the shared world |
| `vehicles` | detection ground truth: `{id: {location, center, extent, angle}}`, `extent` = **half**-dimensions |
| `ego_speed` | km/h, from the odometry twist (or differenced positions if the bag has none) |
| `true_ego_pos`, `predicted_ego_pos`, `plan_trajectory` | present for format compatibility; stock OpenCOOD does not read them |
| `camera0..3` | `cords` / `extrinsic` / `intrinsic`, written only when cameras are configured |

— plus four provenance keys this converter adds (`ros_stamp_ns`,
`ros_frame_stamp_ns`, `source_agent`, `ros_sync`). Extra keys are harmless:
OpenCOOD loads the yaml into a dict and reads what it needs.

`ros_sync` is how synchronous this agent's contribution to this frame actually
is — see [Synchronisation](#synchronisation):

```yaml
ros_sync:
  host: mobile_2
  cloud_dt_ms: 15.2              # signed distance from the frame's reference time
  clock_residual_ms: 0.2         # what correcting the host clocks could not remove
  clock_correction_source: ntp   # ntp | delivery_floor | reference | UNKNOWN | disabled
  clock_residual_source: ntp_spread
  total_ms: 15.4                 # the honest bound on how stale this agent's data is
  pose_interpolation: linear
  camera_dt_ms: {camera0: -8.1}
```

---

## What you must supply

Three things cannot be recovered from a bag, and the converter refuses to run
while any of them is `null` rather than guessing:

1. **`align` / `world_pose` — the shared world frame.** Every agent's odometry
   starts at its own origin. OPV2V assumes one world. Pick one agent's odom
   origin as the world (its `align` is then identity) and give the others the
   measured transform into it. A wrong `align` produces a dataset that loads,
   trains and evaluates while being geometrically meaningless — which is exactly
   why a null is an error and not a default.

   **Unless the bag already carries poses in a shared world frame.** An offline
   mapping pipeline that anchors every robot's trajectory to one surveyed map
   republishes exactly that — in the MIRC coop2 bag, `/mobile_1/global_pose`
   and `/mobile_2/global_pose` (`source: pose`). Point the agents at those and
   `align` is a *measured* identity rather than a declared one: the robots share
   a frame because the anchoring put them there. That removes the single largest
   source of silent error in this conversion, so prefer it whenever the pipeline
   that produced the bag can supply it. Two things to check first:

   * **Which frame do those poses describe?** A pipeline whose state is the
     camera optical frame publishes optical poses, even when the topic's `/tf`
     names a body frame — the two differ by ~90°, and nothing downstream will
     complain. When they are optical, set `child_to_base` to identity and
     remember that the agent's `base` is now that optical frame: every
     `extrinsic` under it is measured **from the camera**, not from `base_link`.
     `geometry.matrix_to_rpy_config` converts a calibration 4×4 into the config
     block so that re-expression is a command, not arithmetic by hand.
   * **Turn `optical_frame` off on depth clouds.** That flag rotates a
     reprojected cloud from the optical convention into ROS body (FLU), which is
     right when the base is a `base_link` and pure error when the base is
     already an optical frame and the extrinsic is optical→optical. It is the
     worst kind of mistake this converter can make: the conversion succeeds, the
     validator passes, and the agent is turned 90°. On the MIRC rig it displaces
     a 3 m return by 4.2 m.
   * **Stamps.** Republished poses normally keep the original bag stamps, so
     they are still on their own host's clock and
     [clock reconciliation](#synchronisation) still applies unchanged.
2. **`extrinsic` — where each sensor sits on its robot.** `base_link -> sensor`.
3. **`object.extent` — the robots' physical half-dimensions**, if you want the
   agents themselves as pseudo-labels.

`scripts/inspect_bag.py` prints the TF tree and every topic's `frame_id`, which
is where to start: if the tree has more than one root, the agents are genuinely
not in a shared frame and (1) has to be measured, not read.

---

## Conventions this converter commits to

### Pose angles are *not* CARLA angles

OpenCOOD turns `lidar_pose` into a matrix with `x_to_world`, whose rotation is
`Rz(yaw) @ Ry(-pitch) @ Rx(-roll)` — CARLA's left-handed convention. Rather than
mirror ROS data into that handedness, the converter stays in the right-handed
ROS world and solves for the `(roll, yaw, pitch)` triple that makes `x_to_world`
reproduce its rotation matrix **exactly**. That parameterisation covers all of
SO(3), so the reconstruction is exact (asserted to 1e-9 over 2000 random
rotations), and it is sufficient because OpenCOOD only ever consumes these
numbers through `x_to_world` / `x1_to_x2` — every relative transform it derives
is then correct.

The practical consequence: **do not compare these angle values numerically
against OPV2V's own yaml files.** Only the matrices they generate are
meaningful. Everything downstream (agent-to-ego projection, box corners,
`commchannel`'s pose noise) is unaffected.

### Intensity travels in the colour channel

OpenCOOD reads a frame as `pcd.points` plus `pcd.colors[:, 0]`. The writer
therefore emits `FIELDS x y z rgb` with `r = g = b = round(intensity * 255)` —
the same layout OPV2V itself uses, and the reason OPV2V intensity is 8-bit
quantised. Verified against `open3d.io.read_point_cloud` in the self-tests:
xyz exact, intensity within 1/255.

Set `intensity.scale` per sensor. Ouster raw counts run to O(1e4), so `1e-4` is
a sensible start; values are clipped to `[0, 1]`. A cloud with no intensity
field (most radars) falls back to `intensity.default` — that is normal input,
not an error.

### Frames are complete or absent

OpenCOOD reads the timestamp list from the *first* agent folder and indexes
every other agent with those same keys. A timestamp missing from one agent is a
`KeyError` at training time, not a skipped sample. So a frame survives only if
every `required: true` agent has a message within `match_tolerance_ms`;
everything else is dropped, and the drop reasons are reported. Set
`required: false` on an agent that may legitimately be absent — but then that
agent's folder will have fewer files than the ego's, so keep it `false` only
for an agent you have also disabled from the timestamp reference.

Tolerance is the main knob, and it is bounded from below: a source at `f` Hz can
be up to `1/(2f)` off any grid, so a tolerance under that drops frames for a
reason no processing can fix. Rather than computing it by hand, read it off the
conversion report's **tightness curve** (`report.tightness`), which lists frames
retained at each budget from 5 ms to 100 ms alongside `structural_floor_ms` and
the agent that sets it. Far above the floor, agents contribute stale data; the
`reuse_rate` column shows when one is repeating messages because it is slower
than the grid. See [Synchronisation](#synchronisation) for what the frames that
survive actually cost.

### Ground lift

`cloud.ground_lift: d` shifts the points **down** by `d` inside the sensor frame
and the sensor pose **up** by `d` in the world. World positions are unchanged, so
ground truth stays aligned (asserted in the self-tests) — what changes is the
apparent sensor height above the floor.

This exists for one purpose: OPV2V's LiDAR sits ~1.9 m up on a car, and every
pretrained checkpoint has learned that ground-plane prior along with a
`cav_lidar_range` of `z ∈ [-3, 1]` *relative to the sensor*. A knee-high robot
LiDAR presents a floor at `z ≈ -0.5` and a ceiling inside the range. Lifting by
~1.4 m makes the geometry look more like what the checkpoint expects. It is a
domain-shift mitigation, not a correction — report it when you report results.

The lift is chosen against the *map's* z, which is meaningful here: the anchoring
rotation is a pure yaw, so the shared frame keeps the mapping session's gravity
axis and `z` really is height.

---

## Synchronisation

The frame table above answers "does every agent have *a* message near this
time". This section is about the harder question — *how near*, and near to
*what clock* — because the answer decides whether a converted dataset can be
used to study timing at all.

Three properties of a multi-robot recording make the naive answer wrong, and
this repository's own results are why they matter: `results/ANALYSIS.md` finds
100 ms of collaborator latency more damaging to fusion than 90% packet loss, and
finds that a *constant* delay is the shape a motion model absorbs quietly rather
than flagging. Any un-modelled asynchrony in the dataset is therefore a latency
impairment sitting inside the baseline of a latency study.

### 1. Three hosts means three clocks

Each machine stamps with its own. An offset between two of them is invisible in
the data — the frames still match inside tolerance, the geometry still looks
right — while acting on one agent's every message as a uniform delay.

Set `clock.enabled: true` and the converter estimates each host's offset three
ways rather than assuming it away (`ros2opv2v/clock.py`):

1. **NTP monitor topics** (`clock.ntp_topics`, host → topic). Direct, but
   self-reported and only present for the hosts that run the monitor.
2. **The delivery floor.** rosbag2 records both `header.stamp` (sender clock) and
   `log_time` (recorder clock at receipt), and their difference is transit plus
   offset. Transit is non-negative, so `min(log − stamp)` approaches the link's
   true floor and differencing two hosts cancels the recorder's clock entirely.
   This is the **only** estimate available for a host with no NTP topic.
3. **The disagreement between the two.** Beyond
   `clock.cross_check_tolerance_ms` the conversion reports `DISAGREE` — one of
   the two is wrong, and neither number should be believed until you know which.

Two things about a custom `NtpStatus` message cannot be read off its field name,
and both are decided from the data rather than guessed:

* **Sign.** `ntpq` and `chrony` disagree about whether "offset" means
  reference-minus-local or local-minus-reference. The wrong sign *doubles* the
  error instead of removing it — strictly worse than not correcting at all.
* **Unit.** Seconds or milliseconds, a factor of 1000 apart. The magnitude
  usually settles it, but at the tens-of-milliseconds scale a wifi-connected
  fleet actually shows, it genuinely does not.

Both are resolved by asking which combination makes the corrected *transit*
floors agree across hosts, since transit floors on one network are similar and
clock offsets are not. A rescale that contradicts the parsed unit is only
accepted when it wins decisively; otherwise the parse stands and the
cross-check is left to flag the disagreement. Pin them with `clock.sign` and
`clock.offset_unit` once you have read the schema — an explicit unit is treated
as an instruction and is never overridden.

**The reference host's correction is zero by definition, not by measurement.**
Nothing in a bag observes its own clock error; whatever it has appears as an
equal and opposite error on every other agent. If it also publishes no NTP
status — as `mobile_1` does not, in the MIRC coop2 bag — the timeline is
internally consistent and externally unanchored. That is fine for relative
inter-agent work and not fine for comparing against anything recorded elsewhere.
`clock.require_measured: true` refuses to convert in that state.

Run `scripts/inspect_bag.py` first: it prints a per-namespace delivery floor, and
namespaces whose floors sit tens of milliseconds apart are either on a much worse
link or on a wrong clock. Either way, turn `clock:` on before trusting the
output.

### 2. Rate asymmetry sets a floor no tolerance can beat

Nearest-neighbour matching cannot do better than **half a publication period**.
In the MIRC coop2 bag the infrastructure camera runs at 10.6 Hz, so no frame
requiring it can be tighter than ±47 ms, while `mobile_2`'s 27.5 Hz camera sits
at ±18 ms. That is a property of the recording, not of the matching.

The conversion report therefore carries a **tightness curve** — frames retained
at each budget from 5 ms to 100 ms — plus `structural_floor_ms` and the agent
that sets it. Pick the operating point off the curve; a
`match_tolerance_ms` below the floor is warned about, because it drops frames
for a reason no amount of processing can fix.

Per-agent, `report.sync[agent]` carries `half_period_ms` (the floor),
`clock_residual_ms` (what correction could not remove) and `reuse_rate` (how
often that agent repeats a message because it is slower than the grid).

### 3. A sweep is not an instant

A 10 Hz spinning LiDAR observes over ~100 ms — the same order as the entire
inter-agent budget — and the resulting smear is azimuth-dependent, so it does not
average out. Set `cloud.point_time_field` (the Ouster's `t`, nanoseconds from the
header stamp) and `cloud.deskew: true`, and every point is moved to where it
would have been observed at the frame's reference instant.

Deskewing targets the *frame time*, not the message stamp, so the correction
absorbs the agent's selection skew as well as the sweep's own duration. Only
relative motion matters, so the operator-supplied `align` transform cancels out
and a wrong alignment cannot corrupt the result. Points are bucketed in time
(`cloud.deskew_buckets`, default 64) and one rigid transform is applied per
bucket; the residual is half a bucket of travel, ~3.5 mm at 5 m/s over a 90 ms
sweep. A cloud whose pose track cannot cover the sweep is left **untouched** and
counted in `report.deskew` — a partially corrected cloud is worse than an
uncorrected one, because it is no longer internally consistent.

### What this does not fix

The residual is *reported*, not eliminated. A frame labelled with a 47 ms camera
skew still has a 47 ms camera skew; what it no longer has is the ability to look
clean. Use `ros_sync.total_ms` downstream to stratify or exclude, and quote the
tightness curve alongside any timing result computed on converted data.

## Ground truth

A bag has no annotations, so the only labels available for free are the agents
themselves: each robot's SLAM pose, plus a configured extent, is an exact 3D box
that the *other* agents should be able to detect. Set `object.emit: true` and an
`extent`. An agent does not label itself (`include_self_in_vehicles: false`,
mirroring OPV2V, where a CAV's box comes from its collaborators' files).

Be clear-eyed about what that is: **two or three boxes per frame, all of them
robots**. It is a geometric sanity signal — do the agents see each other where
the poses say they are? — not a detection benchmark. AP computed against it is
dominated by two objects and is not comparable to any published OPV2V number.

`extent` is **half**-dimensions — a platform 0.6 m long, 0.5 m wide and 0.5 m
tall is `[0.30, 0.25, 0.25]` — and `center` is the offset from the frame's origin
to the middle of the robot, which matters because the origin is a *sensor*, not
the robot's centre.

Both are read in the frame `object.extrinsic` points at (the agent's base by
default). Set it when the base is awkward to describe a robot in: with a camera
optical base, `extent` would otherwise mean [half-width, half-height,
half-length], and a box with its length and height swapped still looks like a
box. `{roll: 90, pitch: -90, yaw: 0}` puts it in ROS body axes at the same
point, so `extent` reads as [half-length, half-width, half-height] and `center`
as [ahead, left, up].

Real labels plug in through `labels.merge_external_labels`, which takes
`{id, location, extent, angle}` in the same world frame. Agent-derived ids are
reserved at 10000+ (CAVs) and 20000+ (RSUs) so a labelling tool's small ids
never collide.

---

## Evaluating pretrained OPV2V checkpoints on converted data

It runs, and it will score badly. Not a bug — an OPV2V checkpoint is a car
detector trained on 64-beam simulated automotive LiDAR at 1.9 m, on ~4 m objects,
outdoors. Pointing it at indoor robot data changes the sensor, the object class,
the scale, the height and the scene statistics at once. Expect near-zero AP
against agent-derived boxes, and treat what you see as a **documented
domain-shift observation**, not a measurement of the method.

What the conversion is genuinely good for, in this repo's terms:

* **visual inspection** — do detections land on real structure, do the agents'
  clouds overlay each other correctly once projected into the ego frame? That is
  a direct test of the `align` transforms and of the whole conversion.
* **a real-data arm for `commchannel/`** — the impairment instrument attaches to
  a *built OpenCOOD dataset*, so once a dataset loads, latency/loss/staleness
  apply to real messages with real inter-agent geometry.
* **training**, if you go that way: emit `train`/`validate` splits and take
  `cav_lidar_range` and the anchor box sizes down to robot scale. Robot-sized
  anchors against car-sized priors is the first thing to change.

The scale mismatch is worth quantifying rather than describing. In the MIRC
coop2 bag the surveyed boards span roughly **8.5 m × 14.3 m** (16.6 m corner to
corner), all within ~15 cm of one height — an indoor room. OPV2V's stock
`cav_lidar_range` reaches ±140 m along x and ±40 m along y, so a stock
configuration spends essentially its whole BEV grid on empty space, and a
collaborator that is 10 m away here would be a near-neighbour there. Anything
derived from the grid — voxel occupancy, the compression ratios in the bandwidth
family, the ego-visible/occluded split in the spatial decomposition — is
measuring the padding, not the scene, until the range is brought down.

To point OpenCOOD at the result, set `validate_dir` in the checkpoint's
`config.yaml` to `<root>/test` and run its `inference.py` with `--show_vis`.

---

## Known limitations

* **Cameras are matched, never interpolated.** An image is a single exposure, so
  its skew is bounded below by half its publication period and nothing removes it.
  Only poses (and, through them, point clouds) can be brought to the frame time
  exactly.
* **Only the frame table is resumable-free** — a re-run redoes everything.
  Conversions are minutes, not hours, so there is no checkpointing.
* **Depth clouds are not de-distorted.** The reprojection uses `k` (or `p`) from
  `CameraInfo` and assumes a rectified image, which `image_rect_raw` is.
* **Camera blocks are best-effort.** Stock OpenCOOD's LiDAR pipeline never reads
  them; they are written for camera-capable forks and for debugging.
* **One RSU per dataset.** OpenCOOD relocates only the *first* negative id to the
  end of the agent list, so a second negative id could be picked as ego. The
  config validator refuses that case.
* **`/tf` is not consumed.** Extrinsics come from the config, not the TF tree —
  a bag's TF is often incomplete across machines, and a config that must be filled
  in makes the operator's assumptions explicit and reviewable.
