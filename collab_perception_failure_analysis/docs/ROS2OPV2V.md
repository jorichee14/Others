# Converting a rosbag2 recording to OPV2V (`ros2opv2v/`)

Turns a multi-agent ROS 2 recording into the directory layout OpenCOOD reads, so
a real testbed can be run through the same code path as the simulated OPV2V
dataset used everywhere else in this study.

```
python scripts/inspect_bag.py     --bag <bag> --emit-config configs/mine.yaml
python scripts/convert_rosbag.py  --config configs/mine.yaml --dry-run
python scripts/convert_rosbag.py  --config configs/mine.yaml
python scripts/validate_opv2v.py  --root <out>/test --with-open3d
python scripts/test_ros2opv2v.py                       # 29 self-tests, no bag needed
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

— plus three provenance keys this converter adds (`ros_stamp_ns`,
`ros_frame_stamp_ns`, `source_agent`). Extra keys are harmless: OpenCOOD loads
the yaml into a dict and reads what it needs.

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

Tolerance is the main knob. Compute it from your *slowest* topic: a source at
`f` Hz can be up to `1/(2f)` off any grid. Below that, frames vanish in bulk;
far above it, agents contribute stale data. The report's `reuse_rate` column
shows when an agent is repeating messages because it is slower than the grid.

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

---

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

To point OpenCOOD at the result, set `validate_dir` in the checkpoint's
`config.yaml` to `<root>/test` and run its `inference.py` with `--show_vis`.

---

## Known limitations

* **Only the frame table is resumable-free** — a re-run redoes everything.
  Conversions are minutes, not hours, so there is no checkpointing.
* **Depth clouds are not de-distorted.** The reprojection uses `k` (or `p`) from
  `CameraInfo` and assumes a rectified image, which `image_rect_raw` is.
* **No motion compensation.** A LiDAR sweep is stamped once and treated as
  instantaneous, exactly as OPV2V does. At walking speed the error is centimetres.
* **Camera blocks are best-effort.** Stock OpenCOOD's LiDAR pipeline never reads
  them; they are written for camera-capable forks and for debugging.
* **One RSU per dataset.** OpenCOOD relocates only the *first* negative id to the
  end of the agent list, so a second negative id could be picked as ego. The
  config validator refuses that case.
* **`/tf` is not consumed.** Extrinsics come from the config, not the TF tree —
  a bag's TF is often incomplete across machines, and a config that must be filled
  in makes the operator's assumptions explicit and reviewable.
