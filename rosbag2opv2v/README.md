# rosbag2opv2v

Convert a rosbag2 (**MCAP**) multi-agent recording into an **OPV2V-format
dataset** that [OpenCOOD](https://github.com/DerrickXuNu/OpenCOOD) can train and
evaluate on, with a ready-made config for the MIRC cooperative recording
(`mirc_dataset_coop2_*.mcap`: Robot A + Robot B + infrastructure node).

No ROS installation is needed — MCAP carries the message definitions, so the
converter decodes `sensor_msgs`, `geometry_msgs` and even the project's custom
messages straight out of the bag.

```bash
pip install -r requirements.txt

# 1. see what is actually in the bag
python -m rosbag2opv2v topics --bag /data/mirc_dataset_coop2_20260828_completed

# 2. plan the conversion without writing anything
python -m rosbag2opv2v convert --bag /data/mirc_dataset_coop2_20260828_completed \
    --config configs/mirc_coop2.yaml --out /data/opv2v_mirc --dry-run

# 3. convert, validate, and emit a matching OpenCOOD training config
python -m rosbag2opv2v convert --bag /data/mirc_dataset_coop2_20260828_completed \
    --config configs/mirc_coop2.yaml --out /data/opv2v_mirc \
    --verify --emit-hypes /data/point_pillar_mirc.yaml
```

## What comes out

```
/data/opv2v_mirc/
├── train/
│   └── mirc_coop2_000/            # one scenario per `scenario_seconds` chunk
│       ├── 1/                     # mobile_1  (Robot A) -- ego: lowest id >= 0
│       │   ├── 000000.pcd         #   point cloud in the LiDAR frame
│       │   ├── 000000.yaml        #   poses + ground-truth boxes
│       │   ├── 000000_camera0.png #   ZED left image (optional)
│       │   └── ...
│       ├── 2/                     # mobile_2  (Robot B, depth-derived cloud)
│       ├── -1/                    # infra_1   (RSU: negative id, never ego)
│       └── scenario_meta.yaml     # provenance: bag, wall-clock times, agents
├── validate/ , test/
└── conversion_report.json         # what was written, what was dropped, and why
```

Each `<timestamp>.yaml` is what OpenCOOD reads:

| key | meaning |
|---|---|
| `lidar_pose` | `[x, y, z, roll, yaw, pitch]` of the point-cloud sensor in the world frame, degrees |
| `true_ego_pos` / `predicted_ego_pos` | the agent's body pose, same encoding |
| `ego_speed` | km/h (OpenCOOD normalises by 30 in the intermediate-fusion dataset) |
| `vehicles` | ground-truth boxes: `location`, `center`, `angle`, `extent` (half-sizes), `speed` |
| `camera0` … | `cords` (world pose), `extrinsic` (camera → LiDAR, 4×4), `intrinsic` (3×3) |
| `plan_trajectory` | empty (no planner in the recording) |
| `RSU: true` | on infrastructure agents, as in V2XSet |
| `mirc` | **extra**, ignored by OpenCOOD: real timestamps, per-stream time offsets, and the time-joined Wi-Fi / NTP telemetry |

### The Euler convention (the part that silently breaks conversions)

OpenCOOD rebuilds every pose with `x_to_world`, which is CARLA's convention:

```
R = Rz(yaw) · Ry(-pitch) · Rx(-roll)          # angles in degrees
```

i.e. a right-handed Z-Y-X rotation with pitch and roll negated. The converter
encodes ROS rotation matrices accordingly (`transforms.matrix_to_opv2v_pose`),
and `tests/test_pipeline.py` checks the round trip against a verbatim copy of
OpenCOOD's `x_to_world`. Because OpenCOOD only ever uses these poses
*relatively* (`x1_to_x2`), a self-consistent export behaves exactly like real
OPV2V data while staying in your own right-handed metric world frame.

### Point clouds

Intensity travels in the PCD `rgb` field, because that is what
`opencood.utils.pcd_utils.pcd_to_np` reads (`np.asarray(pcd.colors)[:, 0]`).
The writer reproduces Open3D's own layout (`FIELDS x y z rgb`, `TYPE F F F F`),
so no Open3D dependency is needed to produce files Open3D reads back exactly —
intensity is quantised to 8 bits, as it is in real OPV2V data. Raw sensor
intensity is rescaled into `[0, 1]` first, either by a fixed
`intensity_scale` or per-cloud by a percentile (`intensity_normalize:
percentile`, the sane default for Ouster signal counts).

## How the recording maps onto OPV2V

| bag | OPV2V agent | cloud | notes |
|---|---|---|---|
| `mobile_1` (Robot A) | id `1`, ego | `/mobile_1/ouster/points` | ZED left as `camera0`; ego radars can be merged in (commented out in the config) |
| `mobile_2` (Robot B) | id `2` | back-projected from `/mobile_2/depth/image_rect_raw` | RGBD-only agent, so the "LiDAR" is the depth cloud, rotated from optical into x-forward/z-up |
| `infra_1` | id `-1`, RSU | `/infra_1/radar/points_all` | negative id keeps OpenCOOD from ever electing it ego |

Ground truth is the dataset's own design (`DATASET_NOVELTY.md` §6.3): the robots
*are* the tracked targets, so each robot's certified pose becomes a
ground-truth box in the other agents' yamls, with the size taken from
`object.extent` in the config.

Timing: the converter samples a common 10 Hz timeline, interpolates each pose
(SLERP + lerp) to the sample instant, and picks the nearest sensor message
within that stream's `max_age`. Frames where a required stream is missing are
dropped; the per-stream mean/max offsets land in `conversion_report.json`, and
per frame in `mirc.<stream>_dt`.

## Before you trust the output

The config ships with placeholders where the recording cannot answer the
question by itself. Check these:

1. **`pose.frame`** — which frame each `*/global_pose` describes (`base_link`
   in the shipped config). It is the starting point for every `source: tf`
   extrinsic lookup.
2. **Extrinsics** — with `source: tf` the converter resolves sensor→body from
   `/tf_static` and prints what it found (`"extrinsic": "tf(base_link<-os_sensor)"`
   in the report). If the path is missing it *warns* and falls back to the
   `xyz` / `rpy_deg` next to it, which are guesses.
3. **`object.extent`** — half `[length, width, height]` of each robot, in
   metres. These are the ground-truth boxes; measure them.
4. **The infrastructure world pose** — `infra_1` is `source: static` at the
   origin because its world extrinsic is the open calibration item
   (`DATASET_NOVELTY.md` §4.2). Until it is measured, the RSU's clouds and
   images are in a placeholder frame and the cross-agent overlap reported by
   `verify` will be poor for that agent.

`python -m rosbag2opv2v verify --root <out>` checks the export the way OpenCOOD
reads it — identical timestamp lists per agent, integer cav folder names, every
frame's `.pcd` present and readable, `lidar_pose` well formed — and then two
geometric checks that catch a wrong extrinsic or a wrong pose:

* `mean_cross_agent_voxel_overlap` — fraction of another agent's voxels that
  coincide with the ego's after projecting with the exported poses. On the
  synthetic fixture this is ~0.7; near zero means an extrinsic or the world
  frame is wrong.
* `gt_boxes_with_ego_points` — fraction of ground-truth boxes that actually
  contain ego LiDAR points. Low values mean the boxes are in the wrong place
  (or genuinely occluded).

## Training with OpenCOOD

`--emit-hypes` writes a PointPillars config sized for *this* dataset rather
than for outdoor traffic: LiDAR range from the exported clouds (symmetric in
x/y, since other agents project into the ego frame), a 0.1 m voxel, anchors
from the measured robot extents, and lower positive/negative IoU thresholds
because sub-metre targets on a 0.2 m anchor grid cannot reach the outdoor 0.6.

```bash
python opencood/tools/train.py --hypes_yaml /data/point_pillar_mirc.yaml
```

Two constants inside OpenCOOD are *not* config-driven and matter indoors:

* `opencood/data_utils/datasets/__init__.py`: `GT_RANGE = [-140, -40, -3, 140, 40, 1]`
  is the evaluation-time box filter. Indoor boxes fall inside it, but only
  because a robot box sits ~0.1–0.3 m *below* the ego LiDAR; `verify` prints a
  NOTE for any box that would be filtered.
* `opencood/data_utils/post_processor/voxel_postprocessor.py`: anchors are
  hardcoded at `cz = -1.0` (a car centre below a roof LiDAR). Anchor/target
  matching is 2-D BEV so this does not break assignment, but it puts a constant
  offset in the z regression target. Set it near `-(lidar height above the
  robot box centre)` if you care.

## Layout

```
rosbag2opv2v/
├── configs/mirc_coop2.yaml      # the recording's agents, topics, extrinsics
├── rosbag2opv2v/
│   ├── bag.py                   # two-pass MCAP reader (index, then decode only what is used)
│   ├── config.py                # config schema + validation
│   ├── convert.py               # sampling, frame planning, yaml/pcd/png writing
│   ├── opencood_hypes.py        # emit a training config sized from the data
│   ├── pcd_io.py                # OPV2V-compatible PCD write/read
│   ├── ros_msgs.py              # PointCloud2 / Image / CameraInfo / depth decoding
│   ├── transforms.py            # quaternions, SLERP, TF graph, the OPV2V pose encoding
│   └── verify.py                # structural + geometric validation
├── tools/make_synthetic_bag.py  # writes a known-truth bag with the same topics
└── tests/test_pipeline.py       # bag -> export -> OpenCOOD-style read, checked against truth
```

## Tests

```bash
python -m unittest discover -s tests
```

The suite generates a synthetic bag (two robots circling a 12 × 10 m room
watched by a fixed node), converts it with the shipped config, and checks the
result against the simulated truth: box centres reconstructed through
OpenCOOD's own `x1_to_x2` math to within 2 mm, the depth agent's cloud landing
inside the room, cross-agent overlap, PCD round trips, and the emitted training
config's grid arithmetic.

## Related tooling in this repo

`analysis/` characterises the same recording rather than exporting it:
`extract_bag.py` dumps the light topics to Parquet with a `stamp_audit` over
every header-bearing topic, and `ntp_analysis.py` reports the cross-agent clock
offsets. Worth running first — this converter associates agents by
`header.stamp`, so its cross-agent alignment is only as good as the NTP sync
that analysis measures.

## Memory and speed

Bulky topics are never fully decoded during planning — only their CDR header is
peeked — and pass 2 decodes exactly the messages the plan selected. A 156 s,
364 k-message recording converts without holding more than one sweep per agent
in memory. Point clouds are written as binary PCD; add `--no-images` to skip
the PNGs, which dominate the output size.
