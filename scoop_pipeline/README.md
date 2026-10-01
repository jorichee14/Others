# scoop_pipeline

Processing for the SCooP dataset (`../scoop_paper`): per-site reference maps
and fiducial boards, then per-sequence reference poses, geometry and release
files. Code lives here; bags, intermediates and the release tree live outside
git (see the plan in the paper's Release Structure section).

```
scoop/               library: everything reusable lives here
  bag.py             reading bags: scans, images, topic schemas (the only reader)
  bagwrite.py        writing bags: BagWriter (MCAP + metadata.yaml), write / write_raw
  rosmsg.py          building ROS messages: header, PointCloud2 (from a dtype), Imu, String
  ouster.py          Ouster packets -> frames / IMU / scans, clock fit, ouster_ros layout
  replay.py          the replay step: packets bag -> points bag, with remaps
  retime.py          sensor time -> capture time (retime_bag.py without ROS)
  zed.py             ZED SVO2 -> bag on capture time (wrapper replay + restamp)
  recording.py       one recording: raw bag -> decoded -> retimed -> GLIM, SVO -> zed
  session.py         one pass: every machine's bags, checked against record.yaml
  merge.py           several bags -> one, in log-time order (zstd optional)
  tftree.py          calibrated transforms added to a bag's TF tree (one parent per frame)
  timing.py          how evenly a topic is spaced: intervals, gaps, repeated frames
configs/recording.yaml  settings for processing a recording (topics, remaps, GLIM config)
configs/zed/         the robot's zed_wrapper config, used to replay SVOs
configs/record.yaml  what each machine records (the robots' record.yaml)
configs/static_tf.yaml  calibrated transforms the merge adds to /tf_static
environment.yml      conda env for all offline processing
scripts/             command lines only: argument parsing around scoop/
  process_recording.py  raw recording -> decoded, retimed, glim, zed (the usual entry point)
  decode_ouster.py   packets bag -> points bag (replaces the ouster_ros replay)
  retime.py          points bag -> retimed bag (same arguments as retime_bag.py)
  run_glim.sh        GLIM (docker) on a bag, dump next to the bag, owned by you
  svo_to_bag.py      one SVO2 -> bag on capture time (the zed step alone)
  merge_session.py   check a pass's bags against record.yaml, merge them into one
  topic_timing.py    intervals, gaps and repeated frames of topics in a bag
  run_zed.sh         ZED wrapper replay + ros2 bag record (here or in the isaac_ros container)
  bag_check.py       verify + time the fast decoder on a real bag
  ouster_check.py    verify packet decoding against a replayed/retimed bag
  env_check.py       is this environment ready?
  test_bag.py        self-tests: reading, writing, messages (no bag needed)
  test_ouster.py     self-tests: packets, clock, replay, remaps (needs ouster-sdk)
  test_zed.py        self-tests: restamp, run_zed.sh with a fake ros2 and docker
  test_merge.py      self-tests: machines, topic check, merge
```

## Environment

Two environments, each for its own kind of work:

| | conda env `scoop` (`environment.yml`) | ROS 2 install |
|---|---|---|
| for | decoding, Ouster packets, map building, refinement, boards, evaluation, figures | recording, `ouster_ros` replay, GLIM, `ros2 bag convert` (merge.yaml), `retime_bag.py` |
| Python | 3.11 from conda | the system's (3.10 on Humble) |

```
cd scoop_pipeline
conda env create -f environment.yml
conda activate scoop
python scripts/env_check.py        # versions, cv2.aruco, open3d, Ouster SDK, GPU
python scripts/test_bag.py && python scripts/test_ouster.py
```

After `environment.yml` changes: `conda env update -f environment.yml --prune`.
For CUDA 11 machines, swap `cupy-cuda12x` for `cupy-cuda11x` first
(`nvidia-smi` shows the version); without a GPU, stage 01 runs on the CPU.

Keep them in separate terminals: `source /opt/ros/<distro>/setup.bash` puts
ROS's Python packages on `PYTHONPATH`, where they shadow the env's.
`env_check.py` warns when that has happened.

## Bag decoding (`scoop/bag.py`)

Built on `../rosbag_to_opv2v`'s `BagReader` (MCAP without ROS, `.db3` via
rosbags). What it adds over the stage scripts' own readers:

- **select before decode**: the header stamp is read from the raw CDR bytes and
  passed to `select(stamp_ns)`; scans without a trajectory pose within
  `time_tol` are never deserialised.
- **direct CDR parsing** of `PointCloud2`, `Image`, `CompressedImage`, with a
  strided zero-copy xyz view for the usual float32 x/y/z layout. Anything else
  falls back to the generic decoder.
- **per-point time** as float64 seconds relative to the header stamp, unit
  detected from the sweep span, same mask as the points.

```python
from scoop import bag
reader = bag.open_bag(cfg["dataset"]["bag"])
topic = bag.detect_points_topic(reader, n_poses=len(traj_t))
for scan in bag.iter_scans(reader, topic, select=bag.nearest_pose(traj_t, tol),
                           min_range=lo, max_range=hi, with_time=True):
    T = traj_T[scan.key]          # scan.xyz (N,3) f32, scan.t (N,) s or None
for fr in bag.iter_images(reader, image_topic, mode="gray", period=0.2):
    ...                           # fr.stamp, fr.image
```

Before using it on a new sensor or driver version:

```
python3 scripts/test_bag.py
python3 scripts/bag_check.py <bag> [--points /ouster/points] [--image <topic>]
```

`bag_check.py` compares the fast path with the generic decoder field by field
on the first 50 messages (exits non-zero on any mismatch) and times both.
On a synthetic uncompressed MCAP of 128x1024 Ouster scans (48-byte points,
6.3 MB each): 67 msg/s for the scripts' decode + extract + range gate, 235
msg/s for `iter_scans` with the same gate, file reading included. Reading
alone runs at ~630 msg/s, so that's the upper limit for a single process.

## Processing a recording

```
python scripts/process_recording.py ~/workspaces/isaac_ros-dev/data/raw/20260924/mapping_A/mobile_1
```

reads the raw recording and writes everything derived to the mirrored folder
below `data/work/`, always under the same names:

```
data/raw/20260924/mapping_A/mobile_1/      as recorded; never written to
    mirc_dataset_survey_1_mapping_20260924_mobile_1/   (the packets bag)
    *.svo2
data/work/20260924/mapping_A/mobile_1/
    mirc_dataset_survey_1_mapping_20260924_mobile_1_decoded/   1  points bag, sensor time
    mirc_dataset_survey_1_mapping_20260924_mobile_1_retimed/   2  capture time; the GLIM input
    glim/          3  GLIM dump: traj_lidar.txt, map
    <svo name>_zed/        4  the SVO2 as a bag, capture time (zed_logs/ inside)
    clock.json        the clock fit (drift, residual, packets) for the paper
    process.yaml      the settings the steps ran with
```

Settings come from `configs/recording.yaml` (`--settings` for another file).
The two bags are named `<original bag>_decoded` and `<original bag>_retimed`
(the .mcap inside too), so a copied bag still says where it came from.
A folder with an SVO but no Ouster packets bag (e.g. an older recording)
does the zed step alone; `--only X` runs just step X. Steps already done are
skipped. `--redo X` redoes X and the steps built from
it: `--redo retimed` redoes retimed and glim, `--redo zed` only zed.
`--until retimed` stops before GLIM; `--status` shows what is done. The
decoded bag may be deleted once retimed and glim exist: it is then skipped
("not needed"), and rebuilt only when retimed is redone. A step is
built in `.partial/<step>/` and only moved into place when it succeeded, so a
step folder is always complete. Steps 1-2 run in the `scoop` env, step 3
starts docker (`scripts/run_glim.sh`), step 4 needs the ZED wrapper
(below).

## Ouster packets: decode, then retime

```
original bag (<ns>/lidar_packets, imu_packets, metadata, other topics)
   |  python scripts/decode_ouster.py <original> <decoded> [--copy-all] [--remap ...]
   v
decoded bag, sensor time (<out-ns>/points, imu, metadata + copied topics)
   |  python scripts/retime.py <original> <decoded> <retimed> /ouster /mobile_1/ouster
   v
retimed bag
   |  scripts/run_glim.sh <retimed>                                 docker
   v
<folder of the bag>/glim_dump/traj_lidar.txt  -> stage 01
```

Both steps run in the `scoop` env. `scripts/retime.py` does what
`retime_bag.py` does, with the same arguments; it rewrites the header stamp in
the message bytes instead of going through rclpy, so it needs no ROS. On
synthetic bags (sensor clock from boot and PTP) its output equals
`retime_bag.py`'s message for message -- topic, log and publish time,
sequence, bytes -- and metadata.yaml is identical.

`decode_ouster.py` replaces `ros2 launch ouster_ros replay.launch.xml` +
`ros2 bag record`: same topics, same organized cloud layout (x y z intensity
t reflectivity ring ambient range), IMU in m/s^2 and rad/s, stamps in sensor
time, `metadata.yaml` included, without waiting for real time. `retime_bag.py`
also still runs on its output unchanged.

```
python scripts/decode_ouster.py <original bag> <decoded bag dir> \
    --packets-ns /ouster --out-ns /mobile_1/ouster --driver-min-range 1.30
```

`--limit 200` stops after 200 scans, for a quick look. `--remap SRC:=DST`
copies a topic of the original bag into the output under a new name, byte for
byte (what `ros2 bag play --remap` did in the replay launch). The replay
launch's set:

```
    --remap /ouster/metadata:=/mobile_1/ouster/metadata \
    --remap /ouster/imu_packets:=/mobile_1/ouster/imu_packets \
    --remap /ouster/lidar_packets:=/mobile_1/ouster/lidar_packets \
    --remap /zed/zed_node/imu/data:=/mobile_1/zed/imu/data \
    --remap /zed/zed_node/imu/mag:=/mobile_1/zed/imu/mag
```

`--copy-all` copies every other topic of the original too (tf, tf_static,
radar, ntp, ...) under its own name, unless a `--remap` renames it;
`--exclude TOPIC` leaves one out. Copied topics keep their QoS from the
original metadata.yaml, so /tf_static stays latched when the bag is played.
Retime passes them all on (times shifted by the same constant as the replay),
so the retimed bag holds the whole recording. `configs/recording.yaml` has
`copy_all: true`.

From Python (a later pipeline stage calls the same function):

```python
from scoop import replay
replay.decode_ouster_bag(original, out_dir, packets_ns="/ouster", out_ns="/mobile_1/ouster",
                         driver_min_range=1.30,
                         remap=replay.parse_remaps(["/zed/zed_node/imu/data:=/mobile_1/zed/imu/data"]))
```

## Ouster packets in Python (`scoop/ouster.py`)

The mapping bags record the Ouster as `<ns>/lidar_packets` + `<ns>/metadata`.
Instead of `ros2 launch ouster_ros replay.launch.xml` (real time) followed by
`retime_bag.py`, the packets are decoded directly from the original bag:

```python
from scoop import bag, ouster
reader = bag.open_bag(packets_bag)
clock = ouster.fit_clock(reader, "/ouster")        # retime_bag's clock fit, ported
for scan in ouster.iter_ouster_scans(reader, "/ouster", clock=clock,
                                     driver_min_range=1.30, frame="lidar"):
    ...                     # same Scan as bag.iter_scans: stamp_ns, xyz, t
```

- stamp: first non-zero column timestamp (sensor clock) mapped to host capture
  time by the same lower-envelope fit `retime_bag.py` uses;
- per-point `t`: column timestamp minus the stamp; points destaggered, in
  the cloud's row-major order;
- `driver_min_range`: the replay's `min_range`, applied to the RANGE field;
- `frame`: `lidar` (os_lidar) or `sensor` (os_sensor).

Which conventions the replayed clouds used is checked, not assumed:

```
python3 scripts/ouster_check.py <packets_bag> --points-bag <retimed bag> \
    [--packets-ns /ouster] [--points-ns /mobile_1/ouster] [--driver-min-range 1.30]
```

It prints the clock fit (compare with `retime_bag.py`'s output), the decode
speed as a multiple of real time, and per scan: stamp difference, point
count, max xyz difference in both frames, and max per-point time difference.
It ends with `MATCH in the <frame> frame` or `NO MATCH`.

## ZED SVO2 -> bag

Step 4 of `process_recording.py`, or alone:

```
python scripts/svo_to_bag.py <file.svo2> <out bag dir>
```

Reading an SVO2 needs the ZED SDK, so the SVO is played through the ZED ROS 2
wrapper and recorded (`scripts/run_zed.sh`):

```
ros2 launch zed_wrapper zed_camera.launch.py camera_model:=zed2i svo_path:=<svo>
    + svo.use_svo_timestamps: true, svo.svo_loop: false, svo.svo_realtime: false
ros2 bag record <zed.topics>          stopped when the wrapper logs "SVO reached the end"
```

recorded in 4 GiB pieces, then restamped (`scoop/zed.py`) piece by piece,
each deleted once read, so the disk needs about one copy of the bag, not two:
log time = header stamp (the SVO's capture
time, the recording PC's clock, as in the raw bag), topics renamed
`/zed/zed_node/...` -> `/mobile_1/zed/...`, QoS kept. The result is on the
same timeline as the retimed bag. The wrapper runs with the robot's own
wrapper config (`configs/zed/common_stereo.yaml`, `configs/zed/zed2i.yaml`:
HD1080 at 15 Hz, NEURAL_PLUS depth, 0.3-10 m, ...), so the replay matches the
live settings; the SVO keys above go on top of it, then `params`. Topics,
camera model and these files are the `zed` section of
`configs/recording.yaml`. `zed.bags` lists one bag per replay; by default
one, `zed`, with the paper's `mobile_1/zed` topics (`tables/topics.tex`: left
image, depth, odom, pose, paths; the IMU is already in the raw bag) and the
right camera. About 100 GB for an 18-minute recording (960x540 at 15 Hz,
stored uncompressed). The
point cloud is left out: it is depth + left camera_info, ~145 GB more.
`svo_to_bag.py --bag zed_right` makes one of them alone. The wrapper log and the parameters used are kept in
`<out>/zed_logs/`.

Where to run: in any terminal. When `ros2` + `zed_wrapper` are not installed
there, `run_zed.sh` runs itself in the running isaac_ros container
(`ZED_CONTAINER`, default `isaac_ros_dev-x86_64-container`) with `docker exec`;
`~/workspaces/isaac_ros-dev` (`ISAAC_ROS_WS`) is `/workspaces/isaac_ros-dev`
in there, so the SVO and the output must be below it (data/raw and data/work
are). Start the container first (`run_dev.sh`).

## Checking and merging a pass

```
python scoop_pipeline/scripts/merge_session.py ~/workspaces/isaac_ros-dev/data/raw/20260924/mapping_A --check
python scoop_pipeline/scripts/merge_session.py ~/workspaces/isaac_ros-dev/data/raw/20260924/mapping_A
```

A pass folder holds one folder per machine (`mobile_1/`, `mobile_2/`,
`infra_1/`, ...). Each bag's machine comes from its name
(`..._<machine>`, `_r2` repeats included) and `configs/record.yaml` -- the
robots' own file -- says what it must contain, like `record.sh`'s preflight:
every topic listed must be there and not empty (`MISSING`, `EMPTY`),
except what the pass mode does not record (`check:` in `record.yaml`:
mapping has no wifi/iperf/ntp/diagnostics; the mode comes from the pass
folder's name, `--mode` sets it). A
machine contributes its processed bags where they exist -- mobile_1: the
retimed bag and the ZED bags, checked for the recorded topics renamed plus
the decoded ones -- and its raw bags otherwise. A new machine (mobile_2,
infra_N) needs only its entry in `record.yaml` and its folder in the pass.

`--check` stops after the check. Otherwise, if nothing is missing (or with
`--allow-missing`), every bag is merged in log-time order -- what
`ros2 bag convert` with `all_topics: true` did, without ROS -- into
`data/work/<date>/<pass>/<prefix>_<pass>_..._<date>_merged/`, byte for byte,
QoS kept, and checked again. It is zstd-compressed inside the MCAP by default
(`ros2 bag play`, GLIM and scoop read it as is), so a pass of raw images
fits next to its inputs; it stops and removes the partial bag before the
disk fills. `--out` puts it on another drive, `--name` names it.

The merge also adds the calibrated transforms of `configs/static_tf.yaml`
(e.g. `os_lidar -> zed_left_camera_optical_frame`) to `/tf_static`. All
static transforms -- the inputs' (Ouster, ZED) and these -- go into ONE
latched `/tf_static` message at the start of the bag, as tf2's static
broadcaster does: `ros2 bag play` replays only the last message of a latched
topic to a late subscriber (rviz, tf2_echo), so separate messages would lose
all but one set of transforms. A frame has one parent in TF, and the ZED's
optical frame already has one (its camera chain), so the merge then adds the
same geometry as `zed_left_camera_optical_frame -> os_sensor` (the root of the
Ouster tree), composed through `os_sensor -> os_lidar`: looking up
`os_lidar -> zed_left_camera_optical_frame` gives exactly the calibration.
It prints what it added; `--static-tf ''` adds nothing.

