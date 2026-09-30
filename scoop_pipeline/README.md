# scoop_pipeline

Processing for the SCooP dataset (`../scoop_paper`): per-site reference maps
and fiducial boards, then per-sequence reference poses, geometry and release
files. Code lives here; bags, intermediates and the release tree live outside
git (see the plan in the paper's Release Structure section).

```
scoop/          library imported by every stage
  bag.py        bag decoding: the only module that opens a bag
  ouster.py     Ouster lidar_packets -> scans offline (no ROS replay)
environment.yml   conda env for all offline processing
scripts/
  env_check.py    is this environment ready?
  test_bag.py     self-tests (no bag needed)
  test_ouster.py  self-tests for packet decoding (needs ouster-sdk)
  bag_check.py    verify + time the fast decoder on a real bag
  ouster_check.py verify packet decoding against a replayed/retimed bag
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

## Ouster packets (`scoop/ouster.py`)

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
