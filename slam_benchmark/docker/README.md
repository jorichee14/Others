# Containers

One image per method. The evaluator never imports a method's code and no method
ever imports the evaluator: the whole interface is three files in `/out`.

## The contract

A container is given:

| | |
|---|---|
| `/bag` | the recording, read-only |
| `/out` | a writable run directory |
| `$SLAM_TOPICS` | comma-separated topics for this stream — **and nothing else is replayed** |
| `$SLAM_RATE` | bag replay rate (1.0 unless a table says otherwise) |
| `$SLAM_PARAMS` | the method config's `params` block as JSON |
| `$SLAM_MODALITY` | `lidar` / `rgbd` / `radar`, from the stream |
| `$SLAM_DEPTH_TOPIC`, `$SLAM_COLOR_TOPIC`, `$SLAM_*_INFO_TOPIC`, `$SLAM_CLOUD_TOPIC` | whichever the stream declares |
| `$SLAM_DEPTH_SCALE`, `$SLAM_RANGE_MIN`, `$SLAM_RANGE_MAX` | the depth stream's units and stated range |
| `$SLAM_IMU_TOPIC` | only for methods that declared `needs_imu` |

and must write:

| | |
|---|---|
| `/out/trajectory.tum` | `timestamp tx ty tz qx qy qz qw`, seconds, metres, **on the bag's clock** |
| `/out/map.ply` | the dense map, if the method makes one, in the same world frame as the trajectory |
| `/out/timing.json` | `{frames, wall_s, cpu_s, peak_rss_mb}` |

Three rules, and each one exists because breaking it produces a run that
completes and is wrong:

1. **Poses describe the sensor frame the method was given, in the method's own
   world frame.** Do not apply an extrinsic. Do not try to output `map`. The
   evaluator does both, from the dataset config, identically for every method —
   a container that "helpfully" pre-transforms is a container whose extrinsic is
   applied twice and nothing downstream can see it.
2. **Stamps are the bag's, not the wall clock.** Replay with `--clock` and run
   every node with `use_sim_time: true`. Wall-clock stamps still associate
   against the reference if the run happens to start near the bag's epoch, which
   is the worst possible failure mode.
3. **`--network none`.** A benchmark run that can reach the network is a
   benchmark run that can download a different checkpoint than the one the
   manifest records. Weights are baked into the image at build time.

## How an image plugs in

Each image ships **`/opt/slambench/launch.sh`**, an executable that starts the
method's nodes and blocks. The shared entrypoint starts the recorder, runs that
script, replays the bag, and shuts everything down.

It is a file and not an environment variable on purpose: a Dockerfile `ENV`
expands `${...}` at *build* time, so a launch line referring to a run-time
variable — the cloud topic, the IMU topic — baked in the empty string and the
method started subscribed to nothing. That shipped once; this is the fix.

An image also declares `SLAM_POSE_TOPIC` (required), `SLAM_MAP_TOPIC`, and
`SLAM_NEEDS_CLOUD=1` if it wants the depth→cloud bridge below.

A method that is **not a ROS node** — MASt3R-SLAM — replaces the entrypoint
entirely with its own `run.sh` and meets the same three-file contract by
unrolling the bag, running offline, and restamping the result.

### Shared pieces, in `docker/common/`

| | |
|---|---|
| `entrypoint.sh` | replay + record + shutdown |
| `record_tum.py` | pose topic → `trajectory.tum`, map topic → `map.ply` |
| `depth_to_cloud.py` | depth image → `PointCloud2`, for geometry-only estimators on a camera. Checks the declared `depth_scale` against the image encoding and **refuses** on a mismatch |
| `sniff_frame.py` | the `frame_id` off the first message of a topic, read from the bag, never assumed |
| `params_to_args.py` | `$SLAM_PARAMS` → RTAB-Map `--Group/Name value` and ROS `-p key:=value` |
| `bag_to_frames.py` | bag → PNG frames + real stamps + a calibration file, for the non-ROS methods |
| `restamp_tum.py` | puts the bag's clock back on a trajectory estimated from a folder |

Every one of these has self-tests in `scripts/test_slambench.py` that run
without ROS, a bag, a GPU or a network.

## Building

```bash
docker build -f docker/Dockerfile.base        -t slambench/base:humble .
docker build -f docker/Dockerfile.kiss-icp    -t slambench/kiss-icp:humble .
docker build -f docker/Dockerfile.rtabmap     -t slambench/rtabmap:humble .
docker build -f docker/Dockerfile.mast3r-slam -t slambench/mast3r-slam:humble .   # GPU, ~10 GB
```

Pin every method to a commit in its Dockerfile, not to a branch. `run.json`
records the image tag; the tag is only useful if it means one thing.
