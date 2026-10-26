# Odometry on a pass, with its cost logged

The methods of `PLAN.md`, each run on one robot of a pass the way it would run
live: the bag at 1x with `--clock`, the method on sim time, and beside it a
probe and a resource sampler (`../monitor/`). One run gives a trajectory and
what it cost.

| method | robot | input | runs in |
|---|---|---|---|
| `kiss_icp` | mobile_1, mobile_2 | Ouster | this image |
| `rko_lio` | mobile_1, mobile_2 | Ouster + its IMU | this image |
| `fast_lio` | mobile_1, mobile_2 | Ouster + its IMU | this image |
| `openvins` | mobile_2 | D455 infrared pair + IMU (needs `--imu-noise`) | this image |
| `rtabmap_rgbd` | mobile_2 | D455 left infrared + depth | this image |
| cuVSLAM | mobile_2 | D455 infrared pair (+ IMU) | the Isaac ROS container: `VSLAM_MONITOR=1 processing/vslam.py ...` |

## Build once

```bash
scoop_pipeline/slam/odometry/build.sh        # scoop/odometry:humble, ~3-4 GB, 20-40 min
```

Every source is pinned in the `Dockerfile`; the image's `/opt/versions.txt`
lists them and RTAB-Map's apt version.

## Run

```bash
conda activate scoop                          # make_config.py reads the bag
B=data/work/20261015/coop/site_1/mirc_dataset_coop_site_1_20261015_merged_final
R=data/processed/20261015/coop/site_1/odometry/methods

# 60 s first: does every method run and write its files
for m in kiss_icp rko_lio fast_lio; do for r in mobile_1 mobile_2; do
  scoop_pipeline/slam/odometry/run_odometry.sh $m $B $r $R/${m}_${r}_60s --start 10 --duration 60
done; done
scoop_pipeline/slam/odometry/run_odometry.sh rtabmap_rgbd $B mobile_2 $R/rtabmap_rgbd_mobile_2_60s --start 10 --duration 60

# then the whole pass (no --start / --duration)
```

`openvins` refuses until `--imu-noise noise.yaml` gives the D455 IMU's Allan
variance (`accelerometer_noise_density`, `accelerometer_random_walk`,
`gyroscope_noise_density`, `gyroscope_random_walk`): record the IMU standing
still for 2-3 h first. cuVSLAM: `VSLAM_MONITOR=1` in front of the usual
`processing/vslam.py` call, at rate 1.

A slice stops the player after its length in wall time (Humble's `ros2 bag
play` has no `--playback-duration`), so it is that long to about a second.
Repeat RTAB-Map at least 3 times: it is not reproducible run to run.

## What a run writes

| file | |
|---|---|
| `run.json` | the method, robot, topics played, the launch line, the output topic, the frame of `trajectory.tum`, every setting changed from its published default and why, the key topic's rate in the bag |
| `config/` | the method's config files, made from the bag (topics, beams, `/tf_static` extrinsics, camera_info intrinsics) |
| `trajectory.tum` | the estimate on the bag's clock, of `run.json`'s `tum_frame` (FAST-LIO2 and OpenVINS: the IMU; the others: the LiDAR or the left camera) |
| `messages.csv` | every input and output message: when its publisher sent it, when it arrived at the probe, stamp, bytes |
| `resources.csv`, `host.json` | the method's processes (CPU, 100 = one core; its own memory, RssAnon, and everything resident, RSS), the machine, the GPU (nvidia-smi; tegrastats on a Jetson) every 0.5 s; CPU and GPU model |
| `accuracy.json` | `score.py`, against stage 04's reference (`traj_<machine>_lidar.tum`), the estimate first put in the LiDAR frame: ATE after a rigid fit (cm, deg), RPE over 1 / 10 / 100 s, drift over 5 / 10 m of travel (% of it), the error at the end with the start aligned, path-length ratio, coverage and gaps, jumps over 3 m/s; also in `summary.json` |
| `summary.json` | per topic Hz, KB, MB/s; latency per input, from the player sending it to the method sending its answer (median, p95, max), drops, whether it kept up (drops <= 1 % and p95 latency <= the input period); CPU mean / p95 / max, peak memory, GPU, power |
| `status.json`, `logs/` | whether the method ran through and the play finished; each process's log |

Scoring against the reference is `check.py trajs` with `trajectory.tum`; for
Swarm-SLAM, `datasets/swarm_slam_lidar.py --odom-tum <machine> trajectory.tum <tum_frame>`.

## Scoring a run again

```bash
python scoop_pipeline/slam/odometry/score.py <run dir> [--ref <T_map_lidar TUM>]
```

The reference is the pipeline's own output: an ATE near its uncertainty (about
1-3 cm) is agreement with it, not accuracy beyond it.

## When a LiDAR method gets its cloud

The pass bags are retimed: a cloud is replayed at its header stamp, the first
point of its sweep, a sweep before a live driver could publish it. Played as
recorded, a LiDAR-inertial method waits that sweep for the IMU to cover the
cloud, and the wait is counted as its latency (RKO-LIO and FAST-LIO2 measured
~110-120 ms that way, KISS-ICP 9). `sweep_relay.py` takes the player's clouds
(renamed to `/replay/...`) and hands each on 0.1 s later, at the sweep's end;
the method and the probe listen to the relay.

## The transport

Every process in the container uses `fastdds.xml`: messages up to 64 MB go
through shared memory in one piece. KISS-ICP, RKO-LIO and FAST-LIO2 subscribe
best-effort (SensorDataQoS); with Fast DDS's default 512 KB shared-memory
segment a 3 MB Ouster cloud goes as fragmented UDP and most are lost before
the method sees them (the first KISS-ICP run: 44 % of scans; a test: 98 of
100). A robot running these methods live needs the same profile, or the
driver and the method in one process.

## Settings changed from the published defaults

Listed per run in `run.json`; in short:

- T_lidar_imu from the bag's `/tf_static`; where the pass bag does not link a
  robot's Ouster LiDAR and IMU (mobile_2 on 20261015), from its GLIM config
  (`configs/glim/<machine>/config_sensors.json`, which `processing/glim_config.py`
  took from that robot's own bag); `run.json` names the source;
- every LiDAR method leaves out points within 1.0 m (RKO-LIO's default;
  KISS-ICP's 0.0 keeps the robot and the operator, FAST-LIO2's 4.0 m is most of
  the room), so KISS-ICP and RKO-LIO differ in the IMU alone; KISS-ICP deskews;
- every method publishes its odometry only (no debug clouds, maps, paths);
- OpenVINS: both infrared cameras (its rs_d455 example is the colour camera
  alone), this IMU's measured noise;
- RTAB-Map: the left infrared image with the depth computed in it (no
  registration; the emitter was off, so the depth is weak on bare walls), a
  reliable subscription.

## Status

The harness (probe, sampler, summary, `inside.sh`'s start / play / slice / stop,
the cuVSLAM hook) has been run end to end in `ros:humble-ros-base` with a
stand-in method; `tests/test_odometry.py` covers `make_config.py` and the
summary. The image has not been built and the five methods have not been run
on a pass yet.
