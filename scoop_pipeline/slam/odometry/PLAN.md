# Odometry and multi-robot SLAM on site_1 / site_2: the final plan

Every method passes five rules: odometry only (no loop closure of its own;
the multi-robot systems excepted), official ROS 2 Humble support, plausibly
real time on the Jetson AGX Orin, sensors recorded well on both sites, and a
question no other entry answers. Workstation runs log resources,
communication and latency now; the same scripts run on the Orin later.

## Methods

### LiDAR, both robots (Ouster, with its own IMU where used)

| method | input | question |
|---|---|---|
| KISS-ICP | LiDAR | the baseline: LiDAR alone |
| RKO-LIO | LiDAR + IMU | what the IMU adds (KISS-ICP's design, the IMU the only difference) |
| FAST-LIO2 | LiDAR + IMU | the standard tightly coupled filter; ties the numbers to published ones |

The better of RKO-LIO and FAST-LIO2 is Swarm-SLAM's odometry.

### Camera, mobile_2 (D455, coop passes: the survey passes skip the infra pair)

| method | input | question | gate |
|---|---|---|---|
| cuVSLAM | infra stereo (+ IMU as an ablation) | camera odometry, the primary | IMU-on run: the measured IMU noise |
| OpenVINS | infra stereo + IMU, the same input as cuVSLAM | is a camera result the camera's or cuVSLAM's | the measured IMU noise (Allan variance); refuses without it |
| RTAB-Map RGB-D odometry | colour + depth, no loop closure | a calibration-free fallback | N >= 3 runs (not reproducible run to run); depth is weak with the emitter off |

### ZED, mobile_1 (only once it records at its nominal rate; 20261015 is ~1 fps)

| method | question |
|---|---|
| cuVSLAM on the ZED stereo | camera vs LiDAR on one rigid body, nothing else varying |
| cuVSLAM IMU on / off | what the IMU adds to a camera |
| ZED SDK positional tracking (`/zed/zed_node/odom`, recorded live) | is the vendor's own tracking enough |

Next recording: SVO H.264/H.265 or HD720 at 15-30 fps; record `/zed/zed_node/odom`.

### Multi-robot (LiDAR)

| entry | question |
|---|---|
| floor: each robot alone, one offline map registration | the least a joint system must beat |
| Swarm-SLAM + the chosen LIO odometry | sparse exchange: accuracy for little data |
| mrg_slam | dense exchange (whole graphs) |
| ceiling: the maps merged with the true inter-robot transform | the most collaboration can give |

## Not in the plan, and why

| method | reason |
|---|---|
| iG-LIO, Voxel-SLAM | no official ROS 2; iG-LIO repeats FAST-LIO2's question, Voxel-SLAM is full SLAM |
| 2Fast-2Lamaa | Jazzy or newer only; its strength (fast non-linear motion) is not these robots' |
| GLIM odometry | the reference is seeded by GLIM |
| Lab-map ICP | it is the reference; localisation in a prior map is a separate experiment |
| DCL-SLAM, DiSCo-SLAM, LAMP 2.0, Kimera-Multi | ROS 1, or (Kimera-Multi) needs the ZED |
| DROID-SLAM, MASt3R-SLAM | not real time on the Orin |

## Environments

| | contents |
|---|---|
| `scoop/odometry:humble` (new) | KISS-ICP, RKO-LIO, FAST-LIO2 (+ Livox-SDK2, livox_ros_driver2 to compile it), OpenVINS, RTAB-Map; each pinned to a commit |
| Isaac ROS container (exists) | cuVSLAM (`processing/vslam.py`, `run_vslam.sh`) |
| Swarm-SLAM container (exists) | `slam/run_swarm_slam.sh` on `datasets/swarm_slam_lidar.py` exports |
| `aserbremen/mrg_slam_humble` | mrg_slam |

## What gets built

1. The image above.
2. A config per method and robot from the bag: topics, beams, T_imu_lidar /
   camera-IMU from `/tf_static`; every indoor change named beside its
   published default; IMU noise `null` until measured (the methods that need
   it refuse).
3. `run_odometry.sh <method> <bag> <machine> <out> [--start --duration]`: the
   bag at 1x with `--clock`, the method on sim time, the estimate as TUM with
   its frame named (ready for `swarm_slam_lidar.py --odom-tum`).
4. The logger, the same for every method (and `run_vslam.sh` with one
   environment variable):
   - resources.csv: the method's processes (CPU % of one core, RSS), the
     machine, the GPU (nvidia-smi; tegrastats on the Orin, same columns);
   - messages.csv: every input and output message (wall time, stamp, bytes):
     Hz, size and MB/s per topic;
   - latency (input to output per scan/frame) and drops (inputs never
     answered);
   - summary.json: all of it, the CPU/GPU model, and whether it kept up.
5. A frame-rate gate for camera methods (`check.py spacing`): refuse under
   90 % of the nominal rate.
6. Self-tests without ROS for the parsers and the summary.

## Order

1. Build 1-6.
2. 60 s of coop site_1: every method runs and writes its files.
3. Record the D455 IMU standing still, 2-3 h; Allan variance -> OpenVINS and
   cuVSLAM's IMU run.
4. The full passes, both sites. `check.py trajs` (ATE, RPE against the
   reference) and the board residual beside the resource numbers, one row
   per method and robot.
5. Choose the odometry; Swarm-SLAM and mrg_slam with the floor and ceiling.
