# Stage 08 for mobile_1: lidar ICP, ZED odom, boards, joint

`08_reference_traj.py` goes next to the other stages (it imports
`pipeline_common` and `pipeline_boards`). Paste the `08_reference` block of
`pipeline_config_08_mobile1.json` into `pipeline_config.json` and run

    python3 08_reference_traj.py pipeline_config.json

## What runs, in order

| step | track | what it produces |
|---|---|---|
| 1 | `mobile_1_lidar` (`lidar_icp`) | every Ouster scan registered to the anchored map, board-free and (after scan 0) odometry-free: each scan is seeded from the lidar's own previous poses, the ZED increment is only a fallback seed, and a scan that fails both is retried with wide gates before being marked unregistered. `traj_mobile_1_lidar.tum` (lidar frame), `traj_mobile_1_lidar_in_cam.tum` (same track as the ZED left optical frame), `traj_mobile_1_lidar_odom_only.tum` (anchored ZED odom at the same stamps, lidar frame), `quality_mobile_1_lidar.csv/png` (per-scan plane rms, observable DOF, seed status, ZED step vs lidar step) |
| 1b | printed | **lidar ICP vs ZED odom**: translation/rotation gap at the same stamps, gap over time, end gap as % of path, and the per-step disagreement with the stamps of every step over 5 cm or 2 deg. One big step is a ZED jump; a run of them is the ZED losing scale or tracking. |
| 2 | `mobile_1_zed` (`arms`, `cloud_source: lidar`) | trajectories of the ZED optical frame from one pose graph: `A_icp` (ZED odom + lidar map factors), `B_boards` (ZED odom + board sightings + session anchor, **started from the odometry, nothing from the lidar** = "boards correct the ZED odom"; the graph distributes the drift measured at each board re-acquisition back over the odometry-only stretch before it, and prints that drift and the applied correction per stretch), `B_breaks` (same graph, but the odometry edges where the ZED step disagrees with the lidar step are freed; only appears when such breaks exist; the difference to `B_boards` is the value of knowing where the ZED broke), `C_joint` (lidar + boards + odom). Plus `traj_mobile_1_zed_odom_only.tum`. |
| 2b | printed | ablation table: board residual / map rms / vs C / vs odom per arm. A's board residual and B's map rms are the independent cells. |
| 3 | printed | cross-check (lidar track scored on the ZED's board sightings, both `T_lidar_camera` conventions), then the rig comparison table: lidar ICP, odom only, A, B, C pairwise, all in the camera frame at the lidar stamps. `compare_mobile_1.csv` has the gap to the lidar track over time. |
| always | `paths.png` | re-saved after every track: overlay of all methods over the map, gap to the lidar track over time (log scale), and one panel per method with the lidar track in grey behind it, start = circle, end = square. It exists even if a later track fails. |

The lidar clouds are reused for the graph (no second ICP pass). Only the
clouds are re-registered inside the graph; the chained lidar POSES are
initialisation only, never measurements (their errors are correlated and would
out-vote the boards).

## Transforms, per case

Notation: `T_a_b` = pose of frame b in frame a = maps b-points into a.

Frames on mobile_1:

* `map` - anchored map (stage 03 output; `ref_map` MUST be that cloud).
* `odom` / `child` - `/mobile_1/zed/odom` header frame and `child_frame_id`.
  The script prints both. **Check the child.** The ZED wrapper publishes
  `zed_camera_link` unless a `base_frame` is set, in which case it is the
  robot base. `cam_extrinsic_xyzquat` must be `T_child_optical` for whatever
  is printed:

      ros2 run tf2_ros tf2_echo <child_frame_id> zed_left_camera_optical_frame

  The value in the config, `[-0.010, 0.060, 0.015, -0.5, 0.5, -0.5, 0.5]`, is
  the ZED2i URDF chain `zed_camera_link -> zed_camera_center (z +0.015) ->
  zed_left_camera_frame (y +0.060) -> optical (rpy -90, 0, -90)`. If the
  child turns out to be a `base_link` the translation is wrong by the mount
  offset; the rotation is still `[-0.5, 0.5, -0.5, 0.5]` only if the base is
  x-forward/z-up and not tilted. Replace with the tf2_echo output.
* `cam` - `zed_left_camera_optical_frame`: what the stage-06 anchor and every
  board detection refer to. **State frame of the arms.**
* `lidar` - the `frame_id` of `/mobile_1/ouster/points` (printed). Ouster
  publishes in `os_lidar` by default, which is `os_sensor` yawed 180 deg and
  shifted ~36 mm. `T_lidar_camera` in `calibration.json` has to be for the
  frame the points are in; stage 01 built the map from the same topic with
  the same calibration, so this is consistent as long as the topic is the same.

Chains used:

| case | chain |
|---|---|
| anchor odometry | `T_map_odom = A @ inv(T_odom_child(t_dwell) @ X)` with `A` = stage-06 `map_to_cam`, `X` = `cam_extrinsic_xyzquat`, evaluated at the dwell end stamp, not at index 0 |
| ZED odom -> camera pose | `T_map_cam(t) = T_map_odom @ T_odom_child(t) @ X` |
| ZED odom -> lidar pose | `T_map_lidar(t) = T_map_odom @ T_odom_child(t) @ T_cl`, `T_cl = X @ inv(T_lidar_camera)` |
| lidar ICP seed, scan 0 | `T_map_lidar` from the line above |
| lidar ICP seed, scan k (`seed: lidar`, default) | `T_prev @ scale(inv(T_prev2) @ T_prev, dt_k / dt_{k-1})` : the lidar's own last motion, extrapolated |
| lidar ICP seed, scan k (`seed: odom`, or fallback) | `T_prev @ inv(T_cl) @ inv(T_odom_child(k-1)) @ T_odom_child(k) @ T_cl` (odometry increment conjugated from child into lidar frame) |
| deskew | same conjugation on the motion over one scan |
| lidar pose -> camera pose | `T_map_cam = T_map_lidar @ T_lidar_camera` |
| lidar cloud -> camera frame (graph factors) | `P_cam = inv(T_lidar_camera) P_lidar` |
| odometry factor in the graph | `Z = inv(X) @ inv(T_odom_child(i)) @ T_odom_child(i+1) @ X`; an edge whose `Z` disagrees with the lidar chain's own increment by more than `odom_jump_m` / `odom_jump_deg` gets its sigma multiplied by 1000 (a ZED jump becomes a free joint instead of being smeared over its neighbours) |
| board factor | `T_map_cam = T_map_board(survey) @ inv(T_cam_board)` |
| lidar-vs-odom comparison | both in the lidar frame at the scan stamps |
| rig comparison table | everything in the camera frame at the lidar stamps |

`T_lidar_camera` is taken as the pose of the camera in the lidar frame. The
cross-check scores the lidar track on the ZED's board sightings with both
`T_lidar_camera` and its inverse: the one that comes out at centimetres is
right. If the inverse wins, set `"invert_T_lidar_camera": true` and rerun.

## Reading the result

* **lidar vs odom gap** grows with time: ZED drift. A step that then stays
  flat: a ZED jump (the per-step line names its stamp; the lidar chain does
  not follow it because it is seeded from itself). A constant offset from
  t=0: the anchor vs the map, or `X`. Unregistered scans are listed with
  their stamps; poses there are extrapolated, not measured.
* **arm B's per-stretch lines** are the drift the odometry accumulated
  between two sighting groups (measured at re-acquisition) and the
  correction the graph distributed over that stretch. A stretch whose drift
  is metres or tens of degrees over a few metres of path is a ZED tracking
  break, not drift, and a pure odometry+boards graph can only smear it over
  the stretch; `B_breaks` shows the same graph with the break located.
* **lidar ICP vs arm B** (two estimates of one body, no shared information)
  is the honest accuracy statement. Constant offset = `T_lidar_camera` or `X`
  error; growth between sightings = ZED drift the boards could not reach.
* **arm C** should match or beat both A's board residual and B's map rms.

## Test

`tests/test_08_synthetic.py` runs the same functions (`chain_lidar`,
`run_arms`, `compare_rig`, `save_paths_png`) on a synthetic room with a
body-frame odometry child, a non-trivial `T_lidar_camera`, a drifting ZED
odometry (65 cm over 12 m) and board sightings. Result: lidar chain 0.1 cm
from truth, B 2.8 cm max, A and C 0.5 cm max.

# mobile_2 (RealSense D455 + Isaac visual SLAM, no lidar)

`pipeline_config_08_mobile2.json` keeps the two mobile_1 tracks with
`"enabled": false` (frozen, outputs untouched in `reference_coop2_mobile1`)
and adds `mobile_2_rs`, an `arms` track with `"cloud_source": "depth"`,
writing to `reference_coop2_mobile2`. Same workflow, one substitution: the
geometry-only reference is the **chained depth ICP** of the D455 against the
map instead of the Ouster.

| step | what | outputs |
|---|---|---|
| 1 | chained depth ICP: every depth frame (10 Hz) is deprojected, range-gated 0.4-3.5 m, edge-rejected, voxelised, and registered to the map. Seeded from the Isaac odometry increment (`"seed": "odom"`; the constant-velocity seed is the fallback), damped toward the seed along unobservable axes (`prior_beta` 0.10). | `traj_mobile_2_rs_depth_icp.tum` (color optical frame), `quality_mobile_2_rs_depth.csv/png` |
| 1b | depth ICP vs anchored Isaac odometry, per-step disagreement | printed |
| 2 | the same A/B/C graph: `A_icp` odom + depth map factors, `B_boards` odom + boards, `C_joint` all. `C` starts from `B` here (`joint_init`): depth ICP can only refine, it cannot relocalise a metre-scale error. `odom_jump_check` is off: a depth chain is not reliable enough to indict the odometry. | `traj_mobile_2_rs_{A_icp,B_boards,C_joint,odom_only}.tum` |
| 3 | comparison table and `paths_mobile_2.png` with the depth chain as the reference | `compare_mobile_2.csv` |

Read the depth chain with more suspicion than the lidar: the D455 sees an
87-degree cone to 3.5 m. Facing a flat wall constrains one axis, a corridor
two. The quality PNG's middle panel (observable DOF) says where. Where it is
below 6 the chain's pose along the missing axes is the odometry seed, so
"depth ICP vs odom" is small there by construction, not by measurement. The
boards are what bound those stretches; `A_icp`'s board residual (A never
used a board) is the honest accuracy number for the depth-only route.

## mobile_2 transforms

| item | value / rule |
|---|---|
| odometry | `/mobile_2/visual_slam/tracking/odometry`, child frame printed at run time |
| `X` = `cam_extrinsic_xyzquat` | `T_child_color_optical`. The config value `[0.0003, 0.0592, -0.0002, 0.4996, -0.4974, 0.5017, -0.5012]` is derived as `R_optical @ inv(T_color_depth)` from your `depth_extrinsic_xyzquat`, which is right ONLY if the odometry child is the RealSense `camera_link` (the depth/left-IR body frame, what Isaac VSLAM uses when `base_frame` = `camera_link`). If the printed child is a `base_link`, replace it with `ros2 run tf2_ros tf2_echo <child> camera_color_optical_frame`. |
| `Xd` = `depth_extrinsic_xyzquat` | `T_color_depth` (pose of the depth frame in the color optical frame): depth clouds are moved into the color frame with it, and the depth chain's state (depth frame) is converted to the color frame with its inverse |
| depth chain seed | `T_map_depth = T_map_odom @ T_odom_child(t) @ X @ Xd` at frame 0, then the odometry increment conjugated by `X @ Xd` |
| anchor | stage-06 `realsense` `map_to_cam` (color optical frame) at the dwell end |

# mobile_1 with ZED depth (`pipeline_config_08_mobile1_zeddepth.json`)

The case where the ZED's own depth is the range sensor. Runs the lidar track
(reference) and `mobile_1_zeddepth`: an `arms` track with
`"cloud_source": "depth"` on `/mobile_1/zed/depth/depth_registered`
(32FC1 metres, already registered to the left rectified camera, so
`depth_extrinsic_xyzquat` is null), ZED odometry, boards. Outputs go to
`reference_coop2_mobile1_zeddepth`; the frozen `mobile_1_zed` (lidar clouds)
track stays disabled. In `paths_mobile_1.png` the ZED depth chain and its
arms `[depth clouds]` are all measured against the lidar ICP.

Settings that differ from mobile_2 and why: `cv_must_agree_with_odom` is
false because this ZED odometry breaks at 45-49 s and the constant-velocity
fallback must be allowed to disagree with it there; `icp_init` is `chained`
because a 95-degree-broken odometry is a worse start for arm A than the
depth chain.

# mobile_1, the five-case run (`pipeline_config_08_mobile1_all.json`)

One run, five curves, all against the lidar, into `reference_coop2_mobile1_all`:

| case | curve in `paths_mobile_1.png` | what it uses |
|---|---|---|
| lidar ICP | `mobile_1_lidar lidar ICP` (reference) | Ouster scans + map |
| pure ZED odom | `mobile_1_lidar odom only` | ZED odometry + session anchor |
| ZED odom + boards | `mobile_1_zed B_boards` | ZED odometry + boards, nothing else |
| ZED pcl ICP | `mobile_1_zed depth ICP chained` | ZED depth clouds + map, ZED odometry only as the seed |
| ZED depth + boards | `mobile_1_zed C_depth [depth clouds]` | ZED odometry + boards + ZED depth clouds; no lidar anywhere, not even the break stamps |
| joint | `mobile_1_zed C_joint [lidar+depth clouds]` | ZED odometry + boards + Ouster clouds (2 cm) + ZED depth clouds (5 cm) in one graph |

Per-source arms available in `arms_run` when a track has both cloud sets:
`A_lidar`, `A_depth` (odometry + that sensor's clouds), `C_lidar`, `C_depth`
(the same plus boards).

`cloud_source: "lidar+depth"` feeds both cloud sets into the graph with
their own sigmas (`icp_sigma_lidar`, `icp_sigma_depth`); `arms_run` limits
the graph to B and C so the output has exactly these cases.

# Depth submaps (`submap_window_s`, default 3.0)

Single depth frames leave at least one axis unconstrained (100% of frames on
this bag). The depth chain therefore no longer registers single frames: every
frame within +-1.5 s of a centre frame is moved into the centre frame with
the odometry's relative motion and stacked (`build_submaps`), and that
submap is registered to the map, then used as the depth map factor in the
graph. A submap taken while the robot turns has seen several directions and
pins the axes a single frustum cannot. The odometry only has to be right
over 3 s for the stitch to hold; a submap spanning a tracking break is
internally inconsistent and shows up as an unregistered centre. Set
`submap_window_s` to 0 for the old single-frame behaviour;
`submap_max_pts` and `submap_stride` bound the cost.

Measured on the coop2 bag: 3 s submaps stitched with the mobile_1 ZED
odometry lost 612 of 1146 centres (the odometry jitters 3 cm per 0.1 s at
the 95th percentile and breaks at 45-49 s), so the mobile_1 configs use
`submap_window_s: 0`. The mobile_2 Isaac odometry (45 cm over 19 m, no
breaks) is good enough to stitch and keeps 3.0.

# `pf` track: the camera-only localiser (both platforms)

A 2D particle filter over the map, the estimator that does not depend on the
odometry never breaking:

* map: points of the anchored cloud in the height band `slice_z` (map z,
  default -0.5..1.2, i.e. walls at sensor height, no floor/ceiling)
  rasterised at `grid_res`, distance-transformed into a likelihood field
* particles: (x, y, yaw) of the LEVEL odometry-child frame; roll/pitch from
  the odometry, z from the session anchor
* motion model: the odometry increment between frames with noise `alpha`
* measurement: each depth frame levelled, height-banded, `scan_pts` points
  scored on the likelihood field (`lf_sigma`, `z_rand`)
* boards: absolute (x, y, yaw) fixes with `board_sigma_m` / `board_sigma_deg`;
  if no particle is near the fix the filter relocalises onto it
* recovery: augmented MCL - when the measurement likelihood collapses,
  a fraction of particles is re-drawn in free space near walls until a
  hypothesis matches again
* output: weighted mean of the cluster around the best particle, composed
  with X into the camera optical frame; `quality_<name>.csv` has the
  particle spread, mean likelihood, injection fraction and board-fix flag
  per frame

It is bounded (5-10 cm class), multi-modal in corridors, self-recovering
after a break, and its accuracy is measured against the lidar on mobile_1
like every other case. Between a break and the next distinctive geometry or
board it reports a large spread rather than a confident wrong pose.

# `odom_source: "imu_vo"` - the odometry replacement for camera-only agents

The platform tracker (ZED odom, Isaac VSLAM) is taken out of the loop and
replaced, inside the same graph, by:

* **heading from the gyro**: `imu_topic` integrated between nodes, bias from
  the first `gyro_bias_window_s` (the dwell), rotated into the camera frame
  with `imu_rot_quat` (or `/tf_static` from the IMU frame to the odometry
  child, else identity). A gyro cannot produce a 95-degree break.
* **translation from RGB-D visual odometry**: ORB features on
  `vo_image_topic` with 3D from the paired depth frame, PnP-RANSAC between
  consecutive frames at `vo_rate_hz`. The VO image must be in the depth
  frame (ZED left + depth_registered; D455 infra1 + depth). If the state
  camera differs from the VO camera (D455 colour vs infra1) give
  `vo_extrinsic_xyzquat` = state -> VO frame (the depth extrinsic).
* steps with fewer than 12 inliers become free edges; 12-40 inliers get 3x
  the sigma; `odom_sigma_t` / `odom_sigma_r` are the per-0.1 s sigmas of a
  good step (1 cm, 0.03 deg by default).

Printed checks: gyro-vs-VO rotation per step (validates the IMU mounting
rotation from data: median under 1 deg is consistent), and the shape of the
new odometry against the platform tracker. Outputs `traj_<name>_imuvo_raw.tum`
and the usual arms; on mobile_1 the lidar remains the reference.

# RTAB-Map as the pose source (`run_rtabmap.py`)

One script, ROS 2 + `rtabmap_ros` required, no other code:

    python3 run_rtabmap.py map      --platform mobile_1 --bag <mapping bag> --db rtab_mobile_1.db
    python3 run_rtabmap.py localize --platform mobile_1 --bag <coop bag>    --db rtab_mobile_1.db --out rtab_out
    (same with --platform mobile_2)

`map` builds the visual database from the mapping bag (RGB-D + IMU visual
odometry, loop closures). `localize` replays a coop bag against it and
writes `rtab_out/rtabmap_<platform>.tum`: the pose of the base frame
(`zed_camera_link` / `camera_link`) in RTAB-Map's map frame, corrected by
its relocalisations. Only images, depth, camera_info, IMU and /tf_static are
played; the bag's /tf (the broken tracker) is not. `--rate 0.5` keeps
rtabmap from dropping frames; `--markers` adds the boards' ArUco markers as
landmarks (one dictionary at a time, 15 mm markers detect only within ~1 m).

Evaluation: the `mobile_1_rtab` / `mobile_2_rtab` tracks (disabled until the
file exists) read the .tum as `odom_file`; stage 08 anchors it on the
session anchor exactly like any odometry, so RTAB-Map's map frame never
needs to be aligned by hand, and compares it to the lidar in
`paths_mobile_1.png` with boards applied on top (`B_boards`).
