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
| 2 | `mobile_1_zed` (`arms`, `cloud_source: lidar`) | three trajectories of the ZED optical frame from one graph: `A_icp` (ZED odom + lidar map factors), `B_boards` (ZED odom + board sightings + session anchor, **started from the odometry** = "boards correct the ZED odom"), `C_joint` (lidar + boards + odom). Plus `traj_mobile_1_zed_odom_only.tum`. |
| 2b | printed | ablation table: board residual / map rms / vs C / vs odom per arm. A's board residual and B's map rms are the independent cells. |
| 3 | printed | cross-check (lidar track scored on the ZED's board sightings, both `T_lidar_camera` conventions), then the rig comparison table: lidar ICP, odom only, A, B, C pairwise, all in the camera frame at the lidar stamps. `compare_mobile_1.csv` has the gap to the lidar track over time; `paths.png` draws everything. |

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
* **arm B vs odom** is the correction the boards applied. The
  board-factor-coverage line says how much of the run the boards can reach;
  outside it B is pure odometry.
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
