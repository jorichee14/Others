# Coop pass, site_1, 20261015 — results

Both robots drive, both carry an Ouster, in the 20261015 mapping pass's
anchored map. mobile_1: OS1-64, ZED. mobile_2: OS1-128, RealSense; parked
at board_4 at the start, which is what places it. infra_1 recorded but not
used. Companion: `RESULTS_survey_site_1_20261015.md` (the survey of the
same day, which placed the boards these runs start from).

## Inputs

| | |
|---|---|
| bag | `work/20261015/coop/site_1/mirc_dataset_coop_site_1_20261015_merged_final` (75 topics, 890 250 messages, 329.2 s) |
| config | `map_stages/pipeline_config_coop_site_1_20261015.json` (written by `run_pass.py --new-config`) |
| reference map | `processed/20261015/mapping/mapping/map_final_mapping_20261015_anchored.pcd` |
| boards, cameras | from `survey/site_1` (`boards_from`, `cameras_from`) |

## Trajectories (stage 04, every scan to the frozen map)

| track | scans | round 1 median / p95 | round 2 | residual | seed held |
|---|---|---|---|---|---|
| mobile_1_lidar | 3 166 | 10.91 / 31.61 cm | 0.06 cm | 1.38 cm | 0 of 3 166 |
| mobile_2_lidar | 2 990 | 106.61 / 124.81 cm | 0.04 cm | 1.27 cm | 1 scan, 0.0 s |

mobile_2's seed came from board_4 through its camera-to-LiDAR calibration,
the untrusted one: its 1.6 deg error over the 20-30 m it drove is the
metre the map took out in round 1. The final LiDAR trajectory does not
depend on that calibration; mobile_2's CAMERA pose still carries it
(8 cm at the start, as in the survey).

GLIM's pose time checked against the reference: on time, offset kept at
46 ms. Path 68.5 m (mobile_1) and 44.0 m (mobile_2); fastest 0.84 and
0.97 m/s.

## Cloud (stage 09)

Both Ousters, each on its own stage-04 trajectory, deskewed, 0.03 m voxel:
6 160 scans. The movers (the other robot, the operators) are removed by
FREE-SPACE CARVING alone — 134 070 voxels of 402 293, 933 930 points —
with stage 01's span test and hit-count guard off, because a run drives
past most of the room once and both would have deleted the walls the
objects stand against (they took 2.05 M points when on). Kept 5 109 551.
The removed points are in `cloud_in_ref_coop_site_1_20261015_dynamic.pcd`.

## Labels (5 static objects, hand-fitted, verified)

Fitted with `label_static.py` from seeds picked in the cloud, each on the
floor measured under it (the local estimate is now confined to the cloud's
own level, or a stairwell 0.4 m down becomes the ground).

| name | class | centre (map) | size L x W x H (m) |
|---|---|---|---|
| chair_1 | chair | -8.812, 10.156, -0.171 | 0.67 x 0.66 x 0.89 |
| chair_2 | chair | -0.633, 10.919, -0.182 | 0.72 x 0.70 x 0.96 |
| potted_plant_1 | potted_plant | -17.286, 13.894, -0.015 | 0.67 x 0.44 x 1.01 |
| potted_plant_2 | potted_plant | -23.633, 14.161, -0.073 | 0.69 x 0.48 x 0.81 |
| potted_plant_3 | potted_plant | -25.550, 13.677, -0.279 | 0.55 x 0.32 x 0.40 |

File: `<InCoP>/labels/site1_20261015_statics.json`. Both robots' boxes come
from their poses (ids 10000, 10001). The calibration-board trolley at
(-4.34, 7.08) was fitted and then dropped: it is testbed equipment, not an
instance of either class, so it is background in both splits.

Checked in the robots' own camera images (`show_labels.py`) and against
their scans (`label_static.py --check`, every 25th frame, 0.10 m margin):

| object | mobile_2 sees it | points (median) | mobile_1 sees it | points (median) |
|---|---|---|---|---|
| chair_1 | 47 % | 471 | 53 % | 40 |
| chair_2 | 53 % | 180 | 82 % | 384 |
| potted_plant_1 | 53 % | 249 | 24 % | 209 |
| potted_plant_2 | 35 % | 292 | 6 % | 154 |
| potted_plant_3 | 29 % | 141 | 18 % | 4 |

Every box holds returns from both robots, which is the independent check
on the labels, the extrinsics and the anchoring at once. Each robot misses
objects the other sees: the collaboration signal the dataset exists to
measure. mobile_2 is the ego (it sees the two chairs least often).

## Dataset (rosbag_to_opv2v)

`configs/mirc_site1_20261015_incop.yaml` ->
`<InCoP>/site1_20261015_real_world/train/case_0/`, agents 0 (mobile_2,
ego) and 1 (mobile_1), 2 973 frames of 3 197 candidates.

| | |
|---|---|
| grid | mobile_1's Ouster, 10 Hz, 55 ms tolerance (two free-running LiDARs sit 31.8 ms apart, 51.6 ms at worst) |
| dropped | 173 + 5 where mobile_2's stream had not started or had ended, 24 with no pose there, 22 scattered RealSense frames |
| clocks | verify; mobile_2's residual 0.37 ms (its source was reachable 75 % of the time) |
| extrinsics | resolved from `/tf_static` (`extrinsic: tf`), both clouds and both cameras |
| points per frame | mobile_1 45 565 mean, mobile_2 78 736 |
| GT per frame | 7 boxes inside the ego's range |
| link state | 12 streams in every frame (`comms`), full-rate tables in `<root>/comms/train/case_0/` |
| ZED view | `..._real_world_zed`, 256 frames (the ZED recorded at 1.3 Hz) |
| sidecars | `heter_modality_assign.json` (both m1), `isaacsim_class_map.json` (+ robot) |

`validate_opv2v.py`: PASSED, no warnings. Inter-agent distance 0.8 to
26.8 m, mean 14.3.

## Open items

- site_2's coop pass is not run yet; it is the held-out split (its
  converter config is `configs/mirc_site2_20261015_incop.yaml`, with the
  start poses to fill from its stage 10 log). Its labels must follow the
  same rule: every chair and every potted plant the robots see.
- mobile_2's camera-to-LiDAR calibration is still the untrusted one. Its
  images are in the dataset and its camera pose carries that error.
- The heterogeneous arms need the sidecar written again with
  `--assign 0=m1 --assign 1=m2`; the tree itself does not change.
