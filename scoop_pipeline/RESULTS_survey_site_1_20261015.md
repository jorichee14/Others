# Survey, site 1, 15 October 2026: results log

Pass `raw/20261015/survey/site_1`, config
`map_stages/pipeline_config_survey_site_1_20261015.json`, run in the same
day's map (`20261015/mapping`, mobile_2's). mobile_1 drove the survey (ZED,
Ouster, radars, the CSI sniffer); mobile_2 was parked as an anchor by
board_4 (RealSense, Ouster); infra_1 recorded the link (arducam, radar).
Processed outputs under `processed/20261015/survey/site_1/`. Code as of
branch `claude/scoop-paper` at 724739d.

## Inputs

| machine | bag | notes |
|---|---|---|
| mobile_1 | `mirc_dataset_survey_1_run_20261015_mobile_1` + `..._mobile_1.svo2` | Ouster on `/mobile_1/ouster`, ZED replayed from the SVO at about 1 Hz (1 497 frames) |
| mobile_1/sniffer | `mirc_dataset_survey_1_run1_20261015_mobile_1_sniffer` | the site 2 sniffer bag (`..._run_site2_...`) had been in this folder at first: swapped, merge redone |
| mobile_2 | `mirc_dataset_survey_1_run1_20261015_mobile_2` | RealSense topics `/mobile_2/color/...` (no `camera/` level: record.yaml corrected), 2.9 s LiDAR gap, IMU gaps while parked |
| infra_1 | `mirc_dataset_survey_1_run1_20261015_infra_1` | `extrinsic_arducam.yaml` next to it |

Merged bag: 63 topics, 2.36 M messages, 52.5 GB; finalized with the pose
bag as `work/20261015/survey/site_1/mirc_dataset_survey_site_1_20261015_merged_final`
(65 topics, 2.42 M messages, TF of 34 static transforms).

## Reference trajectory (03, 04, time check)

- 03: the ZED's opening views of board_1, 101 sightings over 49 s combined
  through GLIM, std 11.8 mm; the parked part alone agrees to 0.4 cm / 0.24 deg.
- 04: 15 115 scans (1 511.8 s, 121.2 m path) registered to
  `map_final_mapping_20261015_anchored.pcd`, deskewed, two rounds.

| round | registered | median correction | p95 | median residual |
|---|---|---|---|---|
| 1 (from GLIM's seed) | 15 115 / 15 115 | 7.85 cm | 44.03 cm | 1.26 cm |
| 2 (from round 1) | 15 115 / 15 115 | 0.03 cm | 0.17 cm | 1.26 cm |

- GLIM seed to final: median 7.9 cm, p95 44.0 cm, max 51.3 cm (GLIM's drift, removed).
- Scans where the map left a direction open and the seed held it: 0.
- Time check: GLIM's heading error follows the turning as −3.0 ms; on time
  with the 46 ms pose-time offset kept.
- Files: `odometry/reference_survey_site_1_20261015/traj_mobile_1_lidar.tum`,
  `traj_mobile_1_lidar_in_cam.tum`, `quality_mobile_1_lidar.csv`,
  `seed_held_mobile_1_lidar.csv`, `summary_mobile_1_lidar.json`.

## Boards (05), all through mobile_1's ZED on the 04 trajectory

Board sizes corrected first: board_3 and board_4 are 2.45 cm squares,
1.75 cm markers (22.1 x 17.2 cm), not 24.5 cm (commit dc702ab); the first
run had put them ten times too far from the camera.

| board | frame | x | y | z (m) | qx qy qz qw | evidence |
|---|---|---|---|---|---|---|
| board_1 | board | 0.0168 | −0.0092 | −0.0461 | −0.0068 −0.0031 −0.0021 0.99997 | 197 views, 4 sections, spread 4.0 mm / 0.77 deg; loop closure start to end (1 471 s) 1.4 mm / 0.34 deg |
| board_2 | board_rs | −3.4168 | 14.3685 | −0.0558 | 0.0035 −0.0113 −0.6978 0.7162 | 8 views, 1 section, std 2.2 mm |
| board_3 | board_3 | −12.6583 | 9.1949 | 0.0204 | 0.0634 −0.0176 0.9972 0.0363 | 8 views in 2 sections 112 s apart agreeing to 1.7 mm / 1.0 deg |
| board_4 | board_4 | −24.4261 | 0.4832 | 0.1252 | −0.0276 −0.0425 −0.7286 0.6830 | 1 view (322 s, 0.2 px): a seed, accepted by `min_section_samples_by_board` |

Board frame: centre, x the outward normal, y left, z up. 49 ZED sightings
fell outside the GLIM span (39 before it started, 10 after). File:
`frames/boards_survey_site_1_20261015.json`; `map -> map_zed` exported.

board_1 against the mapping pass's origin: 49.9 mm / 1.53 deg (mostly −4.6 cm
in z). Internally consistent, so a constant offset between the two rigs'
camera-to-LiDAR chains (mobile_2's RealSense defined the origin). Decision:
trust the LiDAR and mobile_1's chain; mobile_2's calibration is to be redone.

## Cameras (07)

- **mobile_2 (realsense, `camera_color_optical_frame`)**: board_4 read in
  200 RealSense frames from 0.5 m (std 0.5 mm; in the whole stream 1 250
  poses of 1 711 frames, `board_scan.py`), giving the camera at
  (−24.4204, −0.0066, 0.2907). Its own 10 Ouster scans registered to the
  map, RMS 1.4 cm, spread 19.3 mm / max 49.8 mm (the parked robot moved a
  little), average (−24.4656, −0.0127, 0.3716), q (−0.77299, 0.02047,
  −0.01062, 0.634). The board placement is 92.9 mm / 1.59 deg from it. The
  LiDAR average is the one used (`lidar_static_tol_mm: 60`).
- **infra_1 (arducam)**: from `extrinsic_arducam.yaml` at (−4.014, 7.83,
  1.6797), trajectory gap 39 ms. Its ChArUco cross-check reads 31 m / 73 deg
  and is not meaningful (another frame); ignored.
- File: `frames/cameras_in_map.yaml`; TF script `frames/publish_tfs_survey_site_1_20261015.sh`.

## Cloud and poses (09, 10)

- `mapping/cloud_in_ref_survey_site_1_20261015.pcd`: 5 157 915 points at
  3 cm from all 15 117 scans, in the map frame.
- `bags/survey_site_1_20261015_best_poses`: `/mobile_1/global_pose`,
  `/mobile_1/local_pose`, `/mobile_1/camera_pose`, `/tf`, `/tf_static`;
  mobile_1_base_link the root of its sensors; mobile_2 under
  `board_4 -> map_realsense -> camera_link` at its LiDAR pose; the four
  boards, map_zed, the arducam.

## Comms (`comms/`)

| | |
|---|---|
| Wi-Fi | infra_1 RSSI median −31 dBm; mobile_1 −30 dBm (min −70), tx 433 / rx 650 Mbit/s; mobile_2 −70 dBm, tx 195 / rx 292; channel 149, no roams |
| ping | infra_1 89 037 pings, loss 27.6 %, RTT median 10.4 ms, p95 47.2; mobile_1 176 068, loss 30.4 %, 11.5 ms, p95 69.2; mobile_2 39 982, loss 31.8 %, 11 ms, p95 290 |
| NTP | all synchronized 100 %, offsets under 0.03 ms median; mobile_1 the reference, mobile_2 synced to it |
| iperf (mobile_1) | up 104 tests, median 253 Mbit/s (min 67); down 104 (1 failed), median 397 (min 59.2) |
| CSI infra_1 → sniffer | 163 785 packets, 123.8 Hz, longest gap 54.5 s, RSSI median −46 dBm, link 1.93–19.9 m; 101 693 data frames cleaned and posed, link 1.9–18.7 m |
| CSI mobile_2 → sniffer | 19 300 packets, 14.8 Hz, longest gap 22.6 s, RSSI median −57 dBm, link 0.08–26.8 m; 7 606 data frames cleaned and posed |

Tables: `comms/csi/csi.csv` (183 085 rows), `csi_<tx>_to_mobile_1_sniffer.npz`
and `_clean.npz`, `comms/wifi/{wifi,ping,iperf}.csv`, `comms/ntp/ntp.csv`,
`comms/summary.json`, `comms/csi/csi_clean_README.txt`.

## Figures

`analysis/survey_figure.py` wrote `figures/survey/survey_site_1_20261015_overview.{png,pdf,gif,mp4}`;
the layout is parked, to be revisited.

## Decisions and fixes made on this pass

- `record.yaml`: mobile_2's RealSense topics as the bags have them (79fe2bf).
- run_pass: comms reads the bag under its final name (9bd1f07).
- Board sizes board_3 / board_4 (dc702ab); `board_scan.py` (8850b20);
  `05_anchor` `min_section_samples_by_board` (09dc299); 07 reports the
  LiDAR-versus-board delta even when the scans disagree (9b828dc); mobile_2
  placed by LiDAR (2fae641).
- The sniffer bag swapped for site 1's; merge, finalize and comms redone.

## Open

- mobile_2's camera-to-LiDAR calibration: a dedicated recording (the
  RealSense on a board from many angles while its LiDAR is tracked), then
  the mapping pass's 05 re-anchor and the September frames' re-export.
- board_4 as a measurement: a dwell of mobile_1 in front of it on a later pass.
- The infra camera's ChArUco check: the map–map_zed transform it expects.
- The survey figure's layout.

## Related, same day

LOD-3DGS on the mapping pass: October-only model (300k iterations, 1.81 M
Gaussians) 22.58 dB on 174 October views; the combined October + September
model (500k, 1.19 M) 21.73 dB on the same views and 20.88 dB on 132
September ones (`splat_psnr.py`). The October-only model is the one to use
until mobile_2's calibration is fixed; the two sessions' camera chains
disagree by the offset above and the combined model blurs between them.
