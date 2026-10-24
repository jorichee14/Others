# Survey pass, site_2, 20261015 — results

mobile_1 drives (Ouster, ZED) through site_2, in the 20261015 mapping pass's
anchored map. mobile_2 parked there for 92 s with its Ouster and RealSense.
infra_1 recorded. The boards were put up again for this site, so
`dataset.boards_moved` is set and what this run measures replaces the
reference pass's poses. Companions: `RESULTS_survey_site_1_20261015.md`,
`RESULTS_coop_site_1_20261015.md`.

## Inputs

| | |
|---|---|
| bag | `work/20261015/survey/site_2/mirc_dataset_survey_site_2_20261015_merged_final` (70 topics, 538 892 messages, 15.9 GB) |
| spans (UTC) | infra_1 05:42:40 +301 s, mobile_1 05:43:02 +283 s, sniffer 05:41:57 +346 s, mobile_2 05:42:54 +92 s |
| config | `map_stages/pipeline_config_survey_site_2_20261015.json` (`--new-config --boards-moved`) |
| reference map | `processed/20261015/mapping/mapping/map_final_mapping_20261015_anchored.pcd` |

## The run is placed on the map, not on a board

The boards were moved for this site, so the reference pass's board poses say
nothing about where this run is. 03 therefore relocalizes mobile_1's start in
the map itself (`"03_init": {"relocalize": {"enabled": true}}`), which is what
did not move:

- start at map `[-20.13, 7.23, 0.38]`, yaw 88.4 deg: 86.2 % of the first 5 s
  of scans lie within 5 cm of the map, rmse 2.2 cm;
- unique by a wide margin: the next distinct fit, 45 m away, leaves 59.8 % of
  the window off the map against this one's 13.8 % (needs >= 27.7 %);
- confirmed by eye against the map: the start sits 0.4 m from where the robot
  was put down;
- the repeat window 60 s in could not place itself (43.8 % on the map, its
  candidates 40 m apart in places the robot could not have reached: the map
  covers that stretch thinly). ICP from the start carried there settles at
  79.8 % on the map, 16.9 cm away — GLIM's drift over the interval. The start
  explains where the LiDAR is; the comparison check stood down.

The first attempt at this pass had 03 place the run from the ZED's dwell on
board_1 against board_1's pose in the REFERENCE pass, i.e. where the board was
before it was moved. 07 now measures that move: **21 205 mm / 89.6 deg**. The
run started 21 m from where it was, 04 dragged it back as far as it could, and
every number that pass produced was a consequence of that. 03 now refuses a
board-placed start whenever `boards_moved` is set.

## Registration (04, stage defaults)

| round | registered | median corr | p95 corr | median residual |
|---|---|---|---|---|
| 1 | 2 790 / 2 790 | 7.17 cm | 22.17 cm | 1.50 cm |
| 2 | 2 790 / 2 790 | 0.04 cm | 0.39 cm | 1.49 cm |

No scan kept its seed, none lacked correspondences, and no scan sat in a
direction the map leaves open. GLIM seed to final: median 7.2 cm, p95 22.2 cm,
max 31.3 cm. Path 83.2 m, against the 82.22 m `check.py glim` measured on
GLIM's own trajectory — two independent numbers that agree.

`carry`, `max_shift`, `prior_beta` and the round count are all at their
defaults. The corridor ambiguity that forced them off in the first attempt was
a consequence of the bad seed, not of the place.

The time check puts GLIM's heading error at -4.3 ms of the turning (r -0.22):
its poses are its scans', keep the 46 ms export offset.

## Boards (4 written)

| board | map xyz | sightings | spread |
|---|---|---|---|
| board_1 (anchor) | -20.1353, 6.6497, 0.1038 | 5, t=3.2-9.4 s | 1.0 mm / 1.3 mm max |
| board_2 | -20.2789, -25.8491, 0.1809 | 5 of 6, t=76.5-80.1 s | 1.7 mm / 3.3 mm max |
| board_3 | -24.5949, -19.5875, 0.1843 | 5, t=55.0-61.0 s | 0.4 mm / 0.6 mm max |
| board_4 | -24.4249, 0.4780, 0.1259 | 5 of 6, t=24.7-30.9 s | 1.4 mm / 3.5 mm max |

Reprojection 0.18-0.23 px throughout. board_2's two sections close to 1.6 mm /
0.25 deg over 3 s against 5.3 mm of detection noise: not significant (its
second section has one sample, so indicative). No board needed a loosened
gate, and `map->map_zed` is exported.

Two notes on the sightings: 101 of them, including the best-dwelt views of the
anchor board, fall before GLIM's trajectory starts and are dropped —
re-exporting GLIM over the full bag would recover them. And design `board_1`
declares two instances but only one was seen here, which is why 05 reports a
cluster count mismatch; the one it found is the one the run used.

## Cameras, cloud, poses

- arducam (infra_1) placed from its extrinsic
  (`raw/20261015/survey/site_2/infra_1/extrinsic_arducam.yaml`):
  map -> arducam_optical_frame t = [-28.860, -6.950, 1.974], trajectory gap
  10 ms.
- mobile_2 (RealSense) placed from **board_2**, not the board_4 its entry
  names: 200 sightings at 0.15 px, static to 0.6 mm over the stream, averaged.
  map -> camera_color_optical_frame t = [-20.239, -26.363, 0.344], half a
  metre in front of board_2. 07 looks for every board 05 placed when the
  configured one is not detected.
  Its own LiDAR did not improve on that: 10 of 10 scans registered but
  scattered 495 mm, so the board placement was kept; the scans' average sits
  67 mm / 1.41 deg from it. Relocalizing that LiDAR globally in the map had
  already been refused at 44.6 % of the window on the map (floor 50 %), and
  its confident candidate was [-0.14, 7.35] -- 35 m from where the board puts
  the robot. The gate earned its keep. mobile_2 simply does not see enough
  mapped structure from where it is parked.
- `mapping/cloud_in_ref_survey_site_2_20261015.pcd`: 3 167 193 points at
  0.03 m from 2 792 scans, x -50.09..4.82, y -39.50..19.81. 61 854 of 241 314
  voxels (299 341 points) carved out as movers, nothing else removed.
- `bags/survey_site_2_20261015_best_poses`: 11 910 messages; mobile_1's
  base_link starts at [-20.17, 7.186, 0.30].

## Comms

| | |
|---|---|
| ping | infra_1 29 856, loss 27.9 %, RTT 10.4 ms median / 54.7 p95; mobile_1 32 781, 30.2 %, 11.6 / 88.9; mobile_2 9 940, 26.5 %, 9.2 / 98 |
| wifi | RSSI medians -31 (infra_1), -36 (mobile_1), -42 (mobile_2) dBm, channel 149, no roams |
| iperf | mobile_1 up 239 Mbit/s median, down 301 |
| CSI | infra_1 -> sniffer 30 910 packets at 89.4 Hz, RSSI median -55 dBm, link 2.2-21.6 m; mobile_2 -> sniffer 31 014 at 89.7 Hz, -56 dBm |
| NTP | all synchronised, |offset| medians 0.0 to 0.06 ms, no steps |
| csi_clean | infra_1 link 9 953 QoS frames, all posed, link 2.2-21.6 m; mobile_2 link 5 366, 5 348 posed, link 0.1-34.1 m |

## Open items

- Re-export GLIM over the whole bag to recover the 101 opening sightings.
- Site_2's coop pass needs `boards_moved` AND the `relocalize` block, and must
  take its board poses from this run (`boards_from`), not from the mapping
  pass.
