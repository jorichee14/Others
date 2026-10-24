# Survey pass, site_2, 20261015 — results

mobile_1 drives (Ouster, ZED) through site_2, in the 20261015 mapping pass's
anchored map. mobile_2 parked there for 92 s with its Ouster and RealSense.
infra_1 recorded. The boards were put up again for this site, so
`dataset.boards_moved` is set and what this run measures replaces the
reference pass's poses. Companion: `RESULTS_survey_site_1_20261015.md`,
`RESULTS_coop_site_1_20261015.md`.

## Inputs

| | |
|---|---|
| bag | `work/20261015/survey/site_2/mirc_dataset_survey_site_2_20261015_merged_final` (70 topics, 538 892 messages, 15.9 GB) |
| spans (UTC) | infra_1 05:42:40 +301 s, mobile_1 05:43:02 +283 s, sniffer 05:41:57 +346 s, mobile_2 05:42:54 +92 s |
| config | `map_stages/pipeline_config_survey_site_2_20261015.json` (`--new-config --boards-moved`) |
| reference map | `processed/20261015/mapping/mapping/map_final_mapping_20261015_anchored.pcd` |

## The registration had to be re-tuned (and why)

With the stage's defaults (`carry: true`, `max_shift` 0.5 m, two rounds) the
run came out BENT and 10 to 20 m from where it drove: round 1 corrected a
median of 9.7 m, 1 094 of 2 790 scans never registered, and 05 then refused
every board (spreads 23 to 71 mm). The residual stayed at 2.6 cm throughout,
so nothing in the numbers said "wrong place".

`check.py glim` separated the two possible causes. On GLIM's OWN trajectory
the same sightings placed the boards to 2.4, 8.2, 2.7 and 6.9 mm, and
board_2's loop closure over 3 s was 16.7 mm against 11.6 mm of detection
noise, not significant. So the run was good and the REGISTRATION was at
fault: site_2 is a corridor among parallel corridors, a scan fits the map as
well 13 m sideways, and `carry` handed the first slip to every scan after
it (the gate is measured from the carried seed, so it never tripped).

Re-run with the map correcting GLIM instead of replacing it:
`"rounds": 3, "carry": false, "max_shift": 0.3, "max_rot_deg": 2.0,
"prior_beta": 0.5`.

| round | registered | median corr | p95 corr | median residual |
|---|---|---|---|---|
| 1 | 516 / 2 790 | 19.50 cm | 29.29 cm | 2.59 cm |
| 2 | 2 731 / 2 790 | 2.55 cm | 23.11 cm | 2.74 cm |
| 3 | 2 747 / 2 790 | 0.30 cm | 2.45 cm | 2.74 cm |

GLIM seed to final: median 32.8 cm, p95 57.4 cm, max 70.6 cm. 43 scans
interpolated from their neighbours. Path 97.0 m. The time check says GLIM's
poses are its scans' (offset kept at 46 ms). Stage 04 now warns by itself
when a round's median correction is past twice `max_shift`; the rule is in
the README.

## Boards (4 written)

| board | map xyz | sections | spread |
|---|---|---|---|
| board_1 (anchor) | 0.209, 0.059, -0.054 | 1, n=5 | 14.9 mm / 35.8 mm max |
| board_2 | -32.392, 0.152, 0.184 | 1 of 2, n=5 | 16.1 mm / 23.1 mm max |
| board_3 | -26.200, 4.808, 0.130 | 1, n=5 | 21.7 mm / 54.3 mm max |
| board_4 | -6.130, 4.535, -0.026 | 1, n=5 | 12.2 mm / 29.3 mm max |

board_3 needed `"fuse": {"max_std_mm_by_board": {"board_3": 25.0}}`: 21.7 mm
against the 20 mm gate. The spreads are the registered trajectory's own
jitter under a board seen from one spot, not the detector's (the same
sightings on GLIM's smooth trajectory: 2 to 8 mm). Site_1's survey reached 1
to 2 mm, where the robot dwelled at each board.

board_1 sits 223.7 mm / 1.74 deg from where the mapping pass had it, which
is the boards being put up again and is why `boards_moved` is right.

board_2's two sections disagree by 49.3 mm / 0.389 deg over 3 s, 2.2x the
detection noise, flagged DRIFT; its second section has one sample, so it is
indicative. On GLIM's trajectory the same closure was 16.7 mm and not
significant, so this is the registration's jitter again, not GLIM drift.

101 sightings, including the best-dwelt views of the anchor board, fall
before GLIM's trajectory starts and were dropped. Re-exporting GLIM over the
full bag would recover them and tighten every spread above.

## Cameras, cloud, poses

- arducam (infra_1) placed from its extrinsic: map -> arducam_optical_frame
  t = [-13.579, 8.811, 1.807].
- mobile_2 NOT placed: board_4 was never detected in its 1 304 colour
  frames, so 07 had nothing to start its LiDAR registration from. Its CSI is
  therefore unposed in this pass, and site_2's coop pass needs to know which
  board the parked robot actually faces.
- `mapping/cloud_in_ref_survey_site_2_20261015.pcd`: 5 574 564 points at
  0.03 m from 2 749 scans; 115 092 voxels (718 637 points) carved out as
  movers, nothing else removed.
- `bags/survey_site_2_20261015_best_poses`: 11 910 messages; mobile_1 starts
  at [0.787, 0.123, 0.137].

## Comms

| | |
|---|---|
| ping | infra_1 29 856, loss 27.9 %, RTT 10.4 ms median / 54.7 p95; mobile_1 32 781, 30.2 %, 11.6 / 88.9; mobile_2 9 940, 26.5 %, 9.2 / 98 |
| wifi | RSSI medians -31 (infra_1), -36 (mobile_1), -42 (mobile_2) dBm, channel 149, no roams |
| iperf | mobile_1 up 239 Mbit/s median, down 301 |
| CSI | infra_1 -> sniffer 30 910 packets at 89.4 Hz; mobile_2 -> sniffer 31 014 at 89.7 Hz |
| NTP | all synchronised, |offset| medians 0.0 to 0.06 ms, no steps |
| csi_clean | infra_1 link 9 953 QoS frames, all posed; mobile_2 link 5 366, none posed (see above) |

## Open items

- mobile_2 has no placement here. Find the board it faced at site_2 (or
  record one for it) before the coop pass at this site.
- Re-export GLIM over the whole bag to recover the 101 opening sightings,
  then re-run 05: every board's spread should fall toward the 2 to 8 mm the
  GLIM check measured.
- The same `04_reference` settings will be needed for site_2's coop pass:
  the corridor ambiguity does not go away with a second robot.
