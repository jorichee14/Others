# Coop pass, site_2, 20261015 — results

Both robots drive at site_2, in the 20261015 mapping pass's anchored map, with
the boards as `survey/site_2` measured them. infra_1 records; mobile_1 carries
the sniffer. Companions: `RESULTS_survey_site_2_20261015.md`,
`RESULTS_coop_site_1_20261015.md`.

## Inputs

| | |
|---|---|
| bag | `work/20261015/coop/site_2/mirc_dataset_coop_site_2_20261015_merged_final` (75 topics, 684 175 messages, 25.8 GB) |
| spans (local) | mobile_1 13:58:47 +247 s, mobile_2 13:58:56 +231 s, infra_1 13:59:03 +226 s, sniffer 13:59:03 +229 s |
| config | `map_stages/pipeline_config_coop_site_2_20261015.json` (`--new-config --boards-moved`) |
| boards, arducam | `survey/site_2` (`dataset.boards_from`, `cameras_from`), replacing the mapping pass's (`boards_moved`) |
| GLIM | mobile_1 2 434 poses / 243.3 s / 48.1 m; mobile_2 2 247 poses / 224.6 s / 53.3 m |

## Both robots are placed from boards they were not expected to be at

03 looks for every board in the map rather than the one a sensor's entry
names, and here NEITHER robot was at the one the config would have guessed:

| robot | board found | evidence |
|---|---|---|
| mobile_1 (ZED) | **board_2** | 198 sightings; the opening section's 137 over 78.6 s, 103 kept, std 8.7 mm / max 46 mm |
| mobile_2 (RealSense) | **board_1** | 1 060 sightings over 71.2 s at 0.46 m, mean reproj 0.235 px; parked (10.1 mm across all of them), 1 055 kept, std 0.7 mm / max 8.1 mm |

They start 33 m apart, at opposite ends of the site: mobile_1 at
[-20.35, -26.33], mobile_2 at [-20.19, 7.08] — which is where the survey's
mobile_1 began.

**The map could not place this run.** With `relocalize` on, mobile_1's opening
5 s laid only 38.3 % of its points within 5 cm of the map, under lidar_reloc's
50 % floor, so 03 refused. The five surviving candidates agreed on the same
spot, so this is thin map coverage at the start, not ambiguity — the same
thinness that left the survey's window 60 s in unplaceable. The boards are the
better instrument here: `survey/site_2` measured them to 0.4-1.7 mm, in this
site's own geometry, so `relocalize` is off for this pass and the board dwell
places both robots. (It was needed for the survey only because the board poses
available there were the mapping pass's, 21 m stale.)

## Registration (04, stage defaults)

| track | round | registered | median corr | p95 corr | median residual |
|---|---|---|---|---|---|
| mobile_1_lidar | 1 | 2 434 / 2 434 | 13.38 cm | 38.82 cm | 1.38 cm |
| | 2 | 2 434 / 2 434 | 0.05 cm | 0.31 cm | 1.38 cm |
| mobile_2_lidar | 1 | 2 247 / 2 247 | 25.05 cm | 96.89 cm | 1.37 cm |
| | 2 | 2 247 / 2 247 | 0.05 cm | 0.28 cm | 1.36 cm |

Every scan of both robots registered, none kept its seed, and no scan sat in a
direction the map leaves open. GLIM seed to final: mobile_1 median 13.4 cm /
max 42.8 cm, mobile_2 median 25.0 cm / max 106.3 cm — the board start plus
GLIM's drift over the run, which the map takes out in one round. Both settle
at the same 1.36-1.38 cm residual, which is the survey's 1.49 cm or better.

Time check: GLIM's heading error follows the turning as +6.4 ms (r 0.26). Its
poses are its scans'; keep the 46 ms export offset.

## Cloud and poses

- `mapping/cloud_in_ref_coop_site_2_20261015.pcd`: 4 657 328 points at 0.03 m
  from 4 685 scans of both Ousters, x -51.41..4.78, y -40.33..22.30. 161 171
  of 409 495 voxels (643 146 points) carved out as movers — each robot's rays
  carve the other one out.
- `bags/coop_site_2_20261015_best_poses`: 20 190 messages. mobile_1 49.2 m from
  [-20.356, -26.331, 0.391]; mobile_2 54.5 m from [-20.200, 7.064, 0.333].
- The arducam comes from survey/site_2's 07; 07 and 08 are off in a coop pass.

## Comms

| | |
|---|---|
| ping | infra_1 22 001, loss 33.9 %, RTT 15.0 ms median / 55.1 p95; mobile_1 28 577, 38.6 %, 18.3 / 172; mobile_2 26 849, 39.6 %, 21.0 / 117 |
| wifi | RSSI medians -31 (infra_1), -44 (mobile_1), -45 (mobile_2) dBm, channel 149, no roams |
| iperf | mobile_1 up 177 Mbit/s median, down 275; mobile_2 up 189, down 267 |
| CSI | infra_1 -> sniffer 15 591 packets at 68.2 Hz, RSSI -66 dBm, link 3.3-21.3 m; mobile_2 -> sniffer 27 311 at 119.5 Hz, -58 dBm, link 4.1-34.5 m |
| NTP | all synchronised, \|offset\| medians 0.0 to 0.06 ms, no steps |
| csi_clean | infra_1 link 1 085 QoS frames, all posed; mobile_2 link 1 948, all posed |

Both links are posed, because both robots have trajectories. Note how few of
the packets are QoS data here — 1 085 of 15 591 and 1 948 of 27 311, around
7 %, against the survey's 32 %. The rest are control frames. Loss is also
higher than the survey's (34-40 % against 27-30 %) and the infra_1 link's CSI
rate is down to 68 Hz from 89.

## Open items

- mobile_1's ZED covers only the opening: the SVO ends after 17 s, 340 left
  images for a 247 s run. Enough for the board sighting that places it, thin
  for anything that wants camera frames across the run.
- The mapping pass covers the start of this run thinly (38 % of the opening
  window on the map). It did not matter in the end, because the boards placed
  the run and every scan registered, but a run that starts there cannot be
  placed by the map alone.
