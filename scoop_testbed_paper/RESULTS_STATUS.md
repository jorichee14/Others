# SCooP-iT: results in hand and data still lacking

Session: 24 September 2026, site 1 (MIRC 1F). Passes: `mapping_A` (reference
map), `survey_1` (one agent owns the channel, 16 stops), `coop_2` (both robots
moving, contended channel). Earlier pass at the same site: `coop2_0828`.

## 1. Results in hand

### Mapping (`mapping_A`)

| Result | Value | Tool |
|---|---|---|
| Final pose correction at convergence of scan-to-map refinement | 0.15 cm, 0.017° | `02_refine_poses.py` |
| Surface thickness after refinement, walls / floors | 7.9 / 2.1 cm vs noise floor 2.9 / 1.25 cm (confirm this is mapping_A) | `map_quality.py` |
| Agreement with the map built four weeks earlier, walls | 1.06 cm | `map_quality.py` on both maps |
| Relative scale between the two maps | 0.99998 | same |
| Board detector against the mapping pipeline | 8.1 mm / 0.66° ZED, 8.3 mm / 0.58° RealSense | `board_detect.py` |
| Boards posed from the mapping session; infra camera posed through the shared board | done | `05_anchor.py`, `07` |

### Mapping pass 20261015 (`mobile_2`, 1387 s, 544 m): reference with bundle adjustment and TSDF

Method: `supp/refmap_method.tex` (supplementary); main-text version `sections/refmap_main.tex`
(replaces the body of Sec. V-A); tables `tables/refmap_stages.tex`, `tables/glim_vs_ref.tex`.

| Result | Value | Tool |
|---|---|---|
| Board loop closure, GLIM / scan-to-map / scan-to-map + BA / GLIM + BA | 10.8 / 13.7 / 12.1 / 15.1 mm (noise 6-8 mm, none significant) | `loop_check.py` |
| TSDF map thickness, walls / floors: GLIM poses | 14.3 / 3.6 cm, 2.0 % doubled | `map_quality.py` |
| TSDF map thickness, walls / floors: scan-to-map + BA poses | 7.4 / 1.6 cm, 0.7 % doubled | `map_quality.py` |
| Union vs TSDF, same poses (scan-to-map + BA) | walls 12.5 vs 7.4 cm, floors 3.6 vs 1.6 cm; TSDF 98 % complete | `map_quality.py`, `compare_maps.py` |
| Scans as placed, walls, whole run: scan-to-map + BA / GLIM + BA | 8.4 / 8.8 cm (one scan 4.6, 1 s 6.5) | `wall_build.py` |
| BA from two starts (GLIM, scan-to-map) agree | 0.7 cm (median, after alignment), 0.1 deg | `compare_trajs.py` |
| GLIM against GLIM + BA | 5.3 cm median, 0.42 deg; heading error ~0.6 deg within 10 s, not growing | `compare_trajs.py` |
| **Reference = GLIM + BA**: TSDF map walls / floors | 7.5 / 1.7 cm, 0.6 % doubled, 96.8 % of the union's surfaces | `map_quality.py`, `compare_maps.py` |
| ATE / RPE (RMSE) against the reference: GLIM | 3.4 cm / 0.69 deg; RPE 1 s 1.3 cm / 0.26, 10 s 5.7 / 0.93, 100 s 15.0 / 0.96 | `compare_trajs.py --ref ba_glim` |
| same: scan-to-map + BA (the other start) | 1.0 cm / 0.10 deg | same |
| Scans as placed, walls, whole run: GLIM / GLIM + BA (same spots) | 19.2 / 8.8 cm (one scan 4.9) | `wall_build.py` |

Still lacking: control distances.

### Survey (`survey_1`)

| Result | Value | Tool |
|---|---|---|
| RSSI mapped in the reference frame | 177 cells, 2208 samples | `comms_map.py` |
| Uplink goodput mapped | 108 cells, 443 samples | `comms_map.py` |
| Single-stream capacity per mobile agent | 275-282 Mbit/s up, 598-676 Mbit/s down (confirm pass) | `wifi_analysis.py` |
| CSI data-frame rate | 141 Hz infra_1, 104 Hz mobile_2 (confirm pass) | `csi_analysis.py` |
| CSI valid relative speed (3 samples per half wavelength) | about 1.2 m/s | derived |
| chrony RMS offset per client | 10.9 / 42.5 / 28.7 us (confirm pass) | `ntp_analysis.py` |
| 1.5 MB messages within one LiDAR frame, one robot | 86 % | `sim_gap.py` |

### Coop (`coop_2`)

| Result | Value | Tool |
|---|---|---|
| Measured clock offset over the run | <= 0.99 ms | `ntp_analysis.py` |
| Strict chrony error bound | <= 12 ms | `ntp_analysis.py` |
| RGB-D cart in the LiDAR frame, held-out boards | 3.1 cm median, per-pose sigma 5.3 cm | `board_table.py`, `depth_pgo.py` |
| Same without map-fix factors | 284 cm | same |
| Map clearance vs detrended RSSI, mobile_2 CSI link | rho 0.60, 5.4 dB clear-minus-obstructed (passes) | `link_clearance.py` |
| Same, infra_1 link | rho 0.23, 4.7 dB (does not pass) | `link_clearance.py` |
| 1.5 MB messages within one frame, two robots | 59-68 % | `sim_gap.py` |
| Synthetic channels, same messages | 100 % (V2X-ViT constant 100 ms; Where2comm no latency) | `sim_gap.py`, code read |
| RTT idle to under iperf load | 5-10 ms to 25-33 ms | `wifi_analysis.py` |

### Earlier pass (`coop2_0828`), usable as supporting evidence

| Result | Value |
|---|---|
| Held-out board residual, LiDAR robot / RGB-D cart | 10.0 mm / 0.88°, 54.0 mm / 1.38° |
| Real outages | 3, covering 12 % of frames |
| RTT idle to loaded | 3.3 ms to 60-109 ms |
| Single-agent SLAM scored against the reference | RTAB-Map, MASt3R-SLAM, KISS-ICP (slam_benchmark) |

## 2. What each question can say now

| Question | Verdict from 24 Sep |
|---|---|
| 1 Precise | like against like at 1 m/s: measured clock 101x, map refinement 133x, repeatability 19x, detector 24x pass; cart 3.1 cm (6.5x) and strict clock bound 12 ms (8.3x) fall short of 10x (`headroom.py`) |
| 2 Consistent | passes on the mobile_2 link; not on infra_1 |
| 3 Faithful | no data |
| 4 Repeatable | pose half yes (1.06 cm); mode half computable, not computed; day half no data |
| 5 Necessary | answered: 86 % vs 59-68 % vs 100 % |
| 6 Unique | capability table, cells to verify |
| 7 Reproducible | pipeline and spec sheet; costs and setup effort unwritten |

## 3. Lacking data

### A. Computable now from the 24 Sep passes (no recording)

| Item | Feeds | Tool / action |
|---|---|---|
| LiDAR robot (mobile_1) vs boards on coop_2 | Q1, spec sheet | `board_table.py` on coop_2 |
| p95 agent speed per pass | Q1 latency displacement | from reference tracks |
| E1 headroom figure | Q1 | `analysis/headroom.py configs/headroom_terms.json --traj <reference .tum files>`; fill the null terms as they arrive |
| Agent-to-AP clearance vs RSSI and goodput | Q2 | fill `scoop_pipeline/configs/ap.yaml`; `link_clearance.py --all data/processed --ap-file scoop_pipeline/configs/ap.yaml` |
| Pool E2 over survey_1 and coop_2; CI on rho; out-of-run fit | Q2 | `link_clearance.py --all`, small additions |
| Mode distance survey_1 vs coop_2 (RTT, goodput, RSSI on grid) | Q4 mode half | `analysis/comms/link_repeat.py survey_1 coop_2 --label-a survey --label-b coop` (written, self-tested) |
| Cross-robot surface consistency on coop_2 | Q1, GT table | `scan_consistency.py` |
| Sniffer bias | Q1, spec sheet | median of (sniffer receive - mobile_2 send) if the sender logged its stamp; else from the field session |
| Best-fit synthetic baseline for E5 | Q5 | fit constant / Binomial delay to measured delays in `sim_gap.py` |
| Control distances vs map | Q1, absolute scale | only if rangefinder distances and endpoints were logged; short script |
| Radar reflector residuals on held-out positions | Q1, calib table | `radar_camera_calibration` |
| LiDAR-camera reprojection residual per range | calib table | project board returns, per board and range |

### B. Needs one recording session (same route as survey_1, same site)

| Run | Mode | Traffic | Feeds |
|---|---|---|---|
| A1 | survey | iperf + ping | Q3 trace; Q4 day half vs survey_1 |
| B1 | survey | payload node (1 KB / 100 KB / 1.5 MB at 10 Hz) + ping | Q3 live arrivals |
| A2 | cooperation | iperf + ping | Q3 trace; Q4 day half vs coop_2 |
| B2 | cooperation | payload node + ping | Q3 live arrivals |
| optional | either | iperf off, ping only | Q5 probe-off control |

Before the session: payload node (ROS 2 publisher/subscriber, stamps on both
ends, DDS best effort), `twin_run.py`, clocks converged, 802.11ax off, single
stream, power save off. About one hour including setup.

Outputs: per-size, per-mode arrival-time error (median, p95), frame-assignment
agreement (pass >= 90 %), drop agreement, synthetic channel on the same axes;
day-vs-mode distance table; sniffer bias if not obtained in A.

### C. Bookkeeping, no data

- Costs per agent class (bill of materials).
- Per-site setup effort: mapping-session length, number of boards, number of
  rangefinder distances, software versions.
- Verify every cell of the capability table against the cited paper.
- Decide the headline clock figure (measured <= 0.99 ms recommended) and which
  RTT regime to quote (both passes, each with its load).

## 4. Numbers to confirm before they go on a slide

| Number | Doubt |
|---|---|
| 7.9 / 2.1 cm thickness, 1.06 cm cross-session | which two maps: mapping_A and the Aug map? |
| 275-282 / 598-676 Mbit/s | which pass |
| 141 / 104 Hz CSI | which pass |
| 10.9 / 42.5 / 28.7 us chrony RMS | which pass ("single live pass" in the table) |
| 3.3 ms vs 5-10 ms idle RTT | different passes; quote both with load |
