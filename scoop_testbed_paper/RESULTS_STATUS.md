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
| 1 Precise | passes on every term except the cart (3.1 cm is 6x below 0.2 m, not 10x) |
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
| E1 headroom figure | Q1 | one plot script over the rows above |
| Agent-to-AP clearance vs RSSI and goodput | Q2 | read AP position off the anchored map into `ap.yaml`; `link_clearance.py --all` |
| Pool E2 over survey_1 and coop_2; CI on rho; out-of-run fit | Q2 | `link_clearance.py --all`, small additions |
| Mode distance survey_1 vs coop_2 (RTT, goodput, RSSI on grid) | Q4 mode half | write `link_repeat.py` |
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
