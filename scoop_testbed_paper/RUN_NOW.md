# Run now: the remaining compute-only items, in order

Everything below runs on the processed 24 Sep passes on your machine
(`data/processed/20260924/{mapping_A,survey_1,coop_2}`). No recording needed.
Each step names the question it feeds and the file to read the answer from.
Paths assume the repo root and `DATA=~/workspaces/isaac_ros-dev/data`.

## 1. Confirm the four numbers whose pass is unknown (10 min)

```
python scoop_pipeline/analysis/comms/ntp_analysis.py  $DATA/processed/20260924/survey_1   # chrony RMS per client
python scoop_pipeline/analysis/comms/ntp_analysis.py  $DATA/processed/20260924/coop_2     # 0.99 ms / 12 ms should reappear
python scoop_pipeline/analysis/comms/wifi_analysis.py $DATA/processed/20260924/survey_1   # capacity 275-282 / 598-676?
python scoop_pipeline/analysis/comms/csi_analysis.py  $DATA/processed/20260924/survey_1   # 141 / 104 Hz?
python scoop_pipeline/check.py map $DATA/processed/20260924/mapping_A/mapping/denoised_mapping_A_20260924.pcd
```
Read: `comms/ntp/ntp_report.json`, `comms/wifi/wifi_report.json`,
`comms/csi/csi_report.json`, and the thickness printout. Write the pass name
next to each number in `RESULTS_STATUS.md`.

## 2. Q1 Precise: the missing terms (1 h)

```
# LiDAR robot against the boards on coop_2 (same tool as the 0828 result)
python3 slam_benchmark/scripts/board_table.py --runs runs/coop_2_20260924 --agent mobile_1

# p95 speed of the reference tracks, then the headroom figure
python scoop_pipeline/analysis/headroom.py scoop_pipeline/configs/headroom_terms.json \
    --traj $DATA/processed/20260924/coop_2/odometry/reference_*/traj_*.tum \
    --out scoop_testbed_paper/figs/fig_headroom.png
```
Then fill the `null` terms in `scoop_pipeline/configs/headroom_terms.json`
as they arrive (mobile_1 board residual; radar reflector residuals from
`radar_camera_calibration` on held-out positions; LiDAR-camera reprojection;
sniffer bias; control distances) and rerun. The printout is the E1 result
paragraph: margin per term, like against like.

Sniffer bias, if the probe sender logged its transmit stamp:
median of (sniffer receive stamp - mobile_2 send stamp) over coop_2
(`comms/csi/csi.csv` t vs the probe's header stamp). If not logged, it comes
from the field session.

Cross-robot surface consistency:
```
python scoop_pipeline/check.py scans $DATA/processed/20260924/coop_2
```

## 3. Q2 Consistent: AP links, pooling, CI (30 min + the AP position)

1. Read the access point's x y z off the anchored MIRC map (it is in the
   cloud as structure; the `comms_map.py` top view shows it) and fill
   `scoop_pipeline/configs/ap.yaml`.
2. ```
   python scoop_pipeline/analysis/comms/link_clearance.py --all $DATA/processed \
       --ap-file scoop_pipeline/configs/ap.yaml
   ```
   Read `$DATA/processed/clearance/clearance_all.json`: pooled rho and the
   clear-minus-obstructed drop per link kind (csi, ap, ap_iperf). Pass: rho
   > 0.3 and drop > 5 dB on at least one kind.
3. Confidence interval and out-of-run prediction are not in the tool yet:
   bootstrap the windows of `comms/clearance/*.csv` for the CI; for
   out-of-run, take the offset `a` from survey_1's clear windows and score
   coop_2 with it (one flag to add to `link_clearance.py`).

## 4. Q4 Repeatable: the mode half (5 min)

```
python scoop_pipeline/analysis/comms/link_repeat.py \
    $DATA/processed/20260924/survey_1 $DATA/processed/20260924/coop_2 \
    --label-a survey --label-b coop
```
Read `coop_2/comms/repeat/link_repeat_survey_vs_coop.json`: W1 and KS for
RTT, goodput up/down, RSSI, and RSSI on the shared grid cells. This is the
denominator of the E4 criterion. After the repeat day:
```
python scoop_pipeline/analysis/comms/link_repeat.py \
    $DATA/processed/20260924/survey_1 $DATA/processed/<day2>/survey_1 \
    --label-a survey --label-b survey_day2 \
    --against $DATA/processed/20260924/coop_2/comms/repeat/link_repeat_survey_vs_coop.json
```
prints the ratio per quantity; pass < 0.5.

## 5. Q5 Necessary: strengthen (20 min)

```
python scoop_pipeline/analysis/comms/sim_gap.py \
    $DATA/processed/20260924/survey_1 $DATA/processed/20260924/coop_2
```
gives the figure for the paper (`sim_gap_all.png`). The best-fit synthetic
baseline (constant or Binomial delay fitted to the measured delays) is a
small addition to `sim_gap.py`: fit on the logged `delay_ms` column of
`sim_gap.csv`, then report its in-frame share next to the two published
channels.

## 6. Q7 Reproducible: bookkeeping (one afternoon, no tools)

Fill `scoop_testbed_paper/tables/bom.tex` costs per agent class and the
per-site setup line: mapping-session length, number of boards, number of
rangefinder distances, software versions.

## 7. Field session (Q3, Q4 day half, sniffer bias if still missing)

Same route as survey_1, same site, same day: A1 survey (iperf + ping),
B1 survey (payload node + ping), A2 cooperation (iperf + ping), B2
cooperation (payload node + ping), optional probe-off run. Needs the payload
node and `twin_run.py` first. Plan and pass criteria:
`scoop_paper/split/EXPERIMENTS.md`, Phase 2.
