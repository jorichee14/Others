# Run characterization analysis

Scripts that characterize one recorded run (a rosbag2 MCAP) for the dataset paper:
temporal calibration (NTP) first; Wi-Fi, CSI and trajectory geometry follow the
same extract-then-analyze pattern.

No ROS installation is required. Message definitions are read from the schemas
embedded in the MCAP file.

```bash
pip install -r analysis/requirements.txt
```

## 1. Extract the light topics once

```bash
python analysis/extract_bag.py /path/to/mirc_dataset_coop2_20260828_completed_0.mcap --out extracts/coop2
```

By default this decodes every topic except images, point clouds, laser scans,
camera info, paths and raw Ouster packets, and writes one Parquet table per topic
plus `stamp_audit.parquet`, which holds `header.stamp`, publish time and recorder
log time for every message of every header-bearing topic (parsed from the raw
CDR bytes, so it is cheap even for images). The whole output is small enough to
commit or share.

To extract only what the NTP analysis needs:

```bash
python analysis/extract_bag.py BAG.mcap --out extracts/coop2 \
    --topics /infra_1/ntp/status /mobile_2/ntp/status /infra_1/ntp/events /mobile_2/ntp/events
```

(the stamp audit still runs over all topics; add `--no-audit` to skip it.)

## 2. NTP / temporal calibration

```bash
python analysis/ntp_analysis.py --extracts extracts/coop2 --out results/coop2/ntp --run coop2
```

Produces `ntp_roles.csv` (who is server, who are clients), `ntp_summary.csv`
(offset mean / median / p95 / max, delay, jitter, stratum, reach, poll interval,
clock steps per client), `ntp_audit.csv` (header stamp minus recorder log time per
topic, plus each topic's median period), `ntp_summary.md`, a filled-in
`ntp_subsection.tex` for the paper's *Temporal Calibration* subsection, and
`fig_ntp.{pdf,png}`:

- (a) client clock offset over time, clock steps in red, NTP events in yellow;
- (b) ECDF of |offset| per client with the shortest sensor period marked;
- (c) NTP-independent check: ECDF of header stamp minus recorder log time, one
  line per topic, colored by node.

Panel (c) reads as follows. In the recorder's own clock every message arrives
after it was stamped, so its topics sit slightly negative (transport latency). A
node whose topics sit systematically off the recorder's has a clock offset of that
size relative to the recorder, independently of what the NTP monitor reports.

## Tests

`tests/make_synthetic_bag.py` writes a small MCAP with the dataset's NtpStatus
schema so the pipeline can be exercised without the real bag:

```bash
python analysis/tests/make_synthetic_bag.py /tmp/synthetic.mcap
python analysis/extract_bag.py /tmp/synthetic.mcap --out /tmp/extracts
python analysis/ntp_analysis.py --extracts /tmp/extracts --out /tmp/results --run synthetic
```
