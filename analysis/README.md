# Run characterization analysis

One notebook per topic, each characterizing one recorded run (a rosbag2 MCAP) for
the dataset paper. No ROS installation is needed: message definitions are read
from the schemas embedded in the MCAP file.

```
analysis/
├── extract_bag.py          shared: decodes the MCAP into per-topic Parquet tables
├── ntp_analysis.ipynb      notebook 1: NTP / temporal calibration
├── requirements.txt
└── tests/
    └── make_synthetic_bag.py   writes a small fake MCAP to try the notebook without the real bag
```

`extract_bag.py` is a library the notebooks import (`from extract_bag import extract`).
It can also be run from the command line to extract a bag once and share the small
Parquet output instead of the multi-gigabyte bag:

```bash
python analysis/extract_bag.py BAG.mcap --out extracts/coop2
```

## Notebook 1: `ntp_analysis.ipynb`

Keep it in the same folder as `extract_bag.py`. Set `RUN` and `BAG` in section 1.1
and run the cells in order.

| Section | What it does |
|---|---|
| 1 Parameters | paths, thresholds, agent colors |
| 2 Extract Bag | runs `extract()` once; skipped when `extracts/<run>/metadata.json` already exists |
| 3 Load NTP Data | status, events, stamp audit, common time axis |
| 4 NTP Roles | who is server, who are clients |
| 5 Clock Offset | offset statistics per client, clock steps, offset / delay / jitter over time |
| 6 Offset Distribution | ECDF of \|offset\| per client |
| 7 NTP-Independent Check | header stamp − recorder log time per topic, shortest sensor period |
| 8 Figure for the Paper | 3-panel figure, saved as PDF and PNG |
| 9 Paper Text | the *Temporal Calibration* subsection with the numbers filled in |

Outputs go to `results/<run>/ntp/`: `ntp_roles.csv`, `ntp_summary.csv`,
`ntp_audit.csv`, `fig_ntp.pdf`, `fig_ntp.png`, `ntp_subsection.tex`.

Panel (c) of the figure reads as follows. Every sensor is stamped in software on
arrival, so header stamp minus recorder log time is transport latency plus the clock
offset between that node and the recorder. The recorder's own topics sit slightly
negative; a node whose topics sit systematically off them has a clock offset of that
size, independently of what the NTP monitor reports.

## Trying it without the real bag

```bash
python analysis/tests/make_synthetic_bag.py mirc_dataset_coop2_20260828_completed_0.mcap
```

then open the notebook with `BAG` pointing at that file.
