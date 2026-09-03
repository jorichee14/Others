# Run characterization analysis

One script per topic, each characterizing one recorded run (a rosbag2 MCAP) for the
dataset paper. No ROS installation is needed: message definitions are read from the
schemas embedded in the MCAP file.

```
analysis/
├── extract_bag.py          shared: decodes the MCAP into per-topic Parquet tables
├── ntp_analysis.py         NTP / temporal calibration
├── requirements.txt
└── tests/
    └── make_synthetic_bag.py   writes a small fake MCAP to try the scripts without the real bag
```

```bash
pip install -r analysis/requirements.txt
```

## NTP / temporal calibration

```bash
python analysis/ntp_analysis.py --bag /path/to/mirc_dataset_coop2_20260828_completed_0.mcap --run coop2
```

The first call extracts the light topics of the bag into `extracts/coop2/` (a few
minutes; images, point clouds and LiDAR packets are skipped). Later calls reuse that
folder, so re-running the analysis is instant. `--force-extract` redoes it.

Outputs in `results/coop2/ntp/`:

| File | Contents |
|---|---|
| `ntp_roles.csv` | who is server, who are clients: role, hostname, sync source, stratum, rate |
| `ntp_summary.csv` | per client: mean / median / p95 / max offset, delay, jitter, poll, reach, clock steps |
| `ntp_audit.csv` | per topic: median period and header stamp − recorder log time statistics |
| `ntp_summary.md` | all of the above as readable tables plus the NTP event messages |
| `ntp_subsection.tex` | the *Temporal Calibration* subsection with the numbers filled in |
| `fig_ntp.pdf`, `fig_ntp.png` | (a) offset over time, (b) ECDF of \|offset\| vs. shortest sensor period, (c) header stamp − log time per topic |

Panel (c) reads as follows. Every sensor is stamped in software on arrival, so header
stamp minus recorder log time is transport latency plus the clock offset between that
node and the recorder. The recorder's own topics sit slightly negative; a node whose
topics sit systematically off them has a clock offset of that size, independently of
what the NTP monitor reports.

## Extracting separately

`extract_bag.py` can also be run on its own, for example to extract once on the robot
and share the small Parquet folder instead of the bag:

```bash
python analysis/extract_bag.py BAG.mcap --out extracts/coop2
python analysis/ntp_analysis.py --extracts extracts/coop2 --run coop2
```

## Trying it without the real bag

```bash
python analysis/tests/make_synthetic_bag.py synthetic.mcap
python analysis/ntp_analysis.py --bag synthetic.mcap --run synthetic
```
