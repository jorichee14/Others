# Run characterization analysis

One script per topic, each characterizing one recorded run (a rosbag2 MCAP) for the
dataset paper. No ROS installation is needed: message definitions are read from the
schemas embedded in the MCAP file.

```
analysis/
├── extract_bag.py          shared: decodes the MCAP into per-topic Parquet tables
├── ntp_analysis.py         NTP / temporal calibration
├── ntp_issues.md           NTP basics, what went wrong in coop2, and the fixes
├── wifi_analysis.py        Wi-Fi link quality and dual-radio correlation
├── requirements.txt
└── tests/
    └── make_synthetic_bag.py   writes a small fake MCAP to try the scripts without the real bag
```

All analysis scripts share one extraction: run any of them with `--bag` once and the
rest reuse `extracts/<run>/`.

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

## Wi-Fi link quality

```bash
python analysis/wifi_analysis.py --run coop2       # reuses extracts/coop2 from the NTP run
```

Radios are separated by the `interface` field, so an agent publishing two radios on
one topic is analysed as two links. Retry, failure and channel-occupancy rates are
differentiated from the driver's cumulative counters (which reset on reassociation)
rather than read as totals.

Outputs in `results/coop2/wifi/`:

| File | Contents |
|---|---|
| `wifi_links.csv` | per radio: interface, MAC, ESSID/BSSID, band, channel, width, PHY mode, RSSI percentiles, PHY rate, MCS/NSS, retry and failure rates, channel occupancy, missed beacons |
| `wifi_events.csv` | association, BSSID, ESSID and channel changes with their time in the run |
| `wifi_iperf.csv` | one row per iperf test: goodput, retransmits, RTT, direction, and whether it was to the server or robot-to-robot. Reported split by direction, because retry counters describe the uplink only — a station cannot count retries on frames sent *to* it, so reverse-direction tests and `missed_beacon` are the only downlink evidence |
| `wifi_field_availability.csv` | per radio and field, the fraction of samples carrying a real value — `WifiLinkStatus` documents whole groups (station dump, channel survey) as NaN or −1 when the underlying `iw` query is denied, and a statistic over those would be an artefact |
| `wifi_rho.csv` | for an agent with two radios: each radio's Bad fraction, the measured joint-Bad fraction, what independent links would have given, and the conditional and phi correlations |
| `wifi_summary.md` | the above as readable tables |
| `wifi_subsection.tex` | a paragraph for the paper with the numbers filled in |
| `fig_wifi_link.{pdf,png}` | RSSI, PHY rate with iperf goodput overlaid, TX failure rate, channel occupancy, on one time axis |
| `fig_wifi_rho.{pdf,png}` | the two radios' RSSI, their Bad intervals and the overlap, and measured vs independent joint-Bad |

The Bad state is `not associated`, or RSSI at or below `--bad-rssi-dbm` (default −70),
or TX failure rate above `--bad-failure-rate` (default 0.05). Both thresholds are
options because the right values depend on the deployment; whatever you choose is
printed in the figure legend and stated in the summary.

`wifi_rho.csv` and `fig_wifi_rho` are produced only when an agent actually has two radios,
which the script determines from the `interface` field rather than from the topic's publisher
count. Two publishers on one topic can equally mean the monitor node was restarted mid-run.

Every link is agent → access point, since both robots associate with the same AP.
The robot-to-robot iperf still traverses that AP and is reported as the two-hop path
it is, not as a direct link.

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
