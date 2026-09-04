# Run characterization analysis

One script per topic, each characterizing one recorded run (a rosbag2 MCAP) for the
dataset paper. No ROS installation is needed: message definitions are read from the
schemas embedded in the MCAP file.

```
analysis/
├── common.py               shared: agent naming, extraction loaders, pose join, map background
├── extract_bag.py          shared: decodes the MCAP into per-topic Parquet tables
├── ntp_analysis.py         NTP / temporal calibration
├── ntp_issues.md           NTP basics, what went wrong in coop2, and the fixes
├── wifi_analysis.py        Wi-Fi link quality and dual-radio correlation
├── wifi_issues.md          what the Wi-Fi suite measures, coop2 findings, fixes
├── csi_analysis.py         Wi-Fi CSI: multipath structure of the channel
├── csi_core.py             shared CSI maths, numpy only (no pandas, no matplotlib)
├── csi_image_node.py       ROS 2 node: one rendered image per CSI frame, live
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
python analysis/wifi_analysis.py --run coop2 \
    --map map_stages_20260828_outputs/map_final_20260828_nc_anchored.pcd
```

`--map` takes the **anchored** point cloud, drawn as the greyscale background of
the coverage panels; without it the trajectories are drawn on their own. Link
samples are placed by interpolating the `--pose-topic` (default `global_pose`)
at each sample's timestamp, so the coverage map is in the same frame as the
trajectories.

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
| `fig_wifi_link.{pdf,png}` | one coverage panel per radio — the trajectory over the map, coloured by RSSI, start ○ and end ■ marked, iperf tests ✕ — above the RSSI time series for every radio. The other quantities (PHY rate, retries, occupancy) stay in the tables: a panel is spent only on something that varies |

All coverage panels share one colour ramp and one scale, so a colour means the
same RSSI in every panel. The scale is trimmed to the pooled 2nd–98th percentile
so one outlying sample cannot compress every trajectory into one end of the ramp.
Panels are zoomed to their own trajectory rather than to the whole map, since on
a room-sized map a path otherwise occupies a tenth of the panel and its gradient
is unreadable.
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

## Wi-Fi CSI

```bash
python analysis/csi_analysis.py --run coop2 --map <anchored>.pcd
```

A `CsiFrame` is the complex channel response of **one received 802.11 frame**
across OFDM subcarriers — where RSSI is one number, CSI is a vector across
frequency. It measures the propagation environment, not link performance.

Three quantities are derived, all of them ratios and so immune to the receiver's
automatic gain control:

| Quantity | What it says |
|---|---|
| frequency selectivity | spread of \|H\| across subcarriers in dB — flat means one dominant path, notched means multipath cancellation |
| RMS delay spread | how far multipath energy is spread in time, from the power delay profile |
| Rician K-factor | power of the dominant path against everything else — the signature of an unobstructed path |

**Phase is not used.** Every frame carries an unknown carrier and sampling offset
plus a random packet-detection delay, so recorded phase is meaningless across
frames without sanitisation. Only \|H\| and the delay profile survive those
offsets, and both are what the script uses.

**Delay-spread resolution is 1/bandwidth**: 12.5 ns at 80 MHz, 50 ns at 20 MHz.
Indoor delay spreads are tens of nanoseconds, so at 20 MHz the number is one tap
wide and the summary says it should not be quoted.

**Two tests decide whether a run's CSI is usable at all**, and they are the
first thing to read in `csi_summary.md`:

| Test | Column / file | Pass |
|---|---|---|
| is it a channel? | `temporal_coherence` in `csi_inventory.csv` | consecutive frames correlate > 0.8; noise reads near 0 |
| does it follow the robot? | `csi_motion_test.csv`, `fig_csi_motion.png` | change rate rises with ground-truth speed (ρ ≥ 0.3) and is at least 2× higher moving than still |

Every other metric returns a number for noise as readily as for a channel, so
none of them should be quoted for a run that fails the first test. The second
is what ties the CSI to the robots rather than to a radio, and is the result
worth reporting. It needs only the ground-truth pose topics.

Outputs in `results/coop2/csi/`:

| File | Contents |
|---|---|
| `csi_inventory.csv` | per agent: frame rate, channel, capture and occupied bandwidth, usable subcarriers, temporal coherence, tap spacing |
| `csi_transmitters.csv` | per source MAC and frame type: count, share, median RSSI, time span — which transmitter's channel each measurement describes |
| `csi_frames.parquet` | per frame: selectivity, delay spread, K-factor, RSSI |
| `csi_summary.md`, `csi_subsection.tex` | tables and a paragraph for the paper |
| `fig_csi.{pdf,png}` | channel amplitude heat map per agent (diverging about each frame's median, so fades and peaks separate), then delay spread and K-factor over the run |
| `fig_csi_map.{pdf,png}` | the trajectories coloured by K-factor — the same layout as the Wi-Fi coverage map, dark = better channel |
| `csi_motion_test.csv`, `csi_motion_bins.csv`, `fig_csi_motion.{pdf,png}` | the motion test: verdict per agent, the 1 s bins behind it, and change rate against speed |

## Live CSI view in ROS 2

`csi_image_node.py` subscribes to a `CsiFrame` topic and publishes a rendered
`sensor_msgs/Image` (and a JPEG `CompressedImage`) for **every frame**, so the
channel can be watched in `rqt_image_view` while the robot drives.

```bash
python3 analysis/csi_image_node.py --ros-args \
    -p input_topic:=/mobile1/csi -p publish_every_n:=4
```

One panel, three views of the same frame, plus a header with the source MAC,
RSSI, Rician K and RMS delay spread:

| View | What to watch for |
|---|---|
| waterfall | subcarrier against time. A clean channel bands smoothly; a blockage breaks it into deep, fast-moving notches — usually visible before RSSI has moved much |
| \|H\| vs subcarrier | the current frame's frequency response, ±dB about its own median |
| delay profile | rolled onto its strongest tap, so the axis is excess delay past the direct path. One tall tap = direct path dominates; a long tail = scatter |

**Bandwidth.** CSI arrives at ~170 Hz and a 640×480 `bgr8` image is 920 kB, so
one image per frame is roughly 150 MB/s. That is fine over shared memory on the
robot and hopeless over Wi-Fi. The node prints the figure it is about to produce
at startup. Raise `publish_every_n`, or set `publish_raw:=false` and subscribe to
the `/compressed` topic.

`cv2` is optional: without it the node still renders the waterfall and both
plots, but drops the text overlay and the compressed topic.

To check the layout without a robot:

```bash
python3 analysis/csi_image_node.py --selftest /tmp/csi.png
```

That renders the panel from a synthetic channel that switches from clear to
blocked partway through, so the two signatures can be compared side by side.

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
