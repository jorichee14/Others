# SCooP split: testbed paper and dataset paper

Source: the uploaded `SCooP.zip` draft (newer than `scoop_paper/`, which is left
untouched). Two self-contained LaTeX folders were created:

| Folder | Paper | Role |
|---|---|---|
| `scoop_testbed_paper/` | A Heterogeneous Indoor Multi-Robot Testbed for Synchronized Sensing and Wireless Link Measurement with Survey-Free Building-Scale Ground Truth | Systems / instrumentation paper (how the data is acquired and why it can be trusted) |
| `scoop_dataset_paper/` | SCooP: A Real-World Indoor Multi-Agent Dataset for Collaborative SLAM and Perception over Measured Wireless Links | Dataset / benchmark paper (what was recorded, how it is labeled and released, what the benchmark shows) |

Each cites the other (`\cite{scoop_testbed}` / `\cite{scoop_dataset}`; placeholder
bib entries appended to each `references.bib`). Nothing from the original was
deleted; text was reassigned, and bridging text (abstracts, intros,
contributions, summaries, lessons learned, conclusions) was written new.

---

## 1. Testbed paper (`scoop_testbed_paper/main.tex`)

**One-line pitch.** A five-agent indoor platform in which the Wi-Fi link between
every agent pair is a measured stream rather than a transport, and every agent,
with or without a LiDAR, gets a validated pose in one building-scale frame
without a survey scanner.

**Suggested venues.** IEEE T-IM, IEEE Sensors Journal, RA-L (systems), Journal of
Field Robotics, or ICRA/IROS systems track. The "lessons learned" section
suits JFR / Sensors.

### Sections

| # | Section | Content (origin in the SCooP draft) |
|---|---|---|
| I | Introduction | Why a measured link plus indoor GT requires a purpose-built platform; four design principles; four contributions. *(new)* |
| II | Related Work | A. Multi-robot acquisition platforms (C-SLAM datasets, V2X-ReaLO, CooperScene). B. ISAC testbeds (DeepSense 6G, DICHASUS, DeepMIMO, Nexmon). C. Indoor GT strategies (the full taxonomy from old Sec. II-D). |
| III | System Overview | Design principles *(new)*; agents; frames; recording and topics table (old Sec. III). |
| IV | Calibration and Timing | Spatial calibration, chrony sync, sniffer bias, timestamp latency, error bounds Eq. (1)-(2), calibration validation + residual table (old Sec. IV, unchanged). |
| V | Reference Localization | Map + plane-fit ICP, fiducial network with Eq. (3)-(4), validation methodology (old Sec. V-A to V-C). Numbers moved to the Evaluation section. |
| VI | Communication Measurement | Network configuration, link state (survey vs cooperation mode), CSI and the 1.2 m/s rate limit (old Sec. V-E). |
| VII | Post-Processing Pipeline | Decode, reference poses, link/CSI parsing, merge to MCAP (old Sec. VII-B minus labels/geometry/extraction, which stay with the dataset). |
| VIII | Evaluation | Reference accuracy (Table gt, Fig. gt_errors); calibration and timing residuals; link characterization (Fig. linktrace); geometric clearance vs RSSI; end-to-end demo. *(numbers mostly TODO)* |
| IX | Lessons Learned and Limitations | CSI receiver dictates the network; aggregation caps CSI rate; sniffer off a second wireless hop; plane targets vs nearest neighbours; boards posed from the map. *(new, distilled from existing text)* |
| X | Conclusion | *(new)* |
| App. | Sensor intrinsics; Bill of materials | |

### What to highlight

1. **The link is measured, not loaded.** No sensor data crosses Wi-Fi; only chrony
   and probes do, and both are present in every run, so the recorded channel is the
   channel a deployed team would have. This is the single most distinctive design choice.
2. **Three-level link measurement on commodity hardware.** Passive (5 Hz), active
   goodput/RTT (iperf3, 100 Hz ICMP), and per-frame CSI (Nexmon, 242 subcarriers,
   100-140 Hz) on every pair, with cooperation-mode contention captured by design.
3. **Survey-free, annotation-grade ground truth for heterogeneous agents.** Frozen
   self-built map + plane-fit ICP + fiducials posed *from the map*; this is what lets
   an RGB-D pushcart and camera-only masts share the LiDAR frame. The plane-fit
   argument (3 cm offset -> 0.5 cm NN residual) is a concrete technical point reviewers
   will remember.
4. **Out-of-sample validation everywhere.** Second mapping pass (1.06 cm walls, scale
   2e-5), rangefinder control distances, held-out boards, cross-robot consistency;
   held-out reflector positions and boards for extrinsics; probe packets for the
   sniffer bias. Say explicitly which error each number bounds and which it does not.
5. **The timing error budget.** Synchronization contributes < 1 mm at 3 m/s; the
   dominant terms are sensor physics (LiDAR sweep, deskewed) and constant transport
   lags (removable). This pre-empts the usual "how well are the clocks aligned" question.
6. **The CSI rate vs robot speed argument** (half-wavelength 26 mm, 3 samples ->
   1.2 m/s) and why aggregation forbids simply sending more packets. It is an honest,
   quantitative limit and a reusable design rule.
7. **Lessons learned** as a contribution in itself: the CSI receiver forcing 1x1
   802.11ac at 80 MHz, the 2.4 GHz management link bias, the endpoint-clearance workaround.

Numbers still needed before submission: control-distance errors, board residuals,
cross-robot consistency, calibration residual table, sniffer bias, per-site link
characterization, end-to-end demo figure.

---

## 2. Dataset paper (`scoop_dataset_paper/main.tex`)

**One-line pitch.** The first indoor multi-robot dataset on which one can measure
what a real link costs C-SLAM and collaborative perception, and how much of that
cost the mapped geometry could have predicted.

**Suggested venues.** IJRR data paper (default, as in `scoop_paper/README.md`);
T-RO if the Q2 share is strong; RA-L fallback.

### Sections

| # | Section | Content (origin in the SCooP draft) |
|---|---|---|
| I | Introduction | The two questions, the three-layer gap, four contributions (old Sec. I; platform now deferred to companion). |
| II | Related Work | C-SLAM datasets; cooperative perception/comm datasets; ISAC datasets; indoor GT (shortened, points to testbed paper); collaboration geometry (full); positioning + comparison table (old Sec. II). |
| III | Acquisition Platform and Reference | Condensed: agents (new `tables/agents.tex`), reference poses with headline accuracy (new `tables/gt_summary.tex`), communication ground truth as the replayed reference (old Sec. III, V-A..C, V-E compressed). |
| IV | Perception Annotations | Static/dynamic boxes, per-point labels, rendered/projected masks, box-fit check (old Sec. V-D). |
| V | Dataset Design and Collection | Environments; protocol (site/session/sequence, ego/partner, infra never ego); collaboration geometry characterization Eq. (1)-(2), Fresnel clearance, paradigms, params; conditions; dataset statistics *(new placeholder)* (old Sec. VI). |
| VI | Release | Formats (MCAP + OPV2V-style), derived products, structure, three per-frame files with no counterpart, devkit, license (old Sec. VII). |
| VII | Benchmark | Channels (+ new `figures/replay.tex`), T1/T2 baselines, Q1 results, Q2 results, failure cases (old Sec. VIII). |
| VIII | Discussion | Q1, Q2, characterization, limitations, ethics (old Sec. IX). |
| IX | Conclusion | |
| App. | Sequence inventory; extended results | |

### What to highlight

1. **Lead with the two questions**, not with the sensor list. Q1: how much accuracy
   a real indoor link costs a team. Q2: how much of that cost the map already
   predicts. Every section should be visibly in service of one of them.
2. **Comparison table position.** Each prior family covers at most two of the three
   layers; SCooP is the only dataset with measured link + indoor full-trajectory GT
   + cooperative labels + per-frame geometry. Make that the figure reviewers screenshot.
3. **Annotations are cross-agent consistent by construction.** One box in the map
   frame -> same identity, size, position in every agent's view; per-point labels and
   rendered masks fall out for free. Prior datasets author labels independently of
   the reference; SCooP derives them from it, which is why the reference had to be
   annotation grade (hand off to the testbed paper for proof).
4. **Collaboration geometry as a released, per-frame, signal-independent quantity.**
   visible vs observed kept distinct; best helper and aspect angle beta; co-visibility
   C; Fresnel clearance computed from the map and validated against RSSI it did not
   use. Paradigms are predicates over these, assigned per frame, not asserted per run.
5. **The three per-frame files nobody else ships:** `_link.yaml`, `_geom.yaml`,
   `csi/`. This is the concrete artifact that makes bandwidth-, latency-, and
   geometry-conditioned fusion trainable from the frame index.
6. **The benchmark is one controlled experiment.** Same frames, three channels, one
   replay module; the channel is the only variable. Report Delta and retained gain
   1 - Delta/G, and use the synthetic run to show how prior impairment models misstate
   the cost. Leave-one-site-out makes Q2 a transfer test across buildings.
7. **The fourth failure case** (link intact, baseline fails anyway) is what shows the
   benchmark attributes loss to the link only where the link caused it. Keep it.
8. **Heterogeneity and elevation asymmetry** as deliberate: an RGB-D ego with a
   sensing budget, infra never ego, ego alternates so conditions attach to geometry
   rather than platform.

Numbers still needed: session table and totals, statistics figure, all T1/T2
results, Q2 share, failure-case figure, box-fit fraction, headline sentence for
abstract and conclusion.

---

## Overlap policy between the two papers

- Platform, calibration, timing, reference pipeline, link measurement design,
  and their validation numbers live in the **testbed** paper. The dataset paper
  carries a half-page summary plus two summary tables and cites the testbed paper.
- Annotations, geometry characterization, collection protocol, release format,
  devkit, benchmark, and discussion live in the **dataset** paper. The testbed
  paper mentions them only as downstream consumers of its outputs.
- Shared figures (overview, link trace, reference map) appear in both on purpose;
  swap one of them for a different image before submission if the venues overlap.
