# SCooP split: SCooP-iT testbed paper (RA-L) and SCooP dataset paper (T-RO)

Source: the uploaded `SCooP.zip` draft (newer than `scoop_paper/`, which is left
untouched). Two self-contained LaTeX folders, both on the IEEEtran journal class:

| Folder | Paper | Venue | Budget |
|---|---|---|---|
| `scoop_testbed_paper/` | SCooP-iT (SCooP Indoor Testbed): An Indoor Testbed for Collaborative Perception over Measured Wireless Links | IEEE RA-L | 6 pages + at most 2 overlength, references included; no appendix; extra material in `supplementary.tex` |
| `scoop_dataset_paper/` | SCooP: A Real-World Indoor Multi-Agent Dataset for Collaborative SLAM and Perception over Measured Wireless Links | IEEE T-RO regular paper | about 12 pages main text + references, appendices count; verify the current T-RO overlength policy |

Each cites the other (`\cite{scoop_testbed}` / `\cite{scoop_dataset}`). A page
budget per section is written as a comment at the top of each `main.tex`.
Nothing from the original was deleted; text was reassigned, and bridging text
(abstracts, intros, contributions, summaries, lessons learned, conclusions, the
Q1 to Q3 experiment designs) was written new. Submit RA-L first; T-RO cites it.

---

## 1. SCooP-iT testbed paper, Paper A (`scoop_testbed_paper/main.tex`)

**One claim.** A heterogeneous indoor team in which every inter-agent link is
measured alongside the sensors, all agents sit on one clock and in one metric
frame, there is no motion capture or GNSS, and any collaborative method can be
run against the link the agents actually had. Follows
`scoop_paper/split/PAPER_A.md` and `EXPERIMENTS.md`.

### Sections and page budget (total 8.2 pages, trim ~0.3 at camera-ready)

| # | Section | Pages | Floats kept in main text |
|---|---|---|---|
| I | Introduction | 0.8 | |
| II | Related Work: testbeds, comm-aware systems, coop-perception platforms with measured links, indoor GT without a survey instrument | 0.7 | capability table (full width) |
| III | Testbed Design: three agent classes, network, four design decisions with reasons, recording | 1.1 | Fig. 1 overview, platforms, compact sensors table, BOM |
| IV | Time and Space Alignment: clock, sniffer bias, error budget, extrinsics | 0.9 | calibration table (clock rows included) |
| V | Ground Truth Without Motion Capture: map + plane-fit ICP, fiducial chain, four checks, scoring capability | 1.0 | reference pipeline fig, GT table |
| VI | Link Instrumentation: link state, CSI, 1.2 m/s limit, contention knob | 0.6 | link trace fig |
| VII | Channel-in-the-Loop Replay: one-way-delay model, interface, positioning vs netem/Mahimahi/co-simulators | 0.5 | replay fig |
| VIII | Experiments: spec sheet, E1 to E5 | 1.4 | spec table (full width), headroom fig, E2 fig, E3+E5 two-panel fig |
| IX | Lessons Learned and Limitations | 0.35 | |
| X | Conclusion | 0.15 | |
| | References (~35) | 0.7 | |

Moved to `supplementary.tex` (RA-L multimedia attachment): full sensors table,
topics table, per-client clock table, reference-map figure, GT error
distributions, data-flow figure, sensor intrinsics, per-site setup effort.

### The seven questions the paper answers

Five about trust, two about worth. Pass criteria were fixed before the data
were read. "Answer now" uses the 24 Sep session at site 1 (mapping_A, survey_1,
coop_2). This table is `tables/questions.tex` in the paper.

| # | Question | Pass criterion | Answer now (24 Sep) | Remaining data to collect | Where |
|---|---|---|---|---|---|
| 1 | **Is it precise?** Are the testbed's own errors small next to the effects it measures? | every error term >= 10x below the fusion breaking points (0.2 m pose; 100 ms as displacement) | like against like at 1 m/s: measured clock 0.99 ms is 101x below one frame; map refinement 133x, repeatability 19x, board detector 24x below the 0.2 m pose point. Short of 10x: cart 3.1 cm (6.5x), strict clock bound 12 ms (8.3x). Tool: `scoop_pipeline/analysis/headroom.py` | mobile_1 vs boards on coop_2 (tool exists); radar reflector residuals; sniffer bias; measured p95 speed (`headroom.py --traj`); final margin per term | E1 Headroom |
| 2 | **Is it consistent?** Do the poses and the link logs agree when neither was checked by hand? | map clearance predicts detrended RSSI: rho > 0.3 and clear-minus-obstructed drop > 5 dB on one link kind | mobile_2 link rho 0.60, 5.4 dB: passes. infra_1 link rho 0.23, 4.7 dB: does not | agent-to-AP links (write the AP position to `ap.yaml`); pool over survey_1 and coop_2 and over sites; CI on rho; out-of-run prediction; CSI power as second signal | E2 Map Meets Radio |
| 3 | **Is it faithful?** Does replaying a logged link reproduce what the live link did? | >= 90 % agreement on which receiver frame each message lands in, and closer than a synthetic channel | none yet; replay is a model with measured inputs and unvalidated output (abstract says so) | one twin-run session on the survey_1 route: A1/B1 survey, A2/B2 cooperation, back to back; payload node and comparison script; per-size, per-mode table of arrival error, frame agreement, drop agreement | E3 Twin Run |
| 4 | **Is it repeatable?** Does the same setting give the same link on another day? | day-to-day distance < half the survey-to-cooperation distance, for RTT and goodput | pose half only: map repeats to 1.06 cm and 2e-5 in scale four weeks apart; mode half (survey_1 vs coop_2) computable, not yet computed | mode distance: `scoop_pipeline/analysis/comms/link_repeat.py survey_1 coop_2` (written, self-tested); one repeat day (the E3 session gives it); then `--against` prints the day/mode ratio | E4 Same Knob, Same Link |
| 5 | **Is it necessary?** What does a measured link show that a synthetic one hides? | demonstration, no pass | 1.5 MB messages within one LiDAR frame: 86 % alone (survey_1), 59 to 68 % with two robots (coop_2), 100 % under Where2comm / V2X-ViT; RTT 5 to 10 ms idle to 25 to 33 ms loaded; real outages on an earlier pass | best-fit synthetic baseline (constant or Binomial delay fitted to the measured delays); more passes and a second site; probe-off control run | E5 Sim Gap |
| 6 | **Is it unique?** What can it do that no existing platform can? | fills every column of the capability table | indoor, heterogeneous incl. infra, no mocap or GNSS, every pair's link measured, CSI, clock bound reported, replay into methods, perception scoring; CooperScene is the nearest case | verify every cell against the cited paper | Related Work |
| 7 | **Is it reproducible?** Can someone else build and use it? | bill of materials, setup effort per site, released pipeline, spec sheet with bounds | pipeline released; spec sheet drafted | costs per agent class; mapping-session length, number of boards, control distances per site | Design, Lessons |

Numbers whose pass should be confirmed before they go on a slide: surface
thickness 7.9 / 2.1 cm (mapping_A or the Aug map?), capacity 275-282 /
598-676 Mbit/s, CSI 141 / 104 Hz, chrony RMS 10.9 / 42.5 / 28.7 us.

### What to highlight

1. **The spec sheet**: every guarantee with a measured bound and the check behind it.
2. **Headroom**: the instrument is one to three orders of magnitude finer than the effects it measures.
3. **The link is measured, not loaded**, and contention is a knob with a measured effect (RTT 3.3 ms to 60-109 ms).
4. **Building-scale GT from onboard sensing only**, four independent checks, each bounding one error.
5. **E2** validates poses and link logs together with no hand measurement.
6. **Replay with measured one-way delay**, which only a synchronized testbed can do; E3 makes it more than a simulation.
7. **E5 as the reason to exist**: synthetic channels never cross the frame boundary and never produce outages.
8. **Lessons learned** with numbers (1x1 802.11ac at 80 MHz forced by the CSI receiver, aggregation caps CSI, management-link bias).

---

## 2. SCooP dataset paper, Paper B (`scoop_dataset_paper/main.tex`)

**One-line pitch.** The dataset on which collaborative perception for indoor
robot teams can be studied as findings, not only benchmarked: transfer from
vehicles, heterogeneity between collaborators, where the gain comes from, and
what the real link costs. CODa-style per `scoop_paper/split/SPLIT.md`.

### Sections and page budget (total 12.7 pages + appendices)

| # | Section | Pages | Content |
|---|---|---|---|
| I | Introduction | 1.0 | Three-layer gap, the open questions, four contributions (dataset, paradigms, release, benchmarks and findings) |
| II | Related Work | 1.3 | C-SLAM datasets; coop perception and comm datasets; ISAC datasets; indoor GT (short); collaboration geometry; comparison table |
| III | Acquisition Platform and Reference | 0.9 | SCooP-iT summary, agents table, reference-accuracy summary, communication ground truth, link trace |
| IV | Perception Annotations | 0.7 | Static and dynamic boxes, per-point labels, rendered and projected masks, box-fit check |
| V | Dataset Design and Collection | 2.0 | Environments; protocol; geometry characterization and paradigms; conditions; statistics |
| VI | Release | 1.2 | Bag and extracted formats, derived products, structure, devkit, license |
| VII | Benchmarks | 0.8 | T1 and T2 protocol: channels, replay, tasks, baselines, metrics, splits, allowed inputs |
| VIII | Experiments | 2.5 | Q1 transfer, Q2 heterogeneity, Q3 where the gain comes from, Q4 link cost and foreseeable share, failure cases |
| IX | Discussion | 0.8 | Q1-Q2, Q3, Q4 cost, Q4 foresight, limitations, ethics |
| X | Conclusion | 0.2 | |
| | References (~75) | 1.3 | |
| A-D | Appendices | | Sequence inventory; annotation ontology and statistics; full training tables; devkit and release details |

### Experiments needed

| Question | Experiment | Output |
|---|---|---|
| Q1 Transfer | 4 T2 detectors trained on OPV2V, V2V4Real, V2X-Real, DAIR-V2X; direct, fine-tuned, from scratch on SCooP; ideal channel | ego AP and gain G per source and regime (`tables/res_transfer.tex`); hypothesis: G shrinks more than ego AP |
| Q2 Heterogeneity | train-by-test matrix over helper LiDAR resolution (128/64/32/16) and helper modality (LiDAR, RGB-D cart, camera or camera+radar mast), ego fixed; early, intermediate, late fusion | G per train/test pair (`tables/res_hetero.tex`) |
| Q3 Where the gain comes from | stratify G by omega_e, r_e, beta, C from `_geom.yaml`; occlusion rescue vs range rescue | gain share per stratum per baseline (`figures/gain_strata.tex`); first evidence: range rescue 0.008 LiDAR ego vs 0.29 RGB-D ego |
| Q4a Link cost | T1 (Swarm-SLAM, DiSCo-SLAM, COVINS-G, Kimera-Multi, Hydra-Multi) and T2 under ideal / synthetic / logged channels | Delta and retained gain; synthetic vs logged misstatement; T2 on T1 poses; the 100 ms latency-cliff check |
| Q4b Foreseeable share | leave-one-site-out regression of per-frame loss on D, G, G+L | anticipated share per site and task; feature ablation |
| Failure cases | three link-caused, one with the link intact | figure |
| Collection | sessions table, totals, statistics figure, box-fit fraction, clearance vs RSSI, intended vs measured paradigm | tables and figures in Secs. IV-V |

### What to highlight

1. **Findings, not a leaderboard.** Four questions, one controlled experiment each, full training tables in the appendix. This is what carried CODa into T-RO.
2. **Q4 is the question no outdoor dataset can answer**, and the latency cliff (100 ms breaks every fusion method; the logged RTT under contention is 60-109 ms) is the sentence reviewers will quote if it holds.
3. **The comparison table**: each prior family covers at most two of the three layers.
4. **Annotations consistent across agents by construction**, authored in the reference frame.
5. **Collaboration geometry released per frame, independent of the signal it predicts**; it is what makes Q3 and Q4b possible at all.
6. **The three per-frame files nobody else ships**: link, geometry, CSI.
7. **Heterogeneity as a design axis** (RGB-D ego with a sensing budget, infra never ego, ego alternates), which is what Q2 measures.
8. **The fourth failure case** shows loss is attributed to the link only where the link caused it.

---

## Overlap policy

- Platform, calibration, timing, reference pipeline, link measurement design,
  the replay module, and their validation numbers live in **Paper A**. Paper B
  carries a half-page summary, two summary tables, and cites Paper A.
- Annotations, geometry characterization, collection protocol, release format,
  devkit, benchmarks, experiments, and discussion live in **Paper B**. Paper A
  mentions them only as downstream consumers of its outputs.
- No image file is used by both papers. The testbed keeps the map-binned link
  trace and gets a redrawn Fig. 1 (placeholder); the dataset keeps the top-view
  scenario and the reference-map render and gets a time-series link-trace figure
  (placeholder); the testbed supplementary gets a thickness-colored map (placeholder).
