# SCooP split: testbed paper and dataset paper

Source: the uploaded `SCooP.zip` draft (newer than `scoop_paper/`, which is left
untouched). Two self-contained LaTeX folders were created:

| Folder | Paper | Role |
|---|---|---|
| `scoop_testbed_paper/` | SCooP-Bed: An Indoor Multi-Robot Testbed for Collaborative Perception over Measured Wireless Links | Paper A, RA-L testbed paper (the instrument, its guarantees, and the proof it can be trusted) |
| `scoop_dataset_paper/` | SCooP: A Real-World Indoor Multi-Agent Dataset for Collaborative SLAM and Perception over Measured Wireless Links | Paper B, dataset / benchmark paper (what was recorded, how it is labeled and released, what the benchmark shows) |

Each cites the other (`\cite{scoop_testbed}` / `\cite{scoop_dataset}`; placeholder
bib entries appended to each `references.bib`). Nothing from the original was
deleted; text was reassigned, and bridging text (abstracts, intros,
contributions, summaries, lessons learned, conclusions) was written new.

---

## 1. Testbed paper, Paper A (`scoop_testbed_paper/main.tex`)

**Title.** SCooP-Bed: An Indoor Multi-Robot Testbed for Collaborative
Perception over Measured Wireless Links. Follows the plan already on the branch
(`scoop_paper/split/PAPER_A.md`, `EXPERIMENTS.md`).

**One claim.** A heterogeneous indoor team in which every inter-agent link is
measured alongside the sensors, all agents sit on one clock and in one metric
frame, there is no motion capture or GNSS, and any collaborative method can be
run against the link the agents actually had.

**Venue.** RA-L, 6 pages + 2 overlength. Submit first; the dataset paper cites it.

### Sections

| # | Section | Content (origin) |
|---|---|---|
| I | Introduction | Need, three-sentence gap (testbeds / comm-aware systems / coop-perception platforms), three contributions with numbers, highlight sentence (3 ms to 60-109 ms crosses one LiDAR frame). *(new)* |
| II | Related Work | A. Multi-robot testbeds (Robotarium, Duckietown, CPM Lab, ChoiRbot...). B. Communication in multi-robot systems (ROS-NetSim, CHORD/ACHORD, MOCHA...). C. Coop-perception platforms and measured links (CoPeD as closest RA-L neighbour, V2X-ReaLO, CooperScene). D. Indoor GT without a survey instrument (condensed from old Sec. II-D). New **capability table** `tables/capability.tex`. |
| III | Testbed Design | Agents (3 classes), network, **design decisions each with its reason**, recording + topics + data flow (old Sec. III, VII-B). BOM table in appendix. |
| IV | Time and Space Alignment | Clock, sniffer bias ("the measured link calibrates the clock"), error budget Eq. (1)-(2), extrinsics + residual table (old Sec. IV). |
| V | Ground Truth Without Motion Capture | Reference map + plane-fit ICP, fiducial chain Eq. (3)-(4), four independent checks, one-paragraph scoring capability (old Sec. V-A..C). |
| VI | Link Instrumentation | Link state, CSI, 1.2 m/s limit, contention knob (old Sec. V-E). |
| VII | Channel-in-the-Loop Replay | Model with one-way delay d_ab(t), interface, implementation, positioning against trace-driven emulation (netem, Mahimahi, Pantheon) and co-simulators. *(new; was old Sec. VIII-A)* |
| VIII | Experiments | **Spec sheet** `tables/spec.tex` (central table) + E1 Headroom (precise), E2 Map Meets Radio (consistent), E3 Twin Run (faithful), E4 Same Knob Same Link (repeatable), E5 Sim Gap (necessary); each with question, pass criterion, and result so far. |
| IX | Lessons Learned and Limitations | CSI shapes the network; NTP asymmetry; cold-start ICP basins; plane fits vs NN; limits. |
| X | Conclusion | |
| App. | Bill of materials and setup effort; sensor intrinsics | |

### What to highlight

1. **The spec sheet** (Table `tab:spec`): every guarantee with a measured bound
   and the check that produced it. Reviewers of a testbed paper read this first.
2. **The headroom argument (E1).** Testbed error (70 us clock -> 0.2 mm; 1.06 cm
   pose) is one to three orders of magnitude below the errors at which fusion
   breaks (0.2 m pose; ~100 ms delay stated as displacement). Make it the key figure.
3. **The link is measured, not loaded,** and contention is a knob (survey vs
   cooperation), with a measured effect: RTT 3.3 ms idle -> 60-109 ms under load.
4. **Ground truth without mocap or GNSS**, at building scale, from onboard sensing
   only: frozen self-built map, plane-fit ICP (3 cm offset hides behind 0.5 cm NN
   residual), boards posed from the map, infra posed through a shared board view.
   Four independent checks; say which error each bounds.
5. **E2 Map Meets Radio** validates poses and link logs together with no hand
   measurement: rho 0.60, 5.4 dB on the mobile_2 link already passes the
   pre-registered criterion.
6. **Channel-in-the-loop replay** as the testbed's offer to every method, and
   the one-way-delay point: only a synchronized testbed can measure d_ab directly
   rather than assume RTT/2. E3 Twin Run is what makes replay more than a simulation.
7. **E5 Sim Gap** as the reason to exist: raw clouds arrive within one frame 86 %
   alone vs 59-68 % under contention vs 100 % under Where2comm / V2X-ViT channels;
   three real outages (12 % of frames) that no synthetic channel produces.
8. **Lessons learned** with numbers (1x1 802.11ac at 80 MHz forced by the CSI
   receiver; aggregation caps CSI at 100-140 Hz; 2.4 GHz management-link bias).

Still needed (from `EXPERIMENTS.md`): reconcile the clock bound (70 us text vs
42.5 us table vs the coop_2 value), sniffer bias, calibration residuals, control
distances, cross-robot consistency, E2 pooled + agent-to-AP, E4 mode and day
halves, the Phase-2 session for E3, BOM costs, redrawn Fig. 1, all five
experiment figures.

---

## 2. Dataset paper, Paper B (`scoop_dataset_paper/main.tex`)

**One-line pitch.** The first indoor multi-robot dataset on which one can measure
what a real link costs C-SLAM and collaborative perception, and how much of that
cost the mapped geometry could have predicted.

**Venue.** IJRR data paper (default, per `scoop_paper/README.md`); T-RO if
the findings are strong. Note: `scoop_paper/split/SPLIT.md` plans a CODa-style
T-RO version with five questions (Q1 transfer from V2X data, Q2 heterogeneity
between collaborators, Q3 where the indoor gain comes from, Q4 link cost and
foreseeability, Q5 pretraining transfer). The benchmark section here carries the
draft's existing experiment, which is that plan's Q4; a comment at the top of the
section records this.

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
  the channel-replay module, and their validation numbers live in the **testbed**
  paper. The dataset paper
  carries a half-page summary plus two summary tables and cites the testbed paper.
- Annotations, geometry characterization, collection protocol, release format,
  devkit, benchmark, and discussion live in the **dataset** paper. The testbed
  paper mentions them only as downstream consumers of its outputs.
- Shared figures (overview, link trace, reference map) appear in both on purpose;
  swap one of them for a different image before submission if the venues overlap.
