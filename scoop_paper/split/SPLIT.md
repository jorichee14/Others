# Splitting SCooP into two papers

The current `main.tex` carries two papers at once: the instrument (agents,
calibration, clock, link logging, ground truth, replay) and the science (what
the link costs, what the geometry foresees). The split gives each its own
claim, and the dataset paper cites the testbed paper for everything below the
release level.

Order: submit the RA-L testbed paper first. The T-RO paper cites it, keeps
calibration, clock and GT to a one-paragraph summary with the validated
numbers, and adds no new testbed contribution.

---

## Paper A: SCooP-Bed, RA-L testbed (no dataset release; full plan in PAPER_A.md)

**Claim.** A reproducible, heterogeneous indoor multi-robot testbed in which
every frame carries the measured state of every wireless link, on one clock,
in one metric frame, without GNSS or motion capture, plus a replay module
that puts that link back in the loop for any collaborative method.

**What comes from the draft**

| Draft section | Goes to A as |
|---|---|
| System and Sensing Suite | Testbed architecture (agents, sensors, ROS topics, data flow) |
| Calibration (spatial, temporal) | Calibration and synchronization, with the clock numbers |
| Ground Truth: reference map, fiducial network, validation | Indoor ground truth without motion capture |
| Communication Ground Truth | Per-link measurement stack (status, iperf, CSI sniffer, NTP) |
| Benchmark: Channels, replay module | Channel-in-the-loop replay |

**Evaluation.** Experiments E1 to E5 are planned in full in PAPER_A.md:
- E1 Headroom: precision margin;
- E2 Map Meets Radio: clearance against RSSI;
- E3 Twin Run: message-level replay fidelity;
- E4 Same Knob, Same Link: repeatability;
- E5 Sim Gap: logged against synthetic delay.

**Structural models**
- `wilson2021robotarium` (RA-L): frames the testbed itself as the contribution.
- `coped` (RA-L): a heterogeneous team with a sync/calibration/GT pipeline for
  collaborative perception. It is the closest RA-L neighbour, so the
  comparison table must separate SCooP from it.
- `tian2023resilient` (IROS): system, then runs under real communication,
  then lessons learned.
- Communication framing: `calvofullana2021rosnetsim`, `saboia2022achord`,
  `ginting2021chord`, `blumenkamp2022framework`, `cladera2024mocha`.
- Measured-link cooperative perception: `qiu2022autocast`, `zhang2021emp`,
  `hawlader2024v2xexp`, `cooperscene`.
- Infrastructure-sensor systems: `krammer2022providentia`,
  `jimenez2011integrated` (mobile plus static sensors, the closest
  architecture).

---

## Paper B: T-RO dataset, in the vein of CODa

CODa earned T-RO with four empirical questions whose answers outlive the
dataset: the AV-to-robot gap, adaptation, resolution mismatch, and
cross-dataset pretraining. Each question had one controlled experiment, and
the full training tables went in the appendix. The same shape applies here,
with the question moved from "does single-agent perception learned on cars
transfer to robots" to "does *collaborative* perception learned on cars
transfer to robot teams indoors, and what governs its gain there".

**Q1. Transfer.** How well do collaborative perception models trained on V2X
datasets (OPV2V, V2V4Real, V2X-Real, DAIR-V2X) perform on an indoor robot
team? Report the ego AP and the cooperative gain G = A_coop - A_ego
separately, for direct transfer, fine-tuning, and training from scratch on
SCooP. Hypothesis: G shrinks more than ego AP does, because fusion learned
vehicle-scale range, height and object size.
*CODa analogue: Q1/Q2 (Tables IV and V).*

**Q2. Heterogeneity between collaborators.** How does a mismatch between ego
and helper sensors change the gain? This is a train-by-test matrix over the
helper's LiDAR resolution (128, 64, 32, 16 channels, downsampled as CODa did)
and the helper's modality (LiDAR robot, RGB-D cart, camera+radar mast).
*CODa analogue: Q3 (Table VI), moved from one robot to between agents.*

**Q3. Where the gain comes from indoors.** Is the indoor cooperative gain
driven by occlusion, by range, or by viewpoint (elevated infrastructure)?
Stratify G by the per-object occlusion, co-visibility and viewpoint diversity
the release computes (`mirc_dataset_paper/characterization.tex`). The first
evidence already points to occlusion: range rescue is 0.008 for the primary
agent against 0.29 for RGB-D. In driving datasets, range is usually the story.
*New. This is what the characterized trajectory paradigms are for.*

**Q4. What the real indoor link costs, and whether it is foreseeable.** Q1 and
Q2 of the current draft move here, merged:
- Retained gain 1 - Delta/G under the ideal, synthetic and logged channels.
- Delivery against content attribution (absent, late, or fused).
- The H1 latency cliff. In the OPV2V study, 100 ms puts all seven methods
  below the ego floor, and the logged RTT under contention is 60 to 109 ms.
- The geometry-only share of the loss, as a leave-one-site-out regression.
- Fusing with T1 (C-SLAM) poses instead of reference poses goes here as the
  localization half of the same cost.
*CODa has no analogue. This is what no outdoor dataset can answer.*

**Q5 (optional, CODa Q4 analogue).** Does pretraining on SCooP transfer better
than pretraining on V2X data to another real multi-robot dataset? CoPeD is the
candidate target if its labels line up.

**Benchmarks section** (as in CODa Sec. VIII): collaborative 3D detection
(AP at IoU 0.3/0.5/0.7) and collaborative segmentation (mIoU), with splits,
the frame-history rule, and allowed inputs (reference or estimated poses,
logged link).

**Appendix**: full training tables for every number, the annotation ontology,
the devkit, and the release structure, as in CODa Appendices A to E.
