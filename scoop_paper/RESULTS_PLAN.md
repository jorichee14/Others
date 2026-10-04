# SCooP results plan: what to answer, how, and in what order

Goal: turn the two questions of the paper into findings with numbers, so the
abstract's "one sentence with the headline result" can be written and the
venue (IJRR data paper, or T-RO if the finding is strong) can be decided on
evidence. The structure follows what carried CODa into T-RO: explicit
questions, one controlled experiment per question, and full training tables
in the appendix.

## 0. The three answers the paper needs

| | Answer, in the form it will appear | Produced by | Gates |
|---|---|---|---|
| A0 Ground truth | "Reference poses agree with held-out fiducials to X cm and with rangefinder control distances to Y cm; Z % of projected boxes enclose their object's returns." | Phase A | Everything. Without it the labels and the geometry are unverified. |
| Q1 Link cost | "Under the logged channel, collaborative perception retains X % of its cooperative gain and C-SLAM retains Y %; the loss is concentrated in contention-mode runs, where RTT exceeds T ms, and the synthetic channel misstates the cost by Z points." | Phases C, D, E | The headline for an IJRR submission. |
| Q2 Foreseeable share | "Map geometry alone predicts X % of the predictable per-frame loss on a building it was not fitted on; ego-to-helper distance alone predicts Y %." | Phase F | The headline for a T-RO submission if X is large. |

Decision rule for venue (Section 8) is applied after Phase F, not before.

## 1. Hypotheses to state before running anything

Write these into the Benchmark section now, so the results are read as tests
rather than as a search for something to report.

- H1 (latency cliff). In the OPV2V failure-attribution study
  (`../collab_perception_failure_analysis/results/ANALYSIS.md`), 100 ms of
  delay puts all seven fusion methods below the ego-only floor, and dropping
  90 % of messages never does. The logged RTT is 3.3 ms idle and 60 to 109 ms
  under load (Sec. Communication Ground Truth). Prediction: survey-mode
  sequences retain most of the cooperative gain, cooperation-mode sequences
  with contention lose most or all of it, and the loss is a content failure
  (stale messages fused) rather than a delivery failure (messages absent).
  This is the single most citable result available: the real indoor link
  sits exactly at the cliff, and the cliff is caused by queueing, not by
  path loss.
- H2 (message size). Early fusion, which sends raw clouds, is hit first
  because s / G(t) dominates its arrival time; late fusion, which sends
  boxes, is hit only by RTT. Where2comm under a goodput budget sits between.
- H3 (synthetic misstates). A fixed-budget, fixed-delay synthetic channel
  understates the cost because it has no outages and no load-dependent RTT.
- H4 (geometry foresees). Fresnel clearance and co-visibility predict a
  substantial share of per-frame loss across buildings; distance alone does
  not, because indoor loss is dominated by blockage and contention rather
  than range.
- H5 (misalignment valley, optional). Moderate pose error from the link
  (T2 on T1 poses) hurts more than large pose error. Only if time allows.

## 2. Phase A: ground-truth numbers (first, one to two weeks)

Closes todos at main.tex lines 457, 480, 496, 666, 669, 680, 721.

Inputs: `scoop_pipeline/map_stages/03_anchor.py`, `08_reference_traj.py`,
`analysis/mapping/map_quality.py`, `analysis/mapping/scan_consistency.py`,
rangefinder sheet per site, held-out board list.

Outputs:
1. Table `tables/gt.tex`: per site, N control distances, mean and 95th
   percentile of |measured - map| distance; held-out fiducial residual
   (translation, rotation); second-pass map agreement (surface thickness
   and inter-pass point-to-plane residual).
2. Figure `figures/gt_errors.tex`: residual histograms per site.
3. Box-fit fraction: for every projected static box, the fraction of
   enclosed LiDAR returns within d_in of the box surface; report the share
   of frames above the threshold and the threshold used.
4. One sentence on the bias bound (line 457) from the clock calibration.

Acceptance: no site relies on an estimator-only pose; every number has an
independent reference that did not enter the map.

## 3. Phase B: inventory and annotation statistics (parallel with A)

Closes todos at lines 310, 885, 1016, 1209, 1452, 733, 754.

1. Fill `tables/sessions.tex` and `tables/seq.tex`: per session the site,
   date and time, occupancy, idle channel-busy ratio, load mode, furniture
   changes; totals of sessions, sequences, hours, distance, frames, link
   samples per pair, CSI frames, annotated instances, per site.
2. Annotation statistics figure in the style of CODa Fig. 9: log-scale
   count per class, stratified by site and by load mode; list the classes
   with fewer than 100 instances in any stratum rather than hiding them.
3. State the keyframe rate, the instance-identity rule across agents and
   timesteps, and the QA acceptance threshold in one paragraph (CODa:
   95 % boxes, 90 % segmentation valid after internal QA).
4. Put the totals in the first paragraph of the dataset section. The scale
   question will be asked; answer it before it is asked, and argue that
   per-frame link state across every agent pair is the quantity that no
   other dataset has at any scale.

## 4. Phase C: geometry validation, independent of the outcome

Closes todo item (8) at line 1304 and underpins Q2.

1. Clearance against RSSI: for every (pair, frame), scatter Fresnel
   clearance from the map against measured RSSI and goodput; report
   Spearman correlation per site and the RSSI drop between clear (> 0.6)
   and blocked (< 0.6) frames. This proves the geometry labels predict the
   signal they never used.
2. Co-visibility against detection evidence: for every (object, agent,
   frame), ray-cast visibility against the number of returns inside the
   box. Report the confusion between "visible by map" and "has evidence".
3. Add both as a figure in the Conditions or Paradigms section. This is the
   analogue of CODa's split-validation KDE plot and is what makes the
   predictors credible.

## 5. Phase D: T2 under three channels (the critical path)

Closes todos at lines 1223, 1279, 1281, 1304 items (1), (4), (5), (6), (7).

Reuse, do not rebuild: the OpenCOOD-based harness in
`../collab_perception_failure_analysis` already runs early, late, AttFuse,
V2VNet, F-Cooper, CoBEVT and CoAlign with an impairment layer, validated to
±0.001 against published numbers. Port it in this order:

1. Data loader for SCooP (map-frame boxes projected into each agent's frame
   through the reference poses; per-agent clouds; per-pair link trace).
2. Re-parameterisation table (line 1223): detection range, voxel size,
   anchor sizes per class, from the class statistics of Phase B. CODa's
   evidence that outdoor-pretrained detectors transfer poorly and that
   resolution mismatch dominates is the justification; cite it.
3. Train PointPillars-backbone early and late fusion on the ideal channel
   with leave-one-site-out folds. Record every hyperparameter in an
   appendix table (CODa Table X format).
4. Add Where2comm (goodput budget from G(t)) and V2X-ViT (latency
   compensation). The four span message size, which is what the link acts
   on (H2).
5. Replay module: the three channels on the same frames. Synthetic channel
   parameters taken verbatim from Where2comm and V2X-ViT so H3 is a fair
   comparison.
6. Report per task and per output: A_ego, A_ideal, A_synth, A_logged,
   cooperative gain G, cost Delta, retained gain 1 - Delta/G, at IoU 0.3,
   0.5, 0.7 and mIoU. Split every row by load mode (survey vs cooperation)
   to test H1.
7. Attribute the loss: for each logged-channel frame, classify the partner
   message as absent, late beyond the detector's horizon, or fused. Report
   the share of the cost in each bucket. This turns H1 into a number.
8. T2 on T1 poses: appendix table, the extra loss from degraded
   localisation (line 1304 item 7, and H5 if pursued).

Minimum viable set for submission: early, late, Where2comm on detection;
early and late on segmentation. V2X-ViT and the segmentation backbone
choice (Cylinder3D, which CODa benchmarked and which is LiDAR-only) are
second priority.

## 6. Phase E: T1 under three channels

Closes todo at line 1304 items (2), (3).

1. Swarm-SLAM on all three agents with the GLIM single-agent reference.
   This is the comparable system to the one prior measured-link C-SLAM
   study, so it is the primary row and the minimum for submission.
2. DiSCo-SLAM on the two LiDAR agents (what a heterogeneous team loses).
3. COVINS-G centralised (most traffic).
4. Kimera-Multi and Hydra-Multi for the semantic rows, scored against the
   per-point labels. These separate alignment loss from content loss and
   are the stretch goal; drop them if they do not run indoors within a
   bounded effort and say so in Limitations.

Metrics: ATE, RPE against Phase A reference; accepted inter-agent loop
closures; time to first merge; per-exchange-window translation error
increase (the Q2 target for T1).

## 7. Phase F: Q2 regression, pre-registered

Closes todos at lines 1332, 1337.

Fix before fitting: target delta_t per task; feature sets D, G, G ∪ L
exactly as in Sec. "What the Map Could Foresee"; LightGBM with default
depth and early stopping on the training sites; leave-one-site-out, all
three folds; the anticipated share R2_G / R2_{G∪L} on the held-out site.

Report:
1. Share per site and per task, with D as the baseline it must beat.
2. Feature ablation in the appendix (drop one predictor at a time).
3. One qualitative sequence: predicted and actual loss over time, aligned
   with the link trace and the clearance.
4. Position against map-based link prediction work in Discussion.

Interpretation rule: a share above about 0.5 with G clearly above D
supports the T-RO framing (geometry anticipates the cost). A share below
about 0.3 means the paper's headline is the cost measurement (Q1) and the
geometry is released as a characterisation, which is still a valid data
paper claim. Between, report it plainly and let Q1 carry the paper.

## 8. Phase G: write-up

1. Results text in the order already listed at line 1304.
2. Failure cases (lines 1349, 1355) selected from Phase D and E outputs:
   one link-stress, one occlusion × link, one missed rendezvous, one
   failure with an intact link. The fourth is what shows the benchmark
   attributes loss to the link only where the link caused it.
3. Discussion: Q1 in one sentence first, Q2 in one sentence first, then
   the geometry observations, then limits.
4. Abstract headline sentence, conclusion headline sentence.
5. Appendix: full training tables for every number in the main text.

Venue decision: IJRR data paper is the default and the draft is already
shaped for it. Submit to T-RO only if Phase F returns a strong share or if
H1 holds cleanly (the real indoor link sits at the latency cliff, and the
cliff is contention). Either of those is a finding that outlives the
dataset, which is what distinguished CODa from a dataset release.

## 9. Order and dependencies

```
A (GT numbers) ─┬─> C (geometry validation) ─> F (Q2 regression)
                ├─> D (T2 three channels) ────┘
                └─> E (T1 three channels) ────┘
B (inventory, stats) ─> D (re-parameterisation needs class stats)
D, E, F ─> G (write-up)
```

A and B run in parallel now. D starts as soon as A delivers the reference
poses for one site, using that site for development and the other two
untouched until the folds are run. E runs alongside D on a separate
machine. F needs D and E outputs and takes days, not weeks. G is last.

## 10. Three risks, each with its answer

1. An outdoor-origin baseline does not run indoors. Ship the minimum viable
   set of Sections 5 and 6 and name the dropped systems in Limitations.
2. Q2 returns a low share. Apply the interpretation rule of Section 7; the
   paper does not depend on it.
3. Reviewers compare scale to CODa or V2V4Real. Phase B puts the totals up
   front and the per-pair link state is the argument.
