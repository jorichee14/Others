# The paper

One file that says what is being written, what each section claims, which
figure or table proves it, and whether that evidence exists yet. When any other
document disagrees with this one about scope, this one wins.

**Title (working).** *Certified Pseudo Ground Truth Without Motion Capture:
Trajectory References from Sparse Surveyed Anchors*

**Venue.** RA-L or ICRA/IROS. MoCap2GT is RA-L Feb 2026 and must be cited; this
is adjacent to it, not competing with it.

**The one-sentence claim.** A trajectory reference can be built for a platform
carrying no reference sensor, from surveyed anchors visible for a *fraction* of
the run, and its error can be certified on an instrumented twin and transferred
with a stated model.

**The one number.** Construction gain **0.60** — 199.3 mm in the surveyed map
frame with no ground truth used, against 331.3 mm for the same trajectory fitted
*to* ground truth.

**What is explicitly NOT claimed.** The estimator. Factor graph + absolute-pose
factors + linear initialiser is four papers deep. Say so in section 4 and cite
them; the contribution is the regime, the certification and the protocol.

---

## Sections

### 1. Introduction
The gap: every recipe for reference-grade trajectories requires the evaluated
platform to *carry* the reference — a mocap volume, a LiDAR, a total station.
Robots that do not carry one are evaluated against nothing, or against another
SLAM system, which is not evaluation. State the four contributions.

### 2. Related work
Three families, and the sentence that separates this from each:
* **MoCap-fusion GT estimators** (EuRoC 2016, Vicon2GT 2020, Kalibr MoCap
  branch, MoCap2GT RA-L 2026) — make an *already instrumented* trajectory more
  accurate. Absolute pose at 100 Hz, continuously. Cannot run at all where
  there is none.
* **Survey-prior datasets** (Newer College, Oxford Spires ~1-2 cm, ConSLAM,
  Hilti) — the evaluated platform carries the LiDAR.
* **Pseudo-GT from a stronger SLAM** (e.g. PIN-SLAM on KITTI 11-21) — asserts
  the substitute is good enough rather than certifying it.

### 3. Problem and theory
`THEORY_AND_VALIDATION.md` §1-2. Error must not accumulate; three mechanisms
give that property without a LiDAR aboard; accuracy is then set by **absolute
coverage in time**. Lever (unbounded) vs bridge (Brownian, peak at the
midpoint, `sqrt(T)/2`). This section makes a falsifiable prediction, which §6.2
tests.

### 4. Method
The graph, the factor set, the adaptive assembly rule, the observability
refusals. Say plainly that the estimator is standard. The structurally new
factor is the backbone odometry term, present *because* the absolute evidence
is sparse — MoCap2GT has no analogue because it never needs one.

### 5. Certification protocol
Tiers A/B/C. **Every absolute source is a factor or a check, never both.** The
three failures that look like success and the check that catches each.

### 6. Experiments
**6.1 coop2 — the instrument.** Benchmark grid, then the construction, then the
certification battery.
**6.2 Does the theory hold?** Error against time-to-nearest-anchor on
`mobile_1`, measured against the §3 prediction.
**6.3 GT ablation on public datasets.** The coverage sweep, and lever vs bridge
controlled on identical data.
**6.4 Transfer.** `mobile_2`, with the independent checks and the
platform-transfer assumption stated.

### 7. What a dataset must contain
The design rules, each from a measured failure rather than from taste.

### 8. Limitations
Single recording for the native experiments; the transfer assumption; the
sources this recording cannot price.

---

## Figures

| # | what | status |
|---|---|---|
| **F1** | System overview, MoCap2GT Fig. 2 shape: front end → initialisation → graph → **certification → transfer**. The bottom two boxes have no counterpart in their figure and are the contribution | **not drawn** |
| **F2** | Lever vs bridge. The coverage diagram plus the predicted error curve | **not drawn** |
| **F3** | **Error vs time-to-nearest-anchor on `mobile_1`**, measured points over the predicted Brownian-bridge curve | **not run** — cheap, data on disk |
| **F4** | Construction gain vs kept-GT fraction, one line per public dataset | **not run** — the bulk of the work |
| **F5** | Lever vs bridge, controlled: same data, windows at both ends vs one end | **not run** |
| **F6** | Qualitative: backbone, construction and reference in the map frame, anchors marked | **not drawn** |

## Tables

| # | what | status |
|---|---|---|
| **T1** | Benchmark grid, 3 backbones × 2 platforms + the IMU ablation, every tier-1 with its tier-2 and span | **done** |
| **T2** | V1 vs V2c: the matched pair and all five backbones | **done** |
| **T3** | Certification: sigma sweep, hold-one-out, coverage, move by region | **done** |
| **T4** | Cross-dataset construction gain at matched coverage | **not run** |
| **T5** | The six comparable metrics, per dataset | **partial** — coop2 only |

---

## Status, honestly

**Done and defensible:** §1, §2, §3 (written), §4, §5, §6.1, §6.4, §7, T1, T2, T3.

**The two gaps that decide whether this is a paper or a case study:**

1. **F3 — the theory check.** One plot. Needs no new runs. If the measured
   curve matches the prediction, §3 stops being an argument and becomes a
   validated model, and the transfer in §6.4 inherits its authority. If it does
   not match, better to know now than in review.
2. **§6.3 / F4 / F5 / T4 — GT ablation on public datasets.** This is the bulk
   of the remaining work and it is what turns one recording into a controlled
   experiment on five. Without it the obvious review is *"validated on a single
   sequence the authors recorded themselves."* With it, the coverage model is
   measured across platforms, scales and motion types, and the lever is a
   reproducible phenomenon rather than an anecdote.

**External dependency:** the **map cloud**. `map_cloud` is the only null left
that matters, and it is the difference between `mobile_2` having no independent
check and having the strongest one available.

**Order of work.** F3 first — it is a day and it can falsify §3, which
everything downstream leans on. Then §6.3, starting with EuRoC to establish the
ablation protocol, then TUM-VI for its natural gap structure, then Newer
College for mechanism B. Figures last.
