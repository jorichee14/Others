# Literature check, 2026-09-20 — what survives and what has to be engaged

**Method and caveat.** Searched for prior art on each of the four claims in
`PAPER.md`. **arxiv.org and ram-lab.com are blocked by this container's egress
proxy**, so what follows is built from abstracts, indexed summaries and
publisher pages — *not* from full texts. Two papers below are close enough that
they must be read properly before the related-work section is written, and both
are flagged.

---

## The verdict, claim by claim

| claim | status | why |
|---|---|---|
| 1. pseudo-GT for a platform carrying **no** reference sensor | **survives** | every GT estimator found requires the device under test to carry, or be tracked by, the reference |
| 2. a **certified** error figure that never uses the reference | **contested** | PALoc reports uncertainty for prior-map GT. Different kind of uncertainty — see below |
| 3. **transfer** to an uninstrumented agent | **survives** | nothing found that fits an error model on an instrumented twin and evaluates it elsewhere |
| 4. dataset-design rules from measured failures | **survives, and strengthened** | the closest method uses a *single* anchor at a time, which is exactly the lever this project measured at 7.26 m |
| — GT-ablation validation protocol (§6.3) | **NOT novel — and that is good news** | **CORRECTED on a second search.** The established terms are *masking* and *simulated outage*, not "ablation". RTK-SLAM (arXiv 2604.07151) masks the GNSS stream while keeping the dense reference for scoring — the identical protocol. Cite it; claim the USE, not the protocol |

---

## Two papers that must be engaged

### PALoc — TMech 2024, *Prior-Assisted 6-DoF Trajectory Generation and Uncertainty Estimation*

Closest prior art on **claim 2**. Prior-map-assisted framework producing dense
6-DoF GT poses; loosely-coupled factor graph over LO / LIO / LVIO; explicitly
derives covariance **within the factor graph** to analyse pose uncertainty
propagation; handles degenerate and stationary segments. Open source.

**Why it does not close the gap.** Its degeneracy-aware map factor consumes a
prior point cloud **and a LiDAR frame** for point-to-plane optimisation, so the
evaluated platform must carry a LiDAR. `mobile_2` does not. PALoc cannot run
here at all — not badly, at all.

**The distinction to make, and it is one this repo already argues elsewhere:
their uncertainty is the estimator's own covariance, which is PRECISION
conditional on the model being right.** `configs/coop2.yaml` makes exactly this
point about the coop2 pipeline's convergence statistic — *a trajectory and a map
can converge to a consistent solution that is jointly wrong*, which is why the
reference's `uncertainty_m` is taken from the board survey and not from the
convergence table. A graph covariance cannot see a mis-declared extrinsic, a
wrong board convention, or a systematically biased anchor. **Certification
against a held-out sensor can.** That is accuracy, not precision, and it is the
honest way to hold the line on claim 2.

**Do before writing §2:** read the full paper, and check whether it reports any
held-out validation of its covariance — if it does, the distinction narrows and
must be stated more carefully.

### Marker-Constrained Pose-Graph Correction — arXiv 2608.16281, **August 2026**

**The closest prior art on the method itself, and one month old.**
Camouflage-matched CSR fiducial markers as **pre-surveyed visual anchors**;
RGB detection → PnP → coarse similarity alignment → **marker-constrained
pose-graph optimisation**; georeferences both a LiDAR-odometry trajectory and a
dense **RTAB-Map** reconstruction into a common geodetic frame without GNSS.
Reported: self-consistency error down 97.9–99.1% on three of four
session/trajectory combinations; **held-out georeferencing accuracy improved
8–30% over one-time alignment alone**; marker-pose repeatability 15.3–18.8 cm.

The overlap is substantial: surveyed markers, pose-graph correction, RTAB-Map,
and a held-out evaluation. **The "8–30% over one-time alignment" experiment is
the same shape as this project's V1-vs-V2c (331.3 → 199.3 mm, 40%)**, and a
reviewer will put the two side by side.

**Four differences, in descending order of how well they hold:**

1. **Purpose.** They georeference into a geodetic frame so heterogeneous
   pipelines share coordinates. This produces a trajectory *reference for
   benchmarking*, which is a claim about accuracy rather than about
   consistency, and carries the obligations that go with it.
2. **Certification.** Their held-out evidence is another marker. This holds out
   an entire **reference sensor** — a LiDAR that could have produced ground
   truth — which is a strictly stronger test.
3. **Transfer.** They do not carry an error model to a platform with no
   reference. That is claims 1 and 3, and it is the whole point here.
4. **Anchor geometry.** A **single** iMarker relocated among six surveyed
   positions — so at any instant there is one anchor. That is precisely the
   configuration this project measured as a lever: one anchor transports a
   trajectory rigidly without bending it, and the solve moved **7.26 m** while
   reporting convergence at whitened residuals of 0.01/0.08. Their 15.3–18.8 cm
   marker repeatability is consistent with a well-behaved local fix and says
   nothing about the far end of a chain. **This makes claim 4 more valuable,
   not less** — it is a concrete, measured warning to the nearest neighbour in
   the literature.

**Do before writing §2:** read the full paper. Check whether the six marker
positions are ever *simultaneously* surveyed and used, and how far their
evaluated trajectories run from the nearest anchor. If they are bracketed at
both ends, difference 4 weakens.

---

## The §6.3 protocol has a direct precedent — correction to the first pass

The first search for this used the phrasing *"ablate ground truth"* and found
nothing on the SLAM side. That was a search failure, not an absence. The
established vocabulary is **masking** and **simulated outage**, and the
protocol is published:

**RTK-SLAM dataset** (arXiv 2604.07151), *Absolute Accuracy Evaluation in
GNSS-Degraded Environments*: outages are *"simulated by masking the GNSS
measurement stream while the IMU stream continues, with masking keeping the
continuous RTK reference available for every outage"*, which makes the protocol
exactly reproducible. Mask the absolute-measurement stream, keep the dense
reference for scoring. They mask GNSS; §6.3 masks board sightings. Same
structure, same purpose.

**Two further findings in that paper land on choices already made here:**

* *"SE(3)-aligned ATE can underestimate absolute global errors by up to 76%."*
  Independent support for scoring V2c **unaligned**, which is otherwise the
  most unconventional-looking decision in the evaluation.
* They report **three distinct error regimes as GNSS availability degrades** —
  the coverage sweep of §6.3, run with real degradation rather than masking.

**Anchor theory has its own literature too.** *Anchor selection for SLAM based
on graph topology* (UTS) describes anchors as introducing *"zero-uncertainty
loop closures between the anchors and the origin"*, which is precisely why one
anchor transports a chain rigidly and two bracket it. So **where to place
anchors is an established question**, and claim 4 is a contribution to a topic
that already exists rather than an observation nobody asked for.

**Princeton365** reports pseudo-GT *"accurate to approximately 20 cm,
sufficiently accurate to measure keyframe errors larger than 50 cm"* — a
published precedent that a ~20 cm pseudo-GT is a legitimate instrument, at the
same order as the 199 mm here. It answers "what is 20 cm good for" with a
citation instead of an argument.

**What to claim, then.** Not the protocol. **The use:** nobody has swept anchor
coverage to measure the accuracy law, and nobody has used masking to certify a
GT-generation method for a platform that carries no reference at all. A
protocol with a precedent is one a reviewer accepts without argument, which is
a better position than owning it.

SparseGT (Penn, medical imaging) stays as a cross-domain footnote at most; the
RTK-SLAM citation supersedes it.

---

## HortiMulti / Poly-TagSLAM — the closest certification, found on the third pass

[arXiv 2603.20150]. Generates GT by fusing **LiDAR-inertial odometry with
AprilTag detections** in a factor graph, and validates it by **leave-one-out
cross-validation over 35 total-station-surveyed landmarks**: optimise with the
remaining N-1 as tight priors, predict the held-out one, record the 3D error.
**5.9 cm mean / 6.6 cm RMSE.**

So leave-one-out certification of a fiducial-anchored ground truth is
published. It must be cited, and it is the direct comparison for any
certification claim here.

**Two differences, and the second is the contribution.**

1. **Their backbone is LiDAR-inertial** — the platform carries the reference
   modality. The subject platform here does not.
2. **Their 35 landmarks are all the same kind.** Every AprilTag fails for the
   same reasons: range, incidence, blur, lighting, and any systematic bias in
   how the tags were surveyed or how the detector behaves. **Leave-one-out over
   homogeneous landmarks cannot see a failure common to all of them** — each
   held-out tag is predicted well and the trajectory is wrong together.

That second point is this project's own backbone-consensus rule, applied one
level up: *three similar RGB-D systems would agree through a shared failure
mode and certify nothing.* The rule was written for backbones; it applies to
absolute sources identically, and nobody appears to have applied it there.

**The honest framing:** you cannot always survey 35 landmarks. You can usually
arrange three different **ways of being seen** — a fiducial, a static observer,
a peer robot, a prior map — that fail for unrelated reasons. Diversity of kind
substitutes for count of instance, and it certifies something count cannot.

Also found: *Factor Graph-Based Ground Truth Trajectory Estimation by Fusing
Robotic Total Station and Inertial Measurements* (IEEE 2025) — one absolute
source plus IMU, same family.

---

## LaMAR (ECCV 2022) — READ IN FULL 2026-09-21, and it costs three claims

arXiv 2210.10770. GT for HoloLens and phones — devices carrying no reference
sensor — by registering their streams against NavVis laser scans. Open source.
**The closest prior work by a distance, and four of my earlier verdicts about
it were wrong.**

**Their pipeline (§4.1-4.3), accurately:** a reference model built by pairwise
image+ICP registration of scan sessions and a pose-graph global alignment; then
per-sequence: retrieval + local features lifted to 3D through the mesh → P3P
in LO-RANSAC; a **voting** rigid alignment that rejects confident-but-wrong
localizations; a pose graph over tracker relative poses + absolute localization
priors, robustified with **Geman-McClure under Graduated Non-Convexity**;
**guided localization** re-mining pairs by the mesh-based visual-overlap score;
then bundle adjustment with reprojection errors, lidar-sampled 3D points
carrying a prior; then a joint cross-sequence refinement. Ceres.

**RETRACTED, each from my own earlier notes:**

1. *"Nobody has closed the loop on Brachmann."* They cite Brachmann [8] in §4.4
   and answer it directly.
2. *"No per-pose uncertainty."* They invert the Hessian of the refinement and
   calibrate it by empirical keypoint noise (sigma = 1.33 px), report its
   spatial distribution, and **filter on it** — keeping poses correct within
   10 cm at 99.7% confidence (sigma_t <= 3.33 cm), discarding 0.8% of frames.
3. *"Isotropic / no degeneracy handling."* Absolute-term covariance is
   propagated from the localization refinement, relative from the odometry
   pipeline. Fig. 4 shows they observe the degeneracy directly: *"translation
   uncertainties are larger in long corridors and outdoor spaces."*
4. *"One absolute source kind."* Overstated. They fuse visual + inertial +
   lidar structure. The precise statement is **one ABSOLUTE source (the lidar
   reference model) and several RELATIVE ones**.

**WHAT SURVIVES, and it is now a single sharp thing.** Their validation (§4.4)
is exactly two things, both internal:

* the inverted Hessian — precision conditional on the model, and they say so:
  *"We do not claim that our GT is perfect but analyzing the optimization
  uncertainties sheds light on its degree of accuracy."*
* rendering the mesh at the estimated poses and checking pixel alignment — but
  **the poses were fitted to that mesh**. Alignment is near-guaranteed once the
  fit converges, and it cannot reveal a distortion of the mesh, because the
  render and the poses share it.

**No external evidence enters §4.4 anywhere.** So their reply to Brachmann is an
ARGUMENT, not a MEASUREMENT: *"Careful design and propagation of uncertainties
reduces the bias towards one of the sensors."* The residual bias is never
measured against anything independent, because nothing in the pipeline fails to
descend from the reference model.

**The gap, stated precisely:** LaMAR argues that fusing enough complementary
sensors and propagating uncertainty carefully removes the pseudo-GT bias
Brachmann identified. That argument has never been tested, because testing it
needs evidence that does not descend from the reference model. Surveyed
fiducials are such evidence. The experiment is: run the pipeline, then measure
its residual at fiducials it never saw. Consistent confirms their argument for
the first time; inconsistent finds a bias the field's own uncertainty cannot
see.

**Consequences.** Claim 1 (GT for a platform with no reference sensor) is
**taken** — LaMAR did it in 2022, at scale. Claim 3 (the coverage law) is
**weakened**: Fig. 4 is that phenomenon, mapped empirically; a predictive model
in transferable quantities is still distinct but thin, and should be a section
rather than a contribution. **Claims 2 and 5 — certification against evidence
that does not descend from the reference, by leave-one-source-out — are the
paper.** Adopt their pipeline rather than competing with it: Geman-McClure+GNC
beats the Huber implemented here, their covariance propagation is more careful,
and guided localization is the bridge-filler this construction lacks.

Also note Brachmann et al., ICCV 2021, arXiv 2109.00524 — *On the Limits of
Pseudo Ground Truth in Visual Camera Re-localisation*. Shows the reference
algorithm's choice biases rankings toward its own family, overturning in their
data the beliefs that scene-coordinate regression beats feature methods and
that RGB-D beats RGB. **This project measured the same effect independently**:
on `mobile_2`, whose reference is cuVSLAM-derived, ATE ranks the image-based
method first (120 < 135 mm) while the surveyed boards rank it second by a
factor of two (201 vs 94-123 mm at `rs_anchor`). That is Brachmann's effect in
trajectory benchmarking rather than relocalisation, and it is the pilot result
for the experiment above.

---

## Papers to cite, not contest

| paper | why it is in §2 |
|---|---|
| **MoCap2GT** (RA-L 2026) | the named baseline. MoCap + IMU fusion, B-spline, time offset, hand-eye init. Must be cited regardless |
| **XR-HPGT** (TVCG 2025, arXiv 2512.07221) | continuous-time MLE + IMU to compensate mocap jitter, **variable time synchronisation**, screw-congruence pose residual. Confirms that continuous-time representation and time-offset estimation are now standard practice — see the gap below |
| **Vicon2GT** (2020), **EuRoC** (2016), Kalibr MoCap branch | the estimator lineage this design deliberately reuses |
| **RTS-GT** / Vaidis et al. | robotic total stations: genuinely sparse absolute evidence, with an uncertainty analysis. The closest existing work on *sparse* absolute measurement, and the right citation for §3 |
| **Newer College**, **Oxford Spires**, **ConSLAM**, **Hilti** | the survey-prior family; the evaluated platform carries the LiDAR |
| **Boxi** (arXiv 2504.18500) | sensor-suite design decisions against algorithmic performance; supports claim 4's framing |
| **"Look Ma, No Ground Truth!"** (arXiv 2412.01116) | GT-free tuning of SfM/VSLAM — the right neighbour for the tier-C self-consistency arguments |
| **RTK-SLAM** (arXiv 2604.07151) | **the §6.3 protocol's direct precedent** — GNSS masking with the dense reference retained. Also supports unaligned scoring (SE(3) alignment underestimates global error by up to 76%) |
| **Princeton365** | a ~20 cm pseudo-GT used as a legitimate instrument; the precedent for what this accuracy class is good for |
| *Anchor selection for SLAM based on graph topology* (UTS) | anchors as zero-uncertainty loop closures to the origin; establishes anchor PLACEMENT as a recognised question |

---

## Two things this check changes

**1. A visible gap relative to current practice: no time-offset estimation.**
MoCap2GT carries time offset as a B-spline; XR-HPGT makes variable time
synchronisation a headline. This construction assumes a common clock and
interpolates linearly with SLERP. That is defensible here — board detections
and images share a header stamp, and the anchors are sparse enough that
interpolation error is a small share of 199 mm — but it must be *argued* in §4
rather than left unmentioned, or it reads as an oversight. A scalar per-stream
offset is cheap to add and would remove the objection entirely.

**2. A new ablation target worth checking: Hilti-Trimble SLAM Challenge 2026** —
a 360 visual-inertial benchmark **with floor-plan priors**, ground truth
released for all sequences. Floor-plan priors are sparse absolute information,
which is this method's native input, and released GT is exactly what the §6.3
ablation protocol consumes. Check it before finalising the dataset list.

---

## What did NOT turn up, and what that is worth

No hit for: ground truth generated for an agent that lacks the reference
modality; an error model fitted on one platform and evaluated on another;
sweeping anchor coverage to measure the accuracy law; or multi-agent anchoring
for GT production. (The masking protocol itself DOES have a precedent — see
above. That claim was withdrawn on the second search.)

**Read that carefully.** Absence in a search is weak evidence — arXiv was
blocked, the searches were English-language and keyword-driven, and the nearest
paper is one month old, which means the area is moving. Treat the four claims
as *provisionally* clear and re-run this check before submission. The one that
would hurt most if it exists is a 2026 paper doing marker-anchored GT for an
uninstrumented platform; search that specific phrasing again with full arXiv
access.
