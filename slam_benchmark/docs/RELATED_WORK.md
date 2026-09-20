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
| — GT-ablation validation protocol (§6.3) | **survives**, with outside precedent | no SLAM-side hit. The same idea is established in medical image segmentation (SparseGT, Penn) — cite it as precedent rather than claiming the idea outright |

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
| **SparseGT** (Penn, medical imaging) | outside precedent for the GT-ablation protocol. Cite it; do not claim the idea as new in general, only as new here |

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
validating a GT-generation method by ablating existing ground truth; or
multi-agent anchoring for GT production.

**Read that carefully.** Absence in a search is weak evidence — arXiv was
blocked, the searches were English-language and keyword-driven, and the nearest
paper is one month old, which means the area is moving. Treat the four claims
as *provisionally* clear and re-run this check before submission. The one that
would hurt most if it exists is a 2026 paper doing marker-anchored GT for an
uninstrumented platform; search that specific phrasing again with full arXiv
access.
