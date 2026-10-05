# Certifying Ground Truth for a Robot That Cannot Measure It

**Draft.** Every number is measured unless marked `[PENDING]`. Sections whose
evidence does not yet exist are written as specifications of the experiment,
not as results. `PAPER.md` tracks figure/table status; `METHOD.md` holds the
full formulation; `RELATED_WORK.md` holds the literature check and its caveats.

---

## Abstract

Trajectory references for SLAM benchmarking are produced by instrumenting the
platform being measured: a motion-capture volume, a LiDAR registered into a
survey-grade map, a total station, RTK-GNSS. A robot carrying only a camera is
therefore evaluated against another SLAM system, which measures agreement
rather than accuracy, or it is not evaluated at all.

We produce a trajectory reference for such a platform by fusing **heterogeneous
absolute sources with independent failure modes** — surveyed fiducial boards, a
static surveyed observer, a peer robot carrying a reference sensor, and
registration into a prior map — in a batch pose graph over a visual backbone,
and we certify the result by **leave-one-source-out** cross-validation with a
per-source consistency test. Because no single source is continuous, the
construction's accuracy is governed by **absolute coverage in time**; we give a
closed-form model for that dependence and test it.

The procedure is certified on an instrumented twin. On a platform carrying both
a camera and a LiDAR, a construction built from the camera and two surveyed
boards, with the LiDAR held out entirely, is accurate to **199.3 mm** in the
surveyed map frame **with no alignment to ground truth**, against **331.3 mm**
for the same backbone trajectory after an $SE(3)$ fit *to* that ground truth — a
construction gain of **0.60** on strictly less information. A four-fold sweep of
the declared edge weights moves the result **1.1 mm**. Applied to a second
platform carrying a depth camera and nothing else, the construction disagrees
with that platform's own cuVSLAM-derived reference by 277.9 mm, in the direction
the surveyed boards indicate.

We report what the recording could not support — three of six factor classes,
and the platform's independent error bar — with the measurement that establishes
each, and we derive dataset-design requirements from those failures: anchor
coverage at both ends of the route rather than anchor count; fiducials sized for
the lowest-resolution camera; rotational excitation about more than one axis.

---

## 1. Introduction

A SLAM result is quoted against a reference trajectory, and the reference is
produced by adding a sensor the evaluated platform does not normally carry.
EuRoC [Burri16] uses Vicon and Leica; Newer College [Ramezani20] and Oxford
Spires [Tao24] register a platform-mounted LiDAR into a survey-grade prior map;
RTS-GT [Vaidis23] uses robotic total stations; MoCap2GT [MoCap2GT] and
Vicon2GT [Geneva20] refine mocap with the device's own IMU; PALoc [PALoc]
generates dense 6-DoF poses from a prior map with covariance propagated through
the factor graph. Every one of them requires the platform being measured to
carry, or to be tracked by, the reference.

Most deployed robots do not. They carry a consumer depth camera and an IMU. When
such a platform appears in a dataset, its "ground truth" is typically the output
of some other estimator — which makes any evaluation against it a measurement of
agreement between two correlated systems, not of accuracy.

**This paper asks whether a reference can be built for such a platform, and, more
importantly, whether the result can be certified rather than asserted.**

The setting is a single indoor recording containing both cases. One platform
(`mobile_1`) carries an Ouster LiDAR and a ZED 2i stereo camera on one rigid
body. A second (`mobile_2`) carries a RealSense D455 and nothing else. Three
ChArUco boards are fixed to the walls and surveyed. A static node (`infra_1`)
with a surveyed pose observes both platforms. The room is 8.5 × 14.3 m and the
recording is 156 s.

Two properties of this arrangement drive the paper. First, `mobile_1` is an
**instrument**: because it carries a LiDAR that can be *held out*, a procedure
using only its camera can be certified against something it never saw. Second,
the absolute evidence is **sparse and intermittent** — the boards are visible in
two windows totalling ~24 s of 152, so a backbone odometry must carry the
trajectory for the remaining 84%. That sparsity is the technical problem, not an
inconvenience: it is why the construction needs an odometry factor that
mocap-based estimators do not, and why the error is governed by coverage.

**Contributions.**

1. **A construction fusing heterogeneous absolute sources with independent
   failure modes** — fiducial, static observer, peer robot, prior map — and a
   certification by **leave-one-source-out** rather than leave-one-landmark-out.
   A bias shared by every landmark of one kind is invisible to the latter and
   visible to the former.
2. **The instrument/subject split**: certification of the procedure on a
   platform whose reference sensor is held out, then application to a platform
   that has none.
3. **A coverage law** relating error to absolute coverage in time and backbone
   drift rate, both measurable without ground truth; distinguishing an
   unanchored *end* (unbounded) from an unanchored *interior* (bounded, peak at
   the gap midpoint).
4. **Dataset-design requirements derived from measured failures**, including a
   7.26 m failure that a converged solver reported as success.

**What we do not claim.** The estimator. A factor graph over relative motion
with absolute-pose factors and a linear initialiser is EuRoC's own construction,
Vicon2GT's, Kalibr's MoCap branch [Furgale13], and MoCap2GT's. Surveyed fiducial
markers as pose-graph anchors is [MarkerPGC]. Validating by masking an absolute
stream while retaining the dense reference is RTK-SLAM's protocol [RTK-SLAM].
Scoring unaligned is also theirs. Leave-one-out over surveyed landmarks is
HortiMulti's [HortiMulti]. We use all of these and claim none.

---

## 2. Related work

**MoCap-fusion ground-truth estimators.** EuRoC [Burri16], Vicon2GT [Geneva20],
Kalibr's MoCap branch [Furgale13] and MoCap2GT [MoCap2GT] fuse a motion-capture
stream with the device's IMU to remove mocap jitter and recover spatiotemporal
calibration. XR-HPGT [XR-HPGT] adds a continuous-time MLE with variable time
synchronisation. All assume absolute pose at ~100 Hz, continuously, for the
whole sequence; none carries a term to bridge an interval where absolute
evidence is absent, because none has such an interval. They also estimate the
mocap-to-IMU extrinsic, which is unknowable a priori for a marker cluster;
our extrinsics are mechanical constants and the motion could not identify them
anyway (§5.4).

**Survey-prior datasets.** Newer College [Ramezani20], Oxford Spires [Tao24],
ConSLAM and Hilti [Hilti23] register a platform-mounted LiDAR into a prior map
built by survey-grade scanning, reaching 1–2 cm. PALoc [PALoc] is the closest
work on *uncertainty*: a prior-map-assisted factor graph with covariance derived
within the graph, handling degenerate and stationary segments. Its map factor
consumes a prior cloud **and a LiDAR frame**, so it cannot run on a platform
without one. More fundamentally, a covariance propagated through a graph is
*precision conditional on the model*: it cannot see a mis-declared extrinsic, a
wrong fiducial convention, or a biased survey. We therefore certify against
held-out evidence rather than reporting the graph's own covariance as accuracy.

**Fiducial-anchored trajectories.** TagSLAM [TagSLAM] optimises fiducial
observations in a factor graph and is explicitly proposed for ground truth.
[MarkerPGC] georeferences a LiDAR-odometry trajectory and an RTAB-Map
reconstruction using pre-surveyed markers, reporting held-out georeferencing
improvements of 8–30% over one-time alignment. Its evaluation uses a single
marker relocated among six surveyed positions, so at any instant the trajectory
is anchored at one place — the configuration we measure as a 7.26 m failure
mode in §6.4. HortiMulti's Poly-TagSLAM [HortiMulti] fuses LiDAR-inertial
odometry with AprilTags and validates by leave-one-out over 35 total-station
landmarks (5.9 cm mean). It is the closest published certification; it holds out
*instances of one kind*, and its backbone is LiDAR-inertial.

**Sparse absolute measurement.** RTS-GT [Vaidis23] and [Vaidis23b] treat
genuinely sparse total-station measurements and analyse their uncertainty.
RTK-SLAM [RTK-SLAM] evaluates GNSS-degraded SLAM by masking the GNSS stream
while retaining the dense RTK reference — the protocol we adopt in §7 — and
reports that $SE(3)$-aligned ATE can understate absolute global error by up to
76%, which is the independent justification for our unaligned scoring.

**Pseudo-ground-truth.** PIN-SLAM [Pan24] and similar assert that a stronger
SLAM system's output can substitute for ground truth without certifying the
substitute. Princeton365 [Princeton365] reports pseudo-GT at ~20 cm and states
what it is therefore adequate to measure, which is the posture we adopt.

**The gap.** No work found produces a reference for a platform lacking the
reference modality, certifies it by holding out *kinds* of evidence rather than
instances, or states an error model in quantities measurable on an
uninstrumented platform.

---

## 3. Setting and instruments

The recording (156 s, indoor, 8.5 × 14.3 m, 16.6 m diagonal) contains two
platforms and one static node.

`mobile_1` publishes a reference trajectory of 1516 poses at 9.70 Hz — one per
LiDAR sweep — produced by registering each scan into a frozen prior map, with
each scan initialised from its seed pose rather than from the previous scan's
result, so error cannot accumulate along the trajectory. Path length 25.82 m.
Declared uncertainty 15 mm, taken from the worst of the three board surveys.

`mobile_2` publishes 1811 poses at ~11.6 Hz, 24.7 m of path. **Its provenance is
cuVSLAM odometry plus board anchoring**, so it is an odometry product and any
visual method scored against it is partly scored against itself.

Three ChArUco boards are surveyed. `anchor` and `anchor_b` are 18 × 14 cm with
15 mm DICT_4X4_50 markers; `rs_anchor` is 22.5 × 17.5 cm with 18 mm DICT_5X5_50.

**Instrument validation, performed before any method was scored.**

*Board detector.* Reproduces the mapping pipeline's own detections to **8.1 mm /
0.66°** on the ZED and **8.3 mm / 0.58°** on the RealSense, at or below each
board's own scatter (2.0 and 6.9 mm). Lens distortion is handled at ChArUco
corner interpolation; handling it at PnP instead gave 34.2 mm, and undistorting
the image gave 21.2 mm with worse reprojection.

*Board frame convention.* Settled by measurement. All eight planar conventions
(4 spins × 2 chiralities) were re-solved on the same 90 detections and scored
against the pipeline's file: the winner sits at **6.0 mm**, the runner-up at
288 mm. Two conventions read from prose gave 412 mm and 1449 mm.

*References against the boards.* `mobile_1` **0.88° / 10.0 mm**; `mobile_2`
1.38° / 54.0 mm. The first is independent — `mobile_1`'s reference consumes no
board — and it is the measurement that licenses the rest of the paper: two
instruments that know nothing of each other agree to 1 cm. The second is not
independent, because `mobile_2`'s reference consumed the boards.

*Route revisits.* 280 interior closest approaches on `mobile_1` and 140 on
`mobile_2`, each within 0.30 m and at least 20 s apart, measured on the
references. Loop closure is therefore evaluable here. The separations themselves
(medians 133 and 256 mm) are **not** a drift bound: they are censored by the
0.30 m search radius.

---

## 4. Benchmark

Three backbones with independent failure modes — geometric (KISS-ICP), feature
(RTAB-Map), learned-prior (MASt3R-SLAM) — on both platforms, plus the
inertial/IMU-free pair of RTAB-Map. Every tier-1 number is quoted with its
board-residual companion and its **span** (estimate duration ÷ reference
duration), because two ATEs over different fractions of a run are not the same
measurement and the conventional coverage statistic cannot show it.

### 4.1 `mobile_1` (reference: LiDAR, uncertainty 15 mm)

| method | ATE RMSE | rot RMSE | board residual | span |
|---|---:|---:|---|---:|
| `rtabmap_rgbd_imu`, front-end | 372–481 mm | 5.8–7.1° | 467–577 (`anchor`), 350–548 (`rs_anchor`) | 99% |
| `rtabmap_rgbd_imu`, graph | 366–451 mm | 3.4–4.7° | 421–499 (`anchor`) | 78% |
| `rtabmap_rgbd` (IMU-free), graph | **323 mm** | 4.08° | **375** (`anchor`), **336** (`rs_anchor`) | 74% |
| `mast3r_slam` | 1908 mm | 1.91° | 3536 mm | 64% |
| `kiss_icp` | 2.2–6.7 m | 28–165° | 3.3–9.5 m | 99–100% |

The two tiers agree on magnitude and on ranking. That corroborates the reference
and is what licenses `mobile_1` as the instrument for §6.

MASt3R recovers shape but not size: its estimate is **37.1% too large**, and a
$Sim(3)$ fit drops the ATE from 1908 mm to 85 mm. Quoting 85 mm alone would
report a scale model of the room as an accuracy. KISS-ICP fails outright — 126 m
of path for a 25.8 m route — and the board residuals confirm the failure
independently of the reference.

RTAB-Map's loop-closed graph covers only 74–78% because it adds a node on
*motion*, and the cart parks at a board for the final 27 s. That is a property
of the method, stated, not a defect of the evaluation.

### 4.2 `mobile_2` (reference: cuVSLAM-derived, **correlated**)

| method | ATE RMSE | board @ `anchor` | board @ `rs_anchor` | span |
|---|---:|---:|---:|---:|
| `rtabmap_rgbd`, graph | 135 / 149 mm | 55 / 87 mm | 94 / 123 mm | 97 / 95% |
| `mast3r_slam` | **120 mm** | 61 mm | 201 mm | 89% |
| `rtabmap_rgbd_imu`, front-end | 151 mm | 82.5 mm | **77.0 mm** | 97% |
| `kiss_icp` | 5008 mm | 2639 mm | 7379 mm | 100% |

**`mobile_2`'s ATE cannot rank methods.** By ATE, MASt3R wins (120 < 135). By the
surveyed boards, RTAB-Map wins at `rs_anchor` by a factor of two and is
level-or-better at `anchor`. Neither board reproduces the ATE ranking. The
reference is an image-based VIO product, as is MASt3R, and a method correlated
with its reference is flattered by it. Contrast §4.1, where the tiers agree.

**The comparison also spans different runs**: MASt3R covers 89%, RTAB-Map 97%.
The conventional coverage statistic reads 0.97–1.00 for both, because it asks
what fraction of the *estimate* found a reference — near 1 for an estimate that
simply stops early.

### 4.3 An inertial ablation that does not replicate across platforms

`rtabmap_rgbd` and `rtabmap_rgbd_imu` differ in one switch. Compared at the
boards, over identical dwell windows with identical counts:

| | IMU off | IMU on |
|---|---:|---:|
| `mobile_1` front-end @ `anchor` | **422.4 mm** / 6.37° | 466.8–577.0 / 4.41–7.08° |
| `mobile_1` front-end @ `rs_anchor` | **335.6** / **1.48°** | 350.3–547.9 / 3.77–25.40° |
| `mobile_1` graph @ `anchor` | **375.4** / **2.17°** | 421.4–499.3 / 3.45–4.69° |
| `mobile_2` front-end @ `anchor` | **39 / 51 mm** | 82.5 / **2.95°** |
| `mobile_2` front-end @ `rs_anchor` | 95 / 118 | **77.0** / 7.53° |
| `mobile_2` graph @ `rs_anchor` | 94 / 123 / 7.30–7.40° | **86.9** / **3.60°** |

On `mobile_1` the IMU-free run is better in position than all five inertial runs
in all three comparisons. On `mobile_2` the result is mixed: worse at one board,
better at the other, rotation equal or better everywhere.

**So "the IMU hurts" is a platform result, not a property of the method.**
RTAB-Map's IMU enters only as a gravity constraint, and the platforms differ in
where gravity comes from: the ZED SDK's fused orientation on `mobile_1`, a
Madgwick filter over raw gyro and accelerometer on `mobile_2`. A bias in the
former's fusion or declared extrinsic would produce exactly this asymmetry. This
remains a hypothesis; the test compares the IMU-derived gravity direction
against the surveyed map's vertical and needs no new recording `[PENDING]`.

One mechanism the numbers do support: at `rs_anchor`, the first 17.7 s of
`mobile_2`'s run, the IMU halves the graph's rotation residual, 7.30–7.40° →
3.60°. Bounding attitude during initialisation is what a gravity constraint is
for.

---

## 5. Method

### 5.1 State and residuals

$\mathcal{X}=\{\mathbf{x}_k=(\mathtt{R}_k,\mathbf{t}_k)\in SO(3)\times\mathbb{R}^3\}_{k=1}^N$,
the body pose in the surveyed map frame at the backbone's keyframe rate
($N\approx1500$), plus optional per-source bias states $\boldsymbol\beta_s$.
Increments live in the tangent space and are applied by
$\mathbf{t}\mapsto\mathbf{t}+\boldsymbol\rho$,
$\mathtt{R}\mapsto\mathtt{R}\exp(\boldsymbol\phi^\wedge)$.

**Backbone odometry** (binary, consecutive) — the term mocap-based estimators do
not need:

$$\mathbf{r}^{O}_k=\begin{bmatrix}\mathtt{R}_k^\top(\mathbf{t}_{k+1}-\mathbf{t}_k)-\check{\mathbf{t}}_k\\
\log(\check{\mathtt{R}}_k^\top\mathtt{R}_k^\top\mathtt{R}_{k+1})^\vee\end{bmatrix}$$

taken from the *front-end*, not the loop-closed graph: relative poses derived
from an already-optimised trajectory are correlated by the optimiser, which the
independence assumption behind $\Sigma_O$ would violate.

**Surveyed board** (unary), residual on the measurement
$\mathbf{z}^B_a=\mathtt{T}^{\text{cam}}_{\text{board}}$ from ChArUco+PnP:

$$\mathbf{r}^{B}_a=\log\big(\hat{\mathbf{z}}_a^{-1}\mathbf{z}^B_a\big)^\vee,\qquad
\hat{\mathbf{z}}_a=(\mathbf{x}_{k(a)}\mathtt{T}^{\text{body}}_{\text{cam}})^{-1}\mathtt{T}^{\text{map}}_{\text{board}}$$

**Static observer** (unary, 3-DoF — a radar return carries no orientation), with
$\mathbf{p}=(\mathtt{T}^{\text{map}}_{\text{infra}})^{-1}(\mathbf{t}_k+\mathtt{R}_k\boldsymbol\beta_I)$:

$$\mathbf{r}^{I}_k=\big[\rho-\|\mathbf{p}\|,\ \angle(\alpha-\operatorname{atan2}(p_y,p_x)),\ \angle(\epsilon-\arcsin(p_z/\|\mathbf{p}\|))\big]^\top$$

**Peer robot** (unary; the peer's pose is reference-grade and treated as known,
which keeps the factor unary and the system banded):
$\mathbf{r}^{P}_k=\log(\hat{\mathbf{z}}_k^{-1}\mathbf{z}^P_k)^\vee$ with
$\hat{\mathbf{z}}_k=(\mathtt{T}^{\text{map}}_{\text{peer}}(t_k))^{-1}\mathbf{x}_k$.

**Prior-map registration** (unary, per point, so degenerate directions remain
visible rather than being collapsed into a pose):
$\mathbf{r}^{M}_k=\{\mathbf{n}_i^\top(\mathbf{x}_k\mathtt{T}^{\text{body}}_{\text{sens}}\mathbf{p}_i-\mathbf{q}_i)\}$.

### 5.2 Objective

$$(\hat{\mathcal{X}},\hat{\boldsymbol\beta})=\arg\min\sum_{k}\|\mathbf{r}^{O}_k\|^2_{\Sigma_{O,k}}
+\sum_{s\in\mathcal{S}_{\text{fit}}}\sum_{m}\rho_s\big(\|\mathbf{r}^{s}_m\|^2_{\Sigma_{s,m}}\big)
+\sum_{s\in\mathcal{S}_\beta}\|\boldsymbol\beta_s\|^2_{\Sigma_{\beta_s}}$$

with $\rho_s$ a Huber kernel at the 95% point of $\chi^2_d$.

### 5.3 Per-source uncertainty

$$\Sigma_{s,m}=\frac{\Sigma^{\text{meas}}_{s,m}}{n_{\text{eff}}(m)}+\mathtt{J}_{s,m}\Sigma^{\text{anchor}}_s\mathtt{J}_{s,m}^\top+\kappa_s\Sigma_{s,m}$$

Three terms that behave differently. Measurement noise **averages down** with
the effective (not raw) number of independent observations — a stationary dwell
producing 86 near-identical detections is not 86 measurements, and treating it
so over-weights the anchor by $\sqrt{86}$. The anchor's own uncertainty
**does not average**: no number of radar returns makes a static observer better
than its surveyed pose. $\kappa_s$ is a calibration factor set in §5.6.

**Frame ancestry bounds what independence means.** The boards were surveyed
directly; the prior map was built from `mobile_1`'s LiDAR and anchored *to the
boards*; `infra_1`'s pose was recovered from the map; the peer's reference is
registered into the map. A global rigid error of the map frame is common to all
and irrelevant. A *local* distortion is shared by three of the four sources and
not by the boards, so the strongest independent pair is {boards} against
{map-derived sources}.

### 5.4 Declared, not estimated

Camera-to-IMU extrinsics are declared. Cumulative rotation per optical axis is
$[140,1094,101]°$ — the platform yaws and does essentially nothing else — and
the correlation matrix is rank one (singular values 44.238 / 0.014 / 0.005). No
estimator on this recording can self-calibrate that extrinsic; MoCap2GT reports
the same failure under insufficient rotational excitation.

### 5.5 Gauge, observability, solution

At least one absolute source spanning rank 6 must be present, else
$\operatorname{rank}\mathtt{H}\le6N-6$ and the solver **refuses** rather than
returning an arbitrary member of a solution family. The solve reports leading
and trailing unanchored path and the longest interior gap — the inputs to §5.7.

Levenberg–Marquardt on the manifold; Jacobians by central differences on
gathered endpoints. Because odometry joins only consecutive poses and every
other factor is unary, the normal matrix is **block-tridiagonal** and solves
exactly in $O(N)$ by a block Thomas sweep; bias states add an arrow structure
eliminated by Schur complement. Initialisation is an $SE(3)$ — not $Sim(3)$ —
fit of the backbone onto its own absolute observations.

### 5.6 Certification: leave-one-source-out with a consistency test

For each source $s$, solve without it and evaluate its measurements at that
solution: $\mathbf{e}_s=\{\mathbf{r}^{s}_m(\hat{\mathcal{X}}^{(-s)})\}$. Report
the error distribution — median, $p_{95}$, worst stretch, with a **block
bootstrap** over time segments, because trajectory error is autocorrelated and a
per-pose interval would be too narrow by roughly $\sqrt{\text{poses per
episode}}$ — and the normalised error squared,

$$\text{NEES}_s=\frac{1}{d|\mathcal{M}_s|}\sum_m\mathbf{r}_m^{s\top}\big(\Sigma_{s,m}+\mathtt{J}_m\mathtt{P}^{(-s)}\mathtt{J}_m^\top\big)^{-1}\mathbf{r}^{s}_m$$

$\text{NEES}_s\approx1$ means the trajectory *and its claimed covariance* are
both right; $\gg1$ means overconfident. Iterating
$\kappa_s\leftarrow\kappa_s\cdot\text{NEES}_s$ is variance-component estimation
per source: **a source that overclaimed its accuracy is downweighted by exactly
the factor it overclaimed by, as measured against the others.**

**Every absolute source is a factor or a check in a given solve, never both.** A
residual at a consumed source is $\sigma/\sqrt{n}$ — the standard error of the
observations it was fitted to, not evidence. With several sources one rotates;
with one source the choice is forced, which is why source *diversity* is a
requirement rather than a luxury.

If two sources are each consistent alone but jointly inconsistent, they disagree
**systematically** and the graph cannot arbitrate. We then either estimate a
bias of known form via $\boldsymbol\beta_s$, or report the disagreement as a
lower bound on the error of at least one source, using frame ancestry to say
which is likelier. We never resolve it by choosing.

### 5.7 The coverage law

$$e(t)\approx\min_a\big[\sigma_a+D(|t-t_a|)\big]$$

Two regimes. Beyond the last absolute observation nothing constrains the
trajectory and error grows without bound — a **lever**. Between two observations
both ends are pinned and the accumulated drift is distributed; for random-walk
drift this is a Brownian bridge with variance $\sigma_D^2u(T-u)/T$, so the peak
sits at the midpoint and scales as $\sqrt{T}/2$ rather than $T$ — a **bridge**.

Three consequences. Anchor *count* is the wrong statistic and anchor *coverage
in time* is the right one. The quantity to report is the longest unanchored gap,
plus whether either end is unanchored. And a better backbone helps even when the
anchors are unchanged, because it lowers $\sigma_D$.

The law's inputs — coverage and drift rate — are measurable on a platform with
no ground truth, which is what makes it transferable. Its central assumption is
that drift behaves as a random walk; systematic drift would give $\propto T$.
§7.1 tests this.

---

## 6. Certification on the instrument

All results use only `mobile_1`'s camera and two surveyed boards
(`anchor_b` contributes nothing; §8). The LiDAR is held out throughout and used
only to score.

### 6.1 The wiring gate

A construction with odometry factors alone (V1) must reproduce the backbone it
was given. It does: board residuals appear unchanged to the digit (350.3 / 3.77°,
409.1 / 9.06°, 466.8 / 6.28°), and ATE matches the backbone within 1–6 mm across
five runs. The graph adds nothing it was not given.

### 6.2 The headline: one backbone, one run, 100% span both ways

V2c is scored **unaligned** in the surveyed map frame. An $SE(3)$ fit to the
reference before scoring would remove the very placement the anchors provide,
and would use the reference to do it. V1 is in the backbone's own frame and is
aligned like any backbone.

| | ATE | alignment | rot | RPE 1 m |
|---|---:|---|---:|---:|
| V1 | 331.3 mm | $SE(3)$ fit **to the LiDAR** | 4.29° | **91.3 mm** |
| **V2c** | **199.3 mm** | **none — map frame** | **3.45°** | 149.8 mm |

**199 mm using no ground truth, against 331 mm for the same trajectory after
being fitted to ground truth.** Construction gain 0.60. Across all five
backbones the pattern holds: V2c 199–291 mm unaligned against V1 331–411 mm
aligned.

The trade is real and belongs beside the headline: RPE at 1 m worsens from
~91 to ~150 mm. The anchors bend the trajectory to satisfy global constraints
and local smoothness pays. §9 notes that this cost is probably overstated.

### 6.3 Backbone quality propagates

| backbone | its own board residual | its V2c | median move |
|---|---:|---:|---:|
| `rtabmap_rgbd` (IMU-free) | 422 / 336 mm | **199.3 mm** | **186 mm** |
| `rtabmap_rgbd_imu` ×4 | 467–577 / 350–548 mm | 228–291 mm | 336–512 mm |

The anchors do not dominate. The best front-end yields the best construction and
needs the least correction, visible in both the result and the distance the
solve carried the trajectory — as §5.7 predicts, since $\sigma_D$ is the bridge's
only free parameter.

### 6.4 Why the certification holds

**The declared weights do not carry the result.** Over a 4× range of per-edge
sigma the ATE moves **1.1 mm** (290.4 / 291.2 / 290.1 mm). The sigmas derive
from each run's RPE, which was scored against the LiDAR; without this sweep the
LiDAR ATE could be dismissed as circular.

**Leave-one-board-out cannot certify a two-board construction**, and *why* is
the finding:

| build | ATE | median move | independent board residual |
|---|---:|---:|---:|
| `anchor` only | 1163 mm | **7261 mm** | `rs_anchor` 1719 mm |
| `rs_anchor` only | 771 mm | 178 mm | `anchor` 1201 mm |

Holding either board out leaves a lever. The `anchor`-only build moved **7.26 m
while reporting convergence**, at whitened residuals of 0.01 / 0.08. **A
near-zero whitened residual is not a good fit**; combined with a large move it
is the signature of an under-determined system whose solver returned one member
of a family. The builder now warns on that conjunction, and this is the single
strongest argument for source diversity: with $n=2$ landmarks of one kind there
is no leave-one-out, and the held-out LiDAR is the certification of record.

Both single-board builds sit at 94 mm RPE — exactly V1's value — because one
anchor *transports* a trajectory rigidly without bending it. The full build
bends it, and the bending is where the 12 cm improvement comes from.

### 6.5 Coverage: bridges, not levers

| | leading | trailing | widest interior gap |
|---|---:|---:|---:|
| `mobile_1` | ~0 s | ~0 s | 118.2 s |
| `mobile_2` | 0.0 s | 8.5 s | 125.1 s |

Both are long bridges held at both ends. `mobile_2`'s trailing stretch moved
**29 mm** against 142 mm in the anchored part — the tail moved *less* than the
middle, which is what distinguishes an interior gap from a lever.

### 6.6 Transfer to the subject platform

V2c scores **277.9 mm** against `mobile_2`'s cuVSLAM-derived reference where V1
scores 130 mm. **This is disagreement, not error.** The reference and the
backbone share sensors and front-end, so 130 mm is agreement between correlated
estimates. The construction moved 128 mm median *toward the survey and away from
cuVSLAM*; on `mobile_1`, where a held-out LiDAR exists, that same move was
verified correct (377 → 254 mm).

The caveat is stated rather than resolved: `mobile_1`'s backbone was badly wrong
at the boards (421–577 mm) and had much to fix; `mobile_2`'s was already good
(39–95 mm) and still moved 128 mm, most of it inside the 125 s bridge where
nothing measures it.

---

## 7. Generalisation `[PENDING]`

Two experiments remain, and they are what turn a single self-recorded sequence
into evidence.

### 7.1 Does the coverage law hold?

Plot construction error against time-to-nearest-absolute-observation on
`mobile_1`, where held-out LiDAR gives truth everywhere, against §5.7's
prediction: flat near the anchors at $\sigma_a$, rising to a peak at the gap
midpoint, symmetric about it. A curve linear in $T$ rather than $\sqrt{T}$ would
falsify the random-walk assumption and weaken the transfer argument. One figure;
no new runs.

### 7.2 Masked-reference validation on public datasets

Following RTK-SLAM's protocol [RTK-SLAM], take a dataset with continuous ground
truth, retain a windowed fraction as pseudo-anchors, hide the remainder, and
score against what was hidden. This places the method in its native regime on
data where truth is known **in the middle of the bridge** — the one place our
own recording cannot check.

Sweeping the retained fraction from 100% to 2% maps the coverage law as a
measured relation; moving the windows from both ends to one end reproduces the
lever on demand, on identical data. Candidates: EuRoC (standard, smallest),
TUM-VI (mocap covers part of the volume, so the gap structure is natural rather
than imposed), Newer College (tests prior-map registration directly),
Hilti-Trimble 2026 (floor-plan priors — sparse absolute information).

Masked mocap is not a surveyed board: it has no detection failures, no incidence
limits, no blur. The masking experiments therefore test the **estimator and the
coverage law**; the front end is validated separately (§3).

---

## 8. What could not be measured

Reported with the measurement that establishes each, rather than omitted.

**`anchor_b` is below the RealSense's resolving power.** A detector run returned
**0 of 179 frames**; 178 saw no ChArUco corners at all, and the median over the
window was 2 of 48. Not a tuning miss — the identical board design at `anchor`
reads 44 of 48 from the same camera in the same session. At its only observed
range, 0.84 m, its 15 mm markers span ~1.9 px per bit in an 848 × 480 frame.
`mobile_1` never gets closer than 2.48 m. Three surveyed boards; two usable per
robot.

**Three of six factor classes are unavailable.** IMU preintegration is blocked:
the factor covariance needs an Allan-variance noise model and the recording is
156 s. Infrastructure factors were blocked by the static node's pose being
stated to six decimals with no uncertainty — recoverable, since the chain was
validated end-to-end (`mobile_2`: azimuth +0.16° sd 3.51, range +0.16 m sd 0.53,
nearest-return check ±0.01 m). Inter-agent factors are unavailable: the
platforms have 100% place overlap but were never verified to co-observe
simultaneously.

**`mobile_2`'s error bar cannot be priced by backbone consensus.** Only one
front-end is dense enough: MASt3R contributes **6 board observations** across
151 keyframes against 321 for the RGB-D backbone, and KISS-ICP is 5 m off. Six
priors cannot anchor a trajectory.

**Prior-map registration is not run** because the map cloud was not exported.
This is a missing input, not an impossibility, and it is the strongest
independent check available for the subject platform. Note that KISS-ICP's
failure does not predict it: scan-to-*scan* accumulates error, scan-to-*map*
initialises every frame independently and cannot.

---

## 9. Limitations

**Anchor observations are treated as independent when they are not.** The
implementation emits one prior per keyframe in a dwell window, each carrying the
full declared sigma, but during a stationary dwell those priors share their
error. Treating $n=86$ as independent gives an effective anchor of
$7/\sqrt{86}\approx0.75$ mm — a number we observe directly in the self-agreement
residual. The anchors are therefore over-weighted inside the solve, and **the
reported local cost (RPE 91 → 150 mm) is probably overstated**: the trade is
likely better than §6.2 reports, not worse. §5.3's $n_{\text{eff}}$ is the fix
`[PENDING]`.

**The sigma sweep does not cover the independence assumption.** It rescales
magnitude uniformly and never perturbs the correlation structure, which is what
would change the *shape* of the solution rather than its scale.

**Edge sigmas are reference-informed.** They derive from each run's RPE, scored
against the reference. On `mobile_1` that reference is a held-out LiDAR, so the
dependency is real but benign, and the sweep bounds its influence. On `mobile_2`
the reference is cuVSLAM-derived and correlated with the backbone, so the
weights are partly set by the backbone's agreement with itself.

**No time-offset estimation.** MoCap2GT carries a time offset as a B-spline and
XR-HPGT makes variable synchronisation a headline. We assume a common clock,
which is defensible here — board detections and images share a header stamp —
and the anchors are sparse enough that interpolation error is a small share of
199 mm. A scalar per-stream offset would remove the objection.

**Loop closures are not used.** The route demonstrably revisits, and the
backbone's loop-closure links exist but are not exported. In the coverage law,
anchors reduce $T$ while loop closures reduce $\sigma_D$ — they attack the same
quantity from opposite sides, and the error lives in the bridge. Substituting
the loop-closed trajectory would be worse than omitting them, since relative
poses derived from an optimised trajectory are correlated by the optimiser while
the constraint itself remains absent from the cost. Adding them properly costs
the block-tridiagonal structure. Deferred, not rejected.

**Single recording for the native experiments**, and a platform-transfer
assumption between two different cameras that §7.2 is designed to test.

---

## 10. What a dataset must contain

Each rule from a measured failure.

1. **Absolute coverage at both ends of the route.** Not anchor count. Holding
   out one of two boards produced a 7.26 m displacement at a whitened residual of
   0.01 — convergence that means nothing.
2. **Several *kinds* of absolute source, failing for unrelated reasons.** A
   fiducial, a static observer, a peer robot, a prior map. Leave-one-landmark-out
   over homogeneous landmarks cannot detect a bias shared by all of them.
3. **Markers sized for the lowest-resolution camera**, with margin: ≥4 px per
   bit at the maximum intended detection range. 1.9 px/bit succeeded on one
   camera and returned 0 of 179 on another.
4. **Rotational excitation about more than one axis**, or nothing can
   self-calibrate.
5. **Every uncertainty stated**, including the surveyed pose of a static
   observer. An anchored fix cannot be better than its anchor, and that term
   does not average away.
6. **Verified simultaneity** if inter-agent constraints are intended. Place
   overlap is a different statistic.

---

## 11. Conclusion

A trajectory reference can be built for a platform that carries no reference
sensor, and the result can be certified rather than asserted. On an instrumented
twin with its LiDAR held out, a construction from a camera and two surveyed
boards reaches **199.3 mm in the surveyed map frame with no alignment to ground
truth**, against 331.3 mm for the same backbone fitted *to* ground truth — better
on strictly less information, with a 4× weight sweep moving the answer 1.1 mm.

The accuracy is governed by absolute coverage in time, not by anchor count, and
the distinction between an unanchored end and an unanchored interior is the
difference between an unbounded error and a bounded one. Certification requires
holding out *kinds* of evidence rather than instances, because a bias shared by
every landmark of one kind survives leave-one-landmark-out intact.

What remains is to show that the coverage law holds — one figure on data already
in hand, which can falsify it — and to reproduce the regime on public datasets
by masking their ground truth. Both are specified in §7.
