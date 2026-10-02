# Certifiable Ground Truth for a Platform Without a Reference Sensor

**Method document** — motivation, gaps, contributions, system overview, the
optimisation problem, per-source uncertainty, how disagreement between sources
is settled, the reformulation this implies, and references.

This supersedes the scattered statements in `PAPER.md`, `THEORY_AND_VALIDATION.md`
and `FORMULATION.md` where they differ. Loop-closure factors remain **excluded
from the problem** by decision (§8.4).

---

## 1. Motivation

Every method for producing a reference trajectory requires the platform being
measured to carry, or be tracked by, the reference: a motion-capture volume
[MoCap2GT, XR-HPGT, Vicon2GT, EuRoC], a LiDAR registered into a survey-grade
map [Newer College, Oxford Spires, PALoc], a total station [RTS-GT, Mittelstedt],
or RTK-GNSS [RTK-SLAM]. A platform carrying only a camera is evaluated against
another SLAM system — which is agreement, not evaluation — or not at all.

`coop2_20260828` has both cases on one recording. `mobile_1` carries a LiDAR and
a ZED 2i; its reference is LiDAR registration into a board-anchored map, and it
agrees with the surveyed boards independently to 10 mm. `mobile_2` carries a
RealSense D455 and nothing else; its published "reference" is cuVSLAM plus
board anchoring, so scoring any visual method against it is self-agreement.

The room also contains three surveyed ChArUco boards, a static surveyed
radar/camera node (`infra_1`) that observes both carts, and — for `mobile_2` —
a second robot whose pose is reference-grade. None of these is a reference on
its own. Together they may be.

## 2. Gaps

1. **No reference sensor aboard.** All published GT estimators need one on the
   platform (or a mocap volume around it). None runs on `mobile_2` at all.
2. **Single-source certification.** Where GT is validated, it is by holding out
   instances of *one* kind of evidence — 35 AprilTags in HortiMulti, a marker in
   the 2026 marker-correction paper. Leave-one-landmark-out cannot see a bias
   shared by every landmark of that kind (survey offset, detector bias).
3. **Precision reported as accuracy.** PALoc propagates covariance through the
   factor graph; that is precision *conditional on the model*, and it cannot see
   a mis-declared extrinsic or a biased anchor.
4. **No error model that transfers.** A number measured on one platform is not
   a statement about another with a different camera. Nothing found fits an
   error law on an instrumented platform and evaluates it on an uninstrumented
   one at the latter's own measured inputs.
5. **Sparse absolute evidence is treated as an inconvenience**, not modelled.
   MoCap2GT has absolute pose at 100 Hz; here it exists for ~16% of the run.
   How error depends on that coverage is not written down anywhere found.

## 3. Contributions and novelty — after the literature check

**Not claimed:** the estimator. Factor graph + absolute-pose factors + linear
initialiser is EuRoC (2016), Vicon2GT (2020), Kalibr, MoCap2GT (2026).
Surveyed markers as pose-graph anchors is the 2026 marker-correction paper.
The masking protocol is RTK-SLAM. Unaligned scoring is RTK-SLAM. Leave-one-out
over landmarks is HortiMulti.

**Claimed:**

**C1. Certifiable GT for a platform carrying no reference sensor, by fusing
heterogeneous absolute sources with independent failure modes, certified by
leave-one-*source*-out.** Board (fails on range/incidence), static observer
(occlusion/range), peer robot (co-visibility), prior-map registration
(degenerate geometry). Kinds, not instances: a bias shared by every landmark
of one kind is invisible to leave-one-landmark-out and visible to
leave-one-source-out. This is the project's own backbone-consensus rule —
*three similar systems agreeing through a shared failure mode certify
nothing* — applied to the absolute sources.

**C2. The instrument/subject split.** The procedure is certified on a platform
that carries a held-out LiDAR (`mobile_1`: 199.3 mm unaligned in the map
frame vs 331.3 mm for the same trajectory SE(3)-fitted *to* the LiDAR;
construction gain 0.60), then applied to one that carries nothing.

**C3. A coverage law that transfers.** Error as a function of absolute coverage
in time and backbone drift rate — both measurable with no ground truth. Lever
(unbounded) vs bridge (bounded, peak at the gap midpoint). Tested on the
instrument and by the masking protocol on public datasets; evaluated at the
subject's own inputs.

**C4. Dataset-design rules from measured failures**, landing in the existing
anchor-placement literature: coverage at both ends beats landmark count (one
board held out → 7.26 m swing at whitened residual 0.01); markers sized for
the worst camera (15 mm markers unreadable at 1.9 px/bit); rotation about more
than one axis if anything is to self-calibrate.

**C5. Per-source consistency calibration** (§7): declared uncertainties are
tested by held-out NEES and inflated until consistent, so disagreement is
settled by evidence rather than by declaration.

## 4. System overview

```
INPUTS                          FRONT END
RGB-D stream ─────────────────► backbone SLAM (RTAB-Map front-end) ──► relative motion  ΔT̃_k
images + board geometry ──────► ChArUco detect → PnP → axes → compose ─► board sightings  z^B
static node radar ────────────► dynamic-return association ────────────► range/bearing    z^I
peer robot (ref-grade pose) ──► cart detection in peer's LiDAR ─────────► relative pose    z^P
depth cloud + surveyed map ───► point-to-plane registration ────────────► per-point        r^M

                                INITIALISATION
                                SE(3) fit of the backbone onto its own absolute observations
                                (no scale; no gauge prior when any absolute source is present)

                                GRAPH ASSEMBLY  (adaptive: only declared sources enter)
                                odometry factors  σ_O from RPE/√E, reset edges ×100
                                per-source factors with Σ_s = Σ_meas/n_eff + J Σ_anchor Jᵀ
                                robust kernel ρ_s per source
                                observability test → REFUSE if gauge free
                                coverage report: leading / trailing / longest gap

                                OPTIMISATION
                                LM on SO(3)×R³; H block-tridiagonal (+ arrow for bias states)

OUTPUT 1  ─────────────────────  trajectory in the surveyed map frame + graph.json

                                CERTIFICATION
                                leave-one-source-out → error distribution + NEES per source
                                consistency calibration (inflate Σ_s until NEES ≈ 1, re-solve)
                                σ sweeps; under-determination guard; lever/bridge report

OUTPUT 2  ─────────────────────  certified error model  ──►  TRANSFER to the subject platform
```

The bottom two blocks have no counterpart in MoCap2GT's system figure. They are
the contribution.

## 5. The problem

### 5.1 State

$$\mathcal{X}=\{\mathbf{x}_1,\dots,\mathbf{x}_N\},\quad
\mathbf{x}_k=(\mathtt{R}_k,\mathbf{t}_k)\in SO(3)\times\mathbb{R}^3,\qquad
\boldsymbol\beta=\{\boldsymbol\beta_s\}_{s\in\mathcal{S}_\beta}$$

$\mathbf{x}_k$ is the body pose in the surveyed `map` frame at keyframe $k$.
$\boldsymbol\beta_s$ are optional **per-source bias states** (§8.3): a constant
offset for a source with a known systematic, e.g. the radar cluster-centroid to
body-origin lever, or a scalar clock offset. Sensor extrinsics are declared,
not estimated (rank-one rotation: singular values 44.238 / 0.014 / 0.005).

Increments $\boldsymbol\delta_k=(\boldsymbol\rho_k,\boldsymbol\phi_k)$ are applied by
$\mathbf{t}_k\!\leftarrow\!\mathbf{t}_k+\boldsymbol\rho_k$,
$\mathtt{R}_k\!\leftarrow\!\mathtt{R}_k\exp(\boldsymbol\phi_k^\wedge)$.

### 5.2 Residuals

**Odometry** (binary, consecutive), from the backbone's front-end:

$$\mathbf{r}^{O}_k=\begin{bmatrix}\mathtt{R}_k^\top(\mathbf{t}_{k+1}-\mathbf{t}_k)-\check{\mathbf{t}}_k\\
\log\!\big(\check{\mathtt{R}}_k^\top\mathtt{R}_k^\top\mathtt{R}_{k+1}\big)^\vee\end{bmatrix}\in\mathbb{R}^6$$

**Board** (unary), residual **on the measurement** $\mathbf{z}^B_a=\mathtt{T}^{\text{cam}}_{\text{board}}$:

$$\mathbf{r}^{B}_a=\log\!\Big(\hat{\mathbf{z}}_a^{-1}\,\mathbf{z}^B_a\Big)^\vee,\qquad
\hat{\mathbf{z}}_a=\big(\mathbf{x}_{k(a)}\,\mathtt{T}^{\text{body}}_{\text{cam}}\big)^{-1}\mathtt{T}^{\text{map}}_{\text{board}}$$

**Static observer** (unary, 3-DoF), $\mathbf{p}=(\mathtt{T}^{\text{map}}_{\text{infra}})^{-1}(\mathbf{t}_k+\mathtt{R}_k\boldsymbol\beta_I)$:

$$\mathbf{r}^{I}_k=\begin{bmatrix}\rho-\|\mathbf{p}\|\\
\angle\!\big(\alpha-\operatorname{atan2}(p_y,p_x)\big)\\
\angle\!\big(\epsilon-\arcsin(p_z/\|\mathbf{p}\|)\big)\end{bmatrix}\in\mathbb{R}^3$$

**Peer** (unary; the peer's pose is known):

$$\mathbf{r}^{P}_k=\log\!\Big(\hat{\mathbf{z}}_k^{-1}\mathbf{z}^P_k\Big)^\vee,\qquad
\hat{\mathbf{z}}_k=\big(\mathtt{T}^{\text{map}}_{\text{peer}}(t_k)\big)^{-1}\mathbf{x}_k$$

**Prior map** (unary, per point, so degenerate directions stay visible):

$$\mathbf{r}^{M}_{k}=\Big\{\mathbf{n}_i^\top\big(\mathbf{x}_k\mathtt{T}^{\text{body}}_{\text{sens}}\mathbf{p}_i-\mathbf{q}_i\big)\Big\}_{i\in\mathcal{C}_k}$$

### 5.3 Objective

$$\boxed{\;(\hat{\mathcal{X}},\hat{\boldsymbol\beta})=\arg\min\;
\sum_{k=1}^{N-1}\big\|\mathbf{r}^{O}_k\big\|^2_{\Sigma_{O,k}}
\;+\!\sum_{s\in\mathcal{S}_{\text{fit}}}\sum_{m\in\mathcal{M}_s}
\rho_s\!\Big(\big\|\mathbf{r}^{s}_m(\mathcal{X},\boldsymbol\beta_s)\big\|^2_{\Sigma_{s,m}}\Big)
\;+\!\sum_{s\in\mathcal{S}_\beta}\big\|\boldsymbol\beta_s\big\|^2_{\Sigma_{\beta_s}}\;}$$

with $\|\mathbf{r}\|^2_\Sigma=\mathbf{r}^\top\Sigma^{-1}\mathbf{r}$ and $\rho_s$ a
Huber kernel with scale $c_s$ (§7.1). The last term is a weak prior holding
each bias near zero so that it is identifiable only where the data demand it.

### 5.4 Gauge and observability

At least one absolute source must be present and must span rank 6 over the
trajectory, else $\operatorname{rank}\mathtt{H}\le6N-6$ and the solver
**refuses**. (Three-DoF sources such as $\mathbf{r}^I$ fix the gauge only when
observations at three or more non-collinear positions exist.) The solve
reports leading/trailing unanchored path (lever) and the longest interior gap
(bridge) as inputs to the coverage law (§9.3).

### 5.5 Solution

Gauss–Newton with Levenberg–Marquardt damping; Jacobians by central differences
on gathered endpoints; $\mathtt{H}$ block-tridiagonal in $\mathcal{X}$ because
odometry is consecutive and every other factor is unary. Bias states add a
dense row/column (an *arrow* matrix), eliminated by Schur complement in
$O(N)$. Initialisation: $SE(3)$ fit of the backbone onto its own absolute
observations.

## 6. Uncertainty per source of truth

Every source's covariance has three parts, and the parts behave differently:

$$\Sigma_{s,m}=\underbrace{\frac{\Sigma^{\text{meas}}_{s,m}}{n_{\text{eff}}(m)}}_{\text{noise: averages down}}
\;+\;\underbrace{\mathtt{J}_{s,m}\,\Sigma^{\text{anchor}}_s\,\mathtt{J}_{s,m}^\top}_{\text{anchor: does not}}
\;+\;\underbrace{\kappa_s\,\Sigma_{s,m}}_{\text{calibration (§7.2)}}$$

| source | $\Sigma^{\text{meas}}$ | $\Sigma^{\text{anchor}}$ | $n_{\text{eff}}$ | typical magnitude |
|---|---|---|---|---|
| board | PnP covariance from the corner reprojection Hessian — grows with range and incidence | board survey scatter: 6.9 mm `anchor`, ~13 mm `anchor_b`; orientation 0.5–0.7° | detections per dwell **de-correlated** (§8.2); a stationary dwell has $n_{\text{eff}}\ll n$ | 8 mm / 0.7° per view; survey floor 7–13 mm |
| static observer | radar az sd 3.5°, range sd 0.53 m (measured) | mast pose: **must be stated**; currently six decimals and no sigma | returns correlated within a sweep; $\approx$ one per sweep | 0.3–0.5 m per return; averages to ~0.1 m over a pass; **bias from cluster centroid modelled by $\boldsymbol\beta_I$** |
| peer | the peer's detector | the peer's own reference uncertainty (15 mm) | one per detection | ~15–50 mm |
| prior map | per-point plane-fit residual; **degenerate directions carry no information and must not be given any** | the map's own accuracy (anchored to boards: 7–15 mm) | one per registered frame | cm where observable; **unbounded along a corridor** |
| odometry | RPE$_{1\text{m}}/\sqrt{E}$ per edge, $\times100$ across reset gaps | — | assumes independent edges (A1, tested by §7.2) | 11 mm / 0.22° per edge |

Two rules follow, and both are already violated by the current implementation:

* **The anchor term does not average.** Ten thousand radar returns cannot make
  `infra_1` better than the mast survey. Hence the mast needs a sigma.
* **Correlated observations must not be counted as independent.** A stationary
  dwell producing 86 near-identical detections is not 86 measurements; treating
  it so over-weights the anchor by $\sqrt{86}$. See §8.2.

**Frame ancestry — stated, because it bounds what independence means.** The
boards were surveyed directly. The map was built from `mobile_1`'s LiDAR and
anchored *to the boards*. `infra_1`'s pose was recovered from the map. The peer's
reference is registered into the map. So map, observer and peer share the map's
**local geometry**; the boards do not. A global rigid error of the map frame is
common to all four and irrelevant (every trajectory lives in the same frame). A
*local* distortion of the map is shared by three of them and not by the boards.
The strongest independent pair is therefore {boards} against {map-derived
sources}, and leave-one-source-out must be read with that in mind.

## 7. How disagreement between sources is settled

Sources will disagree. Three mechanisms, in order, and the third is a result
rather than a fix.

### 7.1 Inside the solve: information weighting and bounded influence

The estimate is the information-weighted combination of every source: at a
pose constrained by sources $s_1,s_2$, the posterior information is
$\Sigma_{s_1}^{-1}+\Sigma_{s_2}^{-1}$ and each pulls in proportion to its
inverse covariance. That is correct **only if the covariances are honest**.

The Huber kernel bounds what a dishonest or outlying source can do: beyond
its scale $c_s$ the residual contributes gradient $\propto c_s$, not
$\propto\|\mathbf{r}\|$, so an outlier pulls linearly instead of
quadratically. $c_s$ is set at the 95% point of $\chi^2_d$, i.e. a residual
more than $\sim$2.5 σ from the current solution is treated as suspect. This
handles a flipped PnP solution, a mis-associated radar return, an ICP basin
error — **noise-like disagreement**.

### 7.2 Across solves: consistency calibration

Weights settle disagreement correctly only if declared sigmas are true. They
are tested rather than trusted:

$$\text{NEES}_s=\frac{1}{d\,|\mathcal{M}_s|}\sum_{m}\mathbf{r}_m^{s\top}
\Big(\Sigma_{s,m}+\mathtt{J}_m\,\mathtt{P}^{(-s)}\,\mathtt{J}_m^\top\Big)^{-1}\mathbf{r}^{s}_m
\quad\text{at }\hat{\mathcal{X}}^{(-s)}$$

where $\mathtt{P}^{(-s)}$ is the marginal covariance of the relevant pose from
the solve *without* $s$. Under a correct model $\text{NEES}_s\approx1$. Then:

$$\kappa_s\leftarrow\kappa_s\cdot\text{NEES}_s,\qquad\text{re-solve, repeat until all }\text{NEES}_s\in[\chi^2_{d,0.025},\chi^2_{d,0.975}]/d$$

This is variance-component estimation (Helmert) applied per source. **A source
that overclaimed its accuracy is downweighted by exactly the factor it
overclaimed by, as measured against the other sources.** Disagreement is
settled by evidence, not by the declaration. The odometry sigma's
independence assumption (A1) is tested the same way, with $s=O$ held out
against the absolute sources.

### 7.3 Irreconcilable disagreement is a result

If two sources are each consistent when the other is held out
($\text{NEES}\approx1$ both ways) but the joint solve leaves both inconsistent,
they disagree **systematically**, and the graph cannot arbitrate a bias. Then:

1. If a bias of known form exists (centroid lever, clock offset), add
   $\boldsymbol\beta_s$ (§8.3) and re-test. If it resolves, the bias is
   *estimated* and reported.
2. Otherwise report the disagreement magnitude as a **lower bound on the error
   of at least one source**, and use the frame ancestry (§6) to say which is
   likelier: a map-derived source disagreeing with the boards implicates the
   map's local geometry before it implicates the survey.
3. Never resolve it by choosing. A construction that silently prefers one
   source is the failure this project exists to prevent.

## 8. Reformulation — what changes from the current implementation

The current code (`slambench/graph.py`, `scripts/build_pseudo_gt.py`) is the
odometry + board-prior special case of §5. Four changes make it the problem
above. The headline comparison (V1 vs V2c) is unaffected by all four, since
they apply identically to both sides of it.

### 8.1 Board residual on the measurement, with propagated PnP covariance
Replaces the derived camera-in-map pose prior and the flat 7 mm / 1° sigma.
A view at 3 m stops pulling as hard as a view at 0.7 m. (Audit item D2.)

### 8.2 Effective sample size per dwell
Collapse a dwell's detections into one prior per *independent* observation:
$n_{\text{eff}}=n/(1+2\sum_{\ell\ge1}\rho_\ell)$ from the detection stream's
own autocorrelation (`slambench/blockstats.correlation_time`), and inflate the
per-prior sigma by $\sqrt{n/n_{\text{eff}}}$. Predicted effect: the local RPE
cost of anchoring (91 → 150 mm) shrinks, because over-tight anchors were
kinking the trajectory at the dwells. (D1.)

### 8.3 Per-source bias states
$\boldsymbol\beta_I\in\mathbb{R}^3$ for the radar centroid-to-body lever;
scalar clock offsets $\beta_{\tau,s}$ where a source runs on its own clock
(`infra_1` does). Weak zero-mean prior; arrow structure in $\mathtt{H}$.
Identifiable only under sufficient viewing diversity; the solve reports the
posterior sigma of each bias so an unidentified one is visible.

### 8.4 Not changed: loop closures stay out
By decision. RTAB-Map's loop-closure links exist in `rtabmap.db` and are not
exported; the route revisits (280 / 140 interior closest approaches). Adding
them as $\mathbf{r}^{L}_{ij}$, $j\gg i+1$, would reduce $\sigma_D$ in the
coverage law and break the block-tridiagonal structure. Deferred, recorded,
not in the problem.

## 9. Certification

### 9.1 Tiers
| tier | what | available on |
|---|---|---|
| A | held-out reference **sensor** | `mobile_1` |
| B | held-out absolute **source** (leave-one-source-out, §7.2) | any platform with ≥2 sources |
| C | internal: σ sweeps, under-determination guard, gauge refusal, revisit residuals | always |

### 9.2 The rule
**Every absolute source is a factor or a check in a given solve, never both;
with several sources, rotate.** A residual at a consumed source is
$\sigma/\sqrt{n}$ self-agreement, not evidence.

### 9.3 The coverage law and its test
$$e(t)\approx\min_a\big[\sigma_a+D(|t-t_a|)\big],\qquad
e_{\text{peak,bridge}}\approx\tfrac12\sigma_D\sqrt{T}\ \ (\text{random-walk drift}),\qquad
e_{\text{lever}}\ \text{unbounded}$$
Assumption A6: drift is a random walk. Systematic drift gives $\propto T$. The
error-vs-time-to-nearest-anchor curve on `mobile_1` tests this; the masking
protocol [RTK-SLAM] on EuRoC / TUM-VI / Newer College sweeps coverage to map it.

### 9.4 Transfer to the subject platform
State three things together: graph-propagated **precision**; **accuracy**
predicted by the coverage law at the subject's own measured coverage and drift
rate; and **corroboration** by leave-one-source-out on the subject itself
(`infra_1`, prior map). If corroboration contradicts prediction, the transfer
failed and that is reported.

## 10. What is measured, what is assumed, what is not claimed

**Measured:** 199.3 mm unaligned vs 331.3 mm fitted (gain 0.60); rotation
4.29° → 3.45°; RPE$_{1\text{m}}$ 91 → 150 mm; σ sweep ×4 moves 1.1 mm; one-board
hold-out swings 7.26 m at residual 0.01; boards vs LiDAR agree to 10 mm;
detector reproduces the pipeline to 8 mm / 0.66°; anchor coverage `mobile_1`
~0 / ~0 / 118 s, `mobile_2` 0 / 8.5 / 125 s.

**Assumed:** per-edge odometry independence (tested by §7.2, not by the σ
sweep); diagonal covariances; common clock; random-walk drift; peer pose
known. Each is stated in the paper.

**Not claimed:** the estimator; the masking protocol; unaligned scoring;
anchor placement as a question; ~20 cm pseudo-GT as an instrument.

**Not yet run:** the anchor-σ sweep; the theory plot; leave-one-source-out
(needs `map_cloud`, which is null, and `infra_1` factors, which are not
written); the masking experiments.

## 11. References

Confirmed by search this session (arXiv identifiers checked): *

* **MoCap2GT** — *A High-Precision Ground Truth Estimator for SLAM Benchmarking Based on Motion Capture and IMU Fusion.* RA-L 2026. arXiv 2507.12920. *
* **PALoc** — *Advancing SLAM Benchmarking with Prior-Assisted 6-DoF Trajectory Generation and Uncertainty Estimation.* IEEE/ASME TMech 2024. arXiv 2401.17826. *
* **Marker-Constrained Pose-Graph Correction for Cross-Platform Georeferencing in GNSS-Denied Environments.** arXiv 2608.16281, Aug 2026. *
* **RTK-SLAM** — *An RTK-SLAM Dataset for Absolute Accuracy Evaluation in GNSS-Degraded Environments.* arXiv 2604.07151. * (masking protocol; SE(3)-aligned ATE underestimates global error by up to 76%)
* **HortiMulti** — *A Multi-Sensor Dataset for Localisation and Mapping in Horticultural Polytunnels* (Poly-TagSLAM; leave-one-out over 35 surveyed AprilTags, 5.9 cm). arXiv 2603.20150. *
* **XR-HPGT** — *Spatiotemporal Calibration and Ground Truth Estimation for High-Precision SLAM Benchmarking in Extended Reality.* IEEE TVCG 2025. arXiv 2512.07221. *
* **RTS-GT** — Vaidis et al., *Robotic Total Stations Ground Truthing dataset.* arXiv 2309.11935. *
* Vaidis et al., *Uncertainty analysis for accurate ground truth trajectories with robotic total stations.* IROS 2023. arXiv 2308.01553. *
* Mittelstedt et al., *Factor Graph-Based Ground Truth Trajectory Estimation by Fusing Robotic Total Station and Inertial Measurements.* IEEE 2025. *
* **TagSLAM** — Pfrommer & Daniilidis, *Robust SLAM with Fiducial Markers.* arXiv 1910.00679. *
* **Princeton365** — *A Diverse Dataset with Accurate Camera Pose.* arXiv 2506.09035. * (~20 cm pseudo-GT as an instrument)
* *Anchor selection for SLAM based on graph topology and sub-graph…* UTS. * (anchors as zero-uncertainty loop closures to the origin)
* **Hilti SLAM Challenge 2023** — arXiv 2404.09765; **Hilti-Trimble 2026** (floor-plan priors, GT released). *
* Sünderhauf & Protzel, *Switchable constraints for robust pose graph SLAM.* IROS 2012. * (robust loop closure)
* **Boxi** — *Design Decisions in the Context of Algorithmic Performance for Robotics.* arXiv 2504.18500. *
* *Look Ma, No Ground Truth! Ground-Truth-Free Tuning of SfM and Visual SLAM.* arXiv 2412.01116. *

Standard references (from general knowledge; verify before submission):

* Burri et al., *The EuRoC micro aerial vehicle datasets.* IJRR 2016.
* Geneva & Huang, *Vicon2GT: Distortion-free ground truth for MoCap-based visual-inertial datasets.* 2020.
* Furgale, Rehder, Siegwart, *Unified temporal and spatial calibration for multi-sensor systems* (Kalibr). IROS 2013.
* Ramezani et al., *The Newer College Dataset.* IROS 2020.
* Tao et al., *The Oxford Spires Dataset.* 2024.
* Sturm et al., *A benchmark for the evaluation of RGB-D SLAM systems* (ATE/RPE). IROS 2012.
* Umeyama, *Least-squares estimation of transformation parameters between two point patterns.* TPAMI 1991.
* Grisetti, Kümmerle, Stachniss, Burgard, *A tutorial on graph-based SLAM.* ITS Magazine 2010.
* Kümmerle et al., *g2o: A general framework for graph optimization.* ICRA 2011.
* Solà, Deray, Atchuthan, *A micro Lie theory for state estimation in robotics.* 2018.
* Forster, Carlone, Dellaert, Scaramuzza, *On-manifold preintegration for real-time visual-inertial odometry.* T-RO 2017. (the excluded IMU rung)
* Huber, *Robust estimation of a location parameter.* Ann. Math. Stat. 1964.
* Bar-Shalom, Li, Kirubarajan, *Estimation with Applications to Tracking and Navigation.* Wiley 2001. (NEES consistency)
* Helmert, variance-component estimation; see Koch, *Parameter Estimation and Hypothesis Testing in Linear Models.* Springer 1999.
* Künsch, *The jackknife and the bootstrap for general stationary observations.* Ann. Stat. 1989. (block bootstrap)
* Shimodaira, *Improving predictive inference under covariate shift.* JSPI 2000.
* PIN-SLAM, Pan et al., T-RO 2024. (pseudo-GT from a stronger SLAM, asserted not certified)

\* arXiv identifiers and claims marked with an asterisk were confirmed from abstracts and indexed summaries this session; full texts of PALoc and the marker-correction paper were not accessible (arXiv blocked by the container proxy) and must be read before the related-work section is finalised.
