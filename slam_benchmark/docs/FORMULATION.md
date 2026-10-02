# The optimisation problem, formally — and an audit of whether it is sound

`STAGE2_GRAPH.md` describes the design. This file states the problem in full
and then **audits it**, grading every component SOUND, ASSUMED or DEFECT.

**Scope.** All factors here are either binary on consecutive poses or unary on a
single pose. Loop-closure factors are **deliberately excluded** — §10 records
why and what they would cost.

---

## 1. State

$$\mathcal{X}=\{\mathbf{x}_1,\dots,\mathbf{x}_N\},\qquad
\mathbf{x}_k=\mathtt{T}^{\text{map}}_{\text{body}}(t_k)=(\mathtt{R}_k,\mathbf{t}_k)\in SO(3)\times\mathbb{R}^3$$

at the backbone's keyframe rate, in the surveyed `map` frame. $N\approx1500$,
so $6N\approx9000$ free parameters.

Increments live in the tangent space, $\boldsymbol\delta_k=(\boldsymbol\rho_k,\boldsymbol\phi_k)\in\mathbb{R}^6$, applied by

$$\mathcal{X}\oplus\boldsymbol\delta:\quad
\mathbf{t}_k\mapsto\mathbf{t}_k+\boldsymbol\rho_k,\qquad
\mathtt{R}_k\mapsto\mathtt{R}_k\exp(\boldsymbol\phi_k^\wedge)$$

Sensor extrinsics are **declared, never estimated**: cumulative rotation per
optical axis is $[140,1094,101]^\circ$ and the correlation matrix is rank one
(singular values 44.238 / 0.014 / 0.005), so they are unidentifiable on this
motion.

## 2. Residuals

**Backbone odometry**, $k=1\dots N\!-\!1$, measurement $(\check{\mathtt{R}}_k,\check{\mathbf{t}}_k)$ read off the front-end's consecutive relative pose:

$$\mathbf{r}^{O}_k=\begin{bmatrix}
\mathtt{R}_k^\top(\mathbf{t}_{k+1}-\mathbf{t}_k)-\check{\mathbf{t}}_k\\[2pt]
\log\!\big(\check{\mathtt{R}}_k^\top\mathtt{R}_k^\top\mathtt{R}_{k+1}\big)^\vee
\end{bmatrix}\in\mathbb{R}^6$$

**Surveyed-board anchor**, $a\in\mathcal{A}$. *As implemented*, the board
detection is first composed into a camera-in-map pose
$(\bar{\mathtt{R}}_a,\bar{\mathbf{t}}_a)=\mathtt{T}^{\text{map}}_{\text{board}}(\mathtt{T}^{\text{cam}}_{\text{board}})^{-1}$,
resampled onto the keyframe axis, and applied as a pose prior:

$$\mathbf{r}^{B}_a=\begin{bmatrix}\mathbf{t}_a-\bar{\mathbf{t}}_a\\[2pt]
\log\!\big(\bar{\mathtt{R}}_a^\top\mathtt{R}_a\big)^\vee\end{bmatrix}\in\mathbb{R}^6$$

The principled form, residual on the *measurement* rather than a derived pose,
is $\mathbf{r}^{B}_a=\log\big(\hat{\mathbf{z}}_a^{-1}\mathbf{z}_a\big)^\vee$ with
$\hat{\mathbf{z}}_a=(\mathbf{x}_a\mathtt{T}^{\text{body}}_{\text{cam}})^{-1}\mathtt{T}^{\text{map}}_{\text{board}}$.
See audit item **D2**.

**Static surveyed observer** (not yet implemented). Range/bearing only, so it
constrains **3 DoF, not 6**. With $\mathbf{p}=(\mathtt{T}^{\text{map}}_{\text{infra}})^{-1}\mathbf{t}_k$:

$$\mathbf{r}^{I}_k=\begin{bmatrix}\rho-\|\mathbf{p}\|\\
\angle(\alpha-\operatorname{atan2}(p_y,p_x))\\
\angle(\epsilon-\arcsin(p_z/\|\mathbf{p}\|))\end{bmatrix}\in\mathbb{R}^3$$

**Peer robot** (not yet implemented). The peer's pose is treated as **known**,
which makes this unary on $\mathbf{x}_k$:

$$\mathbf{r}^{P}_k=\log\Big(\hat{\mathbf{z}}_k^{-1}\mathbf{z}_k\Big)^\vee,\qquad
\hat{\mathbf{z}}_k=\big(\mathtt{T}^{\text{map}}_{\text{peer}}(t_k)\big)^{-1}\mathbf{x}_k$$

**Prior-map registration** (not yet implemented), point-to-plane, kept per point
so degenerate directions stay visible:

$$\mathbf{r}^{M}_k=\Big\{\mathbf{n}_i^\top\big(\mathbf{x}_k\mathtt{T}^{\text{body}}_{\text{sens}}\mathbf{p}_i-\mathbf{q}_i\big)\Big\}_{i\in\mathcal{C}_k}$$

## 3. Covariances

All diagonal — a survey states a scatter, not a correlation.

$$\Sigma_O=\operatorname{diag}(\sigma_t^2\mathbf{1}_3,\sigma_r^2\mathbf{1}_3)\cdot\gamma_k,
\qquad\gamma_k=\begin{cases}100^2&\Delta t_k>3\widetilde{\Delta t}\\1&\text{else}\end{cases}$$

$$\sigma_t=\frac{\text{RPE}^{\text{trans}}_{1\text{m}}}{\sqrt{E}},\qquad
\sigma_r=\frac{\text{RPE}^{\text{rot}}_{1\text{m}}}{\sqrt{E}},\qquad
E=\frac{N-1}{L}\ \text{edges per metre}$$

Measured: $\sigma_t=11.1$ mm, $\sigma_r=0.223^\circ$ per edge.

$$\Sigma_B=\operatorname{diag}(\sigma_b^2\mathbf{1}_3,\sigma_\psi^2\mathbf{1}_3),
\qquad\sigma_b=7\ \text{mm},\ \sigma_\psi=1.0^\circ\ \text{(per board, declared)}$$

## 4. Objective

$$\boxed{\ \hat{\mathcal{X}}=\arg\min_{\mathcal{X}\in(SO(3)\times\mathbb{R}^3)^N}
\sum_{k=1}^{N-1}\big\|\mathbf{r}^{O}_k\big\|^2_{\Sigma_O}
+\sum_{s\in\mathcal{S}_{\text{fit}}}\sum_{m\in\mathcal{M}_s}
\rho_s\Big(\big\|\mathbf{r}^{s}_m(\mathcal{X})\big\|^2_{\Sigma_s}\Big)\ }$$

$\|\mathbf{r}\|^2_\Sigma=\mathbf{r}^\top\Sigma^{-1}\mathbf{r}$. Currently
$\mathcal{S}_{\text{fit}}=\{B\}$ and $\rho_s=\mathrm{id}$ (audit **D4**).

## 5. Gauge and observability

$$\sum_{s\in\mathcal{S}_{\text{fit}}}|\mathcal{M}_s|\ge1
\quad\text{else}\quad\operatorname{rank}(\mathtt{H})=6N-6\ \Rightarrow\ \textbf{refuse}$$

V1 (no absolute source) adds one synthetic gauge prior at $\sigma=10^{-4}$ m /
$10^{-5}$ rad. V2c must **not** — it would fight the anchors.

Reported beside every solve, because they are the inputs to the coverage model:
leading / trailing unanchored time and path (lever), and the longest interior
gap (bridge).

## 6. Solution

Gauss-Newton with Levenberg-Marquardt damping. Jacobians by central differences
($\varepsilon=10^{-6}$) on **gathered** endpoints. Normal equations assembled
block-wise:

$$\mathtt{H}_{kk}\mathrel{+}=\mathtt{J}_i^\top\Sigma^{-1}\mathtt{J}_i,\quad
\mathtt{H}_{k+1,k+1}\mathrel{+}=\mathtt{J}_j^\top\Sigma^{-1}\mathtt{J}_j,\quad
\mathtt{H}_{k,k+1}\mathrel{+}=\mathtt{J}_i^\top\Sigma^{-1}\mathtt{J}_j$$

Because odometry joins only $(k,k+1)$ and every other factor is unary,
$\mathtt{H}$ is **block-tridiagonal** and solves exactly by block Thomas in
$O(N)$.

$$(\mathtt{H}+\lambda\operatorname{diag}\mathtt{H})\boldsymbol\delta=\mathbf{b},
\qquad\text{accept if }C(\mathcal{X}\oplus\boldsymbol\delta)<C(\mathcal{X})$$

Initialisation: $\mathcal{X}^{(0)}=\mathtt{T}^\star\mathcal{X}^{\text{bb}}$ with
$\mathtt{T}^\star$ the $SE(3)$ (not $Sim(3)$) fit of the backbone onto its own
anchor observations.

## 7. Certification: leave-one-source-out

$$\hat{\mathcal{X}}^{(-s)}=\arg\min_{\mathcal{X}}\Big[\sum_k\|\mathbf{r}^{O}_k\|^2_{\Sigma_O}
+\sum_{s'\neq s}\sum_m\rho_{s'}\big(\|\mathbf{r}^{s'}_m\|^2_{\Sigma_{s'}}\big)\Big],
\qquad\mathbf{e}_s=\big\{\mathbf{r}^{s}_m(\hat{\mathcal{X}}^{(-s)})\big\}$$

Report **(i)** the error distribution of $\|\mathbf{e}_s\|$ — median, $p_{95}$,
worst stretch, block-bootstrap interval — and **(ii)** consistency:

$$\text{NEES}_s=\frac{1}{|\mathcal{M}_s|}\sum_m\mathbf{r}_m^{s\top}
\big(\Sigma_s+\mathtt{H}^{-1}_{(-s)}\big)^{-1}\mathbf{r}^{s}_m
\ \overset{?}{\sim}\ \chi^2_d/d$$

$\approx1$ consistent; $\gg1$ overconfident (the error bar is wrong even if the
trajectory is decent); $\ll1$ conservative.

Leave-one-**source**-out tests what leave-one-**landmark**-out cannot: a bias
shared by every landmark of one kind shifts them together, and each held-out
landmark is still predicted well.

---

## 8. AUDIT

### Sound

| # | component | why |
|---|---|---|
| S1 | $SO(3)\times\mathbb{R}^3$ parameterisation and retraction | rotations are not a vector space; increments in the tangent space with $\exp$ back onto the manifold is the standard correct construction |
| S2 | residual forms $\mathbf{r}^O,\mathbf{r}^B$ | standard between/prior residuals; `so3_log` via quaternion (Shepperd) stays conditioned near $0$ and $\pi$ |
| S3 | whitening by $\Sigma^{-1/2}$ | makes mm and degrees commensurable; the cost is a proper Mahalanobis sum |
| S4 | block-tridiagonal claim | follows from adjacency-only binary + unary-only everything else. `solve()` **refuses** a graph violating it rather than computing something wrong |
| S5 | gauge refusal | with no prior, $\mathtt{H}$ is rank-deficient by exactly 6 and any answer is as good as any other. Refusing is correct |
| S6 | Jacobians on gathered endpoints | perturbing in place folds the $j$-Jacobian into the $i$-Jacobian on a chain and makes $\mathtt{H}$ singular. Verified by test |
| S7 | $SE(3)$ not $Sim(3)$ initialisation | scale is not a free parameter of a metric construction |
| S8 | under-determination guard | low whitened residual **with** large move is the signature of a solution family. Caught the 7.26 m lever, which converged at rms 0.01/0.08 |

### Assumed — defensible, but they are assumptions and must be stated

| # | assumption | exposure |
|---|---|---|
| A1 | **odometry errors independent per edge**, hence $\sigma=\text{RPE}_{1\text{m}}/\sqrt{E}$ | false in general: scale error and bias persist across edges. The $\times0.5/\times2$ sweep moved the answer 1.1 mm, but **that sweep only rescales magnitude uniformly — it does not perturb the correlation structure**, so it does not rule this out. A correlated model changes the *shape* of the solution, not just its scale |
| A2 | **diagonal covariance everywhere** | a survey states scatter, not correlation. Honest, but PnP's actual covariance is strongly non-diagonal (range/bearing anisotropy) |
| A3 | **common clock across streams** | no time-offset state, unlike MoCap2GT (B-spline offset) and XR-HPGT (variable time sync). Defensible — board detections and images share a header stamp — but it must be argued in the paper, not omitted |
| A4 | **linear + SLERP observation resampling** | first-order; the literature uses cubic B-splines. Small here because anchors are sparse and the platform is slow during dwells |
| A5 | **peer pose known, not estimated** | keeps the factor unary and $\mathtt{H}$ banded. Ignores correlation between the peer's own pose errors at different times |
| A6 | **drift is a random walk**, giving $e_{\text{peak}}\propto\sqrt{T}/2$ | systematic drift (scale error) grows as $T$, not $\sqrt{T}$. **This is the model's central assumption and the theory plot is exactly its test.** If the measured curve is linear in $T$, the bridge model is wrong and the transfer argument weakens |

### Defects — real, ordered by how much they could move the answer

**D1 — anchor observations are treated as independent when they are not.**
`anchor_factors` emits one prior per **keyframe in the dwell window**, each
interpolated from a detection stream, each carrying the full declared
$\sigma_b=7$ mm. During a stationary dwell, consecutive keyframes see
essentially the same view through the same detector, so those priors share
their error. Treating $n=86$ of them as independent gives an effective anchor
strength of $7/\sqrt{86}\approx0.75$ mm — and the repo has **observed exactly
that number**, reading 0.7 mm at `anchor` and correctly calling it
self-agreement. The same $\sqrt{n}$ that makes the residual look tiny is
over-weighting the anchor inside the solve, plausibly by nearly an order of
magnitude.

*Consequence, and it is testable:* over-tight anchors kink the trajectory at
the dwell windows. **The reported local cost — RPE $1$ m worsening $91\to150$
mm — may be partly an artefact of this rather than a necessary trade.**

*Fix:* estimate an effective sample size per dwell (e.g. from the detection
stream's own autocorrelation, which `slambench/blockstats.py` already computes)
and inflate $\sigma_b$ by $\sqrt{n/n_{\text{eff}}}$. *Cheap partial check:* an
anchor-$\sigma$ sweep, which has never been run — only the odometry-$\sigma$
sweep has.

**D2 — the board residual is written on a derived pose, not on the measurement.**
The detection is composed into camera-in-map before entering the graph, so
PnP's covariance is not propagated through the composition. Instead a flat
$7$ mm / $1.0^\circ$ is declared per board. **A detection at $0.7$ m and one at
$3$ m therefore pull equally hard**, when PnP's uncertainty grows sharply with
range and with incidence angle. Fix: residual on $\mathbf{z}$ with
$\Sigma_B$ from the PnP Hessian.

**D3 — $\Sigma_B$ omits the survey's own uncertainty.** The declared $7$ mm is
detector scatter. The board's surveyed position has its own scatter
(6.86 mm on `anchor`, ~13 mm on `anchor_b`), which should add in quadrature:
$\Sigma_B=\Sigma_{\text{det}}+\mathtt{J}\Sigma_{\text{survey}}\mathtt{J}^\top$.
This is *"an anchored fix cannot be better than its anchor"*, and the scatter
term averages down across observations while the survey term does not.

**D4 — no robust kernel.** $\rho_s=\mathrm{id}$. A planar board admits two PnP
solutions and a flipped one enters at full weight. Currently mitigated
upstream by `min_ambiguity_ratio` and by `board_detect.py` gating solved poses
against the reference trajectory — **which is reference-informed**, so the
mitigation is not available to a platform without a reference. With four
sources and their association failures, a Huber kernel stops being optional.

**D5 — $\sigma_O$ is reference-informed, and for `mobile_2` it is nearly
circular.** $\sigma_O$ derives from the run's RPE, scored against the reference.
On `mobile_1` that reference is a held-out LiDAR, so the dependency is real but
benign. On `mobile_2` the reference is **cuVSLAM-derived and correlated with
the backbone**, so the weights are set by a quantity that is partly the
backbone's agreement with itself. `run.json` records
`reference_informed: true`, but the `mobile_2` case is worse than that flag
conveys.

**D6 — `reset_edges` is a heuristic with a declared blast radius.** Edges
crossing a gap $>3\times$ the median period get $\sigma\times100$. It declines
entirely above 10% flagged, after misfiring on MASt3R (65 of 150 edges, 43% of
the chain freed, residuals driven to 0.00). Documented and guarded, but it is
still a threshold, not a measurement.

### What the audit does **not** overturn

The headline comparison stands. V1 and V2c share the same backbone, the same
$\sigma_O$, the same solver and the same anchor treatment, so D1–D5 apply
**identically to both**. **199.3 mm unaligned against 331.3 mm $SE(3)$-fitted
to the reference is a fair comparison**, and D1 predicts the *local* RPE cost
is overstated — i.e. the trade is better than reported, not worse.

---

## 9. Priority

1. **D1** — anchor $\sigma$ sweep first (one command, never run), then effective
   sample size. Largest plausible effect, and it may recover the RPE trade.
2. **A6** — the theory plot. Cheap, and it can falsify the coverage model that
   the whole transfer argument rests on.
3. **D3** — quadrature with the survey scatter. Minutes.
4. **D2, D4** — needed before the multi-source extension, not before the
   current result.
5. **D5** — state it in the paper for `mobile_2`; the fix is a
   reference-free $\sigma_O$ (e.g. from backbone self-consistency at revisits).

---

## 10. Excluded: loop-closure factors

**Not in the problem above, by decision.** Recorded here because the omission is
substantive and should not be discovered by a reviewer.

The construction consumes `odometry.tum`, the dense front-end, and never the
loop-closure links RTAB-Map found. `docker/rtabmap/post.sh` exports optimised
poses from `rtabmap.db` and does not read the link table. So:

* **Loop closures are discarded.** The route *does* revisit — 280 interior
  closest approaches on `mobile_1`, 140 on `mobile_2`, within 0.30 m and
  $\ge20$ s apart — so this is information that exists and is unused.
* **Substituting `graph.tum` would be worse, not better.** Re-deriving
  consecutive relative poses from an *optimised* trajectory yields measurements
  the optimiser has already correlated, violating A1 far more severely, while
  leaving the loop-closure constraint itself absent from the cost — so the
  anchors could undo it at zero cost.

The principled version consumes RTAB-Map's **measurements** (neighbour links +
loop-closure links) rather than its answer, and re-optimises with the anchors
added. The factor is an ordinary between factor with $j\gg i+1$:

$$\mathbf{r}^{L}_{ij}=\begin{bmatrix}\mathtt{R}_i^\top(\mathbf{t}_j-\mathbf{t}_i)-\check{\mathbf{t}}_{ij}\\[2pt]
\log\!\big(\check{\mathtt{R}}_{ij}^\top\mathtt{R}_i^\top\mathtt{R}_j\big)^\vee\end{bmatrix}$$

**Why it would matter here:** in $e_{\text{peak}}\approx\tfrac12\sigma_D\sqrt{T}$,
anchors reduce $T$ while **loop closures reduce $\sigma_D$** — the two attack
the same quantity from opposite sides, and the error lives in the 118 s bridge.
It is also the only drift-bounding source needing no new data and no map cloud.

**Cost:** an off-band block at $(i,j)$ destroys the block-tridiagonal structure
and `solve()`'s $O(N)$ exact sweep. At $N\approx1500$ a general sparse Cholesky
is milliseconds, so the loss is elegance rather than performance.

Deferred, not rejected.
