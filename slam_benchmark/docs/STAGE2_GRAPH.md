# Stage 2 — the factor graph

The construction that produces the pseudo ground truth. `docs/PLAN.md` names
the rungs; this file says what is actually optimised, what each rung switches
on, and which rungs coop2 cannot price.

**The design is deliberately not novel.** It is the standard MoCap-fusion
estimator — EuRoC's own (2016), Vicon2GT (2020), Kalibr's MoCap branch,
MoCap2GT (RA-L Feb 2026) — with **surveyed board anchors substituted for the
motion-capture system**. Writing it this way is a choice: the machinery is
established, so the paper's claim rests on the regime and the certification
rather than on the solver. See *Novelty* at the end, and cite MoCap2GT.

---

## 1. What is being estimated

One trajectory per agent, expressed in the surveyed `map` frame, at the
backbone's own keyframe rate.

```
X = [ X_1 ... X_n , X_calib ]

X_k     = [ p_k, v_k, q_k, b_a,k, b_g,k ]     per keyframe k
X_calib = [ dt_1 ... dt_m ]                   time-offset control points
```

`p, q` are the body pose in `map`; `v` is velocity; `b_a, b_g` are IMU biases.

**What is NOT in the state vector, and why — this is where we depart from
MoCap2GT.**

| their state | ours | why |
|---|---|---|
| `T_MI` (MoCap↔IMU extrinsic) | **fixed, declared** | their extrinsic is estimated because a marker cluster's pose relative to the IMU is unknowable a priori. Ours is a mechanical constant recovered from the device's published geometry and cross-checked three ways (`configs/coop2.yaml`, `mobile_2.imu`). Estimating it here would be *refusing a measurement we have* — and the motion cannot identify it anyway: the cart yaws about one axis, the correlation matrix is rank one (44.238 / 0.014 / 0.005), and MoCap2GT's own Fig. 4 reports the same failure under insufficient rotational excitation. |
| `[θ_roll, θ_pitch]` gravity alignment | **not estimated** | their MoCap world frame is arbitrary relative to gravity. Our `map` frame is already gravity-aligned by the survey. If that assumption is wrong the board residuals will say so. |
| — | time offset control points | kept. See §4. |

---

## 2. The factors

Three classes, one of which has no analogue in the MoCap literature.

### 2.1 Backbone odometry factor — `r_O`

Between consecutive keyframes, the relative pose the backbone front-end
reports:

```
r_O,k = log( ΔT̃_k^-1 · (T_k^-1 · T_k+1) )        ∈ R^6
```

`ΔT̃_k` comes from `odometry.tum` (RTAB-Map's dense front-end, 2227 poses,
99% span — **not** the loop-closed graph, which stops 27 s early on mobile_1
because RTAB-Map adds nodes on motion and the cart parks).

This factor does not exist in MoCap2GT and its presence is the structural
difference between the two problems. They have absolute pose at 100 Hz
continuously, so nothing needs to carry the trajectory between measurements.
We have absolute pose in **four windows totalling ~30 s of 150**. The backbone
is what spans the other 120 s; the IMU alone would not survive it.

Covariance: **derived from the run's own RPE at 1 m** — per-edge σ = RMSE₁ₘ /
√(edges per metre), treating per-edge errors as independent
(`build_pseudo_gt.py --rung`; override with `--sigma-t/--sigma-r`). Two
honest caveats, both recorded in `run.json`: the RPE was scored against the
LiDAR, so this is a *reference-informed weight*, not an independent
measurement; and on synthetic data with planted 4 mm / 0.05° per edge the
derivation returned **6.4 mm / 0.09°** — a 1.6–1.8× over-estimate, because a
1 m RPE also carries rotation compounded into translation and the model
charges all of it to translation white noise. It loosens the odometry, so the
anchors pull somewhat harder than they should. Check the sensitivity (×0.5,
×2) before quoting any number that leans on it. Edges that cross a stamp gap
wider than 3× the median period are RTAB-Map resets and get ×100 σ: the chain
stays connected, the edge carries almost no weight.

Implemented: `slambench/graph.py`. Chain of between factors plus single-pose
priors → block-tridiagonal normal equations → exact O(N) block Thomas solve,
pure numpy. Jacobians numerical, on gathered endpoints (batching the
perturbation in place folds the j-Jacobian into the i-Jacobian and makes the
system singular — found the hard way, 2026-09-20).

### 2.2 Board anchor factor — `r_B`

For each frame in a dwell window where the ChArUco detector solved a pose:

```
r_B,j = [ p_kj  -  p̃_board_cam,j
          2·[ q̃_board_cam,j^-1 ⊗ q_kj ]_vec ]     ∈ R^6
```

where `p̃, q̃` is the camera pose in `map` derived from the surveyed board and
PnP (`scripts/board_detect.py`, validated to 8.1–8.3 mm / 0.58–0.66° against
the mapping pipeline's own detections).

Covariance: **the survey's own uncertainty**, which is the entire point —
7 mm at `anchor`, 15 mm at `rs_anchor`, and these are declared per board in
`configs/coop2.yaml` with their provenance. This is the only factor in the
graph whose error bar does not come from something being tested.

Available pairs (measured, 2026-09-20):

| board | mobile_1 | mobile_2 |
|---|---|---|
| `anchor` | 86 obs, σ 7 mm | 190 obs, σ 7 mm |
| `rs_anchor` | 353 obs, σ 15 mm | 92 obs, σ 15 mm |
| `anchor_b` | — | — |

`anchor_b` is unusable by **both** agents and that is a measurement, not a gap:
15 mm markers at 0.84 m in 848×480 give ~1.9 px/bit and the detector returned
0 of 179 frames. Each agent gets two boards.

**Interpolation.** A keyframe rarely coincides with a detection. MoCap2GT
interpolates its absolute measurement with a **cumulative cubic B-spline on
SE(3)** rather than linearly, for C² continuity and to suppress jitter. Adopt
that — our tier-2 rows already carry `interpolated across 2.4 s` caveats and
linear interpolation across a gap that long is the weakest link in the current
table.

### 2.3 IMU preintegration factor — `r_I`  (V2a, **blocked**)

Standard 9-D preintegration residual on Δp, Δv, Δq, plus bias random-walk
factors between consecutive keyframes.

**Blocked on `noise_model`, which is null and must stay null.** The factor's
covariance is not a cosmetic parameter — it *is* the factor's weight. Too
tight and the IMU drags the solution; too loose and the rung measures zero.
Reporting "what preintegration buys in centimetres" when that number is a
direct function of a guessed covariance is not a result.

Allan variance needs hours of static logging — the bias random-walk terms only
emerge at cluster times of 1e2–1e3 s — and the recording is 156 s. A static
segment inside it bounds the white noise only.

**What it would buy here, honestly:** not scale (RGB-D is already metric) and
not much attitude (the backbone already carries a gravity constraint). It
would buy **rigidity across vision dropouts**, which is paper task B5. That is
a real motivation, not a box to tick.

**To unblock:** leave the D455 still on a desk overnight logging IMU and run
Allan variance. Per-unit, one-time, serves every future recording.

---

## 3. The objective

```
argmin_X   Σ_k ‖r_O,k‖²_Σ_O  +  Σ_j ‖r_B,j‖²_Σ_B
                              [ + Σ_k ‖r_I,k‖²_Σ_I + bias random walk ]
```

Nonlinear least squares, Levenberg–Marquardt, sparse Cholesky. GTSAM if it
installs cleanly; the problem is small enough (≈2000 keyframes × 6 DoF, plus
~700 anchor factors) that a scipy sparse solve is a viable fallback and keeps
the dependency surface honest.

---

## 4. Initialisation

The backbone trajectory, rigidly aligned to the board observations. That is
already implemented — `slambench/align.py` `fit(mode="se3")` — and it is a
much better starting point than MoCap2GT's linear initialiser, because we
start from a converged SLAM solution rather than from nothing.

**What we keep from their initialiser:** nothing structural. `qMI` is declared,
not solved. But `scripts/imu_extrinsic.py` already implements their eq. 9 (the
hand-eye SVD) and it stays in the repo as the *check* that our declared
extrinsic is consistent with the gyro — it refuses on rank rather than
reporting, which is the correct behaviour on this motion.

**Time offset.** MoCap2GT models it as a linear B-spline over control points
because consumer IMU clocks drift >2 ms/min. `imu_extrinsic.py` measured +4 ms
between the RealSense IMU and the reference, assuming it constant. At their
quoted drift rate, 150 s gives ~5 ms of variation — **the same size as the
offset itself**. So the control points stay in the state vector even though
the current measurement is a single number.

---

## 5. Degeneracy rejection

Adopt MoCap2GT §IV-B3, adapted. Split into windows of `w = 5 s`; compute the
maximum rotation angle in each. Below a threshold the window is degenerate and
the anchor factor constrains only the pose states, contributing no gradient to
the calibration parameters.

Their threshold is 10° in 5 s. Ours: rotation is plentiful (median 3.77 °/s,
≈19° per 5 s window, 61% of intervals above 2 °/s) but **rank one**. So the
translation-degeneracy case they guard against is not our failure mode; ours is
axis degeneracy, and the guard that matters is the one already in
`imu_extrinsic.py` — refuse the calibration, keep the poses.

---

## 6. The rungs, and which ones coop2 can price

| rung | factors on | status |
|---|---|---|
| **V1** backbone only | `r_O` | **built and gated on synthetic** — reproduces its input exactly; awaiting the real run |
| **V2a** + IMU | `r_O + r_I` | **blocked** — noise model (§2.3) |
| **V2b** + radar ego-velocity | `r_O + r_V` | feasible; radar1/radar2 at 15.8 Hz, Doppler confirmed (I12), sign convention open. mobile_1 only |
| **V2c** + board anchors | `r_O + r_B` | **built and gated on synthetic** — 4 real pairs, detector at 8 mm; awaiting the real run |
| **V2d** + infrastructure | `r_O + r_B + r_N` | **blocked** — I9: `infra_1`'s pose has no stated uncertainty anywhere in the dataset, and an anchored fix cannot be better than its anchor |
| **V2e** + inter-agent | all | **unavailable** — needs simultaneous co-observation; the agents swap corners rather than share one. `docs/PLAN.md`'s next-session list already says to route them through a shared corridor |

**Two rungs are buildable, one more is plausible, three are not.** Report it
that way. Three unpriceable evidence classes each with a specific measured
reason is a finding about what a dataset must contain, which is a dataset
paper's business (rule 8).

---

## 7. Certification

V1 and V2c are both run on **mobile_1** and scored against the held-out LiDAR
reference — the instrument that never feeds the construction.

**Scoring rule, and it is not optional:** a V2c row is scored **unaligned**
(`alignment: none`, declared in `configs/methods/pseudo_gt_v2c.yaml`). Its
poses are in the surveyed map frame, placed by the boards; an se3 fit to the
LiDAR before scoring would remove exactly the placement the anchors provide
and re-use the reference doing it. V1 is in the backbone's own frame and is
aligned like any backbone. So the comparison the paper makes is: *the
construction's error in the map frame, with no use of the LiDAR*, against
*the backbone's error after an oracle fit to the LiDAR*. If the first beats
the second, the pseudo-GT stands on its own.

**Measured on synthetic data, 2026-09-20** (planted 4 mm / 0.05° per edge,
boards 7 / 15 mm, 0.5°, 150 s at 14.7 Hz, through the real `eval_run.py`):

| row | ATE | tier 2 |
|---|---|---|
| backbone, se3-aligned | 92.9 mm / 2.08° | 187 / 118 mm |
| **V1** | **92.9 mm / 2.08°** — identical, the gate passes | same |
| **V2c**, unaligned | **79.2 mm / 1.66°** | 8.9 / 20.6 mm (consumed: self-agreement) |
| V2c, `rs_anchor` held out, unaligned | 373 mm | 8.9 mm at `anchor`, **405 mm at `rs_anchor`** |

**And on REAL data the same test is far harsher.** Holding out `rs_anchor`
leaves `anchor` alone, 86 observations in a 6.5 s window at one end of a 152 s
run: the solve converges, whitened residuals fall to 0.01 / 0.08, and the
trajectory moves **7.3 m median / 9.4 m max**. Near-zero residuals with a large
move mean the system is under-determined, not that the fit is good;
`build_pseudo_gt.py` warns on that conjunction. `rs_anchor` alone (378
observations over 26.6 s) moved 178 mm and behaved. **What conditions this
graph is anchor coverage in TIME, not the number of boards** — which is the
sharpest thing this sweep says about `mobile_2`, whose two windows are 6.5 s
and 12 s.

Two things the earlier synthetic row says. A tier-2 residual at a *consumed* board is the
board's own scatter, not evidence; only a held-out board and the LiDAR are
independent (`--hold-out`). And **one board is worth nothing beyond fixing
the gauge**: with a single window the far end of the run drifts the full
150 s (405 mm), and under se3 alignment that row's ATE was indistinguishable
from the backbone's. Two boards at opposite ends of the route is the minimum
that buys anything. That is the shape of the claim for `mobile_2`, which has
exactly two.

**MEASURED ON REAL DATA, 2026-09-20** — `mobile_1`, four backbone runs, the
held-out LiDAR scoring and never feeding:

| row | ATE vs LiDAR | alignment | RPE 1 m | tier 2 |
|---|---|---|---|---|
| backbone (`odometry.tum`) | 372–481 mm | se3 **to the LiDAR** | ~94 mm | 350–549 mm |
| **V1** | **372 / 377 / 411 mm** | se3 **to the LiDAR** | ~94 mm | identical to the backbone, to the digit |
| **V2c** | **228 / 254 / 254 / 291 mm** | **none** | ~154 mm | 0.7 / 3.8–4.7 mm (consumed: self-agreement) |

The anchors buy ~12 cm *and* remove the oracle alignment. They cost local
smoothness: RPE at 1 m goes 94 → 154 mm, because the trajectory is bent to
satisfy global constraints. Report both.

**CERTIFIED 2026-09-20.** Four backbone runs, plus a sigma sweep and both
leave-one-board-out builds, on `mobile_1`:

| variant | ATE | RPE 1 m | moved | independent evidence |
|---|---|---|---|---|
| V1 (odometry only), se3 **to the LiDAR** | 372 / 377 / 411 mm | 91–96 mm | 0 | boards 350–549 mm |
| **V2c** (both boards), **unaligned** | **228 / 254 / 254 / 291 mm** | 132–176 mm | 336–512 mm | **LiDAR ATE** |
| V2c, σ ×0.5 | 290.4 mm | 175 mm | 340 mm | — |
| V2c, σ ×2 | 290.1 mm | 178 mm | 332 mm | — |
| V2c, `anchor` only | 1163 mm | 94 mm | **7261 mm** ⚠ | `rs_anchor` 1719 mm |
| V2c, `rs_anchor` only | 771 mm | 95 mm | 178 mm | `anchor` 1201 mm |

Three readings.

**The sigma model does not carry the result.** A 4× range of declared per-edge
sigma moves the ATE by 1.1 mm (290.4 / 291.2 / 290.1 on one run). So the fact
that the sigmas are derived from an RPE scored against the LiDAR does not make
the LiDAR ATE circular — that ATE is independent certification.

**Leave-one-board-out cannot certify a two-board construction.** With n=2,
holding one out leaves a lever, and the 1201 / 1719 mm independent residuals
measure the *one-board* construction rather than the shipped one. This is a
property of having two boards, not a defect to fix, and it is why the LiDAR is
the certification of record.

**The RPE column is the mechanism.** Both single-board builds sit at 94 mm,
exactly V1's value, because one board transports the trajectory rigidly
without bending it. The full build bends it — that bending is where the 12 cm
comes from, and local smoothness is what it costs.

Two numbers matter:

1. **V1 must roughly reproduce the backbone's own error.** If it does not, the
   graph is wired wrong and nothing downstream is trustworthy. This is the
   sanity gate, and it should be run before V2c is even written.
2. **V2c − V1 is the answer to "what do the anchors buy, in centimetres."**
   That is the deliverable.

Then Stage 3's covariate model, which is the part MoCap2GT does not have: fit
error against observables that exist without a reference — dropout fraction,
longest starved stretch, anchor visibility, time since the last absolute
factor — and validate leave-one-segment-out. A scalar accuracy figure does not
transfer to `mobile_2`. A function does.

---

## 8. Novelty — state it precisely

**Not novel:** the estimator. Factor graph + preintegration + absolute-pose
factors + a linear initialiser is four papers deep. Framing the contribution as
"a pose graph that fuses anchors and odometry" invites exactly those citations
as prior art.

**Novel:**

1. **Pseudo-GT with no motion-capture volume.** Every method above needs a
   Vicon lab. This needs three printed boards and a survey, and the absolute
   evidence is *sparse and intermittent* rather than continuous at 100 Hz. The
   sparsity is the hard part, not an inconvenience — it is why §2.1 exists.
2. **An error model that transfers.** MoCap2GT reports its accuracy on its own
   data and offers no way to state uncertainty on a sequence with no ground
   truth. Stage 3 does.
3. **Transfer to an agent with no reference at all**, with the board residuals
   as a falsification test that can come back negative.
4. **Multi-agent anchoring** — absent from this literature.

Claim the regime and the certification. "High-precision GT estimator" is taken.
