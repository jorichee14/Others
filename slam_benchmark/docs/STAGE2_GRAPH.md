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

Covariance: from the front-end's own reported covariance where it exists, else
fitted from the RPE-1m statistics already measured in tier 1. **Declared in
the config, never tuned per run** (rule 7).

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
| **V1** backbone only | `r_O` | **ready** — the no-anchor floor |
| **V2a** + IMU | `r_O + r_I` | **blocked** — noise model (§2.3) |
| **V2b** + radar ego-velocity | `r_O + r_V` | feasible; radar1/radar2 at 15.8 Hz, Doppler confirmed (I12), sign convention open. mobile_1 only |
| **V2c** + board anchors | `r_O + r_B` | **ready** — 4 pairs, detector at 8 mm |
| **V2d** + infrastructure | `r_O + r_B + r_N` | **blocked** — I9: `infra_1`'s pose has no stated uncertainty anywhere in the dataset, and an anchored fix cannot be better than its anchor |
| **V2e** + inter-agent | all | **unavailable** — needs simultaneous co-observation; the agents swap corners rather than share one. `docs/PLAN.md`'s next-session list already says to route them through a shared corridor |

**Two rungs are buildable, one more is plausible, three are not.** Report it
that way. Three unpriceable evidence classes each with a specific measured
reason is a finding about what a dataset must contain, which is a dataset
paper's business (rule 8).

---

## 7. Certification

V1 and V2c are both run on **mobile_1** and scored against the held-out LiDAR
reference — the instrument that never feeds the construction. Two numbers
matter:

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
