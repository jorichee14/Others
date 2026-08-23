# Three-radar extrinsic calibration on the Ouster rig — full report

**Rig:** Ouster OS1 lidar, ZED stereo camera, Arducam, three TI IWR6843ISK radars
**Dates:** 2026-07-15 (ChArUco attempts) → 2026-08-19 (radar↔lidar, final)
**Deliverable:** `T_camera_radarN` for all three radars, plus `T_os_lidar_radarN`
**Status:** final. All three radars solved, cross-checked and published.

---

## 1. What this is and why the method changed

The goal is the rigid transform between each radar and the camera it will be
fused with, good enough that a radar return projects onto the right object in
the image.

### 1.1 The first method (radar ↔ ChArUco camera board) and why it was dropped

The July sessions solved radar→camera directly, using a corner reflector taped
to a ChArUco board: the camera got the board pose from the markers, the radar
got the reflector, and the pair was matched. Two things broke it.

* **The board has to face the camera and the reflector has to face the radar.**
  Those two constraints fight each other. A trihedral returns strongly only
  inside a narrow cone; a ChArUco board detects well only near frontal. The
  usable overlap is a small pose set, and a small pose set is exactly what a
  radar with one weak angular axis cannot afford.
* **Camera pose error enters the solve twice** — once through the board's own
  6-DoF estimate (which is soft in depth for a small board) and once through
  the reflector-to-board offset, which must be measured by hand.

### 1.2 The method that was used (radar ↔ lidar, then compose)

Replace the camera with the lidar as the reference sensor:

1. Solve `T_lidar_radarN` directly — the lidar sees the trihedral apex to a
   few millimetres, at any orientation, anywhere in a 360° ring.
2. Compose with the already-trusted lidar↔camera extrinsic:

   ```
   T_cam_radar = T_cam_lidar · T_lidar_radar
   ```

This is strictly better because the lidar is a far better reference target
locator than the camera, the capture geometry is unconstrained (the reflector
only has to face the radar), and the lidar↔camera transform is solved once,
carefully, by a dedicated method (GLIM for the ZED, a separate solve for the
Arducam) rather than re-estimated per capture.

---

## 2. The solver

`sessions/solve_radar_lidar.py` and the live node `radar_lidar_calib.py` share
one estimator.

### 2.1 Measurement-space maximum likelihood, not Cartesian Kabsch

The naive approach converts the radar detection to XYZ and runs Kabsch
(closed-form point-set alignment). That is **biased** here, because Kabsch
assumes isotropic noise and this radar's noise is anything but:

| axis | 1σ used |
|---|---|
| range | 5 cm |
| azimuth | 3° |
| elevation | 8° |

An 8° elevation error at 4 m is 56 cm of cross-range displacement. In Cartesian
space that swamps the 5 cm range measurement, so an isotropic fit throws away
the one axis the sensor is actually good at.

Instead the residual is formed **in the radar's own measurement space**:

```
predict  p_radar = Rᵀ (p_lidar − t)
convert  (r, az, el) = cart_to_raz(p_radar)
residual [(r−r̂)/σ_r, (az−âz)/σ_az, (el−êl)/σ_el]
```

Each axis is weighted by what it is worth. A dead elevation channel then
contributes almost nothing instead of contributing garbage with full weight,
and the solve leans on range and azimuth, which is exactly right.

### 2.2 Robustness

* **Huber loss**, `f_scale = 1.5`, so a moderately bad capture is down-weighted
  rather than deleted.
* **Iterative sigma-gated rejection**, run to convergence:
  * `reject_sigma = 4.0` — RMS across the three axes, divided by √3.
  * `reject_axis_sigma = 3.5` — per-axis maximum. This second gate is what
    catches elevation *mirror ghosts*, where a multipath return lands at the
    negative of the true elevation; those pass an RMS gate but fail a per-axis
    one.

### 2.3 Independent estimators used as cross-checks

* **Range-only position estimator.** For a target at lidar direction `û`,
  `lidar_r − radar_r ≈ t·û` to first order. Linear least squares over all
  captures recovers `t` using **no radar angles and no rotation at all**. When
  a translation component is disputed, this is the arbiter — it cannot be
  contaminated by a bad angle channel or a rotation-translation trade.
* **Channel slope diagnostic.** Regress the reported angle against the angle
  the solved extrinsic predicts. Slope ≈ +1.00 means the channel carries real
  information. Slope ≈ 0 means the channel is emitting a constant.
* **Split-half test.** Split one dataset in half at random, solve each half,
  compare the two translations. This is the honest uncertainty number.
* **χ² observability test.** Freeze the inlier set, disable rejection, force a
  translation axis N cm off, and read the total chi-square penalty. >25 is
  decisive, 9–25 weak, <9 means the data genuinely cannot tell.

---

## 3. Results

All three solved on the Ouster, 2026-08-19. `n` = captures, `inl` = inliers
after rejection, `rms` = normalised residual, `loo` = leave-one-out,
`cond` = condition number of the normal equations,
`split-half` = translation disagreement between two random halves.

### 3.1 radar1 — ZED, upright mount

```
n=54  inl=46  rms=1.29  loo=1.36  cond=5.2  split-half = 8.4 cm

T_os_lidar_radar1   t = [ 0.033393,  0.140555, -0.168519]
                    q = [ 0.123377,  0.001261,  0.992298, -0.010988]

T_zed_radar1        t = [ 0.207505,  0.076150, -0.108883]
                    q = [-0.551338,  0.560743, -0.443155, -0.430356]

camera-frame 1σ rot = 4.29 / 0.91 / 2.98 deg
camera-frame 1σ t   = 19.3 / 40.9 / 8.7 mm
lidar-frame  1σ t   =  8.8 / 19.3 / 40.9 mm
residual bias       = +16.8 / -49.3 / +12.6 mm
residual RMS        =  185  /  352  /  102  mm
channels            az = 0.99   el = 0.14
azimuth spread      81 deg ; near captures 13 ; range 0.79 – 4.42 m
range fit           a = 0.9982   b = +5.9 mm
```

### 3.2 radar2 — ZED, rolled 90°

```
n=42  inl=33  rms=1.61  loo=1.78  cond=3.8  split-half = 13.1 cm

T_os_lidar_radar2   t = [ 0.036705, -0.120823, -0.139143]
                    q = [ 0.013827,  0.736618,  0.675767,  0.023290]

T_zed_radar2        t = [-0.053850,  0.046553, -0.112039]
                    q = [ 0.724722,  0.038269,  0.687594, -0.022953]

camera-frame 1σ rot = 3.43 / 3.87 / 2.95 deg
camera-frame 1σ t   = 37.5 / 36.4 / 10.1 mm
lidar-frame  1σ t   = 10.1 / 37.5 / 36.4 mm
residual bias       =  -7.4 / +12.3 / +24.3 mm
residual RMS        =  513  /  197  /   68  mm
channels            az = 1.09   el = -0.07
azimuth spread      70 deg ; near captures 15 ; range 0.79 – 3.78 m
range fit           a = 1.0191   b = -36.5 mm
```

### 3.3 radar3 — Arducam, inverted mount

```
n=54  inl=44  rms=1.31  loo=1.36  cond=5.3  split-half = 8.3 cm

T_os_lidar_radar3   t = [ 0.041839, -0.101587,  0.161252]
                    q = [-0.007406,  0.986026,  0.007861,  0.166243]

T_arducam_radar3    t = [ 0.017773, -0.208023, -0.155788]
                    q = [ 0.571431,  0.563856,  0.432969, -0.409965]

camera-frame 1σ rot = 1.90 / 3.33 / 1.27 deg
camera-frame 1σ t   = 21.2 / 40.6 / 10.2 mm
lidar-frame  1σ t   = 11.2 / 21.3 / 40.3 mm
residual bias       =  -9.4 / +51.0 / +30.0 mm
residual RMS        =  192  /  366  /   91  mm
channels            az = 0.89   el = 0.06
azimuth spread      71 deg ; near captures 11 ; range 0.94 – 4.26 m
range fit           a = 1.0146   b = -26.1 mm
```

### 3.4 Fusion-node parameters (camera→radar, ready to paste)

```
r1_t_xyz:="[0.2075,0.0762,-0.1089]"   r1_quat_xyzw:="[-0.5513,0.5607,-0.4432,-0.4304]"   # ZED
r2_t_xyz:="[-0.0538,0.0466,-0.1120]"  r2_quat_xyzw:="[0.7247,0.0383,0.6876,-0.0230]"     # ZED
r3_t_xyz:="[0.0178,-0.2080,-0.1558]"  r3_quat_xyzw:="[0.5714,0.5639,0.4330,-0.4100]"     # Arducam
```

### 3.5 Camera transforms used in the composition

Both used exactly as published — no inverse, no sign flip.

```
os_lidar → zed_left_camera_optical_frame     (GLIM)
  t = [-0.074928, -0.066971, -0.091627]
  q = [-0.497829, -0.498035,  0.501789,  0.502329]

os_lidar → arducam_optical_frame
  t = [-0.105424, -0.121850, -0.052669]
  q = [ 0.514446,  0.505671, -0.492421, -0.486994]
  rpy = -92.392 / 0.810 / 89.792 deg
```

---

## 4. The central finding: the elevation channel is dead on all three radars

This is the single most important result of the campaign, and it shaped every
decision after it was found.

### 4.1 The evidence

Regressing the radar's reported `z` against the target's true range and true
height across all captures gives

```
z_radar = -0.208 · range + 0.171 · height + 0.066
```

The height coefficient **should be ≈ 1.0**. It is 0.17. The range coefficient
should be ≈ 0; it is −0.208, which is the signature of a **fixed cone**: the
radar reports a near-constant elevation angle of about −11.5° regardless of
where the target actually is, so the reported `z` just tracks range.

The channel slope diagnostic says the same thing three times:

| radar | azimuth slope | elevation slope |
|---|---|---|
| radar1 | +0.99 | **+0.14** |
| radar2 | +1.09 | **−0.07** |
| radar3 | +0.89 | **+0.06** |

Azimuth is essentially perfect on all three. Elevation carries no information
on any of them.

### 4.2 What was ruled out

Three different physical mountings, three different brackets, same failure.
That already points at the chip or the config rather than any one bracket.
Specifically ruled out:

* **TX antenna masking** — `channelCfg 15 7 0` enables all three TX.
* **Antenna geometry table** — `antGeometry1` has two distinct rows, so the
  elevation pair is declared.
* **FoV clipping** — `fovCfg -1 60.0 20.0` allows ±20° elevation, wider than
  anything measured.
* **Quantisation** — the reported values vary continuously, they are just
  uncorrelated with truth.
* **RX phase calibration** — `compRangeBiasAndRxChanPhase` verified consistent
  by the user.

### 4.3 What is still open

Every capture in this campaign came from the demo's **static** angle chain
(`staticRangeAngleCfg`). A trihedral on a tripod is a zero-Doppler target, and
the capture node gates on `max_abs_doppler`, so the dynamic chain was never
exercised. If the elevation estimate is only computed in the dynamic path, that
would explain all of it. **This is the first thing to test if elevation is ever
needed.** Nothing in this campaign proves the hardware cannot do elevation — it
proves that in this configuration, on static targets, it does not.

### 4.4 Why it matters for the extrinsic

A rotation δ about axis `k` displaces a point by `δ (k × p)`. It is observable
only if that displacement lands on an axis the radar can actually measure.
So:

* **yaw** comes from azimuth spread — well constrained, azimuth works.
* **pitch and roll** come from elevation spread — poorly constrained, because
  elevation does not work.
* **translation vs rotation** is separated by range spread — which is why
  captures very close in (< 1.5 m) matter far more than their count suggests.

And because cross-range error is `r · sin(σ_angle)`, an angular error turns
into a distance error that grows with range. That is why near captures were
chased so hard.

---

## 5. Complementary mounting geometry

The three radars are not mounted alike, and that turns out to be useful:

| radar | mount | dead axis in the world |
|---|---|---|
| radar1 | upright | **vertical** |
| radar2 | rolled 90° | **horizontal** |
| radar3 | inverted | **vertical** |

Because radar2 is rolled, its dead elevation channel points sideways, so it is
blind in the horizontal direction while radar1 and radar3 are blind vertically.
No single radar constrains all three translation axes well, but **the pair
does** — radar2 pins the vertical that radar1/radar3 cannot, and vice versa.

This also changed the capture guidance mid-campaign: telling the operator to
move "high/low" means something different for radar2 than for radar1. The live
HUD was taught to work this out from the current extrinsic and phrase its hints
in world terms rather than radar-frame terms.

---

## 6. Observability, measured rather than assumed

Forcing each translation axis 14 cm off a **frozen** inlier set (rejection
disabled) and reading the total χ² penalty:

| | x (fwd/back) | y (lateral) | z (vertical) |
|---|---|---|---|
| **radar1** | 27 — decisive | 50 — decisive | 12 — weak |
| **radar2** | 36 — decisive | 14 — weak | 20 — weak |
| **radar3** | 68 — decisive | 29 — decisive | 8 — floats |

Read this as the honest statement of what each dataset actually pins down.
radar3's vertical is essentially unconstrained by the data; radar1's and
radar2's are weak. Depth (x) is decisive everywhere, which matters in §8.

**Freezing the inlier set is essential.** The first version of this test let
rejection re-run, so forcing an axis off simply shrank the inlier set until χ²
came back down — radar2's x showed a *negative* penalty of −80, making a
well-constrained axis look free. That was a bug in the test, not a finding.

---

## 7. Uncertainty: split-half beats the covariance

The covariance-derived 1σ from the solver is **optimistic**, badly so on the
dead axes. The split-half disagreements

| radar | split-half |
|---|---|
| radar1 | 8.4 cm |
| radar2 | 13.1 cm |
| radar3 | 8.3 cm |

are several times the reported per-axis 1σ. Use split-half as the number to
quote. The covariance assumes the residuals are independent zero-mean Gaussian
with the declared sigmas; when one channel is emitting a near-constant, that
assumption fails in exactly the direction that makes the answer look better
than it is.

**Related correction made during the campaign.** An early test of "does more
data settle this?" used residual **RMS**, which dilutes with N and therefore
always looks like it improves. Redone as **total χ²** — the correct
likelihood-ratio quantity — the picture was honest.

A second over-claim was made and withdrawn: *"the geometry is degenerate, more
poses cannot help."* Simulation showed clean 1/√N convergence. It is not an
impossibility, it is an **exchange rate** — roughly 1900 poses, about 14 hours
of capture, to match what a 3 cm tape measurement gives for free. That is why
tape was used as a cross-check on axes the data leaves weak, and why it was
*not* used to override an axis the data pins decisively.

---

## 8. The radar3 depth investigation — settled in favour of the point cloud

This consumed the most effort and is worth recording in full.

**The disagreement.** The solve puts radar3 at `t_x = +4.2 cm` (behind the
lidar) while the Arducam sits at −10.5 cm (in front of it), giving a composed
camera→radar depth of **15.6 cm**. Tape measurements suggested 1–3 cm, later
revised to about 6 cm. A 10 cm discrepancy on the axis the radar measures best
demanded an explanation.

**Every independent check agreed with the point cloud:**

* **Range-only estimator** (no radar angles, no rotation): `+3.97 ± 0.89 cm`.
  That is **8.9σ** away from a hypothetical −4 cm.
* **Bootstrap**, 2000 resamples: **0 of 2000** fell below −4 cm.
* **χ²** rules out anything at or below zero.
* **Nine rotation-flip restarts** — deliberately seeded at wrong rotations to
  see if a mirror solution existed — all converged to the identical answer.
* **radar1 and radar2 solve independently** to +3.3 cm and +3.7 cm on the same
  axis, with completely separate datasets.
* **Raw range differences** across the three radars: +2.7 / +4.1 / +4.3 cm.

**Decision:** keep the point-cloud answer, 15.6 cm, transform published
unflipped. The user's call, and the right one — depth is the decisively
observable axis (χ² = 68 for radar3) and six independent lines of evidence
converge on it.

**A mistake made along the way.** The Arducam x sign was flipped once, on the
basis of a verbal "10 cm behind" without pinning down the reference direction.
The camera and the lidar face opposite ways, so "behind" is ambiguous until
the frame is named. The flip was reverted; the published transform is unchanged
from the one that was verified by measurement.

**A second correction.** The claim *"depth is the best axis, it cannot be 14 cm
wrong"* was made citing the 11.9 mm covariance. Testing showed that forcing
14 cm cost only 0.07σ by that metric — the covariance was optimistic (§7). The
conclusion survived, but on the strength of the range-only estimator and the
χ² test, not the covariance.

---

## 9. Outlier and round rejection

**radar2 rounds 1 and 2 were excluded** from the final solve. The reasons:

* They land 22–30 cm away from rounds 3 and 4.
* Round 2 missed the tape-measured radar1↔radar2 separation (29.6 cm) by 24 cm
  — *while having the best residual in the entire project*.

That last point is the standing lesson of the campaign: **residual and
leave-one-out do not detect this failure mode.** A dataset whose captures all
sit in a degenerate arrangement will fit itself beautifully and still be wrong.
Judge by split-half agreement and by an independent physical measurement, not
by how small the residual is.

**A wrong instruction was given and later corrected.** After radar2 round 1 the
advice was to "stay inside ±10° horizontally". That reasoning was backwards —
horizontal spread is precisely what breaks the planar degeneracy for a
90°-rolled radar. Those captures were being rejected because the dead channel
mispredicts them, not because the poses were bad.

---

## 10. Centroid vs three-plane apex — why centroid

The lidar apex can be estimated two ways: the **centroid** of the reflector
points, or `planes3`, a three-plane RANSAC intersection that recovers the
geometric corner. `planes3` is the more "correct" corner. Centroid was used
anyway, deliberately:

1. **Calibration needs a consistent reference, not the true corner.** Any fixed
   offset between the estimator's point and the physical apex is common-mode
   and cancels out of the extrinsic.
2. **The radar's own return centre is not the corner either.** Matching a
   geometric corner in the lidar to a scattering centroid in the radar
   introduces a bias rather than removing one.
3. **Repeatability, which is what actually matters:** lidar apex repeatability
   is **2.3 mm**; radar range noise is about **50 mm**. The lidar side is 20×
   better either way — the estimator choice is far below the noise floor.
4. **`planes3` was unavailable on 129 of 129 captures.** In practice the choice
   did not exist.
5. **Consistency:** centroid was already used for the ZED calibration, so
   keeping it makes the whole chain share one convention.

---

## 11. Tooling built during the campaign

### `radar_lidar_calib.py` — the live capture node

* **Coverage HUD.** Not just progress bars — a spatial **coverage map**. The
  azimuth×elevation plane is split into a 3×3 grid (`AZ_EDGES` ±20/±60°,
  `EL_EDGES` ±10/±40°) plus three range bands, and the overlay shades the cells
  that still have no capture in them. Targets: 1.5 m range spread, 60° azimuth,
  30° elevation, balance on the thinner side of boresight, ≥6 captures inside
  1.5 m, ≥8 of the 9 cells filled.
* **Direction-aware hints.** Because radar2's elevation axis is physically
  horizontal, the hint text is derived from the current extrinsic: if the
  radar's +Z is within 45° of lidar-up, elevation reads "high/low", otherwise
  "left/right".
* **Capture race fix.** The capture window used to close as soon as the radar
  quota was met — but the radar publishes far faster than the lidar spins, so
  captures were being refused for "too few lidar detections". `_maybe_finish()`
  now requires **both** quotas, with a new `capture_frames_lidar` parameter
  (default 3) and a timeout swept from the heartbeat.
* **Live channel diagnostic.** Prints the az/el slopes after every solve, with
  a `!!` flag when a slope is more than 0.3 from unity. This is what surfaced
  the dead elevation channel in the first place.
* **TF-sourced camera transform** (`camera_transform_from_tf`), looked up once
  from `lidar_frame → camera_frame` and retried quietly.
* **In-node rectification** (`rectify_image`, `rectify_alpha`) via
  `initUndistortRectifyMap` / `remap`, with `K`/`D` replaced by the new
  camera matrix and zero distortion.
* **Display fixes:** HUD drawn after the `debug_scale` resize, panel opacity
  raised 0.45 → 0.78, range strip spans the range actually in use rather than
  `lidar_max_range`.
* **Profile overrides:** `main(default_overrides=None)` so thin wrappers can
  preset parameters.

### `radar_lidar_calib_arducam.py`

Thin wrapper presetting only the camera-side parameters — `/arducam/image_raw`,
`/arducam/camera_info`, `arducam_optical_frame`, rectification on with
`alpha = 0.0`. Everything else is inherited.

### `sessions/solve_radar_lidar.py`

Offline re-solver over a saved pose JSON. Supports per-axis translation priors
(`t_prior` / `t_sigma` as vectors), rotation priors, the coverage table with
the `[X]/[ ]` cell grid, and `--cam-quat` / `--cam-xyz` to compose straight to
the camera frame. *Note:* argparse needs the `--cam-quat=...` form when the
value starts with a minus sign.

### `sessions/plan_captures.py`

Turns an extrinsic plus the lidar half-FoV into concrete tripod positions —
forward / side / height relative to the radar — bisecting for the height band
that stays inside the lidar FoV.

### `sessions/plot_extrinsics.py`, `sessions/plot_cam_radar.py`

Diagram generators. `plot_cam_radar.py` draws three auto-fitting orthographic
views in the camera frame.

---

## 12. Errors made and corrected during the campaign

Recorded because several of them are easy to repeat.

1. **"The radar is pitched 13° down."** Wrong — the boresight z component is
   **+0.22**, so it is pitched **up** 12.9°. Cause: `plot_extrinsics.py` took
   `abs(R_lr[2,0])` and the direction word was written by hand. Fixed by
   deriving the word from the sign. This mattered practically: the el = 0 line
   *rises* 22 cm per metre, so positive elevation is only reachable close in.
2. **RMS used where χ² was needed** (§7).
3. **Observability test let rejection re-run** (§6).
4. **"The geometry is degenerate, more poses cannot help"** — withdrawn, it is
   an exchange rate (§7).
5. **A proposed static-vs-dynamic test was unusable** — it said to "watch the
   radar z in the log", but the log prints range, not z.
6. **"Stay inside ±10° horizontally"** — backwards (§9).
7. **The Arducam x flip** — reverted (§8).
8. **radar3's +76 mm vertical bias attributed to the centroid estimator** —
   over-attributed; it cannot be separated from the dead elevation channel.
9. **"Depth cannot be 14 cm wrong, the covariance says 11.9 mm"** — the
   covariance was optimistic (§8).
10. **`ndarray.ptp()`** removed in numpy 2.x — switched to `np.ptp(...)`.

---

## 13. Open items

* **The dynamic angle chain has never been exercised** (§4.3). A trihedral on a
  tripod is zero-Doppler and the node gates on `max_abs_doppler`, so if
  elevation is only computed in the dynamic path, that alone explains the dead
  channel. Highest-value next test by a wide margin.
* **Vertical translation stays weak on all three** — radar3's especially
  (χ² = 8, essentially floating). If the vertical ever needs to be better than
  ~10 cm, it will have to come from tape or from a fixed dynamic elevation
  channel, not from more static captures.
* **radar2 has the worst split-half** (13.1 cm) on 33 inliers from two rounds.
  Another round would help it more than it would help the other two.

---

## 14. Data files

| file | contents |
|---|---|
| `2026-08-19_ouster_radar1_poses.json` | 54 captures |
| `2026-08-19_ouster_radar1_lidar.md` | radar1 write-up |
| `2026-08-19_ouster_radar2_poses.json` | 42 captures (rounds 3 + 4) |
| `2026-08-19_ouster_radar2.md` | radar2 write-up |
| `2026-08-19_ouster_radar3_poses.json` | 54 captures (three rounds) |
| `2026-08-19_ouster_radar3.md` | radar3 write-up |
| `2026-08-19_ouster_radar1_extrinsics.svg` | lidar-frame diagram |
| `2026-08-19_cam_radar.svg` | camera-frame diagram |
