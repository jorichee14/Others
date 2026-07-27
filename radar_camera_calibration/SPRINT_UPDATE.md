# Sprint Update — Radar ↔ Camera Extrinsic Calibration

**Focus:** extrinsic calibration between a TI **IWR6843ISK** mmWave radar and a
**ZED** camera, using a **ChArUco board + trihedral corner reflector** rig.
**Tool:** `radar_camera_calib.py` (packaged as `wicoms_utils/radar_camera_calibration`).

> Status: tool validated and hardened through several live sessions. Live
> calibration now **captures cleanly** but has **not yet converged** to a
> trustworthy extrinsic — blocked mainly by pose diversity (azimuth spread) and
> hardware (small board, low camera resolution, large reflector).

---

## Goal

Recover `T_cam_radar` (radar pose in the camera optical frame) so radar
detections can be projected into the camera / fused. One rigid target carries
both cues: the camera localizes the **ChArUco board**, the radar localizes the
**corner reflector** rigidly fixed to it at a fixed offset.

Governing equation, per pose `i`:
```
R_board,i · a + t_board,i  =  R · q_i + t
```
`(R_board,i, t_board,i)` = board pose from ChArUco, `q_i` = radar-measured
reflector point, `a` = apex offset (board frame), `(R,t)` = the extrinsic.
Solve 9 unknowns `(R, t, a)`; **board tilt** makes `a` observable, **radar-point
spread** makes `(R,t)` observable.

---

## Hardware / software context

| Item | Value |
|---|---|
| Radar | IWR6843ISK, **3D People-Counting** firmware (group tracker) |
| Radar topic | `/radar1/radar/points_all` (also `points_static`, `points_dynamic`), ~10 Hz |
| Radar fields | `x, y, z, doppler, intensity` — **SNR field is `intensity`** |
| Radar frame | `radar1_link`; **X=forward, Y=lateral, Z=vertical**; range = ‖xyz‖ |
| Radar FoV | azimuth **±60°**, elevation **±20°** |
| Camera | ZED left, `/zed/zed_node/left/…`; optical frame X=right, Y=down, Z=forward |
| Camera res | **960×540** in sessions — too low; markers die past ~0.4–1 m |
| Board | small: 20 mm squares / 15 mm markers, DICT_4X4_50 (hand-held) |
| Reflector | large gold trihedral, ~15 cm |

Frames do **not** share a convention (radar forward = X, camera forward = Z), so
the extrinsic `R` is a ~90°-ish rotation plus the mounting. Verified on a live
correspondence: radar range ≈ camera range (same physical point), and the
mounting is "same facing, 180° rolled about boresight" → `prior_rpy_deg ≈ [-90,-90,0]`.

---

## Method (final)

**Kabsch init → measurement-space maximum-likelihood** in `(range, azimuth,
elevation)` weighted by real radar σ (range precise, angle poor, cross-range
error grows with range) → **Huber robust loss + iterative σ-gated outlier
rejection** → **joint apex-offset estimation (MAP)** toward the measured value →
**per-DOF covariance / observability** reporting. 2-D radar auto-detected.
Optional **extrinsic prior (MAP)** on `(R,t)`.

Monte-Carlo (offline): ML ≈ 1.0°/38 mm vs Kabsch ≈ 1.7°/120 mm; robust loss
holds ~1° with gross outliers (L2 → 50°); offset recoverable from scratch to
~15 mm given close-range high-tilt poses.

---

## What worked ✅

- **Measurement-space ML solver**, joint offset (MAP), robust rejection,
  covariance/observability, LOO-CV, planar/2-D detection, range diagnostic.
- **SNR selection via `intensity`** — highest-SNR return = the trihedral.
- **Camera-range gate** ("highest SNR around the board") — gate radar points to
  the camera's board distance `|p_cam|` (rotation-invariant), then max SNR.
- **Gate fallback** — range gate always applied; predicted (prior/solve) gate
  only *tightens*, with fallback so a wrong prior never rejects everything.
- **Stage-by-stage gate diagnostic** — `[gate] … total N → range k → bg k →
  cam±m k → final k`, plus `best snr X < min_snr` — pinpoints which gate drops
  the reflector.
- **Split stability** (`stable_std` cam vs `stable_std_radar`) + **min_baseline
  skip message** ("steady but only X cm from last capture — MOVE").
- **Continuous (sweep) capture mode** — no stillness required; capture every
  `min_baseline` of movement. Made collection practical.
- **`min_snr` strict-capture gate** — rejects weak (mis-associable) returns; the
  single biggest quality lever once captures were flowing.
- **Priors** (offset + extrinsic, each a center+sigma pair) — stabilise
  under-constrained DOFs; extrinsic prior also seeds the radar search from frame 1.
- **Doppler gates** (`min/max_abs_doppler`) + `points_dynamic` support.
- **Display** — native window + debug topic: board axes, apex reticle,
  **whole radar cloud projected via solved-or-prior extrinsic (yellow)**, and the
  **selected reflector point in magenta with range** — so you can *see* whether
  the radar lands on the reflector before any solve.

---

## What didn't work / problems found 🔧

| Symptom | Root cause | Fix |
|---|---|---|
| `InvalidParameterType BOOL` | `-p pc_field_y:=y` → YAML parses `y` as `true` | drop `pc_field_x/y/z` (defaults) or quote `:="'y'"` |
| No SNR selection | field is `intensity`, not `snr` | `pc_field_snr:=intensity` |
| Never captured | 10 mm stability gate on jittery radar | `stable_std_radar` (0.08) |
| `no board n=0..7` | 15 mm markers unreadable past ~0.4 m at 960×540; motion blur | HD1080/HD2K, `min_corners:=6`, **bigger board** |
| Picking far clutter (4–7 m) | global SNR-max | camera-range gate |
| Reflector SNR 300→20 | trihedral aimed away (~±30° usable beam) | re-aim at radar; tilt ≤ ±30° |
| **Radar point never moved** (14 poses) | **reflector not attached to board** (sitting on desk) | **rigidly bolt reflector to board** |
| Prior on → `no gated radar return` (all frames) | extrinsic prior replaced range gate; imperfect prior misses reflector | range gate always-on; prior only tightens + fallback |
| `still? True` but no capture #2 | rig not moved `min_baseline` from last | skip message added; **move ≥10 cm** |
| Dynamic sweep → SNR 5–100, 3-D 200 mm | `min_abs_doppler` grabbed weak moving clutter (hand/body), `min_snr:=0` | drop doppler filter, `min_snr:=100` |
| Offset pulls to Y≈0.30 vs measured 0.23 | large trihedral's **radar phase center ~7 cm behind geometric apex** | loosen `offset_prior_sigma_m:=0.05`, let it refine |
| **Rotation reads as a "huge error"** | **gimbal lock**: the extrinsic sits at pitch −90, the singularity of the `xyz` euler convention. A **1° real change rewrites the printed triple by 90°** (`[-90,-90,0]`→`[0,-89,-90]`); scipy zeroes the third angle. Pasting a printed rpy back in as `prior_rpy_deg` (as the README used to advise) then makes it a **real** error | report quat + axis-angle + **radar→camera axis map**; compare rotations geodesically only; `prior_quat_xyzw` param + a paste-back line printed every solve; warn when within 8° of the lock |
| Wrong prior invisible to every metric | a prior 35° off still fits at **0.53 σ** while dragging the answer 15°; the prior was also the *only* initialisation, so the data never contradicted it | solve **twice** (data-only from Kabsch, and with the prior, multi-start) and report the **geodesic gap**; verdict check at 5° |
| `1σ` looks tight on an under-constrained rotation | covariance included the prior rows, so the prior's own width read as data-derived confidence (6.8°→5.3° from adding a prior alone) | report **`1σ data`** with prior rows excluded, and inflate by the residual |
| `LOO ≈ 3σ` while residual ≈ 2.3σ | LOO refit **without** the priors — a different estimator, so folds scattered | LOO now carries the same priors |
| `~/solve` with exactly 3 captures | `np.sort(pn)[3]` on a length-3 array | **`IndexError` crash** — clamp the inlier floor to `len(P)` |
| Board rotation noisy / occasionally flipped | **planar PnP two-fold ambiguity**, unguarded: measured ~3.9° median and **7/80 flips >15°** at 2 m with the 160 mm board; each degree is multiplied by the 25 cm apex offset | `solvePnPGeneric`+IPPE, reject when the two hypotheses are within `pnp_ambiguity_ratio` (removes all flips) |
| 3-D radar silently demoted to 2-D | `use_el` keyed on `std(|z|/r)`, which measures **pose diversity**, not whether the radar reports elevation | key on `z ≡ 0`; cost of the old behaviour measured at ~2° rotation and a doubled 1σ |
| `reproj` always ≫ 20 px | 20 px is unrealistic for close-range radar | judge by **3-D mm** + overlay; `val_pass_reproj_px:=200` |

---

## Key findings / physics 📐

1. **Radar noise is anisotropic** (range precise, angle poor, cross-range ∝
   range) → measurement-space weighting, not isotropic XYZ.
2. **IWR6843ISK is 3-D but elevation is weak** (±20° FoV) → large `sigma_el_deg`.
3. **The rig must be ONE rigid body** — the single biggest failure was the
   reflector not fixed to the board (radar tracked a static object while the
   camera tracked the moving board). No solver fixes a broken rig.
4. **A corner reflector is directional** (~±30° beam) — tilting the board for
   diversity dims it. Keep it aimed; tilt ≤ ±30°. This capped offset
   observability (which wants high tilt) — a real tension.
5. **A large reflector's radar phase center ≠ its geometric apex** (~7 cm behind)
   → the solver correctly refines the offset away from the measured value; a
   smaller / point-like reflector would tighten correspondences.
6. **Two independent observability needs:** board **rotation** → offset; radar
   **point spread (range/az/el)** → extrinsic. Both must be satisfied.
7. **Camera-side is the bottleneck hardware** here: 960×540 + 15 mm markers →
   frequent dropouts and no range diversity. Higher res + a bigger board would
   unlock far poses and the translation/rotation accuracy.
8. **Translation is well-observed; rotation is not** — rotation needs wide
   **azimuth** spread; clustered poses leave it under-constrained (and a bad
   basin can hide behind a small local `1σ`). Extrinsic prior guards against this.
9. **A ~90° extrinsic cannot be read, compared, or stored as roll/pitch/yaw.**
   It lands on the euler singularity, where the triple is neither unique nor
   continuous. Use the quaternion to store it, the **axis map** to read it, and
   the **geodesic angle** to compare it. Most of what looked like a "huge
   rotation error" in the live sessions was this representation blowing up.
10. **No metric the fit produces can validate the prior it was given.** Residual,
   RMS and reprojection are all just as happy with a 35°-wrong prior. Only
   re-solving *without* the prior and comparing exposes it — which is why the
   solve now always does both.

---

## Current status 🚦

- Tooling: **working, validated, well-instrumented.** Full pipeline runs live.
- Best live solve so far (continuous, `min_snr:=100`): `|t|≈18 cm` (plausible
  baseline), `apex ✓ data-determined`, `1σ t ≈ 25–44 mm`, but **`3-D ≈ 165 mm`,
  `residual ≈ 2.3σ`, `LOO ≈ 3σ`, `VERDICT ✗`** — not yet trustworthy.
- Blockers, in priority order:
  1. **Azimuth spread too small (~20°)** → rotation under-constrained. Sweep the
     rig far **left ↔ right** (target az spread 40–50°+).
  2. **Noise model too tight for this rig** → loosen `sigma_range:=0.05`,
     `sigma_az:=3`, `offset_prior_sigma:=0.05` so residual → ~1σ.
  3. **Camera hardware**: raise ZED to HD1080/HD2K; ideally print a **~100 mm
     ChArUco** to enable far poses (range diversity → accuracy).
  4. Consider a **smaller / point-like reflector** to reduce phase-center wander.

---

## Recommended parameters (current best)

```
-p radar_topic:=/radar1/radar/points_all -p pc_field_snr:=intensity -p pc_field_doppler:=doppler
-p sigma_range_m:=0.05 -p sigma_az_deg:=3.0 -p sigma_el_deg:=10.0
-p min_range:=0.5 -p max_range:=2.5 -p range_gate_margin_m:=0.5
-p max_abs_doppler:=-1.0 -p min_abs_doppler:=-1.0
-p reflector_offset_x:=0.10 -p reflector_offset_y:=0.23 -p reflector_offset_z:=-0.05
-p solve_offset:=true -p offset_prior_sigma_m:=0.05
-p use_extrinsic_prior:=true -p prior_t_xyz:="[0.207,0.016,0.020]"
-p prior_quat_xyzw:="[-0.5,-0.5,-0.5,0.5]"   # NOT prior_rpy_deg — gimbal lock
-p prior_t_sigma_m:=0.05 -p prior_rot_sigma_deg:=15.0 -p gate_radius:=0.5
-p pnp_ambiguity_ratio:=1.2 -p prior_disagree_warn_deg:=5.0
-p capture_mode:=continuous -p min_baseline:=0.10 -p min_snr:=100.0
-p val_pass_reproj_px:=200.0 -p val_pass_3d_mm:=150.0 -p min_points:=20
-p parent_frame:=zed_left_camera_optical_frame -p child_frame:=radar1_link -p show_window:=true
```
Convergence targets: `az spread > 40°`, `3-D < ~150 mm`, `residual ≈ 1σ`,
`LOO ≈ residual`, `apex ✓ data-determined`, **`1σ data` (not just the posterior)
small**, **`prior` gap < 5°**, the printed **axis map matching the physical
mounting** — and, the real judge, the **magenta radar dot glued to the
reflector** across the whole FoV. Ignore `rpy(deg)` entirely.

---

## Collection technique (what actually works)

1. Rigidly attach the reflector to the board; keep its opening toward the radar.
2. `capture_mode:=continuous`, `min_snr:=100` → sweep slowly, it captures every
   `min_baseline` of movement; each capture shows `snr ≥ 100`.
3. **Sweep WIDE in azimuth** (far left ↔ far right) and near ↔ far; moderate tilt.
4. Watch `RADAR spread: az` climb and the magenta dot track the reflector.
5. Solve auto-refreshes past `min_points`; `~/save` writes YAML/JSON + TF cmd.

---

## Next steps

1. Collect ~20–30 clean poses with **wide azimuth** (the immediate blocker).
2. Raise **ZED resolution**; print a **~100 mm board** for far poses.
3. Evaluate a **smaller reflector** (less phase-center wander).
4. Add a **launch file** (`launch/iwr6843isk.launch.py`, present) to `setup.py`
   `data_files` so params are typed (avoids the YAML-boolean trap).
5. Once converged, validate with a fresh independent pose set + the live overlay.
