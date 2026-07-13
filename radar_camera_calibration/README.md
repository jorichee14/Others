# Radar ↔ Camera Extrinsic Calibration (ChArUco board + trihedral reflector)

Estimate the rigid extrinsic **`T_cam_radar`** between a camera and a radar using
one rigid target: a **ChArUco board with a trihedral corner reflector** bolted to
it at a known, fixed offset.

- `radar_camera_calib.py` — the calibration node (this task).
- `general_charuco.py` — the original **camera-to-camera** tool you started from,
  kept for reference.

---

## Method

The trihedral corner reflector is the trick. It is the **single brightest radar
return** in the scene, and it is **rigidly fixed to the ChArUco board**, so both
sensors can locate the same physical point (the reflector apex):

```
camera : detect ChArUco board → T_cam_board (6-DOF) → p_cam = T_cam_board · apex_board
radar  : reflector = strongest gated return          → p_radar   (a 3-D point)
```

A radar point has **no orientation**, so one view can't give a 6-DOF transform.
Move the rig to `N ≥ 3` non-collinear positions, collect corresponding point
pairs, and solve for `X = T_cam_radar` (`p_cam = R·p_radar + t`).

### The estimator — why not plain Kabsch

A radar measures **range precisely (~cm)** but **angle poorly (degrees)**, and
cross-range error **grows with range** (≈ `range·σ_az` — ~9 cm at 5 m for
σ_az = 1°, ~17 cm at 10 m). Isotropic Cartesian Kabsch
(`min ‖R·p_radar + t − p_cam‖²`) weights that large, range-dependent angular
error the same as the tiny range error, so it is **biased**. Monte-Carlo on this
exact geometry: **Kabsch ≈ 1.7° / 120 mm** vs the estimator below
**≈ 1.0° / 38 mm** (≈1.6× better rotation, ≈3× better translation).

So the tool does **maximum-likelihood estimation in the radar's measurement
space**: predict each radar measurement from the (accurate) camera apex via the
current `(R,t)`, convert to `(range, azimuth, elevation)`, and minimise
residuals weighted by each component's real σ:

```
predicted radar pt = R^T (p_cam − t)
min_{R,t} Σ ρ( [ (r_m−r_p)/σ_r , (az_m−az_p)/σ_az , (el_m−el_p)/σ_el ] )
```

- `ρ` = **Huber** + iterative σ-gated **outlier rejection** — one wrong radar
  match blows plain least-squares up to ~50°; the robust solver holds ~1°.
- **Kabsch** supplies the initial guess.
- **2-D radar** (no elevation) is auto-detected and the elevation residual is
  dropped — but then **out-of-plane rotation and height are unobservable**, and
  the per-DOF covariance readout flags exactly which parameters are weak.

Set `sigma_range_m`, `sigma_az_deg`, `sigma_el_deg` from your radar's spec — the
*relative* sizes are what steer the weighting.

### The apex offset — measured, refined, or calculated (`solve_offset`)

You don't have to trust a hand-measurement. The offset `a` (board→apex, in the
board frame) enters as `p_cam = board_R·a + board_t`, and because the board
**rotates** between poses, `a` is separable from the constant extrinsic
translation — so the tool can **jointly estimate it** (MAP: a free offset
regularised toward your measured value). Verified in simulation:

| your measurement | fixed | **joint solve** |
|---|---|---|
| good (~5 mm) | 0.82° / 23 mm | 0.77° / 20 mm — doesn't hurt |
| wrong (44 mm) | 0.89° / 44 mm | 0.59° / 20 mm — repairs it, offset → ~15 mm |
| **none** (seed 0) | — | 0.38° / 25 mm, offset **calculated** 114 → ~15 mm |

**The catch — observability.** The offset is only visible where radar cross-range
noise (≈ `range·σ_az`) is *smaller* than the offset, i.e. at **close range
(1.5–3 m) with high board tilt (±45–55°)**. At long range it's swamped and the
solve just returns your prior. The reported **apex 1σ** tells you which happened
(`✓ data-determined` vs `trusting your prior`), and the debug overlay confirms
the apex visually either way.

**How to use it:**
- Measured it well → leave `solve_offset:=true`, `offset_prior_sigma_m:=0.02`
  (tight). It stays put; nothing lost.
- Measured it roughly → `offset_prior_sigma_m:=0.05`. It refines toward truth.
- Can't measure it → set `reflector_offset_*` to a rough guess (or 0),
  `offset_prior_sigma_m:=0.10` (loose), and **include a batch of close-range,
  high-tilt poses** so it's observable.

**Pose diversity is everything** — spread captures across range, azimuth, and
**height**. Collinear/planar poses make out-of-plane rotation unobservable; the
tool detects this (condition number, planar singular value, and per-DOF σ) and
warns. Aim for **8–15 well-spread poses**.

### Why the camera side uses the board, not depth

A bare corner reflector is specular: stereo/ToF depth on bare metal is
unreliable and a hand-click on it is ~10 px noisy. The **ChArUco board pose is
metric** (from the known square size), sub-mm, and fully automatic. You only
measure the apex→board offset once. This is the main upgrade over click-and-depth
calibration tools — no depth topic is even required.

### Radar side (robust reflector identification)

1. **Background subtraction** — pool `bg_accum_frames` radar frames with the rig
   *out* of the scene (`~/background`). A live point counts as "new" only if it's
   farther than `bg_match_dist` from every background point. Kills static clutter.
2. **Gating** — range window `[min_range, max_range]`; optional `|doppler|`
   window (the held-still reflector is ≈0 doppler, so walking people are
   rejected); and, once an extrinsic exists, proximity to the camera-predicted
   apex (`gate_radius`).
3. **Highest-SNR selection** — among the survivors, take **argmax(SNR)**. The
   trihedral is built to be the strongest reflector, so this *is* the apex. One
   bright background-subtracted point — no clustering needed.
4. **Doppler ↔ motion consistency** (`use_doppler_consistency`, for a *moving /
   hand-held* rig) — while you sweep, the reflector is rigidly tied to the board,
   so its radar radial velocity must equal the rate of change of the camera's
   range to the apex. Your hand, arm, and body also move, but at a *different*
   radial velocity, so a static `|doppler|≈0` gate can't separate them — this
   consistency check can. It keeps only survivors whose measured doppler matches
   the camera-predicted `v_pred = (p_cam − t_ext)·ṗ_cam / |p_cam − t_ext|`
   (rotation-free) within `doppler_match_tol`, then takes argmax(SNR); it falls
   back to the full set if that would empty it, and the sign convention is
   auto-learned. **This is the primary fix for "the dynamic mode picks the wrong
   feature."** See *Moving / hand-held rigs* below.

---

## Moving / hand-held rigs (the "dynamic" mode)

Sweeping a hand-held rig is the easy way to get pose diversity, but motion
introduces three correspondence errors that a static capture never has:

1. **Time mismatch** — the camera and radar are sampled at slightly different
   instants; at hand speed `v` and sync gap `Δt`, `p_cam` and `p_radar` are
   `v·Δt` apart (0.3 m/s × 60 ms ≈ 18 mm) even though they should be the *same*
   point. Guard it with `max_sync_dt` (drops badly-aligned pairs) and by moving
   slowly.
2. **Tracker lag** — people-counting firmware smooths a moving target, so its
   reported position lags the truth by the filter's time constant. This scales
   with speed too; keep speed modest.
3. **Feature ambiguity** — your hand/arm/body are strong, *moving* reflectors, so
   `min_abs_doppler`/`max_abs_doppler` alone can't isolate the reflector.
   `use_doppler_consistency` resolves this using the motion itself (above).

**Two ways to run it, most-robust first:**

- **Step-and-settle (recommended).** `capture_mode:=auto`; move, pause ~0.5 s,
  let it capture during the settle, move again. You get wide diversity *and* zero
  moving-correspondence error (errors 1–2 vanish when stationary). This is almost
  always the most accurate hand-held option.
- **Continuous sweep.** `capture_mode:=continuous` with
  `use_doppler_consistency:=true`, `max_sync_dt:=0.03`, and
  `max_capture_speed:=0.25` (skips captures while you move too fast). Turn motion
  into your discriminator instead of fighting it.

Extra params for these modes:

| Param | Meaning |
|---|---|
| `use_doppler_consistency` | keep radar pts whose doppler matches the camera-predicted radial velocity |
| `doppler_match_tol` (m/s) | match tolerance (default 0.30) |
| `doppler_sign` | `auto` (learn) \| `1` \| `-1` — TI radial-velocity sign |
| `min_motion_mps` | only apply the doppler gate above this apex speed |
| `max_sync_dt` (s) | drop image/radar pairs whose stamps differ by more than this (`-1` off) |
| `max_capture_speed` (m/s) | continuous mode: skip captures while moving faster than this (`-1` off) |

---

## Measuring `apex_in_board`

`reflector_offset_{x,y,z}` is the vector from the **ChArUco board origin** to the
**reflector apex**, in the board frame. You can measure it, and the tool will
also **refine or fully calculate** it (`solve_offset`, see above) — a rough value
here just seeds/anchors the estimate. Still, a good measurement is the tight
prior that makes everything robust, so measure it if you can.

Board frame (OpenCV ChArUco): origin at the first inner chessboard corner,
**+x** along `squares_x`, **+y** along `squares_y`, **+z** out of the board plane
toward the camera. Example: apex 3 cm right, 10 cm down, 4 cm proud of the board:

```
reflector_offset_x: 0.03
reflector_offset_y: 0.10
reflector_offset_z: 0.04
```

**Verify visually first.** View `debug_image_topic` — the tool draws the board
axes and a dot at the projected apex. That dot must land on the real reflector.
If not, fix the offset (usually a sign). 30 seconds here saves the calibration.

---

## Inputs / outputs

**Subscribes**
| topic (param) | type | purpose |
|---|---|---|
| `image_topic` | `sensor_msgs/Image` | camera image (ChArUco) |
| `info_topic` | `sensor_msgs/CameraInfo` | intrinsics K, D |
| `radar_topic` | `sensor_msgs/PointCloud2` | radar detections (`/points_all`: x,y,z,snr,doppler) |
| `~/background` `~/capture` `~/solve` `~/reset` `~/save` | `std_msgs/Empty` | control |

**Publishes**
- static TF `parent_frame → child_frame` (camera optical → radar), if `publish_tf`.
- `debug_image_topic` — annotated image: board axes, projected apex, and (once
  solved) the **whole radar cloud projected onto the feed, coloured by depth**.
  This live overlay is the most trustworthy check — walk the reflector around and
  confirm the dots stay glued to it near/far and left/right/high/low.
- `extrinsic_<cam>__<radar>.yaml` / `.json` — `T_cam_radar`, its inverse, RMS,
  LOO-CV, per-axis bias, condition number, verdict, and a ready-to-run
  `static_transform_publisher` command.

---

## Two scripts: static vs dynamic

The calibration comes in two profiles over one shared implementation
(`radar_camera_calib.py` — same solver, gating, validation, save/TF):

| Script | For | Capture | Extras enabled |
|---|---|---|---|
| **`radar_camera_calib_static.py`** | rig held **still** at each pose (step-and-settle) — **most accurate** | `capture_mode:=auto` (stability-gated) | **diversity HUD** on |
| **`radar_camera_calib_dynamic.py`** | rig you keep **moving** (continuous sweep) | `capture_mode:=continuous` | Doppler↔motion consistency, `max_sync_dt`, `max_capture_speed` |

Each is a thin profile that just presets sensible defaults; every parameter is
still overridable on the command line. Prefer **static** whenever you can pause —
stopping removes the time-mismatch and tracker-lag errors that a moving rig
suffers (see *Moving / hand-held rigs*). Use **dynamic** only if you truly can't
stop. The old `radar_camera_calib.py` still runs both via `capture_mode`.

> Package entry points (add to your `setup.py` `console_scripts`):
> ```python
> 'radar_camera_calib_static  = wicoms_utils.radar_camera_calib_static:main',
> 'radar_camera_calib_dynamic = wicoms_utils.radar_camera_calib_dynamic:main',
> ```

### Diversity HUD — "is my pose set good enough for rotation?"

A single reflector gives good **translation** but **bad rotation** unless the
poses are diverse — the #1 static-calibration failure. The static script draws a
live HUD (`show_diversity_hud:=true`) with six bars that go **green** when each
crosses the target that makes rotation (and, via the offset, translation)
well-observed:

- **PITCH / ROLL / YAW** — spread of the **board orientation** (camera frame:
  pitch=about X, yaw=about Y, roll=about Z). Board tilt is what makes the apex
  **offset** observable. Targets: 40° / 30° / 40°.
- **AZ / EL / RANGE** — spread of the **radar points**. This is the lever arm
  that makes the **extrinsic rotation** observable (rotation error ≈
  cross-range noise ÷ point-cloud extent). Targets: 40° / 15° / 0.30 m.

When all bars are green, `n ≥ min_points`, and the measured **`rot 1σ`** (shown
after a solve) is small, the HUD reads **READY — rotation observable**. In
practice: **tilt** the board in pitch and yaw, **roll** it, *and* **move** the
rig near↔far, left↔right, and up↔down. Watch the red bars fill.

---

## Usage

```bash
pip install -r requirements.txt        # numpy, scipy, opencv-contrib-python
# plus ROS 2: rclpy, cv_bridge, sensor_msgs_py, message_filters, tf2_ros
```

Static (recommended), with the diversity HUD:

```bash
python3 radar_camera_calib_static.py --ros-args \
  -p image_topic:=/zed/zed_node/left/image_rect_color \
  -p info_topic:=/zed/zed_node/left/camera_info \
  -p radar_topic:=/points_all -p pc_field_snr:=snr \
  -p squares_x:=9 -p squares_y:=7 -p square_len:=0.020 -p marker_len:=0.015 \
  -p reflector_offset_x:=0.03 -p reflector_offset_y:=0.10 -p reflector_offset_z:=0.04 \
  -p parent_frame:=zed_left_camera_optical_frame -p child_frame:=radar_link \
  -p min_points:=8 -p show_window:=true
```

Dynamic (only if you can't stop moving):

```bash
python3 radar_camera_calib_dynamic.py --ros-args \
  -p radar_topic:=/radar1/radar/points_all -p pc_field_snr:=intensity \
  -p use_doppler_consistency:=true -p max_sync_dt:=0.03 -p max_capture_speed:=0.25 \
  # ...same camera/board/offset/frame params as above...
```

Or the fully-explicit shared node:

```bash
python3 radar_camera_calib.py --ros-args \
  -p image_topic:=/zed/zed_node/left/image_rect_color \
  -p info_topic:=/zed/zed_node/left/camera_info \
  -p radar_topic:=/points_all \
  -p pc_field_snr:=snr -p pc_field_doppler:=doppler \
  -p squares_x:=9 -p squares_y:=7 -p square_len:=0.020 -p marker_len:=0.015 \
  -p dictionary:=DICT_4X4_50 \
  -p reflector_offset_x:=0.03 -p reflector_offset_y:=0.10 -p reflector_offset_z:=0.04 \
  -p parent_frame:=zed_left_camera_optical_frame -p child_frame:=radar_link \
  -p capture_mode:=auto -p min_points:=6 -p min_baseline:=0.15
```

Then:
1. Rig **out** of the scene → `ros2 topic pub -1 /radar_camera_calib/background std_msgs/msg/Empty {}`.
2. Watch `debug_image_topic` — confirm the apex dot sits on the reflector.
3. Bring the rig in; move it to a spot seen by **both** sensors and hold ~1 s.
   Auto mode captures when stable and moved ≥ `min_baseline` from prior captures.
   (Or publish on `~/capture`.)
4. Repeat across a spread of ranges/angles/heights (8–15 poses). The extrinsic
   solves and re-validates after each capture; force with `~/solve`, write with
   `~/save`, clear with `~/reset`.

---

## What "good" looks like

Printed on every solve and checked against thresholds (radar angular noise is a
few degrees, so pixel error is naturally larger than a camera-camera rig — don't
chase sub-pixel numbers):

| metric | threshold (param) | meaning |
|---|---|---|
| residual | ≈ 1 σ | measurement-space fit vs the radar noise model; ≫1 ⇒ σ too small, bad matches, or a real misfit |
| per-DOF 1σ (rot/t) | small | from the covariance; a large σ on a DOF ⇒ that parameter is weak/unobservable (esp. 2-D radar) |
| LOO-CV | ≈ residual σ | honest generalisation; ≫ residual ⇒ too few / clustered poses |
| mean reproj | `val_pass_reproj_px` (20 px) | radar-predicted apex lands on the visual apex |
| mean 3-D | `val_pass_3d_mm` (150 mm) | Cartesian agreement (cross-check) |
| max per-axis bias | `val_pass_bias_mm` (50 mm) | **signed** residual — a non-zero mean on an axis is a systematic error RMS would hide |
| \|t\| | `measured_baseline_m` ± `baseline_tol_m` | tape-measure the radar↔camera distance for metric ground truth |
| condition number | < ~200 | pose spread; high ⇒ add diversity |
| inliers / rejected | — | how many matches the robust solver kept vs threw out |
| live overlay | — | dots track the reflector everywhere in the FoV |

The **VERDICT** line fails if any threshold is exceeded **or** any DOF is flagged
unobservable (e.g. a 2-D radar can never pin out-of-plane rotation from points).

**Range diagnostic** — if radar range disagrees with the camera, the solve prints
a `cam_r = a·radar_r + b` fit and suggests `radar_range_scale` / `radar_range_bias_m`
(applied at ingest, before everything). `a ≠ 1` is a scale error; `b ≠ 0` a bias.

---

## Key parameters

| param | default | meaning |
|---|---|---|
| `radar_topic` | `/points_all` | radar PointCloud2 |
| `pc_field_x/y/z/snr/doppler` | `x/y/z/snr/doppler` | field names (missing ones tolerated) |
| `select_by` | `snr` | `snr` (highest return) or `nearest` |
| `min_range`/`max_range` | `0.3`/`20` m | range gate |
| `max_abs_doppler` | `-1` (off) | keep `|doppler| ≤` this — rejects moving clutter |
| `gate_radius` | `1.0` m | proximity gate to predicted apex once solved |
| `bg_accum_frames` | `15` | frames pooled on `~/background` |
| `bg_match_dist` | `0.2` m | "new point" distance threshold |
| `require_background` | `false` | refuse to capture until background pooled |
| `radar_range_scale`/`_bias_m` | `1.0`/`0.0` | ingest range correction |
| `sigma_range_m` | `0.05` | radar range noise (drives ML weighting) |
| `sigma_az_deg` | `2.0` | radar azimuth noise — the dominant cross-range term |
| `sigma_el_deg` | `5.0` | radar elevation noise (usually the worst) |
| `force_2d_radar` | `false` | ignore elevation entirely; auto-detected otherwise |
| `huber_f_scale` / `reject_sigma` | `1.5` / `4.0` | robust-loss knee / outlier cutoff (σ units) |
| `capture_mode` | `auto` | `auto` or `manual` (`~/capture`) |
| `stable_window`/`stable_std` | `12`/`0.01 m` | per-pose stability gate |
| `min_baseline`/`min_points` | `0.15 m`/`6` | capture spacing / count |
| `reflector_offset_{x,y,z}` | `0` | apex offset in board frame (measured value / **offset prior centre**) |
| `solve_offset` | `true` | jointly estimate the apex offset (MAP) |
| `offset_prior_sigma_m` | `0.03` | **offset prior** width — tight if measured well, `0.10` if not |
| `use_extrinsic_prior` | `false` | regularise + initialise the extrinsic toward a known mounting |
| `prior_t_xyz` | `[0,0,0]` | **extrinsic prior**: radar position in camera frame (m) |
| `prior_rpy_deg` | `[0,0,0]` | extrinsic prior: radar orientation in camera frame (xyz euler) |
| `prior_t_sigma_m` / `prior_rot_sigma_deg` | `0.05` / `10` | extrinsic prior widths (tight = trust it more) |
| `min_snr` | `0` (off) | **strict capture**: reject a pick whose reflector SNR is below this |
| `measured_baseline_m` | `-1` (off) | tape-measured `|t|` for the baseline check |

---

## Priors and strict captures (for under-constrained / close-range rigs)

When poses are few or clustered, the extrinsic is under-determined (large `1σ`).
Two knobs stabilise it.

**Offset prior** (always on): `reflector_offset_{x,y,z}` is the prior centre and
`offset_prior_sigma_m` its width. Measure the offset and set a tight sigma
(`0.02`) to pin it; use `0.10` to let the data drive it.

**Extrinsic prior** (opt-in): give a rough known radar-in-camera pose from a tape
measure / CAD and the solve is both **initialised from it and regularised toward
it** (MAP). It caps the uncertainty of poorly-observed DOFs at the prior width
instead of letting them blow up.
```
-p use_extrinsic_prior:=true \
-p prior_t_xyz:="[0.20, 0.0, 0.0]" \      # radar ~20 cm along camera +x, measured
-p prior_rpy_deg:="[-115, -70, -130]" \   # rough radar orientation in camera frame
-p prior_t_sigma_m:=0.05 \                # tighten to trust the prior more
-p prior_rot_sigma_deg:=10
```
Get `prior_rpy_deg` from the nominal mounting (or from a first rough solve's
`rpy(deg)` line), and `prior_t_xyz` by tape-measuring the radar position relative
to the camera optical centre. Tighten the sigmas only as far as you trust the
measurement — a wrong-but-tight prior biases the result.

**Strict captures**: `min_snr` rejects any capture whose reflector return is
weaker than the threshold (weak returns are the ones most likely mis-associated
with clutter). Also tighten `stable_std_radar` and/or raise `stable_window` so
only rock-steady poses are accepted. These are the cheapest defences against a
bad correspondence poisoning the solve (on top of the built-in Huber +
`reject_sigma` outlier rejection).

## Coordinate conventions

- radar: `X=forward, Y=left, Z=up` (automotive)
- camera: `X=right, Y=down, Z=forward` (optical/pinhole)

Raw radar points go straight into Kabsch, so the solved `R` absorbs the **full**
~90° frame difference **and** the mounting rotation — there is no separate remap
step to get wrong. A large-looking `R` (≈90°) is therefore expected and correct.
`T_cam_radar` (and its inverse) in the YAML already map raw radar points into the
camera frame; the emitted `static_transform_publisher` command is ready to use.
