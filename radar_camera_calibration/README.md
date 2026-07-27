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
3. **Reflector selection** (`select_by`) — identify the reflector among the
   survivors:
   - `snr` (default) — **argmax(SNR)**; the trihedral is the strongest reflector.
   - `cluster` — group the survivors (connected components within `cluster_eps`,
     min `min_cluster_size` points) and take the **SNR-weighted centroid** of the
     reflector blob (the cluster holding the brightest return, or the one nearest
     the predicted apex once an extrinsic/prior exists). More robust than a lone
     argmax spike: it **averages the blob's returns down** (less angular noise)
     and **rejects an isolated bright clutter point** that SNR-max would grab.
     Falls back to `snr` if no cluster meets `min_cluster_size`.
   - `nearest` — nearest the predicted apex / radar origin.

   **Cluster around the estimated apex.** Once the extrinsic (from a solve or an
   extrinsic prior) predicts where the reflector should be, `cluster` mode
   restricts to points within `cluster_apex_radius` of that prediction and takes
   the cluster nearest it — so a *brighter* off-apex blob can't win. With
   `cluster_strict:=true`, a capture with **no** cluster within that radius is
   **rejected** (no SNR fall-back) — the strictest, cleanest gate for the
   reflector once you have a decent prior.
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
- `extrinsic_<cam>__<radar>_session.json` — full **reproducible session record**:
  ISO timestamp, every parameter used, every capture (radar point + camera apex +
  full board pose), and the solved result. Re-solve or audit it offline with
  `sessions/solve_from_poses.py` (no ROS needed). Written on every solve/`~/save`.

---

## Two scripts: static vs dynamic

The calibration comes in two profiles over one shared implementation
(`radar_camera_calib.py` — same solver, gating, validation, save/TF):

| Script | For | Capture | Extras enabled |
|---|---|---|---|
| **`radar_camera_calib_static.py`** | rig held **still** at each pose (step-and-settle) — **most accurate** | `capture_mode:=auto` (stability-gated) | **diversity HUD** on |
| **`radar_camera_calib_dynamic.py`** | rig you keep **moving** (continuous sweep) | `capture_mode:=continuous` | Doppler↔motion consistency, `max_sync_dt`, `max_capture_speed` |
| **`radar_camera_calib_arducam.py`** | **Arducam** (raw/unrectified feed), step-and-settle | `capture_mode:=auto` (stability-gated) | static profile **+ `rectify_image`** (in-node undistort) |

Each is a thin profile that just presets sensible defaults; every parameter is
still overridable on the command line. Prefer **static** whenever you can pause —
stopping removes the time-mismatch and tracker-lag errors that a moving rig
suffers (see *Moving / hand-held rigs*). Use **dynamic** only if you truly can't
stop. The old `radar_camera_calib.py` still runs both via `capture_mode`.

> Package entry points (add to your `setup.py` `console_scripts`):
> ```python
> 'radar_camera_calib_static  = wicoms_utils.radar_camera_calib_static:main',
> 'radar_camera_calib_dynamic = wicoms_utils.radar_camera_calib_dynamic:main',
> 'radar_camera_calib_arducam = wicoms_utils.radar_camera_calib_arducam:main',
> 'radar_fusion_reflector     = wicoms_utils.radar_fusion_reflector:main',
> 'radar_cloud_fusion         = wicoms_utils.radar_cloud_fusion:main',
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

## Full run reference — every parameter

Two ways to run each node: a **launch file** (recommended — typed, so the CLI
YAML-boolean trap `-p pc_field_y:=y → true` can't happen) or an **explicit CLI
command** listing every parameter. The launch files double as the parameter
reference.

### A. Calibration node — launch file (recommended)

```bash
ros2 launch wicoms_utils iwr6843isk.launch.py     # edit launch/iwr6843isk.launch.py for your rig
```

### B. Calibration node — full CLI (all parameters)

```bash
ros2 run wicoms_utils radar_camera_calibration --ros-args \
  `# camera` \
  -p image_topic:=/zed/zed_node/left/image_rect_color \
  -p info_topic:=/zed/zed_node/left/camera_info \
  `# board (edit to your printed ChArUco)` \
  -p squares_x:=4 -p squares_y:=4 -p square_len:=0.12 -p marker_len:=0.09 \
  -p dictionary:=DICT_4X4_50 -p min_corners:=4 -p max_reproj_px:=1.5 \
  `# radar topic + fields` \
  -p radar_topic:=/radar1/radar/points_all \
  -p pc_field_x:=x -p pc_field_y:=y -p pc_field_z:=z \
  -p pc_field_snr:=intensity -p pc_field_doppler:=doppler \
  `# reflector selection` \
  -p select_by:=cluster -p cluster_eps:=0.20 -p min_cluster_size:=1 \
  -p cluster_apex_radius:=0.40 -p cluster_strict:=false \
  `# range / doppler gating` \
  -p min_range:=0.5 -p max_range:=2.5 -p range_gate_margin_m:=0.5 -p gate_radius:=0.5 \
  -p max_abs_doppler:=-1.0 -p min_abs_doppler:=-1.0 \
  -p use_doppler_consistency:=false -p doppler_match_tol:=0.30 \
  -p doppler_sign:=auto -p min_motion_mps:=0.05 \
  `# background subtraction` \
  -p bg_accum_frames:=15 -p bg_match_dist:=0.2 -p require_background:=false \
  `# range correction (per-radar; applied at ingest)` \
  -p radar_range_scale:=1.0 -p radar_range_bias_m:=0.0 \
  `# radar noise model` \
  -p sigma_range_m:=0.05 -p sigma_az_deg:=3.0 -p sigma_el_deg:=8.0 -p force_2d_radar:=false \
  `# robust solver` \
  -p huber_f_scale:=1.5 -p reject_sigma:=4.0 -p reject_axis_sigma:=0.0 \
  `# apex offset + offset prior` \
  -p reflector_offset_x:=0.0 -p reflector_offset_y:=0.0 -p reflector_offset_z:=0.0 \
  -p solve_offset:=true -p offset_prior_sigma_m:=0.05 \
  `# extrinsic prior (opt-in)` \
  -p use_extrinsic_prior:=false \
  -p prior_t_xyz:="[0.0,0.0,0.0]" -p prior_rpy_deg:="[0.0,0.0,0.0]" \
  -p prior_t_sigma_m:=0.05 -p prior_rot_sigma_deg:=10.0 \
  `# capture / strictness` \
  -p capture_mode:=auto -p stable_window:=12 -p stable_std:=0.02 -p stable_std_radar:=0.10 \
  -p min_baseline:=0.15 -p min_points:=25 -p min_snr:=100.0 -p sync_slop:=0.06 \
  -p max_sync_dt:=-1.0 -p max_capture_speed:=-1.0 \
  `# validation thresholds` \
  -p val_pass_reproj_px:=20.0 -p val_pass_3d_mm:=150.0 -p val_pass_bias_mm:=50.0 \
  -p measured_baseline_m:=-1.0 -p baseline_tol_m:=0.03 \
  `# frames / output / display` \
  -p parent_frame:=zed_left_camera_optical_frame -p child_frame:=radar1_link \
  -p camera_name:=zed_left -p radar_name:=radar1 -p output_path:="" \
  -p publish_tf:=true -p debug_image:=true \
  -p debug_image_topic:=/radar_camera_calib/debug_image \
  -p show_window:=true -p show_diversity_hud:=true -p radar_watchdog_s:=3.0
```

> **Dynamic profile:** flip `-p capture_mode:=continuous -p use_doppler_consistency:=true
> -p max_sync_dt:=0.03 -p max_capture_speed:=0.25`. **Multipath-ghost rejection:**
> add `-p reject_axis_sigma:=3.5`. Every `*_poses.json` / `*_session.json` in
> `sessions/` lists the exact params used for that run.

### C. Fusion + tracking node — launch file

```bash
ros2 launch wicoms_utils radar_fusion.launch.py   # edit launch/radar_fusion.launch.py with YOUR extrinsics
```

### D. Fusion + tracking node — full CLI (all parameters)

```bash
ros2 run wicoms_utils radar_fusion_reflector --ros-args \
  `# topics` \
  -p image_topic:=/zed/zed_node/left/image_rect_color \
  -p info_topic:=/zed/zed_node/left/camera_info \
  -p radar1_topic:=/radar1/radar/points_all -p radar2_topic:=/radar2/radar/points_all \
  -p pc_field_x:=x -p pc_field_y:=y -p pc_field_z:=z -p pc_field_snr:=intensity \
  `# per-radar extrinsics T_cam_radar (FINAL values — replace if re-solved)` \
  -p r1_t_xyz:="[0.2368,0.0190,-0.0542]" \
  -p r1_quat_xyzw:="[-0.4995,0.6007,-0.4224,-0.4596]" \
  -p r2_t_xyz:="[-0.1194,-0.0096,-0.0157]" \
  -p r2_quat_xyzw:="[0.7572,0.0539,0.6506,-0.0217]" \
  -p r1_range_scale:=0.958 -p r2_range_scale:=0.967 \
  `# radar noise model (drives fusion weighting)` \
  -p sigma_range_m:=0.05 -p sigma_az_deg:=3.0 -p sigma_el_deg:=8.0 \
  `# reflector selection` \
  -p min_range:=0.3 -p max_range:=6.0 -p min_snr:=100.0 -p select_radius_m:=0.5 \
  `# maneuvering-target tracker` \
  -p process_accel:=1.0 -p maneuver_gain:=3.0 -p maneuver_deadband:=0.15 \
  -p innov_gate_chi2:=11.35 -p adapt_window:=12 -p adapt_max_scale:=4.0 \
  -p reinit_gap_s:=1.0 -p coast_s:=0.5 -p trail_len:=60 -p trail_s:=3.0 \
  `# output / display` \
  -p publish_point:=true -p debug_image_topic:=/radar_fusion/debug_image -p show_window:=true
```

> The `` `# comment` `` lines are bash no-ops (command substitution of a comment) —
> they group the flags and are safe to keep or delete. Inline-array params
> (`prior_t_xyz`, `r1_t_xyz`, …) **must** be quoted as shown.

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
| 3-D error RMS X/Y/Z | — | the 3-D error **split per axis** (camera frame: X/Y = cross-range, Z = range); a big X/Y ⇒ radar angular error, a big Z ⇒ a range error |
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
| `rectify_image` | `false` | undistort each frame in-node from `camera_info` — for a RAW feed (e.g. Arducam `image_raw`); detection & projection then use the rectified K and zero D. No-op when D≈0 (already-rectified feeds like ZED `image_rect_color`). |
| `rectify_alpha` | `0.0` | undistort crop: `0`=only valid pixels (zoomed), `1`=keep whole FoV (black borders) |
| `debug_every_n` | `1` | build/publish the debug overlay only every Nth pair. Visualization only; capture/solve unaffected. |
| `debug_scale` | `1.0` | downscale the published/shown debug image (`0.5`=half) — cheaper bandwidth/render on a big feed |
| `debug_raw` | `true` | publish the raw `bgr8` debug Image (~MBs/frame). Set `false` on a remote node to send only the compressed stream. |
| `debug_compressed` | `true` | also publish a JPEG frame on `<debug_image_topic>/compressed` (~tens of KB) — the real-time fix for a remote/big feed; overlay stays live every frame. View it in `rqt_image_view`. |
| `debug_jpeg_quality` | `40` | JPEG quality 1–100 for the compressed stream; live-settable via `ros2 param set`. |
| `radar_topic` | `/points_all` | radar PointCloud2 |
| `pc_field_x/y/z/snr/doppler` | `x/y/z/snr/doppler` | field names (missing ones tolerated) |
| `select_by` | `snr` | `snr` (highest return), `cluster` (blob centroid), or `nearest` |
| `cluster_eps` | `0.15` m | `select_by:=cluster`: points within this join one cluster |
| `min_cluster_size` | `2` | `select_by:=cluster`: min points to accept a cluster (else fall back to SNR) |
| `cluster_apex_radius` | `0.30` m | cluster only points within this radius of the **predicted apex** (once solved/prior) |
| `cluster_strict` | `false` | `true` → reject the capture if no cluster forms within that radius of the predicted apex (no SNR fall-back) |
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
| `huber_f_scale` / `reject_sigma` | `1.5` / `4.0` | robust-loss knee / outlier cutoff — the **RMS-across-axes** residual (σ units) |
| `reject_axis_sigma` | `0` (off) | opt-in **per-axis** cutoff: also drop a match if *any single* axis (range/az/el) exceeds this σ. Catches a multipath ghost that's wrong in range **and** elevation but clean in azimuth, whose RMS therefore hides under `reject_sigma`. Try `3.5`. |
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

---

## Calibrated results — this rig (2026-07-15)

Two IWR6843ISK radars calibrated to `zed_left_camera_optical_frame`; radar2 rolled
~87° (orthogonal). Full records + offline cross-checks in
[`sessions/`](sessions/): [radar1](sessions/2026-07-15_zed_radar1.md) ·
[radar2](sessions/2026-07-15_zed_radar2.md) ·
**[radar1-vs-radar2 comparison](sessions/two_radars.md)**.

| TF (zed_left → radarN_link) | translation xyz (m) | quaternion xyzw |
|---|---|---|
| **radar1** | `+0.2218  −0.0067  −0.1721` | `−0.5345  +0.5853  −0.4196  −0.4424` |
| **radar2** | `−0.0999  −0.0124  −0.0011` | `+0.7882  −0.0406  +0.6121  +0.0499` |

**Independent offset cross-check (the key validation):** the corner-reflector
apex offset is a property of the *rig*, so both radars must recover the same value.
They agree to **8 mm (X)** and **34 mm (Y)** — both inside their combined 1σ — on
the two well-observed in-plane axes (the board-normal Z differs by 157 mm because
radar2's Z was weakly observed and its own solve flagged it). Two radars mounted
87° apart, sharing only the board, landing on the same offset ⇒ both extrinsics
are sound. Systematic bias is ~1 mm (radar2) and a few mm (radar1, except its
weak-elevation vertical); the large per-detection RMS is random radar angular
noise that averages out in the fusion/tracking below.

---

## Fusing two radars to locate the reflector (`radar_fusion_reflector.py`)

Once **both** radars are calibrated, this node fuses their detections to place
the corner reflector accurately and draws it on the ZED image. It exists because
a single IWR6843 is **anisotropic**: precise in range, moderate in azimuth, poor
in elevation. We mounted **radar2 rolled ~90°** vs radar1, so their weak axes are
**perpendicular** in the camera frame:

- **radar1** — sharp horizontal, soft vertical
- **radar2** — sharp vertical, soft horizontal

Each detection becomes a 3-D point **plus an anisotropic covariance**
(σ_range along the radial, `range·σ_az` / `range·σ_el` across it), rotated into
the camera frame by that radar's calibrated extrinsic.

**A tracker, not a per-frame combine.** A memoryless per-frame BLUE is only as
steady as the raw detections — which hop between multipath returns → a *jumpy*
output. Instead the node runs a **constant-velocity Kalman filter** (tuned for a
**moving / dynamic** reflector) in the camera frame. Both radars update it
**asynchronously** with their full 3-D covariance, so every axis is constrained by
whichever radar sees it sharply **and** the estimate is smoothed over time (it
takes the **horizontal from radar1** and the **vertical from radar2** — both, over
time, not one component):

```
predict:  x⁻ = F x,   P⁻ = F P Fᵀ + Q(σ_a)
update :  y = z_i − H x⁻,   S = H P⁻ Hᵀ + R_i,   K = P⁻ Hᵀ S⁻¹
          x = x⁻ + K y,     P = (I − K H) P⁻
R_i    = R_model_i (anisotropic, from the extrinsic) + Cov(recent innovations of i)
```

**Adaptive covariance from recent errors.** Each radar's `R_i` is inflated by the
sample covariance of its *recent innovations* (how far its detections have been
landing from the track), capped at `adapt_max_scale × model`. A radar that is
currently jumpy gets automatically trusted less. Every measurement is
**Mahalanobis-gated** (`innov_gate_chi2`) so a clutter jump is rejected before it
can move the estimate; selection itself is the SNR-weighted **centroid** of the
blob within `select_radius_m` of the prediction, not the single brightest pixel.

**Tracing a dynamic reflector (maneuvering-target model).** A *fixed* process
noise forces a bad tradeoff — small σ_a is smooth on a still reflector but **lags a
fast one** (sim: 392 mm RMS on a ~2.7 m/s sweep); large σ_a follows the fast one
but is jittery when still. So the process-noise accel std is **speed-adaptive**:

```
σ_a = process_accel + maneuver_gain · max(0, speed − maneuver_deadband)
```

Still → smooth (the deadband ignores measurement-noise-driven phantom velocity);
moving → agile. In sim this stays within ~10 mm of the *best fixed filter in every
regime* (static 86, slow 90, aggressive 119 mm RMS) instead of blowing up in one.
Versus the raw per-frame detections it cuts frame-to-frame **jitter ~3×** (234 →
~70 mm) and RMS error ~40%.

**Display**: radar1 (cyan) and radar2 (orange) show each radar's **raw** detection
with a bar along its blind axis; the **tracked** point (green cross) is the smooth
KF estimate with a **fading motion trace** of its recent path, a **heading arrow**
when it moves, and its per-axis 1σ (mm) + speed. Coasts on prediction up to
`coast_s` if both radars drop; hard-reinitialises after `reinit_gap_s`. Publishes
the tracked reflector on `/radar_fusion/reflector` (camera frame) and the
annotated image on `debug_image_topic`.

The extrinsic defaults are already the solved values for this rig; override any
with params.

```bash
python3 radar_fusion_reflector.py --ros-args \
  -p image_topic:=/zed/zed_node/left/image_rect_color \
  -p info_topic:=/zed/zed_node/left/camera_info \
  -p radar1_topic:=/radar1/radar/points_all \
  -p radar2_topic:=/radar2/radar/points_all \
  -p pc_field_snr:=intensity \
  -p min_snr:=100 -p assoc_gate_m:=0.6 -p show_window:=true
# override extrinsics if re-solved:
#   -p r1_t_xyz:="[...]" -p r1_quat_xyzw:="[...]" (same for r2), r{1,2}_range_scale
```

Needs **both** radars streaming (works with one — it just tracks that radar's
axes). Tuning knobs for the tracker:

| param | default | effect |
|---|---|---|
| `process_accel` | `1.0` m/s² | **quiet-state** smoothing floor. ↓ = steadier when still |
| `maneuver_gain` | `3.0` 1/s | how hard σ_a ramps with speed. ↑ = snappier follow of a fast reflector |
| `maneuver_deadband` | `0.15` m/s | speed below this doesn't ramp σ_a — keeps a still reflector smooth |
| `innov_gate_chi2` | `11.35` | Mahalanobis gate (3-DOF 99%). ↓ rejects more outliers |
| `adapt_window` / `adapt_max_scale` | `12` / `4.0` | how many recent innovations set the adaptive R, and the cap on its inflation |
| `select_radius_m` | `0.5` m | blob-centroid radius around the prediction (stabilises selection) |
| `reinit_gap_s` / `coast_s` | `1.0` / `0.5` s | hard-reinit after this gap; keep drawing the tracked point this long after last update |
| `trail_len` / `trail_s` | `60` / `3.0` s | length / max age of the motion trace |
| `sigma_az_deg` / `sigma_el_deg` | `3.0` / `8.0` | the model-floor angular noise (should match calibration) — sets each axis's baseline trust |

Because σ_a is speed-adaptive you rarely need to touch it, but: if the tracked
point **lags** a fast sweep, raise `maneuver_gain` (or lower `maneuver_deadband`);
if it's **jumpy** when the reflector is still, lower `process_accel` (or raise
`maneuver_deadband`).

---

## Fusing two radars into a better SCENE cloud (`radar_cloud_fusion.py`)

`radar_fusion_reflector.py` tracks **one** corner reflector. `radar_cloud_fusion.py`
is the cloud-level sibling: it fuses the **entire** point cloud of both radars into
a single, denser, less-noisy cloud in the camera frame, projects it onto the ZED
image, and **validates the fusion live** — for perception, not a single target.

**Why fusing helps (same anisotropy trick).** Each IWR6843 is sharp in range,
moderate in azimuth, poor in elevation; radar2 is rolled ~90°, so their weak axes
are perpendicular in the camera frame. Where **both** radars see the same physical
point, an information-form BLUE merge is tight on *both* axes:

```
C_f = (C1⁻¹ + C2⁻¹)⁻¹      p_f = C_f (C1⁻¹ p1 + C2⁻¹ p2)
```

radar1 supplies the horizontal, radar2 the vertical — a real accuracy gain, not
just more dots. Points seen by only one radar are kept (flagged `n_radars=1`) with
that radar's honest anisotropic covariance, so downstream code can weight them.

**The pipeline** — ingest (range_scale → gate → transform → per-point covariance)
→ **associate** the two clouds by Mahalanobis distance under `C1+C2` (optimal
Hungarian assignment, χ²-gated at `assoc_gate_chi2`) → **fuse** matched pairs with
the BLUE above, pass unmatched points through → optional temporal **voxel merge**
(`accum_s`, static scenes only) → **project** onto the image, coloured by depth,
each point drawn with its **1σ uncertainty ellipse** (the 3-D covariance through
the projection Jacobian; confirmed 2-radar points filled, single-radar hollow).

**Validation is built in — this is the point of "then validate".** Matched pairs
are 3-DOF, so their Mahalanobis **χ² should average ≈ 3** when the extrinsics *and*
the noise model are right. A mean ≫ 3 means the cloud is *not* trustworthy
(miscalibration or too-small σ), regardless of how good the overlay looks. Every
frame the node reports:

| metric | ideal | meaning |
|---|---|---|
| **mean χ²** of matched pairs | ≈ 3 | cross-radar consistency; ≫3 ⇒ extrinsics/σ wrong |
| within-gate % | high | fraction of pairs that associated under the χ² gate |
| **σ shrink** (fused ÷ best single) | < 1 | fusion genuinely tightened the estimate |
| pairs/frame, 2-radar vs 1-radar counts | — | overlap / confirmation rate |

It publishes the fused cloud on `fused_cloud_topic` (PointCloud2:
`x,y,z,intensity,n_radars,sigma_mm`) and the annotated image on
`debug_image_topic`, and logs a `VALID`/`CHECK` verdict every `report_every_s`.

**Offline proof (no ROS needed).** The math is self-validating on synthetic data
using this rig's final extrinsics:

```bash
python3 radar_cloud_fusion.py --selftest
```

It confirms the fused cloud beats **both** radars in 3-D RMS, that radar1 wins
horizontally / radar2 wins vertically (and fusion inherits the best of each), and
that matched-pair χ² is statistically consistent (mean ≈ 3). Representative run:

```
  axis        radar1 RMS    radar2 RMS     FUSED RMS
  X (horiz)      203mm         519mm         184mm
  Y (vert)       506mm         225mm         210mm
  Z (range)      104mm         135mm          67mm
  |3D|           321mm         336mm         166mm      matched-pair mean χ² = 2.79
```

**Run (live):**

```bash
python3 radar_cloud_fusion.py --ros-args \
  -p image_topic:=/zed/zed_node/left/image_rect_color \
  -p info_topic:=/zed/zed_node/left/camera_info \
  -p radar1_topic:=/radar1/radar/points_all \
  -p radar2_topic:=/radar2/radar/points_all \
  -p pc_field_snr:=intensity \
  -p show_window:=true
# extrinsics default to the FINAL rig values; override r{1,2}_t_xyz / _quat_xyzw /
# _range_scale if re-solved. Or: ros2 launch wicoms_utils radar_cloud_fusion.launch.py
```

Key knobs (full list in `launch/radar_cloud_fusion.launch.py`):

| param | default | effect |
|---|---|---|
| `assoc_gate_chi2` | `7.815` | cross-radar match gate (3-DOF 95%). Lower rejects looser pairs |
| `require_both` | `false` | `true` → publish **only** 2-radar confirmed points (max robustness, sparser) |
| `sync_s` | `0.15` s | both clouds must be within this window to cross-fuse |
| `accum_s` / `voxel_m` | `0.0` / `0.10` m | temporal accumulate + voxel-merge for a denser cloud — **static scenes only** (it smears motion) |
| `min_range`/`max_range`/`min_snr` | `0.3`/`8.0`/`0` | ingest gates |
| `max_points` | `400` | per-cloud cap (keeps the brightest) so association stays O(n·m)-bounded |
| `draw_ellipse` | `true` | per-point 1σ projected uncertainty ellipse on the overlay |
| `sigma_range_m`/`sigma_az_deg`/`sigma_el_deg` | `0.05`/`3.0`/`8.0` | radar noise model — **must match calibration**; it sets both the covariances and the χ² scale, so a wrong σ makes the χ² verdict lie |
| `rN_ext_sigma_t_m` / `rN_ext_sigma_rot_deg` | `0.03`/`4°` | the extrinsic's own 1σ (from the solve), folded into each point's covariance. This is what calibrates the cross-radar χ² to ≈3 on real data — omit it and χ² reads ~2× high (the few-degree / few-cm extrinsic error is unmodelled) |
| `valid_chi2_max` / `stats_window` | `6.0` / `300` | VALID threshold on the windowed mean χ² (2× the ideal 3), and how many recent matches the reported χ²/shrink average over |

**Colours in the overlay** — point/ellipse colour is **depth** (JET: red=near → blue=far over `[min_range,max_range]`), *not* radar identity. Radar identity is the **fill**: filled = 2-radar confirmed (fused), hollow = single-radar. The ellipse is the point's **1σ** uncertainty. The `VALIDATE:` HUD line is green when VALID, orange when CHECK.

---

## Coordinate conventions

- radar: `X=forward, Y=left, Z=up` (automotive)
- camera: `X=right, Y=down, Z=forward` (optical/pinhole)

Raw radar points go straight into Kabsch, so the solved `R` absorbs the **full**
~90° frame difference **and** the mounting rotation — there is no separate remap
step to get wrong. A large-looking `R` (≈90°) is therefore expected and correct.
`T_cam_radar` (and its inverse) in the YAML already map raw radar points into the
camera frame; the emitted `static_transform_publisher` command is ready to use.
