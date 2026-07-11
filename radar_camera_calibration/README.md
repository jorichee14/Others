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
pairs, and solve the rigid registration (Kabsch/Umeyama, scale = 1):

```
min_{R,t} Σ ‖ R·p_radar^i + t − p_cam^i ‖²   ⇒   X = T_cam_radar
p_cam = R · p_radar + t      (X maps a radar point into the camera frame)
```

**Pose diversity is everything** — spread captures across range, azimuth, and
**height**. Collinear/planar poses make out-of-plane rotation unobservable; the
tool detects this (condition number + planar singular value) and warns. Aim for
**8–15 well-spread poses**.

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

---

## Measuring `apex_in_board` (the one number that matters most)

`reflector_offset_{x,y,z}` is the vector from the **ChArUco board origin** to the
**reflector apex**, in the board frame. The whole calibration is biased by this
error — measure it carefully.

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

## Usage

```bash
pip install -r requirements.txt        # numpy, scipy, opencv-contrib-python
# plus ROS 2: rclpy, cv_bridge, sensor_msgs_py, message_filters, tf2_ros
```

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
| mean reproj | `val_pass_reproj_px` (20 px) | radar-predicted apex lands on the visual apex |
| mean 3-D | `val_pass_3d_mm` (150 mm) | 3-D agreement |
| max per-axis bias | `val_pass_bias_mm` (50 mm) | **signed** residual — a non-zero mean on an axis is a systematic error RMS would hide |
| LOO-CV RMS | ≈ in-sample RMS | honest generalisation; ≫ in-sample ⇒ overfit / too few / clustered |
| \|t\| | `measured_baseline_m` ± `baseline_tol_m` | tape-measure the radar↔camera distance for metric ground truth |
| condition number | < ~200 | pose spread; high ⇒ add diversity |
| live overlay | — | dots track the reflector everywhere in the FoV |

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
| `capture_mode` | `auto` | `auto` or `manual` (`~/capture`) |
| `stable_window`/`stable_std` | `12`/`0.01 m` | per-pose stability gate |
| `min_baseline`/`min_points` | `0.15 m`/`6` | capture spacing / count |
| `reflector_offset_{x,y,z}` | `0` | **apex offset in board frame — measure this!** |
| `measured_baseline_m` | `-1` (off) | tape-measured `|t|` for the baseline check |

---

## Coordinate conventions

- radar: `X=forward, Y=left, Z=up` (automotive)
- camera: `X=right, Y=down, Z=forward` (optical/pinhole)

Raw radar points go straight into Kabsch, so the solved `R` absorbs the **full**
~90° frame difference **and** the mounting rotation — there is no separate remap
step to get wrong. A large-looking `R` (≈90°) is therefore expected and correct.
`T_cam_radar` (and its inverse) in the YAML already map raw radar points into the
camera frame; the emitted `static_transform_publisher` command is ready to use.
