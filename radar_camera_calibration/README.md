# Radar ↔ Camera Extrinsic Calibration (ChArUco board + trihedral reflector)

Estimate the rigid extrinsic **`T_cam_radar`** between a camera and a radar using
a single rigid target: a **ChArUco board with a trihedral corner reflector**
bolted to it at a known, fixed offset.

- `radar_camera_calib.py` — the calibration tool (this task).
- `general_charuco.py` — the original **camera-to-camera** tool you started from,
  kept for reference. It does not apply to radar (a radar return has no
  orientation), but the ChArUco detection code is shared in spirit.

---

## Why a corner reflector?

A radar doesn't see a checkerboard — it sees a sparse set of range/azimuth
(/elevation) detections. A **trihedral corner reflector** is engineered to be
the single brightest return in the scene, so the radar reliably reports **one
point**: the reflector's apex.

That apex is rigidly fixed to the ChArUco board, so the **camera** can also
locate it:

```
camera:  detect board → T_cam_board (6-DOF) → p_cam = T_cam_board · apex_in_board
radar :  strongest gated return                → p_radar   (a 3-D point)
```

Each pose gives one **corresponding 3-D point pair** `(p_cam, p_radar)`.

## Why you must move the rig

A single point has no orientation, so **one view cannot recover a 6-DOF
transform.** Collect `N ≥ 3` non-collinear pairs by moving the rig around the
shared field of view, then solve the rigid point-set registration
(Kabsch / Umeyama, scale fixed at 1):

```
min_{R,t}  Σ_i ‖ R·p_radar^i + t − p_cam^i ‖²      ⇒   T_cam_radar
p_cam = R · p_radar + t          (X = T_cam_radar maps a radar point into the camera frame)
```

**Pose diversity is everything.** Spread captures across **range, azimuth, and
height**. If all captures lie in a plane or a line the out-of-plane rotation is
unobservable — the tool detects this (singular-value check) and warns.
Aim for **8–15 well-spread poses**.

---

## Measuring `apex_in_board` (the one number that matters most)

`reflector_offset_{x,y,z}` is the vector from the **ChArUco board origin** to the
**reflector apex**, expressed **in the board frame**. The whole calibration is
biased by exactly this error, so measure it carefully.

Board frame (OpenCV ChArUco convention):
- **origin** = the first inner chessboard corner (top-left of the board grid),
- **+x** along the `squares_x` direction,
- **+y** along the `squares_y` direction,
- **+z** out of the board plane, toward the camera side.

Measure the apex position with a ruler/caliper along those three axes. If the
reflector sits, say, 3 cm to the right of the origin, 10 cm below it, and its
apex stands 4 cm proud of the board surface:

```
reflector_offset_x:  0.03
reflector_offset_y:  0.10
reflector_offset_z:  0.04
```

**Verify visually before capturing:** enable `debug_image` (on by default) and
view `/radar_camera_calib/debug_image`. The tool draws the board axes and a dot
at the projected apex. That dot must land on the real reflector apex in the
image. If it's off, fix the offset (usually a sign). This 30-second check saves
the whole calibration.

---

## Inputs / outputs

**Subscribes**
| topic (param) | type | purpose |
|---|---|---|
| `image_topic` | `sensor_msgs/Image` | camera image (ChArUco detection) |
| `info_topic` | `sensor_msgs/CameraInfo` | intrinsics K, D |
| `radar_topic` | `sensor_msgs/PointCloud2` **or** `radar_msgs/RadarScan` | radar detections |
| `~/capture` `~/solve` `~/reset` `~/save` | `std_msgs/Empty` | manual control |

**Publishes**
- static TF `parent_frame → child_frame` (camera optical → radar), if `publish_tf`.
- `debug_image_topic` — annotated image (axes + projected apex), if `debug_image`.
- files `extrinsic_<cam>__<radar>.yaml` and `.json` with `T_cam_radar`, its
  inverse `T_radar_cam`, RMS residual, capture count, and a planar-warning flag.

---

## Usage

Install deps (ROS 2 + OpenCV with aruco + these):

```bash
pip install -r requirements.txt        # numpy, scipy, opencv-contrib-python
# plus ROS 2: rclpy, cv_bridge, message_filters, tf2_ros, sensor_msgs_py
# for RadarScan input: the radar_msgs package
```

Run (edit params to your rig):

```bash
python3 radar_camera_calib.py --ros-args \
  -p image_topic:=/zed/zed_node/left/image_rect_color \
  -p info_topic:=/zed/zed_node/left/camera_info \
  -p radar_topic:=/radar/points \
  -p radar_type:=pointcloud2 \
  -p pc_field_intensity:=rcs \
  -p squares_x:=9 -p squares_y:=7 -p square_len:=0.020 -p marker_len:=0.015 \
  -p dictionary:=DICT_4X4_50 \
  -p reflector_offset_x:=0.03 -p reflector_offset_y:=0.10 -p reflector_offset_z:=0.04 \
  -p parent_frame:=zed_left_camera_optical_frame \
  -p child_frame:=radar_link \
  -p capture_mode:=auto -p min_points:=6 -p min_baseline:=0.15
```

Then:
1. Watch `/radar_camera_calib/debug_image` — confirm the apex dot is on the reflector.
2. Move the rig to a spot in view of **both** sensors, hold still ~1 s. In
   `auto` mode it captures once stable and far enough from prior captures.
   (Or `ros2 topic pub -1 /radar_camera_calib/capture std_msgs/msg/Empty {}`.)
3. Repeat across a spread of ranges / angles / heights (8–15 poses).
4. The extrinsic solves and refines after each capture and is saved on lock /
   `~/save` / Ctrl-C. Force a solve any time with `~/solve`; clear with `~/reset`.

---

## Key parameters

| param | default | meaning |
|---|---|---|
| `radar_type` | `pointcloud2` | `pointcloud2` or `radarscan` |
| `pc_field_x/y/z` | `x`/`y`/`z` | PointCloud2 coordinate field names |
| `pc_field_intensity` | `intensity` | intensity/RCS field for picking the strongest return (`''` disables) |
| `pick_by_intensity` | `true` | pick strongest return; else nearest. A prior estimate always gates by proximity |
| `min_range`/`max_range` | `0.3`/`20` m | radar range gate (clutter rejection) |
| `gate_radius` | `1.0` m | once an extrinsic exists, radar pt must be within this of the predicted apex |
| `capture_mode` | `auto` | `auto` (stable + moved) or `manual` (`~/capture`) |
| `stable_window` | `12` | frames averaged per pose |
| `stable_t_std` | `0.004` m | jitter threshold to call the rig "still" |
| `min_baseline` | `0.12` m | minimum move between auto-captures |
| `min_points` | `4` | captures before the first solve (≥3 required) |
| `max_reproj_px` | `1.5` | reject bad board detections |
| `reflector_offset_{x,y,z}` | `0` | **apex offset in board frame — measure this!** |

---

## 2-D radar (range + azimuth only)

If your radar has no elevation, all points have `z ≈ 0`. The tool auto-detects
this (planar singular value) and warns: with in-plane points you can recover the
in-plane rotation (yaw) and the in-plane translation, but the out-of-plane
rotation (pitch/roll) and the height offset are unobservable from point
correspondences alone. Options:
- Mount the reflector at several **different heights** so the camera-side points
  span z even if the radar's don't — this constrains the fit through the known
  board geometry.
- Or fix the unobservable DOFs from CAD / a spirit level and only trust yaw +
  planar translation from this tool.

---

## Sanity checks

- **RMS residual** printed on every solve should fall to a few mm–cm as poses
  accumulate. A residual that stays large ⇒ wrong `apex_in_board`, wrong radar
  field/units, or bad time sync (`sync_slop`).
- **Singular values** of the radar point spread are printed; the smallest being
  near zero ⇒ poses too planar/collinear, add diversity.
- Reproject: with `publish_tf` on, `ros2 run tf2_ros tf2_echo parent child`
  should match the saved YAML.
