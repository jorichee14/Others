# Finalized calibration — ZED left ↔ radar1 & radar2 (2026-07-22)

Standardized measurement-first process (see `CALIBRATION_PROTOCOL.md`), static
profile, range scale tuned to `a≈1`. Both extrinsics are `T_cam_radar`
(parent `zed_left_camera_optical_frame` → child `radarN_link`).

## Deployable transforms

```bash
# radar1  (radar_range_scale = 0.958)
ros2 run tf2_ros static_transform_publisher \
  0.2368 0.0190 -0.0542  -0.4995 0.6007 -0.4224 -0.4596 \
  zed_left_camera_optical_frame radar1_link

# radar2  (radar_range_scale = 0.967)
ros2 run tf2_ros static_transform_publisher \
  -0.1194 -0.0096 -0.0157  0.7572 0.0539 0.6506 -0.0217 \
  zed_left_camera_optical_frame radar2_link
```

## Errors

| | radar1 | radar2 |
|---|---|---|
| t (m) | [+0.2368, +0.0190, −0.0542] | [−0.1194, −0.0096, −0.0157] |
| \|t\| | 24.4 cm | 12.1 cm |
| quat xyzw | [−0.4995, 0.6007, −0.4224, −0.4596] | [0.7572, 0.0539, 0.6506, −0.0217] |
| rot 1σ (deg) | [4.85, 3.62, 4.19] | [3.06, 4.08, 3.58] |
| t 1σ (mm) | [27.3, 39.8, 30.5] | [32.7, 29.1, 23.0] |
| 3-D RMS (mm) X/Y/Z | [97, 94, 42] | [228, 119, 58] |
| signed bias (mm) | [+6.5, −5.4, +3.7] | [−5.9, −13.6, +11.9] |
| residual / LOO | 1.07σ / 1.14σ | 1.31σ / 1.45σ |
| range fit a | 1.00 | 0.999 |
| soft axis | vertical (Y) — weak elevation | horizontal (X) — rolled 90° mount |

Bias ≤ a few mm on both; the large single-axis 3-D RMS is the radar's random
angular noise on its soft axis (averages out in fusion), not a calibration error.

## Rig cross-check (apex offset — a property of the shared board)

| board axis | radar1 (mm) | radar2 (mm) | Δ |
|---|---|---|---|
| X (in-plane) | 256 | 250 | **6** ✅ |
| Y (in-plane) | 539 | 544 | **5** ✅ |
| Z (board normal) | −20 | −55 | 35 (weak axis both) |

Two radars mounted ~90° apart, sharing only the board, recovered the same apex
offset to **5–6 mm on X/Y** — independent evidence both extrinsics are correct.

## Notes
- radar1 top UP; radar2 rolled 90° CW about boresight (top points RIGHT →
  `prior_rpy_deg [90,-90,90]`). Soft axes are perpendicular, so fusing the two
  constrains every axis.
- Offline reproduced from the logged pose sets with `solve_from_poses_joint.py`.
