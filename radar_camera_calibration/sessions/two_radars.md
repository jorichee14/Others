# Two-radar calibration — radar1 vs radar2 (from the ZED frame)

Both IWR6843ISK radars calibrated to `zed_left_camera_optical_frame` on
**2026-07-15**. radar2 is mounted **rolled ~87° about its boresight** vs radar1,
so their weak angular axes are perpendicular (radar1 soft vertical, radar2 soft
horizontal). Per-radar sessions:
[radar1](2026-07-15_zed_radar1.md) · [radar2](2026-07-15_zed_radar2.md).

## TF transforms (parent `zed_left_camera_optical_frame` → child `radarN_link`)

| | translation xyz (m) | \|t\| | quaternion xyzw | rpy (deg) |
|---|---|---|---|---|
| **radar1** | `+0.2218  −0.0067  −0.1721` | 28.1 cm | `−0.5345  +0.5853  −0.4196  −0.4424` | `−175.91  −75.11  −98.34` |
| **radar2** | `−0.0999  −0.0124  −0.0011` | 10.1 cm | `+0.7882  −0.0406  +0.6121  +0.0499` | `+173.30  −75.68  −0.69` |

```bash
# ready to publish (args: x y z qx qy qz qw parent child)
ros2 run tf2_ros static_transform_publisher \
  0.2218 -0.0067 -0.1721  -0.5345 0.5853 -0.4196 -0.4424 \
  zed_left_camera_optical_frame radar1_link

ros2 run tf2_ros static_transform_publisher \
  -0.0999 -0.0124 -0.0011  0.7882 -0.0406 0.6121 0.0499 \
  zed_left_camera_optical_frame radar2_link
```

**Geometry check:** the relative rotation between the two radar frames is **87.2°**
about an axis 96 % aligned with radar1's X (boresight) — i.e. a pure ~90° **roll**,
confirming the orthogonal mount. The two radars sit **36.4 cm** apart in the ZED
frame (radar1 low-and-right at 28 cm, radar2 near the optical centre at 10 cm).

## Offset cross-check — do the two calibrations agree? ✅ (X, Y) / ⚠️ (Z)

The **apex offset** (corner-reflector position in the *board* frame) is a physical
property of the **rig**, not of either radar — so an independent calibration from
each radar must recover the *same* offset. This is the strongest available
cross-validation, because the two radars share nothing but the board.

| board-frame axis | radar1 (mm) | radar2 (mm) | diff | combined 1σ | agreement |
|---|---|---|---|---|---|
| **X** (in-plane) | +288 ± 23 | +280 ± 27 | **8 mm** | 35 | **0.23σ — MATCH** ✅ |
| **Y** (in-plane) | +614 ± 21 | +648 ± 23 | **34 mm** | 31 | **1.09σ — MATCH** ✅ |
| **Z** (board normal) | −9 ± 38 | −166 ± 32 | 157 mm | 50 | **3.16σ — mismatch** ⚠️ |

**Verdict:** the two independent calibrations agree on the offset to **8 mm / 34 mm**
in the two well-observed in-plane axes — well inside their combined 1σ. That two
radars mounted 87° apart, sharing only the board, land on the same X/Y offset is
strong evidence **both extrinsics are correct**.

The **Z (board-normal) component disagrees by 157 mm**, and this is *expected, not
a contradiction*: radar2's own solve flagged its offset-Z as **weakly observed**
(“moved 127 mm from measured — add close-range high-tilt poses”), with radar1's Z
better constrained (1σ 38 vs radar2 pulled by its prior). Observing the
board-normal offset needs strong **pitch/roll tilt** diversity; radar2's pose set
didn't tilt enough, so trust **radar1's Z (−9 mm)** as the rig value. Best combined
apex offset estimate: **X ≈ +284, Y ≈ +631, Z ≈ −9 mm** (board frame).

## Accuracy summary

| | signed bias XYZ (mm) | 3-D RMS XYZ (mm) | residual | soft axis |
|---|---|---|---|---|
| radar1 | −1, **−48**, −6 | 92, **269**, 89 | 1.03σ | **vertical (Y)** — weak elevation |
| radar2 | +0, +1, +1 | **215**, 150, 74 | 1.25σ | **horizontal (X)** — rolled 90° |

- **Systematic (bias) accuracy is a few mm** on both, except radar1's vertical
  (−48 mm) — the IWR6843's weak-elevation floor.
- The **large RMS is random per-detection angular noise**, not a solve error; it
  **averages out in tracking/fusion**.
- Crucially the **soft axes are perpendicular**: radar1 is sharp horizontally,
  radar2 sharp vertically. Fusing them constrains *every* axis → see
  [`radar_fusion_reflector.py`](../radar_fusion_reflector.py), which drops the
  fused 1σ to ≈ `[53, 69, 29]` mm (from ≈ `[112, 325]` / `[287, 112]` single-radar)
  and, as a smooth maneuvering-target tracker, cuts frame-to-frame jitter ~3×.

## Why radar2's t_x had to be measured, not solved

radar2's X (in the camera frame) is along its **blind horizontal axis**, so the
points barely constrain it — across rounds it floated −14 → +12 → −3.5 cm. It was
finally **pinned to a tape measure (−0.10 m)** with a tight prior
(`prior_t_sigma_m:=0.04`). This is the rotation↔translation coupling of a
single-reflector rig on the poorly-observed axis, and is why the extrinsic prior
exists. radar1's blind axis is vertical, so *its* t_z is the analogously soft one.
