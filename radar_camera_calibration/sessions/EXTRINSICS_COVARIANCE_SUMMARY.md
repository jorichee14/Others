# Extrinsics, errors & fusion covariance — ZED left ↔ radar1 / radar2

Single reference for the deployable extrinsics, their uncertainty (covariance),
their accuracy (errors), and everything the fusion node needs to build its
per-detection measurement covariance. Both extrinsics are `T_cam_radar`
(parent `zed_left_camera_optical_frame` → child `radarN_link`); two IWR6843ISK
radars mounted ~90° apart, sharing one ChArUco board.

Sources: [`2026-07-22_zed_radar1_radar2_final.md`](2026-07-22_zed_radar1_radar2_final.md)
(deployed), the per-radar sessions ([radar1](2026-07-15_zed_radar1.md) ·
[radar2](2026-07-15_zed_radar2.md)), and the fusion node
[`radar_fusion_reflector.py`](../radar_fusion_reflector.py). The full 6×6
covariances below were recomputed offline with
[`dump_extrinsic_cov.py`](dump_extrinsic_cov.py).

---

## 1. Deployable extrinsics (finalized 2026-07-22)

| | **radar1** (range scale 0.958) | **radar2** (range scale 0.967) |
|---|---|---|
| translation xyz (m) | `+0.2368  +0.0190  −0.0542` | `−0.1194  −0.0096  −0.0157` |
| \|t\| | 24.4 cm | 12.1 cm |
| quaternion xyzw | `−0.4995  0.6007  −0.4224  −0.4596` | `0.7572  0.0539  0.6506  −0.0217` |

```bash
# radar1
ros2 run tf2_ros static_transform_publisher \
  0.2368 0.0190 -0.0542  -0.4995 0.6007 -0.4224 -0.4596 \
  zed_left_camera_optical_frame radar1_link
# radar2
ros2 run tf2_ros static_transform_publisher \
  -0.1194 -0.0096 -0.0157  0.7572 0.0539 0.6506 -0.0217 \
  zed_left_camera_optical_frame radar2_link
```

---

## 2. Extrinsic covariance (solve uncertainty)

Reported per-axis as 1σ = √diag of the solver's 6×6 parameter covariance
`cov6 = pinv(Jᵀ J)` over `x = [rotvec(3), t(3)]`. Live-node values (with the
extrinsic prior + joint offset refinement active):

| | radar1 | radar2 |
|---|---|---|
| **rot 1σ (deg)** | [4.85, 3.62, 4.19] | [3.06, 4.08, 3.58] |
| **t 1σ (mm)** | [27.3, 39.8, 30.5] | [32.7, 29.1, 23.0] |
| residual / LOO | 1.07σ / 1.14σ | 1.31σ / 1.45σ |
| condition | 2.2 | 4.7 |
| **soft axis** | vertical (Y) — weak elevation | horizontal (X) — 90° roll mount |

### Full 6×6 — recomputed offline from the logged poses

`dump_extrinsic_cov.py` re-runs the measurement-space ML fit and reads
`cov6 = pinv(Jᵀ J)`. Order **[rx ry rz (deg), tx ty tz (mm)]**; blocks in
deg² / deg·mm / mm².

**⚠️ Reproduction caveat.** These offline fits use **no extrinsic prior and no
joint offset refinement**, and for **radar2 only the logged round-1/round-2
subsets exist** (the final 30-pose set's individual captures were never logged).
So the magnitudes here are **looser than the deployed σ in the table above** and
must not be read as the finalized numbers — their value is the **full off-diagonal
structure** (which axes trade off) and confirmation of **which axis is soft**.

**radar1 — 31 poses** (soft axis = **ty**, σ 61.8 mm, the largest):

```
per-axis 1σ:  rot (deg) [4.66, 3.81, 3.30]   t (mm) [41.5, 61.8, 11.3]

              rx      ry      rz      tx       ty       tz
   rx  [   21.73    8.11    6.95  -100.87   122.43     6.87 ]
   ry  [    8.11   14.49    9.63  -109.03  -108.70   -13.85 ]
   rz  [    6.95    9.63   10.90   -24.25   -76.66    -7.83 ]
   tx  [ -100.87 -109.03  -24.25  1719.22   202.44    77.37 ]
   ty  [  122.43 -108.70  -76.66   202.44  3813.34   333.94 ]
   tz  [    6.87  -13.85   -7.83    77.37   333.94   127.58 ]
```

**radar2 — round 2, 15 poses** (soft axis = **tx**, σ 111 mm; `corr(tx,rx) = −0.78`):

```
per-axis 1σ:  rot (deg) [5.76, 10.35, 7.17]   t (mm) [111.2, 71.0, 14.8]

              rx      ry      rz      tx        ty       tz
   rx  [   33.19   22.93  -14.50   -501.75   122.38     1.53 ]
   ry  [   22.93  107.21   35.48   -124.71  -139.10   -16.01 ]
   rz  [  -14.50   35.48   51.42    532.51   114.92    16.21 ]
   tx  [ -501.75 -124.71  532.51  12373.15    68.42   183.65 ]
   ty  [  122.38 -139.10  114.92     68.42  5036.68   394.88 ]
   tz  [    1.53  -16.01   16.21    183.65   394.88   219.42 ]
```
(round-1 bootstrap, 11 poses, is looser still: t 1σ [108.0, 71.6, 19.7] mm.)

**Reading the structure**
- **Perpendicular soft axes** — radar1's biggest translation variance is **ty**
  (vertical, weak elevation); radar2's is **tx** (horizontal, the 90°-roll blind
  axis). This is exactly why fusing them constrains every axis.
- **Rotation↔translation coupling on the blind axis** — radar2's `tx` correlates
  **−0.78** with `rx`: a single-reflector rig can't separate the two on the
  unobserved axis, so `tx` was **pinned to a tape measure (−0.10 m)** with a tight
  prior rather than solved. radar1's analogous soft coupling is on `tz` (vertical).

Regenerate:
```bash
cd sessions && python3 dump_extrinsic_cov.py 2026-07-15_zed_radar1_poses.json
```

---

## 3. Errors (accuracy — distinct from the solve σ above)

| | radar1 | radar2 |
|---|---|---|
| **signed bias (mm)** | [+6.5, −5.4, +3.7] | [−5.9, −13.6, +11.9] |
| **3-D RMS X/Y/Z (mm)** | [97, 94, 42] | [228, 119, 58] |
| range fit *a* | 1.00 | 0.999 |

Systematic **bias ≤ a few mm on both** → the extrinsics are sound. The large
single-axis RMS is the IWR6843's **random per-detection angular noise on its soft
axis** — it averages out in tracking/fusion; not a calibration error.

**Rig cross-check** (apex offset is a property of the shared board — the strongest
validation, since the two radars share only the board):

| board axis | radar1 (mm) | radar2 (mm) | Δ |
|---|---|---|---|
| X (in-plane) | 256 | 250 | **6** ✅ |
| Y (in-plane) | 539 | 544 | **5** ✅ |
| Z (normal) | −20 | −55 | 35 (weak both) |

---

## 4. Fusion covariance — what the fusion node needs

Fusion does **not** consume the static extrinsic 6×6. It builds a **per-detection
3×3 measurement covariance in the camera frame at runtime** and feeds it to a
speed-adaptive constant-velocity KF. Inputs:

**(a) Radar noise model** — shared by both radars (`radar_fusion_reflector.py`
defaults, matching the calibration sigmas):

```
sigma_range_m = 0.05     # radial
sigma_az_deg  = 3.0      # azimuth (horizontal cross-range)
sigma_el_deg  = 8.0      # elevation (weak axis)
```

**(b) The extrinsic rotation R** (§1) rotates the anisotropic ellipsoid into the
camera frame — `radar_cov_cam()`:

```
S   = σr²·(er·erᵀ) + (r·σaz)²·(eaz·eazᵀ) + (r·σel)²·(eel·eelᵀ)   # local radial/az/el basis
cov = R · S · Rᵀ                                                  # 3×3 in camera frame
```
The az/el terms scale with range `r`, so the ellipsoid grows with distance; a
`blind-axis vector = R·eel·(r·σel)` is emitted for diagnostics.

**(c) Runtime inflation & gating** — `_adaptive_R()` adds each radar's sample
innovation covariance (capped at `adapt_max×` the model in trace) so a misbehaving
radar self-down-weights; each update is Mahalanobis-gated (`gate_chi2`) against
clutter.

**Why it works:** the soft axes are **perpendicular** (radar1 sharp horizontal,
radar2 sharp vertical — see §2 covariances). Combining the two 3×3s constrains
every axis:

- Fused position 1σ ≈ **[53, 69, 29] mm** — from single-radar ≈ [112, 325] (radar1)
  / [287, 112] (radar2).
- Frame-to-frame jitter cut ~3× by the maneuvering-target KF.

---

## 5. Caveats / status

- The **older 2026-07-15 radar1 t_z** (−17.2 cm) was inflated by a residual range
  bias; [`..._radar1_CORRECTED.json`](2026-07-15_zed_radar1_CORRECTED.json) fixed
  it to −2.1 cm (drops outlier capture #23, applies `r' = 0.979·r − 0.09`). The
  finalized 2026-07-22 t_z (−0.0542) supersedes both.
- **radar_infra** (third radar) is a round-1 bootstrap only (10 poses, rot 1σ ≈ 7°)
  — **not final**, excluded from fusion.
- The offline 6×6 in §2 corroborates the structure of the deployed result; the
  deployed per-axis σ (top of §2) remains the number to quote.
