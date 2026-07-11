# Sprint Update — Radar ↔ Camera Extrinsic Calibration

**Focus:** extrinsic calibration between a TI **IWR6843ISK** mmWave radar and a
**ZED** camera, using a **ChArUco board + trihedral corner reflector** rig.
**Tool:** `radar_camera_calib.py` (packaged as `wicoms_utils/radar_camera_calibration`).

---

## Goal

Recover `T_cam_radar` (the 6-DOF pose of the radar in the camera optical frame)
so radar detections can be projected into the camera / fused. One rigid target
carries both cues: the camera localizes the **ChArUco board**, the radar localizes
the **corner reflector** bolted to it at a fixed offset.

---

## Hardware / software context

| Item | Value |
|---|---|
| Radar | IWR6843ISK, **3D People-Counting** firmware (group tracker) |
| Radar topic | `/radar1/radar/points_all` (PointCloud2), ~10 Hz, 40 pts/frame |
| Radar fields | `x, y, z, doppler, intensity` (SNR field is **`intensity`**, not `snr`) |
| Radar frame | `radar1_link`; x/y horizontal, **z = elevation**; range = ‖xyz‖ |
| Radar FoV | azimuth **±60°**, elevation **±20°** (`fovCfg -1 60 20`) |
| Range res | ~0.16 m (cfg `..._36230023`, preferred) vs 0.28 m (`..._36230670`) |
| Range bias | already compensated in-chip (`compRangeBiasAndRxChanPhase`) |
| Camera | ZED, `/zed/zed_node/left/image_rect_color` + `/left/camera_info` |
| Board | small: 20 mm squares / 15 mm markers, DICT_4X4_50 (hand-held) |

Confirmed from `TrackingModule.c`: point cloud uses `azimuth=atan2(x,y)`,
`elev=asin(z/range)`, and the tilt/height correction is a **no-op**
(`az=el=height=0`) → published points are the **raw sensor frame**. `/points_all`
is the **detection cloud**, not tracker centroids — correct for our use.

---

## Method (and why it evolved)

### v1 — plain rigid registration (rejected)
Collect corresponding 3-D points `(p_radar, p_cam)` and solve
`min ‖R·p_radar + t − p_cam‖²` (Kabsch/Umeyama). **Problem:** treats the radar
point as isotropically accurate. It is not — radar **range is precise (~cm)** but
**angle is poor (degrees)**, and cross-range error grows with range
(≈ range·σ_az → ~9 cm at 5 m for 1°). Isotropic Kabsch is biased.
*Monte-Carlo: 1.7°/120 mm.*

### v2 — measurement-space maximum likelihood (adopted)
Predict each radar measurement from the (accurate) camera apex and compare in the
radar's native coordinates, weighted by real noise:
```
q_pred = Rᵀ(p_cam − t);   residual = [(r−r̂)/σ_r, (az−âz)/σ_az, (el−êl)/σ_el]
```
Huber robust loss + iterative σ-gated outlier rejection; Kabsch for the initial
guess; auto-detect 2-D radar and drop elevation. *Monte-Carlo: 1.0°/38 mm — ≈1.6×
better rotation, ≈3× better translation than Kabsch; recovers a known transform to
0.34°/27 mm under 2 cm noise, rejects gross outliers (L2 → 52°, robust → ~1°).*

### v3 — joint apex-offset estimation (adopted)
The apex offset `a` (board→apex, board frame) enters as
`p_cam = R_board·a + t_board`. Because the **board rotates** between poses,
`R_board·a` changes pose-to-pose in a way constant `t` can't mimic → `a` is
**separable** from the extrinsic and can be **calculated**, not just measured.
Solved as MAP (free `a` regularized toward the measured value).
*Monte-Carlo: measured-well → no harm; measured-wrong (44 mm) → repaired to ~15 mm;
not measured (seed 0) → calculated to ~15 mm, extrinsic 0.38°/25 mm.*

**Governing equation (per pose):** `R_board,i·a + t_board,i = R·q_i + t`
→ solve 9 unknowns `(R, t, a)`; **board tilt** makes `a` observable, **radar-point
spread** makes `(R, t)` observable.

---

## What worked ✅

- **Measurement-space ML solver** — validated in simulation, clearly beats Kabsch;
  correctly down-weights noisy elevation on this radar (few elevation antennas).
- **Joint offset (MAP)** — recovers/repairs the offset from radar; reports its own
  1σ so you know if it was data-determined or fell back to the prior.
- **Robust rejection + covariance/observability readout** — per-DOF 1σ, LOO-CV,
  planar/2-D detection, pass/fail verdict, range scale/bias diagnostic.
- **SNR selection via `intensity`** — highest-SNR return = the trihedral, once the
  field name was corrected.
- **Camera-range gate ("highest SNR around the board")** — user's idea: gate radar
  points to those whose range matches the camera's board distance `|p_cam|`
  (rotation-invariant), then take max SNR. Rejected 4–7 m clutter directly from the
  camera cue, no background needed. Verified on live logs (0.44 m board → kept the
  0.85 m reflector, dropped 4.4/7.0/6.6 m).
- **Split stability threshold** (`stable_std` camera 10 mm vs `stable_std_radar`
  80 mm) — radar jitter is cm-dm; the old shared 10 mm gate meant captures never
  fired. After the split, auto-capture works.
- **Native display window** (`show_window`) with an apex reticle + status banner,
  plus the debug-image topic and live radar-cloud overlay.
- **Background subtraction** — pool N frames with the rig out; removes static
  clutter (incl. a strong fixed reflector at the same range as the target, which
  the range gate cannot).
- **Operational hardening** — radar watchdog, depth-unit guard, startup topic echo,
  ready-to-run `static_transform_publisher` line in the YAML/JSON output.

---

## What didn't work / problems found 🔧

| Symptom | Root cause | Fix |
|---|---|---|
| `InvalidParameterType … BOOL` on launch | `-p pc_field_y:=y` — YAML parses `y` as **true** | drop `pc_field_x/y/z` (defaults) or quote `:="'y'"` |
| No SNR selection | field is named **`intensity`**, not `snr` | `-p pc_field_snr:=intensity` |
| Never captured (`still? False`) | 10 mm stability gate applied to jittery radar | added `stable_std_radar` (0.08 m) |
| Board drops out (`no board n=0..7`) | 15 mm markers unreadable past ~0.4–1 m; also motion blur / out-of-frame | max ZED resolution, `min_corners:=6`, hold still, **bigger board needed** |
| Picking far clutter (4–7 m, high SNR) | global SNR-max grabs room reflectors | camera-range gate + optional background |
| Reflector return weak (SNR 300→20) | trihedral aimed away from radar (limited ~±30° beam) | re-aim reflector at radar; keep tilts ≤ ±30° |
| Two candidate clusters flip-flop | range gate margin too loose | `range_gate_margin_m:=0.5` |
| **Radar point never moved** (`[0.62,0,-0.11]` for 14 poses) while camera moved 30 cm | **reflector NOT attached to the board** — sitting on the desk while the board moved in hand (seen in the screencast) | **rigidly bolt the reflector to the board** |
| Degenerate solves (1σ ~12°/100 mm, negative range fit, reproj 1000s px, 7/14 rejected) | under-constrained: no radar-point spread + inconsistent/again-not-rigid rig | attach reflector, then collect many well-spread poses |

---

## Key findings / physics 📐

1. **Radar noise is anisotropic** — range precise, angle poor, cross-range error
   ∝ range. Must weight in measurement space, not isotropic XYZ.
2. **IWR6843ISK is 3-D but elevation is weak** (±20° FoV, ~2 el rows) → use a large
   `sigma_el_deg` (≈10°); out-of-plane DOFs are the least observable.
3. **Offset ≠ freely recoverable everywhere** — radar cross-range noise (~10 cm)
   exceeds the offset (~5 cm) at long range, so the offset is only observable at
   **close range + high board tilt**. MAP prior prevents it going wild.
4. **A corner reflector is directional** (~±30° usable beam) — tilting the board
   for diversity fights keeping the reflector lit. Compromise at ±20–30°, or put
   the reflector on a mount that keeps facing the radar.
5. **The rig must be ONE rigid body.** The single biggest failure this sprint was
   the reflector not being fixed to the board: the radar tracked a stationary
   object while the camera tracked the moving board → the correspondence was
   meaningless. **No amount of solver cleverness fixes a broken rig.**
6. **Two observability requirements, independent:** board **rotation** diversity →
   offset; radar-**point** spread (range/az/el) → extrinsic. Both must be satisfied.

---

## Current status 🚦

- Tooling: **working and validated in simulation.** Live pipeline runs end-to-end
  (detect → select → capture → solve → report → TF/YAML).
- Live calibration: **not yet converged.** Latest solve `1σ rot ~12°, 1σ t ~50–100 mm`,
  `reproj` huge, `7/14` rejected — **under-constrained**.
- Blockers, in priority order:
  1. **Rigidly attach the reflector to the board** (was separate — root cause).
  2. **Verify the radar point tracks the rig** and spans range + azimuth + elevation
     across captures (not near-constant).
  3. **Board too small** for range diversity — decodes only to ~0.4–1 m even at 2K.
     A ~100 mm-square board would unlock 3–5 m poses and tighten translation a lot.
  4. Sanity-check the ~2× camera-vs-radar range discrepancy (baseline vs a possible
     range scale; tool suggested `radar_range_scale≈0.51`).

---

## Recommended parameters (IWR6843ISK + ZED, offset unmeasured)

```
-p radar_topic:=/radar1/radar/points_all
-p pc_field_snr:=intensity -p pc_field_doppler:=doppler -p select_by:=snr
-p sigma_range_m:=0.03 -p sigma_az_deg:=1.5 -p sigma_el_deg:=10.0
-p radar_range_bias_m:=0.0 -p force_2d_radar:=false
-p min_range:=0.5 -p max_range:=2.5 -p range_gate_margin_m:=0.5
-p reflector_offset_x:=0.0 -p reflector_offset_y:=0.0 -p reflector_offset_z:=0.0
-p solve_offset:=true -p offset_prior_sigma_m:=0.10
-p min_corners:=6 -p stable_std_radar:=0.08 -p min_baseline:=0.10
-p parent_frame:=zed_left_camera_optical_frame -p child_frame:=radar1_link
-p show_window:=true
```
Convergence targets before trusting the result: `inliers ≈ all`, `1σ rot < ~2°`,
`1σ t < ~30 mm`, `LOO ≈ residual`, `apex ✓ data-determined`, `VERDICT ✔`, and the
live overlay dots stay glued to the reflector across the FoV.

---

## Next steps

1. **Bolt the reflector to the board** as one rigid panel; keep its opening toward
   the radar.
2. **Print a ~100 mm-square ChArUco board** to enable 1.5–5 m poses (range
   diversity → translation accuracy).
3. Recollect **~30 poses** spanning range + azimuth + elevation + moderate tilt;
   confirm captured `radar […]` values differ pose-to-pose.
4. Add a **diversity guard** to the tool: warn when captured radar points are
   near-constant / high condition number ("not enough radar spread").
5. Resolve the range-scale question with a tape-measure check.
6. Package an `iwr6843isk.launch.py` with the params above (typed → avoids the
   YAML-boolean trap).
