# Standardized Radar ↔ Camera Calibration Process

One identical procedure for **every** radar. You **measure first** (rig offset +
extrinsic prior), seed those as priors, then collect one diverse pose set and
solve. There is **no per-radar improvisation** and **no no-prior bootstrap round**
— measuring the extrinsic up front replaces it.

The only things that differ between radars are the **measured numbers** (each
radar sits in a different place — unavoidable) and which translation axis is
"blind." Everything else — parameters, steps, gates — is frozen below.

---

## 0. Measure BEFORE any capture

### 0a. The rig offset — measure **ONCE**, reuse for every radar
The board + corner reflector is one rigid rig. The **apex offset** (reflector apex
position in the *board* frame) is a property of the *rig*, not the radar, so you
measure it a single time and reuse it for radar1, radar2, … It also becomes the
cross-check between radars (all must recover the same offset).

Measure from the **board origin** (first inner chessboard corner) to the
**reflector apex**, in board axes (+x along squares_x, +y along squares_y, +z out
of the board toward the camera):
```
reflector_offset_x, reflector_offset_y, reflector_offset_z
```

### 0b. The extrinsic prior — measure **per radar**
Two numbers per radar, taken before collecting:

1. **Translation** `prior_t_xyz` = radar position in the **ZED left optical frame**,
   tape-measured from the **left lens center** (X = right, Y = down, Z = forward).
   *This is the key measurement.* It seeds the solve into the right basin AND
   anchors the radar's **blind axis** — the one translation direction the radar's
   weak angular resolution (elevation for an upright IWR6843) cannot observe. The
   data pins the other two axes; only the blind one relies on this tape.

2. **Rotation** `prior_rpy_deg` = the radar's rough orientation in the camera frame
   from the nominal mounting (xyz euler). This only needs to be *roughly* right —
   rotation IS observable, so the data refines it. A loose prior is fine.

### 0c. Range scale — verify per radar (once, early)
On the first ~10 captures the solve prints `cam_r = a·radar_r + b`. **Tune
`radar_range_scale` until the slope `a ≈ 1.00`** — that is the whole criterion.
The printed `set radar_range_scale=…` value already folds in the scale you have
applied, so just set it (don't add to the old value); re-check that `a` landed on
1.00 and freeze it. These IWR6843s read range slightly *long*, so the scale is
typically **below 1** (~0.96), not above. Leave bias at 0 unless it persists with
wide range spread. A residual range error doesn't show up in the residual — it
leaks into `t_z` (depth), so getting `a ≈ 1` is what keeps the radar from being
placed too far forward/back.

> **Why measurement-first?** A single-reflector radar has one translation axis it
> physically cannot see (its weak angular direction). No pose set fixes that — it
> must come from a tape measure. Measuring the extrinsic up front supplies that
> anchor AND removes the need for a separate no-prior bootstrap round. The radar
> then *confirms your tape on the two axes it CAN see* (they'll agree to a few cm),
> which is exactly what tells you the tape is trustworthy on the blind third axis.

---

## 1. Command = camera profile + frozen block

The *process* is camera-agnostic. Only the **camera profile** (which script + which
image/info topics + which optical frame) differs; everything else is frozen.

### 1a. Camera profile — pick ONE (the only camera-specific lines)

**ZED** (already-rectified feed — `image_rect_color`):
```bash
ros2 run wicoms_utils radar_camera_calib_static --ros-args \
  -p image_topic:=/zed/zed_node/left/image_rect_color \
  -p info_topic:=/zed/zed_node/left/camera_info \
  -p parent_frame:=zed_left_camera_optical_frame \
```

**Arducam** (RAW/unrectified feed — undistorted in-node; separate script, `rectify_image` on):
```bash
ros2 run wicoms_utils radar_camera_calib_arducam --ros-args \
  -p image_topic:=/arducam/image_raw \
  -p info_topic:=/arducam/camera_info \
  -p parent_frame:=arducam_optical_frame \
```
The Arducam script *is* the static profile with `rectify_image:=true` baked in — it
undistorts each frame from `camera_info`, then runs the identical solver/HUD/gates.
Set your real Arducam topic names and optical-frame id. (If you ever point it at an
already-rectified feed, rectification auto-disables and it behaves like the ZED.)

### 1b. FROZEN block (camera-agnostic — identical every radar, both cameras)

```bash
  -p pc_field_snr:=intensity -p pc_field_doppler:=doppler \
  -p squares_x:=4 -p squares_y:=4 -p square_len:=0.12 -p marker_len:=0.09 \
  -p dictionary:=DICT_4X4_50 -p min_corners:=4 \
  -p min_range:=0.5 -p max_range:=2.5 -p range_gate_margin_m:=0.5 \
  -p sigma_range_m:=0.05 -p sigma_az_deg:=3.0 -p sigma_el_deg:=8.0 \
  -p solve_offset:=true -p offset_prior_sigma_m:=0.05 \
  -p use_extrinsic_prior:=true -p prior_t_sigma_m:=0.05 -p prior_rot_sigma_deg:=30.0 \
  -p select_by:=cluster -p cluster_eps:=0.20 -p min_cluster_size:=1 -p cluster_strict:=false \
  -p stable_std:=0.02 -p stable_std_radar:=0.10 -p min_snr:=100.0 \
  -p capture_mode:=auto -p min_baseline:=0.12 -p min_points:=25 \
  -p show_window:=true -p show_diversity_hud:=true
```

The three prior *widths* are frozen on purpose:
- `offset_prior_sigma_m:=0.05` — keeps the offset near your measured rig value while
  letting the data refine it (offset_z only refines with high tilt — see step 2).
- `prior_t_sigma_m:=0.05` — a **uniform moderate** translation prior. This is what
  makes it standardized: it automatically **dominates the blind axis** (where the
  data is uninformative, the tape wins) and is automatically **overridden on the
  two observable axes** (where the data is tight, ~3–4 cm, the data wins and
  happens to agree with your tape). You never have to identify which axis is blind.
- `prior_rot_sigma_deg:=30` — loose; the data pins rotation.

## 2. PER-RADAR inputs (fill from step 0 — the only things that change)

```bash
-p radar_topic:=/radarN/radar/points_all \
-p child_frame:=radarN_link -p radar_name:=radarN \
-p radar_range_scale:=<from 0c> \
-p reflector_offset_x:=<0a> -p reflector_offset_y:=<0a> -p reflector_offset_z:=<0a> \
-p prior_t_xyz:="[<x>,<y>,<z>]"      # 0b tape \
-p prior_rpy_deg:="[<r>,<p>,<y>]"    # 0b mounting
```

## 3. The run — same four steps every radar

**Step 1 — Setup & background.** Mount, confirm topics publishing. Rig **out** →
`~/background`. In the debug view confirm the apex reticle **and** the magenta
radar dot both sit on the reflector. *(Also read the range fit → set 0c.)*

**Step 2 — Collect one diverse pose set.** `capture_mode:=auto` captures each held
pose. Collect **25–30 poses until all six diversity HUD bars are green**
(pitch 40° / roll 30° / yaw 40°, az 40° / el 15° / range 0.30 m). Keep the
reflector **aimed at the radar** (SNR ≥ 100), board at **0.8–1.8 m**, and include
**3–4 close-range (~0.6–0.7 m) high pitch/roll tilt** poses (these make
`offset_z` observable). Ignore `no board` / `REJECTED-ALL` chatter between poses.

**Step 3 — Solve (automatic).** The solve refreshes after every capture. The
frozen moderate priors do the right thing with no fiddling: tape anchors the blind
axis, data owns the rest, rotation is data-driven.

**Step 4 — Validate & save.** Pass **all** of section 4, then `~/save` (writes
YAML + JSON + `_session.json`). Move the session file into `sessions/` as
`<date>_zed_<radar>_session.json`.

## 4. "Done" gate (identical every radar)

| gate | threshold |
|---|---|
| residual / LOO | ≈ 1σ, LOO ≈ residual |
| condition number | ≲ 5 |
| rot 1σ | ≲ 4° (worst axis) |
| all six HUD bars | green |
| `|t|` observable axes vs tape | agree within ~5 cm |
| **live overlay** | magenta dot glued to the reflector across the whole FoV, **including up/down** |
| offline re-solve | `solve_from_poses_joint.py` reproduces to < 1° / < 40 mm |

The overlay is the final, tape-free judge — if the dot tracks the reflector
everywhere (near/far, left/right, **high/low**), the extrinsic is correct,
blind-axis anchor included.

**Not a gate:** the large 3-D **RMS** on the blind axis is the radar's random
angular noise (weak elevation / rolled mount). It averages out in fusion. Judge by
signed bias, per-DOF 1σ, and the overlay — never RMS.

## 5. Multi-radar cross-check (when you calibrate more than one)

The apex offset must match across radars (rig property). After each radar, compare
its solved `apex_offset_in_board_m` to the others — agreement on the observable
in-plane axes (X, Y) within combined 1σ is the strongest evidence both extrinsics
are correct. Do this only after each radar independently passes section 4.

---

## What changed vs the old flow (and why)

- **Measurement-first replaces the no-prior "Round 1."** You always measure the
  extrinsic prior, so there is never a cold-start bootstrap. One path for all radars.
- **Uniform moderate translation prior** handles the blind axis automatically — no
  per-radar reasoning about which axis to pin.
- **Offset measured once as a rig property**, reused and cross-checked across radars.
- Offline re-solve is `solve_from_poses_joint.py` (jointly refines the offset), so a
  saved session can be re-solved / audited at any prior width without recollecting.
- **Camera-agnostic:** the same process runs on the **ZED** (`radar_camera_calib_static`)
  or an **Arducam** (`radar_camera_calib_arducam`, raw feed undistorted in-node). Only
  the camera profile in §1a changes — the frozen block, steps, and gates are identical.
