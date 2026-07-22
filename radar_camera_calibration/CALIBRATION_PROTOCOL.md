# Standardized Radar ↔ Camera Calibration Protocol

A repeatable procedure for calibrating each IWR6843ISK radar to the ZED left
camera with the ChArUco + trihedral rig. It exists because the 2026-07-15 runs,
while successful, drifted between radars (sigmas, gates, priors, the reflector
offset seed) and improvised their rounds — and two sets were never logged, so
they can't be re-solved. This freezes everything that must not change and defines
numeric gates for what does.

**Order: calibrate `radar1` first, then `radar2`.** radar1's well-observed apex
offset is the cross-check reference for every later radar.

Tool: `radar_camera_calib.py` (static profile). See `README.md` for parameter
meanings, `sessions/` for the reference runs.

---

## Part A — FROZEN block (identical for every radar, every round)

Never change these between radars or rounds. They are the settings that produced
`residual 1.03σ / cond 2.2` on radar1.

```bash
-p image_topic:=/zed/zed_node/left/image_rect_color \
-p info_topic:=/zed/zed_node/left/camera_info \
-p pc_field_snr:=intensity -p pc_field_doppler:=doppler \
-p squares_x:=4 -p squares_y:=4 -p square_len:=0.12 -p marker_len:=0.09 \
-p dictionary:=DICT_4X4_50 -p min_corners:=4 \
-p min_range:=0.5 -p max_range:=2.5 -p range_gate_margin_m:=0.5 \
-p sigma_range_m:=0.05 -p sigma_az_deg:=3.0 -p sigma_el_deg:=8.0 \
-p reflector_offset_x:=0.284 -p reflector_offset_y:=0.631 -p reflector_offset_z:=-0.009 \
-p solve_offset:=true -p offset_prior_sigma_m:=0.05 \
-p select_by:=cluster -p cluster_eps:=0.20 -p min_cluster_size:=1 -p cluster_strict:=false \
-p stable_std:=0.02 -p stable_std_radar:=0.10 -p min_snr:=100.0 \
-p parent_frame:=zed_left_camera_optical_frame \
-p show_window:=true -p show_diversity_hud:=true
```

**Reflector offset is frozen to the combined rig value** (X +284, Y +631,
Z −9 mm; Z from radar1, the well-observed axis). The offset is a property of the
*rig*, not the radar — every radar must recover the same X/Y, so it is seeded, not
re-guessed. `solve_offset:=true` still lets it refine and doubles as a cross-check.

---

## Part B — Per-radar deltas (the ONLY things that change)

| param | **radar1** | radar2 |
|---|---|---|
| `radar_topic` | `/radar1/radar/points_all` | `/radar2/radar/points_all` |
| `child_frame` / `radar_name` | `radar1_link` / `radar1` | `radar2_link` / `radar2` |
| `radar_range_scale` | `1.039` *(re-verify in R1)* | `1.026` *(re-verify in R1)* |
| `cluster_apex_radius` | `0.40` | `0.50` |
| `gate_radius` | `0.50` | `0.60` |
| **blind translation axis** | **t_z (vertical)** | **t_x (horizontal)** — tape ≈ −0.10 m |

The blind axis is the soft translation DOF of a single-reflector rig: radar1's is
vertical (t_z), radar2's is horizontal (t_x, its rolled mount). It is pinned by
tape measure in Round 3, not solved.

---

## Part C — The four rounds (same procedure for every radar)

### Round 0 — Setup & background *(no captures)*
1. Mount rig; confirm `radar_topic`, `image_topic`, `info_topic` all publishing.
2. Rig **out** of scene → pool background (`~/background`).
3. In the debug view confirm **both**: the apex reticle sits on the reflector
   (offset sign correct) **and** the magenta radar dot lands on it.

✅ **Gate:** apex reticle + magenta radar dot both on the reflector; ≥1 clean
gated return in the `[gate]` line.

### Round 1 — Bootstrap, no prior
Add: `-p use_extrinsic_prior:=false -p min_points:=10`
1. Collect **10–15** poses, spread as wide as the geometry allows.
2. Read the range diagnostic `cam_r = a·radar_r + b`; set `radar_range_scale` so
   **a ≈ 1, b ≈ 0** (re-verifies the per-radar scale rather than trusting the
   stored value).
3. **Save the pose JSON** (mandatory — every round is logged).

✅ **Gate:** residual ≤ ~1.5σ, plausible `|t|`, no gross outliers. Record the
printed `prior_t_xyz` / `prior_rpy_deg` → these seed Round 2.

### Round 2 — Prior-seeded diverse collection
Add: `-p use_extrinsic_prior:=true -p prior_t_xyz:="[…R1…]" -p prior_rpy_deg:="[…R1…]"
-p prior_t_sigma_m:=0.10 -p prior_rot_sigma_deg:=30 -p min_points:=25`
1. Keep collecting to **25–30** poses until **all six diversity HUD bars are
   green** (targets: pitch 40° / roll 30° / yaw 40°, az 40° / el 15° / range 0.30 m).
2. Vary board tilt (pitch/yaw/roll) **and** rig position (near↔far, left↔right,
   up↔down) — orientation spread makes the offset observable, radar-point spread
   makes rotation observable. Both are required.

✅ **Gate:** all six bars green, `cond ≲ 5`, residual ≈ 1σ, LOO ≈ residual.

### Round 3 — Pin blind axis & finalize
1. From the per-DOF `1σ t` readout, identify the soft translation axis
   (radar1 → t_z; radar2 → t_x). Tape-measure it and re-solve with a tight prior
   on that axis: `-p prior_t_sigma_m:=0.04` (keep the rest of the R2 prior).
2. Save YAML/JSON + the emitted `static_transform_publisher` command.
3. Offline re-solve the saved poses:
   `cd sessions && python3 solve_from_poses.py <radarN>_poses.json` — must
   reproduce the live extrinsic to **< 1° / < 40 mm**.

✅ **DONE gate** — all of Part D must hold.

---

## Part D — "Done" definition (all must hold)

| gate | threshold | notes |
|---|---|---|
| residual / LOO | ≈ 1σ, LOO ≈ residual | noise model matches reality |
| condition number | ≲ 5 | pose diversity |
| rot 1σ | ≲ 3° | needs wide azimuth spread |
| t 1σ | ≲ 40 mm | on observed axes |
| diversity HUD | all six bars green | — |
| offline re-solve | < 1° / < 40 mm vs live | independent confirmation |
| apex offset | X/Y match the frozen rig value within combined 1σ | radar1 is the reference |

**Not a gate:** the vertical (radar1) / horizontal (radar2) 3-D **RMS** is the
IWR6843's random per-detection angular noise (weak elevation / rolled mount). It
averages out in fusion. Judge accuracy by **signed bias** and **1σ**, never RMS.

---

## Part E — Per-round log template (fill one per round, per radar)

Keeps every round reproducible (the gap that made radar2-final and radar_infra
un-resolvable). Save to `sessions/<date>_zed_<radar>_round<N>.md` + a poses JSON.

```
radar: radarN      round: N      date/time (UTC):
poses: __ (inliers __)      radar_range_scale: __      prior: none | from R(N-1)
|t| (m): [ , , ]   rpy(deg): [ , , ]   |t|=__
1σ rot (deg): [ , , ]        1σ t (mm): [ , , ]
apex offset (m): [ , , ]  1σ (mm): [ , , ]
residual / LOO / cond: __σ / __σ / __
signed bias (mm): X __  Y __  Z __
diversity HUD: pitch_ roll_ yaw_ | az_ el_ range_   (green/red each)
gate result: PASS advance to R(N+1)  |  HOLD — collect: ______
poses JSON: <file>
```

---

## Cross-checks between radars

1. **Apex offset must agree.** After radar2, its solved offset X/Y must match
   radar1's within combined 1σ (radar1: X +288±23, Y +614±21 mm). A match is the
   strongest evidence both extrinsics are correct — the radars share only the rig.
2. **Blind axes are complementary.** radar1 is sharp horizontally / soft
   vertically; radar2 (rolled ~90°) is the reverse. Fusing both constrains every
   axis — see `radar_fusion_reflector.py` / `sessions/two_radars.md`.
