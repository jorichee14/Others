# Calibration session — ZED left ↔ radar_infra (IWR6843ISK) — ROUND 1 (bootstrap)

- **Date/time (UTC):** 2026-07-15
- **Profile:** STATIC / step-and-settle (`radar_camera_calibration`)
- **Frames:** parent `zed_left_camera_optical_frame` (X=right, Y=down, Z=fwd) ← child `radar_infra_link` (X=fwd, Y=left, Z=up)
- **Status:** ⚠️ **ROUND 1, 10 poses — NOT FINAL.** The fit is self-consistent (residual 1.15σ, cond 3.6) but the per-DOF uncertainty is still loose (rot 1σ ≈ 7°, t 1σ up to 106 mm). Use as a **seed / extrinsic prior** for round 2, not as a deployed transform.

## Live node result (10 captures)

| quantity | value |
|---|---|
| `T_cam_radar` translation (m) | **[−0.0165, −0.0205, −0.0666]**  (\|t\| 7.2 cm) |
| quaternion (xyzw) | [−0.4735, −0.4611, −0.4587, +0.5939]  (\|q\| = 1.000) |
| rpy (deg) | [−47.79, −79.16, −35.13] |
| **1σ rot (deg)** | **[7.15, 6.90, 6.56]** — loose |
| **1σ t (mm)** | **[82.2, 106.1, 56.7]** — loose |
| captures / inliers | 10 |
| in-sample RMS | 214.7 mm |
| residual / cond | 1.15σ / 3.6 |

**Read:** the residual (1.15σ) says the model fits the data well and there are no
gross outliers, but the wide 1σ on every DOF says the **pose set is too small /
not diverse enough** to pin the extrinsic yet. Typical round-1 behaviour — collect
20–30 poses with wider azimuth/elevation and board tilt (watch the diversity HUD),
using this result as the prior, then re-solve.

## Next step (round 2)

Re-run with this result as the extrinsic prior and keep collecting:

```bash
-p use_extrinsic_prior:=true \
-p prior_t_xyz:="[-0.0165,-0.0205,-0.0666]" \
-p prior_rpy_deg:="[-47.79,-79.16,-35.13]" \
-p prior_t_sigma_m:=0.10 -p prior_rot_sigma_deg:=30.0 \
-p child_frame:=radar_infra_link -p radar_name:=radar_infra \
-p min_points:=25
```

Target for “done”: rot 1σ ≲ 3°, t 1σ ≲ 40 mm, all diversity bars green, and the
apex offset matching radar1's well-observed value ([+0.288, +0.614, −0.009] m) as a
cross-check (see [`two_radars.md`](two_radars.md)).

> Raw per-pose captures were not logged for this round (only the solve summary
> above was emitted), so there is no offline `solve_from_poses.py` reproduction for
> this set. Enable session saving on the round-2 run to get a reproducible record.
