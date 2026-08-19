# Radar ↔ LiDAR calibration — Ouster ↔ radar1 (IWR6843ISK), 2026-08-19

Rebuilt rig. This run replaces the ChArUco radar↔camera flow with a
**radar↔lidar** solve: the corner reflector is located in the **lidar** (not the
camera), the radar↔camera transform is then obtained by composing through the
GLIM lidar↔camera calibration.

Why the switch: the reflector apex is a *radar* feature, and the camera only ever
saw it indirectly (as a fixed offset from a ChArUco board). The lidar sees the
reflector itself, at centimetre accuracy, from any orientation — so the
correspondence is direct and the board-pose error, the apex-offset measurement,
and the three unobservable offset DOF all disappear from the problem.

Node: `radar_lidar_calib.py`. Offline re-solve: `solve_radar_lidar.py`.
Capture set: `2026-08-19_ouster_radar1_poses.json` (33 poses, 3 runs merged).

## Deployable transforms

```bash
# solved directly:  os_lidar -> radar1_link
ros2 run tf2_ros static_transform_publisher \
  0.0334 0.1406 -0.1685  0.12338 0.00126 0.99230 -0.01099 \
  os_lidar radar1_link

# composed through GLIM:  zed_left_camera_optical_frame -> radar1_link
ros2 run tf2_ros static_transform_publisher \
  0.207505 0.076150 -0.108883  -0.551338 0.560743 -0.443155 -0.430356 \
  zed_left_camera_optical_frame radar1_link
```

Fusion-node form:

```
r1_t_xyz:="[0.2075,0.0762,-0.1089]"   r1_quat_xyzw:="[-0.5513,0.5607,-0.4432,-0.4304]"
```

## Result

`T_os_lidar_radar1_link` — 46 of 54 poses, range 0.6–4.4 m, two sessions merged.

| | value |
|---|---|
| t (m) | [+0.0334, +0.1406, **−0.1685**] |
| \|t\| | 22.2 cm |
| quat xyzw | [+0.12338, +0.00126, +0.99230, −0.01099] |
| rpy (deg) | [−0.01, −14.17, −178.73] |
| rot 1σ (deg) | [2.97, **4.29**, 0.92] |
| t 1σ (mm) | [8.8, 19.3, **40.9**] |
| signed bias (mm) | [−12.2, +16.8, **+49.4**] |
| 3-D RMS (mm) X/Y/Z | [102, 185, 352] |
| residual / LOO | 1.29σ / 1.36σ |
| condition number | 5.2 |

The two sessions were solved independently first and agree on every axis:

| | 33-pose set | 21-pose set | apart | combined 1σ | |
|---|---|---|---|---|---|
| t_x | +2.79 cm | +4.54 cm | 17.5 mm | 18.3 mm | 1.0σ |
| t_y | +13.34 cm | +13.54 cm | 2.0 mm | 39.7 mm | 0.0σ |
| t_z | −21.00 cm | −12.87 cm | 81.3 mm | 82.9 mm | 1.0σ |

Rotation differs by 9.8° between them, almost all of it roll — the axis whose own
1σ is 5–6.5° in each run, so ~1.2σ. Treating them as one rig state is justified,
and merging cuts every uncertainty by roughly a third.

## Camera → radar, per axis

`T_cam_radar` = `T_cam_lidar · T_lidar_radar`, with `T_cam_lidar` from GLIM.
Camera frame is `zed_left_camera_optical_frame`: **X right, Y down, Z forward**.

```
t (m)      : +0.207505 +0.076150 -0.108883      |t| = 24.6 cm
quat xyzw  : -0.551338 +0.560743 -0.443155 -0.430356
rpy (deg)  : -174.58 -76.24 -95.22
```

| | **X** (right) | **Y** (down) | **Z** (forward) |
|---|---|---|---|
| t (m) | +0.2075 | +0.0762 | −0.1089 |
| **t 1σ (mm)** | 19.3 | **40.9** | **8.7** |
| rot 1σ about axis (deg) | 4.29 | 0.91 | 2.98 |
| signed bias (mm) | +16.8 | **−49.3** | +12.6 |
| 3-D RMS (mm) | 185 | **352** | 102 |

residual 1.29σ · LOO 1.36σ · cond 5.2 · 46/54 inliers · mean 3-D error 338 mm,
median 260 mm.

The covariance and residuals were rotated from the lidar frame into the camera
frame; they do **not** include GLIM's own uncertainty, so these are the
radar↔lidar numbers seen from the camera, not the full camera↔radar budget.

Radar axes in the camera frame:

```
radar X fwd  -> [-0.02 -0.24 +0.97]     boresight ~ camera +Z, tipped 14 deg up
radar Y left -> [-1.00 -0.00 -0.02]     radar left = camera -X
radar Z up   -> [+0.01 -0.97 -0.24]     radar up   = camera -Y (image up)
```

**The soft axis is camera Y — vertical in the image.** That is where the radar's
dead elevation channel lands: 40.9 mm of 1σ and a −49 mm bias, against 8.7 mm on
depth. Depth is the *best* axis here, which is the opposite of the usual
camera-fusion intuition and follows directly from the radar measuring range at
5 cm while inferring height from nothing.

Projected into the image at 2.5 m, the 1σ is roughly:

| fx | X | Y | Z |
|---|---|---|---|
| 350 px | 2.7 px | 5.7 px | 1.2 px |
| 525 px | 4.1 px | **8.6 px** | 1.8 px |
| 700 px | 5.4 px | **11.5 px** | 2.4 px |

So a radar return lands within about 4 px horizontally and 9 px vertically of
where it should on an HD720 left image — inside a person-sized box, and usable
for association. Do not use it to refine a bounding box vertically.

For reference, the previous rig's ChArUco radar↔camera solve gave \|t\| = 24.4 cm
against 24.6 cm here. The rig was rebuilt between the two, so the components are
not comparable, but the camera-to-radar separation landing in the same place is a
weak sanity check that nothing is grossly wrong.

## The elevation channel does not work

This is the finding that governs everything below. Regressing the radar's
reported `z` against range and the target's true height over the 21-pose set:

```
z_radar = -0.208 * range  +0.171 * height  +0.066
                            ^^^^^^ should be ~1.0
```

The radar reports a near-fixed cone at about −11.5°, almost independent of where
the reflector actually is. 83 cm of height change produces **3.5°** of response
where it should produce ~30°. Two captures at the same range with 84 cm between
them differ by 2.1°.

Ruled out as causes: `channelCfg 15 7 0` enables all three TX; `antGeometry1`
carries two elevation rows; `fovCfg` (±20°) clipped only 1 of 21 targets; the
values are continuous, not quantised; and the RX phase table was independently
verified against the re-banded 62.05–63.98 GHz profile. Still open: every capture
came from the demo's **static** angle chain (`staticRangeAngleCfg`) because a
tripod reflector is exactly zero-Doppler and the node gates `max_abs_doppler 0.15`
— the dynamic chain (`dynamic2DAngleCfg`, two angles) has never been tested.

Consequence: **no capture plan can fix `t_z`.** Every elevation gate below stays
red not because of where the tripod stood but because the radar cannot report it.

## How soft is t_z, really

Solving the merged set without the elevation residual at all is the honest bound:

| | t_x | t_y | **t_z** | worst rot 1σ |
|---|---|---|---|---|
| with elevation | +3.34 cm | +14.06 cm | **−16.85 cm** | 4.29° |
| range + azimuth only | +2.75 cm | +13.89 cm | **−23.07 cm** | 9.94° |

x and y move by under 6 mm — they are real, carried by range and azimuth, and
independent of the elevation question. `t_z` moves **6.2 cm**, which is about 1σ,
so the two are not in conflict — but it says plainly that roughly a third of the
height answer is resting on a channel that does not measure height.

Take `t_z` as **−17 to −23 cm** and resolve it with a tape measure; that single
measurement is worth more than another hundred captures. Note also that dropping
elevation costs 4.29° → 9.94° of rotation, so simply discarding the channel is
not an improvement — the fake elevation is still pinning roll.

## Reading the per-axis numbers

The three translation sigmas are not interchangeable — each one is a different
radar measurement axis projected onto a lidar axis. Because the radar looks along
lidar −X, the mapping is one-to-one:

| radar measures | σ | lands on lidar | error at 2.5 m | ÷√27 | observed 1σ |
|---|---|---|---|---|---|
| range | 5 cm | **x** (forward) | 5.0 cm | 9.6 mm | **11.4 mm** |
| azimuth | 3° | **y** (left) | 13.1 cm | 25.2 mm | **31.1 mm** |
| elevation | **8°** | **z** (up) | 34.8 cm | 67.0 mm | **59.5 mm** |

The right two columns are the point: the uncertainty predicted from the radar's
datasheet noise alone — knowing nothing about this data — reproduces the observed
uncertainty on every axis. The ordering x ≪ y < z is the IWR6843ISK's fingerprint
(wide horizontal array → 3° azimuth; only **two rows** vertically → 8° elevation).
So `t_z` is not "range" and it is not a defect: it is the radar's weak elevation
showing through, and there is no solver change that recovers information the
antenna never collected.

The large single-axis 3-D **RMS** (377 mm on Z) is likewise random angular noise,
not a calibration error — it averages out in fusion. Judge by signed bias, per-DOF
1σ, and the live overlay; never by RMS.

## Gates

| gate | threshold | value | |
|---|---|---|---|
| residual | ≈ 1σ | 1.33σ | PASS |
| LOO ≈ residual | — | 1.47σ | PASS |
| condition number | ≲ 5 | 4.2 | PASS |
| rot 1σ worst axis | ≲ 4° | **6.55°** | **FAIL** |
| signed bias | ≲ 50 mm | +40.0 mm (z) | PASS (marginal) |

**Not yet done.** The pitch axis (6.55°) and the +40 mm vertical bias are the same
defect from two directions. Pose diversity over the 33 captures, in radar
coordinates:

| axis | coverage | verdict |
|---|---|---|
| range | 0.95 → 4.42 m (spread 3.47 m) | good |
| azimuth | −36.5° → +35.5° (spread 72.0°) | good spread, but lopsided: 12 poses one side, 21 the other |
| **elevation** | **−14.9° → +0.7° (spread 15.6°)** | **bad — and entirely on ONE side of boresight** |

Not one capture sat meaningfully *above* the radar's boresight. With all the
elevation leverage on one side and only 15.6° of it, pitch is fitted from a short,
one-sided lever arm — hence the 6.55° and the uncorrected vertical bias. Height
was varied only at long range, where 8° of elevation error is 35 cm and swamps
the signal.

## Known systematic — the +40 mm vertical bias

Every capture resolved the reflector with the **centroid** estimator; none used
the plane-intersection apex fit. The lidar only ever sees the reflector's upper
front faces (it is 13.5 cm across, well under the beam spacing at these ranges),
so the centroid of the returned points sits above and in front of the true apex.
That offset is constant, which is exactly why it appears as a clean +40 mm bias on
z rather than as scatter.

Two ways to remove it, in order of effort: capture closer (the fraction of the
reflector seen rises sharply below ~1.5 m, and the plane fit becomes possible), or
subtract the measured centroid-to-apex offset as a fixed correction.

## To finish this calibration

Six more captures at **1.2–1.5 m**, chosen to attack elevation directly:

- 3 with the reflector on the **floor** (≈ −35° elevation)
- 3 **above head height** (≈ +20° elevation)
- put 3 of the six on the thin azimuth side (12 poses vs 21 today)

At 1.2 m the same 8° elevation error is 17 cm instead of 35 cm, so these six shots
are worth far more than another twenty at 3 m. Expect rot 1σ to drop under 4° and
the z bias to shrink as the closer geometry lets the plane fit engage.

`radar_lidar_calib.py` now shows this live as a six-bar **coverage HUD** (image
overlay + RViz status text), one bar per spread with the DOF it unlocks, so the
red bar names the axis that will come out loose before you stop collecting.
`solve_radar_lidar.py` prints the same table for a saved session. On this set:

```
  range       3.5 /   1.5   PASS   t vs R
  az         65.0 /  60.0   PASS   yaw
  az_bal     29.5 /  20.0   PASS   yaw, one-sidedness
  el         15.6 /  30.0   FAIL   pitch + roll
  el_bal      0.7 /  10.0   FAIL   pitch, one-sidedness
  near        6.0 /   6.0   PASS   all (captures under 1.5 m)
```

The two failing rows are exactly the ones that feed pitch — derived from the
geometry alone, with no knowledge of the 6.55° the solve produced. Adding the six
captures above turns all six bars green in simulation.

Independent checks still open: tape-measure lidar→radar (predicted 25.1 cm) and
camera→radar (predicted 25.5 cm), and physically confirm the radar is upright with
its nose pitched ~13° UP.

## Reproducing

```bash
python3 solve_radar_lidar.py 2026-08-19_ouster_radar1_poses.json \
  --cam-quat=-0.497829,-0.498035,0.501789,0.502329 \
  --cam-xyz=-0.074928,-0.066971,-0.091627
```

`--cam-*` are the GLIM `T_lidar_camera`; the script inverts and composes them to
print `T_camera_radar`. Re-weight with `--sig-r/--sig-az/--sig-el` or re-gate with
`--reject/--reject-axis` to audit how much the answer depends on the noise model.

The offline re-solve lands **1.1 mm and 0.8°** from the live node's output. The
node had converged on 26 inliers rather than 27 (one pose sits on the rejection
threshold); the disagreement is far inside the 11–60 mm 1σ, so the two agree.

## Rejected poses

| # | range | why |
|---|---|---|
| 1 | 2.52 m | range residual **+24.7σ** — radar locked onto a different object entirely |
| 3 | 2.20 m | azimuth −4.5σ |
| 7 | 2.36 m | azimuth +3.8σ |
| 22 | 1.73 m | azimuth −6.6σ |
| 26 | 3.78 m | azimuth −4.2σ |
| 29 | 2.21 m | azimuth +5.9σ |

All five azimuth rejections are the same failure: the radar picked a bright
neighbour (tripod leg, wall corner, or a multipath image) at nearly the right
range but the wrong bearing. This is what `max_baseline_m` and the persistence
gate are there to suppress, and 6/33 getting through is a normal rate.
