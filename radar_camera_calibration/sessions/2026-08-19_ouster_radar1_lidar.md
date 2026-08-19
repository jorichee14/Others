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
  0.0279 0.1334 -0.2100  0.11161 0.02673 0.99330 -0.01330 \
  os_lidar radar1_link

# composed through GLIM:  zed_left_camera_optical_frame -> radar1_link
ros2 run tf2_ros static_transform_publisher \
  0.200334 0.117691 -0.103728  0.557558 -0.543698 0.463363 0.422867 \
  zed_left_camera_optical_frame radar1_link
```

Fusion-node form:

```
r1_t_xyz:="[0.2003,0.1177,-0.1037]"   r1_quat_xyzw:="[0.5576,-0.5437,0.4634,0.4229]"
```

## Result

`T_os_lidar_radar1_link` — 27 of 33 poses, range 0.95–4.42 m.

| | value |
|---|---|
| t (m) | [+0.0279, +0.1334, −0.2100] |
| \|t\| | 25.0 cm |
| quat xyzw | [+0.11161, +0.02673, +0.99330, −0.01330] |
| rpy (deg) | [+2.95, −12.85, −178.80] |
| rot 1σ (deg) | [3.75, **6.55**, 1.18] |
| t 1σ (mm) | [11.4, 31.1, **59.5**] |
| signed bias (mm) | [−7.9, −0.1, **+40.0**] |
| 3-D RMS (mm) X/Y/Z | [102, 212, 377] |
| residual / LOO | 1.33σ / 1.47σ |
| condition number | 4.2 |

Radar axes expressed in the lidar frame:

```
X -> [-0.97 -0.02 +0.22]     boresight, pointing along lidar -X (yawed ~180 deg)
Y -> [+0.03 -1.00 +0.05]     radar +Y is lidar -Y
Z -> [+0.22 +0.06 +0.97]     upright, pitched ~13 deg down
```

The radar is **upright** (its +Z lands within 13° of lidar +Z), sits **21 cm
below** and **13 cm to the left of** the lidar, and is **yawed 180°** — it faces
along lidar −X. Earlier radar↔camera work called this mount inverted; the lidar
data says it is not.

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

Independent checks still open: tape-measure lidar→radar (predicted 25.1 cm) and
camera→radar (predicted 25.5 cm), and physically confirm the radar is upright and
pitched ~14° down.

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
